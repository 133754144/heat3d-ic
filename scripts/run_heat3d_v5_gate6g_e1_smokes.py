#!/usr/bin/env python3
"""Run Gate 6G e1-only compile/reload/memory smokes from registered e200 configs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


CONFIGS = (
    "V4P5_22_gate6g_control_constlr",
    "V4P5_23_gate6g_stopgrad_constlr",
    "V4P5_24_gate6g_shape_attention_constlr",
    "V4P5_25_gate6g_scale_attention_constlr",
    "V4P5_26_gate6g_shape_attention_stopgrad_constlr",
    "V4P5_27_gate6g_deep_scale_head_constlr",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", choices=CONFIGS, action="append", default=[])
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "output/heat3d_v5_gate6g_smoke/e1_smoke_summary.json",
    )
    return parser.parse_args()


def _resolved(config_id: str) -> dict[str, Any]:
    path = ROOT / f"configs/heat3d_v5/generated/{config_id}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved = copy.deepcopy(resolved)
    resolved["run"].update({
        "epochs": 1,
        "report_every": 1,
        "train_metrics_schedule": "final_only",
        "final_probe_eval_after_training": False,
        "post_training_diagnostics": False,
        "profile_timing": True,
        "memory_audit_every_batch": True,
        "memory_audit_jsonl": f"output/heat3d_v5_gate6g_smoke_memory/{config_id}.jsonl",
        "final_probe_output_dir": f"output/heat3d_v5_gate6g_smoke_final_probe/{config_id}",
        "post_training_diagnostics_output_dir": f"output/heat3d_v5_gate6g_smoke_diagnostics/{config_id}",
    })
    resolved["export"].update({
        "output_dir": f"output/heat3d_v5_gate6g_smoke_runs/{config_id}",
        "run_name": f"{config_id}_e1_smoke",
    })
    validate_v2_config(resolved, config_path=path)
    return resolved


def _memory(path: Path) -> dict[str, Any]:
    maxima = {
        "peak_rss_mb": 0.0,
        "peak_live_device_bytes": None,
        "peak_reserved_device_bytes": None,
        "peak_pool_bytes": None,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("rss_mb") is not None:
                maxima["peak_rss_mb"] = max(maxima["peak_rss_mb"], float(row["rss_mb"]))
            for device in row.get("jax_memory", {}).get("jax_devices", []):
                groups = {
                    "peak_live_device_bytes": ("bytes_in_use_mb", "peak_bytes_in_use_mb"),
                    "peak_reserved_device_bytes": ("bytes_reserved_mb", "peak_bytes_reserved_mb"),
                    "peak_pool_bytes": ("pool_bytes_mb", "peak_pool_bytes_mb"),
                }
                for output_name, keys in groups.items():
                    values = [float(device[key]) * 1024.0 * 1024.0 for key in keys if device.get(key) is not None]
                    if values:
                        maxima[output_name] = max(float(maxima[output_name] or 0.0), *values)
    return maxima


def _run(config_id: str) -> dict[str, Any]:
    resolved = _resolved(config_id)
    output_dir = ROOT / resolved["export"]["output_dir"]
    memory_path = ROOT / resolved["run"]["memory_audit_jsonl"]
    # A complete loss summary is the atomic resume marker. This lets a failed
    # collector resume without retraining a successful e1 model smoke.
    if not (output_dir / "loss_summary.json").is_file():
        command = build_training_command(resolved, python_executable=sys.executable)
        environment = dict(os.environ)
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    loss = json.loads((output_dir / "loss_summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    with (output_dir / "params_final.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    reload_entries = loss["checkpoint_prediction_reload_audit"]["entries"]
    assert len(reload_entries) >= 5 and all(bool(row["passed"]) for row in reload_entries)
    assert loss["status_ok"] is True and loss["grad_finite"] is True
    assert run_config["global_context"]["standardizer"]["fit_population"] == "train_only"
    assert int(run_config["global_context"]["standardizer"]["fit_sample_count"]) == 672
    return {
        "config_id": config_id,
        "status": "passed",
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "nodes_per_sample": 1024,
        "batch_size": int(run_config["batch_size"]),
        "parameter_count": int(checkpoint["param_count"]),
        "shape_attention_mode": run_config["model_config"].get("shape_attention_mode", "none"),
        "scale_attention_mode": run_config["model_config"].get("scale_attention_mode", "none"),
        "pooled_latent_stop_gradient": bool(run_config["model_config"].get("pooled_latent_stop_gradient", False)),
        "scale_head_depth": int(run_config["model_config"].get("scale_head_depth", 1)),
        "checkpoint_reload_entry_count": len(reload_entries),
        "checkpoint_reload_passed": True,
        "grad_finite": True,
        "valid_base_mse": float(loss["final_valid_base_mse"]),
        "valid_point_global_relative_rmse_pct": float(loss["rel_rmse_v4_pct"]),
        "output_dir": str(resolved["export"]["output_dir"]),
        "memory_audit_jsonl": str(resolved["run"]["memory_audit_jsonl"]),
        **_memory(memory_path),
    }


def main() -> int:
    args = _args()
    selected = tuple(args.config_id) if args.config_id else CONFIGS
    existing = {row["config_id"]: row for row in (
        json.loads(args.summary_json.read_text(encoding="utf-8")).get("results", [])
        if args.summary_json.is_file() else []
    )}
    for config_id in selected:
        existing[config_id] = _run(config_id)
        payload = {
            "schema_version": "heat3d_v5_gate6g_e1_smoke_v1",
            "status": "completed" if all(name in existing for name in CONFIGS) else "in_progress",
            "results": [existing[name] for name in CONFIGS if name in existing],
            "roles_accessed": ["train", "valid_iid"],
            "forbidden_roles_accessed": [],
            "sealed_iid_accessed": False,
            "long_training_started": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
