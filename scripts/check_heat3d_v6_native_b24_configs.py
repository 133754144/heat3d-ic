#!/usr/bin/env python3
"""Validate native-B24 V6 formal configs and their isolated e1 preflights."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


PAIRS = {
    "V6_01_V4best": (
        ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
        ROOT / "configs/heat3d_v6/preflight/V6_01_V4best_B24_native_e1_gpu_preflight.yaml",
        "wsl2",
    ),
    "V6_02_V5best": (
        ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
        ROOT / "configs/heat3d_v6/preflight/V6_02_V5best_B24_native_e1_gpu_preflight.yaml",
        "devbox",
    ),
}


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = resolve_inherited_yaml(payload, path)
    value["config_id"] = payload["config_id"]
    return value


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaf_diffs(left.get(key), right.get(key), path))
        return result
    return [] if left == right else [prefix]


def _runtime_identity_free(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("config_id", None)
    value.get("run", {}).pop("epochs", None)
    for key in (
        "report_every", "train_metrics_schedule", "grad_norm_report_every",
        "profile_timing", "profile_timing_json", "memory_audit_jsonl",
        "memory_audit_every_batch", "memory_audit_gc",
    ):
        value.get("run", {}).pop(key, None)
    for key in ("output_dir", "run_name"):
        value.get("export", {}).pop(key, None)
    value.pop("metadata", None)
    return value


def main() -> int:
    reports = []
    output_paths: set[str] = set()
    for config_id, (formal_path, preflight_path, host) in PAIRS.items():
        formal = _resolved(formal_path)
        preflight = _resolved(preflight_path)
        validate_v2_config(formal, config_path=formal_path)
        validate_v2_config(preflight, config_path=preflight_path)
        assert formal["run"]["batch_size"] == 24
        assert formal["run"]["micro_batch_size"] == 24
        assert formal["run"]["validation_batch_size"] == 32
        assert formal["run"]["prediction_batch_size"] == 32
        assert formal["run"]["drop_last"] is False
        assert formal["run"]["epochs"] == 600
        assert formal["metadata"]["training_started"] is True
        assert formal["metadata"]["micro_batches_per_epoch"] == 32
        assert formal["metadata"]["optimizer_updates_per_epoch"] == 32
        assert formal["metadata"]["final_partial_effective_batch_size"] is None
        assert formal["metadata"]["b24_execution_mode"] == "one_real_B24_forward_backward_per_update"
        command = build_training_command(formal)
        assert "--batch-size" in command and command[command.index("--batch-size") + 1] == "24"
        assert "--micro-batch-size" in command and command[command.index("--micro-batch-size") + 1] == "24"

        assert preflight["run"]["epochs"] == 1
        assert preflight["run"]["train_metrics_schedule"] == "final_only"
        assert preflight["run"]["memory_audit_every_batch"] is True
        assert preflight["metadata"]["preflight_only"] is True
        assert preflight["metadata"]["assigned_host"] == host
        assert preflight["metadata"]["batching_adaptation"] == "native_B24_single_forward_backward"
        assert _runtime_identity_free(preflight) == _runtime_identity_free(formal)
        assert preflight["export"]["output_dir"] != formal["export"]["output_dir"]
        assert preflight["export"]["output_dir"] not in output_paths
        output_paths.add(preflight["export"]["output_dir"])
        assert not (ROOT / preflight["export"]["output_dir"]).exists()
        reports.append(
            {
                "config_id": config_id,
                "host": host,
                "formal_epochs": 600,
                "preflight_epochs": 1,
                "train_batches_per_epoch": 32,
                "forward_backward_per_epoch": 32,
                "optimizer_updates_per_epoch": 32,
                "batch_size": 24,
                "micro_batch_size": 24,
                "preflight_diff_paths": _leaf_diffs(formal, preflight),
            }
        )
    print(json.dumps({"status": "passed", "runs": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
