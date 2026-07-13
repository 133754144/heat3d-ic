#!/usr/bin/env python3
"""Validate Gate-5 real-P5 smoke/calibration configs without training."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _combine_metric_payloads,
)


CONFIGS = {
    "N0_smoke": ("V5G5_N0_execution_smoke_e1.yaml", 1, "physics_only"),
    "N1_smoke": ("V5G5_N1_execution_smoke_e1.yaml", 1, "physics_plus_pooled_latent"),
    "N0_calibration": ("V5G5_N0_calibration_e10.yaml", 10, "physics_only"),
    "N1_calibration": ("V5G5_N1_calibration_e10.yaml", 10, "physics_plus_pooled_latent"),
}


def main() -> int:
    combined = _combine_metric_payloads(
        [
            (1, {
                "normalized_loss": 1.0,
                "raw_delta_mse": 4.0,
                "recovered_temperature_mse": 4.0,
                "mean_abs_true_deltaT": 2.0,
                "scale_log_abs_error": 0.25,
                "joint_relative_rmse": 0.5,
                "finite_ok": True,
                "shape_ok": True,
            }),
            (3, {
                "normalized_loss": 3.0,
                "raw_delta_mse": 16.0,
                "recovered_temperature_mse": 16.0,
                "mean_abs_true_deltaT": 4.0,
                "scale_log_abs_error": 0.75,
                "joint_relative_rmse": 1.5,
                "finite_ok": True,
                "shape_ok": True,
            }),
        ]
    )
    assert combined["scale_log_abs_error"] == 0.625
    assert combined["joint_relative_rmse"] == 1.25
    config_dir = ROOT / "configs/heat3d_v5/preflight"
    reports = {}
    output_dirs = set()
    for label, (filename, epochs, scale_mode) in CONFIGS.items():
        path = config_dir / filename
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        resolved = resolve_inherited_yaml(source, path)
        validate_v2_config(resolved, config_path=path)
        run = resolved["run"]
        model = resolved["model"]
        dataset = resolved["dataset"]
        export = resolved["export"]
        assert run["epochs"] == epochs
        assert run["batch_size"] == 28
        assert run["validation_batch_size"] == 128
        assert run["prediction_batch_size"] == 128
        assert run["train_metrics_schedule"] == "every_epoch"
        assert run["grad_norm_report_every"] == 1
        assert run["profile_timing"] is True
        expected_memory_every_batch = label.endswith("smoke")
        assert run["memory_audit_every_batch"] is expected_memory_every_batch
        assert run["final_probe_eval_after_training"] is False
        assert run["post_training_diagnostics"] is False
        assert dataset["subset_path"] == "data/heat3d_v4_p5_clean_nohard_v0"
        assert "train672_valid128" in dataset["split_map_path"]
        assert model["node_latent_size"] == 96
        assert model["native_output_mode"] == "native_shape_scale"
        assert model["native_branch_mode"] == "joint"
        assert model["scale_head_mode"] == scale_mode
        assert export["selection_metric"] == "valid_base_mse"
        assert resolved["metadata"]["formal_performance_result"] is False
        assert export["output_dir"] not in output_dirs
        output_dirs.add(export["output_dir"])
        command = build_training_command(resolved, python_executable="python")
        joined = shlex.join(command)
        for fragment in (
            f"--epochs {epochs}",
            "--batch-size 28",
            "--validation-batch-size 128",
            "--profile-timing",
            "--memory-audit-jsonl",
            "--no-final-probe-eval-after-training",
            "--no-post-training-diagnostics",
            f"--scale-head-mode {scale_mode}",
        ):
            assert fragment in joined, f"{label}: dry-run missing {fragment}"
        assert ("--memory-audit-every-batch" in joined) is expected_memory_every_batch
        reports[label] = {
            "config": str(path.relative_to(ROOT)),
            "epochs": epochs,
            "scale_head_mode": scale_mode,
            "output_dir": export["output_dir"],
            "training_started": False,
        }
    print(json.dumps({"status": "passed", "configs": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
