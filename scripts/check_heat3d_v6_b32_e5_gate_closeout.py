#!/usr/bin/env python3
"""Validate the frozen two-host V6 B32 e5 selective-launch closeout."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "configs/heat3d_v6/v6_b32_e5_gate_closeout.json"
PREREG = ROOT / "configs/heat3d_v6/v6_b32_selective_launch_gate.json"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"

EXPECTED = {
    "V6_01_V4best_B32": {
        "host": "devbox",
        "steady_reference_s": 444.9469199340092,
        "b24_peak_memory_mb": 2569.2861328125,
    },
    "V6_02_V5best_B32": {
        "host": "wsl2",
        "steady_reference_s": 825.0061747839209,
        "b24_peak_memory_mb": 2894.994384765625,
    },
}


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert report["schema_version"] == "heat3d_v6_b32_e5_gate_closeout_v1"
    assert report["status"] == "completed"
    assert report["formal_training_started"] is False
    assert report["e600_started"] is False
    assert report["roles_materialized"] == ["train", "valid_iid"]
    assert report["forbidden_roles_materialized"] == []
    assert report["dataset"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert report["pre_registered_gate"] == str(PREREG.relative_to(ROOT))
    assert report["effective_batch"] == {
        "configured": 32,
        "effective": 32,
        "micro_batch_size": 8,
        "micro_batches_per_epoch": 96,
        "micro_batch_sample_counts_unique": [8],
        "micro_batches_per_update": 4,
        "updates_per_epoch": 24,
        "effective_batch_sample_counts_unique": [32],
        "tail": None,
        "geometry_split": False,
        "padding_policy": "repeat_existing_dummy_edges_only",
    }

    derived_passes: dict[str, bool] = {}
    for config_id, expected in EXPECTED.items():
        run = report["runs"][config_id]
        assert run["status"] == "completed_e5"
        assert run["host"] == expected["host"]
        assert run["epochs"] == 5 and run["random_initialization"] is True
        assert run["sample_counts"] == {"train": 768, "valid_iid": 128}
        assert run["group_counts"] == {
            "train_micro_batches_per_epoch": 96,
            "valid_iid_batches": 4,
            "test_iid": 0,
            "all": 0,
        }
        assert run["optimizer_updates_per_epoch"] == 24
        assert run["finite_loss_gradient_update"] is True
        assert run["oom_nan_inf"] is False
        times = run["timing"]["epoch_s"]
        assert [entry["epoch"] for entry in times] == [1, 2, 3, 4, 5]
        assert all(_finite_positive(entry["seconds"]) for entry in times)
        mean_2_5 = sum(entry["seconds"] for entry in times[1:]) / 4.0
        assert math.isclose(run["timing"]["mean_epoch_2_to_5_s"], mean_2_5)
        assert math.isclose(run["timing"]["b24_steady_reference_s"], expected["steady_reference_s"])
        speedup = 100.0 * (1.0 - mean_2_5 / expected["steady_reference_s"])
        assert math.isclose(run["timing"]["speedup_vs_b24_pct"], speedup)

        device = run["device"]
        assert device["platform"] == "gpu"
        assert _finite_positive(device["peak_allocator_memory_mb"])
        assert math.isclose(device["b24_peak_allocator_memory_mb"], expected["b24_peak_memory_mb"])
        memory_change = 100.0 * (
            device["peak_allocator_memory_mb"] / expected["b24_peak_memory_mb"] - 1.0
        )
        assert math.isclose(device["allocator_memory_change_pct"], memory_change)
        assert _finite_positive(device["allocator_memory_limit_mb"])
        assert device["peak_allocator_memory_mb"] < device["allocator_memory_limit_mb"]
        assert device["nvml"]["sample_count"] > 0
        assert 0 <= device["nvml"]["gpu_utilization_mean_pct"] <= 100
        assert 0 <= device["nvml"]["gpu_utilization_max_pct"] <= 100

        reload = run["checkpoint_prediction_reload_audit"]
        assert reload["status"] == "passed"
        assert reload["entries"]
        for entry in reload["entries"]:
            assert entry["passed"] is True
            assert entry["parameter_reload_max_abs_error"] == 0.0
            assert entry["npz_reload_max_abs_error_K"] == 0.0
            assert entry["checkpoint_reload_max_abs_error_K"] <= 0.1
            assert entry["checkpoint_reload_rmse_K"] <= 0.01
        assert all(len(value) == 64 for value in run["artifact_sha256"].values())
        assert all(len(value) == 64 for value in run["evidence_sha256"].values())

        expected_checks = {
            "e5_completed": True,
            "epochs_2_to_5_finite": True,
            "no_oom_nan_inf": True,
            "checkpoint_export_reload_passed": True,
            "test_and_all_group_counts_zero": True,
            "peak_gpu_memory_mb_not_above_B24_reference": memory_change <= 0.0,
            "mean_epoch_2_to_5_s_strictly_below_B24_steady_reference": speedup > 0.0,
        }
        assert run["gate_checks"] == expected_checks
        derived_passes[config_id] = all(expected_checks.values())

    assert report["joint_gate"]["per_run_pass"] == derived_passes
    joint_pass = all(derived_passes.values())
    assert report["joint_gate"]["passed"] is joint_pass
    expected_decision = "B32" if joint_pass else "unified_B24_fallback"
    assert report["joint_gate"]["decision"] == expected_decision
    assert report["joint_gate"]["formal_launch_deferred_by_user"] is True
    assert report["joint_gate"]["selected_configs"] == (
        [
            "configs/heat3d_v6/V6_01_V4best_B32.yaml",
            "configs/heat3d_v6/V6_02_V5best_B32.yaml",
        ]
        if joint_pass
        else [
            "configs/heat3d_v6/V6_01_V4best.yaml",
            "configs/heat3d_v6/V6_02_V5best.yaml",
        ]
    )
    context = report["runs"]["V6_02_V5best_B32"]["global_context"]
    assert context == {
        "dimension": 24,
        "fit_population": "train_only",
        "fit_sample_count": 768,
        "target_or_label_derived_inputs": False,
    }
    assert prereg["formal_training_started"] is False
    print(json.dumps({"status": "passed", "decision": expected_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
