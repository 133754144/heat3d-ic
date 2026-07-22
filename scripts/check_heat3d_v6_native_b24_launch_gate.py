#!/usr/bin/env python3
"""Validate native-B24 e1 evidence and the formal launch state."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "configs/heat3d_v6/v6_native_b24_launch_gate.json"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
EXPECTED_HOSTS = {"V6_01_V4best": "wsl2", "V6_02_V5best": "devbox"}


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "heat3d_v6_native_b24_launch_gate_v1"
    assert report["status"] in {"passed_ready_to_launch", "e600_started"}
    assert report["dataset"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert report["dataset"]["roles_materialized"] == ["train", "valid_iid"]
    assert report["dataset"]["forbidden_roles_materialized"] == []
    assert report["effective_batch"] == {
        "batch_size": 24,
        "micro_batch_size": 24,
        "forward_backward_per_epoch": 32,
        "optimizer_updates_per_epoch": 32,
        "train_batch_start_count": 32,
        "micro_batch_count_per_update_unique": [1],
        "effective_sample_count_unique": [24],
        "tail": None,
        "geometry_split": False,
    }
    assert report["gate"]["both_passed"] is True
    for config_id, host in EXPECTED_HOSTS.items():
        run = report["runs"][config_id]
        assert run["host"] == host and run["status"] == "passed_e1"
        assert run["train_batch_count"] == 32
        assert run["train_batch_start_count"] == 32
        assert run["finite_loss_gradient_update"] is True
        assert run["oom_nan_inf"] is False
        assert run["test_iid_group_count"] == 0 and run["all_groups_count"] == 0
        device = run["device"]
        expected_fraction = device["peak_allocator_memory_mb"] / device["allocator_memory_limit_mb"]
        assert math.isclose(device["peak_allocator_fraction"], expected_fraction)
        assert device["peak_allocator_fraction"] <= report["gate"]["peak_allocator_fraction_limit"]
        reload = run["checkpoint_prediction_reload_audit"]
        assert reload["status"] == "passed" and reload["entries"]
        for entry in reload["entries"]:
            assert entry["passed"] is True
            assert entry["parameter_reload_max_abs_error"] == 0.0
            assert entry["npz_reload_max_abs_error_K"] == 0.0
            assert entry["checkpoint_reload_max_abs_error_K"] <= 0.1
            assert entry["checkpoint_reload_rmse_K"] <= 0.01
        assert all(len(value) == 64 for value in run["artifact_sha256"].values())
        assert all(len(value) == 64 for value in run["evidence_sha256"].values())
        formal = report["formal_runs"][config_id]
        assert formal["host"] == host and formal["launch_method"] == "nohup"
        if report["e600_started"]:
            assert report["formal_training_started"] is True
            assert isinstance(formal["pid"], int) and formal["pid"] > 0
        else:
            assert report["formal_training_started"] is False
            assert formal["pid"] is None
    context = report["runs"]["V6_02_V5best"]["global_context"]
    assert context == {
        "dimension": 24,
        "fit_population": "train_only",
        "fit_sample_count": 768,
        "target_or_label_derived_inputs": False,
    }
    print(json.dumps({"status": "passed", "launch_status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
