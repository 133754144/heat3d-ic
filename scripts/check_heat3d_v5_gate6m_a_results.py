#!/usr/bin/env python3
"""Check the persisted Gate 6M A valid-only result registration."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs/heat3d_v5/gate6m/gate6m_a_valid_only_metrics.json"
CSV_RESULT = ROOT / "configs/heat3d_v5/gate6m/gate6m_a_valid_only_metrics.csv"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6m_registry.csv"
REQUIRED = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
)


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heat3d_v5_gate6m_a_result_summary_v1"
    assert payload["status"] == "completed_valid_iid_only"
    assert payload["config_id"] == "V4P5_35_gate6m_v32_scale_head_only_e100"
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["evaluation_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert not scope["test_accessed"]
    assert not scope["hard_accessed"]
    assert not scope["sealed_iid_accessed"]
    assert not scope["training_started"]
    assert payload["normalization_and_context"]["normalization_recomputed_from_train_only"]
    assert payload["normalization_and_context"]["context_recomputed_from_train_only"]
    assert payload["normalization_and_context"]["fit_sample_count"] == 672
    assert payload["split"]["train_count"] == 672
    assert payload["split"]["valid_iid_count"] == 128
    assert payload["training_completion"]["final_epoch"] == 100
    assert payload["training_completion"]["epoch_history_contiguous_1_to_100"]
    assert payload["training_completion"]["declared_log_exists"] is False

    assert set(payload["metrics"]) == {
        "point_global_best",
        "sample_first_best",
        "legacy_best",
        "final",
    }
    for summary in payload["metrics"].values():
        assert set(REQUIRED) <= set(summary)
        assert all(math.isfinite(float(summary[field])) for field in REQUIRED)
    metadata = payload["checkpoint_metadata"]
    assert set(metadata) == set(payload["metrics"])
    assert {row["parameter_count"] for row in metadata.values()} == {893736}
    assert {row["epoch"] for row in metadata.values()} == {18, 25, 100}
    for row in metadata.values():
        assert len(row["sha256"]) == 64
        assert row["parameter_reload_max_abs_error"] == 0.0
        audit = row["training_reload_audit"]
        assert audit["passed"]
        assert audit["checkpoint_reload_max_abs_error_K"] <= audit["tolerance_K"]
        assert audit["npz_reload_max_abs_error_K"] == 0.0
    for epoch in ("e18", "e25", "e100"):
        native = payload["native_diagnostics_selected"][epoch]
        assert all(math.isfinite(float(value)) for value in native.values())

    with CSV_RESULT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["checkpoint"] for row in rows} == set(payload["metrics"])
    for row in rows:
        for field in REQUIRED:
            assert math.isfinite(float(row[field]))

    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        registry_rows = {row["config_id"]: row for row in csv.DictReader(handle)}
    a = registry_rows[payload["config_id"]]
    b = registry_rows["V4P5_36_gate6m_v32_epoch_regroup_e600"]
    # The phase registry preserves the user-managed lifecycle label; the
    # separate result columns carry the verified completed e100 evidence.
    assert a["execution_status"] == "started_user_managed"
    assert a["training_started"] == "true"
    assert a["result_v5_status"] == "completed_valid_only"
    assert a["result_v5_source"] == "wsl2"
    assert a["result_v5_required_metrics_complete"] == "true"
    assert a["result_v5_primary_epoch"] == "18"
    assert a["result_v5_legacy_epoch"] == "18"
    assert a["result_v5_primary_test_point_global_relative_rmse_pct"] == ""
    assert a["result_v5_threshold_pass"] == "valid_only_fail"
    assert b["execution_status"] == "not_started"
    assert b["training_started"] == "false"
    print(
        json.dumps(
            {
                "status": "passed",
                "config_id": payload["config_id"],
                "checkpoint_count": 4,
                "roles": ["train", "valid_iid"],
                "test_hard_sealed_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
