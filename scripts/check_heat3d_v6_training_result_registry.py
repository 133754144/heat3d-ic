#!/usr/bin/env python3
"""Check the frozen valid-only V6 result registry and diagnostic payload."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/heat3d_v6/v6_training_result_registry.csv"
CHECKPOINTS = ROOT / "configs/heat3d_v6/v6_training_checkpoint_metrics.csv"
PAYLOAD = ROOT / "configs/heat3d_v6/v6_latest_training_results.json"


def main() -> int:
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with CHECKPOINTS.open(newline="", encoding="utf-8") as handle:
        checkpoints = list(csv.DictReader(handle))
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))

    expected = {
        "V6_01_V4best",
        "V6_02_V5best",
        "V6_03_V5best_P1h",
        "V6_04_V5best_P1h_DualAttention",
    }
    assert {row["config_id"] for row in rows} == expected
    assert len(rows) == 4 and len(checkpoints) == 14
    assert payload["status"] == "passed"
    assert payload["evaluation_scope"] == "valid_iid_saved_predictions_only"
    assert payload["test_accessed"] is False
    assert payload["training_started"] is False
    assert payload["checkpoint_inference_executed"] is False

    for row in rows:
        assert row["execution_status"] == "completed_e600"
        assert row["evaluation_status"] == "completed_valid_iid_saved_predictions"
        assert row["final_epoch"] == "600"
        assert row["valid_sample_count"] == "128"
        assert row["threshold_lt20"] == "True"
        assert row["result_scope"] == "valid_iid_saved_predictions_only"
        for key in (
            "point_global_relative_rmse_pct",
            "sample_first_relative_rmse_pct",
            "raw_rmse_K",
            "base_mse",
            "amplitude_ratio",
            "spatial_correlation",
            "shape_cv_rmse",
            "scale_log_rmse",
        ):
            assert math.isfinite(float(row[key])), f"{row['config_id']}:{key}"
    for row in checkpoints:
        assert len(row["checkpoint_sha256"]) == 64
        assert len(row["prediction_sha256"]) == 64
        assert row["valid_sample_count"] == "128"
        assert row["node_count"] == "1024"

    by_id = {row["config_id"]: row for row in rows}
    assert float(by_id["V6_04_V5best_P1h_DualAttention"]["point_global_relative_rmse_pct"]) < float(
        by_id["V6_03_V5best_P1h"]["point_global_relative_rmse_pct"]
    )
    assert by_id["V6_03_V5best_P1h"]["dataset_id"] == (
        by_id["V6_04_V5best_P1h_DualAttention"]["dataset_id"]
    )
    assert payload["ranking_by_primary_point_global"][0]["config_id"] == (
        "V6_04_V5best_P1h_DualAttention"
    )
    assert payload["paired_v6_03_v6_04"]["sample_count"] == 128
    print(
        json.dumps(
            {
                "status": "passed",
                "run_count": len(rows),
                "checkpoint_count": len(checkpoints),
                "best_config": payload["ranking_by_primary_point_global"][0]["config_id"],
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
