#!/usr/bin/env python3
"""Check frozen V6_03 seed0/1/2 saved-valid result artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/heat3d_v6/v6_multiseed_training_results.csv"
CHECKPOINTS = (
    ROOT / "configs/heat3d_v6/v6_multiseed_checkpoint_metrics.csv"
)
PAYLOAD = ROOT / "configs/heat3d_v6/v6_multiseed_training_results.json"
REPORT = ROOT / "docs/v6_multiseed_training_results.md"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    registry = _rows(REGISTRY)
    checkpoints = _rows(CHECKPOINTS)
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    if (
        payload["status"] != "passed"
        or payload["evaluation_scope"]
        != "valid_iid_saved_predictions_only"
        or payload["test_accessed"]
        or payload["hard_accessed"]
        or payload["sealed_accessed"]
        or payload["training_started"]
        or payload["checkpoint_inference_executed"]
        or payload["remote_inference_executed"]
    ):
        raise AssertionError("top-level multi-seed contract failed")
    if [int(row["seed"]) for row in registry] != [0, 1, 2]:
        raise AssertionError("registry seed set/order drifted")
    if len(checkpoints) != 12:
        raise AssertionError("expected 3 seeds x 4 checkpoints")
    for row in registry:
        if (
            row["execution_status"] != "completed_e600"
            or row["evaluation_status"]
            != "completed_valid_iid_saved_predictions"
            or row["final_epoch"] != "600"
            or row["valid_sample_count"] != "128"
            or row["threshold_lt20"] != "True"
            or row["result_scope"]
            != "valid_iid_saved_predictions_only"
        ):
            raise AssertionError(f"seed{row['seed']}: lifecycle drift")
        for key in (
            "point_global_relative_rmse_pct",
            "sample_first_relative_rmse_pct",
            "raw_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
        ):
            if not math.isfinite(float(row[key])):
                raise AssertionError(f"seed{row['seed']}:{key}")
        if not (
            float(row["final_point_global_relative_rmse_pct"])
            > float(row["point_global_relative_rmse_pct"])
        ):
            raise AssertionError(f"seed{row['seed']}: no best-to-final degradation")
    for row in checkpoints:
        if (
            len(row["checkpoint_sha256"]) != 64
            or len(row["prediction_sha256"]) != 64
            or row["valid_sample_count"] != "128"
            or row["node_count"] != "1024"
        ):
            raise AssertionError("checkpoint binding drifted")
    ranking = payload["ranking_by_point_global"]
    if ranking[0]["seed"] != 2:
        raise AssertionError("latest seed ranking drifted")
    stats = payload["seed_statistics"][
        "point_global_relative_rmse_pct"
    ]
    if not (
        0.0 < float(stats["std_sample"]) < 0.1
        and float(stats["max"]) < 1.0
    ):
        raise AssertionError("point-global seed statistics drifted")
    report = REPORT.read_text(encoding="utf-8")
    for token in ("seed0/1/2", "valid_iid", "volume-representative"):
        if token not in report:
            raise AssertionError(f"report missing {token!r}")
    print(
        json.dumps(
            {
                "status": "passed",
                "run_count": len(registry),
                "checkpoint_count": len(checkpoints),
                "best_seed": ranking[0]["seed"],
                "test_hard_sealed_accessed": False,
                "checkpoint_inference_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
