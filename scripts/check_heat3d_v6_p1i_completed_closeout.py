#!/usr/bin/env python3
"""Check the V6 P1i e600 prediction-only recovery closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "configs/heat3d_v6_p1i/v6_p1i_seed0_B24_valid_closeout.json"
DEFAULT_CSV = ROOT / "configs/heat3d_v6_p1i/v6_p1i_seed0_B24_valid_metrics.csv"
DEFAULT_REGISTRY = ROOT / "configs/heat3d_v6_p1i/v6_p1i_training_registry.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    closeout = json.loads(args.closeout.read_text(encoding="utf-8"))
    if closeout["status"] != "completed_e600_export_failed_valid_predictions_evaluated":
        raise AssertionError("closeout status drifted")
    if closeout["evaluation_scope"] != "valid_iid_only":
        raise AssertionError("evaluation scope is not valid-only")
    if closeout["test_access"] or closeout["sealed_access"]:
        raise AssertionError("test/sealed access must remain false")
    if closeout["training_commit"] != "dfe3cf6":
        raise AssertionError("training commit drifted")
    if closeout["dataset_manifest_sha256"] != (
        "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
    ):
        raise AssertionError("manifest SHA256 drifted")
    metrics = closeout["metrics"]
    if [row["checkpoint_epoch"] for row in metrics] != [542, 600]:
        raise AssertionError("checkpoint epochs drifted")
    for row in metrics:
        required = (
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
        if any(not math.isfinite(float(row[field])) for field in required):
            raise AssertionError("closeout contains a missing/non-finite metric")
        if row["checkpoint_available"]:
            raise AssertionError("missing checkpoints must not be reported available")
    if metrics[0]["point_global_relative_rmse_pct"] >= 20.0:
        raise AssertionError("recorded point-global gate should pass")
    paired = closeout["best_to_final"]
    if not 2.0 < float(paired["point_sse_change_pct"]) < 2.3:
        raise AssertionError("best-to-final point-SSE attribution drifted")
    if int(paired["point_sse_final_wins"]) != 61:
        raise AssertionError("paired point-SSE win count drifted")
    if int(paired["sample_relative_final_wins"]) != 57:
        raise AssertionError("paired sample-relative win count drifted")

    csv_rows = list(csv.DictReader(args.metrics.open(encoding="utf-8")))
    if [row["checkpoint_label"] for row in csv_rows] != ["point_global_best", "final"]:
        raise AssertionError("metrics CSV rows drifted")
    for csv_row, json_row in zip(csv_rows, metrics):
        for field in (
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
            "legacy_normalized_valid_base_mse",
        ):
            if float(csv_row[field]) != float(json_row[field]):
                raise AssertionError(f"CSV/JSON mismatch for {field}")

    registry_rows = list(csv.DictReader(args.registry.open(encoding="utf-8")))
    registry = next(
        row for row in registry_rows if row["config_id"] == "V6_05_V5best_P1i_seed0_B24"
    )
    if registry["execution_status"] != "completed_e600_export_failed":
        raise AssertionError("registry execution status drifted")
    if registry["evaluation_status"] != "completed_valid_only_from_predictions":
        raise AssertionError("registry evaluation status drifted")
    if registry["test_access"] != "closed_audited_holdout":
        raise AssertionError("test role was not kept closed")
    if registry["sealed_access"] != "closed_confirmatory":
        raise AssertionError("sealed role was not kept closed")

    if args.run_dir is not None:
        for row in metrics:
            name = "best_predictions.npz" if row["checkpoint_label"] == "point_global_best" else "predictions.npz"
            path = args.run_dir / name
            if _sha256(path) != row["prediction_sha256"]:
                raise AssertionError(f"prediction SHA256 mismatch: {path}")
        unexpected = sorted(args.run_dir.glob("params*.pkl"))
        if unexpected:
            raise AssertionError(f"closeout says checkpoints are missing but found {unexpected}")

    print("V6 P1i completed-run closeout checker: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
