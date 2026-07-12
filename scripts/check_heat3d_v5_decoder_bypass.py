#!/usr/bin/env python3
"""Analytic fixture checks for V5 decoder-bypass audit semantics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_bypass_audit import (  # noqa: E402
    bypass_structure_recommendation,
    classify_feature_node_variation,
    compare_bypass_metric_rows,
)
from rigno.heat3d_v5_metrics import compute_sample_metrics, control_volume_weights  # noqa: E402


AUDIT_JSON = REPO_ROOT / "configs/heat3d_v5/v5_decoder_bypass_audit.json"
AUDIT_CSV = REPO_ROOT / "configs/heat3d_v5/v5_decoder_bypass_audit.csv"


def _metric_row(sample_id: str, prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict:
    return compute_sample_metrics(
        {
            "sample_id": sample_id,
            "split": "valid_iid",
            "prediction_deltaT_K": prediction,
            "target_deltaT_K": target,
            "control_volumes_m3": weights,
            "q_W_m3": np.arange(1.0, target.size + 1.0),
            "prediction_normalized": prediction,
            "target_normalized": target,
        }
    )


def main() -> int:
    artifact = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    csv_bytes = AUDIT_CSV.read_bytes()
    if b"\r\n" in csv_bytes:
        raise AssertionError("committed bypass audit CSV must use LF line endings")
    actual_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    table = artifact["per_sample_table"]
    if actual_sha256 != table["sha256"]:
        raise AssertionError("bypass audit JSON does not match the committed CSV bytes")
    with AUDIT_CSV.open("r", encoding="utf-8", newline="") as handle:
        table_rows = list(csv.reader(handle))
    if len(table_rows) - 1 != int(table["row_count"]) or len(table_rows[0]) != int(table["column_count"]):
        raise AssertionError("bypass audit CSV row/column metadata drifted")

    feature_names = ("k_z", "q", "is_bottom", "top_h", "log_Lx")
    first = np.asarray(
        [
            [2.0, 0.0, 1.0, 5.0, 1.0],
            [2.0, 3.0, 0.0, 5.0, 1.0],
            [4.0, 7.0, 0.0, 5.0, 1.0],
        ]
    )
    second = np.asarray(
        [
            [3.0, 0.0, 1.0, 6.0, 2.0],
            [3.0, 2.0, 0.0, 6.0, 2.0],
            [3.0, 8.0, 0.0, 6.0, 2.0],
        ]
    )
    variation = classify_feature_node_variation(feature_names, [first, second])
    by_name = {row["feature_name"]: row for row in variation}
    assert by_name["q"]["classification"] == "node_varying"
    assert by_name["k_z"]["classification"] == "mixed_node_variation"
    assert by_name["top_h"]["classification"] == "sample_global_broadcast"
    assert by_name["log_Lx"]["duplicate_of_sample_global_context"] is True
    decision = bypass_structure_recommendation(variation)
    assert decision["decision"] == "retain_local_bypass_and_remove_global_broadcast_duplicates"
    assert set(decision["local_bypass_feature_names"]) == {"k_z", "q", "is_bottom"}

    coords = np.asarray([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    weights = control_volume_weights(coords)
    target = np.linspace(0.2, 2.0, 8)
    full_rows = [_metric_row("a", target * 1.02, target, weights)]
    without_rows = [_metric_row("a", target * 1.25, target, weights)]
    comparison = compare_bypass_metric_rows(full_rows, without_rows)
    reduction = comparison["bypass_error_reduction_positive_is_better"]
    assert reduction["raw_cv_weighted_rmse_K"] > 0.0
    assert abs(reduction["shape_cv_rmse"]) < 1.0e-12
    print(
        json.dumps(
            {
                "status": "passed",
                "decision": decision["decision"],
                "raw_cv_rmse_reduction": reduction["raw_cv_weighted_rmse_K"],
                "audit_csv_sha256": actual_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
