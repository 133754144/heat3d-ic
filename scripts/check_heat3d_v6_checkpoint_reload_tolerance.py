#!/usr/bin/env python3
"""Regression checks for sparse-safe V6 checkpoint prediction replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


MAX_TOLERANCE_K = 0.5
RMSE_TOLERANCE_K = 0.01
OUTLIER_ABS_THRESHOLD_K = 0.1
OUTLIER_FRACTION_TOLERANCE = 0.001


def _passes(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    parameter_error: float = 0.0,
    npz_error: float = 0.0,
) -> bool:
    difference = runner._prediction_difference_summary(expected, actual)
    npz_difference = {
        "max_abs_K": npz_error,
        "rmse_K": npz_error,
        "mean_abs_K": npz_error,
        "p9999_abs_K": npz_error,
        "count_gt_0p1_K": int(npz_error > 0.1),
        "point_count": difference["point_count"],
    }
    return runner._checkpoint_prediction_consistency_passes(
        parameter_max_abs=parameter_error,
        checkpoint_difference=difference,
        npz_difference=npz_difference,
        max_tolerance=MAX_TOLERANCE_K,
        rmse_tolerance=RMSE_TOLERANCE_K,
        outlier_abs_threshold=OUTLIER_ABS_THRESHOLD_K,
        outlier_fraction_tolerance=OUTLIER_FRACTION_TOLERANCE,
    )


def main() -> int:
    expected = {"sample": np.zeros((128, 1024, 1), dtype=np.float64)}

    exact = {"sample": np.zeros((128, 1024, 1), dtype=np.float64)}
    assert _passes(expected, exact)

    sparse_gpu_order_spike = {
        "sample": np.zeros((128, 1024, 1), dtype=np.float64)
    }
    sparse_gpu_order_spike["sample"].reshape(-1)[12345] = 0.3
    sparse_summary = runner._prediction_difference_summary(
        expected, sparse_gpu_order_spike
    )
    assert sparse_summary["max_abs_K"] == 0.3
    assert sparse_summary["rmse_K"] < RMSE_TOLERANCE_K
    assert sparse_summary["count_gt_0p1_K"] == 1
    assert _passes(expected, sparse_gpu_order_spike)

    excessive_single_point = {
        "sample": np.zeros((128, 1024, 1), dtype=np.float64)
    }
    excessive_single_point["sample"].reshape(-1)[12345] = 0.6
    assert not _passes(expected, excessive_single_point)

    broad_drift = {
        "sample": np.full((128, 1024, 1), 0.02, dtype=np.float64)
    }
    assert not _passes(expected, broad_drift)

    high_tail = {"sample": np.zeros((128, 1024, 1), dtype=np.float64)}
    high_tail["sample"].reshape(-1)[:200] = 0.2
    high_tail_summary = runner._prediction_difference_summary(expected, high_tail)
    assert high_tail_summary["rmse_K"] < RMSE_TOLERANCE_K
    assert (
        high_tail_summary["count_gt_0p1_K"] / high_tail_summary["point_count"]
        > OUTLIER_FRACTION_TOLERANCE
    )
    assert not _passes(expected, high_tail)

    assert not _passes(expected, exact, parameter_error=1e-12)
    assert not _passes(expected, exact, npz_error=1e-12)

    print(
        "checkpoint reload tolerance regression passed: "
        "exact serialization, sparse-safe GPU replay, broad-drift rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
