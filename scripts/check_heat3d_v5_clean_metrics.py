#!/usr/bin/env python3
"""Analytic fixture checks for the V5 clean-first metric contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_metrics import (  # noqa: E402
    REQUIRED_SUMMARY_FIELDS,
    control_volume_weights,
    decompose_shape_scale,
    evaluate_metric_suite,
    project_raw_dirichlet,
    reconstruct_shape_scale,
    validate_metric_suite,
)


def _grid_coords() -> np.ndarray:
    axes = (np.array([0.0, 1.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0]))
    return np.asarray([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]], dtype=np.float64)


def main() -> int:
    coords = _grid_coords()
    weights = control_volume_weights(coords)
    assert weights.shape == (8,)
    assert np.allclose(weights, np.ones(8) * 0.75)

    target = np.array([0.2, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3, 3.1], dtype=np.float64)
    prediction = target * 1.1
    q = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0], dtype=np.float64)
    target_scale, target_shape = decompose_shape_scale(target, weights)
    assert np.allclose(reconstruct_shape_scale(target_scale, target_shape), target, atol=1.0e-12)

    suite = evaluate_metric_suite(
        [
            {
                "sample_id": "fixture_a",
                "split": "valid_iid",
                "prediction_deltaT_K": prediction,
                "target_deltaT_K": target,
                "control_volumes_m3": weights,
                "q_W_m3": q,
                "prediction_normalized": prediction,
                "target_normalized": target,
            },
            {
                "sample_id": "fixture_b",
                "split": "valid_iid",
                "prediction_deltaT_K": target,
                "target_deltaT_K": target,
                "control_volumes_m3": weights,
                "q_W_m3": q,
                "prediction_normalized": target,
                "target_normalized": target,
            },
        ]
    )
    summary = suite["summary"]
    validate_metric_suite(summary)
    assert set(REQUIRED_SUMMARY_FIELDS).issubset(summary)
    assert 0.0 < summary["point_global_relative_rmse_pct"] < 100.0
    assert 0.0 < summary["sample_first_cv_relative_rmse_pct"] < 100.0
    assert 0.0 <= summary["low_deltaT_background_over_ratio"] <= 1.0
    assert summary["strong_q_sample_count"] == 2

    raw_temperature = target + 300.0
    projected = project_raw_dirichlet(
        raw_temperature,
        np.array([True] + [False] * 7),
        np.array([300.0] * 8),
    )
    assert projected[0] == 300.0
    assert np.allclose(projected[1:], raw_temperature[1:])

    print(
        json.dumps(
            {
                "status": "passed",
                "metric_fields": len(REQUIRED_SUMMARY_FIELDS),
                "point_global_relative_rmse_pct": summary["point_global_relative_rmse_pct"],
                "sample_first_cv_relative_rmse_pct": summary["sample_first_cv_relative_rmse_pct"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
