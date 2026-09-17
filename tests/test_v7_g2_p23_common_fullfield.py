"""Unit tests for P23 metric definitions using a synthetic non-P1i fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p23_eval", ROOT / "scripts/evaluate_v7_g2_p23_common_fullfield.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_layout_conversion_zxy_to_xyz_flat():
    zxy = np.arange(57 * 65 * 65, dtype=np.float64).reshape(57, 65, 65)
    expected = np.transpose(zxy, (1, 2, 0)).reshape(-1)
    assert np.array_equal(MODULE.as_xyz_flat(zxy, "z_x_y"), expected)


def test_metrics_use_truth_defined_hotspot_and_finite_values():
    truth = np.arange(100, dtype=np.float64) + 1.0
    pred = truth.copy()
    pred[-1] += 2.0
    weights = np.ones_like(truth)
    metrics = MODULE.one_case_metrics(pred, truth, weights)
    assert metrics["sample_first_relative_rmse_pct"] > 0.0
    assert metrics["point_global_relative_rmse_pct"] > 0.0
    assert metrics["peak_temperature_absolute_error_K"] == 2.0
    # ceil(1% of 100) selects the single hottest truth node.
    assert metrics["true_hotspot_region_rmse_K"] == 2.0
