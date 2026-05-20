#!/usr/bin/env python3
"""Synthetic smoke checks for Heat3D v2 field-shape diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_field_shape_diagnostics import (  # noqa: E402
    aggregate_field_shape_metrics,
    compute_field_shape_metrics,
)


def _assert_close(actual: float | None, expected: float, *, name: str, atol: float = 1.0e-10) -> None:
    if actual is None:
        raise AssertionError(f"{name}: expected {expected}, got None")
    if abs(float(actual) - expected) > atol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> int:
    true = np.array([0.0, 0.2, 0.8, 1.5, 3.0, 5.0, 8.0, 13.0], dtype=np.float64)
    perfect = compute_field_shape_metrics(true, true.copy(), top_k=5, sample_id="perfect", split="valid")
    _assert_close(perfect["field_variance_ratio"], 1.0, name="perfect field_variance_ratio")
    _assert_close(perfect["field_std_ratio"], 1.0, name="perfect field_std_ratio")
    _assert_close(perfect["centered_spatial_correlation"], 1.0, name="perfect centered correlation")
    _assert_close(perfect["uncentered_cosine_similarity"], 1.0, name="perfect cosine")
    _assert_close(perfect["amplitude_ratio"], 1.0, name="perfect amplitude_ratio")
    _assert_close(perfect["peak_abs_error"], 0.0, name="perfect peak_abs_error")
    _assert_close(perfect["top_k_overlap"], 1.0, name="perfect top_k_overlap")

    true_mean = float(np.mean(true))
    smoothed_values = true_mean + 0.5 * (true - true_mean)
    smoothed = compute_field_shape_metrics(true, smoothed_values, top_k=5, sample_id="smoothed", split="valid")
    _assert_close(smoothed["field_variance_ratio"], 0.25, name="smoothed field_variance_ratio")
    _assert_close(smoothed["field_std_ratio"], 0.5, name="smoothed field_std_ratio")
    _assert_close(smoothed["centered_spatial_correlation"], 1.0, name="smoothed centered correlation")
    _assert_close(smoothed["amplitude_ratio"], 0.5, name="smoothed amplitude_ratio")
    if not (smoothed["field_variance_ratio"] < perfect["field_variance_ratio"]):
        raise AssertionError("smoothed prediction should lower variance ratio")

    constant_values = np.full_like(true, true_mean)
    constant = compute_field_shape_metrics(true, constant_values, top_k=5, sample_id="constant", split="valid")
    _assert_close(constant["field_variance_ratio"], 0.0, name="constant field_variance_ratio")
    _assert_close(constant["field_std_ratio"], 0.0, name="constant field_std_ratio")
    _assert_close(constant["amplitude_ratio"], 0.0, name="constant amplitude_ratio")
    if constant["centered_spatial_correlation"] is not None:
        raise AssertionError("constant prediction should have null centered correlation")
    if not constant["warnings"]:
        raise AssertionError("constant prediction should record denominator warning")
    if not (constant["field_variance_ratio"] < smoothed["field_variance_ratio"]):
        raise AssertionError("constant prediction should lower variance ratio below smoothed")

    aggregate = aggregate_field_shape_metrics([perfect, smoothed, constant])
    if aggregate["valid_sample_count"] != 3:
        raise AssertionError("aggregate valid_sample_count should be 3")
    if aggregate["field_variance_ratio"] is None:
        raise AssertionError("aggregate field_variance_ratio should be present")

    print("Heat3D v2 field-shape diagnostics smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
