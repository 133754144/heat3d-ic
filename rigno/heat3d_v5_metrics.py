"""Clean-first V5 Heat3D field, shape, and scale evaluation metrics.

The V5 metric contract deliberately distinguishes two relative-error views:

* point-global relative RMSE preserves the V4 reporting denominator: the
  point-global raw-DeltaT RMSE divided by the point-global mean absolute true
  DeltaT;
* sample-first CV-relative RMSE first evaluates every sample with its own
  control-volume (CV) weights, then takes an unweighted sample mean.

All functions in this module are read-only NumPy utilities.  They accept
prediction/target fields only at evaluation time; no routine here constructs a
model input or exposes target-derived quantities as inference features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


EPS = 1.0e-12
METRIC_SCHEMA_VERSION = "heat3d_v5_clean_metrics_v1"
BACKGROUND_QUANTILE = 0.50
HOTSPOT_FRACTION = 0.05
TOP_K = 5
STRONG_Q_QUANTILE = 0.90

REQUIRED_SUMMARY_FIELDS = (
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


class V5MetricError(ValueError):
    """Raised for malformed V5 metric inputs."""


def control_volume_weights(coords: np.ndarray) -> np.ndarray:
    """Infer rectilinear control-volume weights for a ``[N, 3]`` coordinate set.

    The P5 subset is a complete rectilinear grid.  Explicitly requiring that
    condition catches accidental point ordering or incomplete-grid errors
    instead of silently assigning invalid weights.
    """

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise V5MetricError(f"coords must have shape [N, 3], got {points.shape}")
    if points.shape[0] < 8 or not np.all(np.isfinite(points)):
        raise V5MetricError("coords must be finite and contain a 3D grid")

    axes = [np.unique(points[:, axis]) for axis in range(3)]
    if any(axis.size < 2 for axis in axes):
        raise V5MetricError("each coordinate axis needs at least two values")
    expected_count = int(np.prod([axis.size for axis in axes]))
    if expected_count != points.shape[0]:
        raise V5MetricError(
            "coords must be a complete rectilinear product grid: "
            f"expected {expected_count} points, found {points.shape[0]}"
        )

    widths = [_control_widths(axis, label=f"axis_{index}") for index, axis in enumerate(axes)]
    indices = [np.searchsorted(axis, points[:, index]) for index, axis in enumerate(axes)]
    weights = widths[0][indices[0]] * widths[1][indices[1]] * widths[2][indices[2]]
    _require_positive_finite(weights, "control-volume weights")
    return weights


def decompose_shape_scale(
    delta_t_K: np.ndarray,
    control_volumes_m3: np.ndarray,
    *,
    eps: float = EPS,
) -> tuple[float, np.ndarray]:
    """Return ``s=CV-RMS(deltaT)`` and ``phi=deltaT/s``.

    V5 uses this exact native convention.  A non-positive/near-zero target
    scale cannot define a stable logarithmic scale target and is rejected.
    """

    delta = _flat_finite(delta_t_K, "delta_t_K")
    weights = _flat_weights(control_volumes_m3, delta.size)
    scale = _weighted_rms(delta, weights)
    if scale <= eps:
        raise V5MetricError(f"CV-RMS scale must exceed {eps:g}, found {scale:g}")
    return scale, delta / scale


def reconstruct_shape_scale(scale_K: float, shape: np.ndarray) -> np.ndarray:
    """Reconstruct DeltaT from a positive V5 scale and unit-CV-RMS shape."""

    if not math.isfinite(float(scale_K)) or float(scale_K) <= 0.0:
        raise V5MetricError(f"scale_K must be finite and positive, got {scale_K!r}")
    return float(scale_K) * _flat_finite(shape, "shape")


def project_raw_dirichlet(
    raw_temperature_K: np.ndarray,
    dirichlet_mask: np.ndarray,
    prescribed_temperature_K: np.ndarray,
) -> np.ndarray:
    """Project raw temperature values at Dirichlet nodes without altering others."""

    temperature = _flat_finite(raw_temperature_K, "raw_temperature_K").copy()
    mask = np.asarray(dirichlet_mask, dtype=bool).reshape(-1)
    prescribed = _flat_finite(prescribed_temperature_K, "prescribed_temperature_K")
    if mask.shape != temperature.shape or prescribed.shape != temperature.shape:
        raise V5MetricError("Dirichlet mask and prescribed temperatures must match field shape")
    temperature[mask] = prescribed[mask]
    return temperature


def compute_sample_metrics(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Compute all contract metrics for one raw-DeltaT sample.

    Required sample fields are ``sample_id``, ``prediction_deltaT_K``,
    ``target_deltaT_K``, and ``control_volumes_m3``.  ``q_W_m3`` is required
    for the strong-q metric.  Legacy normalized fields are optional for
    prediction archives that cannot preserve them, but V5 training reports
    must provide them so the aggregate legacy metric is not null.
    """

    sample_id = str(sample.get("sample_id") or "")
    if not sample_id:
        raise V5MetricError("sample_id is required")
    prediction = _flat_finite(sample.get("prediction_deltaT_K"), f"{sample_id}.prediction_deltaT_K")
    target = _flat_finite(sample.get("target_deltaT_K"), f"{sample_id}.target_deltaT_K")
    if prediction.shape != target.shape:
        raise V5MetricError(f"{sample_id}: prediction and target shapes differ")
    weights = _flat_weights(sample.get("control_volumes_m3"), target.size)
    error = prediction - target

    true_scale, true_shape = decompose_shape_scale(target, weights)
    pred_scale, pred_shape = decompose_shape_scale(prediction, weights)
    background_mask = target <= float(np.quantile(target, BACKGROUND_QUANTILE))
    hotspot_mask = _top_fraction_mask(target, HOTSPOT_FRACTION)
    top5_mask = _top_k_mask(target, TOP_K)
    strong_q_mask = _strong_q_mask(sample.get("q_W_m3"), target.size)

    legacy_base_mse = _legacy_base_mse(sample, target.size)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": sample.get("split"),
        "point_count": int(target.size),
        "cv_volume_m3": float(np.sum(weights)),
        "point_error_squared_sum": float(np.sum(np.square(error))),
        "point_true_abs_sum": float(np.sum(np.abs(target))),
        "point_count_for_global": int(target.size),
        "raw_cv_error_squared_integral_K2_m3": float(np.sum(np.square(error) * weights)),
        "raw_cv_volume_m3": float(np.sum(weights)),
        "sample_cv_relative_rmse": _weighted_rms(error, weights) / true_scale,
        "raw_cv_weighted_rmse_K": _weighted_rms(error, weights),
        "amplitude_ratio": pred_scale / true_scale,
        "spatial_correlation": _weighted_centered_correlation(target, prediction, weights),
        "hotspot_cv_weighted_rmse_K": _masked_weighted_rmse(error, weights, hotspot_mask),
        "top5_cv_weighted_rmse_K": _masked_weighted_rmse(error, weights, top5_mask),
        "strong_q_cv_weighted_rmse_K": (
            _masked_weighted_rmse(error, weights, strong_q_mask)
            if strong_q_mask is not None
            else None
        ),
        "strong_q_active": bool(strong_q_mask is not None),
        "low_deltaT_background_bias_K": _masked_weighted_mean(error, weights, background_mask),
        "low_deltaT_background_rmse_K": _masked_weighted_rmse(error, weights, background_mask),
        "low_deltaT_background_over_ratio": _masked_weighted_mean(
            (error > 0.0).astype(np.float64), weights, background_mask
        ),
        "shape_cv_rmse": _weighted_rms(pred_shape - true_shape, weights),
        "scale_log_error": float(math.log(pred_scale / true_scale)),
        "scale_log_squared_error": float(math.log(pred_scale / true_scale) ** 2),
        "true_scale_cv_rms_K": float(true_scale),
        "pred_scale_cv_rms_K": float(pred_scale),
        "legacy_normalized_base_mse": legacy_base_mse,
    }
    return row


def summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample metric rows according to the V5 clean contract."""

    if not rows:
        raise V5MetricError("at least one metric row is required")
    point_error_squared_sum = _sum_finite(rows, "point_error_squared_sum")
    point_true_abs_sum = _sum_finite(rows, "point_true_abs_sum")
    point_count = _sum_finite(rows, "point_count_for_global")
    cv_error_squared_integral = _sum_finite(rows, "raw_cv_error_squared_integral_K2_m3")
    cv_volume = _sum_finite(rows, "raw_cv_volume_m3")
    if point_count <= 0.0 or point_true_abs_sum <= EPS or cv_volume <= EPS:
        raise V5MetricError("invalid aggregate metric denominator")

    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "point_count": int(round(point_count)),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "aggregation": {
            "point_global_relative_rmse_pct": (
                "sqrt(sum_point_error_squared / N_points) / "
                "(sum_abs_true_deltaT / N_points) * 100"
            ),
            "sample_first_cv_relative_rmse_pct": (
                "mean_samples(CV_RMS(pred-target) / CV_RMS(target)) * 100"
            ),
            "raw_cv_weighted_rmse_K": "sqrt(sum(error^2 * CV) / sum(CV))",
            "amplitude_ratio": "mean_samples(CV_RMS(pred) / CV_RMS(target))",
            "spatial_correlation": "mean_samples(CV-weighted centered correlation)",
        },
        "point_global_relative_rmse_pct": float(
            math.sqrt(point_error_squared_sum / point_count) / (point_true_abs_sum / point_count) * 100.0
        ),
        "sample_first_cv_relative_rmse_pct": _mean_finite(rows, "sample_cv_relative_rmse", required=True) * 100.0,
        "raw_cv_weighted_rmse_K": float(math.sqrt(cv_error_squared_integral / cv_volume)),
        "amplitude_ratio": _mean_finite(rows, "amplitude_ratio", required=True),
        "spatial_correlation": _mean_finite(rows, "spatial_correlation", required=True),
        "hotspot_cv_weighted_rmse_K": _mean_finite(rows, "hotspot_cv_weighted_rmse_K", required=True),
        "top5_cv_weighted_rmse_K": _mean_finite(rows, "top5_cv_weighted_rmse_K", required=True),
        "strong_q_cv_weighted_rmse_K": _mean_finite(rows, "strong_q_cv_weighted_rmse_K", required=False),
        "low_deltaT_background_bias_K": _mean_finite(rows, "low_deltaT_background_bias_K", required=True),
        "low_deltaT_background_rmse_K": _mean_finite(rows, "low_deltaT_background_rmse_K", required=True),
        "low_deltaT_background_over_ratio": _mean_finite(rows, "low_deltaT_background_over_ratio", required=True),
        "shape_cv_rmse": _mean_finite(rows, "shape_cv_rmse", required=True),
        "scale_log_rmse": float(math.sqrt(_mean_finite(rows, "scale_log_squared_error", required=True))),
        "legacy_normalized_valid_base_mse": _mean_finite(
            rows, "legacy_normalized_base_mse", required=False
        ),
        "strong_q_sample_count": int(sum(bool(row.get("strong_q_active")) for row in rows)),
    }
    return summary


def evaluate_metric_suite(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the full V5 clean metric suite and validate required outputs."""

    rows = [compute_sample_metrics(sample) for sample in samples]
    summary = summarize_metric_rows(rows)
    validate_metric_suite(summary, require_legacy=True)
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "summary": summary,
        "per_sample": rows,
    }


def validate_metric_suite(summary: Mapping[str, Any], *, require_legacy: bool = True) -> None:
    """Ensure a V5 report contains finite, contract-complete summary fields."""

    missing = []
    non_finite = []
    for field in REQUIRED_SUMMARY_FIELDS:
        value = summary.get(field)
        if value is None:
            if field == "legacy_normalized_valid_base_mse" and not require_legacy:
                continue
            missing.append(field)
            continue
        if not _is_finite_number(value):
            non_finite.append(field)
    if missing or non_finite:
        raise V5MetricError(
            "V5 metric suite incomplete: "
            f"missing={missing} non_finite={non_finite}"
        )
    if float(summary["point_global_relative_rmse_pct"]) < 0.0:
        raise V5MetricError("point_global_relative_rmse_pct must be non-negative")
    if not 0.0 <= float(summary["low_deltaT_background_over_ratio"]) <= 1.0:
        raise V5MetricError("low_deltaT_background_over_ratio must be in [0, 1]")


def _control_widths(axis: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise V5MetricError(f"{label} must be finite, strictly increasing, and have >=2 entries")
    widths = np.empty_like(values)
    widths[0] = 0.5 * (values[1] - values[0])
    widths[-1] = 0.5 * (values[-1] - values[-2])
    if values.size > 2:
        widths[1:-1] = 0.5 * (values[2:] - values[:-2])
    _require_positive_finite(widths, f"{label} control widths")
    return widths


def _flat_finite(value: Any, label: str) -> np.ndarray:
    if value is None:
        raise V5MetricError(f"{label} is required")
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise V5MetricError(f"{label} must be a non-empty finite array")
    return values


def _flat_weights(value: Any, expected_size: int) -> np.ndarray:
    weights = _flat_finite(value, "control_volumes_m3")
    if weights.size != expected_size:
        raise V5MetricError(
            f"control_volumes_m3 count {weights.size} does not match field count {expected_size}"
        )
    _require_positive_finite(weights, "control_volumes_m3")
    return weights


def _require_positive_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise V5MetricError(f"{label} must be finite and positive")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= EPS:
        raise V5MetricError("control-volume total must be positive")
    return float(np.sum(values * weights) / total_weight)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(math.sqrt(_weighted_mean(np.square(values), weights)))


def _weighted_centered_correlation(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    target_centered = target - _weighted_mean(target, weights)
    prediction_centered = prediction - _weighted_mean(prediction, weights)
    denominator = _weighted_rms(target_centered, weights) * _weighted_rms(prediction_centered, weights)
    if denominator <= EPS:
        # A spatially constant target/prediction cannot define correlation.  The
        # P5 fields are non-constant, so surfacing this is safer than inventing 1.
        raise V5MetricError("spatial correlation is undefined for a constant field")
    return _weighted_mean(target_centered * prediction_centered, weights) / denominator


def _top_k_mask(values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k < 1:
        raise V5MetricError("top_k must be positive")
    count = min(int(top_k), int(values.size))
    order = np.argsort(values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[-count:]] = True
    return mask


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise V5MetricError("top fraction must be in (0, 1]")
    return _top_k_mask(values, max(1, int(math.ceil(values.size * fraction))))


def _strong_q_mask(q_value: Any, expected_size: int) -> np.ndarray | None:
    if q_value is None:
        return None
    q = _flat_finite(q_value, "q_W_m3")
    if q.size != expected_size:
        raise V5MetricError("q_W_m3 count does not match field count")
    positive = q > 0.0
    if not np.any(positive):
        return None
    threshold = float(np.quantile(q[positive], STRONG_Q_QUANTILE))
    mask = np.logical_and(positive, q >= threshold)
    return mask if np.any(mask) else None


def _masked_weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool).reshape(-1)
    if active.shape != values.shape or not np.any(active):
        raise V5MetricError("metric mask must match fields and select at least one node")
    return _weighted_mean(values[active], weights[active])


def _masked_weighted_rmse(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return _weighted_rms(values[np.asarray(mask, dtype=bool)], weights[np.asarray(mask, dtype=bool)])


def _legacy_base_mse(sample: Mapping[str, Any], expected_size: int) -> float | None:
    prediction = sample.get("prediction_normalized")
    target = sample.get("target_normalized")
    if prediction is None and target is None:
        return None
    if prediction is None or target is None:
        raise V5MetricError("legacy normalized prediction and target must be supplied together")
    pred_values = _flat_finite(prediction, "prediction_normalized")
    target_values = _flat_finite(target, "target_normalized")
    if pred_values.size != expected_size or target_values.size != expected_size:
        raise V5MetricError("legacy normalized fields must match DeltaT point count")
    return float(np.mean(np.square(pred_values - target_values)))


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_finite(rows: Sequence[Mapping[str, Any]], field: str, *, required: bool) -> float | None:
    values = [float(row[field]) for row in rows if _is_finite_number(row.get(field))]
    if not values:
        if required:
            raise V5MetricError(f"no finite per-sample values for {field}")
        return None
    return float(np.mean(values))


def _sum_finite(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if _is_finite_number(row.get(field))]
    if len(values) != len(rows):
        raise V5MetricError(f"missing/non-finite per-sample field {field}")
    return float(np.sum(values))
