"""Read-only field-shape diagnostics for Heat3D v2 prediction outputs.

This module computes diagnostics on already-generated DeltaT fields. It does
not import training code, model code, JAX, Flax, or Optax.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np


EPS = 1.0e-12
METRIC_FIELDS = (
    "true_mean",
    "pred_mean",
    "error_mean",
    "true_std",
    "pred_std",
    "field_variance_ratio",
    "field_std_ratio",
    "centered_spatial_correlation",
    "uncentered_cosine_similarity",
    "amplitude_ratio",
    "p95_error",
    "p99_error",
    "p95_amplitude_ratio",
    "p99_amplitude_ratio",
    "peak_true",
    "peak_pred",
    "peak_abs_error",
    "top_k_overlap",
)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _flatten_field(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    metric_name: str,
    warnings: list[str],
    eps: float,
) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        warnings.append(f"{metric_name}: non-finite numerator or denominator")
        return None
    if abs(denominator) <= eps:
        warnings.append(f"{metric_name}: denominator near zero")
        return None
    return _json_float(numerator / denominator)


def _centered_correlation(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    warnings: list[str],
    eps: float,
) -> float | None:
    true_centered = true_values - float(np.mean(true_values))
    pred_centered = pred_values - float(np.mean(pred_values))
    denominator = float(np.linalg.norm(true_centered) * np.linalg.norm(pred_centered))
    if denominator <= eps:
        warnings.append("centered_spatial_correlation: denominator near zero")
        return None
    return _json_float(float(np.dot(true_centered, pred_centered)) / denominator)


def _uncentered_cosine(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    warnings: list[str],
    eps: float,
) -> float | None:
    denominator = float(np.linalg.norm(true_values) * np.linalg.norm(pred_values))
    if denominator <= eps:
        warnings.append("uncentered_cosine_similarity: denominator near zero")
        return None
    return _json_float(float(np.dot(true_values, pred_values)) / denominator)


def _top_k_overlap(true_values: np.ndarray, pred_values: np.ndarray, top_k: int) -> float:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    k = min(int(top_k), int(true_values.size))
    true_top = set(np.argpartition(true_values, -k)[-k:].tolist())
    pred_top = set(np.argpartition(pred_values, -k)[-k:].tolist())
    return float(len(true_top & pred_top) / k)


def compute_field_shape_metrics(
    true_delta_t: np.ndarray,
    pred_delta_t: np.ndarray,
    *,
    top_k: int = 5,
    sample_id: str | None = None,
    split: str | None = None,
    eps: float = EPS,
) -> dict[str, Any]:
    """Compute per-sample field-shape metrics on flattened DeltaT fields."""

    true_values = _flatten_field(true_delta_t, "true_delta_t")
    pred_values = _flatten_field(pred_delta_t, "pred_delta_t")
    if true_values.shape != pred_values.shape:
        raise ValueError(
            "true_delta_t and pred_delta_t must have the same flattened shape, "
            f"found {true_values.shape} and {pred_values.shape}"
        )

    warnings: list[str] = []
    error = pred_values - true_values
    abs_error = np.abs(error)
    true_var = float(np.var(true_values))
    pred_var = float(np.var(pred_values))
    true_std = float(np.std(true_values))
    pred_std = float(np.std(pred_values))
    true_range = float(np.max(true_values) - np.min(true_values))
    pred_range = float(np.max(pred_values) - np.min(pred_values))
    true_p95 = float(np.percentile(true_values, 95))
    pred_p95 = float(np.percentile(pred_values, 95))
    true_p99 = float(np.percentile(true_values, 99))
    pred_p99 = float(np.percentile(pred_values, 99))
    peak_true = float(np.max(true_values))
    peak_pred = float(np.max(pred_values))

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "point_count": int(true_values.size),
        "top_k": int(top_k),
        "true_mean": _json_float(float(np.mean(true_values))),
        "pred_mean": _json_float(float(np.mean(pred_values))),
        "error_mean": _json_float(float(np.mean(error))),
        "true_std": _json_float(true_std),
        "pred_std": _json_float(pred_std),
        "field_variance_ratio": _safe_ratio(
            pred_var,
            true_var,
            metric_name="field_variance_ratio",
            warnings=warnings,
            eps=eps,
        ),
        "field_std_ratio": _safe_ratio(
            pred_std,
            true_std,
            metric_name="field_std_ratio",
            warnings=warnings,
            eps=eps,
        ),
        "centered_spatial_correlation": _centered_correlation(
            true_values,
            pred_values,
            warnings=warnings,
            eps=eps,
        ),
        "uncentered_cosine_similarity": _uncentered_cosine(
            true_values,
            pred_values,
            warnings=warnings,
            eps=eps,
        ),
        "amplitude_ratio": _safe_ratio(
            pred_range,
            true_range,
            metric_name="amplitude_ratio",
            warnings=warnings,
            eps=eps,
        ),
        "p95_error": _json_float(float(np.percentile(abs_error, 95))),
        "p99_error": _json_float(float(np.percentile(abs_error, 99))),
        "p95_amplitude_ratio": _safe_ratio(
            pred_p95,
            true_p95,
            metric_name="p95_amplitude_ratio",
            warnings=warnings,
            eps=eps,
        ),
        "p99_amplitude_ratio": _safe_ratio(
            pred_p99,
            true_p99,
            metric_name="p99_amplitude_ratio",
            warnings=warnings,
            eps=eps,
        ),
        "peak_true": _json_float(peak_true),
        "peak_pred": _json_float(peak_pred),
        "peak_abs_error": _json_float(abs(peak_pred - peak_true)),
        "top_k_overlap": _json_float(_top_k_overlap(true_values, pred_values, top_k)),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return result


def _mean_metric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return _json_float(float(np.mean(values)))


def aggregate_field_shape_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample field-shape metrics by sample mean."""

    valid_rows = [row for row in rows if not row.get("failed")]
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "failed_sample_count": len(rows) - len(valid_rows),
        "point_count": int(sum(int(row.get("point_count") or 0) for row in valid_rows)),
        "warning_count": int(sum(int(row.get("warning_count") or 0) for row in valid_rows)),
        "aggregation": "sample_mean_ignoring_null_metrics",
    }
    for field in METRIC_FIELDS:
        result[field] = _mean_metric(valid_rows, field)
    return result


def splitwise_field_shape_summary(
    rows: list[dict[str, Any]],
    *,
    split_field: str = "split",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(split_field) or "unknown")].append(row)
    summary = []
    for split, split_rows in sorted(grouped.items()):
        item = {"split": split}
        item.update(aggregate_field_shape_metrics(split_rows))
        summary.append(item)
    return summary


def build_field_shape_report(
    rows: list[dict[str, Any]],
    *,
    warnings: list[str] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an aggregate report from per-sample rows."""

    return {
        "overall": aggregate_field_shape_metrics(rows),
        "split_summary": splitwise_field_shape_summary(rows),
        "per_sample": rows,
        "warnings": warnings or [],
        "failures": failures or [],
    }
