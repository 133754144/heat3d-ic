"""Pure helpers for the V5 frozen decoder-bypass audit.

This module classifies the actual within-sample variation of each bypass input
and compares metric summaries with and without a frozen bypass residual.  It
does not import JAX, models, checkpoints, or dataset loaders, so its fixture
can verify the audit semantics without a remote GPU/runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_v5_metrics import summarize_metric_rows


LOWER_IS_BETTER_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_rmse_K",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
)


class BypassAuditError(ValueError):
    """Raised for malformed frozen-bypass audit data."""


def classify_feature_node_variation(
    feature_names: Sequence[str],
    per_sample_features: Sequence[np.ndarray],
    *,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-9,
) -> list[dict[str, Any]]:
    """Classify each condition feature from actual sample-node values.

    ``sample_global_broadcast`` means no sampled clean scene has within-sample
    variation. ``node_varying`` means every sampled clean scene varies across
    nodes. ``mixed_node_variation`` means it is local for some scenes and
    constant for others; it must still be retained as a local-capable channel.
    """

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise BypassAuditError("feature_names must not be empty")
    if not per_sample_features:
        raise BypassAuditError("at least one sample feature matrix is required")
    arrays = []
    for index, value in enumerate(per_sample_features):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(names) or array.shape[0] < 1:
            raise BypassAuditError(
                f"sample feature matrix {index} must have shape [nodes, {len(names)}], got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise BypassAuditError(f"sample feature matrix {index} contains non-finite values")
        arrays.append(array)

    result: list[dict[str, Any]] = []
    for feature_index, name in enumerate(names):
        ranges = np.asarray([np.ptp(array[:, feature_index]) for array in arrays], dtype=np.float64)
        max_abs = np.asarray(
            [np.max(np.abs(array[:, feature_index])) for array in arrays], dtype=np.float64
        )
        thresholds = absolute_tolerance + relative_tolerance * np.maximum(max_abs, 1.0)
        variable = ranges > thresholds
        varying_count = int(np.sum(variable))
        invariant_count = len(arrays) - varying_count
        if varying_count == 0:
            classification = "sample_global_broadcast"
        elif invariant_count == 0:
            classification = "node_varying"
        else:
            classification = "mixed_node_variation"
        result.append(
            {
                "feature_index": int(feature_index),
                "feature_name": name,
                "classification": classification,
                "node_varying_sample_count": varying_count,
                "node_invariant_sample_count": invariant_count,
                "sample_count": len(arrays),
                "max_within_sample_range": float(np.max(ranges)),
                "median_within_sample_range": float(np.median(ranges)),
                "max_within_sample_tolerance": float(np.max(thresholds)),
                "retain_as_local_bypass_input": bool(varying_count > 0),
                "duplicate_of_sample_global_context": bool(varying_count == 0),
            }
        )
    return result


def compare_bypass_metric_rows(
    full_bypass_rows: Sequence[Mapping[str, Any]],
    without_bypass_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize frozen full-bypass versus residual-disabled predictions.

    Positive reductions mean the full bypass improves a lower-is-better metric.
    Amplitude is reported as a reduction in absolute deviation from the ideal
    ratio one; spatial correlation is reported as a direct full-minus-disabled
    gain.
    """

    full = summarize_metric_rows(full_bypass_rows)
    without = summarize_metric_rows(without_bypass_rows)
    reductions: dict[str, float | None] = {}
    for metric in LOWER_IS_BETTER_METRICS:
        full_value = full.get(metric)
        without_value = without.get(metric)
        if _finite(full_value) and _finite(without_value):
            reductions[metric] = float(without_value) - float(full_value)
        else:
            reductions[metric] = None
    full_corr = full.get("spatial_correlation")
    without_corr = without.get("spatial_correlation")
    full_amp = full.get("amplitude_ratio")
    without_amp = without.get("amplitude_ratio")
    return {
        "with_full_bypass": full,
        "without_bypass": without,
        "bypass_error_reduction_positive_is_better": reductions,
        "bypass_spatial_correlation_gain_positive_is_better": (
            float(full_corr) - float(without_corr)
            if _finite(full_corr) and _finite(without_corr)
            else None
        ),
        "bypass_amplitude_abs_error_reduction_positive_is_better": (
            abs(float(without_amp) - 1.0) - abs(float(full_amp) - 1.0)
            if _finite(full_amp) and _finite(without_amp)
            else None
        ),
    }


def bypass_structure_recommendation(feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the architecture branch required by observed channel variation."""

    local = [str(row["feature_name"]) for row in feature_rows if bool(row["retain_as_local_bypass_input"])]
    global_broadcast = [
        str(row["feature_name"])
        for row in feature_rows
        if bool(row["duplicate_of_sample_global_context"])
    ]
    if local:
        decision = "retain_local_bypass_and_remove_global_broadcast_duplicates"
        rationale = (
            "At least one frozen full_condition channel is node-varying in clean scenes; "
            "the V5 local bypass must remain a separately switchable module."
        )
    else:
        decision = "disable_existing_bypass_and_use_global_film_only"
        rationale = (
            "Every frozen full_condition channel is sample-global broadcast, so the existing "
            "post-decoder bypass duplicates Global FiLM context rather than carrying local signal."
        )
    return {
        "decision": decision,
        "rationale": rationale,
        "local_bypass_feature_names": local,
        "global_broadcast_feature_names": global_broadcast,
        "all_effective_bypass_channels_node_invariant": not bool(local),
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
