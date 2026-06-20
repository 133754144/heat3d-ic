"""Legacy Heat3D v1 normalization helpers.

The functions in this module preserve the existing V4 `legacy_zscore` behavior:
train-only coordinate min/max scaling, per-feature condition z-score,
normalized DeltaT target, and raw DeltaT/temperature recovery.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rigno.heat3d_v1_training_semantics import build_legacy_zero_delta_bridge


LEGACY_ZSCORE_EPS = 1.0e-8


def safe_stats(array: np.ndarray, eps: float = LEGACY_ZSCORE_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Return train mean and std with the legacy zero-variance guard."""

    mean = np.mean(array, axis=0, keepdims=True)
    std = np.std(array, axis=0, keepdims=True)
    return mean, np.where(std < eps, 1.0, std)


def legacy_train_only_stats(
    examples: list[Any],
    *,
    bridge_fn: Callable[[Any], Any] = build_legacy_zero_delta_bridge,
    eps: float = LEGACY_ZSCORE_EPS,
) -> dict[str, Any]:
    """Compute the current train-only legacy normalization statistics."""

    c_values = []
    delta_values = []
    coord_values = []
    feature_names = None
    for example in examples:
        bridge = bridge_fn(example)
        names = bridge.condition_feature_names
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Relative condition feature-name mismatch in train split")

        c_values.append(np.asarray(bridge.legacy_inputs.c).reshape(-1, len(names)))
        delta_values.append(np.asarray(bridge.target_delta_u).reshape(-1, 1))
        coord_values.append(np.asarray(bridge.legacy_inputs.x_inp).reshape(-1, 3))

    c_all = np.concatenate(c_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    c_mean, c_std = safe_stats(c_all, eps=eps)
    delta_mean, delta_std = safe_stats(delta_all, eps=eps)
    coord_min = np.min(coord_all, axis=0, keepdims=True)
    coord_max = np.max(coord_all, axis=0, keepdims=True)
    coord_span = np.where((coord_max - coord_min) < eps, 1.0, coord_max - coord_min)
    return {
        "feature_names": tuple(feature_names or ()),
        "condition_mean": c_mean.reshape(1, 1, 1, -1),
        "condition_std": c_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
    }


def normalize_coords(coords: Any, stats: dict[str, Any]) -> Any:
    """Map physical coordinates to the legacy train min/max unit box."""

    return 2.0 * ((coords - stats["coord_min"]) / stats["coord_span"]) - 1.0


def normalize_condition(raw_c: Any, stats: dict[str, Any]) -> Any:
    """Apply legacy per-feature z-score to condition channels."""

    return (raw_c - stats["condition_mean"]) / stats["condition_std"]


def recover_raw_condition(normalized_c: Any, stats: dict[str, Any]) -> Any:
    """Invert legacy condition z-score."""

    return normalized_c * stats["condition_std"] + stats["condition_mean"]


def normalize_target_delta(target_delta: Any, stats: dict[str, Any]) -> Any:
    """Apply legacy train DeltaT mean/std target normalization."""

    return (target_delta - stats["target_delta_mean"]) / stats["target_delta_std"]


def normalized_delta_to_raw(pred_normalized: Any, stats: dict[str, Any]) -> Any:
    """Recover raw DeltaT from normalized model output."""

    return pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]


def recover_temperature_from_normalized_delta(pred_normalized: Any, t_ref: Any, stats: dict[str, Any]) -> Any:
    """Recover raw temperature from normalized model output and T_ref."""

    return t_ref + normalized_delta_to_raw(pred_normalized, stats)
