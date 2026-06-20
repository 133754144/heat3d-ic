"""Legacy Heat3D v1 normalization helpers.

The functions in this module preserve the existing V4 `legacy_zscore` behavior:
train-only coordinate min/max scaling, per-feature condition z-score,
normalized DeltaT target, and raw DeltaT/temperature recovery.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v1_training_semantics import (
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
    NORMALIZATION_PROFILES,
    build_legacy_zero_delta_bridge,
)


LEGACY_ZSCORE_EPS = 1.0e-8
SEMANTIC_LOG_EPS = 1.0e-12
BC_FLAG_FEATURES = ("is_top", "is_bottom", "is_side", "is_interior")
K_FEATURES = ("k_x", "k_y", "k_z")
Q_FEATURES = ("q",)
TOP_H_FEATURES = ("top_h",)
RELATIVE_BC_TEMPERATURE_FEATURES = (
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
)
TRANSFORM_LINEAR_ZSCORE = "linear_zscore"
TRANSFORM_LOG_K_ZSCORE = "log_k_zscore"
TRANSFORM_SIGNED_LOG1P_Q_ZSCORE = "signed_log1p_q_zscore"
TRANSFORM_BINARY_PASSTHROUGH = "binary_passthrough"
TRANSFORM_TOP_H_ZSCORE = "top_h_independent_zscore"
TRANSFORM_RELATIVE_BC_TEMPERATURE_ZSCORE = "relative_bc_temperature_zscore"
CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE = "legacy_zscore_all_condition_features"
CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL = (
    "semantic_v1_logk_signedlog1p_q_binary_bcflags_independent_bc_scalars"
)
CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY = (
    "semantic_v1_bc_flags_binary_passthrough_only"
)
CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY = "semantic_v1_q_signedlog1p_only"
CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY = "semantic_v1_k_log_only"
CONDITION_FEATURE_TRANSFORMS = (
    CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
)
SEMANTIC_CONDITION_FEATURE_TRANSFORMS = (
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
)


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
        "normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
        "condition_feature_transform": CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE,
        "feature_names": tuple(feature_names or ()),
        "condition_feature_transforms": tuple(
            TRANSFORM_LINEAR_ZSCORE for _ in tuple(feature_names or ())
        ),
        "condition_mean": c_mean.reshape(1, 1, 1, -1),
        "condition_std": c_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
    }


def semantic_normalization_v1_train_only_stats(
    examples: list[Any],
    *,
    condition_feature_transform: str = CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    bridge_fn: Callable[[Any], Any] = build_legacy_zero_delta_bridge,
    eps: float = LEGACY_ZSCORE_EPS,
) -> dict[str, Any]:
    """Compute train-only stats for the opt-in semantic normalization profile."""

    _check_condition_feature_transform(
        condition_feature_transform,
        normalization_profile=NORMALIZATION_PROFILE_SEMANTIC_V1,
    )
    c_values = []
    transformed_values = []
    delta_values = []
    coord_values = []
    sample_extents = []
    feature_names = None
    transforms = None
    for example in examples:
        bridge = bridge_fn(example)
        names = bridge.condition_feature_names
        if feature_names is None:
            feature_names = names
            transforms = tuple(
                _semantic_transform_for_feature(name, condition_feature_transform)
                for name in names
            )
        elif feature_names != names:
            raise ValueError("Relative condition feature-name mismatch in train split")

        raw_c = np.asarray(bridge.legacy_inputs.c, dtype=np.float64).reshape(-1, len(names))
        c_values.append(raw_c)
        transformed_values.append(_semantic_transform_condition_np(raw_c, tuple(transforms)))
        delta_values.append(np.asarray(bridge.target_delta_u).reshape(-1, 1))
        coords = np.asarray(bridge.legacy_inputs.x_inp).reshape(-1, 3)
        coord_values.append(coords)
        sample_extents.append(np.max(coords, axis=0) - np.min(coords, axis=0))

    transformed_all = np.concatenate(transformed_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    transformed_mean, transformed_std = _semantic_condition_stats(
        transformed_all, tuple(transforms or ()), eps=eps
    )
    delta_mean, delta_std = safe_stats(delta_all, eps=eps)
    coord_min = np.min(coord_all, axis=0, keepdims=True)
    coord_max = np.max(coord_all, axis=0, keepdims=True)
    coord_span = np.where((coord_max - coord_min) < eps, 1.0, coord_max - coord_min)
    extents = np.asarray(sample_extents, dtype=np.float64)
    aspect = np.max(extents, axis=1) / np.maximum(np.min(extents, axis=1), eps)
    return {
        "normalization_profile": NORMALIZATION_PROFILE_SEMANTIC_V1,
        "condition_feature_transform": condition_feature_transform,
        "feature_names": tuple(feature_names or ()),
        "condition_feature_transforms": tuple(transforms or ()),
        "condition_mean": transformed_mean.reshape(1, 1, 1, -1),
        "condition_std": transformed_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
        "physical_extent_min": np.min(extents, axis=0, keepdims=True).reshape(1, 1, 1, 3),
        "physical_extent_max": np.max(extents, axis=0, keepdims=True).reshape(1, 1, 1, 3),
        "physical_extent_mean": np.mean(extents, axis=0, keepdims=True).reshape(1, 1, 1, 3),
        "aspect_ratio_min": float(np.min(aspect)),
        "aspect_ratio_max": float(np.max(aspect)),
        "aspect_ratio_mean": float(np.mean(aspect)),
    }


def training_normalization_stats(
    examples: list[Any],
    *,
    normalization_profile: str = NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    condition_feature_transform: str | None = None,
    bridge_fn: Callable[[Any], Any] = build_legacy_zero_delta_bridge,
    eps: float = LEGACY_ZSCORE_EPS,
) -> dict[str, Any]:
    """Compute train-only stats for a supported Heat3D normalization profile."""

    _check_normalization_profile(normalization_profile)
    transform = _default_condition_feature_transform(
        normalization_profile, condition_feature_transform
    )
    if normalization_profile == NORMALIZATION_PROFILE_LEGACY_ZSCORE:
        _check_condition_feature_transform(
            transform, normalization_profile=normalization_profile
        )
        return legacy_train_only_stats(examples, bridge_fn=bridge_fn, eps=eps)
    return semantic_normalization_v1_train_only_stats(
        examples,
        condition_feature_transform=transform,
        bridge_fn=bridge_fn,
        eps=eps,
    )


def normalize_coords(coords: Any, stats: dict[str, Any]) -> Any:
    """Map physical coordinates to the legacy train min/max unit box."""

    return 2.0 * ((coords - stats["coord_min"]) / stats["coord_span"]) - 1.0


def normalize_condition(raw_c: Any, stats: dict[str, Any]) -> Any:
    """Apply legacy per-feature z-score to condition channels."""

    if _stats_profile(stats) == NORMALIZATION_PROFILE_SEMANTIC_V1:
        return _normalize_semantic_condition(raw_c, stats)
    return (raw_c - stats["condition_mean"]) / stats["condition_std"]


def recover_raw_condition(normalized_c: Any, stats: dict[str, Any]) -> Any:
    """Invert legacy condition z-score."""

    if _stats_profile(stats) == NORMALIZATION_PROFILE_SEMANTIC_V1:
        return _recover_semantic_condition(normalized_c, stats)
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


def _check_normalization_profile(normalization_profile: str) -> None:
    if normalization_profile not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"normalization_profile must be one of {NORMALIZATION_PROFILES}, "
            f"found {normalization_profile!r}"
        )


def _default_condition_feature_transform(
    normalization_profile: str, condition_feature_transform: str | None
) -> str:
    if condition_feature_transform:
        return condition_feature_transform
    if normalization_profile == NORMALIZATION_PROFILE_SEMANTIC_V1:
        return CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL
    return CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE


def _check_condition_feature_transform(
    condition_feature_transform: str, *, normalization_profile: str
) -> None:
    if condition_feature_transform not in CONDITION_FEATURE_TRANSFORMS:
        raise ValueError(
            "condition_feature_transform must be one of "
            f"{CONDITION_FEATURE_TRANSFORMS}, found {condition_feature_transform!r}"
        )
    if normalization_profile == NORMALIZATION_PROFILE_LEGACY_ZSCORE:
        if condition_feature_transform != CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE:
            raise ValueError(
                "legacy_zscore requires condition_feature_transform="
                f"{CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE!r}"
            )
        return
    if condition_feature_transform not in SEMANTIC_CONDITION_FEATURE_TRANSFORMS:
        raise ValueError(
            "semantic_normalization_v1 requires a semantic "
            f"condition_feature_transform, got {condition_feature_transform!r}"
        )


def _stats_profile(stats: dict[str, Any]) -> str:
    profile = stats.get("normalization_profile", NORMALIZATION_PROFILE_LEGACY_ZSCORE)
    _check_normalization_profile(profile)
    return str(profile)


def _semantic_transform_for_feature(
    feature_name: str, condition_feature_transform: str
) -> str:
    if (
        condition_feature_transform
        in {
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
        }
        and feature_name in K_FEATURES
    ):
        return TRANSFORM_LOG_K_ZSCORE
    if (
        condition_feature_transform
        in {
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
        }
        and feature_name in Q_FEATURES
    ):
        return TRANSFORM_SIGNED_LOG1P_Q_ZSCORE
    if (
        condition_feature_transform
        in {
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
            CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
        }
        and feature_name in BC_FLAG_FEATURES
    ):
        return TRANSFORM_BINARY_PASSTHROUGH
    if (
        condition_feature_transform == CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL
        and feature_name in TOP_H_FEATURES
    ):
        return TRANSFORM_TOP_H_ZSCORE
    if (
        condition_feature_transform == CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL
        and feature_name in RELATIVE_BC_TEMPERATURE_FEATURES
    ):
        return TRANSFORM_RELATIVE_BC_TEMPERATURE_ZSCORE
    return TRANSFORM_LINEAR_ZSCORE


def _semantic_transform_condition_np(raw_c: np.ndarray, transforms: tuple[str, ...]) -> np.ndarray:
    columns = []
    for index, transform in enumerate(transforms):
        values = raw_c[:, index : index + 1]
        if transform == TRANSFORM_LOG_K_ZSCORE:
            columns.append(np.log(np.maximum(values, SEMANTIC_LOG_EPS)))
        elif transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
            columns.append(np.sign(values) * np.log1p(np.abs(values)))
        else:
            columns.append(values)
    transformed = np.concatenate(columns, axis=-1)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("semantic_normalization_v1 produced non-finite condition features")
    return transformed


def _semantic_condition_stats(
    transformed_all: np.ndarray,
    transforms: tuple[str, ...],
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = safe_stats(transformed_all, eps=eps)
    passthrough = np.asarray(
        [transform == TRANSFORM_BINARY_PASSTHROUGH for transform in transforms],
        dtype=bool,
    ).reshape(1, -1)
    mean = np.where(passthrough, 0.0, mean)
    std = np.where(passthrough, 1.0, std)
    return mean, std


def _normalize_semantic_condition(raw_c: Any, stats: dict[str, Any]) -> Any:
    transforms = tuple(stats.get("condition_feature_transforms") or ())
    if not transforms:
        raise ValueError("semantic stats missing condition_feature_transforms")
    mean = stats["condition_mean"]
    std = stats["condition_std"]
    columns = []
    for index, transform in enumerate(transforms):
        raw_column = raw_c[..., index : index + 1]
        transformed = _semantic_transform_column(raw_column, transform)
        columns.append((transformed - mean[..., index : index + 1]) / std[..., index : index + 1])
    return jnp.concatenate(columns, axis=-1)


def _recover_semantic_condition(normalized_c: Any, stats: dict[str, Any]) -> Any:
    transforms = tuple(stats.get("condition_feature_transforms") or ())
    if not transforms:
        raise ValueError("semantic stats missing condition_feature_transforms")
    mean = stats["condition_mean"]
    std = stats["condition_std"]
    columns = []
    for index, transform in enumerate(transforms):
        transformed = normalized_c[..., index : index + 1] * std[..., index : index + 1]
        transformed = transformed + mean[..., index : index + 1]
        columns.append(_semantic_inverse_transform_column(transformed, transform))
    return jnp.concatenate(columns, axis=-1)


def _semantic_transform_column(raw_column: Any, transform: str) -> Any:
    if transform == TRANSFORM_LOG_K_ZSCORE:
        return jnp.log(jnp.maximum(raw_column, SEMANTIC_LOG_EPS))
    if transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
        return jnp.sign(raw_column) * jnp.log1p(jnp.abs(raw_column))
    return raw_column


def _semantic_inverse_transform_column(transformed_column: Any, transform: str) -> Any:
    if transform == TRANSFORM_LOG_K_ZSCORE:
        return jnp.exp(transformed_column)
    if transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
        return jnp.sign(transformed_column) * jnp.expm1(jnp.abs(transformed_column))
    return transformed_column
