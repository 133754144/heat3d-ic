"""Stable Heat3D v1 training-semantics helpers.

This module names the current legacy V4 route without changing behavior. It is
intended to keep smoke scripts and medium runners from each defining their own
bridge policy strings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from rigno.models.operator import Inputs


NORMALIZATION_PROFILE_LEGACY_ZSCORE = "legacy_zscore"
NORMALIZATION_PROFILE_SEMANTIC_V1 = "semantic_normalization_v1"
NORMALIZATION_PROFILES = (
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
)
BRIDGE_POLICY_ZERO_DELTA_U = "zero_delta_u_bridge"
TARGET_MODE_NORMALIZED_DELTAT = "normalized_deltaT"
COORD_POLICY_TRAIN_MINMAX_UNIT_BOX = "train_minmax_to_unit_box"
COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC = "sample_local_isotropic"
COORD_POLICIES = (
    COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC,
)
INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS = "legacy_bc_flags"
INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT = "boundary_distance_replacement"
INPUT_FEATURE_SCHEMAS = (
    INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS,
    INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT,
)
EXTENT_FEATURE_POLICY_NONE = "none"
EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST = "log_extent_broadcast"
EXTENT_FEATURE_POLICIES = (
    EXTENT_FEATURE_POLICY_NONE,
    EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST,
)
TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF = "deltaT_norm_to_K_plus_T_ref"
LEGACY_ROUTE_DESCRIPTION = "relative BC features + zero_delta_u_bridge + normalized DeltaT target"
BC_FLAG_FEATURES = ("is_top", "is_bottom", "is_side", "is_interior")
BOUNDARY_DISTANCE_FEATURES = (
    "d_xmin",
    "d_xmax",
    "d_ymin",
    "d_ymax",
    "d_bottom",
    "d_top",
)
EXTENT_BROADCAST_FEATURES = (
    "log_Lx",
    "log_Ly",
    "log_Lz",
    "log_Lx_over_Lz",
    "log_Ly_over_Lz",
)
BASE_CONDITION_FEATURES = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "top_h",
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
)
_GEOMETRY_EPS = 1.0e-12


def build_legacy_zero_delta_bridge(example: Any) -> Any:
    """Build the current V4 legacy bridge for one native Heat3D example."""

    return example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy=BRIDGE_POLICY_ZERO_DELTA_U
    )


def build_configured_zero_delta_bridge(
    example: Any,
    *,
    input_feature_schema: str = INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS,
    coord_policy: str = COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    extent_feature_policy: str = EXTENT_FEATURE_POLICY_NONE,
) -> Any:
    """Build the zero-delta bridge with opt-in P2 condition feature views."""

    _check_input_feature_schema(input_feature_schema)
    _check_coord_policy(coord_policy)
    _check_extent_feature_policy(extent_feature_policy)
    bridge = build_legacy_zero_delta_bridge(example)
    if (
        input_feature_schema == INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS
        and extent_feature_policy == EXTENT_FEATURE_POLICY_NONE
    ):
        return bridge

    raw_c = np.asarray(bridge.legacy_inputs.c, dtype=np.float64)
    flat_c = raw_c.reshape(-1, raw_c.shape[-1])
    coords = np.asarray(bridge.legacy_inputs.x_inp, dtype=np.float64).reshape(-1, 3)
    transformed_c, transformed_names = transform_condition_feature_view(
        flat_c,
        tuple(bridge.condition_feature_names),
        coords,
        input_feature_schema=input_feature_schema,
        coord_policy=coord_policy,
        extent_feature_policy=extent_feature_policy,
    )
    c = jnp.asarray(transformed_c.reshape(raw_c.shape[:-1] + (-1,)))
    legacy_inputs = Inputs(
        u=bridge.legacy_inputs.u,
        c=c,
        x_inp=bridge.legacy_inputs.x_inp,
        x_out=bridge.legacy_inputs.x_out,
        t=bridge.legacy_inputs.t,
        tau=bridge.legacy_inputs.tau,
    )
    return replace(
        bridge,
        legacy_inputs=legacy_inputs,
        condition_feature_names=tuple(transformed_names),
    )


def transform_condition_feature_view(
    condition_features: np.ndarray,
    feature_names: tuple[str, ...],
    coords: np.ndarray,
    *,
    input_feature_schema: str,
    coord_policy: str,
    extent_feature_policy: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the P2 feature-schema view before condition normalization."""

    _check_input_feature_schema(input_feature_schema)
    _check_coord_policy(coord_policy)
    _check_extent_feature_policy(extent_feature_policy)
    features = np.asarray(condition_features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"condition_features must be rank-2, got {features.shape}")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            "condition feature width does not match feature_names: "
            f"features={features.shape[1]} names={len(feature_names)}"
        )
    coords_array = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if coords_array.shape[0] != features.shape[0]:
        raise ValueError(
            "coords and condition feature rows must match: "
            f"coords={coords_array.shape[0]} features={features.shape[0]}"
        )

    transformed = features
    names = tuple(feature_names)
    if input_feature_schema == INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT:
        transformed, names = _replace_bc_flags_with_boundary_distances(
            transformed,
            names,
            coords_array,
            coord_policy=coord_policy,
        )
    if extent_feature_policy == EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST:
        transformed, names = _append_log_extent_broadcast(
            transformed,
            names,
            coords_array,
        )
    return transformed, names


def decoder_bypass_required_full_condition_features(
    *,
    input_feature_schema: str,
    extent_feature_policy: str,
) -> tuple[str, ...]:
    """Return the required `full_condition` names for the active feature view."""

    _check_input_feature_schema(input_feature_schema)
    _check_extent_feature_policy(extent_feature_policy)
    boundary_features = (
        BOUNDARY_DISTANCE_FEATURES
        if input_feature_schema == INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT
        else BC_FLAG_FEATURES
    )
    required = (
        "k_x",
        "k_y",
        "k_z",
        "q",
        *boundary_features,
        "top_h",
        "top_T_inf_minus_T_ref",
        "bottom_T_fixed_minus_T_ref",
    )
    if extent_feature_policy == EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST:
        required = (*required, *EXTENT_BROADCAST_FEATURES)
    return required


def legacy_training_semantics_manifest() -> dict[str, str]:
    """Return provenance labels for the current no-behavior-change route."""

    return {
        "normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
        "bridge_policy": BRIDGE_POLICY_ZERO_DELTA_U,
        "target_mode": TARGET_MODE_NORMALIZED_DELTAT,
        "coord_policy": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
        "input_feature_schema": INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS,
        "extent_feature_policy": EXTENT_FEATURE_POLICY_NONE,
        "target_recovery_policy": TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF,
        "route": LEGACY_ROUTE_DESCRIPTION,
    }


def _replace_bc_flags_with_boundary_distances(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    coords: np.ndarray,
    *,
    coord_policy: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    missing = [name for name in BC_FLAG_FEATURES if name not in feature_names]
    if missing:
        raise ValueError(
            "boundary_distance_replacement requires legacy BC flag features; "
            f"missing={missing} available={feature_names}"
        )
    distance_features = _boundary_distance_features(coords, coord_policy=coord_policy)
    distance_inserted = False
    columns: list[np.ndarray] = []
    names: list[str] = []
    for index, name in enumerate(feature_names):
        if name in BC_FLAG_FEATURES:
            if not distance_inserted:
                columns.append(distance_features)
                names.extend(BOUNDARY_DISTANCE_FEATURES)
                distance_inserted = True
            continue
        columns.append(features[:, index : index + 1])
        names.append(name)
    return np.concatenate(columns, axis=1), tuple(names)


def _boundary_distance_features(coords: np.ndarray, *, coord_policy: str) -> np.ndarray:
    _check_coord_policy(coord_policy)
    mins = np.min(coords, axis=0, keepdims=True)
    maxs = np.max(coords, axis=0, keepdims=True)
    spans = np.maximum(maxs - mins, _GEOMETRY_EPS)
    if coord_policy == COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC:
        l_ref = np.maximum(np.max(spans, axis=1, keepdims=True), _GEOMETRY_EPS)
        denominators = np.repeat(l_ref, repeats=3, axis=1)
    else:
        denominators = spans
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    z = coords[:, 2:3]
    return np.concatenate(
        (
            (x - mins[:, 0:1]) / denominators[:, 0:1],
            (maxs[:, 0:1] - x) / denominators[:, 0:1],
            (y - mins[:, 1:2]) / denominators[:, 1:2],
            (maxs[:, 1:2] - y) / denominators[:, 1:2],
            (z - mins[:, 2:3]) / denominators[:, 2:3],
            (maxs[:, 2:3] - z) / denominators[:, 2:3],
        ),
        axis=1,
    )


def _append_log_extent_broadcast(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    coords: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    duplicate = [name for name in EXTENT_BROADCAST_FEATURES if name in feature_names]
    if duplicate:
        raise ValueError(f"extent broadcast feature(s) already present: {duplicate}")
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    spans = np.maximum(maxs - mins, _GEOMETRY_EPS)
    lx, ly, lz = (float(value) for value in spans)
    values = np.asarray(
        [
            np.log(lx),
            np.log(ly),
            np.log(lz),
            np.log(lx / lz),
            np.log(ly / lz),
        ],
        dtype=np.float64,
    ).reshape(1, -1)
    broadcast = np.repeat(values, repeats=features.shape[0], axis=0)
    return (
        np.concatenate((features, broadcast), axis=1),
        (*feature_names, *EXTENT_BROADCAST_FEATURES),
    )


def _check_input_feature_schema(value: str) -> None:
    if value not in INPUT_FEATURE_SCHEMAS:
        raise ValueError(
            f"input_feature_schema must be one of {INPUT_FEATURE_SCHEMAS}, "
            f"found {value!r}"
        )


def _check_coord_policy(value: str) -> None:
    if value not in COORD_POLICIES:
        raise ValueError(
            f"coord_policy must be one of {COORD_POLICIES}, found {value!r}"
        )


def _check_extent_feature_policy(value: str) -> None:
    if value not in EXTENT_FEATURE_POLICIES:
        raise ValueError(
            f"extent_feature_policy must be one of {EXTENT_FEATURE_POLICIES}, "
            f"found {value!r}"
        )
