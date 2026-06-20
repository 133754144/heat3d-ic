"""Stable Heat3D v1 training-semantics helpers.

This module names the current legacy V4 route without changing behavior. It is
intended to keep smoke scripts and medium runners from each defining their own
bridge policy strings.
"""

from __future__ import annotations

from typing import Any


NORMALIZATION_PROFILE_LEGACY_ZSCORE = "legacy_zscore"
BRIDGE_POLICY_ZERO_DELTA_U = "zero_delta_u_bridge"
TARGET_MODE_NORMALIZED_DELTAT = "normalized_deltaT"
COORD_POLICY_TRAIN_MINMAX_UNIT_BOX = "train_minmax_to_unit_box"
TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF = "deltaT_norm_to_K_plus_T_ref"
LEGACY_ROUTE_DESCRIPTION = "relative BC features + zero_delta_u_bridge + normalized DeltaT target"


def build_legacy_zero_delta_bridge(example: Any) -> Any:
    """Build the current V4 legacy bridge for one native Heat3D example."""

    return example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy=BRIDGE_POLICY_ZERO_DELTA_U
    )


def legacy_training_semantics_manifest() -> dict[str, str]:
    """Return provenance labels for the current no-behavior-change route."""

    return {
        "normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
        "bridge_policy": BRIDGE_POLICY_ZERO_DELTA_U,
        "target_mode": TARGET_MODE_NORMALIZED_DELTAT,
        "coord_policy": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
        "target_recovery_policy": TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF,
        "route": LEGACY_ROUTE_DESCRIPTION,
    }
