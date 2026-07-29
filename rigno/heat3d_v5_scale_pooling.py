"""Input-only regional pooling features for V5 native shape--scale heads.

This module intentionally contains no target-temperature inputs.  The
``qk_gated`` pooling branch receives one feature vector per RIGNO regional
node, deterministically aggregated from raw coordinates, conductivity, heat
source and boundary-condition fields through the already-built P2R graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SCALE_POOLING_MODES = (
    "mean",
    "mean_std",
    "mean_max",
    "pre_film_mean_std",
    "latent_attention",
    "qk_gated",
)

REGIONAL_ATTENTION_MODES = ("none", "physics_gate")
QK_REGION_FEATURE_VERSIONS = ("bugged_v1", "sparse_safe_v2")

# These versioned schemas are frozen provenance.  ``bugged_v1`` remains the
# default so historical checkpoints replay exactly.  Both schemas are derived
# exclusively from raw coords/k/q/BC before inference; no temperature or
# target-derived quantity may enter either list.
QK_REGION_FEATURE_SCHEMAS = {
    "bugged_v1": (
        "log1p_q_relative",
        "log_inverse_kz_relative",
        "log1p_q_inverse_kz_relative",
        "q_high_inverse_kz_overlap",
        "source_z_normalized",
        "is_top_fraction",
        "is_bottom_fraction",
        "is_side_fraction",
        "is_interior_fraction",
        "log1p_top_h_relative",
        "bottom_bc_offset_relative",
    ),
    "sparse_safe_v2": (
        "log1p_q_relative",
        "log_inverse_kz_relative",
        "log1p_q_inverse_kz_relative",
        "source_present_fraction",
        "region_z_normalized",
        "is_top_fraction",
        "is_bottom_fraction",
        "is_side_fraction",
        "is_interior_fraction",
        "log1p_top_h_relative",
        "bottom_bc_offset_relative",
    ),
}
QK_REGION_FEATURES = QK_REGION_FEATURE_SCHEMAS["bugged_v1"]

_REQUIRED_RAW_FEATURES = (
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
    "top_h",
)

EPS = 1.0e-12


def validate_scale_pooling_mode(mode: str) -> str:
    """Return a checked scale pooling mode."""

    value = str(mode)
    if value not in SCALE_POOLING_MODES:
        raise ValueError(
            "scale_pooling must be one of "
            f"{list(SCALE_POOLING_MODES)}, got {value!r}"
        )
    return value


def validate_qk_region_feature_version(version: str) -> str:
    """Return a checked QK regional feature schema version."""

    value = str(version)
    if value not in QK_REGION_FEATURE_VERSIONS:
        raise ValueError(
            "qk_region_feature_version must be one of "
            f"{list(QK_REGION_FEATURE_VERSIONS)}, got {value!r}"
        )
    return value


def qk_region_feature_names(version: str = "bugged_v1") -> tuple[str, ...]:
    """Return the frozen feature names for ``version``."""

    return QK_REGION_FEATURE_SCHEMAS[validate_qk_region_feature_version(version)]


def qk_region_features_from_raw(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    p2r_edge_indices: np.ndarray,
    rnode_count: int,
    feature_version: str = "bugged_v1",
) -> np.ndarray:
    """CV-free P2R regional aggregation for q--k gated pooling.

    ``p2r_edge_indices`` uses the RIGNO P2R sender/receiver convention
    ``[physical_node, regional_node]``.  Dummy graph nodes and edges are
    excluded before aggregation.  Per-sample relative scalings are derived
    only from the raw input itself so the output remains available at
    inference time and cannot leak labels.
    """

    version = validate_qk_region_feature_version(feature_version)
    feature_names = qk_region_feature_names(version)
    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(raw_condition, dtype=np.float64)
    names = tuple(str(name) for name in condition_feature_names)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,3], got {points.shape}")
    if values.ndim != 2 or values.shape[0] != points.shape[0]:
        raise ValueError(
            "raw_condition must have shape [N,F] aligned with coords, "
            f"got {values.shape} for {points.shape}"
        )
    if int(rnode_count) < 1:
        raise ValueError("rnode_count must be >= 1")
    missing = [name for name in _REQUIRED_RAW_FEATURES if name not in names]
    if missing:
        raise ValueError(f"qk_gated pooling lacks raw condition features: {missing}")
    if "bottom_T_fixed_minus_T_ref" not in names and "bottom_h" not in names:
        raise ValueError(
            "qk_gated pooling requires either the legacy bottom Dirichlet offset "
            "or the V6 bottom Robin coefficient"
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("qk_gated raw coords/condition must be finite")

    index = {name: names.index(name) for name in _REQUIRED_RAW_FEATURES}
    q = np.maximum(values[:, index["q"]], 0.0)
    kz = np.maximum(values[:, index["k_z"]], EPS)
    inv_kz = 1.0 / kz
    q_positive = q[q > 0.0]
    q_reference = float(np.mean(q_positive)) if q_positive.size else 1.0
    inv_reference = max(float(np.mean(inv_kz)), EPS)
    q_relative = q / max(q_reference, EPS)
    inv_relative = inv_kz / inv_reference
    q_inv_relative = q_relative * inv_relative
    if version == "bugged_v1":
        q_high = q >= np.quantile(q, 0.75)
        inv_high = inv_kz >= np.quantile(inv_kz, 0.75)
        fourth_feature = (q_high & inv_high).astype(np.float64)
    else:
        fourth_feature = (q > EPS).astype(np.float64)

    z = points[:, 2]
    z_extent = max(float(np.max(z) - np.min(z)), EPS)
    z_normalized = (z - float(np.min(z))) / z_extent
    top_h = np.maximum(values[:, index["top_h"]], 0.0)
    top_h_reference = max(float(np.mean(top_h)), EPS)
    if "bottom_h" in names:
        bottom_h = np.maximum(values[:, names.index("bottom_h")], 0.0)
        bottom_reference = max(float(np.mean(bottom_h)), EPS)
        bottom_bc_feature = np.log1p(bottom_h / bottom_reference)
    else:
        bottom_offset = values[:, names.index("bottom_T_fixed_minus_T_ref")]
        bottom_reference = max(float(np.mean(np.abs(bottom_offset))), EPS)
        bottom_bc_feature = bottom_offset / bottom_reference
    physical_features = np.stack(
        [
            np.log1p(q_relative),
            np.log(np.maximum(inv_relative, EPS)),
            np.log1p(q_inv_relative),
            fourth_feature,
            z_normalized,
            values[:, index["is_top"]],
            values[:, index["is_bottom"]],
            values[:, index["is_side"]],
            values[:, index["is_interior"]],
            np.log1p(top_h / top_h_reference),
            bottom_bc_feature,
        ],
        axis=-1,
    )
    if physical_features.shape[1] != len(feature_names):
        raise AssertionError("qk regional feature schema width drifted")

    edges = np.asarray(p2r_edge_indices, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"p2r_edge_indices must have shape [E,2], got {edges.shape}")
    physical_indices, regional_indices = edges[:, 0], edges[:, 1]
    valid = (
        (physical_indices >= 0)
        & (physical_indices < points.shape[0])
        & (regional_indices >= 0)
        & (regional_indices < int(rnode_count))
    )
    if not np.any(valid):
        raise ValueError("qk_gated pooling found no non-dummy P2R edges")
    physical_indices = physical_indices[valid]
    regional_indices = regional_indices[valid]
    regional_sum = np.zeros((int(rnode_count), len(feature_names)), dtype=np.float64)
    regional_count = np.zeros((int(rnode_count), 1), dtype=np.float64)
    np.add.at(regional_sum, regional_indices, physical_features[physical_indices])
    np.add.at(regional_count[:, 0], regional_indices, 1.0)
    regional = regional_sum / np.maximum(regional_count, 1.0)
    if not np.all(np.isfinite(regional)):
        raise ValueError("qk_gated regional features are non-finite")
    return regional.astype(np.float32)
