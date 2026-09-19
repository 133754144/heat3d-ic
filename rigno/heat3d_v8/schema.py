"""Typed, provenance-explicit V8 MASS-HBM physics schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np


class ProvenanceClass(str, Enum):
    PRE_SOLVE = "PRE_SOLVE"
    ORACLE = "ORACLE"
    LABEL = "LABEL"
    METADATA = "METADATA"


class SupportProvenance(str, Enum):
    """Whether support strata can exist before the coupled solve."""

    ORACLE_SUPPORT = "ORACLE_SUPPORT"
    PRE_SOLVE_SUPPORT = "PRE_SOLVE_SUPPORT"


@dataclass(frozen=True)
class V8BoundaryRepresentation:
    """Generic Robin/flux boundary faces plus a node-aggregated candidate.

    Every face follows ``Q_face = G_face * (T - T_sink)`` with
    ``G_face = h * area``.  Internal and external sinks use the same law;
    no categorical liquid-cooling flag is present.
    """

    cell_index: np.ndarray
    area_m2: np.ndarray
    outward_normal: np.ndarray
    conductance_W_K: np.ndarray
    sink_temperature_K: np.ndarray
    prescribed_flux_W_m2: np.ndarray
    is_internal: np.ndarray
    node_feature_names: tuple[str, ...]
    node_features: np.ndarray
    provenance: Mapping[str, ProvenanceClass]
    semantic_status: str


@dataclass(frozen=True)
class V8InterfaceRepresentation:
    """Interface physics kept separate from boundary physics."""

    mode: str
    lower_cell_index: np.ndarray
    upper_cell_index: np.ndarray
    area_m2: np.ndarray
    normal: np.ndarray
    tbr_m2K_W: np.ndarray
    conductance_W_K: np.ndarray
    node_feature_names: tuple[str, ...]
    node_features: np.ndarray
    provenance: Mapping[str, ProvenanceClass]
    double_count_guard: str
    explicit_feature_safe: bool


@dataclass(frozen=True)
class MassHBMCase:
    dataset_root: Path
    case_directory: str
    sample_id: str
    architecture_metadata: str
    shape_zyx: tuple[int, int, int]
    coords_m: np.ndarray
    control_volume_m3: np.ndarray
    valid_cell_mask: np.ndarray
    geometry_metadata: Mapping[str, object]
    boundary: V8BoundaryRepresentation
    interface: V8InterfaceRepresentation


@dataclass(frozen=True)
class V8OraclePhysicsView:
    case: MassHBMCase
    k_diag_W_mK: np.ndarray
    q_W_m3: np.ndarray
    target_temperature_K: np.ndarray
    local_feature_names: tuple[str, ...]
    local_features: np.ndarray
    feature_provenance: Mapping[str, ProvenanceClass]
    target_provenance: ProvenanceClass = ProvenanceClass.LABEL
    metadata: Mapping[str, object] = field(default_factory=dict)


def reject_oracle_features(provenance: Mapping[str, ProvenanceClass]) -> None:
    """Future Track-B fail-closed provenance gate."""

    forbidden = sorted(
        name
        for name, source in provenance.items()
        if ProvenanceClass(source) == ProvenanceClass.ORACLE
    )
    if forbidden:
        raise ValueError(
            "Track-B rejects ORACLE features: " + ", ".join(forbidden)
        )
