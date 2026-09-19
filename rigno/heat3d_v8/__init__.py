"""Independent V8 MASS-HBM physics contract.

This package does not import or modify the frozen V7/P1i dataset contract.
"""

from .adapter import MassHBMReadOnlyAdapter, normalize_oracle_local_features
from .fingerprint import GeometryFingerprint, geometry_fingerprint
from .schema import (
    MassHBMCase,
    ProvenanceClass,
    V8BoundaryRepresentation,
    V8InterfaceRepresentation,
    V8OraclePhysicsView,
    reject_oracle_features,
)
from .support import V8SupportSelection, select_v8_support

__all__ = [
    "GeometryFingerprint",
    "MassHBMCase",
    "MassHBMReadOnlyAdapter",
    "ProvenanceClass",
    "V8BoundaryRepresentation",
    "V8InterfaceRepresentation",
    "V8OraclePhysicsView",
    "V8SupportSelection",
    "geometry_fingerprint",
    "normalize_oracle_local_features",
    "reject_oracle_features",
    "select_v8_support",
]
