"""Independent V8 MASS-HBM physics contract.

This package does not import or modify the frozen V7/P1i dataset contract.
"""

from .adapter import (
    MassHBMReadOnlyAdapter,
    V8_PHYSICAL_SCALE_FEATURES,
    canonical_oracle_condition,
    normalize_oracle_local_features,
    physical_scale_features,
)
from .fingerprint import GeometryFingerprint, geometry_fingerprint
from .context import V8_GLOBAL_CONTEXT_FEATURES, oracle_global_context
from .bridge import V8CanonicalBridge, build_canonical_bridge
from .schema import (
    MassHBMCase,
    ProvenanceClass,
    SupportProvenance,
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
    "SupportProvenance",
    "V8BoundaryRepresentation",
    "V8CanonicalBridge",
    "V8_GLOBAL_CONTEXT_FEATURES",
    "V8InterfaceRepresentation",
    "V8OraclePhysicsView",
    "V8SupportSelection",
    "geometry_fingerprint",
    "canonical_oracle_condition",
    "build_canonical_bridge",
    "normalize_oracle_local_features",
    "physical_scale_features",
    "oracle_global_context",
    "V8_PHYSICAL_SCALE_FEATURES",
    "reject_oracle_features",
    "select_v8_support",
]
