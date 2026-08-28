"""Stable V7 Heat3D inference runtime public surface.

The package intentionally keeps imports lazy.  Importing the namespace does
not initialize JAX, Flax, a model, a checkpoint, or a dataset.  The concrete
runtime objects are loaded only when requested by a caller.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CheckpointBundle",
    "load_checkpoint",
    "materialize_checkpoint_stats",
    "resolve_model_config",
    "device_params",
    "FeatureTransform",
    "GroupBuilder",
    "RuntimeSession",
    "FullFieldGeometry",
    "SupportArtifact",
    "HighNCase",
    "HighNRuntime",
    "UHighNCase",
    "UHighNRuntime",
    "u_v2_asymmetric_metadata",
    "ComparisonReport",
    "compare_named_arrays",
    "compare_metadata",
    "snapshot_group",
    "EXECUTION_ROLES",
    "SemanticContractError",
    "validate_semantic_contract",
    "load_registered_route",
    "bind_registered_route",
    "EvaluationSample",
    "EvaluationCore",
    "TimingCore",
    "TimingLifecycle",
    "ABSOLUTE_TEMPERATURE_K",
    "DELTA_T_K",
    "TemperatureRepresentationError",
    "temperature_K_to_deltaT_K",
    "load_temperature_contract",
    "validate_temperature_contract",
    "PredictionOnlyRecord",
    "FormalEvaluationOrchestrator",
]

_EXPORTS = {
    "CheckpointBundle": ("rigno.heat3d_runtime.checkpoint", "CheckpointBundle"),
    "load_checkpoint": ("rigno.heat3d_runtime.checkpoint", "load_checkpoint"),
    "materialize_checkpoint_stats": (
        "rigno.heat3d_runtime.checkpoint",
        "materialize_checkpoint_stats",
    ),
    "resolve_model_config": (
        "rigno.heat3d_runtime.checkpoint",
        "resolve_model_config",
    ),
    "device_params": ("rigno.heat3d_runtime.checkpoint", "device_params"),
    "FeatureTransform": ("rigno.heat3d_runtime.features", "FeatureTransform"),
    "GroupBuilder": ("rigno.heat3d_runtime.grouping", "GroupBuilder"),
    "RuntimeSession": ("rigno.heat3d_runtime.session", "RuntimeSession"),
    "FullFieldGeometry": ("rigno.heat3d_runtime.high_n", "FullFieldGeometry"),
    "SupportArtifact": ("rigno.heat3d_runtime.high_n", "SupportArtifact"),
    "HighNCase": ("rigno.heat3d_runtime.high_n", "HighNCase"),
    "HighNRuntime": ("rigno.heat3d_runtime.high_n", "HighNRuntime"),
    "UHighNCase": ("rigno.heat3d_runtime.u_split", "UHighNCase"),
    "UHighNRuntime": ("rigno.heat3d_runtime.u_split", "UHighNRuntime"),
    "u_v2_asymmetric_metadata": (
        "rigno.heat3d_runtime.u_split",
        "u_v2_asymmetric_metadata",
    ),
    "ComparisonReport": ("rigno.heat3d_runtime.equivalence", "ComparisonReport"),
    "compare_named_arrays": (
        "rigno.heat3d_runtime.equivalence",
        "compare_named_arrays",
    ),
    "compare_metadata": (
        "rigno.heat3d_runtime.equivalence",
        "compare_metadata",
    ),
    "snapshot_group": ("rigno.heat3d_runtime.equivalence", "snapshot_group"),
    "EXECUTION_ROLES": ("rigno.heat3d_runtime.preflight", "EXECUTION_ROLES"),
    "SemanticContractError": (
        "rigno.heat3d_runtime.preflight",
        "SemanticContractError",
    ),
    "validate_semantic_contract": (
        "rigno.heat3d_runtime.preflight",
        "validate_semantic_contract",
    ),
    "load_registered_route": (
        "rigno.heat3d_runtime.preflight",
        "load_registered_route",
    ),
    "bind_registered_route": (
        "rigno.heat3d_runtime.preflight",
        "bind_registered_route",
    ),
    "EvaluationSample": ("rigno.heat3d_runtime.evaluation", "EvaluationSample"),
    "EvaluationCore": ("rigno.heat3d_runtime.evaluation", "EvaluationCore"),
    "TimingCore": ("rigno.heat3d_runtime.timing", "TimingCore"),
    "TimingLifecycle": ("rigno.heat3d_runtime.timing", "TimingLifecycle"),
    "ABSOLUTE_TEMPERATURE_K": (
        "rigno.heat3d_runtime.temperature",
        "ABSOLUTE_TEMPERATURE_K",
    ),
    "DELTA_T_K": ("rigno.heat3d_runtime.temperature", "DELTA_T_K"),
    "TemperatureRepresentationError": (
        "rigno.heat3d_runtime.temperature",
        "TemperatureRepresentationError",
    ),
    "temperature_K_to_deltaT_K": (
        "rigno.heat3d_runtime.temperature",
        "temperature_K_to_deltaT_K",
    ),
    "load_temperature_contract": (
        "rigno.heat3d_runtime.temperature",
        "load_temperature_contract",
    ),
    "validate_temperature_contract": (
        "rigno.heat3d_runtime.temperature",
        "validate_temperature_contract",
    ),
    "PredictionOnlyRecord": (
        "rigno.heat3d_runtime.orchestration",
        "PredictionOnlyRecord",
    ),
    "FormalEvaluationOrchestrator": (
        "rigno.heat3d_runtime.orchestration",
        "FormalEvaluationOrchestrator",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
