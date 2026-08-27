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
    "ComparisonReport",
    "compare_named_arrays",
    "compare_metadata",
    "snapshot_group",
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
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
