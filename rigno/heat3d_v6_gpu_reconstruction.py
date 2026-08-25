"""GPU apply path for frozen Heat3D V6 reconstruction maps.

The map builder, selected indices, neighbour indices, and interpolation weights
remain owned by :mod:`rigno.heat3d_v6_full_field`.  This module only transfers
an already-built map to a JAX device and applies the same weighted gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v6_full_field import ReconstructionMap


@dataclass(frozen=True)
class DeviceReconstructionMap:
    """Device-resident immutable view of a frozen reconstruction map."""

    neighbor_local_indices: Any
    neighbor_weights: Any
    support_node_count: int
    full_node_count: int

    def reconstruct(self, support_values: Any) -> Any:
        values = jnp.asarray(support_values)
        if values.ndim != 1 or values.shape[0] != self.support_node_count:
            raise ValueError("support value shape does not match reconstruction map")
        weights = self.neighbor_weights.astype(values.dtype)
        return jnp.sum(values[self.neighbor_local_indices] * weights, axis=1)


def to_device_reconstruction_map(
    mapping: ReconstructionMap,
    *,
    device: Any | None = None,
) -> DeviceReconstructionMap:
    """Transfer a frozen CPU map to one device without rebuilding it."""

    target = device if device is not None else jax.devices()[0]
    indices = jax.device_put(
        np.asarray(mapping.neighbor_local_indices, dtype=np.int32), target
    )
    weights = jax.device_put(
        np.asarray(mapping.neighbor_weights, dtype=np.float64), target
    )
    return DeviceReconstructionMap(
        neighbor_local_indices=indices,
        neighbor_weights=weights,
        support_node_count=int(len(mapping.support_indices)),
        full_node_count=int(len(mapping.neighbor_local_indices)),
    )
