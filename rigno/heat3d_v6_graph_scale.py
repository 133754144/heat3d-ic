"""Label-independent graph-scale candidates for P1i High-N ablation.

This module does not alter the semantics of the existing
``discrete_physical_coverage`` policy.  The native-1024 candidate builds the
unchanged native anchor graph first, then transfers its local normalized
coverage radii to High-N regional nodes by geometry-only nearest assignment.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.models.rigno import RegionInteractionGraphBuilder


NATIVE_POLICY = "native1024_physical_coverage_v1"


def _nearest_reference_values(
    query: np.ndarray,
    reference_coords: np.ndarray,
    reference_values: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Nearest assignment using the graph's float32 normalized L2 convention."""

    query = np.asarray(query, dtype=np.float32)
    reference_coords = np.asarray(reference_coords, dtype=np.float32)
    reference_values = np.asarray(reference_values, dtype=np.float32)
    result = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk_size):
        block = query[start : start + chunk_size]
        distance = np.linalg.norm(
            block[:, None, :] - reference_coords[None, :, :], axis=-1
        )
        result[start : start + len(block)] = reference_values[
            np.argmin(distance, axis=1)
        ]
    return result


class _NativeCoverageRegionBuilder(RegionInteractionGraphBuilder):
    def __init__(
        self,
        *,
        native_rnodes: np.ndarray,
        native_radii: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._native_rnodes = np.asarray(native_rnodes, dtype=np.float32)
        self._native_radii = np.asarray(native_radii, dtype=np.float32)

    def _compute_discrete_physical_coverage_radius(self, centers, points):
        del points
        transferred = _nearest_reference_values(
            np.asarray(centers, dtype=np.float32),
            self._native_rnodes,
            self._native_radii,
        )
        return jnp.asarray(transferred, dtype=centers.dtype)


class Native1024PhysicalCoverageGraphBuilder:
    """High-N builder with native-1024 geometry-only coverage radii."""

    def __init__(
        self,
        *,
        anchor_coords: np.ndarray,
        graph_config: Mapping[str, Any],
        graph_key: Any,
        regional_subsample_factor: int = 4,
    ) -> None:
        base_config = dict(graph_config)
        base_config["subsample_factor"] = 4
        base_config["radius_policy"] = "discrete_physical_coverage"
        native_builder = Heat3DGraphBuilder(**base_config)
        native = native_builder.build_metadata(anchor_coords, key=graph_key)
        coordinate_dimension = int(np.asarray(anchor_coords).shape[-1])
        native_rnodes = np.asarray(native.x_rnodes)[0, :-1, :coordinate_dimension]
        native_radii = np.asarray(native.r_rnodes)[0, :-1]

        candidate = dict(graph_config)
        candidate["subsample_factor"] = int(regional_subsample_factor)
        candidate["radius_policy"] = "discrete_physical_coverage"
        self.config = dict(candidate)
        self.config["reported_graph_mode"] = NATIVE_POLICY
        self.config["native_reference_subsample_factor"] = 4
        self._native_reference = {
            "rnodes": native_rnodes,
            "radii": native_radii,
        }
        self.builder = _NativeCoverageRegionBuilder(
            periodic=False,
            rmesh_levels=int(candidate["rmesh_levels"]),
            subsample_factor=float(candidate["subsample_factor"]),
            overlap_factor_p2r=float(candidate["overlap_factor_p2r"]),
            overlap_factor_r2p=float(candidate["overlap_factor_r2p"]),
            node_coordinate_encoding=str(candidate["node_coordinate_encoding"]),
            node_coordinate_freqs=int(candidate["node_coordinate_freqs"]),
            coverage_repair_policy=str(candidate["coverage_repair_policy"]),
            radius_policy="discrete_physical_coverage",
            repair_p2r=bool(candidate["repair_p2r"]),
            repair_r2p=bool(candidate["repair_r2p"]),
            min_physical_coverage=int(candidate["min_physical_coverage"]),
            discrete_graph_backend=str(candidate["discrete_graph_backend"]),
            discrete_graph_chunk_size=int(candidate["discrete_graph_chunk_size"]),
            discrete_coverage_multiplier=float(candidate["discrete_coverage_multiplier"]),
            native_rnodes=native_rnodes,
            native_radii=native_radii,
        )

    @property
    def native_reference(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value).copy() for key, value in self._native_reference.items()}

    def build_metadata(self, coords: np.ndarray, key=None):
        coords = jnp.asarray(coords)
        domain = jnp.asarray([coords.min(axis=0), coords.max(axis=0)])
        return self.builder.build_metadata(
            x_inp=coords,
            x_out=coords,
            domain=domain,
            key=key,
        )

    def build_graphs(self, metadata):
        return self.builder.build_graphs(metadata)
