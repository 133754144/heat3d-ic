"""Stable V7 anchor-derived high-resolution inference primitives.

This module is an explicit adapter around the frozen V6/P1i high-N contract.
It reads existing support/full-field artifacts, preserves the registered
source-aware support order, uses the existing sparse KD-tree graph backend,
and delegates reconstruction to the existing V6 implementation.  It never
generates support data, invokes a solver, reads prediction targets while
building model inputs, or mutates another module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import jax
import numpy as np
from scipy.spatial import cKDTree

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_graph_cache import (
    cache_key,
    cache_key_payload,
    file_sha256,
    graph_hash,
    graph_builder_code_fingerprint,
    load_metadata,
    metadata_hash,
    save_metadata,
)
from rigno.heat3d_runtime.checkpoint import file_sha256 as checkpoint_file_sha256
from rigno.heat3d_runtime.grouping import EDGE_FIELDS, GroupBuilder
from rigno.heat3d_runtime.session import RuntimeSession
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v6_dataset import (
    V6DualRobinExample,
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.heat3d_v6_full_field import (
    FullFieldMetricAccumulator,
    ReconstructionMap,
    build_reconstruction_map,
)
from rigno.heat3d_v6_p1i_anchor_query import array_sha256


REFERENCE_K = 300.0
TRAINING_ANCHOR_COUNT = 1024
SPARSE_GRAPH_BACKEND = "sparse_kdtree_v1"


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        if leaf is None or not hasattr(leaf, "shape"):
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(index).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping_sha256(mapping: ReconstructionMap) -> str:
    digest = hashlib.sha256()
    for name in (
        "support_indices",
        "neighbor_local_indices",
        "neighbor_weights",
        "domain_code",
    ):
        array = np.ascontiguousarray(np.asarray(getattr(mapping, name)))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    digest.update("\n".join(mapping.domain_names).encode())
    return digest.hexdigest()


def _boundary_flags(coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    top = np.isclose(points[:, 2], np.max(points[:, 2]), atol=1.0e-15)
    bottom = np.isclose(points[:, 2], np.min(points[:, 2]), atol=1.0e-15)
    side = (
        np.isclose(points[:, 0], np.min(points[:, 0]), atol=1.0e-15)
        | np.isclose(points[:, 0], np.max(points[:, 0]), atol=1.0e-15)
        | np.isclose(points[:, 1], np.min(points[:, 1]), atol=1.0e-15)
        | np.isclose(points[:, 1], np.max(points[:, 1]), atol=1.0e-15)
    ) & ~top & ~bottom
    interior = ~(top | bottom | side)
    flags = np.column_stack((top, bottom, side, interior)).astype(np.float64)
    if not np.array_equal(np.sum(flags, axis=1), np.ones(len(points))):
        raise ValueError("high-N boundary flags are not one-hot")
    return flags


def _boundaries(example: V6DualRobinExample, minimum_z: float) -> np.ndarray:
    values = [float(minimum_z)]
    layers = example.meta.get("layers_bottom_to_top") or example.meta["physics"][
        "layers_bottom_to_top"
    ]
    for layer in layers:
        values.append(values[-1] + float(layer["thickness_m"]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (10,):
        raise ValueError(f"{example.sample_id}: expected nine-layer boundaries")
    return result


@dataclass(frozen=True)
class FullFieldGeometry:
    """Shared solver geometry loaded without reading temperature labels."""

    path: Path
    coords: np.ndarray
    control_volume: np.ndarray
    layer_id: np.ndarray
    sample_ids: tuple[str, ...]
    split_roles: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "FullFieldGeometry":
        archive_path = Path(path).resolve()
        with h5py.File(archive_path, "r") as archive:
            coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
            control_volume = np.asarray(
                archive["shared/control_volume_m3"][:], dtype=np.float64
            )
            layer_id = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
            sample_ids = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in archive["samples/sample_id"][:]
            )
            split_roles = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in archive["samples/split_role"][:]
            )
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("full-field coordinates must have shape [M,3]")
        if control_volume.shape != (len(coords),) or np.any(control_volume <= 0.0):
            raise ValueError("full-field control-volume contract is invalid")
        if layer_id.shape != (len(coords),):
            raise ValueError("full-field layer-id contract is invalid")
        if len(sample_ids) != len(split_roles) or len(sample_ids) != 1024:
            raise ValueError("full-field sample population contract is invalid")
        return cls(
            path=archive_path,
            coords=coords,
            control_volume=control_volume,
            layer_id=layer_id,
            sample_ids=sample_ids,
            split_roles=split_roles,
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": checkpoint_file_sha256(self.path),
            "node_count": int(len(self.coords)),
            "coords_sha256": array_sha256(self.coords),
            "control_volume_sha256": array_sha256(self.control_volume),
            "layer_id_sha256": array_sha256(self.layer_id),
            "sample_population": len(self.sample_ids),
        }

    def valid_truth(self, sample_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Read only requested valid_iid labels for post-inference metrics."""

        positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        result: dict[str, np.ndarray] = {}
        with h5py.File(self.path, "r") as archive:
            for sample_id in sample_ids:
                index = positions.get(str(sample_id))
                if index is None:
                    raise KeyError(f"unknown full-field sample: {sample_id}")
                if self.split_roles[index] != "valid_iid":
                    raise ValueError(f"refusing non-valid_iid label: {sample_id}")
                result[str(sample_id)] = np.asarray(
                    archive["samples/deltaT_K"][index], dtype=np.float64
                )
        return result


@dataclass(frozen=True)
class SupportArtifact:
    """One pre-existing source-aware support artifact."""

    path: Path
    selected_indices: np.ndarray
    operator_control_volume: np.ndarray
    k_xyz: np.ndarray
    q_W_m3: np.ndarray
    layer_id: np.ndarray
    sha256: str

    @classmethod
    def from_arrays(
        cls,
        *,
        selected_indices: np.ndarray,
        operator_control_volume: np.ndarray,
        k_xyz: np.ndarray,
        q_W_m3: np.ndarray,
        layer_id: np.ndarray,
        path: str | Path = "<in-memory-compatibility-fixture>",
        sha256: str = "<in-memory>",
    ) -> "SupportArtifact":
        """Materialize a label-independent support fixture without writing it."""

        selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        operator_cv = np.asarray(operator_control_volume, dtype=np.float64).reshape(-1)
        conductivity = np.asarray(k_xyz, dtype=np.float64)
        source = np.asarray(q_W_m3, dtype=np.float64).reshape(-1)
        layers = np.asarray(layer_id, dtype=np.int32).reshape(-1)
        resolution = len(selected)
        if (
            len(np.unique(selected)) != resolution
            or operator_cv.shape != (resolution,)
            or conductivity.shape != (resolution, 3)
            or source.shape != (resolution,)
            or layers.shape != (resolution,)
            or np.any(operator_cv <= 0.0)
            or np.any(conductivity <= 0.0)
            or np.any(source < 0.0)
        ):
            raise ValueError("in-memory support artifact shape/value contract is invalid")
        return cls(
            path=Path(path),
            selected_indices=selected,
            operator_control_volume=operator_cv,
            k_xyz=conductivity,
            q_W_m3=source,
            layer_id=layers,
            sha256=str(sha256),
        )

    @classmethod
    def load(cls, path: str | Path, *, expected_resolution: int | None = None) -> "SupportArtifact":
        support_path = Path(path).resolve()
        with np.load(support_path, allow_pickle=False) as payload:
            required = {
                "selected_indices",
                "operator_control_volume",
                "k_xyz",
                "q_W_m3",
                "layer_id",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"support artifact is missing fields: {missing}")
            selected = np.asarray(payload["selected_indices"], dtype=np.int64).reshape(-1)
            operator_cv = np.asarray(
                payload["operator_control_volume"], dtype=np.float64
            ).reshape(-1)
            k_xyz = np.asarray(payload["k_xyz"], dtype=np.float64)
            q = np.asarray(payload["q_W_m3"], dtype=np.float64).reshape(-1)
            layer = np.asarray(payload["layer_id"], dtype=np.int32).reshape(-1)
        resolution = len(selected)
        if expected_resolution is not None and resolution != int(expected_resolution):
            raise ValueError(
                f"support resolution mismatch: observed={resolution} "
                f"expected={expected_resolution}"
            )
        if (
            len(np.unique(selected)) != resolution
            or operator_cv.shape != (resolution,)
            or k_xyz.shape != (resolution, 3)
            or q.shape != (resolution,)
            or layer.shape != (resolution,)
            or np.any(operator_cv <= 0.0)
            or np.any(k_xyz <= 0.0)
            or np.any(q < 0.0)
        ):
            raise ValueError("support artifact shape/value contract is invalid")
        return cls(
            path=support_path,
            selected_indices=selected,
            operator_control_volume=operator_cv,
            k_xyz=k_xyz,
            q_W_m3=q,
            layer_id=layer,
            sha256=checkpoint_file_sha256(support_path),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "resolution": int(len(self.selected_indices)),
            "selected_indices_sha256": array_sha256(self.selected_indices.astype(np.int32)),
            "k_sha256": array_sha256(self.k_xyz),
            "q_sha256": array_sha256(self.q_W_m3),
            "operator_control_volume_sha256": array_sha256(self.operator_control_volume),
            "layer_id_sha256": array_sha256(self.layer_id),
        }


@dataclass(frozen=True)
class GraphBuildRecord:
    metadata: Any
    audit: dict[str, Any]


@dataclass(frozen=True)
class HighNCase:
    resolution: int
    anchor: V6DualRobinExample
    example: V6DualRobinExample
    support: SupportArtifact
    group: dict[str, Any]
    graph: GraphBuildRecord


@dataclass
class HighNRuntime:
    """Explicit high-N runtime sharing the V7 checkpoint/session semantics."""

    session: RuntimeSession
    geometry: FullFieldGeometry
    graph_config: dict[str, Any]
    graph_builder_fingerprint: str

    @classmethod
    def from_session(
        cls,
        session: RuntimeSession,
        geometry: FullFieldGeometry,
        *,
        graph_builder_fingerprint: str | None = None,
    ) -> "HighNRuntime":
        graph_config = dict(session.graph_config)
        graph_config["discrete_graph_backend"] = SPARSE_GRAPH_BACKEND
        configured_session = replace(
            session,
            graph_config=graph_config,
            group_builder=GroupBuilder(
                feature_transform=session.feature_transform,
                graph_config=graph_config,
                graph_seed=int(session.run_config.get("graph_seed", 0)),
            ),
        )
        return cls(
            session=configured_session,
            geometry=geometry,
            graph_config=graph_config,
            graph_builder_fingerprint=(
                graph_builder_fingerprint or graph_builder_code_fingerprint()
            ),
        )

    def anchor_support(self, anchor: V6DualRobinExample) -> SupportArtifact:
        if len(anchor.condition.coords) != TRAINING_ANCHOR_COUNT:
            raise ValueError(f"{anchor.sample_id}: expected 1024 anchor points")
        distance, indices = cKDTree(self.geometry.coords).query(
            np.asarray(anchor.condition.coords, dtype=np.float64), k=1
        )
        if float(np.max(distance)) > 1.0e-14 or len(np.unique(indices)) != TRAINING_ANCHOR_COUNT:
            raise ValueError(f"{anchor.sample_id}: anchor is not an exact full-mesh subset")
        features = np.asarray(anchor.condition.condition_features, dtype=np.float64)
        return SupportArtifact(
            path=Path("<anchor-embedded>"),
            selected_indices=np.asarray(indices, dtype=np.int64),
            operator_control_volume=np.asarray(anchor.operator_point_weights, dtype=np.float64),
            k_xyz=features[:, :3].copy(),
            q_W_m3=features[:, 3].copy(),
            layer_id=self.geometry.layer_id[np.asarray(indices, dtype=np.int64)],
            sha256="<anchor-embedded>",
        )

    def graph_config_for_resolution(self, resolution: int) -> dict[str, Any]:
        """Resolve the frozen E graph policy for a conditioning/query pair.

        V6 uses factor four for the native 1024 conditioning graph and
        resolution/256 for a direct E query.  Keeping this route policy
        explicit prevents a high-resolution E query from accidentally reusing
        the native regional mesh.  U-v2 has its own stable runtime because its
        output query reuses the native regional graph.
        """

        resolved = dict(self.graph_config)
        if int(resolution) > TRAINING_ANCHOR_COUNT:
            resolved.update(
                {
                    "subsample_factor": float(int(resolution) / 256.0),
                    "reuse_exact_p2r_for_r2p": True,
                }
            )
        return resolved

    @staticmethod
    def compatible_edge_targets(
        metadata: Any,
        edge_targets: Mapping[str, int | None] | None,
    ) -> dict[str, int | None] | None:
        """Keep registered capacities while omitting intentionally absent families."""

        if edge_targets is None:
            return None
        return {
            field: (edge_targets.get(field) if getattr(metadata, field) is not None else None)
            for field in EDGE_FIELDS
        }

    def query_example(
        self,
        anchor: V6DualRobinExample,
        support: SupportArtifact,
    ) -> V6DualRobinExample:
        indices = np.asarray(support.selected_indices, dtype=np.int64)
        coords = self.geometry.coords[indices]
        count = len(indices)
        flags = _boundary_flags(self.geometry.coords)[indices]
        features = np.column_stack(
            (
                support.k_xyz,
                support.q_W_m3,
                flags,
                np.full(count, float(anchor.meta["top_h_W_m2K"])),
                np.full(count, float(anchor.meta["bottom_h_W_m2K"])),
                np.zeros(count),
            )
        )
        if features.shape != (count, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
            raise ValueError(f"{anchor.sample_id}: query feature schema drifted")
        meta = deepcopy(anchor.meta)
        meta["v6_adapter"] = dict(meta["v6_adapter"])
        meta["v6_adapter"]["operator_point_measure"] = (
            "same_layer_nearest_solver_cv_partition_v1"
        )
        return V6DualRobinExample(
            sample_id=anchor.sample_id,
            condition=V1SteadyConditionInput(
                coords=coords,
                condition_features=features,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(
                target_u=np.full((count, 1), REFERENCE_K, dtype=np.float64)
            ),
            meta=meta,
            operator_point_weights=support.operator_control_volume.copy(),
        )

    def graph_metadata(
        self,
        example: V6DualRobinExample,
        *,
        support_hash: str,
        graph_config: Mapping[str, Any] | None = None,
        cache_dir: str | Path | None = None,
        write_cache: bool = False,
    ) -> GraphBuildRecord:
        if len(support_hash) != 64:
            raise ValueError("support_hash must be a SHA256 value")
        resolved_graph_config = dict(graph_config or self.graph_config)
        builder = Heat3DGraphBuilder(**resolved_graph_config)
        key_payload = cache_key_payload(
            support_hash=support_hash,
            graph_config=dict(builder.config),
            graph_seed=int(self.session.run_config.get("graph_seed", 0)),
            graph_builder_fingerprint=self.graph_builder_fingerprint,
        )
        key = cache_key(key_payload)
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir).resolve() / (
                f"{example.sample_id}_{len(example.condition.coords)}_{key}.npz"
            )
        graph_coords = self.session.feature_transform.transform(example).graph_coords
        cache_hit = bool(cache_path is not None and cache_path.is_file())
        fresh = None
        save_audit = None
        if cache_hit:
            metadata, load_audit = load_metadata(cache_path)
        else:
            metadata = builder.build_metadata(
                graph_coords,
                key=jax.random.PRNGKey(int(self.session.run_config.get("graph_seed", 0))),
            )
            load_audit = None
            if write_cache:
                if cache_path is None:
                    raise ValueError("write_cache requires cache_dir")
                save_audit = save_metadata(cache_path, metadata)
        if cache_hit:
            fresh = builder.build_metadata(
                graph_coords,
                key=jax.random.PRNGKey(int(self.session.run_config.get("graph_seed", 0))),
            )
        fresh_metadata_hash = None if fresh is None else metadata_hash(fresh)
        observed_metadata_hash = metadata_hash(metadata)
        fresh_graph_hash = None
        if fresh is not None:
            fresh_graph_hash = graph_hash(builder.build_graphs(fresh))
        observed_graph_hash = graph_hash(builder.build_graphs(metadata))
        hash_exact = fresh is None or (
            fresh_metadata_hash == observed_metadata_hash
            and fresh_graph_hash == observed_graph_hash
        )
        if not hash_exact:
            raise ValueError(f"{example.sample_id}: cached/fresh graph hash mismatch")
        audit = {
            "cache_key": key,
            "cache_key_payload": key_payload,
            "cache_file": None if cache_path is None else str(cache_path),
            "cache_file_sha256": (
                None if cache_path is None or not cache_path.is_file() else file_sha256(cache_path)
            ),
            "cache_hit": cache_hit,
            "cache_written": save_audit is not None,
            "metadata_hash": observed_metadata_hash,
            "graph_hash": observed_graph_hash,
            "fresh_metadata_hash": fresh_metadata_hash,
            "fresh_graph_hash": fresh_graph_hash,
            "cached_fresh_hash_exact": hash_exact,
            "edge_counts": {
                field: (
                    None
                    if getattr(metadata, field) is None
                    else int(getattr(metadata, field).shape[1])
                )
                for field in EDGE_FIELDS
            },
        }
        return GraphBuildRecord(metadata=metadata, audit=audit)

    def build_case(
        self,
        anchor: V6DualRobinExample,
        resolution: int,
        *,
        support_path: str | Path | None = None,
        edge_targets: dict[str, int | None] | None = None,
        cache_dir: str | Path | None = None,
        write_cache: bool = False,
    ) -> HighNCase:
        resolution = int(resolution)
        if resolution == TRAINING_ANCHOR_COUNT:
            support = self.anchor_support(anchor)
            example = anchor
        else:
            if support_path is None:
                raise FileNotFoundError(
                    f"missing pre-existing support artifact for resolution={resolution}"
                )
            support = SupportArtifact.load(support_path, expected_resolution=resolution)
            example = self.query_example(anchor, support)
        return self.build_case_from_support(
            anchor,
            resolution,
            support=support,
            edge_targets=edge_targets,
            cache_dir=cache_dir,
            write_cache=write_cache,
        )

    def build_case_from_support(
        self,
        anchor: V6DualRobinExample,
        resolution: int,
        *,
        support: SupportArtifact,
        edge_targets: dict[str, int | None] | None = None,
        cache_dir: str | Path | None = None,
        write_cache: bool = False,
    ) -> HighNCase:
        """Build from an already-materialized support, including temp fixtures."""

        resolution = int(resolution)
        if resolution != len(support.selected_indices):
            raise ValueError(
                f"support resolution mismatch: support={len(support.selected_indices)} "
                f"requested={resolution}"
            )
        if resolution == TRAINING_ANCHOR_COUNT:
            expected = self.anchor_support(anchor)
            if not np.array_equal(support.selected_indices, expected.selected_indices):
                raise ValueError("native support does not match the anchor subset")
            example = anchor
        else:
            example = self.query_example(anchor, support)
        support_hash = array_sha256(support.selected_indices.astype(np.int32))
        graph = self.graph_metadata(
            example,
            support_hash=support_hash,
            graph_config=self.graph_config_for_resolution(resolution),
            cache_dir=cache_dir,
            write_cache=write_cache,
        )
        group_session = self.session
        query_graph_config = self.graph_config_for_resolution(resolution)
        if query_graph_config != self.session.graph_config:
            group_session = replace(
                self.session,
                graph_config=query_graph_config,
                group_builder=GroupBuilder(
                    feature_transform=self.session.feature_transform,
                    graph_config=query_graph_config,
                    graph_seed=int(self.session.run_config.get("graph_seed", 0)),
                ),
            )
        group = group_session.build_group_from_metadata(
            [example],
            graph.metadata,
            name=f"v7_high_n_valid_iid_{resolution}",
            edge_targets=self.compatible_edge_targets(graph.metadata, edge_targets),
            context_examples=[anchor],
        )
        group["support_indices_sha256"] = support_hash
        group["graph_tensors_sha256"] = _tree_sha256(group["graphs"])
        return HighNCase(
            resolution=resolution,
            anchor=anchor,
            example=example,
            support=support,
            group=group,
            graph=graph,
        )

    def reconstruction(
        self,
        case: HighNCase,
    ) -> tuple[ReconstructionMap, dict[str, Any]]:
        boundaries = _boundaries(case.anchor, float(np.min(self.geometry.coords[:, 2])))
        mapping, audit = build_reconstruction_map(
            coords=self.geometry.coords,
            layer_id=self.geometry.layer_id,
            boundaries=boundaries,
            support_indices=case.support.selected_indices.astype(np.int32),
            empty_domain_fallback="same_layer",
        )
        audit = dict(audit)
        audit.update(
            {
                "mapping_hash": _mapping_sha256(mapping),
                "support_hash": array_sha256(
                    case.support.selected_indices.astype(np.int32)
                ),
                "boundary_hash": array_sha256(boundaries),
            }
        )
        return mapping, audit

    @staticmethod
    def apply_anchor_scale(
        raw_temperature: np.ndarray,
        anchor_scale: float,
        operator_weights: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(raw_temperature, dtype=np.float64)
        delta = raw - REFERENCE_K
        weights = np.asarray(operator_weights, dtype=np.float64).reshape(-1)
        normalized = weights / float(np.sum(weights))
        query_scale = math.sqrt(float(np.sum(normalized * delta * delta)))
        if not np.isfinite(query_scale) or query_scale <= 0.0 or anchor_scale <= 0.0:
            raise ValueError("non-positive/non-finite shape-scale reconstruction")
        return REFERENCE_K + delta / query_scale * float(anchor_scale)

    def metric_accumulator(self, anchor: V6DualRobinExample) -> FullFieldMetricAccumulator:
        """Return the existing V6 metric accumulator for valid_iid reporting."""

        return FullFieldMetricAccumulator(
            control_volume=self.geometry.control_volume,
            layer_id=self.geometry.layer_id,
            boundaries=_boundaries(
                anchor,
                float(np.min(self.geometry.coords[:, 2])),
            ),
            coords=self.geometry.coords,
        )
