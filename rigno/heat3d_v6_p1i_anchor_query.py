"""Checkpoint-preserving anchor/query adapter for sample-varying P1i."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput
from rigno.heat3d_v6_dataset import V6DualRobinExample


TRAINING_ANCHOR_COUNT = 1024
HIGH_N_SELECTION_SEED = 20260808
HIGH_N_STRATUM_FRACTIONS = {
    "source": 0.35,
    "interface": 0.15,
    "robin": 0.10,
    "volume": 0.40,
}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorPayload:
    sample_id: str
    coords: np.ndarray
    condition_features: np.ndarray
    operator_point_weights: np.ndarray
    condition_feature_names: tuple[str, ...]
    k_encoding_mode: str
    support_hashes: Mapping[str, str]


class P1iSampleVaryingAnchorQueryAdapter:
    """Copy one frozen P1i sample without changing any R0 model input."""

    def __init__(self, example: V6DualRobinExample) -> None:
        coords = np.asarray(example.condition.coords)
        features = np.asarray(example.condition.condition_features)
        weights = np.asarray(example.operator_point_weights)
        if coords.shape != (TRAINING_ANCHOR_COUNT, 3):
            raise ValueError(f"{example.sample_id}: invalid anchor shape {coords.shape}")
        if features.shape[0] != TRAINING_ANCHOR_COUNT:
            raise ValueError(f"{example.sample_id}: feature/support count drift")
        if weights.reshape(-1).shape != (TRAINING_ANCHOR_COUNT,):
            raise ValueError(f"{example.sample_id}: control-volume/support count drift")
        self._example = example
        self.anchor = AnchorPayload(
            sample_id=example.sample_id,
            coords=coords.copy(),
            condition_features=features.copy(),
            operator_point_weights=weights.copy(),
            condition_feature_names=tuple(example.condition.condition_feature_names),
            k_encoding_mode=str(example.condition.k_encoding_mode),
            support_hashes={
                "coords": array_sha256(coords),
                "condition_features": array_sha256(features),
                "operator_point_weights": array_sha256(weights),
            },
        )

    def r0_example(self) -> V6DualRobinExample:
        """Return an independent, byte-equivalent 1024 anchor/query example."""
        meta: dict[str, Any] = deepcopy(self._example.meta)
        meta["p1i_anchor_query_adapter"] = {
            "mode": "R0_exact_anchor_query_identity",
            "anchor_count": TRAINING_ANCHOR_COUNT,
            "target_or_label_used": False,
            "support_hashes": dict(self.anchor.support_hashes),
        }
        return V6DualRobinExample(
            sample_id=self.anchor.sample_id,
            condition=V1SteadyConditionInput(
                coords=self.anchor.coords.copy(),
                condition_features=self.anchor.condition_features.copy(),
                condition_feature_names=self.anchor.condition_feature_names,
                k_encoding_mode=self.anchor.k_encoding_mode,
            ),
            # Required by the legacy group container, but not read by adapter.
            target=self._example.target,
            meta=meta,
            operator_point_weights=self.anchor.operator_point_weights.copy(),
        )

    def r0_input_equivalence(self, adapted: V6DualRobinExample) -> dict[str, Any]:
        observed = {
            "coords": np.asarray(adapted.condition.coords),
            "condition_features": np.asarray(adapted.condition.condition_features),
            "operator_point_weights": np.asarray(adapted.operator_point_weights),
        }
        expected = {
            "coords": self.anchor.coords,
            "condition_features": self.anchor.condition_features,
            "operator_point_weights": self.anchor.operator_point_weights,
        }
        rows = {}
        for name in expected:
            exact = bool(np.array_equal(expected[name], observed[name]))
            rows[name] = {
                "exact": exact,
                "reference_sha256": array_sha256(expected[name]),
                "adapter_sha256": array_sha256(observed[name]),
                "max_abs_error": float(np.max(np.abs(
                    np.asarray(expected[name], dtype=np.float64)
                    - np.asarray(observed[name], dtype=np.float64)
                ))),
            }
        order_exact = adapted.sample_id == self.anchor.sample_id
        schema_exact = (
            tuple(adapted.condition.condition_feature_names)
            == self.anchor.condition_feature_names
            and adapted.condition.k_encoding_mode == self.anchor.k_encoding_mode
        )
        return {
            "passed": bool(all(row["exact"] for row in rows.values()) and order_exact and schema_exact),
            "sample_id_order_exact": order_exact,
            "feature_schema_exact": schema_exact,
            "arrays": rows,
        }


def _hash_order(
    sample_id: str,
    seed: int,
    indices: np.ndarray,
    profile: dict[str, float] | None = None,
) -> np.ndarray:
    hash_started = time.perf_counter()
    decorated = [
        (hashlib.sha256(f"{seed}:{sample_id}:{index}".encode()).digest(), int(index))
        for index in map(int, indices)
    ]
    hash_seconds = time.perf_counter() - hash_started
    sort_started = time.perf_counter()
    decorated.sort(key=lambda item: (item[0], item[1]))
    sort_seconds = time.perf_counter() - sort_started
    if profile is not None:
        profile["sha256_seconds"] = profile.get("sha256_seconds", 0.0) + hash_seconds
        profile["sort_seconds"] = profile.get("sort_seconds", 0.0) + sort_seconds
    return np.fromiter((item[1] for item in decorated), dtype=np.int64, count=len(decorated))


def _weighted_interleave(buckets: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Deterministically interleave finite queues by largest quota deficit."""
    # Keep the frozen queue contents and tie-breaking semantics, but advance a
    # cursor instead of repeatedly shifting a Python list with ``pop(0)``.
    # The latter is quadratic in each bucket length and dominates high-N
    # support preparation even though no scientific work is being performed.
    queues = {name: tuple(map(int, values)) for name, values in buckets.items()}
    names = tuple(sorted(queues))
    rank = {name: index for index, name in enumerate(names)}
    cursor = {name: 0 for name in queues}
    consumed = {name: 0 for name in queues}
    result: list[int] = []
    remaining = sum(len(values) for values in queues.values())
    while remaining:
        active = [name for name in names if cursor[name] < len(queues[name])]
        total_weight = sum(float(weights[name]) for name in active)
        step = len(result) + 1
        chosen = max(
            active,
            key=lambda name: (
                float(weights[name]) / total_weight * step - consumed[name],
                -rank[name],
            ),
        )
        result.append(queues[chosen][cursor[chosen]])
        cursor[chosen] += 1
        consumed[chosen] += 1
        remaining -= 1
    return np.asarray(result, dtype=np.int64)


@dataclass(frozen=True)
class NestedQueryGeometryCache:
    """Label-independent static geometry used by lazy high-N selection."""

    coords: np.ndarray
    control_volume: np.ndarray
    layer_id: np.ndarray
    layer_indices: tuple[np.ndarray, ...]
    top_indices: np.ndarray
    bottom_indices: np.ndarray
    interface_indices: np.ndarray
    layer_boundaries_m: np.ndarray
    static_hashes: Mapping[str, str]


def prepare_nested_query_geometry_cache(
    *,
    full_coords: np.ndarray,
    full_control_volume: np.ndarray,
    full_layer_id: np.ndarray,
    layer_boundaries_m: np.ndarray,
) -> NestedQueryGeometryCache:
    """Freeze reusable layer/surface/interface partitions without labels."""

    coords = np.asarray(full_coords, dtype=np.float64)
    cv = np.asarray(full_control_volume, dtype=np.float64).reshape(-1)
    layer = np.asarray(full_layer_id, dtype=np.int32).reshape(-1)
    boundaries = np.asarray(layer_boundaries_m, dtype=np.float64).reshape(-1)
    if len(coords) != len(cv) or len(coords) != len(layer) or np.any(cv <= 0.0):
        raise ValueError("invalid nested-query geometry arrays")
    z = coords[:, 2]
    internal = boundaries[1:-1]
    interface = np.flatnonzero(
        np.any(np.isclose(z[:, None], internal[None, :], atol=1.0e-15), axis=1)
    )
    top = np.flatnonzero(np.isclose(z, np.max(z), atol=1.0e-15))
    bottom = np.flatnonzero(np.isclose(z, np.min(z), atol=1.0e-15))
    layer_indices = tuple(
        np.flatnonzero(layer == layer_value)
        for layer_value in sorted(map(int, np.unique(layer)))
    )
    return NestedQueryGeometryCache(
        coords=coords,
        control_volume=cv,
        layer_id=layer,
        layer_indices=layer_indices,
        top_indices=top,
        bottom_indices=bottom,
        interface_indices=interface,
        layer_boundaries_m=boundaries,
        static_hashes={
            "coords": array_sha256(coords),
            "control_volume": array_sha256(cv),
            "layer_id": array_sha256(layer),
            "top_indices": array_sha256(top),
            "bottom_indices": array_sha256(bottom),
            "interface_indices": array_sha256(interface),
            "layer_boundaries_m": array_sha256(boundaries),
        },
    )


class _LazyWeightedInterleave:
    """Exact deficit-round-robin cursor with bounded prefix emission."""

    def __init__(
        self,
        buckets: Mapping[str, np.ndarray | "_LazyWeightedInterleave"],
        weights: Mapping[str, float],
        *,
        profile: dict[str, float] | None = None,
        profile_key: str | None = None,
    ) -> None:
        self._buckets = dict(buckets)
        self._names = tuple(sorted(self._buckets))
        self._rank = {name: index for index, name in enumerate(self._names)}
        self._cursor = {name: 0 for name in self._names}
        self._consumed = {name: 0 for name in self._names}
        self._weights = {name: float(weights[name]) for name in self._names}
        self._emitted = 0
        self._remaining_by_name = {
            name: (
                self._buckets[name].remaining
                if isinstance(self._buckets[name], _LazyWeightedInterleave)
                else len(self._buckets[name])
            )
            for name in self._names
        }
        self._remaining = sum(self._remaining_by_name.values())
        self._active = tuple(
            name for name in self._names if self._remaining_by_name[name] > 0
        )
        self._total_weight = sum(self._weights[name] for name in self._active)
        self._profile = profile
        self._profile_key = profile_key

    @property
    def remaining(self) -> int:
        return self._remaining

    def pop_next(self) -> int:
        started = time.perf_counter() if self._profile is not None else 0.0
        if not self._active:
            raise StopIteration
        step = self._emitted + 1
        chosen = max(
            self._active,
            key=lambda name: (
                self._weights[name] / self._total_weight * step - self._consumed[name],
                -self._rank[name],
            ),
        )
        bucket = self._buckets[chosen]
        if isinstance(bucket, _LazyWeightedInterleave):
            value = bucket.pop_next()
        else:
            value = int(bucket[self._cursor[chosen]])
            self._cursor[chosen] += 1
        self._consumed[chosen] += 1
        self._emitted += 1
        self._remaining_by_name[chosen] -= 1
        self._remaining -= 1
        if self._remaining_by_name[chosen] == 0:
            self._active = tuple(
                name for name in self._active if self._remaining_by_name[name] > 0
            )
            # Recompute in the same sorted-name order used by the historical
            # implementation. Subtraction would change floating-point order.
            self._total_weight = sum(self._weights[name] for name in self._active)
        if self._profile is not None and self._profile_key is not None:
            self._profile[self._profile_key] = (
                self._profile.get(self._profile_key, 0.0)
                + (time.perf_counter() - started)
            )
        return value

    def take(self, count: int) -> np.ndarray:
        if count < 0 or count > self.remaining:
            raise ValueError("lazy interleave prefix count is out of bounds")
        return np.fromiter((self.pop_next() for _ in range(count)), dtype=np.int64, count=count)


def deterministic_nested_query_prefix(
    *,
    sample_id: str,
    anchor_indices: np.ndarray,
    full_q: np.ndarray,
    target_count: int,
    geometry_cache: NestedQueryGeometryCache,
    selection_seed: int = HIGH_N_SELECTION_SEED,
    profile: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact historical prefix without materializing the full order."""

    coords = geometry_cache.coords
    cv = geometry_cache.control_volume
    layer = geometry_cache.layer_id
    q = np.asarray(full_q, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    node_count = len(coords)
    if target_count < len(anchors) or target_count > node_count:
        raise ValueError("target_count must lie between anchor and solver node counts")
    if len(q) != node_count or len(anchors) != TRAINING_ANCHOR_COUNT:
        raise ValueError("lazy prefix inputs do not match frozen P1i sizes")
    mask_started = time.perf_counter()
    available = np.ones(node_count, dtype=bool)
    available[anchors] = False
    q_eps = max(1.0e-30, float(np.max(np.abs(q))) * 1.0e-12)
    source = available & (q > q_eps)
    interface_static = np.zeros(node_count, dtype=bool)
    interface_static[geometry_cache.interface_indices] = True
    robin_static = np.zeros(node_count, dtype=bool)
    robin_static[geometry_cache.top_indices] = True
    robin_static[geometry_cache.bottom_indices] = True
    interface = available & ~source & interface_static
    robin = available & ~source & ~interface & robin_static
    volume = available & ~source & ~interface & ~robin
    masks = {"source": source, "interface": interface, "robin": robin, "volume": volume}
    if profile is not None:
        profile.clear()
        profile["mask_seconds"] = time.perf_counter() - mask_started
        profile["sha256_seconds"] = 0.0
        profile["sort_seconds"] = 0.0
        profile["inner_interleave_seconds"] = 0.0
    strata: dict[str, _LazyWeightedInterleave] = {}
    counts: dict[str, int] = {}
    for name, mask in masks.items():
        layer_buckets: dict[str, np.ndarray] = {}
        layer_weights: dict[str, float] = {}
        for layer_index, static_indices in enumerate(geometry_cache.layer_indices):
            selected = static_indices[mask[static_indices]]
            if not len(selected):
                continue
            key = f"layer_{layer_index:02d}"
            layer_buckets[key] = _hash_order(
                sample_id, selection_seed, selected, profile=profile
            )
            layer_weights[key] = float(np.sum(cv[selected]))
        if layer_buckets:
            strata[name] = _LazyWeightedInterleave(
                layer_buckets, layer_weights,
                profile=profile, profile_key="inner_interleave_seconds",
            )
        counts[name] = int(np.sum(mask))
    outer = _LazyWeightedInterleave(strata, HIGH_N_STRATUM_FRACTIONS)
    interleave_started = time.perf_counter()
    added = outer.take(target_count - len(anchors))
    total_interleave_seconds = time.perf_counter() - interleave_started
    if profile is not None:
        profile["outer_interleave_seconds"] = max(
            0.0, total_interleave_seconds - profile["inner_interleave_seconds"]
        )
    prefix = np.concatenate((anchors, added))
    if len(prefix) != target_count or len(np.unique(prefix)) != target_count:
        raise RuntimeError("lazy nested-query prefix is not unique or complete")
    return prefix, {
        "algorithm": "anchored_stratified_deficit_round_robin_lazy_prefix_v1",
        "selection_seed": int(selection_seed),
        "target_count": int(target_count),
        "anchor_count": int(len(anchors)),
        "anchor_order_preserved": bool(np.array_equal(prefix[:len(anchors)], anchors)),
        "stratum_fractions": dict(HIGH_N_STRATUM_FRACTIONS),
        "stratum_candidate_counts": counts,
        "inner_and_outer_early_stop": True,
        "geometry_static_hashes": dict(geometry_cache.static_hashes),
        "profile_seconds": dict(profile) if profile is not None else None,
        "target_or_temperature_used": False,
        "prefix_sha256": array_sha256(prefix),
    }


def deterministic_nested_query_order(
    *,
    sample_id: str,
    anchor_indices: np.ndarray,
    full_coords: np.ndarray,
    full_control_volume: np.ndarray,
    full_layer_id: np.ndarray,
    full_q: np.ndarray,
    layer_boundaries_m: np.ndarray,
    selection_seed: int = HIGH_N_SELECTION_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one label-independent order whose prefixes define every high N.

    The exact original anchors always lead the order. Added solver nodes use
    exclusive source/interface/Robin/volume strata, deterministic hash order,
    layer-volume interleaving, and a documented exhausted-stratum fallback.
    """
    coords = np.asarray(full_coords, dtype=np.float64)
    cv = np.asarray(full_control_volume, dtype=np.float64).reshape(-1)
    layer = np.asarray(full_layer_id, dtype=np.int32).reshape(-1)
    q = np.asarray(full_q, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    node_count = len(coords)
    if len(anchors) != TRAINING_ANCHOR_COUNT or len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchor indices must be the unique original ordered 1024 nodes")
    if any(len(value) != node_count for value in (cv, layer, q)):
        raise ValueError("full-field coordinate/kind arrays have inconsistent counts")
    if np.any(anchors < 0) or np.any(anchors >= node_count) or np.any(cv <= 0.0):
        raise ValueError("invalid solver indices or control volumes")
    available = np.ones(node_count, dtype=bool)
    available[anchors] = False
    z = coords[:, 2]
    q_eps = max(1.0e-30, float(np.max(np.abs(q))) * 1.0e-12)
    source = available & (q > q_eps)
    internal = np.asarray(layer_boundaries_m, dtype=np.float64).reshape(-1)[1:-1]
    interface = available & ~source
    interface &= np.any(np.isclose(z[:, None], internal[None, :], atol=1.0e-15), axis=1)
    robin = available & ~source & ~interface
    robin &= np.isclose(z, np.min(z), atol=1.0e-15) | np.isclose(z, np.max(z), atol=1.0e-15)
    volume = available & ~source & ~interface & ~robin
    masks = {"source": source, "interface": interface, "robin": robin, "volume": volume}
    stratum_sequences: dict[str, np.ndarray] = {}
    stratum_counts = {}
    for name, mask in masks.items():
        layer_buckets, layer_weights = {}, {}
        for layer_id in sorted(map(int, np.unique(layer[mask]))):
            selected = np.flatnonzero(mask & (layer == layer_id))
            key = f"layer_{layer_id:02d}"
            layer_buckets[key] = _hash_order(sample_id, selection_seed, selected)
            layer_weights[key] = float(np.sum(cv[selected]))
        stratum_sequences[name] = (
            _weighted_interleave(layer_buckets, layer_weights)
            if layer_buckets else np.empty(0, dtype=np.int64)
        )
        stratum_counts[name] = int(np.sum(mask))
    added = _weighted_interleave(stratum_sequences, HIGH_N_STRATUM_FRACTIONS)
    order = np.concatenate((anchors, added))
    if len(order) != node_count or len(np.unique(order)) != node_count:
        raise RuntimeError("nested query order is not a permutation of solver nodes")
    return order, {
        "algorithm": "anchored_stratified_deficit_round_robin_v1",
        "selection_seed": int(selection_seed),
        "anchor_count": int(len(anchors)),
        "anchor_order_preserved": bool(np.array_equal(order[:len(anchors)], anchors)),
        "stratum_fractions": dict(HIGH_N_STRATUM_FRACTIONS),
        "stratum_candidate_counts": stratum_counts,
        "within_stratum_order": "per-layer SHA256(seed:sample_id:solver_index), volume-weighted deficit interleave",
        "fallback": "exhausted strata/layers are removed and remaining weights renormalized",
        "target_or_temperature_used": False,
        "order_sha256": array_sha256(order),
    }


def conservative_selected_control_volume(
    *,
    full_coords: np.ndarray,
    full_control_volume: np.ndarray,
    full_layer_id: np.ndarray,
    selected_indices: np.ndarray,
    query_workers: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Partition every solver CV to its nearest selected node in the same layer."""
    coords = np.asarray(full_coords, dtype=np.float64)
    cv = np.asarray(full_control_volume, dtype=np.float64).reshape(-1)
    layer = np.asarray(full_layer_id, dtype=np.int32).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    result = np.zeros(len(selected), dtype=np.float64)
    for layer_id in sorted(map(int, np.unique(layer))):
        full_local = np.flatnonzero(layer == layer_id)
        support_local = np.flatnonzero(layer[selected] == layer_id)
        if not len(support_local):
            raise RuntimeError(f"selected support has no node in layer {layer_id}")
        nearest = cKDTree(coords[selected[support_local]]).query(
            coords[full_local], k=1, workers=int(query_workers)
        )[1]
        np.add.at(result, support_local[np.asarray(nearest, dtype=np.int64)], cv[full_local])
    if np.any(result <= 0.0):
        raise RuntimeError("selected support contains a zero-measure node")
    relative_error = abs(float(np.sum(result) - np.sum(cv))) / float(np.sum(cv))
    return result, {
        "algorithm": "same_layer_nearest_solver_cv_partition_v1",
        "full_volume_m3": float(np.sum(cv)),
        "selected_volume_m3": float(np.sum(result)),
        "relative_volume_error": relative_error,
        "label_or_temperature_used": False,
        "query_workers": int(query_workers),
        "weights_sha256": array_sha256(result),
    }
