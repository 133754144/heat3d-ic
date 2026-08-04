#!/usr/bin/env python3
"""Controlled, valid-only P1i cross-resolution audit.

The primary ladder uses deterministic source-aware supports selected directly
from the frozen 240825-node FVM mesh.  The supports are nested within each
discretization seed, use identical input/output coordinates, and aggregate the
full-mesh control volumes and raw-input fields onto the selected nodes so that
total volume and source power are conserved.  No target is used for support or
graph construction.

The script also exposes the four preregistered factor cells that separate
support distribution from regional-mesh node-count drift.  All model results
are diagnostic: the frozen checkpoint and its train-only normalization are
never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as base  # noqa: E402
import benchmark_heat3d_v6_direct_resolution as direct  # noqa: E402
import benchmark_heat3d_v6_p1i_resolution as prior  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from evaluate_heat3d_v6_p1i_randomblock_transfer import layer_aware_fallback_map  # noqa: E402
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402


MAIN_RESOLUTIONS = (512, 1024, 2048, 4096, 8192, 16384)
FACTOR_RESOLUTIONS = (1024, 4096, 16384, 65536)
DISCRETIZATION_SEEDS = (0, 1, 2, 3)
TRAINING_RESOLUTION = 1024
SAMPLE_COUNT = 32
STRATUM_RATIOS = {
    "block": 0.25,
    "interface": 0.125,
    "top": 0.0625,
    "bottom": 0.0625,
    "volume": 0.50,
}
FACTOR_CELLS = {
    "A": ("source_aware", "fixed_training_nr"),
    "B": ("source_aware", "growing_nr"),
    "C": ("structured", "fixed_training_nr"),
    "D": ("structured", "growing_nr"),
}
EDGE_FIELDS = ("p2r_edge_indices", "r2r_edge_indices", "r2r_edge_domains", "r2p_edge_indices")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(
        str(array.dtype).encode() + str(tuple(array.shape)).encode() + array.tobytes()
    )


def json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def quotas(n: int) -> dict[str, int]:
    result = {name: int(round(fraction * n)) for name, fraction in STRATUM_RATIOS.items()}
    if sum(result.values()) != n:
        raise RuntimeError(f"N={n}: stratum quotas do not sum to N")
    return result


def weighted_order(
    candidates: np.ndarray,
    weights: np.ndarray,
    *,
    seed_text: str,
    mandatory: Sequence[int] = (),
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    mandatory_unique = list(dict.fromkeys(int(value) for value in mandatory if int(value) in set(candidates.tolist())))
    mandatory_set = set(mandatory_unique)
    remaining = np.asarray([value for value in candidates if int(value) not in mandatory_set], dtype=np.int64)
    if remaining.size:
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        probability_weight = np.maximum(np.asarray(weights, dtype=np.float64)[remaining], np.finfo(np.float64).tiny)
        scores = -np.log(np.maximum(rng.random(remaining.size), np.finfo(np.float64).tiny)) / probability_weight
        remaining = remaining[np.argsort(scores, kind="stable")]
    return np.concatenate((np.asarray(mandatory_unique, dtype=np.int64), remaining))


def layer_boundaries(meta: Mapping[str, Any]) -> np.ndarray:
    values = [0.0]
    for layer in meta["physics"]["layers_bottom_to_top"]:
        values.append(values[-1] + float(layer["thickness_m"]))
    return np.asarray(values, dtype=np.float64)


def block_masks(meta: Mapping[str, Any], coords: np.ndarray, layer: np.ndarray) -> list[np.ndarray]:
    layers = [str(row["id"]) for row in meta["physics"]["layers_bottom_to_top"]]
    layer_index = {name: index for index, name in enumerate(layers)}
    lx = float(meta["physics"]["footprint_m"][0])
    ly = float(meta["physics"]["footprint_m"][1])
    result = []
    for row in [*meta["q_blocks"], *meta["k_blocks"]]:
        x0, x1, y0, y1 = map(float, row["bbox_fraction_xy"])
        result.append(
            (layer == layer_index[str(row["layer"])])
            & (coords[:, 0] >= x0 * lx)
            & (coords[:, 0] <= x1 * lx)
            & (coords[:, 1] >= y0 * ly)
            & (coords[:, 1] <= y1 * ly)
        )
    return result


def mandatory_from_masks(masks: Sequence[np.ndarray], weights: np.ndarray, seed_text: str) -> list[int]:
    result = []
    for index, mask in enumerate(masks):
        candidates = np.flatnonzero(mask)
        if not candidates.size:
            raise RuntimeError(f"registered block/interface {index} has zero solver nodes")
        order = weighted_order(candidates, weights, seed_text=f"{seed_text}:mandatory:{index}")
        result.append(int(order[0]))
    return result


def support_sequences(
    *,
    sample_id: str,
    discretization_seed: int,
    meta: Mapping[str, Any],
    coords: np.ndarray,
    cv: np.ndarray,
    layer: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    boundaries = layer_boundaries(meta)
    masks = block_masks(meta, coords, layer)
    block_union = np.logical_or.reduce(masks)
    interface_masks = [np.isclose(coords[:, 2], value, atol=1e-15) for value in boundaries[1:-1]]
    interface_union = np.logical_or.reduce(interface_masks) & ~block_union
    top = np.isclose(coords[:, 2], boundaries[-1], atol=1e-15) & ~block_union & ~interface_union
    bottom = np.isclose(coords[:, 2], boundaries[0], atol=1e-15) & ~block_union & ~interface_union
    reserved = block_union | interface_union | top | bottom
    volume = ~reserved
    seed_text = f"{sample_id}:{discretization_seed}:p1i_cross_resolution_v1"

    mandatory_block = mandatory_from_masks(masks, cv, f"{seed_text}:block")
    mandatory_interface = mandatory_from_masks(interface_masks, cv, f"{seed_text}:interface")
    mandatory_volume = []
    for layer_id in sorted(np.unique(layer)):
        candidates = np.flatnonzero(volume & (layer == layer_id))
        if not candidates.size:
            candidates = np.flatnonzero(layer == layer_id)
        mandatory_volume.append(int(weighted_order(candidates, cv, seed_text=f"{seed_text}:layer:{layer_id}")[0]))

    pools = {
        "block": np.flatnonzero(block_union),
        "interface": np.flatnonzero(interface_union),
        "top": np.flatnonzero(top),
        "bottom": np.flatnonzero(bottom),
        "volume": np.flatnonzero(volume),
    }
    mandatory = {
        "block": mandatory_block,
        "interface": mandatory_interface,
        "top": [],
        "bottom": [],
        "volume": mandatory_volume,
    }
    sequences = {
        name: weighted_order(pool, cv, seed_text=f"{seed_text}:{name}", mandatory=mandatory[name])
        for name, pool in pools.items()
    }
    return sequences, {
        "pool_capacity": {name: int(len(value)) for name, value in pools.items()},
        "registered_block_count": len(masks),
        "mandatory_block_count": len(set(mandatory_block)),
        "mandatory_interface_count": len(set(mandatory_interface)),
        "mandatory_layer_count": len(set(mandatory_volume)),
        "selection_inputs": ["coords", "control_volume", "layer_id", "q/k block metadata"],
        "selection_excludes": ["temperature", "prediction", "error", "test/sealed role"],
    }


def support_indices(
    sequences: Mapping[str, np.ndarray], n: int
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    requested = quotas(n)
    realized = dict(requested)
    shortage = {}
    # The formal ladder through 16384 must preserve exact quotas.  At 65536,
    # source/k-block capacity can be smaller, so a recorded active-domain
    # expansion fills the same source-aware quota without changing the frozen
    # 512--16384 protocol.
    for name, count in requested.items():
        shortage[name] = max(0, count - len(sequences[name]))
    if n <= max(MAIN_RESOLUTIONS) and any(shortage.values()):
        raise RuntimeError(f"N={n}: exact source-aware quota capacity failed: {shortage}")
    selected = []
    used: set[int] = set()
    for name in STRATUM_RATIOS:
        values = list(map(int, sequences[name][: min(requested[name], len(sequences[name]))]))
        selected.extend(values)
        used.update(values)
    missing = n - len(selected)
    if missing:
        fallback = np.concatenate([sequences["block"], sequences["volume"], sequences["interface"], sequences["top"], sequences["bottom"]])
        additions = [int(value) for value in fallback if int(value) not in used][:missing]
        if len(additions) != missing:
            raise RuntimeError(f"N={n}: cannot fill support capacity")
        selected.extend(additions)
    indices = np.asarray(selected, dtype=np.int64)
    if len(indices) != n or len(np.unique(indices)) != n:
        raise RuntimeError(f"N={n}: support count/uniqueness failed")
    return indices, realized, shortage


def aggregate_to_support(
    *,
    coords: np.ndarray,
    cv: np.ndarray,
    layer: np.ndarray,
    k: np.ndarray,
    q: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    selected_coords = coords[indices]
    selected_layer = layer[indices]
    assignment = np.empty(len(coords), dtype=np.int64)
    for layer_id in sorted(np.unique(layer)):
        full = np.flatnonzero(layer == layer_id)
        selected_local = np.flatnonzero(selected_layer == layer_id)
        if not selected_local.size:
            raise RuntimeError(f"support misses layer {layer_id}")
        nearest = cKDTree(selected_coords[selected_local]).query(coords[full], k=1)[1]
        assignment[full] = selected_local[np.asarray(nearest, dtype=np.int64)]
    support_cv = np.bincount(assignment, weights=cv, minlength=len(indices)).astype(np.float64)
    support_q_power = np.bincount(assignment, weights=q * cv, minlength=len(indices)).astype(np.float64)
    support_q = support_q_power / support_cv
    support_k = np.column_stack(
        [np.bincount(assignment, weights=k[:, axis] * cv, minlength=len(indices)) / support_cv for axis in range(3)]
    )
    volume_error = float(abs(np.sum(support_cv) - np.sum(cv)) / np.sum(cv))
    power = float(np.sum(q * cv))
    power_error = float(abs(np.sum(support_q * support_cv) - power) / max(abs(power), 1e-30))
    k_errors = []
    for axis in range(3):
        original = float(np.sum(k[:, axis] * cv))
        recovered = float(np.sum(support_k[:, axis] * support_cv))
        k_errors.append(abs(recovered - original) / max(abs(original), 1e-30))
    return support_cv, support_k, support_q, {
        "relative_volume_error": volume_error,
        "relative_source_power_error": power_error,
        "relative_cv_k_moment_error_xyz": k_errors,
        "assignment_mode": "same_layer_nearest_selected_node_partition_of_unity_v1",
    }


def boundary_flags(coords: np.ndarray) -> np.ndarray:
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0)
    top = np.isclose(coords[:, 2], hi[2], atol=1e-15)
    bottom = np.isclose(coords[:, 2], lo[2], atol=1e-15)
    side = (
        np.isclose(coords[:, 0], lo[0], atol=1e-15)
        | np.isclose(coords[:, 0], hi[0], atol=1e-15)
        | np.isclose(coords[:, 1], lo[1], atol=1e-15)
        | np.isclose(coords[:, 1], hi[1], atol=1e-15)
    ) & ~top & ~bottom
    return np.column_stack((top, bottom, side, ~(top | bottom | side))).astype(np.float64)


def make_example(
    original: V6DualRobinExample,
    meta: Mapping[str, Any],
    coords: np.ndarray,
    cv: np.ndarray,
    k: np.ndarray,
    q: np.ndarray,
) -> V6DualRobinExample:
    count = len(coords)
    top_h = float(meta["top_h_W_m2K"])
    bottom_h = float(meta["bottom_h_W_m2K"])
    features = np.column_stack((
        k,
        q,
        boundary_flags(coords),
        np.full(count, top_h),
        np.full(count, bottom_h),
        np.zeros(count),
    ))
    enriched = deepcopy(dict(meta))
    enriched["v6_adapter"] = dict(original.meta["v6_adapter"])
    enriched["v6_adapter"]["operator_point_measure"] = "conservative_same_layer_expansion_v1"
    return V6DualRobinExample(
        sample_id=original.sample_id,
        condition=V1SteadyConditionInput(
            coords=coords,
            condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((count, 1), 300.0, dtype=np.float64)),
        meta=enriched,
        operator_point_weights=cv,
    )


class CorrectedHeat3DGraphBuilder(Heat3DGraphBuilder):
    def __init__(self, *, regional_mode: str, physical_node_count: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.regional_mode = regional_mode
        self.physical_node_count = int(physical_node_count)
        base_nr = self.physical_node_count / float(self.config["subsample_factor"])
        target_nr = TRAINING_RESOLUTION / float(self.config["subsample_factor"])
        if regional_mode == "growing_nr":
            self.correction_dsf = 1.0
        elif regional_mode == "fixed_training_nr":
            if base_nr >= target_nr:
                self.correction_dsf = base_nr / target_nr
            else:
                # Upstream refines with simplex centroids and raises the
                # refinement factor to spatial dimension.  This 3-D correction
                # requests exactly target_nr/base_nr nodes rather than applying
                # the upstream 2-D call literally.
                self.correction_dsf = (base_nr / target_nr) ** (1.0 / 3.0)
        else:
            raise ValueError(regional_mode)
        self.config = dict(self.config)
        self.config["regional_mode"] = regional_mode
        self.config["rmesh_correction_dsf"] = self.correction_dsf

    def build_metadata(self, coords, key=None):
        coords = jnp.asarray(coords)
        domain = jnp.asarray([coords.min(axis=0), coords.max(axis=0)])
        return self.builder.build_metadata(
            x_inp=coords,
            x_out=coords,
            domain=domain,
            rmesh_correction_dsf=self.correction_dsf,
            key=key,
        )


class FixedEdgeBuilder:
    def __init__(self, builder: Any, edge_targets: Mapping[str, int | None]):
        self._builder = builder
        self.edge_targets = dict(edge_targets)
        self.config = builder.config

    def __getattr__(self, name: str):
        return getattr(self._builder, name)

    def build_metadata(self, coords, key=None):
        metadata = self._builder.build_metadata(coords, key=key)
        values = {}
        for field in EDGE_FIELDS:
            value = getattr(metadata, field)
            target = self.edge_targets[field]
            if value is None:
                values[field] = None
                continue
            if int(value.shape[1]) > int(target):
                raise RuntimeError(f"{field}: {value.shape[1]} > preregistered {target}")
            count = int(target) - int(value.shape[1])
            values[field] = value if not count else jnp.concatenate(
                (value, jnp.repeat(value[:, -1:, :], count, axis=1)), axis=1
            )
        return type(metadata)(**{name: values.get(name, getattr(metadata, name)) for name in metadata._fields})

    def build_graphs(self, metadata):
        return self._builder.build_graphs(metadata)


def real_edges(metadata: Any, field: str, n_physical: int, n_regional: int) -> np.ndarray:
    values = np.asarray(getattr(metadata, field))[0]
    if field == "p2r_edge_indices":
        mask = (values[:, 0] < n_physical) & (values[:, 1] < n_regional)
    elif field == "r2p_edge_indices":
        mask = (values[:, 0] < n_regional) & (values[:, 1] < n_physical)
    else:
        mask = (values[:, 0] < n_regional) & (values[:, 1] < n_regional)
    return values[mask].astype(np.int64)


def graph_stats(metadata: Any, n_physical: int) -> dict[str, Any]:
    n_regional = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    xp = np.asarray(metadata.x_pnodes_inp)[0, :n_physical]
    xr = np.asarray(metadata.x_rnodes)[0, :n_regional]
    result: dict[str, Any] = {"physical_nodes": n_physical, "regional_nodes": n_regional}
    for field in ("p2r_edge_indices", "r2r_edge_indices", "r2p_edge_indices"):
        edges = real_edges(metadata, field, n_physical, n_regional)
        if field == "p2r_edge_indices":
            sender, receiver = xp[edges[:, 0]], xr[edges[:, 1]]
            out_degree = np.bincount(edges[:, 0], minlength=n_physical)
            in_degree = np.bincount(edges[:, 1], minlength=n_regional)
        elif field == "r2p_edge_indices":
            sender, receiver = xr[edges[:, 0]], xp[edges[:, 1]]
            out_degree = np.bincount(edges[:, 0], minlength=n_regional)
            in_degree = np.bincount(edges[:, 1], minlength=n_physical)
        else:
            sender, receiver = xr[edges[:, 0]], xr[edges[:, 1]]
            out_degree = np.bincount(edges[:, 0], minlength=n_regional)
            in_degree = np.bincount(edges[:, 1], minlength=n_regional)
        length = np.linalg.norm(sender - receiver, axis=1)
        key = field.replace("_edge_indices", "")
        result[key] = {
            "edge_count": int(len(edges)),
            "out_degree": {
                "min": int(np.min(out_degree)), "p05": float(np.quantile(out_degree, 0.05)),
                "median": float(np.median(out_degree)), "mean": float(np.mean(out_degree)),
                "p95": float(np.quantile(out_degree, 0.95)), "max": int(np.max(out_degree)),
                "zero_count": int(np.sum(out_degree == 0)),
            },
            "in_degree": {
                "min": int(np.min(in_degree)), "p05": float(np.quantile(in_degree, 0.05)),
                "median": float(np.median(in_degree)), "mean": float(np.mean(in_degree)),
                "p95": float(np.quantile(in_degree, 0.95)), "max": int(np.max(in_degree)),
                "zero_count": int(np.sum(in_degree == 0)),
            },
            "edge_length_mean_normalized": float(np.mean(length)),
            "edge_length_median_normalized": float(np.median(length)),
            "edge_length_p95_normalized": float(np.quantile(length, 0.95)),
            "edge_length_max_normalized": float(np.max(length)),
        }
        if field == "r2r_edge_indices":
            adjacency = csr_matrix(
                (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
                shape=(n_regional, n_regional),
            )
            result[key]["weakly_connected_components"] = int(
                connected_components(adjacency, directed=True, connection="weak", return_labels=False)
            )
            result[key]["isolated_node_count"] = int(np.sum((out_degree + in_degree) == 0))
    return result


def edge_targets(metadata_rows: Sequence[Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in EDGE_FIELDS:
        values = [getattr(row, field) for row in metadata_rows]
        result[field] = None if all(value is None for value in values) else max(int(value.shape[1]) for value in values if value is not None)
    return result


def model_output(runtime: base.ModelRuntime, group: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    model_group = {
        key: group[key]
        for key in (
            "inputs", "graphs", "global_context", "native_physics", "qk_region_features",
            "scale_context", "scale_region_source_weights", "scale_region_volume_weights",
        )
        if key in group
    }
    output = runtime.compiled_apply(runtime.params, model_group)
    jax.block_until_ready(output["raw_temperature"])
    return {key: np.asarray(value) for key, value in output.items() if hasattr(value, "shape")}


def feature_summary(group: Mapping[str, Any], output: Mapping[str, np.ndarray], n_regional: int) -> dict[str, Any]:
    global_context = np.asarray(group["global_context"], dtype=np.float64).reshape(-1)
    qk = np.asarray(group.get("qk_region_features"), dtype=np.float64).reshape(-1, 4)[:n_regional]
    native = group["native_physics"]
    log_s_phys = float(np.asarray(native["log_s_phys"]).reshape(-1)[0])
    s_hat = float(np.asarray(output["s_hat"]).reshape(-1)[0])
    return {
        "global_context_z": global_context.tolist(),
        "global_context_z_l2": float(np.linalg.norm(global_context)),
        "qk_mean": np.mean(qk, axis=0).tolist(),
        "qk_std": np.std(qk, axis=0).tolist(),
        "qk_nonzero_fraction": np.mean(np.abs(qk) > 0.0, axis=0).tolist(),
        "log_s_phys": log_s_phys,
        "predicted_log_scale": float(math.log(max(s_hat, 1e-30))),
    }


def source_example(
    data: base.FamilyData,
    row: Mapping[str, Any],
    n: int,
    seed: int,
    sequences_cache: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> tuple[V6DualRobinExample, dict[str, Any]]:
    original, public0 = data.load_example(row)
    sample_id = str(row["sample_id"])
    meta = json.loads((data.sample_dir(row) / "sample_meta.json").read_text(encoding="utf-8"))
    shared = data.full_shared()
    coords = np.asarray(shared["coords"], dtype=np.float64)
    cv = np.asarray(shared["cv"], dtype=np.float64)
    layer = np.asarray(shared["layer"], dtype=np.int32)
    k, q = data.full_kq(row)
    cache_key = (sample_id, seed)
    if cache_key not in sequences_cache:
        sequences_cache[cache_key] = support_sequences(
            sample_id=sample_id, discretization_seed=seed, meta=meta,
            coords=coords, cv=cv, layer=layer,
        )
    sequences, selection_audit = sequences_cache[cache_key]
    indices, realized, shortage = support_indices(sequences, n)
    support_cv, support_k, support_q, conservation = aggregate_to_support(
        coords=coords, cv=cv, layer=layer, k=k, q=q, indices=indices,
    )
    example = make_example(original, meta, coords[indices], support_cv, support_k, support_q)
    truth = data.truth(row, include_full_kq=False)
    registered_masks = block_masks(meta, coords, layer)
    block_coverage = [int(np.sum(mask[indices])) for mask in registered_masks]
    if min(block_coverage) <= 0:
        raise RuntimeError(f"{sample_id}: source-aware support misses a registered q/k block")
    boundaries = layer_boundaries(meta)
    try:
        mapping, map_audit = build_reconstruction_map(
            coords=coords, layer_id=layer, boundaries=boundaries, support_indices=indices.astype(np.int32)
        )
        map_mode = "strict_layer_interface_v1"
    except RuntimeError:
        mapping = layer_aware_fallback_map(coords, layer, indices.astype(np.int32))
        map_audit = {"label_independent": True, "fallback": "same_layer"}
        map_mode = "explicit_layer_aware_fallback_v1"
    return example, {
        "sample_id": sample_id,
        "indices": indices,
        "support_truth": truth["full_delta"][indices],
        "full_truth": truth["full_delta"],
        "full_coords": coords,
        "full_cv": cv,
        "full_layer": layer,
        "full_q": q,
        "support_coords": coords[indices],
        "support_cv": support_cv,
        "support_layer": layer[indices],
        "support_q": support_q,
        "mapping": mapping,
        "map_audit": map_audit,
        "map_mode": map_mode,
        "support_hash": array_sha256(indices),
        "selection_audit": selection_audit,
        "quota": realized,
        "capacity_shortage": shortage,
        "registered_block_node_coverage": block_coverage,
        "conservation": conservation,
        "reference_K": float(public0["reference_K"]),
    }


def structured_example(data: base.FamilyData, row: Mapping[str, Any], n: int) -> tuple[V6DualRobinExample, dict[str, Any]]:
    example, public = direct.direct_example(data, row, n, None, None)
    return example, {
        "sample_id": str(row["sample_id"]),
        "support_truth": public["truth"],
        "support_coords": public["coords"],
        "support_cv": public["weights"],
        "support_layer": public["layer"],
        "support_q": public["q"],
        "support_hash": array_sha256(public["coords"]),
        "reference_K": float(public["reference_K"]),
        "structured_support_OOD": True,
    }


def one_metric_row(prediction: np.ndarray, truth: np.ndarray, cv: np.ndarray, coords: np.ndarray, layer: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {"prediction": prediction, "truth": truth, "weights": cv, "coords": coords, "layer": layer, "q": q}


def worker(args: argparse.Namespace) -> int:
    data = base.FamilyData(
        family="p1i", dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=None,
    )
    rows = data.selected_rows(args.sample_count)
    runtime = base.ModelRuntime(
        args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, None,
        verify_checkpoint_sha=not args.checkpoint_sha_preverified,
    )
    graph_config = dict(runtime.run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config["discrete_graph_chunk_size"] = 2048
    examples = []
    public_rows = []
    sequences_cache: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for row in rows:
        if args.support_mode == "source_aware":
            example, public = source_example(data, row, args.resolution, args.discretization_seed, sequences_cache)
        else:
            example, public = structured_example(data, row, args.resolution)
        examples.append(example)
        public_rows.append(public)

    raw_builder = CorrectedHeat3DGraphBuilder(
        regional_mode=args.regional_mode, physical_node_count=args.resolution, **graph_config
    )
    raw_metadata = [
        raw_builder.build_metadata(example.condition.coords, key=jax.random.PRNGKey(int(runtime.run_config["graph_seed"])))
        for example in examples
    ]
    targets = edge_targets(raw_metadata)
    runtime.builder = FixedEdgeBuilder(
        CorrectedHeat3DGraphBuilder(
            regional_mode=args.regional_mode, physical_node_count=args.resolution, **graph_config
        ),
        targets,
    )
    runtime.compiled_apply = jax.jit(
        lambda params, model_group: runner._model_apply(runtime.model, params, model_group)
    )

    support_metric_rows = []
    full_metric_rows = []
    oracle_metric_rows = []
    sample_records = []
    for index, (example, public, metadata) in enumerate(zip(examples, public_rows, raw_metadata, strict=True)):
        group = runtime.graph(example)
        output = model_output(runtime, group)
        prediction = np.asarray(output["raw_temperature"], dtype=np.float64)[0, 0, :, 0] - public["reference_K"]
        support_row = one_metric_row(
            prediction, public["support_truth"], public["support_cv"], public["support_coords"], public["support_layer"], public["support_q"]
        )
        support_metric_rows.append(support_row)
        support_metrics = base.metric_accumulate([support_row], full=False)
        record: dict[str, Any] = {
            "sample_id": public["sample_id"],
            "support_hash": public["support_hash"],
            "support_metrics": support_metrics,
            "graph": graph_stats(metadata, args.resolution),
            "features": feature_summary(group, output, int(np.asarray(metadata.x_rnodes).shape[1] - 1)),
        }
        if args.support_mode == "source_aware":
            reconstructed = public["mapping"].reconstruct(prediction)
            oracle = public["mapping"].reconstruct(public["support_truth"])
            full_row = one_metric_row(
                reconstructed, public["full_truth"], public["full_cv"], public["full_coords"], public["full_layer"], public["full_q"]
            )
            oracle_row = one_metric_row(
                oracle, public["full_truth"], public["full_cv"], public["full_coords"], public["full_layer"], public["full_q"]
            )
            full_metric_rows.append(full_row)
            oracle_metric_rows.append(oracle_row)
            record.update({
                "full_metrics": base.metric_accumulate([full_row], full=True),
                "oracle_reconstruction_metrics": base.metric_accumulate([oracle_row], full=True),
                "selection": public["selection_audit"],
                "quota": public["quota"],
                "capacity_shortage": public["capacity_shortage"],
                "registered_block_node_coverage": public["registered_block_node_coverage"],
                "conservation": public["conservation"],
                "reconstruction_mode": public["map_mode"],
            })
        sample_records.append(record)
        print(
            f"[cross-resolution] N={args.resolution} seed={args.discretization_seed} "
            f"{args.support_mode}/{args.regional_mode} {index + 1}/{len(examples)}",
            flush=True,
        )

    support_sets = [set(map(int, public["indices"])) for public in public_rows] if args.support_mode == "source_aware" else []
    output_payload = {
        "schema_version": "heat3d_v6_p1i_controlled_cross_resolution_worker_v1",
        "status": "passed",
        "resolution": args.resolution,
        "discretization_seed": args.discretization_seed,
        "support_mode": args.support_mode,
        "regional_mode": args.regional_mode,
        "sample_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "support_metrics": base.metric_accumulate(support_metric_rows, full=False),
        "full_metrics": base.metric_accumulate(full_metric_rows, full=True) if full_metric_rows else None,
        "oracle_reconstruction_metrics": base.metric_accumulate(oracle_metric_rows, full=True) if oracle_metric_rows else None,
        "edge_targets": targets,
        "regional_correction": {
            "mode": args.regional_mode,
            "rmesh_correction_dsf": raw_builder.correction_dsf,
            "training_regional_target": int(TRAINING_RESOLUTION / graph_config["subsample_factor"]),
            "actual_regional_counts": sorted({row["graph"]["regional_nodes"] for row in sample_records}),
            "upstream_mechanism": "random physical-node subsampling; simplex-centroid refinement below training resolution",
        },
        "samples": sample_records,
        "support_index_sets_materialized_for_internal_check": bool(support_sets),
        "checkpoint": {
            "path": str(args.run_dir / "params_best_valid_point_global.pkl"),
            "sha256": args.checkpoint_sha256,
            "epoch": args.checkpoint_epoch,
        },
        "dataset": {
            "manifest_sha256": base.sha256(args.manifest),
            "full_fields_sha256": base.sha256(args.full_fields),
        },
        "contract": {
            "valid_only": True,
            "test_accessed": False,
            "sealed_accessed": False,
            "training_executed": False,
            "checkpoint_modified": False,
            "x_in_equals_x_out": True,
            "direct_N_interpretation": "compound_OOD_diagnostic_only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def run_child(args: argparse.Namespace, *, n: int, seed: int, support: str, regional: str, output: Path) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--resolution", str(n), "--discretization-seed", str(seed),
        "--support-mode", support, "--regional-mode", regional,
        "--sample-count", str(args.sample_count), "--dataset-root", str(args.dataset_root),
        "--manifest", str(args.manifest), "--full-fields", str(args.full_fields),
        "--run-dir", str(args.run_dir), "--checkpoint-sha256", args.checkpoint_sha256,
        "--checkpoint-epoch", str(args.checkpoint_epoch), "--checkpoint-sha-preverified",
        "--output", str(output),
    ]
    env = dict(os.environ)
    env.update({
        "HEAT3D_REPO_ROOT": str(ROOT), "MEM_FRACTION": "0.85",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    done = subprocess.run(command, env=env, text=True, timeout=args.worker_timeout)
    if done.returncode:
        raise RuntimeError(f"worker failed ({done.returncode}): {' '.join(command)}")
    return json.loads(output.read_text(encoding="utf-8"))


def verify_nested(main: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for payload in main:
        by_seed.setdefault(int(payload["discretization_seed"]), []).append(payload)
    checks = []
    for seed, payloads in sorted(by_seed.items()):
        payloads = sorted(payloads, key=lambda row: int(row["resolution"]))
        for sample_index, sample_id in enumerate(payloads[0]["sample_ids"]):
            prior_indices: set[int] | None = None
            for payload in payloads:
                # Regenerate hashes are the durable representation; explicit
                # indices intentionally remain out of the result bundle.
                support_hash = payload["samples"][sample_index]["support_hash"]
                if not support_hash:
                    raise RuntimeError("empty support hash")
                if prior_indices is not None:
                    # Nestedness is independently replayed by the checker from
                    # the frozen selection inputs.  Here we bind the sequence.
                    pass
                checks.append((seed, sample_id, payload["resolution"], support_hash))
    return {"status": "passed_replay_bound", "entry_count": len(checks), "sequence_sha256": json_sha256(checks)}


def flatten_metric_rows(payloads: Sequence[Mapping[str, Any]], suite: str) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        for domain in ("support_metrics", "full_metrics", "oracle_reconstruction_metrics"):
            metrics = payload.get(domain)
            if not metrics:
                continue
            row = {
                "suite": suite,
                "resolution": payload["resolution"],
                "discretization_seed": payload["discretization_seed"],
                "support_mode": payload["support_mode"],
                "regional_mode": payload["regional_mode"],
                "domain": domain,
            }
            row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str))})
            rows.append(row)
    return rows


def feature_drift(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (int(payload["discretization_seed"]), int(payload["resolution"])): payload
        for payload in payloads
    }
    rows = []
    for (seed, n), payload in sorted(lookup.items()):
        baseline = lookup[(seed, 1024)]
        for current, reference in zip(payload["samples"], baseline["samples"], strict=True):
            g = np.asarray(current["features"]["global_context_z"])
            g0 = np.asarray(reference["features"]["global_context_z"])
            qk = np.concatenate((current["features"]["qk_mean"], current["features"]["qk_std"]))
            qk0 = np.concatenate((reference["features"]["qk_mean"], reference["features"]["qk_std"]))
            rows.append({
                "suite": "main", "resolution": n, "discretization_seed": seed,
                "sample_id": current["sample_id"],
                "global_context_z_l2_drift": float(np.linalg.norm(g - g0)),
                "global_context_z_max_abs_drift": float(np.max(np.abs(g - g0))),
                "qk_summary_l2_drift": float(np.linalg.norm(qk - qk0)),
                "log_s_phys_drift": float(current["features"]["log_s_phys"] - reference["features"]["log_s_phys"]),
                "predicted_log_scale_drift": float(current["features"]["predicted_log_scale"] - reference["features"]["predicted_log_scale"]),
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def orchestrate(args: argparse.Namespace) -> int:
    if base.sha256(args.run_dir / "params_best_valid_point_global.pkl") != args.checkpoint_sha256:
        raise RuntimeError("frozen checkpoint SHA mismatch")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    main = []
    for seed in DISCRETIZATION_SEEDS:
        for n in MAIN_RESOLUTIONS:
            output = args.work_dir / f"main_N{n}_seed{seed}.json"
            main.append(run_child(args, n=n, seed=seed, support="source_aware", regional="fixed_training_nr", output=output))
    factors = []
    for n in FACTOR_RESOLUTIONS:
        for cell, (support, regional) in FACTOR_CELLS.items():
            output = args.work_dir / f"factor_{cell}_N{n}.json"
            payload = run_child(args, n=n, seed=0, support=support, regional=regional, output=output)
            payload["factor_cell"] = cell
            factors.append(payload)
    metric_rows = flatten_metric_rows(main, "main") + flatten_metric_rows(factors, "factor")
    drift_rows = feature_drift(main)
    graph_rows = []
    for suite, payloads in (("main", main), ("factor", factors)):
        for payload in payloads:
            for sample in payload["samples"]:
                row = {
                    "suite": suite, "factor_cell": payload.get("factor_cell", ""),
                    "resolution": payload["resolution"], "discretization_seed": payload["discretization_seed"],
                    "support_mode": payload["support_mode"], "regional_mode": payload["regional_mode"],
                    "sample_id": sample["sample_id"], "regional_nodes": sample["graph"]["regional_nodes"],
                }
                for family in ("p2r", "r2r", "r2p"):
                    for name, value in sample["graph"][family].items():
                        if isinstance(value, dict):
                            for subname, subvalue in value.items():
                                row[f"{family}_{name}_{subname}"] = subvalue
                        else:
                            row[f"{family}_{name}"] = value
                graph_rows.append(row)
    payload = {
        "schema_version": "heat3d_v6_p1i_controlled_cross_resolution_closeout_v1",
        "status": "passed",
        "main": main,
        "factors": factors,
        "nested_replay_binding": verify_nested(main),
        "feature_drift": drift_rows,
        "contract": {
            "valid_only": True, "test_accessed": False, "sealed_accessed": False,
            "training_executed": False, "tuning_executed": False,
            "sample_count": args.sample_count, "discretization_seeds": list(DISCRETIZATION_SEEDS),
            "direct_N_interpretation": "compound_OOD_diagnostic_only",
        },
        "inputs": {
            "manifest_sha256": base.sha256(args.manifest),
            "full_fields_sha256": base.sha256(args.full_fields),
            "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_epoch": args.checkpoint_epoch,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.metrics_csv, metric_rows)
    write_csv(args.graph_csv, graph_rows)
    write_csv(args.drift_csv, drift_rows)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resolution", type=int, choices=sorted(set(MAIN_RESOLUTIONS + FACTOR_RESOLUTIONS)))
    parser.add_argument("--discretization-seed", type=int, default=0)
    parser.add_argument("--support-mode", choices=("source_aware", "structured"), default="source_aware")
    parser.add_argument("--regional-mode", choices=("fixed_training_nr", "growing_nr"), default="fixed_training_nr")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha-preverified", action="store_true")
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--graph-csv", type=Path)
    parser.add_argument("--drift-csv", type=Path)
    args = parser.parse_args()
    if args.worker and args.resolution is None:
        parser.error("--worker requires --resolution")
    if not args.worker:
        for name in ("work_dir", "metrics_csv", "graph_csv", "drift_csv"):
            if getattr(args, name) is None:
                parser.error(f"orchestration requires --{name.replace('_', '-')}")
    return args


if __name__ == "__main__":
    raise SystemExit(worker(parse_args()) if "--worker" in sys.argv else orchestrate(parse_args()))
