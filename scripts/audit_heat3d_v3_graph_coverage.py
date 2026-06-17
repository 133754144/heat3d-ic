#!/usr/bin/env python3
"""Formal read-only Heat3D v3 P0/P1 p2r/r2p graph coverage audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_schema import find_sample_dirs, load_sample_meta  # noqa: E402


DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_SPLIT_MAP = (
    REPO_ROOT
    / "configs"
    / "heat3d_v2"
    / "medium1024_gapA_stratified_split_seed0.json"
)
DEFAULT_SPLITS = ("train", "valid_iid", "valid_stress")
SYNTHETIC_GRID_SHAPES = ((8, 8, 6), (12, 12, 8), (16, 16, 12))
POLICY_CURRENT = "current"
POLICY_NO_HARD_RESET = "candidate_no_hard_reset"
POLICY_NO_GLOBAL_CLIP = "candidate_no_global_clip"
POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP = "candidate_no_hard_reset_no_global_clip"
POLICY_NEAREST_REPAIR = "candidate-nearest-repair"
POLICY_DISCRETE_COVERAGE_RADIUS = "candidate-discrete-coverage-radius"
POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR = (
    "candidate-discrete-coverage-radius-with-nearest-repair"
)
P1_POLICIES = {
    POLICY_NEAREST_REPAIR,
    POLICY_DISCRETE_COVERAGE_RADIUS,
    POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
}
P0_SIMULATED_POLICIES = {
    POLICY_NO_HARD_RESET,
    POLICY_NO_GLOBAL_CLIP,
    POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--split-map", type=Path, default=DEFAULT_SPLIT_MAP)
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names, or all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of selected samples; 0 audits all selected samples.",
    )
    parser.add_argument("--rmesh-seeds", default="0", help="Comma-separated integer seeds.")
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument(
        "--boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_false",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--candidate-no-hard-reset", action="store_true")
    parser.add_argument("--candidate-no-global-clip", action="store_true")
    parser.add_argument("--candidate-no-hard-reset-no-global-clip", action="store_true")
    parser.add_argument("--candidate-nearest-repair", action="store_true")
    parser.add_argument("--candidate-discrete-coverage-radius", action="store_true")
    parser.add_argument(
        "--candidate-discrete-coverage-radius-with-nearest-repair",
        action="store_true",
    )
    parser.add_argument(
        "--coverage-repair-policy",
        choices=("none", "nearest_rnode"),
        default="none",
    )
    parser.add_argument(
        "--radius-policy",
        choices=("legacy_kdtree_mean4", "discrete_physical_coverage"),
        default="legacy_kdtree_mean4",
    )
    parser.add_argument("--repair-p2r", dest="repair_p2r", action="store_true", default=True)
    parser.add_argument("--no-repair-p2r", dest="repair_p2r", action="store_false")
    parser.add_argument("--repair-r2p", dest="repair_r2p", action="store_true", default=True)
    parser.add_argument("--no-repair-r2p", dest="repair_r2p", action="store_false")
    parser.add_argument("--min-physical-coverage", type=int, default=1)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--rmesh-seeds must contain at least one integer")
    return seeds


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split or all")
    return splits


def selected_policies(args: argparse.Namespace) -> list[str]:
    policies = [POLICY_CURRENT]
    if args.candidate_no_hard_reset:
        policies.append(POLICY_NO_HARD_RESET)
    if args.candidate_no_global_clip:
        policies.append(POLICY_NO_GLOBAL_CLIP)
    if args.candidate_no_hard_reset_no_global_clip:
        policies.append(POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP)
    if args.candidate_nearest_repair:
        policies.append(POLICY_NEAREST_REPAIR)
    if args.candidate_discrete_coverage_radius:
        policies.append(POLICY_DISCRETE_COVERAGE_RADIUS)
    if args.candidate_discrete_coverage_radius_with_nearest_repair:
        policies.append(POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR)

    configured_policy = _configured_policy_name(
        radius_policy=args.radius_policy,
        coverage_repair_policy=args.coverage_repair_policy,
    )
    if configured_policy is not None:
        policies.append(configured_policy)
    return list(dict.fromkeys(policies))


def _configured_policy_name(
    *,
    radius_policy: str,
    coverage_repair_policy: str,
) -> str | None:
    if radius_policy == "discrete_physical_coverage":
        if coverage_repair_policy == "nearest_rnode":
            return POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR
        return POLICY_DISCRETE_COVERAGE_RADIUS
    if coverage_repair_policy == "nearest_rnode":
        return POLICY_NEAREST_REPAIR
    return None


def all_policies() -> list[str]:
    return [
        POLICY_CURRENT,
        POLICY_NO_HARD_RESET,
        POLICY_NO_GLOBAL_CLIP,
        POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP,
        POLICY_NEAREST_REPAIR,
        POLICY_DISCRETE_COVERAGE_RADIUS,
        POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
    ]


def _policy_switches(policy: str) -> tuple[bool, bool]:
    if policy == POLICY_CURRENT:
        return True, True
    if policy == POLICY_NO_HARD_RESET:
        return False, True
    if policy == POLICY_NO_GLOBAL_CLIP:
        return True, False
    if policy == POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP:
        return False, False
    if policy == POLICY_NEAREST_REPAIR:
        return True, True
    if policy in {
        POLICY_DISCRETE_COVERAGE_RADIUS,
        POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
    }:
        return False, False
    raise ValueError(f"Unknown graph coverage policy: {policy}")


def _percentile_stats(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _coverage_stats(degree: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(degree, dtype=np.int64).reshape(-1)
    stats: dict[str, int | float | None] = _percentile_stats(values)
    low_count = int(np.sum(values <= 1))
    stats.update(
        {
            "zero_count": int(np.sum(values == 0)),
            "low_coverage_count": low_count,
            "low_coverage_ratio": float(low_count / values.size) if values.size else None,
        }
    )
    return stats


def _shape_signature(tree: Any) -> list[list[int]]:
    return [
        list(leaf.shape)
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "shape")
    ]


def _all_finite(tree: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "shape")
    )


def _real_edges(edges: Any, sender_count: int, receiver_count: int) -> np.ndarray:
    values = np.asarray(edges).reshape(-1, 2).astype(np.int64)
    return values[
        (values[:, 0] >= 0)
        & (values[:, 0] < sender_count)
        & (values[:, 1] >= 0)
        & (values[:, 1] < receiver_count)
    ]


def _degrees(edges: np.ndarray, node_count: int, column: int) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(node_count, dtype=np.int64)
    return np.bincount(edges[:, column], minlength=node_count).astype(np.int64)


def _radius_stages(raw: np.ndarray, overlap_factor: float, policy: str) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if policy in {
        POLICY_DISCRETE_COVERAGE_RADIUS,
        POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
    }:
        overlap = raw.copy()
        legacy_hard_reset_candidates = overlap >= 0.5
        return {
            "raw": _percentile_stats(raw),
            "overlap": _percentile_stats(overlap),
            "clipped": _percentile_stats(overlap),
            "hard_reset": _percentile_stats(overlap),
            "hard_reset_trigger_count": 0,
            "hard_reset_candidate_count": int(np.sum(legacy_hard_reset_candidates)),
            "coverage_guarantee_preserved": True,
            "effective_overlap_factor": 1.0,
            "legacy_overlap_factor_not_applied": float(overlap_factor),
        }

    use_hard_reset, use_global_clip = _policy_switches(policy)
    overlap = overlap_factor * raw
    clipped = np.minimum(overlap, np.max(raw)) if use_global_clip else overlap.copy()
    hard_reset_candidates = clipped >= 0.5
    final = np.where(hard_reset_candidates, 0.2, clipped) if use_hard_reset else clipped.copy()
    return {
        "raw": _percentile_stats(raw),
        "overlap": _percentile_stats(overlap),
        "clipped": _percentile_stats(clipped),
        "hard_reset": _percentile_stats(final),
        "hard_reset_trigger_count": int(np.sum(hard_reset_candidates)) if use_hard_reset else 0,
        "hard_reset_candidate_count": int(np.sum(hard_reset_candidates)),
        "coverage_guarantee_preserved": False,
        "effective_overlap_factor": float(overlap_factor),
        "legacy_overlap_factor_not_applied": None,
    }


def _candidate_edges(
    distance: np.ndarray,
    radii: np.ndarray,
    *,
    reverse: bool,
    dtype: np.dtype,
    n_physical_nodes: int,
    n_rnodes: int,
) -> jax.Array:
    pairs = np.argwhere(distance <= radii.reshape(1, -1))
    if reverse:
        pairs = np.flip(pairs, axis=1)
        dummy = np.array([[n_rnodes, n_physical_nodes]], dtype=np.int64)
    else:
        dummy = np.array([[n_physical_nodes, n_rnodes]], dtype=np.int64)
    pairs = np.concatenate([pairs, dummy], axis=0).astype(dtype, copy=False)
    return jnp.expand_dims(jnp.asarray(pairs), axis=0)


def _boundary_group_masks(coords: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    lower = np.min(coords, axis=0)
    upper = np.max(coords, axis=0)
    span = upper - lower
    scale = np.maximum(np.maximum(np.abs(lower), np.abs(upper)), span)
    atol = np.maximum(scale * 1.0e-8, np.finfo(np.float64).eps * 32)
    at_lower = np.isclose(coords, lower, atol=atol, rtol=0.0)
    at_upper = np.isclose(coords, upper, atol=atol, rtol=0.0)
    on_axis_boundary = at_lower | at_upper
    on_side = np.any(on_axis_boundary[:, :2], axis=1)
    on_z_boundary = on_axis_boundary[:, 2]
    return {
        "top": at_upper[:, 2],
        "bottom": at_lower[:, 2],
        "side": on_side,
        "corner": np.sum(on_axis_boundary, axis=1) == 3,
        "interior": ~(on_side | on_z_boundary),
    }


def _interface_mask(coords: np.ndarray, layer_id: np.ndarray) -> np.ndarray:
    z_values = np.asarray(coords, dtype=np.float64)[:, 2]
    layers = np.asarray(layer_id).reshape(-1)
    unique_z = np.unique(z_values)
    z_scale = max(float(np.max(np.abs(z_values))), float(np.ptp(z_values)))
    z_atol = max(z_scale * 1.0e-8, np.finfo(np.float64).eps * 32)
    interface_z: list[float] = []
    previous_layers: set[int] | None = None
    previous_z: float | None = None
    for z_value in unique_z:
        plane_layers = {
            int(value)
            for value in np.unique(layers[np.isclose(z_values, z_value, atol=z_atol, rtol=0.0)])
        }
        if len(plane_layers) > 1:
            interface_z.append(float(z_value))
        if previous_layers is not None and plane_layers != previous_layers:
            assert previous_z is not None
            interface_z.extend([previous_z, float(z_value)])
        previous_layers = plane_layers
        previous_z = float(z_value)
    mask = np.zeros(z_values.shape[0], dtype=bool)
    for z_value in interface_z:
        mask |= np.isclose(z_values, z_value, atol=z_atol, rtol=0.0)
    return mask


def _group_stats(
    masks: dict[str, np.ndarray],
    p2r_degree: np.ndarray,
    r2p_degree: np.ndarray,
) -> dict[str, Any]:
    return {
        str(name): {
            "node_count": int(np.sum(mask)),
            "p2r": _coverage_stats(p2r_degree[mask]),
            "r2p": _coverage_stats(r2p_degree[mask]),
        }
        for name, mask in masks.items()
    }


def _grouped_coverage(
    coords: np.ndarray,
    p2r_degree: np.ndarray,
    r2p_degree: np.ndarray,
    layer_id: np.ndarray | None,
    material_id: np.ndarray | None,
) -> dict[str, Any]:
    n_nodes = coords.shape[0]
    reasons: list[str] = []
    result: dict[str, Any] = {
        "boundary": _group_stats(_boundary_group_masks(coords), p2r_degree, r2p_degree)
    }

    if layer_id is None or np.asarray(layer_id).reshape(-1).size != n_nodes:
        reasons.append("layer_id metadata is missing or has incompatible shape")
    else:
        layer_values = np.asarray(layer_id).reshape(-1)
        layer_masks = {
            str(int(value)): layer_values == value for value in np.unique(layer_values)
        }
        result["layer"] = _group_stats(layer_masks, p2r_degree, r2p_degree)
        result["interface"] = _group_stats(
            {"interface": _interface_mask(coords, layer_values)},
            p2r_degree,
            r2p_degree,
        )

    if material_id is None or np.asarray(material_id).reshape(-1).size != n_nodes:
        reasons.append("material_id metadata is missing or has incompatible shape")
    else:
        material_values = np.asarray(material_id).reshape(-1)
        material_masks = {
            str(int(value)): material_values == value for value in np.unique(material_values)
        }
        result["material"] = _group_stats(material_masks, p2r_degree, r2p_degree)

    return {
        "grouped_coverage_available": not reasons,
        "grouped_coverage_unavailable_reasons": reasons,
        "grouped_coverage": result,
    }


def _builder_for_policy(
    policy: str,
    *,
    repair_p2r: bool,
    repair_r2p: bool,
    min_physical_coverage: int,
) -> Heat3DGraphBuilder:
    if policy == POLICY_NEAREST_REPAIR:
        return Heat3DGraphBuilder(
            radius_policy="legacy_kdtree_mean4",
            coverage_repair_policy="nearest_rnode",
            repair_p2r=repair_p2r,
            repair_r2p=repair_r2p,
            min_physical_coverage=min_physical_coverage,
        )
    if policy == POLICY_DISCRETE_COVERAGE_RADIUS:
        return Heat3DGraphBuilder(radius_policy="discrete_physical_coverage")
    if policy == POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR:
        return Heat3DGraphBuilder(
            radius_policy="discrete_physical_coverage",
            coverage_repair_policy="nearest_rnode",
            repair_p2r=repair_p2r,
            repair_r2p=repair_r2p,
            min_physical_coverage=min_physical_coverage,
        )
    return Heat3DGraphBuilder(
        radius_policy="legacy_kdtree_mean4",
        coverage_repair_policy="none",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    )


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(sender), int(receiver)) for sender, receiver in np.asarray(edges)}


def _edge_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, int | float]:
    candidate_set = _edge_set(candidate)
    baseline_set = _edge_set(baseline)
    return {
        "added_edge_count": len(candidate_set - baseline_set),
        "removed_edge_count": len(baseline_set - candidate_set),
        "edge_ratio": float(len(candidate) / len(baseline)) if len(baseline) else float("inf"),
    }


def _repaired_physical_count(
    candidate_degree: np.ndarray,
    baseline_degree: np.ndarray,
    min_physical_coverage: int,
) -> int:
    return int(
        np.sum(
            (np.asarray(baseline_degree) < min_physical_coverage)
            & (np.asarray(candidate_degree) >= min_physical_coverage)
        )
    )


def _node_and_r2r_stable(candidate: Any, legacy: Any) -> bool:
    fields = (
        "x_pnodes_inp",
        "x_pnodes_out",
        "x_rnodes",
        "r2r_edge_indices",
        "r2r_edge_domains",
    )
    return all(
        np.array_equal(np.asarray(getattr(candidate, field)), np.asarray(getattr(legacy, field)))
        for field in fields
    )


def audit_coords(
    *,
    sample_id: str,
    split: str,
    coords: np.ndarray,
    seeds: Iterable[int],
    policies: Iterable[str],
    layer_id: np.ndarray | None = None,
    material_id: np.ndarray | None = None,
    repair_p2r: bool = True,
    repair_r2p: bool = True,
    min_physical_coverage: int = 1,
) -> list[dict[str, Any]]:
    coords = np.asarray(coords, dtype=np.float64)
    seeds = list(seeds)
    policies = list(policies)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{sample_id}: coords must have shape (N, 3), found {coords.shape}")

    if min_physical_coverage < 1:
        raise ValueError("min_physical_coverage must be at least 1")

    records: list[dict[str, Any]] = []
    legacy_builder = Heat3DGraphBuilder(
        radius_policy="legacy_kdtree_mean4",
        coverage_repair_policy="none",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    )
    for seed in seeds:
        metadata_start = time.perf_counter()
        legacy_metadata = legacy_builder.build_metadata(coords, key=jax.random.PRNGKey(seed))
        legacy_metadata_build_time = time.perf_counter() - metadata_start

        n_physical_nodes = coords.shape[0]
        n_rnodes = int(np.asarray(legacy_metadata.x_rnodes).shape[1] - 1)
        if n_rnodes < 5:
            raise ValueError(f"{sample_id}: need at least 5 real regional nodes, found {n_rnodes}")

        x_pnodes = np.asarray(legacy_metadata.x_pnodes_inp)[0, :n_physical_nodes]
        x_rnodes = np.asarray(legacy_metadata.x_rnodes)[0, :n_rnodes]
        legacy_raw_radius = np.asarray(legacy_metadata.r_rnodes)[0, :n_rnodes]
        distance = np.linalg.norm(x_pnodes[:, None, :] - x_rnodes[None, :, :], axis=-1)
        current_dtype = np.asarray(legacy_metadata.p2r_edge_indices).dtype
        legacy_p2r = _real_edges(
            legacy_metadata.p2r_edge_indices, n_physical_nodes, n_rnodes
        )
        legacy_r2p_values = legacy_metadata.r2p_edge_indices
        if legacy_r2p_values is None:
            legacy_r2p_values = jnp.flip(legacy_metadata.p2r_edge_indices, axis=-1)
        legacy_r2p = _real_edges(legacy_r2p_values, n_rnodes, n_physical_nodes)
        legacy_p2r_degree = _degrees(legacy_p2r, n_physical_nodes, column=0)
        legacy_r2p_degree = _degrees(legacy_r2p, n_physical_nodes, column=1)

        p1_metadata: dict[str, Any] = {}
        p1_build_times: dict[str, float] = {}
        p1_builders: dict[str, Heat3DGraphBuilder] = {}
        policies_to_build = set(policies) & P1_POLICIES
        if POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR in policies_to_build:
            policies_to_build.add(POLICY_DISCRETE_COVERAGE_RADIUS)
        for p1_policy in sorted(policies_to_build):
            p1_builder = _builder_for_policy(
                p1_policy,
                repair_p2r=repair_p2r,
                repair_r2p=repair_r2p,
                min_physical_coverage=min_physical_coverage,
            )
            metadata_start = time.perf_counter()
            candidate_metadata = p1_builder.build_metadata(
                coords,
                key=jax.random.PRNGKey(seed),
            )
            p1_build_times[p1_policy] = time.perf_counter() - metadata_start
            p1_builders[p1_policy] = p1_builder
            p1_metadata[p1_policy] = candidate_metadata

        for policy in policies:
            graph_start = time.perf_counter()
            if policy == POLICY_CURRENT:
                policy_builder = legacy_builder
                policy_metadata = legacy_metadata
                metadata_build_time = legacy_metadata_build_time
            elif policy in P1_POLICIES:
                policy_builder = p1_builders[policy]
                policy_metadata = p1_metadata[policy]
                metadata_build_time = p1_build_times[policy]
            elif policy in P0_SIMULATED_POLICIES:
                policy_builder = legacy_builder
                metadata_build_time = legacy_metadata_build_time
                use_hard_reset, use_global_clip = _policy_switches(policy)

                def final_radii(overlap_factor: float) -> np.ndarray:
                    radii = overlap_factor * legacy_raw_radius
                    if use_global_clip:
                        radii = np.minimum(radii, np.max(legacy_raw_radius))
                    if use_hard_reset:
                        radii = np.where(radii < 0.5, radii, 0.2)
                    return radii

                p2r_edges = _candidate_edges(
                    distance,
                    final_radii(legacy_builder.config["overlap_factor_p2r"]),
                    reverse=False,
                    dtype=current_dtype,
                    n_physical_nodes=n_physical_nodes,
                    n_rnodes=n_rnodes,
                )
                r2p_edges = _candidate_edges(
                    distance,
                    final_radii(legacy_builder.config["overlap_factor_r2p"]),
                    reverse=True,
                    dtype=current_dtype,
                    n_physical_nodes=n_physical_nodes,
                    n_rnodes=n_rnodes,
                )
                policy_metadata = type(legacy_metadata)(
                    x_pnodes_inp=legacy_metadata.x_pnodes_inp,
                    x_pnodes_out=legacy_metadata.x_pnodes_out,
                    x_rnodes=legacy_metadata.x_rnodes,
                    r_rnodes=legacy_metadata.r_rnodes,
                    p2r_edge_indices=p2r_edges,
                    r2r_edge_indices=legacy_metadata.r2r_edge_indices,
                    r2r_edge_domains=legacy_metadata.r2r_edge_domains,
                    r2p_edge_indices=r2p_edges,
                )
            else:
                raise ValueError(f"Unsupported audit policy: {policy}")

            raw_radius = np.asarray(policy_metadata.r_rnodes)[0, :n_rnodes]
            p2r_stages = _radius_stages(
                raw_radius,
                policy_builder.config["overlap_factor_p2r"],
                policy,
            )
            r2p_stages = _radius_stages(
                raw_radius,
                policy_builder.config["overlap_factor_r2p"],
                policy,
            )
            graphs = policy_builder.build_graphs(policy_metadata)
            graph_build_time = time.perf_counter() - graph_start

            p2r_real = _real_edges(
                policy_metadata.p2r_edge_indices, n_physical_nodes, n_rnodes
            )
            r2r_real = _real_edges(policy_metadata.r2r_edge_indices, n_rnodes, n_rnodes)
            r2p_values = policy_metadata.r2p_edge_indices
            if r2p_values is None:
                r2p_values = jnp.flip(policy_metadata.p2r_edge_indices, axis=-1)
            r2p_real = _real_edges(r2p_values, n_rnodes, n_physical_nodes)
            p2r_degree = _degrees(p2r_real, n_physical_nodes, column=0)
            r2p_degree = _degrees(r2p_real, n_physical_nodes, column=1)
            p2r_delta = _edge_delta(p2r_real, legacy_p2r)
            r2p_delta = _edge_delta(r2p_real, legacy_r2p)

            if policy == POLICY_NEAREST_REPAIR:
                repair_base_p2r = legacy_p2r
                repair_base_r2p = legacy_r2p
                repair_base_p2r_degree = legacy_p2r_degree
                repair_base_r2p_degree = legacy_r2p_degree
            elif policy == POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR:
                discrete_metadata = p1_metadata[POLICY_DISCRETE_COVERAGE_RADIUS]
                repair_base_p2r = _real_edges(
                    discrete_metadata.p2r_edge_indices, n_physical_nodes, n_rnodes
                )
                discrete_r2p_values = discrete_metadata.r2p_edge_indices
                if discrete_r2p_values is None:
                    discrete_r2p_values = jnp.flip(
                        discrete_metadata.p2r_edge_indices,
                        axis=-1,
                    )
                repair_base_r2p = _real_edges(
                    discrete_r2p_values, n_rnodes, n_physical_nodes
                )
                repair_base_p2r_degree = _degrees(
                    repair_base_p2r, n_physical_nodes, column=0
                )
                repair_base_r2p_degree = _degrees(
                    repair_base_r2p, n_physical_nodes, column=1
                )
            else:
                repair_base_p2r = p2r_real
                repair_base_r2p = r2p_real
                repair_base_p2r_degree = p2r_degree
                repair_base_r2p_degree = r2p_degree
            p2r_repair_delta = _edge_delta(p2r_real, repair_base_p2r)
            r2p_repair_delta = _edge_delta(r2p_real, repair_base_r2p)

            record: dict[str, Any] = {
                "sample_id": sample_id,
                "split": split,
                "seed": int(seed),
                "policy": policy,
                "n_physical_nodes_real": int(n_physical_nodes),
                "n_rnodes_real": int(n_rnodes),
                "p2r_real_edge_count": int(p2r_real.shape[0]),
                "r2p_real_edge_count": int(r2p_real.shape[0]),
                "r2r_real_edge_count": int(r2r_real.shape[0]),
                "legacy_p2r_real_edge_count": int(legacy_p2r.shape[0]),
                "legacy_r2p_real_edge_count": int(legacy_r2p.shape[0]),
                "p2r_physical_node_coverage": _coverage_stats(p2r_degree),
                "r2p_physical_node_coverage": _coverage_stats(r2p_degree),
                "radius_stages": {"p2r": p2r_stages, "r2p": r2p_stages},
                "builder_config": policy_builder.config,
                "edge_delta_vs_legacy": {"p2r": p2r_delta, "r2p": r2p_delta},
                "edge_ratio_vs_legacy": {
                    "p2r": p2r_delta["edge_ratio"],
                    "r2p": r2p_delta["edge_ratio"],
                },
                "repaired_edge_count": {
                    "p2r": p2r_repair_delta["added_edge_count"],
                    "r2p": r2p_repair_delta["added_edge_count"],
                },
                "repaired_physical_count": {
                    "p2r": _repaired_physical_count(
                        p2r_degree,
                        repair_base_p2r_degree,
                        min_physical_coverage,
                    ),
                    "r2p": _repaired_physical_count(
                        r2p_degree,
                        repair_base_r2p_degree,
                        min_physical_coverage,
                    ),
                },
                "metadata_build_time_seconds": float(metadata_build_time),
                "graph_build_time_seconds": float(graph_build_time),
                "metadata_shape_signature": _shape_signature(policy_metadata),
                "graph_leaf_shape_signature": _shape_signature(graphs),
                "metadata_all_finite": _all_finite(policy_metadata),
                "graph_all_finite": _all_finite(graphs),
                "node_and_r2r_stable_vs_legacy": _node_and_r2r_stable(
                    policy_metadata,
                    legacy_metadata,
                ),
                "dummy_excluded": True,
            }
            record.update(
                _grouped_coverage(coords, p2r_degree, r2p_degree, layer_id, material_id)
            )
            records.append(record)
    return records


def _synthetic_grid(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    x = np.linspace(0.0, 0.01, nx)
    y = np.linspace(0.0, 0.01, ny)
    z = np.linspace(0.0, 0.002, nz)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    z_index = np.tile(np.arange(nz), nx * ny)
    layer_id = np.minimum((3 * z_index) // nz, 2).astype(np.int32)
    material_id = layer_id.copy()
    return grid, layer_id, material_id


def run_synthetic_probes(
    *,
    seeds: Iterable[int],
    policies: Iterable[str],
    repair_p2r: bool = True,
    repair_r2p: bool = True,
    min_physical_coverage: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shape in SYNTHETIC_GRID_SHAPES:
        coords, layer_id, material_id = _synthetic_grid(shape)
        records.extend(
            audit_coords(
                sample_id=f"synthetic_{shape[0]}x{shape[1]}x{shape[2]}",
                split="synthetic",
                coords=coords,
                seeds=seeds,
                policies=policies,
                layer_id=layer_id,
                material_id=material_id,
                repair_p2r=repair_p2r,
                repair_r2p=repair_r2p,
                min_physical_coverage=min_physical_coverage,
            )
        )
    return records


def _load_split_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sample_splits", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: expected mapping or sample_splits mapping")
    return {str(sample_id): str(split) for sample_id, split in mapping.items()}


def _load_optional_array(sample_dir: Path, name: str) -> np.ndarray | None:
    path = sample_dir / name
    return np.load(path) if path.is_file() else None


def audit_real_dataset(
    *,
    subset: Path,
    split_map: Path | None,
    splits: Iterable[str],
    max_samples: int,
    seeds: Iterable[int],
    policies: Iterable[str],
    k_encoding_mode: str,
    boundary_mask_fallback: bool,
    repair_p2r: bool = True,
    repair_r2p: bool = True,
    min_physical_coverage: int = 1,
) -> list[dict[str, Any]]:
    subset = subset.resolve()
    if not subset.is_dir():
        raise FileNotFoundError(f"Heat3D subset does not exist: {subset}")
    # Graph topology depends only on coordinates. These options are accepted and
    # recorded by callers so audit invocations mirror the corresponding loader.
    if not k_encoding_mode:
        raise ValueError("--k-encoding-mode must not be empty")
    _ = boundary_mask_fallback

    split_by_id = _load_split_map(split_map.resolve() if split_map is not None else None)
    requested_splits = set(splits)
    select_all = "all" in requested_splits
    selected: list[tuple[Path, str, str]] = []
    for sample_dir in find_sample_dirs(subset):
        meta = load_sample_meta(sample_dir)
        sample_id = str(meta.get("sample_id", sample_dir.name))
        split = split_by_id.get(sample_id, str(meta.get("split", "unknown")))
        if select_all or split in requested_splits:
            selected.append((sample_dir, sample_id, split))
    selected.sort(key=lambda item: item[1])
    if max_samples > 0:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(
            f"No samples selected from {subset}; requested splits={sorted(requested_splits)}"
        )

    records: list[dict[str, Any]] = []
    for sample_dir, sample_id, split in selected:
        coords_path = sample_dir / "coords.npy"
        if not coords_path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: missing required graph coordinate file: {coords_path}"
            )
        records.extend(
            audit_coords(
                sample_id=sample_id,
                split=split,
                coords=np.load(coords_path),
                seeds=seeds,
                policies=policies,
                layer_id=_load_optional_array(sample_dir, "layer_id.npy"),
                material_id=_load_optional_array(sample_dir, "material_id.npy"),
                repair_p2r=repair_p2r,
                repair_r2p=repair_r2p,
                min_physical_coverage=min_physical_coverage,
            )
        )
    return records


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["policy"]].append(record)
    summary: dict[str, Any] = {}
    for policy, policy_records in sorted(grouped.items()):
        p2r_edge_count = int(sum(row["p2r_real_edge_count"] for row in policy_records))
        r2p_edge_count = int(sum(row["r2p_real_edge_count"] for row in policy_records))
        legacy_p2r_edge_count = int(
            sum(row["legacy_p2r_real_edge_count"] for row in policy_records)
        )
        legacy_r2p_edge_count = int(
            sum(row["legacy_r2p_real_edge_count"] for row in policy_records)
        )
        p2r_zero_count = int(
            sum(row["p2r_physical_node_coverage"]["zero_count"] for row in policy_records)
        )
        r2p_zero_count = int(
            sum(row["r2p_physical_node_coverage"]["zero_count"] for row in policy_records)
        )
        summary[policy] = {
            "record_count": len(policy_records),
            "p2r_zero_count_total": p2r_zero_count,
            "r2p_zero_count_total": r2p_zero_count,
            "p2r_low_coverage_count_total": int(
                sum(
                    row["p2r_physical_node_coverage"]["low_coverage_count"]
                    for row in policy_records
                )
            ),
            "r2p_low_coverage_count_total": int(
                sum(
                    row["r2p_physical_node_coverage"]["low_coverage_count"]
                    for row in policy_records
                )
            ),
            "p2r_real_edge_count_total": p2r_edge_count,
            "r2p_real_edge_count_total": r2p_edge_count,
            "r2r_real_edge_count_total": int(
                sum(row["r2r_real_edge_count"] for row in policy_records)
            ),
            "p2r_repaired_edge_count_total": int(
                sum(row["repaired_edge_count"]["p2r"] for row in policy_records)
            ),
            "r2p_repaired_edge_count_total": int(
                sum(row["repaired_edge_count"]["r2p"] for row in policy_records)
            ),
            "p2r_repaired_physical_count_total": int(
                sum(row["repaired_physical_count"]["p2r"] for row in policy_records)
            ),
            "r2p_repaired_physical_count_total": int(
                sum(row["repaired_physical_count"]["r2p"] for row in policy_records)
            ),
            "p2r_edge_ratio_vs_legacy": (
                float(p2r_edge_count / legacy_p2r_edge_count)
                if legacy_p2r_edge_count
                else None
            ),
            "r2p_edge_ratio_vs_legacy": (
                float(r2p_edge_count / legacy_r2p_edge_count)
                if legacy_r2p_edge_count
                else None
            ),
            "p2r_hard_reset_trigger_count_total": int(
                sum(
                    row["radius_stages"]["p2r"]["hard_reset_trigger_count"]
                    for row in policy_records
                )
            ),
            "r2p_hard_reset_trigger_count_total": int(
                sum(
                    row["radius_stages"]["r2p"]["hard_reset_trigger_count"]
                    for row in policy_records
                )
            ),
            "graph_build_time_seconds_total": float(
                sum(row["graph_build_time_seconds"] for row in policy_records)
            ),
            "metadata_build_time_seconds_total": float(
                sum(row["metadata_build_time_seconds"] for row in policy_records)
            ),
            "all_metadata_finite": all(row["metadata_all_finite"] for row in policy_records),
            "all_graphs_finite": all(row["graph_all_finite"] for row in policy_records),
            "all_node_and_r2r_stable_vs_legacy": all(
                row["node_and_r2r_stable_vs_legacy"] for row in policy_records
            ),
            "all_dummy_excluded": all(row["dummy_excluded"] for row in policy_records),
            "coverage_gate_passed": p2r_zero_count == 0 and r2p_zero_count == 0,
        }
    return summary


def make_payload(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    return {
        "schema_version": "heat3d_v3_graph_coverage_audit_v2",
        "diagnostic_scope": scope,
        "config": config,
        "summary": summarize_records(records),
        "records": records,
    }


def write_json_ignored(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        check = subprocess.run(
            ["git", "check-ignore", "-q", str(relative)],
            cwd=REPO_ROOT,
            check=False,
        )
        if check.returncode != 0:
            raise ValueError(f"Refusing to write non-ignored audit JSON inside repo: {relative}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def print_summary(summary: dict[str, Any]) -> None:
    for policy, values in summary.items():
        print(
            f"{policy}: records={values['record_count']} "
            f"p2r_zero={values['p2r_zero_count_total']} "
            f"r2p_zero={values['r2p_zero_count_total']} "
            f"p2r_edges={values['p2r_real_edge_count_total']} "
            f"r2p_edges={values['r2p_real_edge_count_total']} "
            f"edge_ratio={values['p2r_edge_ratio_vs_legacy']:.3f}/"
            f"{values['r2p_edge_ratio_vs_legacy']:.3f}"
        )


def main() -> int:
    args = parse_args()
    policies = selected_policies(args)
    seeds = parse_seeds(args.rmesh_seeds)
    splits = parse_splits(args.splits)
    records = audit_real_dataset(
        subset=args.subset,
        split_map=args.split_map,
        splits=splits,
        max_samples=args.max_samples,
        seeds=seeds,
        policies=policies,
        k_encoding_mode=args.k_encoding_mode,
        boundary_mask_fallback=args.boundary_mask_fallback,
        repair_p2r=args.repair_p2r,
        repair_r2p=args.repair_r2p,
        min_physical_coverage=args.min_physical_coverage,
    )
    payload = make_payload(
        records=records,
        scope="real Heat3D sample graph coverage audit; no training",
        config={
            "subset": str(args.subset),
            "split_map": str(args.split_map),
            "splits": splits,
            "max_samples": args.max_samples,
            "rmesh_seeds": seeds,
            "policies": policies,
            "k_encoding_mode": args.k_encoding_mode,
            "boundary_mask_fallback": args.boundary_mask_fallback,
            "coverage_repair_policy": args.coverage_repair_policy,
            "radius_policy": args.radius_policy,
            "repair_p2r": args.repair_p2r,
            "repair_r2p": args.repair_r2p,
            "min_physical_coverage": args.min_physical_coverage,
        },
    )
    print_summary(payload["summary"])
    if args.output_json is not None:
        print(f"wrote={write_json_ignored(args.output_json, payload)}")
    print("Heat3D v3 real-data graph coverage audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
