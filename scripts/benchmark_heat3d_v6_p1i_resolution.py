#!/usr/bin/env python3
"""P1i valid-only direct/interpolated/FVM accuracy-latency benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import jax
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_p1i_valid_full_field as full_eval  # noqa: E402
import heat3d_v6_p1i_continuous_core as core  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadyTarget,
)
from rigno.heat3d_v6_dataset import (  # noqa: E402
    Heat3DV6DualRobinDataset,
    V6DualRobinExample,
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    stats_from_checkpoint_payload,
)


SHAPES = {
    1024: (8, 8, 16),
    4096: (16, 16, 16),
    16384: (32, 32, 16),
    65536: (64, 64, 16),
    240825: (65, 65, 57),
}
COARSE_Z_INTERVALS = (2, 2, 1, 1, 2, 1, 2, 1, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _gpu_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "device_kind": device.device_kind,
        "bytes_limit": int(stats.get("bytes_limit", 0)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "peak_bytes_reserved": int(stats.get("peak_bytes_reserved", 0)),
    }


def _target_physics(meta: dict[str, Any], resolution: int) -> dict[str, Any]:
    physics = deepcopy(meta["physics"])
    shape = SHAPES[resolution]
    physics["solver_mesh_intervals_xyz"] = [value - 1 for value in shape]
    if resolution != 240825:
        for layer, intervals in zip(
            physics["layers_bottom_to_top"], COARSE_Z_INTERVALS, strict=True
        ):
            layer["z_intervals"] = int(intervals)
    return physics


def _dual_bounds(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    middle = 0.5 * (axis[:-1] + axis[1:])
    lower = np.concatenate(([axis[0]], middle))
    upper = np.concatenate((middle, [axis[-1]]))
    return lower, upper


def _overlap(lower: np.ndarray, upper: np.ndarray, left: float, right: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(upper, right) - np.maximum(lower, left))


def _continuous_fields(
    meta: dict[str, Any], mesh: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    k = np.asarray(mesh["base_k_diag"], dtype=np.float64).copy()
    q = np.zeros(int(mesh["node_count"]), dtype=np.float64)
    x, y = np.asarray(mesh["x"]), np.asarray(mesh["y"])
    xl, xu = _dual_bounds(x)
    yl, yu = _dual_bounds(y)
    xw = xu - xl
    yw = yu - yl
    layer_ids = np.asarray(mesh["layer_ids"])
    grid = np.asarray(mesh["grid"])
    weights = np.asarray(mesh["weights"], dtype=np.float64)
    lx, ly = float(x[-1]), float(y[-1])
    layer_index = mesh["layer_index"]

    for block, value in zip(meta["k_blocks"], meta["k_block_values_W_mK"], strict=True):
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        fx = _overlap(xl, xu, x0 * lx, x1 * lx) / xw
        fy = _overlap(yl, yu, y0 * ly, y1 * ly) / yw
        fraction_xy = fx[:, None] * fy[None, :]
        layer = int(layer_index[str(block["layer"])])
        fraction = np.broadcast_to(fraction_xy[:, :, None], grid.shape).reshape(-1)
        fraction = np.where(layer_ids == layer, fraction, 0.0)
        k = k * (1.0 - fraction[:, None]) + float(value) * fraction[:, None]

    source_rows = []
    total_power = float(meta["package_total_power_W"])
    for block, fraction_power in zip(
        meta["q_blocks"], meta["q_block_power_fractions"], strict=True
    ):
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        ox = _overlap(xl, xu, x0 * lx, x1 * lx)
        oy = _overlap(yl, yu, y0 * ly, y1 * ly)
        fraction_xy = (ox / xw)[:, None] * (oy / yw)[None, :]
        layer = int(layer_index[str(block["layer"])])
        fraction = np.broadcast_to(fraction_xy[:, :, None], grid.shape).reshape(-1)
        fraction = np.where(layer_ids == layer, fraction, 0.0)
        represented_volume = weights * fraction
        volume = float(np.sum(represented_volume))
        if volume <= 0.0:
            raise RuntimeError("conservative source projection produced zero volume")
        power = total_power * float(fraction_power)
        q += power * represented_volume / volume / weights
        source_rows.append(
            {
                "layer": block["layer"],
                "power_W": power,
                "represented_volume_m3": volume,
                "nonzero_node_count": int(np.sum(represented_volume > 0.0)),
            }
        )
    applied = float(np.sum(q * weights))
    return k, q, {
        "requested_power_W": total_power,
        "applied_power_W": applied,
        "relative_power_error": abs(applied - total_power) / total_power,
        "sources": source_rows,
    }


def _example(
    original: V6DualRobinExample,
    meta: dict[str, Any],
    mesh: dict[str, Any],
    k: np.ndarray,
    q: np.ndarray,
) -> V6DualRobinExample:
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    flags = core.boundary_flags(coords, mesh)
    count = len(coords)
    top_h = float(meta["top_h_W_m2K"])
    bottom_h = float(meta["bottom_h_W_m2K"])
    features = np.column_stack(
        (
            k,
            q,
            flags,
            np.full(count, top_h),
            np.full(count, bottom_h),
            np.zeros(count),
        )
    )
    if features.shape != (count, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
        raise RuntimeError("direct-support condition schema drifted")
    enriched = deepcopy(meta)
    enriched["v6_adapter"] = dict(original.meta["v6_adapter"])
    return V6DualRobinExample(
        sample_id=original.sample_id,
        condition=V1SteadyConditionInput(
            coords=coords,
            condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((count, 1), 300.0)),
        meta=enriched,
        operator_point_weights=np.asarray(mesh["weights"], dtype=np.float64),
    )


def _build_group(
    example: V6DualRobinExample,
    train_examples: list[V6DualRobinExample],
    stats: dict[str, Any],
    model_config: dict[str, Any],
    graph_config: dict[str, Any],
    graph_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    started = time.perf_counter()
    builder = runner.Heat3DGraphBuilder(**graph_config)
    groups = runner._make_v6_padded_groups_with_progress(
        [example], stats, builder, "valid_iid_benchmark", False, "off", graph_seed,
        batch_size=1, drop_last=False,
    )
    lookup, context_payload = runner._prepare_global_context_lookup(
        model_config, train_examples=train_examples, required_examples=[example]
    )
    runner._attach_global_context_to_groups(
        groups, lookup,
        expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
    )
    examples = {example.sample_id: example}
    runner._attach_native_physics_to_groups(groups, examples)
    if (
        model_config.get("scale_pooling") == "qk_gated"
        or model_config.get("shape_attention_mode") != "none"
        or model_config.get("scale_attention_mode") != "none"
    ):
        runner._attach_qk_region_features_to_groups(
            groups,
            examples,
            feature_version=str(model_config.get("qk_region_feature_version")),
        )
    return groups[0], context_payload, float(time.perf_counter() - started)


def _predict_once(model, params, group, stats) -> np.ndarray:
    result = runner._predict_temperatures(model, params, [group], stats)
    return np.asarray(result[group["sample_ids"][0]], dtype=np.float64).reshape(-1)


def _rgi(mesh: dict[str, Any], values: np.ndarray, query: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (mesh["x"], mesh["y"], mesh["z"]),
        np.asarray(values, dtype=np.float64).reshape(mesh["shape"]),
        bounds_error=True,
    )
    return np.asarray(interpolator(query), dtype=np.float64)


def _assemble(
    mesh: dict[str, Any], k: np.ndarray, q: np.ndarray, top_h: float, bottom_h: float
):
    i, j, g = core._neighbor_faces(mesh, k)
    n = int(mesh["node_count"])
    diagonal = np.bincount(
        np.concatenate((i, j)), weights=np.concatenate((g, g)), minlength=n
    )
    rhs = q * np.asarray(mesh["weights"], dtype=np.float64)
    grid = np.asarray(mesh["grid"], dtype=np.int64)
    dx, dy, _ = mesh["widths"]
    area = (dx[:, None] * dy[None, :]).reshape(-1)
    top_nodes = grid[:, :, -1].reshape(-1)
    bottom_nodes = grid[:, :, 0].reshape(-1)
    top_robin = top_h * area
    bottom_robin = bottom_h * area
    diagonal[top_nodes] += top_robin
    diagonal[bottom_nodes] += bottom_robin
    rhs[top_nodes] += top_robin * 300.0
    rhs[bottom_nodes] += bottom_robin * 300.0
    rows = np.concatenate((i, j, np.arange(n, dtype=np.int64)))
    cols = np.concatenate((j, i, np.arange(n, dtype=np.int64)))
    values = np.concatenate((-g, -g, diagonal))
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    preconditioner = LinearOperator(
        (n, n),
        matvec=lambda value: np.asarray(value, dtype=np.float64) / diagonal,
        dtype=np.float64,
    )
    return matrix, rhs, preconditioner


def _solve(matrix, rhs, preconditioner) -> np.ndarray:
    temperature, info = cg(
        matrix,
        rhs,
        x0=np.full(len(rhs), 300.0),
        rtol=1.0e-10,
        atol=0.0,
        maxiter=20000,
        M=preconditioner,
    )
    if info != 0:
        raise RuntimeError(f"CG failed: {info}")
    return np.asarray(temperature, dtype=np.float64)


def _metric_summary(
    full_coords, full_cv, full_layer, boundaries, truth, prediction, meta
) -> dict[str, Any]:
    accumulator = full_eval.Metrics(full_cv, full_coords, full_layer, boundaries)
    source = full_eval._source_mask(full_coords, full_layer, meta)
    accumulator.add(meta["sample_id"], prediction, truth, source)
    return accumulator.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=sorted(SHAPES), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 20:
        raise ValueError("formal timing requires at least 20 repeats")

    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    install_checkpoint_feature_hooks(checkpoint["train_only_normalization"])
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root, args.manifest, include_roles={"train", "valid_iid"}
    )
    index = dataset.sample_index_by_id()
    train_examples = [dataset[index[value]] for value in dataset.split_ids["train"]]
    valid_id = dataset.split_ids["valid_iid"][0]
    original = dataset[index[valid_id]]
    stats = stats_from_checkpoint_payload(
        checkpoint["train_only_normalization"], train_examples
    )
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    meta = deepcopy(original.meta)
    meta.pop("v6_adapter", None)
    meta["sample_id"] = valid_id

    with h5py.File(args.full_fields, "r") as archive:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in archive["samples/sample_id"][:]]
        full_index = ids.index(valid_id)
        full_coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        full_cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        full_layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
        truth = np.asarray(archive["samples/deltaT_K"][full_index], dtype=np.float64)
    boundaries = full_eval._boundaries(meta, float(np.min(full_coords[:, 2])))
    full_mesh = core.build_mesh(meta["physics"])

    data_started = time.perf_counter()
    target_mesh = core.build_mesh(_target_physics(meta, args.resolution))
    k, q, power_audit = _continuous_fields(meta, target_mesh)
    direct_example = _example(original, meta, target_mesh, k, q)
    data_prepare_seconds = time.perf_counter() - data_started
    if int(target_mesh["node_count"]) != args.resolution:
        raise RuntimeError("exact target node count contract failed")
    if power_audit["relative_power_error"] > 1.0e-12:
        raise RuntimeError("target-grid power conservation failed")

    # Route C: the same exact structured target mesh.
    assembly_times, solve_times, solver_e2e = [], [], []
    solver_temperature = None
    for _ in range(args.repeats + 1):
        e2e_started = time.perf_counter()
        started = time.perf_counter()
        matrix, rhs, preconditioner = _assemble(
            target_mesh, k, q, float(meta["top_h_W_m2K"]), float(meta["bottom_h_W_m2K"])
        )
        assembly = time.perf_counter() - started
        started = time.perf_counter()
        solver_temperature = _solve(matrix, rhs, preconditioner)
        solve = time.perf_counter() - started
        e2e = time.perf_counter() - e2e_started
        if assembly_times:
            assembly_times.append(assembly)
            solve_times.append(solve)
            solver_e2e.append(e2e)
        else:
            assembly_times.append(assembly)
            solve_times.append(solve)
            solver_e2e.append(e2e)
    cold_solver = {
        "assembly_seconds": assembly_times.pop(0),
        "linear_solve_seconds": solve_times.pop(0),
        "end_to_end_seconds": solver_e2e.pop(0),
    }
    solver_full = _rgi(target_mesh, solver_temperature - 300.0, full_coords)

    # Route B: original frozen 1024 inference and layer-aware full reconstruction,
    # then resample through the exact target grid.
    original_graph_config = dict(run_config["graph_config"])
    original_group, original_context, original_graph_seconds = _build_group(
        original, train_examples, stats, model_config, original_graph_config,
        int(run_config["graph_seed"]),
    )
    started = time.perf_counter()
    original_prediction = _predict_once(model, params, original_group, stats)
    original_jit_seconds = time.perf_counter() - started
    original_forward_times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        original_prediction = _predict_once(model, params, original_group, stats)
        original_forward_times.append(time.perf_counter() - started)
    support_coords = np.asarray(original.condition.coords, dtype=np.float64)
    distance, support_indices = cKDTree(full_coords).query(support_coords, k=1)
    if float(np.max(distance)) > 1.0e-14:
        raise RuntimeError("frozen 1024 support is not a solver-node subset")
    mapping, mapping_audit = runner.build_reconstruction_map(
        coords=full_coords,
        layer_id=full_layer,
        boundaries=boundaries,
        support_indices=np.asarray(support_indices, dtype=np.int32),
        empty_domain_fallback="same_layer",
    ) if hasattr(runner, "build_reconstruction_map") else (None, None)
    if mapping is None:
        from rigno.heat3d_v6_full_field import build_reconstruction_map
        mapping, mapping_audit = build_reconstruction_map(
            coords=full_coords, layer_id=full_layer, boundaries=boundaries,
            support_indices=np.asarray(support_indices, dtype=np.int32),
            empty_domain_fallback="same_layer",
        )
    reconstructed_1024 = mapping.reconstruct(original_prediction - 300.0)
    oracle_1024 = mapping.reconstruct(
        np.asarray(original.target.target_u, dtype=np.float64).reshape(-1) - 300.0
    )
    interpolation_times = []
    interpolated_full = oracle_full = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        target_prediction = _rgi(full_mesh, reconstructed_1024, target_mesh["coords"])
        interpolated_full = _rgi(target_mesh, target_prediction, full_coords)
        target_oracle = _rgi(full_mesh, oracle_1024, target_mesh["coords"])
        oracle_full = _rgi(target_mesh, target_oracle, full_coords)
        interpolation_times.append(time.perf_counter() - started)

    payload: dict[str, Any] = {
        "schema_version": "heat3d_v6_p1i_resolution_benchmark_v1",
        "status": "partial_before_direct_route",
        "resolution": args.resolution,
        "actual_target_node_count": int(target_mesh["node_count"]),
        "target_mesh_shape": list(target_mesh["shape"]),
        "sample_ids": [valid_id],
        "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path), "epoch": int(checkpoint["epoch"])},
        "dataset": {"manifest_sha256": _sha256(args.manifest), "full_fields_sha256": _sha256(args.full_fields)},
        "power_audit": power_audit,
        "timing_contract": {"repeats": args.repeats, "batch_size": 1, "warmup_repeats": 1, "no_cross_run_addition": True},
        "data_prepare_seconds": data_prepare_seconds,
        "route_B_1024_plus_interpolation": {
            "graph_build_seconds": original_graph_seconds,
            "jit_first_seconds": original_jit_seconds,
            "steady_model_inference_seconds": _distribution(original_forward_times),
            "interpolation_seconds": _distribution(interpolation_times),
            "steady_end_to_end_seconds": _distribution([a + b for a, b in zip(original_forward_times, interpolation_times, strict=True)]),
            "metrics": _metric_summary(full_coords, full_cv, full_layer, boundaries, truth, interpolated_full, meta),
            "oracle_reconstruction_metrics": _metric_summary(full_coords, full_cv, full_layer, boundaries, truth, oracle_full, meta),
            "reconstruction_map_audit": mapping_audit,
            "global_context_fit_population": original_context["standardizer"]["fit_population"],
        },
        "route_C_structured_FVM": {
            "cold": cold_solver,
            "assembly_seconds": _distribution(assembly_times),
            "linear_solve_seconds": _distribution(solve_times),
            "steady_end_to_end_seconds": _distribution(solver_e2e),
            "solver_tolerance": {"rtol": 1.0e-10, "atol": 0.0, "maxiter": 20000},
            "metrics": _metric_summary(full_coords, full_cv, full_layer, boundaries, truth, solver_full, meta),
        },
        "process_peak_ram_bytes_before_direct": _rss_bytes(),
        "gpu_memory_before_direct": _gpu_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # Route A: direct graph/model at the exact target resolution.  Sparse KD-tree
    # is a graph-builder backend substitution only; scientific graph parameters
    # remain identical to the frozen training config.
    direct_config = dict(run_config["graph_config"])
    direct_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    direct_config["discrete_graph_chunk_size"] = 2048
    try:
        direct_group, direct_context, direct_graph_seconds = _build_group(
            direct_example, train_examples, stats, model_config, direct_config,
            int(run_config["graph_seed"]),
        )
        started = time.perf_counter()
        direct_prediction = _predict_once(model, params, direct_group, stats)
        direct_jit_seconds = time.perf_counter() - started
        direct_forward_times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            direct_prediction = _predict_once(model, params, direct_group, stats)
            direct_forward_times.append(time.perf_counter() - started)
        direct_interp_times = []
        direct_full = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            direct_full = _rgi(target_mesh, direct_prediction - 300.0, full_coords)
            direct_interp_times.append(time.perf_counter() - started)
        payload["route_A_direct"] = {
            "status": "passed",
            "graph_backend": "sparse_kdtree_v1",
            "scientific_graph_parameters_unchanged": True,
            "graph_build_seconds": direct_graph_seconds,
            "jit_first_seconds": direct_jit_seconds,
            "steady_model_inference_seconds": _distribution(direct_forward_times),
            "full_field_interpolation_seconds": _distribution(direct_interp_times),
            "steady_end_to_end_seconds": _distribution([a + b for a, b in zip(direct_forward_times, direct_interp_times, strict=True)]),
            "metrics": _metric_summary(full_coords, full_cv, full_layer, boundaries, truth, direct_full, meta),
            "global_context_fit_population": direct_context["standardizer"]["fit_population"],
        }
        payload["status"] = "passed"
    except Exception as error:  # OOM/incompatibility is a required result.
        payload["route_A_direct"] = {
            "status": "failed_incompatible_or_oom",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "graph_backend": "sparse_kdtree_v1",
            "scientific_graph_parameters_unchanged": True,
        }
        payload["status"] = "passed_with_direct_route_failure_recorded"
    payload["process_peak_ram_bytes"] = _rss_bytes()
    payload["gpu_memory"] = _gpu_memory()
    payload["environment"] = {
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "jax": jax.__version__, "numpy": np.__version__,
        "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
        "cpu_count": os.cpu_count(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "resolution": args.resolution, "route_A": payload["route_A_direct"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
