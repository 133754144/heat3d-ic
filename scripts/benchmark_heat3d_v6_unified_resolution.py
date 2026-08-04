#!/usr/bin/env python3
"""Unified valid-only 1024-inference/reconstruction and matched-count FVM ladder.

The model route always evaluates the frozen 1024 source-aware support and then
reconstructs directly onto an N-node, label-independent support containing all
1024 anchors plus volume-PPS query nodes.  The FVM route uses a legal structured
mesh with exactly N nodes.  The two routes therefore match node count, not DOF
placement; that distinction is recorded explicitly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import jax
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
import yaml


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

_base_override = Path(__file__).with_name("benchmark_heat3d_v6_inference_qualification.py")
if _base_override.resolve() != (ROOT / "scripts/benchmark_heat3d_v6_inference_qualification.py").resolve():
    _base_spec = importlib.util.spec_from_file_location("benchmark_heat3d_v6_inference_qualification", _base_override)
    if _base_spec is None or _base_spec.loader is None:
        raise RuntimeError(f"cannot import benchmark base override {_base_override}")
    base = importlib.util.module_from_spec(_base_spec)
    _base_spec.loader.exec_module(base)
else:
    import benchmark_heat3d_v6_inference_qualification as base  # noqa: E402
import benchmark_heat3d_v6_p1i_resolution as p1i  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402


RESOLUTIONS = (4096, 8192, 16384, 32768, 65536, 240825)
SHAPES = {
    4096: (16, 16, 16),
    8192: (16, 32, 16),
    16384: (32, 32, 16),
    32768: (32, 64, 16),
    65536: (64, 64, 16),
    240825: (65, 65, 57),
}
COARSE_Z_INTERVALS = (2, 2, 1, 1, 2, 1, 2, 1, 3)
STATES = ("process_cold", "new_topology", "known_topology_new_physics", "fully_cached")
ROUTES = ("production_reconstruction", "fvm")


def distribution(values: Sequence[float], *, require_formal_count: bool = True) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if require_formal_count and array.size < 20:
        raise RuntimeError(f"formal distribution requires >=20 values, got {array.size}")
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


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def serialize(array: np.ndarray) -> int:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return len(buffer.getvalue())


def boundaries(example: Any) -> np.ndarray:
    meta = example.meta
    raw = meta.get("physics", {}).get("layer_boundaries_m")
    if raw:
        return np.asarray(raw, dtype=np.float64)
    layers = meta.get("layers_bottom_to_top") or meta.get("physics", {}).get("layers_bottom_to_top")
    values = [0.0]
    for layer in layers:
        values.append(values[-1] + float(layer["thickness_m"]))
    return np.asarray(values, dtype=np.float64)


def target_support(
    family: str,
    sample_id: str,
    example: Any,
    shared: Mapping[str, np.ndarray],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    support = np.asarray(example.condition.coords, dtype=np.float64)
    distance, anchor_full = cKDTree(shared["coords"]).query(support, k=1)
    if float(np.max(distance)) > 1e-14 or len(np.unique(anchor_full)) != 1024:
        raise RuntimeError("1024 anchors are not unique solver nodes")
    anchor_full = np.asarray(anchor_full, dtype=np.int64)
    if resolution == len(shared["coords"]):
        extra = np.setdiff1d(np.arange(resolution, dtype=np.int64), anchor_full, assume_unique=False)
    else:
        remaining = np.setdiff1d(
            np.arange(len(shared["coords"]), dtype=np.int64), anchor_full, assume_unique=False
        )
        seed = int.from_bytes(
            hashlib.sha256(f"{family}|{sample_id}|{resolution}|volume_pps_v1".encode()).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        probability = np.asarray(shared["cv"], dtype=np.float64)[remaining]
        probability = probability / np.sum(probability)
        extra = rng.choice(remaining, size=resolution - 1024, replace=False, p=probability)
    indices = np.concatenate((anchor_full, np.sort(np.asarray(extra, dtype=np.int64))))
    if indices.shape != (resolution,) or len(np.unique(indices)) != resolution:
        raise RuntimeError("target support cardinality contract failed")
    coords = np.asarray(shared["coords"])[indices]
    layers = np.asarray(shared["layer"])[indices]
    mapping, audit = build_reconstruction_map(
        coords=coords,
        layer_id=layers,
        boundaries=boundaries(example),
        support_indices=np.arange(1024, dtype=np.int32),
        empty_domain_fallback="same_layer",
    )
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    return indices, layers, mapping, {
        "selection": "all_1024_source_aware_anchors_plus_label_independent_volume_PPS_v1",
        "indices_sha256": digest,
        "anchor_count": 1024,
        "query_count": resolution - 1024,
        "mapping_audit": audit,
    }


def model_once(
    data: base.FamilyData,
    runtime: base.ModelRuntime,
    row: Mapping[str, Any],
    resolution: int,
    caches: dict[str, Any],
    *,
    state: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    e2e = time.perf_counter()
    started = time.perf_counter(); example, public = data.load_example(row); data_s = time.perf_counter() - started
    sample_id = str(row["sample_id"])
    key = base.support_key(example)
    started = time.perf_counter()
    group = caches["groups"].get(sample_id)
    if group is None:
        group = runtime.graph(example)
    graph_s = time.perf_counter() - started
    started = time.perf_counter(); prediction = runtime.forward(group); forward_s = time.perf_counter() - started
    shared = caches["shared"]
    started = time.perf_counter()
    map_item = caches["maps"].get((key, resolution))
    if map_item is None:
        indices, layer, mapping, audit = target_support(data.family, sample_id, example, shared, resolution)
        map_item = (indices, layer, mapping, audit)
    map_build_s = time.perf_counter() - started
    indices, layer, mapping, audit = map_item
    started = time.perf_counter(); output = mapping.reconstruct(prediction - public["reference_K"]); map_apply_s = time.perf_counter() - started
    started = time.perf_counter(); output_bytes = serialize(output); output_s = time.perf_counter() - started
    serialization_cutoff = time.perf_counter()
    wall = time.perf_counter() - e2e
    truth = data.truth(row, include_full_kq=False)["full_delta"][indices]
    q = data.full_kq(row)[1][indices]
    metric_row = {
        "prediction": output,
        "truth": truth,
        "weights": np.asarray(shared["cv"])[indices],
        "coords": np.asarray(shared["coords"])[indices],
        "layer": layer,
        "q": q,
    }
    return {
        "data_seconds": data_s,
        "graph_seconds": graph_s,
        "jit_or_forward_seconds": forward_s,
        "map_build_seconds": map_build_s,
        "map_apply_seconds": map_apply_s,
        "output_seconds": output_s,
        "continuous_wall_seconds": wall,
        "output_bytes": output_bytes,
        "prediction_serialization_completed_monotonic_s": serialization_cutoff,
    }, {"metric_row": metric_row, "target_support": audit}


def target_physics(physics: Mapping[str, Any], resolution: int) -> dict[str, Any]:
    result = deepcopy(dict(physics))
    shape = SHAPES[resolution]
    result["solver_mesh_intervals_xyz"] = [value - 1 for value in shape]
    if resolution != 240825:
        for layer, count in zip(result["layers_bottom_to_top"], COARSE_Z_INTERVALS, strict=True):
            layer["z_intervals"] = count
    return result


def regular_truth(shared: Mapping[str, np.ndarray], values: np.ndarray, query: np.ndarray) -> np.ndarray:
    coords = np.asarray(shared["coords"], dtype=np.float64)
    x, y, z = (np.unique(coords[:, axis]) for axis in range(3))
    if x.size * y.size * z.size != len(coords):
        raise RuntimeError("full-field coordinates are not a structured tensor grid")
    interpolator = RegularGridInterpolator(
        (x, y, z), np.asarray(values, dtype=np.float64).reshape((x.size, y.size, z.size)),
        method="linear", bounds_error=True,
    )
    return np.asarray(interpolator(query), dtype=np.float64)


def import_random_core(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_randomblock_core_unified", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import random-block core {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def random_resolution_preflight(
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = import_random_core(args.randomblock_core)
    config = yaml.safe_load(args.randomblock_config.read_text())
    physics = target_physics(config["physics"], args.resolution)
    mesh = core.build_mesh(physics)
    groups = {str(item["group_id"]): item for item in config["layout_groups"]}
    unique_group_ids = sorted({str(row["group_id"]) for row in selected})
    checked = []
    for group_id in unique_group_ids:
        try:
            core.validate_layout(groups[group_id], mesh)
        except Exception as error:
            return {
                "status": "infeasible_under_frozen_block_resolution_contract",
                "resolution": args.resolution,
                "mesh_shape": list(SHAPES[args.resolution]),
                "failed_group_id": group_id,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "silent_projection_or_validation_bypass_used": False,
            }
        checked.append(group_id)
    return {
        "status": "passed",
        "resolution": args.resolution,
        "mesh_shape": list(SHAPES[args.resolution]),
        "validated_layout_group_count": len(checked),
    }


def fvm_case(
    data: base.FamilyData,
    row: Mapping[str, Any],
    resolution: int,
    random_core: Any | None,
    random_config: Mapping[str, Any] | None,
    cache: dict[str, Any],
    *,
    fully_cached: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    e2e = time.perf_counter(); sample_id = str(row["sample_id"])
    system_key = (sample_id, resolution)
    started = time.perf_counter(); cached = cache.get(system_key)
    if cached is None:
        example, _ = data.load_example(row)
        if data.family == "p1i":
            meta = json.loads((data.sample_dir(row) / "sample_meta.json").read_text())
            physics = target_physics(meta["physics"], resolution)
            mesh = p1i.core.build_mesh(physics)
            k, q, power = p1i._continuous_fields(meta, mesh)
            if power["relative_power_error"] > 1e-12:
                raise RuntimeError("P1i coarse-grid source power drift")
            top_h, bottom_h = float(meta["top_h_W_m2K"]), float(meta["bottom_h_W_m2K"])
        else:
            case_by_id = {str(item["sample_id"]): item for item in random_config["cases"]}
            group_by_id = {str(item["group_id"]): item for item in random_config["layout_groups"]}
            case = case_by_id[sample_id]; group = group_by_id[str(case["group_id"])]
            physics = target_physics(random_config["physics"], resolution)
            mesh = random_core.build_mesh(physics)
            mesh["_randomblock_core"] = random_core
            layout = random_core.validate_layout(group, mesh)
            k, q, _ = random_core.build_case_fields(case, group, mesh, layout)
            top_h, bottom_h = float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"])
        system = None
    else:
        mesh, q, system = cached
    data_s = time.perf_counter() - started
    started = time.perf_counter()
    if system is None:
        system = base.randomblock_assemble(mesh, k, q, top_h, bottom_h) if data.family == "randomblock" else p1i._assemble(mesh, k, q, top_h, bottom_h)
    assembly_s = time.perf_counter() - started
    started = time.perf_counter()
    matrix, rhs, preconditioner = system; iterations = 0
    def callback(_):
        nonlocal iterations
        iterations += 1
    temperature, info = cg(matrix, rhs, x0=np.full(rhs.size, 300.0), rtol=1e-10, atol=0.0, maxiter=20000, M=preconditioner, callback=callback)
    if info != 0:
        raise RuntimeError(f"CG failed: {info}")
    solve_s = time.perf_counter() - started
    started = time.perf_counter(); output_bytes = serialize(temperature - 300.0); output_s = time.perf_counter() - started
    serialization_cutoff = time.perf_counter()
    wall = time.perf_counter() - e2e
    truth_full = data.truth(row, include_full_kq=False)["full_delta"]
    shared = data.full_shared()
    truth = regular_truth(shared, truth_full, np.asarray(mesh["coords"], dtype=np.float64))
    weights = np.asarray(mesh.get("weights") if "weights" in mesh else mesh["info"]["weights"], dtype=np.float64)
    layer = np.asarray(mesh.get("layer_ids"), dtype=np.int32)
    metric_row = {
        "prediction": np.asarray(temperature) - 300.0,
        "truth": truth,
        "weights": weights,
        "coords": np.asarray(mesh["coords"]),
        "layer": layer,
        "q": np.asarray(q),
    }
    return {
        "data_seconds": data_s,
        "assembly_seconds": assembly_s,
        "linear_solve_seconds": solve_s,
        "output_seconds": output_s,
        "continuous_wall_seconds": wall,
        "cg_iterations": iterations,
        "output_bytes": output_bytes,
        "prediction_serialization_completed_monotonic_s": serialization_cutoff,
    }, {"metric_row": metric_row, "mesh_shape": list(SHAPES[resolution]), "cache_item": (mesh, q, system)}


def summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    return {
        key: distribution(
            [float(row[key]) for row in rows],
            require_formal_count=len(rows) >= 20,
        )
        for key in sorted(rows[0]) if key.endswith("_seconds")
    }


def worker(args: argparse.Namespace) -> int:
    data = base.FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    selected = data.selected_rows(args.sample_count)
    if args.sample_id:
        selected = [data.row_by_id[args.sample_id]]
    runtime = None
    if args.route == "production_reconstruction":
        runtime = base.ModelRuntime(
            args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, args.edge_targets,
            verify_checkpoint_sha=not args.checkpoint_sha_preverified,
        )
    random_core = import_random_core(args.randomblock_core) if args.family == "randomblock" and args.route == "fvm" else None
    random_config = yaml.safe_load(args.randomblock_config.read_text()) if args.family == "randomblock" else None
    caches: dict[str, Any] = {"groups": {}, "maps": {}, "shared": data.full_shared()}
    system_cache: dict[str, Any] = {}
    cache_prep = 0.0
    if args.state == "new_topology":
        if args.family != "p1i" or args.route != "production_reconstruction":
            raise RuntimeError("new_topology is only valid for varying-support P1i model inference")
        warm = data.warmup_rows(selected)
        for row in warm:
            example, _ = data.load_example(row); runtime.forward(runtime.graph(example))
        warm_hashes = {base.support_key(data.load_example(row)[0]) for row in warm}
        measured_hashes = {base.support_key(data.load_example(row)[0]) for row in selected}
        if warm_hashes & measured_hashes:
            raise RuntimeError("new topology cache was prefilled")
    elif args.state == "known_topology_new_physics":
        if args.family != "randomblock" and args.route != "fvm":
            raise RuntimeError("P1i model supports have no preregistered same-topology physics pairs")
        if args.route == "production_reconstruction":
            for row in data.warmup_rows(selected):
                example, _ = data.load_example(row); runtime.forward(runtime.graph(example))
    elif args.state == "fully_cached":
        started = time.perf_counter()
        for row in selected:
            if args.route == "production_reconstruction":
                example, _ = data.load_example(row); sample_id = str(row["sample_id"])
                caches["groups"][sample_id] = runtime.graph(example); runtime.forward(caches["groups"][sample_id])
                key = base.support_key(example)
                indices, layer, mapping, audit = target_support(args.family, sample_id, example, caches["shared"], args.resolution)
                caches["maps"][(key, args.resolution)] = (indices, layer, mapping, audit)
            else:
                _, extra = fvm_case(data, row, args.resolution, random_core, random_config, {}, fully_cached=False)
                system_cache[(str(row["sample_id"]), args.resolution)] = extra["cache_item"]
        cache_prep = time.perf_counter() - started
    measurements = []; metric_rows = []; support_audits = []; cg_values = []
    for row in selected:
        if args.route == "production_reconstruction":
            measurement, extra = model_once(data, runtime, row, args.resolution, caches, state=args.state)
            support_audits.append(extra["target_support"])
        else:
            measurement, extra = fvm_case(data, row, args.resolution, random_core, random_config, system_cache, fully_cached=args.state == "fully_cached")
            cg_values.append(int(measurement["cg_iterations"]))
        measurements.append(measurement); metric_rows.append(extra["metric_row"])
    metrics = base.metric_accumulate(metric_rows, full=False)
    metrics["domain"] = f"resolution_{args.resolution}"
    payload = {
        "schema_version": "heat3d_v6_unified_resolution_worker_v1",
        "status": "passed", "family": args.family, "route": args.route,
        "state": args.state, "resolution": args.resolution,
        "sample_ids": [str(row["sample_id"]) for row in selected],
        "measurements": measurements, "stage_timing": summarize(measurements),
        "metrics": metrics,
        "cg_iterations": distribution(cg_values, require_formal_count=len(cg_values) >= 20) if cg_values else None,
        "cache_preparation_seconds_outside_timing": cache_prep,
        "target_support_audits": support_audits,
        "process_peak_ram_bytes": rss_bytes(),
        "device_memory": base.device_memory(),
        "production_contract": {
            "batch_size": 1,
            "node_count_matched_but_dof_placement_nonmatched": True,
            "model_route": "frozen_1024_source_aware_inference_to_N_node_reconstruction_and_output",
            "fvm_route": "legal_structured_FVM_exact_N_nodes",
            "metrics_oracle_hash_json_checker_outside_timing": True,
        },
        "accessed_roles": ["valid_iid", "train_frozen_normalization_metadata"] if args.route.startswith("production") else ["valid_iid"],
        "test_accessed": False, "sealed_accessed": False, "training_executed": False,
        "environment": {"host": platform.node(), "python": sys.version, "jax": jax.__version__, "device": str(jax.devices()[0]), "cpu_count": os.cpu_count()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "family": args.family, "route": args.route, "state": args.state, "resolution": args.resolution}))
    return 0


def command(args: argparse.Namespace, route: str, state: str, output: Path, sample_id: str | None = None) -> list[str]:
    values = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--family", args.family,
        "--route", route, "--state", state, "--resolution", str(args.resolution),
        "--sample-count", str(args.sample_count), "--dataset-root", str(args.dataset_root),
        "--manifest", str(args.manifest), "--full-fields", str(args.full_fields),
        "--run-dir", str(args.run_dir), "--checkpoint-sha256", args.checkpoint_sha256,
        "--checkpoint-epoch", str(args.checkpoint_epoch), "--edge-targets", str(args.edge_targets),
        "--output", str(output), "--checkpoint-sha-preverified",
    ]
    if args.randomblock_config: values += ["--randomblock-config", str(args.randomblock_config)]
    if args.randomblock_core: values += ["--randomblock-core", str(args.randomblock_core)]
    if sample_id: values += ["--sample-id", sample_id]
    return values


def run_child(values: list[str], *, cpu: bool) -> tuple[dict[str, Any], float]:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MEM_FRACTION": "0.85", "HEAT3D_REPO_ROOT": str(ROOT)})
    if cpu: env["JAX_PLATFORMS"] = "cpu"; env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter(); done = subprocess.run(values, env=env, text=True, capture_output=True)
    if done.returncode:
        raise RuntimeError(f"child failed: {' '.join(values)}\n{done.stdout}\n{done.stderr}")
    payload = json.loads(Path(values[values.index("--output") + 1]).read_text())
    wall = time.perf_counter() - started
    if payload["state"] == "process_cold":
        wall = float(payload["measurements"][-1]["prediction_serialization_completed_monotonic_s"]) - started
    return payload, wall


def orchestrate(args: argparse.Namespace) -> int:
    checkpoint = args.run_dir / "params_best_valid_point_global.pkl"
    if base.sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA preflight failed")
    data = base.FamilyData(family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest, full_fields_path=args.full_fields, randomblock_config=args.randomblock_config)
    selected = data.selected_rows(args.sample_count); args.work_dir.mkdir(parents=True, exist_ok=True)
    results = {}; commands = []
    for route in ROUTES:
        if route == "fvm" and args.family == "randomblock":
            preflight = random_resolution_preflight(args, selected)
            if preflight["status"] != "passed":
                results[route] = {
                    state: {"status": "not_run_resolution_infeasible", "preflight": preflight}
                    for state in STATES
                }
                continue
        result = {}
        for state in STATES:
            applicable = not (
                state == "new_topology" and (route == "fvm" or args.family == "randomblock")
                or state == "known_topology_new_physics" and args.family == "p1i" and route != "fvm"
            )
            if not applicable:
                result[state] = {"status": "not_applicable_under_frozen_contract"}
                continue
            if state == "process_cold":
                payloads = []; walls = []
                for index, row in enumerate(selected):
                    output = args.work_dir / f"{args.family}_{args.resolution}_{route}_cold_{index:02d}.json"
                    values = command(args, route, state, output, str(row["sample_id"])); commands.append(" ".join(values))
                    payload, wall = run_child(values, cpu=route == "fvm"); payloads.append(payload); walls.append(wall)
                    print(f"[unified] {args.family} N={args.resolution} {route} cold {index+1}/32 {wall:.3f}s", flush=True)
                result[state] = {
                    "status": "passed", "external_continuous_wall_seconds": distribution(walls),
                    "stage_timing": {key: distribution([float(item["measurements"][0][key]) for item in payloads]) for key in payloads[0]["stage_timing"]},
                    "peak_ram_bytes": max(int(item["process_peak_ram_bytes"]) for item in payloads),
                    "peak_device_bytes": max(int(item["device_memory"]["peak_bytes_in_use"]) for item in payloads),
                    "metrics": base.metric_accumulate([item["metrics"] for item in []], full=False) if False else None,
                    "sample_ids": [item["sample_ids"][0] for item in payloads],
                    "worker_metrics": [item["metrics"] for item in payloads],
                    "cg_iterations": distribution([float(item["measurements"][0].get("cg_iterations", 0)) for item in payloads]) if route == "fvm" else None,
                }
            else:
                output = args.work_dir / f"{args.family}_{args.resolution}_{route}_{state}.json"
                values = command(args, route, state, output); commands.append(" ".join(values))
                payload, wall = run_child(values, cpu=route == "fvm"); payload["external_process_wall_seconds"] = wall
                result[state] = payload
                print(f"[unified] {args.family} N={args.resolution} {route} {state} {wall:.3f}s", flush=True)
        results[route] = result
    output = {
        "schema_version": "heat3d_v6_unified_resolution_family_v1", "status": "passed",
        "family": args.family, "resolution": args.resolution, "sample_count": len(selected),
        "sample_ids": [str(row["sample_id"]) for row in selected], "routes": results,
        "commands": commands,
        "checkpoint": {"sha256": args.checkpoint_sha256, "epoch": args.checkpoint_epoch},
        "dataset": {"manifest_sha256": base.sha256(args.manifest), "full_fields_sha256": base.sha256(args.full_fields)},
        "contract": {"valid_only": True, "test_accessed": False, "sealed_accessed": False, "batch_size": 1, "fixed_cpu_threads": 1, "sample_count": 32},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", choices=("p1i", "randomblock"), required=True)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--state", choices=STATES)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-id")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha-preverified", action="store_true")
    parser.add_argument("--edge-targets", type=Path, required=True)
    parser.add_argument("--randomblock-config", type=Path)
    parser.add_argument("--randomblock-core", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/v6_unified_resolution"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker and (args.route is None or args.state is None): parser.error("worker requires route/state")
    if args.family == "randomblock" and (args.randomblock_config is None or args.randomblock_core is None): parser.error("randomblock requires config/core")
    return args


if __name__ == "__main__":
    parsed = parse_args(); raise SystemExit(worker(parsed) if parsed.worker else orchestrate(parsed))
