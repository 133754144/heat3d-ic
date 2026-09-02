#!/usr/bin/env python3
"""Truth-free Gate A graph replay for the V7 G1 H2 closeout.

The only data input accepted by this tool is the 1024-point support
``coords.npy`` for ``v6p1if1_0993``.  It never opens manifests, sample metadata,
full-field archives, receipts, checkpoints, labels, predictions, or metrics.
The parent process runs a fixed implementation/profile/execution-placement
matrix so a backend/device explanation cannot be selected after observing
counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


def _load_current_sanitizer() -> Any:
    """Load the current guard without importing the historical ``rigno`` package."""

    path = Path(__file__).resolve().parents[1] / "rigno/heat3d_v7_h2_geometry_sanitizer.py"
    spec = importlib.util.spec_from_file_location("v7_h2_geometry_sanitizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("current sanitizer module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SANITIZER = _load_current_sanitizer()
assert_geometry_only_output = _SANITIZER.assert_geometry_only_output
guard_geometry_input_paths = _SANITIZER.guard_geometry_input_paths


SAMPLE_ID = "v6p1if1_0993"
HISTORICAL_COMMIT = "05b32ce"
HISTORICAL_SANITY = {"p2r_count": 3074, "r2r_count": 4075}
GRAPH_SEED = 0
EXPECTED_SUPPORT_COUNT = 1024

FROZEN_GRAPH_CONFIG = {
    "rmesh_levels": 3,
    "subsample_factor": 4.0,
    "overlap_factor_p2r": 1.5,
    "overlap_factor_r2p": 2.0,
    "node_coordinate_encoding": "raw",
    "node_coordinate_freqs": 4,
    "coverage_repair_policy": "none",
    "radius_policy": "discrete_physical_coverage",
    "repair_p2r": True,
    "repair_r2p": True,
    "min_physical_coverage": 1,
    "discrete_graph_chunk_size": 1024,
    "discrete_coverage_multiplier": 1.0,
}

PROFILES = {
    "historical_u2_frozen": {
        **FROZEN_GRAPH_CONFIG,
        "discrete_graph_backend": "sparse_kdtree_v1",
        "reuse_exact_p2r_for_r2p": True,
        "profile_basis": "historical U-v2 runner graph_config update",
    },
    "current_restored_h2": {
        **FROZEN_GRAPH_CONFIG,
        "discrete_graph_backend": "sparse_kdtree_v1",
        "reuse_exact_p2r_for_r2p": True,
        "profile_basis": "current UHighNRuntime.from_session frozen override",
    },
    "current_source_run_config": {
        **FROZEN_GRAPH_CONFIG,
        "discrete_graph_backend": "dense_reference",
        "reuse_exact_p2r_for_r2p": False,
        "profile_basis": "current H2 source run_config control only",
    },
}

EXECUTION_PLACEMENTS = ("cpu", "default")
CURRENT_RESTORED_SANITY = {"p2r_count": 3083, "r2r_count": 4075}

STAGES = (
    "support_indices",
    "support_coordinates",
    "normalized_coordinates",
    "rnodes",
    "radius_config",
    "raw_p2r_edge_multiset",
    "repair_edge_multiset",
    "final_p2r_edge_multiset",
    "final_r2r_edge_multiset",
)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _edge_multiset_sha256(value: Any) -> str:
    edges = np.asarray(value, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    edges = edges.reshape(-1, 2)
    if len(edges):
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
    return _array_sha256(edges)


def _added_edge_multiset(before: Any, after: Any) -> np.ndarray:
    before_rows = [tuple(row) for row in np.asarray(before, dtype=np.int64).reshape(-1, 2)]
    after_rows = [tuple(row) for row in np.asarray(after, dtype=np.int64).reshape(-1, 2)]
    counts = Counter(before_rows)
    added: list[tuple[int, int]] = []
    for row in after_rows:
        if counts[row]:
            counts[row] -= 1
        else:
            added.append(row)
    return np.asarray(added, dtype=np.int64).reshape(-1, 2)


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metadata_real(metadata: Any, field: str, dummy: tuple[int, int]) -> np.ndarray:
    value = np.asarray(getattr(metadata, field))
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 2:
        raise RuntimeError(f"unexpected graph metadata shape for {field}")
    if not np.array_equal(value[0, -1], np.asarray(dummy, dtype=value.dtype)):
        raise RuntimeError(f"missing graph dummy edge for {field}")
    return np.asarray(value[0, :-1], dtype=np.int64)


def _worker_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--execution-placement", choices=EXECUTION_PLACEMENTS, required=True)
    parser.add_argument("--graph-seed", type=int, default=GRAPH_SEED)
    return parser


def _run_worker(args: argparse.Namespace) -> int:
    guard_geometry_input_paths([args.coords])
    if args.implementation not in {"historical_05b32ce", "current_restored_h2"}:
        raise RuntimeError("unsupported implementation label")
    if args.profile not in PROFILES:
        raise RuntimeError("unsupported frozen graph profile")
    if args.execution_placement not in EXECUTION_PLACEMENTS:
        raise RuntimeError("unsupported frozen graph execution placement")
    if args.graph_seed != GRAPH_SEED:
        raise RuntimeError("Gate A graph seed is not frozen")

    coordinates_path = args.coords.resolve()
    coordinates = np.load(coordinates_path, allow_pickle=False)
    if coordinates.shape != (EXPECTED_SUPPORT_COUNT, 3):
        raise RuntimeError("0993 support coordinate shape drifted")
    if coordinates.dtype != np.dtype("float64"):
        raise RuntimeError("0993 support coordinate dtype drifted")
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError("0993 support coordinates are not finite")

    # Imports are deliberately inside the worker.  The parent never imports
    # either implementation, so historical and current module caches cannot
    # contaminate one another.
    import jax
    import jaxlib
    import scipy
    from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
    from rigno.models.rigno import RegionInteractionGraphBuilder

    profile = dict(PROFILES[args.profile])
    profile.pop("profile_basis")
    builder = Heat3DGraphBuilder(**profile)
    inner = builder.builder
    trace: dict[str, Any] = {"effective_radii": [], "raw_support_edges": []}

    original_radius = inner._compute_discrete_physical_coverage_radius

    def traced_radius(centers: Any, points: Any) -> Any:
        value = original_radius(centers, points)
        trace["coverage_centers"] = np.asarray(centers).copy()
        trace["coverage_points"] = np.asarray(points).copy()
        trace["coverage_radius"] = np.asarray(value).copy()
        return value

    inner._compute_discrete_physical_coverage_radius = traced_radius
    original_effective = inner._get_effective_support_radii

    def traced_effective(radii: Any, overlap_factor: float) -> Any:
        value = original_effective(radii, overlap_factor)
        trace["effective_radii"].append({
            "overlap_factor": float(overlap_factor),
            "input": np.asarray(radii).copy(),
            "output": np.asarray(value).copy(),
        })
        return value

    inner._get_effective_support_radii = traced_effective
    original_supported = inner._get_supported_pnodes_by_rnodes

    def traced_supported(centers: Any, points: Any, radii: Any, **kwargs: Any) -> Any:
        value = original_supported(centers, points, radii, **kwargs)
        trace["raw_support_edges"].append({
            "centers": np.asarray(centers).copy(),
            "points": np.asarray(points).copy(),
            "radii": np.asarray(radii).copy(),
            "edges": np.asarray(value).copy(),
        })
        return value

    inner._get_supported_pnodes_by_rnodes = traced_supported
    original_repair = inner._repair_physical_node_coverage

    def traced_repair(edge_indices: Any, centers: Any, points: Any) -> Any:
        value = original_repair(edge_indices, centers, points)
        trace["repair_before"] = np.asarray(edge_indices).copy()
        trace["repair_after"] = np.asarray(value).copy()
        return value

    inner._repair_physical_node_coverage = traced_repair
    original_r2r = inner._get_r2r_edges

    def traced_r2r(rnodes: Any) -> Any:
        edges, domains = original_r2r(rnodes)
        trace["r2r_raw"] = np.asarray(edges).copy()
        trace["r2r_domains_raw"] = np.asarray(domains).copy()
        return edges, domains

    inner._get_r2r_edges = traced_r2r

    cpu = jax.devices("cpu")[0]
    if args.execution_placement == "cpu":
        execution_device = cpu
        with jax.default_device(cpu):
            metadata = builder.build_metadata(
                coordinates,
                key=jax.random.PRNGKey(int(args.graph_seed)),
            )
    else:
        execution_device = jax.devices()[0]
        metadata = builder.build_metadata(
            coordinates,
            key=jax.random.PRNGKey(int(args.graph_seed)),
        )

    raw_p2r = np.asarray(trace["raw_support_edges"][0]["edges"], dtype=np.int64)
    final_p2r = _metadata_real(metadata, "p2r_edge_indices", (len(coordinates), len(trace["coverage_centers"])))
    final_r2r = _metadata_real(metadata, "r2r_edge_indices", (len(trace["coverage_centers"]), len(trace["coverage_centers"])))
    if "repair_before" in trace:
        repair_edges = _added_edge_multiset(trace["repair_before"], trace["repair_after"])
    else:
        repair_edges = np.empty((0, 2), dtype=np.int64)

    effective = trace["effective_radii"]
    if not effective:
        raise RuntimeError("missing effective graph radii trace")
    effective_p2r = np.asarray(effective[0]["output"])
    normalized_coordinates = np.asarray(metadata.x_pnodes_inp)[0, :-1]
    rnodes = np.asarray(metadata.x_rnodes)[0, :-1]
    if normalized_coordinates.shape != coordinates.shape:
        raise RuntimeError("normalized coordinate shape drifted")
    if len(final_p2r) != len(raw_p2r):
        raise RuntimeError("repair unexpectedly changed p2r count without trace")

    graph_config = dict(profile)
    graph_config_sha = _json_sha256(graph_config)
    source_root = args.implementation_root.resolve()
    source_paths = {
        "rigno_graph": source_root / "rigno/models/rigno.py",
        "graph_wrapper": source_root / "rigno/graphBuilder_Heat3D.py",
        "u2_runner": source_root / "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py",
        "anchor_query": source_root / "rigno/heat3d_v6_p1i_anchor_query.py",
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise RuntimeError("historical/current provenance source is incomplete")

    graph_builder_source = inspect.getsource(RegionInteractionGraphBuilder)
    graph_builder_source_sha = hashlib.sha256(graph_builder_source.encode("utf-8")).hexdigest()
    output = {
        "schema_version": "geometry_only_graph_manifest_v1",
        "sample_id": SAMPLE_ID,
        "geometry": {
            "support_coordinates_sha256": _array_sha256(coordinates),
            "normalized_coordinates_sha256": _array_sha256(normalized_coordinates),
            "normalized_coordinates_shape": list(normalized_coordinates.shape),
            "rnodes_sha256": _array_sha256(rnodes),
            "rnodes_count": int(len(rnodes)),
            "domain_sha256": _array_sha256(np.asarray([
                np.min(coordinates, axis=0),
                np.max(coordinates, axis=0),
            ], dtype=np.float64)),
        },
        "support": {
            "support_indices_sha256": _array_sha256(np.arange(EXPECTED_SUPPORT_COUNT, dtype=np.int64)),
            "support_coordinates_sha256": _array_sha256(coordinates),
            "support_count": EXPECTED_SUPPORT_COUNT,
            "support_source": "stored 1024 solver-node support coordinates",
            "support_order": "stored coordinate row order",
        },
        "graph": {
            "implementation": args.implementation,
            "profile": args.profile,
            "execution_placement": args.execution_placement,
            "graph_config": graph_config,
            "config_sha256": graph_config_sha,
            "coverage_radius_sha256": _array_sha256(trace["coverage_radius"]),
            "effective_p2r_radius_sha256": _array_sha256(effective_p2r),
            "raw_p2r_edge_multiset_sha256": _edge_multiset_sha256(raw_p2r),
            "raw_p2r_count": int(len(raw_p2r)),
            "repair_edge_multiset_sha256": _edge_multiset_sha256(repair_edges),
            "repair_edge_count": int(len(repair_edges)),
            "final_p2r_edge_multiset_sha256": _edge_multiset_sha256(final_p2r),
            "final_p2r_count": int(len(final_p2r)),
            "final_r2r_edge_multiset_sha256": _edge_multiset_sha256(final_r2r),
            "final_r2r_count": int(len(final_r2r)),
            "final_r2r_domains_sha256": _array_sha256(
                np.asarray(trace["r2r_domains_raw"], dtype=np.int64)
            ),
        },
        "dependency": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "execution_device": str(execution_device),
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "platform": platform.platform(),
        },
        "provenance": {
            "historical_commit": HISTORICAL_COMMIT,
            "graph_code_sha256": _file_sha256(source_paths["rigno_graph"]),
            "graph_wrapper_sha256": _file_sha256(source_paths["graph_wrapper"]),
            "u2_runner_sha256": _file_sha256(source_paths["u2_runner"]),
            "anchor_query_sha256": _file_sha256(source_paths["anchor_query"]),
            "region_graph_builder_source_sha256": graph_builder_source_sha,
            "graph_seed": int(args.graph_seed),
            "input_coordinates_sha256": _file_sha256(coordinates_path),
        },
    }
    assert_geometry_only_output(output)
    sys.stdout.write(_safe_json(output) + "\n")
    return 0


def _parent_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-repo", type=Path, required=True)
    parser.add_argument("--historical-repo", type=Path, required=True)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _run_cell(
    *,
    script: Path,
    implementation_root: Path,
    coords: Path,
    implementation: str,
    profile: str,
    execution_placement: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--implementation-root",
        str(implementation_root.resolve()),
        "--coords",
        str(coords.resolve()),
        "--implementation",
        implementation,
        "--profile",
        profile,
        "--execution-placement",
        execution_placement,
        "--graph-seed",
        str(GRAPH_SEED),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(implementation_root.resolve()), str((implementation_root / "scripts").resolve())]
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    # stderr is deliberately discarded; it is never copied into a log or
    # decision, and stdout must be exactly one sanitized JSON object.
    if completed.returncode != 0:
        raise RuntimeError("geometry-only worker failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("geometry-only worker emitted non-JSON output") from error
    assert_geometry_only_output(result)
    return result


def _stage_fingerprints(result: Mapping[str, Any]) -> dict[str, str]:
    geometry = result["geometry"]
    support = result["support"]
    graph = result["graph"]
    return {
        "support_indices": str(support["support_indices_sha256"]),
        "support_coordinates": str(support["support_coordinates_sha256"]),
        "normalized_coordinates": str(geometry["normalized_coordinates_sha256"]),
        "rnodes": str(geometry["rnodes_sha256"]),
        "radius_config": _json_sha256({
            "config_sha256": graph["config_sha256"],
            "coverage_radius_sha256": graph["coverage_radius_sha256"],
            "effective_p2r_radius_sha256": graph["effective_p2r_radius_sha256"],
        }),
        "raw_p2r_edge_multiset": str(graph["raw_p2r_edge_multiset_sha256"]),
        "repair_edge_multiset": str(graph["repair_edge_multiset_sha256"]),
        "final_p2r_edge_multiset": str(graph["final_p2r_edge_multiset_sha256"]),
        "final_r2r_edge_multiset": str(graph["final_r2r_edge_multiset_sha256"]),
    }


def _compare_stages(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_values = _stage_fingerprints(left)
    right_values = _stage_fingerprints(right)
    rows = {
        stage: {
            "exact": left_values[stage] == right_values[stage],
            "left_sha256": left_values[stage],
            "right_sha256": right_values[stage],
        }
        for stage in STAGES
    }
    first = next((stage for stage in STAGES if not rows[stage]["exact"]), None)
    return {"stages": rows, "first_divergence_stage": first}


def _parent_main(args: argparse.Namespace) -> int:
    guard_geometry_input_paths([args.coords])
    current_repo = args.current_repo.resolve()
    historical_repo = args.historical_repo.resolve()
    coords = args.coords.resolve()
    if not current_repo.is_dir() or not historical_repo.is_dir() or not coords.is_file():
        raise FileNotFoundError("Gate A input path is missing")

    cells: dict[str, dict[str, Any]] = {}
    for implementation, root in (
        ("historical_05b32ce", historical_repo),
        ("current_restored_h2", current_repo),
    ):
        for profile in PROFILES:
            for execution_placement in EXECUTION_PLACEMENTS:
                key = f"{implementation}__{profile}__{execution_placement}"
                cells[key] = _run_cell(
                    script=Path(__file__).resolve(),
                    implementation_root=root,
                    coords=coords,
                    implementation=implementation,
                    profile=profile,
                    execution_placement=execution_placement,
                )

    historical_reference = cells["historical_05b32ce__historical_u2_frozen__cpu"]
    current_restored = cells["current_restored_h2__current_restored_h2__default"]
    historical_cpu_default = _compare_stages(
        cells["historical_05b32ce__historical_u2_frozen__cpu"],
        cells["historical_05b32ce__historical_u2_frozen__default"],
    )
    current_cpu_default = _compare_stages(
        cells["current_restored_h2__current_restored_h2__cpu"],
        cells["current_restored_h2__current_restored_h2__default"],
    )
    implementation_default = _compare_stages(
        cells["historical_05b32ce__current_restored_h2__default"],
        cells["current_restored_h2__current_restored_h2__default"],
    )
    implementation_cpu = _compare_stages(
        cells["historical_05b32ce__current_restored_h2__cpu"],
        cells["current_restored_h2__current_restored_h2__cpu"],
    )
    historical_profile_cpu = _compare_stages(
        cells["historical_05b32ce__historical_u2_frozen__cpu"],
        cells["historical_05b32ce__current_restored_h2__cpu"],
    )
    current_profile_default = _compare_stages(
        cells["current_restored_h2__historical_u2_frozen__default"],
        cells["current_restored_h2__current_restored_h2__default"],
    )
    primary_pair = _compare_stages(historical_reference, current_restored)

    historical_counts = {
        "p2r_count": int(historical_reference["graph"]["final_p2r_count"]),
        "r2r_count": int(historical_reference["graph"]["final_r2r_count"]),
    }
    current_counts = {
        "p2r_count": int(current_restored["graph"]["final_p2r_count"]),
        "r2r_count": int(current_restored["graph"]["final_r2r_count"]),
    }
    historical_sanity_exact = historical_counts == HISTORICAL_SANITY
    current_sanity_exact = {
        "p2r_count": int(current_restored["graph"]["final_p2r_count"]),
        "r2r_count": int(current_restored["graph"]["final_r2r_count"]),
    } == CURRENT_RESTORED_SANITY
    placement_effect_proven = (
        historical_cpu_default["stages"]["support_indices"]["exact"]
        and historical_cpu_default["stages"]["support_coordinates"]["exact"]
        and historical_cpu_default["stages"]["normalized_coordinates"]["exact"]
        and historical_cpu_default["stages"]["rnodes"]["exact"] is False
        and historical_cpu_default["stages"]["final_p2r_edge_multiset"]["exact"] is False
        and historical_cpu_default["stages"]["final_r2r_edge_multiset"]["exact"] is False
        and current_cpu_default["stages"]["support_coordinates"]["exact"]
        and current_cpu_default["stages"]["normalized_coordinates"]["exact"]
        and implementation_default["stages"]["final_p2r_edge_multiset"]["exact"]
        and implementation_default["stages"]["final_r2r_edge_multiset"]["exact"]
    )
    if historical_sanity_exact and current_sanity_exact and placement_effect_proven:
        root_cause_status = "EXECUTION_DEVICE_PLACEMENT_DIFFERENCE_PROVEN"
        root_cause_detail = "historical U-v2 CPU placement reproduces 3074/4075 while the current restored UHighN default-device placement reproduces 3083/4075; fixed support/normalized coordinates are exact and historical/current graph implementations are edge-exact under the same placement"
    else:
        root_cause_status = "NOT_PROVEN_FAIL_CLOSED"
        root_cause_detail = "fixed implementation/profile/execution-placement matrix did not establish a unique historical/current explanation"

    dependencies = {
        key: cells[key]["dependency"]
        for key in sorted(cells)
    }
    provenance = {
        "gate_a_status": (
            "PASS" if historical_sanity_exact and root_cause_status == "GRAPH_PROFILE_BACKEND_DIFFERENCE_PROVEN"
            else "FAIL_CLOSED"
        ),
        "historical_sanity_status": "EXACT_COUNTS" if historical_sanity_exact else "NOT_EXACT",
        "current_restored_sanity_status": "EXPECTED_COUNTS" if current_sanity_exact else "UNEXPECTED_COUNTS",
        "root_cause_status": root_cause_status,
        "root_cause_detail": root_cause_detail,
        "primary_first_divergence_stage": primary_pair["first_divergence_stage"],
        "historical_commit": HISTORICAL_COMMIT,
        "sample_id": SAMPLE_ID,
        "graph_seed": GRAPH_SEED,
        "cell_count": len(cells),
        "execution_placements": list(EXECUTION_PLACEMENTS),
        "test_sealed_access": "not accepted as an input by this tool",
        "training_or_model_execution": "not performed",
    }
    manifest = {
        "schema_version": "heat3d_v7_g1_h2_gate_a_geometry_recovery_manifest_v1",
        "sample_id": SAMPLE_ID,
        "geometry": {
            "support_count": EXPECTED_SUPPORT_COUNT,
            "support_coordinates_sha256": historical_reference["support"]["support_coordinates_sha256"],
            "normalized_coordinates_sha256": historical_reference["geometry"]["normalized_coordinates_sha256"],
            "historical_current_exact": primary_pair["stages"]["normalized_coordinates"]["exact"],
            "historical": historical_reference["geometry"],
            "current": current_restored["geometry"],
        },
        "support": {
            "support_indices_sha256": historical_reference["support"]["support_indices_sha256"],
            "support_coordinates_sha256": historical_reference["support"]["support_coordinates_sha256"],
            "support_count": EXPECTED_SUPPORT_COUNT,
            "historical_current_exact": primary_pair["stages"]["support_coordinates"]["exact"],
        },
        "graph": {
            "sanity": HISTORICAL_SANITY,
            "historical_counts": historical_counts,
            "current_counts": current_counts,
            "historical": historical_reference["graph"],
            "current": current_restored["graph"],
            "comparison": {
                "primary": primary_pair,
                "historical_cpu_vs_default": historical_cpu_default,
                "current_cpu_vs_default": current_cpu_default,
                "implementation_effect_default": implementation_default,
                "implementation_effect_cpu": implementation_cpu,
                "historical_profile_effect_cpu": historical_profile_cpu,
                "current_profile_effect_default": current_profile_default,
                "historical_sanity_exact": historical_sanity_exact,
                "current_restored_sanity_exact": current_sanity_exact,
                "execution_placement_effect_proven": placement_effect_proven,
            },
        },
        "dependency": {
            "cells": dependencies,
            "historical_current_versions_exact": len({
                _safe_json(row) for row in dependencies.values()
            }) == 1,
        },
        "provenance": provenance,
    }
    assert_geometry_only_output(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if provenance["gate_a_status"] == "PASS" else 2


def main() -> int:
    if "--worker" in sys.argv[1:]:
        return _run_worker(_worker_args().parse_args())
    return _parent_main(_parent_args().parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Never print exception text: it could include an un-sanitized input
        # path or dependency detail.  A nonzero exit is the only worker signal.
        raise SystemExit(3)
