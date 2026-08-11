#!/usr/bin/env python3
"""P5-A3 actual-data coverage-radius timing and graph-hash gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import jax
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import array_sha256  # noqa: E402
from rigno.models.rigno import RegionInteractionGraphBuilder  # noqa: E402


def _reference_coverage(self: Any, centers: Any, points: Any) -> Any:
    centers_array = np.asarray(centers)
    points_array = np.asarray(points)
    tree = cKDTree(np.asarray(centers_array, dtype=np.float64))
    candidate_count = min(16, len(centers_array))
    _, nearest_candidates = tree.query(
        np.asarray(points_array, dtype=np.float64), k=candidate_count, workers=1
    )
    nearest_candidates = np.asarray(nearest_candidates, dtype=np.int64)
    if candidate_count == 1:
        nearest_candidates = nearest_candidates[:, None]
    candidate_distance = np.linalg.norm(
        points_array[:, None, :] - centers_array[nearest_candidates], axis=-1
    )
    minimum = np.min(candidate_distance, axis=1, keepdims=True)
    tied = candidate_distance == minimum
    nearest = np.min(
        np.where(tied, nearest_candidates, len(centers_array)), axis=1
    )
    nearest_distance = np.linalg.norm(
        points_array - centers_array[nearest], axis=1
    )
    radii = np.zeros(len(centers_array), dtype=points_array.dtype)
    np.maximum.at(radii, nearest, nearest_distance)
    radii = np.nextafter(radii, np.asarray(np.inf, dtype=radii.dtype))
    return jax.numpy.asarray(radii)


def _details(centers: np.ndarray, points: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(np.asarray(centers, dtype=np.float64))
    count = min(16, len(centers))
    _, candidates = tree.query(np.asarray(points, dtype=np.float64), k=count, workers=workers)
    candidates = np.asarray(candidates, dtype=np.int64)
    if count == 1:
        candidates = candidates[:, None]
    distance = np.linalg.norm(points[:, None, :] - centers[candidates], axis=-1)
    minimum = np.min(distance, axis=1, keepdims=True)
    nearest = np.min(np.where(distance == minimum, candidates, len(centers)), axis=1)
    nearest_distance = np.linalg.norm(points - centers[nearest], axis=1)
    radii = np.zeros(len(centers), dtype=points.dtype)
    np.maximum.at(radii, nearest, nearest_distance)
    return nearest, np.nextafter(radii, np.asarray(np.inf, dtype=radii.dtype))


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"median_seconds": float(np.median(a)), "mean_seconds": float(np.mean(a)), "p95_seconds": float(np.quantile(a, 0.95))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    binding = json.loads(args.binding.read_text())
    if binding.get("status") != "frozen_after_three_seed_r0_pass":
        raise RuntimeError("high-N binding status drifted")
    runtime = highn._checkpoint_runtime(args)
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    support_rows = {str(n): {row["sample_id"]: row for row in preflight["supports"][str(n)]} for n in (8192, 32768)}
    with h5py.File(args.full_fields, "r") as archive:
        full_coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    original = RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius
    rows: list[dict[str, Any]] = []
    for resolution, route, factor in ((8192, "B8192", 8.0), (32768, "E32768", 128.0)):
        for number, anchor in enumerate(anchors, start=1):
            support = highn._load_support(Path(support_rows[str(resolution)][anchor.sample_id]["support_file"]))
            example = highn._query_example(anchor, support, full_coords)
            coords = highn.runner._graph_coords_for_example(example, runtime["stats"])
            config = dict(runtime["graph_config"])
            config["subsample_factor"] = factor
            config["discrete_graph_backend"] = "sparse_kdtree_v1"
            ref_builder = Heat3DGraphBuilder(**config)
            RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius = _reference_coverage
            try:
                ref_metadata = ref_builder.build_metadata(coords, key=graph_key)
                jax.block_until_ready(ref_metadata.r_rnodes)
            finally:
                RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius = original
            candidate_builder = Heat3DGraphBuilder(**config)
            candidate_metadata = candidate_builder.build_metadata(coords, key=graph_key)
            jax.block_until_ready(candidate_metadata.r_rnodes)
            centers = np.asarray(candidate_metadata.x_rnodes)[0, :-1]
            points = np.concatenate([
                np.asarray(candidate_metadata.x_pnodes_inp)[0, :-1],
                np.asarray(candidate_metadata.x_pnodes_out)[0, :-1],
            ], axis=0)
            ref_nearest, ref_radii = _details(centers, points, 1)
            cand_nearest, cand_radii = _details(centers, points, -1)
            row = {
                "route": route, "resolution": resolution, "sample_id": anchor.sample_id,
                "nearest_assignment_equal": bool(np.array_equal(ref_nearest, cand_nearest)),
                "radii_array_equal": bool(np.array_equal(ref_radii, cand_radii)),
                "radii_sha256_equal": array_sha256(ref_radii) == array_sha256(cand_radii),
                "final_graph_hash_equal": graph_hash(ref_metadata) == graph_hash(candidate_metadata),
                "reference_graph_hash": graph_hash(ref_metadata),
                "candidate_graph_hash": graph_hash(candidate_metadata),
                "reference_regional_prepare_seconds": float(ref_builder.builder.last_build_timings["regional_prepare_seconds"]),
                "candidate_regional_prepare_seconds": float(candidate_builder.builder.last_build_timings["regional_prepare_seconds"]),
                "reference_coverage_seconds": float(ref_builder.builder.last_build_timings["coverage_radius_seconds"]),
                "candidate_coverage_seconds": float(candidate_builder.builder.last_build_timings["coverage_radius_seconds"]),
            }
            rows.append(row)
            print(f"[P5-A3] {route} {number}/32 graph_exact={row['final_graph_hash_equal']}", flush=True)
    gates = ("nearest_assignment_equal", "radii_array_equal", "radii_sha256_equal", "final_graph_hash_equal")
    hard_gate = bool(all(row[key] for row in rows for key in gates))
    summary = {}
    for route in ("B8192", "E32768", "pooled"):
        selected = rows if route == "pooled" else [row for row in rows if row["route"] == route]
        ref = _stats([row["reference_coverage_seconds"] for row in selected])
        cand = _stats([row["candidate_coverage_seconds"] for row in selected])
        summary[route] = {"reference": ref, "candidate": cand, "median_speedup": ref["median_seconds"] / max(cand["median_seconds"], 1e-30), "regional_prepare": _stats([row["candidate_regional_prepare_seconds"] for row in selected])}
    promoted = bool(hard_gate and summary["pooled"]["median_speedup"] > 1.0)
    payload = {"schema_version": "heat3d_v6_p1i_p5_a3_result_v1", "status": "passed" if hard_gate else "failed", "phase": protocol["phase"], "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "hard_gate_passed": hard_gate, "timer_boundary_corrected": True, "candidate_promoted": promoted, "decision": "GO_parallel_coverage" if promoted else "NO_GO_single_worker_coverage", "summary": summary, "samples": rows, "role_contract": protocol["role_contract"]}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not hard_gate:
        raise RuntimeError("P5-A3 exact-equivalence gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
