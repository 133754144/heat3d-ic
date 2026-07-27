#!/usr/bin/env python3
"""Benchmark the frozen 240825-node P1h FVM solver on one valid sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import numpy as np

import heat3d_v6_p1d_core as core


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def _source_resolution_audit(
    *,
    dataset: Path,
    valid_rows: list[dict[str, Any]],
    layers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit lower-DOF P1h meshes using only registered source metadata."""
    results = []
    for lateral_intervals in (48, 52, 56, 60, 64):
        physics = core.build_physics(
            top_h=1000.0,
            bottom_h=20.0,
            mesh_intervals=(lateral_intervals, lateral_intervals, 56),
            layers=layers,
        )
        mesh = core.build_mesh(physics)
        coords = np.asarray(mesh["coords"], dtype=np.float64)
        layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
        source_count = 0
        failed_count = 0
        min_cv_count = 1 << 30
        min_in_plane_intervals = 1 << 30
        for row in valid_rows:
            meta = json.loads(
                (dataset / row["sample_dir"] / "sample_meta.json").read_text(
                    encoding="utf-8"
                )
            )
            for source in meta["sources"]:
                source_count += 1
                layer_index = int(mesh["layer_index"][source["active_layer"]])
                bbox = source["bbox_m"]
                mask = (
                    (layer_ids == layer_index)
                    & (coords[:, 0] >= float(bbox["x"][0]))
                    & (coords[:, 0] <= float(bbox["x"][1]))
                    & (coords[:, 1] >= float(bbox["y"][0]))
                    & (coords[:, 1] <= float(bbox["y"][1]))
                )
                if source["active_layer"] == "silicon_die_lower":
                    mask &= ~np.isclose(
                        coords[:, 2],
                        float(mesh["boundaries"][layer_index]),
                        atol=1.0e-15,
                    )
                cv_count = int(np.sum(mask))
                x_intervals = int(np.unique(coords[mask, 0]).size - 1)
                y_intervals = int(np.unique(coords[mask, 1]).size - 1)
                in_plane = min(x_intervals, y_intervals)
                min_cv_count = min(min_cv_count, cv_count)
                min_in_plane_intervals = min(min_in_plane_intervals, in_plane)
                if cv_count < 128 or in_plane < 7:
                    failed_count += 1
        results.append(
            {
                "mesh_intervals_xyz": [lateral_intervals, lateral_intervals, 56],
                "node_count": int(len(coords)),
                "valid_source_metadata_count": source_count,
                "failed_source_resolution_count": failed_count,
                "minimum_source_control_volume_count": min_cv_count,
                "minimum_source_in_plane_intervals": min_in_plane_intervals,
                "source_resolution_contract_passed": failed_count == 0,
                "target_fields_accessed": False,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solve-repeats", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") not in {None, "cpu"}:
        raise RuntimeError("solver benchmark must not be mixed with a GPU run")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    valid_rows = [row for row in manifest["samples"] if row["split_role"] == "valid"]
    if len(valid_rows) != 128:
        raise RuntimeError("valid_iid population drifted")
    row = valid_rows[0]
    meta_path = args.dataset / row["sample_dir"] / "sample_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["split_role"] != "valid":
        raise RuntimeError("benchmark sample is not valid_iid")
    intervals = tuple(int(value) for value in meta["solver_mesh"]["intervals_xyz"])
    top = meta["boundary_conditions"]["top"]
    bottom = meta["boundary_conditions"]["bottom"]
    physics = core.build_physics(
        top_h=float(top["h_W_m2K"]),
        bottom_h=float(bottom["h_W_m2K"]),
        mesh_intervals=intervals,
        layers=meta["layers_bottom_to_top"],
    )
    total_started = time.perf_counter()
    mesh_started = time.perf_counter()
    mesh = core.build_mesh(physics)
    mesh_seconds = time.perf_counter() - mesh_started
    archive_path = args.dataset / "full_fields.h5"
    archive_row = int(row["full_field_archive_row"])
    with h5py.File(archive_path, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        q = np.asarray(handle["samples/q_W_m3"][archive_row], dtype=np.float64)
        frozen_temperature = np.asarray(
            handle["samples/temperature_K"][archive_row], dtype=np.float64
        )
    if not np.array_equal(coords, np.asarray(mesh["coords"], dtype=np.float64)):
        raise RuntimeError("rebuilt solver mesh does not match frozen full-field archive")
    assembly_started = time.perf_counter()
    solver = core.DualRobinSolver(mesh, physics)
    assembly_seconds = time.perf_counter() - assembly_started
    solve_seconds: list[float] = []
    audits: list[dict[str, float]] = []
    temperatures: list[np.ndarray] = []
    for _ in range(args.solve_repeats):
        started = time.perf_counter()
        temperature, audit = solver.solve(q)
        solve_seconds.append(time.perf_counter() - started)
        temperatures.append(temperature)
        audits.append(audit)
    total_seconds = time.perf_counter() - total_started
    replay_error = np.asarray(temperatures[0]) - frozen_temperature
    if float(np.max(np.abs(replay_error))) > 1.0e-8:
        raise RuntimeError("full solver replay drifted from frozen archive")
    node_count = int(len(coords))
    if node_count != 240825:
        raise RuntimeError("registered P1h full solver node count drifted")
    source_resolution = _source_resolution_audit(
        dataset=args.dataset,
        valid_rows=valid_rows,
        layers=meta["layers_bottom_to_top"],
    )
    smallest_source_legal = next(
        row for row in source_resolution if row["source_resolution_contract_passed"]
    )
    legacy_convergence_path = (
        Path(__file__).resolve().parents[1]
        / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_mesh_convergence.json"
    )
    legacy_convergence = json.loads(
        legacy_convergence_path.read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": "heat3d_v6_full_solver_cpu_benchmark_v1",
        "status": "passed",
        "platform": "local_CPU",
        "sample_id": row["sample_id"],
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "solver": {
            "family": "layer_aligned_control_volume_finite_volume",
            "node_count": node_count,
            "mesh_intervals_xyz": list(intervals),
            "matrix_shape": list(solver.matrix.shape),
            "matrix_nnz": int(solver.matrix.nnz),
            "linear_solver": "scipy.sparse.linalg.cg",
            "relative_tolerance": 1.0e-11,
            "preconditioner": "Jacobi",
        },
        "timing_seconds": {
            "mesh_build": float(mesh_seconds),
            "matrix_assembly": float(assembly_seconds),
            "solve": _summary(solve_seconds),
            "cold_total_mesh_assembly_first_solve": float(
                mesh_seconds + assembly_seconds + solve_seconds[0]
            ),
            "script_total_including_repeats": float(total_seconds),
        },
        "replay": {
            "frozen_archive_sha256": _sha256(archive_path),
            "temperature_max_abs_error_K": float(np.max(np.abs(replay_error))),
            "temperature_rmse_K": float(np.sqrt(np.mean(np.square(replay_error)))),
            "repeat_max_abs_error_K": float(
                max(np.max(np.abs(value - temperatures[0])) for value in temperatures)
            ),
            "linear_residual_max": float(
                max(value["linear_residual"] for value in audits)
            ),
        },
        "resources": {
            "peak_ram_bytes": _rss_bytes(),
            "gpu_memory": "N/A_CPU_only",
        },
        "dof_comparability": {
            "accuracy_equivalent_similar_dof_mesh_available": False,
            "source_resolution_audit_valid_iid_metadata_only": source_resolution,
            "smallest_source_resolution_legal_candidate": smallest_source_legal,
            "legacy_p1d_mesh_convergence_evidence": {
                "path": str(
                    legacy_convergence_path.relative_to(Path(__file__).resolve().parents[1])
                ),
                "sha256": _sha256(legacy_convergence_path),
                "passed": bool(legacy_convergence["passed"]),
                "limitation": (
                    "The legacy coarse 48x48x48 evidence covers two P1d pilot "
                    "geometries, not the P1h randomized source family; the frozen "
                    "P1h layer partition also requires 56 z intervals."
                ),
            },
            "reason": (
                "The frozen and replay-qualified physical solver uses the registered "
                "64x64x56 layer-aligned mesh (240825 nodes). The smallest audited "
                "mesh satisfying every valid source-resolution constraint is still "
                f"{smallest_source_legal['node_count']} nodes, and it has no P1h "
                "mesh-convergence qualification. Therefore model/solver speedups are "
                "explicitly nonmatched-DOF."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "sample_id": row["sample_id"], "node_count": node_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
