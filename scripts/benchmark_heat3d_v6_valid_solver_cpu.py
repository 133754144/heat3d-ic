#!/usr/bin/env python3
"""Benchmark frozen P1h FVM cold and warm solves on all valid_iid samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import numpy as np

import heat3d_v6_p1d_core as core


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def _physics(meta: dict[str, Any]) -> dict[str, Any]:
    top = meta["boundary_conditions"]["top"]
    bottom = meta["boundary_conditions"]["bottom"]
    return core.build_physics(
        top_h=float(top["h_W_m2K"]),
        bottom_h=float(bottom["h_W_m2K"]),
        mesh_intervals=tuple(meta["solver_mesh"]["intervals_xyz"]),
        layers=meta["layers_bottom_to_top"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [row for row in manifest["samples"] if row["split_role"] == "valid"]
    if len(rows) != 128:
        raise RuntimeError("valid_iid population drifted")
    archive_path = args.dataset / "full_fields.h5"
    with h5py.File(archive_path, "r") as handle:
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        archive_index = {sample_id: index for index, sample_id in enumerate(ids)}
        q_rows = {
            str(row["sample_id"]): np.asarray(
                handle["samples/q_W_m3"][archive_index[str(row["sample_id"])]],
                dtype=np.float64,
            )
            for row in rows
        }
        frozen_temperature = {
            str(row["sample_id"]): np.asarray(
                handle["samples/temperature_K"][
                    archive_index[str(row["sample_id"])]
                ],
                dtype=np.float64,
            )
            for row in rows
        }
    meta_rows = {
        str(row["sample_id"]): json.loads(
            (
                args.dataset / row["sample_dir"] / "sample_meta.json"
            ).read_text(encoding="utf-8")
        )
        for row in rows
    }
    cold_mesh, cold_assembly, cold_solve, cold_total = [], [], [], []
    cold_replay_max = 0.0
    cold_started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        physics = _physics(meta_rows[sample_id])
        started = time.perf_counter()
        mesh = core.build_mesh(physics)
        mesh_seconds = time.perf_counter() - started
        started = time.perf_counter()
        solver = core.DualRobinSolver(mesh, physics)
        assembly_seconds = time.perf_counter() - started
        started = time.perf_counter()
        temperature, _ = solver.solve(q_rows[sample_id])
        solve_seconds = time.perf_counter() - started
        replay = float(
            np.max(np.abs(temperature - frozen_temperature[sample_id]))
        )
        cold_replay_max = max(cold_replay_max, replay)
        cold_mesh.append(mesh_seconds)
        cold_assembly.append(assembly_seconds)
        cold_solve.append(solve_seconds)
        cold_total.append(mesh_seconds + assembly_seconds + solve_seconds)
        if index % 16 == 0:
            print(
                f"[solver-cold] {index}/128 elapsed={time.perf_counter() - cold_started:.2f}s",
                flush=True,
            )
    warm_solvers: dict[tuple[float, float], core.DualRobinSolver] = {}
    warm_setup_seconds: dict[str, float] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        meta = meta_rows[sample_id]
        key = (
            float(meta["boundary_conditions"]["top"]["h_W_m2K"]),
            float(meta["boundary_conditions"]["bottom"]["h_W_m2K"]),
        )
        if key in warm_solvers:
            continue
        started = time.perf_counter()
        physics = _physics(meta)
        mesh = core.build_mesh(physics)
        warm_solvers[key] = core.DualRobinSolver(mesh, physics)
        warm_setup_seconds[f"top{key[0]:g}_bottom{key[1]:g}"] = float(
            time.perf_counter() - started
        )
    warm_solve = []
    warm_replay_max = 0.0
    warm_started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        meta = meta_rows[sample_id]
        key = (
            float(meta["boundary_conditions"]["top"]["h_W_m2K"]),
            float(meta["boundary_conditions"]["bottom"]["h_W_m2K"]),
        )
        started = time.perf_counter()
        temperature, _ = warm_solvers[key].solve(q_rows[sample_id])
        warm_solve.append(time.perf_counter() - started)
        replay = float(
            np.max(np.abs(temperature - frozen_temperature[sample_id]))
        )
        warm_replay_max = max(warm_replay_max, replay)
        if index % 16 == 0:
            print(
                f"[solver-warm] {index}/128 elapsed={time.perf_counter() - warm_started:.2f}s",
                flush=True,
            )
    if max(cold_replay_max, warm_replay_max) > 1.0e-8:
        raise RuntimeError("valid_iid solver replay drifted from frozen archive")
    payload = {
        "schema_version": "heat3d_v6_valid_solver_cpu_benchmark_v1",
        "status": "passed",
        "evaluation_role": "valid_iid",
        "sample_count": 128,
        "test_hard_accessed": False,
        "training_executed": False,
        "platform": "local_CPU",
        "solver_node_count": 240825,
        "dof_matched_to_model": False,
        "nonmatched_dof_statement": (
            "The only P1h replay- and accuracy-qualified FVM uses 240825 nodes; "
            "model resolution comparisons are explicitly nonmatched-DOF."
        ),
        "cold_per_sample": {
            "mesh_seconds": _distribution(cold_mesh),
            "assembly_seconds": _distribution(cold_assembly),
            "solve_seconds": _distribution(cold_solve),
            "mesh_assembly_solve_seconds": _distribution(cold_total),
        },
        "warm_reused_by_boundary_pair": {
            "boundary_pair_count": len(warm_solvers),
            "setup_seconds_by_pair": warm_setup_seconds,
            "solve_seconds": _distribution(warm_solve),
        },
        "replay": {
            "cold_temperature_max_abs_error_K": cold_replay_max,
            "warm_temperature_max_abs_error_K": warm_replay_max,
        },
        "resources": {
            "peak_ram_bytes": _rss_bytes(),
            "gpu_memory": "N/A_CPU_only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "cold_mean_seconds": payload["cold_per_sample"][
                    "mesh_assembly_solve_seconds"
                ]["mean"],
                "warm_mean_seconds": payload["warm_reused_by_boundary_pair"][
                    "solve_seconds"
                ]["mean"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
