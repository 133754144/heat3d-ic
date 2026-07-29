#!/usr/bin/env python3
"""Measure legal structured-FVM mesh sensitivity against 240825-node fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import heat3d_v6_p1d_core as core


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _physics(meta: dict[str, Any], lateral: int) -> dict[str, Any]:
    return core.build_physics(
        top_h=float(meta["boundary_conditions"]["top"]["h_W_m2K"]),
        bottom_h=float(meta["boundary_conditions"]["bottom"]["h_W_m2K"]),
        mesh_intervals=(lateral, lateral, 56),
        layers=meta["layers_bottom_to_top"],
    )


def _q_from_sources(
    mesh: dict[str, Any], meta: dict[str, Any]
) -> tuple[np.ndarray, dict[str, int]]:
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
    q = np.zeros(len(coords), dtype=np.float64)
    occupied = np.zeros(len(coords), dtype=bool)
    minimum_cv = 1 << 30
    minimum_in_plane = 1 << 30
    for source in meta["sources"]:
        layer_name = str(source["active_layer"])
        layer_index = int(mesh["layer_index"][layer_name])
        bbox = source["bbox_m"]
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= float(bbox["x"][0]))
            & (coords[:, 0] <= float(bbox["x"][1]))
            & (coords[:, 1] >= float(bbox["y"][0]))
            & (coords[:, 1] <= float(bbox["y"][1]))
        )
        if layer_name == "silicon_die_lower":
            mask &= ~np.isclose(
                coords[:, 2],
                float(mesh["boundaries"][layer_index]),
                atol=1.0e-15,
            )
        if not np.any(mask) or np.any(mask & occupied):
            raise RuntimeError("illegal coarse source mapping")
        occupied |= mask
        volume = float(np.sum(weights[mask]))
        q[mask] = float(source["source_power_W"]) / volume
        count = int(np.sum(mask))
        x_intervals = int(np.unique(coords[mask, 0]).size - 1)
        y_intervals = int(np.unique(coords[mask, 1]).size - 1)
        minimum_cv = min(minimum_cv, count)
        minimum_in_plane = min(
            minimum_in_plane, x_intervals, y_intervals
        )
    declared = float(sum(source["source_power_W"] for source in meta["sources"]))
    realized = float(np.dot(q, weights))
    if not np.isclose(declared, realized, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("coarse FVM source power is not conserved")
    return q, {
        "minimum_source_control_volume_count": minimum_cv,
        "minimum_source_in_plane_intervals": minimum_in_plane,
    }


def _interpolate_to_reference(
    mesh: dict[str, Any], temperature: np.ndarray, reference_coords: np.ndarray
) -> np.ndarray:
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in mesh["info"]["axes"])
    grid = np.asarray(mesh["info"]["grid"], dtype=np.int64)
    values = np.asarray(temperature, dtype=np.float64)[grid]
    interpolator = RegularGridInterpolator(
        axes, values, method="linear", bounds_error=True
    )
    return np.asarray(interpolator(reference_coords), dtype=np.float64)


def _metrics(
    sse: float,
    energy: float,
    sample_relative: list[float],
    peak_errors: list[float],
    sample_count: int,
    weight_sum: float,
) -> dict[str, float]:
    return {
        "cv_weighted_point_global_relative_rmse_pct": float(
            100.0 * np.sqrt(sse / energy)
        ),
        "sample_first_cv_relative_rmse_pct": float(
            100.0 * np.mean(sample_relative)
        ),
        "raw_cv_weighted_rmse_K": float(
            np.sqrt(sse / (sample_count * weight_sum))
        ),
        "peak_error_rmse_K": float(np.sqrt(np.mean(np.square(peak_errors)))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-benchmark", type=Path, required=True)
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pareto-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
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
        reference_coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        reference_cv = np.asarray(
            handle["mesh/control_volume"], dtype=np.float64
        )
        reference_temperature = {
            str(row["sample_id"]): np.asarray(
                handle["samples/temperature_K"][
                    archive_index[str(row["sample_id"])]
                ],
                dtype=np.float64,
            )
            for row in rows
        }
    metas = {
        str(row["sample_id"]): json.loads(
            (
                args.dataset / row["sample_dir"] / "sample_meta.json"
            ).read_text(encoding="utf-8")
        )
        for row in rows
    }
    results: dict[str, Any] = {}
    # The smallest frozen P1h source boxes require at least seven in-plane
    # intervals.  A label-independent mesh audit over all valid layouts showed
    # that 48/52/56 fail that contract (minimum 5/6/6), while 60 and 62 pass.
    # Keep both comparison meshes strictly below the 64x64 reference mesh.
    for label, lateral in (("coarse", 60), ("medium", 62)):
        cold, warm = [], []
        sse = 0.0
        energy = 0.0
        sample_relative: list[float] = []
        peak_errors: list[float] = []
        minimum_cv = 1 << 30
        minimum_in_plane = 1 << 30
        solver_cache: dict[tuple[float, float], tuple[dict[str, Any], Any]] = {}
        for index, row in enumerate(rows, start=1):
            sample_id = str(row["sample_id"])
            meta = metas[sample_id]
            pair = (
                float(meta["boundary_conditions"]["top"]["h_W_m2K"]),
                float(meta["boundary_conditions"]["bottom"]["h_W_m2K"]),
            )
            cold_started = time.perf_counter()
            cold_physics = _physics(meta, lateral)
            cold_mesh = core.build_mesh(cold_physics)
            cold_solver = core.DualRobinSolver(cold_mesh, cold_physics)
            cold_q, source_audit = _q_from_sources(cold_mesh, meta)
            cold_temperature, _ = cold_solver.solve(cold_q)
            cold.append(time.perf_counter() - cold_started)
            minimum_cv = min(
                minimum_cv,
                source_audit["minimum_source_control_volume_count"],
            )
            minimum_in_plane = min(
                minimum_in_plane,
                source_audit["minimum_source_in_plane_intervals"],
            )
            if pair not in solver_cache:
                physics = _physics(meta, lateral)
                mesh = core.build_mesh(physics)
                solver_cache[pair] = (mesh, core.DualRobinSolver(mesh, physics))
            mesh, solver = solver_cache[pair]
            q, _ = _q_from_sources(mesh, meta)
            warm_started = time.perf_counter()
            temperature, _ = solver.solve(q)
            warm.append(time.perf_counter() - warm_started)
            prediction = _interpolate_to_reference(
                mesh, temperature, reference_coords
            )
            truth = reference_temperature[sample_id]
            reference = float(
                meta["boundary_conditions"]["bottom"]["T_inf_K"]
            )
            truth_delta = truth - reference
            error = prediction - truth
            sample_sse = float(np.dot(reference_cv, error * error))
            sample_energy = float(
                np.dot(reference_cv, truth_delta * truth_delta)
            )
            sse += sample_sse
            energy += sample_energy
            sample_relative.append(np.sqrt(sample_sse / sample_energy))
            peak_errors.append(
                float(np.max(prediction) - np.max(truth))
            )
            if index % 16 == 0:
                print(f"[{label}] {index}/128", flush=True)
        legal = minimum_cv >= 128 and minimum_in_plane >= 7
        if not legal:
            raise RuntimeError(f"{label} FVM violates source resolution contract")
        results[label] = {
            "mesh_intervals_xyz": [lateral, lateral, 56],
            "solver_node_count": int(len(solver_cache[next(iter(solver_cache))][0]["coords"])),
            "source_resolution": {
                "minimum_source_control_volume_count": minimum_cv,
                "minimum_source_in_plane_intervals": minimum_in_plane,
                "passed": legal,
            },
            "cold_mesh_assembly_solve_seconds": _distribution(cold),
            "warm_solve_seconds": _distribution(warm),
            "accuracy_vs_240825_reference": _metrics(
                sse,
                energy,
                sample_relative,
                peak_errors,
                len(rows),
                float(np.sum(reference_cv)),
            ),
        }
    reference = json.loads(args.reference_benchmark.read_text(encoding="utf-8"))
    results["reference"] = {
        "mesh_intervals_xyz": [64, 64, 56],
        "solver_node_count": 240825,
        "source_resolution": {
            "minimum_source_control_volume_count": min(
                int(meta["solver_mesh"]["minimum_source_control_volume_count"])
                for meta in metas.values()
            ),
            "minimum_source_in_plane_intervals": min(
                int(meta["solver_mesh"]["minimum_source_in_plane_interval_count"])
                for meta in metas.values()
            ),
            "passed": True,
        },
        "cold_mesh_assembly_solve_seconds": reference["cold_per_sample"][
            "mesh_assembly_solve_seconds"
        ],
        "warm_solve_seconds": reference["warm_reused_by_boundary_pair"][
            "solve_seconds"
        ],
        "accuracy_vs_240825_reference": {
            "cv_weighted_point_global_relative_rmse_pct": 0.0,
            "sample_first_cv_relative_rmse_pct": 0.0,
            "raw_cv_weighted_rmse_K": 0.0,
            "peak_error_rmse_K": 0.0,
        },
        "timing_source": str(args.reference_benchmark),
    }
    model_rows = json.loads(args.model_metrics.read_text(encoding="utf-8"))
    pareto = []
    for label, row in results.items():
        pareto.append(
            {
                "method": f"FVM_{label}",
                "nodes": row["solver_node_count"],
                "runtime_seconds_per_sample": row[
                    "warm_solve_seconds"
                ]["mean"],
                "raw_cv_weighted_rmse_K": row[
                    "accuracy_vs_240825_reference"
                ]["raw_cv_weighted_rmse_K"],
                "role": "valid_iid",
            }
        )
    for resolution in (4096, 8192, 16384):
        metric = model_rows["mean_std"][str(resolution)]["mean"]
        pareto.append(
            {
                "method": f"V6_anchor_{resolution}",
                "nodes": resolution,
                "runtime_seconds_per_sample": None,
                "raw_cv_weighted_rmse_K": metric["full_raw_cv_rmse_K"],
                "role": "valid_iid",
            }
        )
    args.pareto_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.pareto_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(pareto[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(pareto)
    payload = {
        "schema_version": (
            "heat3d_v6_legal_structured_fvm_mesh_sensitivity_v1"
        ),
        "status": "passed",
        "evaluation_role": "valid_iid",
        "sample_count": 128,
        "reference_node_count": 240825,
        "meshes": results,
        "pareto_rows": pareto,
        "source_aware_points_used_as_fvm_mesh": False,
        "test_hard_accessed": False,
        "training_executed": False,
        "process_peak_ram_bytes": _rss_bytes(),
    }
    # Persist the numerical benchmark before rendering the convenience figure.
    # A headless plotting backend must never discard an otherwise completed
    # 128-sample solver benchmark.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fvm_rows = [row for row in pareto if row["method"].startswith("FVM_")]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        [row["runtime_seconds_per_sample"] for row in fvm_rows],
        [row["raw_cv_weighted_rmse_K"] for row in fvm_rows],
        "o-",
        label="FVM",
    )
    for row in fvm_rows:
        plt.annotate(row["method"].removeprefix("FVM_"), (
            row["runtime_seconds_per_sample"],
            row["raw_cv_weighted_rmse_K"],
        ))
    plt.xlabel("Runtime per sample (s)")
    plt.ylabel("CV-weighted full-field RMSE (K)")
    plt.title("V6 valid_iid legal structured-FVM mesh sensitivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=180)
    plt.close()
    print(json.dumps({"status": "passed", "meshes": list(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
