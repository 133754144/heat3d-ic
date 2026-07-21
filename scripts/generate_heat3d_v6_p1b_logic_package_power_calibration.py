#!/usr/bin/env python3
"""Generate the fixed V6-P1b 16-sample logic-package calibration dataset.

This entry point performs deterministic finite-volume data generation only. It
never trains or runs a learned model, never infers power from temperature, and
never filters or replaces a sample based on the solved temperature field.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "heat3d_v6_p1b_logic_package_power_calibration_v1"
DEFAULT_CASES = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_cases.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1b_logic_package_power_calibration16_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_manifest.json"
DEFAULT_SAMPLES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_samples.csv"
DEFAULT_SOURCES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_sources.csv"


class GenerationError(RuntimeError):
    pass


def _load_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenerationError("P1b registry must be a YAML object")
    if payload.get("schema_version") != "heat3d_v6_p1b_logic_package_power_cases_v1":
        raise GenerationError("unexpected P1b registry schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise GenerationError("P1b requires exactly 16 cases")
    if len({str(case["id"]) for case in cases}) != 16:
        raise GenerationError("P1b case IDs must be unique")
    return payload


def _bbox_geometry(
    bbox_fraction_xy: Sequence[float],
    footprint_m: Sequence[float],
) -> tuple[dict[str, list[float]], float]:
    fx0, fx1, fy0, fy1 = map(float, bbox_fraction_xy)
    lx, ly = map(float, footprint_m)
    if not (0.0 <= fx0 < fx1 <= 1.0 and 0.0 <= fy0 < fy1 <= 1.0):
        raise GenerationError(f"invalid source bbox fraction: {bbox_fraction_xy}")
    bbox = {"x": [fx0 * lx, fx1 * lx], "y": [fy0 * ly, fy1 * ly]}
    area = (bbox["x"][1] - bbox["x"][0]) * (bbox["y"][1] - bbox["y"][0])
    return bbox, float(area)


def _build_area_weighted_sources(
    case: Mapping[str, Any],
    registry: Mapping[str, Any],
    mesh: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    physics = registry["physics"]
    topology_id = str(case["topology"])
    topology = registry["topologies"][topology_id]
    package_power = float(case["package_total_power_W"])
    source_specs = topology["sources"]
    source_geometry = [
        _bbox_geometry(source["bbox_fraction_xy"], physics["footprint_m"])
        for source in source_specs
    ]
    total_area = float(sum(area for _, area in source_geometry))
    expected_area = float(registry["power_contract"]["topology_total_declared_source_area_m2"])
    if not math.isclose(total_area, expected_area, rel_tol=0.0, abs_tol=1.0e-15):
        raise GenerationError(f"{topology_id}: total source area {total_area} != {expected_area}")

    coords = np.asarray(mesh["coords"], dtype=np.float64)
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
    bottom_z = float(mesh["z"][0])
    boundaries = mesh["boundaries"]
    q = np.zeros(coords.shape[0], dtype=np.float64)
    occupied = np.zeros(coords.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    layer_powers: defaultdict[str, float] = defaultdict(float)

    for source_index, (source, (bbox_xy, declared_area)) in enumerate(
        zip(source_specs, source_geometry, strict=True)
    ):
        layer_name = str(source["layer"])
        layer_index = int(mesh["layer_index"][layer_name])
        if physics["layers_bottom_to_top"][layer_index]["role"] != "active":
            raise GenerationError(f"{case['id']}: source assigned to non-active layer {layer_name}")
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= bbox_xy["x"][0])
            & (coords[:, 0] <= bbox_xy["x"][1])
            & (coords[:, 1] >= bbox_xy["y"][0])
            & (coords[:, 1] <= bbox_xy["y"][1])
            & (~np.isclose(coords[:, 2], bottom_z, atol=1.0e-15))
        )
        if np.any(mask & occupied):
            raise GenerationError(f"{case['id']}: overlapping source control volumes")
        occupied |= mask
        count = int(np.sum(mask))
        if count < int(physics["source_min_control_volume_count"]):
            raise GenerationError(f"{case['id']} source {source_index}: only {count} control volumes")
        x_count = int(np.unique(coords[mask, 0]).size)
        y_count = int(np.unique(coords[mask, 1]).size)
        z_count = int(np.unique(coords[mask, 2]).size)
        x_intervals = max(x_count - 1, 0)
        y_intervals = max(y_count - 1, 0)
        min_intervals = int(physics["source_min_in_plane_intervals"])
        if min(x_intervals, y_intervals) < min_intervals:
            raise GenerationError(f"{case['id']} source {source_index}: underresolved in-plane footprint")

        area_fraction = declared_area / total_area
        source_power = package_power * area_fraction
        surface_density = source_power / declared_area
        realized_volume = float(np.sum(weights[mask]))
        q_density = source_power / realized_volume
        q[mask] = q_density
        layer_powers[layer_name] += source_power
        layer_thickness = float(physics["layers_bottom_to_top"][layer_index]["thickness_m"])
        bbox = {
            **bbox_xy,
            "z": [float(boundaries[layer_index]), float(boundaries[layer_index + 1])],
        }
        rows.append(
            {
                "sample_id": str(case["id"]),
                "source_id": f"src_{source_index:02d}",
                "topology": topology_id,
                "power_provenance": "explicit_user_instruction_area_weighted",
                "active_layer": layer_name,
                "active_layer_index": layer_index,
                "bbox_m": bbox,
                "declared_source_area_m2": declared_area,
                "source_area_fraction": area_fraction,
                "source_power_W": source_power,
                "surface_power_density_W_m2": surface_density,
                "geometric_source_volume_m3": declared_area * layer_thickness,
                "realized_source_control_volume_m3": realized_volume,
                "q_W_m3": q_density,
                "covered_control_volume_count": count,
                "covered_x_node_count": x_count,
                "covered_y_node_count": y_count,
                "covered_z_node_count": z_count,
                "resolved_x_interval_count": x_intervals,
                "resolved_y_interval_count": y_intervals,
                "physical_layer_z_interval_count": int(
                    physics["layers_bottom_to_top"][layer_index]["z_intervals"]
                ),
            }
        )

    realized_power = float(np.dot(q, weights))
    if not math.isclose(realized_power, package_power, rel_tol=1.0e-12, abs_tol=1.0e-13):
        raise GenerationError(f"{case['id']}: integrated source power mismatch")
    return q, rows, dict(sorted(layer_powers.items()))


def _point_inputs(
    points: np.ndarray,
    physics: Mapping[str, Any],
    mesh: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer_ids, k, q = p1a._values_at_points(points, physics, mesh, source_rows)
    q[np.isclose(points[:, 2], float(mesh["z"][0]), atol=1.0e-15)] = 0.0
    return layer_ids, k, q


def _relative_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_registry(args.cases)
    physics = registry["physics"]
    mesh = p1a._build_mesh(physics)
    plan = [
        {
            "sample_id": str(case["id"]),
            "topology": str(case["topology"]),
            "package_total_power_W": float(case["package_total_power_W"]),
            "source_count": int(registry["topologies"][case["topology"]]["source_count"]),
        }
        for case in registry["cases"]
    ]
    if args.dry_run:
        return {
            "schema_version": SCHEMA,
            "mode": "dry_run",
            "dataset_id": registry["dataset_id"],
            "sample_count": len(plan),
            "case_plan": plan,
            "solver_calls": 0,
            "dataset_writes": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
            "peak_deltaT_filtering": False,
            "power_back_calculation": False,
        }
    if args.dataset.exists():
        raise GenerationError(f"refusing to overwrite existing dataset: {args.dataset}")

    solver = p1a.LayeredFvmSolver(mesh, physics)
    sample_rows: list[dict[str, Any]] = []
    source_csv_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{args.dataset.name}.", dir=args.dataset.parent))
    try:
        for case in registry["cases"]:
            sample_id = str(case["id"])
            topology_id = str(case["topology"])
            q, source_rows, layer_powers = _build_area_weighted_sources(case, registry, mesh)
            points, point_strata, point_seed = p1a._sample_points_before_labels(
                base_seed=int(registry["seed"]),
                sample_id=sample_id,
                physics=physics,
                mesh=mesh,
                sources=source_rows,
            )
            points_sha = p1a._array_sha256(points)
            temperature, solver_audit = solver.solve(q)
            metrics = p1a._sample_metrics(
                temperature=temperature,
                q=q,
                mesh=mesh,
                solver_audit=solver_audit,
                bottom_t=solver.bottom_t,
            )
            interpolator = RegularGridInterpolator(
                (mesh["x"], mesh["y"], mesh["z"]),
                temperature.reshape(mesh["info"]["shape"]),
                method="linear",
                bounds_error=True,
            )
            point_temperature = np.asarray(interpolator(points), dtype=np.float64)
            point_layers, point_k, point_q = _point_inputs(points, physics, mesh, source_rows)
            point_bc = p1a._bc_features(points, physics, mesh)
            projection_peak_gap = float(
                metrics["peak_deltaT_K"] - np.max(point_temperature - solver.bottom_t)
            )
            arrays = {
                "coords.npy": points.astype(np.float64),
                "temperature.npy": point_temperature[:, None].astype(np.float64),
                "deltaT.npy": (point_temperature - solver.bottom_t)[:, None].astype(np.float64),
                "k_field.npy": point_k.astype(np.float64),
                "q_field.npy": point_q[:, None].astype(np.float64),
                "layer_id.npy": point_layers[:, None].astype(np.int32),
                "bc_features.npy": point_bc.astype(np.float64),
            }
            package_power = float(case["package_total_power_W"])
            source_payload: list[dict[str, Any]] = []
            for source in source_rows:
                row = dict(source)
                row["active_layer_power_W"] = layer_powers[row["active_layer"]]
                row["package_total_power_W"] = package_power
                source_payload.append(row)
                source_csv_rows.append(
                    {
                        key: (
                            value
                            if isinstance(value, (str, int))
                            else f"{float(value):.17g}"
                        )
                        for key, value in row.items()
                        if key not in {"bbox_m", "active_layer_index"}
                    }
                )
            common_surface_density = package_power / float(
                registry["power_contract"]["topology_total_declared_source_area_m2"]
            )
            meta = {
                "schema_version": SCHEMA,
                "sample_id": sample_id,
                "dataset_id": registry["dataset_id"],
                "stack_template_id": "logic_package",
                "stack_realization": physics["stack_realization"],
                "topology": topology_id,
                "package_power_provenance": registry["power_contract"]["provenance"],
                "power_allocation_mode": "proportional_to_declared_source_planform_area",
                "power_was_Rth_inferred": False,
                "direct_package_power_to_local_small_source": False,
                "topology_total_declared_source_area_m2": float(
                    registry["power_contract"]["topology_total_declared_source_area_m2"]
                ),
                "common_surface_power_density_W_m2": common_surface_density,
                "active_layers": sorted(layer_powers),
                "active_layer_power_W": layer_powers,
                "sources": source_payload,
                "boundary_conditions": physics["boundary_conditions"],
                "contact": physics["contact"],
                "solver_mesh": {
                    "type": "layer_aligned_node_control_volume",
                    "shape": list(mesh["info"]["shape"]),
                    "intervals_xyz": list(map(int, physics["solver_mesh_intervals_xyz"])),
                    "node_count": int(mesh["coords"].shape[0]),
                    "axis_sha256": {
                        "x": p1a._array_sha256(mesh["x"]),
                        "y": p1a._array_sha256(mesh["y"]),
                        "z": p1a._array_sha256(mesh["z"]),
                    },
                    "minimum_source_control_volume_count": min(
                        int(source["covered_control_volume_count"]) for source in source_rows
                    ),
                    "minimum_source_in_plane_interval_count": min(
                        min(
                            int(source["resolved_x_interval_count"]),
                            int(source["resolved_y_interval_count"]),
                        )
                        for source in source_rows
                    ),
                    "source_on_bottom_dirichlet_node_count": int(
                        np.sum((q != 0.0) & np.isclose(mesh["coords"][:, 2], mesh["z"][0]))
                    ),
                },
                "operator_projection": {
                    "point_count": 1024,
                    "point_seed": point_seed,
                    "point_schema": "v6_p1b_irregular_points_v1",
                    "point_coordinates_sha256": points_sha,
                    "point_coordinates_frozen_before_temperature_solve": True,
                    "strata_counts": dict(sorted(Counter(point_strata).items())),
                    "interpolation_method": "scipy_regular_grid_linear",
                    "solver_peak_minus_projected_peak_K": projection_peak_gap,
                    "label_inputs_used_for_point_selection": [],
                },
                "metrics": metrics,
                "guardrails": {
                    "peak_deltaT_filtering": False,
                    "peak_deltaT_resampling": False,
                    "power_back_calculation": False,
                    "sample_replacement": False,
                    "training_runs": 0,
                    "model_inference_runs": 0,
                },
            }
            file_hashes = p1a._write_sample(temp_root / sample_id, arrays, meta)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "topology": topology_id,
                    "package_total_power_W": package_power,
                    "sample_dir": sample_id,
                    "file_sha256": file_hashes,
                    "point_coordinates_sha256": points_sha,
                }
            )
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "topology": topology_id,
                    "source_count": len(source_rows),
                    "active_layer_count": len(layer_powers),
                    "active_layer_power_W": json.dumps(layer_powers, sort_keys=True),
                    "topology_total_declared_source_area_m2": float(
                        registry["power_contract"]["topology_total_declared_source_area_m2"]
                    ),
                    "common_surface_power_density_W_m2": common_surface_density,
                    "package_total_power_W": metrics["package_total_power_W"],
                    "peak_deltaT_K": metrics["peak_deltaT_K"],
                    "mean_deltaT_K": metrics["mean_deltaT_K"],
                    "Rth_peak_K_W": metrics["Rth_peak_K_W"],
                    "top_heat_fraction": metrics["top_heat_fraction"],
                    "bottom_heat_fraction": metrics["bottom_heat_fraction"],
                    "energy_balance_relative_error": metrics["energy_balance_relative_error"],
                    "linear_residual": metrics["linear_residual"],
                    "min_source_control_volume_count": min(
                        int(source["covered_control_volume_count"]) for source in source_rows
                    ),
                    "min_source_in_plane_interval_count": min(
                        min(
                            int(source["resolved_x_interval_count"]),
                            int(source["resolved_y_interval_count"]),
                        )
                        for source in source_rows
                    ),
                    "in_30_80_K_window": metrics["in_30_80_K_window"],
                    "solver_peak_minus_projected_peak_K": projection_peak_gap,
                }
            )

        manifest = {
            "schema_version": SCHEMA,
            "dataset_id": registry["dataset_id"],
            "case_registry": _relative_repo_path(args.cases),
            "case_registry_sha256": p1a._sha256(args.cases),
            "sample_count": len(manifest_rows),
            "samples": manifest_rows,
            "guardrails": {
                "peak_deltaT_filtering": False,
                "peak_deltaT_resampling": False,
                "power_back_calculation": False,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
        }
        p1a._json_dump(temp_root / "manifest.json", manifest)
        temp_root.rename(args.dataset)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    p1a._json_dump(args.manifest_json, manifest)
    numeric = lambda key: np.asarray([float(row[key]) for row in sample_rows], dtype=np.float64)
    hits = [str(row["sample_id"]) for row in sample_rows if row["in_30_80_K_window"]]
    by_topology: dict[str, Any] = {}
    for topology_id in registry["topologies"]:
        subset = [row for row in sample_rows if row["topology"] == topology_id]
        by_topology[topology_id] = {
            "sample_count": len(subset),
            "window_hit_sample_ids": [row["sample_id"] for row in subset if row["in_30_80_K_window"]],
            "peak_deltaT_K_by_power_W": {
                f"{float(row['package_total_power_W']):g}": float(row["peak_deltaT_K"])
                for row in subset
            },
            "Rth_peak_K_W": {
                "min": min(float(row["Rth_peak_K_W"]) for row in subset),
                "max": max(float(row["Rth_peak_K_W"]) for row in subset),
            },
        }
    by_power: dict[str, Any] = {}
    for power in registry["power_contract"]["package_total_power_W"]:
        subset = [
            row for row in sample_rows
            if math.isclose(float(row["package_total_power_W"]), float(power), rel_tol=1.0e-12)
        ]
        by_power[f"{float(power):g}"] = {
            "sample_count": len(subset),
            "peak_deltaT_K": {
                "min": min(float(row["peak_deltaT_K"]) for row in subset),
                "median": float(np.median([float(row["peak_deltaT_K"]) for row in subset])),
                "max": max(float(row["peak_deltaT_K"]) for row in subset),
            },
            "window_hit_count": sum(bool(row["in_30_80_K_window"]) for row in subset),
        }
    audit = {
        "schema_version": SCHEMA,
        "dataset_id": registry["dataset_id"],
        "dataset_path": _relative_repo_path(args.dataset),
        "dataset_manifest_sha256": p1a._sha256(args.dataset / "manifest.json"),
        "tracked_manifest_sha256": p1a._sha256(args.manifest_json),
        "sample_count": len(sample_rows),
        "source_count": len(source_csv_rows),
        "topology_case_counts": dict(sorted(Counter(row["topology"] for row in sample_rows).items())),
        "power_definition": {
            "package_total_power_provenance": "explicit_user_instruction",
            "source_power": "package power times declared source-area fraction",
            "active_layer_power": "sum of area-weighted source power in layer",
            "surface_power_density": "source power divided by declared planform area",
            "volumetric_q": "source power divided by realized source control-volume volume",
            "Rth_power_inference_used": False,
        },
        "peak_deltaT_evaluation_window_K": [30.0, 80.0],
        "window_hit_sample_ids": hits,
        "window_hit_count": len(hits),
        "by_topology": by_topology,
        "by_power": by_power,
        "summary": {
            key: {
                "min": float(np.min(numeric(key))),
                "median": float(np.median(numeric(key))),
                "max": float(np.max(numeric(key))),
            }
            for key in (
                "package_total_power_W",
                "common_surface_power_density_W_m2",
                "peak_deltaT_K",
                "mean_deltaT_K",
                "Rth_peak_K_W",
                "top_heat_fraction",
                "bottom_heat_fraction",
                "energy_balance_relative_error",
                "solver_peak_minus_projected_peak_K",
            )
        },
        "integrity": {
            "all_metrics_finite": bool(
                all(
                    np.all(np.isfinite(numeric(key)))
                    for key in (
                        "package_total_power_W",
                        "peak_deltaT_K",
                        "mean_deltaT_K",
                        "Rth_peak_K_W",
                        "top_heat_fraction",
                        "bottom_heat_fraction",
                        "energy_balance_relative_error",
                        "linear_residual",
                    )
                )
            ),
            "max_abs_energy_balance_relative_error": float(
                np.max(np.abs(numeric("energy_balance_relative_error")))
            ),
            "min_source_control_volume_count": min(
                int(row["covered_control_volume_count"]) for row in source_csv_rows
            ),
            "min_source_in_plane_interval_count": min(
                min(int(row["resolved_x_interval_count"]), int(row["resolved_y_interval_count"]))
                for row in source_csv_rows
            ),
            "source_on_bottom_dirichlet_node_count": 0,
            "point_count_per_sample": 1024,
            "point_coordinates_frozen_before_labels": True,
        },
        "guardrails": {
            "generated_samples": 16,
            "expanded_samples": 0,
            "peak_deltaT_filtering": False,
            "peak_deltaT_resampling": False,
            "power_back_calculation": False,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
        "samples": sample_rows,
    }
    p1a._json_dump(args.audit_json, audit)
    p1a._csv_write(
        args.samples_csv,
        [
            {
                key: value if isinstance(value, (str, int, bool)) else f"{float(value):.17g}"
                for key, value in row.items()
            }
            for row in sample_rows
        ],
        list(sample_rows[0]),
    )
    p1a._csv_write(args.sources_csv, source_csv_rows, list(source_csv_rows[0]))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES_CSV)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("cases", "dataset", "audit_json", "manifest_json", "samples_csv", "sources_csv"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (REPO_ROOT / value).resolve())
    print(json.dumps(generate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
