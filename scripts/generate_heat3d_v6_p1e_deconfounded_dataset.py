#!/usr/bin/env python3
"""Generate a preregistered V6-P1e paired or 1024 dataset; never train/infer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import yaml

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import heat3d_v6_p1d_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
DEFAULT_CONFIG = CONFIG_DIR / "v6_p1e_deconfounded1024.yaml"
DEFAULT_DATASET = ROOT / "data/heat3d_v6_p1e_deconfounded1024_v0"
STRATUM_CODES = {"volume": 0, "source": 1, "interface": 2, "top": 3, "bottom": 4}


class P1eError(RuntimeError):
    pass


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise P1eError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _physics(top_h: float, bottom_h: float, intervals: Sequence[int]) -> dict[str, Any]:
    if not (500.0 <= top_h <= 2500.0 and 1.0 <= bottom_h <= 200.0):
        raise P1eError(f"unregistered BC: top={top_h}, bottom={bottom_h}")
    return {
        "footprint_m": [0.01, 0.01],
        "layers_bottom_to_top": core.load_p1c_stack(),
        "solver_mesh_intervals_xyz": list(map(int, intervals)),
        "source_min_control_volume_count": 128,
        "source_min_in_plane_intervals": 7,
        "boundary_conditions": {
            "top": {"type": "robin", "h_W_m2K": top_h, "T_inf_K": 300.0},
            "bottom": {"type": "robin", "h_W_m2K": bottom_h, "T_inf_K": 300.0},
            "sides": {"type": "adiabatic"},
        },
        "contact": {"type": "perfect", "R_contact_m2K_W": 0.0},
        "operator_projection": {
            "point_count": 1024,
            "strata": {"volume": 512, "source": 256, "interface": 128, "top": 64, "bottom": 64},
        },
    }


def _load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") not in {
        "heat3d_v6_p1e_deconfounded_dataset_v1",
        "heat3d_v6_p1f_unified_layered_dataset_v1",
    }:
        raise P1eError("unexpected P1e/P1f schema")
    if int(config["sample_count"]) != len(config["cases"]):
        raise P1eError("sample count mismatch")
    if len({case["id"] for case in config["cases"]}) != len(config["cases"]):
        raise P1eError("duplicate sample ID")
    for key in (
        "model_training", "model_inference", "peak_deltaT_filtering",
        "peak_deltaT_resampling", "sample_replacement",
        "per_sample_Rth_power_back_calculation",
    ):
        if config["scope"][key] is not False:
            raise P1eError(f"forbidden scope: {key}")
    post_solve_key = (
        "post_solve_case_or_seed_selection"
        if "post_solve_case_or_seed_selection" in config["scope"]
        else "post_solve_factor_or_seed_selection"
    )
    if config["scope"][post_solve_key] is not False:
        raise P1eError(f"forbidden scope: {post_solve_key}")
    groups = {group["group_id"]: group for group in config["geometry_groups"]}
    if len(groups) != len(config["geometry_groups"]):
        raise P1eError("duplicate geometry group")
    common = set(map(float, config["factor_contract"].get(
        "common_package_power_levels_W_for_every_BC_family",
        config["factor_contract"].get("package_power_W", []),
    )))
    by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in config["cases"]:
        if case["group_id"] not in groups or case["split_role"] != groups[case["group_id"]]["split_role"]:
            raise P1eError("group/split mismatch")
        if float(case["package_total_power_W"]) not in common:
            raise P1eError("BC-specific power detected")
        by_group[case["group_id"]].append(case)
    if any(len({case["split_role"] for case in rows}) != 1 for rows in by_group.values()):
        raise P1eError("split leakage within geometry group")
    return config


def _build_sources(
    sample_id: str, power: float, group: Mapping[str, Any], physics: Mapping[str, Any], mesh: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
    q = np.zeros(coords.shape[0], dtype=np.float64)
    occupied = np.zeros(coords.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    layer_powers: defaultdict[str, float] = defaultdict(float)
    for index, source in enumerate(group["sources"]):
        fx0, fx1, fy0, fy1 = map(float, source["bbox_fraction_xy"])
        x0, x1, y0, y1 = fx0 * 0.01, fx1 * 0.01, fy0 * 0.01, fy1 * 0.01
        layer_name = str(source["layer"])
        layer_index = int(mesh["layer_index"][layer_name])
        lower_face = float(mesh["boundaries"][layer_index])
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= x0) & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
        )
        if layer_name == "silicon_die_lower":
            mask &= ~np.isclose(coords[:, 2], lower_face, atol=1e-15)
        if np.any(mask & occupied):
            raise P1eError(f"{sample_id}: overlapping source masks")
        occupied |= mask
        count = int(np.sum(mask))
        x_count = int(np.unique(coords[mask, 0]).size)
        y_count = int(np.unique(coords[mask, 1]).size)
        z_count = int(np.unique(coords[mask, 2]).size)
        if count < int(physics["source_min_control_volume_count"]):
            raise P1eError(f"{sample_id}: source CV count {count}")
        if min(x_count - 1, y_count - 1) < int(physics["source_min_in_plane_intervals"]):
            raise P1eError(f"{sample_id}: source planform underresolved")
        source_power = power * float(source["package_power_fraction"])
        if source_power > 8.0 + 1e-12:
            raise P1eError(f"{sample_id}: single source power above 8 W")
        volume = float(np.sum(weights[mask]))
        q_density = source_power / volume
        area = (x1 - x0) * (y1 - y0)
        surface_density = source_power / area
        if q_density > 1.5e10 or surface_density > 1.5e6:
            raise P1eError(f"{sample_id}: q/power-density constraint")
        q[mask] = q_density
        layer_powers[layer_name] += source_power
        rows.append({
            "sample_id": sample_id, "source_id": f"src_{index:02d}",
            "active_layer": layer_name, "active_layer_index": layer_index,
            "slot_index": int(source["slot_index"]),
            "bbox_m": {"x": [x0, x1], "y": [y0, y1], "z": [float(mesh["boundaries"][layer_index]), float(mesh["boundaries"][layer_index + 1])]},
            "declared_source_area_m2": area, "package_power_fraction": float(source["package_power_fraction"]),
            "source_power_W": source_power, "surface_power_density_W_m2": surface_density,
            "surface_power_density_W_cm2": surface_density / 1e4,
            "realized_source_control_volume_m3": volume, "q_W_m3": q_density,
            "covered_control_volume_count": count, "covered_x_node_count": x_count,
            "covered_y_node_count": y_count, "covered_z_node_count": z_count,
            "resolved_x_interval_count": x_count - 1, "resolved_y_interval_count": y_count - 1,
        })
    realized = float(np.dot(q, weights))
    if not math.isclose(realized, power, rel_tol=1e-12, abs_tol=1e-12):
        raise P1eError(f"{sample_id}: integrated power mismatch")
    return q, rows, dict(sorted(layer_powers.items()))


def _write_sample(target: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> dict[str, str]:
    target.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name, array in arrays.items():
        path = target / name
        np.save(path, np.asarray(array), allow_pickle=False)
        hashes[name] = core.sha256(path)
    path = target / "sample_meta.json"
    _json(path, meta)
    hashes[path.name] = core.sha256(path)
    return hashes


def generate(config_path: Path, dataset: Path, artifact_stem: str, dry_run: bool) -> dict[str, Any]:
    config = _load(config_path)
    phase = "p1f_unified_layered" if config["schema_version"].startswith("heat3d_v6_p1f") else "p1e_deconfounded"
    cases = config["cases"]
    if dry_run:
        return {
            "status": "dry_run_ok", "sample_count": len(cases), "solver_calls": len(cases),
            "temperature_filtering": False, "sample_replacement": False,
            "training_runs": 0, "model_inference_runs": 0,
        }
    if dataset.exists():
        raise P1eError(f"refusing to overwrite {dataset}")
    intervals = config["physics"]["solver_mesh_intervals_xyz"]
    initial = cases[0]
    base_physics = _physics(float(initial["top_h_W_m2K"]), float(initial["bottom_h_W_m2K"]), intervals)
    mesh = core.build_mesh(base_physics)
    groups = {group["group_id"]: group for group in config["geometry_groups"]}
    solver_cache: dict[tuple[float, float], tuple[dict[str, Any], core.DualRobinSolver]] = {}
    using_bc_ood_domain = False
    group_projection: dict[str, tuple[np.ndarray, list[str], int, str]] = {}
    sample_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=dataset.parent))
    try:
        for index, case in enumerate(cases):
            if case["split_role"] == "bc_ood" and not using_bc_ood_domain:
                # BC-OOD groups are ordered last.  Their 16 held-out matrices
                # replace, rather than accumulate beside, the 16 main matrices.
                solver_cache.clear()
                using_bc_ood_domain = True
            group = groups[str(case["group_id"])]
            sample_id = str(case["id"])
            top_h, bottom_h = float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"])
            key = (top_h, bottom_h)
            if key not in solver_cache:
                physics = _physics(top_h, bottom_h, intervals)
                solver_cache[key] = (physics, core.DualRobinSolver(mesh, physics))
            physics, solver = solver_cache[key]
            power = float(case["package_total_power_W"])
            q, sources, layer_power = _build_sources(sample_id, power, group, physics, mesh)
            if group["group_id"] not in group_projection:
                points, strata, point_seed = p1a._sample_points_before_labels(
                    base_seed=int(config["seed"]), sample_id=str(group["group_id"]),
                    physics=physics, mesh=mesh, sources=sources,
                )
                group_projection[str(group["group_id"])] = (
                    points, strata, point_seed, p1a._array_sha256(points),
                )
            points, strata, point_seed, point_sha = group_projection[str(group["group_id"])]
            temperature, solve_audit = solver.solve(q)
            metrics = core.field_metrics(
                temperature=temperature, q=q, total_power_W=power,
                mesh=mesh, solver_audit=solve_audit,
            )
            interpolator = RegularGridInterpolator(
                (mesh["x"], mesh["y"], mesh["z"]),
                temperature.reshape(mesh["info"]["shape"]), method="linear", bounds_error=True,
            )
            point_temperature = np.asarray(interpolator(points), dtype=np.float64)
            point_layer, point_k, point_q = core.point_inputs(points, physics, mesh, sources)
            point_bc = p1a._bc_features(points, physics, mesh)
            coverage = core.point_coverage(points, strata, point_layer, mesh, physics)
            metrics["solver_peak_minus_projected_peak_K"] = float(metrics["peak_deltaT_K"] - np.max(point_temperature - 300.0))
            metrics["projected_field_cv_rms_deltaT_K"] = float(np.sqrt(np.mean((point_temperature - 300.0) ** 2)))
            meta = {
                "schema_version": f"heat3d_v6_{phase}_sample_v1",
                "sample_id": sample_id, "dataset_id": config["dataset_id"],
                "group_id": group["group_id"], "split_role": case["split_role"],
                "design_block": case["design_block"], "stack_template_id": "logic_package_complete_B_path",
                "layout_kind": group["layout_kind"], "alignment_relation": group["alignment_relation"],
                "source_count": int(group["source_count"]), "active_layer_power_W": layer_power,
                "package_power_provenance": "common_preregistered_grid_not_BC_specific_not_Rth_inferred",
                "power_was_Rth_inferred": False, "sample_was_temperature_filtered": False,
                "sources": sources, "layers_bottom_to_top": physics["layers_bottom_to_top"],
                "boundary_conditions": physics["boundary_conditions"], "contact": physics["contact"],
                "solver_mesh": {
                    "type": "layer_aligned_node_control_volume", "shape": list(mesh["info"]["shape"]),
                    "intervals_xyz": list(map(int, intervals)), "node_count": int(mesh["coords"].shape[0]),
                    "minimum_source_control_volume_count": min(int(row["covered_control_volume_count"]) for row in sources),
                    "minimum_source_in_plane_interval_count": min(min(int(row["resolved_x_interval_count"]), int(row["resolved_y_interval_count"])) for row in sources),
                },
                "operator_projection": {
                    "point_count": 1024, "point_seed": point_seed,
                    "point_seed_key": group["group_id"], "point_coordinates_sha256": point_sha,
                    "coordinates_reused_within_geometry_group": True,
                    "point_coordinates_frozen_before_temperature_solve": True,
                    "label_inputs_used_for_point_selection": [], "strata_counts": dict(sorted(Counter(strata).items())),
                    "coverage": coverage,
                },
                "metrics": metrics,
                "guardrails": {
                    "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
                    "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
                    "post_solve_factor_or_seed_selection": False, "training_runs": 0, "model_inference_runs": 0,
                },
            }
            arrays = {
                "coords.npy": points.astype(np.float64),
                "temperature.npy": point_temperature[:, None].astype(np.float64),
                "deltaT.npy": (point_temperature - 300.0)[:, None].astype(np.float64),
                "k_field.npy": point_k.astype(np.float64), "q_field.npy": point_q[:, None].astype(np.float64),
                "layer_id.npy": point_layer[:, None].astype(np.int32),
                "bc_features.npy": point_bc.astype(np.float64),
                "bc_parameters.npy": np.tile(np.asarray([top_h, bottom_h, 300.0, 300.0]), (1024, 1)),
                "sampling_stratum.npy": np.asarray([STRATUM_CODES[value] for value in strata], dtype=np.int8)[:, None],
            }
            hashes = _write_sample(temporary / sample_id, arrays, meta)
            manifest_rows.append({
                "sample_id": sample_id, "group_id": group["group_id"], "split_role": case["split_role"],
                "sample_dir": sample_id, "point_coordinates_sha256": point_sha, "file_sha256": hashes,
            })
            source_rows.extend({
                **{key_name: value for key_name, value in row.items() if key_name not in {"bbox_m", "active_layer_index"}},
                "group_id": group["group_id"], "split_role": case["split_role"],
            } for row in sources)
            sample_rows.append({
                "sample_id": sample_id, "group_id": group["group_id"], "split_role": case["split_role"],
                "design_block": case["design_block"], "top_h_W_m2K": top_h, "bottom_h_W_m2K": bottom_h,
                "package_total_power_W": power, "source_count": int(group["source_count"]),
                "total_source_area_mm2": float(group["total_source_area_mm2"]),
                "layout_kind": group["layout_kind"], "alignment_relation": group["alignment_relation"],
                **metrics,
                "minimum_source_control_volume_count": meta["solver_mesh"]["minimum_source_control_volume_count"],
                "minimum_source_in_plane_interval_count": meta["solver_mesh"]["minimum_source_in_plane_interval_count"],
                "all_layers_covered_by_1024_points": coverage["all_layers_covered"],
                "all_interfaces_covered_by_1024_points": coverage["all_interfaces_covered"],
            })
            if (index + 1) % 16 == 0:
                print(f"generated {index + 1}/{len(cases)}", flush=True)
        manifest = {
            "schema_version": f"heat3d_v6_{phase}_manifest_v1",
            "dataset_id": config["dataset_id"], "config": str(config_path.relative_to(ROOT)),
            "config_sha256": core.sha256(config_path), "sample_count": len(cases), "samples": manifest_rows,
            "guardrails": {"temperature_filtered_samples": 0, "sample_replacements": 0, "training_runs": 0, "model_inference_runs": 0},
        }
        _json(temporary / "manifest.json", manifest)
        temporary.rename(dataset)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    manifest_path = CONFIG_DIR / f"{artifact_stem}_manifest.json"
    samples_path = CONFIG_DIR / f"{artifact_stem}_samples.csv"
    sources_path = CONFIG_DIR / f"{artifact_stem}_sources.csv"
    split_path = CONFIG_DIR / f"{artifact_stem}_split_map.csv"
    audit_path = CONFIG_DIR / f"{artifact_stem}_audit.json"
    _json(manifest_path, manifest)
    _csv(samples_path, sample_rows)
    _csv(sources_path, source_rows)
    _csv(split_path, [{
        "group_id": group["group_id"], "split_role": group["split_role"],
        "source_count": group["source_count"], "layout_kind": group["layout_kind"],
        "case_count": sum(case["group_id"] == group["group_id"] for case in cases),
    } for group in config["geometry_groups"]])
    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in sample_rows])
    factors = np.asarray([[float(row[key]) for key in ("top_h_W_m2K", "bottom_h_W_m2K", "package_total_power_W")] for row in sample_rows])
    audit = {
        "schema_version": f"heat3d_v6_{phase}_audit_v1", "dataset_id": config["dataset_id"],
        "dataset_path": str(dataset.relative_to(ROOT)), "dataset_manifest_sha256": core.sha256(dataset / "manifest.json"),
        "sample_count": len(sample_rows), "source_count": len(source_rows),
        "window_hit_count": int(np.sum((peaks >= 30.0) & (peaks <= 80.0))),
        "window_hit_fraction": float(np.mean((peaks >= 30.0) & (peaks <= 80.0))),
        "factor_pearson_correlation": np.corrcoef(factors, rowvar=False).tolist(),
        "split_role_counts": dict(sorted(Counter(row["split_role"] for row in sample_rows).items())),
        "source_count_distribution": dict(sorted(Counter(str(row["source_count"]) for row in sample_rows).items())),
        "summary": {key: {
            "min": float(np.min([float(row[key]) for row in sample_rows])),
            "median": float(np.median([float(row[key]) for row in sample_rows])),
            "max": float(np.max([float(row[key]) for row in sample_rows])),
        } for key in ("package_total_power_W", "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction", "energy_balance_relative_error", "solver_peak_minus_projected_peak_K")},
        "integrity": {
            "minimum_source_control_volume_count": min(int(row["minimum_source_control_volume_count"]) for row in sample_rows),
            "minimum_source_in_plane_interval_count": min(int(row["minimum_source_in_plane_interval_count"]) for row in sample_rows),
            "maximum_q_W_m3": max(float(row["q_W_m3"]) for row in source_rows),
            "maximum_single_source_power_W": max(float(row["source_power_W"]) for row in source_rows),
            "maximum_surface_power_density_W_cm2": max(float(row["surface_power_density_W_cm2"]) for row in source_rows),
            "max_abs_energy_balance_relative_error": max(abs(float(row["energy_balance_relative_error"])) for row in sample_rows),
            "all_layers_covered": all(bool(row["all_layers_covered_by_1024_points"]) for row in sample_rows),
            "all_interfaces_covered": all(bool(row["all_interfaces_covered_by_1024_points"]) for row in sample_rows),
        },
        "guardrails": {
            "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
            "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
            "post_solve_factor_or_seed_selection": False, "training_runs": 0, "model_inference_runs": 0,
        },
    }
    _json(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-stem", default="v6_p1e_deconfounded1024")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    result = generate(config.resolve(), dataset.resolve(), args.artifact_stem, args.dry_run)
    print(json.dumps({key: value for key, value in result.items() if key not in {"samples"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
