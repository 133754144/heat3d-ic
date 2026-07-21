#!/usr/bin/env python3
"""Generate a frozen V6-P1d asymmetric dual-Robin dataset without ML work."""

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
import yaml
from scipy.interpolate import RegularGridInterpolator

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import heat3d_v6_p1d_core as core


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "heat3d_v6_p1d_asymmetric_dual_robin_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_pilot16.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1d_asymmetric_dual_robin16_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_manifest.json"
DEFAULT_SAMPLES = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_samples.csv"
DEFAULT_SOURCES = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_sources.csv"
DEFAULT_LAYERS = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_layer_drops.csv"
DEFAULT_INTERFACES = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_interface_drops.csv"
DEFAULT_MESH_JSON = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_mesh_convergence.json"
DEFAULT_MESH_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_mesh_convergence.csv"
STRATUM_CODES = {"volume": 0, "source": 1, "interface": 2, "top": 3, "bottom": 4}


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise core.P1dError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "heat3d_v6_p1d_dual_robin_dataset_v1":
        raise core.P1dError("unexpected P1d dataset schema")
    count = int(payload["sample_count"])
    if count != len(payload.get("cases", [])) or count not in {16, 64, 1024}:
        raise core.P1dError("P1d dataset count must be 16, 64, or 1024")
    if len({case["id"] for case in payload["cases"]}) != count:
        raise core.P1dError("duplicate P1d case ID")
    scope = payload["scope"]
    for key in ("model_training", "model_inference", "peak_deltaT_filtering", "peak_deltaT_resampling", "sample_replacement", "per_sample_Rth_power_back_calculation", "material_parameter_tuning"):
        if scope[key] is not False:
            raise core.P1dError(f"forbidden P1d scope: {key}")
    if count == 16:
        for key in ("literature_matrix", "exploration_config", "exploration_json", "exploration_attempts_csv"):
            artifact = REPO_ROOT / payload["provenance"][key]
            if core.sha256(artifact) != payload["provenance"][f"{key}_sha256"]:
                raise core.P1dError(f"provenance SHA mismatch: {key}")
    for case in payload["cases"]:
        core.build_physics(top_h=float(case["top_h_W_m2K"]), bottom_h=float(case["bottom_h_W_m2K"]))
        if float(case["total_source_area_mm2"]) not in {16.0, 32.0, 48.0, 64.0}:
            raise core.P1dError("unregistered source area")
    return payload


def _write_sample(target: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> dict[str, str]:
    target.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name, array in arrays.items():
        path = target / name
        np.save(path, np.asarray(array), allow_pickle=False)
        hashes[name] = core.sha256(path)
    meta_path = target / "sample_meta.json"
    _json_dump(meta_path, meta)
    hashes["sample_meta.json"] = core.sha256(meta_path)
    return hashes


def _sample_row(case: Mapping[str, Any], metrics: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": case["id"], "family_id": case["family_id"],
        "selection_bin": case["selection_bin"],
        "top_h_W_m2K": float(case["top_h_W_m2K"]),
        "bottom_h_W_m2K": float(case["bottom_h_W_m2K"]),
        "package_total_power_W": float(case["package_total_power_W"]),
        "total_source_area_mm2": float(case["total_source_area_mm2"]),
        "layout_seed": int(case["layout_seed"]),
        **metrics,
        "minimum_source_control_volume_count": meta["solver_mesh"]["minimum_source_control_volume_count"],
        "minimum_source_in_plane_interval_count": meta["solver_mesh"]["minimum_source_in_plane_interval_count"],
        "all_layers_covered_by_1024_points": meta["operator_projection"]["coverage"]["all_layers_covered"],
        "all_interfaces_covered_by_1024_points": meta["operator_projection"]["coverage"]["all_interfaces_covered"],
    }


def _mesh_convergence(
    config: Mapping[str, Any], cases_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = config.get("mesh_convergence_contract")
    if not contract:
        return {"status": "not_requested_for_expansion"}, []
    base_layers = core.load_p1c_stack()
    rows: list[dict[str, Any]] = []
    by_sample: dict[str, Any] = {}
    for sample_id in contract["representative_sample_ids"]:
        case = cases_by_id[sample_id]
        sample_rows: list[dict[str, Any]] = []
        for mesh_name, mesh_spec in contract["meshes"].items():
            layers = [dict(layer) for layer in base_layers]
            z_intervals = list(map(int, mesh_spec["layer_z_intervals_bottom_to_top"]))
            if len(z_intervals) != len(layers):
                raise core.P1dError("mesh convergence layer interval count")
            for layer, intervals in zip(layers, z_intervals, strict=True):
                layer["z_intervals"] = intervals
            mesh_intervals = [*map(int, mesh_spec["intervals_xy"]), sum(z_intervals)]
            physics = core.build_physics(
                top_h=float(case["top_h_W_m2K"]), bottom_h=float(case["bottom_h_W_m2K"]),
                mesh_intervals=mesh_intervals, layers=layers,
            )
            physics["source_min_control_volume_count"] = 64
            physics["source_min_in_plane_intervals"] = 5
            mesh = core.build_mesh(physics)
            solver = core.DualRobinSolver(mesh, physics)
            q, _, _ = core.build_sources(
                sample_id=f"{sample_id}_{mesh_name}",
                total_power_W=float(case["package_total_power_W"]),
                total_area_m2=float(case["total_source_area_mm2"]) * 1e-6,
                layout_seed=int(case["layout_seed"]), physics=physics, mesh=mesh,
            )
            temperature, solver_audit = solver.solve(q)
            metrics = core.field_metrics(
                temperature=temperature, q=q,
                total_power_W=float(case["package_total_power_W"]),
                mesh=mesh, solver_audit=solver_audit,
            )
            row = {
                "sample_id": sample_id, "mesh": mesh_name,
                "intervals_x": mesh_intervals[0], "intervals_y": mesh_intervals[1],
                "intervals_z": mesh_intervals[2], "node_count": int(mesh["coords"].shape[0]),
                **{key: metrics[key] for key in (
                    "peak_deltaT_K", "mean_deltaT_K", "top_heat_fraction",
                    "bottom_heat_fraction", "top_branch_R_ambient_K_W",
                    "bottom_branch_R_ambient_K_W", "energy_balance_relative_error",
                )},
            }
            rows.append(row)
            sample_rows.append(row)
        index = {row["mesh"]: row for row in sample_rows}
        comparisons = {}
        for key, tolerance_key in (
            ("peak_deltaT_K", "peak_deltaT"),
            ("mean_deltaT_K", "mean_deltaT"),
            ("top_heat_fraction", "top_heat_fraction"),
            ("top_branch_R_ambient_K_W", "top_branch_R"),
        ):
            base = float(index["base"][key])
            fine = float(index["fine"][key])
            relative = abs(base - fine) / max(abs(fine), 1e-15)
            tolerance = float(contract["base_to_fine_relative_tolerance"][tolerance_key])
            comparisons[key] = {"base_to_fine_relative_difference": relative, "tolerance": tolerance, "passed": relative <= tolerance}
        by_sample[sample_id] = {"meshes": sample_rows, "comparisons": comparisons, "passed": all(item["passed"] for item in comparisons.values())}
    payload = {
        "schema_version": "heat3d_v6_p1d_mesh_convergence_v1",
        "representative_sample_ids": list(contract["representative_sample_ids"]),
        "samples": by_sample,
        "passed": all(value["passed"] for value in by_sample.values()),
    }
    return payload, rows


def generate(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    cases = config["cases"]
    if args.dry_run:
        return {
            "schema_version": SCHEMA, "mode": "dry_run",
            "dataset_id": config["dataset_id"], "sample_count": len(cases),
            "solver_calls": len(cases), "dataset_writes": len(cases),
            "training_runs": 0, "model_inference_runs": 0,
            "peak_deltaT_filtering": False, "sample_replacement": False,
        }
    if args.dataset.exists():
        raise core.P1dError(f"refusing to overwrite dataset: {args.dataset}")

    mesh_intervals = config["physics"]["solver_mesh_intervals_xyz"]
    first = cases[0]
    base_physics = core.build_physics(
        top_h=float(first["top_h_W_m2K"]), bottom_h=float(first["bottom_h_W_m2K"]),
        mesh_intervals=mesh_intervals,
    )
    mesh = core.build_mesh(base_physics)
    solver_cache: dict[tuple[float, float], tuple[dict[str, Any], core.DualRobinSolver]] = {}
    for case in cases:
        key = (float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]))
        if key not in solver_cache:
            physics = core.build_physics(top_h=key[0], bottom_h=key[1], mesh_intervals=mesh_intervals)
            solver_cache[key] = (physics, core.DualRobinSolver(mesh, physics))

    sample_rows: list[dict[str, Any]] = []
    source_csv_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{args.dataset.name}.", dir=args.dataset.parent))
    try:
        for case_index, case in enumerate(cases):
            sample_id = str(case["id"])
            key = (float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]))
            physics, solver = solver_cache[key]
            power = float(case["package_total_power_W"])
            area_m2 = float(case["total_source_area_mm2"]) * 1e-6
            q, sources, layer_powers = core.build_sources(
                sample_id=sample_id, total_power_W=power, total_area_m2=area_m2,
                layout_seed=int(case["layout_seed"]), physics=physics, mesh=mesh,
            )
            points, point_strata, point_seed = p1a._sample_points_before_labels(
                base_seed=int(config["seed"]), sample_id=sample_id,
                physics=physics, mesh=mesh, sources=sources,
            )
            point_sha = p1a._array_sha256(points)
            temperature, solver_audit = solver.solve(q)
            metrics = core.field_metrics(
                temperature=temperature, q=q, total_power_W=power,
                mesh=mesh, solver_audit=solver_audit,
            )
            layers, interfaces = core.layer_interface_diagnostics(
                sample_id=sample_id, temperature=temperature, total_power_W=power,
                mesh=mesh, physics=physics,
            )
            for row in layers:
                row.update({"family_id": case["family_id"], "selection_bin": case["selection_bin"]})
            for row in interfaces:
                row.update({"family_id": case["family_id"], "selection_bin": case["selection_bin"]})
            layer_rows.extend(layers)
            interface_rows.extend(interfaces)
            interpolator = RegularGridInterpolator(
                (mesh["x"], mesh["y"], mesh["z"]),
                temperature.reshape(mesh["info"]["shape"]), method="linear", bounds_error=True,
            )
            point_temperature = np.asarray(interpolator(points), dtype=np.float64)
            point_layers, point_k, point_q = core.point_inputs(points, physics, mesh, sources)
            point_masks = p1a._bc_features(points, physics, mesh)
            bc_parameters = np.tile(np.asarray([key[0], key[1], core.AMBIENT_K, core.AMBIENT_K]), (1024, 1))
            stratum_codes = np.asarray([STRATUM_CODES[name] for name in point_strata], dtype=np.int8)
            coverage = core.point_coverage(points, point_strata, point_layers, mesh, physics)
            peak_gap = float(metrics["peak_deltaT_K"] - np.max(point_temperature - core.AMBIENT_K))
            mean_gap = float(metrics["mean_deltaT_K"] - np.mean(point_temperature - core.AMBIENT_K))
            metrics["solver_peak_minus_projected_peak_K"] = peak_gap
            metrics["solver_cv_mean_minus_projected_unweighted_mean_K"] = mean_gap
            source_payload = []
            for source in sources:
                enriched = {
                    **source, "active_layer_power_W": layer_powers[source["active_layer"]],
                    "package_total_power_W": power, "total_source_area_mm2": float(case["total_source_area_mm2"]),
                }
                source_payload.append(enriched)
                source_csv_rows.append({
                    key_name: value for key_name, value in enriched.items()
                    if key_name not in {"bbox_m", "active_layer_index"}
                })
            meta = {
                "schema_version": SCHEMA, "sample_id": sample_id,
                "dataset_id": config["dataset_id"], "family_id": case["family_id"],
                "selection_bin": case["selection_bin"],
                "exploration_attempt_id": case.get("exploration_attempt_id"),
                "stack_template_id": "logic_package_complete_B_path",
                "topology": "dual_layer_multi_source",
                "package_power_provenance": "frozen_config_not_Rth_inferred",
                "power_was_Rth_inferred": False, "sample_was_temperature_filtered": False,
                "material_parameters_tuned_after_solve": False,
                "total_source_area_mm2": float(case["total_source_area_mm2"]),
                "layout_seed": int(case["layout_seed"]),
                "active_layer_power_W": layer_powers, "sources": source_payload,
                "layers_bottom_to_top": physics["layers_bottom_to_top"],
                "boundary_conditions": physics["boundary_conditions"], "contact": physics["contact"],
                "branch_resistance_contract": config["branch_resistance_contract"],
                "solver_mesh": {
                    "type": "layer_aligned_node_control_volume",
                    "shape": list(mesh["info"]["shape"]),
                    "intervals_xyz": list(map(int, mesh_intervals)),
                    "node_count": int(mesh["coords"].shape[0]),
                    "minimum_source_control_volume_count": min(int(source["covered_control_volume_count"]) for source in sources),
                    "minimum_source_in_plane_interval_count": min(min(int(source["resolved_x_interval_count"]), int(source["resolved_y_interval_count"])) for source in sources),
                    "axis_sha256": {axis: p1a._array_sha256(mesh[axis]) for axis in ("x", "y", "z")},
                },
                "operator_projection": {
                    "point_count": 1024, "point_seed": point_seed,
                    "point_schema": "v6_p1d_irregular_points_v1",
                    "point_coordinates_sha256": point_sha,
                    "point_coordinates_frozen_before_temperature_solve": True,
                    "strata_counts": dict(sorted(Counter(point_strata).items())),
                    "stratum_codes": STRATUM_CODES,
                    "coverage": coverage,
                    "solver_peak_minus_projected_peak_K": peak_gap,
                    "solver_cv_mean_minus_projected_unweighted_mean_K": mean_gap,
                    "label_inputs_used_for_point_selection": [],
                },
                "metrics": metrics,
                "guardrails": {
                    "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
                    "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
                    "material_parameter_tuning": False, "training_runs": 0, "model_inference_runs": 0,
                },
            }
            arrays = {
                "coords.npy": points.astype(np.float64),
                "temperature.npy": point_temperature[:, None].astype(np.float64),
                "deltaT.npy": (point_temperature - core.AMBIENT_K)[:, None].astype(np.float64),
                "k_field.npy": point_k.astype(np.float64),
                "q_field.npy": point_q[:, None].astype(np.float64),
                "layer_id.npy": point_layers[:, None].astype(np.int32),
                "bc_features.npy": point_masks.astype(np.float64),
                "bc_parameters.npy": bc_parameters.astype(np.float64),
                "sampling_stratum.npy": stratum_codes[:, None],
            }
            hashes = _write_sample(temp_root / sample_id, arrays, meta)
            manifest_rows.append({
                "sample_id": sample_id, "family_id": case["family_id"],
                "selection_bin": case["selection_bin"], "sample_dir": sample_id,
                "file_sha256": hashes, "point_coordinates_sha256": point_sha,
            })
            sample_rows.append(_sample_row(case, metrics, meta))
            if (case_index + 1) % max(1, min(32, len(cases) // 8 or 1)) == 0:
                print(f"generated {case_index + 1}/{len(cases)}", flush=True)
        manifest = {
            "schema_version": SCHEMA, "dataset_id": config["dataset_id"],
            "config": _repo_path(args.config), "config_sha256": core.sha256(args.config),
            "sample_count": len(cases), "samples": manifest_rows,
            "guardrails": {"temperature_filtered_samples": 0, "sample_replacements": 0, "training_runs": 0, "model_inference_runs": 0},
        }
        _json_dump(temp_root / "manifest.json", manifest)
        temp_root.rename(args.dataset)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    mesh_payload, mesh_rows = _mesh_convergence(config, {str(case["id"]): case for case in cases})
    if mesh_rows:
        _json_dump(args.mesh_json, mesh_payload)
        _csv_write(args.mesh_csv, mesh_rows)
        if not mesh_payload["passed"]:
            raise core.P1dError("mesh convergence failed")
    _json_dump(args.manifest_json, manifest)
    _csv_write(args.samples_csv, sample_rows)
    _csv_write(args.sources_csv, source_csv_rows)
    _csv_write(args.layers_csv, layer_rows)
    _csv_write(args.interfaces_csv, interface_rows)

    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in sample_rows])
    bins = {"30_42p5": 0, "42p5_55": 0, "55_67p5": 0, "67p5_80": 0, "outside": 0}
    for peak in peaks:
        if 30 <= peak < 42.5: bins["30_42p5"] += 1
        elif 42.5 <= peak < 55: bins["42p5_55"] += 1
        elif 55 <= peak < 67.5: bins["55_67p5"] += 1
        elif 67.5 <= peak <= 80: bins["67p5_80"] += 1
        else: bins["outside"] += 1
    audit = {
        "schema_version": SCHEMA, "dataset_id": config["dataset_id"],
        "dataset_path": _repo_path(args.dataset),
        "dataset_manifest_sha256": core.sha256(args.dataset / "manifest.json"),
        "sample_count": len(sample_rows), "source_count": len(source_csv_rows),
        "window_hit_count": int(np.sum((peaks >= 30) & (peaks <= 80))),
        "temperature_bin_counts": bins,
        "family_counts": dict(sorted(Counter(row["family_id"] for row in sample_rows).items())),
        "selection_bin_counts": dict(sorted(Counter(row["selection_bin"] for row in sample_rows).items())),
        "source_area_counts_mm2": dict(sorted(Counter(str(int(float(row["total_source_area_mm2"]))) for row in sample_rows).items())),
        "summary": {
            key: {"min": float(np.min([float(row[key]) for row in sample_rows])), "median": float(np.median([float(row[key]) for row in sample_rows])), "max": float(np.max([float(row[key]) for row in sample_rows]))}
            for key in ("package_total_power_W", "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction", "top_branch_R_ambient_K_W", "bottom_branch_R_ambient_K_W", "energy_balance_relative_error", "solver_peak_minus_projected_peak_K")
        },
        "mesh_convergence": mesh_payload,
        "integrity": {
            "all_layers_covered_by_every_sample": all(bool(row["all_layers_covered_by_1024_points"]) for row in sample_rows),
            "all_interfaces_covered_by_every_sample": all(bool(row["all_interfaces_covered_by_1024_points"]) for row in sample_rows),
            "max_abs_energy_balance_relative_error": max(abs(float(row["energy_balance_relative_error"])) for row in sample_rows),
            "max_abs_parallel_branch_relative_closure_error": max(abs(float(row["parallel_branch_relative_closure_error"])) for row in sample_rows),
            "minimum_source_control_volume_count": min(int(row["minimum_source_control_volume_count"]) for row in sample_rows),
            "minimum_source_in_plane_interval_count": min(int(row["minimum_source_in_plane_interval_count"]) for row in sample_rows),
            "max_abs_projection_peak_gap_K": max(abs(float(row["solver_peak_minus_projected_peak_K"])) for row in sample_rows),
        },
        "guardrails": {
            "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
            "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
            "material_parameter_tuning": False, "training_runs": 0, "model_inference_runs": 0,
        },
        "samples": sample_rows,
    }
    _json_dump(args.audit_json, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--layers-csv", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--interfaces-csv", type=Path, default=DEFAULT_INTERFACES)
    parser.add_argument("--mesh-json", type=Path, default=DEFAULT_MESH_JSON)
    parser.add_argument("--mesh-csv", type=Path, default=DEFAULT_MESH_CSV)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, key, (REPO_ROOT / value).resolve())
    result = generate(args)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
