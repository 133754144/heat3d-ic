#!/usr/bin/env python3
"""Validate the frozen V6-P1b logic-package power-calibration dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import generate_heat3d_v6_p1b_logic_package_power_calibration as generator


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_cases.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1b_logic_package_power_calibration16_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_manifest.json"
DEFAULT_SAMPLES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_samples.csv"
DEFAULT_SOURCES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1b_logic_package_power_calibration_sources.csv"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def _bbox_area(source: Mapping[str, Any], footprint: Sequence[float]) -> float:
    _, area = generator._bbox_geometry(source["bbox_fraction_xy"], footprint)
    return area


def _check_registry(registry: Mapping[str, Any]) -> None:
    _expect(registry["sample_count"] == 16 and len(registry["cases"]) == 16, "sample count must be 16")
    scope = registry["scope"]
    _expect(scope["layered_stacks_only"] is True, "P1b must remain layered")
    _expect(scope["stack_template_id"] == "logic_package", "wrong stack template")
    _expect(scope["peak_deltaT_filtering"] is False, "temperature filtering forbidden")
    _expect(scope["peak_deltaT_resampling"] is False, "temperature resampling forbidden")
    _expect(scope["power_back_calculation"] is False, "power back-calculation forbidden")
    _expect(scope["model_training"] is False and scope["model_inference"] is False, "model work forbidden")

    physics = registry["physics"]
    _expect(physics["footprint_m"] == [0.010, 0.010], "footprint must be 10 x 10 mm")
    _expect(physics["stack_realization"] == "repeated_active_die_logic_package", "stack realization mismatch")
    expected_layers = [
        ("silicon_die_lower", 0.00015, "silicon_bulk", 120.0, "active", 4),
        ("tim_between_dies", 0.00005, "tim_effective", 4.0, "interface", 4),
        ("silicon_die_upper", 0.00015, "silicon_bulk", 120.0, "active", 4),
        ("tim_to_spreader", 0.00005, "tim_effective", 4.0, "interface", 4),
        ("spreader", 0.001, "copper_bulk", 400.0, "passive", 16),
    ]
    realized_layers = [
        (
            layer["id"],
            float(layer["thickness_m"]),
            layer["material"],
            float(layer["k_W_mK"]),
            layer["role"],
            int(layer["z_intervals"]),
        )
        for layer in physics["layers_bottom_to_top"]
    ]
    _expect(realized_layers == expected_layers, "formal logic_package layer realization changed")
    _expect(physics["solver_mesh_intervals_xyz"] == [64, 64, 32], "native mesh changed")
    _expect(sum(layer[-1] for layer in expected_layers) == 32, "z interval total mismatch")
    _expect(physics["boundary_conditions"] == {
        "top": {"type": "robin", "h_W_m2K": 500.0, "T_inf_K": 300.0},
        "bottom": {"type": "dirichlet", "T_K": 300.0},
        "sides": {"type": "adiabatic"},
    }, "boundary-condition mismatch")
    _expect(physics["contact"] == {"type": "perfect", "R_contact_m2K_W": 0.0}, "contact mismatch")
    _expect(sum(physics["operator_projection"]["strata"].values()) == 1024, "point strata mismatch")

    contract = registry["power_contract"]
    _expect(contract["package_total_power_W"] == [0.5, 1.0, 2.0, 4.0], "power grid changed")
    _expect(contract["provenance"] == "explicit_user_instruction", "power provenance mismatch")
    _expect(contract["allocation_mode"] == "proportional_to_declared_source_planform_area", "allocation mode mismatch")
    _expect(contract["direct_package_power_to_local_small_source_forbidden"] is True, "small-source guard missing")
    expected_topology_contract = {
        "single_layer_single_source": (1, 1),
        "single_layer_multi_source": (1, 4),
        "dual_layer_few_source": (2, 2),
        "dual_layer_multi_source": (2, 8),
    }
    _expect(set(registry["topologies"]) == set(expected_topology_contract), "topology set mismatch")
    footprint = physics["footprint_m"]
    expected_total_area = float(contract["topology_total_declared_source_area_m2"])
    for topology_id, topology in registry["topologies"].items():
        active_count, source_count = expected_topology_contract[topology_id]
        _expect(topology["active_layer_count"] == active_count, f"{topology_id}: active-layer count")
        _expect(topology["source_count"] == source_count == len(topology["sources"]), f"{topology_id}: source count")
        areas = [_bbox_area(source, footprint) for source in topology["sources"]]
        _expect(math.isclose(sum(areas), expected_total_area, abs_tol=1.0e-15), f"{topology_id}: total source area")
        _expect(min(areas) >= 8.0e-6 - 1.0e-15, f"{topology_id}: local source is too small")
        layers = {source["layer"] for source in topology["sources"]}
        _expect(len(layers) == active_count, f"{topology_id}: realized active layers")
        _expect(layers <= {"silicon_die_lower", "silicon_die_upper"}, f"{topology_id}: source outside silicon")
        for i, left in enumerate(topology["sources"]):
            lx0, lx1, ly0, ly1 = map(float, left["bbox_fraction_xy"])
            for right in topology["sources"][i + 1:]:
                if left["layer"] != right["layer"]:
                    continue
                rx0, rx1, ry0, ry1 = map(float, right["bbox_fraction_xy"])
                overlap = min(lx1, rx1) > max(lx0, rx0) and min(ly1, ry1) > max(ly0, ry0)
                _expect(not overlap, f"{topology_id}: overlapping source areas")

    expected_cases = {
        (topology, power)
        for topology in expected_topology_contract
        for power in (0.5, 1.0, 2.0, 4.0)
    }
    realized_cases = {
        (str(case["topology"]), float(case["package_total_power_W"]))
        for case in registry["cases"]
    }
    _expect(realized_cases == expected_cases, "cases are not the 4 x 4 Cartesian contract")


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return roots


def _check_generator_contract() -> None:
    forbidden = {"jax", "flax", "optax", "rigno"}
    _expect(not (forbidden & _import_roots(Path(generator.__file__))), "P1b generator imports model stack")
    _expect(not (forbidden & _import_roots(Path(p1a.__file__))), "shared solver imports model stack")
    source = Path(generator.__file__).read_text(encoding="utf-8")
    _expect(
        source.index("points, point_strata, point_seed") < source.index("temperature, solver_audit"),
        "point coordinates must freeze before solving temperature",
    )
    _expect("package_power * area_fraction" in source, "source power is not area allocated")
    _expect("source_power / realized_volume" in source, "q is not volume calibrated")


def _check_sample(
    sample_dir: Path,
    manifest_row: Mapping[str, Any],
    case: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, expected_hash in manifest_row["file_sha256"].items():
        path = sample_dir / name
        _expect(path.is_file(), f"{sample_dir.name}: missing {name}")
        _expect(p1a._sha256(path) == expected_hash, f"{sample_dir.name}: hash mismatch {name}")
    coords = np.load(sample_dir / "coords.npy")
    temperature = np.load(sample_dir / "temperature.npy")
    delta_t = np.load(sample_dir / "deltaT.npy")
    k = np.load(sample_dir / "k_field.npy")
    q = np.load(sample_dir / "q_field.npy")
    layer = np.load(sample_dir / "layer_id.npy")
    bc = np.load(sample_dir / "bc_features.npy")
    _expect(coords.shape == (1024, 3), f"{sample_dir.name}: coordinate shape")
    _expect(temperature.shape == delta_t.shape == q.shape == layer.shape == (1024, 1), f"{sample_dir.name}: scalar shapes")
    _expect(k.shape == (1024, 3) and bc.shape == (1024, 4), f"{sample_dir.name}: feature shapes")
    _expect(all(np.all(np.isfinite(array)) for array in (coords, temperature, delta_t, k, q, layer, bc)), f"{sample_dir.name}: non-finite array")
    _expect(len(np.unique(coords, axis=0)) == 1024, f"{sample_dir.name}: duplicate points")
    _expect(np.max(np.abs(delta_t[:, 0] - (temperature[:, 0] - 300.0))) < 1.0e-12, f"{sample_dir.name}: DeltaT mismatch")
    bottom = bc[:, 1] == 1.0
    _expect(int(np.sum(bc[:, 0])) == 64 and int(np.sum(bottom)) == 64, f"{sample_dir.name}: boundary strata")
    _expect(np.allclose(temperature[bottom, 0], 300.0, atol=1.0e-10), f"{sample_dir.name}: bottom Dirichlet")
    _expect(np.all(q[bottom, 0] == 0.0), f"{sample_dir.name}: source on bottom Dirichlet points")

    meta = _read_json(sample_dir / "sample_meta.json")
    _expect(meta["stack_template_id"] == "logic_package", f"{sample_dir.name}: stack ID")
    _expect(meta["topology"] == case["topology"] == manifest_row["topology"], f"{sample_dir.name}: topology")
    _expect(meta["power_allocation_mode"] == "proportional_to_declared_source_planform_area", f"{sample_dir.name}: allocation")
    _expect(meta["power_was_Rth_inferred"] is False, f"{sample_dir.name}: inferred power")
    _expect(meta["direct_package_power_to_local_small_source"] is False, f"{sample_dir.name}: direct local package power")
    guards = meta["guardrails"]
    _expect(guards == {
        "model_inference_runs": 0,
        "peak_deltaT_filtering": False,
        "peak_deltaT_resampling": False,
        "power_back_calculation": False,
        "sample_replacement": False,
        "training_runs": 0,
    }, f"{sample_dir.name}: guardrails")
    _expect(meta["contact"] == {"R_contact_m2K_W": 0.0, "type": "perfect"}, f"{sample_dir.name}: contact")
    mesh_meta = meta["solver_mesh"]
    _expect(mesh_meta["intervals_xyz"] == [64, 64, 32] and mesh_meta["node_count"] == 139425, f"{sample_dir.name}: mesh")
    _expect(mesh_meta["source_on_bottom_dirichlet_node_count"] == 0, f"{sample_dir.name}: forbidden source nodes")
    _expect(mesh_meta["minimum_source_in_plane_interval_count"] >= 8, f"{sample_dir.name}: source resolution")
    projection = meta["operator_projection"]
    _expect(projection["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_dir.name}: point freeze")
    _expect(projection["label_inputs_used_for_point_selection"] == [], f"{sample_dir.name}: point label leak")
    _expect(projection["strata_counts"] == {"bottom": 64, "interface": 128, "source": 256, "top": 64, "volume": 512}, f"{sample_dir.name}: point strata")
    _expect(p1a._array_sha256(coords) == projection["point_coordinates_sha256"] == manifest_row["point_coordinates_sha256"], f"{sample_dir.name}: point SHA")

    topology = registry["topologies"][case["topology"]]
    sources = meta["sources"]
    _expect(len(sources) == topology["source_count"], f"{sample_dir.name}: source count")
    expected_area = float(registry["power_contract"]["topology_total_declared_source_area_m2"])
    package_power = float(case["package_total_power_W"])
    expected_surface_density = package_power / expected_area
    source_power_sum = 0.0
    source_area_sum = 0.0
    layer_power: dict[str, float] = {}
    for source in sources:
        area = float(source["declared_source_area_m2"])
        power = float(source["source_power_W"])
        volume = float(source["realized_source_control_volume_m3"])
        density = float(source["q_W_m3"])
        _expect(math.isclose(power, package_power * area / expected_area, rel_tol=1.0e-12), f"{sample_dir.name}: source area allocation")
        _expect(math.isclose(float(source["surface_power_density_W_m2"]), expected_surface_density, rel_tol=1.0e-12), f"{sample_dir.name}: surface density")
        _expect(math.isclose(power, density * volume, rel_tol=1.0e-12, abs_tol=1.0e-13), f"{sample_dir.name}: q-volume-power")
        _expect(int(source["covered_control_volume_count"]) >= 256, f"{sample_dir.name}: source CV coverage")
        _expect(min(int(source["resolved_x_interval_count"]), int(source["resolved_y_interval_count"])) >= 8, f"{sample_dir.name}: in-plane intervals")
        _expect(int(source["physical_layer_z_interval_count"]) >= 4, f"{sample_dir.name}: layer z intervals")
        source_area_sum += area
        source_power_sum += power
        layer_name = str(source["active_layer"])
        layer_power[layer_name] = layer_power.get(layer_name, 0.0) + power
    _expect(math.isclose(source_area_sum, expected_area, abs_tol=1.0e-15), f"{sample_dir.name}: total source area")
    _expect(math.isclose(source_power_sum, package_power, rel_tol=1.0e-12), f"{sample_dir.name}: package power")
    _expect(all(math.isclose(value, float(meta["active_layer_power_W"][name]), rel_tol=1.0e-12) for name, value in layer_power.items()), f"{sample_dir.name}: layer power")
    metrics = meta["metrics"]
    _expect(math.isclose(float(metrics["package_total_power_W"]), package_power, rel_tol=1.0e-12), f"{sample_dir.name}: integrated power")
    _expect(all(math.isfinite(float(value)) for key, value in metrics.items() if key != "in_30_80_K_window"), f"{sample_dir.name}: finite metrics")
    _expect(abs(float(metrics["energy_balance_relative_error"])) < 1.0e-8, f"{sample_dir.name}: energy balance")
    _expect(abs(float(metrics["top_heat_fraction"]) + float(metrics["bottom_heat_fraction"]) - 1.0) < 1.0e-8, f"{sample_dir.name}: heat fractions")
    expected_window = 30.0 <= float(metrics["peak_deltaT_K"]) <= 80.0
    _expect(metrics["in_30_80_K_window"] is expected_window, f"{sample_dir.name}: window flag")
    return meta, sources


def check(args: argparse.Namespace) -> dict[str, Any]:
    registry = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    _check_registry(registry)
    _check_generator_contract()
    manifest = _read_json(args.dataset / "manifest.json")
    tracked_manifest = _read_json(args.manifest_json)
    _expect(tracked_manifest == manifest, "tracked manifest differs from dataset manifest")
    _expect(manifest["sample_count"] == 16 and len(manifest["samples"]) == 16, "manifest sample count")
    _expect(manifest["case_registry_sha256"] == p1a._sha256(args.cases), "case registry SHA")
    _expect(manifest["guardrails"] == {
        "model_inference_runs": 0,
        "peak_deltaT_filtering": False,
        "peak_deltaT_resampling": False,
        "power_back_calculation": False,
        "training_runs": 0,
    }, "manifest guardrails")
    cases = {str(case["id"]): case for case in registry["cases"]}
    manifest_ids = [str(row["sample_id"]) for row in manifest["samples"]]
    _expect(len(set(manifest_ids)) == 16 and set(manifest_ids) == set(cases), "manifest IDs")
    dataset_dirs = {path.name for path in args.dataset.iterdir() if path.is_dir()}
    _expect(dataset_dirs == set(cases), "missing or extra dataset directories")
    source_count = 0
    for row in manifest["samples"]:
        _, sources = _check_sample(args.dataset / row["sample_id"], row, cases[row["sample_id"]], registry)
        source_count += len(sources)

    audit = _read_json(args.audit_json)
    _expect(audit["dataset_manifest_sha256"] == p1a._sha256(args.dataset / "manifest.json"), "dataset manifest SHA")
    _expect(audit["tracked_manifest_sha256"] == p1a._sha256(args.manifest_json), "tracked manifest SHA")
    _expect(audit["sample_count"] == 16 and audit["source_count"] == source_count, "audit counts")
    _expect(audit["topology_case_counts"] == {
        "dual_layer_few_source": 4,
        "dual_layer_multi_source": 4,
        "single_layer_multi_source": 4,
        "single_layer_single_source": 4,
    }, "topology balance")
    _expect(audit["integrity"]["all_metrics_finite"] is True, "audit finite flag")
    _expect(audit["integrity"]["min_source_control_volume_count"] >= 256, "audit CV coverage")
    _expect(audit["integrity"]["min_source_in_plane_interval_count"] >= 8, "audit source intervals")
    _expect(audit["integrity"]["source_on_bottom_dirichlet_node_count"] == 0, "audit forbidden source nodes")
    _expect(audit["guardrails"] == {
        "expanded_samples": 0,
        "generated_samples": 16,
        "model_inference_runs": 0,
        "peak_deltaT_filtering": False,
        "peak_deltaT_resampling": False,
        "power_back_calculation": False,
        "training_runs": 0,
    }, "audit guardrails")
    with args.samples_csv.open(encoding="utf-8", newline="") as handle:
        sample_csv = list(csv.DictReader(handle))
    with args.sources_csv.open(encoding="utf-8", newline="") as handle:
        source_csv = list(csv.DictReader(handle))
    _expect(len(sample_csv) == 16 and len(source_csv) == source_count, "CSV row counts")
    _expect({row["sample_id"] for row in sample_csv} == set(cases), "sample CSV IDs")
    return {
        "schema_version": "heat3d_v6_p1b_checker_v1",
        "passed": True,
        "sample_count": 16,
        "source_count": source_count,
        "window_hit_count": audit["window_hit_count"],
        "max_abs_energy_balance_relative_error": audit["integrity"]["max_abs_energy_balance_relative_error"],
        "min_source_control_volume_count": audit["integrity"]["min_source_control_volume_count"],
        "min_source_in_plane_interval_count": audit["integrity"]["min_source_in_plane_interval_count"],
        "power_back_calculation": False,
        "training_runs": 0,
        "model_inference_runs": 0,
        "expanded_samples": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("cases", "dataset", "audit_json", "manifest_json", "samples_csv", "sources_csv"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (REPO_ROOT / value).resolve())
    print(json.dumps(check(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
