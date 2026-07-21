#!/usr/bin/env python3
"""Validate V6-P1c package-path calibration inputs, artifacts, and guardrails."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import generate_heat3d_v6_p1c_package_path_calibration as generator


REPO_ROOT = Path(__file__).resolve().parent.parent


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{path}: expected object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return roots


def _check_registry(registry: Mapping[str, Any]) -> None:
    _expect(registry["schema_version"] == "heat3d_v6_p1c_package_path_cases_v1", "schema")
    _expect(registry["sample_count"] == 8 and len(registry["cases"]) == 8, "sample count")
    scope = registry["scope"]
    for key in ("baseline_A_regeneration", "peak_deltaT_filtering", "peak_deltaT_resampling", "power_back_calculation", "material_parameter_tuning", "model_training", "model_inference"):
        _expect(scope[key] is False, f"forbidden scope enabled: {key}")
    _expect(scope["layered_stacks_only"] is True, "non-layered scope")
    _expect(registry["common_physics"]["footprint_m"] == [0.010, 0.010], "footprint")
    _expect(registry["common_physics"]["top"] == {"type": "robin", "h_W_m2K": 500.0, "T_inf_K": 300.0}, "top BC")
    _expect(registry["common_physics"]["sides"] == {"type": "adiabatic"}, "side BC")
    _expect(registry["common_physics"]["contact"] == {"type": "perfect", "R_contact_m2K_W": 0.0}, "contact")
    _expect(registry["common_physics"]["source_face_policy"] == "preserve_p1b_source_z_node_policy", "source z policy")
    _expect(sum(registry["common_physics"]["operator_projection"]["strata"].values()) == 1024, "projection count")

    baseline = registry["baseline_A"]
    baseline_manifest = REPO_ROOT / baseline["manifest_path"]
    _expect(generator._sha256(baseline_manifest) == baseline["manifest_sha256"], "baseline A manifest SHA")
    _expect((REPO_ROOT / baseline["dataset_path"]).is_dir(), "baseline A dataset unavailable")

    b = registry["paths"]["B_remote_dirichlet"]
    _expect(b["bottom"] == {"type": "dirichlet", "T_K": 300.0, "location": "pcb_exterior_bottom"}, "B bottom")
    expected_added = [
        ("pcb_fr4_equivalent", 0.0016, [0.8, 0.8, 0.3], 8),
        ("bt_substrate_with_vias", 0.001, [0.2, 0.2, 0.49], 8),
        ("silicon_interposer_tsv_0p1", 0.0001, [148.3, 148.3, 151.0], 4),
        ("bump_underfill_under_interposer", 0.000075, [0.6, 0.6, 4.9], 4),
    ]
    realized_added = [
        (layer["id"], float(layer["thickness_m"]), list(map(float, layer["k_xyz_W_mK"])), int(layer["z_intervals"]))
        for layer in b["prepended_layers_bottom_to_top"]
    ]
    _expect(realized_added == expected_added, "B literature layers changed")
    _expect(b["solver_mesh_intervals_xyz"] == [64, 64, 56], "B mesh")
    c0 = registry["paths"]["C0_bottom_adiabatic"]
    _expect(c0["bottom"] == {"type": "adiabatic", "location": "lower_die_exterior_bottom"}, "C0 bottom")
    _expect(c0["prepended_layers_bottom_to_top"] == [] and c0["solver_mesh_intervals_xyz"] == [64, 64, 32], "C0 stack")

    _expect(registry["power_contract"]["package_total_power_W"] == [1.0, 4.0], "power grid")
    _expect(set(registry["topologies"]) == {"single_layer_single_source", "dual_layer_multi_source"}, "topologies")
    expected_cases = {
        (path, topology, power)
        for path in ("B_remote_dirichlet", "C0_bottom_adiabatic")
        for topology in ("single_layer_single_source", "dual_layer_multi_source")
        for power in (1.0, 4.0)
    }
    realized_cases = {(case["path"], case["topology"], float(case["package_total_power_W"])) for case in registry["cases"]}
    _expect(realized_cases == expected_cases, "cases are not 2x2x2")


def _check_generator() -> None:
    forbidden = {"jax", "flax", "optax", "rigno"}
    _expect(not (_import_roots(Path(generator.__file__)) & forbidden), "generator imports model stack")
    source = Path(generator.__file__).read_text(encoding="utf-8")
    generation_loop = source[source.index("temp_root = Path(tempfile.mkdtemp"):]
    _expect(generation_loop.index("points, point_strata, point_seed") < generation_loop.index("temperature, solver_audit = solver.solve"), "points not frozen before solve")
    _expect("baseline_A_regenerated\": False" in source, "baseline A guard missing")
    _expect("package_power * area / total_area" in source, "area power allocation missing")


def _check_sample(
    sample_dir: Path, manifest_row: Mapping[str, Any], case: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    for name, expected in manifest_row["file_sha256"].items():
        path = sample_dir / name
        _expect(path.is_file() and p1a._sha256(path) == expected, f"{sample_dir.name}: {name} hash")
    coords = np.load(sample_dir / "coords.npy")
    temperature = np.load(sample_dir / "temperature.npy")
    delta = np.load(sample_dir / "deltaT.npy")
    k = np.load(sample_dir / "k_field.npy")
    q = np.load(sample_dir / "q_field.npy")
    layer = np.load(sample_dir / "layer_id.npy")
    bc = np.load(sample_dir / "bc_features.npy")
    _expect(coords.shape == (1024, 3), f"{sample_dir.name}: coords")
    _expect(temperature.shape == delta.shape == q.shape == layer.shape == (1024, 1), f"{sample_dir.name}: scalar shapes")
    _expect(k.shape == (1024, 3) and bc.shape == (1024, 4), f"{sample_dir.name}: feature shapes")
    _expect(all(np.all(np.isfinite(value)) for value in (coords, temperature, delta, k, q, layer, bc)), f"{sample_dir.name}: finite")
    _expect(len(np.unique(coords, axis=0)) == 1024, f"{sample_dir.name}: unique points")
    _expect(np.max(np.abs(delta[:, 0] - (temperature[:, 0] - 300.0))) < 1e-11, f"{sample_dir.name}: deltaT")
    _expect(int(np.sum(bc[:, 0])) == 64 and int(np.sum(bc[:, 1])) == 64, f"{sample_dir.name}: boundary strata")
    meta = _read_json(sample_dir / "sample_meta.json")
    _expect(meta["package_path"] == case["path"] == manifest_row["path"], f"{sample_dir.name}: path")
    _expect(meta["topology"] == case["topology"] == manifest_row["topology"], f"{sample_dir.name}: topology")
    _expect(meta["power_was_Rth_inferred"] is False and meta["material_parameters_tuned_after_solve"] is False, f"{sample_dir.name}: tuning")
    _expect(meta["operator_projection"]["label_inputs_used_for_point_selection"] == [], f"{sample_dir.name}: label leak")
    _expect(meta["operator_projection"]["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_dir.name}: point freeze")
    _expect(p1a._array_sha256(coords) == meta["operator_projection"]["point_coordinates_sha256"] == manifest_row["point_coordinates_sha256"], f"{sample_dir.name}: point SHA")
    _expect(meta["operator_projection"]["strata_counts"] == {"bottom": 64, "interface": 128, "source": 256, "top": 64, "volume": 512}, f"{sample_dir.name}: strata")
    guards = meta["guardrails"]
    for key in ("baseline_A_regenerated", "peak_deltaT_filtering", "peak_deltaT_resampling", "power_back_calculation", "material_parameter_tuning", "sample_replacement"):
        _expect(guards[key] is False, f"{sample_dir.name}: guard {key}")
    _expect(guards["training_runs"] == guards["model_inference_runs"] == 0, f"{sample_dir.name}: model work")

    path_id = str(case["path"])
    if path_id == "B_remote_dirichlet":
        _expect(meta["solver_mesh"]["intervals_xyz"] == [64, 64, 56] and meta["solver_mesh"]["node_count"] == 240825, f"{sample_dir.name}: B mesh")
        bottom = bc[:, 1] == 1.0
        _expect(np.allclose(temperature[bottom, 0], 300.0, atol=1e-8), f"{sample_dir.name}: remote Dirichlet")
        _expect(len(meta["layers_bottom_to_top"]) == 9, f"{sample_dir.name}: B layers")
    else:
        _expect(meta["solver_mesh"]["intervals_xyz"] == [64, 64, 32] and meta["solver_mesh"]["node_count"] == 139425, f"{sample_dir.name}: C0 mesh")
        _expect(len(meta["layers_bottom_to_top"]) == 5, f"{sample_dir.name}: C0 layers")

    expected_area = float(registry["power_contract"]["topology_total_declared_source_area_m2"])
    power = float(case["package_total_power_W"])
    sources = meta["sources"]
    _expect(math.isclose(sum(float(source["declared_source_area_m2"]) for source in sources), expected_area, abs_tol=1e-15), f"{sample_dir.name}: source area")
    _expect(math.isclose(sum(float(source["source_power_W"]) for source in sources), power, rel_tol=1e-12), f"{sample_dir.name}: power")
    for source in sources:
        _expect(math.isclose(float(source["source_power_W"]), float(source["q_W_m3"]) * float(source["realized_source_control_volume_m3"]), rel_tol=1e-12), f"{sample_dir.name}: q power")
        _expect(int(source["covered_control_volume_count"]) >= 256, f"{sample_dir.name}: source resolution")

    metrics = meta["metrics"]
    finite_required = [value for key, value in metrics.items() if value is not None and key not in {"junction_to_board_status", "in_30_80_K_window"}]
    _expect(all(math.isfinite(float(value)) for value in finite_required), f"{sample_dir.name}: metric finite")
    _expect(abs(float(metrics["energy_balance_relative_error"])) < 1e-8, f"{sample_dir.name}: energy")
    _expect(abs(float(metrics["top_heat_fraction"]) + float(metrics["bottom_heat_fraction"]) - 1.0) < 1e-8, f"{sample_dir.name}: heat fractions")
    if path_id == "B_remote_dirichlet":
        _expect(float(metrics["top_heat_fraction"]) > 0.5 and float(metrics["bottom_heat_fraction"]) > 0.0, f"{sample_dir.name}: B path split")
        _expect(metrics["junction_to_board_R_K_W"] is not None, f"{sample_dir.name}: board R")
    else:
        _expect(float(metrics["bottom_heat_fraction"]) == 0.0 and math.isclose(float(metrics["top_heat_fraction"]), 1.0, abs_tol=1e-8), f"{sample_dir.name}: C0 path")
        _expect(metrics["junction_to_board_R_K_W"] is None and metrics["junction_to_board_status"].startswith("not_applicable"), f"{sample_dir.name}: C0 board R")
    return meta


def check(args: argparse.Namespace) -> dict[str, Any]:
    registry = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    _check_registry(registry)
    _check_generator()
    manifest = _read_json(args.dataset / "manifest.json")
    tracked = _read_json(args.manifest_json)
    _expect(manifest == tracked, "tracked manifest mismatch")
    _expect(manifest["sample_count"] == 8 and len(manifest["samples"]) == 8, "manifest count")
    _expect(manifest["guardrails"] == {"baseline_A_regenerated": False, "generated_samples": 8, "model_inference_runs": 0, "training_runs": 0}, "manifest guards")
    cases = {str(case["id"]): case for case in registry["cases"]}
    metas = [
        _check_sample(args.dataset / row["sample_dir"], row, cases[str(row["sample_id"])], registry)
        for row in manifest["samples"]
    ]
    _expect({meta["sample_id"] for meta in metas} == set(cases), "sample ID set")

    samples = _read_csv(args.samples_csv)
    sources = _read_csv(args.sources_csv)
    layers = _read_csv(args.layers_csv)
    interfaces = _read_csv(args.interfaces_csv)
    paired = _read_csv(args.paired_csv)
    _expect(len(samples) == 8 and len(sources) == 36, "sample/source CSV counts")
    _expect(len(layers) == 4 * 5 + 4 * 9 + 4 * 5, "layer CSV includes A/B/C0")
    _expect(len(interfaces) == 4 * 4 + 4 * 8 + 4 * 4, "interface CSV includes A/B/C0")
    _expect(len(paired) == 8, "paired CSV count")
    _expect(all(float(row["perfect_contact_temperature_jump_K"]) == 0.0 for row in interfaces), "perfect-contact jump")
    _expect({row["comparison_path"] for row in paired} == {"B_remote_dirichlet", "C0_bottom_adiabatic"}, "paired paths")

    audit = _read_json(args.audit_json)
    _expect(audit["sample_count_new"] == 8 and audit["baseline_A_generated_samples"] == 0, "audit counts")
    _expect(audit["guardrails"]["training_runs"] == audit["guardrails"]["model_inference_runs"] == 0, "audit model work")
    _expect(audit["top_path_assessment"] == {
        "B_remote_dirichlet_top_dominant_all_cases": True,
        "C0_adiabatic_top_fraction_equals_one": True,
        "C0_board_resistance_defined": False,
    }, "top path assessment")
    _expect(audit["integrity"]["max_abs_energy_balance_relative_error"] < 1e-8, "audit energy")
    dry = generator.generate(argparse.Namespace(
        cases=args.cases, dataset=args.dataset, audit_json=args.audit_json,
        manifest_json=args.manifest_json, samples_csv=args.samples_csv,
        sources_csv=args.sources_csv, layers_csv=args.layers_csv,
        interfaces_csv=args.interfaces_csv, paired_csv=args.paired_csv, dry_run=True,
    ))
    _expect(dry["sample_count"] == 8 and dry["baseline_A_dataset_writes"] == 0, "dry-run scope")
    return {
        "status": "ok", "sample_count": 8, "source_count": len(sources),
        "paired_comparison_count": len(paired),
        "max_abs_energy_balance_relative_error": audit["integrity"]["max_abs_energy_balance_relative_error"],
        "max_abs_projection_peak_gap_K": audit["integrity"]["max_abs_projection_peak_gap_K"],
        "baseline_A_regenerated": False, "training_runs": 0, "model_inference_runs": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=generator.DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=generator.DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=generator.DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=generator.DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=generator.DEFAULT_SAMPLES)
    parser.add_argument("--sources-csv", type=Path, default=generator.DEFAULT_SOURCES)
    parser.add_argument("--layers-csv", type=Path, default=generator.DEFAULT_LAYERS)
    parser.add_argument("--interfaces-csv", type=Path, default=generator.DEFAULT_INTERFACES)
    parser.add_argument("--paired-csv", type=Path, default=generator.DEFAULT_PAIRED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(check(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
