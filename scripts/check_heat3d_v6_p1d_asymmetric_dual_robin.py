#!/usr/bin/env python3
"""Check the V6-P1d search, frozen configs, generated data, and guardrails."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import generate_heat3d_v6_p1d_asymmetric_dual_robin as generator
import heat3d_v6_p1d_core as core
import prepare_heat3d_v6_p1d_expansion_config as expansion


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs/heat3d_v6"
LITERATURE = REPO_ROOT / "docs/v6_p1d_literature_matrix.csv"
SEARCH_CONFIG = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_search.yaml"
EXPLORATION = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_exploration.json"
EXPLORATION_CSV = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_exploration_attempts.csv"
PILOT16_CONFIG = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_pilot16.yaml"
TRIAL1_64_CONFIG = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_pilot64.yaml"
BALANCED64_CONFIG = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin_pilot64_balanced.yaml"
FINAL1024_CONFIG = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin1024.yaml"
DATASETS = {
    "pilot16": REPO_ROOT / "data/heat3d_v6_p1d_asymmetric_dual_robin16_v0",
    "balanced64": REPO_ROOT / "data/heat3d_v6_p1d_asymmetric_dual_robin64_balanced_v1",
    "final1024": REPO_ROOT / "data/heat3d_v6_p1d_asymmetric_dual_robin1024_v0",
}
ARTIFACT_STEMS = {
    "pilot16": "v6_p1d_asymmetric_dual_robin16",
    "balanced64": "v6_p1d_asymmetric_dual_robin64_balanced",
    "final1024": "v6_p1d_asymmetric_dual_robin1024",
}
CONFIGS = {
    "pilot16": PILOT16_CONFIG,
    "balanced64": BALANCED64_CONFIG,
    "final1024": FINAL1024_CONFIG,
}
EXPECTED_COUNTS = {"pilot16": 16, "balanced64": 64, "final1024": 1024}
EXPECTED_BINS = {"pilot16": None, "balanced64": 16, "final1024": 256}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expect(isinstance(payload, dict), f"{path}: object expected")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def import_roots(path: Path) -> set[str]:
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


def check_literature_and_search() -> dict[str, Any]:
    literature = read_csv(LITERATURE)
    expect(len(literature) >= 8, "literature matrix has fewer than eight traceable sources")
    expect(sum(bool(row["doi"]) for row in literature) >= 5, "insufficient DOI evidence")
    expect(all(row["url"].startswith("https://") for row in literature), "literature URL missing")
    expect(sum("book chapter" not in row["limitation"] for row in literature) >= 7, "primary evidence count")

    search = yaml.safe_load(SEARCH_CONFIG.read_text(encoding="utf-8"))
    expect(search["scope"]["model_training"] is False, "search training scope")
    expect(search["scope"]["model_inference"] is False, "search inference scope")
    exploration = read_json(EXPLORATION)
    attempts = read_csv(EXPLORATION_CSV)
    expect(exploration["attempt_count"] == len(attempts) == 64, "exploration count")
    expect(exploration["all_attempts_retained"] is True, "attempt retention")
    expect(exploration["attempt_deletion_count"] == 0, "attempt deletion")
    expect(all(row["attempt_retained"] == "True" for row in attempts), "unretained exploration row")
    expect(exploration["guardrails"] == {
        "final_sample_generation": False,
        "model_inference_runs": 0,
        "per_sample_Rth_power_back_calculation": False,
        "result_dependent_attempt_deletion": False,
        "training_runs": 0,
    }, "exploration guardrails")
    expect(core.sha256(LITERATURE) == exploration["literature_matrix_sha256"], "literature SHA")
    expect(core.sha256(SEARCH_CONFIG) == exploration["search_config_sha256"], "search config SHA")
    return {"literature_rows": len(literature), "exploration_attempts": len(attempts)}


def check_config(name: str, path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    count = EXPECTED_COUNTS[name]
    expect(config["sample_count"] == len(config["cases"]) == count, f"{name}: config count")
    expect(len({case["id"] for case in config["cases"]}) == count, f"{name}: duplicate ID")
    scope = config["scope"]
    for key in (
        "model_training", "model_inference", "peak_deltaT_filtering",
        "peak_deltaT_resampling", "sample_replacement",
        "per_sample_Rth_power_back_calculation", "material_parameter_tuning",
    ):
        expect(scope[key] is False, f"{name}: forbidden scope {key}")
    physics = config["physics"]
    expect(physics["footprint_m"] == [0.01, 0.01], f"{name}: footprint")
    expect(physics["solver_mesh_intervals_xyz"] == [64, 64, 56], f"{name}: mesh")
    bc = physics["boundary_common"]
    expect(bc["top_ambient_K"] == bc["bottom_ambient_K"] == 300.0, f"{name}: ambient")
    expect(bc["sides"] == "adiabatic", f"{name}: side BC")
    expect(bc["contact"] == {"type": "perfect", "R_contact_m2K_W": 0.0}, f"{name}: contact")
    expect(config["source_contract"]["topology"] == "dual_layer_multi_source", f"{name}: topology")
    expect(config["source_contract"]["source_count"] == 8, f"{name}: source count")
    expect(sum(config["operator_projection"]["strata"].values()) == 1024, f"{name}: projection")
    branch = config["branch_resistance_contract"]
    expect(branch["top_branch_R_formula"] == "(T_junction - T_inf_top) / Q_top", f"{name}: top branch definition")
    expect(branch["bottom_branch_R_formula"] == "(T_junction - T_inf_bottom) / Q_bottom", f"{name}: bottom branch definition")
    expect(branch["parallel_closure_formula"].startswith("R_effective == 1 /"), f"{name}: parallel closure")

    families = Counter(case["family_id"] for case in config["cases"])
    areas = Counter(float(case["total_source_area_mm2"]) for case in config["cases"])
    expect(set(families) == set(expansion.BALANCED_POWER_GRID_W), f"{name}: BC family set")
    expect(set(areas) == set(expansion.AREAS), f"{name}: source areas")
    for case in config["cases"]:
        top_h = float(case["top_h_W_m2K"])
        bottom_h = float(case["bottom_h_W_m2K"])
        expect(500.0 <= top_h <= 2500.0, f"{name}: top h")
        expect(bottom_h in {1.0, 2.0} or 20.0 <= bottom_h <= 200.0, f"{name}: bottom h")
    if name in {"balanced64", "final1024"}:
        trial1 = REPO_ROOT / config["provenance"]["trial1_complete_cartesian_audit"]
        expect(core.sha256(trial1) == config["provenance"]["trial1_complete_cartesian_audit_sha256"], f"{name}: trial1 provenance SHA")
        expect("never individual-sample Rth inversion" in config["provenance"]["power_grid_rule"], f"{name}: power freeze rule")
        expect(config["source_contract"]["balanced_power_grid_W"] == {
            family: list(powers) for family, powers in expansion.BALANCED_POWER_GRID_W.items()
        }, f"{name}: power grid")
        expected_per_bin = EXPECTED_BINS[name]
        slots = Counter(case["selection_bin"] for case in config["cases"])
        expect(slots == Counter({slot: expected_per_bin for slot in expansion.TEMPERATURE_SLOTS}), f"{name}: slot balance")
        expect(all(value == count // 8 for value in families.values()), f"{name}: family balance")
        expect(all(value == count // 4 for value in areas.values()), f"{name}: area balance")
    return {"count": count, "families": dict(families), "areas": dict(areas)}


def check_sample(sample_dir: Path, manifest_row: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    for name, expected_sha in manifest_row["file_sha256"].items():
        path = sample_dir / name
        expect(path.is_file() and core.sha256(path) == expected_sha, f"{sample_dir.name}: hash {name}")
    arrays = {
        name: np.load(sample_dir / name, allow_pickle=False)
        for name in (
            "coords.npy", "temperature.npy", "deltaT.npy", "k_field.npy", "q_field.npy",
            "layer_id.npy", "bc_features.npy", "bc_parameters.npy", "sampling_stratum.npy",
        )
    }
    expect(arrays["coords.npy"].shape == (1024, 3), f"{sample_dir.name}: coords")
    expect(arrays["temperature.npy"].shape == arrays["deltaT.npy"].shape == (1024, 1), f"{sample_dir.name}: labels")
    expect(arrays["k_field.npy"].shape == (1024, 3), f"{sample_dir.name}: k")
    expect(arrays["q_field.npy"].shape == arrays["layer_id.npy"].shape == (1024, 1), f"{sample_dir.name}: inputs")
    expect(arrays["bc_features.npy"].shape == (1024, 4), f"{sample_dir.name}: BC masks")
    expect(arrays["bc_parameters.npy"].shape == (1024, 4), f"{sample_dir.name}: BC parameters")
    expect(arrays["sampling_stratum.npy"].shape == (1024, 1), f"{sample_dir.name}: strata")
    expect(all(np.all(np.isfinite(array)) for array in arrays.values()), f"{sample_dir.name}: finite arrays")
    expect(np.max(np.abs(arrays["temperature.npy"] - 300.0 - arrays["deltaT.npy"])) < 1e-10, f"{sample_dir.name}: deltaT")
    expected_bc = np.array([
        float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]), 300.0, 300.0,
    ])
    expect(np.allclose(arrays["bc_parameters.npy"], expected_bc[None, :]), f"{sample_dir.name}: BC values")

    meta = read_json(sample_dir / "sample_meta.json")
    expect(meta["sample_id"] == case["id"] == manifest_row["sample_id"], f"{sample_dir.name}: ID")
    expect(meta["topology"] == "dual_layer_multi_source", f"{sample_dir.name}: topology")
    expect(len(meta["layers_bottom_to_top"]) == 9, f"{sample_dir.name}: complete B stack")
    expect(meta["solver_mesh"]["intervals_xyz"] == [64, 64, 56], f"{sample_dir.name}: native mesh")
    expect(meta["solver_mesh"]["node_count"] == 240825, f"{sample_dir.name}: node count")
    expect(meta["operator_projection"]["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_dir.name}: point freeze")
    expect(meta["operator_projection"]["label_inputs_used_for_point_selection"] == [], f"{sample_dir.name}: label leak")
    coverage = meta["operator_projection"]["coverage"]
    expect(coverage["all_layers_covered"] and coverage["all_interfaces_covered"], f"{sample_dir.name}: coverage")
    expect(len(meta["sources"]) == 8, f"{sample_dir.name}: sources")
    expect({source["active_layer"] for source in meta["sources"]} == {"silicon_die_lower", "silicon_die_upper"}, f"{sample_dir.name}: active layers")
    expect(math.isclose(
        sum(float(source["source_power_W"]) for source in meta["sources"]),
        float(case["package_total_power_W"]), rel_tol=1e-12,
    ), f"{sample_dir.name}: power")
    expect(math.isclose(sum(float(source["declared_source_area_m2"]) for source in meta["sources"]), float(case["total_source_area_mm2"]) * 1e-6, rel_tol=0, abs_tol=1e-15), f"{sample_dir.name}: area")
    expect(min(int(source["covered_control_volume_count"]) for source in meta["sources"]) >= 240, f"{sample_dir.name}: source CV")
    expect(min(min(int(source["resolved_x_interval_count"]), int(source["resolved_y_interval_count"])) for source in meta["sources"]) >= 7, f"{sample_dir.name}: source intervals")
    metrics = meta["metrics"]
    expect(abs(float(metrics["energy_balance_relative_error"])) < 1e-8, f"{sample_dir.name}: energy")
    expect(abs(float(metrics["parallel_branch_relative_closure_error"])) < 1e-8, f"{sample_dir.name}: parallel closure")
    tj = float(metrics["junction_temperature_K"])
    qtop = float(metrics["top_heat_flux_W"])
    qbottom = float(metrics["bottom_heat_flux_W"])
    expect(math.isclose(float(metrics["top_branch_R_ambient_K_W"]), (tj - 300.0) / qtop, rel_tol=1e-12), f"{sample_dir.name}: top branch R")
    expect(math.isclose(float(metrics["bottom_branch_R_ambient_K_W"]), (tj - 300.0) / qbottom, rel_tol=1e-12), f"{sample_dir.name}: bottom branch R")
    guards = meta["guardrails"]
    expect(guards["training_runs"] == guards["model_inference_runs"] == 0, f"{sample_dir.name}: model work")
    expect(not any(guards[key] for key in ("peak_deltaT_filtering", "peak_deltaT_resampling", "sample_replacement", "per_sample_Rth_power_back_calculation", "material_parameter_tuning")), f"{sample_dir.name}: guardrails")


def check_dataset(name: str) -> dict[str, Any]:
    count = EXPECTED_COUNTS[name]
    config_path = CONFIGS[name]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in config["cases"]}
    dataset = DATASETS[name]
    manifest = read_json(dataset / "manifest.json")
    stem = ARTIFACT_STEMS[name]
    tracked_manifest = read_json(CONFIG_DIR / f"{stem}_manifest.json")
    expect(manifest == tracked_manifest, f"{name}: tracked manifest")
    expect(manifest["config_sha256"] == core.sha256(config_path), f"{name}: config SHA")
    expect(manifest["sample_count"] == len(manifest["samples"]) == count, f"{name}: manifest count")
    expect(manifest["guardrails"] == {"temperature_filtered_samples": 0, "sample_replacements": 0, "training_runs": 0, "model_inference_runs": 0}, f"{name}: manifest guardrails")
    for row in manifest["samples"]:
        check_sample(dataset / row["sample_dir"], row, cases[row["sample_id"]])
    expect({row["sample_id"] for row in manifest["samples"]} == set(cases), f"{name}: sample set")

    sample_rows = read_csv(CONFIG_DIR / f"{stem}_samples.csv")
    source_rows = read_csv(CONFIG_DIR / f"{stem}_sources.csv")
    layer_rows = read_csv(CONFIG_DIR / f"{stem}_layer_drops.csv")
    interface_rows = read_csv(CONFIG_DIR / f"{stem}_interface_drops.csv")
    expect(len(sample_rows) == count, f"{name}: sample CSV")
    expect(len(source_rows) == 8 * count, f"{name}: source CSV")
    expect(len(layer_rows) == 9 * count, f"{name}: layer CSV")
    expect(len(interface_rows) == 8 * count, f"{name}: interface CSV")

    audit = read_json(CONFIG_DIR / f"{stem}_audit.json")
    expect(audit["sample_count"] == count and audit["source_count"] == 8 * count, f"{name}: audit count")
    expect(audit["window_hit_count"] == count, f"{name}: 30--80 K window")
    expect(audit["temperature_bin_counts"]["outside"] == 0, f"{name}: outside window")
    expect(audit["integrity"]["all_layers_covered_by_every_sample"], f"{name}: layer coverage")
    expect(audit["integrity"]["all_interfaces_covered_by_every_sample"], f"{name}: interface coverage")
    expect(audit["integrity"]["max_abs_energy_balance_relative_error"] < 1e-8, f"{name}: audit energy")
    expect(audit["integrity"]["max_abs_parallel_branch_relative_closure_error"] < 1e-8, f"{name}: audit branch")
    if name in {"balanced64", "final1024"}:
        per_bin = EXPECTED_BINS[name]
        expect(audit["temperature_bin_counts"] == {
            "30_42p5": per_bin, "42p5_55": per_bin, "55_67p5": per_bin,
            "67p5_80": per_bin, "outside": 0,
        }, f"{name}: realized temperature balance")
    return {
        "sample_count": count,
        "peak_deltaT_K": audit["summary"]["peak_deltaT_K"],
        "temperature_bin_counts": audit["temperature_bin_counts"],
        "manifest_sha256": audit["dataset_manifest_sha256"],
    }


def check_mesh_convergence() -> dict[str, Any]:
    payload = read_json(CONFIG_DIR / "v6_p1d_asymmetric_dual_robin16_mesh_convergence.json")
    rows = read_csv(CONFIG_DIR / "v6_p1d_asymmetric_dual_robin16_mesh_convergence.csv")
    expect(payload["passed"] is True, "mesh convergence")
    expect(len(rows) == 6, "mesh convergence row count")
    expect({row["mesh"] for row in rows} == {"coarse", "base", "fine"}, "mesh levels")
    return {"passed": True, "representatives": payload["representative_sample_ids"]}


def main() -> int:
    forbidden = {"jax", "flax", "optax", "rigno"}
    for script in (Path(generator.__file__), Path(core.__file__), Path(expansion.__file__)):
        expect(not (import_roots(script) & forbidden), f"model import in {script.name}")
    trial1 = read_json(CONFIG_DIR / "v6_p1d_asymmetric_dual_robin64_audit.json")
    expect(trial1["sample_count"] == trial1["window_hit_count"] == 64, "trial1 count/window")
    expect(trial1["temperature_bin_counts"] == {
        "30_42p5": 20, "42p5_55": 12, "55_67p5": 5,
        "67p5_80": 27, "outside": 0,
    }, "trial1 retained temperature distribution")
    result = {
        "status": "ok",
        "literature_search": check_literature_and_search(),
        "configs": {name: check_config(name, path) for name, path in CONFIGS.items()},
        "datasets": {name: check_dataset(name) for name in CONFIGS},
        "mesh_convergence": check_mesh_convergence(),
        "trial1_64_retained": True,
        "training_runs": 0,
        "model_inference_runs": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
