#!/usr/bin/env python3
"""Check the complete P1f pilot/final dataset contract and tracked artifacts."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import heat3d_v6_p1d_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
SCRIPTS = (
    ROOT / "scripts/prepare_heat3d_v6_p1f_unified_layered_config.py",
    ROOT / "scripts/generate_heat3d_v6_p1f_unified_layered_dataset.py",
    ROOT / "scripts/generate_heat3d_v6_p1e_deconfounded_dataset.py",
    ROOT / "scripts/audit_heat3d_v6_p1f_unified_layered_dataset.py",
)
ARTIFACTS = {
    "pilot": {
        "stem": "v6_p1f_temperature_shaping_pilot128",
        "config": CONFIG_DIR / "v6_p1f_temperature_shaping_pilot128.yaml",
        "dataset": ROOT / "data/heat3d_v6_p1f_temperature_shaping_pilot128_v0",
        "groups": 16, "samples": 128,
    },
    "final": {
        "stem": "v6_p1f_unified_layered1024",
        "config": CONFIG_DIR / "v6_p1f_unified_layered1024.yaml",
        "dataset": ROOT / "data/heat3d_v6_p1f_unified_layered1024_v0",
        "groups": 128, "samples": 1024,
    },
}
EXPECTED_GATE = {
    "peak_deltaT_below_30_count_max": 0,
    "peak_deltaT_30_80_fraction_min": 0.80,
    "peak_deltaT_above_100_fraction_max": 0.05,
    "peak_deltaT_above_120_count_max": 0,
}


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


def check_config(stage: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(Path(spec["config"]).read_text(encoding="utf-8"))
    expect(config["schema_version"] == "heat3d_v6_p1f_unified_layered_dataset_v1", f"{stage}: schema")
    expect(config["stage"] == stage and config["status"] == "frozen_before_generation", f"{stage}: lifecycle")
    expect(config["sample_count"] == len(config["cases"]) == spec["samples"], f"{stage}: cases")
    expect(config["geometry_group_count"] == len(config["geometry_groups"]) == spec["groups"], f"{stage}: groups")
    expect(config["physics"]["top_h_W_m2K"] == [1000.0, 1400.0], f"{stage}: top h")
    expect(config["physics"]["bottom_h_W_m2K"] == [20.0, 120.0], f"{stage}: bottom h")
    expect(config["factor_contract"]["package_power_W"] == [4.0, 6.0], f"{stage}: power")
    expect(config["factor_contract"]["temperature_gate"] == EXPECTED_GATE, f"{stage}: gate")
    expect(config["factor_contract"]["same_distribution_across_train_valid_test"] is True, f"{stage}: unified factors")
    expect(config["source_contract"]["same_generator_across_train_valid_test"] is True, f"{stage}: unified source generator")
    expect(config["physics"]["material_distribution"] == {
        "mode": "fixed_profile", "profile_id": "logic_package_complete_B_fixed_materials_v1",
        "identical_across_train_valid_test": True,
    }, f"{stage}: material distribution")
    for key in (
        "model_training", "model_inference", "peak_deltaT_filtering", "peak_deltaT_resampling",
        "sample_replacement", "per_sample_Rth_power_back_calculation", "post_solve_case_or_seed_selection",
    ):
        expect(config["scope"][key] is False, f"{stage}: forbidden {key}")
    groups = {group["group_id"]: group for group in config["geometry_groups"]}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in config["cases"]:
        grouped[case["group_id"]].append(case)
        expect(case["split_role"] == groups[case["group_id"]]["split_role"], f"{stage}: group split")
    expected_factorial = {
        (top, bottom, power)
        for top in (1000.0, 1400.0)
        for bottom in (20.0, 120.0)
        for power in (4.0, 6.0)
    }
    for group_id, cases in grouped.items():
        expect(len(cases) == 8, f"{stage}: {group_id} case count")
        expect({
            (float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]), float(case["package_total_power_W"]))
            for case in cases
        } == expected_factorial, f"{stage}: {group_id} factorial")
    if stage == "pilot":
        expect(set(group["split_role"] for group in groups.values()) == {"pilot_only"}, "pilot role")
    else:
        expect(set(group["split_role"] for group in groups.values()) == {"train", "valid", "test"}, "final roles")
        expect(not any("ood" in group["split_role"].lower() for group in groups.values()), "final OOD role")
        expect(Counter(group["split_role"] for group in groups.values()) == Counter({"train": 96, "valid": 16, "test": 16}), "final group counts")
        for role, count_per_value in (("train", 12), ("valid", 2), ("test", 2)):
            role_groups = [group for group in groups.values() if group["split_role"] == role]
            expect(Counter(int(group["source_count"]) for group in role_groups) == Counter({value: count_per_value for value in range(3, 11)}), f"{role}: source count distribution")
            expect(Counter(group["layout_kind"] for group in role_groups) == Counter({value: len(role_groups) // 4 for value in ("distributed", "clustered", "mixed", "edge_balanced")}), f"{role}: layout distribution")
    for group in groups.values():
        expect(len(group["sources"]) == int(group["source_count"]), f"{stage}: source payload")
        expect(math.isclose(sum(float(row["package_power_fraction"]) for row in group["sources"]), 1.0, abs_tol=1e-12), f"{stage}: power fractions")
        expect(group["material_profile_id"] == "logic_package_complete_B_fixed_materials_v1", f"{stage}: material profile")
    return {"seed": int(config["seed"]), "group_ids": set(groups), "sample_ids": {case["id"] for case in config["cases"]}}


def check_sample(dataset: Path, manifest_row: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    sample = dataset / manifest_row["sample_dir"]
    for name, digest in manifest_row["file_sha256"].items():
        expect((sample / name).is_file() and core.sha256(sample / name) == digest, f"{manifest_row['sample_id']}: {name} hash")
    arrays = {name: np.load(sample / name, allow_pickle=False) for name in (
        "coords.npy", "temperature.npy", "deltaT.npy", "k_field.npy", "q_field.npy",
        "layer_id.npy", "bc_features.npy", "bc_parameters.npy", "sampling_stratum.npy",
    )}
    expect(arrays["coords.npy"].shape == (1024, 3), f"{manifest_row['sample_id']}: coords")
    expect(arrays["temperature.npy"].shape == arrays["deltaT.npy"].shape == (1024, 1), f"{manifest_row['sample_id']}: labels")
    expect(arrays["k_field.npy"].shape == (1024, 3), f"{manifest_row['sample_id']}: k")
    expect(arrays["bc_parameters.npy"].shape == (1024, 4), f"{manifest_row['sample_id']}: BC")
    expect(all(np.all(np.isfinite(value)) for value in arrays.values()), f"{manifest_row['sample_id']}: finite")
    expect(np.max(np.abs(arrays["temperature.npy"] - 300.0 - arrays["deltaT.npy"])) < 1e-10, f"{manifest_row['sample_id']}: deltaT")
    expected_bc = np.asarray([case["top_h_W_m2K"], case["bottom_h_W_m2K"], 300.0, 300.0])
    expect(np.allclose(arrays["bc_parameters.npy"], expected_bc), f"{manifest_row['sample_id']}: BC values")
    meta = read_json(sample / "sample_meta.json")
    expect(meta["sample_id"] == manifest_row["sample_id"] == case["id"], f"{manifest_row['sample_id']}: ID")
    expect(meta["group_id"] == manifest_row["group_id"] and meta["split_role"] == manifest_row["split_role"], f"{manifest_row['sample_id']}: group role")
    projection = meta["operator_projection"]
    expect(projection["point_seed_key"] == meta["group_id"], f"{manifest_row['sample_id']}: point seed")
    expect(projection["point_coordinates_frozen_before_temperature_solve"] is True, f"{manifest_row['sample_id']}: point freeze")
    expect(projection["label_inputs_used_for_point_selection"] == [], f"{manifest_row['sample_id']}: label leak")
    expect(projection["coverage"]["all_layers_covered"] and projection["coverage"]["all_interfaces_covered"], f"{manifest_row['sample_id']}: coverage")
    expect(meta["solver_mesh"]["minimum_source_control_volume_count"] >= 128, f"{manifest_row['sample_id']}: source CV")
    expect(meta["solver_mesh"]["minimum_source_in_plane_interval_count"] >= 7, f"{manifest_row['sample_id']}: source intervals")
    expect(abs(float(meta["metrics"]["energy_balance_relative_error"])) <= 1e-8, f"{manifest_row['sample_id']}: energy")
    expect(meta["guardrails"]["training_runs"] == meta["guardrails"]["model_inference_runs"] == 0, f"{manifest_row['sample_id']}: model work")


def check_dataset(stage: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    stem = str(spec["stem"])
    config_path = Path(spec["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in config["cases"]}
    dataset = Path(spec["dataset"])
    tracked_manifest = read_json(CONFIG_DIR / f"{stem}_manifest.json")
    expect(tracked_manifest == read_json(dataset / "manifest.json"), f"{stage}: manifest mirror")
    expect(tracked_manifest["config_sha256"] == core.sha256(config_path), f"{stage}: config SHA")
    expect(tracked_manifest["sample_count"] == len(tracked_manifest["samples"]) == spec["samples"], f"{stage}: manifest count")
    groups: defaultdict[str, set[str]] = defaultdict(set)
    coordinates: defaultdict[str, set[str]] = defaultdict(set)
    for row in tracked_manifest["samples"]:
        check_sample(dataset, row, cases[row["sample_id"]])
        groups[row["group_id"]].add(row["split_role"])
        coordinates[row["group_id"]].add(row["point_coordinates_sha256"])
    expect(all(len(value) == 1 for value in groups.values()), f"{stage}: split leakage")
    expect(all(len(value) == 1 for value in coordinates.values()), f"{stage}: coordinate leakage")
    qualification = read_json(CONFIG_DIR / f"{stem}_qualification.json")
    expect(qualification["passed"] is True and qualification["gate"] == EXPECTED_GATE, f"{stage}: qualification")
    expect(all(qualification["gate_checks"].values()), f"{stage}: gate checks")
    expect(all(qualification["integrity"]["checks"].values()), f"{stage}: integrity checks")
    return {
        "sample_count": qualification["sample_count"],
        "geometry_group_count": qualification["geometry_group_count"],
        "gate_counts": qualification["gate_counts"],
        "gate_fractions": qualification["gate_fractions"],
        "peak_deltaT_K": qualification["peak_deltaT_K"],
    }


def main() -> int:
    forbidden = {"jax", "flax", "optax", "rigno"}
    for path in SCRIPTS:
        expect(not (import_roots(path) & forbidden), f"model import in {path.name}")
    p1d = CONFIG_DIR / "v6_p1d_asymmetric_dual_robin1024.yaml"
    p1e = CONFIG_DIR / "v6_p1e_deconfounded1024.yaml"
    expect(p1d.is_file() and core.sha256(p1d) == "58cff515dc6af27b2b262535101318c01069ff84788a9c45c17efd6339502fcc", "P1d provenance changed")
    expect(p1e.is_file() and core.sha256(p1e) == "8d1448005a2afb3267c891dfb5660cf5d6e2ea3e9ca6bce6abee755b3f1ae1e3", "P1e provenance changed")
    config_checks = {stage: check_config(stage, spec) for stage, spec in ARTIFACTS.items()}
    expect(config_checks["pilot"]["seed"] != config_checks["final"]["seed"], "pilot/final seed overlap")
    expect(not (config_checks["pilot"]["group_ids"] & config_checks["final"]["group_ids"]), "pilot/final group overlap")
    expect(not (config_checks["pilot"]["sample_ids"] & config_checks["final"]["sample_ids"]), "pilot/final sample overlap")
    result = {
        "status": "ok", "configs": {
            stage: {"seed": value["seed"], "group_count": len(value["group_ids"]), "sample_count": len(value["sample_ids"])}
            for stage, value in config_checks.items()
        },
        "datasets": {stage: check_dataset(stage, spec) for stage, spec in ARTIFACTS.items()},
        "P1d_P1e_provenance_retained": True,
        "pilot_samples_retained_in_final": False,
        "training_runs": 0, "model_inference_runs": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
