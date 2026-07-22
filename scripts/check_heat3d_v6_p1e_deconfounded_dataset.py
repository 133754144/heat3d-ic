#!/usr/bin/env python3
"""Validate P1e configs, literature, orthogonal audits, datasets, and guardrails."""

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
LITERATURE = ROOT / "docs/v6_p1e_literature_matrix.csv"
SCRIPTS = (
    ROOT / "scripts/prepare_heat3d_v6_p1e_deconfounded_config.py",
    ROOT / "scripts/generate_heat3d_v6_p1e_deconfounded_dataset.py",
    ROOT / "scripts/audit_heat3d_v6_p1e_deconfounding.py",
    ROOT / "scripts/audit_heat3d_v6_p1e_p1d_baseline.py",
)
DATASETS = {
    "v6_p1e_deconfounded_paired128": ROOT / "data/heat3d_v6_p1e_deconfounded_paired128_v0",
    "v6_p1e_deconfounded1024": ROOT / "data/heat3d_v6_p1e_deconfounded1024_v0",
}
CONFIGS = {
    "v6_p1e_deconfounded_paired128": CONFIG_DIR / "v6_p1e_deconfounded_paired128.yaml",
    "v6_p1e_deconfounded1024": CONFIG_DIR / "v6_p1e_deconfounded1024.yaml",
}
EXPECTED = {"v6_p1e_deconfounded_paired128": 128, "v6_p1e_deconfounded1024": 1024}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expect(isinstance(payload, dict), f"{path}: JSON object expected")
    return payload


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


def check_literature() -> dict[str, Any]:
    rows = read_csv(LITERATURE)
    expect(len(rows) >= 9, "P1e literature matrix too short")
    expect(sum(row["uniform_h_numeric_use"] == "allowed_endpoints" for row in rows) >= 1, "no direct asymmetric h evidence")
    text = " ".join(row["boundary_evidence"] + " " + row["p1e_support_scope"] for row in rows).lower()
    for word in ("pcb", "chassis", "natural", "forced"):
        expect(word in text, f"missing literature scope: {word}")
    mass = next((row for row in rows if row["id"] == "P1E-L09"), None)
    expect(mass is not None, "MASS-HBM restriction missing")
    expect(mass["uniform_h_numeric_use"] == "forbidden", "MASS-HBM used as scalar h")
    expect("architecture relevance" in mass["p1e_support_scope"].lower(), "MASS-HBM scope too broad")
    return {"rows": len(rows), "mass_hbm_uniform_h_forbidden": True}


def check_config(stem: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIGS[stem].read_text(encoding="utf-8"))
    count = EXPECTED[stem]
    expect(config["schema_version"] == "heat3d_v6_p1e_deconfounded_dataset_v1", f"{stem}: schema")
    expect(config["sample_count"] == len(config["cases"]) == count, f"{stem}: count")
    for key in (
        "model_training", "model_inference", "peak_deltaT_filtering", "peak_deltaT_resampling",
        "sample_replacement", "per_sample_Rth_power_back_calculation", "post_solve_factor_or_seed_selection",
    ):
        expect(config["scope"][key] is False, f"{stem}: forbidden scope {key}")
    expect(config["factor_contract"]["common_package_power_levels_W_for_every_BC_family"] == [2.0, 6.0, 10.0, 14.0], f"{stem}: common powers")
    expect(config["factor_contract"]["temperature_window_role"] == "report_only_not_filter_or_replacement_rule", f"{stem}: window role")
    groups = {group["group_id"]: group for group in config["geometry_groups"]}
    by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in config["cases"]:
        by_group[case["group_id"]].append(case)
        expect(case["split_role"] == groups[case["group_id"]]["split_role"], f"{stem}: group split")
    expect(all(len({row["split_role"] for row in cases}) == 1 for cases in by_group.values()), f"{stem}: split leakage")
    complete = [case for case in config["cases"] if case["design_block"] == "complete_factorial"]
    expect(len(complete) >= 128, f"{stem}: paired cases")
    for group_id in sorted({case["group_id"] for case in complete}):
        rows = [case for case in complete if case["group_id"] == group_id]
        expect(len(rows) == 64, f"{stem}: incomplete factorial {group_id}")
        expect({float(row["package_total_power_W"]) for row in rows} == {2.0, 6.0, 10.0, 14.0}, f"{stem}: power coverage")
    source_counts = {int(group["source_count"]) for group in groups.values()}
    expect(len(source_counts) > 1, f"{stem}: source count fixed")
    for group in groups.values():
        fractions = [float(row["package_power_fraction"]) for row in group["sources"]]
        expect(len(fractions) == int(group["source_count"]), f"{stem}: source count payload")
        expect(math.isclose(sum(fractions), 1.0, abs_tol=1e-12), f"{stem}: source power fractions")
        expect(float(group["max_preregistered_source_surface_power_density_W_m2"]) <= 1.5e6, f"{stem}: density")
    if count == 1024:
        roles = Counter(case["split_role"] for case in config["cases"])
        expect(roles == Counter({
            "train": 640, "valid_iid": 128, "test_iid": 128,
            "layout_ood": 32, "source_count_ood": 32, "power_density_ood": 32, "bc_ood": 32,
        }), f"{stem}: role counts {roles}")
        expect(source_counts.issuperset({2, 12}), f"{stem}: source-count OOD")
    return {"sample_count": count, "geometry_groups": len(groups), "source_counts": sorted(source_counts)}


def check_sample(dataset: Path, row: Mapping[str, Any], config_case: Mapping[str, Any]) -> None:
    sample = dataset / row["sample_dir"]
    for name, digest in row["file_sha256"].items():
        expect((sample / name).is_file() and core.sha256(sample / name) == digest, f"{row['sample_id']}: hash {name}")
    arrays = {name: np.load(sample / name, allow_pickle=False) for name in (
        "coords.npy", "temperature.npy", "deltaT.npy", "k_field.npy", "q_field.npy",
        "layer_id.npy", "bc_features.npy", "bc_parameters.npy", "sampling_stratum.npy",
    )}
    expect(arrays["coords.npy"].shape == (1024, 3), f"{row['sample_id']}: coords")
    expect(arrays["temperature.npy"].shape == arrays["deltaT.npy"].shape == (1024, 1), f"{row['sample_id']}: labels")
    expect(arrays["k_field.npy"].shape == (1024, 3), f"{row['sample_id']}: k")
    expect(arrays["bc_parameters.npy"].shape == (1024, 4), f"{row['sample_id']}: BC")
    expect(all(np.all(np.isfinite(value)) for value in arrays.values()), f"{row['sample_id']}: finite")
    expect(np.max(np.abs(arrays["temperature.npy"] - 300.0 - arrays["deltaT.npy"])) < 1e-10, f"{row['sample_id']}: deltaT")
    expected_bc = np.asarray([config_case["top_h_W_m2K"], config_case["bottom_h_W_m2K"], 300.0, 300.0])
    expect(np.allclose(arrays["bc_parameters.npy"], expected_bc), f"{row['sample_id']}: BC values")
    meta = read_json(sample / "sample_meta.json")
    expect(meta["sample_id"] == row["sample_id"] == config_case["id"], f"{row['sample_id']}: identity")
    expect(meta["group_id"] == row["group_id"] and meta["split_role"] == row["split_role"], f"{row['sample_id']}: group/role")
    expect(meta["operator_projection"]["point_seed_key"] == meta["group_id"], f"{row['sample_id']}: point seed")
    expect(meta["operator_projection"]["label_inputs_used_for_point_selection"] == [], f"{row['sample_id']}: label leak")
    expect(meta["operator_projection"]["coverage"]["all_layers_covered"], f"{row['sample_id']}: layers")
    expect(meta["operator_projection"]["coverage"]["all_interfaces_covered"], f"{row['sample_id']}: interfaces")
    expect(meta["solver_mesh"]["minimum_source_control_volume_count"] >= 128, f"{row['sample_id']}: source CV")
    expect(meta["solver_mesh"]["minimum_source_in_plane_interval_count"] >= 7, f"{row['sample_id']}: source intervals")
    expect(abs(float(meta["metrics"]["energy_balance_relative_error"])) < 1e-8, f"{row['sample_id']}: energy")
    expect(meta["guardrails"]["training_runs"] == meta["guardrails"]["model_inference_runs"] == 0, f"{row['sample_id']}: model work")


def check_dataset(stem: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIGS[stem].read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in config["cases"]}
    dataset = DATASETS[stem]
    manifest = read_json(CONFIG_DIR / f"{stem}_manifest.json")
    expect(manifest == read_json(dataset / "manifest.json"), f"{stem}: tracked manifest")
    expect(manifest["config_sha256"] == core.sha256(CONFIGS[stem]), f"{stem}: config SHA")
    expect(manifest["sample_count"] == len(manifest["samples"]) == EXPECTED[stem], f"{stem}: manifest count")
    for row in manifest["samples"]:
        check_sample(dataset, row, cases[row["sample_id"]])
    groups: defaultdict[str, set[str]] = defaultdict(set)
    point_hash: defaultdict[str, set[str]] = defaultdict(set)
    for row in manifest["samples"]:
        groups[row["group_id"]].add(row["split_role"])
        point_hash[row["group_id"]].add(row["point_coordinates_sha256"])
    expect(all(len(value) == 1 for value in groups.values()), f"{stem}: manifest split leakage")
    expect(all(len(value) == 1 for value in point_hash.values()), f"{stem}: coordinates not group-frozen")
    audit = read_json(CONFIG_DIR / f"{stem}_audit.json")
    ortho = read_json(CONFIG_DIR / f"{stem}_orthogonal_audit.json")
    expect(audit["sample_count"] == EXPECTED[stem] and ortho["passed"] is True, f"{stem}: audits")
    expect(audit["guardrails"]["training_runs"] == audit["guardrails"]["model_inference_runs"] == 0, f"{stem}: audit model work")
    return {
        "sample_count": audit["sample_count"], "window_hit_count": audit["window_hit_count"],
        "peak_deltaT_K": audit["summary"]["peak_deltaT_K"],
        "BC_power_max_abs_pearson": max(abs(ortho["pearson_correlation"][0][2]), abs(ortho["pearson_correlation"][1][2])),
    }


def main() -> int:
    forbidden = {"jax", "flax", "optax", "rigno"}
    for path in SCRIPTS:
        expect(not (import_roots(path) & forbidden), f"model import in {path.name}")
    baseline = read_json(CONFIG_DIR / "v6_p1e_p1d_baseline_deconfounding_audit.json")
    expect(baseline["decision"]["selected_policy"] == "rebuild_new_p1e1024_keep_p1d_as_provenance", "baseline decision")
    result = {
        "status": "ok", "literature": check_literature(),
        "configs": {stem: check_config(stem) for stem in CONFIGS},
        "datasets": {stem: check_dataset(stem) for stem in DATASETS},
        "p1d_retained_as_provenance": True, "training_runs": 0, "model_inference_runs": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
