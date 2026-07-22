#!/usr/bin/env python3
"""Check the frozen P1g config, generated dataset, audits, and P1f provenance."""

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
STEM = "v6_p1g_geometry_deconfounded1024"
CONFIG = CONFIG_DIR / f"{STEM}.yaml"
DATASET = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v0"
P1F_CONFIG = CONFIG_DIR / "v6_p1f_unified_layered1024.yaml"
P1F_MANIFEST = CONFIG_DIR / "v6_p1f_unified_layered1024_manifest.json"
P1F_CONFIG_SHA = "6b05c889760675954300428066f3ff6a12109725073bf0edb336fc9eb04e0fda"
P1F_MANIFEST_SHA = "fd311b9b8c19b1f578f2cbc7c8322826766d22bfc75b5067820799abd34c2e03"
SCRIPTS = (
    ROOT / "scripts/prepare_heat3d_v6_p1g_geometry_deconfounded_config.py",
    ROOT / "scripts/generate_heat3d_v6_p1g_geometry_deconfounded_dataset.py",
    ROOT / "scripts/generate_heat3d_v6_p1e_deconfounded_dataset.py",
    ROOT / "scripts/audit_heat3d_v6_p1g_geometry_deconfounding.py",
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expect(isinstance(payload, dict), f"{path}: JSON object expected")
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


def check_scientific_contract(config: Mapping[str, Any], p1f: Mapping[str, Any]) -> None:
    expect(config["physics"] == p1f["physics"], "P1g physics differs from P1f")
    expect(config["factor_contract"] == p1f["factor_contract"], "P1g BC/power/gate differs from P1f")
    expect(config["source_contract"] == p1f["source_contract"], "P1g source guardrails differ from P1f")
    expect(config["split_contract"] == p1f["split_contract"], "P1g split differs from P1f")
    p1g_projection = dict(config["operator_projection"])
    diagnostic = p1g_projection.pop("full_field_reconstruction_diagnostic")
    expect(p1g_projection == p1f["operator_projection"], "P1g point projection contract differs from P1f")
    expect(diagnostic["purpose"] == "representation_QC_only_not_model_inference", "projection diagnostic scope")
    for key in (
        "model_training", "model_inference", "peak_deltaT_filtering", "peak_deltaT_resampling",
        "sample_replacement", "per_sample_Rth_power_back_calculation", "post_solve_case_or_seed_selection",
    ):
        expect(config["scope"][key] is False, f"forbidden scope {key}")


def check_assignments(config: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    groups = config["geometry_groups"]
    expect(len(groups) == 128, "geometry group count")
    expect(Counter(group["split_role"] for group in groups) == Counter({"train": 96, "valid": 16, "test": 16}), "split group counts")
    triplets = Counter((int(group["source_count"]), group["layout_kind"], group["alignment_relation"]) for group in groups)
    expect(len(triplets) == 64 and set(triplets.values()) == {2}, "global factor triplet replication")
    for role, groups_expected in (("train", 96), ("valid", 16), ("test", 16)):
        scoped = [group for group in groups if group["split_role"] == role]
        expect(len(scoped) == groups_expected, f"{role}: count")
        expect(Counter(int(group["source_count"]) for group in scoped) == Counter({value: groups_expected // 8 for value in range(3, 11)}), f"{role}: source-count margin")
        expect(Counter(group["layout_kind"] for group in scoped) == Counter({value: groups_expected // 4 for value in ("distributed", "clustered", "mixed", "edge_balanced")}), f"{role}: layout margin")
        expect(Counter(group["alignment_relation"] for group in scoped) == Counter({"offset": groups_expected // 2, "partly_aligned": groups_expected // 2}), f"{role}: alignment margin")
        for count in range(3, 11):
            subset = [group for group in scoped if int(group["source_count"]) == count]
            expect(len({group["layout_kind"] for group in subset}) >= 2, f"{role}: count {count} layout support")
            expect(len({group["alignment_relation"] for group in subset}) == 2, f"{role}: count {count} alignment support")
    # Explicitly reject the three old ordinal formulas as a joint assignment.
    old_formula_matches = []
    for group in groups:
        ordinal = int(group["split_ordinal"])
        old_formula_matches.append(
            int(group["source_count"]) == 3 + ordinal % 8
            and group["layout_kind"] == ("distributed", "clustered", "mixed", "edge_balanced")[ordinal % 4]
            and group["alignment_relation"] == ("partly_aligned" if (ordinal // 2) % 2 == 0 else "offset")
        )
        expect(group["assignment_table_version"] == "p1g_balanced_joint_v1", "assignment version")
        expect("pre_shuffle_assignment_id" in group and "assignment_shuffle_seed" in group, "assignment provenance")
    expect(not all(old_formula_matches), "P1f ordinal mapping survived")
    expect(audit["split_margins_identical"] is True, "audit split margins")
    expect(all(audit["alignment_geometry_semantics"]["semantic_checks"].values()), "alignment geometry semantics")
    for scope in ("all", "train"):
        for association in audit["association"][scope].values():
            expect(abs(float(association["cramers_v"])) <= 1e-12, f"{scope}: Cramer's V")
            expect(abs(float(association["mutual_information_nats"])) <= 1e-12, f"{scope}: MI")
    for scope in ("valid", "test"):
        expect(abs(float(audit["association"][scope]["source_count__alignment_relation"]["cramers_v"])) <= 1e-12, f"{scope}: count/alignment")
        expect(abs(float(audit["association"][scope]["layout_kind__alignment_relation"]["cramers_v"])) <= 1e-12, f"{scope}: layout/alignment")
    valid_v = float(audit["association"]["valid"]["source_count__layout_kind"]["cramers_v"])
    test_v = float(audit["association"]["test"]["source_count__layout_kind"]["cramers_v"])
    expect(math.isclose(valid_v, test_v, rel_tol=0.0, abs_tol=1e-12), "valid/test structural association mismatch")


def check_dataset(config: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    tracked_manifest = read_json(CONFIG_DIR / f"{STEM}_manifest.json")
    expect(tracked_manifest == read_json(DATASET / "manifest.json"), "manifest mirror")
    expect(tracked_manifest["config_sha256"] == core.sha256(CONFIG), "manifest config hash")
    expect(tracked_manifest["sample_count"] == len(tracked_manifest["samples"]) == 1024, "manifest count")
    cases = {case["id"]: case for case in config["cases"]}
    group_roles: defaultdict[str, set[str]] = defaultdict(set)
    group_coordinates: defaultdict[str, set[str]] = defaultdict(set)
    max_energy_error = 0.0
    full_field_rmse: list[float] = []
    peaks: list[float] = []
    for row in tracked_manifest["samples"]:
        sample_id = row["sample_id"]
        sample = DATASET / row["sample_dir"]
        expect(sample_id in cases, f"{sample_id}: case")
        for name, digest in row["file_sha256"].items():
            expect((sample / name).is_file() and core.sha256(sample / name) == digest, f"{sample_id}: {name} hash")
        coords = np.load(sample / "coords.npy", mmap_mode="r", allow_pickle=False)
        temperature = np.load(sample / "temperature.npy", mmap_mode="r", allow_pickle=False)
        expect(coords.shape == (1024, 3) and temperature.shape == (1024, 1), f"{sample_id}: array shape")
        expect(np.all(np.isfinite(coords)) and np.all(np.isfinite(temperature)), f"{sample_id}: finite arrays")
        meta = read_json(sample / "sample_meta.json")
        projection = meta["operator_projection"]
        coverage = projection["coverage"]
        reconstruction = projection["full_field_reconstruction"]
        expect(projection["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_id}: point freeze")
        expect(projection["label_inputs_used_for_point_selection"] == [], f"{sample_id}: label leakage")
        expect(coverage["all_layers_covered"] and coverage["all_interfaces_covered"], f"{sample_id}: coverage")
        expect(len(coverage["layer_point_counts"]) == len(core.load_p1c_stack()), f"{sample_id}: layer counts")
        expect(reconstruction["purpose"] == "representation_QC_only_not_model_inference", f"{sample_id}: reconstruction scope")
        expect(len(reconstruction["per_layer"]) == len(coverage["layer_point_counts"]), f"{sample_id}: per-layer diagnostics")
        numeric = [
            reconstruction["full_field_cv_rmse_K"], reconstruction["full_field_cv_relative_rmse"],
            reconstruction["full_field_max_abs_error_K"], reconstruction["max_abs_layer_average_error_K"],
            reconstruction["max_abs_layer_drop_error_K"],
            *[layer["layer_average_signed_error_K"] for layer in reconstruction["per_layer"]],
            *[layer["layer_drop_signed_error_K"] for layer in reconstruction["per_layer"]],
        ]
        expect(np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))), f"{sample_id}: reconstruction finite")
        full_field_rmse.append(float(reconstruction["full_field_cv_rmse_K"]))
        peaks.append(float(meta["metrics"]["peak_deltaT_K"]))
        max_energy_error = max(max_energy_error, abs(float(meta["metrics"]["energy_balance_relative_error"])))
        group_roles[row["group_id"]].add(row["split_role"])
        group_coordinates[row["group_id"]].add(row["point_coordinates_sha256"])
    expect(all(len(value) == 1 for value in group_roles.values()), "group split leakage")
    expect(all(len(value) == 1 for value in group_coordinates.values()), "group coordinate mismatch")
    expect(max_energy_error <= 1e-8, "energy conservation")
    gate_counts = {
        "below_30": sum(value < 30.0 for value in peaks),
        "in_30_80": sum(30.0 <= value <= 80.0 for value in peaks),
        "above_100": sum(value > 100.0 for value in peaks),
        "above_120": sum(value > 120.0 for value in peaks),
    }
    expect(gate_counts == audit["temperature_gate_counts"], "temperature gate recount")
    expect(audit["temperature_gate_passed"] is False, "P1g-v0 failure state changed")
    expect(audit["temperature_gate_checks"] == {
        "below_30": False, "in_30_80": True, "above_100": True, "above_120": True,
    }, "temperature gate failure attribution")
    expect(audit["near_duplicate_geometry"]["exact_signature_duplicate_count"] == 0, "exact duplicate geometry")
    expect(audit["near_duplicate_geometry"]["near_duplicate_pair_count"] == 0, "near duplicate geometry")
    expect(audit["coverage"]["all_samples_cover_all_layers"] and audit["coverage"]["all_samples_cover_all_interfaces"], "audit coverage")
    return {
        "max_energy_balance_relative_error": max_energy_error,
        "full_field_cv_rmse_K_median": float(np.median(full_field_rmse)),
        "temperature_gate_counts": gate_counts,
    }


def main() -> int:
    forbidden = {"jax", "flax", "optax", "rigno"}
    for path in SCRIPTS:
        expect(not (import_roots(path) & forbidden), f"model import in {path.name}")
    expect(core.sha256(P1F_CONFIG) == P1F_CONFIG_SHA, "P1f config changed")
    expect(core.sha256(P1F_MANIFEST) == P1F_MANIFEST_SHA, "P1f manifest changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p1f = yaml.safe_load(P1F_CONFIG.read_text(encoding="utf-8"))
    expect(config["schema_version"] == "heat3d_v6_p1g_geometry_deconfounded_dataset_v1", "schema")
    expect(config["sample_count"] == len(config["cases"]) == 1024, "sample count")
    expect(config["geometry_group_count"] == len(config["geometry_groups"]) == 128, "group count")
    expect(config["provenance"]["p1f_config_sha256"] == P1F_CONFIG_SHA, "P1f config provenance")
    expect(config["provenance"]["p1f_manifest_sha256"] == P1F_MANIFEST_SHA, "P1f manifest provenance")
    check_scientific_contract(config, p1f)
    expected_factorial = {
        (top, bottom, power) for top in (1000.0, 1400.0) for bottom in (20.0, 120.0) for power in (4.0, 6.0)
    }
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in config["cases"]:
        grouped[case["group_id"]].append(case)
    for group_id, cases in grouped.items():
        expect(len(cases) == 8, f"{group_id}: case count")
        expect({(float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]), float(case["package_total_power_W"])) for case in cases} == expected_factorial, f"{group_id}: factorial")
    audit = read_json(CONFIG_DIR / f"{STEM}_geometry_audit.json")
    qualification = read_json(CONFIG_DIR / f"{STEM}_qualification.json")
    check_assignments(config, audit)
    dataset_summary = check_dataset(config, audit)
    expect(qualification["integrity_passed"] is True, "integrity qualification")
    expect(qualification["formal_training_qualified"] is False, "formal training qualification")
    expect(qualification["decision"] == "rejected_for_formal_training_failed_frozen_temperature_gate", "qualification decision")
    expect(read_csv(CONFIG_DIR / f"{STEM}_joint_contingency.csv"), "contingency CSV empty")
    expect(len(read_csv(CONFIG_DIR / f"{STEM}_projection_diagnostics.csv")) == 1024, "projection CSV count")
    expect(len(read_csv(CONFIG_DIR / f"{STEM}_layer_projection_errors.csv")) > 1024, "layer CSV count")
    result = {
        "status": "ok_integrity_but_not_training_qualified", "dataset_id": config["dataset_id"], "sample_count": 1024,
        "geometry_group_count": 128, "split_margins_identical": True,
        "temperature_gate_passed": False, "formal_training_qualified": False, "P1f_v0_immutable": True,
        **dataset_summary, "training_runs": 0, "model_inference_runs": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
