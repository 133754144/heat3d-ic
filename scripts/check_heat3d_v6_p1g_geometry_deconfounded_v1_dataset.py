#!/usr/bin/env python3
"""Check P1g-v1 config-only preflight or the complete generated qualification."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import check_heat3d_v6_p1g_geometry_deconfounded_dataset as v0check
import heat3d_v6_p1d_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
STEM = "v6_p1g_geometry_deconfounded1024_v1"
CONFIG = CONFIG_DIR / f"{STEM}.yaml"
DATASET = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v1"
PARENT_CONFIG = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024.yaml"
PARENT_MANIFEST = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024_manifest.json"
PARENT_QUALIFICATION = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024_qualification.json"
PARENT_CONFIG_SHA = "ab162724af61c745f82571e9c8f07102d5262c70a4817ace0900e894bfc4af83"
PARENT_MANIFEST_SHA = "e5329d5cd6253510d87a4432d5f2ddae67259637810c29fdfb6ddf42621875a4"
PARENT_QUALIFICATION_SHA = "31f33bfa535153981279293c55c74a0f19db6484a08f4fb50ce95b7cdaf1a141"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_invariant(source: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(source["layer"]), round(float(source["width_mm"]), 15),
        round(float(source["height_mm"]), 15), round(float(source["declared_area_mm2"]), 15),
        round(float(source["package_power_fraction"]), 15),
    )


def _center_mm(source: Mapping[str, Any]) -> tuple[float, float]:
    x0, x1, y0, y1 = map(float, source["bbox_fraction_xy"])
    return 5.0 * (x0 + x1), 5.0 * (y0 + y1)


def check_config() -> tuple[dict[str, Any], dict[str, Any]]:
    v0check.expect(core.sha256(PARENT_CONFIG) == PARENT_CONFIG_SHA, "P1g-v0 config changed")
    v0check.expect(core.sha256(PARENT_MANIFEST) == PARENT_MANIFEST_SHA, "P1g-v0 manifest changed")
    v0check.expect(core.sha256(PARENT_QUALIFICATION) == PARENT_QUALIFICATION_SHA, "P1g-v0 qualification changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parent = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    p1f = yaml.safe_load(v0check.P1F_CONFIG.read_text(encoding="utf-8"))
    v0check.expect(config["schema_version"] == "heat3d_v6_p1g_geometry_deconfounded_dataset_v2", "schema")
    v0check.expect(config["dataset_id"] == "heat3d_v6_p1g_geometry_deconfounded1024_v1", "dataset ID")
    v0check.expect(config["seed"] == parent["seed"], "seed changed")
    v0check.expect(config["sample_count"] == len(config["cases"]) == 1024, "sample count")
    v0check.expect(config["geometry_group_count"] == len(config["geometry_groups"]) == 128, "group count")
    v0check.check_scientific_contract(config, p1f)
    provenance = config["provenance"]
    v0check.expect(provenance["parent_P1g_v0_config_sha256"] == PARENT_CONFIG_SHA, "parent config provenance")
    v0check.expect(provenance["parent_P1g_v0_manifest_sha256"] == PARENT_MANIFEST_SHA, "parent manifest provenance")
    v0check.expect(provenance["parent_P1g_v0_qualification_sha256"] == PARENT_QUALIFICATION_SHA, "parent qualification provenance")
    v0check.expect(provenance["seed_changed_or_searched"] is False, "seed search")
    v0check.expect(provenance["per_sample_filtering_replacement_or_patch"] is False, "sample patch")
    parent_groups = {group["group_id"]: group for group in parent["geometry_groups"]}
    directions: Counter[tuple[float, float]] = Counter()
    displacement: dict[str, list[float]] = {"offset": [], "partly_aligned": []}
    triplets: Counter[tuple[Any, ...]] = Counter()
    for group in config["geometry_groups"]:
        parent_group = parent_groups[group["parent_p1g_v0_group_id"]]
        for key in (
            "split_role", "geometry_seed", "split_ordinal", "assignment_table_version",
            "assignment_shuffle_seed", "pre_shuffle_assignment_id", "role_shuffle_position",
            "material_profile_id", "layout_kind", "alignment_relation", "source_count",
            "upper_layer_power_fraction", "total_source_area_mm2",
            "maximum_preregistered_source_surface_power_density_W_m2",
        ):
            v0check.expect(group[key] == parent_group[key], f"{group['group_id']}: changed {key}")
        v0check.expect(group["projection_seed_key"] == parent_group["group_id"], f"{group['group_id']}: projection seed key")
        v0check.expect(
            sorted(_source_invariant(source) for source in group["sources"])
            == sorted(_source_invariant(source) for source in parent_group["sources"]),
            f"{group['group_id']}: source size/area/power changed",
        )
        lower_by_slot = {
            int(source["slot_index"]): source for source in group["sources"] if source["layer"] == "silicon_die_lower"
        }
        for source in group["sources"]:
            reference = source.get("alignment_reference_lower_slot")
            if source["layer"] != "silicon_die_upper" or reference is None:
                continue
            lower_x, lower_y = _center_mm(lower_by_slot[int(reference)])
            upper_x, upper_y = _center_mm(source)
            displacement[group["alignment_relation"]].append(math.hypot(upper_x - lower_x, upper_y - lower_y))
        transform = group["alignment_transform"]
        v0check.expect(transform["label_or_temperature_inputs"] == [], f"{group['group_id']}: transform label leak")
        if group["alignment_relation"] == "offset":
            directions[tuple(map(float, transform["upper_translation_mm"]))] += 1
        triplets[(int(group["source_count"]), group["layout_kind"], group["alignment_relation"])] += 1
    v0check.expect(len(triplets) == 64 and set(triplets.values()) == {2}, "factor triplet balance")
    v0check.expect(all(abs(value) <= 1e-12 for value in displacement["partly_aligned"]), "aligned displacement")
    v0check.expect(all(abs(value - 0.3125) <= 1e-12 for value in displacement["offset"]), "offset displacement")
    v0check.expect(len(directions) >= 4, "offset direction diversity")
    for role, expected_count in (("train", 96), ("valid", 16), ("test", 16)):
        scoped = [group for group in config["geometry_groups"] if group["split_role"] == role]
        v0check.expect(len(scoped) == expected_count, f"{role}: group count")
        v0check.expect(Counter(int(group["source_count"]) for group in scoped) == Counter({value: expected_count // 8 for value in range(3, 11)}), f"{role}: source margin")
        v0check.expect(Counter(group["layout_kind"] for group in scoped) == Counter({value: expected_count // 4 for value in ("distributed", "clustered", "mixed", "edge_balanced")}), f"{role}: layout margin")
        v0check.expect(Counter(group["alignment_relation"] for group in scoped) == Counter({"offset": expected_count // 2, "partly_aligned": expected_count // 2}), f"{role}: alignment margin")
    expected_factorial = {
        (top, bottom, power) for top in (1000.0, 1400.0) for bottom in (20.0, 120.0) for power in (4.0, 6.0)
    }
    cases_by_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in config["cases"]:
        cases_by_group[case["group_id"]].append(case)
    for group_id, cases in cases_by_group.items():
        v0check.expect(len(cases) == 8, f"{group_id}: case count")
        v0check.expect({
            (float(case["top_h_W_m2K"]), float(case["bottom_h_W_m2K"]), float(case["package_total_power_W"]))
            for case in cases
        } == expected_factorial, f"{group_id}: factorial")
    return config, parent


def check_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = v0check.read_json(CONFIG_DIR / f"{STEM}_manifest.json")
    v0check.expect(manifest == v0check.read_json(DATASET / "manifest.json"), "manifest mirror")
    v0check.expect(manifest["config_sha256"] == core.sha256(CONFIG), "config SHA")
    v0check.expect(manifest["sample_count"] == len(manifest["samples"]) == 1024, "manifest count")
    cases = {case["id"]: case for case in config["cases"]}
    groups = {group["group_id"]: group for group in config["geometry_groups"]}
    group_roles: defaultdict[str, set[str]] = defaultdict(set)
    group_coordinates: defaultdict[str, set[str]] = defaultdict(set)
    peaks: list[float] = []
    max_energy_error = 0.0
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        sample = DATASET / row["sample_dir"]
        for name, digest in row["file_sha256"].items():
            v0check.expect((sample / name).is_file() and core.sha256(sample / name) == digest, f"{sample_id}: {name} hash")
        coords = np.load(sample / "coords.npy", mmap_mode="r", allow_pickle=False)
        temperature = np.load(sample / "temperature.npy", mmap_mode="r", allow_pickle=False)
        v0check.expect(coords.shape == (1024, 3) and temperature.shape == (1024, 1), f"{sample_id}: array shape")
        v0check.expect(np.all(np.isfinite(coords)) and np.all(np.isfinite(temperature)), f"{sample_id}: finite arrays")
        meta = v0check.read_json(sample / "sample_meta.json")
        case = cases[sample_id]
        group = groups[case["group_id"]]
        projection = meta["operator_projection"]
        reconstruction = projection["full_field_reconstruction"]
        v0check.expect(projection["point_seed_key"] == group["projection_seed_key"], f"{sample_id}: point seed key")
        v0check.expect(projection["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_id}: point freeze")
        v0check.expect(projection["label_inputs_used_for_point_selection"] == [], f"{sample_id}: point label leak")
        v0check.expect(projection["coverage"]["all_layers_covered"] and projection["coverage"]["all_interfaces_covered"], f"{sample_id}: coverage")
        v0check.expect(len(reconstruction["per_layer"]) == len(core.load_p1c_stack()), f"{sample_id}: per-layer diagnostics")
        numbers = [
            reconstruction["full_field_cv_rmse_K"], reconstruction["full_field_cv_relative_rmse"],
            reconstruction["full_field_max_abs_error_K"], reconstruction["max_abs_layer_average_error_K"],
            reconstruction["max_abs_layer_drop_error_K"],
        ]
        v0check.expect(np.all(np.isfinite(np.asarray(numbers))), f"{sample_id}: projection metrics")
        peaks.append(float(meta["metrics"]["peak_deltaT_K"]))
        max_energy_error = max(max_energy_error, abs(float(meta["metrics"]["energy_balance_relative_error"])))
        group_roles[row["group_id"]].add(row["split_role"])
        group_coordinates[row["group_id"]].add(row["point_coordinates_sha256"])
    v0check.expect(all(len(values) == 1 for values in group_roles.values()), "group split leakage")
    v0check.expect(all(len(values) == 1 for values in group_coordinates.values()), "group coordinate mismatch")
    v0check.expect(max_energy_error <= 1e-8, "energy conservation")
    audit = v0check.read_json(CONFIG_DIR / f"{STEM}_geometry_audit.json")
    qualification = v0check.read_json(CONFIG_DIR / f"{STEM}_qualification.json")
    v0check.check_assignments(config, audit)
    v0check.expect(audit["split_margins_identical"] is True, "split margins")
    for scope in ("all", "train"):
        for association in audit["association"][scope].values():
            v0check.expect(abs(float(association["cramers_v"])) <= 1e-12, f"{scope}: Cramer's V")
            v0check.expect(abs(float(association["mutual_information_nats"])) <= 1e-12, f"{scope}: MI")
    v0check.expect(all(audit["alignment_geometry_semantics"]["semantic_checks"].values()), "alignment semantics")
    parent_diff = audit["P1g_v1_parent_scientific_diff"]
    for key in (
        "same_seed", "same_physics", "same_BC_power_temperature_gate", "same_source_guardrails",
        "same_operator_projection_contract", "same_split_contract",
    ):
        v0check.expect(parent_diff[key] is True, f"parent diff: {key}")
    v0check.expect(parent_diff["groups_with_exact_factor_assignment"] == 128, "parent factor assignment")
    v0check.expect(parent_diff["groups_with_exact_source_size_area_power"] == 128, "parent source invariants")
    v0check.expect(parent_diff["changed_lower_source_bbox_count"] == 0, "lower geometry changed")
    v0check.expect(parent_diff["changed_upper_source_bbox_count"] == 384, "alignment transform coverage")
    v0check.expect(parent_diff["label_or_temperature_inputs"] == [], "parent diff label leak")
    v0check.expect(audit["near_duplicate_geometry"]["near_duplicate_pair_count"] == 0, "near duplicate geometry")
    v0check.expect(audit["coverage"]["all_samples_cover_all_layers"] and audit["coverage"]["all_samples_cover_all_interfaces"], "coverage audit")
    v0check.expect(audit["temperature_gate_passed"] is True and all(audit["temperature_gate_checks"].values()), "temperature gate")
    v0check.expect(qualification["integrity_passed"] is True, "integrity qualification")
    v0check.expect(qualification["formal_training_qualified"] is True, "formal training qualification")
    v0check.expect(qualification["decision"] == "qualified_for_formal_training", "qualification decision")
    v0check.expect(len(_read_csv(CONFIG_DIR / f"{STEM}_projection_diagnostics.csv")) == 1024, "projection CSV")
    v0check.expect(len(_read_csv(CONFIG_DIR / f"{STEM}_layer_projection_errors.csv")) == 1024 * len(core.load_p1c_stack()), "layer CSV")
    return {
        "peak_deltaT_K_min": min(peaks), "peak_deltaT_K_max": max(peaks),
        "window_30_80_count": sum(30.0 <= value <= 80.0 for value in peaks),
        "max_energy_balance_relative_error": max_energy_error,
        "dataset_manifest_sha256": core.sha256(DATASET / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    config, _ = check_config()
    result: dict[str, Any] = {
        "status": "config_only_ok" if args.config_only else "ok",
        "dataset_id": config["dataset_id"], "geometry_group_count": 128, "sample_count": 1024,
        "seed_changed_or_searched": False, "training_runs": 0, "model_inference_runs": 0,
    }
    if not args.config_only:
        result.update(check_dataset(config))
        result["formal_training_qualified"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
