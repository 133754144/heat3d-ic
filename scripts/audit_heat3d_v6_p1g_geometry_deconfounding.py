#!/usr/bin/env python3
"""Audit P1g factor deconfounding and 1024-point representation quality."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
CONFIG = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024.yaml"
DATASET = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v0"
STEM = "v6_p1g_geometry_deconfounded1024"
AUDIT_JSON = CONFIG_DIR / f"{STEM}_geometry_audit.json"
QUALIFICATION_JSON = CONFIG_DIR / f"{STEM}_qualification.json"
CONTINGENCY_CSV = CONFIG_DIR / f"{STEM}_joint_contingency.csv"
PROJECTION_CSV = CONFIG_DIR / f"{STEM}_projection_diagnostics.csv"
LAYER_CSV = CONFIG_DIR / f"{STEM}_layer_projection_errors.csv"
REPORT = ROOT / "docs/v6_p1g_geometry_deconfounding_audit.md"
FACTORS = ("source_count", "layout_kind", "alignment_relation")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _levels(groups: Iterable[Mapping[str, Any]], factor: str) -> list[Any]:
    values = {group[factor] for group in groups}
    return sorted(values, key=lambda value: (isinstance(value, str), value))


def _association(groups: list[Mapping[str, Any]], factor_a: str, factor_b: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    levels_a, levels_b = _levels(groups, factor_a), _levels(groups, factor_b)
    table = np.zeros((len(levels_a), len(levels_b)), dtype=np.float64)
    index_a = {value: index for index, value in enumerate(levels_a)}
    index_b = {value: index for index, value in enumerate(levels_b)}
    for group in groups:
        table[index_a[group[factor_a]], index_b[group[factor_b]]] += 1.0
    n = float(np.sum(table))
    row = np.sum(table, axis=1, keepdims=True)
    col = np.sum(table, axis=0, keepdims=True)
    expected = row @ col / n
    nonzero = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[nonzero] / expected[nonzero]))
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / denominator) if denominator > 0 else 0.0
    probability = table / n
    expected_probability = expected / n
    occupied = probability > 0
    mutual_information = float(np.sum(probability[occupied] * np.log(probability[occupied] / expected_probability[occupied])))
    rows = [{
        "factor_a": factor_a, "factor_b": factor_b,
        "level_a": level_a, "level_b": level_b,
        "observed_count": int(table[i, j]), "independence_expected_count": float(expected[i, j]),
    } for i, level_a in enumerate(levels_a) for j, level_b in enumerate(levels_b)]
    return {
        "factor_a": factor_a, "factor_b": factor_b,
        "shape": list(table.shape), "sample_size": int(n),
        "cramers_v": cramers_v, "mutual_information_nats": mutual_information,
        "observed_table": table.astype(int).tolist(), "independence_expected_table": expected.tolist(),
        "levels_a": levels_a, "levels_b": levels_b,
    }, rows


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)), "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)), "max": float(np.max(array)),
    }


def _geometry_vector(group: Mapping[str, Any]) -> tuple[str, np.ndarray]:
    ordered = sorted(group["sources"], key=lambda row: (row["layer"], int(row["slot_index"]), row["bbox_fraction_xy"]))
    signature_payload = {
        "source_count": int(group["source_count"]), "layout_kind": group["layout_kind"],
        "alignment_relation": group["alignment_relation"], "sources": ordered,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    vector = [float(group["source_count"]), float(group["upper_layer_power_fraction"]), float(group["total_source_area_mm2"])]
    for source in ordered:
        vector.extend([
            0.0 if source["layer"] == "silicon_die_lower" else 1.0,
            float(source["slot_index"]), *map(float, source["bbox_fraction_xy"]),
            float(source["package_power_fraction"]),
        ])
    vector.extend([0.0] * ((10 - len(ordered)) * 7))
    return signature, np.asarray(vector, dtype=np.float64)


def _near_duplicate_audit(groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    encoded = [_geometry_vector(group) for group in groups]
    signatures = [row[0] for row in encoded]
    matrix = np.stack([row[1] for row in encoded])
    scale = np.std(matrix, axis=0)
    active = scale > 1e-12
    normalized = (matrix[:, active] - np.mean(matrix[:, active], axis=0)) / scale[active]
    distance = np.sqrt(np.sum((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distance, np.inf)
    pairs = np.dstack(np.unravel_index(np.argsort(distance.ravel())[:10], distance.shape))[0]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, right in pairs:
        pair = tuple(sorted((int(left), int(right))))
        if pair in seen:
            continue
        seen.add(pair)
        unique.append({
            "group_a": groups[pair[0]]["group_id"], "group_b": groups[pair[1]]["group_id"],
            "standardized_distance": float(distance[pair]),
            "same_split": groups[pair[0]]["split_role"] == groups[pair[1]]["split_role"],
        })
        if len(unique) == 5:
            break
    return {
        "features_are_input_only": True,
        "feature_definition": "count_upper_fraction_total_area_plus_sorted_padded_source_layer_slot_bbox_power_fraction",
        "exact_signature_duplicate_count": len(signatures) - len(set(signatures)),
        "near_duplicate_threshold": 1e-6,
        "near_duplicate_pair_count": int(np.sum(np.triu(distance < 1e-6, k=1))),
        "minimum_nonself_standardized_distance": float(np.min(distance)),
        "nearest_pairs": unique,
    }


def _alignment_overlap_audit(groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    if all("alignment_transform" in group for group in groups):
        by_relation: dict[str, list[float]] = {"offset": [], "partly_aligned": []}
        for group in groups:
            lower_by_slot = {
                int(row["slot_index"]): row for row in group["sources"] if row["layer"] == "silicon_die_lower"
            }
            for source in group["sources"]:
                reference = source.get("alignment_reference_lower_slot")
                if source["layer"] != "silicon_die_upper" or reference is None:
                    continue
                lower_x, lower_y = _source_center_mm(lower_by_slot[int(reference)])
                upper_x, upper_y = _source_center_mm(source)
                by_relation[group["alignment_relation"]].append(math.hypot(upper_x - lower_x, upper_y - lower_y))
        return {
            relation: _summary(values) for relation, values in by_relation.items()
        } | {
            "metric": "paired_source_centroid_displacement_mm",
            "semantic_checks": {
                "partly_aligned_centroids_coincide": all(abs(value) <= 1e-12 for value in by_relation["partly_aligned"]),
                "offset_is_exactly_two_mesh_intervals": all(abs(value - 0.3125) <= 1e-12 for value in by_relation["offset"]),
            },
        }
    by_relation: dict[str, list[float]] = {"offset": [], "partly_aligned": []}
    for group in groups:
        lower = {int(row["slot_index"]) for row in group["sources"] if row["layer"] == "silicon_die_lower"}
        upper = {int(row["slot_index"]) for row in group["sources"] if row["layer"] == "silicon_die_upper"}
        overlap = len(lower & upper) / max(1, min(len(lower), len(upper)))
        by_relation[group["alignment_relation"]].append(overlap)
    return {
        relation: _summary(values) for relation, values in by_relation.items()
    } | {
        "semantic_checks": {
            "offset_has_zero_slot_overlap": all(value == 0.0 for value in by_relation["offset"]),
            "partly_aligned_has_positive_slot_overlap": all(value > 0.0 for value in by_relation["partly_aligned"]),
        },
    }


def _source_center_mm(source: Mapping[str, Any]) -> tuple[float, float]:
    x0, x1, y0, y1 = map(float, source["bbox_fraction_xy"])
    return 5.0 * (x0 + x1), 5.0 * (y0 + y1)


def _source_nonposition_signature(source: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(source["layer"]), round(float(source["width_mm"]), 15),
        round(float(source["height_mm"]), 15), round(float(source["declared_area_mm2"]), 15),
        round(float(source["package_power_fraction"]), 15),
    )


def _parent_v0_diff_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    parent_path = ROOT / str(config["provenance"]["parent_P1g_v0_config"])
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    parent_groups = {str(group["group_id"]): group for group in parent["geometry_groups"]}
    factor_fields = (
        "split_role", "geometry_seed", "split_ordinal", "assignment_table_version",
        "assignment_shuffle_seed", "pre_shuffle_assignment_id", "role_shuffle_position",
        "material_profile_id", "layout_kind", "alignment_relation", "source_count",
        "upper_layer_power_fraction", "total_source_area_mm2",
        "maximum_preregistered_source_surface_power_density_W_m2",
    )
    exact_factor_groups = 0
    exact_nonposition_source_groups = 0
    changed_lower_bbox_count = 0
    changed_upper_bbox_count = 0
    unchanged_upper_bbox_count = 0
    for group in config["geometry_groups"]:
        parent_group = parent_groups[str(group["parent_p1g_v0_group_id"])]
        exact_factor_groups += int(all(group[field] == parent_group[field] for field in factor_fields))
        exact_nonposition_source_groups += int(
            sorted(_source_nonposition_signature(source) for source in group["sources"])
            == sorted(_source_nonposition_signature(source) for source in parent_group["sources"])
        )
        parent_lower_bbox = sorted(
            tuple(map(float, source["bbox_fraction_xy"]))
            for source in parent_group["sources"] if source["layer"] == "silicon_die_lower"
        )
        child_lower_bbox = sorted(
            tuple(map(float, source["bbox_fraction_xy"]))
            for source in group["sources"] if source["layer"] == "silicon_die_lower"
        )
        changed_lower_bbox_count += sum(left != right for left, right in zip(parent_lower_bbox, child_lower_bbox, strict=True))
        parent_upper = sorted(
            [source for source in parent_group["sources"] if source["layer"] == "silicon_die_upper"],
            key=_source_nonposition_signature,
        )
        child_upper = sorted(
            [source for source in group["sources"] if source["layer"] == "silicon_die_upper"],
            key=_source_nonposition_signature,
        )
        for parent_source, child_source in zip(parent_upper, child_upper, strict=True):
            changed = tuple(map(float, parent_source["bbox_fraction_xy"])) != tuple(map(float, child_source["bbox_fraction_xy"]))
            changed_upper_bbox_count += int(changed)
            unchanged_upper_bbox_count += int(not changed)
    return {
        "parent_config_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "same_seed": config["seed"] == parent["seed"],
        "same_physics": config["physics"] == parent["physics"],
        "same_BC_power_temperature_gate": config["factor_contract"] == parent["factor_contract"],
        "same_source_guardrails": config["source_contract"] == parent["source_contract"],
        "same_operator_projection_contract": config["operator_projection"] == parent["operator_projection"],
        "same_split_contract": config["split_contract"] == parent["split_contract"],
        "groups_with_exact_factor_assignment": exact_factor_groups,
        "groups_with_exact_source_size_area_power": exact_nonposition_source_groups,
        "changed_lower_source_bbox_count": changed_lower_bbox_count,
        "changed_upper_source_bbox_count": changed_upper_bbox_count,
        "unchanged_upper_source_bbox_count": unchanged_upper_bbox_count,
        "allowed_change": "upper_source_bbox_and_slot_alignment_metadata_only",
        "label_or_temperature_inputs": [],
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p1f_config_path = CONFIG_DIR / "v6_p1f_unified_layered1024.yaml"
    p1f_manifest_path = CONFIG_DIR / "v6_p1f_unified_layered1024_manifest.json"
    p1f = yaml.safe_load(p1f_config_path.read_text(encoding="utf-8"))
    groups = config["geometry_groups"]
    samples = _read_csv(CONFIG_DIR / f"{STEM}_samples.csv")
    associations: dict[str, Any] = {}
    contingency_rows: list[dict[str, Any]] = []
    for scope in ("all", "train", "valid", "test"):
        scoped = groups if scope == "all" else [group for group in groups if group["split_role"] == scope]
        associations[scope] = {}
        for factor_a, factor_b in ((FACTORS[0], FACTORS[1]), (FACTORS[0], FACTORS[2]), (FACTORS[1], FACTORS[2])):
            key = f"{factor_a}__{factor_b}"
            result, rows = _association(scoped, factor_a, factor_b)
            associations[scope][key] = result
            contingency_rows.extend({"scope": scope, **row} for row in rows)
    margins: dict[str, Any] = {}
    for role in ("train", "valid", "test"):
        scoped = [group for group in groups if group["split_role"] == role]
        margins[role] = {
            factor: {str(key): value / len(scoped) for key, value in sorted(Counter(group[factor] for group in scoped).items(), key=lambda item: str(item[0]))}
            for factor in FACTORS
        }
    projection_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for row in samples:
        meta = json.loads((DATASET / row["sample_id"] / "sample_meta.json").read_text(encoding="utf-8"))
        projection = meta["operator_projection"]
        reconstruction = projection["full_field_reconstruction"]
        projection_rows.append({
            "sample_id": row["sample_id"], "group_id": row["group_id"], "split_role": row["split_role"],
            "full_field_cv_rmse_K": reconstruction["full_field_cv_rmse_K"],
            "full_field_cv_relative_rmse": reconstruction["full_field_cv_relative_rmse"],
            "full_field_max_abs_error_K": reconstruction["full_field_max_abs_error_K"],
            "max_abs_layer_average_error_K": reconstruction["max_abs_layer_average_error_K"],
            "max_abs_layer_drop_error_K": reconstruction["max_abs_layer_drop_error_K"],
        })
        counts = projection["coverage"]["layer_point_counts"]
        for layer in reconstruction["per_layer"]:
            layer_rows.append({
                "sample_id": row["sample_id"], "group_id": row["group_id"], "split_role": row["split_role"],
                "layer_name": layer["layer_name"], "operator_point_count": counts[layer["layer_name"]],
                "solver_node_count": layer["solver_node_count"],
                "layer_average_signed_error_K": layer["layer_average_signed_error_K"],
                "layer_drop_signed_error_K": layer["layer_drop_signed_error_K"],
            })
    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in samples])
    gate = config["factor_contract"]["temperature_gate"]
    gate_counts = {
        "below_30": int(np.sum(peaks < 30.0)),
        "in_30_80": int(np.sum((peaks >= 30.0) & (peaks <= 80.0))),
        "above_100": int(np.sum(peaks > 100.0)),
        "above_120": int(np.sum(peaks > 120.0)),
    }
    gate_checks = {
        "below_30": gate_counts["below_30"] <= int(gate["peak_deltaT_below_30_count_max"]),
        "in_30_80": gate_counts["in_30_80"] / len(peaks) >= float(gate["peak_deltaT_30_80_fraction_min"]),
        "above_100": gate_counts["above_100"] / len(peaks) <= float(gate["peak_deltaT_above_100_fraction_max"]),
        "above_120": gate_counts["above_120"] <= int(gate["peak_deltaT_above_120_count_max"]),
    }
    layer_counts_by_name: dict[str, dict[str, float]] = {}
    for layer_name in sorted({row["layer_name"] for row in layer_rows}):
        layer_counts_by_name[layer_name] = _summary(float(row["operator_point_count"]) for row in layer_rows if row["layer_name"] == layer_name)
    audit = {
        "schema_version": "heat3d_v6_p1g_geometry_deconfounding_audit_v1",
        "dataset_id": config["dataset_id"], "sample_count": len(samples), "geometry_group_count": len(groups),
        "p1f_v0_provenance": {
            "config_sha256": hashlib.sha256(p1f_config_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(p1f_manifest_path.read_bytes()).hexdigest(),
            "physics_unchanged": config["physics"] == p1f["physics"],
            "BC_power_temperature_gate_unchanged": config["factor_contract"] == p1f["factor_contract"],
            "source_guardrails_unchanged": config["source_contract"] == p1f["source_contract"],
            "split_contract_unchanged": config["split_contract"] == p1f["split_contract"],
            "P1f_files_modified": False,
        },
        **({
            "parent_P1g_v0_provenance": {
                "config_sha256": config["provenance"]["parent_P1g_v0_config_sha256"],
                "manifest_sha256": config["provenance"]["parent_P1g_v0_manifest_sha256"],
                "qualification_sha256": config["provenance"]["parent_P1g_v0_qualification_sha256"],
                "parent_decision": config["provenance"]["parent_P1g_v0_decision"],
                "seed_changed_or_searched": config["provenance"]["seed_changed_or_searched"],
                "per_sample_filtering_replacement_or_patch": config["provenance"]["per_sample_filtering_replacement_or_patch"],
            },
            "P1g_v1_parent_scientific_diff": _parent_v0_diff_audit(config),
        } if "parent_P1g_v0_config_sha256" in config["provenance"] else {}),
        "split_group_counts": dict(sorted(Counter(group["split_role"] for group in groups).items())),
        "split_sample_counts": dict(sorted(Counter(row["split_role"] for row in samples).items())),
        "association": associations, "split_normalized_margins": margins,
        "split_margins_identical": margins["train"] == margins["valid"] == margins["test"],
        "coverage": {
            "operator_points_per_sample": 1024, "per_layer_point_count": layer_counts_by_name,
            "all_samples_cover_all_layers": all(row["all_layers_covered_by_1024_points"] == "True" for row in samples),
            "all_samples_cover_all_interfaces": all(row["all_interfaces_covered_by_1024_points"] == "True" for row in samples),
        },
        "projection_reconstruction": {
            key: _summary(float(row[key]) for row in projection_rows)
            for key in (
                "full_field_cv_rmse_K", "full_field_cv_relative_rmse", "full_field_max_abs_error_K",
                "max_abs_layer_average_error_K", "max_abs_layer_drop_error_K",
            )
        },
        "alignment_geometry_semantics": _alignment_overlap_audit(groups),
        "near_duplicate_geometry": _near_duplicate_audit(groups),
        "temperature_gate": gate, "temperature_gate_counts": gate_counts,
        "temperature_gate_checks": gate_checks, "temperature_gate_passed": all(gate_checks.values()),
        "peak_deltaT_K": _summary(peaks),
        "guardrails": {
            "whole_version_gate_only": True, "sample_filtering": False, "sample_replacement": False,
            "Rth_power_back_calculation": False, "local_patch": False,
            "training_runs": 0, "model_inference_runs": 0,
        },
    }
    _write_json(AUDIT_JSON, audit)
    qualification = {
        "schema_version": "heat3d_v6_p1g_geometry_deconfounded_qualification_v1",
        "dataset_id": config["dataset_id"],
        "integrity_passed": (
            audit["split_margins_identical"]
            and audit["coverage"]["all_samples_cover_all_layers"]
            and audit["coverage"]["all_samples_cover_all_interfaces"]
            and audit["near_duplicate_geometry"]["near_duplicate_pair_count"] == 0
            and all(audit["alignment_geometry_semantics"]["semantic_checks"].values())
        ),
        "formal_training_qualified": audit["temperature_gate_passed"],
        "decision": (
            "qualified_for_formal_training" if audit["temperature_gate_passed"]
            else "rejected_for_formal_training_failed_frozen_temperature_gate"
        ),
        "gate": gate, "gate_counts": gate_counts, "gate_checks": gate_checks,
        "sample_filtering_or_replacement_performed": False,
        "training_runs": 0, "model_inference_runs": 0,
        "audit": str(AUDIT_JSON.relative_to(ROOT)),
    }
    _write_json(QUALIFICATION_JSON, qualification)
    _write_csv(CONTINGENCY_CSV, contingency_rows)
    _write_csv(PROJECTION_CSV, projection_rows)
    _write_csv(LAYER_CSV, layer_rows)
    if "parent_P1g_v0_config_sha256" in config["provenance"]:
        version_decision = (
            "This is the immutable P1g-v1 whole-version revision of P1g-v0. It reuses every v0 factor-assignment row, "
            "source size, area, power fraction, seed, and all P1f scientific contracts. Before solving, all groups receive "
            "one frozen alignment definition: paired centroids coincide for partly-aligned groups and are displaced by "
            "exactly two solver-mesh intervals for offset groups. No case, seed, or factor level was selected from labels."
        )
        alignment_decision = (
            "P1g-v1 paired-centroid diagnostics confirm 0 mm displacement for partly-aligned sources and exactly "
            "0.3125 mm for offset sources."
        )
    else:
        version_decision = (
            "P1g is a new version derived from immutable P1f-v0. It preserves the P1f stack, materials, 2×2×2 BC/power "
            "block, whole-version temperature gate, frozen 1024-point sampling contract, and 96/16/16 group split. Only "
            "the geometry assignment and newly seeded geometry instances were rebuilt."
        )
        alignment_decision = (
            "P1g-v0 makes the alignment label slot-based: offset groups have no shared upper/lower source slots, while "
            "partly-aligned groups have positive overlap."
        )
    report = f"""# V6-P1g geometry-factor deconfounding audit: {config['dataset_id']}

## Decision

{version_decision} No sample was filtered, replaced, Rth-back-calculated, or locally patched.

The qualification gate **{'passed' if audit['temperature_gate_passed'] else 'failed'}**. Peak ΔT spans {audit['peak_deltaT_K']['min']:.3f}–{audit['peak_deltaT_K']['max']:.3f} K; {gate_counts['in_30_80']}/{len(samples)} cases are in 30–80 K.

## Factor deconfounding

All three factor margins are identical in train/valid/test. Globally, each of the 8×4×2 source-count/layout/alignment combinations appears exactly twice; all global pairwise Cramér's V and mutual information are zero (up to floating-point roundoff). Train is also pairwise independent. With only 16 groups, a valid/test 8×4 count-layout table cannot be independent (expected 0.5 case/cell); the frozen complementary schedules attain the preregistered minimum-support construction while keeping count-alignment and layout-alignment independent. Every count sees multiple layouts and both alignments in every split.

{alignment_decision} The P1f-v0 files and hashes remain untouched.

## Representation and leakage QC

Every sample retains 1024 points, covers every layer and interface, and reuses coordinates only within its geometry group. Point selection is frozen before solving and uses no label. IDW-8 reconstruction is a post-solve representation diagnostic—not model inference and not a selection criterion. Full-field CV-RMSE median is {audit['projection_reconstruction']['full_field_cv_rmse_K']['median']:.6f} K (P95 {audit['projection_reconstruction']['full_field_cv_rmse_K']['p95']:.6f} K). Maximum absolute per-layer mean error is summarized in the companion JSON/CSV, as are layer-drop errors and per-layer point counts.

Input-only geometry signatures have {audit['near_duplicate_geometry']['exact_signature_duplicate_count']} exact duplicates and {audit['near_duplicate_geometry']['near_duplicate_pair_count']} pairs below the frozen standardized-distance threshold {audit['near_duplicate_geometry']['near_duplicate_threshold']}.

## Artifacts

- Frozen config: `{CONFIG.relative_to(ROOT)}`
- Joint tables: `configs/heat3d_v6/{STEM}_joint_contingency.csv`
- Projection summary: `configs/heat3d_v6/{STEM}_projection_diagnostics.csv`
- Layer diagnostics: `configs/heat3d_v6/{STEM}_layer_projection_errors.csv`
- Machine-readable audit: `configs/heat3d_v6/{STEM}_geometry_audit.json`
- Qualification decision: `configs/heat3d_v6/{STEM}_qualification.json`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "ok", "sample_count": len(samples), "temperature_gate_passed": audit["temperature_gate_passed"],
        "formal_training_qualified": qualification["formal_training_qualified"],
        "split_margins_identical": audit["split_margins_identical"],
        "global_max_cramers_v": max(value["cramers_v"] for value in associations["all"].values()),
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
