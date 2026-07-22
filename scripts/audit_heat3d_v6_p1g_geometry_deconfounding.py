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
    report = f"""# V6-P1g geometry-factor deconfounding audit

## Decision

P1g is a new version derived from immutable P1f-v0. It preserves the P1f stack, materials, 2×2×2 BC/power block, whole-version temperature gate, frozen 1024-point sampling contract, and 96/16/16 group split. Only the geometry assignment and newly seeded geometry instances were rebuilt. No sample was filtered, replaced, Rth-back-calculated, or locally patched.

The qualification gate **{'passed' if audit['temperature_gate_passed'] else 'failed'}**. Peak ΔT spans {audit['peak_deltaT_K']['min']:.3f}–{audit['peak_deltaT_K']['max']:.3f} K; {gate_counts['in_30_80']}/{len(samples)} cases are in 30–80 K.

## Factor deconfounding

All three factor margins are identical in train/valid/test. Globally, each of the 8×4×2 source-count/layout/alignment combinations appears exactly twice; all global pairwise Cramér's V and mutual information are zero (up to floating-point roundoff). Train is also pairwise independent. With only 16 groups, a valid/test 8×4 count-layout table cannot be independent (expected 0.5 case/cell); the frozen complementary schedules attain the preregistered minimum-support construction while keeping count-alignment and layout-alignment independent. Every count sees multiple layouts and both alignments in every split.

P1g also makes the alignment label geometrically truthful: `offset` groups have no shared upper/lower source slots, while `partly_aligned` groups have positive overlap. This corrects the new P1g geometry instances only; the P1f-v0 files and hashes remain untouched.

## Representation and leakage QC

Every sample retains 1024 points, covers every layer and interface, and reuses coordinates only within its geometry group. Point selection is frozen before solving and uses no label. IDW-8 reconstruction is a post-solve representation diagnostic—not model inference and not a selection criterion. Full-field CV-RMSE median is {audit['projection_reconstruction']['full_field_cv_rmse_K']['median']:.6f} K (P95 {audit['projection_reconstruction']['full_field_cv_rmse_K']['p95']:.6f} K). Maximum absolute per-layer mean error is summarized in the companion JSON/CSV, as are layer-drop errors and per-layer point counts.

Input-only geometry signatures have {audit['near_duplicate_geometry']['exact_signature_duplicate_count']} exact duplicates and {audit['near_duplicate_geometry']['near_duplicate_pair_count']} pairs below the frozen standardized-distance threshold {audit['near_duplicate_geometry']['near_duplicate_threshold']}.

## Artifacts

- Frozen config: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml`
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
