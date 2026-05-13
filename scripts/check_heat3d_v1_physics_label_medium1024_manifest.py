#!/usr/bin/env python3
"""Check the planned or Gap-A Heat3D v1 medium1024 manifest.

This checker validates manifest structure only. It does not require generated
samples and does not write data. The medium1024 manifests are research
benchmark-candidate designs, not completed benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium1024_manifest.json"
EXPECTED_TOTAL_COUNT = 1024
EXPECTED_SPLIT_COUNTS = {
    "train": 768,
    "valid": 128,
    "test_id": 64,
    "test_ood_bc_candidate": 24,
    "test_ood_stack_candidate": 24,
    "test_ood_combined_candidate": 16,
}
COUNT_SECTIONS = (
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
)
DIAG3_MIN_FRACTION = 0.25
DIAG3_MAX_FRACTION = 0.375


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the planned Heat3D v1 medium1024 manifest structure. "
            "No generated data are required."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sum_counts(counts: dict[str, Any]) -> int:
    total = 0
    for key, value in counts.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"count for {key!r} must be a non-negative integer, found {value!r}")
        total += value
    return total


def _section_counts(manifest: dict[str, Any], section: str) -> dict[str, int]:
    planned = manifest.get("coverage_summary_planned", {})
    counts = planned.get(section)
    if not isinstance(counts, dict):
        raise ValueError(f"coverage_summary_planned.{section} must be an object")
    return counts


def _target_values(manifest: dict[str, Any], section: str) -> set[str]:
    targets = manifest.get("coverage_targets", {}).get(section, [])
    if not isinstance(targets, list):
        raise ValueError(f"coverage_targets.{section} must be a list")
    return {str(item) for item in targets}


def _planned_only_values(manifest: dict[str, Any], section: str) -> set[str]:
    planned = manifest.get("implementation_status", {}).get("planned_only_not_yet_consumed_by_generator", {})
    values = planned.get(section, [])
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values}


def _implemented_values(manifest: dict[str, Any], section: str) -> set[str]:
    implemented = manifest.get("implementation_status", {}).get("implemented_now", {})
    values = implemented.get(section, [])
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values}


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_name = str(manifest.get("dataset_name", ""))
    manifest_kind = str(manifest.get("manifest_kind", ""))
    is_gap_a = dataset_name.endswith("_medium1024_gapA") or manifest_kind == "generation_ready_candidate_manifest"
    split_counts = manifest.get("split_counts", {})
    if not isinstance(split_counts, dict):
        errors.append("split_counts must be an object")
        split_counts = {}
    try:
        split_total = _sum_counts(split_counts)
    except ValueError as exc:
        errors.append(str(exc))
        split_total = 0
    if split_total != EXPECTED_TOTAL_COUNT:
        errors.append(f"split_counts must sum to {EXPECTED_TOTAL_COUNT}, found {split_total}")
    if split_counts != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split_counts must be {EXPECTED_SPLIT_COUNTS}, found {split_counts}")

    coverage_totals: dict[str, int] = {}
    for section in COUNT_SECTIONS:
        try:
            counts = _section_counts(manifest, section)
            coverage_totals[section] = _sum_counts(counts)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if coverage_totals[section] != EXPECTED_TOTAL_COUNT:
            errors.append(
                f"coverage_summary_planned.{section} must sum to {EXPECTED_TOTAL_COUNT}, "
                f"found {coverage_totals[section]}"
            )
        target_values = _target_values(manifest, section)
        count_values = set(counts)
        missing_counts = target_values - count_values
        extra_counts = count_values - target_values
        if missing_counts:
            errors.append(f"coverage_summary_planned.{section} missing targets: {sorted(missing_counts)}")
        if extra_counts:
            errors.append(f"coverage_summary_planned.{section} has extra keys: {sorted(extra_counts)}")

    k_field_counts = manifest.get("coverage_summary_planned", {}).get("k_field_mode", {})
    diag3_count = int(k_field_counts.get("diag3", 0)) if isinstance(k_field_counts, dict) else 0
    diag3_fraction = diag3_count / EXPECTED_TOTAL_COUNT
    if not DIAG3_MIN_FRACTION <= diag3_fraction <= DIAG3_MAX_FRACTION:
        errors.append(
            f"diag3 fraction must be in [{DIAG3_MIN_FRACTION}, {DIAG3_MAX_FRACTION}], "
            f"found {diag3_fraction:.6f}"
        )

    policy = manifest.get("candidate_split_policy", {})
    held_out_bc = set(policy.get("held_out_bc_categories", []))
    held_out_bc_allowed = set(policy.get("held_out_bc_allowed_splits", []))
    held_out_stack = set(policy.get("held_out_stack_templates", []))
    held_out_stack_allowed = set(policy.get("held_out_stack_allowed_splits", []))
    if held_out_bc_allowed != {"test_ood_bc_candidate", "test_ood_combined_candidate"}:
        errors.append(f"held_out_bc_allowed_splits unexpected: {sorted(held_out_bc_allowed)}")
    if held_out_stack_allowed != {"test_ood_stack_candidate", "test_ood_combined_candidate"}:
        errors.append(f"held_out_stack_allowed_splits unexpected: {sorted(held_out_stack_allowed)}")

    split_plan = manifest.get("split_composition_plan", {})
    if not isinstance(split_plan, dict):
        errors.append("split_composition_plan must be an object")
        split_plan = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        plan = split_plan.get(split)
        if not isinstance(plan, dict):
            errors.append(f"split_composition_plan.{split} missing")
            continue
        if plan.get("sample_count") != expected_count:
            errors.append(
                f"split_composition_plan.{split}.sample_count must be {expected_count}, "
                f"found {plan.get('sample_count')}"
            )
    held_out_bc_split_count = sum(
        int((split_plan.get(split, {}) or {}).get("held_out_bc_count", 0))
        for split in ("test_ood_bc_candidate", "test_ood_combined_candidate")
    )
    held_out_stack_split_count = sum(
        int((split_plan.get(split, {}) or {}).get("held_out_stack_count", 0))
        for split in ("test_ood_stack_candidate", "test_ood_combined_candidate")
    )
    bc_counts = manifest.get("coverage_summary_planned", {}).get("bc_category", {})
    stack_counts = manifest.get("coverage_summary_planned", {}).get("stack_template", {})
    held_out_bc_count = sum(int(bc_counts.get(item, 0)) for item in held_out_bc) if isinstance(bc_counts, dict) else 0
    held_out_stack_count = (
        sum(int(stack_counts.get(item, 0)) for item in held_out_stack) if isinstance(stack_counts, dict) else 0
    )
    if held_out_bc_count != held_out_bc_split_count:
        errors.append(
            f"held-out BC count {held_out_bc_count} must match allowed split count {held_out_bc_split_count}"
        )
    if held_out_stack_count != held_out_stack_split_count:
        errors.append(
            f"held-out stack count {held_out_stack_count} must match allowed split count {held_out_stack_split_count}"
        )

    if (split_plan.get("test_ood_bc_candidate", {}) or {}).get("held_out_stack_count", 0) != 0:
        errors.append("test_ood_bc_candidate must not contain held-out stack")
    if (split_plan.get("test_ood_stack_candidate", {}) or {}).get("held_out_bc_count", 0) != 0:
        errors.append("test_ood_stack_candidate must not contain held-out BC")
    combined = split_plan.get("test_ood_combined_candidate", {}) or {}
    if combined.get("held_out_bc_count") != combined.get("sample_count"):
        errors.append("test_ood_combined_candidate must contain held-out BC for every sample")
    if combined.get("held_out_stack_count") != combined.get("sample_count"):
        errors.append("test_ood_combined_candidate must contain held-out stack for every sample")

    for section in ("source_pattern_tag", "k_region_mode", "stack_template", "bc_category"):
        planned_only = _planned_only_values(manifest, section)
        implemented = _implemented_values(manifest, section)
        targets = _target_values(manifest, section)
        overlap = planned_only & implemented
        if overlap:
            errors.append(f"implementation status overlap in {section}: {sorted(overlap)}")
        if is_gap_a and planned_only:
            errors.append(f"Gap-A manifest must not contain planned-only values in {section}: {sorted(planned_only)}")
        if is_gap_a:
            not_implemented = targets - implemented
            if not_implemented:
                errors.append(f"Gap-A coverage targets not marked implemented in {section}: {sorted(not_implemented)}")
        if planned_only:
            warnings.append(f"{section} planned-only values are not yet generator-consumed: {sorted(planned_only)}")

    samples = manifest.get("samples", [])
    if samples and not is_gap_a:
        warnings.append("medium1024 manifest includes samples; checker treats this as a plan-only manifest")
    if not is_gap_a and manifest.get("full_generation_ready") is True:
        errors.append("medium1024 manifest must not mark full_generation_ready=true in this planning stage")
    if is_gap_a:
        if manifest.get("full_generation_ready") is not True:
            errors.append("Gap-A manifest must mark full_generation_ready=true")
        generation_plan = manifest.get("sample_generation_plan", {})
        if not isinstance(generation_plan, dict):
            errors.append("Gap-A manifest must include sample_generation_plan")
        elif generation_plan.get("strategy") != "gapA_deterministic_balanced_cycle":
            errors.append(
                "Gap-A sample_generation_plan.strategy must be gapA_deterministic_balanced_cycle"
            )
        if samples:
            warnings.append("Gap-A manifest includes explicit samples; deterministic plan is preferred")

    summary = {
        "split_total": split_total,
        "coverage_totals": coverage_totals,
        "diag3_count": diag3_count,
        "diag3_fraction": diag3_fraction,
        "held_out_bc_count": held_out_bc_count,
        "held_out_stack_count": held_out_stack_count,
        "is_gap_a": is_gap_a,
        "warnings": warnings,
    }
    return errors, summary


def main() -> int:
    args = parse_args()
    manifest = _read_json(args.manifest)
    errors, summary = validate_manifest(manifest)

    print("Heat3D v1 medium1024 manifest checker")
    print(f"manifest: {args.manifest}")
    print("scope: manifest dry-run only; no data generation; not a formal benchmark")
    print(f"dataset_name: {manifest.get('dataset_name')}")
    print(f"status: {manifest.get('status')}")
    print(f"split_counts: {manifest.get('split_counts')}")
    print(f"split_total: {summary['split_total']}")
    print(f"coverage_totals: {summary['coverage_totals']}")
    print(f"diag3_count: {summary['diag3_count']}")
    print(f"diag3_fraction: {summary['diag3_fraction']:.6f}")
    print(f"held_out_bc_count: {summary['held_out_bc_count']}")
    print(f"held_out_stack_count: {summary['held_out_stack_count']}")
    print(f"warnings: {summary['warnings']}")
    print(f"errors: {errors}")
    print("medium1024_manifest_ok:", not errors)
    if summary["is_gap_a"]:
        print("medium1024_gapA_manifest_ok:", not errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
