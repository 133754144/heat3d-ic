#!/usr/bin/env python3
"""Freeze P1g-v1 by applying one global alignment-only transform to P1g-v0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
PARENT_CONFIG = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024.yaml"
PARENT_MANIFEST = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024_manifest.json"
PARENT_QUALIFICATION = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024_qualification.json"
OUTPUT = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024_v1.yaml"
OFFSET_MM = 0.3125  # exactly two 10 mm / 64 solver-mesh intervals
FOOTPRINT_MIN_MM = 0.05
FOOTPRINT_MAX_MM = 9.95


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _center_mm(source: Mapping[str, Any]) -> tuple[float, float]:
    x0, x1, y0, y1 = map(float, source["bbox_fraction_xy"])
    return 5.0 * (x0 + x1), 5.0 * (y0 + y1)


def _set_center(source: dict[str, Any], center_x_mm: float, center_y_mm: float) -> None:
    half_width = float(source["width_mm"]) / 2.0
    half_height = float(source["height_mm"]) / 2.0
    x0, x1 = center_x_mm - half_width, center_x_mm + half_width
    y0, y1 = center_y_mm - half_height, center_y_mm + half_height
    if not (
        FOOTPRINT_MIN_MM <= x0 < x1 <= FOOTPRINT_MAX_MM
        and FOOTPRINT_MIN_MM <= y0 < y1 <= FOOTPRINT_MAX_MM
    ):
        raise AssertionError("alignment transform placed a source outside the frozen footprint margin")
    source["bbox_fraction_xy"] = [x0 / 10.0, x1 / 10.0, y0 / 10.0, y1 / 10.0]


def _boxes_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    lx0, lx1, ly0, ly1 = map(float, left["bbox_fraction_xy"])
    rx0, rx1, ry0, ry1 = map(float, right["bbox_fraction_xy"])
    return not (lx1 <= rx0 or rx1 <= lx0 or ly1 <= ry0 or ry1 <= ly0)


def _candidate_vectors(group_id: str) -> list[tuple[float, float]]:
    diagonal = OFFSET_MM / math.sqrt(2.0)
    candidates = [
        (OFFSET_MM, 0.0), (-OFFSET_MM, 0.0), (0.0, OFFSET_MM), (0.0, -OFFSET_MM),
        (diagonal, diagonal), (-diagonal, diagonal), (diagonal, -diagonal), (-diagonal, -diagonal),
    ]
    # The rotation is input-ID-derived and frozen before solving.  It prevents
    # every offset group from sharing a direction without consulting labels.
    rotation = int(hashlib.sha256(f"{group_id}:p1g_v1_offset_direction".encode()).hexdigest()[:8], 16) % len(candidates)
    return candidates[rotation:] + candidates[:rotation]


def _try_transform(
    upper: list[dict[str, Any]], lower: list[dict[str, Any]], vector: tuple[float, float], relation: str,
) -> list[dict[str, Any]] | None:
    transformed = copy.deepcopy(upper)
    paired_count = min(len(lower), len(transformed))
    for index in range(paired_count):
        lower_x, lower_y = _center_mm(lower[index])
        dx, dy = vector if relation == "offset" else (0.0, 0.0)
        try:
            _set_center(transformed[index], lower_x + dx, lower_y + dy)
        except AssertionError:
            return None
        transformed[index]["slot_index"] = int(lower[index]["slot_index"])
        transformed[index]["alignment_reference_lower_slot"] = int(lower[index]["slot_index"])
        transformed[index]["alignment_center_displacement_mm"] = math.hypot(dx, dy)
    for index in range(paired_count, len(transformed)):
        transformed[index]["alignment_reference_lower_slot"] = None
        transformed[index]["alignment_center_displacement_mm"] = None
    for left in range(len(transformed)):
        for right in range(left + 1, len(transformed)):
            if _boxes_overlap(transformed[left], transformed[right]):
                return None
    return transformed


def _transform_group(parent: Mapping[str, Any], global_index: int) -> dict[str, Any]:
    group = copy.deepcopy(parent)
    parent_group_id = str(parent["group_id"])
    group["group_id"] = f"p1g_v1_g{global_index:03d}"
    group["projection_seed_key"] = parent_group_id
    group["parent_p1g_v0_group_id"] = parent_group_id
    lower = sorted(
        [copy.deepcopy(source) for source in parent["sources"] if source["layer"] == "silicon_die_lower"],
        key=lambda source: (int(source["slot_index"]), source["bbox_fraction_xy"]),
    )
    upper = sorted(
        [copy.deepcopy(source) for source in parent["sources"] if source["layer"] == "silicon_die_upper"],
        key=lambda source: (int(source["slot_index"]), source["bbox_fraction_xy"]),
    )
    remaining_upper = list(upper)
    paired_upper: list[dict[str, Any]] = []
    for lower_source in lower:
        match_index = next((
            index for index, upper_source in enumerate(remaining_upper)
            if int(upper_source["slot_index"]) == int(lower_source["slot_index"])
        ), 0)
        paired_upper.append(remaining_upper.pop(match_index))
    upper = paired_upper + remaining_upper
    relation = str(parent["alignment_relation"])
    vectors = [(0.0, 0.0)] if relation == "partly_aligned" else _candidate_vectors(parent_group_id)
    transformed_upper = None
    selected_vector = None
    for vector in vectors:
        transformed_upper = _try_transform(upper, lower, vector, relation)
        if transformed_upper is not None:
            selected_vector = vector
            break
    if transformed_upper is None or selected_vector is None:
        raise AssertionError(f"{parent_group_id}: no safe frozen alignment transform")
    group["sources"] = lower + transformed_upper
    group["alignment_transform"] = {
        "version": "two_mesh_interval_centroid_offset_v1",
        "paired_source_count": min(len(lower), len(transformed_upper)),
        "upper_translation_mm": list(map(float, selected_vector)),
        "translation_magnitude_mm": float(math.hypot(*selected_vector)),
        "selection_inputs": ["parent_source_geometry", "footprint_margin", "group_id_hash"],
        "label_or_temperature_inputs": [],
    }
    # No size, area, fraction, material, or power change is permitted.
    parent_by_layer = CounterLike(parent["sources"])
    child_by_layer = CounterLike(group["sources"])
    if parent_by_layer != child_by_layer:
        raise AssertionError(f"{parent_group_id}: non-position source attributes changed")
    return group


def CounterLike(sources: list[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted((
        str(source["layer"]), round(float(source["width_mm"]), 15), round(float(source["height_mm"]), 15),
        round(float(source["declared_area_mm2"]), 15), round(float(source["package_power_fraction"]), 15),
    ) for source in sources)


def _payload() -> dict[str, Any]:
    parent = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    parent_qualification = json.loads(PARENT_QUALIFICATION.read_text(encoding="utf-8"))
    if parent_qualification["formal_training_qualified"] is not False:
        raise AssertionError("P1g-v1 contract expects the preserved P1g-v0 failed-gate provenance")
    groups = [_transform_group(group, index) for index, group in enumerate(parent["geometry_groups"])]
    group_id_map = {
        str(parent_group["group_id"]): str(group["group_id"])
        for parent_group, group in zip(parent["geometry_groups"], groups, strict=True)
    }
    cases: list[dict[str, Any]] = []
    for parent_case in parent["cases"]:
        case = copy.deepcopy(parent_case)
        parent_group_id = str(case["group_id"])
        group_id = group_id_map[parent_group_id]
        case["parent_p1g_v0_sample_id"] = str(case["id"])
        suffix = str(case["id"])[len(parent_group_id):]
        case["id"] = f"{group_id}{suffix}"
        case["group_id"] = group_id
        cases.append(case)
    payload = copy.deepcopy(parent)
    payload.update({
        "schema_version": "heat3d_v6_p1g_geometry_deconfounded_dataset_v2",
        "dataset_id": "heat3d_v6_p1g_geometry_deconfounded1024_v1",
        "status": "frozen_before_generation",
        "geometry_groups": groups,
        "cases": cases,
    })
    payload["provenance"] = {
        **copy.deepcopy(parent["provenance"]),
        "parent_P1g_v0_config": str(PARENT_CONFIG.relative_to(ROOT)),
        "parent_P1g_v0_config_sha256": _sha256(PARENT_CONFIG),
        "parent_P1g_v0_manifest": str(PARENT_MANIFEST.relative_to(ROOT)),
        "parent_P1g_v0_manifest_sha256": _sha256(PARENT_MANIFEST),
        "parent_P1g_v0_qualification": str(PARENT_QUALIFICATION.relative_to(ROOT)),
        "parent_P1g_v0_qualification_sha256": _sha256(PARENT_QUALIFICATION),
        "parent_P1g_v0_decision": parent_qualification["decision"],
        "whole_version_revision_reason": "P1g_v0_failed_frozen_below30_subgate",
        "revision_scope": "uniform_alignment_geometry_definition_only",
        "per_sample_filtering_replacement_or_patch": False,
        "seed_changed_or_searched": False,
        "generator_contract": "scripts/prepare_heat3d_v6_p1g_geometry_deconfounded_v1_config.py",
    }
    payload["geometry_assignment_contract"] = {
        **copy.deepcopy(parent["geometry_assignment_contract"]),
        "parent_assignment_rows_reused_exactly": True,
        "alignment_geometry_definition": {
            "partly_aligned": "paired upper/lower source centroids coincide",
            "offset": "paired upper centroids translated exactly two solver-mesh intervals",
            "offset_magnitude_mm": OFFSET_MM,
            "direction_selection": "first collision-free direction in group-id-hash-rotated frozen candidate list",
            "label_or_temperature_inputs": [],
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = _payload()
    if len(payload["geometry_groups"]) != 128 or len(payload["cases"]) != 1024:
        raise AssertionError("P1g-v1 requires 128 geometry groups and 1024 cases")
    output.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
    print(json.dumps({
        "status": "ok", "output": str(output), "dataset_id": payload["dataset_id"],
        "geometry_groups": len(payload["geometry_groups"]), "sample_count": len(payload["cases"]),
        "seed_changed_or_searched": False, "training_runs": 0, "model_inference_runs": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
