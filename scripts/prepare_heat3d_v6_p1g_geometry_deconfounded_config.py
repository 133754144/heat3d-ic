#!/usr/bin/env python3
"""Freeze the V6-P1g geometry-deconfounded 1024-sample configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import prepare_heat3d_v6_p1f_unified_layered_config as p1f
import heat3d_v6_p1d_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
OUTPUT = CONFIG_DIR / "v6_p1g_geometry_deconfounded1024.yaml"
P1F_CONFIG = CONFIG_DIR / "v6_p1f_unified_layered1024.yaml"
P1F_MANIFEST = CONFIG_DIR / "v6_p1f_unified_layered1024_manifest.json"
SEED = 6301001
LAYOUTS = ("distributed", "clustered", "mixed", "edge_balanced")
ALIGNMENTS = ("offset", "partly_aligned")


def _seed(label: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{label}:p1g_geometry_v1".encode()).hexdigest()[:16], 16) % (2**32)


def _assignment_rows(role: str) -> list[dict[str, Any]]:
    """Build a balanced joint table before an independent role-local shuffle."""
    rows: list[dict[str, Any]] = []
    if role == "train":
        for count_index, source_count in enumerate(range(3, 11)):
            for layout_index, layout_kind in enumerate(LAYOUTS):
                # Three replicates per count/layout.  Alternating 2:1 alignment
                # balance makes all train pairwise margins exactly independent.
                alignments = (0, 0, 1) if (count_index + layout_index) % 2 == 0 else (0, 1, 1)
                for replicate, alignment_index in enumerate(alignments):
                    rows.append({
                        "pre_shuffle_assignment_id": f"train_c{source_count}_l{layout_index}_a{alignment_index}_r{replicate}",
                        "source_count": source_count,
                        "layout_kind": layout_kind,
                        "alignment_relation": ALIGNMENTS[alignment_index],
                    })
    else:
        layout_offset = 0 if role == "valid" else 2
        for count_index, source_count in enumerate(range(3, 11)):
            for local_index in range(2):
                layout_index = (count_index + layout_offset + local_index) % len(LAYOUTS)
                # Fill the alignment underrepresented in train for this cell.
                alignment_index = 1 if (count_index + layout_index) % 2 == 0 else 0
                rows.append({
                    "pre_shuffle_assignment_id": f"{role}_c{source_count}_l{layout_index}_a{alignment_index}",
                    "source_count": source_count,
                    "layout_kind": LAYOUTS[layout_index],
                    "alignment_relation": ALIGNMENTS[alignment_index],
                })
    expected = {"train": 96, "valid": 16, "test": 16}[role]
    if len(rows) != expected:
        raise AssertionError(f"{role}: expected {expected} assignment rows")
    rng = np.random.default_rng(_seed(f"assignment_shuffle:{role}"))
    order = rng.permutation(len(rows))
    return [{**rows[int(index)], "role_shuffle_position": position} for position, index in enumerate(order)]


def _geometry(global_index: int, split_ordinal: int, role: str, assignment: dict[str, Any]) -> dict[str, Any]:
    group_id = f"p1g_g{global_index:03d}"
    rng = np.random.default_rng(_seed(group_id))
    source_count = int(assignment["source_count"])
    layout_kind = str(assignment["layout_kind"])
    aligned = assignment["alignment_relation"] == "partly_aligned"
    lower_count = source_count // 2
    upper_count = source_count - lower_count
    counts = {p1f.ACTIVE_LAYERS[0]: lower_count, p1f.ACTIVE_LAYERS[1]: upper_count}
    nominal_upper = upper_count / source_count
    upper_fraction = (
        0.5 if min(lower_count, upper_count) == 1
        else float(np.clip(nominal_upper + rng.uniform(-0.07, 0.07), 0.38, 0.62))
    )
    layer_fraction = {p1f.ACTIVE_LAYERS[0]: 1.0 - upper_fraction, p1f.ACTIVE_LAYERS[1]: upper_fraction}
    x_centers = np.asarray([1.25, 3.75, 6.25, 8.75])
    y_centers = np.asarray([1.67, 5.0, 8.33])
    slots = np.asarray([(x, y) for y in y_centers for x in x_centers], dtype=np.float64)
    sources: list[dict[str, Any]] = []
    lower_slots: list[int] = []
    for layer_index, layer in enumerate(p1f.ACTIVE_LAYERS):
        count = counts[layer]
        if layout_kind == "clustered":
            candidates = np.asarray([5, 6, 1, 2, 9, 10, 4, 7, 0, 3, 8, 11])
        elif layout_kind == "edge_balanced":
            candidates = np.asarray([0, 3, 8, 11, 4, 7, 1, 2, 9, 10, 5, 6])
        elif layout_kind == "mixed":
            candidates = np.asarray([0, 5, 10, 3, 6, 9, 1, 4, 7, 11, 2, 8])
        else:
            candidates = rng.permutation(12)
        if layer_index == 1 and aligned and lower_slots:
            shared = lower_slots[: min(count, len(lower_slots))]
            remaining = [int(value) for value in candidates if int(value) not in shared]
            selected = shared + remaining[: count - len(shared)]
        elif layer_index == 1 and lower_slots:
            # P1f labelled this branch "offset" but deterministic layout
            # candidates could still repeat all lower-layer slots.  P1g makes
            # the frozen alignment factor geometrically truthful.
            remaining = [int(value) for value in candidates if int(value) not in lower_slots]
            selected = remaining[:count]
        else:
            selected = [int(value) for value in candidates[:count]]
        if layer_index == 0:
            lower_slots = selected
        local_rows: list[dict[str, Any]] = []
        raw_weights: list[float] = []
        for slot_index in selected:
            width, height = p1f._dimensions_mm(rng, source_count)
            center = slots[slot_index] + rng.uniform([-0.08, -0.08], [0.08, 0.08])
            x0, x1 = center[0] - width / 2.0, center[0] + width / 2.0
            y0, y1 = center[1] - height / 2.0, center[1] + height / 2.0
            if not (0.05 <= x0 < x1 <= 9.95 and 0.05 <= y0 < y1 <= 9.95):
                raise AssertionError(f"{group_id}: source outside footprint")
            area = float(width * height)
            raw_weights.append(area * float(rng.uniform(0.90, 1.10)))
            local_rows.append({
                "layer": layer,
                "slot_index": slot_index,
                "bbox_fraction_xy": [float(x0 / 10), float(x1 / 10), float(y0 / 10), float(y1 / 10)],
                "width_mm": width,
                "height_mm": height,
                "declared_area_mm2": area,
            })
        normalized = np.asarray(raw_weights) / np.sum(raw_weights)
        for row, fraction in zip(local_rows, normalized, strict=True):
            row["package_power_fraction"] = float(layer_fraction[layer] * fraction)
            sources.append(row)
    if not math.isclose(sum(float(row["package_power_fraction"]) for row in sources), 1.0, abs_tol=1e-14):
        raise AssertionError(f"{group_id}: source fractions")
    maximum_density = max(
        max(p1f.POWERS) * float(row["package_power_fraction"]) / (float(row["declared_area_mm2"]) * 1e-6)
        for row in sources
    )
    if maximum_density > 1.5e6:
        raise AssertionError(f"{group_id}: source density")
    return {
        "group_id": group_id,
        "split_role": role,
        "geometry_seed": _seed(group_id),
        "split_ordinal": split_ordinal,
        "assignment_table_version": "p1g_balanced_joint_v1",
        "assignment_shuffle_seed": _seed(f"assignment_shuffle:{role}"),
        "pre_shuffle_assignment_id": assignment["pre_shuffle_assignment_id"],
        "role_shuffle_position": int(assignment["role_shuffle_position"]),
        "material_profile_id": "logic_package_complete_B_fixed_materials_v1",
        "layout_kind": layout_kind,
        "alignment_relation": assignment["alignment_relation"],
        "source_count": source_count,
        "upper_layer_power_fraction": upper_fraction,
        "total_source_area_mm2": float(sum(float(row["declared_area_mm2"]) for row in sources)),
        "maximum_preregistered_source_surface_power_density_W_m2": maximum_density,
        "sources": sources,
    }


def _groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    global_index = 0
    for role in ("train", "valid", "test"):
        for split_ordinal, assignment in enumerate(_assignment_rows(role)):
            groups.append(_geometry(global_index, split_ordinal, role, assignment))
            global_index += 1
    return groups


def _payload(groups: list[dict[str, Any]]) -> dict[str, Any]:
    cases = p1f._cases(groups)
    payload = p1f._payload("final", groups, cases)
    payload.update({
        "schema_version": "heat3d_v6_p1g_geometry_deconfounded_dataset_v1",
        "dataset_id": "heat3d_v6_p1g_geometry_deconfounded1024_v0",
        "seed": SEED,
    })
    payload["status"] = "frozen_before_generation"
    payload["provenance"] = {
        "parent_version": "P1f-v0",
        "p1f_config": str(P1F_CONFIG.relative_to(ROOT)),
        "p1f_config_sha256": core.sha256(P1F_CONFIG),
        "p1f_manifest": str(P1F_MANIFEST.relative_to(ROOT)),
        "p1f_manifest_sha256": core.sha256(P1F_MANIFEST),
        "version_policy": "new_P1g_version_keep_P1f_v0_immutable",
        "only_rebuilt_component": "geometry_assignment_and_geometry_instances",
        "generator_contract": "scripts/prepare_heat3d_v6_p1g_geometry_deconfounded_config.py",
    }
    payload["geometry_assignment_contract"] = {
        "version": "p1g_balanced_joint_v1",
        "factors": ["source_count", "layout_kind", "alignment_relation"],
        "source_count_levels": list(range(3, 11)),
        "layout_levels": list(LAYOUTS),
        "alignment_levels": list(ALIGNMENTS),
        "assignment_method": "explicit_balanced_joint_table_then_independent_role_seeded_shuffle",
        "factor_generation_from_split_ordinal": False,
        "global_factor_triplet_replicates": 2,
        "train_count_layout_replicates": 3,
        "same_factor_marginals_across_train_valid_test": True,
        "valid_test_count_layout_structural_note": (
            "16 groups cannot make an 8x4 count-layout table independent; each count covers two layouts "
            "and valid/test use complementary minimum-association schedules"
        ),
    }
    payload["operator_projection"]["full_field_reconstruction_diagnostic"] = {
        "method": "inverse_distance_weighted_8_nearest_points",
        "purpose": "representation_QC_only_not_model_inference",
        "reported_metrics": [
            "full_field_cv_rmse_K", "full_field_cv_relative_rmse", "full_field_max_abs_error_K",
            "per_layer_average_error_K", "per_layer_drop_error_K",
        ],
        "point_selection_unchanged_from_P1f": True,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    groups = _groups()
    payload = _payload(groups)
    if len(groups) != 128 or len(payload["cases"]) != 1024:
        raise AssertionError("P1g requires 128 groups and 1024 cases")
    output.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
    print(json.dumps({
        "status": "ok", "output": str(output), "geometry_groups": len(groups),
        "sample_count": len(payload["cases"]), "seed": SEED,
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
