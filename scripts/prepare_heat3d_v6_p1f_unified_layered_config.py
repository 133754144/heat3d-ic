#!/usr/bin/env python3
"""Freeze P1f unified pilot/final configs; final requires a passing pilot gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
PILOT_CONFIG = CONFIG_DIR / "v6_p1f_temperature_shaping_pilot128.yaml"
FINAL_CONFIG = CONFIG_DIR / "v6_p1f_unified_layered1024.yaml"
PILOT_QUALIFICATION = CONFIG_DIR / "v6_p1f_temperature_shaping_pilot128_qualification.json"
PILOT_SEED = 6201001
FINAL_SEED = 6202001
TOP_H = (1000.0, 1400.0)
BOTTOM_H = (20.0, 120.0)
POWERS = (4.0, 6.0)
ACTIVE_LAYERS = ("silicon_die_lower", "silicon_die_upper")
GATE = {
    "peak_deltaT_below_30_count_max": 0,
    "peak_deltaT_30_80_fraction_min": 0.80,
    "peak_deltaT_above_100_fraction_max": 0.05,
    "peak_deltaT_above_120_count_max": 0,
}


def _seed(base_seed: int, label: str) -> int:
    return int(hashlib.sha256(f"{base_seed}:{label}:p1f_geometry_v1".encode()).hexdigest()[:16], 16) % (2**32)


def _role_schedule(stage: str) -> list[str]:
    if stage == "pilot":
        return ["pilot_only"] * 16
    return ["train"] * 96 + ["valid"] * 16 + ["test"] * 16


def _dimensions_mm(rng: np.random.Generator, source_count: int) -> tuple[float, float]:
    if source_count <= 3:
        width = rng.uniform(1.90, 2.15)
        height = rng.uniform(2.55, 2.85)
    elif source_count <= 5:
        width = rng.uniform(1.65, 2.10)
        height = rng.uniform(2.20, 2.80)
    else:
        width = rng.uniform(1.35, 2.05)
        height = rng.uniform(1.60, 2.70)
    return float(width), float(height)


def _geometry(stage: str, global_index: int, split_ordinal: int, role: str) -> dict[str, Any]:
    base_seed = PILOT_SEED if stage == "pilot" else FINAL_SEED
    prefix = "p1f_pilot" if stage == "pilot" else "p1f"
    group_id = f"{prefix}_g{global_index:03d}"
    rng = np.random.default_rng(_seed(base_seed, group_id))
    source_count = 3 + split_ordinal % 8
    lower_count = source_count // 2
    upper_count = source_count - lower_count
    counts = {ACTIVE_LAYERS[0]: lower_count, ACTIVE_LAYERS[1]: upper_count}
    nominal_upper = upper_count / source_count
    upper_fraction = 0.5 if min(lower_count, upper_count) == 1 else float(np.clip(nominal_upper + rng.uniform(-0.07, 0.07), 0.38, 0.62))
    layer_fraction = {ACTIVE_LAYERS[0]: 1.0 - upper_fraction, ACTIVE_LAYERS[1]: upper_fraction}
    layout_kind = ("distributed", "clustered", "mixed", "edge_balanced")[split_ordinal % 4]
    aligned = bool((split_ordinal // 2) % 2 == 0)
    x_centers = np.asarray([1.25, 3.75, 6.25, 8.75])
    y_centers = np.asarray([1.67, 5.0, 8.33])
    slots = np.asarray([(x, y) for y in y_centers for x in x_centers], dtype=np.float64)
    sources: list[dict[str, Any]] = []
    lower_slots: list[int] = []
    for layer_index, layer in enumerate(ACTIVE_LAYERS):
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
        else:
            selected = [int(value) for value in candidates[:count]]
        if layer_index == 0:
            lower_slots = selected
        local_rows: list[dict[str, Any]] = []
        raw_weights: list[float] = []
        for slot_index in selected:
            width, height = _dimensions_mm(rng, source_count)
            center = slots[slot_index] + rng.uniform([-0.08, -0.08], [0.08, 0.08])
            x0, x1 = center[0] - width / 2.0, center[0] + width / 2.0
            y0, y1 = center[1] - height / 2.0, center[1] + height / 2.0
            if not (0.05 <= x0 < x1 <= 9.95 and 0.05 <= y0 < y1 <= 9.95):
                raise AssertionError(f"{group_id}: source outside footprint")
            area = float(width * height)
            raw_weights.append(area * float(rng.uniform(0.90, 1.10)))
            local_rows.append({
                "layer": layer, "slot_index": slot_index,
                "bbox_fraction_xy": [float(x0 / 10), float(x1 / 10), float(y0 / 10), float(y1 / 10)],
                "width_mm": width, "height_mm": height, "declared_area_mm2": area,
            })
        normalized = np.asarray(raw_weights) / np.sum(raw_weights)
        for row, fraction in zip(local_rows, normalized, strict=True):
            row["package_power_fraction"] = float(layer_fraction[layer] * fraction)
            sources.append(row)
    if not math.isclose(sum(float(row["package_power_fraction"]) for row in sources), 1.0, abs_tol=1e-14):
        raise AssertionError(f"{group_id}: source fractions")
    maximum_density = max(
        max(POWERS) * float(row["package_power_fraction"]) / (float(row["declared_area_mm2"]) * 1e-6)
        for row in sources
    )
    if maximum_density > 1.5e6:
        raise AssertionError(f"{group_id}: source density")
    return {
        "group_id": group_id, "split_role": role,
        "geometry_seed": _seed(base_seed, group_id), "split_ordinal": split_ordinal,
        "material_profile_id": "logic_package_complete_B_fixed_materials_v1",
        "layout_kind": layout_kind,
        "alignment_relation": "partly_aligned" if aligned else "offset",
        "source_count": source_count, "upper_layer_power_fraction": upper_fraction,
        "total_source_area_mm2": float(sum(float(row["declared_area_mm2"]) for row in sources)),
        "maximum_preregistered_source_surface_power_density_W_m2": maximum_density,
        "sources": sources,
    }


def _cases(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for group in groups:
        for top_index in range(2):
            for bottom_index in range(2):
                for power_index in range(2):
                    cases.append({
                        "id": f"{group['group_id']}_t{top_index}_b{bottom_index}_p{power_index}",
                        "group_id": group["group_id"], "split_role": group["split_role"],
                        "design_block": "complete_factorial_2x2x2",
                        "top_h_W_m2K": TOP_H[top_index], "bottom_h_W_m2K": BOTTOM_H[bottom_index],
                        "package_total_power_W": POWERS[power_index],
                        "top_level_index": top_index, "bottom_level_index": bottom_index,
                        "power_level_index": power_index,
                    })
    return cases


def _payload(stage: str, groups: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    seed = PILOT_SEED if stage == "pilot" else FINAL_SEED
    return {
        "schema_version": "heat3d_v6_p1f_unified_layered_dataset_v1",
        "dataset_id": f"heat3d_v6_p1f_{'temperature_shaping_pilot128' if stage == 'pilot' else 'unified_layered1024'}_v0",
        "stage": stage, "status": "frozen_before_generation", "sample_count": len(cases),
        "geometry_group_count": len(groups), "seed": seed,
        "scope": {
            "model_training": False, "model_inference": False,
            "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
            "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
            "post_solve_case_or_seed_selection": False,
        },
        "provenance": {
            "p1d_config": "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024.yaml",
            "p1e_config": "configs/heat3d_v6/v6_p1e_deconfounded1024.yaml",
            "version_policy": "new_dataset_keep_P1d_P1e_immutable",
            "pilot_samples_retained_in_final": False,
            "generator_contract": "scripts/prepare_heat3d_v6_p1f_unified_layered_config.py",
        },
        "physics": {
            "footprint_m": [0.01, 0.01],
            "stack_source": "configs/heat3d_v6/v6_p1c_package_path_calibration_cases.yaml#B_remote_dirichlet",
            "material_distribution": {
                "mode": "fixed_profile", "profile_id": "logic_package_complete_B_fixed_materials_v1",
                "identical_across_train_valid_test": True,
            },
            "solver_mesh_intervals_xyz": [64, 64, 56],
            "boundary_common": {
                "top_ambient_K": 300.0, "bottom_ambient_K": 300.0,
                "sides": "adiabatic", "contact": {"type": "perfect", "R_contact_m2K_W": 0.0},
            },
            "top_h_W_m2K": list(TOP_H), "bottom_h_W_m2K": list(BOTTOM_H),
        },
        "factor_contract": {
            "package_power_W": list(POWERS),
            "cases_per_geometry_group": 8,
            "orthogonal_design": "complete_factorial_2x2x2",
            "same_distribution_across_train_valid_test": True,
            "temperature_shaping_is_global_contract_only": True,
            "temperature_gate": GATE,
        },
        "source_contract": {
            "active_layers": list(ACTIVE_LAYERS), "source_count_range": [3, 10],
            "randomized_factors": ["source_count", "area_allocation", "aspect_ratio", "clustering", "upper_lower_power_fraction", "alignment"],
            "same_generator_across_train_valid_test": True,
            "maximum_surface_power_density_W_m2": 1.5e6,
            "maximum_surface_power_density_W_cm2": 150.0,
            "maximum_q_W_m3": 1.5e10, "maximum_single_source_power_W": 8.0,
            "minimum_source_control_volume_count": 128, "minimum_source_in_plane_intervals": 7,
        },
        "operator_projection": {
            "point_count": 1024, "point_seed_key": "geometry_group_not_case",
            "coordinates_reused_within_geometry_group": True,
            "strata": {"volume": 512, "source": 256, "interface": 128, "top": 64, "bottom": 64},
            "label_inputs_used_for_point_selection": [],
        },
        "split_contract": {
            "group_key": "group_id", "all_cases_in_group_stay_in_one_split": True,
            "roles": ["pilot_only"] if stage == "pilot" else ["train", "valid", "test"],
            "group_counts": {"pilot_only": 16} if stage == "pilot" else {"train": 96, "valid": 16, "test": 16},
            "OOD_roles": [],
        },
        "geometry_groups": groups, "cases": cases,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")


def _groups(stage: str) -> list[dict[str, Any]]:
    roles = _role_schedule(stage)
    per_role: dict[str, int] = {}
    groups: list[dict[str, Any]] = []
    for global_index, role in enumerate(roles):
        ordinal = per_role.get(role, 0)
        groups.append(_geometry(stage, global_index, ordinal, role))
        per_role[role] = ordinal + 1
    return groups


def _require_pilot_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True or payload.get("gate") != GATE:
        raise AssertionError("final config requires the exact passing pilot gate")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-qualification", type=Path, default=PILOT_QUALIFICATION)
    args = parser.parse_args()
    output = args.output or (PILOT_CONFIG if args.stage == "pilot" else FINAL_CONFIG)
    if args.stage == "final":
        qualification = args.pilot_qualification if args.pilot_qualification.is_absolute() else ROOT / args.pilot_qualification
        _require_pilot_gate(qualification)
    groups = _groups(args.stage)
    cases = _cases(groups)
    expected = 128 if args.stage == "pilot" else 1024
    if len(cases) != expected:
        raise AssertionError(f"{args.stage}: expected {expected} cases")
    _write(output, _payload(args.stage, groups, cases))
    print(json.dumps({
        "status": "ok", "stage": args.stage, "output": str(output),
        "geometry_groups": len(groups), "sample_count": len(cases),
        "seed": PILOT_SEED if args.stage == "pilot" else FINAL_SEED,
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
