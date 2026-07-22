#!/usr/bin/env python3
"""Freeze the V6-P1e paired-128 and deconfounded-1024 designs before solving."""

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
PILOT_CONFIG = CONFIG_DIR / "v6_p1e_deconfounded_paired128.yaml"
FINAL_CONFIG = CONFIG_DIR / "v6_p1e_deconfounded1024.yaml"
SEED = 6105
TOP_MAIN = (500.0, 1000.0, 1750.0, 2500.0)
BOTTOM_MAIN = (1.0, 20.0, 80.0, 200.0)
TOP_OOD = (650.0, 1250.0, 2000.0, 2375.0)
BOTTOM_OOD = (5.0, 40.0, 120.0, 160.0)
POWERS = (2.0, 6.0, 10.0, 14.0)
ACTIVE_LAYERS = ("silicon_die_lower", "silicon_die_upper")


def _seed(label: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{label}".encode()).hexdigest()[:16], 16) % (2**32)


def _roles() -> list[str]:
    return (
        ["train"] * 6 + ["valid_iid", "test_iid"]
        + ["train"] * 16 + ["valid_iid"] * 4 + ["test_iid"] * 4
        + ["layout_ood"] * 2 + ["source_count_ood"] * 2
        + ["power_density_ood"] * 2 + ["bc_ood"] * 2
    )


def _source_count(group_index: int, role: str) -> int:
    if role == "source_count_ood":
        return (2, 12)[group_index % 2]
    return (3, 4, 5, 6, 7, 8, 9, 10)[group_index % 8]


def _dimensions_mm(rng: np.random.Generator, count: int, role: str) -> tuple[float, float]:
    if role == "power_density_ood":
        low_density = bool(rng.integers(0, 2))
        if low_density:
            width = rng.uniform(2.0, 2.2)
            height = rng.uniform(2.65, 2.95)
        else:
            width = rng.uniform(1.25, 1.45)
            height = rng.uniform(1.25, 1.55)
    elif count <= 3:
        width = rng.uniform(1.9, 2.15)
        height = rng.uniform(2.55, 2.85)
    elif count <= 5:
        width = rng.uniform(1.65, 2.1)
        height = rng.uniform(2.2, 2.8)
    else:
        width = rng.uniform(1.3, 2.05)
        height = rng.uniform(1.55, 2.75)
    return float(width), float(height)


def _geometry(group_index: int, role: str) -> dict[str, Any]:
    group_id = f"p1e_g{group_index:02d}"
    rng = np.random.default_rng(_seed(group_id))
    count = _source_count(group_index, role)
    lower_count = count // 2
    upper_count = count - lower_count
    if count == 2:
        lower_count = upper_count = 1
    counts = {ACTIVE_LAYERS[0]: lower_count, ACTIVE_LAYERS[1]: upper_count}
    nominal_upper = upper_count / count
    upper_fraction = 0.5 if min(lower_count, upper_count) == 1 else float(np.clip(nominal_upper + rng.uniform(-0.08, 0.08), 0.35, 0.65))
    layer_fractions = {ACTIVE_LAYERS[0]: 1.0 - upper_fraction, ACTIVE_LAYERS[1]: upper_fraction}
    layout_kind = (
        ("edge_band", "diagonal")[group_index % 2]
        if role == "layout_ood" else ("distributed", "clustered", "mixed")[group_index % 3]
    )
    aligned = bool(group_index % 2 == 0)
    x_centers = np.asarray([1.25, 3.75, 6.25, 8.75])
    y_centers = np.asarray([1.67, 5.0, 8.33])
    slots = np.asarray([(x, y) for y in y_centers for x in x_centers], dtype=np.float64)
    sources: list[dict[str, Any]] = []
    lower_slot_indices: list[int] = []
    for layer_index, layer in enumerate(ACTIVE_LAYERS):
        layer_count = counts[layer]
        if layout_kind == "clustered":
            candidate = np.asarray([5, 6, 1, 2, 9, 10, 4, 7, 0, 3, 8, 11])
        elif layout_kind == "edge_band":
            candidate = np.asarray([0, 3, 8, 11, 4, 7, 1, 2, 9, 10, 5, 6])
        elif layout_kind == "diagonal":
            candidate = np.asarray([0, 5, 10, 3, 6, 9, 1, 4, 7, 11, 2, 8])
        else:
            candidate = rng.permutation(12)
        if layer_index == 1 and aligned and lower_slot_indices:
            shared = lower_slot_indices[: min(layer_count, len(lower_slot_indices))]
            remaining = [int(value) for value in candidate if int(value) not in shared]
            selected = shared + remaining[: layer_count - len(shared)]
        else:
            selected = [int(value) for value in candidate[:layer_count]]
        if layer_index == 0:
            lower_slot_indices = selected
        raw_weights: list[float] = []
        local_rows: list[dict[str, Any]] = []
        for slot_index in selected:
            width, height = _dimensions_mm(rng, count, role)
            center = slots[slot_index] + rng.uniform([-0.08, -0.08], [0.08, 0.08])
            x0, x1 = center[0] - width / 2.0, center[0] + width / 2.0
            y0, y1 = center[1] - height / 2.0, center[1] + height / 2.0
            if not (0.05 <= x0 < x1 <= 9.95 and 0.05 <= y0 < y1 <= 9.95):
                raise AssertionError(f"{group_id}: source outside footprint")
            area = width * height
            raw_weight = area * float(rng.uniform(0.88, 1.12))
            raw_weights.append(raw_weight)
            local_rows.append({
                "layer": layer, "slot_index": slot_index,
                "bbox_fraction_xy": [float(x0 / 10.0), float(x1 / 10.0), float(y0 / 10.0), float(y1 / 10.0)],
                "width_mm": width, "height_mm": height, "declared_area_mm2": area,
            })
        normalized = np.asarray(raw_weights) / np.sum(raw_weights)
        for row, fraction in zip(local_rows, normalized, strict=True):
            row["package_power_fraction"] = float(layer_fractions[layer] * fraction)
            sources.append(row)
    power_fraction_sum = sum(float(source["package_power_fraction"]) for source in sources)
    if not math.isclose(power_fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError(f"{group_id}: power fractions do not sum to one")
    max_surface_density = max(
        max(POWERS) * float(source["package_power_fraction"]) / (float(source["declared_area_mm2"]) * 1e-6)
        for source in sources
    )
    if max_surface_density > 1.5e6:
        raise AssertionError(f"{group_id}: source exceeds 150 W/cm2")
    return {
        "group_id": group_id, "split_role": role, "geometry_seed": _seed(group_id),
        "layout_kind": layout_kind, "alignment_relation": "partly_aligned" if aligned else "offset",
        "source_count": count, "upper_layer_power_fraction": upper_fraction,
        "total_source_area_mm2": float(sum(source["declared_area_mm2"] for source in sources)),
        "max_preregistered_source_surface_power_density_W_m2": max_surface_density,
        "sources": sources,
    }


def _cases(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        role = group["split_role"]
        top_levels = TOP_OOD if role == "bc_ood" else TOP_MAIN
        bottom_levels = BOTTOM_OOD if role == "bc_ood" else BOTTOM_MAIN
        if group_index < 8:
            design = "complete_factorial"
            factor_rows = [(ti, bi, pi) for ti in range(4) for bi in range(4) for pi in range(4)]
        else:
            design = "orthogonal_array_16"
            offset = group_index % 4
            factor_rows = [(ti, bi, (ti + bi + offset) % 4) for ti in range(4) for bi in range(4)]
        for ti, bi, pi in factor_rows:
            cases.append({
                "id": f"{group['group_id']}_t{ti}_b{bi}_p{pi}",
                "group_id": group["group_id"], "split_role": role, "design_block": design,
                "top_h_W_m2K": top_levels[ti], "bottom_h_W_m2K": bottom_levels[bi],
                "package_total_power_W": POWERS[pi], "top_level_index": ti,
                "bottom_level_index": bi, "power_level_index": pi,
            })
    return cases


def _payload(groups: list[dict[str, Any]], cases: list[dict[str, Any]], dataset_id: str, pilot: bool) -> dict[str, Any]:
    return {
        "schema_version": "heat3d_v6_p1e_deconfounded_dataset_v1",
        "dataset_id": dataset_id, "status": "frozen_before_generation", "sample_count": len(cases),
        "seed": SEED,
        "scope": {
            "model_training": False, "model_inference": False,
            "peak_deltaT_filtering": False, "peak_deltaT_resampling": False,
            "sample_replacement": False, "per_sample_Rth_power_back_calculation": False,
            "post_solve_factor_or_seed_selection": False,
        },
        "provenance": {
            "p1d_baseline_config": "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024.yaml",
            "p1d_policy": "immutable_provenance_not_formal_training_dataset",
            "literature_matrix": "docs/v6_p1e_literature_matrix.csv",
            "design_freeze_stage": "before_any_P1e_temperature_solve",
            "paired128_is_qualification_only": pilot,
        },
        "physics": {
            "footprint_m": [0.01, 0.01],
            "stack_source": "configs/heat3d_v6/v6_p1c_package_path_calibration_cases.yaml#B_remote_dirichlet",
            "retain_complete_package_path": True,
            "solver_mesh_intervals_xyz": [64, 64, 56],
            "boundary_common": {
                "top_ambient_K": 300.0, "bottom_ambient_K": 300.0,
                "sides": "adiabatic", "contact": {"type": "perfect", "R_contact_m2K_W": 0.0},
            },
            "main_top_h_W_m2K": list(TOP_MAIN), "main_bottom_h_W_m2K": list(BOTTOM_MAIN),
            "bc_ood_top_h_W_m2K": list(TOP_OOD), "bc_ood_bottom_h_W_m2K": list(BOTTOM_OOD),
        },
        "factor_contract": {
            "common_package_power_levels_W_for_every_BC_family": list(POWERS),
            "paired_complete_factorial_group_count": len(groups) if pilot else 8,
            "paired_complete_factorial_case_count": len(cases) if pilot else 512,
            "power_was_BC_specific": False,
            "temperature_window_K": [30.0, 80.0],
            "temperature_window_role": "report_only_not_filter_or_replacement_rule",
        },
        "source_contract": {
            "active_layers": list(ACTIVE_LAYERS), "source_count_IID_range": [3, 10],
            "source_count_OOD_values": [2, 12], "source_count_fixed": False,
            "randomized_factors": ["source_count", "area_allocation", "aspect_ratio", "clustering", "upper_lower_power_fraction", "alignment"],
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
            "group_key": "group_id", "all_BC_and_power_cases_stay_in_one_split": True,
            "roles": ["train", "valid_iid", "test_iid", "layout_ood", "bc_ood", "source_count_ood", "power_density_ood"],
            "model_selection_roles": [], "dataset_qualification_only": True,
        },
        "geometry_groups": groups, "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", type=Path, default=PILOT_CONFIG)
    parser.add_argument("--final-config", type=Path, default=FINAL_CONFIG)
    args = parser.parse_args()
    roles = _roles()
    if len(roles) != 40:
        raise AssertionError("group role schedule must contain 40 groups")
    groups = [_geometry(index, role) for index, role in enumerate(roles)]
    cases = _cases(groups)
    if len(cases) != 1024:
        raise AssertionError(f"expected 1024 cases, got {len(cases)}")
    pilot_groups = groups[:2]
    pilot_cases = [case for case in cases if case["group_id"] in {group["group_id"] for group in pilot_groups}]
    if len(pilot_cases) != 128:
        raise AssertionError("paired qualification block must contain 128 cases")
    args.pilot_config.write_text(yaml.safe_dump(
        _payload(pilot_groups, pilot_cases, "heat3d_v6_p1e_deconfounded_paired128_v0", True),
        sort_keys=False, width=120,
    ), encoding="utf-8")
    args.final_config.write_text(yaml.safe_dump(
        _payload(groups, cases, "heat3d_v6_p1e_deconfounded1024_v0", False),
        sort_keys=False, width=120,
    ), encoding="utf-8")
    print(json.dumps({
        "status": "ok", "pilot_cases": len(pilot_cases), "final_cases": len(cases),
        "geometry_groups": len(groups), "common_power_levels_W": POWERS,
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
