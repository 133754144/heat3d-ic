#!/usr/bin/env python3
"""Prepare deterministic balanced 64/1024-sample P1d expansion configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

import heat3d_v6_p1d_core as core


REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_CONFIG = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_pilot16.yaml"
PILOT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_audit.json"
PILOT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin16_manifest.json"
TRIAL1_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin64_audit.json"
DEFAULT_64 = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_pilot64_balanced.yaml"
DEFAULT_1024 = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024.yaml"
AREAS = (16.0, 32.0, 48.0, 64.0)

# These are family-level, preregistered discrete powers.  They were frozen only
# after retaining the complete 64-case Cartesian trial, and are deliberately
# not computed from any individual sample's Rth or label.  Four slots target
# the four 30--80 K reporting intervals without filtering solved samples.
BALANCED_POWER_GRID_W = {
    "f00_near_adiabatic": (1.8, 2.4, 3.0, 3.6),
    "f01_low_low": (1.8, 2.5, 3.1, 3.7),
    "f02_750_50": (2.8, 3.7, 4.6, 5.5),
    "f03_1000_100": (3.7, 5.0, 6.2, 7.3),
    "f04_1500_200": (5.5, 7.2, 8.9, 10.7),
    "f05_2000_20": (6.8, 9.0, 11.2, 13.4),
    "f06_2500_100": (8.5, 11.0, 13.8, 16.5),
    "f07_2500_200": (8.5, 11.0, 13.8, 16.5),
}
TEMPERATURE_SLOTS = ("q1_30_42p5", "q2_42p5_55", "q3_55_67p5", "q4_67p5_80")


def prepare(sample_count: int) -> dict[str, Any]:
    if sample_count not in {64, 1024}:
        raise core.P1dError("P1d expansion count must be 64 or 1024")
    pilot = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))
    if pilot.get("sample_count") != 16 or len(pilot.get("cases", [])) != 16:
        raise core.P1dError("frozen pilot16 unavailable")
    payload = copy.deepcopy(pilot)
    payload["dataset_id"] = (
        "heat3d_v6_p1d_asymmetric_dual_robin64_balanced_v1"
        if sample_count == 64 else "heat3d_v6_p1d_asymmetric_dual_robin1024_v0"
    )
    payload["status"] = f"frozen_{sample_count}_before_generation"
    payload["sample_count"] = sample_count
    payload["scope"]["expansion_beyond_16"] = True
    payload["provenance"] = {
        "pilot16_config": str(PILOT_CONFIG.relative_to(REPO_ROOT)),
        "pilot16_config_sha256": core.sha256(PILOT_CONFIG),
        "pilot16_audit": str(PILOT_AUDIT.relative_to(REPO_ROOT)),
        "pilot16_audit_sha256": core.sha256(PILOT_AUDIT),
        "pilot16_manifest": str(PILOT_MANIFEST.relative_to(REPO_ROOT)),
        "pilot16_manifest_sha256": core.sha256(PILOT_MANIFEST),
        "trial1_complete_cartesian_audit": str(TRIAL1_AUDIT.relative_to(REPO_ROOT)),
        "trial1_complete_cartesian_audit_sha256": core.sha256(TRIAL1_AUDIT),
        "power_grid_rule": (
            "family-level discrete four-slot grid frozen from the fully retained trial1; "
            "never individual-sample Rth inversion, filtering, replacement, or seed search"
        ),
        "expansion_rule": (
            "8 BC families x 4 frozen power slots x 2 area/layout assignments"
            if sample_count == 64 else
            "8 BC families x 4 frozen power slots x 4 source areas x 8 layout seeds"
        ),
    }
    payload.pop("mesh_convergence_contract", None)
    payload["source_contract"][f"final{sample_count}_area_counts"] = {
        "16": sample_count // 4, "32": sample_count // 4,
        "48": sample_count // 4, "64": sample_count // 4,
    }
    payload["source_contract"]["layout_seeds"] = [0, 1] if sample_count == 64 else list(range(8))
    payload["source_contract"]["balanced_power_grid_W"] = {
        family: list(powers) for family, powers in BALANCED_POWER_GRID_W.items()
    }
    payload["source_contract"]["preregistered_temperature_slots"] = list(TEMPERATURE_SLOTS)
    cases = []
    family_bc = {}
    for base in pilot["cases"]:
        family_bc.setdefault(base["family_id"], (
            float(base["top_h_W_m2K"]), float(base["bottom_h_W_m2K"]),
        ))
    if set(family_bc) != set(BALANCED_POWER_GRID_W):
        raise core.P1dError("balanced power grid does not match pilot16 BC families")
    if sample_count == 64:
        for family_index, (family, powers) in enumerate(BALANCED_POWER_GRID_W.items()):
            top_h, bottom_h = family_bc[family]
            for slot_index, (slot, power) in enumerate(zip(TEMPERATURE_SLOTS, powers, strict=True)):
                # Rotating pairs give each area twice per family and 16 times globally.
                for replicate in range(2):
                    area_index = (2 * slot_index + replicate + family_index) % len(AREAS)
                    area = AREAS[area_index]
                    cases.append({
                        "id": f"p1d64b_{family}_{slot}_a{int(area)}_l{replicate}",
                        "family_id": family, "selection_bin": slot,
                        "top_h_W_m2K": top_h, "bottom_h_W_m2K": bottom_h,
                        "package_total_power_W": power,
                        "total_source_area_mm2": area, "layout_seed": replicate,
                    })
    else:
        for family, powers in BALANCED_POWER_GRID_W.items():
            top_h, bottom_h = family_bc[family]
            for slot, power in zip(TEMPERATURE_SLOTS, powers, strict=True):
                for area in AREAS:
                    for layout_seed in range(8):
                        cases.append({
                            "id": f"p1d1024_{family}_{slot}_a{int(area)}_l{layout_seed:02d}",
                            "family_id": family, "selection_bin": slot,
                            "top_h_W_m2K": top_h, "bottom_h_W_m2K": bottom_h,
                            "package_total_power_W": power,
                            "total_source_area_mm2": area, "layout_seed": layout_seed,
                        })
    if len(cases) != sample_count or len({case["id"] for case in cases}) != sample_count:
        raise core.P1dError("expansion case count mismatch")
    payload["cases"] = cases
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, choices=(64, 1024), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or (DEFAULT_64 if args.sample_count == 64 else DEFAULT_1024)
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    payload = prepare(args.sample_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:
        output_display = str(output.relative_to(REPO_ROOT))
    except ValueError:
        output_display = str(output)
    print(json.dumps({
        "output": output_display, "sample_count": len(payload["cases"]),
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
