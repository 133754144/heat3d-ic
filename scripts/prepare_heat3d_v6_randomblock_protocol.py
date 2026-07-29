#!/usr/bin/env python3
"""Prepare deterministic V6-RandomBlock smoke/pilot/formal protocol YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

import heat3d_v6_randomblock_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_randomblock"
STAGE_COUNTS = {"smoke16": 2, "pilot128": 16, "formal1024": 128}
PROTOCOL_ATTEMPT = 1
STAGE_OUTPUTS = {
    stage: CONFIG_DIR / f"v6_randomblock_{stage}_v{PROTOCOL_ATTEMPT}.yaml"
    for stage in STAGE_COUNTS
}
SEEDS = {
    "layout": 20260730,
    "physics": 20260731,
    "split": 20260732,
    "support": 20260733,
}
VARIANTS = (
    {
        "variant_id": "v0",
        "intended_temperature_bin": 0,
        "package_total_power_W": 1.8,
        "top_h_W_m2K": 333.33,
        "bottom_h_W_m2K": 333.33,
        "power_provenance": ["RB-X02", "RB-L04", "RB-L01"],
        "bc_provenance": ["RB-L05"],
    },
    {
        "variant_id": "v1",
        "intended_temperature_bin": 0,
        "package_total_power_W": 2.5,
        "top_h_W_m2K": 500.0,
        "bottom_h_W_m2K": 333.33,
        "power_provenance": ["RB-X02", "RB-L01", "RB-L03"],
        "bc_provenance": ["RB-L05"],
    },
    {
        "variant_id": "v2",
        "intended_temperature_bin": 1,
        "package_total_power_W": 7.0,
        "top_h_W_m2K": 1000.0,
        "bottom_h_W_m2K": 333.33,
        "power_provenance": ["RB-X02", "RB-L01", "RB-L03"],
        "bc_provenance": ["RB-L03", "RB-L05"],
    },
    {
        "variant_id": "v3",
        "intended_temperature_bin": 1,
        "package_total_power_W": 7.0,
        "top_h_W_m2K": 1000.0,
        "bottom_h_W_m2K": 500.0,
        "power_provenance": ["RB-X02", "RB-L03", "RB-L01"],
        "bc_provenance": ["RB-L05"],
    },
    {
        "variant_id": "v4",
        "intended_temperature_bin": 2,
        "package_total_power_W": 16.5,
        "top_h_W_m2K": 2050.0,
        "bottom_h_W_m2K": 500.0,
        "power_provenance": ["RB-X02", "RB-L03", "RB-L01", "RB-L02"],
        "bc_provenance": ["RB-L02", "RB-L05"],
    },
    {
        "variant_id": "v5",
        "intended_temperature_bin": 2,
        "package_total_power_W": 17.0,
        "top_h_W_m2K": 2050.0,
        "bottom_h_W_m2K": 1000.0,
        "power_provenance": ["RB-X02", "RB-L01", "RB-L02"],
        "bc_provenance": ["RB-L02", "RB-L03"],
    },
    {
        "variant_id": "v6",
        "intended_temperature_bin": 3,
        "package_total_power_W": 25.0,
        "top_h_W_m2K": 2500.0,
        "bottom_h_W_m2K": 1000.0,
        "power_provenance": ["RB-X02", "RB-L02"],
        "bc_provenance": ["RB-L07", "RB-L03"],
    },
    {
        "variant_id": "v7",
        "intended_temperature_bin": 3,
        "package_total_power_W": 25.0,
        "top_h_W_m2K": 2500.0,
        "bottom_h_W_m2K": 1000.0,
        "power_provenance": ["RB-X02", "RB-L01", "RB-L02"],
        "bc_provenance": ["RB-L07", "RB-L03"],
    },
)


def _stack() -> list[dict[str, Any]]:
    """Frozen P1h/P1g package stack, bottom to top."""

    return [
        {
            "id": "pcb_fr4_equivalent",
            "thickness_m": 0.0016,
            "k_xyz_W_mK": [0.8, 0.8, 0.3],
            "role": "board",
            "z_intervals": 8,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "bt_substrate_with_vias",
            "thickness_m": 0.001,
            "k_xyz_W_mK": [0.2, 0.2, 0.49],
            "role": "substrate",
            "z_intervals": 8,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "silicon_interposer_tsv_0p1",
            "thickness_m": 0.0001,
            "k_xyz_W_mK": [148.3, 148.3, 151.0],
            "role": "interposer",
            "z_intervals": 4,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "bump_underfill_under_interposer",
            "thickness_m": 0.000075,
            "k_xyz_W_mK": [0.6, 0.6, 4.9],
            "role": "interface",
            "z_intervals": 4,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "silicon_die_lower",
            "thickness_m": 0.00015,
            "k_W_mK": 120.0,
            "role": "active",
            "z_intervals": 4,
            "provenance": ["V6-P1h frozen stack", "RB-L03"],
        },
        {
            "id": "tim_between_dies",
            "thickness_m": 0.00005,
            "k_W_mK": 4.0,
            "role": "interface",
            "z_intervals": 4,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "silicon_die_upper",
            "thickness_m": 0.00015,
            "k_W_mK": 120.0,
            "role": "active",
            "z_intervals": 4,
            "provenance": ["V6-P1h frozen stack", "RB-L03"],
        },
        {
            "id": "tim_to_spreader",
            "thickness_m": 0.00005,
            "k_W_mK": 4.0,
            "role": "interface",
            "z_intervals": 4,
            "provenance": "V6-P1h frozen stack",
        },
        {
            "id": "spreader",
            "thickness_m": 0.001,
            "k_W_mK": 400.0,
            "role": "passive",
            "z_intervals": 16,
            "provenance": ["V6-P1h frozen stack", "RB-L02"],
        },
    ]


def _rng_for(group_index: int, suffix: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{SEEDS['layout']}:{group_index}:{suffix}".encode("utf-8")
    ).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(
        a[2], b[2]
    )


def _place_blocks(
    *,
    group_index: int,
    family: str,
    count: int,
    margin: float,
    total_area_mm2: float | None,
) -> list[dict[str, Any]]:
    rng = _rng_for(group_index, family)
    if family == "q":
        assert total_area_mm2 is not None
        floor = 0.5
        remaining = float(total_area_mm2) - floor * count
        if remaining <= 0.0:
            raise core.RandomBlockError("q area floor exceeds total area")
        areas = floor + remaining * rng.dirichlet(np.full(count, 5.0))
    else:
        areas = rng.uniform(0.8, 3.5, size=count)
    blocks: list[dict[str, Any]] = []
    per_layer: dict[str, list[list[float]]] = {
        layer: [] for layer in core.ACTIVE_LAYERS
    }
    for index, area in enumerate(areas):
        layer = core.ACTIVE_LAYERS[(index + group_index) % 2]
        minimum_fraction = (4.1 if family == "q" else 5.1) / 64.0
        for _ in range(4000):
            aspect = float(
                np.exp(rng.uniform(math.log(0.65), math.log(1.55)))
            )
            width = math.sqrt(float(area) * aspect) / 10.0
            height = math.sqrt(float(area) / aspect) / 10.0
            if min(width, height) >= minimum_fraction:
                break
        else:
            raise core.RandomBlockError(
                f"group {group_index}: {family} block cannot meet mesh resolution"
            )
        placed = None
        for _ in range(4000):
            x0 = float(rng.uniform(margin, 1.0 - margin - width))
            y0 = float(rng.uniform(margin, 1.0 - margin - height))
            bbox = [x0, x0 + width, y0, y0 + height]
            if not any(_overlap(bbox, other) for other in per_layer[layer]):
                placed = bbox
                break
        if placed is None:
            raise core.RandomBlockError(
                f"group {group_index}: could not place {family} block {index}"
            )
        per_layer[layer].append(placed)
        blocks.append(
            {
                "block_id": f"{family}{index:02d}",
                "layer": layer,
                "bbox_fraction_xy": [round(value, 10) for value in placed],
                "nominal_area_mm2": float(area),
            }
        )
    return blocks


def _overlap_pair_count(
    k_blocks: Sequence[Mapping[str, Any]],
    q_blocks: Sequence[Mapping[str, Any]],
) -> int:
    result = 0
    for k_block in k_blocks:
        for q_block in q_blocks:
            if (
                k_block["layer"] == q_block["layer"]
                and _overlap(
                    k_block["bbox_fraction_xy"],
                    q_block["bbox_fraction_xy"],
                )
            ):
                result += 1
    return result


def _split_roles() -> dict[str, str]:
    ranked = sorted(
        range(128),
        key=lambda index: hashlib.sha256(
            f"{SEEDS['split']}:rb_g{index:03d}".encode("utf-8")
        ).hexdigest(),
    )
    roles: dict[str, str] = {}
    for rank, index in enumerate(ranked):
        role = "train" if rank < 96 else "valid" if rank < 112 else "test"
        roles[f"rb_g{index:03d}"] = role
    return roles


def _group(group_index: int, role: str) -> dict[str, Any]:
    q_count = 3 + group_index % 8
    k_count = 2 + (group_index * 3) % 7
    total_area = 6.0 + 0.18 * math.sin(group_index * 0.73)
    q_blocks = _place_blocks(
        group_index=group_index,
        family="q",
        count=q_count,
        margin=0.05,
        total_area_mm2=total_area,
    )
    k_blocks = _place_blocks(
        group_index=group_index,
        family="k",
        count=k_count,
        margin=0.025,
        total_area_mm2=None,
    )
    identity = {
        "group_id": f"rb_g{group_index:03d}",
        "split_role": role,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "cross_family_overlap_pair_count": _overlap_pair_count(
            k_blocks, q_blocks
        ),
    }
    identity["layout_hash"] = core.canonical_json_sha256(identity)
    return identity


def _bounded_normalize(
    raw: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Scale then clip positive weights onto a box-constrained simplex."""

    raw = np.asarray(raw, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (
        raw.ndim != 1
        or raw.shape != lower.shape
        or raw.shape != upper.shape
        or np.any(raw <= 0.0)
        or np.any(lower < 0.0)
        or np.any(upper <= lower)
        or float(np.sum(lower)) > 1.0 + 1.0e-12
        or float(np.sum(upper)) < 1.0 - 1.0e-12
    ):
        raise core.RandomBlockError("infeasible source-fraction bounds")
    lo, hi = 0.0, 1.0
    while float(np.sum(np.clip(raw * hi, lower, upper))) < 1.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(raw * mid, lower, upper))) < 1.0:
            lo = mid
        else:
            hi = mid
    result = np.clip(raw * hi, lower, upper)
    # Correct the sub-ulp bisection remainder without violating bounds.
    remainder = 1.0 - float(np.sum(result))
    if remainder > 0.0:
        capacity = upper - result
    else:
        capacity = result - lower
    index = int(np.argmax(capacity))
    result[index] += remainder
    if not math.isclose(float(np.sum(result)), 1.0, abs_tol=2.0e-15):
        raise core.RandomBlockError("source fractions do not sum to one")
    return result


def _q_fractions(
    group: Mapping[str, Any],
    variant_index: int,
    group_index: int,
    total_power_W: float,
    mesh: Mapping[str, Any],
    layout_audit: Mapping[str, Any],
) -> list[float]:
    areas = np.asarray(
        [
            (
                float(block["bbox_fraction_xy"][1])
                - float(block["bbox_fraction_xy"][0])
            )
            * float(mesh["x"][-1])
            * (
                float(block["bbox_fraction_xy"][3])
                - float(block["bbox_fraction_xy"][2])
            )
            * float(mesh["y"][-1])
            for block in group["q_blocks"]
        ],
        dtype=np.float64,
    )
    volumes = np.asarray(
        [
            float(np.sum(np.asarray(mesh["weights"])[mask]))
            for mask in layout_audit["masks"]["q"]
        ],
        dtype=np.float64,
    )
    strength = 0.04 if variant_index >= 6 else 0.12
    phase = np.arange(volumes.size, dtype=np.float64) + group_index * 0.37
    multiplier = np.exp(strength * np.sin(phase + variant_index * 0.83))
    raw = volumes * multiplier
    lower = 14.5 * 1.0e4 * areas / float(total_power_W)
    upper = np.minimum(
        1000.0 * 1.0e4 * areas / float(total_power_W),
        6.666666666666667e10 * volumes / float(total_power_W),
    )
    values = _bounded_normalize(raw, lower, upper)
    return [float(value) for value in values]


def _k_values(
    group: Mapping[str, Any], variant_index: int, group_index: int
) -> list[float]:
    palette = core.K_PALETTE_W_MK
    return [
        float(palette[(group_index * 3 + index * 2 + variant_index) % len(palette)])
        for index in range(len(group["k_blocks"]))
    ]


def prepare(stage: str) -> dict[str, Any]:
    if stage not in STAGE_COUNTS:
        raise core.RandomBlockError(f"unsupported stage: {stage}")
    roles = _split_roles()
    if stage == "smoke16":
        indices = [0, 23]
    else:
        indices = list(range(STAGE_COUNTS[stage]))
    physics = {
        "footprint_m": [0.01, 0.01],
        "layers_bottom_to_top": _stack(),
        "solver_mesh_intervals_xyz": [64, 64, 56],
        "boundary_schema": {
            "top": "robin",
            "bottom": "robin",
            "sides": "adiabatic",
            "top_T_inf_K": 300.0,
            "bottom_T_inf_K": 300.0,
            "contact": "perfect",
        },
        "operator_projection": {
            "point_count": 1024,
            "strata": {
                "volume": 512,
                "block": 256,
                "interface": 128,
                "top": 64,
                "bottom": 64,
            },
            "selection_uses_temperature_or_labels": False,
            "shared_within_layout_group": True,
        },
        "literature_contract": "configs/heat3d_v6_randomblock/v6_randomblock_literature.json",
        "k_block_palette_W_mK": list(core.K_PALETTE_W_MK),
        "surface_power_density_W_cm2": [14.5, 1000.0],
        "volumetric_q_max_W_m3": 6.666666666666667e10,
        "package_power_range_W": [1.0, 47.8],
    }
    groups = [_group(index, roles[f"rb_g{index:03d}"]) for index in indices]
    mesh = core.build_mesh(physics)
    layout_audits = {
        str(group["group_id"]): core.validate_layout(group, mesh)
        for group in groups
    }
    cases: list[dict[str, Any]] = []
    for group_index, group in zip(indices, groups):
        for variant_index, variant in enumerate(VARIANTS):
            cases.append(
                {
                    "sample_id": f"{group['group_id']}_{variant['variant_id']}",
                    "group_id": group["group_id"],
                    "split_role": group["split_role"],
                    **variant,
                    "k_block_values_W_mK": _k_values(
                        group, variant_index, group_index
                    ),
                    "q_block_power_fractions": _q_fractions(
                        group,
                        variant_index,
                        group_index,
                        float(variant["package_total_power_W"]),
                        mesh,
                        layout_audits[str(group["group_id"])],
                    ),
                    "k_provenance": [
                        "RB-L01",
                        "RB-L02",
                        "RB-L03",
                        "RB-L04",
                        "RB-X03",
                    ],
                    "q_provenance": ["RB-L02", "RB-L03", "RB-X01"],
                }
            )
    dataset_id = f"heat3d_v6_randomblock_{stage}_v{PROTOCOL_ATTEMPT}"
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "stage": stage,
        "protocol_attempt": PROTOCOL_ATTEMPT,
        "sample_count": len(cases),
        "group_count": len(groups),
        "variants_per_group": 8,
        "seeds": dict(SEEDS),
        "physics": physics,
        "temperature_bin_contract": {
            "edges_K": list(core.TEMPERATURE_BIN_EDGES_K),
            "intended_slots_per_group": [2, 2, 2, 2],
            "post_solve_filtering": False,
            "post_solve_replacement": False,
            "pilot_may_change_only_global_rules_then_restart": True,
        },
        "layout_groups": groups,
        "physics_variants": [dict(value) for value in VARIANTS],
        "cases": cases,
        "guardrails": {
            "training": False,
            "model_inference": False,
            "post_solve_temperature_filtering": False,
            "post_solve_sample_replacement": False,
            "group_split_leakage": False,
            "canonical_p1h_modified": False,
        },
        "provenance": {
            "base_main_commit": "332ef3f463d91442632c3ebddd4f7549c7895b8d",
            "literature_schema": "heat3d_v6_randomblock_literature_v1",
            "protocol_generator": "scripts/prepare_heat3d_v6_randomblock_protocol.py",
            "protocol_hash_excludes": ["provenance.protocol_sha256"],
            "global_rule_revision": {
                "source_attempt": "smoke16_v0",
                "source_manifest_sha256": "a32850d10775d346bd4764765335c91b088e78dfbb18001a24bb02675ce9ada2",
                "method": "global per-variant power table adjusted from two-group aggregate peak response; no per-sample inverse calibration, filtering, replacement, or BC change",
            },
        },
    }
    payload["provenance"]["protocol_sha256"] = core.canonical_json_sha256(
        {
            **payload,
            "provenance": {
                key: value
                for key, value in payload["provenance"].items()
                if key != "protocol_sha256"
            },
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_COUNTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload = prepare(args.stage)
    output = args.output or STAGE_OUTPUTS[args.stage]
    output = output if output.is_absolute() else ROOT / output
    if args.check_only:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "sample_count": payload["sample_count"],
                    "group_count": payload["group_count"],
                    "protocol_sha256": payload["provenance"]["protocol_sha256"],
                    "output": str(output.relative_to(ROOT)),
                    "write_executed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "sample_count": payload["sample_count"],
                "group_count": payload["group_count"],
                "protocol_sha256": payload["provenance"]["protocol_sha256"],
                "output": str(output.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
