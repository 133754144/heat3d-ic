#!/usr/bin/env python3
"""Versioned V6-P1i v8 generator adapter.

The frozen v0/v1/v2 generator and core are imported read-only. This adapter
retains the v3 split/support contract and adds one population-level monotonic
severity curve to the power rule. It never reads targets when constructing a
case.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

import generate_heat3d_v6_p1i_continuous_dataset as base
import heat3d_v6_p1i_continuous_core as core


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    ROOT / "configs/heat3d_v6_p1i/v6_p1i_pilot128_v8.yaml"
)
_ORIGINAL_CASE_BUILDER = base._case_from_sobol
_ORIGINAL_SPLIT_MAP = base._split_map
_ORIGINAL_SUPPORT_SELECTOR = core.select_group_support


def case_from_sobol_v3(
    index: int,
    values: np.ndarray,
    config: Mapping[str, Any],
    split_role: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cursor = base.Cursor(values)
    physics = copy.deepcopy(dict(config["physics"]))
    layers, background_rows = base._sample_layer_backgrounds(
        physics["layers_bottom_to_top"], cursor
    )
    physics["layers_bottom_to_top"] = layers
    sampling = config["sampling"]
    top_h = base._log_lerp(
        sampling["top_h_W_m2K"]["range"], cursor.take()
    )
    bottom_h = base._log_lerp(
        sampling["bottom_h_W_m2K"]["range"], cursor.take()
    )
    severity = cursor.take()
    independent_power_latent = cursor.take()
    power_spec = sampling["power_W"]
    if (
        power_spec["distribution"]
        != "monotonic_curved_severity_top_h_power_law_with_independent_jitter"
    ):
        raise core.ContinuousPhysicsError("v8 power distribution mismatch")
    severity_curve_exponent = float(power_spec["severity_curve_exponent"])
    if not 0.0 < severity_curve_exponent <= 1.0:
        raise core.ContinuousPhysicsError(
            "v8 severity curve exponent must be in (0, 1]"
        )
    base_power = base._lerp(
        power_spec["base_range_W"],
        severity ** severity_curve_exponent,
    )
    top_h_factor = (
        top_h / float(power_spec["top_h_reference_W_m2K"])
    ) ** float(power_spec["top_h_exponent"])
    multiplier_spec = power_spec["independent_multiplier"]
    if multiplier_spec["distribution"] != "log_uniform":
        raise core.ContinuousPhysicsError("v8 independent multiplier mismatch")
    independent_multiplier = base._log_lerp(
        multiplier_spec["range"], independent_power_latent
    )
    power = base_power * top_h_factor * independent_multiplier
    low, high = map(float, power_spec["allowed_range_W"])
    if not low <= power <= high:
        raise core.ContinuousPhysicsError(
            f"v8 global power rule produced {power} W outside {low, high}"
        )
    source_count = base._count(
        sampling["source_count"]["range_inclusive"], cursor.take()
    )
    source_layer_probability = base._lerp(
        sampling["source_layer_upper_probability"]["range"], cursor.take()
    )
    source_spec = sampling["source_planform_fraction"]
    if float(source_spec.get("size_severity_weight", 0.0)) != 0.0:
        raise core.ContinuousPhysicsError(
            "v3 source size must be independent of severity"
        )
    q_blocks = base._rectangles(
        cursor,
        count=source_count,
        width_range=source_spec["width_range"],
        height_range=source_spec["height_range"],
        margin=float(source_spec["center_margin"]),
        gap=float(source_spec["same_family_gap"]),
        layer_probability=source_layer_probability,
        size_bias=None,
        size_bias_weight=0.0,
    )
    intensity_bounds = sampling["source_power_heterogeneity"]["range"]
    raw_power = []
    for block in q_blocks:
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        raw_power.append(
            (x1 - x0)
            * (y1 - y0)
            * base._log_lerp(intensity_bounds, cursor.take())
        )
    q_fractions = np.asarray(raw_power, dtype=np.float64)
    q_fractions /= np.sum(q_fractions)
    k_count = base._count(
        sampling["k_region_count"]["range_inclusive"], cursor.take()
    )
    k_spec = sampling["k_region_planform_fraction"]
    k_blocks = base._rectangles(
        cursor,
        count=k_count,
        width_range=k_spec["width_range"],
        height_range=k_spec["height_range"],
        margin=float(k_spec["center_margin"]),
        gap=float(k_spec["same_family_gap"]),
        layer_probability=cursor.take(),
    )
    k_values = [
        base._log_lerp(
            sampling["local_k_W_mK"]["range"], cursor.take()
        )
        for _ in k_blocks
    ]
    sample_id = f"{config['sample_id_prefix']}{index:04d}"
    group = {
        "group_id": sample_id,
        "split_role": split_role,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "cross_family_overlap_pair_count": base._cross_overlap_count(
            k_blocks, q_blocks
        ),
    }
    case = {
        "sample_id": sample_id,
        "group_id": sample_id,
        "split_role": split_role,
        "package_total_power_W": power,
        "power_base_W": base_power,
        "power_top_h_factor": top_h_factor,
        "power_independent_latent": independent_power_latent,
        "power_independent_multiplier": independent_multiplier,
        "continuous_severity": severity,
        "top_h_W_m2K": top_h,
        "bottom_h_W_m2K": bottom_h,
        "k_block_values_W_mK": k_values,
        "q_block_power_fractions": q_fractions.tolist(),
        "q_bounds_W_m3": [1.0e8, 8.0e10],
        "surface_power_density_bounds_W_cm2": [1.0, 1000.0],
        "sobol_index": index,
        "sobol_dimensions_consumed": cursor.offset,
        "source_size_coupled_to_severity": False,
    }
    latent = {
        "sample_id": sample_id,
        "sobol_index": index,
        "sobol_dimensions_consumed": cursor.offset,
        "source_count": source_count,
        "k_region_count": k_count,
        "source_layer_upper_probability": source_layer_probability,
        "top_h_W_m2K": top_h,
        "bottom_h_W_m2K": bottom_h,
        "package_total_power_W": power,
        "continuous_severity": severity,
        "power_independent_latent": independent_power_latent,
        "power_independent_multiplier": independent_multiplier,
    }
    for row in background_rows:
        row["sample_id"] = sample_id
    return case, group, physics, background_rows


def split_map_v3(
    sample_ids: list[str],
    counts: Mapping[str, int],
    split_spec: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if (
        not split_spec
        or split_spec.get("method")
        != "frozen_balanced_pre_solve_assignment_v1"
        or split_spec.get("target_values_used") is not False
    ):
        raise core.ContinuousPhysicsError("v3 frozen split contract mismatch")
    path = ROOT / str(split_spec["assignment_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignment = {
        str(key): str(value) for key, value in payload["assignment"].items()
    }
    if set(assignment) != set(sample_ids):
        raise core.ContinuousPhysicsError("v3 split sample IDs mismatch")
    if core.canonical_json_sha256(dict(sorted(assignment.items()))) != payload[
        "assignment_sha256"
    ]:
        raise core.ContinuousPhysicsError("v3 split assignment SHA drift")
    realized = {
        role: sum(value == role for value in assignment.values())
        for role in ("train", "valid_iid", "test_iid")
    }
    if realized != {key: int(value) for key, value in counts.items()}:
        raise core.ContinuousPhysicsError(
            f"v3 split count mismatch: {realized}"
        )
    if payload["target_values_used"] is not False:
        raise core.ContinuousPhysicsError("v3 split used targets")
    return assignment


def select_support_v3(
    group: Mapping[str, Any],
    mesh: Mapping[str, Any],
    layout_audit: Mapping[str, Any],
    support_seed: int,
) -> dict[str, Any]:
    for attempt in range(64):
        result = _ORIGINAL_SUPPORT_SELECTOR(
            group,
            mesh,
            layout_audit,
            int(support_seed) + 104729 * attempt,
        )
        if min(result["block_coverage"]) >= 4:
            result["selection_attempt"] = attempt
            result["minimum_required_block_coverage"] = 4
            return result
    raise core.ContinuousPhysicsError(
        f"{group['group_id']}: unable to obtain >=4 support per local region"
    )


@contextmanager
def patched_v3_runtime() -> Iterator[None]:
    base._case_from_sobol = case_from_sobol_v3
    base._split_map = split_map_v3
    core.select_group_support = select_support_v3
    try:
        yield
    finally:
        base._case_from_sobol = _ORIGINAL_CASE_BUILDER
        base._split_map = _ORIGINAL_SPLIT_MAP
        core.select_group_support = _ORIGINAL_SUPPORT_SELECTOR


def preflight(config_path: Path) -> dict[str, Any]:
    with patched_v3_runtime():
        result = base.preflight(config_path)
    result["generator_adapter"] = str(Path(__file__).relative_to(ROOT))
    result["minimum_support_contract"] = 4
    return result


def generate(
    config_path: Path,
    output_root: Path,
    *,
    replace: bool,
    resume: bool,
) -> dict[str, Any]:
    with patched_v3_runtime():
        return base.generate(
            config_path,
            output_root,
            replace=replace,
            resume=resume,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    if args.preflight_only:
        print(json.dumps(preflight(config), indent=2))
        return 0
    manifest = generate(
        config,
        args.output_root.resolve(),
        replace=args.replace,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "dataset_id",
                    "sample_count",
                    "dataset_root",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
