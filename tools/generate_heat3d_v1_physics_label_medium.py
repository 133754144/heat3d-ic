#!/usr/bin/env python3
"""Generate Heat3D v1 physics-label medium-style subsets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_heat3d_v1_physics_label_medium_expansion import (  # noqa: E402
    _read_json,
    _select_samples,
    _write_sample,
)


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium_manifest.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_v2"
)
DEFAULT_MEDIUM256_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium256_v2"
)
PROTECTED_SUBSET_NAMES = {
    "v1_multilayer_bc_eq_demo",
    "v1_multilayer_bc_eq_supervised_smoke",
    "v1_multilayer_bc_eq_supervised_small",
    "v1_multilayer_bc_eq_physics_label_small_v2",
    "v1_multilayer_bc_eq_physics_label_medium_pilot_v2",
    "v1_multilayer_bc_eq_physics_label_medium_expansion_v2",
}
GAP_A_VARIANT_VERSION = "medium1024_gapA_diversity_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Heat3D v1 physics-label medium-style subsets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-subset",
        type=Path,
        default=None,
        help=(
            "Output subset path. Defaults to medium_v2 for the original manifest "
            "and medium256_v2 for the medium256 manifest."
        ),
    )
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--sample-limit", "--max-samples", dest="sample_limit", type=int, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_output_path(path: Path, overwrite: bool) -> Path:
    output_subset = path.resolve()
    if output_subset.name in PROTECTED_SUBSET_NAMES:
        raise ValueError(f"refusing to write protected subset: {output_subset.name}")
    try:
        output_subset.relative_to(REPO_ROOT / "data")
    except ValueError as exc:
        raise ValueError(f"output subset must be under ignored data/: {output_subset}") from exc
    if output_subset.exists() and not overwrite:
        raise FileExistsError(f"output subset exists: {output_subset}; use --overwrite")
    return output_subset


def _default_output_subset_for_manifest(manifest_path: Path) -> Path:
    if manifest_path.name == "heat3d_v1_physics_label_medium256_manifest.json":
        return DEFAULT_MEDIUM256_OUTPUT_SUBSET
    return DEFAULT_OUTPUT_SUBSET


def _is_gap_a_manifest(manifest: dict[str, Any]) -> bool:
    plan = manifest.get("sample_generation_plan", {})
    return isinstance(plan, dict) and plan.get("strategy") == "gapA_deterministic_balanced_cycle"


def _apply_sample_limit(
    samples: list[dict],
    sample_limit: int | None,
    manifest: dict[str, Any],
    *,
    balanced: bool,
) -> list[dict]:
    if sample_limit is None:
        return samples
    if sample_limit < 1:
        raise ValueError("--sample-limit must be >= 1")
    if sample_limit > len(samples):
        raise ValueError(f"--sample-limit must be <= {len(samples)}")
    if balanced and _is_gap_a_manifest(manifest):
        return _balanced_gap_a_sample_limit(samples, sample_limit, manifest)
    return samples[:sample_limit]


def _repeat_counts(counts: dict[str, Any]) -> list[str]:
    values: list[str] = []
    remaining: dict[str, int] = {}
    for key, count in counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid count for {key!r}: {count!r}")
        remaining[str(key)] = count
    while any(count > 0 for count in remaining.values()):
        for key in remaining:
            if remaining[key] > 0:
                values.append(key)
                remaining[key] -= 1
    return values


def _active_layer_for_stack(stack_template: str, secondary: bool = False) -> str:
    if secondary and stack_template == "dual_active_4_layer":
        return "active_die_1"
    return "active_die_0"


def _unit_interval(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _range_value(seed: int, key: str, low: float, high: float) -> float:
    return low + (high - low) * _unit_interval(seed, key)


def _signed_value(seed: int, key: str, amplitude: float) -> float:
    return (2.0 * _unit_interval(seed, key) - 1.0) * amplitude


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pattern_seed(sample_id: str, index: int, split: str, source: str, k_region: str, k_field: str, stack: str, bc: str) -> int:
    key = f"{GAP_A_VARIANT_VERSION}:{sample_id}:{index}:{split}:{source}:{k_region}:{k_field}:{stack}:{bc}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _base_source_regions_for_pattern(source_pattern: str, stack_template: str, index: int) -> list[dict[str, Any]]:
    active0 = _active_layer_for_stack(stack_template)
    active1 = _active_layer_for_stack(stack_template, secondary=True)
    jitter = (index % 5) * 0.025
    if source_pattern == "centered_single_hotspot":
        return [{
            "region_id": "src_center",
            "layer": active0,
            "center_xy_fraction": [0.50, 0.50],
            "size_xy_fraction": [0.22, 0.22],
            "q_scale_category": "nominal",
        }]
    if source_pattern == "shifted_single_hotspot":
        return [{
            "region_id": "src_shifted",
            "layer": active0,
            "center_xy_fraction": [0.34 + jitter, 0.58 - 0.5 * jitter],
            "size_xy_fraction": [0.20, 0.22],
            "q_scale_category": "low",
        }]
    if source_pattern == "edge_or_corner_hotspot":
        return [{
            "region_id": "src_edge_corner",
            "layer": active0,
            "center_xy_fraction": [0.75, 0.25 + jitter],
            "size_xy_fraction": [0.18, 0.18],
            "q_scale_category": "high",
        }]
    if source_pattern == "two_hotspots_same_layer":
        return [
            {
                "region_id": "src_left",
                "layer": active0,
                "center_xy_fraction": [0.30, 0.50],
                "size_xy_fraction": [0.18, 0.18],
                "q_scale_category": "nominal",
            },
            {
                "region_id": "src_right",
                "layer": active0,
                "center_xy_fraction": [0.70, 0.50],
                "size_xy_fraction": [0.18, 0.18],
                "q_scale_category": "low",
            },
        ]
    if source_pattern == "dual_active_layer_hotspots":
        return [
            {
                "region_id": "src_active0",
                "layer": active0,
                "center_xy_fraction": [0.34, 0.42],
                "size_xy_fraction": [0.18, 0.18],
                "q_scale_category": "nominal",
            },
            {
                "region_id": "src_active1",
                "layer": active1,
                "center_xy_fraction": [0.66, 0.60],
                "size_xy_fraction": [0.18, 0.18],
                "q_scale_category": "low",
            },
        ]
    if source_pattern == "broad_block_power":
        return [{
            "region_id": "src_broad",
            "layer": active0,
            "center_xy_fraction": [0.50, 0.50],
            "size_xy_fraction": [0.44, 0.36],
            "q_scale_category": "low",
        }]
    if source_pattern == "multi_block_power":
        return [
            {
                "region_id": "src_multi_a",
                "layer": active0,
                "center_xy_fraction": [0.28, 0.30],
                "size_xy_fraction": [0.16, 0.16],
                "q_scale_category": "nominal",
            },
            {
                "region_id": "src_multi_b",
                "layer": active0,
                "center_xy_fraction": [0.70, 0.42],
                "size_xy_fraction": [0.18, 0.16],
                "q_scale_category": "low",
            },
            {
                "region_id": "src_multi_c",
                "layer": active0,
                "center_xy_fraction": [0.48, 0.72],
                "size_xy_fraction": [0.16, 0.18],
                "q_scale_category": "high",
            },
        ]
    if source_pattern == "low_power_near_zero_background_cases":
        return [
            {
                "region_id": "src_low_background_a",
                "layer": active0,
                "center_xy_fraction": [0.38 + 0.5 * jitter, 0.46],
                "size_xy_fraction": [0.20, 0.20],
                "q_scale_category": "very_low",
            },
            {
                "region_id": "src_low_background_b",
                "layer": active0,
                "center_xy_fraction": [0.62, 0.64 - 0.5 * jitter],
                "size_xy_fraction": [0.16, 0.16],
                "q_scale_category": "trace",
            },
        ]
    if source_pattern == "high_dynamic_range_power_cases":
        return [
            {
                "region_id": "src_hdr_hotspot",
                "layer": active0,
                "center_xy_fraction": [0.42 + 0.5 * jitter, 0.48],
                "size_xy_fraction": [0.14, 0.14],
                "q_scale_category": "very_high",
            },
            {
                "region_id": "src_hdr_background",
                "layer": active0,
                "center_xy_fraction": [0.58, 0.56],
                "size_xy_fraction": [0.46, 0.38],
                "q_scale_category": "trace",
            },
        ]
    raise ValueError(f"unsupported source_pattern_tag for generated samples: {source_pattern}")


def _source_scale_range(source_pattern: str, q_category: str) -> tuple[float, float]:
    if source_pattern == "low_power_near_zero_background_cases":
        return (0.70, 1.45)
    if source_pattern == "high_dynamic_range_power_cases":
        if q_category == "very_high":
            return (0.85, 1.30)
        return (0.55, 1.65)
    if q_category == "trace":
        return (0.75, 1.35)
    if q_category == "very_low":
        return (0.75, 1.40)
    if q_category == "high":
        return (0.85, 1.22)
    return (0.85, 1.15)


def _apply_gap_a_source_variation(
    regions: list[dict[str, Any]],
    source_pattern: str,
    pattern_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    varied: list[dict[str, Any]] = []
    global_dx = _signed_value(pattern_seed, "global_dx", 0.035)
    global_dy = _signed_value(pattern_seed, "global_dy", 0.035)
    base_sx = _range_value(pattern_seed, "global_sx", 0.88, 1.18)
    base_sy = _range_value(pattern_seed, "global_sy", 0.88, 1.18)
    q_scales: list[float] = []

    for region_index, region in enumerate(regions):
        item = dict(region)
        cx, cy = item["center_xy_fraction"]
        sx, sy = item["size_xy_fraction"]
        local_dx = _signed_value(pattern_seed, f"region_{region_index}_dx", 0.028)
        local_dy = _signed_value(pattern_seed, f"region_{region_index}_dy", 0.028)
        local_sx = base_sx * _range_value(pattern_seed, f"region_{region_index}_sx", 0.92, 1.12)
        local_sy = base_sy * _range_value(pattern_seed, f"region_{region_index}_sy", 0.92, 1.12)
        new_sx = _clamp(float(sx) * local_sx, 0.145, 0.56)
        new_sy = _clamp(float(sy) * local_sy, 0.145, 0.56)
        item["center_xy_fraction"] = [
            _clamp(float(cx) + global_dx + local_dx, 0.14, 0.86),
            _clamp(float(cy) + global_dy + local_dy, 0.14, 0.86),
        ]
        item["size_xy_fraction"] = [new_sx, new_sy]
        z_shrink = _range_value(pattern_seed, f"region_{region_index}_z_span", 0.82, 1.0)
        z_center = _clamp(0.50 + _signed_value(pattern_seed, f"region_{region_index}_z_center", 0.08), 0.35, 0.65)
        item["z_center_fraction"] = z_center
        item["z_span_fraction"] = z_shrink
        scale_low, scale_high = _source_scale_range(source_pattern, str(item["q_scale_category"]))
        q_scale = _range_value(pattern_seed, f"region_{region_index}_q_scale", scale_low, scale_high)
        item["q_density_scale"] = q_scale
        item["geometry_variant_id"] = int(pattern_seed % 1_000_000) + region_index
        q_scales.append(q_scale)
        varied.append(item)

    if source_pattern == "broad_block_power" and _unit_interval(pattern_seed, "broad_aux") > 0.45:
        aux = dict(varied[0])
        aux["region_id"] = "src_broad_aux"
        aux["center_xy_fraction"] = [
            _clamp(0.28 + _signed_value(pattern_seed, "broad_aux_dx", 0.09), 0.14, 0.86),
            _clamp(0.70 + _signed_value(pattern_seed, "broad_aux_dy", 0.09), 0.14, 0.86),
        ]
        aux["size_xy_fraction"] = [
            _range_value(pattern_seed, "broad_aux_sx", 0.16, 0.24),
            _range_value(pattern_seed, "broad_aux_sy", 0.16, 0.24),
        ]
        aux["q_scale_category"] = "trace"
        aux["q_density_scale"] = _range_value(pattern_seed, "broad_aux_q", 0.60, 1.30)
        aux["geometry_variant_id"] = int(pattern_seed % 1_000_000) + 99
        q_scales.append(float(aux["q_density_scale"]))
        varied.append(aux)

    if source_pattern == "multi_block_power" and _unit_interval(pattern_seed, "multi_aux") > 0.35:
        active_layer = varied[0]["layer"]
        aux = {
            "region_id": "src_multi_d",
            "layer": active_layer,
            "center_xy_fraction": [
                _range_value(pattern_seed, "multi_aux_cx", 0.18, 0.82),
                _range_value(pattern_seed, "multi_aux_cy", 0.18, 0.82),
            ],
            "size_xy_fraction": [
                _range_value(pattern_seed, "multi_aux_sx", 0.145, 0.22),
                _range_value(pattern_seed, "multi_aux_sy", 0.145, 0.22),
            ],
            "q_scale_category": "low",
            "q_density_scale": _range_value(pattern_seed, "multi_aux_q", 0.75, 1.25),
            "geometry_variant_id": int(pattern_seed % 1_000_000) + 199,
            "z_center_fraction": _range_value(pattern_seed, "multi_aux_zc", 0.42, 0.58),
            "z_span_fraction": _range_value(pattern_seed, "multi_aux_zs", 0.82, 1.0),
        }
        q_scales.append(float(aux["q_density_scale"]))
        varied.append(aux)

    metadata = {
        "source_center_shift": [global_dx, global_dy],
        "source_size_scale": [base_sx, base_sy],
        "q_scale_factor": float(sum(q_scales) / max(len(q_scales), 1)),
        "q_geometry_variant": int(pattern_seed % 1_000_000),
        "source_region_variant_count": len(varied),
    }
    return varied, metadata


def _source_regions_for_pattern(
    source_pattern: str,
    stack_template: str,
    index: int,
    pattern_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    regions = _base_source_regions_for_pattern(source_pattern, stack_template, index)
    if pattern_seed is None:
        return regions, {}
    return _apply_gap_a_source_variation(regions, source_pattern, pattern_seed)


def _power_scale_category(source_pattern: str) -> str:
    if source_pattern == "low_power_near_zero_background_cases":
        return "low_power"
    if source_pattern == "high_dynamic_range_power_cases":
        return "high_dynamic_range"
    return "nominal"


def _top_h_value_for_category(bc: str, pattern_seed: int) -> float:
    ranges = {
        "nominal_top_h": (900.0, 1100.0),
        "low_top_h": (420.0, 640.0),
        "high_top_h": (1320.0, 1720.0),
        "held_out_top_h_candidate": (1820.0, 2300.0),
        "very_low_top_h_candidate": (180.0, 320.0),
        "very_high_top_h_candidate": (2600.0, 3400.0),
    }
    low, high = ranges.get(bc, (900.0, 1100.0))
    return _range_value(pattern_seed, "top_h", low, high)


def _k_variation(k_region: str, pattern_seed: int) -> dict[str, Any]:
    layer_names = (
        "substrate",
        "active_die_0",
        "active_die_1",
        "tim_equivalent",
        "interposer_equivalent",
        "interposer_like_equivalent",
        "heat_spreader_equivalent",
    )
    layer_scales = {
        name: _range_value(pattern_seed, f"k_layer_{name}", 0.88, 1.16)
        for name in layer_names
    }
    variation = {
        "layer_scale_factors": layer_scales,
        "block_x_threshold": _range_value(pattern_seed, "block_x_threshold", 0.0043, 0.0057),
        "block_y_threshold": _range_value(pattern_seed, "block_y_threshold", 0.0043, 0.0064),
        "block_low_scale": _range_value(pattern_seed, "block_low_scale", 0.64, 0.88),
        "block_high_scale": _range_value(pattern_seed, "block_high_scale", 1.08, 1.36),
        "interposer_scale": _range_value(pattern_seed, "interposer_scale", 0.86, 1.24),
        "diag_ratios": [
            _range_value(pattern_seed, "diag_x", 1.05, 1.35),
            _range_value(pattern_seed, "diag_y", 0.78, 1.05),
            _range_value(pattern_seed, "diag_z", 0.45, 0.72),
        ],
        "high_contrast_scale": _range_value(pattern_seed, "high_contrast_scale", 0.85, 1.22),
        "barrier_scale": _range_value(pattern_seed, "barrier_scale", 0.72, 1.18),
    }
    if k_region == "high_contrast_interface_k":
        variation["k_contrast_category"] = "high_contrast"
    if k_region == "low_k_barrier_or_TIM_variation":
        variation["barrier_k_category"] = "low_k"
    return variation


def _sample_from_conditions(index: int, split: str, source: str, k_region: str, k_field: str, stack: str, bc: str) -> dict[str, Any]:
    sample_id = f"medium_gapA_{index:04d}"
    pattern_seed = _pattern_seed(sample_id, index, split, source, k_region, k_field, stack, bc)
    source_regions, source_variant = _source_regions_for_pattern(source, stack, index, pattern_seed)
    k_variation = _k_variation(k_region, pattern_seed)
    top_h_value = _top_h_value_for_category(bc, pattern_seed)
    sample = {
        "sample_id": sample_id,
        "split": split,
        "stack_template": stack,
        "source_regions": source_regions,
        "source_pattern_tag": source,
        "power_scale_category": _power_scale_category(source),
        "q_policy": "fixed_density",
        "source_assignment": "volume_fraction",
        "k_region_mode": k_region,
        "k_field_mode": k_field,
        "bc_category": bc,
        "top_h_value": top_h_value,
        "bc_value_variant": {"top_h_W_m2K": top_h_value},
        "variant_id": f"gapA_v2_{index:04d}",
        "pattern_seed": pattern_seed,
        "generation_variant_version": GAP_A_VARIANT_VERSION,
        "k_scale_factor": float(sum(k_variation["layer_scale_factors"].values()) / len(k_variation["layer_scale_factors"])),
        "k_variant_id": int((pattern_seed >> 16) % 1_000_000),
        "k_variation": k_variation,
        **source_variant,
        "resolution_category": "medium_expansion_mid",
        "purpose_tag": f"{split} {source} / {k_region} / {stack} / {bc}",
    }
    if k_region == "high_contrast_interface_k":
        sample["k_contrast_category"] = "high_contrast"
    if k_region == "low_k_barrier_or_TIM_variation":
        sample["barrier_k_category"] = "low_k"
    return sample


def _build_gap_a_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    plan = manifest.get("sample_generation_plan", {})
    if plan.get("strategy") != "gapA_deterministic_balanced_cycle":
        raise ValueError("unsupported generated-sample manifest strategy")

    split_counts = dict(manifest.get("split_counts", {}))
    coverage = dict(manifest.get("coverage_summary_planned", {}))
    target_count = int(plan.get("target_sample_count", sum(int(v) for v in split_counts.values())))

    bc_counts = dict(coverage.get("bc_category", {}))
    stack_counts = dict(coverage.get("stack_template", {}))
    policy = manifest.get("candidate_split_policy", {})
    held_out_bc = set(policy.get("held_out_bc_categories", []))
    held_out_stack = set(policy.get("held_out_stack_templates", []))

    probes = []
    for probe in plan.get("smoke_probe_conditions", []):
        probes.append(
            _sample_from_conditions(
                len(probes),
                str(probe["split"]),
                str(probe["source_pattern_tag"]),
                str(probe["k_region_mode"]),
                str(probe["k_field_mode"]),
                str(probe["stack_template"]),
                str(probe["bc_category"]),
            )
        )

    counters = Counter()
    for sample in probes:
        counters[("split", sample["split"])] += 1
        counters[("source", sample["source_pattern_tag"])] += 1
        counters[("k_region", sample["k_region_mode"])] += 1
        counters[("k_field", sample["k_field_mode"])] += 1
        counters[("stack", sample["stack_template"])] += 1
        counters[("bc", sample["bc_category"])] += 1

    split_remaining = []
    for split, count in split_counts.items():
        remaining = int(count) - counters[("split", split)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed split count for {split}")
        split_remaining.extend([split] * remaining)
    source_remaining = []
    for source, count in coverage.get("source_pattern_tag", {}).items():
        remaining = int(count) - counters[("source", source)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed source count for {source}")
        source_remaining.extend([source] * remaining)
    k_region_remaining = []
    for k_region, count in coverage.get("k_region_mode", {}).items():
        remaining = int(count) - counters[("k_region", k_region)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed k-region count for {k_region}")
        k_region_remaining.extend([k_region] * remaining)
    k_field_remaining = []
    for k_field, count in coverage.get("k_field_mode", {}).items():
        remaining = int(count) - counters[("k_field", k_field)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed k-field count for {k_field}")
        k_field_remaining.extend([k_field] * remaining)

    regular_bc_remaining = []
    held_out_bc_remaining = []
    for bc, count in bc_counts.items():
        remaining = int(count) - counters[("bc", bc)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed BC count for {bc}")
        if bc in held_out_bc:
            held_out_bc_remaining.extend([bc] * remaining)
        else:
            regular_bc_remaining.extend([bc] * remaining)
    regular_stack_remaining = []
    held_out_stack_remaining = []
    for stack, count in stack_counts.items():
        remaining = int(count) - counters[("stack", stack)]
        if remaining < 0:
            raise ValueError(f"smoke probes exceed stack count for {stack}")
        if stack in held_out_stack:
            held_out_stack_remaining.extend([stack] * remaining)
        else:
            regular_stack_remaining.extend([stack] * remaining)

    expected_remaining = target_count - len(probes)
    if len(split_remaining) != expected_remaining:
        raise ValueError("split remaining count does not match target")
    for name, remaining in (
        ("source", source_remaining),
        ("k_region", k_region_remaining),
        ("k_field", k_field_remaining),
    ):
        if len(remaining) != expected_remaining:
            raise ValueError(f"{name} remaining count does not match target")

    samples = list(probes)
    regular_bc_index = 0
    held_out_bc_index = 0
    regular_stack_index = 0
    held_out_stack_index = 0
    for offset, split in enumerate(split_remaining):
        sample_index = len(samples)
        source = source_remaining[offset]
        k_region = k_region_remaining[offset]
        k_field = k_field_remaining[offset]
        if split in {"test_ood_bc_candidate", "test_ood_combined_candidate"}:
            bc = held_out_bc_remaining[held_out_bc_index]
            held_out_bc_index += 1
        else:
            bc = regular_bc_remaining[regular_bc_index]
            regular_bc_index += 1
        if split in {"test_ood_stack_candidate", "test_ood_combined_candidate"}:
            stack = held_out_stack_remaining[held_out_stack_index]
            held_out_stack_index += 1
        else:
            stack = regular_stack_remaining[regular_stack_index]
            regular_stack_index += 1
        samples.append(_sample_from_conditions(sample_index, split, source, k_region, k_field, stack, bc))

    if len(samples) != target_count:
        raise ValueError(f"generated sample count must be {target_count}, found {len(samples)}")
    return samples


def _largest_remainder_counts(total: int, split_counts: dict[str, Any]) -> dict[str, int]:
    manifest_total = sum(int(value) for value in split_counts.values())
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for split, count in split_counts.items():
        exact = total * int(count) / manifest_total
        floor = int(exact)
        floors[str(split)] = floor
        remainders.append((exact - floor, str(split)))
    remaining = total - sum(floors.values())
    for _, split in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        floors[split] += 1
    return floors


def _target_split_counts(
    sample_limit: int,
    split_counts: dict[str, Any],
    mandatory: list[dict[str, Any]],
) -> dict[str, int]:
    target = _largest_remainder_counts(sample_limit, split_counts)
    mandatory_counts = Counter(sample["split"] for sample in mandatory)
    for split, count in mandatory_counts.items():
        target[split] = max(target.get(split, 0), int(count))
    extra = sum(target.values()) - sample_limit
    while extra > 0:
        candidates = [
            split
            for split in target
            if target[split] > mandatory_counts.get(split, 0)
        ]
        if not candidates:
            raise ValueError("mandatory Gap-A probes exceed requested sample limit")
        split = max(candidates, key=lambda item: (target[item], int(split_counts.get(item, 0)), item))
        target[split] -= 1
        extra -= 1
    return target


def _diverse_condition_select(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("cannot select a negative sample count")
    if count > len(samples):
        raise ValueError(f"requested {count} samples from bucket with {len(samples)}")
    remaining = list(samples)
    selected: list[dict[str, Any]] = []
    condition_counts = {
        "source_pattern_tag": Counter(),
        "k_region_mode": Counter(),
        "k_field_mode": Counter(),
        "stack_template": Counter(),
        "bc_category": Counter(),
        "power_scale_category": Counter(),
    }
    while len(selected) < count:
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                condition_counts["source_pattern_tag"][remaining[idx].get("source_pattern_tag")],
                condition_counts["k_region_mode"][remaining[idx].get("k_region_mode")],
                condition_counts["stack_template"][remaining[idx].get("stack_template")],
                condition_counts["bc_category"][remaining[idx].get("bc_category")],
                condition_counts["k_field_mode"][remaining[idx].get("k_field_mode")],
                condition_counts["power_scale_category"][remaining[idx].get("power_scale_category")],
                idx,
            ),
        )
        sample = remaining.pop(best_idx)
        selected.append(sample)
        for key, counts in condition_counts.items():
            counts[sample.get(key)] += 1
    return selected


def _balanced_gap_a_sample_limit(
    samples: list[dict[str, Any]],
    sample_limit: int,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    smoke_probe_count = int(
        manifest.get("sample_generation_plan", {}).get("smoke_probe_sample_count", 16)
    )
    if sample_limit <= smoke_probe_count:
        return samples[:sample_limit]

    mandatory = samples[:smoke_probe_count]
    target_counts = _target_split_counts(sample_limit, manifest.get("split_counts", {}), mandatory)
    selected = list(mandatory)
    selected_ids = {sample["sample_id"] for sample in selected}
    buckets: dict[str, list[dict[str, Any]]] = {split: [] for split in target_counts}
    for sample in samples:
        if sample["sample_id"] in selected_ids:
            continue
        buckets.setdefault(sample["split"], []).append(sample)

    for split in manifest.get("split_counts", {}):
        need = target_counts.get(split, 0) - sum(1 for sample in selected if sample["split"] == split)
        if need <= 0:
            continue
        selected.extend(_diverse_condition_select(buckets.get(split, []), need))

    if len(selected) != sample_limit:
        raise ValueError(f"balanced selection expected {sample_limit}, found {len(selected)}")
    return selected


def _materialized_samples(manifest: dict[str, Any], sample_ids: list[str] | None) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    if isinstance(samples, list) and samples:
        return _select_samples(manifest, sample_ids)
    plan = manifest.get("sample_generation_plan", {})
    if isinstance(plan, dict) and plan.get("strategy") == "gapA_deterministic_balanced_cycle":
        generated = _build_gap_a_samples(manifest)
        if sample_ids is None:
            return generated
        requested = set(sample_ids)
        filtered = [sample for sample in generated if sample["sample_id"] in requested]
        missing = sorted(requested - {sample["sample_id"] for sample in filtered})
        if missing:
            raise ValueError(f"requested sample ids missing from generated manifest plan: {missing}")
        return filtered
    return _select_samples(manifest, sample_ids)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    samples = _apply_sample_limit(
        _materialized_samples(manifest, args.sample_ids),
        args.sample_limit,
        manifest,
        balanced=args.sample_ids is None,
    )
    output_subset_arg = args.output_subset or _default_output_subset_for_manifest(manifest_path)
    output_subset = _validate_output_path(output_subset_arg, overwrite=args.overwrite)

    print("Heat3D v1 physics-label medium generator")
    print(f"manifest: {manifest_path}")
    print(f"output_subset: {output_subset}")
    print(f"selected_sample_count: {len(samples)}")
    print(f"split_counts: {dict(Counter(sample['split'] for sample in samples))}")
    print(
        "scope: medium-style generation smoke / research reference labels / "
        "benchmark-candidate dataset preparation only"
    )
    print("source_assignment: volume_fraction")
    print("q_policy: fixed_density")
    if not args.write:
        print("write_enabled: False")
        print("no_data_written: True")
        return 0

    if output_subset.exists() and args.overwrite:
        shutil.rmtree(output_subset)
    samples_dir = output_subset / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    summaries = [_write_sample(samples_dir, manifest, manifest_path, sample) for sample in samples]

    print("write_enabled: True")
    print(f"wrote_sample_count: {len(summaries)}")
    for summary in summaries:
        print(
            "- "
            f"{summary['sample_id']} split={summary['split']} "
            f"source={summary['source_pattern_tag']} stack={summary['stack_template']} "
            f"k={summary['k_region_mode']} bc={summary['bc_category']} "
            f"k_shape={summary['k_shape']} source_missed={summary['source_missed']} "
            f"active_volume={summary['active_source_volume_discrete']:.6e} "
            f"integrated_power={summary['integrated_q_power']:.6e} "
            f"power_rel_error={summary['integrated_q_power_relative_error']:.6e} "
            f"T_range=[{summary['T_min']:.6f}, {summary['T_max']:.6f}] "
            f"converged={summary['convergence_flag']} "
            f"residual_norm={summary['residual_norm']:.6e} "
            f"bottom_error={summary['bottom_dirichlet_error']:.6e}"
        )
    print("temperature_written: True")
    print("label_meta_written: True")
    print("formal_benchmark_generated: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
