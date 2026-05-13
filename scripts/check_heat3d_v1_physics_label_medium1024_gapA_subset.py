#!/usr/bin/env python3
"""Check generated Heat3D v1 medium1024 Gap-A subsets.

This checker validates generated samples and coverage summaries. It does not
require manifest.samples and is not a formal benchmark validator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import diagnose_sample  # noqa: E402
from rigno.heat3d_v1_reference_solver_v2 import BOTTOM_TOL_K, RESIDUAL_TOL  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium1024_gapA_manifest.json"
DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_v0_candidate"
)
REQUIRED_FILES = (
    "coords.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
    "label_meta.json",
    "metadata.json",
)
FINITE_ARRAYS = (
    "coords.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
)
METADATA_REQUIRED_FIELDS = (
    "sample_id",
    "split",
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
    "power_scale_category",
    "k_contrast_category",
    "barrier_k_category",
)
GAP_A_SOURCE_MODES = {
    "low_power_near_zero_background_cases",
    "high_dynamic_range_power_cases",
}
GAP_A_K_MODES = {
    "high_contrast_interface_k",
    "low_k_barrier_or_TIM_variation",
}
GAP_A_BC_MODES = {
    "very_low_top_h_candidate",
    "very_high_top_h_candidate",
}
POWER_REL_TOL = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a generated Heat3D v1 medium1024 Gap-A subset."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sample_dirs(subset: Path) -> list[Path]:
    root = subset / "samples" if (subset / "samples").is_dir() else subset
    if root.is_dir() and (root / "metadata.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    )


def _finite_array(path: Path) -> tuple[bool, list[int] | None, str | None]:
    try:
        array = np.load(path)
    except Exception:
        return False, None, None
    return bool(np.all(np.isfinite(array))), list(array.shape), str(array.dtype)


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "<missing>" if value is None else str(value)


def _manifest_values(manifest: dict[str, Any], section: str) -> set[str]:
    values = manifest.get("coverage_targets", {}).get(section, [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def main() -> int:
    args = parse_args()
    if args.expected_count < 1:
        raise ValueError("--expected-count must be >= 1")

    manifest = _read_json(args.manifest)
    sample_dirs = _sample_dirs(args.subset)
    errors: list[str] = []
    warnings: list[str] = []
    if len(sample_dirs) != args.expected_count:
        errors.append(f"expected {args.expected_count} samples, found {len(sample_dirs)}")

    counts = {
        "split": Counter(),
        "source_pattern_tag": Counter(),
        "k_region_mode": Counter(),
        "k_field_mode": Counter(),
        "stack_template": Counter(),
        "bc_category": Counter(),
        "power_scale_category": Counter(),
    }
    source_missed_count = 0
    max_power_rel_error = 0.0
    max_residual_norm = 0.0
    max_bottom_error = 0.0
    fail_samples: list[str] = []
    warning_samples: list[str] = []

    for sample_dir in sample_dirs:
        missing_files = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
        if missing_files:
            errors.append(f"{sample_dir.name} missing required files: {missing_files}")
            continue

        for name in FINITE_ARRAYS:
            finite, shape, dtype = _finite_array(sample_dir / name)
            if not finite:
                errors.append(f"{sample_dir.name} {name} contains non-finite values or cannot be read")
            if name == "coords.npy" and shape is not None and (len(shape) != 2 or shape[1] != 3):
                errors.append(f"{sample_dir.name} coords.npy must have shape (N, 3), found {shape}")
            if name in {"k_field.npy", "q_field.npy", "temperature.npy"} and shape is not None and len(shape) != 2:
                errors.append(f"{sample_dir.name} {name} must be a 2D array, found {shape}")

        metadata = _read_json(sample_dir / "metadata.json")
        label_meta = _read_json(sample_dir / "label_meta.json")
        missing_metadata = [field for field in METADATA_REQUIRED_FIELDS if field not in metadata]
        if missing_metadata:
            errors.append(f"{sample_dir.name} metadata.json missing fields: {missing_metadata}")

        if str(metadata.get("sample_id")) != sample_dir.name:
            errors.append(
                f"{sample_dir.name} metadata sample_id mismatch: {metadata.get('sample_id')}"
            )

        for key in counts:
            counts[key][_metadata_value(metadata, key)] += 1

        for section in (
            "source_pattern_tag",
            "k_region_mode",
            "k_field_mode",
            "stack_template",
            "bc_category",
        ):
            allowed = _manifest_values(manifest, section)
            value = _metadata_value(metadata, section)
            if allowed and value not in allowed:
                errors.append(f"{sample_dir.name} {section}={value!r} not in manifest coverage targets")

        source_missed = bool(metadata.get("source_missed"))
        source_missed_count += int(source_missed)
        if source_missed:
            errors.append(f"{sample_dir.name} source_missed=true")

        active_volume = float(metadata.get("active_source_volume_discrete_m3", 0.0))
        if active_volume <= 0.0:
            errors.append(f"{sample_dir.name} active_source_volume_discrete_m3 <= 0")

        power_rel = float(metadata.get("integrated_q_power_relative_error", 1.0))
        max_power_rel_error = max(max_power_rel_error, power_rel)
        if power_rel > POWER_REL_TOL:
            errors.append(f"{sample_dir.name} integrated_q_power_relative_error exceeds tolerance")

        if label_meta.get("convergence_flag") is not True:
            errors.append(f"{sample_dir.name} label_meta.convergence_flag is not true")
        residual_norm = float(label_meta.get("residual_norm", 1.0))
        max_residual_norm = max(max_residual_norm, residual_norm)
        if residual_norm > RESIDUAL_TOL:
            errors.append(f"{sample_dir.name} residual_norm exceeds tolerance")
        bottom_error = float(label_meta.get("bottom_dirichlet_error", 1.0))
        max_bottom_error = max(max_bottom_error, bottom_error)
        if bottom_error > BOTTOM_TOL_K:
            errors.append(f"{sample_dir.name} bottom_dirichlet_error exceeds tolerance")

        report = diagnose_sample(sample_dir)
        status = report.get("overall_status")
        if status == "fail":
            fail_samples.append(sample_dir.name)
            errors.append(f"{sample_dir.name} label diagnostics failed")
        elif status == "warning":
            warning_samples.append(sample_dir.name)

    if args.expected_count >= 16:
        missing_gap_a_sources = sorted(GAP_A_SOURCE_MODES - set(counts["source_pattern_tag"]))
        missing_gap_a_k = sorted(GAP_A_K_MODES - set(counts["k_region_mode"]))
        missing_gap_a_bc = sorted(GAP_A_BC_MODES - set(counts["bc_category"]))
        if missing_gap_a_sources:
            errors.append(f"missing Gap-A source modes: {missing_gap_a_sources}")
        if missing_gap_a_k:
            errors.append(f"missing Gap-A k-region modes: {missing_gap_a_k}")
        if missing_gap_a_bc:
            errors.append(f"missing Gap-A BC modes: {missing_gap_a_bc}")

    print("Heat3D v1 medium1024 Gap-A subset checker")
    print(f"subset: {args.subset}")
    print(f"manifest: {args.manifest}")
    print(f"expected_count: {args.expected_count}")
    print("scope: generated subset diagnostics only; not a formal benchmark")
    print(f"sample_count: {len(sample_dirs)}")
    for key in (
        "split",
        "source_pattern_tag",
        "k_region_mode",
        "k_field_mode",
        "stack_template",
        "bc_category",
        "power_scale_category",
    ):
        print(f"{key}_counts: {dict(counts[key])}")
    print(f"source_missed_count: {source_missed_count}")
    print(f"max_integrated_q_power_relative_error: {max_power_rel_error:.6e}")
    print(f"max_residual_norm: {max_residual_norm:.6e}")
    print(f"max_bottom_dirichlet_error: {max_bottom_error:.6e}")
    print(f"warning_samples: {warning_samples}")
    print(f"fail_samples: {fail_samples}")
    print(f"warnings: {warnings}")
    print(f"errors: {errors}")
    print("diagnostics_scope: Gap-A subset smoke/pilot diagnostics only")
    print(f"medium1024_gapA_subset_ok: {not errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
