#!/usr/bin/env python3
"""Run a small medium1024 Gap-A generation smoke.

This writes only an ignored local subset under data/. It is a diagnostics
smoke for the generation-ready candidate, not a formal benchmark generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import diagnose_sample, find_sample_dirs  # noqa: E402
from rigno.heat3d_v1_reference_solver_v2 import BOTTOM_TOL_K, RESIDUAL_TOL  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium1024_gapA_manifest.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_local_smoke_v2"
)
REQUIRED_FILES = (
    "coords.npy",
    "layer_id.npy",
    "region_id.npy",
    "material_id.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
    "sample_meta.json",
    "label_meta.json",
)
REQUIRED_GAP_A_SOURCE = {
    "low_power_near_zero_background_cases",
    "high_dynamic_range_power_cases",
}
REQUIRED_GAP_A_K = {
    "high_contrast_interface_k",
    "low_k_barrier_or_TIM_variation",
}
REQUIRED_GAP_A_BC = {
    "very_low_top_h_candidate",
    "very_high_top_h_candidate",
}
POWER_REL_TOL = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and check a 16/32-sample Heat3D v1 medium1024 Gap-A local smoke."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument("--sample-limit", type=int, default=16)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sample_plan(meta: dict[str, Any]) -> dict[str, Any]:
    generation = meta.get("generation_config", {})
    if isinstance(generation, dict):
        plan = generation.get("sample_plan", {})
        if isinstance(plan, dict):
            return plan
    return {}


def _check_arrays(sample_dir: Path, errors: list[str]) -> None:
    for name in ("coords.npy", "k_field.npy", "q_field.npy", "temperature.npy"):
        values = np.load(sample_dir / name)
        if not np.all(np.isfinite(values)):
            errors.append(f"{sample_dir.name} {name} contains NaN/Inf")


def main() -> int:
    args = parse_args()
    if args.sample_limit not in {16, 32}:
        raise ValueError("--sample-limit should be 16 or 32 for this local smoke")

    print("Heat3D v1 medium1024 Gap-A generation smoke")
    print(f"manifest: {args.manifest}")
    print(f"output_subset: {args.output_subset}")
    print(f"sample_limit: {args.sample_limit}")
    print("scope: generation smoke only; not a formal benchmark")

    _run([
        sys.executable,
        "scripts/check_heat3d_v1_physics_label_medium1024_manifest.py",
        "--manifest",
        str(args.manifest),
    ])
    _run([
        sys.executable,
        "tools/generate_heat3d_v1_physics_label_medium.py",
        "--manifest",
        str(args.manifest),
        "--output-subset",
        str(args.output_subset),
        "--sample-limit",
        str(args.sample_limit),
        "--write",
        "--overwrite",
    ])
    _run([
        sys.executable,
        "scripts/check_heat3d_v1_label_diagnostics.py",
        "--subset",
        str(args.output_subset),
    ])

    sample_dirs = find_sample_dirs(args.output_subset)
    errors: list[str] = []
    if len(sample_dirs) != args.sample_limit:
        errors.append(f"expected {args.sample_limit} samples, found {len(sample_dirs)}")

    source_counts: Counter[str] = Counter()
    k_counts: Counter[str] = Counter()
    bc_counts: Counter[str] = Counter()
    power_scale_counts: Counter[str] = Counter()
    for sample_dir in sample_dirs:
        missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
        if missing:
            errors.append(f"{sample_dir.name} missing files: {missing}")
            continue
        _check_arrays(sample_dir, errors)
        meta = _read_json(sample_dir / "sample_meta.json")
        label_meta = _read_json(sample_dir / "label_meta.json")
        plan = _sample_plan(meta)
        source_summary = meta.get("source_diagnostics", {})
        source_counts[str(plan.get("source_pattern_tag"))] += 1
        k_counts[str(plan.get("k_region_mode"))] += 1
        bc_counts[str(plan.get("bc_category"))] += 1
        power_scale_counts[str(plan.get("power_scale_category"))] += 1
        if source_summary.get("source_missed"):
            errors.append(f"{sample_dir.name} source_missed=true")
        if float(source_summary.get("active_source_volume_discrete", 0.0)) <= 0.0:
            errors.append(f"{sample_dir.name} active source volume <= 0")
        if float(source_summary.get("integrated_q_power_relative_error", 1.0)) > POWER_REL_TOL:
            errors.append(f"{sample_dir.name} integrated power relative error exceeds tolerance")
        if label_meta.get("convergence_flag") is not True:
            errors.append(f"{sample_dir.name} convergence_flag is not true")
        if float(label_meta.get("residual_norm", 1.0)) > RESIDUAL_TOL:
            errors.append(f"{sample_dir.name} residual_norm exceeds tolerance")
        if float(label_meta.get("bottom_dirichlet_error", 1.0)) > BOTTOM_TOL_K:
            errors.append(f"{sample_dir.name} bottom_dirichlet_error exceeds tolerance")
        report = diagnose_sample(sample_dir)
        if report.get("overall_status") == "fail":
            errors.append(f"{sample_dir.name} label diagnostics failed")

    missing_sources = sorted(REQUIRED_GAP_A_SOURCE - set(source_counts))
    missing_k = sorted(REQUIRED_GAP_A_K - set(k_counts))
    missing_bc = sorted(REQUIRED_GAP_A_BC - set(bc_counts))
    if missing_sources:
        errors.append(f"missing Gap-A source modes in smoke: {missing_sources}")
    if missing_k:
        errors.append(f"missing Gap-A k modes in smoke: {missing_k}")
    if missing_bc:
        errors.append(f"missing Gap-A BC modes in smoke: {missing_bc}")

    print("summary")
    print(f"  sample_count: {len(sample_dirs)}")
    print(f"  source_pattern_counts: {dict(source_counts)}")
    print(f"  k_region_mode_counts: {dict(k_counts)}")
    print(f"  bc_category_counts: {dict(bc_counts)}")
    print(f"  power_scale_category_counts: {dict(power_scale_counts)}")
    print(f"  errors: {errors}")
    print("  diagnostics_scope: local generation smoke only; not formal benchmark")
    print(f"  medium1024_gapA_generation_smoke_ok: {not errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
