#!/usr/bin/env python3
"""Generate the Heat3D v1 64-sample physics-label medium smoke subset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys


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
PROTECTED_SUBSET_NAMES = {
    "v1_multilayer_bc_eq_demo",
    "v1_multilayer_bc_eq_supervised_smoke",
    "v1_multilayer_bc_eq_supervised_small",
    "v1_multilayer_bc_eq_physics_label_small_v2",
    "v1_multilayer_bc_eq_physics_label_medium_pilot_v2",
    "v1_multilayer_bc_eq_physics_label_medium_expansion_v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 64-sample Heat3D v1 physics-label medium smoke subset."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument("--sample-ids", nargs="*", default=None)
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


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    samples = _select_samples(manifest, args.sample_ids)
    output_subset = _validate_output_path(args.output_subset, overwrite=args.overwrite)

    print("Heat3D v1 physics-label medium generator")
    print(f"manifest: {manifest_path}")
    print(f"output_subset: {output_subset}")
    print(f"selected_sample_count: {len(samples)}")
    print(f"split_counts: {dict(Counter(sample['split'] for sample in samples))}")
    print("scope: 64-sample medium generation smoke / research reference labels / benchmark candidate only")
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
