#!/usr/bin/env python3
"""Generate and diagnose a small medium1024 Gap-A diversity smoke subset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium1024_gapA_manifest.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_diversity_local_smoke_v2"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v1_medium_runs" / "medium1024_gapA_diversity_generation_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and diagnose a 16/32-sample Heat3D v1 medium1024 Gap-A diversity smoke."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=16)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sample_dirs(subset: Path) -> list[Path]:
    root = subset / "samples" if (subset / "samples").is_dir() else subset
    return sorted(child for child in root.iterdir() if child.is_dir())


def main() -> int:
    args = parse_args()
    if args.sample_limit not in {16, 32}:
        raise ValueError("--sample-limit should be 16 or 32 for this local smoke")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diversity_json = args.output_dir / "diversity.json"
    diversity_md = args.output_dir / "diversity.md"

    print("Heat3D v1 medium1024 Gap-A diversity generation smoke", flush=True)
    print(f"manifest: {args.manifest}", flush=True)
    print(f"output_subset: {args.output_subset}", flush=True)
    print(f"sample_limit: {args.sample_limit}", flush=True)
    print("scope: generation diversity smoke only; not a formal benchmark", flush=True)

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
        "scripts/check_heat3d_v1_physics_label_medium1024_gapA_subset.py",
        "--manifest",
        str(args.manifest),
        "--subset",
        str(args.output_subset),
        "--expected-count",
        str(args.sample_limit),
    ])
    _run([
        sys.executable,
        "scripts/check_heat3d_v1_label_diagnostics.py",
        "--subset",
        str(args.output_subset),
    ])
    _run([
        sys.executable,
        "scripts/analyze_heat3d_v1_medium1024_gapA_diversity.py",
        "--subset",
        str(args.output_subset),
        "--output-json",
        str(diversity_json),
        "--output-md",
        str(diversity_md),
        "--top-n",
        "30",
    ])

    sample_dirs = _sample_dirs(args.output_subset)
    metadata_present = all((sample_dir / "metadata.json").is_file() for sample_dir in sample_dirs)
    source_missed_count = sum(
        1
        for sample_dir in sample_dirs
        if bool(_read_json(sample_dir / "metadata.json").get("source_missed"))
    )
    diversity = _read_json(diversity_json)
    flags = diversity["diagnostic_flags"]
    checks = {
        "metadata_present": metadata_present,
        "sample_count_ok": diversity["sample_count"] == args.sample_limit,
        "q_hash_not_all_repeated": diversity["unique_q_hash_count"] > 1,
        "k_hash_not_all_repeated": diversity["unique_k_hash_count"] > 1,
        "temperature_hash_not_all_repeated": diversity["unique_temperature_hash_count"] > 1,
        "gap_a_modes_covered": flags["diversity_ready_for_training_smoke"],
        "no_nonfinite_arrays": diversity["nonfinite_array_count"] == 0,
        "source_missed_count_zero": source_missed_count == 0,
        "diversity_outputs_written": diversity_json.is_file() and diversity_md.is_file(),
    }
    ok = all(checks.values())
    print(f"checks: {checks}", flush=True)
    print(f"diversity_flags: {flags}", flush=True)
    print(f"source_missed_count: {source_missed_count}", flush=True)
    print(f"diversity_json: {diversity_json}", flush=True)
    print(f"diversity_md: {diversity_md}", flush=True)
    print(f"medium1024_gapA_diversity_generation_smoke_ok: {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
