#!/usr/bin/env python3
"""Run a small medium1024 Gap-A generation smoke.

This writes only an ignored local subset under data/. It is a diagnostics
smoke for the generation-ready candidate, not a formal benchmark generation.
"""

from __future__ import annotations

import argparse
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
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_local_smoke_v2"
)

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
        "scripts/check_heat3d_v1_physics_label_medium1024_gapA_subset.py",
        "--manifest",
        str(args.manifest),
        "--subset",
        str(args.output_subset),
        "--expected-count",
        str(args.sample_limit),
    ])
    print("medium1024_gapA_generation_smoke_ok: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
