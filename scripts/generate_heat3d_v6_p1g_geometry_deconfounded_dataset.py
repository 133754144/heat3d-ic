#!/usr/bin/env python3
"""Generate the frozen V6-P1g geometry-deconfounded dataset; never train/infer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_heat3d_v6_p1e_deconfounded_dataset as shared


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml"
DEFAULT_DATASET = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v0"
ARTIFACT_STEM = "v6_p1g_geometry_deconfounded1024"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    result = shared.generate(config.resolve(), dataset.resolve(), ARTIFACT_STEM, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
