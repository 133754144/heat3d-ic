#!/usr/bin/env python3
"""Generate a frozen P1f unified layered pilot or final dataset; no ML work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_heat3d_v6_p1e_deconfounded_dataset import ROOT, generate


DEFAULT_CONFIG = ROOT / "configs/heat3d_v6/v6_p1f_temperature_shaping_pilot128.yaml"
DEFAULT_DATASET = ROOT / "data/heat3d_v6_p1f_temperature_shaping_pilot128_v0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-stem", default="v6_p1f_temperature_shaping_pilot128")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    result = generate(config.resolve(), dataset.resolve(), args.artifact_stem, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
