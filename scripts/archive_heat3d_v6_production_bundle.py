#!/usr/bin/env python3
"""Archive immutable V6 production inference dependencies with SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_graph_cache import graph_builder_code_fingerprint  # noqa: E402
import evaluate_heat3d_v6_anchored_resolution as anchored  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--reconstruction-maps", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tracked-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists() and any(args.archive.iterdir()):
        raise SystemExit("archive destination must be empty")
    args.archive.mkdir(parents=True, exist_ok=True)
    inputs = []
    inputs.extend(
        (path, Path("graph_cache") / path.name)
        for path in sorted(args.graph_cache.glob("*.npz"))
    )
    inputs.extend(
        (path, Path("reconstruction_maps") / path.name)
        for path in sorted(args.reconstruction_maps.glob("*.npz"))
    )
    for name in (
        "params_best_valid_point_global.pkl",
        "run_config.json",
        "loss_summary.json",
    ):
        inputs.append((args.checkpoint_dir / name, Path("checkpoint") / name))
    inputs.append(
        (
            ROOT / "scripts/preflight_heat3d_v6_production_bundle.py",
            Path("preflight_heat3d_v6_production_bundle.py"),
        )
    )
    files = []
    for source, relative in inputs:
        if not source.is_file():
            raise SystemExit(f"missing archive input: {source}")
        destination = args.archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    spec = anchored.SEED_SPECS["seed0"]
    manifest = {
        "schema_version": "heat3d_v6_production_bundle_v1",
        "status": "archived",
        "archive_path": str(args.archive.resolve()),
        "graph_builder_code_fingerprint": graph_builder_code_fingerprint(),
        "cache_key_contract": [
            "support_hash",
            "resolved_graph_config",
            "graph_seed",
            "graph_builder_code_fingerprint",
        ],
        "checkpoint": spec,
        "production_resolutions": [1024, 2048, 4096, 8192, 16384],
        "experimental_resolution": 32768,
        "files": files,
        "training_executed": False,
        "test_hard_accessed": False,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.archive / "bundle_manifest.json").write_text(encoded, encoding="utf-8")
    args.tracked_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.tracked_manifest.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "archive": str(args.archive.resolve()),
                "file_count": len(files),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
