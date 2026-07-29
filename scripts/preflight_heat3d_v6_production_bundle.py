#!/usr/bin/env python3
"""One-command integrity preflight for the archived V6 production bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_graph_cache import (  # noqa: E402
    graph_builder_code_fingerprint,
    load_metadata,
)
from rigno.heat3d_v6_full_field import load_reconstruction_map  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.archive / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["graph_builder_code_fingerprint"] != (
        graph_builder_code_fingerprint()
    ):
        raise SystemExit("graph-builder fingerprint drifted")
    for row in manifest["files"]:
        path = args.archive / row["relative_path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise SystemExit(f"missing/size drift: {path}")
        if _sha256(path) != row["sha256"]:
            raise SystemExit(f"SHA256 drift: {path}")
    graph_files = sorted((args.archive / "graph_cache").glob("*.npz"))
    reconstruction_files = sorted(
        (args.archive / "reconstruction_maps").glob("*.npz")
    )
    if len(graph_files) < 5 or len(reconstruction_files) < 5:
        raise SystemExit("production resolution archive coverage drifted")
    for path in graph_files:
        load_metadata(path)
    for path in reconstruction_files:
        mapping, _ = load_reconstruction_map(path)
        if len(mapping.support_indices) not in {
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        }:
            raise SystemExit(f"unexpected reconstruction support: {path}")
    checkpoint_path = (
        args.archive / "checkpoint/params_best_valid_point_global.pkl"
    )
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != int(manifest["checkpoint"]["epoch"]):
        raise SystemExit("checkpoint epoch drifted")
    print(
        json.dumps(
            {
                "status": "passed",
                "archive": str(args.archive.resolve()),
                "file_count": len(manifest["files"]),
                "graph_cache_count": len(graph_files),
                "reconstruction_map_count": len(reconstruction_files),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "training_executed": False,
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
