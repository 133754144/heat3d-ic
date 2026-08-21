#!/usr/bin/env python3
"""Hash a copied authoritative-valid32 raw/log artifact directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.artifact_dir.resolve()
    output = args.output.resolve()
    files = sorted(path for path in directory.iterdir() if path.is_file() and path != output)
    cell_json = [path for path in files if path.name.endswith(("_serial.json", "_Q2.json"))]
    cell_logs = [path for path in files if path.name.endswith(("_serial.log", "_Q2.log"))]
    if len(cell_json) != 30 or len(cell_logs) != 30:
        raise RuntimeError(f"expected 30 cell JSON and 30 cell logs, got {len(cell_json)}/{len(cell_logs)}")
    if not any(path.name == "authoritative_valid32_raw.json" for path in files):
        raise RuntimeError("authoritative raw summary absent")
    entries = []
    for path in files:
        try:
            relative = path.relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError(f"artifact is outside repository: {path}") from error
        entries.append({
            "path": str(relative), "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    result = {
        "schema_version": "heat3d_v6_publication_authoritative_valid32_artifact_manifest_v1",
        "status": "complete", "artifact_count": len(entries), "artifacts": entries,
        "role_contract": {"training": False, "test": False, "sealed": False},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
