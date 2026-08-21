#!/usr/bin/env python3
"""Hash a fail-closed authoritative benchmark attempt without requiring 30 cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.artifact_dir.resolve()
    output = args.output.resolve()
    raw_path = directory / "authoritative_valid32_raw.json"
    raw = json.loads(raw_path.read_text())
    if raw["status"] != "failed_fail_closed":
        raise RuntimeError("failure manifest requires a fail-closed raw artifact")
    files = sorted(path for path in directory.iterdir() if path.is_file() and path != output)
    entries = []
    for path in files:
        relative = path.relative_to(ROOT)
        entries.append({
            "path": str(relative), "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    result = {
        "schema_version": "heat3d_v6_publication_authoritative_valid32_failure_manifest_v2",
        "status": "complete_failure_evidence",
        "attempted": True, "completed": False, "publication_results_generated": False,
        "completed_cell_count": len(raw["rows"]),
        "attempted_process_count": len(raw["process_records"]),
        "failed_cell": {
            key: raw["failure"][key]
            for key in ("route", "seed", "service_mode", "returncode", "wall_seconds")
        },
        "artifact_count": len(entries), "artifacts": entries,
        "role_contract": {"training": False, "test": False, "sealed": False},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
