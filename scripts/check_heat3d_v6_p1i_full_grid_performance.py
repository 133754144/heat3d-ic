#!/usr/bin/env python3
"""Fail-closed checks for the P1i full-grid performance closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    assert payload["status"] == "preregistered_before_full_grid_execution"
    assert payload["base_commit"].startswith("9f439ba")
    assert payload["scientific_contract"]["resolution"] == 240825
    assert payload["scientific_contract"]["direct_output"] is True
    assert payload["scientific_contract"]["test_accessed"] is False
    assert payload["scientific_contract"]["sealed_accessed"] is False
    assert payload["scientific_contract"]["training"] is False
    assert payload["policies"]["B"]["regional_node_count"] == 30103
    assert payload["policies"]["E"]["regional_node_count"] == 256
    for row in payload["historical_inputs"].values():
        target = ROOT / row["path"]
        assert target.is_file(), target
        assert sha256(target) == row["sha256"], target
    return payload


def check_closeout(path: Path, table: Path) -> None:
    payload = json.loads(path.read_text())
    assert payload["role_contract"] == {
        "training": False, "test": False, "sealed": False,
    }
    assert payload["historical_artifacts_reexecuted"] is False
    rows = list(csv.DictReader(table.open(newline="")))
    required = {("A", str(n)) for n in (1024, 4096, 8192, 16384, 32768)}
    required |= {("B", str(n)) for n in (1024, 4096, 8192, 16384, 32768, 240825)}
    required |= {("E", str(n)) for n in (1024, 4096, 8192, 16384, 32768, 240825)}
    assert required.issubset({(row["policy"], row["resolution"]) for row in rows})
    for row in rows:
        assert row["timing_protocol"]
        assert row["provenance"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_full_grid_performance_protocol.json")
    parser.add_argument("--closeout", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    check_protocol(args.protocol)
    if args.closeout or args.table:
        if args.closeout is None or args.table is None:
            raise SystemExit("--closeout and --table must be supplied together")
        check_closeout(args.closeout, args.table)
    print(json.dumps({"status": "passed", "protocol_checked": True, "closeout_checked": bool(args.closeout)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
