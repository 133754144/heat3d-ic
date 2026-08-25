#!/usr/bin/env python3
"""Fail-closed validation of the publication-compatible supplemental closeout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/heat3d_v6_supplemental_publication"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    payload = json.loads((BASE / "known_topology_publication_closeout.json").read_text())
    if payload["status"] != "passed" or payload["gates"]["S0"] != "PASS":
        raise RuntimeError("closeout status drift")
    if payload["execution_commit"] != "8a812619ab0112b4ecfc37ef18189f731180059d":
        raise RuntimeError("execution commit drift")
    if payload["invalidated_attempt"]["head"][:7] != "171cd5e":
        raise RuntimeError("invalidated attempt provenance drift")
    for phase, expected in (("S1", 4), ("S2", 8), ("S3", 32)):
        for route, gate in payload["gates"][phase].items():
            if gate["status"] != "PASS" or int(gate["case_count"]) != expected:
                raise RuntimeError(f"{phase}/{route}: gate drift")
            path = BASE / "known_topology_results_8a81261" / phase / f"{route}.json"
            if sha256(path) != payload["result_artifact_sha256"][phase][route]:
                raise RuntimeError(f"{phase}/{route}: artifact SHA drift")
    with (BASE / "known_topology_publication_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16 or len(payload["result_table"]) != 16:
        raise RuntimeError("formal result table population drift")
    if any(row["mode"] == "known_topology_new_physics" and not row["speedup_vs_fresh_median"] for row in rows):
        raise RuntimeError("known-topology speedup missing")
    if any(payload["guardrails"].values()):
        raise RuntimeError("forbidden scope was enabled")
    print(json.dumps({"S0": "PASS", "S1": "PASS", "S2": "PASS", "S3": "PASS", "rows": 16}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
