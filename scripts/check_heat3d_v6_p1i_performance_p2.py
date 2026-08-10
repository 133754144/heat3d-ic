#!/usr/bin/env python3
"""Checker for isolated P2 workload semantics."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"; DOC = ROOT / "docs"


def main() -> int:
    protocol = json.loads((CFG / "v6_p1i_performance_p2_protocol.json").read_text())
    result = json.loads((CFG / "v6_p1i_performance_p2_closeout.json").read_text())
    assert protocol["p1_commit"] == "e514103" and result["status"] == "completed"
    assert result["p1_frozen_sha256"] == hashlib.sha256((CFG / "v6_p1i_performance_p1_closeout.json").read_bytes()).hexdigest()
    assert all(not result["role_contract"][key] for key in ("training", "test", "sealed", "accuracy_recomputed", "checkpoint_modified", "dataset_modified", "graph_policy_search"))
    for gate in result["known_support_gates"].values():
        assert gate == {"graph_repeat_exact": True, "metrics_computed": False, "status": "passed",
                        "temperature_read": False, "unique_group_count": 32,
                        "unique_physics_signature_count": 32}
    with (DOC / "v6_p1i_workload_semantics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    routes = {row["route"] for row in rows}; assert routes == {"B8192_recon", "E32768_recon", "B240825_direct", "E240825_direct", "FVM240825"}
    assert all(len([row for row in rows if row["route"] == route]) == 4 for route in routes)
    for row in rows:
        if row["state"] == "resident_runtime_graph_rebuild":
            assert row["speedup_vs_semantically_matched_fvm"].startswith("N/A")
    for path, expected in result["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert "32 个不同 group" in (DOC / "v6_p1i_workload_semantics.md").read_text()
    print(json.dumps({"status": "passed", "rows": len(rows), "known_support_routes": 4}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
