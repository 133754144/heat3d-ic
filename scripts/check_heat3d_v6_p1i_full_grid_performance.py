#!/usr/bin/env python3
"""Fail-closed checks for the P1i full-grid performance closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    assert payload["timing_protocols_pooled"] is False
    assert payload["process_cold_repeat_counts"] == {"B": 10, "E": 10}
    for policy in ("B", "E"):
        feasibility = payload["sample1_feasibility"][policy]
        assert feasibility["status"] == "passed"
        assert feasibility["undercovered_fraction"] == 0.0
        assert feasibility["r2r_components"] == 1.0
    optimization = payload["optimization"]
    assert optimization["status"] == "completed_no_promotion"
    assert optimization["promoted_cells"] == []
    assert optimization["B_optimization_applied"] is False
    for name in ("shared_reverse", "gpu_tiled"):
        equivalence = optimization[name]["equivalence"]
        assert equivalence["status"] == "passed"
        assert all(
            equivalence["edge_summary"][field]["all_samples_exact"]
            for field in ("p2r", "r2r", "r2p")
        )
        assert equivalence["full_prediction_difference"]["rmse_K"] < 0.01
        assert equivalence["full_prediction_difference"]["max_abs_K"] < 0.1
        assert max(
            abs(value)
            for value in equivalence["metric_delta_candidate_minus_reference"].values()
        ) < 0.001
    assert optimization["shared_reverse"]["process_cold_unpaired_bootstrap"]["ci95_seconds"][0] <= 0.0
    assert optimization["gpu_tiled"]["gain_pct"]["graph_construction_pct"] < 0.0
    rows = list(csv.DictReader(table.open(newline="")))
    required = {("A", str(n)) for n in (1024, 4096, 8192, 16384, 32768)}
    required |= {("B", str(n)) for n in (1024, 4096, 8192, 16384, 32768, 240825)}
    required |= {("E", str(n)) for n in (1024, 4096, 8192, 16384, 32768, 240825)}
    assert required.issubset({(row["policy"], row["resolution"]) for row in rows})
    for row in rows:
        assert row["timing_protocol"]
        assert row["provenance"]
        for field in ("point_global_pct", "raw_cv_rmse_K"):
            if row[field] not in ("", "not_available"):
                assert math.isfinite(float(row[field]))
    for policy in ("B", "E"):
        row = next(
            row for row in rows
            if row["policy"] == policy and row["resolution"] == "240825"
        )
        assert row["status"] == "new_full_grid_valid32"
        assert float(row["process_cold_median_s"]) > 0.0
        assert float(row["fresh_topology_median_s"]) > 0.0
        assert float(row["warm_resident_median_s"]) > 0.0
        assert row["fresh_topology_speedup_vs_fvm"] == "not_comparable_no_fvm_unseen_topology_state"


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
