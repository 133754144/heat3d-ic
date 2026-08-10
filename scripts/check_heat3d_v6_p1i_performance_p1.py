#!/usr/bin/env python3
"""Deterministic checker for the isolated P1 closeout."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOC = ROOT / "docs"


def main() -> int:
    protocol = json.loads((CFG / "v6_p1i_performance_p1_protocol.json").read_text())
    result = json.loads((CFG / "v6_p1i_performance_p1_closeout.json").read_text())
    assert protocol["base_commit"] == "2ccb77926d849ccbcfbd9a5e3ee43f8117d2f2a3"
    assert result["status"] == "completed" and len(result["rows"]) == 5
    assert result["new_computation"] == {"B8192_process_cold_independent_runs": 10}
    assert result["historical_accuracy_recomputed"] is False and result["fvm_recomputed"] is False
    assert all(result["pareto"].values())
    assert result["role_contract"] == protocol["role_contract"]
    assert not any(result["role_contract"][key] for key in ("training", "test", "sealed", "checkpoint_modified", "dataset_modified", "graph_policy_search"))
    with (DOC / "v6_p1i_optimal_resolution_full_grid_comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["route"] for row in rows] == ["B8192_recon", "E32768_recon", "B240825_direct", "E240825_direct", "FVM240825"]
    assert all(row["measurement_domain"] in {"reconstructed_full_240825", "direct_full_grid_240825", "full_240825_reference"} for row in rows)
    assert all(float(row["reconstruction_median_s"]) > 0 for row in rows[:2])
    assert all(float(row["reconstruction_median_s"]) == 0 for row in rows[2:])
    assert all(row["fresh_speedup_vs_fvm"].startswith("N/A") for row in rows[:4])
    assert len(list((CFG / "v6_p1i_performance_p1_raw/B8192_process").glob("run*.json"))) == 10
    for path, expected in result["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    text = (DOC / "v6_p1i_optimal_resolution_full_grid_comparison.md").read_text()
    assert "模型推理与重建 apply 独立" in text and "Pareto decision" in text
    print(json.dumps({"status": "passed", "p1_rows": len(rows), "role_contract": result["role_contract"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
