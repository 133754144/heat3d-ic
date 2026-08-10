#!/usr/bin/env python3
"""Fail-closed checks for the frozen P1i graph-resolution closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"
PROTOCOL = CONFIG / "v6_p1i_graph_resolution_closeout_protocol.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["status"] == "preregistered_before_graph_resolution_closeout", "protocol status")
    require(protocol["base_commit"].startswith("d3337f2"), "base commit")
    require(protocol["resolutions"] == [1024, 4096, 8192, 16384, 32768], "resolution ladder")
    scope = protocol["scope"]
    require(not scope["training"] and not scope["test_accessed"] and not scope["sealed_accessed"], "role scope")
    require(not scope["valid96_confirmation_reexecuted"] and not scope["factor_or_radius_search"], "forbidden work")
    require(protocol["policies"]["A"]["regional_subsample_factor"] == 4, "A definition")
    require(protocol["policies"]["B"]["regional_subsample_factor"] == 8, "B definition")
    require(protocol["policies"]["E"]["regional_node_count"] == 256, "E definition")
    expected_cells = {
        ("B", 1024), ("B", 4096), ("B", 32768),
        ("E", 4096), ("E", 16384), ("E", 32768),
    }
    require({(r["policy"], r["resolution"]) for r in protocol["new_execution_cells"]} == expected_cells, "new cells")
    for name, item in protocol["provenance"].items():
        path = ROOT / item["path"]
        require(path.is_file() and sha(path) == item["sha256"], f"provenance {name}")
    if args.results:
        final = json.loads((CONFIG / "v6_p1i_graph_resolution_closeout.json").read_text())
        require(final["status"] == "completed", "final status")
        require(len(final["rows"]) == 15, "A/B/E x five resolutions")
        require(final["role_contract"] == {
            "training": False, "test": False, "sealed": False,
            "valid96_confirmation_reexecuted": False,
        }, "final role contract")
        observed = {(r["policy"], int(r["resolution"])) for r in final["rows"]}
        require(observed == {(p, n) for p in "ABE" for n in protocol["resolutions"]}, "final matrix")
        for row in final["rows"]:
            require(row["sample_count"] == 32, "valid32 sample count")
            require(row["coverage_passed"], "coverage")
            require(row["provenance_class"] in {"historical_reuse", "new_execution", "exact_policy_alias"}, "provenance class")
        with (ROOT / "docs/v6_p1i_graph_resolution_closeout.csv").open(newline="") as handle:
            require(len(list(csv.DictReader(handle))) == 15, "CSV rows")
        md = (ROOT / "docs/v6_p1i_graph_resolution_closeout.md").read_text()
        require("valid32" in md and "valid96" in md and "不混合" in md, "provenance warning")
    print(json.dumps({"status": "passed", "results_checked": args.results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
