#!/usr/bin/env python3
"""Check P5-S2 exact gates and report completeness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol["status"] == "preregistered_before_candidate_execution"
    assert protocol["population"] == "frozen valid32"
    assert protocol["role_contract"] == {
        "training": False, "test": False, "sealed": False, "inference": False,
        "temperature_or_prediction_used": False, "checkpoint_modified": False,
        "graph_policy_modified": False,
    }
    result_checked = False
    if args.json:
        payload = json.loads(args.json.read_text())
        assert payload["status"] == "passed" and payload["hard_gate_passed"] is True
        assert len(payload["samples"]) == 64
        assert "layer_boundaries_m" in payload["geometry_static_hashes"]
        for row in payload["samples"]:
            assert all(row["gates"].values())
            assert set(row["ordering_profile"]) == {
                "mask_seconds", "sha256_seconds", "sort_seconds",
                "inner_interleave_seconds", "outer_interleave_seconds",
            }
        assert payload["decision"]["c_cpp"] == "NOT_IMPLEMENTED"
        assert payload["decision"]["approximate_q_cluster_cache"] == "NOT_IMPLEMENTED"
        assert not any(payload["role_contract"].values())
        result_checked = True
    if args.csv:
        rows = list(csv.DictReader(args.csv.open()))
        assert len(rows) == 32
    if args.md:
        text = args.md.read_text()
        for section in ("# V6 P1i P5-S2", "## B8192", "## E32768", "## Decision"):
            assert section in text
    print(json.dumps({"p5s2_protocol_checked": True, "result_checked": result_checked}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
