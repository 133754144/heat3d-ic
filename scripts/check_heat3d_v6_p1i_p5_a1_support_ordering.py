#!/usr/bin/env python3
"""Check the tracked P5-A1 support-ordering closeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["hard_gate_passed"] is True
    assert len(payload["samples"]) == 32
    assert payload["role_contract"] == {
        "training": False,
        "test": False,
        "sealed": False,
        "checkpoint_modified": False,
        "dataset_modified": False,
        "graph_policy_modified": False,
        "temperature_or_prediction_used": False,
    }
    for row in payload["samples"]:
        assert row["selected_indices_array_equal"]
        assert row["selected_indices_sha256_equal"]
        assert row["full_order_is_permutation"]
        assert row["anchor_prefix_exact"]
    print(json.dumps({
        "status": "passed",
        "samples": 32,
        "candidate_promoted": payload["candidate_promoted"],
        "median_speedup": payload["timing"]["full_order_seconds"]["median_speedup"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
