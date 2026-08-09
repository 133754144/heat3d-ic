#!/usr/bin/env python3
"""Fail-closed validation for final P1i A/B graph-policy confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)


def main() -> int:
    protocol_path = CONFIG / "v6_p1i_graph_policy_ab_confirmation_protocol.json"
    compact_path = CONFIG / "v6_p1i_graph_policy_confirmation_compact.json"
    final_path = CONFIG / "v6_p1i_graph_policy_final.json"
    protocol = json.loads(protocol_path.read_text())
    compact = json.loads(compact_path.read_text())
    final = json.loads(final_path.read_text())
    require(protocol["status"] == "frozen_after_E_no_go_before_confirmation", "protocol")
    require(compact["status"] == "passed" and len(compact["cells"]) == 12, "matrix")
    expected = {(s, p, n) for s in (0, 1, 2) for p in ("A", "B") for n in (8192, 16384)}
    observed = {(int(r["seed"]), r["policy"], int(r["resolution"])) for r in compact["cells"]}
    require(observed == expected, "cell identities")
    sample_order = None
    for row in compact["cells"]:
        role = row["role_contract"]
        require(not role["training"] and not role["test"] and not role["sealed"], "role access")
        require(role["confirmation_remaining_valid96"] and not role["prediction_artifact_saved"], "confirmation role")
        require(len(row["sample_ids"]) == 96 and len(row["per_sample_metrics"]) == 96, "sample count")
        if sample_order is None: sample_order = row["sample_ids"]
        require(row["sample_ids"] == sample_order, "sample order")
    require(final["protocol"]["sha256"] == sha(protocol_path), "protocol SHA")
    require(final["compact_input"]["sha256"] == sha(compact_path), "compact SHA")
    require(final["E_decision"] == "NO_GO", "E decision")
    require(final["decision"] in ("B_GO", "B_NO_GO_RETAIN_A"), "final decision")
    require(len(final["paired"]) == 8, "paired metric count")
    with (ROOT / "docs/v6_p1i_graph_policy_confirmation.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    with (ROOT / "docs/v6_p1i_graph_policy_confirmation_paired.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    require(len(table) == 12 and len(paired) == 8, "CSV rows")
    require(not (CONFIG / "v6_p1i_graph_policy_e_raw/E_16384.json").exists(), "E 16384 must not exist")
    md = (ROOT / "docs/v6_p1i_graph_policy_confirmation.md").read_text()
    require("剩余 valid96" in md and "test/sealed" in md, "report scope")
    print(json.dumps({"status": "passed", "decision": final["decision"], "cells": 12, "paired_rows": 8}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
