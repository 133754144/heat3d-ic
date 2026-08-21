#!/usr/bin/env python3
"""Validate a fail-closed authoritative-valid32 attempt and its evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closeout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    closeout = json.loads(args.closeout.read_text())
    manifest = json.loads(args.manifest.read_text())
    attempt = closeout["authoritative_attempt2"]
    require(closeout["status"] == "failed_fail_closed_q2_result_assembly", "closeout status")
    require(attempt["attempted"] and not attempt["completed"], "attempt lifecycle")
    require(not attempt["publication_results_generated"], "publication results must be absent")
    require(attempt["completed_cells"] == 5 and attempt["attempted_processes"] == 6,
            "cell/process count")
    require(closeout["decision"]["publication_timing_freeze"] == "NO_GO", "freeze status")
    require(not closeout["decision"]["collector_executed"], "collector unexpectedly executed")
    require(not closeout["decision"]["single_cell_rerun"], "single-cell rerun recorded")
    raw_path = ROOT / attempt["raw_path"]
    require(sha(raw_path) == attempt["raw_sha256"], "raw SHA drift")
    raw = json.loads(raw_path.read_text())
    require(raw["status"] == "failed_fail_closed", "raw failure status")
    require(len(raw["rows"]) == 5 and len(raw["process_records"]) == 6, "raw counts")
    require(raw["failure"]["route"] == "E16384_reconstruction", "failed route")
    require(raw["failure"]["service_mode"] == "Q2", "failed mode")
    require(raw["failure"]["seed"] == 20260814, "failed seed")
    require(manifest["status"] == "complete_failure_evidence", "manifest status")
    require(manifest["completed_cell_count"] == 5, "manifest completed cells")
    for entry in manifest["artifacts"]:
        path = ROOT / entry["path"]
        require(path.is_file(), f"missing artifact: {entry['path']}")
        require(path.stat().st_size == entry["size_bytes"], f"size drift: {entry['path']}")
        require(sha(path) == entry["sha256"], f"SHA drift: {entry['path']}")
    role = closeout["role_contract"]
    require(not any(role[key] for key in (
        "training", "accuracy_tuning", "test", "sealed", "checkpoint_modified",
        "graph_policy_modified", "route_semantics_modified",
    )), "forbidden role/change")
    print(json.dumps({
        "status": "passed_failure_closeout", "publication_timing_freeze": "NO_GO",
        "completed_cells": 5, "attempted_processes": 6,
        "training": False, "test": False, "sealed": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
