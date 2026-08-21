#!/usr/bin/env python3
"""Fail-closed comparison and freeze for two graph-only qualification runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def require(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_payload(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": run["sample_count"],
        "train_only_warmup_count": run["train_only_warmup_count"],
        "sample_ids": run["sample_ids"],
        "train_only_warmup_sample_id": run["train_only_warmup_sample_id"],
        "routes": run["routes"],
        "records": run["records"],
        "observed_max_envelopes": run["observed_max_envelopes"],
        "historical_golden_checks": run["historical_golden_checks"],
        "graph_contract": run["graph_contract"],
        "role_contract": run["role_contract"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--attempt1-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_a = json.loads(args.run_a.read_text())
    run_b = json.loads(args.run_b.read_text())
    for label, run in (("A", run_a), ("B", run_b)):
        require(run["status"] == "passed", f"run {label} status")
        require(run["sample_count"] == 32 and len(run["records"]) == 165,
                f"run {label} population")
        require(all(row["real_unpadded_graph_hash_exact"]
                    for row in run["historical_golden_checks"]), f"run {label} golden")
        require(run["graph_contract"] == {
            "backend": "sparse_kdtree_v1", "graph_seed": 0,
            "radius_changed": False, "backend_semantics_changed": False,
            "u_v2_repair_changed": False, "model_inference_executed": False,
            "padding_applied": False,
        }, f"run {label} graph contract")
        require(run["role_contract"]["temperature_truth_read"] is False,
                f"run {label} truth access")
    require(exact_payload(run_a) == exact_payload(run_b),
            "independent graph-only runs differ in count/hash/metadata")

    attempt = json.loads(args.attempt1_raw.read_text())
    require(attempt["status"] == "failed_fail_closed", "Attempt 1 status")
    require(attempt["failure"]["returncode"] == 1, "Attempt 1 attempted")
    require(len(attempt["rows"]) == 0, "Attempt 1 completed cell count")
    require("2911 exceeds target 2905" in attempt["failure"]["stderr_tail"],
            "Attempt 1 padding failure")

    envelopes = run_a["observed_max_envelopes"]
    require(envelopes["native1024_anchor"]["p2r_edge_indices"] >= 2911,
            "native envelope did not derive failing capacity")
    result = {
        "schema_version": "heat3d_v6_publication_graph_envelope_qualification_v1",
        "status": "passed", "envelope_qualification": "GO",
        "ready_for_padding_equivalence": "GO",
        "qualification_runs": [
            {"path": str(args.run_a), "sha256": sha(args.run_a)},
            {"path": str(args.run_b), "sha256": sha(args.run_b)},
        ],
        "independent_process_count": 2,
        "per_sample_count_hash_metadata_exact": True,
        "historical_real_unpadded_graph_golden_exact": True,
        "padding_envelopes": envelopes,
        "capacity_derivation": "max_over_frozen_valid32_plus_one_target_free_train_static_warmup",
        "manual_capacity_override": False,
        "dummy_capacity_only": True,
        "attempt1_semantics_amendment": {
            "attempted": True, "completed": False,
            "publication_results_generated": False,
            "completed_cells": 0, "remaining_cells_not_started": 29,
            "failure_artifact_unchanged": True,
        },
        "role_contract": {
            "training": False, "model_inference": False, "accuracy_tuning": False,
            "test": False, "sealed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "envelope_qualification": "GO"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
