#!/usr/bin/env python3
"""Fail-closed checker for the V6 pre-measurement publication seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from seal_heat3d_v6_publication_benchmark_pre_measurement import exact_records

ROOT = Path(__file__).resolve().parents[1]


def require(value: Any, message: str) -> None:
    if not value:
        raise SystemExit(f"ERROR: {message}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seal = json.loads(args.seal.read_text())
    require(protocol["status"] == "frozen_before_full_valid32_measurement", "protocol status")
    require(seal["status"] == "passed", "seal status")
    require(seal["pre_measurement_seal"] == "GO", "seal GO")
    require(seal["publication_timing_freeze"] == "NO_GO_ready_for_full_valid32", "timing status")
    golden = protocol["golden_exactness"]
    blob = subprocess.check_output(
        ("git", "show", f"{golden['source_commit']}:{golden['source_path']}"), cwd=ROOT)
    require(hashlib.sha256(blob).hexdigest() == golden["source_sha256"], "historical blob SHA")
    reference_records = exact_records(json.loads(blob))
    candidate_records = exact_records(json.loads(args.candidate.read_text()))
    require(candidate_records == reference_records, "candidate/reference direct SHA records")
    require(seal["historical_golden"]["records"] == reference_records, "seal/reference records")
    require(seal["historical_golden"]["source_sha256"] == golden["source_sha256"], "golden binding")
    require(seal["historical_golden"]["record_count"] == 12, "golden record count")
    require(seal["historical_golden"]["candidate_reference_direct_SHA_equal"], "direct golden compare marker")
    require(not seal["historical_golden"]["current_implementation_self_replay_is_only_reference"], "self replay reference")
    for record in seal["historical_golden"]["records"]:
        require(record["resolution"] in (16384, 240825), "golden resolution")
        require(record["native1024_graph_hashes"]["combined_sha256"], "native metadata/edge hash")
        require(record["query_graph_hashes"]["combined_sha256"], "query metadata/edge hash")
        require(record["prepared_payload_sha256"], "payload hash")
    expected_workloads = {
        "cold_service_first_case", "fresh_distinct_case", "repeat_case_cache_hot",
        "resident_core", "true_B16_to_B32_marginal", "Q1_closed_loop", "real_Q2",
    }
    require(set(protocol["complete_workloads"]) == expected_workloads, "workload contract")
    e_text = (ROOT / "scripts/benchmark_heat3d_v6_p1i_final_e_service.py").read_text()
    u_text = (ROOT / "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py").read_text()
    runner_text = (ROOT / "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py").read_text()
    for token in ("--publication-v1-1", "repeat_case_cache_hot", "resident_core",
                  "true_B16_to_B32_marginal_seconds"):
        require(token in e_text, f"E workload implementation: {token}")
    for token in ("--publication-v1-1", "repeat_case_cache_hot", "resident_core",
                  "actual_B16_to_B32_marginal_seconds", "actual_concurrent_execution"):
        require(token in u_text, f"U workload implementation: {token}")
    for token in ("--formal-measurement", "summed_process_HWM_upper_bound_bytes",
                  "prepared_system_solve_only", "serial_prepass_of_Q2_samples"):
        require(token in runner_text, f"unified runner workload implementation: {token}")
    require(protocol["FVM_contract"]["fresh_Q1"] == "persistent_in_process_P1_one_thread", "FVM P1")
    require(protocol["FVM_contract"]["Q2"] == "persistent_P2_each_one_thread", "FVM P2")
    require(protocol["runtime_state"]["memory_reporting"]["simultaneous_aggregate_RSS_claimed"] is False, "RAM semantics")
    require(len(seal["runtime_state"]) == 5, "five runtime states")
    for state in seal["runtime_state"].values():
        require(state["state_sha256"], "runtime state hash")
    for rel, expected in seal["frozen_implementation_sha256"].items():
        require(sha(ROOT / rel) == expected, f"frozen file SHA: {rel}")
    collector = (ROOT / "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py").read_text()
    for token in ("BOOTSTRAP_SEED = 20260821", "BOOTSTRAP_RESAMPLES = 20000",
                  "median_of_three_ratios", "pooled_96_ratio_used"):
        require(token in collector, f"collector freeze: {token}")
    for key in ("training", "test", "sealed", "accuracy_tuning",
                "full_valid32_timing_executed_in_this_seal",
                "formal_latency_or_speedup_generated_in_this_seal"):
        require(seal["role_contract"][key] is False, f"forbidden role: {key}")
    require(seal["new_benchmark_execution"]["case_count"] <= 2, "low-cost seal")
    print(json.dumps({
        "status": "passed", "pre_measurement_seal": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "golden_records": 12, "formal_latency_generated": False,
        "training": False, "test": False, "sealed": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
