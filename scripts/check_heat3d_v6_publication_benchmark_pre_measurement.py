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
from heat3d_v6_publication_lifecycle_schema import validate_cell

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
    parser.add_argument("--authoritative-raw", type=Path)
    parser.add_argument("--collector-result", type=Path)
    parser.add_argument("--artifact-sha-manifest", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seal = json.loads(args.seal.read_text())
    require(protocol["status"] == "frozen_before_full_valid32_measurement", "protocol status")
    require(seal["status"] == "passed", "seal status")
    require(seal["pre_measurement_seal"] == "GO", "seal GO")
    require(seal["ready_for_authoritative_valid32"] == "GO", "authoritative valid32 readiness")
    require(seal["publication_timing_freeze"] == "NO_GO_ready_for_full_valid32", "timing status")
    require(seal.get("benchmark_lifecycle_schema") == "GO", "lifecycle schema gate")
    require(seal.get("benchmark_runtime_isolation") == "GO", "runtime isolation gate")
    runtime_record = seal["runtime_isolation_regression"]
    runtime_path = ROOT / runtime_record["artifact_path"]
    require(runtime_path.is_file() and sha(runtime_path) == runtime_record["artifact_sha256"],
            "runtime isolation fixture SHA")
    runtime_result = json.loads(runtime_path.read_text())
    require(runtime_result["status"] == "passed", "runtime isolation fixture status")
    require(runtime_result["benchmark_runtime_isolation"] == "GO", "runtime isolation GO")
    require(runtime_record["audit_outside_production_timing"], "audit timing isolation")
    require(runtime_record["audit_outside_Q2_completion_refill"], "audit Q2 isolation")
    require(runtime_record["service_HWM_captured_before_untimed_audit"], "HWM isolation")
    require(runtime_record["inner_failure_artifact_verified"], "failure observability")
    schema_record = seal["lifecycle_schema_regression"]
    schema_path = ROOT / schema_record["artifact_path"]
    require(schema_path.is_file() and sha(schema_path) == schema_record["artifact_sha256"],
            "lifecycle schema fixture SHA")
    schema_result = json.loads(schema_path.read_text())
    require(schema_result["status"] == "passed", "lifecycle fixture status")
    require(schema_result["fixture_count"] == 10, "lifecycle fixture count")
    require(schema_result["collector_parsed_all_10"], "collector fixture parsing")
    require(schema_result["benchmark_lifecycle_schema"] == "GO", "lifecycle fixture GO")
    golden = protocol["golden_exactness"]
    blob = subprocess.check_output(
        ("git", "show", f"{golden['source_commit']}:{golden['source_path']}"), cwd=ROOT)
    require(hashlib.sha256(blob).hexdigest() == golden["source_sha256"], "historical blob SHA")
    reference_records = exact_records(json.loads(blob))
    candidate_records = exact_records(json.loads(args.candidate.read_text()))
    require(candidate_records == reference_records, "candidate/reference direct SHA records")
    final_padding = seal.get("final_padding_gate")
    if final_padding is None:
        require(seal["historical_golden"]["records"] == reference_records,
                "seal/reference records")
        require(seal["historical_golden"]["source_sha256"] == golden["source_sha256"],
                "golden binding")
    else:
        padding_golden_path = ROOT / final_padding["padding_golden_path"]
        padding_golden = json.loads(padding_golden_path.read_text())
        require(sha(padding_golden_path) == final_padding["padding_golden_sha256"],
                "padding-adjusted golden SHA")
        require(seal["historical_golden"]["records"] == padding_golden["records"],
                "seal/padding-adjusted records")
        require(seal["historical_golden"]["source_sha256"] == sha(padding_golden_path),
                "padding-adjusted golden binding")
        old_by_key = {
            (row["route"], row["order_seed"], row["sample_id"]): row
            for row in reference_records
        }
        require(len(padding_golden["records"]) == len(old_by_key),
                "padding-adjusted record count")
        for row in padding_golden["records"]:
            old = old_by_key[(row["route"], row["order_seed"], row["sample_id"])]
            require(row["native1024_graph_hashes"] == old["native1024_graph_hashes"],
                    "padding-adjusted native real graph drift")
            require(row["query_graph_hashes"] == old["query_graph_hashes"],
                    "padding-adjusted query real graph drift")
        padding_manifest = ROOT / final_padding["padding_manifest_path"]
        padding_result = ROOT / final_padding["result_path"]
        require(sha(padding_manifest) == final_padding["padding_manifest_sha256"],
                "final padding manifest SHA")
        require(sha(padding_result) == final_padding["result_sha256"],
                "final padding result SHA")
        gate = json.loads(padding_result.read_text())
        require(gate["padding_numerical_equivalence"] == "GO", "padding numerical gate")
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
                  "service_process_HWM_bytes", "in_process_persistent_P1_one_thread",
                  "prepared_system_solve_only", "serial_prepass_of_Q2_samples",
                  "formal implementation SHA drift"):
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
                  "paired_workload_bootstrap", "three_seed_bootstrap_used",
                  "Q2_and_B16_to_B32", "pooled_96_ratio_used"):
        require(token in collector, f"collector freeze: {token}")
    sanity_record = seal["FVM_in_process_P1_sanity"]
    sanity_path = ROOT / sanity_record["artifact_path"]
    sanity = json.loads(sanity_path.read_text())
    require(sha(sanity_path) == sanity_record["artifact_sha256"], "FVM sanity SHA")
    require(sanity["execution_model"] == "in_process_persistent_P1_one_thread", "FVM sanity model")
    require(sanity["worker_pids"] == [sanity["process_id"]], "FVM sanity PID")
    require(sanity["worker_count"] == 1 and not sanity["IPC_used_in_fresh_Q1"], "FVM sanity no IPC")
    require(len(sanity["rows"]) in (1, 2), "FVM sanity case count")
    for key in ("training", "test", "sealed", "accuracy_tuning",
                "full_valid32_timing_executed_in_this_seal",
                "formal_latency_or_speedup_generated_in_this_seal"):
        require(seal["role_contract"][key] is False, f"forbidden role: {key}")
    maximum_seal_cases = 8 if final_padding is not None else 2
    require(seal["new_benchmark_execution"]["case_count"] <= maximum_seal_cases,
            "low-cost seal")
    optional = (args.authoritative_raw, args.collector_result, args.artifact_sha_manifest)
    require(all(value is None for value in optional) or all(value is not None for value in optional),
            "authoritative closeout inputs must be supplied together")
    authoritative_checked = False
    if args.authoritative_raw is not None:
        raw = json.loads(args.authoritative_raw.read_text())
        collected = json.loads(args.collector_result.read_text())
        manifest = json.loads(args.artifact_sha_manifest.read_text())
        require(raw["status"] == "passed", "authoritative raw hard gates")
        require(raw["formal_measurement_attempted"] is True, "formal measurement attempted")
        require(raw["formal_matrix_completed"] is True, "formal matrix completed")
        require(raw["publication_results_generated"] is False,
                "raw matrix must predate publication result generation")
        require(raw["authoritative_full_valid32"] == "completed_hard_gates_passed", "valid32 completion")
        require(raw["publication_timing_freeze"] == "NO_GO_pending_collector", "raw freeze state")
        require(raw["sample_count"] == 32 and len(raw["rows"]) == 30, "30-cell valid32 matrix")
        require(raw["independent_process_count"] == 30, "independent process count")
        require(len({row["process_id"] for row in raw["rows"]}) == 30, "independent process PIDs")
        require(raw["same_seed_cross_route_order_exact"], "cross-route sample order")
        require(raw["Q2_without_serial_prepass"], "Q2 no serial prepass")
        require(raw["role_contract"]["test"] is False and raw["role_contract"]["sealed"] is False,
                "authoritative forbidden roles")
        for row in raw["rows"]:
            require(row["status"] == "passed", f"formal cell status: {row['route']}")
            validate_cell(row, formal=True)
            if row["route"] == "FVM240825_reference" and row["service_mode"] == "serial":
                require(row["execution_model"] == "in_process_persistent_P1_one_thread", "formal FVM P1")
                require(row["worker_pids"] == [row["process_id"]], "formal FVM P1 PID")
                require(not row["IPC_used_in_fresh_Q1"], "formal FVM P1 IPC")
        require(collected["publication_timing_freeze"] == "GO", "collector publication freeze")
        require(collected["formal_measurement_attempted"] is True, "collector attempted provenance")
        require(collected["formal_matrix_completed"] is True, "collector matrix provenance")
        require(collected["publication_results_generated"] is True,
                "collector publication provenance")
        require(collected["aggregation_contract"]["bootstrap_seed"] == 20260821, "collector bootstrap seed")
        require(collected["aggregation_contract"]["bootstrap_resamples"] == 20000, "collector resamples")
        require(not collected["aggregation_contract"]["pooled_96_ratio_used"], "collector pooled96")
        require(collected["aggregation_contract"]["three_lifecycle_repeats"] ==
                "median_and_min_max_only", "three lifecycle uncertainty")
        entries = manifest["artifacts"]
        require(len(entries) >= 61, "raw/log/SHA artifact coverage")
        for entry in entries:
            path = ROOT / entry["path"]
            require(path.is_file(), f"artifact exists: {entry['path']}")
            require(path.stat().st_size == entry["size_bytes"], f"artifact size: {entry['path']}")
            require(sha(path) == entry["sha256"], f"artifact SHA: {entry['path']}")
        authoritative_checked = True
    print(json.dumps({
        "status": "passed", "pre_measurement_seal": "GO",
        "ready_for_authoritative_valid32": "GO",
        "benchmark_lifecycle_schema": "GO",
        "benchmark_runtime_isolation": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "golden_records": 12, "formal_latency_generated": False,
        "training": False, "test": False, "sealed": False,
        "authoritative_valid32_checked": authoritative_checked,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
