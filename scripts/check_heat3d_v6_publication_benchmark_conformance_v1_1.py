#!/usr/bin/env python3
"""Fail-closed checker for the real-route benchmark v1.1 smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROUTES = {
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control", "FVM240825_reference",
}


def require(value, message: str) -> None:
    if not value:
        raise SystemExit(f"ERROR: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    result = json.loads(args.result.read_text())
    require(protocol["schema_version"] == "heat3d_v6_publication_benchmark_standard_v1_1", "protocol schema")
    require(protocol["status"] == "frozen_before_real_route_conformance_smoke", "protocol status")
    require(set(protocol["routes"]) == ROUTES, "route set")
    require(protocol["randomized_order_seeds"] == [20260814, 20260815, 20260816], "order seeds")
    require(protocol["lifecycle"]["service_modes"] == ["serial", "Q2"], "service modes")
    require(protocol["lifecycle"]["required_full_measurement_processes"] == 30, "lifecycle count")
    require(protocol["timing_boundary"] == "in_memory_k_q_BC_to_synchronized_240825_result", "boundary")
    require(protocol["residual_gate"]["synthetic_other_or_residual_stage_forbidden"], "synthetic residual stage")
    require(result["status"] == "passed", "smoke status")
    require(result["benchmark_protocol_v1_1"] == "GO", "protocol GO")
    require(result["benchmark_implementation_freeze"] == "GO", "implementation GO")
    require(result["publication_timing_freeze"] == "NO_GO_pending_full_valid32", "timing remains NO-GO")
    require(result["sample_count"] in (2, 3, 4), "low-cost population")
    rows = result["rows"]
    require(len(rows) == 30 and result["independent_process_count"] == 30, "30 rows")
    require(len({int(row["process_id"]) for row in rows}) == 30, "independent PIDs")
    require({row["route"] for row in rows} == ROUTES, "all routes")
    require({row["service_mode"] for row in rows} == {"serial", "Q2"}, "all modes")
    for seed in protocol["randomized_order_seeds"]:
        selected = [row for row in rows if row["order_seed"] == seed]
        require(len(selected) == 10, f"seed {seed} row count")
        orders = []
        for row in selected:
            value = row["ordered_sample_ids"]
            orders.append(value[str(seed)] if isinstance(value, dict) else value)
        require(all(order == orders[0] for order in orders), f"seed {seed} cross-route order")
    for row in rows:
        if not row["route"].startswith("FVM"):
            pools = row["timing_pool_classification"]
            require(not pools["pools_mixed"], "timing pools mixed")
            require(pools["repeat_case_cache_hot"] == "not_measured_in_conformance_smoke", "cache-hot mislabeled")
        if row["route"].startswith("E"):
            require(row["checkpoint_unchanged"], f"{row['route']} checkpoint")
            require(row["warmup"]["source_split"] == "train", f"{row['route']} warmup split")
            require(not row["warmup"]["target_read"], f"{row['route']} warmup target")
            require(not row["warmup"]["timed_graph_or_packing_prebuilt"], f"{row['route']} timed prewarm")
            if row["service_mode"] == "serial":
                require(row["serial_orders"] and not row["Q2_orders"], "E serial pool")
                for order in row["serial_orders"]:
                    for service in order["rows"]:
                        require(0 <= service["residual_seconds"] <= service["residual_limit_seconds"], "E residual")
            else:
                require(not row["serial_orders"] and row["Q2_orders"], "E Q2 pool")
                require(all(item["status"] == "passed" for item in row["Q2_orders"]), "E Q2")
                for order in row["Q2_orders"]:
                    for service in order["rows"]:
                        require(0 <= service["service_residual_seconds"] <= service["service_residual_limit_seconds"], "E Q2 residual")
        elif row["route"].startswith("U"):
            require(row["checkpoint_parameters_unchanged"], f"{row['route']} checkpoint")
            require(row["warmup"]["source_split"] == "train", f"{row['route']} warmup split")
            require(not row["warmup"]["target_read"], f"{row['route']} warmup target")
            require(row["concurrent_only"] == (row["service_mode"] == "Q2"), "U serial/Q2 pool")
            if row["service_mode"] == "Q2":
                require(row["true_concurrent_streaming"]["actual_concurrent_execution"], "U real Q2")
                require(row["true_concurrent_streaming"]["queue_depth"] == 2, "U Q2 depth")
            for service in row["samples"]:
                residual = service["stages"]["e2e_minus_exclusive_stages"]
                limit = service["timing_audit"]["exclusive_residual_limit_seconds"]
                require(0 <= residual <= limit, "U residual")
        else:
            require(row["worker_count"] == (1 if row["service_mode"] == "serial" else 2), "FVM P1/P2")
            require(row["classification"]["repeat_case_cache_hot"] == "not_measured_in_conformance_smoke", "FVM cache pool")
            for service in row["rows"]:
                require(0 <= service["residual_seconds"] <= service["residual_limit_seconds"], "FVM residual")
        require(int(row["aggregate_service_worker_peak_RAM_bytes"]) > 0, "aggregate RAM")
    neural_policies = {
        json.dumps(row["cpu_policy"], sort_keys=True)
        for row in rows if not row["route"].startswith("FVM")
    }
    require(len(neural_policies) == 1, "E/U CPU policy differs")
    exact = result["exactness_provenance"]
    require(exact["status"] == "passed", "exactness status")
    require(exact["1024"]["metadata_candidate_sha256"] == exact["1024"]["metadata_reference_sha256"], "1024 metadata")
    require(exact["1024"]["edge_candidate_sha256"] == exact["1024"]["edge_reference_sha256"], "1024 edges")
    require(exact["16384_direct_audit_count"] == 9, "16384 direct hash audits")
    require(exact["240825_direct_audit_count"] == 9, "240825 direct hash audits")
    require(not exact["prewritten_boolean_used_without_hash_comparison"], "prewritten exactness boolean")
    for key in ("training", "test", "sealed", "accuracy_tuning", "full_valid32_or_valid96_timing", "formal_speedup_generated"):
        require(result["role_contract"][key] is False, f"forbidden role {key}")
    require(result["publication_numbers_generated"] is False, "publication numbers")
    print(json.dumps({
        "status": "passed", "benchmark_protocol_v1_1": "GO",
        "benchmark_implementation_freeze": "GO",
        "publication_timing_freeze": "NO_GO_pending_full_valid32",
        "real_route_processes": 30, "test": False, "sealed": False, "training": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
