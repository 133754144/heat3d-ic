#!/usr/bin/env python3
"""Fail-closed checker for the V6 publication benchmark standard freeze."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROUTES = {
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control", "FVM240825_reference",
}


def require(value, message):
    if not value: raise SystemExit(f"ERROR: {message}")


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--smoke",type=Path,required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());smoke=json.loads(a.smoke.read_text())
    require(protocol["status"]=="frozen_before_publication_measurement","protocol status")
    require(set(protocol["routes"])==ROUTES,"route set")
    require(protocol["output_domain"]["timing_boundary"]=="in_memory_k_q_BC_to_synchronized_240825_result","boundary")
    require(protocol["lifecycle"]["randomized_order_seeds"]==[20260814,20260815,20260816],"seeds")
    require(protocol["lifecycle"]["independent_python_process_per_route_and_order"],"independent lifecycle")
    require(protocol["lifecycle"]["route_specific_shape_prewarm_forbidden"],"shape prewarm")
    require(protocol["lifecycle"]["warmup_case"]["must_not_belong_to_timed_population"],"warmup population")
    require(protocol["lifecycle"]["warmup_case"]["timed_case_graph_or_packing_shape_prewarm_forbidden"],"timed shape prewarm")
    require(protocol["fairness_gates"]["E_and_U_identical_neural_CPU_policy"],"CPU fairness")
    require(protocol["fairness_gates"]["Q2_serial_population_prepass_forbidden"],"Q2 prepass")
    require(protocol["fairness_gates"]["exclusive_residual_hard_gate"]=="0 <= residual <= max(0.025_seconds, 0.05_times_E2E)","residual gate")
    require(protocol["exactness_gates"]["resolutions"]==[1024,16384,240825],"exact coverage")
    for key in ("training","test","sealed","accuracy_tuning","full_valid32_or_valid96_timing","final_speedup_generated"):
        require(protocol["role_contract"][key] is False,f"forbidden role {key}")
    require(smoke["status"]=="passed","smoke status")
    require(smoke["benchmark_standard_freeze"]=="GO","standard freeze")
    require(smoke["publication_timing_freeze"]=="NO_GO_pending_full_measurement","timing freeze")
    require(smoke["route_order_process_count"]==15 and smoke["all_processes_independent"],"process lifecycle")
    for key in ("all_warmups_excluded","all_warmup_shapes_disjoint","E_U_CPU_policy_equal","all_Q2_real_concurrent","all_Q2_without_serial_prepass","all_residual_gates_executable","all_smoke_rows_passed"):
        require(smoke[key],key)
    require(smoke["publication_numbers_generated"] is False,"publication number leak")
    require(set(smoke["exactness_coverage"])=={"1024","16384","240825"},"exactness resolution keys")
    for resolution,row in smoke["exactness_coverage"].items():
        for key,value in row.items():
            if key not in {"source"}: require(value is True,f"{resolution} exactness {key}")
    pids=[row["pid"] for row in smoke["rows"]]
    require(len(pids)==len(set(pids))==15,"unique PIDs")
    require({row["route"] for row in smoke["rows"]}==ROUTES,"smoke route set")
    for row in smoke["rows"]:
        require(row["warmup_split"]=="train_input_only" and not row["warmup_target_read"],"warmup target")
        require(row["warmup_shape"]!=row["timed_shape"],"shape disjoint")
        require(row["Q2"]["worker_count"]==2 and row["Q2"]["distinct_k_q_BC_tokens"],"real Q2")
        require(not row["Q2"]["serial_prepass_of_Q2_samples"],"Q2 serial prepass")
        require(not row["publication_timing_eligible"],"smoke timing mislabeled")
    locations=smoke["packing_cache_audit"]["locations"]
    require(all(row["line_numbers"] for row in locations.values()),"packing/cache source locations")
    print(json.dumps({"status":"passed","benchmark_standard_freeze":"GO","publication_timing_freeze":"NO_GO_pending_full_measurement","processes":15,"test":False,"sealed":False,"training":False}))
    return 0
if __name__=="__main__":raise SystemExit(main())
