#!/usr/bin/env python3
"""Deterministic gate for the V6 performance final correction."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOC = ROOT / "docs"


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def main():
    p = json.loads((CFG / "v6_p1i_performance_final_correction_protocol.json").read_text())
    c = json.loads((CFG / "v6_p1i_performance_final_correction_closeout.json").read_text())
    if c["status"] != "passed" or c["performance_freeze"] != "GO": fail("closeout did not pass")
    if c["shared_graph_runtime_fix"]["status"] != "passed" or not c["shared_graph_runtime_fix"]["byte_exact"]: fail("graph exact gate")
    if c["shared_graph_runtime_fix"]["route_specific_prewarm"]: fail("route-specific prewarm")
    if c["native1024_encoder_graph"] != "unchanged": fail("native graph changed")
    roles = c["role_contract"]
    for key in ("training","test","sealed","checkpoint_modified","dataset_modified","manifest_modified","graph_policy_modified","accuracy_driven_tuning"):
        if roles[key]: fail(f"forbidden role/state: {key}")
    expected={"E16384_reconstruction","U_v2_16384_reconstruction","U_v2_direct240825","E240825_direct_control","FVM240825_reference"}
    rows=c["timing_rows"]
    if {r["route"] for r in rows} != expected: fail("route set")
    for r in rows:
        if r["output_nodes"] != 240825: fail("non-240825 output")
        if not r["Q2_all_orders_passed"]: fail(f"Q2 failed: {r['route']}")
        if r["timing_boundary"] != "in_memory_k_q_BC_to_synchronized_240825_result": fail("boundary")
        for key in ("fresh_median_s","fresh_p95_s","resident_core_median_s","B16_to_B32_marginal_median_s","Q2_submit_to_result_median_s","Q2_inter_completion_median_s","Q2_samples_per_s_median"):
            if not isinstance(r[key],(int,float)) or r[key] <= 0: fail(f"non-finite/non-positive {r['route']} {key}")
    if not c["historical_and_pre_fix_failed_Q2_orders"]["retained"]: fail("failed Q2 evidence lost")
    dep=set(p["evidence_correction"]["deprecated"])
    if {"E16384_fresh_2.383_seconds","E240825_fresh_2.042_seconds","all_speedups_derived_from_those_values"} - dep: fail("deprecated evidence missing")
    if p["evidence_correction"]["U_v2_1.520_seconds_role"] != "steady_shape_fresh_not_unseen_shape_fresh": fail("U 1.520 role")
    with (DOC/"v6_p1i_performance_final_correction.csv").open() as handle:
        csv_rows=list(csv.DictReader(handle))
    if len(csv_rows)!=5: fail("performance CSV row count")
    with (DOC/"v6_p1i_performance_final_resolution_accuracy.csv").open() as handle:
        resolution=list(csv.DictReader(handle))
    if not resolution: fail("resolution CSV empty")
    summary=(DOC/"v6_stage_summary_and_performance.md").read_text()
    for phrase in ("shared shape-compile contamination", "steady-shape fresh", "U-v2 16384", "output-query R2P"):
        if phrase not in summary: fail(f"summary phrase missing: {phrase}")
    print(json.dumps({"status":"passed","routes":len(rows),"q2_all_passed":True,"test":False,"sealed":False,"training":False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
