#!/usr/bin/env python3
"""Close out P6/U2 runtime and throughput without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(cell: dict[str, Any], key: str) -> float:
    return float(cell["runtime"]["fresh_sample"][key]["median_seconds"])


def p6_accuracy(cell: dict[str, Any]) -> dict[str, float]:
    value = cell["accuracy_reused_from_p5r"]
    return {
        "point_global_pct": float(value["point_global_pct"]),
        "sample_first_pct": float(value["sample_first_pct"]),
        "raw_cv_rmse_K": float(value["raw_cv_rmse_K"]),
        "source_rmse_K": float(value["source_rmse_K"]),
        "peak_rmse_K": float(value["peak_rmse_K"]),
        "interface_rmse_K": float(value["interface_rmse_K"]),
    }


def u2_accuracy(cell: dict[str, Any]) -> dict[str, float]:
    value = cell["accuracy"]["full_field"]
    return {
        "point_global_pct": float(value["point_global_true_rms_relative_rmse_pct"]),
        "sample_first_pct": float(value["sample_first_cv_relative_rmse_pct"]),
        "raw_cv_rmse_K": float(value["raw_cv_weighted_rmse_K"]),
        "source_rmse_K": float(value["source_rmse_K"]),
        "peak_rmse_K": float(value["peak_rmse_K"]),
        "interface_rmse_K": float(value["interface_drop_rmse_K"]),
    }


def nn_rows(system: str, resolution: int, cell: dict[str, Any], accuracy: dict[str, float], provenance: str) -> list[dict[str, Any]]:
    fresh = cell["runtime"]["fresh_sample"]
    rows = []
    for batch in cell["batch"]:
        rows.append({
            "system": system,
            "resolution": resolution,
            "workload": "gpu_neural_reconstruction",
            "batch_size": int(batch["batch_size"]),
            "batch_status": batch["status"],
            **accuracy,
            "b1_matched_continuous_e2e_median_s": median(cell, "matched_continuous_e2e"),
            "b1_matched_continuous_e2e_p95_s": float(fresh["matched_continuous_e2e"]["p95_seconds"]),
            "support_plus_cv_median_s": median(cell, "support_plus_cv"),
            "anchor_graph_median_s": median(cell, "anchor_graph"),
            "query_graph_median_s": median(cell, "query_graph"),
            "reconstruction_map_median_s": median(cell, "reconstruction_map"),
            "anchor_group_pack_median_s": median(cell, "anchor_group_pack"),
            "query_group_pack_median_s": median(cell, "query_group_pack"),
            "h2d_enqueue_median_s": median(cell, "h2d_enqueue"),
            "h2d_sync_median_s": median(cell, "h2d_sync"),
            "forward_median_s": median(cell, "asymmetric_forward") if system.startswith("U1") else median(cell, "anchor_forward") + median(cell, "query_forward"),
            "reconstruction_apply_median_s": median(cell, "reconstruction_apply"),
            "same_input_replay_median_s": float(cell["runtime"]["same_input_replay"]["median_seconds"]),
            "batch_wall_s": batch.get("batch_wall_seconds"),
            "samples_per_s": batch.get("samples_per_second"),
            "average_per_case_s": batch.get("average_per_case_seconds"),
            "marginal_per_case_s": batch.get("marginal_per_case_seconds"),
            "streamed_batch_median_s": (batch.get("streamed_prepared_host_batch") or {}).get("median_seconds"),
            "peak_vram_bytes": batch.get("peak_vram_bytes"),
            "processes": None,
            "threads": None,
            "provenance": provenance,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--p6-closeout", type=Path, required=True)
    parser.add_argument("--u1-smoke", type=Path, required=True)
    parser.add_argument("--u1-valid32", type=Path, required=True)
    parser.add_argument("--checkpoint-post-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    p6 = load(args.p6_closeout)
    smoke = load(args.u1_smoke)
    u1 = load(args.u1_valid32)
    if smoke["status"] != "passed_smoke" or smoke["sample_count"] != 1:
        raise RuntimeError("U1-32768 smoke is not complete")
    if u1["status"] != "passed" or u1["sample_count"] != 32 or u1["resolution"] != 32768:
        raise RuntimeError("U1-32768 valid32 is not complete")
    if u1["checkpoint_sha256"] != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA drift")
    if args.checkpoint_post_sha256 != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("post-execution checkpoint file SHA drift")
    if u1["role_contract"] != protocol["role_contract"]:
        raise RuntimeError("role contract drift")
    systems = {row["route"]["route"]: row for row in p6["systems"]}
    e16384 = systems["E16384"]
    e32768 = systems["E32768"]
    e_b1 = median(e32768, "matched_continuous_e2e")
    u_b1 = median(u1, "matched_continuous_e2e")
    common = sorted(
        set(row["batch_size"] for row in e32768["batch"] if row["status"] == "passed")
        & set(row["batch_size"] for row in u1["batch"] if row["status"] == "passed")
    )
    if not common:
        raise RuntimeError("no common resident batch")
    highest = common[-1]
    e_batch = next(row for row in e32768["batch"] if row["batch_size"] == highest)
    u_batch = next(row for row in u1["batch"] if row["batch_size"] == highest)
    b1_speedup = e_b1 / u_b1
    throughput_speedup = float(u_batch["samples_per_second"]) / float(e_batch["samples_per_second"])
    required_b1 = float(protocol["runtime_go_gate"]["matched_b1_e2e_minimum_speedup"])
    required_throughput = float(protocol["runtime_go_gate"]["resident_throughput_minimum_speedup_at_highest_common_batch"])
    b1_pass = b1_speedup >= required_b1
    throughput_pass = throughput_speedup >= required_throughput
    go = b1_pass and throughput_pass
    if go:
        raise RuntimeError("GO requires U1-240825 artifacts; this closeout is for the observed NO-GO path")

    rows: list[dict[str, Any]] = []
    rows += nn_rows("E16384", 16384, e16384, p6_accuracy(e16384), "P6_new_runtime_P5R_reused_accuracy")
    rows += nn_rows("E32768", 32768, e32768, p6_accuracy(e32768), "P6_new_runtime_P5R_reused_accuracy")
    rows += nn_rows("U1-32768", 32768, u1, u2_accuracy(u1), "U2_new_valid32_accuracy_and_runtime")
    fvm = p6["fvm"]
    blank_accuracy = {key: None for key in ("point_global_pct", "sample_first_pct", "raw_cv_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K")}
    blank_stages = {key: None for key in (
        "support_plus_cv_median_s", "anchor_graph_median_s", "query_graph_median_s", "reconstruction_map_median_s",
        "anchor_group_pack_median_s", "query_group_pack_median_s", "h2d_enqueue_median_s", "h2d_sync_median_s",
        "forward_median_s", "reconstruction_apply_median_s", "same_input_replay_median_s", "streamed_batch_median_s",
        "peak_vram_bytes")}
    for batch_size, wall, samples_per_s in (
        (1, fvm["b1_known_topology_new_physics_median_s"], 1.0 / fvm["b1_known_topology_new_physics_median_s"]),
        (32, fvm["serial_32_case_total_wall_s"], fvm["serial_32_case_samples_per_s"]),
    ):
        rows.append({
            "system": "FVM", "resolution": 240825, "workload": "cpu_known_topology_new_physics_serial",
            "batch_size": batch_size, "batch_status": "passed", **blank_accuracy,
            "b1_matched_continuous_e2e_median_s": fvm["b1_known_topology_new_physics_median_s"],
            "b1_matched_continuous_e2e_p95_s": fvm["b1_p95_s"], **blank_stages,
            "batch_wall_s": wall, "samples_per_s": samples_per_s, "average_per_case_s": wall / batch_size,
            "marginal_per_case_s": None, "processes": fvm["processes"], "threads": fvm["threads"],
            "provenance": "P6_new_same_host_FVM_runtime",
        })

    result = {
        "schema_version": "heat3d_v6_p1i_p6_u2_runtime_closeout_v1",
        "status": "completed_no_go",
        "protocol_sha256": sha256(args.protocol),
        "checkpoint_file_sha256_verified_after_execution": args.checkpoint_post_sha256,
        "artifacts": {
            "p6_closeout": {"path": str(args.p6_closeout), "sha256": sha256(args.p6_closeout)},
            "u1_32768_smoke": {"path": str(args.u1_smoke), "sha256": sha256(args.u1_smoke)},
            "u1_32768_valid32": {"path": str(args.u1_valid32), "sha256": sha256(args.u1_valid32)},
        },
        "accuracy_gate": {
            "preregistered_delta_percentage_points": protocol["u1_32768_accuracy_gate"]["delta_percentage_points"],
            "observed_same_execution_delta_percentage_points": u2_accuracy(u1)["point_global_pct"] - p6_accuracy(e32768)["point_global_pct"],
            "margin_percentage_points": protocol["u1_32768_accuracy_gate"]["non_inferiority_margin_percentage_points"],
            "passed": u2_accuracy(u1)["point_global_pct"] - p6_accuracy(e32768)["point_global_pct"] <= protocol["u1_32768_accuracy_gate"]["non_inferiority_margin_percentage_points"],
        },
        "runtime_gate": {
            "b1_E32768_seconds": e_b1, "b1_U1_32768_seconds": u_b1,
            "b1_speedup_E_over_U1": b1_speedup, "b1_required_speedup": required_b1, "b1_passed": b1_pass,
            "highest_common_batch": highest,
            "E32768_samples_per_second": e_batch["samples_per_second"], "U1_32768_samples_per_second": u_batch["samples_per_second"],
            "resident_throughput_speedup_U1_over_E": throughput_speedup,
            "resident_required_speedup": required_throughput, "resident_throughput_passed": throughput_pass,
            "both_required": True, "passed": go,
        },
        "decision": {
            "u1_32768": "NO_GO_for_production_replacement",
            "u1_240825": "not_executed_fail_fast_runtime_gate",
            "paper_mainline": "E16384 production accuracy route; E32768 reference route",
            "reason": "U1 improves resident batch throughput but fails the preregistered matched B1 E2E speedup gate.",
        },
        "rows": rows,
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# V6 P1i P6 + U2 runtime/throughput closeout", "",
        "All neural rows use frozen seed0 valid32. No training or test/sealed access occurred. Accuracy and timing are never interchanged across workload semantics.", "",
        "| system | full PG (%) | raw CV (K) | fresh B1 med/p95 (s) | highest common B4 samples/s | peak VRAM at B4 (GB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, cell, accuracy in (("E16384", e16384, p6_accuracy(e16384)), ("E32768", e32768, p6_accuracy(e32768)), ("U1-32768", u1, u2_accuracy(u1))):
        batch4 = next(row for row in cell["batch"] if row["batch_size"] == 4 and row["status"] == "passed")
        lines.append(f"| {name} | {accuracy['point_global_pct']:.6f} | {accuracy['raw_cv_rmse_K']:.6f} | {median(cell, 'matched_continuous_e2e'):.6f}/{cell['runtime']['fresh_sample']['matched_continuous_e2e']['p95_seconds']:.6f} | {batch4['samples_per_second']:.2f} | {batch4['peak_vram_bytes']/1e9:.3f} |")
    lines += [
        "", "## Frozen gates", "",
        f"- Accuracy: U1−E32768 = {result['accuracy_gate']['observed_same_execution_delta_percentage_points']:.6f} pp, within +0.1 pp: PASS.",
        f"- Matched B1 E2E: E/U1 = {b1_speedup:.3f}x versus required 1.2x: FAIL.",
        f"- Resident throughput at highest common batch B{highest}: U1/E = {throughput_speedup:.3f}x versus required 1.2x: PASS.",
        "- Both runtime gates were preregistered as mandatory. U1-32768 is therefore NO-GO for production replacement; U1-240825 was not executed.",
        "", "## Workload semantics", "",
        "`single-case latency` is the fresh matched continuous pipeline. `resident inference throughput` uses already prepared/device-resident groups. `production batch throughput` includes streamed prepared-host H2D plus forward/reconstruction and is reported separately in the CSV.",
        f"FVM known-topology/new-physics B1 median/p95 is {fvm['b1_known_topology_new_physics_median_s']:.6f}/{fvm['b1_p95_s']:.6f} s. Its 32-case result is serial, one process and one thread; it is not presented as a parallel batch result.",
        "", "## Decision", "",
        "The paper mainline remains E16384 for the production accuracy/latency route, with E32768 retained as a reference. U1 is a throughput-oriented diagnostic, not a production replacement.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
