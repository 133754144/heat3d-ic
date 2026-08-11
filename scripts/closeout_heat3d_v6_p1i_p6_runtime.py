#!/usr/bin/env python3
"""Aggregate P6 B1 runtime, throughput, and serial FVM results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def med(payload: dict, key: str) -> float:
    return float(payload["runtime"]["fresh_sample"][key]["median_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--e16384", type=Path, required=True)
    parser.add_argument("--e32768", type=Path, required=True)
    parser.add_argument("--fvm", type=Path, required=True)
    parser.add_argument("--p5r-raw-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    cells = [load(args.native), load(args.e16384), load(args.e32768)]
    fvm = load(args.fvm)
    if any(row["status"] != "passed" or row["sample_count"] != 32 for row in cells):
        raise RuntimeError("P6 NN cells incomplete")
    if fvm["status"] != "passed" or fvm["sample_count"] != 32:
        raise RuntimeError("P6 FVM incomplete")
    fvm_b1 = float(fvm["stage_timing"]["continuous_wall_seconds"]["median"])
    fvm_p95 = float(fvm["stage_timing"]["continuous_wall_seconds"]["p95"])
    fvm_batch_wall = float(sum(row["continuous_wall_seconds"] for row in fvm["measurements"]))
    fvm_batch_throughput = 32.0 / fvm_batch_wall
    old_names = {
        "native1024": "native1024_reconstruction.json",
        "E16384": "E16384_reconstruction.json",
        "E32768": "E32768_reconstruction.json",
    }
    rows = []
    diagnoses = []
    for cell in cells:
        route = cell["route"]["route"]
        accuracy = cell["accuracy_reused_from_p5r"]
        fresh = cell["runtime"]["fresh_sample"]
        old = load(args.p5r_raw_root / old_names[route])
        old_graph = float(old["timing"]["graph"]["median_seconds"])
        old_group = float(old["timing"]["group_and_h2d"]["median_seconds"])
        graph = med(cell, "anchor_graph") + med(cell, "query_graph")
        group = (
            med(cell, "anchor_group_pack") + med(cell, "query_group_pack")
            + med(cell, "h2d_enqueue") + med(cell, "h2d_sync")
        )
        diagnoses.append({
            "route": route,
            "p5r_graph_combined_median_s": old_graph,
            "p6_anchor_plus_query_graph_median_s": graph,
            "graph_ratio_p5r_over_p6": old_graph / graph,
            "p5r_group_and_h2d_median_s": old_group,
            "p6_group_pack_plus_h2d_median_s": group,
            "group_ratio_p5r_over_p6": old_group / group,
            "explanation": "P6 forces graph/group construction onto the CPU host, synchronizes each named boundary, and reuses native1024 anchor/query graph+group; P5-R used a coarser combined GPU-default span.",
        })
        for batch in cell["batch"]:
            row = {
                "system": route,
                "resolution": int(cell["route"]["resolution"]),
                "batch_size": int(batch["batch_size"]),
                "status": batch["status"],
                "point_global_pct": accuracy["point_global_pct"],
                "raw_cv_rmse_K": accuracy["raw_cv_rmse_K"],
                "source_rmse_K": accuracy["source_rmse_K"],
                "peak_rmse_K": accuracy["peak_rmse_K"],
                "interface_rmse_K": accuracy["interface_rmse_K"],
                "fresh_b1_median_s": med(cell, "matched_continuous_e2e"),
                "fresh_b1_p95_s": fresh["matched_continuous_e2e"]["p95_seconds"],
                "anchor_graph_median_s": med(cell, "anchor_graph"),
                "query_graph_median_s": med(cell, "query_graph"),
                "anchor_group_pack_median_s": med(cell, "anchor_group_pack"),
                "query_group_pack_median_s": med(cell, "query_group_pack"),
                "h2d_enqueue_median_s": med(cell, "h2d_enqueue"),
                "h2d_sync_median_s": med(cell, "h2d_sync"),
                "anchor_forward_median_s": med(cell, "anchor_forward"),
                "query_forward_median_s": med(cell, "query_forward"),
                "reconstruction_median_s": med(cell, "reconstruction_apply"),
                "same_input_replay_median_s": cell["runtime"]["same_input_replay"]["median_seconds"],
                "batch_wall_s": batch.get("batch_wall_seconds"),
                "samples_per_s": batch.get("samples_per_second"),
                "average_per_case_s": batch.get("average_per_case_seconds"),
                "marginal_per_case_s": batch.get("marginal_per_case_seconds"),
                "streamed_batch_median_s": (batch.get("streamed_prepared_host_batch") or {}).get("median_seconds"),
                "peak_vram_bytes": batch.get("peak_vram_bytes"),
                "b1_speedup_vs_serial_fvm": fvm_b1 / med(cell, "matched_continuous_e2e"),
                "batch_speedup_vs_serial_fvm_cases": (
                    None if batch.get("batch_wall_seconds") is None
                    else fvm_b1 * batch["batch_size"] / batch["batch_wall_seconds"]
                ),
                "provenance": "P6_new_runtime_P5R_reused_accuracy",
            }
            rows.append(row)
    result = {
        "schema_version": "heat3d_v6_p1i_p6_runtime_closeout_v1",
        "status": "completed",
        "protocol": protocol,
        "systems": cells,
        "fvm": {
            "b1_known_topology_new_physics_median_s": fvm_b1,
            "b1_p95_s": fvm_p95,
            "serial_32_case_total_wall_s": fvm_batch_wall,
            "serial_32_case_samples_per_s": fvm_batch_throughput,
            "processes": 1,
            "threads": 1,
            "semantic_note": "serial CPU independent cases; GPU batch speedups are explicitly against this serial reference",
        },
        "p5r_runtime_diagnosis": diagnoses,
        "rows": rows,
        "decision": {
            "support_optimization_stopped": True,
            "native1024_reuse_promoted": True,
            "production_accuracy_route_unchanged": "E16384",
            "throughput_note": "resident throughput is not single-case production latency",
        },
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# V6 P1i P6 runtime and throughput closeout", "",
        "Accuracy is reused from P5-R because graph/model/support semantics are unchanged. All latency values below are newly measured on WSL2 (RTX 5070 / Ryzen 7 9700X).", "",
        "| route | fresh B1 med/p95 (s) | resident replay (s) | B16 samples/s | B16 avg (ms) | B1 speedup vs serial FVM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        route = cell["route"]["route"]
        batch16 = next((row for row in cell["batch"] if row["batch_size"] == 16 and row["status"] == "passed"), None)
        lines.append(
            f"| {route} | {med(cell, 'matched_continuous_e2e'):.6f} / {cell['runtime']['fresh_sample']['matched_continuous_e2e']['p95_seconds']:.6f} | "
            f"{cell['runtime']['same_input_replay']['median_seconds']:.6f} | "
            f"{(batch16 or {}).get('samples_per_second', float('nan')):.2f} | "
            f"{1000*(batch16 or {}).get('average_per_case_seconds', float('nan')):.3f} | "
            f"{fvm_b1/med(cell, 'matched_continuous_e2e'):.2f}x |"
        )
    lines += [
        "", f"Serial FVM B1 median/p95: {fvm_b1:.6f}/{fvm_p95:.6f} s; 32-case serial wall {fvm_batch_wall:.6f} s ({fvm_batch_throughput:.3f} samples/s), one process and one thread.",
        "", "## P5-R timer explanation", "",
    ]
    for row in diagnoses:
        lines.append(
            f"- {row['route']}: graph {row['p5r_graph_combined_median_s']:.3f}→{row['p6_anchor_plus_query_graph_median_s']:.3f} s; "
            f"group+H2D {row['p5r_group_and_h2d_median_s']:.3f}→{row['p6_group_pack_plus_h2d_median_s']:.3f} s. "
            "The old values were coarse GPU-default combined spans; P6 uses synchronized CPU-host graph/pack and explicit H2D boundaries."
        )
    lines += [
        "", "Single-case latency, resident inference throughput, and streamed prepared-host throughput are separate workload semantics and must not be interchanged.",
        "No training, test/sealed access, checkpoint change, dataset change, or graph semantic change occurred.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
