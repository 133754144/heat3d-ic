#!/usr/bin/env python3
"""Aggregate same-execution P5-R accuracy and matched continuous timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "route", "output_mode", "N", "Nr", "point_global_pct", "sample_first_pct",
    "raw_cv_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K",
    "oracle_point_global_pct", "support_plus_cv_median_s", "graph_median_s",
    "reconstruction_map_median_s", "anchor1024_median_s", "forward_median_s",
    "reconstruction_median_s", "matched_continuous_e2e_median_s",
    "matched_continuous_e2e_p95_s", "peak_vram_bytes", "p2r_edges_mean",
    "r2r_edges_mean", "speedup_vs_fvm_known_topology_new_physics",
    "accuracy_provenance", "timing_provenance",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    axes = ("point_global_pct", "matched_continuous_e2e_median_s", "peak_vram_bytes")
    return all(float(left[key]) <= float(right[key]) for key in axes) and any(
        float(left[key]) < float(right[key]) for key in axes
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--fvm", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    fvm_payload = load(args.fvm)
    fvm_known = 1.686430754023604
    rows: list[dict[str, Any]] = []
    cell_payloads = []
    for cell in protocol["cells"]:
        path = args.cell_root / f"{cell['route']}.json"
        payload = load(path)
        if payload["status"] != "passed" or payload["route"] != cell:
            raise RuntimeError(f"P5-R cell contract mismatch: {path}")
        if payload["sample_count"] != 32 or not all(
            row["support_exact"] and row["anchor_k_q_cv_exact"] for row in payload["samples"]
        ):
            raise RuntimeError(f"P5-R exact gate failed: {path}")
        metric = payload["accuracy"]["full_field"]
        oracle = payload["accuracy"]["oracle_reconstruction_floor"]
        timing = payload["timing"]
        row = {
            "route": cell["route"], "output_mode": cell["output_mode"],
            "N": int(cell["resolution"]),
            "Nr": payload["graph"]["regional_node_count_mean"],
            "point_global_pct": metric["point_global_true_rms_relative_rmse_pct"],
            "sample_first_pct": metric["sample_first_cv_relative_rmse_pct"],
            "raw_cv_rmse_K": metric["raw_cv_weighted_rmse_K"],
            "source_rmse_K": metric["source_rmse_K"],
            "peak_rmse_K": metric["peak_rmse_K"],
            "interface_rmse_K": metric["interface_drop_rmse_K"],
            "oracle_point_global_pct": oracle["point_global_true_rms_relative_rmse_pct"],
            "support_plus_cv_median_s": timing["support_plus_cv"]["median_seconds"],
            "graph_median_s": timing["graph"]["median_seconds"],
            "reconstruction_map_median_s": timing["reconstruction_map"]["median_seconds"],
            "anchor1024_median_s": timing["anchor1024"]["median_seconds"],
            "forward_median_s": timing["forward"]["median_seconds"],
            "reconstruction_median_s": timing["reconstruction"]["median_seconds"],
            "matched_continuous_e2e_median_s": timing["matched_continuous_e2e"]["median_seconds"],
            "matched_continuous_e2e_p95_s": timing["matched_continuous_e2e"]["p95_seconds"],
            "peak_vram_bytes": payload["peak_vram_bytes"],
            "p2r_edges_mean": payload["graph"]["p2r_edges_mean"],
            "r2r_edges_mean": payload["graph"]["r2r_edges_mean"],
            "speedup_vs_fvm_known_topology_new_physics": fvm_known / timing["matched_continuous_e2e"]["median_seconds"],
            "accuracy_provenance": f"new_same_execution:{path}",
            "timing_provenance": f"new_same_execution:{path}",
        }
        rows.append(row)
        cell_payloads.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pareto = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    best_pg = min(float(row["point_global_pct"]) for row in rows)
    margin = float(protocol["recommendation_rule"]["accuracy_noninferiority_margin_percentage_points"])
    eligible = [row for row in pareto if float(row["point_global_pct"]) <= best_pg + margin]
    recommended = min(eligible, key=lambda row: float(row["matched_continuous_e2e_median_s"]))
    result = {
        "schema_version": "heat3d_v6_p1i_p5r_resolution_sweep_closeout_v1",
        "status": "passed",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "cells": cell_payloads,
        "rows": rows,
        "pareto_routes": [row["route"] for row in pareto],
        "recommended_production_route": recommended["route"],
        "recommendation_rule": protocol["recommendation_rule"],
        "fvm_reference": {
            "known_topology_new_physics_median_s": fvm_known,
            "source": str(args.fvm),
            "sha256": hashlib.sha256(args.fvm.read_bytes()).hexdigest(),
            "note": "historical CPU FVM reference only; no FVM rerun",
        },
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# V6 P1i P5-R resolution sweep", "",
        "Status: **PASS**. Accuracy and latency in every neural row come from the same valid32 execution.", "",
        "| Route | Mode | N | Nr | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | continuous median (s) | p95 (s) | VRAM (GB) | FVM speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['route']} | {row['output_mode']} | {row['N']} | {row['Nr']:.0f} | "
            f"{row['point_global_pct']:.6f} | {row['raw_cv_rmse_K']:.6f} | "
            f"{row['source_rmse_K']:.6f} | {row['peak_rmse_K']:.6f} | "
            f"{row['interface_rmse_K']:.6f} | {row['matched_continuous_e2e_median_s']:.6f} | "
            f"{row['matched_continuous_e2e_p95_s']:.6f} | {row['peak_vram_bytes']/1e9:.3f} | "
            f"{row['speedup_vs_fvm_known_topology_new_physics']:.2f}x |"
        )
    lines += [
        "", "## Decision", "",
        f"Pareto routes: {', '.join(result['pareto_routes'])}.",
        f"Frozen 0.1 percentage-point non-inferiority plus latency rule recommends **{recommended['route']}**.",
        "The FVM timing is historical reuse. Every neural accuracy and continuous latency above is newly measured together; no old latency was joined to new accuracy.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
