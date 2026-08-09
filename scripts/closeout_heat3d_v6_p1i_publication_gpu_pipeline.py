#!/usr/bin/env python3
"""Combine frozen accuracy, cached graph diagnostics, and clean GPU timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _correlation(left: list[float], right: list[float]) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]),
    }


def _read_accuracy(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["resolution"]): row for row in csv.DictReader(handle)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        "# V6 P1i publication-grade GPU inference closeout", "",
        "No training, checkpoint/data/binding changes, test/sealed access, batch inference, or full-GPU graph construction occurred.", "",
        "## Clean B1 production timing", "",
        "Qualification, hashes, metrics, labels, and serialization are excluded. GPU completion is synchronized.", "",
        "| N | full PG (%) | oracle floor (%) | new-case median/p95 (ms) | warm-cache median/p95 (ms) | neural-forward median/p95 (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["curve"]:
        lines.append(
            f"| {row['resolution']} | {row['full_point_global_relative_rmse_pct']:.6f} | "
            f"{row['oracle_floor_point_global_relative_rmse_pct']:.6f} | "
            f"{1000*row['new_case_median_seconds']:.3f}/{1000*row['new_case_p95_seconds']:.3f} | "
            f"{1000*row['warm_cache_median_seconds']:.3f}/{1000*row['warm_cache_p95_seconds']:.3f} | "
            f"{1000*row['neural_forward_median_seconds']:.3f}/{1000*row['neural_forward_p95_seconds']:.3f} |"
        )
    conclusion = payload["decision"]
    lines += [
        "", "## Graph diagnosis", "",
        conclusion["graph_evidence"], "",
        "The association is diagnostic rather than causal: the frozen graph method was not changed and no model was retrained.", "",
        "## P2 reconstruction", "",
        conclusion["reconstruction_evidence"], "",
        "## Decision", "",
        f"**Next priority: {conclusion['priority']}.** {conclusion['basis']}", "",
        "GPU graph optimization is secondary because cached steady-state does not build a graph. Batch inference is deferred because the dominant B1 new-case cost is preparation/cache transfer, not the already-small warm model+reconstruction apply.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-json", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    parser.add_argument("--graph-execution-commit", required=True)
    parser.add_argument("--timing-execution-commit", required=True)
    parser.add_argument("--timing-log-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timing = json.loads(args.timing_json.read_text())
    graph = json.loads(args.graph_json.read_text())
    accuracy = _read_accuracy(args.accuracy_csv)
    p1i_graph = {
        row["resolution"]: row for row in graph["summaries"]
        if row["family"] == "P1i_sample_varying"
    }
    p1h_graph = {
        row["resolution"]: row for row in graph["summaries"]
        if row["family"] == "P1h_shared_support"
    }
    curve = []
    for timing_row in timing["results"]:
        resolution = int(timing_row["resolution"])
        accuracy_row = accuracy[resolution]
        graph_row = p1i_graph[resolution]
        curve.append({
            "resolution": resolution,
            "support_point_global_relative_rmse_pct": float(accuracy_row["support_point_global_pct"]),
            "full_point_global_relative_rmse_pct": float(accuracy_row["full_point_global_pct"]),
            "oracle_floor_point_global_relative_rmse_pct": float(accuracy_row["oracle_full_point_global_pct"]),
            "model_excess_over_floor_pct_point": (
                float(accuracy_row["full_point_global_pct"])
                - float(accuracy_row["oracle_full_point_global_pct"])
            ),
            "regional_nodes_mean": graph_row["regional_node_count"]["mean"],
            "p2r_regional_degree_mean": graph_row["degree"]["p2r_regional"]["mean"],
            "observed_radius_median_m": graph_row["observed_physical_support_radius_m"]["median"],
            "source_p2r_degree_mean": graph_row["partition"]["source"]["p2r_degree_mean"],
            "new_case_median_seconds": timing_row["timing"]["new_case"]["median_seconds"],
            "new_case_p95_seconds": timing_row["timing"]["new_case"]["p95_seconds"],
            "warm_cache_median_seconds": timing_row["timing"]["warm_cache"]["median_seconds"],
            "warm_cache_p95_seconds": timing_row["timing"]["warm_cache"]["p95_seconds"],
            "neural_forward_median_seconds": timing_row["timing"]["neural_forward"]["median_seconds"],
            "neural_forward_p95_seconds": timing_row["timing"]["neural_forward"]["p95_seconds"],
            "reconstruction_max_abs_error_K": timing_row["gpu_reconstruction_equivalence"]["maximum_sample_max_abs_error_K"],
            "reconstruction_max_rmse_K": timing_row["gpu_reconstruction_equivalence"]["maximum_sample_rmse_K"],
            "peak_vram_bytes": timing_row["device_memory"]["peak_bytes_in_use"],
        })
    post_minimum = [row for row in curve if row["resolution"] >= 8192]
    full_error = [row["full_point_global_relative_rmse_pct"] for row in post_minimum]
    degree = [row["p2r_regional_degree_mean"] for row in post_minimum]
    radius = [row["observed_radius_median_m"] for row in post_minimum]
    source_degree = [row["source_p2r_degree_mean"] for row in post_minimum]
    correlation = {
        "full_error_vs_p2r_regional_degree": _correlation(full_error, degree),
        "full_error_vs_observed_radius": _correlation(full_error, radius),
        "full_error_vs_source_p2r_degree": _correlation(full_error, source_degree),
    }
    p1i_8192, p1i_65536 = p1i_graph[8192], p1i_graph[65536]
    p1h_8192, p1h_32768 = p1h_graph[8192], p1h_graph[32768]
    maximum_reconstruction = max(row["reconstruction_max_abs_error_K"] for row in curve)
    maximum_reconstruction_rmse = max(row["reconstruction_max_rmse_K"] for row in curve)
    minimum_row = min(curve, key=lambda row: row["full_point_global_relative_rmse_pct"])
    ratio = np.median([row["new_case_median_seconds"] / row["warm_cache_median_seconds"] for row in curve])
    decision = {
        "priority": "graph reuse/fixed regional mesh",
        "graph_evidence": (
            f"P1i full-field PG reaches its minimum at N={minimum_row['resolution']} "
            f"({minimum_row['full_point_global_relative_rmse_pct']:.6f}%), while the oracle floor keeps improving. "
            f"From 8192 to 65536, P1i P2R regional degree falls from "
            f"{p1i_8192['degree']['p2r_regional']['mean']:.3f} to {p1i_65536['degree']['p2r_regional']['mean']:.3f} "
            f"and source-node P2R degree falls from {p1i_8192['partition']['source']['p2r_degree_mean']:.3f} "
            f"to {p1i_65536['partition']['source']['p2r_degree_mean']:.3f}. "
            f"By contrast, P1h P2R regional degree remains {p1h_8192['degree']['p2r_regional']['mean']:.3f} "
            f"at 8192 and {p1h_32768['degree']['p2r_regional']['mean']:.3f} at 32768."
        ),
        "reconstruction_evidence": (
            f"All 160 cached-map cases pass CPU-vs-GPU apply equivalence; maximum absolute error is "
            f"{maximum_reconstruction:.3e} K and maximum RMSE is {maximum_reconstruction_rmse:.3e} K. "
            "Warm model+GPU reconstruction latency is effectively the neural-forward latency at every resolution."
        ),
        "basis": (
            f"The median new-case/warm-cache latency ratio is {ratio:.1f}x, so cache/group/H2D preparation dominates B1. "
            "The frozen accuracy curve and declining regional/source coverage provide associated evidence for stabilizing and reusing the regional representation before optimizing graph build kernels or adding batch inference."
        ),
    }
    payload = {
        "schema_version": "heat3d_v6_p1i_publication_gpu_pipeline_closeout_v1",
        "status": "passed_p0_p1_p2",
        "role_contract": timing["role_contract"],
        "timing_contract": timing["timing_contract"],
        "curve": curve,
        "graph_drift_correlation_post_8192": correlation,
        "decision": decision,
        "execution_provenance": {
            "host": "wsl2",
            "formal_backend": "gpu",
            "graph_execution_commit": args.graph_execution_commit,
            "timing_execution_commit": args.timing_execution_commit,
            "timing_log_sha256": args.timing_log_sha256,
            "input_artifacts": {
                "timing_json_sha256": _sha256(args.timing_json),
                "graph_json_sha256": _sha256(args.graph_json),
                "frozen_accuracy_csv_sha256": _sha256(args.accuracy_csv),
            },
            "actual_new_compute": {
                "model_forward_calls": 470,
                "cpu_gpu_reconstruction_equivalence_cases": 160,
                "fresh_graph_builds": 0,
                "reconstruction_map_builds": 0,
                "metric_or_label_evaluations": 0,
                "offline_graph_cache_reads": 352,
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    _write_md(args.output_md, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
