#!/usr/bin/env python3
"""Collect the frozen V6 source-aware ladder, multi-seed, and solver results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


MODES = (
    "upstream_like_joint_context_scale",
    "anchor_derived_context_scale_diagnostic",
)
METRICS = {
    "point_global_pct": "point_global_cv_relative_rmse_pct",
    "sample_first_pct": "sample_first_cv_relative_rmse_pct",
    "raw_rmse_K": "raw_cv_weighted_rmse_K",
    "shape_cv_rmse": "shape_cv_rmse",
    "scale_log_rmse": "scale_log_rmse",
    "field_bias_K": "field_cv_weighted_bias_K",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _e2e_excluding_warm(runtime: dict[str, Any]) -> float:
    if "end_to_end_seconds_valid128_excluding_warm_benchmark" in runtime:
        return float(runtime["end_to_end_seconds_valid128_excluding_warm_benchmark"])
    overhead = sum(runtime["warm_inference_seconds_batch1"]["values"])
    return float(runtime["end_to_end_seconds_valid128"]) - float(overhead)


def _flatten(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        metrics = payload["modes"][mode]
        row = {
            "seed": payload["seed"],
            "config_id": payload["config_id"],
            "checkpoint_epoch": payload["runtime"]["checkpoint_epoch"],
            "checkpoint_sha256": payload["runtime"]["checkpoint_sha256"],
            "resolution": payload["resolution"],
            "mode": mode,
            **{name: metrics[key] for name, key in METRICS.items()},
            "peak_rmse_K": metrics["peak"]["rmse_K"],
            "source_rmse_K": metrics["source_region"]["cv_weighted_rmse_K"],
            "layer_mean_rmse_K": metrics["layer_mean"]["rmse_K"],
            "layer_drop_rmse_K": metrics["layer_drop"]["rmse_K"],
            "top_surface_rmse_K": metrics["top_surface"]["cv_weighted_rmse_K"],
            "bottom_surface_rmse_K": metrics["bottom_surface"][
                "cv_weighted_rmse_K"
            ],
            "gate_passed": payload["resolution_gate"]["passed"],
            "stop_reasons": "|".join(payload["resolution_gate"]["stop_reasons"]),
            "test_hard_accessed": payload["test_hard_accessed"],
            "raw_payload_sha256": _sha256(path),
        }
        if mode == MODES[0]:
            runtime = payload["runtime"]
            row.update(
                {
                    "graph_build_seconds_valid128": runtime[
                        "graph_build_seconds_valid128"
                    ],
                    "first_compile_seconds_batch1": runtime[
                        "first_compile_inference_seconds_batch1"
                    ],
                    "warm_inference_seconds_batch1_mean": runtime[
                        "warm_inference_seconds_batch1"
                    ]["mean"],
                    "warm_repeat_count": runtime["warm_repeat_count"],
                    "formal_inference_seconds_valid128": runtime[
                        "formal_inference_seconds_valid128_including_first_compile"
                    ],
                    "end_to_end_seconds_valid128_excluding_warm_benchmark": (
                        _e2e_excluding_warm(runtime)
                    ),
                    "peak_ram_bytes": runtime["peak_ram_bytes"],
                    "gpu_memory": runtime["gpu_memory"],
                }
            )
        rows.append(row)
    return payload, rows


def _group_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["resolution"]), str(row["mode"])), []).append(row)
    result = []
    for (resolution, mode), values in sorted(grouped.items()):
        if {value["seed"] for value in values} != {"seed0", "seed1", "seed2"}:
            continue
        out: dict[str, Any] = {
            "resolution": resolution,
            "mode": mode,
            "seed_count": 3,
        }
        for name in METRICS:
            numbers = [float(value[name]) for value in values]
            out[f"{name}_mean"] = mean(numbers)
            out[f"{name}_std"] = stdev(numbers)
        for name in (
            "peak_rmse_K",
            "source_rmse_K",
            "layer_mean_rmse_K",
            "layer_drop_rmse_K",
            "top_surface_rmse_K",
            "bottom_surface_rmse_K",
        ):
            numbers = [float(value[name]) for value in values]
            out[f"{name}_mean"] = mean(numbers)
            out[f"{name}_std"] = stdev(numbers)
        result.append(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--timeout-resolution", type=int, default=32768)
    parser.add_argument("--timeout-seconds-lower-bound", type=float, required=True)
    parser.add_argument("--timeout-observed-rss-bytes", type=int, required=True)
    args = parser.parse_args()
    expected = [
        *(f"seed0_{value}.json" for value in (1024, 2048, 4096, 8192, 16384)),
        *(f"{seed}_{value}.json" for seed in ("seed1", "seed2") for value in (1024, 4096, 16384)),
    ]
    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name in expected:
        payload, flat = _flatten(args.raw_dir / name)
        payloads.append(payload)
        rows.extend(flat)
    seed0 = sorted(
        (
            row
            for row in rows
            if row["seed"] == "seed0" and row["mode"] == MODES[0]
        ),
        key=lambda row: int(row["resolution"]),
    )
    if not all(row["gate_passed"] for row in seed0):
        raise RuntimeError("a completed seed0 resolution unexpectedly failed")
    max_stable = int(seed0[-1]["resolution"])
    if args.timeout_resolution != 32768 or max_stable != 16384:
        raise RuntimeError("observed stop resolution drifted from completed search")
    timeout_row = {
        "seed": "seed0",
        "config_id": seed0[0]["config_id"],
        "checkpoint_epoch": seed0[0]["checkpoint_epoch"],
        "checkpoint_sha256": seed0[0]["checkpoint_sha256"],
        "resolution": args.timeout_resolution,
        "mode": MODES[0],
        "gate_passed": False,
        "stop_reasons": "unacceptable_end_to_end_runtime_before_graph_completion",
        "test_hard_accessed": False,
        "graph_build_seconds_valid128": f">{args.timeout_seconds_lower_bound}",
        "first_compile_seconds_batch1": None,
        "warm_inference_seconds_batch1_mean": None,
        "warm_repeat_count": 0,
        "formal_inference_seconds_valid128": None,
        "end_to_end_seconds_valid128_excluding_warm_benchmark": (
            f">{args.timeout_seconds_lower_bound}"
        ),
        "observed_rss_bytes_at_termination": args.timeout_observed_rss_bytes,
        "gpu_memory": "N/A_CPU_only",
        "metric_status": "not_available_inference_not_reached",
    }
    rows.append(timeout_row)
    solver = json.loads(args.solver.read_text(encoding="utf-8"))
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    solver_warm = float(solver["timing_seconds"]["solve"]["mean"])
    solver_cold = float(
        solver["timing_seconds"]["cold_total_mesh_assembly_first_solve"]
    )
    timing_rows = []
    for row in seed0:
        warm_model = float(row["warm_inference_seconds_batch1_mean"])
        amortized = (
            float(row["end_to_end_seconds_valid128_excluding_warm_benchmark"]) / 128
        )
        timing_rows.append(
            {
                "resolution": row["resolution"],
                "model_nodes": row["resolution"],
                "solver_nodes": solver["solver"]["node_count"],
                "dof_matched": False,
                "model_graph_build_seconds_valid128": row[
                    "graph_build_seconds_valid128"
                ],
                "model_first_compile_seconds_batch1": row[
                    "first_compile_seconds_batch1"
                ],
                "model_warm_inference_seconds_batch1": warm_model,
                "model_end_to_end_seconds_per_sample_amortized_valid128": amortized,
                "model_peak_ram_bytes": row["peak_ram_bytes"],
                "solver_warm_solve_seconds": solver_warm,
                "solver_cold_mesh_assembly_solve_seconds": solver_cold,
                "solver_peak_ram_bytes": solver["resources"]["peak_ram_bytes"],
                "warm_solve_over_model_inference_speedup": solver_warm / warm_model,
                "cold_solver_over_amortized_model_speedup": solver_cold / amortized,
                "gpu_memory": "N/A_CPU_only",
            }
        )
    timing_rows.append(
        {
            "resolution": args.timeout_resolution,
            "model_nodes": args.timeout_resolution,
            "solver_nodes": solver["solver"]["node_count"],
            "dof_matched": False,
            "model_graph_build_seconds_valid128": (
                f">{args.timeout_seconds_lower_bound}"
            ),
            "model_first_compile_seconds_batch1": "N/A_not_reached",
            "model_warm_inference_seconds_batch1": "N/A_not_reached",
            "model_end_to_end_seconds_per_sample_amortized_valid128": (
                "N/A_inference_not_reached"
            ),
            "model_observed_rss_bytes_at_termination": (
                args.timeout_observed_rss_bytes
            ),
            "solver_warm_solve_seconds": solver_warm,
            "solver_cold_mesh_assembly_solve_seconds": solver_cold,
            "solver_peak_ram_bytes": solver["resources"]["peak_ram_bytes"],
            "warm_solve_over_model_inference_speedup": "N/A_not_reached",
            "cold_solver_over_amortized_model_speedup": "N/A_not_reached",
            "gpu_memory": "N/A_CPU_only",
        }
    )
    multi_seed = _group_stats(rows)
    aggregate = {
        "schema_version": "heat3d_v6_source_aware_resolution_closeout_v1",
        "status": "completed",
        "scope": {
            "evaluation_role": "valid_iid",
            "test_hard_accessed": False,
            "training_executed": False,
            "checkpoint_modified": False,
            "formal_platform": "local_CPU",
        },
        "search": {
            "seed0_completed_resolutions": [row["resolution"] for row in seed0],
            "first_failed_resolution": args.timeout_resolution,
            "first_failure_reasons": [
                "unacceptable_end_to_end_runtime_before_graph_completion"
            ],
            "failure_elapsed_seconds_lower_bound": args.timeout_seconds_lower_bound,
            "failure_observed_rss_bytes_at_termination": (
                args.timeout_observed_rss_bytes
            ),
            "failure_metric_status": "not_available_inference_not_reached",
            "maximum_stable_resolution": max_stable,
            "next_ratio_exact_resolution_status": ladder["next_resolution_status"],
            "next_ratio_exact_resolution": ladder["next_resolution"],
            "next_ratio_exact_capacity_audit": ladder[
                "next_resolution_capacity_audit"
            ],
        },
        "workflow_decision": {
            "accepted": all(
                row["point_global_pct_mean"] < 20.0
                for row in multi_seed
                if row["mode"] == MODES[1]
            ),
            "minimum_error_scheme": (
                "1024 source-aware conditioning anchors plus nested source-aware "
                "high-resolution query nodes plus anchor-derived context and "
                "anchor-only scale pooling"
            ),
            "conditioning_query_distinction": (
                "conditioning support supplies q/k/BC/global-scale information; "
                "query support supplies field evaluation nodes. The frozen model "
                "uses xinp=xout=N in the main path, while anchor-derived context/scale "
                "is inference-only diagnostic evidence."
            ),
            "applicability": "P1h source-aware support family only",
            "error_attribution": {
                "main_finding": (
                    "The dominant upstream-like degradation with N is scale/context "
                    "distribution shift: positive field bias and scale-log error grow, "
                    "while anchor-derived context/scale removes most of that increase."
                ),
                "secondary_finding": (
                    "Shape CV-RMSE still rises gradually under anchor-derived scale, "
                    "consistent with residual regional-graph/query-distribution shift."
                ),
                "runtime_limit": (
                    "The 32768 stop is graph-construction scaling, not an observed "
                    "non-finite prediction or model-accuracy failure."
                ),
            },
        },
        "seed0_rows": seed0,
        "multi_seed_mean_std": multi_seed,
        "solver_benchmark": solver,
        "timing_rows": timing_rows,
        "raw_input_sha256": {
            path.name: _sha256(path) for path in (args.raw_dir / name for name in expected)
        },
    }
    args.json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    _write_csv(args.metrics_csv, rows + multi_seed)
    _write_csv(args.timing_csv, timing_rows)
    main_multi = [row for row in multi_seed if row["mode"] == MODES[0]]
    anchor_multi = [row for row in multi_seed if row["mode"] == MODES[1]]
    seed0_anchor = sorted(
        (
            row
            for row in rows
            if row.get("seed") == "seed0" and row.get("mode") == MODES[1]
        ),
        key=lambda row: int(row["resolution"]),
    )
    md = [
        "# V6 source-aware resolution limit and solver timing",
        "",
        "All formal model inference used local CPU, valid_iid only, batch=1. "
        "No training, test/hard access, or checkpoint mutation occurred.",
        "",
        f"- First failed resolution: {args.timeout_resolution} "
        f"(graph construction still incomplete after >{args.timeout_seconds_lower_bound:.0f} s; "
        "inference and field metrics were not reached)",
        f"- Maximum stable resolution: {max_stable}",
        f"- Full physical solver: {solver['solver']['node_count']} nodes; comparison is nonmatched-DOF.",
        "",
        "## Support qualification",
        "",
        "Every probe retains the original ordered 1024 P1h anchors and uses exact "
        "assigned-stratum ratios: source 50%, volume 25%, interface 12.5%, top "
        "6.25%, bottom 6.25%. Selection is label-independent.",
        "",
        "| N | min nodes/source box | p05 | median | 9 layers | 8 interfaces | top/bottom |",
        "|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for resolution in (1024, 2048, 4096, 8192, 16384, 32768):
        probe = ladder["probes"][str(resolution)]
        coverage = probe["source_box_coverage"]["node_count_per_source_box"]
        md.append(
            f"| {resolution} | {coverage['min']} | {coverage['p05']:.1f} | "
            f"{coverage['median']:.1f} | "
            f"{'yes' if probe['all_layers_covered'] else 'no'} | "
            f"{'yes' if probe['all_interfaces_covered'] else 'no'} | "
            f"{'yes' if probe['top_count'] and probe['bottom_count'] else 'no'} |"
        )
    md.extend(
        [
            "",
            "The next doubling (65536) cannot preserve the exact unique-node ratio: "
            f"the source stratum requires "
            f"{ladder['next_resolution_capacity_audit']['source']['required']} nodes "
            f"but has capacity "
            f"{ladder['next_resolution_capacity_audit']['source']['capacity']} "
            f"(shortfall "
            f"{ladder['next_resolution_capacity_audit']['source']['shortfall']}).",
            "",
        "## Seed0 upstream-like main path",
        "",
        "| N | point-global % | sample-first % | raw K | graph s | warm s | E2E valid128 s | RAM GiB | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in seed0:
        md.append(
            f"| {row['resolution']} | {row['point_global_pct']:.6f} | "
            f"{row['sample_first_pct']:.6f} | {row['raw_rmse_K']:.6f} | "
            f"{row['graph_build_seconds_valid128']:.3f} | "
            f"{row['warm_inference_seconds_batch1_mean']:.6f} | "
            f"{row['end_to_end_seconds_valid128_excluding_warm_benchmark']:.3f} | "
            f"{float(row['peak_ram_bytes']) / 2**30:.3f} | "
            f"{'pass' if row['gate_passed'] else 'fail'} |"
        )
    md.append(
        f"| {args.timeout_resolution} | N/A | N/A | N/A | "
        f">{args.timeout_seconds_lower_bound:.0f} | N/A | "
        f">{args.timeout_seconds_lower_bound:.0f} | "
        f"{args.timeout_observed_rss_bytes / 2**30:.3f} observed | fail |"
    )
    md.extend(
        [
            "",
            "## Three-seed stability",
            "",
            "| N | point-global mean±std % | sample-first mean±std % | raw mean±std K |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in main_multi:
        md.append(
            f"| {row['resolution']} | {row['point_global_pct_mean']:.6f} ± "
            f"{row['point_global_pct_std']:.6f} | "
            f"{row['sample_first_pct_mean']:.6f} ± {row['sample_first_pct_std']:.6f} | "
            f"{row['raw_rmse_K_mean']:.6f} ± {row['raw_rmse_K_std']:.6f} |"
        )
    md.extend(
        [
            "",
            "## Anchor-derived context and scale-pooling diagnostic",
            "",
            "| N | seed0 point-global % | seed0 sample-first % | seed0 raw K | 3-seed point-global mean±std % |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    anchor_multi_by_resolution = {
        int(row["resolution"]): row for row in anchor_multi
    }
    for row in seed0_anchor:
        resolution = int(row["resolution"])
        aggregate_row = anchor_multi_by_resolution.get(resolution)
        aggregate_text = (
            f"{aggregate_row['point_global_pct_mean']:.6f} ± "
            f"{aggregate_row['point_global_pct_std']:.6f}"
            if aggregate_row
            else "seed0 only"
        )
        md.append(
            f"| {resolution} | {row['point_global_pct']:.6f} | "
            f"{row['sample_first_pct']:.6f} | {row['raw_rmse_K']:.6f} | "
            f"{aggregate_text} |"
        )
    md.extend(
        [
            "",
            "## Same-host physical-solver comparison",
            "",
            f"The frozen 240825-node FVM replayed the archived valid sample with "
            f"`max_abs_error=0 K`; warm solve mean was `{solver_warm:.6f} s`, "
            f"cold mesh+assembly+first-solve was `{solver_cold:.6f} s`, and peak "
            f"RAM was `{solver['resources']['peak_ram_bytes'] / 2**30:.3f} GiB`.",
            f"The valid source-metadata audit found the smallest source-resolution-legal "
            f"P1h candidate at "
            f"`{solver['dof_comparability']['smallest_source_resolution_legal_candidate']['node_count']}` "
            "nodes (60x60x56), still far above the model ladder and without a P1h "
            "mesh-convergence qualification; the frozen 240825-node solver is therefore "
            "the only accuracy-qualified replay comparator.",
            "",
            "| model N | model warm batch1 s | warm solver/model ratio | model E2E/sample amortized s | cold solver/model ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows[:-1]:
        md.append(
            f"| {row['resolution']} | "
            f"{row['model_warm_inference_seconds_batch1']:.6f} | "
            f"{row['warm_solve_over_model_inference_speedup']:.3f}× | "
            f"{row['model_end_to_end_seconds_per_sample_amortized_valid128']:.6f} | "
            f"{row['cold_solver_over_amortized_model_speedup']:.3f}× |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main path is the upstream-like xinp=xout=N graph with context and "
            "joint pooling recomputed from N source-aware nodes.",
            "- The lowest-error frozen inference scheme uses the canonical 1024 "
            "source-aware anchors for context and scale pooling, plus nested "
            "high-resolution source-aware query nodes. It does not change the frozen checkpoint.",
            "- The upstream-like error increase is dominated by scale/context "
            "distribution shift: positive field bias and scale-log error rise with N. "
            "Anchor-derived context/scale removes most of it; the smaller remaining "
            "shape increase is consistent with regional-graph/query distribution shift.",
            "- 32768 failed on graph-construction runtime before inference, so this is "
            "an engineering scaling limit, not evidence of non-finite prediction or "
            "point-global accuracy failure.",
            "- Physical-solver speedups are nonmatched-DOF because the only frozen, "
            "replay-qualified solver mesh has 240825 nodes; no similarly sized mesh "
            "has an accuracy-equivalent mesh-convergence qualification.",
            "",
        ]
    )
    args.markdown.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "completed", "maximum_stable_resolution": max_stable}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
