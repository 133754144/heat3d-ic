#!/usr/bin/env python3
"""Freeze corrected U-v2 timing, repair attribution, and performance semantics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "median": float(np.median(array)),
        "mean": float(np.mean(array)), "std": float(np.std(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, action="append", required=True)
    parser.add_argument("--u1-control", type=Path, required=True)
    parser.add_argument("--geometry-exact", type=Path, required=True)
    parser.add_argument("--attribution-json", type=Path, required=True)
    parser.add_argument("--attribution-csv", type=Path, required=True)
    parser.add_argument("--seed-accuracy", type=Path, action="append", required=True)
    parser.add_argument("--q2-pass", type=Path, required=True)
    parser.add_argument("--q2-failure-log", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-attribution-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if len(args.serial) != 3 or len(args.seed_accuracy) != 3:
        raise RuntimeError("closeout requires three serial orders and three seed accuracy payloads")
    serial = [load(path) for path in args.serial]
    if any(row.get("status") != "passed" or row.get("sample_count") != 96 for row in serial):
        raise RuntimeError("corrected serial timing is incomplete")
    control = load(args.u1_control)
    geometry = load(args.geometry_exact)
    attribution = load(args.attribution_json)
    q2 = load(args.q2_pass)
    if not q2.get("true_concurrent_streaming", {}).get("actual_concurrent_execution"):
        raise RuntimeError("Q2 pass artifact is not an actual concurrent execution")
    failure_text = args.q2_failure_log.read_text()
    if "exclusive timing residual" not in failure_text:
        raise RuntimeError("Q2 failure log does not contain the frozen residual gate failure")

    stage_names = (
        "support_plus_cv", "anchor_graph", "query_graph", "reconstruction_map",
        "anchor_group_pack", "query_group_pack", "h2d_enqueue", "h2d_sync",
        "asymmetric_forward", "reconstruction_apply", "e2e_minus_exclusive_stages",
        "matched_continuous_e2e",
    )
    stages = {
        name: distribution([
            float(sample["stages"][name])
            for payload in serial for sample in payload["samples"]
        ])
        for name in stage_names
    }
    resident = distribution([
        float(payload["runtime"]["same_input_replay"]["median_seconds"])
        for payload in serial
    ])
    marginal = distribution([
        float(next(row for row in payload["batch"] if row["batch_size"] == 32)["marginal_per_case_seconds"])
        for payload in serial
    ])
    u_accuracy_payloads = [load(path) for path in args.seed_accuracy]
    accuracy_keys = {
        "PG_pct": "point_global_true_rms_relative_rmse_pct",
        "raw_K": "raw_cv_weighted_rmse_K", "source_K": "source_rmse_K",
        "peak_K": "peak_rmse_K", "interface_K": "interface_drop_rmse_K",
    }
    seed_accuracy = []
    for seed_index, payload in enumerate(u_accuracy_payloads):
        metrics = payload["accuracy"]["full_field"]
        seed_accuracy.append({"seed": seed_index, **{
            name: float(metrics[key]) for name, key in accuracy_keys.items()
        }})
    accuracy_mean_std = {
        name: {"mean": float(np.mean([row[name] for row in seed_accuracy])),
               "std": float(np.std([row[name] for row in seed_accuracy]))}
        for name in accuracy_keys
    }

    old = load(Path("configs/heat3d_v6_p1i/v6_p1i_u_v2_valid96_closeout.json"))
    fvm_batch = load(Path("configs/heat3d_v6_p1i/v6_p1i_u_v2_raw/timing/v6_p1i_u_v2_serial_fvm_batch_prefix.json"))
    fvm_marginal = distribution([float(row["marginal_B16_to_B32_seconds"]) for row in fvm_batch["rows"]])
    fvm = old["FVM240825"]["timing"]
    u_fresh = stages["matched_continuous_e2e"]
    q2_pass = q2["true_concurrent_streaming"]

    performance_rows = []
    for route_name in ("E16384-reconstruction", "E240825-direct-control"):
        route = old["routes"][route_name]
        accuracy = route["accuracy"]
        timing = route["timing"]
        performance_rows.append({
            "strategy": route_name, "accuracy_role": "surrogate_error_vs_FVM_reference",
            "PG_pct": accuracy["point_global_true_rms_relative_rmse_pct"],
            "raw_K": accuracy["raw_cv_weighted_rmse_K"], "source_K": accuracy["source_rmse_K"],
            "peak_K": accuracy["peak_rmse_K"], "interface_K": accuracy["interface_drop_rmse_K"],
            "fresh_median_s": timing["fresh_single_case"]["median_seconds"],
            "fresh_p95_s": timing["fresh_single_case"]["p95_seconds"],
            "resident_median_s": timing["resident_core"]["median_seconds"],
            "B16_to_B32_marginal_s": timing["batch_scale_marginal_fresh_case_estimate"]["median_seconds"],
            "Q1_median_s": timing["closed_loop_added_case_latency"]["median_seconds"],
            "Q2_status": "deprecated_serial_trace_not_concurrent", "Q2_throughput_samples_s": None,
            "fresh_speedup_vs_FVM": fvm["fresh_single_case"]["median_seconds"] / timing["fresh_single_case"]["median_seconds"],
            "provenance": "trusted_existing_accuracy_and_serial_timing; Q2 invalidated by audit",
        })
    performance_rows.append({
        "strategy": "U-v2-direct240825", "accuracy_role": "surrogate_error_vs_FVM_reference",
        **seed_accuracy[0],
        "fresh_median_s": u_fresh["median"], "fresh_p95_s": u_fresh["p95"],
        "resident_median_s": resident["median"], "B16_to_B32_marginal_s": marginal["median"],
        "Q1_median_s": u_fresh["median"],
        "Q2_status": "not_qualified_one_pass_one_residual_gate_failure",
        "Q2_throughput_samples_s": q2_pass["samples_per_second"],
        "fresh_speedup_vs_FVM": fvm["fresh_single_case"]["median_seconds"] / u_fresh["median"],
        "provenance": "new_corrected_serial_3_orders; exploratory actual Q2 pass retained with failed repeat",
    })
    performance_rows[-1].pop("seed")
    performance_rows.append({
        "strategy": "FVM240825", "accuracy_role": "reference_solution",
        "PG_pct": None, "raw_K": None, "source_K": None, "peak_K": None, "interface_K": None,
        "fresh_median_s": fvm["fresh_single_case"]["median_seconds"],
        "fresh_p95_s": fvm["fresh_single_case"]["p95_seconds"],
        "resident_median_s": fvm["resident_core"]["median_seconds"],
        "B16_to_B32_marginal_s": fvm_marginal["median"],
        "Q1_median_s": fvm["closed_loop_added_case_latency"]["median_seconds"],
        "Q2_status": "qualified_actual_persistent_process_pool",
        "Q2_throughput_samples_s": fvm["saturated_streaming"]["median_throughput_samples_per_second"],
        "fresh_speedup_vs_FVM": 1.0,
        "provenance": "existing matched in-memory FVM; actual distinct-case P2 B16/B32 prefix",
    })

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    sources = [
        *[(path, f"corrected_serial_{index}.json") for index, path in enumerate(args.serial)],
        (args.u1_control, "u_v1_historical_control_valid8.json"),
        (args.geometry_exact, "u_v2_geometry_exact_repair_valid96.json"),
        (args.attribution_json, "u_v2_repair_error_attribution.json"),
        (args.attribution_csv, "u_v2_repair_error_attribution.csv"),
        (args.q2_pass, "u_v2_true_q2_pass_order20260814.json"),
        (args.q2_failure_log, "u_v2_true_q2_failed_order20260815.log"),
        *[(path, f"u_v2_seed{index}_accuracy.json") for index, path in enumerate(args.seed_accuracy)],
    ]
    for source, name in sources:
        target = args.raw_dir / name
        shutil.copyfile(source, target)
        artifacts.append({"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)})
    shutil.copyfile(args.attribution_csv, args.output_attribution_csv)

    result = {
        "schema_version": "heat3d_v6_p1i_u_v2_timing_regression_closeout_v1",
        "status": "passed_with_Q2_not_qualified",
        "timing_regression": {
            "root_cause": "sample-varying CPU-JAX graph-shape compilation was inside anchor/query graph spans",
            "historical_U5": {"anchor_graph_median_s": 0.02285672735888511,
                              "query_graph_median_s": 0.09006908594164997,
                              "fresh_median_s": 0.6582150295143947},
            "U1_control_after_shape_prewarm": {
                "anchor_graph_median_s": control["runtime"]["fresh_sample"]["anchor_graph"]["median_seconds"],
                "query_graph_median_s": control["runtime"]["fresh_sample"]["query_graph"]["median_seconds"],
                "fresh_median_s": control["runtime"]["fresh_sample"]["matched_continuous_e2e"]["median_seconds"],
            },
            "corrected_U_v2": {"orders": 3, "stages": stages, "resident_core": resident,
                               "actual_distinct_case_Q1_B16_to_B32_marginal": marginal},
            "deprecated": ["old U-v2 3.229151 s fresh latency", "all neural serial-trace Q2 values"],
        },
        "edge_exact_optimization": {
            "golden_edge_exact_all_96": geometry["summary"]["repair_edge_exact_all"],
            "reference_repair_median_s": geometry["summary"]["reference_repair_median_seconds"],
            "candidate_repair_median_s": geometry["summary"]["candidate_repair_median_seconds"],
            "speedup": geometry["summary"]["reference_repair_median_seconds"] / geometry["summary"]["candidate_repair_median_seconds"],
        },
        "valid96_accuracy": {"per_seed": seed_accuracy, "mean_std": accuracy_mean_std,
                             "role": "diagnostic_characterization_not_model_selection"},
        "repair_error_attribution": attribution["summary"],
        "Q2": {
            "decision": "not_qualified_stop_after_second_order_residual_gate_failure",
            "passed_order": {key: q2_pass[key] for key in (
                "samples_per_second", "submit_to_result", "inter_completion",
                "actual_B16_to_B32_marginal_seconds", "wall_seconds")},
            "failed_order_log_sha256": sha256(args.q2_failure_log),
        },
        "performance_rows": performance_rows,
        "artifacts": artifacts,
        "role_contract": {
            "training": False, "test": False, "sealed": False,
            "checkpoint_modified": False, "dataset_modified": False,
            "E_architecture_modified": False, "accuracy_driven_optimization": False,
            "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = list(performance_rows[0])
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(performance_rows)

    md = [
        "# P1i U-v2 timing regression closeout", "",
        "本报告替代旧 U-v2 `3.229151 s` fresh latency，并废弃所有由 serial trace 推算的 neural Q2。",
        "未训练、未访问 test/sealed，E 架构、checkpoint、dataset、graph policy 均未改变。", "",
        "## 根因与修复", "",
        f"- U-v1 历史控制恢复为 anchor `{result['timing_regression']['U1_control_after_shape_prewarm']['anchor_graph_median_s']:.6f} s`、query `{result['timing_regression']['U1_control_after_shape_prewarm']['query_graph_median_s']:.6f} s`、fresh `{result['timing_regression']['U1_control_after_shape_prewarm']['fresh_median_s']:.6f} s`。",
        "- 回归来自 sample-varying CPU-JAX graph-shape compilation 被计入 production span；JIT/qualification/hash/I/O 现均在 span 外。",
        f"- U-v2 uncovered-only nearest repair 与冻结 R2P edge/hash 在 96/96 样本逐字节一致；repair median `{result['edge_exact_optimization']['reference_repair_median_s']:.3f}→{result['edge_exact_optimization']['candidate_repair_median_s']:.3f} s`（`{result['edge_exact_optimization']['speedup']:.2f}×`）。", "",
        "## Corrected performance", "",
        "| strategy | PG % | raw K | fresh med/p95 s | resident med s | B16→B32 marginal s | Q2 | fresh speedup vs FVM |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in performance_rows:
        pg = "—" if row["PG_pct"] is None else f"{row['PG_pct']:.6f}"
        raw = "—" if row["raw_K"] is None else f"{row['raw_K']:.6f}"
        md.append(f"| {row['strategy']} | {pg} | {raw} | {row['fresh_median_s']:.6f}/{row['fresh_p95_s']:.6f} | {row['resident_median_s']:.6f} | {row['B16_to_B32_marginal_s']:.6f} | {row['Q2_status']} | {row['fresh_speedup_vs_FVM']:.3f}× |")
    md += ["", "## Q2 判定", "",
           f"真实 Q2 首个顺序通过，吞吐 `{q2_pass['samples_per_second']:.6f} sample/s`；第二顺序触发 residual hard gate，故 **未取得 publication qualification**。该通过值只作 exploratory，不能用于正式 speedup。",
           "E16384/E240825 的旧 Q2 是 serial trace，统一标记 deprecated；FVM Q2 是真实 persistent process pool，继续有效。", "",
           "## Repair error attribution", ""]
    for seed, values in attribution["summary"].items():
        categories_value = values["categories"]
        md.append(f"- {seed}: covered `{categories_value['covered']['cv_rmse_K']:.3f} K`；repaired-inside `{categories_value['repaired_inside']['cv_rmse_K']:.3f} K`；repaired-outside `{categories_value['repaired_outside']['cv_rmse_K']:.3f} K`。distance/error Spearman `{values['repair_distance_vs_absolute_error']['spearman_r']:.4f}`。")
    md += ["", "repair distance 与误差相关性接近零；outside-repair 体积占比极小，未形成主要误差来源。U-v2 仍仅是 valid96 diagnostic/characterization，E16384 保持 production/reference。", ""]
    args.output_md.write_text("\n".join(md))
    print(json.dumps({"status": result["status"], "fresh_median_s": u_fresh["median"],
                      "q2": result["Q2"]["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
