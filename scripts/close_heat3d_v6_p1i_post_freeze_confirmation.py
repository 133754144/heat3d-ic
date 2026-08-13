#!/usr/bin/env python3
"""Freeze unified 240825-node timing and independent valid96 confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROUTES = ("E16384_reconstruction", "U_direct240825", "E240825_direct")
METRICS = {
    "point_global_true_rms_relative_rmse_pct": "PG_pct",
    "raw_cv_weighted_rmse_K": "raw_K",
    "source_rmse_K": "source_K",
    "peak_rmse_K": "peak_K",
    "interface_drop_rmse_K": "interface_K",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float], unit: str = "seconds") -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        f"median_{unit}": float(np.median(array)),
        f"mean_{unit}": float(np.mean(array)),
        f"p95_{unit}": float(np.quantile(array, 0.95)),
    }


def resolve_artifact(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.exists() else root / path.name


def load_matrix(
    root: Path, *, require_passed: bool = True,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path, dict[str, Any]]]]:
    state_path = root / "execution_state.json"
    state = json.loads(state_path.read_text())
    if require_passed and state.get("status") != "passed":
        raise RuntimeError(f"matrix is not passed: {root}")
    rows = []
    for cell in state["cells"]:
        if cell["returncode"] != 0:
            if require_passed:
                raise RuntimeError(f"failed matrix cell: {cell}")
            continue
        path = resolve_artifact(root, cell["result"])
        payload = json.loads(path.read_text())
        if not str(payload.get("status", "")).startswith("passed"):
            raise RuntimeError(f"result is not passed: {path}")
        rows.append((payload, path, cell))
    return state, rows


def timing_payload(result: dict[str, Any]) -> dict[str, Any]:
    if "route" in result:
        fresh = result["timing"]["matched_continuous_e2e"]
        resident = result["resident_core"]
    else:
        fresh = result["runtime"]["fresh_sample"]["matched_continuous_e2e"]
        resident = result["runtime"]["same_input_replay"]
    stream = result["streaming"]
    marginal = (float(stream["wall_seconds"]) - float(fresh["median_seconds"])) / 31.0
    return {
        "fresh_single_case": fresh,
        "resident_core": resident,
        "batch_scale_marginal_fresh_case_estimate_seconds": marginal,
        "batch_scale_estimate_definition": "(continuous valid32 stream wall - fresh median) / 31",
        "true_streaming_added_case_latency": stream["submit_to_result"],
        "inter_completion_interval": stream["inter_completion"],
        "stream_wall_seconds": float(stream["wall_seconds"]),
        "stream_throughput_samples_per_second": float(stream["samples_per_second"]),
    }


def route_name(result: dict[str, Any], path: Path) -> str:
    if "route" in result:
        route = result["route"]
        return str(route["route"] if isinstance(route, dict) else route)
    if "U_direct240825" in path.name:
        return "U_direct240825"
    raise RuntimeError(f"cannot identify route for {path}")


def exact_payload(result: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for row in result["samples"]:
        # Group trees include the order-dependent sample_idx leaf. It is
        # intentionally excluded from a randomized-order equivalence gate;
        # the frozen support, real graph, native anchor inputs and conserved
        # physical totals are the model semantics that must remain exact.
        keys = (
            "support_hash", "graph_hash", "anchor_k_q_cv_exact",
            "selected_power_W", "full_power_W", "selected_volume_m3", "full_volume_m3",
        ) if "support_hash" in row else (
            "prepared_payload_sha256", "input_physics_context_sha256",
            "support_1024_hash", "output_support_hash", "graph_hash",
        )
        payload = {key: row[key] for key in keys if key in row}
        output[row["sample_id"]] = payload
    return output


def exact_payload_by_sample(result: dict[str, Any]) -> dict[str, Any]:
    return exact_payload(result)


def aggregate_components(rows: list[dict[str, float | int]], indices: np.ndarray | None = None) -> dict[str, float]:
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    total = lambda key: float(sum(float(row[key]) for row in selected))
    return {
        "point_global_true_rms_relative_rmse_pct": math.sqrt(total("point_sse") / total("point_energy")) * 100.0,
        "raw_cv_weighted_rmse_K": math.sqrt(total("weighted_sse") / total("volume")),
        "source_rmse_K": math.sqrt(total("source_sse") / total("source_volume")),
        "peak_rmse_K": math.sqrt(total("peak_error_squared") / len(selected)),
        "interface_drop_rmse_K": math.sqrt(total("interface_error_squared_sum") / total("interface_error_count")),
    }


def bootstrap_difference(
    left: list[dict[str, float | int]], right: list[dict[str, float | int]], seed: int, repeats: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("paired rows have different lengths")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(left), size=(repeats, len(left)), endpoint=False)
    output = {}
    observed_left = aggregate_components(left)
    observed_right = aggregate_components(right)
    left_columns = {key: np.asarray([float(row[key]) for row in left]) for key in left[0]}
    right_columns = {key: np.asarray([float(row[key]) for row in right]) for key in right[0]}
    def sampled(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total = lambda key: np.sum(columns[key][draws], axis=1)
        return {
            "point_global_true_rms_relative_rmse_pct": np.sqrt(total("point_sse") / total("point_energy")) * 100.0,
            "raw_cv_weighted_rmse_K": np.sqrt(total("weighted_sse") / total("volume")),
            "source_rmse_K": np.sqrt(total("source_sse") / total("source_volume")),
            "peak_rmse_K": np.sqrt(total("peak_error_squared") / len(left)),
            "interface_drop_rmse_K": np.sqrt(total("interface_error_squared_sum") / total("interface_error_count")),
        }
    left_samples = sampled(left_columns); right_samples = sampled(right_columns)
    for metric in METRICS:
        deltas = left_samples[metric] - right_samples[metric]
        output[metric] = {
            "left_minus_right": observed_left[metric] - observed_right[metric],
            "bootstrap_95pct_CI": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--performance-root", type=Path, required=True)
    parser.add_argument("--fvm-result", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--performance-csv", type=Path, required=True)
    parser.add_argument("--confirmation-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    perf_state, perf_cells = load_matrix(args.performance_root)
    confirm_states = []
    confirm_cells = []
    for root in args.confirmation_root:
        state, cells = load_matrix(root, require_passed=False)
        confirm_states.append((root, state))
        confirm_cells.extend(cells)
    fvm = json.loads(args.fvm_result.read_text())
    if fvm.get("status") != "passed":
        raise RuntimeError("FVM result did not pass")

    by_route: dict[str, list[tuple[dict[str, Any], Path]]] = {route: [] for route in ROUTES}
    for result, path, cell in perf_cells:
        by_route[route_name(result, path)].append((result, path, cell))
    performance_rows = []
    performance = {}
    for route, cells in by_route.items():
        if len(cells) != 3:
            raise RuntimeError(f"{route}: expected three randomized orders")
        reference_exact = exact_payload_by_sample(cells[0][0])
        if any(exact_payload_by_sample(result) != reference_exact for result, _, _ in cells[1:]):
            raise RuntimeError(f"{route}: exact prepared payload changed across orders")
        route_timing = [timing_payload(result) for result, _, _ in cells]
        fresh = [float(row["fresh_single_case"]["median_seconds"]) for row in route_timing]
        fresh_p95 = [float(row["fresh_single_case"]["p95_seconds"]) for row in route_timing]
        resident = [float(row["resident_core"]["median_seconds"]) for row in route_timing]
        resident_p95 = [float(row["resident_core"]["p95_seconds"]) for row in route_timing]
        stream = [float(row["true_streaming_added_case_latency"]["median_seconds"]) for row in route_timing]
        stream_p95 = [float(row["true_streaming_added_case_latency"]["p95_seconds"]) for row in route_timing]
        interval = [float(row["inter_completion_interval"]["median_seconds"]) for row in route_timing]
        throughput = [float(row["stream_throughput_samples_per_second"]) for row in route_timing]
        marginal = [float(row["batch_scale_marginal_fresh_case_estimate_seconds"]) for row in route_timing]
        accuracy_rows = [result["accuracy"]["full_field"] for result, _, _ in cells]
        accuracy = dict(accuracy_rows[0])
        accuracy_drift = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in accuracy_rows]
            accuracy[metric] = float(np.mean(values))
            accuracy_drift[metric] = float(max(values) - min(values))
            if accuracy_drift[metric] > 0.005:
                raise RuntimeError(f"{route}: randomized-order numerical drift exceeded 0.005: {metric}")
        memory = max(int(result.get("peak_vram_bytes", result.get("memory", {}).get("peak_bytes_in_use", 0))) for result, _, _ in cells)
        performance[route] = {
            "accuracy_240825": accuracy,
            "fresh_single_case": {"median_across_orders_seconds": float(np.median(fresh)), "p95_across_orders_seconds": float(np.max(fresh_p95))},
            "resident_core": {"median_across_orders_seconds": float(np.median(resident)), "p95_across_orders_seconds": float(np.max(resident_p95))},
            "batch_scale_marginal_fresh_case_estimate": stats(marginal),
            "true_streaming_added_case_latency": {"median_across_orders_seconds": float(np.median(stream)), "p95_across_orders_seconds": float(np.max(stream_p95))},
            "inter_completion_interval": {"median_across_orders_seconds": float(np.median(interval)), "p95_across_orders_seconds": float(np.max([row["inter_completion_interval"]["p95_seconds"] for row in route_timing]))},
            "throughput": stats(throughput, "samples_per_second"),
            "peak_vram_bytes": memory,
            "randomized_order_count": 3,
            "exact_model_semantics_across_orders": True,
            "group_tree_hash_excluded_reason": "contains order-dependent sample_idx leaf",
            "randomized_order_accuracy_range": accuracy_drift,
        }
        performance_rows.append({
            "strategy": route, "output_nodes": 240825,
            **{column: accuracy[metric] for metric, column in METRICS.items()},
            "fresh_single_case_median_s": float(np.median(fresh)), "fresh_single_case_p95_s": float(np.max(fresh_p95)),
            "resident_core_median_s": float(np.median(resident)), "resident_core_p95_s": float(np.max(resident_p95)),
            "batch_scale_marginal_estimate_s": float(np.median(marginal)),
            "true_streaming_added_case_median_s": float(np.median(stream)), "true_streaming_added_case_p95_s": float(np.max(stream_p95)),
            "inter_completion_median_s": float(np.median(interval)), "throughput_samples_s": float(np.median(throughput)),
            "peak_vram_bytes": memory, "timing_semantics": "unified_post_freeze_240825",
        })

    p1 = next(row for row in fvm["rows"] if row["process_count"] == 1)
    saturated = next(row for row in fvm["rows"] if row["process_count"] == fvm["saturation_process_count"])
    fvm_summary = {
        "fresh_single_case": {
            "median_seconds": float(np.median([row["fresh_single_case"]["median_seconds"] for row in p1["repeats"]])),
            "p95_seconds": float(max(row["fresh_single_case"]["p95_seconds"] for row in p1["repeats"])),
        },
        "resident_core_solve_only": {
            "median_seconds": float(np.median([row["resident_core_solve_only"]["median_seconds"] for row in p1["repeats"]])),
            "p95_seconds": float(max(row["resident_core_solve_only"]["p95_seconds"] for row in p1["repeats"])),
        },
        "batch_scale_marginal_fresh_case_estimate_seconds": float(np.median([
            (row["streaming_wall_seconds"] - row["fresh_single_case"]["median_seconds"]) / 31.0
            for row in saturated["repeats"]
        ])),
        "true_streaming_added_case_latency": {
            "median_seconds": float(np.median([row["stream_submit_to_result"]["median_seconds"] for row in saturated["repeats"]])),
            "p95_seconds": float(max(row["stream_submit_to_result"]["p95_seconds"] for row in saturated["repeats"])),
        },
        "inter_completion_interval": {
            "median_seconds": float(np.median([row["stream_inter_completion"]["median_seconds"] for row in saturated["repeats"]])),
            "p95_seconds": float(max(row["stream_inter_completion"]["p95_seconds"] for row in saturated["repeats"])),
        },
        "streaming_throughput": saturated["streaming_throughput"],
        "saturation_process_count": fvm["saturation_process_count"],
        "resident_core_is_prepared_system_solve_only_not_e2e": True,
    }
    for route, values in performance.items():
        values["comparisons_to_FVM"] = {
            "fresh_single_case_speedup": fvm_summary["fresh_single_case"]["median_seconds"] / values["fresh_single_case"]["median_across_orders_seconds"],
            "resident_core_ratio_not_E2E_speedup": fvm_summary["resident_core_solve_only"]["median_seconds"] / values["resident_core"]["median_across_orders_seconds"],
            "true_streaming_throughput_ratio": values["throughput"]["median_samples_per_second"] / fvm_summary["streaming_throughput"]["median_seconds"],
        }
    performance_rows.append({
        "strategy": "FVM240825", "output_nodes": 240825,
        **{column: "reference" for column in METRICS.values()},
        "fresh_single_case_median_s": fvm_summary["fresh_single_case"]["median_seconds"],
        "fresh_single_case_p95_s": fvm_summary["fresh_single_case"]["p95_seconds"],
        "resident_core_median_s": fvm_summary["resident_core_solve_only"]["median_seconds"],
        "resident_core_p95_s": fvm_summary["resident_core_solve_only"]["p95_seconds"],
        "batch_scale_marginal_estimate_s": fvm_summary["batch_scale_marginal_fresh_case_estimate_seconds"],
        "true_streaming_added_case_median_s": fvm_summary["true_streaming_added_case_latency"]["median_seconds"],
        "true_streaming_added_case_p95_s": fvm_summary["true_streaming_added_case_latency"]["p95_seconds"],
        "inter_completion_median_s": fvm_summary["inter_completion_interval"]["median_seconds"],
        "throughput_samples_s": fvm_summary["streaming_throughput"]["median_seconds"],
        "peak_vram_bytes": "N/A", "timing_semantics": "persistent_FVM_240825",
    })

    confirm_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    confirm_artifacts = []
    for result, path, cell in confirm_cells:
        route = route_name(result, path)
        seed = int(cell["seed"])
        confirm_by_seed.setdefault(seed, {})[route] = result
        confirm_artifacts.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    u_failures = []
    for root, state in confirm_states:
        for cell in state["cells"]:
            if cell.get("route") != "U_direct240825" or int(cell.get("returncode", 0)) == 0:
                continue
            log = resolve_artifact(root, cell["log"])
            if "query outside native domain" not in log.read_text(errors="replace"):
                raise RuntimeError(f"unexpected U-direct failure: {log}")
            u_failures.append({"seed": int(cell["seed"]), "reason": "query_outside_native_domain", "log": str(log), "log_sha256": sha256(log), "log_bytes": log.stat().st_size})
    if not u_failures:
        raise RuntimeError("missing preregistered U-direct fail-closed evidence")
    confirmation_rows = []
    paired = {}
    for seed, routes in sorted(confirm_by_seed.items()):
        expected = {"E16384_reconstruction", "E240825_direct"}
        if set(routes) != expected:
            raise RuntimeError(f"seed{seed}: incomplete executable confirmation routes")
        sample_orders = {route: result["sample_ids"] if "sample_ids" in result else [row["sample_id"] for row in result["samples"]] for route, result in routes.items()}
        reference_ids = sample_orders[ROUTES[0]]
        if any(ids != reference_ids for ids in sample_orders.values()):
            raise RuntimeError(f"seed{seed}: paired sample order mismatch")
        components = {route: [row["full_field_metric_components"] for row in routes[route]["samples"]] for route in expected}
        paired[f"seed{seed}"] = {}
        for left, right in (("E16384_reconstruction", "E240825_direct"),):
            paired[f"seed{seed}"][f"{left}_minus_{right}"] = bootstrap_difference(
                components[left], components[right], int(protocol["confirmation"]["paired_bootstrap"]["seed"]),
                int(protocol["confirmation"]["paired_bootstrap"]["replicates"]),
            )
        for route, result in routes.items():
            accuracy = result["accuracy"]["full_field"]
            confirmation_rows.append({"seed": seed, "route": route, "population": "valid96", **{column: accuracy[metric] for metric, column in METRICS.items()}})

    mean_std = {}
    for route in ("E16384_reconstruction", "E240825_direct"):
        route_rows = [row for row in confirmation_rows if row["route"] == route]
        mean_std[route] = {metric: {"mean": float(np.mean([row[column] for row in route_rows])), "std": float(np.std([row[column] for row in route_rows], ddof=1))} for metric, column in METRICS.items()}

    artifacts = [artifact_record(path) for _, path, _ in perf_cells]
    artifacts += confirm_artifacts + [artifact_record(args.fvm_result)]
    for root in [args.performance_root, *args.confirmation_root, args.fvm_result.parent]:
        for path in sorted(root.glob("*.log")) + sorted(root.glob("execution_state.json")):
            record = artifact_record(path)
            if record["path"] not in {row["path"] for row in artifacts}:
                artifacts.append(record)
    result = {
        "schema_version": "heat3d_v6_p1i_post_freeze_closeout_v1", "status": "passed_with_u_direct_fail_closed",
        "protocol_sha256": sha256(args.protocol), "output_domain_nodes": 240825,
        "valid32_subset_formal_valid128": True, "valid96_count": 96,
        "performance": performance, "fvm": fvm_summary,
        "confirmation_valid96_three_seed_mean_std": mean_std, "paired_bootstrap": paired,
        "u_direct_valid96": {"status": "fail_closed_not_evaluated", "reason": "query_outside_native_domain", "accuracy_or_CI_available": False, "failure_evidence": u_failures, "interpretation": "valid32 feasibility did not generalize to the frozen formal-valid population"},
        "artifacts": artifacts, "role_contract": protocol["role_contract"],
        "decision": {
            "valid32_optimization_closed": True,
            "B_E_U_are_parallel_inference_strategies": True,
            "E16384_reconstruction_is_reference_production_strategy": True,
            "U_direct240825_is_direct_full_grid_strategy": True,
            "U_direct240825_formal_population_qualified": False,
            "E240825_direct_is_architecture_control_only": True,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(args.performance_csv, performance_rows); write_csv(args.confirmation_csv, confirmation_rows)
    lines = ["# P1i post-freeze confirmation", "", "所有主比较均输出 240825 nodes；test/sealed 未访问，未训练。", "", "## Unified performance", "", "| strategy | PG % | raw K | fresh median/p95 s | resident median/p95 s | streaming added median/p95 s | throughput samples/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in performance_rows:
        lines.append(f"| {row['strategy']} | {row['PG_pct']} | {row['raw_K']} | {row['fresh_single_case_median_s']:.6f}/{row['fresh_single_case_p95_s']:.6f} | {row['resident_core_median_s']:.6f}/{row['resident_core_p95_s']:.6f} | {row['true_streaming_added_case_median_s']:.6f}/{row['true_streaming_added_case_p95_s']:.6f} | {row['throughput_samples_s']:.6f} |")
    lines += ["", "FVM resident-core 是 prepared-system solve-only，不是 E2E。batch-scale marginal 是预注册估计量；true streaming 行才是 persistent service 的 submit-to-result 实测。", "", "## Independent valid96 confirmation (three-seed mean ± std)", "", "| route | PG % | raw K | source K | peak K | interface K |", "|---|---:|---:|---:|---:|---:|"]
    for route, values in mean_std.items():
        cells = [f"{values[metric]['mean']:.6f} ± {values[metric]['std']:.6f}" for metric in METRICS]
        lines.append(f"| {route} | " + " | ".join(cells) + " |")
    lines += ["", "Paired bootstrap 95% CI 详见机器可读 JSON。valid96 在任何结果查看前由 formal valid128 减 frozen valid32 唯一确定。", "", "U-direct 在 seed0 valid96 首个不满足 native-domain 覆盖的样本上 fail-closed；因此没有 valid96 accuracy/CI，也未继续 seed1/2。该结果是适用边界，不通过修改 route/packing/graph 绕开。", "", "## Freeze", "", "B/E/U 冻结为并列 inference strategy 定义；正式 production/reference 冻结为 E16384-reconstruction。U-direct 仅保留 valid32 架构证明，不具备 formal-valid population 部署资格。不再返回 valid32 做 route/graph/packing/model 优化。FVM 仍是物理 reference，surrogate 不声明比 FVM 更精确。"]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed_with_u_direct_fail_closed", "performance_rows": len(performance_rows), "confirmation_rows": len(confirmation_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
