#!/usr/bin/env python3
"""Build the unified A/B/E/FVM full-grid performance closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "system", "policy", "resolution", "measurement_domain", "sample_count",
    "point_global_pct", "sample_first_pct", "raw_cv_rmse_K", "source_rmse_K",
    "peak_rmse_K", "interface_rmse_K", "regional_nodes", "p2r_edges", "r2r_edges",
    "process_cold_median_s", "process_cold_p95_s", "fresh_topology_median_s",
    "fresh_topology_p95_s", "warm_resident_median_s", "warm_resident_p95_s",
    "neural_forward_median_s", "neural_forward_p95_s", "peak_vram_bytes",
    "process_cold_speedup_vs_fvm", "fresh_topology_speedup_vs_fvm",
    "warm_resident_speedup_vs_fvm", "timing_protocol", "provenance", "status",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def dist(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    index = 0.95 * (len(values) - 1)
    low = int(index)
    high = min(low + 1, len(values) - 1)
    p95 = values[low] + (values[high] - values[low]) * (index - low)
    return statistics.median(values), p95


def timing_runs(root: Path, name: str) -> tuple[float | str, float | str, int]:
    paths = sorted((root / name).glob("run*.json"))
    if not paths:
        return "", "", 0
    values = [json.loads(path.read_text())["timing"]["process_cold_continuous_seconds"] for path in paths]
    median, p95 = dist(values)
    return median, p95, len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-timing-root", type=Path, required=True)
    parser.add_argument("--optimization-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    graph_closeout_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_resolution_closeout.json"
    graph_closeout = json.loads(graph_closeout_path.read_text())
    rows: list[dict] = []
    for source in graph_closeout["rows"]:
        rows.append({
            "system": "GPU_RIGNO", "policy": source["policy"], "resolution": source["resolution"],
            "measurement_domain": "reconstructed_full_240825", "sample_count": source["sample_count"],
            "point_global_pct": source["full_point_global_pct"], "sample_first_pct": "",
            "raw_cv_rmse_K": source["full_raw_cv_rmse_K"], "source_rmse_K": source["source_rmse_K"],
            "peak_rmse_K": source["peak_rmse_K"], "interface_rmse_K": source["interface_rmse_K"],
            "regional_nodes": source["regional_nodes"], "p2r_edges": source["p2r_edges_mean"],
            "r2r_edges": source["r2r_edges_mean"], "process_cold_median_s": "",
            "process_cold_p95_s": "", "fresh_topology_median_s": source["fresh_median_s"],
            "fresh_topology_p95_s": source["fresh_p95_s"], "warm_resident_median_s": source["warm_median_s"],
            "warm_resident_p95_s": source["warm_p95_s"], "neural_forward_median_s": "",
            "neural_forward_p95_s": "", "peak_vram_bytes": source["peak_vram_bytes"],
            "process_cold_speedup_vs_fvm": "", "fresh_topology_speedup_vs_fvm": "",
            "warm_resident_speedup_vs_fvm": "", "timing_protocol": "historical_mixed_protocol_not_pooled",
            "provenance": source["provenance_class"], "status": "historical_reuse",
        })

    unified_path = ROOT / "configs/heat3d_v6_p1i/v6_unified_performance_timing.csv"
    with unified_path.open(newline="") as handle:
        fvm_timing = [row for row in csv.DictReader(handle) if row["family"] == "p1i" and row["route"] == "fvm"]
    fvm_state = {row["state"]: row for row in fvm_timing}
    fvm_cold = float(fvm_state["process_cold"]["continuous_wall_median_s"])
    fvm_cold_p95 = float(fvm_state["process_cold"]["continuous_wall_p95_s"])
    fvm_warm = float(fvm_state["fully_cached"]["continuous_wall_median_s"])
    fvm_warm_p95 = float(fvm_state["fully_cached"]["continuous_wall_p95_s"])

    full_paths = {
        policy: ROOT / f"configs/heat3d_v6_p1i/v6_p1i_full_grid_raw/{policy}_240825.json"
        for policy in ("B", "E")
    }
    feasibility_paths = {
        policy: ROOT / f"configs/heat3d_v6_p1i/v6_p1i_full_grid_raw/{policy}_240825_sample1_feasibility.json"
        for policy in ("B", "E")
    }
    process_counts = {}
    for policy, path in full_paths.items():
        source = json.loads(path.read_text())
        metric = source["accuracy"]["full_field"]
        graph = source["graph_diagnostics"]
        timing = source["timing"]
        cold, cold_p95, cold_count = timing_runs(args.process_timing_root, f"{policy}240825_baseline")
        process_counts[policy] = cold_count
        rows.append({
            "system": "GPU_RIGNO", "policy": policy, "resolution": 240825,
            "measurement_domain": "direct_full_grid_240825", "sample_count": 32,
            "point_global_pct": metric["point_global_true_rms_relative_rmse_pct"],
            "sample_first_pct": metric["sample_first_cv_relative_rmse_pct"],
            "raw_cv_rmse_K": metric["raw_cv_weighted_rmse_K"], "source_rmse_K": metric["source_rmse_K"],
            "peak_rmse_K": metric["peak_rmse_K"], "interface_rmse_K": metric["interface_drop_rmse_K"],
            "regional_nodes": graph["regional_node_count"], "p2r_edges": graph["edge_count"]["p2r"],
            "r2r_edges": graph["edge_count"]["r2r"], "process_cold_median_s": cold,
            "process_cold_p95_s": cold_p95, "fresh_topology_median_s": timing["new_case_e2e"]["median_seconds"],
            "fresh_topology_p95_s": timing["new_case_e2e"]["p95_seconds"],
            "warm_resident_median_s": timing["warm_cache_e2e"]["median_seconds"],
            "warm_resident_p95_s": timing["warm_cache_e2e"]["p95_seconds"],
            "neural_forward_median_s": timing["neural_core"]["median_seconds"],
            "neural_forward_p95_s": timing["neural_core"]["p95_seconds"],
            "peak_vram_bytes": source["device_memory"]["peak_bytes_in_use"],
            "process_cold_speedup_vs_fvm": fvm_cold / cold if cold else "",
            "fresh_topology_speedup_vs_fvm": "not_comparable_no_fvm_unseen_topology_state",
            "warm_resident_speedup_vs_fvm": fvm_warm / timing["warm_cache_e2e"]["median_seconds"],
            "timing_protocol": "new_matched_B1_GPU_sync_valid32_plus_independent_process_cold",
            "provenance": f"new:{path.relative_to(ROOT)}", "status": "new_full_grid_valid32",
        })

    comparison_path = ROOT / "docs/v6_p1i_resolution_performance_comparison.csv"
    with comparison_path.open(newline="") as handle:
        comparison = [row for row in csv.DictReader(handle) if row.get("system") == "CPU_FVM"]
    for source in comparison:
        resolution = int(source["resolution"])
        row = {
            "system": "CPU_FVM", "policy": "FVM", "resolution": resolution,
            "measurement_domain": source["measurement_domain"], "sample_count": 32,
            "point_global_pct": source["point_global_pct"], "sample_first_pct": source["sample_first_pct"],
            "raw_cv_rmse_K": source["raw_cv_rmse_K"], "source_rmse_K": "", "peak_rmse_K": "",
            "interface_rmse_K": "", "regional_nodes": "", "p2r_edges": "", "r2r_edges": "",
            "process_cold_median_s": "", "process_cold_p95_s": "",
            "fresh_topology_median_s": source["new_case_median_s"], "fresh_topology_p95_s": source["new_case_p95_s"],
            "warm_resident_median_s": source["warm_cache_median_s"], "warm_resident_p95_s": source["warm_cache_p95_s"],
            "neural_forward_median_s": "", "neural_forward_p95_s": "", "peak_vram_bytes": 0,
            "process_cold_speedup_vs_fvm": 1.0 if resolution == 240825 else "",
            "fresh_topology_speedup_vs_fvm": 1.0, "warm_resident_speedup_vs_fvm": 1.0,
            "timing_protocol": "historical_legal_structured_FVM_mesh_sensitivity",
            "provenance": source["timing_evidence"], "status": "historical_reuse",
        }
        if resolution == 240825:
            row.update({
                "process_cold_median_s": fvm_cold, "process_cold_p95_s": fvm_cold_p95,
                "fresh_topology_median_s": "not_available", "fresh_topology_p95_s": "not_available",
                "warm_resident_median_s": fvm_warm, "warm_resident_p95_s": fvm_warm_p95,
            })
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    optimization = json.loads(args.optimization_summary.read_text())
    decomposition = {
        policy: json.loads(
            (args.process_timing_root / f"{policy}240825_baseline/decomposition.json").read_text()
        )["timing"]
        for policy in ("B", "E")
    }
    closeout = {
        "schema_version": "heat3d_v6_p1i_full_grid_performance_closeout_v1",
        "status": "completed",
        "rows": rows,
        "process_cold_repeat_counts": process_counts,
        "optimization": optimization,
        "first_inference_latency_decomposition": decomposition,
        "sample1_feasibility": {
            policy: {
                "status": json.loads(path.read_text())["status"],
                "undercovered_fraction": json.loads(path.read_text())["graph_diagnostics"]["undercovered_fraction"],
                "r2r_components": json.loads(path.read_text())["graph_diagnostics"]["r2r_connected_components"]["max"],
                "path": str(path.relative_to(ROOT)), "sha256": sha(path),
            }
            for policy, path in feasibility_paths.items()
        },
        "historical_artifacts_reexecuted": False,
        "timing_protocols_pooled": False,
        "role_contract": {"training": False, "test": False, "sealed": False},
        "sources": {
            "graph_resolution": {"path": str(graph_closeout_path.relative_to(ROOT)), "sha256": sha(graph_closeout_path)},
            "fvm_timing": {"path": str(unified_path.relative_to(ROOT)), "sha256": sha(unified_path)},
            "B_full_grid": {"path": str(full_paths["B"].relative_to(ROOT)), "sha256": sha(full_paths["B"])},
            "E_full_grid": {"path": str(full_paths["E"].relative_to(ROOT)), "sha256": sha(full_paths["E"])},
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    b = next(row for row in rows if row["policy"] == "B" and row["resolution"] == 240825)
    e = next(row for row in rows if row["policy"] == "E" and row["resolution"] == 240825)
    lines = [
        "# V6/P1i full-grid performance closeout", "",
        "本表不混算不同 timing protocol；空值或 `not_comparable` 表示没有语义匹配的历史状态。", "",
        "## 240825-node matched summary", "",
        "| policy | PG % | raw K | source K | peak K | process-cold s | fresh s | warm s | VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (b, e):
        lines.append(
            f"| {row['policy']} | {float(row['point_global_pct']):.4f} | {float(row['raw_cv_rmse_K']):.4f} | "
            f"{float(row['source_rmse_K']):.4f} | {float(row['peak_rmse_K']):.4f} | "
            f"{float(row['process_cold_median_s']):.4f} | {float(row['fresh_topology_median_s']):.4f} | "
            f"{float(row['warm_resident_median_s']):.4f} | {float(row['peak_vram_bytes']) / 2**30:.3f} |"
        )
    lines += [
        f"| FVM | 0.0696 | 0.0366 | N/A | N/A | {fvm_cold:.4f} | N/A | {fvm_warm:.4f} | 0 |", "",
        "## Optimization decision", "",
        optimization["decision"], "",
        "- GPU tiled exact：边可做到完全一致，但图构建更慢，NO-GO。",
        "- P2R/R2P reverse reuse：仅在 fresh 路径有小幅收益；独立 process-cold bootstrap CI 含 0，因此未推广。",
        "- padding/bucketing：未进入，因为前两步已定位主要瓶颈且无进一步预注册收益依据。", "",
        "## Original first-inference bottleneck", "",
        (
            f"- B@240825：连续 process-cold {decomposition['B']['process_cold_continuous_seconds']:.3f} s；"
            f"CUDA init {decomposition['B']['decomposition']['cuda_context_init_seconds']:.3f} s，"
            f"graph {decomposition['B']['decomposition']['graph_construction_seconds']:.3f} s，"
            f"packing/padding {decomposition['B']['decomposition']['packing_padding_seconds']:.3f} s，"
            f"JIT+首 forward+sync {decomposition['B']['decomposition']['jit_plus_first_forward_and_sync_seconds']:.3f} s。"
        ),
        (
            f"- E@240825：连续 process-cold {decomposition['E']['process_cold_continuous_seconds']:.3f} s；"
            f"CUDA init {decomposition['E']['decomposition']['cuda_context_init_seconds']:.3f} s，"
            f"graph {decomposition['E']['decomposition']['graph_construction_seconds']:.3f} s，"
            f"packing/padding {decomposition['E']['decomposition']['packing_padding_seconds']:.3f} s，"
            f"JIT+首 forward+sync {decomposition['E']['decomposition']['jit_plus_first_forward_and_sync_seconds']:.3f} s。"
        ),
        "- direct full-grid output 与 solver grid 同序，reconstruction map/build/apply 均为 0；同步已包含在首 forward 或 warm forward span。", "",
        "## Timing interpretation", "",
        "- process-cold：独立进程连续 wall-clock；与 FVM process-cold 比较。",
        "- fresh-topology：进程已驻留、重新构图；历史 FVM 没有语义相同的 unseen-topology 状态，因此 speedup 标记 N/A。",
        "- warm-resident：固定 support/graph/JIT 重复分析；只与 FVM fully-cached lower bound 比较。",
        "- 历史 A/B/E 行保留各自 provenance，不与本轮 full-grid timing 合并统计。", "",
        "## GO / NO-GO", "",
        (
            f"- B/E@240825 implementation feasibility 均 GO；但 process-cold 相对 full FVM 仅 "
            f"{float(b['process_cold_speedup_vs_fvm']):.3f}x / {float(e['process_cold_speedup_vs_fvm']):.3f}x，"
            "不构成 cold/new-case production speedup。"
        ),
        (
            f"- 固定 support 的 warm repeated-analysis 为 {float(b['warm_resident_speedup_vs_fvm']):.2f}x / "
            f"{float(e['warm_resident_speedup_vs_fvm']):.2f}x；该结论只适用于 fully-cached 语义。"
        ),
        "- E 的 full-grid PG/raw 优于 B，但 source/peak 更差且 VRAM 更高；不据此替换既有 B@8192 推荐分辨率。",
        "- graph optimization 总结为 NO-GO：不推广 shared-reverse、GPU tiled 或 padding/bucketing，不修改 frozen graph policy。", "",
        "完整逐分辨率表见 `docs/v6_p1i_full_grid_performance_closeout.csv`。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "row_count": len(rows), "B_process_n": process_counts["B"], "E_process_n": process_counts["E"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
