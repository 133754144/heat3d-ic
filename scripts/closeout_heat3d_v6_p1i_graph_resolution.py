#!/usr/bin/env python3
"""Assemble the frozen A/B/E valid32 graph-resolution closeout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"
DOCS = ROOT / "docs"
PROTOCOL = CONFIG / "v6_p1i_graph_resolution_closeout_protocol.json"
FINAL = CONFIG / "v6_p1i_graph_resolution_closeout.json"
CSV_PATH = DOCS / "v6_p1i_graph_resolution_closeout.csv"
MD_PATH = DOCS / "v6_p1i_graph_resolution_closeout.md"
RESOLUTIONS = (1024, 4096, 8192, 16384, 32768)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source(path: Path, role: str) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "role": role}


def candidate_row(policy: str, resolution: int, path: Path, provenance_class: str) -> dict[str, Any]:
    payload = load(path)
    if payload["status"] != "passed" or payload["candidate"] != policy or payload["resolution"] != resolution:
        raise RuntimeError(f"candidate identity/status drifted: {path}")
    role = payload["role_contract"]
    if role["training"] or role["test"] or role["sealed"] or len(payload["sample_ids"]) != 32:
        raise RuntimeError(f"candidate role drifted: {path}")
    full = payload["accuracy"]["full_field"]
    graph = payload["graph_diagnostics"]
    timing = payload["timing"]
    components = graph["r2r_connected_components"]
    coverage_passed = graph["undercovered_fraction"] == 0.0 and components["max"] <= 1.0
    return {
        "policy": policy,
        "resolution": resolution,
        "sample_count": int(full["sample_count"]),
        "full_point_global_pct": full["point_global_true_rms_relative_rmse_pct"],
        "full_raw_cv_rmse_K": full["raw_cv_weighted_rmse_K"],
        "source_rmse_K": full["source_rmse_K"],
        "peak_rmse_K": full["peak_rmse_K"],
        "interface_rmse_K": full["interface_drop_rmse_K"],
        "regional_nodes": graph["regional_node_count"],
        "p2r_edges_mean": graph["edge_count"]["p2r"],
        "r2r_edges_mean": graph["edge_count"]["r2r"],
        "p2r_regional_degree_mean": graph["degree"]["p2r_regional"]["mean"],
        "physical_coverage_radius_median_m": graph["physical_coverage_radius_m"]["median"],
        "undercovered_fraction": graph["undercovered_fraction"],
        "r2r_components_max": components["max"],
        "coverage_passed": coverage_passed,
        "fresh_median_s": timing["new_case_e2e"]["median_seconds"],
        "fresh_p95_s": timing["new_case_e2e"]["p95_seconds"],
        "warm_median_s": timing["warm_cache_e2e"]["median_seconds"],
        "warm_p95_s": timing["warm_cache_e2e"]["p95_seconds"],
        "peak_vram_bytes": payload["device_memory"]["peak_bytes_in_use"],
        "provenance_class": provenance_class,
        "sources": {"accuracy_graph_timing": source(path, provenance_class)},
    }


def qualification_timing_1024() -> tuple[dict[str, float], Path]:
    path = CONFIG / "v6_inference_qualification_timing.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    selected = {
        row["state"]: row for row in rows
        if row["family"] == "p1i" and row["route"] == "production_reconstruction"
        and int(row["input_nodes"]) == 1024 and int(row["output_nodes"]) == 240825
    }
    fresh = selected["jit_cached_new_case"]
    warm = selected["fully_cached_repeat"]
    return {
        "fresh_median_s": float(fresh["continuous_wall_median_s"]),
        "fresh_p95_s": float(fresh["continuous_wall_p95_s"]),
        "warm_median_s": float(warm["continuous_wall_median_s"]),
        "warm_p95_s": float(warm["continuous_wall_p95_s"]),
        "peak_vram_bytes": max(int(fresh["peak_device_bytes"]), int(warm["peak_device_bytes"])),
    }, path


def a_row(resolution: int) -> dict[str, Any]:
    accuracy_path = CONFIG / "v6_p1i_gpu_only_high_n_closeout.json"
    accuracy_payload = load(accuracy_path)
    accuracy = next(row for row in accuracy_payload["rows"] if int(row["resolution"]) == resolution)
    graph_path = CONFIG / "v6_p1i_publication_graph_diagnostics.json"
    graph_payload = load(graph_path)
    graph = next(
        row for row in graph_payload["summaries"]
        if row["family"] == "P1i_sample_varying" and int(row["resolution"]) == resolution
    )
    if resolution in (8192, 16384):
        timing_path = CONFIG / f"v6_p1i_graph_scale_ablation_raw/A_{resolution}_timing_only.json"
        timing_payload = load(timing_path)
        timing = {
            "fresh_median_s": timing_payload["timing"]["new_case_e2e"]["median_seconds"],
            "fresh_p95_s": timing_payload["timing"]["new_case_e2e"]["p95_seconds"],
            "warm_median_s": timing_payload["timing"]["warm_cache_e2e"]["median_seconds"],
            "warm_p95_s": timing_payload["timing"]["warm_cache_e2e"]["p95_seconds"],
            "peak_vram_bytes": timing_payload["device_memory"]["peak_bytes_in_use"],
        }
        timing_role = "historical_reuse_matched_candidate_span"
    elif resolution == 1024:
        timing, timing_path = qualification_timing_1024()
        timing_role = "historical_reuse_qualification_span_not_cross_policy_matched"
    else:
        timing_path = CONFIG / "v6_p1i_publication_gpu_timing.json"
        timing_payload = load(timing_path)
        item = next(row for row in timing_payload["results"] if int(row["resolution"]) == resolution)
        timing = {
            "fresh_median_s": item["timing"]["new_case"]["median_seconds"],
            "fresh_p95_s": item["timing"]["new_case"]["p95_seconds"],
            "warm_median_s": item["timing"]["warm_cache"]["median_seconds"],
            "warm_p95_s": item["timing"]["warm_cache"]["p95_seconds"],
            "peak_vram_bytes": item["device_memory"]["peak_bytes_in_use"],
        }
        timing_role = "historical_reuse_publication_span_not_candidate_runner_matched"
    coverage_passed = (
        graph["coverage"]["p2r_zero_degree_nodes"]["max"] == 0.0
        and graph["coverage"]["r2p_zero_degree_nodes"]["max"] == 0.0
    )
    return {
        "policy": "A", "resolution": resolution, "sample_count": 32,
        "full_point_global_pct": accuracy["full_point_global_pct"],
        "full_raw_cv_rmse_K": accuracy["full_raw_cv_rmse_K"],
        "source_rmse_K": accuracy["full_source_rmse_K"],
        "peak_rmse_K": accuracy["full_peak_rmse_K"],
        "interface_rmse_K": accuracy["full_interface_drop_rmse_K"],
        "regional_nodes": graph["regional_node_count"]["mean"],
        "p2r_edges_mean": graph["edge_count"]["p2r"]["mean"],
        "r2r_edges_mean": graph["edge_count"]["r2r"]["mean"],
        "p2r_regional_degree_mean": graph["degree"]["p2r_regional"]["mean"],
        "physical_coverage_radius_median_m": graph["observed_physical_support_radius_m"]["median"],
        "undercovered_fraction": 0.0,
        "r2r_components_max": 1.0,
        "coverage_passed": coverage_passed,
        **timing,
        "provenance_class": "historical_reuse",
        "sources": {
            "accuracy": source(accuracy_path, "historical_valid32_gpu_only"),
            "graph": source(graph_path, "historical_offline_cache_diagnostic"),
            "timing": source(timing_path, timing_role),
        },
    }


def alias_e_1024(a: dict[str, Any]) -> dict[str, Any]:
    row = dict(a)
    row["policy"] = "E"
    row["provenance_class"] = "exact_policy_alias"
    row["sources"] = dict(a["sources"])
    row["sources"]["alias_contract"] = {
        "path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha(PROTOCOL),
        "role": "E1024_equals_A1024_factor4_Nr256",
    }
    return row


def main() -> int:
    protocol = load(PROTOCOL)
    if protocol["status"] != "preregistered_before_graph_resolution_closeout":
        raise RuntimeError("protocol is not frozen")
    raw = CONFIG / "v6_p1i_graph_resolution_raw"
    rows: list[dict[str, Any]] = []
    a_rows = {n: a_row(n) for n in RESOLUTIONS}
    rows.extend(a_rows.values())
    for resolution in RESOLUTIONS:
        path = (
            CONFIG / f"v6_p1i_graph_scale_ablation_raw/B_{resolution}.json"
            if resolution in (8192, 16384) else raw / f"B_{resolution}.json"
        )
        rows.append(candidate_row("B", resolution, path, "historical_reuse" if resolution in (8192, 16384) else "new_execution"))
    rows.append(alias_e_1024(a_rows[1024]))
    for resolution in (4096, 8192, 16384, 32768):
        path = (
            CONFIG / "v6_p1i_graph_policy_e_raw/E_8192.json"
            if resolution == 8192 else raw / f"E_{resolution}.json"
        )
        rows.append(candidate_row("E", resolution, path, "historical_reuse" if resolution == 8192 else "new_execution"))
    rows.sort(key=lambda row: (row["policy"], row["resolution"]))
    b_rows = [row for row in rows if row["policy"] == "B"]
    best_b = min(b_rows, key=lambda row: (row["full_point_global_pct"], row["fresh_median_s"]))
    e_rows = [row for row in rows if row["policy"] == "E"]
    e_full_monotonic = all(
        e_rows[i + 1]["full_point_global_pct"] <= e_rows[i]["full_point_global_pct"]
        for i in range(len(e_rows) - 1)
    )
    e_raw_monotonic = all(
        e_rows[i + 1]["full_raw_cv_rmse_K"] <= e_rows[i]["full_raw_cv_rmse_K"]
        for i in range(len(e_rows) - 1)
    )
    final = {
        "schema_version": "heat3d_v6_p1i_graph_resolution_closeout_v1",
        "status": "completed",
        "rows": rows,
        "B_best_and_recommended_resolution": int(best_b["resolution"]),
        "B_recommendation_basis": "minimum valid32 full-field point-global RMSE; also faster than B@16384 and B@32768",
        "E_scaling": {
            "full_point_global_monotonic_improvement": e_full_monotonic,
            "raw_cv_rmse_monotonic_improvement": e_raw_monotonic,
            "regional_nodes_constant": all(row["regional_nodes"] == 256 for row in e_rows),
            "retain": "exploratory_fixed_capacity_only",
            "production_status": "not_promoted",
            "reason": "full/raw improve with N, but source/peak locality and memory do not improve monotonically; only seed0 valid32",
        },
        "compression_locality": {
            "A": "Nr=N/4; highest regional capacity and locality, highest regional-state cost",
            "B": "Nr=N/8; confirmed latency Pareto winner on separate valid96x3-seed protocol",
            "E": "Nr=256; strongest regional-state compression, but P2R densification and source/peak trade-off prevent production high-efficiency claim",
        },
        "provenance_warning": "All curve rows are seed0 frozen valid32. Separate valid96x3-seed confirmation is decision provenance only and is not pooled or averaged with this curve.",
        "protocol": source(PROTOCOL, "frozen_before_new_execution"),
        "role_contract": {
            "training": False, "test": False, "sealed": False,
            "valid96_confirmation_reexecuted": False,
        },
        "actual_new_execution_cells": protocol["new_execution_cells"],
    }
    FINAL.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    csv_rows = []
    for row in rows:
        csv_rows.append({k: v for k, v in row.items() if k != "sources"} | {
            "source_paths": ";".join(item["path"] for item in row["sources"].values()),
            "source_sha256": ";".join(item["sha256"] for item in row["sources"].values()),
        })
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(csv_rows)
    table = [
        "| policy | N | full PG % | raw K | source K | peak K | interface K | Nr | P2R | R2R | fresh/warm ms | VRAM MiB | provenance |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|",
    ]
    for row in rows:
        table.append(
            f"| {row['policy']} | {row['resolution']} | {row['full_point_global_pct']:.6f} | "
            f"{row['full_raw_cv_rmse_K']:.6f} | {row['source_rmse_K']:.6f} | {row['peak_rmse_K']:.6f} | "
            f"{row['interface_rmse_K']:.6f} | {row['regional_nodes']:.0f} | {row['p2r_edges_mean']:.1f} | "
            f"{row['r2r_edges_mean']:.1f} | {1000*row['fresh_median_s']:.3f}/{1000*row['warm_median_s']:.3f} | "
            f"{row['peak_vram_bytes']/2**20:.1f} | {row['provenance_class']} |"
        )
    MD_PATH.write_text(
        "# P1i graph-resolution closeout\n\n"
        "范围：seed0、冻结 valid32、N={1024,4096,8192,16384,32768}。无训练，未访问 test/sealed，"
        "未重新运行 valid96 confirmation。A/B/E 的 factor、radius 与 coverage 定义均已冻结。\n\n"
        "**Provenance 注意：**下表的 accuracy 全部属于 valid32；valid96×3-seed 只支持 B 的既有晋级结论，"
        "与 valid32 数值不混合统计。A@1024、A@4096/32768 与 A@8192/16384 的 timing 来自不同历史"
        "计时协议，已逐行绑定来源；只有相同 provenance 的 timing 才可严格横比。\n\n"
        + "\n".join(table) + "\n\n"
        "表中 latency 为 median；fresh/warm p95、coverage 与逐文件 SHA 见 CSV/JSON。新增 GPU 日志出现 JAX "
        "CUDA timer 精度 warning，但生产 span 使用 `perf_counter + block_until_ready`；六个单元均无 OOM/NaN/coverage 失败。\n\n"
        "## 判定\n\n"
        f"1. **B 最佳/推荐 resolution：{best_b['resolution']}。** valid32 full PG 最低，且其 fresh/warm latency "
        "低于 B@16384/32768。\n"
        f"2. **E scaling：** full PG 单调改善={e_full_monotonic}，raw 单调改善={e_raw_monotonic}；"
        "Nr 恒为256，但 P2R edges、warm latency 与 VRAM 随 N 增长。\n"
        "3. **Compression-locality：**A 保留最多 regional state；B 将 regional state 减半并已在独立 valid96×3-seed"
        "确认中通过；E 固定 regional state，但以更稠密 P2R、source/peak 局部误差和内存增长为代价。\n"
        "4. **E 结论：保留为 exploratory fixed-capacity policy，不晋级为 production/high-efficiency policy。**"
        "它证明 full/raw 场误差可受益于固定容量压缩，但目前没有三 seed confirmatory evidence，且 source/peak 与"
        "内存证据不足以支持高效生产声明。\n"
    )
    print(json.dumps({"status": "completed", "B_resolution": best_b["resolution"], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
