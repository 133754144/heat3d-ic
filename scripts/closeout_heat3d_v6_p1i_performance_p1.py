#!/usr/bin/env python3
"""Close P1 using frozen valid32 and independent process-cold evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOC = ROOT / "docs"
FIELDS = [
    "route", "system", "policy", "support_resolution", "output_resolution",
    "measurement_domain", "sample_count", "point_global_pct", "sample_first_pct",
    "raw_cv_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K",
    "regional_nodes", "p2r_edges", "r2r_edges", "model_inference_median_s",
    "model_inference_p95_s", "reconstruction_median_s", "reconstruction_p95_s",
    "process_cold_median_s", "process_cold_p95_s", "fresh_total_median_s",
    "fresh_total_p95_s", "warm_total_median_s", "warm_total_p95_s",
    "peak_vram_bytes", "process_cold_speedup_vs_fvm", "warm_speedup_vs_fvm",
    "fresh_speedup_vs_fvm", "accuracy_provenance", "timing_provenance", "status",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    x = 0.95 * (len(values) - 1)
    lo = int(x); hi = min(lo + 1, len(values) - 1)
    return statistics.median(values), values[lo] + (values[hi] - values[lo]) * (x - lo)


def process_dist(path: Path) -> tuple[float, float, int]:
    files = sorted(path.glob("run*.json"))
    values = [float(load(item)["timing"]["process_cold_continuous_seconds"]) for item in files]
    median, p95 = dist(values)
    return median, p95, len(values)


def gpu_row(route: str, policy: str, resolution: int, path: Path, process_dir: Path) -> dict:
    payload = load(path)
    metric = payload["accuracy"]["full_field"]
    timing = payload["timing"]
    graph = payload["graph_diagnostics"]
    process_median, process_p95, process_count = process_dist(process_dir)
    direct = resolution == 240825
    return {
        "route": route, "system": "GPU_RIGNO", "policy": policy,
        "support_resolution": resolution, "output_resolution": 240825,
        "measurement_domain": "direct_full_grid_240825" if direct else "reconstructed_full_240825",
        "sample_count": metric["sample_count"],
        "point_global_pct": metric["point_global_true_rms_relative_rmse_pct"],
        "sample_first_pct": metric["sample_first_cv_relative_rmse_pct"],
        "raw_cv_rmse_K": metric["raw_cv_weighted_rmse_K"],
        "source_rmse_K": metric["source_rmse_K"], "peak_rmse_K": metric["peak_rmse_K"],
        "interface_rmse_K": metric["interface_drop_rmse_K"],
        "regional_nodes": graph["regional_node_count"], "p2r_edges": graph["edge_count"]["p2r"],
        "r2r_edges": graph["edge_count"]["r2r"],
        "model_inference_median_s": timing["neural_core"]["median_seconds"],
        "model_inference_p95_s": timing["neural_core"]["p95_seconds"],
        "reconstruction_median_s": 0.0 if direct else timing["reconstruction_apply_gpu"]["median_seconds"],
        "reconstruction_p95_s": 0.0 if direct else timing["reconstruction_apply_gpu"]["p95_seconds"],
        "process_cold_median_s": process_median, "process_cold_p95_s": process_p95,
        "fresh_total_median_s": timing["new_case_e2e"]["median_seconds"],
        "fresh_total_p95_s": timing["new_case_e2e"]["p95_seconds"],
        "warm_total_median_s": timing["warm_cache_e2e"]["median_seconds"],
        "warm_total_p95_s": timing["warm_cache_e2e"]["p95_seconds"],
        "peak_vram_bytes": payload["device_memory"]["peak_bytes_in_use"],
        "process_cold_speedup_vs_fvm": "", "warm_speedup_vs_fvm": "",
        "fresh_speedup_vs_fvm": "N/A:no_semantically_matched_FVM_fresh_topology",
        "accuracy_provenance": str(path.relative_to(ROOT)),
        "timing_provenance": f"{path.relative_to(ROOT)};{process_dir.relative_to(ROOT)};process_n={process_count}",
        "status": "historical_reuse" if "performance_p1_raw" not in str(process_dir) else "reused_accuracy_plus_new_process_timing",
    }


def main() -> int:
    protocol = load(CFG / "v6_p1i_performance_p1_protocol.json")
    assert protocol["status"] == "frozen_before_p1_closeout"
    rows = [
        gpu_row("B8192_recon", "B", 8192,
                CFG / "v6_p1i_graph_scale_ablation_raw/B_8192.json",
                CFG / "v6_p1i_performance_p1_raw/B8192_process"),
        gpu_row("E32768_recon", "E", 32768,
                CFG / "v6_p1i_graph_resolution_raw/E_32768.json",
                CFG / "v6_p1i_full_grid_raw/process_timing/E32768_reference"),
        gpu_row("B240825_direct", "B", 240825,
                CFG / "v6_p1i_full_grid_raw/B_240825.json",
                CFG / "v6_p1i_full_grid_raw/process_timing/B240825_baseline"),
        gpu_row("E240825_direct", "E", 240825,
                CFG / "v6_p1i_full_grid_raw/E_240825.json",
                CFG / "v6_p1i_full_grid_raw/process_timing/E240825_baseline"),
    ]
    with (CFG / "v6_unified_performance_accuracy.csv").open(newline="") as handle:
        accuracy = next(row for row in csv.DictReader(handle)
                        if row["family"] == "p1i" and row["route"] == "fvm" and row["resolution"] == "240825")
    with (CFG / "v6_unified_performance_timing.csv").open(newline="") as handle:
        timing_rows = [row for row in csv.DictReader(handle) if row["family"] == "p1i" and row["route"] == "fvm"]
    timing = {row["state"]: row for row in timing_rows}
    cold = timing["process_cold"]; known = timing["known_topology_new_physics"]; warm = timing["fully_cached"]
    fvm = {
        "route": "FVM240825", "system": "CPU_FVM", "policy": "legal_structured_FVM",
        "support_resolution": 240825, "output_resolution": 240825,
        "measurement_domain": "full_240825_reference", "sample_count": 32,
        "point_global_pct": accuracy["point_global_true_rms_relative_rmse_pct"],
        "sample_first_pct": accuracy["sample_first_cv_relative_rmse_pct"],
        "raw_cv_rmse_K": accuracy["raw_cv_weighted_rmse_K"],
        "source_rmse_K": accuracy["source_rmse_K"], "peak_rmse_K": accuracy["peak_rmse_K"],
        "interface_rmse_K": accuracy["interface_drop_rmse_K"], "regional_nodes": "",
        "p2r_edges": "", "r2r_edges": "", "model_inference_median_s": "",
        "model_inference_p95_s": "", "reconstruction_median_s": 0.0,
        "reconstruction_p95_s": 0.0,
        "process_cold_median_s": cold["continuous_wall_median_s"],
        "process_cold_p95_s": cold["continuous_wall_p95_s"],
        "fresh_total_median_s": known["continuous_wall_median_s"],
        "fresh_total_p95_s": known["continuous_wall_p95_s"],
        "warm_total_median_s": warm["continuous_wall_median_s"],
        "warm_total_p95_s": warm["continuous_wall_p95_s"], "peak_vram_bytes": 0,
        "process_cold_speedup_vs_fvm": 1.0, "warm_speedup_vs_fvm": 1.0,
        "fresh_speedup_vs_fvm": 1.0,
        "accuracy_provenance": "configs/heat3d_v6_p1i/v6_unified_performance_accuracy.csv",
        "timing_provenance": "configs/heat3d_v6_p1i/v6_unified_performance_timing.csv",
        "status": "historical_reuse",
    }
    rows.append(fvm)
    fvm_cold = float(fvm["process_cold_median_s"]); fvm_warm = float(fvm["warm_total_median_s"])
    for row in rows[:-1]:
        row["process_cold_speedup_vs_fvm"] = fvm_cold / float(row["process_cold_median_s"])
        row["warm_speedup_vs_fvm"] = fvm_warm / float(row["warm_total_median_s"])

    csv_path = DOC / "v6_p1i_optimal_resolution_full_grid_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    by_route = {row["route"]: row for row in rows}
    pareto = {
        "B8192_recon_vs_B240825_direct": all(
            float(by_route["B8192_recon"][key]) < float(by_route["B240825_direct"][key])
            for key in ("point_global_pct", "raw_cv_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K",
                        "process_cold_median_s", "fresh_total_median_s", "warm_total_median_s", "peak_vram_bytes")
        ),
        "E32768_recon_vs_E240825_direct": all(
            float(by_route["E32768_recon"][key]) < float(by_route["E240825_direct"][key])
            for key in ("point_global_pct", "raw_cv_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K",
                        "process_cold_median_s", "fresh_total_median_s", "warm_total_median_s", "peak_vram_bytes")
        ),
    }
    sources = {}
    for row in rows:
        for item in str(row["accuracy_provenance"]).split(";") + str(row["timing_provenance"]).split(";"):
            path = ROOT / item
            if path.is_file(): sources[item] = sha(path)
    closeout = {
        "schema_version": "heat3d_v6_p1i_performance_p1_closeout_v1", "status": "completed",
        "base_commit": protocol["base_commit"], "rows": rows, "pareto": pareto,
        "new_computation": {"B8192_process_cold_independent_runs": 10},
        "historical_accuracy_recomputed": False, "fvm_recomputed": False,
        "sources": sources, "role_contract": protocol["role_contract"],
    }
    json_path = CFG / "v6_p1i_performance_p1_closeout.json"
    json_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    lines = [
        "# V6/P1i P1 optimal-resolution + full-grid comparison", "",
        "所有 accuracy 均为冻结 `seed0 + valid32`；本阶段没有重算历史 accuracy/FVM。模型推理与重建 apply 独立列出。", "",
        "| route | PG % | raw K | source K | peak K | interface K | model ms | recon ms | cold s | fresh s | warm s | VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str, scale: float = 1.0) -> str:
            value = row[key]
            return "N/A" if value == "" else f"{float(value) * scale:.4f}"
        lines.append(
            f"| {row['route']} | {fmt('point_global_pct')} | {fmt('raw_cv_rmse_K')} | {fmt('source_rmse_K')} | "
            f"{fmt('peak_rmse_K')} | {fmt('interface_rmse_K')} | {fmt('model_inference_median_s',1000)} | "
            f"{fmt('reconstruction_median_s',1000)} | {fmt('process_cold_median_s')} | {fmt('fresh_total_median_s')} | "
            f"{fmt('warm_total_median_s')} | {fmt('peak_vram_bytes',1/2**30)} |"
        )
    lines += ["", "## Pareto decision", "",
              f"- B8192-recon strictly dominates B240825-direct on all registered accuracy/latency/VRAM fields: **{pareto['B8192_recon_vs_B240825_direct']}**.",
              f"- E32768-recon strictly dominates E240825-direct on all registered accuracy/latency/VRAM fields: **{pareto['E32768_recon_vs_E240825_direct']}**.",
              "- FVM is far more accurate. RIGNO process-cold is slower than FVM process-cold; warm replay speedups apply only to fixed-input resident semantics.",
              "- Fresh graph has no semantically matched FVM unseen-topology state, so no fresh speedup is computed.", "",
              "## Provenance", "",
              "- New: only 10 independent B8192 process-cold timing runs.",
              "- Reused: B/E accuracy, fresh/warm timing, E32768/B240825/E240825 process timing, and FVM accuracy/timing."]
    (DOC / "v6_p1i_optimal_resolution_full_grid_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "rows": len(rows), "pareto": pareto}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
