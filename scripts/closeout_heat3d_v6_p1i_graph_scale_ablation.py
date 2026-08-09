#!/usr/bin/env python3
"""Create the frozen P1i graph-scale and CPU-FVM/GPU-RIGNO closeout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_ablation_raw"
BASELINE = ROOT / "configs/heat3d_v6_p1i/v6_p1i_gpu_only_high_n_closeout.json"
PUB_TIMING = ROOT / "configs/heat3d_v6_p1i/v6_p1i_publication_gpu_timing.csv"
PUB_GRAPH = ROOT / "configs/heat3d_v6_p1i/v6_p1i_publication_graph_diagnostics.csv"
UNIFIED_RESOLUTION = ROOT / "configs/heat3d_v6_p1i/v6_unified_performance_resolution.csv"
UNIFIED_TIMING = ROOT / "configs/heat3d_v6_p1i/v6_unified_performance_timing.csv"
POLICY = ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_policy_closeout.json"
APPLY_FIRST = ROOT / "configs/heat3d_v6_p1i/v6_p1i_reconstruction_apply_only_timing_first.json"
APPLY_REPLAY = ROOT / "configs/heat3d_v6_p1i/v6_p1i_reconstruction_apply_only_timing_replay8192.json"
GRAPH_CSV = ROOT / "docs/v6_p1i_graph_scale_ablation.csv"
GRAPH_MD = ROOT / "docs/v6_p1i_graph_scale_ablation.md"
PERF_CSV = ROOT / "docs/v6_p1i_resolution_performance_comparison.csv"
PERF_MD = ROOT / "docs/v6_p1i_resolution_performance_comparison.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def distribution_value(payload: dict, key: str) -> float:
    return float(payload[key])


def candidate_row(candidate: str, resolution: int, payload: dict) -> dict[str, object]:
    full = payload["accuracy"]["full_field"]
    support = payload["accuracy"]["support"]
    graph = payload["graph_diagnostics"]
    timing = payload["timing"]
    memory = payload["device_memory"]
    return {
        "candidate": candidate,
        "resolution": resolution,
        "evidence": "new_valid32_seed0",
        "factor": payload["policy"]["subsample_factor"],
        "coverage_mode": payload["policy"]["coverage_mode"],
        "support_pg_pct": support["point_global_true_rms_relative_rmse_pct"],
        "support_sample_first_pct": support["sample_first_cv_relative_rmse_pct"],
        "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
        "full_pg_pct": full["point_global_true_rms_relative_rmse_pct"],
        "full_sample_first_pct": full["sample_first_cv_relative_rmse_pct"],
        "full_raw_cv_rmse_K": full["raw_cv_weighted_rmse_K"],
        "source_rmse_K": full["source_rmse_K"],
        "background_rmse_K": full["background_rmse_K"],
        "interface_rmse_K": full["interface_drop_rmse_K"],
        "peak_rmse_K": full["peak_rmse_K"],
        "oracle_floor_pg_pct": payload["accuracy"]["oracle_reconstruction_floor_reused"]["point_global_true_rms_relative_rmse_pct"],
        "oracle_floor_raw_K": payload["accuracy"]["oracle_reconstruction_floor_reused"]["raw_cv_weighted_rmse_K"],
        "common_anchor_response_rmse_K": payload["common_anchor_response_drift"]["rmse_K"],
        "common_anchor_response_max_K": payload["common_anchor_response_drift"]["max_abs_K"],
        "delta_k_max": payload["common_anchor_input_drift"]["delta_k"]["max_abs"],
        "delta_q_max": payload["common_anchor_input_drift"]["delta_q"]["max_abs"],
        "delta_cv_max": payload["common_anchor_input_drift"]["delta_cv"]["max_abs"],
        "regional_nodes": graph["regional_node_count"],
        "p2r_edges": graph["edge_count"]["p2r"],
        "r2r_edges": graph["edge_count"]["r2r"],
        "r2p_edges": graph["edge_count"]["r2p"],
        "physical_degree_p5": graph["degree"]["p2r_physical"]["p5"],
        "physical_degree_median": graph["degree"]["p2r_physical"]["median"],
        "physical_degree_mean": graph["degree"]["p2r_physical"]["mean"],
        "physical_degree_p95": graph["degree"]["p2r_physical"]["p95"],
        "regional_degree_p5": graph["degree"]["p2r_regional"]["p5"],
        "regional_degree_median": graph["degree"]["p2r_regional"]["median"],
        "regional_degree_mean": graph["degree"]["p2r_regional"]["mean"],
        "regional_degree_p95": graph["degree"]["p2r_regional"]["p95"],
        "r2r_degree_mean": graph["degree"]["r2r_regional"]["mean"],
        "r2p_regional_degree_mean": graph["degree"]["r2p_regional"]["mean"],
        "r2p_physical_degree_mean": graph["degree"]["r2p_physical"]["mean"],
        "source_degree_mean": graph["partition"]["source"]["p2r_degree"]["mean"],
        "interface_degree_mean": graph["partition"]["interface"]["p2r_degree"]["mean"],
        "background_degree_mean": graph["partition"]["background"]["p2r_degree"]["mean"],
        "target_radius_p5": graph["target_normalized_radius"]["p5"],
        "target_radius_median": graph["target_normalized_radius"]["median"],
        "target_radius_p95": graph["target_normalized_radius"]["p95"],
        "target_radius_max": graph["target_normalized_radius"]["max"],
        "realized_radius_p5": graph["realized_normalized_radius"]["p5"],
        "realized_radius_median": graph["realized_normalized_radius"]["median"],
        "realized_radius_p95": graph["realized_normalized_radius"]["p95"],
        "physical_coverage_radius_median_m": graph["physical_coverage_radius_m"]["median"],
        "normalized_p2r_edge_length_median": graph["normalized_edge_length"]["p2r"]["median"],
        "physical_p2r_edge_length_median_m": graph["physical_edge_length_m"]["p2r"]["median"],
        "undercovered_fraction": graph["undercovered_fraction"],
        "r2r_components_max": graph["r2r_connected_components"]["max"],
        "receptive_field_proxy_m": graph["effective_physical_receptive_field_proxy_m"],
        "graph_median_s": timing["graph_construction"]["median_seconds"],
        "graph_p95_s": timing["graph_construction"]["p95_seconds"],
        "warm_median_s": timing["warm_cache_e2e"]["median_seconds"],
        "warm_p95_s": timing["warm_cache_e2e"]["p95_seconds"],
        "new_case_median_s": timing["new_case_e2e"]["median_seconds"],
        "new_case_p95_s": timing["new_case_e2e"]["p95_seconds"],
        "peak_vram_bytes": memory["peak_bytes_in_use"],
        "artifact_sha256": sha256(RAW / f"{candidate}_{resolution}.json"),
    }


def baseline_rows() -> list[dict[str, object]]:
    closeout = json.loads(BASELINE.read_text())
    metric = {int(row["resolution"]): row for row in closeout["rows"]}
    timing = {int(row["resolution"]): row for row in load_csv(PUB_TIMING)}
    graph = {
        int(row["resolution"]): row for row in load_csv(PUB_GRAPH)
        if row["family"] == "P1i_sample_varying"
    }
    direct = {
        int(row["resolution"]): row for row in load_csv(UNIFIED_RESOLUTION)
        if row["family"] == "p1i" and row["route"] == "production_reconstruction"
        and row["state"] == "new_topology" and row["status"] == "passed"
    }
    rows = []
    for resolution in (8192, 16384):
        m, t, g = metric[resolution], timing[resolution], graph[resolution]
        rows.append({
            "candidate": "A", "resolution": resolution,
            "evidence": "historical_read_only_accuracy_graph_plus_direct_wall", "factor": 4,
            "coverage_mode": "discrete_physical_coverage",
            "support_pg_pct": m["support_point_global_pct"],
            "support_sample_first_pct": m["support_sample_first_pct"],
            "support_raw_cv_rmse_K": m["support_raw_cv_rmse_K"],
            "full_pg_pct": m["full_point_global_pct"],
            "full_sample_first_pct": m["full_sample_first_pct"],
            "full_raw_cv_rmse_K": m["full_raw_cv_rmse_K"],
            "source_rmse_K": m["full_source_rmse_K"],
            "background_rmse_K": m["full_background_rmse_K"],
            "interface_rmse_K": m["full_interface_drop_rmse_K"],
            "peak_rmse_K": m["full_peak_rmse_K"],
            "oracle_floor_pg_pct": m["oracle_full_point_global_pct"],
            "oracle_floor_raw_K": m["oracle_full_raw_cv_rmse_K"],
            "common_anchor_response_rmse_K": "", "common_anchor_response_max_K": "",
            "delta_k_max": "", "delta_q_max": "", "delta_cv_max": "",
            "regional_nodes": float(g["regional_nodes_mean"]),
            "p2r_edges": float(g["p2r_edges_mean"]), "r2r_edges": float(g["r2r_edges_mean"]),
            "r2p_edges": float(g["r2p_edges_mean"]),
            "physical_degree_p5": "", "physical_degree_median": "", "physical_degree_mean": "", "physical_degree_p95": "",
            "regional_degree_p5": "", "regional_degree_median": "",
            "regional_degree_mean": float(g["p2r_regional_degree_mean"]), "regional_degree_p95": "",
            "r2r_degree_mean": float(g["r2r_out_degree_mean"]),
            "r2p_regional_degree_mean": float(g["r2p_regional_degree_mean"]),
            "r2p_physical_degree_mean": "",
            "source_degree_mean": float(g["source_p2r_degree_mean"]),
            "interface_degree_mean": float(g["interface_p2r_degree_mean"]),
            "background_degree_mean": float(g["background_p2r_degree_mean"]),
            "target_radius_p5": "", "target_radius_median": float(g["normalized_radius_median"]),
            "target_radius_p95": "", "target_radius_max": "", "realized_radius_p5": "",
            "realized_radius_median": "", "realized_radius_p95": "",
            "physical_coverage_radius_median_m": float(g["observed_radius_median_m"]),
            "normalized_p2r_edge_length_median": "",
            "physical_p2r_edge_length_median_m": float(g["p2r_edge_length_median_m"]),
            "undercovered_fraction": 0.0, "r2r_components_max": 1.0,
            "receptive_field_proxy_m": "",
            "graph_median_s": float(direct[resolution]["graph_s_median"]),
            "graph_p95_s": float(direct[resolution]["graph_s_p95"]),
            "warm_median_s": float(t["warm_cache_median_seconds"]),
            "warm_p95_s": float(t["warm_cache_p95_seconds"]),
            "new_case_median_s": float(direct[resolution]["continuous_wall_median_s"]),
            "new_case_p95_s": float(direct[resolution]["continuous_wall_p95_s"]),
            "peak_vram_bytes": int(t["peak_vram_bytes"]),
            "artifact_sha256": m["result_sha256"],
        })
    return rows


def p1h_graph_context_rows() -> list[dict[str, object]]:
    graph = [row for row in load_csv(PUB_GRAPH) if row["family"] == "P1h_shared_support"]
    result = []
    for g in graph:
        resolution = int(g["resolution"])
        if resolution not in (8192, 16384):
            continue
        blank = {key: "" for key in baseline_rows()[0]}
        blank.update({
            "candidate": "P1h_context", "resolution": resolution,
            "evidence": "historical_graph_only_read_only", "factor": 8,
            "coverage_mode": "discrete_physical_coverage",
            "regional_nodes": float(g["regional_nodes_mean"]),
            "p2r_edges": float(g["p2r_edges_mean"]), "r2r_edges": float(g["r2r_edges_mean"]),
            "r2p_edges": float(g["r2p_edges_mean"]),
            "regional_degree_mean": float(g["p2r_regional_degree_mean"]),
            "r2r_degree_mean": float(g["r2r_out_degree_mean"]),
            "r2p_regional_degree_mean": float(g["r2p_regional_degree_mean"]),
            "source_degree_mean": float(g["source_p2r_degree_mean"]),
            "interface_degree_mean": float(g["interface_p2r_degree_mean"]),
            "background_degree_mean": float(g["background_p2r_degree_mean"]),
            "target_radius_median": float(g["normalized_radius_median"]),
            "physical_coverage_radius_median_m": float(g["observed_radius_median_m"]),
            "physical_p2r_edge_length_median_m": float(g["p2r_edge_length_median_m"]),
            "undercovered_fraction": 0.0, "r2r_components_max": 1.0,
        })
        result.append(blank)
    return result


def clear_improvement(candidate: str, rows: list[dict[str, object]]) -> dict[str, object]:
    base = {int(row["resolution"]): row for row in rows if row["candidate"] == "A"}
    cand = {int(row["resolution"]): row for row in rows if row["candidate"] == candidate}
    pg_each = all(float(cand[n]["full_pg_pct"]) <= float(base[n]["full_pg_pct"]) for n in (8192, 16384))
    raw_each = all(float(cand[n]["full_raw_cv_rmse_K"]) <= float(base[n]["full_raw_cv_rmse_K"]) for n in (8192, 16384))
    mean_pg_gain = sum(float(base[n]["full_pg_pct"]) - float(cand[n]["full_pg_pct"]) for n in (8192, 16384)) / 2
    mean_raw_gain = sum(float(base[n]["full_raw_cv_rmse_K"]) - float(cand[n]["full_raw_cv_rmse_K"]) for n in (8192, 16384)) / 2
    source_peak_ok = all(
        float(cand[n][field]) <= float(base[n][field]) * 1.01
        for n in (8192, 16384) for field in ("source_rmse_K", "peak_rmse_K")
    )
    passed = pg_each and raw_each and mean_pg_gain >= 0.01 and mean_raw_gain >= 0.01 and source_peak_ok
    return {
        "candidate": candidate, "passed": passed, "pg_non_regression_each": pg_each,
        "raw_non_regression_each": raw_each, "mean_pg_gain_percentage_points": mean_pg_gain,
        "mean_raw_gain_K": mean_raw_gain, "source_peak_non_regression": source_peak_ok,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def performance_rows() -> list[dict[str, object]]:
    closeout = json.loads(BASELINE.read_text())
    metric = {int(row["resolution"]): row for row in closeout["rows"]}
    pub = {int(row["resolution"]): row for row in load_csv(PUB_TIMING)}
    unified = load_csv(UNIFIED_RESOLUTION) + load_csv(UNIFIED_TIMING)
    old_accuracy = {
        4096: {"point_global_pct": 1.190491, "sample_first_pct": 1.962932, "raw_cv_rmse_K": 1.518489},
        8192: {"point_global_pct": 1.191106, "sample_first_pct": 1.963303, "raw_cv_rmse_K": 1.518798},
        16384: {"point_global_pct": 1.191235, "sample_first_pct": 1.963571, "raw_cv_rmse_K": 1.519003},
        32768: {"point_global_pct": 1.191553, "sample_first_pct": 1.963771, "raw_cv_rmse_K": 1.519157},
        240825: {"point_global_pct": 0.069648, "sample_first_pct": 0.045814, "raw_cv_rmse_K": 0.036553},
    }
    apply_rows = {int(row["resolution"]): row for row in json.loads(APPLY_FIRST.read_text())["results"]}
    replay = json.loads(APPLY_REPLAY.read_text())
    apply_rows[8192] = replay["results"][0]

    def timed(n: int, route: str, state: str, family: str = "p1i") -> dict[str, str]:
        hits = [row for row in unified if row["family"] == family and row["resolution"] == str(n)
                and row["route"] == route and row["state"] == state and row["status"] == "passed"]
        if not hits: raise RuntimeError(f"missing timing {n}/{route}/{state}")
        return hits[0]

    rows: list[dict[str, object]] = []
    native = metric[1024]
    new1024 = timed(1024, "production_reconstruction", "jit_cached_new_topology")
    warm1024 = timed(1024, "production_reconstruction", "fully_cached")
    for domain, pg, sf, raw in (
        ("native_support_1024", native["support_point_global_pct"], native["support_sample_first_pct"], native["support_raw_cv_rmse_K"]),
        ("reconstructed_full_240825_from_1024", native["full_point_global_pct"], native["full_sample_first_pct"], native["full_raw_cv_rmse_K"]),
    ):
        rows.append({
            "resolution": 1024, "system": "GPU_RIGNO", "graph_policy": "A_frozen",
            "measurement_domain": domain, "point_global_pct": pg, "sample_first_pct": sf,
            "raw_cv_rmse_K": raw, "accuracy_evidence": "historical_valid32",
            "neural_core_median_s": "", "neural_core_p95_s": "",
            "graph_construction_median_s": new1024["graph_s_median"], "graph_construction_p95_s": new1024["graph_s_p95"],
            "reconstruction_map_build_median_s": new1024["map_build_s_median"], "reconstruction_map_build_p95_s": new1024["map_build_s_p95"],
            "reconstruction_apply_median_s": warm1024["map_apply_s_median"], "reconstruction_apply_p95_s": warm1024["map_apply_s_p95"],
            "warm_cache_median_s": warm1024["continuous_wall_median_s"], "warm_cache_p95_s": warm1024["continuous_wall_p95_s"],
            "new_case_median_s": new1024["continuous_wall_median_s"], "new_case_p95_s": new1024["continuous_wall_p95_s"],
            "peak_vram_bytes": new1024["peak_device_bytes"], "new_case_speedup_vs_fvm": "",
            "warm_speedup_vs_fvm": "", "neural_core_over_fvm": "",
            "timing_evidence": "historical_unified_direct_wall",
        })
    for n in (4096, 8192, 16384, 32768):
        m, p = metric[n], pub[n]
        new = timed(n, "production_reconstruction", "new_topology")
        warm = p
        fvm_new = timed(n, "fvm", "known_topology_new_physics")
        fvm_warm = timed(n, "fvm", "fully_cached")
        rows.append({
            "resolution": n, "system": "GPU_RIGNO", "graph_policy": "A_frozen_no_go_retained",
            "measurement_domain": f"reconstructed_full_240825_from_{n}",
            "point_global_pct": m["full_point_global_pct"], "sample_first_pct": m["full_sample_first_pct"],
            "raw_cv_rmse_K": m["full_raw_cv_rmse_K"], "accuracy_evidence": "historical_valid32",
            "neural_core_median_s": p["neural_forward_median_seconds"], "neural_core_p95_s": p["neural_forward_p95_seconds"],
            "graph_construction_median_s": new["graph_s_median"], "graph_construction_p95_s": new["graph_s_p95"],
            "reconstruction_map_build_median_s": new["map_build_s_median"], "reconstruction_map_build_p95_s": new["map_build_s_p95"],
            "reconstruction_apply_median_s": apply_rows[n]["median_seconds"],
            "reconstruction_apply_p95_s": apply_rows[n]["p95_seconds"],
            "warm_cache_median_s": warm["warm_cache_median_seconds"], "warm_cache_p95_s": warm["warm_cache_p95_seconds"],
            "new_case_median_s": new["continuous_wall_median_s"], "new_case_p95_s": new["continuous_wall_p95_s"],
            "peak_vram_bytes": p["peak_vram_bytes"],
            "new_case_speedup_vs_fvm": float(fvm_new["continuous_wall_median_s"]) / float(new["continuous_wall_median_s"]),
            "warm_speedup_vs_fvm": float(fvm_warm["continuous_wall_median_s"]) / float(warm["warm_cache_median_seconds"]),
            "neural_core_over_fvm": float(p["neural_forward_median_seconds"]) / float(fvm_new["continuous_wall_median_s"]),
            "timing_evidence": "historical_direct_new_topology_plus_publication_cached",
        })
        a = old_accuracy[n]
        rows.append({
            "resolution": n, "system": "CPU_FVM", "graph_policy": "legal_structured_mesh",
            "measurement_domain": f"structured_{n}_vs_240825_reference",
            "point_global_pct": a["point_global_pct"], "sample_first_pct": a["sample_first_pct"],
            "raw_cv_rmse_K": a["raw_cv_rmse_K"], "accuracy_evidence": "historical_read_only",
            "neural_core_median_s": "", "neural_core_p95_s": "",
            "graph_construction_median_s": "", "graph_construction_p95_s": "",
            "reconstruction_map_build_median_s": "", "reconstruction_map_build_p95_s": "",
            "reconstruction_apply_median_s": "", "reconstruction_apply_p95_s": "",
            "warm_cache_median_s": fvm_warm["continuous_wall_median_s"], "warm_cache_p95_s": fvm_warm["continuous_wall_p95_s"],
            "new_case_median_s": fvm_new["continuous_wall_median_s"], "new_case_p95_s": fvm_new["continuous_wall_p95_s"],
            "peak_vram_bytes": 0, "new_case_speedup_vs_fvm": 1.0, "warm_speedup_vs_fvm": 1.0,
            "neural_core_over_fvm": "", "timing_evidence": "historical_unified_CPU",
        })
    fnew = timed(240825, "fvm", "known_topology_new_physics")
    fwarm = timed(240825, "fvm", "fully_cached")
    a = old_accuracy[240825]
    rows.append({
        "resolution": 240825, "system": "CPU_FVM", "graph_policy": "reference_structured_mesh",
        "measurement_domain": "full_240825_reference", "point_global_pct": a["point_global_pct"],
        "sample_first_pct": a["sample_first_pct"], "raw_cv_rmse_K": a["raw_cv_rmse_K"],
        "accuracy_evidence": "historical_read_only", "neural_core_median_s": "", "neural_core_p95_s": "",
        "graph_construction_median_s": "", "graph_construction_p95_s": "",
        "reconstruction_map_build_median_s": "", "reconstruction_map_build_p95_s": "",
        "reconstruction_apply_median_s": "", "reconstruction_apply_p95_s": "",
        "warm_cache_median_s": fwarm["continuous_wall_median_s"], "warm_cache_p95_s": fwarm["continuous_wall_p95_s"],
        "new_case_median_s": fnew["continuous_wall_median_s"], "new_case_p95_s": fnew["continuous_wall_p95_s"],
        "peak_vram_bytes": 0, "new_case_speedup_vs_fvm": 1.0, "warm_speedup_vs_fvm": 1.0,
        "neural_core_over_fvm": "", "timing_evidence": "historical_unified_CPU",
    })
    return rows


def main() -> int:
    candidates = [candidate_row(c, n, json.loads((RAW / f"{c}_{n}.json").read_text()))
                  for c in ("B", "C") for n in (8192, 16384)]
    rows = baseline_rows() + candidates
    decisions = {c: clear_improvement(c, rows) for c in ("B", "C")}
    if decisions["B"]["passed"] and decisions["C"]["passed"]:
        raise RuntimeError("D is required by preregistration but was not supplied")
    outcome = "no_go_keep_A"
    policy = {
        "schema_version": "heat3d_v6_p1i_graph_scale_policy_closeout_v1",
        "status": "completed_no_go", "decision": outcome,
        "native1024_physical_coverage_hypothesis_supported": False,
        "candidate_production_graph_policy": None,
        "retained_production_graph_policy": "A_factor4_discrete_physical_coverage",
        "D_executed": False, "D_reason": "B_and_C_did_not_both_pass_clear_improvement",
        "winner_extension_executed": False,
        "selection": decisions,
        "causal_conclusion": (
            "Preserving native1024 local coverage radius did not improve both 8192 and 16384; "
            "the observed high-N degradation is not explained by radius/receptive-field drift alone."
        ),
        "next_priority": "GPU graph optimization for sample-varying new topology; reuse fixed graphs when support repeats",
        "frozen_roles": {"training": False, "test": False, "sealed": False, "valid32_seed0_only": True},
        "candidate_execution_code_commit": "bb259d3",
        "raw_artifacts": {f"{c}_{n}": {"path": str(RAW.relative_to(ROOT) / f"{c}_{n}.json"), "sha256": sha256(RAW / f"{c}_{n}.json")}
                          for c in ("B", "C") for n in (8192, 16384)},
        "timing_only_artifacts": {
            "first": {"path": str(APPLY_FIRST.relative_to(ROOT)), "sha256": sha256(APPLY_FIRST)},
            "replay8192": {"path": str(APPLY_REPLAY.relative_to(ROOT)), "sha256": sha256(APPLY_REPLAY),
                            "reason": "first-pass CUDA timer anomaly; frozen 100-repeat replay"},
        },
    }
    POLICY.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    graph_csv_rows = rows + p1h_graph_context_rows()
    write_csv(GRAPH_CSV, graph_csv_rows)
    GRAPH_MD.write_text(
        "# V6/P1i high-N graph-scale 因果消融\n\n"
        "范围：冻结 seed0 checkpoint 与 frozen valid32；无训练、无 test/sealed 访问。"
        "A 与 FVM 证据只读复用，本轮仅新增 B/C 的 8192/16384。\n\n"
        "| 策略 | N | full PG % | sample-first % | raw K | source K | peak K | anchor drift K | Nr | under-covered | warm/new-case ms | VRAM MiB |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['candidate']} | {r['resolution']} | {float(r['full_pg_pct']):.6f} | {float(r['full_sample_first_pct']):.6f} | "
            f"{float(r['full_raw_cv_rmse_K']):.6f} | {float(r['source_rmse_K']):.6f} | {float(r['peak_rmse_K']):.6f} | "
            f"{('-' if r['common_anchor_response_rmse_K']=='' else f'{float(r['common_anchor_response_rmse_K']):.6f}')} | "
            f"{float(r['regional_nodes']):.0f} | {float(r['undercovered_fraction']):.6f} | "
            f"{float(r['warm_median_s'])*1000:.3f}/{float(r['new_case_median_s'])*1000:.3f} | {int(r['peak_vram_bytes'])/1048576:.1f} |"
            for r in rows
        ) +
        "\n\n## 冻结判定\n\n"
        f"- B 明确改善 gate：**{decisions['B']['passed']}**；平均 PG/raw 改善为 "
        f"{decisions['B']['mean_pg_gain_percentage_points']:.6f} pp / {decisions['B']['mean_raw_gain_K']:.6f} K，"
        f"但 source/peak 非退化={decisions['B']['source_peak_non_regression']}。\n"
        f"- C 明确改善 gate：**{decisions['C']['passed']}**；它在 8192 退化并产生可测 under-coverage。\n"
        "- 未触发 D；未冻结 winner，因此没有运行 4096/32768 扩展。生产策略继续使用 A。\n"
        "- native-1024 physical-coverage 假设**不成立**。仅保持半径不能解释 8192+ 趋势；"
        "support 分布、P2R 稀疏度、context/scale 表示和图响应仍是复合因素。\n"
        "- 历史 P1h 对照：8192/16384 的 P2R regional degree 为 28.225/28.726，Nr=1024/2048；"
        "P1i A 对应为 11.084/9.451，Nr=2048/4096。B 虽把 Nr 降至 P1h 密度，却未复现 P1h 的 physical/source coverage。\n"
        "- 候选 timing 使用同步连续 span，但独立 neural/apply 子段包含一次首调用 JIT；它们不进入正式性能表。"
        "warm-cache/new-case 连续 span 与 accuracy 仍有效。\n"
        "- A 的 new-case 来自历史 unified direct-wall new-topology，B/C 来自本轮连续 fresh-graph span；"
        "硬件相同但并非同期代码，故 latency 仅作工程参考，且没有参与已失败的 accuracy gate。\n"
        "- 实际新增计算：B8192 因初始误把 report-only k/q/CV 当 hard gate，在保留结果前有两次工程重试；"
        "最终 B8192/B16384/C8192/C16384 均由 SHA 绑定。没有重跑 A/FVM accuracy，也没有运行 D、winner 扩展或 timing-only 补测。\n"
        "\n## 工程优先级\n\n"
        "冻结策略仍为 A；新拓扑图构建主导端到端延迟，而 warm neural 仅为毫秒级。下一步优先 GPU 图构建优化。"
        "只有 support hash 重复时才优先固定图复用；对当前 B1 瓶颈，batch inference 优先级更低。\n"
    )
    perf = performance_rows()
    write_csv(PERF_CSV, perf)
    rigno = [r for r in perf if r["system"] == "GPU_RIGNO" and int(r["resolution"]) >= 4096]
    fvm = [r for r in perf if r["system"] == "CPU_FVM"]
    PERF_MD.write_text(
        "# V6/P1i CPU FVM 与 GPU RIGNO 跨分辨率性能\n\n"
        "硬件合同：同一 WSL2 主机的 RTX 5070 GPU 与 Ryzen 7 9700X CPU，B1。"
        "所有生产 GPU 区间以 `block_until_ready` 同步；metrics、hash、oracle 与 serialization 均在计时外。\n\n"
        "评价域不混淆：RIGNO 在共同 240825-node full field 上重建评价；structured-FVM 则在各自合法结构网格上与 240825 参考场比较。\n\n"
        "| N | RIGNO full PG % / raw K | RIGNO new / warm ms | FVM PG % / raw K | FVM new-physics / cached ms | new speedup | warm speedup | VRAM MiB |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['resolution']} | {float(r['point_global_pct']):.6f} / {float(r['raw_cv_rmse_K']):.6f} | "
            f"{float(r['new_case_median_s'])*1000:.3f} / {float(r['warm_cache_median_s'])*1000:.3f} | "
            f"{float(next(x for x in fvm if x['resolution']==r['resolution'])['point_global_pct']):.6f} / "
            f"{float(next(x for x in fvm if x['resolution']==r['resolution'])['raw_cv_rmse_K']):.6f} | "
            f"{float(next(x for x in fvm if x['resolution']==r['resolution'])['new_case_median_s'])*1000:.3f} / "
            f"{float(next(x for x in fvm if x['resolution']==r['resolution'])['warm_cache_median_s'])*1000:.3f} | "
            f"{float(r['new_case_speedup_vs_fvm']):.3f}x | {float(r['warm_speedup_vs_fvm']):.3f}x | {int(r['peak_vram_bytes'])/1048576:.1f} |"
            for r in rigno
        ) +
        "\n\n在完整 240825 节点上，CPU FVM new-physics/cached median 为 "
        f"{float(next(x for x in fvm if x['resolution']==240825)['new_case_median_s']):.6f}/"
        f"{float(next(x for x in fvm if x['resolution']==240825)['warm_cache_median_s']):.6f} s. "
        "历史 1024-RIGNO 加完整场重建路线在 JIT-cached 新拓扑下为 1.763765 s、fully cached 为 0.010003 s；"
        "相对 240825 FVM，new-case 为 0.962x，重复已知样本为 166.92x。CSV 明确标记其评价域和较早 timing 协议。\n\n"
        "`new_case_speedup_vs_fvm < 1` 表示 sample-varying 新 support 的 RIGNO 端到端更慢。"
        "warm-cache ratio 只比较重复已知样本 lower bound。CSV 单列 neural-core/FVM ratio，禁止称为 E2E speedup。\n"
        "\n证据来源：A accuracy/graph、GPU clean cached timing、unified direct-wall timing 与 FVM field 全部是历史只读复用。"
        "本轮只新增 B/C valid32 accuracy/graph 与同步 candidate warm/new-case span，并补了缺失的 GPU reconstruction-apply timing-only；"
        "未重跑 FVM 或 A accuracy。8192 首轮受 CUDA timer 抖动影响，正式值采用预先不看 accuracy 的 100-repeat 复核。\n"
    )
    print(json.dumps({"status": policy["status"], "decision": outcome, "graph_rows": len(graph_csv_rows), "performance_rows": len(perf)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
