#!/usr/bin/env python3
"""Close the preregistered fixed-Nr E screen and freeze A/B confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"
PREREG = CONFIG / "v6_p1i_graph_policy_e_confirmation_preregistration.json"
E_RESULT = CONFIG / "v6_p1i_graph_policy_e_raw/E_8192.json"
A_RESULT = CONFIG / "v6_p1i_gpu_only_high_n_closeout.json"
A_TIMING = CONFIG / "v6_p1i_graph_scale_ablation_raw/A_8192_timing_only.json"
SCREEN_JSON = CONFIG / "v6_p1i_graph_policy_e_screen_closeout.json"
SCREEN_MD = ROOT / "docs/v6_p1i_graph_policy_e_screen_closeout.md"
CONFIRM = CONFIG / "v6_p1i_graph_policy_ab_confirmation_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    prereg = json.loads(PREREG.read_text())
    e = json.loads(E_RESULT.read_text())
    a_close = json.loads(A_RESULT.read_text())
    a_timing = json.loads(A_TIMING.read_text())
    assert prereg["status"] == "preregistered_before_E_execution"
    assert e["candidate"] == "E" and e["resolution"] == 8192
    a_row = next(row for row in a_close["rows"] if int(row["resolution"]) == 8192)
    e_full = e["accuracy"]["full_field"]
    margins = prereg["historical_margin_source"]["values"]
    fields = {
        "full_point_global_pct": ("full_point_global_pct", "point_global_true_rms_relative_rmse_pct"),
        "full_raw_cv_rmse_K": ("full_raw_cv_rmse_K", "raw_cv_weighted_rmse_K"),
        "source_rmse_K": ("full_source_rmse_K", "source_rmse_K"),
        "peak_rmse_K": ("full_peak_rmse_K", "peak_rmse_K"),
        "interface_rmse_K": ("full_interface_drop_rmse_K", "interface_drop_rmse_K"),
    }
    accuracy = {}
    for name, (a_key, e_key) in fields.items():
        delta = float(e_full[e_key]) - float(a_row[a_key])
        accuracy[name] = {
            "A": float(a_row[a_key]), "E": float(e_full[e_key]),
            "E_minus_A": delta, "margin": float(margins[name]),
            "passed": delta <= float(margins[name]),
        }
    graph = e["graph_diagnostics"]
    coverage = {
        "regional_node_count_exact": float(graph["regional_node_count"]) == 256.0,
        "undercovered_fraction": float(graph["undercovered_fraction"]),
        "max_isolated_physical_fraction": max(
            float(row["isolated_fraction"]["p2r_physical"])
            for row in e["graph_diagnostics_per_sample"]
        ),
        "r2r_connected_components_max": float(graph["r2r_connected_components"]["max"]),
    }
    coverage["passed"] = bool(
        coverage["regional_node_count_exact"]
        and coverage["undercovered_fraction"] == 0.0
        and coverage["max_isolated_physical_fraction"] == 0.0
        and coverage["r2r_connected_components_max"] == 1.0
    )
    latency = {}
    for name in ("new_case_e2e", "warm_cache_e2e", "graph_construction"):
        av = float(a_timing["timing"][name]["median_seconds"])
        ev = float(e["timing"][name]["median_seconds"])
        latency[name] = {"A_seconds": av, "E_seconds": ev, "improvement_fraction": 1.0 - ev / av}
    vram_ratio = (
        float(e["device_memory"]["peak_bytes_in_use"])
        / float(a_timing["device_memory"]["peak_bytes_in_use"])
    )
    latency.update({
        "A_peak_vram_bytes": int(a_timing["device_memory"]["peak_bytes_in_use"]),
        "E_peak_vram_bytes": int(e["device_memory"]["peak_bytes_in_use"]),
        "peak_vram_ratio": vram_ratio,
        "speed_passed": max(
            latency["new_case_e2e"]["improvement_fraction"],
            latency["warm_cache_e2e"]["improvement_fraction"],
        ) >= 0.05,
        "vram_passed": vram_ratio <= 1.10,
    })
    latency["passed"] = bool(latency["speed_passed"] and latency["vram_passed"])
    passed = bool(all(row["passed"] for row in accuracy.values()) and coverage["passed"] and latency["passed"])
    assert not passed, "E unexpectedly passed; protocol requires 16384 before closeout"
    closeout = {
        "schema_version": "heat3d_v6_p1i_graph_policy_e_screen_closeout_v1",
        "status": "completed_E_no_go_stopped_after_8192",
        "decision": "E_NO_GO",
        "reason": "peak_vram_ratio_exceeded_preregistered_1.10_limit",
        "E_16384_executed": False,
        "confirmation_policy_set": ["A", "B"],
        "accuracy": accuracy, "coverage": coverage, "latency": latency,
        "artifacts": {
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "E_8192": {"path": str(E_RESULT.relative_to(ROOT)), "sha256": sha256(E_RESULT)},
            "failed_pre_graph_launch_log": {"remote_path": "/tmp/v6_p1i_graph_policy_e_e856_E8192.log", "sha256": "fd495c7afc37c1e6a2ef8303f39a026e90769c159ebb61ad5ce710d20652e7f6"},
            "successful_log": {"remote_path": "/tmp/v6_p1i_graph_policy_e_e856_E8192_retry.log", "sha256": "152c7918d3e641dd49da8498afe4cde519049838be14810b9669186f631df900"}
        },
        "role_contract": {"training": False, "test": False, "sealed": False, "valid32_seed0_only": True},
    }
    SCREEN_JSON.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    confirmation = {
        "schema_version": "heat3d_v6_p1i_graph_policy_ab_confirmation_protocol_v1",
        "status": "frozen_after_E_no_go_before_confirmation",
        "screen_closeout": {"path": str(SCREEN_JSON.relative_to(ROOT)), "sha256": sha256(SCREEN_JSON)},
        "policies": ["A", "B"], "primary_candidate": "B",
        "resolutions": [8192, 16384],
        "population": {
            "role": "valid_iid", "count": 96,
            "rule": "ascending SHA256(sample_id), exclude frozen first32",
            "frozen_valid32_recomputed": False,
        },
        "checkpoints": {
            "0": {"config_id": "V6_06_V5best_P1i_seed0_reliable_B24", "epoch": 559, "sha256": "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e", "run_config_sha256": "c814b74be5b0d6a316b99e4be41db312a2075a3a01752ca5fe1e65d9b8c77ef8"},
            "1": {"config_id": "V6_07_V5best_P1i_seed1_reliable_B24", "epoch": 455, "sha256": "7197157969278d99648ef9b40d74005d759f32e52e9282e72a5586003d1e71f7", "run_config_sha256": "7592aab6605fca69d080c20478564a41e75934966f35926971ada638f04e57e6"},
            "2": {"config_id": "V6_08_V5best_P1i_seed2_reliable_B24", "epoch": 587, "sha256": "d67e0dac2e8ed8009ce7dcdf0b2de4543b10bc005c0bfaa51027ea721bb2ab49", "run_config_sha256": "97090f042109aea5bc861e01879728dee3a3b53f60e1175a8d91a04b6b009cbf"}
        },
        "accuracy_metrics_K": ["full_rmse_K", "source_rmse_K", "peak_rmse_K", "interface_rmse_K"],
        "non_inferiority_margins": margins,
        "paired_bootstrap": {"seed": 20260810, "replicates": 10000, "confidence": 0.95, "cluster_unit": "sample_id_with_all_three_seeds", "difference": "B_minus_A"},
        "accuracy_gate": "upper_95pct_CI <= frozen margin for all four K metrics at both resolutions",
        "latency_pareto_gate": {"fresh_median_B_le_A": True, "warm_median_B_le_A": True, "peak_vram_ratio_max": 1.05, "at_least_one_latency_improvement_fraction": 0.05, "required_at_both_resolutions": True},
        "selection": "B_GO only if accuracy gate and latency Pareto gate pass; otherwise retain A",
        "role_contract": {"training": False, "test": False, "sealed": False, "no_factor_or_radius_search": True},
    }
    CONFIRM.write_text(json.dumps(confirmation, indent=2, sort_keys=True) + "\n")
    SCREEN_MD.write_text(
        "# P1i graph policy E screen closeout\n\n"
        "E 在 8192 的 accuracy 与 coverage 均通过冻结门；fresh/warm 分别改善 "
        f"{latency['new_case_e2e']['improvement_fraction']:.2%}/{latency['warm_cache_e2e']['improvement_fraction']:.2%}。"
        f"但峰值 VRAM 比为 {vram_ratio:.4f}，超过预注册 1.10，因此 **E NO-GO**，未运行 16384。\n\n"
        "确认协议冻结为 A/B，使用剩余 valid96、三 seed、8192/16384；不重算冻结 valid32。\n"
    )
    print(json.dumps({"status": closeout["status"], "decision": closeout["decision"], "confirmation": ["A", "B"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
