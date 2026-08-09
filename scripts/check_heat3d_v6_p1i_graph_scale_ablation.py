#!/usr/bin/env python3
"""Fail-closed checks for the preregistered P1i graph-scale ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GRAPH_BUILDER_SHA = "fce189e90aa3e182a418cd1ef50a9b5d24558fc3d24e50f9d6d1e734c3129cc3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    require(payload["status"] == "preregistered_before_candidate_execution", "protocol status")
    scope = payload["scientific_scope"]
    require(scope["mandatory_candidate_resolutions"] == [8192, 16384], "mandatory resolutions")
    require(scope["winner_extension_resolutions"] == [4096, 32768], "winner extensions")
    require(scope["forbidden_resolutions"] == [65536], "forbidden resolution")
    require(not scope["training"] and not scope["test_accessed"] and not scope["sealed_accessed"], "role contract")
    candidates = payload["candidates"]
    require(candidates["A"]["run"] is False, "baseline A must be reused")
    require(candidates["D"]["trigger"] == "B_and_C_both_clear_improvement_over_A", "D trigger")
    native = payload["native1024_physical_coverage_v1"]
    require(native["uses_temperature_prediction_or_error"] is False, "native policy leakage")
    require(native["changes_legacy_discrete_physical_coverage"] is False, "legacy semantics")
    require(sha256(ROOT / "rigno/graphBuilder_Heat3D.py") == EXPECTED_GRAPH_BUILDER_SHA, "legacy graph builder drift")
    implementation = (ROOT / "rigno/heat3d_v6_graph_scale.py").read_text()
    for forbidden in ("target_u", "temperature", "prediction", "truth", "q_W_m3"):
        require(forbidden not in implementation, f"label/error-dependent graph implementation: {forbidden}")
    timing_amendment = json.loads(
        (ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_a_timing_amendment.json").read_text()
    )
    require(
        timing_amendment["status"] == "preregistered_before_a_timing_only_execution",
        "A timing amendment",
    )
    timing_scope = timing_amendment["authorized_scope"]
    require(timing_scope["resolutions"] == [8192, 16384], "A timing resolutions")
    require(timing_scope["timing_only"] and not timing_scope["accuracy_recomputed"], "A timing scope")
    return {"protocol_checked": True, "legacy_graph_builder_unchanged": True}


def check_result(path: Path, candidate: str, resolution: int) -> dict[str, object]:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed", "candidate execution failed")
    require(payload["candidate"] == candidate and payload["resolution"] == resolution, "identity")
    require(len(payload["sample_ids"]) == 32 and len(set(payload["sample_ids"])) == 32, "valid32")
    role = payload["role_contract"]
    require(not role["training"] and not role["test"] and not role["sealed"], "role access")
    require(not role["checkpoint_modified"] and not role["support_or_physics_modified"], "frozen inputs")
    for name in ("delta_k", "delta_q", "delta_cv"):
        drift = payload["common_anchor_input_drift"][name]
        require(float(drift["max_abs"]) >= 0.0 and float(drift["max_rmse"]) >= 0.0, f"{name} drift")
    require(
        payload["common_anchor_input_drift_interpretation"]
        == "report_only_frozen_high_n_overlap_fields_vs_native1024_binary_mask_fields",
        "anchor input drift semantics",
    )
    graph = payload["graph_diagnostics"]
    require(0.0 <= float(graph["undercovered_fraction"]) <= 1.0, "under-covered fraction")
    require(float(graph["r2r_connected_components"]["max"]) >= 1.0, "connected-component diagnostic")
    for domain in ("support", "full_field"):
        values = payload["accuracy"][domain]
        for key in (
            "point_global_true_rms_relative_rmse_pct", "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K", "source_rmse_K", "background_rmse_K",
            "interface_drop_rmse_K", "peak_rmse_K",
        ):
            require(float(values[key]) >= 0.0, f"missing/non-finite {domain}.{key}")
    for name in ("neural_core", "reconstruction_apply_gpu", "warm_cache_e2e", "new_case_e2e"):
        timing = payload["timing"][name]
        require(timing["count"] >= 20 and timing["median_seconds"] > 0 and timing["p95_seconds"] > 0, f"timing {name}")
    return {"result_checked": True, "candidate": candidate, "resolution": resolution}


def check_a_timing_result(path: Path, resolution: int) -> dict[str, object]:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed_timing_only", "A timing-only status")
    require(payload["candidate"] == "A" and payload["resolution"] == resolution, "A timing identity")
    require(len(payload["sample_ids"]) == 32, "A timing valid32")
    require(payload["accuracy"] == {
        "status": "not_evaluated_timing_only", "metrics_evaluated": False,
        "accuracy_recomputed": False,
    }, "A timing accuracy boundary")
    role = payload["role_contract"]
    require(role["timing_only"] and not role["metrics_evaluated"], "A timing role")
    require(not role["training"] and not role["test"] and not role["sealed"], "A timing access")
    require(not role["prediction_artifact_saved"] and "prediction_artifact" not in payload, "A timing prediction")
    for name in ("graph_construction", "warm_cache_e2e", "new_case_e2e"):
        timing = payload["timing"][name]
        require(timing["count"] >= 20 and timing["median_seconds"] > 0.0, f"A timing {name}")
    return {"a_timing_result_checked": True, "resolution": resolution}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_ablation_protocol.json")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--candidate", choices=["B", "C", "D"])
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--closeout", action="store_true")
    parser.add_argument("--a-timing-result", type=Path)
    args = parser.parse_args()
    report = check_protocol(args.protocol)
    if args.result:
        require(args.candidate is not None and args.resolution is not None, "result identity arguments")
        report.update(check_result(args.result, args.candidate, args.resolution))
    if args.a_timing_result:
        require(args.resolution in (8192, 16384), "A timing resolution")
        report.update(check_a_timing_result(args.a_timing_result, args.resolution))
    if args.closeout:
        policy_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_policy_closeout.json"
        policy = json.loads(policy_path.read_text())
        require(policy["status"] == "completed_no_go", "closeout status")
        require(policy["decision"] == "no_go_keep_A", "closeout decision")
        require(policy["candidate_production_graph_policy"] is None, "unexpected winner")
        require(policy["D_executed"] is False and policy["winner_extension_executed"] is False, "unexpected execution")
        require(policy["frozen_roles"] == {
            "training": False, "test": False, "sealed": False, "valid32_seed0_only": True,
        }, "closeout roles")
        for candidate in ("B", "C"):
            require(policy["selection"][candidate]["passed"] is False, f"{candidate} decision")
            for resolution in (8192, 16384):
                raw = ROOT / f"configs/heat3d_v6_p1i/v6_p1i_graph_scale_ablation_raw/{candidate}_{resolution}.json"
                check_result(raw, candidate, resolution)
                require(sha256(raw) == policy["raw_artifacts"][f"{candidate}_{resolution}"]["sha256"], "raw SHA")
        require(
            policy["timing_comparison_status"]
            == "same_executor_continuous_span_completed_for_A_B_C",
            "matched timing comparison status",
        )
        for resolution in (8192, 16384):
            raw = ROOT / (
                "configs/heat3d_v6_p1i/v6_p1i_graph_scale_ablation_raw/"
                f"A_{resolution}_timing_only.json"
            )
            check_a_timing_result(raw, resolution)
            frozen = policy["matched_A_timing_only"][f"A_{resolution}"]
            require(frozen["same_executor_span_as_B_C"], "A matched timing semantics")
            require(frozen["accuracy_recomputed"] is False, "A accuracy boundary")
            require(sha256(raw) == frozen["sha256"], "A timing raw SHA")
        for key in ("first", "replay8192"):
            timing = policy["timing_only_artifacts"][key]
            path = ROOT / timing["path"]
            payload = json.loads(path.read_text())
            require(payload["status"] == "passed", "timing-only status")
            require(payload["timing_contract"]["labels_or_metrics_read"] is False, "timing-only labels")
            require(payload["role_contract"]["training"] is False, "timing-only training")
            require(sha256(path) == timing["sha256"], "timing-only SHA")
        with (ROOT / "docs/v6_p1i_graph_scale_ablation.csv").open(newline="") as handle:
            graph_rows = list(csv.DictReader(handle))
        with (ROOT / "docs/v6_p1i_resolution_performance_comparison.csv").open(newline="") as handle:
            performance_rows = list(csv.DictReader(handle))
        require(len(graph_rows) == 8, "graph CSV row count")
        require(len(performance_rows) == 11, "performance CSV row count")
        require({row["candidate"] for row in graph_rows} == {"A", "B", "C", "P1h_context"}, "graph CSV roles")
        require(all(row["measurement_domain"] for row in performance_rows), "performance domain")
        for row in performance_rows:
            if row["system"] == "GPU_RIGNO" and int(row["resolution"]) >= 4096:
                require(float(row["reconstruction_apply_median_s"]) > 0.0, "GPU apply timing")
            if row["system"] == "GPU_RIGNO" and int(row["resolution"]) in (8192, 16384):
                require(
                    row["timing_evidence"] == "new_matched_A_timing_only_current_executor",
                    "performance A matched timing evidence",
                )
        a_rows = {
            int(row["resolution"]): row for row in graph_rows if row["candidate"] == "A"
        }
        for resolution in (8192, 16384):
            require(
                a_rows[resolution]["evidence"]
                == "historical_accuracy_plus_new_matched_timing_only",
                "graph table A matched timing evidence",
            )
            require(
                a_rows[resolution]["artifact_sha256"]
                == policy["matched_A_timing_only"][f"A_{resolution}"]["sha256"],
                "graph table A timing SHA",
            )
        graph_md = (ROOT / "docs/v6_p1i_graph_scale_ablation.md").read_text()
        perf_md = (ROOT / "docs/v6_p1i_resolution_performance_comparison.md").read_text()
        for phrase in ("不成立", "未触发 D", "GPU 图构建优化", "同一 executor"):
            require(phrase in graph_md, f"missing graph conclusion: {phrase}")
        for phrase in ("评价域", "重复已知样本", "neural-core/FVM"):
            require(phrase in perf_md, f"missing timing qualification: {phrase}")
        report["closeout_checked"] = True
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
