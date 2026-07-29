#!/usr/bin/env python3
"""Deterministic checker for the V6 final performance closeout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"
PREREG_COMMIT = "53fc4c334fd1f20813f1d32eb8cff9a22efc1c17"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    closeout = _load(CONFIG / "v6_final_performance_closeout.json")
    assert closeout["status"] == "passed"
    assert closeout["preregistration_commit"] == PREREG_COMMIT
    assert closeout["checkpoint_modified"] is False
    assert closeout["training_executed"] is False
    assert closeout["schema_version"] == (
        "heat3d_v6_final_performance_governance_v2"
    )
    assert closeout["confirmatory_holdout_classification"] == (
        "corrected_confirmatory_holdout"
    )
    assert closeout["confirmatory_holdout_opened_after_preregistration"] is True
    assert closeout["confirmatory_holdout_used_for_selection"] is False
    assert closeout["hard_accessed"] is False
    opening = closeout["protocol_deviation"]
    assert opening["status"] == "completed_with_corrected_command_input"
    assert opening["selection_or_workflow_changed"] is False
    assert opening["hard_accessed"] is False
    assert {
        row["resolution"] for row in opening["excluded_temporary_results"]
    } == {4096, 8192}
    assert all(
        row["status"] == "excluded_wrong_ladder_input"
        and row["used_for_selection_or_reporting"] is False
        for row in opening["excluded_temporary_results"]
    )
    assert set(opening["formal_result_sha256"]) == {
        "4096",
        "8192",
        "16384",
    }
    assert all(
        len(value) == 64 for value in opening["ladder_sha256"].values()
    )
    assert len(set(opening["ladder_sha256"].values())) == 2
    assert closeout["frozen_decision_unchanged"] == {
        "default_hotspot_oriented": 4096,
        "balanced_full_field": 8192,
        "iid_average_best_full_field_accuracy": 16384,
        "experimental_excluded_from_primary_test_table": 32768,
    }
    assert closeout["decision_basis"]["selection_source"] == (
        "valid_iid_timing_and_accuracy_before_holdout_open"
    )
    assert closeout["decision_basis"]["confirmatory_holdout_role"] == (
        "descriptive_confirmation_only"
    )
    timing = _rows(CONFIG / "v6_final_performance_timing.csv")
    persistent = _rows(CONFIG / "v6_final_persistent_gpu.csv")
    comparison = _rows(CONFIG / "v6_final_solver_inference_comparison.csv")
    pareto = _rows(CONFIG / "v6_final_accuracy_runtime_pareto.csv")
    test = _rows(
        CONFIG / "v6_final_corrected_confirmatory_holdout_metrics.csv"
    )
    assert len(timing) == 27
    assert len(persistent) == 6
    assert len(comparison) == 4
    assert len(pareto) == 6
    assert len(test) == 3
    assert {int(row["resolution"]) for row in test} == {4096, 8192, 16384}
    assert {int(row["query_resolution"]) for row in comparison} == {
        4096,
        8192,
        16384,
        32768,
    }
    assert all(row["nonmatched_dof"] == "True" for row in comparison)
    assert all(
        float(row["cpu_cold_model_core_s"])
        < float(row["cpu_cold_production_with_graph_build_s"])
        and float(row["gpu_b8_cold_model_core_s"])
        < float(row["gpu_b8_cold_production_with_graph_build_s"])
        for row in comparison
    )
    assert all(row["hard_accessed"] == "False" for row in test)
    assert all(row["used_for_selection"] == "False" for row in test)
    assert all(
        row["role_classification"] == "corrected_confirmatory_holdout"
        for row in test
    )
    for row in timing:
        model = float(row["model_core_seconds"])
        production = float(row["full_field_production_seconds"])
        evaluation = float(row["evaluation_seconds"])
        assert 0.0 < model <= production <= evaluation
        assert row["direct_single_cycle_measurements"] == "True"
        assert row["cross_run_phase_addition"] == "False"
    for row in persistent:
        assert int(row["batch_size"]) in {8, 16}
        assert int(row["resolution"]) in {4096, 8192, 16384}
        assert float(row["samples_per_second"]) > 0.0
        assert float(row["fvm_cold_speedup"]) > 1.0
        assert float(row["fvm_warm_speedup"]) > 1.0
    fvm = closeout["legal_structured_fvm_mesh_sensitivity"]
    assert fvm["status"] == "passed"
    assert fvm["schema_version"] == (
        "heat3d_v6_legal_structured_fvm_mesh_sensitivity_v1"
    )
    assert fvm["evaluation_role"] == "valid_iid"
    assert fvm["source_aware_points_used_as_fvm_mesh"] is False
    assert fvm["test_hard_accessed"] is False
    assert set(fvm["meshes"]) == {"coarse", "medium", "reference"}
    assert all(
        row["source_resolution"]["passed"]
        for row in fvm["meshes"].values()
    )
    runner = closeout["runner_graph_reuse"]
    assert runner["status"] == "passed"
    assert runner["run_level_cache_audit"]["shared_topology_build_calls"] == 1
    assert runner["forward_max_abs_error_K"] == 0.0
    assert runner["loss_abs_error"] == 0.0
    assert runner["gradient_max_abs_error"] == 0.0
    assert runner["updated_parameter_max_abs_error"] == 0.0
    assert runner["varying_support_fallback"][
        "metadata_hash_equal_to_legacy"
    ] is True
    for payload in closeout["corrected_confirmatory_holdout"].values():
        assert payload["role"] == "test_iid"
        assert payload["hard_accessed"] is False
        assert payload["training_executed"] is False
        assert payload["checkpoint_modified"] is False
        assert payload["checkpoint"]["sha256"] == (
            "3ad58c2b34a46481acb74722c80bdcadb"
            "f55a0d613bc25c4fe2d7646b91aa1f2"
        )
    numeric_columns = (
        "support_point_global_pct",
        "support_sample_first_pct",
        "support_raw_cv_rmse_K",
        "full_point_global_pct",
        "full_sample_first_pct",
        "full_raw_cv_rmse_K",
    )
    assert all(
        math.isfinite(float(row[key]))
        for row in test
        for key in numeric_columns
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "direct_timing_rows": len(timing),
                "persistent_gpu_rows": len(persistent),
                "solver_comparison_rows": len(comparison),
                "pareto_rows": len(pareto),
                "confirmatory_holdout_rows": len(test),
                "confirmatory_holdout_used_for_selection": False,
                "hard_accessed": False,
                "training_executed": False,
                "checkpoint_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
