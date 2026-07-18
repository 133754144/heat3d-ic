#!/usr/bin/env python3
"""Validate the frozen Gate 6L valid-only closeout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "configs/heat3d_v5/gate6l"
RESULT = RESULT_DIR / "gate6l_valid_only_evaluation.json"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6k_v32_single_variable_registry.csv"
REPORT = ROOT / "docs/v5_gate6l_valid_only_closeout.md"
EXPECTED_EVALUATOR = "ebbeed01b34b2e790bc3d7b87a6d64a8c6c70d8b"
EXPECTED_TRAINING = "461d810"
CHECKPOINTS = {"point_global_best", "sample_first_best", "legacy_best", "final"}
MODELS = {
    "O075": {
        "config_id": "V4P5_33_gate6k_o075_log_scale",
        "epochs": {
            "point_global_best": 280,
            "sample_first_best": 305,
            "legacy_best": 280,
            "final": 600,
        },
    },
    "Dual": {
        "config_id": "V4P5_34_gate6k_dual_physics_attention",
        "epochs": {
            "point_global_best": 298,
            "sample_first_best": 316,
            "legacy_best": 298,
            "final": 600,
        },
    },
}
REQUIRED_METRICS = {
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
}
REQUIRED_STRATA = {
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "nominal_to_hard",
    "Q2_intersection_nominal_to_hard",
    "scale_abs_error_top10pct",
    "scale_signed_low_p10",
    "scale_signed_central_p10_p90",
    "scale_signed_high_p90",
}


def _finite_metrics(summary: dict[str, object]) -> None:
    assert REQUIRED_METRICS <= set(summary)
    for key in REQUIRED_METRICS:
        assert math.isfinite(float(summary[key])), key


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "completed_valid_iid_only"
    assert (
        payload["schema_version"]
        == "heat3d_v5_gate6l_valid_only_evaluation_v1"
    )
    assert payload["metric_schema_version"] == "heat3d_v5_clean_metrics_v2_true_rms"
    assert payload["evaluator_commit"] == EXPECTED_EVALUATOR
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["evaluation_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["sample_count"] == 128 and scope["nodes_per_sample"] == 1024
    assert scope["training_started"] is False
    assert scope["model_parameters_modified"] is False
    assert scope["checkpoint_selection_modified"] is False
    assert not scope["test_accessed"]
    assert not scope["hard_accessed"]
    assert not scope["sealed_iid_accessed"]
    assert payload["split"]["train_count"] == 672
    assert payload["split"]["valid_iid_count"] == 128
    assert payload["normalization_and_context"]["fit_roles"] == ["train"]
    assert payload["normalization_and_context"]["target_or_label_features"] == []

    for model_name, expected in MODELS.items():
        model = payload["models"][model_name]
        assert model["config_id"] == expected["config_id"]
        assert model["training_commit"] == EXPECTED_TRAINING
        assert set(model["metrics"]) == CHECKPOINTS
        assert set(model["checkpoint_metadata"]) == CHECKPOINTS
        assert set(model["reload_audit"]) == CHECKPOINTS
        for checkpoint in CHECKPOINTS:
            metadata = model["checkpoint_metadata"][checkpoint]
            assert metadata["epoch"] == expected["epochs"][checkpoint]
            assert len(metadata["sha256"]) == 64
            assert metadata["parameter_count"] > 0
            assert metadata["parameter_reload_max_abs_error"] == 0.0
            replay = model["reload_audit"][checkpoint]
            assert replay["passed"] is True
            assert replay["max_abs_error_K"] < replay["tolerance_K"] == 0.02
            _finite_metrics(model["metrics"][checkpoint]["summary"])
            reports = model["strata"][checkpoint]["reports"]
            assert REQUIRED_STRATA <= set(reports)
            for stratum in REQUIRED_STRATA:
                assert reports[stratum]["sample_count"] > 0
                _finite_metrics(reports[stratum]["metrics"])

    pairs = payload["paired_primary_point_global_best"]
    assert set(pairs) == {
        "O075_minus_V32",
        "Dual_minus_V32",
        "Dual_minus_O075",
    }
    expected_seeds = {
        "O075_minus_V32": 2026071802,
        "Dual_minus_V32": 2026071803,
        "Dual_minus_O075": 2026071804,
    }
    for pair, comparison in pairs.items():
        assert comparison["resamples"] == 20000
        assert comparison["seed"] == expected_seeds[pair]
        assert (
            comparison["difference_orientation"]
            == "right_minus_left; negative is improvement"
        )
        for metric in (
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
        ):
            result = comparison["metrics"][metric]
            for key in (
                "observed_difference",
                "bootstrap_probability_right_improves",
                "per_sample_win_rate",
                "per_sample_median_difference",
            ):
                assert math.isfinite(float(result[key]))
            ci_low, ci_high = result["bootstrap_95pct_ci"]
            assert math.isfinite(float(ci_low))
            assert math.isfinite(float(ci_high))
            assert ci_low <= ci_high
            assert 0.0 <= result["per_sample_win_rate"] <= 1.0
        tail = comparison["tail_contribution"]
        assert tail["point_sse_absolute_delta_sum_K2"] >= 0.0
        assert tail["left_model_scale_tail_sample_count"] > 0

    for name in (
        "gate6l_checkpoint_comparison.csv",
        "gate6l_paired_samples.csv",
        "gate6l_paired_bootstrap.csv",
        "gate6l_strata.csv",
        "gate6l_o075_valid_only_metrics.json",
        "gate6l_dual_valid_only_metrics.json",
    ):
        assert (RESULT_DIR / name).is_file(), name
    assert REPORT.is_file()

    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        model_name = next(
            name
            for name, expected in MODELS.items()
            if expected["config_id"] == row["config_id"]
        )
        expected = MODELS[model_name]
        assert row["plan_status"] == "completed"
        assert row["execution_status"] == "completed_e600"
        assert row["evaluation_status"] == "completed_valid_iid_four_checkpoint"
        assert row["training_started"] == "true"
        assert row["test_accessed"] == "false"
        assert row["hard_accessed"] == "false"
        assert row["sealed_iid_accessed"] == "false"
        assert row["result_v5_status"] == "completed_valid_only"
        assert row["result_v5_required_metrics_complete"] == "true"
        assert row["result_v5_missing_metrics"] == ""
        assert row["result_v5_primary_epoch"] == str(
            expected["epochs"]["point_global_best"]
        )
        assert row["result_v5_legacy_epoch"] == str(
            expected["epochs"]["legacy_best"]
        )
        assert row["gate6l_status"] == "completed_valid_iid_four_checkpoint"
        assert row["gate6l_evaluator_commit"] == EXPECTED_EVALUATOR
        assert row["gate6l_training_commit"] == EXPECTED_TRAINING
        assert row["gate6l_roles_accessed"] == "train|valid_iid"
        assert row["gate6l_no_auto_advancement"] == "true"
        hashes = json.loads(row["gate6l_checkpoint_sha256_json"])
        reload_audit = json.loads(row["gate6l_reload_audit_json"])
        assert set(hashes) == CHECKPOINTS
        assert all(len(value) == 64 for value in hashes.values())
        assert set(reload_audit) == CHECKPOINTS
        assert all(entry["passed"] for entry in reload_audit.values())

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "output/heat3d_v5_gate6l_eval_inputs",
            "output/heat3d_v5_gate6k_o075_runs",
            "output/heat3d_v5_gate6k_dual_runs",
            "checkpoints",
            "logs",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""
    print(
        json.dumps(
            {
                "status": "passed",
                "models": sorted(MODELS),
                "checkpoints_per_model": 4,
                "roles_accessed": ["train", "valid_iid"],
                "training_started": False,
                "automatic_advancement": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
