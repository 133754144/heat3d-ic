#!/usr/bin/env python3
"""Validate the V32 valid-only closeout without loading model checkpoints."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
RESULT_DIR = ROOT / "configs/heat3d_v5/gate6h/v32_closeout"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6h_attention_fix_registry.csv"
METRICS_JSON = RESULT_DIR / "v32_valid_only_metrics.json"
ATTENTION_JSON = RESULT_DIR / "v32_attention_diagnostics.json"
COMPARISON_CSV = RESULT_DIR / "v32_checkpoint_comparison.csv"
CHECKPOINTS = {
    "point_global_best": (
        474,
        "f3063b53ca26a2b91fffc090ad4de98fe260ac5d7b669bcfbfd77c1fcf045d24",
    ),
    "sample_first_best": (
        366,
        "7e2c62667e6aed4d862214a384f9eeddc374b45ea6512a94d7a8a98c183b5b2e",
    ),
    "legacy_best": (
        474,
        "9a107249f5dd5f08d58fa0ea084198a77863e0fe86aa7ffb90e6dff69bc5995f",
    ),
    "final": (
        600,
        "27bf9b51a07567edf91e73e6af77c8a05c88017d8e9072f77bcec86342c6cefa",
    ),
}
FROZEN_TAG_TARGET = "96fa6fb5af451d8bafd6219f4372e36c792648d6"
METRICS = (
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
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    payload = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    assert payload["config_id"] == CONFIG_ID
    assert payload["training_commit"] == "fcdb01d"
    assert payload["evaluator_commit"] == (
        "370ed5cb661da5809d7d34e40cbf49011592d023"
    )
    assert payload["metric_schema_version"] == (
        "heat3d_v5_clean_metrics_v2_true_rms"
    )
    assert payload["training_completion"]["final_epoch"] == 600
    assert payload["training_completion"]["epoch_history_contiguous_1_to_600"]
    assert payload["training_completion"]["grad_finite"]
    assert payload["training_completion"]["declared_log_exists"] is False
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["evaluation_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["training_started"] is False
    assert scope["model_parameters_modified"] is False
    assert scope["checkpoint_selection_modified"] is False
    for role in ("test", "hard", "sealed_iid"):
        assert scope[f"{role}_accessed"] is False
    assert payload["normalization_and_context"]["fit_roles"] == ["train"]
    assert payload["normalization_and_context"]["fit_sample_count"] == 672
    assert payload["split"]["valid_iid_count"] == 128

    for checkpoint, (epoch, sha256) in CHECKPOINTS.items():
        metadata = payload["checkpoint_metadata"][checkpoint]
        assert metadata["epoch"] == epoch
        assert metadata["sha256"] == sha256
        assert metadata["parameter_count"] == 893736
        assert metadata["parameter_reload_max_abs_error"] == 0.0
        assert metadata["training_replay_passed"] is True
        summary = payload["metrics"][checkpoint]["summary"]
        assert all(_finite(summary[metric]) for metric in METRICS)
        diagnostic = payload["attention_diagnostics"][checkpoint]
        assert diagnostic["finite"] is True
        assert diagnostic["sample_count"] == 128
        assert diagnostic["classification"] == "effective_regional_selection"
        assert _finite(diagnostic["evaluator_replay_max_abs_error_K"])
        assert diagnostic["evaluator_replay_max_abs_error_K"] < 0.02
        for feature in (
            "source_present_fraction",
            "log1p_q_relative",
            "log_inverse_kz_relative",
            "log1p_q_inverse_kz_relative",
        ):
            correlations = diagnostic[
                "weight_feature_correlations_computed_per_sample_then_aggregated"
            ][feature]
            assert correlations["pearson"]["finite_sample_count"] == 128
            assert correlations["spearman"]["finite_sample_count"] == 128

    caveat = payload["checkpoint_selection_caveat"]
    assert caveat["saved_tie_break_actual"] == "ordinary_raw_rmse_K"
    assert caveat["saved_tie_break_condition"] == (
        "exact_primary_metric_equality_only"
    )
    assert caveat["correct_cv_metric_role"] == "post_hoc_diagnostic_only"
    assert caveat["checkpoint_reselection_performed"] is False

    attention = json.loads(ATTENTION_JSON.read_text(encoding="utf-8"))
    assert attention["scope"] == scope
    assert set(attention["attention_diagnostics"]) == set(CHECKPOINTS)
    with COMPARISON_CSV.open(newline="", encoding="utf-8") as handle:
        comparison = list(csv.DictReader(handle))
    assert [row["checkpoint"] for row in comparison] == [
        "point_global_best",
        "legacy_best",
        "sample_first_best",
        "final",
    ]

    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["config_id"] == CONFIG_ID
    row = rows[0]
    assert row["plan_status"] == "frozen_closed"
    assert row["execution_status"] == "completed_e600"
    assert row["evaluation_status"] == "completed_valid_iid_four_checkpoint"
    assert row["long_training_started"] == "true"
    assert row["test_accessed"] == "false"
    assert row["hard_accessed"] == "false"
    assert row["sealed_iid_accessed"] == "false"
    assert row["result_v5_status"] == "completed_valid_only"
    assert row["result_v5_required_metrics_complete"] == "true"
    assert row["result_v5_missing_metrics"] == ""
    assert row["result_v5_primary_epoch"] == "474"
    assert row["result_v5_legacy_epoch"] == "474"
    assert row["result_v5_threshold_pass"] == "valid_only_fail"
    assert row["result_v5_final_probe_status"] == "not_applicable_valid_only"
    assert (
        row["result_v5_post_training_diagnostics_status"]
        == "not_applicable_valid_only"
    )
    embedded = json.loads(row["result_v5_metrics_json"])
    assert embedded["evaluator_commit"] == payload["evaluator_commit"]
    assert set(embedded["metrics"]) == set(CHECKPOINTS)

    for name in (
        "v32_valid_only_metrics.md",
        "v32_attention_diagnostics.md",
        "v32_vs_v13_closeout.md",
        "v32_next_optimization_options.md",
    ):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "test/hard/sealed" in text or "test, hard, or sealed-IID" in text

    tag_target = subprocess.check_output(
        ["git", "rev-list", "-n", "1", "v5-gate6h-frozen"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert tag_target == FROZEN_TAG_TARGET
    print(
        json.dumps(
            {
                "status": "passed",
                "config_id": CONFIG_ID,
                "checkpoints": list(CHECKPOINTS),
                "roles": scope["roles_accessed"],
                "frozen_tag_target": tag_target,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
