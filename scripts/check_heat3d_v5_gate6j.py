#!/usr/bin/env python3
"""Validate the tracked Gate 6J valid-only causal diagnostic."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "configs/heat3d_v5/gate6j"
RESULT_JSON = RESULT_DIR / "gate6j_causal_diagnostic.json"
PAIRED_CSV = RESULT_DIR / "gate6j_paired_samples.csv"
ALPHA_CSV = RESULT_DIR / "gate6j_alpha_sweep.csv"
STRATA_CSV = RESULT_DIR / "gate6j_strata.csv"
REPORT_MD = ROOT / "docs/v5_gate6j_valid_causal_diagnostic.md"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6h_attention_fix_registry.csv"
EVALUATOR_COMMIT = "3d94dacba922c4288edb09adfdb47614e370234b"
V13_SHA256 = (
    "dac34633392015d7a1752367cca5ed9cb58fdb62331c46cdf31b0105fc49923d"
)
V32_SHA256 = (
    "f3063b53ca26a2b91fffc090ad4de98fe260ac5d7b669bcfbfd77c1fcf045d24"
)
FROZEN_TAG_TARGET = "96fa6fb5af451d8bafd6219f4372e36c792648d6"
REQUIRED_METRICS = (
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


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heat3d_v5_gate6j_closeout_v1"
    assert payload["evaluator_commit"] == EVALUATOR_COMMIT
    assert payload["metric_schema_version"] == (
        "heat3d_v5_clean_metrics_v2_true_rms"
    )
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["evaluation_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["sample_count"] == 128
    assert scope["nodes_per_sample"] == 1024
    for field in (
        "training_started",
        "model_parameters_modified",
        "checkpoints_modified",
        "checkpoint_selection_performed",
        "test_accessed",
        "hard_accessed",
        "sealed_iid_accessed",
    ):
        assert scope[field] is False
    normalization = payload["normalization_and_context"]
    assert normalization["fit_roles"] == ["train"]
    assert normalization["fit_sample_count"] == 672
    assert normalization["target_or_label_features"] == []

    assert payload["models"]["v13"]["epoch"] == 318
    assert payload["models"]["v13"]["sha256"] == V13_SHA256
    assert payload["models"]["v32"]["epoch"] == 474
    assert payload["models"]["v32"]["sha256"] == V32_SHA256
    for model in ("v13", "v32"):
        summary = payload["models"][model]["metrics"]["summary"]
        assert all(_finite(summary[metric]) for metric in REQUIRED_METRICS)

    bootstrap = payload["paired_bootstrap"]
    assert bootstrap["seed"] == 2026071801
    assert bootstrap["resamples"] == 20_000
    for metric in (
        "point_global_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
        "shape_cv_rmse",
        "scale_log_rmse",
    ):
        row = bootstrap["metrics"][metric]
        assert _finite(row["observed_aggregate_difference"])
        assert len(row["bootstrap_95pct_ci"]) == 2
        assert all(_finite(value) for value in row["bootstrap_95pct_ci"])
        assert 0.0 <= row["per_sample_win_rate"] <= 1.0
        assert _finite(row["per_sample_median_difference"])

    strata = payload["stratified_paired_analysis"]
    assert set(strata) == {
        "true_cv_rms_deltaT_K",
        "total_power_W",
        "source_occupancy_fraction",
        "q_weighted_inverse_kz_mK_W",
        "generator_condition_category",
    }
    for feature in (
        "true_cv_rms_deltaT_K",
        "total_power_W",
        "source_occupancy_fraction",
        "q_weighted_inverse_kz_mK_W",
    ):
        assert [row["sample_count"] for row in strata[feature]["bins"]] == [
            32,
            32,
            32,
            32,
        ]
    categories = {
        row["label"]: row["sample_count"]
        for row in strata["generator_condition_category"]["bins"]
    }
    assert categories == {
        "low": 23,
        "low_to_nominal": 24,
        "nominal_to_hard": 81,
    }

    residual = payload["attention_residual_analysis"]
    for field in (
        "attention_residual_mean_pool_cosine",
        "attention_residual_to_mean_pool_norm_ratio",
    ):
        distribution = residual["distributions"][field]
        assert all(_finite(value) for value in distribution.values())
        correlations = residual["correlations"][field]
        assert len(correlations["error_change"]) == 5
        assert len(correlations["physics_context"]) == 4
        for group in correlations.values():
            for row in group.values():
                assert row["sample_count"] == 128
                assert _finite(row["pearson"])
                assert _finite(row["spearman"])

    alpha = payload["alpha_sweep"]
    assert alpha["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert alpha["inference_only"] is True
    assert alpha["checkpoint_selection_performed"] is False
    assert alpha["replay"]["passed"] is True
    assert alpha["replay"]["manual_alpha1_vs_saved_max_abs_error_K"] < 0.02
    for value in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        summary = alpha["metrics"][value]["summary"]
        assert all(_finite(summary[metric]) for metric in REQUIRED_METRICS)

    route = payload["route_decision"]
    assert route["successful_small_alphas"] == []
    assert route["optimal_sensitivity_interval_alpha"] == [1.0, 1.0]
    assert route["recommendation"] == "objective_alignment"
    assert all(
        row["q1_sample_first_delta_vs_v13_pct_points"] > 0.0
        for row in route["rows"]
    )
    assert [
        row["alpha"]
        for row in route["rows"]
        if row["point_global_gain_preserved_vs_v13"]
    ] == [1.0]
    assert not any(
        row["sample_first_recovered_vs_v13"] for row in route["rows"]
    )

    paired_rows = _csv(PAIRED_CSV)
    alpha_rows = _csv(ALPHA_CSV)
    strata_rows = _csv(STRATA_CSV)
    assert len(paired_rows) == 128
    assert len({row["sample_id"] for row in paired_rows}) == 128
    assert len(alpha_rows) == 5
    assert len(strata_rows) == 19
    assert all(
        row["generator_condition_category"]
        in {"low", "low_to_nominal", "nominal_to_hard"}
        for row in paired_rows
    )

    registry_rows = _csv(REGISTRY)
    assert len(registry_rows) == 1
    registry = registry_rows[0]
    assert registry["gate6j_status"] == "completed_valid_iid_no_training"
    assert registry["gate6j_evaluator_commit"] == EVALUATOR_COMMIT
    assert registry["gate6j_recommendation"] == "objective_alignment"
    assert registry["gate6j_alpha_sensitivity_interval"] == "1|1"
    assert registry["gate6j_roles_accessed"] == "train|valid_iid"
    for field in (
        "gate6j_training_started",
        "gate6j_test_accessed",
        "gate6j_hard_accessed",
        "gate6j_sealed_iid_accessed",
    ):
        assert registry[field] == "false"

    report = REPORT_MD.read_text(encoding="utf-8")
    assert "objective_alignment" in report
    assert "test/hard/sealed" in report
    assert "No training" in report
    tag_target = subprocess.check_output(
        ["git", "rev-list", "-n", "1", "v5-gate6h-frozen"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert tag_target == FROZEN_TAG_TARGET
    assert not subprocess.check_output(
        ["git", "ls-files", "output/heat3d_v5_gate6j_diagnostics"],
        cwd=ROOT,
        text=True,
    ).strip()
    print(
        json.dumps(
            {
                "status": "passed",
                "roles": scope["roles_accessed"],
                "paired_samples": len(paired_rows),
                "alpha_count": len(alpha_rows),
                "strata_rows": len(strata_rows),
                "recommendation": route["recommendation"],
                "frozen_tag_target": tag_target,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
