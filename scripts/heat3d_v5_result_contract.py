"""Shared V5 result-registry fields and frozen metric names."""

from __future__ import annotations

V5_REPORT_ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
V5_CHECKPOINTS = ("primary_relative", "legacy_metric", "best", "final")
V5_FROZEN_METRICS = (
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

# V4-compatible scalar diagnostics retained for quick registry comparisons.
V4_VALUE_FIELDS = (
    "result_v4_best_valid_base_mse",
    "result_v4_final_valid_base_mse",
    "result_v4_best_valid_iid_loss",
    "result_v4_final_valid_iid_loss",
    "result_v4_best_valid_raw_deltaT_rmse_K",
    "result_v4_final_valid_raw_deltaT_rmse_K",
    "result_v4_best_valid_recovered_T_rmse_K",
    "result_v4_final_valid_recovered_T_rmse_K",
    "result_v4_best_valid_relative_rmse_pct_v4",
    "result_v4_final_valid_relative_rmse_pct_v4",
    "result_v4_corr_iid",
    "result_v4_amp",
    "result_v4_valid_iid_topk",
    "result_v4_strong_q_rmse",
    "result_v4_hotspot_mae",
    "result_v4_peak_abs",
    "result_v4_peak_rel",
    "result_v4_zrmse",
    "result_v4_final_probe_status",
    "result_v4_post_training_diagnostics_status",
)

V5_RESULT_FIELDS = (
    "result_v5_status",
    "result_v5_source",
    "result_v5_updated_at",
    "result_v5_commit",
    "result_v5_run_dir",
    "result_v5_log_path",
    "result_v5_loss_summary",
    "result_v5_metrics_json",
    "result_v5_required_metrics_complete",
    "result_v5_missing_metrics",
    "result_v5_primary_checkpoint",
    "result_v5_primary_epoch",
    "result_v5_legacy_checkpoint",
    "result_v5_legacy_epoch",
    "result_v5_primary_valid_point_global_relative_rmse_pct",
    "result_v5_primary_valid_sample_first_cv_relative_rmse_pct",
    "result_v5_primary_valid_raw_cv_weighted_rmse_K",
    "result_v5_primary_test_point_global_relative_rmse_pct",
    "result_v5_primary_test_sample_first_cv_relative_rmse_pct",
    "result_v5_primary_test_raw_cv_weighted_rmse_K",
    "result_v5_legacy_valid_base_mse",
    "result_v5_legacy_test_point_global_relative_rmse_pct",
    "result_v5_threshold_pass",
    "result_v5_final_probe_status",
    "result_v5_post_training_diagnostics_status",
    "result_v5_notes",
)

V5_REGISTRY_RESULT_FIELDS = V4_VALUE_FIELDS + V5_RESULT_FIELDS


def required_metric_paths() -> tuple[str, ...]:
    """Return checkpoint/role/metric paths required for a complete V5 report."""

    return tuple(
        f"{checkpoint}.{role}.{metric}"
        for checkpoint in ("primary_relative", "legacy_metric")
        for role in V5_REPORT_ROLES
        for metric in V5_FROZEN_METRICS
    )
