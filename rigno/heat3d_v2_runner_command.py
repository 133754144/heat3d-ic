"""Dry-run command builders for Heat3D v2 configs.

The helpers in this module translate draft v2 YAML configs into command lists
for the existing v1 runner and diagnostics scripts. They do not execute those
commands, import training code, read datasets, or create output directories.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
import shlex
from typing import Any

from rigno.heat3d_v2_config import validate_v2_config


TRAINING_SCRIPT = "scripts/run_heat3d_v1_medium_controlled_training_export.py"
V4_TRAINING_SCRIPT = "scripts/run_heat3d_v4_controlled_training.py"
NORMALIZATION_PROFILE_LEGACY_ZSCORE = "legacy_zscore"
NORMALIZATION_PROFILE_SEMANTIC_V1 = "semantic_normalization_v1"
NORMALIZATION_PROFILES = {
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
}
INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS = "legacy_bc_flags"
INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT = "boundary_distance_replacement"
INPUT_FEATURE_SCHEMAS = {
    INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS,
    INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT,
}
COORD_POLICY_TRAIN_MINMAX_UNIT_BOX = "train_minmax_to_unit_box"
COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC = "sample_local_isotropic"
COORD_POLICIES = {
    COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC,
}
EXTENT_FEATURE_POLICY_NONE = "none"
EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST = "log_extent_broadcast"
EXTENT_FEATURE_POLICIES = {
    EXTENT_FEATURE_POLICY_NONE,
    EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST,
}
CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE = "legacy_zscore_all_condition_features"
CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL = (
    "semantic_v1_logk_signedlog1p_q_binary_bcflags_independent_bc_scalars"
)
CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY = (
    "semantic_v1_bc_flags_binary_passthrough_only"
)
CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY = "semantic_v1_q_signedlog1p_only"
CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY = "semantic_v1_k_log_only"
CONDITION_FEATURE_TRANSFORMS = {
    CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
}
DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP = (
    "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json"
)
COMPARISON_SCRIPT = "scripts/compare_heat3d_v1_medium_baselines.py"
ERROR_BINS_SCRIPT = "scripts/analyze_heat3d_v1_medium_error_bins.py"
RUN_SUMMARY_SCRIPT = "scripts/analyze_heat3d_v1_medium_run_summary.py"
CONDITION_DIAGNOSTICS_SCRIPT = (
    "scripts/analyze_heat3d_v1_medium_condition_diagnostics.py"
)
FIELD_SHAPE_DIAGNOSTICS_SCRIPT = (
    "scripts/analyze_heat3d_v2_field_shape_diagnostics.py"
)
LEGACY_OPTIMIZER_NAME = "manual_full_batch_gradient_descent"
NON_EXECUTION_NOTE = (
    "Dry-run only: generated commands are not executed and no output "
    "directories or files are created."
)


def build_training_command(
    config: Mapping[str, Any], *, python_executable: str = "python3"
) -> list[str]:
    """Build the v1 controlled training/export command without executing it."""

    validate_v2_config(config)
    dataset = _section(config, "dataset")
    model = _section(config, "model")
    run = _section(config, "run")
    optimizer = _section(config, "optimizer")
    loss = _section(config, "loss")
    export = _section(config, "export")
    graph_section = config.get("graph")
    graph = graph_section if isinstance(graph_section, Mapping) else {}

    normalization_profile = _normalization_profile(config)
    condition_feature_transform = _condition_feature_transform(config)
    input_feature_schema = _input_feature_schema(config)
    coord_policy = _coord_policy(config)
    extent_feature_policy = _extent_feature_policy(config)
    command = [python_executable, _training_script_for_config(config)]
    if _requires_v4_training_wrapper(config):
        _append_option(command, "--normalization-profile", normalization_profile)
        _append_option(
            command,
            "--condition-feature-transform",
            condition_feature_transform,
        )
        _append_option(command, "--input-feature-schema", input_feature_schema)
        _append_option(command, "--coord-policy", coord_policy)
        _append_option(command, "--extent-feature-policy", extent_feature_policy)
    _append_option(command, "--subset", dataset.get("subset_path"))
    _append_option(command, "--split-map", _split_map_path_for_dataset(dataset))
    _append_option(command, "--dataset-loader", dataset.get("loader"))
    _append_option(command, "--dataset-manifest", dataset.get("manifest_path"))
    if dataset.get("boundary_mask_fallback") is True:
        command.append("--boundary-mask-fallback")
    elif dataset.get("boundary_mask_fallback") is False:
        command.append("--no-boundary-mask-fallback")
    _append_option(
        command,
        "--node-coordinate-encoding",
        graph.get("node_coordinate_encoding"),
    )
    _append_option(command, "--node-coordinate-freqs", graph.get("node_coordinate_freqs"))
    _append_option(command, "--epochs", run.get("epochs"))
    _append_option(command, "--node-latent-size", model.get("node_latent_size"))
    _append_option(command, "--edge-latent-size", model.get("edge_latent_size"))
    _append_option(command, "--processor-steps", model.get("processor_steps"))
    _append_option(command, "--mlp-hidden-layers", model.get("mlp_hidden_layers"))
    _append_option(command, "--p-edge-masking", model.get("p_edge_masking"))
    _append_option(command, "--edge-masking-scope", model.get("edge_masking_scope"))
    _append_option(command, "--decoder-bypass-mode", model.get("decoder_bypass_mode"))
    _append_option(command, "--decoder-bypass-features", model.get("decoder_bypass_features"))
    _append_option(
        command,
        "--decoder-bypass-feature-source",
        model.get("decoder_bypass_feature_source"),
    )
    _append_option(
        command,
        "--decoder-bypass-local-feature-names",
        _csv_names(model.get("decoder_bypass_local_feature_names")),
    )
    _append_option(
        command,
        "--decoder-bypass-output-space",
        model.get("decoder_bypass_output_space"),
    )
    _append_option(
        command,
        "--decoder-bypass-hidden-size",
        model.get("decoder_bypass_hidden_size"),
    )
    _append_option(command, "--decoder-bypass-layers", model.get("decoder_bypass_layers"))
    _append_option(command, "--decoder-bypass-init", model.get("decoder_bypass_init"))
    _append_option(
        command,
        "--decoder-bypass-residual-scale",
        model.get("decoder_bypass_residual_scale"),
    )
    _append_option(command, "--global-context-mode", model.get("global_context_mode"))
    _append_option(
        command,
        "--global-context-feature-names",
        _csv_names(model.get("global_context_feature_names")),
    )
    _append_option(command, "--film-target", model.get("film_target"))
    _append_option(command, "--film-init", model.get("film_init"))
    _append_option(command, "--film-hidden-size", model.get("film_hidden_size"))
    _append_option(command, "--native-output-mode", model.get("native_output_mode"))
    _append_option(command, "--native-branch-mode", model.get("native_branch_mode"))
    _append_option(command, "--scale-head-mode", model.get("scale_head_mode"))
    _append_option(command, "--scale-pooling", model.get("scale_pooling"))
    _append_option(command, "--scale-head-hidden-size", model.get("scale_head_hidden_size"))
    _append_option(command, "--scale-head-depth", model.get("scale_head_depth"))
    _append_option(command, "--shape-attention-mode", model.get("shape_attention_mode"))
    _append_option(command, "--scale-attention-mode", model.get("scale_attention_mode"))
    _append_option(
        command,
        "--regional-attention-hidden-size",
        model.get("regional_attention_hidden_size"),
    )
    _append_option(
        command,
        "--qk-region-feature-version",
        model.get("qk_region_feature_version"),
    )
    _append_option(command, "--scale-context-mode", model.get("scale_context_mode"))
    _append_option(
        command,
        "--scale-context-feature-names",
        _csv_names(model.get("scale_context_feature_names")),
    )
    _append_option(command, "--scale-deepsets-mode", model.get("scale_deepsets_mode"))
    _append_option(
        command,
        "--scale-deepsets-hidden-size",
        model.get("scale_deepsets_hidden_size"),
    )
    if model.get("pooled_latent_stop_gradient") is True:
        command.append("--pooled-latent-stop-gradient")
    elif model.get("pooled_latent_stop_gradient") is False:
        command.append("--no-pooled-latent-stop-gradient")
    _append_option(command, "--batch-size", run.get("batch_size"))
    _append_option(command, "--validation-batch-size", run.get("validation_batch_size"))
    _append_option(command, "--prediction-batch-size", run.get("prediction_batch_size"))
    _append_option(command, "--init-mode", run.get("init_mode"))
    _append_option(command, "--init-checkpoint", run.get("init_checkpoint"))
    _append_option(command, "--checkpoint-load-strict", run.get("checkpoint_load_strict"))
    _append_option(command, "--partial-load-policy", run.get("partial_load_policy"))
    final_probe_eval = run.get("final_probe_eval_after_training", True)
    command.append(
        "--final-probe-eval-after-training"
        if final_probe_eval is not False
        else "--no-final-probe-eval-after-training"
    )
    _append_option(command, "--final-probe-output-dir", run.get("final_probe_output_dir"))
    _append_option(command, "--final-probe-checkpoint-kind", run.get("final_probe_checkpoint_kind"))
    _append_option(command, "--final-probe-subset", run.get("final_probe_subset"))
    _append_option(command, "--final-probe-provenance", run.get("final_probe_provenance"))
    _append_option(command, "--final-probe-batch-size", run.get("final_probe_batch_size"))
    post_training_diagnostics = run.get("post_training_diagnostics", True)
    command.append(
        "--post-training-diagnostics"
        if post_training_diagnostics is not False
        else "--no-post-training-diagnostics"
    )
    _append_option(
        command,
        "--post-training-diagnostics-output-dir",
        run.get("post_training_diagnostics_output_dir"),
    )
    _append_option(command, "--batch-plan", run.get("batch_plan"))
    _append_option(command, "--batch-build-seed", run.get("batch_build_seed"))
    _append_option(command, "--sample-weight-policy", run.get("sample_weight_policy"))
    _append_option(command, "--sample-weight-json", run.get("sample_weight_json"))
    _append_option(command, "--sample-weight-default", run.get("sample_weight_default"))
    if run.get("sample_weight_normalize") is True:
        command.append("--sample-weight-normalize")
    if run.get("shuffle_train_batches") is True:
        command.append("--shuffle-train-batches")
    if run.get("epoch_wise_batch_regrouping") is True:
        command.append("--epoch-wise-batch-regrouping")
    if run.get("drop_last") is True:
        command.append("--drop-last")
    _append_option(command, "--optimizer", _runner_optimizer_name(optimizer.get("name")))
    _append_option(command, "--lr", optimizer.get("lr"))
    _append_option(
        command,
        "--scale-head-lr-multiplier",
        optimizer.get("scale_head_lr_multiplier"),
    )
    _append_option(
        command,
        "--native-trainable-scope",
        optimizer.get("native_trainable_scope"),
    )
    _append_option(command, "--lr-schedule", optimizer.get("lr_schedule"))
    _append_option(command, "--warmup-epochs", optimizer.get("warmup_epochs"))
    _append_option(command, "--min-lr", optimizer.get("min_lr"))
    _append_option(
        command, "--second-stage-epoch", optimizer.get("second_stage_epoch")
    )
    _append_option(command, "--second-stage-lr", optimizer.get("second_stage_lr"))
    _append_option(command, "--lr-init", optimizer.get("lr_init"))
    _append_option(command, "--lr-peak", optimizer.get("lr_peak"))
    _append_option(command, "--lr-base", optimizer.get("lr_base"))
    _append_option(command, "--lr-lowr", optimizer.get("lr_lowr"))
    _append_option(command, "--pct-start", optimizer.get("pct_start"))
    _append_option(command, "--pct-final", optimizer.get("pct_final"))
    _append_option(command, "--gradient-clip-norm", optimizer.get("gradient_clip_norm"))
    _append_option(command, "--weight-decay", optimizer.get("weight_decay"))
    _append_option(command, "--seed", optimizer.get("seed"))
    _append_option(command, "--model-seed", optimizer.get("model_seed"))
    _append_option(command, "--batch-order-seed", optimizer.get("batch_order_seed"))
    _append_option(command, "--graph-seed", optimizer.get("graph_seed"))
    _append_option(command, "--output-dir", export.get("output_dir"))
    _append_option(command, "--prediction-split", export.get("prediction_split"))
    _append_option(command, "--radius-policy", graph.get("radius_policy"))
    _append_option(command, "--coverage-repair-policy", graph.get("coverage_repair_policy"))
    if graph.get("repair_p2r") is True:
        command.append("--repair-p2r")
    elif graph.get("repair_p2r") is False:
        command.append("--no-repair-p2r")
    if graph.get("repair_r2p") is True:
        command.append("--repair-r2p")
    elif graph.get("repair_r2p") is False:
        command.append("--no-repair-r2p")
    _append_option(command, "--min-physical-coverage", graph.get("min_physical_coverage"))

    if export.get("save_final_predictions") is False:
        command.append("--no-save-predictions")
    elif export.get("save_final_predictions") is True:
        command.append("--save-predictions")
    if export.get("save_best_predictions") is False:
        command.append("--no-save-best-predictions")
    elif export.get("save_best_predictions") is True:
        command.append("--save-best-predictions")
    _append_option(command, "--best-predictions-name", export.get("best_predictions_name"))
    if export.get("save_point_global_best_checkpoint") is True:
        command.append("--save-point-global-best-checkpoint")
    _append_option(
        command,
        "--point-global-best-checkpoint-name",
        export.get("point_global_best_checkpoint_name"),
    )
    if export.get("save_base_mse_best_checkpoint") is True:
        command.append("--save-base-mse-best-checkpoint")
    _append_option(
        command,
        "--base-mse-best-checkpoint-name",
        export.get("base_mse_best_checkpoint_name"),
    )
    if export.get("save_sample_first_best_checkpoint") is True:
        command.append("--save-sample-first-best-checkpoint")
    _append_option(
        command,
        "--sample-first-best-checkpoint-name",
        export.get("sample_first_best_checkpoint_name"),
    )
    _append_option(command, "--report-every", run.get("report_every"))
    _append_option(command, "--train-metrics-schedule", run.get("train_metrics_schedule"))
    _append_option(command, "--grad-norm-report-every", run.get("grad_norm_report_every"))
    _append_option(command, "--log-mode", run.get("log_mode"))
    command.append("--progress-log" if run.get("progress_log", True) else "--no-progress-log")
    _append_option(command, "--progress-detail", run.get("progress_detail"))
    if run.get("profile_timing") is True:
        command.append("--profile-timing")
    _append_option(command, "--profile-timing-json", run.get("profile_timing_json"))
    _append_option(command, "--memory-audit-jsonl", run.get("memory_audit_jsonl"))
    if run.get("memory_audit_every_batch") is True:
        command.append("--memory-audit-every-batch")
    if run.get("memory_audit_gc") is True:
        command.append("--memory-audit-gc")
    _append_option(command, "--selection-metric", export.get("selection_metric"))

    _append_option(command, "--loss-mode", loss.get("mode"))
    _append_option(command, "--background-quantile", loss.get("background_quantile"))
    _append_option(command, "--hotspot-quantile", loss.get("hotspot_quantile"))
    _append_option(command, "--strong-q-quantile", loss.get("strong_q_quantile"))
    _append_option(command, "--background-weight", loss.get("background_weight"))
    _append_option(command, "--hotspot-weight", loss.get("hotspot_weight"))
    _append_option(command, "--strong-q-weight", loss.get("strong_q_weight"))
    _append_option(command, "--native-shape-cv-weight", loss.get("native_shape_cv_weight"))
    _append_option(command, "--native-log-scale-weight", loss.get("native_log_scale_weight"))
    _append_option(command, "--native-relative-field-weight", loss.get("native_relative_field_weight"))
    _append_option(command, "--native-raw-field-weight", loss.get("native_raw_field_weight"))
    _append_option(command, "--native-raw-loss-mode", loss.get("native_raw_loss_mode"))
    _append_option(
        command,
        "--native-log-scale-weight-mode",
        loss.get("native_log_scale_weight_mode"),
    )
    _append_option(
        command,
        "--native-log-scale-weight-clip-min",
        loss.get("native_log_scale_weight_clip_min"),
    )
    _append_option(
        command,
        "--native-log-scale-weight-clip-max",
        loss.get("native_log_scale_weight_clip_max"),
    )
    _append_option(command, "--background-l1-weight", loss.get("background_l1_weight"))
    _append_option(
        command, "--background-bias-weight", loss.get("background_bias_weight")
    )
    _append_option(command, "--background-over-weight", loss.get("background_over_weight"))
    _append_option(
        command,
        "--background-relative-weight",
        loss.get("background_relative_weight"),
    )
    _append_option(command, "--relative-floor", loss.get("relative_floor"))
    _append_option(command, "--relative-floor-mode", loss.get("relative_floor_mode"))
    _append_option(
        command, "--pseudo-negative-quantile", loss.get("pseudo_negative_quantile")
    )
    _append_option(
        command,
        "--pseudo-negative-delta-threshold",
        loss.get("pseudo_negative_delta_threshold"),
    )
    _append_option(command, "--pseudo-negative-weight", loss.get("pseudo_negative_weight"))
    _append_option(
        command,
        "--pseudo-negative-over-margin",
        loss.get("pseudo_negative_over_margin"),
    )
    _append_option(
        command, "--pseudo-negative-min-count", loss.get("pseudo_negative_min_count")
    )
    _append_option(
        command, "--pseudo-negative-loss-type", loss.get("pseudo_negative_loss_type")
    )
    _append_option(
        command,
        "--pseudo-negative-relative-floor",
        loss.get("pseudo_negative_relative_floor"),
    )
    _append_option(command, "--loss-weight-schedule", loss.get("weight_schedule"))
    _append_option(command, "--loss-transition-epoch", loss.get("transition_epoch"))
    return command


def build_baseline_comparison_command(
    config: Mapping[str, Any],
    *,
    prediction_label: str,
    predictions_path: str,
    python_executable: str = "python3",
) -> list[str]:
    """Build a v1 baseline comparison command for one prediction archive."""

    validate_v2_config(config)
    _validate_prediction_label(prediction_label)
    dataset = _section(config, "dataset")
    diagnostics = _section(config, "diagnostics")
    output_dir = _output_dir(config)
    return [
        python_executable,
        COMPARISON_SCRIPT,
        "--subset",
        _stringify(dataset.get("subset_path")),
        *_split_map_args_for_dataset(dataset),
        "--trained-predictions",
        predictions_path,
        "--output-json",
        _join_output(output_dir, f"baseline_comparison_{prediction_label}.json"),
        "--top-k",
        _stringify(diagnostics.get("top_k", 5)),
        "--stdout-mode",
        _stdout_mode(config),
    ]


def build_error_bins_command(
    config: Mapping[str, Any],
    *,
    prediction_label: str,
    predictions_path: str,
    python_executable: str = "python3",
) -> list[str]:
    """Build a v1 error-bin diagnostics command for one prediction archive."""

    validate_v2_config(config)
    _validate_prediction_label(prediction_label)
    dataset = _section(config, "dataset")
    diagnostics = _section(config, "diagnostics")
    output_dir = _output_dir(config)
    return [
        python_executable,
        ERROR_BINS_SCRIPT,
        "--subset",
        _stringify(dataset.get("subset_path")),
        *_split_map_args_for_dataset(dataset),
        "--trained-predictions",
        predictions_path,
        "--output-json",
        _join_output(output_dir, f"error_bins_{prediction_label}.json"),
        "--output-md",
        _join_output(output_dir, f"error_bins_{prediction_label}.md"),
        "--bins",
        _stringify(diagnostics.get("deltaT_bins", "p50,p75,p90,p95")),
        "--stdout-mode",
        _stdout_mode(config),
    ]


def build_run_summary_command(
    config: Mapping[str, Any],
    *,
    prediction_label: str,
    python_executable: str = "python3",
) -> list[str]:
    """Build a v1 run-summary command for one prediction label."""

    validate_v2_config(config)
    _validate_prediction_label(prediction_label)
    diagnostics = _section(config, "diagnostics")
    output_dir = _output_dir(config)
    command = [
        python_executable,
        RUN_SUMMARY_SCRIPT,
        "--run-dir",
        output_dir,
        "--loss-summary",
        _join_output(output_dir, "loss_summary.json"),
        "--baseline-comparison-json",
        _join_output(output_dir, f"baseline_comparison_{prediction_label}.json"),
        "--error-bins-json",
        _join_output(output_dir, f"error_bins_{prediction_label}.json"),
        "--prediction-label",
        prediction_label,
        "--output-json",
        _join_output(output_dir, f"run_analysis_{prediction_label}.json"),
        "--output-md",
        _join_output(output_dir, f"run_analysis_{prediction_label}.md"),
        "--stdout-mode",
        _stdout_mode(config),
    ]
    metric_set = diagnostics.get("metric_set")
    if metric_set:
        command.append("--metric-set")
        if isinstance(metric_set, str):
            command.append(metric_set)
        else:
            command.extend(_stringify(item) for item in metric_set)
    return command


def build_condition_diagnostics_command(
    config: Mapping[str, Any],
    *,
    prediction_label: str,
    predictions_path: str,
    python_executable: str = "python3",
) -> list[str]:
    """Build a v1 condition diagnostics command for one prediction archive."""

    validate_v2_config(config)
    _validate_prediction_label(prediction_label)
    dataset = _section(config, "dataset")
    diagnostics = _section(config, "diagnostics")
    output_dir = _output_dir(config)
    return [
        python_executable,
        CONDITION_DIAGNOSTICS_SCRIPT,
        "--subset",
        _stringify(dataset.get("subset_path")),
        *_split_map_args_for_dataset(dataset),
        "--trained-predictions",
        predictions_path,
        "--output-json",
        _join_output(output_dir, f"condition_diagnostics_{prediction_label}.json"),
        "--output-md",
        _join_output(output_dir, f"condition_diagnostics_{prediction_label}.md"),
        "--prediction-label",
        prediction_label,
        "--bins",
        _stringify(diagnostics.get("deltaT_bins", "p50,p75,p90,p95")),
        "--q-power-bins",
        _stringify(diagnostics.get("q_power_bins", "p33,p66")),
        "--stdout-mode",
        _stdout_mode(config),
    ]


def build_field_shape_diagnostics_command(
    config: Mapping[str, Any],
    *,
    prediction_label: str,
    predictions_path: str,
    python_executable: str = "python3",
) -> list[str]:
    """Build a v2 field-shape diagnostics command for one prediction archive."""

    validate_v2_config(config)
    _validate_prediction_label(prediction_label)
    dataset = _section(config, "dataset")
    diagnostics = _section(config, "diagnostics")
    output_dir = _output_dir(config)
    return [
        python_executable,
        FIELD_SHAPE_DIAGNOSTICS_SCRIPT,
        "--subset",
        _stringify(dataset.get("subset_path")),
        *_split_map_args_for_dataset(dataset),
        "--trained-predictions",
        predictions_path,
        "--prediction-label",
        prediction_label,
        "--output-json",
        _join_output(output_dir, f"field_shape_diagnostics_{prediction_label}.json"),
        "--output-md",
        _join_output(output_dir, f"field_shape_diagnostics_{prediction_label}.md"),
        "--top-k",
        _stringify(diagnostics.get("top_k", 5)),
        "--stdout-mode",
        _stdout_mode(config),
    ]


def build_v2_command_plan(
    config: Mapping[str, Any], *, python_executable: str = "python3"
) -> dict[str, Any]:
    """Build a complete dry-run command plan for a v2 smoke/controlled config."""

    validate_v2_config(config)
    role = config.get("config_role")
    if role == "baseline_reference":
        raise ValueError("baseline_reference configs do not map to runner commands")
    model = _section(config, "model")
    dataset = _section(config, "dataset")
    graph_section = config.get("graph")
    graph = graph_section if isinstance(graph_section, Mapping) else {}

    plan: dict[str, Any] = {
        "config_name": _config_name(config),
        "config_role": role,
        "training_command": build_training_command(
            config, python_executable=python_executable
        ),
        "normalization_profile": _normalization_profile(config),
        "input_feature_schema": _input_feature_schema(config),
        "coord_policy": _coord_policy(config),
        "extent_feature_policy": _extent_feature_policy(config),
        "condition_feature_transform": _condition_feature_transform(config),
        "split_map_path": _split_map_path_for_dataset(dataset),
        "training_script": _training_script_for_config(config),
        "node_coordinate_encoding": graph.get("node_coordinate_encoding", "raw"),
        "node_coordinate_freqs": graph.get("node_coordinate_freqs", 4),
        "decoder_bypass_mode": model.get("decoder_bypass_mode", "none"),
        "decoder_bypass_features": model.get("decoder_bypass_features", "none"),
        "decoder_bypass_feature_source": model.get(
            "decoder_bypass_feature_source", "normalized_c"
        ),
        "decoder_bypass_local_feature_names": list(
            model.get("decoder_bypass_local_feature_names") or ()
        ),
        "global_context_mode": model.get("global_context_mode", "none"),
        "global_context_feature_names": list(
            model.get("global_context_feature_names") or ()
        ),
        "film_target": model.get("film_target", "rnodes_processed"),
        "film_init": model.get("film_init", "identity"),
        "diagnostics_commands": [],
        "mapped_fields": _mapped_fields(config),
        "unmapped_fields": _unmapped_fields(config),
        "warnings": _warnings(config),
        "non_execution_note": NON_EXECUTION_NOTE,
    }

    diagnostics = _section(config, "diagnostics")
    prediction_paths = _prediction_paths(config)
    labels = _prediction_labels(config)
    for label in labels:
        predictions_path = prediction_paths[label]
        if diagnostics.get("run_baseline_comparison"):
            plan["diagnostics_commands"].append(
                _diagnostic_entry(
                    "baseline_comparison",
                    label,
                    build_baseline_comparison_command(
                        config,
                        prediction_label=label,
                        predictions_path=predictions_path,
                        python_executable=python_executable,
                    ),
                )
            )
        if diagnostics.get("run_error_bins"):
            plan["diagnostics_commands"].append(
                _diagnostic_entry(
                    "error_bins",
                    label,
                    build_error_bins_command(
                        config,
                        prediction_label=label,
                        predictions_path=predictions_path,
                        python_executable=python_executable,
                    ),
                )
            )
        if diagnostics.get("run_summary"):
            plan["diagnostics_commands"].append(
                _diagnostic_entry(
                    "run_summary",
                    label,
                    build_run_summary_command(
                        config,
                        prediction_label=label,
                        python_executable=python_executable,
                    ),
                )
            )
        if diagnostics.get("run_condition_diagnostics"):
            plan["diagnostics_commands"].append(
                _diagnostic_entry(
                    "condition_diagnostics",
                    label,
                    build_condition_diagnostics_command(
                        config,
                        prediction_label=label,
                        predictions_path=predictions_path,
                        python_executable=python_executable,
                    ),
                )
            )
        if _field_shape_enabled(diagnostics):
            plan["diagnostics_commands"].append(
                _diagnostic_entry(
                    "field_shape_diagnostics",
                    label,
                    build_field_shape_diagnostics_command(
                        config,
                        prediction_label=label,
                        predictions_path=predictions_path,
                        python_executable=python_executable,
                    ),
                )
            )

    return plan


def summarize_command_plan(plan: Mapping[str, Any]) -> str:
    """Render a compact human-readable summary of a dry-run command plan."""

    lines = [
        f"config: {plan.get('config_name')}",
        f"role: {plan.get('config_role')}",
        f"normalization_profile: {plan.get('normalization_profile')}",
        f"input_feature_schema: {plan.get('input_feature_schema')}",
        f"coord_policy: {plan.get('coord_policy')}",
        f"extent_feature_policy: {plan.get('extent_feature_policy')}",
        f"condition_feature_transform: {plan.get('condition_feature_transform')}",
        f"split_map_path: {plan.get('split_map_path')}",
        f"node_coordinate_encoding: {plan.get('node_coordinate_encoding')}",
        f"node_coordinate_freqs: {plan.get('node_coordinate_freqs')}",
        f"decoder_bypass_mode: {plan.get('decoder_bypass_mode')}",
        f"decoder_bypass_features: {plan.get('decoder_bypass_features')}",
        f"decoder_bypass_feature_source: {plan.get('decoder_bypass_feature_source')}",
        f"decoder_bypass_local_feature_names: {','.join(plan.get('decoder_bypass_local_feature_names', [])) or 'none'}",
        f"global_context_mode: {plan.get('global_context_mode')}",
        f"global_context_feature_count: {len(plan.get('global_context_feature_names', []))}",
        f"film_target: {plan.get('film_target')}",
        f"film_init: {plan.get('film_init')}",
        f"training_script: {plan.get('training_script')}",
        f"training: {shlex.join(plan['training_command'])}",
    ]
    diagnostics_commands = plan.get("diagnostics_commands", [])
    lines.append(f"diagnostics_commands: {len(diagnostics_commands)}")
    for entry in diagnostics_commands:
        lines.append(
            "  "
            f"{entry['prediction_label']}:{entry['kind']}: "
            f"{shlex.join(entry['command'])}"
        )
    lines.append(f"mapped_fields: {len(plan.get('mapped_fields', []))}")
    lines.append(f"unmapped_fields: {len(plan.get('unmapped_fields', []))}")
    if plan.get("warnings"):
        lines.append(f"warnings: {len(plan['warnings'])}")
    lines.append(str(plan.get("non_execution_note")))
    return "\n".join(lines)


def _diagnostic_entry(kind: str, prediction_label: str, command: list[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "prediction_label": prediction_label,
        "command": command,
    }


def _mapped_fields(config: Mapping[str, Any]) -> list[dict[str, str]]:
    mappings = [
        ("dataset.subset_path", "training --subset"),
        ("dataset.split_map_path", "training/diagnostics --split-map"),
        ("dataset.boundary_mask_fallback", "training --boundary-mask-fallback/--no-boundary-mask-fallback"),
        ("dataset.normalization_profile", "training script selection and optional --normalization-profile"),
        ("dataset.condition_feature_transform", "V4 training --condition-feature-transform"),
        ("dataset.input_feature_schema", "V4 training --input-feature-schema"),
        ("dataset.coord_policy", "V4 training --coord-policy"),
        ("dataset.extent_feature_policy", "V4 training --extent-feature-policy"),
        ("graph.node_coordinate_encoding", "training --node-coordinate-encoding"),
        ("graph.node_coordinate_freqs", "training --node-coordinate-freqs"),
        ("model.node_latent_size", "training --node-latent-size"),
        ("model.edge_latent_size", "training --edge-latent-size"),
        ("model.processor_steps", "training --processor-steps"),
        ("model.mlp_hidden_layers", "training --mlp-hidden-layers"),
        ("model.decoder_bypass_mode", "training --decoder-bypass-mode"),
        ("model.decoder_bypass_features", "training --decoder-bypass-features"),
        (
            "model.decoder_bypass_feature_source",
            "training --decoder-bypass-feature-source",
        ),
        (
            "model.decoder_bypass_local_feature_names",
            "training --decoder-bypass-local-feature-names",
        ),
        ("model.decoder_bypass_output_space", "training --decoder-bypass-output-space"),
        ("model.decoder_bypass_hidden_size", "training --decoder-bypass-hidden-size"),
        ("model.decoder_bypass_layers", "training --decoder-bypass-layers"),
        ("model.decoder_bypass_init", "training --decoder-bypass-init"),
        (
            "model.decoder_bypass_residual_scale",
            "training --decoder-bypass-residual-scale",
        ),
        ("model.global_context_mode", "training --global-context-mode"),
        (
            "model.global_context_feature_names",
            "training --global-context-feature-names",
        ),
        ("model.film_target", "training --film-target"),
        ("model.film_init", "training --film-init"),
        ("model.film_hidden_size", "training --film-hidden-size"),
        ("model.native_output_mode", "training --native-output-mode"),
        ("model.native_branch_mode", "training --native-branch-mode"),
        ("model.scale_head_mode", "training --scale-head-mode"),
        ("model.scale_pooling", "training --scale-pooling"),
        ("model.scale_head_hidden_size", "training --scale-head-hidden-size"),
        ("model.scale_head_depth", "training --scale-head-depth"),
        ("model.pooled_latent_stop_gradient", "training --pooled-latent-stop-gradient"),
        ("model.shape_attention_mode", "training --shape-attention-mode"),
        ("model.scale_attention_mode", "training --scale-attention-mode"),
        (
            "model.regional_attention_hidden_size",
            "training --regional-attention-hidden-size",
        ),
        (
            "model.qk_region_feature_version",
            "training --qk-region-feature-version",
        ),
        ("model.scale_context_mode", "training --scale-context-mode"),
        (
            "model.scale_context_feature_names",
            "training --scale-context-feature-names",
        ),
        ("model.scale_deepsets_mode", "training --scale-deepsets-mode"),
        (
            "model.scale_deepsets_hidden_size",
            "training --scale-deepsets-hidden-size",
        ),
        (
            "optimizer.native_trainable_scope",
            "training --native-trainable-scope",
        ),
        ("run.epochs", "training --epochs"),
        ("run.report_every", "training --report-every"),
        ("run.train_metrics_schedule", "training --train-metrics-schedule"),
        ("run.grad_norm_report_every", "training --grad-norm-report-every"),
        ("run.log_mode", "training --log-mode"),
        ("run.progress_log", "training --progress-log/--no-progress-log"),
        ("run.progress_detail", "training --progress-detail"),
        ("run.profile_timing", "training --profile-timing"),
        ("run.profile_timing_json", "training --profile-timing-json"),
        ("run.memory_audit_jsonl", "training --memory-audit-jsonl"),
        ("run.memory_audit_every_batch", "training --memory-audit-every-batch"),
        ("run.memory_audit_gc", "training --memory-audit-gc"),
        ("run.batch_size", "training --batch-size"),
        ("run.validation_batch_size", "training --validation-batch-size"),
        ("run.prediction_batch_size", "training --prediction-batch-size"),
        ("run.init_mode", "training --init-mode"),
        ("run.init_checkpoint", "training --init-checkpoint"),
        ("run.checkpoint_load_strict", "training --checkpoint-load-strict"),
        ("run.partial_load_policy", "training --partial-load-policy"),
        ("run.final_probe_eval_after_training", "training --final-probe-eval-after-training/--no-final-probe-eval-after-training"),
        ("run.final_probe_output_dir", "training --final-probe-output-dir"),
        ("run.final_probe_checkpoint_kind", "training --final-probe-checkpoint-kind"),
        ("run.final_probe_subset", "training --final-probe-subset"),
        ("run.final_probe_provenance", "training --final-probe-provenance"),
        ("run.final_probe_batch_size", "training --final-probe-batch-size"),
        ("run.post_training_diagnostics", "training --post-training-diagnostics/--no-post-training-diagnostics"),
        ("run.post_training_diagnostics_output_dir", "training --post-training-diagnostics-output-dir"),
        ("run.batch_plan", "training --batch-plan"),
        ("run.batch_build_seed", "training --batch-build-seed"),
        ("run.sample_weight_policy", "training --sample-weight-policy"),
        ("run.sample_weight_json", "training --sample-weight-json"),
        ("run.sample_weight_default", "training --sample-weight-default"),
        ("run.sample_weight_normalize", "training --sample-weight-normalize"),
        ("run.shuffle_train_batches", "training --shuffle-train-batches"),
        (
            "run.epoch_wise_batch_regrouping",
            "training --epoch-wise-batch-regrouping",
        ),
        ("run.drop_last", "training --drop-last"),
        ("optimizer.name", "training --optimizer"),
        ("optimizer.lr", "training --lr"),
        ("optimizer.lr_schedule", "training --lr-schedule"),
        ("optimizer.warmup_epochs", "training --warmup-epochs"),
        ("optimizer.min_lr", "training --min-lr"),
        ("optimizer.second_stage_epoch", "training --second-stage-epoch"),
        ("optimizer.second_stage_lr", "training --second-stage-lr"),
        ("optimizer.lr_init", "training --lr-init"),
        ("optimizer.lr_peak", "training --lr-peak"),
        ("optimizer.lr_base", "training --lr-base"),
        ("optimizer.lr_lowr", "training --lr-lowr"),
        ("optimizer.pct_start", "training --pct-start"),
        ("optimizer.pct_final", "training --pct-final"),
        ("optimizer.gradient_clip_norm", "training --gradient-clip-norm"),
        ("optimizer.weight_decay", "training --weight-decay"),
        ("optimizer.seed", "training --seed"),
        ("optimizer.model_seed", "training --model-seed"),
        ("optimizer.batch_order_seed", "training --batch-order-seed"),
        ("optimizer.graph_seed", "training --graph-seed"),
        ("loss.mode", "training --loss-mode"),
        ("loss.background_quantile", "training --background-quantile"),
        ("loss.hotspot_quantile", "training --hotspot-quantile"),
        ("loss.strong_q_quantile", "training --strong-q-quantile"),
        ("loss.background_weight", "training --background-weight"),
        ("loss.hotspot_weight", "training --hotspot-weight"),
        ("loss.strong_q_weight", "training --strong-q-weight"),
        ("loss.native_shape_cv_weight", "training --native-shape-cv-weight"),
        ("loss.native_log_scale_weight", "training --native-log-scale-weight"),
        (
            "loss.native_relative_field_weight",
            "training --native-relative-field-weight",
        ),
        ("loss.native_raw_field_weight", "training --native-raw-field-weight"),
        ("loss.native_raw_loss_mode", "training --native-raw-loss-mode"),
        (
            "loss.native_log_scale_weight_mode",
            "training --native-log-scale-weight-mode",
        ),
        (
            "loss.native_log_scale_weight_clip_min",
            "training --native-log-scale-weight-clip-min",
        ),
        (
            "loss.native_log_scale_weight_clip_max",
            "training --native-log-scale-weight-clip-max",
        ),
        ("loss.background_l1_weight", "training --background-l1-weight"),
        ("loss.background_bias_weight", "training --background-bias-weight"),
        ("loss.background_over_weight", "training --background-over-weight"),
        ("loss.background_relative_weight", "training --background-relative-weight"),
        ("loss.relative_floor", "training --relative-floor"),
        ("loss.relative_floor_mode", "training --relative-floor-mode"),
        ("loss.pseudo_negative_quantile", "training --pseudo-negative-quantile"),
        (
            "loss.pseudo_negative_delta_threshold",
            "training --pseudo-negative-delta-threshold",
        ),
        ("loss.pseudo_negative_weight", "training --pseudo-negative-weight"),
        (
            "loss.pseudo_negative_over_margin",
            "training --pseudo-negative-over-margin",
        ),
        ("loss.pseudo_negative_min_count", "training --pseudo-negative-min-count"),
        ("loss.pseudo_negative_loss_type", "training --pseudo-negative-loss-type"),
        (
            "loss.pseudo_negative_relative_floor",
            "training --pseudo-negative-relative-floor",
        ),
        ("loss.weight_schedule", "training --loss-weight-schedule"),
        ("loss.transition_epoch", "training --loss-transition-epoch"),
        ("export.output_dir", "training --output-dir"),
        ("export.prediction_split", "training --prediction-split"),
        ("export.save_final_predictions", "training --save-predictions"),
        ("export.save_best_predictions", "training --save-best-predictions"),
        ("export.best_predictions_name", "training --best-predictions-name"),
        ("export.selection_metric", "training --selection-metric"),
        ("graph.radius_policy", "training --radius-policy"),
        ("graph.coverage_repair_policy", "training --coverage-repair-policy"),
        ("graph.repair_p2r", "training --repair-p2r/--no-repair-p2r"),
        ("graph.repair_r2p", "training --repair-r2p/--no-repair-r2p"),
        ("graph.min_physical_coverage", "training --min-physical-coverage"),
        ("diagnostics.top_k", "comparison --top-k"),
        ("diagnostics.deltaT_bins", "error bins/condition diagnostics --bins"),
        (
            "diagnostics.q_power_bins",
            "condition diagnostics --q-power-bins",
        ),
        (
            "diagnostics.field_shape_metrics",
            "field-shape diagnostics command group",
        ),
        ("diagnostics.prediction_labels", "final/best diagnostics command groups"),
        ("diagnostics.metric_set", "run summary --metric-set"),
    ]
    return [
        {"field": field, "target": target}
        for field, target in mappings
        if _get_dotted(config, field) is not None
    ]


def _unmapped_fields(config: Mapping[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for field in (
        "model.report_parameter_count",
        "model.report_memory_estimate",
        "optimizer.multi_seed",
        "run.micro_batch_size",
        "diagnostics.p_quantiles",
        "baseline_reference.path",
        "dataset.k_encoding_mode",
        "dataset.sample_limit",
        "run.device_policy",
        "export.save_run_config",
        "export.save_loss_summary",
    ):
        value = _get_dotted(config, field)
        if value is not None:
            entries.append({"field": field, "reason": _unmapped_reason(field)})

    final_name = _get_dotted(config, "export.final_predictions_name")
    if final_name is not None and final_name != "predictions.npz":
        entries.append(
            {
                "field": "export.final_predictions_name",
                "reason": "v1 runner has no CLI flag for the final prediction archive name.",
            }
        )

    return entries


def _warnings(config: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _get_dotted(config, "run.micro_batch_size") is not None:
        warnings.append(
            "run.micro_batch_size is a future gradient-accumulation field and "
            "is not passed to the current v1 runner command."
        )
    if _get_dotted(config, "baseline_reference.path") is not None:
        warnings.append(
            "baseline_reference.path is checked by config validation only; it "
            "is not passed to the v1 runner."
        )

    return warnings


def _field_shape_enabled(diagnostics: Mapping[str, Any]) -> bool:
    if "run_field_shape_diagnostics" in diagnostics:
        return bool(diagnostics.get("run_field_shape_diagnostics"))
    return "field_shape_metrics" in diagnostics


def _unmapped_reason(field: str) -> str:
    if field.startswith("model."):
        return "model reporting field is not a runner CLI parameter."
    if field == "optimizer.multi_seed":
        return "multi-seed execution is outside this dry-run command builder."
    if field == "run.micro_batch_size":
        return "future micro-batch gradient accumulation field; not passed to current runner CLI."
    if field.startswith("diagnostics."):
        return "draft v2 diagnostics field is not implemented by current v1 scripts."
    if field == "baseline_reference.path":
        return "used for reference-path validation/explanation only."
    if field == "dataset.k_encoding_mode":
        return "currently implicit in the v1 dataset loader; not passed through CLI."
    if field == "dataset.sample_limit":
        return "current v1 runner has no sample-limit CLI."
    if field == "run.device_policy":
        return "local/SSH policy field; not a runner CLI parameter."
    if field in {"export.save_run_config", "export.save_loss_summary"}:
        return "v1 runner writes these files implicitly."
    return "not mapped to current v1 CLI."


def _normalization_profile(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    profile = dataset.get("normalization_profile") or NORMALIZATION_PROFILE_LEGACY_ZSCORE
    if profile not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"dataset.normalization_profile must be one of {sorted(NORMALIZATION_PROFILES)}, "
            f"got {profile!r}"
        )
    return str(profile)


def _condition_feature_transform(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    profile = _normalization_profile(config)
    default = (
        CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL
        if profile == NORMALIZATION_PROFILE_SEMANTIC_V1
        else CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE
    )
    transform = dataset.get("condition_feature_transform") or default
    if transform not in CONDITION_FEATURE_TRANSFORMS:
        raise ValueError(
            "dataset.condition_feature_transform must be one of "
            f"{sorted(CONDITION_FEATURE_TRANSFORMS)}, got {transform!r}"
        )
    if (
        profile == NORMALIZATION_PROFILE_LEGACY_ZSCORE
        and transform != CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE
    ):
        raise ValueError(
            "legacy_zscore requires dataset.condition_feature_transform="
            f"{CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE!r}"
        )
    if (
        profile == NORMALIZATION_PROFILE_SEMANTIC_V1
        and transform == CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE
    ):
        raise ValueError(
            "semantic_normalization_v1 requires a semantic "
            f"condition_feature_transform, got {transform!r}"
        )
    return str(transform)


def _input_feature_schema(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    value = dataset.get("input_feature_schema") or INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS
    if value not in INPUT_FEATURE_SCHEMAS:
        raise ValueError(
            f"dataset.input_feature_schema must be one of {sorted(INPUT_FEATURE_SCHEMAS)}, "
            f"got {value!r}"
        )
    return str(value)


def _coord_policy(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    value = dataset.get("coord_policy") or COORD_POLICY_TRAIN_MINMAX_UNIT_BOX
    if value not in COORD_POLICIES:
        raise ValueError(
            f"dataset.coord_policy must be one of {sorted(COORD_POLICIES)}, got {value!r}"
        )
    return str(value)


def _extent_feature_policy(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    value = dataset.get("extent_feature_policy") or EXTENT_FEATURE_POLICY_NONE
    if value not in EXTENT_FEATURE_POLICIES:
        raise ValueError(
            "dataset.extent_feature_policy must be one of "
            f"{sorted(EXTENT_FEATURE_POLICIES)}, got {value!r}"
        )
    return str(value)


def _requires_v4_training_wrapper(config: Mapping[str, Any]) -> bool:
    return (
        _normalization_profile(config) == NORMALIZATION_PROFILE_SEMANTIC_V1
        or _input_feature_schema(config) != INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS
        or _coord_policy(config) != COORD_POLICY_TRAIN_MINMAX_UNIT_BOX
        or _extent_feature_policy(config) != EXTENT_FEATURE_POLICY_NONE
    )


def _training_script_for_config(config: Mapping[str, Any]) -> str:
    if _requires_v4_training_wrapper(config):
        return V4_TRAINING_SCRIPT
    return TRAINING_SCRIPT


def _training_script_for_profile(normalization_profile: str) -> str:
    if normalization_profile == NORMALIZATION_PROFILE_SEMANTIC_V1:
        return V4_TRAINING_SCRIPT
    return TRAINING_SCRIPT


def _prediction_labels(config: Mapping[str, Any]) -> list[str]:
    diagnostics = _section(config, "diagnostics")
    raw_labels = diagnostics.get("prediction_labels") or []
    if isinstance(raw_labels, str):
        labels = [raw_labels]
    else:
        labels = list(raw_labels)
    for label in labels:
        _validate_prediction_label(label)
    return labels


def _prediction_paths(config: Mapping[str, Any]) -> dict[str, str]:
    export = _section(config, "export")
    output_dir = _output_dir(config)
    return {
        "final": _join_output(output_dir, export.get("final_predictions_name") or "predictions.npz"),
        "best": _join_output(output_dir, export.get("best_predictions_name") or "best_predictions.npz"),
    }


def _validate_prediction_label(prediction_label: str) -> None:
    if prediction_label not in {"final", "best"}:
        raise ValueError(
            f"prediction_label must be 'final' or 'best', got {prediction_label!r}"
        )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config field {name!r} must be a mapping")
    return section


def _output_dir(config: Mapping[str, Any]) -> str:
    output_dir = _section(config, "export").get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("config field 'export.output_dir' must be a non-empty string")
    return output_dir


def _stdout_mode(config: Mapping[str, Any]) -> str:
    value = _section(config, "run").get("log_mode", "compact")
    if value in {"compact", "full", "quiet"}:
        return str(value)
    return "compact"


def _config_name(config: Mapping[str, Any]) -> str:
    export = config.get("export")
    if isinstance(export, Mapping) and export.get("run_name"):
        return str(export["run_name"])
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping) and dataset.get("name"):
        return str(dataset["name"])
    return str(config.get("config_role", "unknown_config"))


def _csv_names(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise ValueError(f"feature-name configuration must be a string or sequence, got {value!r}")


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, _stringify(value)])


def _runner_optimizer_name(value: Any) -> str | None:
    if value is None:
        return None
    if value == LEGACY_OPTIMIZER_NAME:
        return "manual_gd"
    return _stringify(value)


def _split_map_path_for_dataset(dataset: Mapping[str, Any]) -> Any:
    split_map_path = dataset.get("split_map_path")
    if split_map_path:
        return split_map_path
    if _is_medium1024_gapA_dataset(dataset):
        return DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP
    return None


def _split_map_args_for_dataset(dataset: Mapping[str, Any]) -> list[str]:
    split_map_path = _split_map_path_for_dataset(dataset)
    if not split_map_path:
        return []
    return ["--split-map", _stringify(split_map_path)]


def _is_medium1024_gapA_dataset(dataset: Mapping[str, Any]) -> bool:
    dataset_name = str(dataset.get("name") or "")
    subset_path = str(dataset.get("subset_path") or "")
    return (
        dataset_name == "medium1024_gapA_full1024_v2"
        or "medium1024_gapA_full1024_v2" in subset_path
    )


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _join_output(output_dir: str, filename: str) -> str:
    return str(PurePosixPath(output_dir) / filename)


def _get_dotted(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value
