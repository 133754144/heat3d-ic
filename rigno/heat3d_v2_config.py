"""Read-only Heat3D v2 config loading and validation helpers.

This module intentionally stays outside the training stack. It reads YAML
configs, validates the draft schema, and returns compact summaries without
importing JAX, Flax, Optax, runner code, model code, or dataset loaders.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without PyYAML.
    raise ImportError("PyYAML is required to load Heat3D v2 configs.") from exc


CONFIG_SCHEMA_VERSION = "heat3d_v2_config_draft_v0"
REFERENCE_SCHEMA_VERSION = "heat3d_v2_reference_draft_v0"
CONFIG_ROLES = {"smoke", "controlled", "baseline_reference"}
RUN_CONFIG_REQUIRED_SECTIONS = (
    "dataset",
    "model",
    "optimizer",
    "loss",
    "run",
    "export",
    "diagnostics",
)
BATCH_SIZE_FIELDS = (
    "batch_size",
    "micro_batch_size",
    "validation_batch_size",
    "prediction_batch_size",
)
BATCH_BOOL_FIELDS = ("shuffle_train_batches", "drop_last")
TRAIN_METRICS_SCHEDULES = {"every_epoch", "half_and_final", "final_only", "none"}
PREDICTION_SPLITS = {"all", "train", "valid_iid", "valid_stress"}
RADIUS_POLICIES = {"legacy_kdtree_mean4", "discrete_physical_coverage"}
COVERAGE_REPAIR_POLICIES = {"none", "nearest_rnode"}
BATCH_PLANS = {"current_graph_shape", "sample_shuffle"}
NORMALIZATION_PROFILES = {"legacy_zscore", "semantic_normalization_v1"}
CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE = "legacy_zscore_all_condition_features"
CONDITION_FEATURE_TRANSFORMS = {
    CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE,
    "semantic_v1_logk_signedlog1p_q_binary_bcflags_independent_bc_scalars",
    "semantic_v1_bc_flags_binary_passthrough_only",
    "semantic_v1_q_signedlog1p_only",
    "semantic_v1_k_log_only",
}
DECODER_BYPASS_MODES = {"none", "post_decoder_residual"}
DECODER_BYPASS_FEATURES = {"none", "full_condition"}
DECODER_BYPASS_FEATURE_SOURCES = {"normalized_c"}
DECODER_BYPASS_INITS = {"zero_residual"}
INIT_MODES = {"real_first_batch", "upstream_dummy"}
PARTIAL_LOAD_POLICIES = {"matching", "skip_decoder", "encoder_processor_only"}
FINAL_PROBE_CHECKPOINT_KINDS = {"best", "final", "both"}
SAMPLE_WEIGHT_POLICIES = {"none", "hard_sample_list"}
LR_SCHEDULES = {
    "constant",
    "warmup_cosine",
    "rapid_decay",
    "two_stage",
    "second_stage",
    "upstream_onecycle",
}
LOSS_MODES = {
    "mse",
    "background_hotspot",
    "background_l1_bias",
    "background_l1_relative",
    "background_pseudo_negative",
    "hotspot_strong_q",
}

_MISSING = object()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk without applying training side effects."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"{config_path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{config_path}: failed to read YAML config: {exc}") from exc

    if loaded is None:
        raise ValueError(f"{config_path}: YAML config is empty")
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path}: YAML config must be a mapping")
    return loaded


def load_v2_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Heat3D v2 smoke, controlled, or reference config."""

    config = load_yaml_config(path)
    validate_v2_config(config, config_path=path)
    return config


def load_baseline_reference(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen baseline reference config."""

    config = load_yaml_config(path)
    validate_v2_config(config, config_path=path)
    role = config.get("config_role")
    if role != "baseline_reference":
        raise ValueError(
            f"{path}: field 'config_role' must be 'baseline_reference', got {role!r}"
        )
    return config


def validate_v2_config(
    config: Mapping[str, Any], *, config_path: str | Path | None = None
) -> Mapping[str, Any]:
    """Validate the current Heat3D v2 draft config schema."""

    label = _config_label(config_path)
    if not isinstance(config, Mapping):
        raise ValueError(f"{label}: config must be a mapping")

    schema_version = _required_field(config, "schema_version", label)
    role = _required_field(config, "config_role", label)
    if role not in CONFIG_ROLES:
        allowed = ", ".join(sorted(CONFIG_ROLES))
        raise ValueError(
            f"{label}: invalid field 'config_role'={role!r}; expected one of {allowed}"
        )

    _validate_schema_version(schema_version, role, label)

    if role in {"smoke", "controlled"}:
        _validate_run_config(config, role=role, config_path=config_path, label=label)
    else:
        _validate_baseline_reference(config, label=label)

    return config


def summarize_v2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, side-effect-free summary for logs and smoke checks."""

    dataset = _mapping_or_empty(config.get("dataset"))
    model = _mapping_or_empty(config.get("model"))
    optimizer = _mapping_or_empty(config.get("optimizer"))
    loss = _mapping_or_empty(config.get("loss"))
    run = _mapping_or_empty(config.get("run"))
    export = _mapping_or_empty(config.get("export"))
    diagnostics = _mapping_or_empty(config.get("diagnostics"))
    baseline_reference = _mapping_or_empty(config.get("baseline_reference"))
    training = _mapping_or_empty(config.get("training"))
    graph = _mapping_or_empty(config.get("graph"))

    summary: dict[str, Any] = {
        "config_role": config.get("config_role"),
        "dataset_name": dataset.get("name"),
        "model_architecture": model.get("architecture"),
        "model_node_latent_size": model.get("node_latent_size"),
        "model_edge_latent_size": model.get("edge_latent_size"),
        "model_processor_steps": model.get("processor_steps"),
        "model_p_edge_masking": model.get("p_edge_masking"),
        "optimizer_name": optimizer.get("name"),
        "optimizer_lr": optimizer.get("lr"),
        "optimizer_lr_schedule": optimizer.get("lr_schedule"),
        "optimizer_lr_init": optimizer.get("lr_init"),
        "optimizer_lr_peak": optimizer.get("lr_peak"),
        "optimizer_lr_base": optimizer.get("lr_base"),
        "optimizer_lr_lowr": optimizer.get("lr_lowr"),
        "optimizer_pct_start": optimizer.get("pct_start"),
        "optimizer_pct_final": optimizer.get("pct_final"),
        "optimizer_seed": optimizer.get("seed"),
        "model_seed": optimizer.get("model_seed"),
        "batch_order_seed": optimizer.get("batch_order_seed"),
        "graph_seed": optimizer.get("graph_seed"),
        "loss_mode": loss.get("mode"),
        "loss_hotspot_quantile": loss.get("hotspot_quantile"),
        "loss_hotspot_weight": loss.get("hotspot_weight"),
        "loss_strong_q_quantile": loss.get("strong_q_quantile"),
        "loss_strong_q_weight": loss.get("strong_q_weight"),
        "run_mode": run.get("mode"),
        "run_epochs": run.get("epochs"),
        "init_mode": run.get("init_mode"),
        "init_checkpoint": run.get("init_checkpoint"),
        "checkpoint_load_strict": run.get("checkpoint_load_strict"),
        "partial_load_policy": run.get("partial_load_policy"),
        "final_probe_eval_after_training": run.get("final_probe_eval_after_training"),
        "final_probe_checkpoint_kind": run.get("final_probe_checkpoint_kind"),
        "final_probe_output_dir": run.get("final_probe_output_dir"),
        "post_training_diagnostics": run.get("post_training_diagnostics"),
        "post_training_diagnostics_output_dir": run.get("post_training_diagnostics_output_dir"),
        "sample_weight_policy": run.get("sample_weight_policy"),
        "sample_weight_json": run.get("sample_weight_json"),
        "sample_weight_default": run.get("sample_weight_default"),
        "sample_weight_normalize": run.get("sample_weight_normalize"),
        "batch_plan": run.get("batch_plan"),
        "batch_build_seed": run.get("batch_build_seed"),
        "export_output_dir": export.get("output_dir"),
        "diagnostics_enabled": _summarize_diagnostics(diagnostics),
        "graph_radius_policy": graph.get("radius_policy"),
        "graph_coverage_repair_policy": graph.get("coverage_repair_policy"),
    }

    if config.get("config_role") == "baseline_reference":
        summary["optimizer_name"] = training.get("optimizer")
        summary["optimizer_lr"] = training.get("lr")
        summary["loss_mode"] = training.get("loss_mode")
        summary["run_epochs"] = training.get("epochs")
        summary["baseline_reference_name"] = _baseline_value(config, "name")
        summary["baseline_reference_best_epoch"] = _baseline_value(
            config, "best_epoch"
        )
    elif baseline_reference:
        summary["baseline_reference_path"] = baseline_reference.get("path")
        summary["baseline_reference_compare_against"] = baseline_reference.get(
            "compare_against"
        )

    return summary


def resolve_baseline_reference(
    config: Mapping[str, Any], *, base_dir: str | Path | None = None
) -> dict[str, Any] | None:
    """Load the baseline reference pointed to by a run config, if present."""

    if config.get("config_role") == "baseline_reference":
        validate_v2_config(config)
        return dict(config)

    baseline_reference = config.get("baseline_reference")
    if not isinstance(baseline_reference, Mapping):
        return None

    reference_path = baseline_reference.get("path")
    if reference_path in (None, ""):
        return None
    if not isinstance(reference_path, str):
        raise ValueError("field 'baseline_reference.path' must be a string")

    resolved = _resolve_path(reference_path, base_dir=base_dir)
    if not resolved.exists():
        raise ValueError(
            "field 'baseline_reference.path' points to a missing file: "
            f"{reference_path!r} resolved as {resolved}"
        )
    return load_baseline_reference(resolved)


def _validate_run_config(
    config: Mapping[str, Any],
    *,
    role: str,
    config_path: str | Path | None,
    label: str,
) -> None:
    for section in RUN_CONFIG_REQUIRED_SECTIONS:
        _required_mapping(config, section, label)

    run = _required_mapping(config, "run", label)
    run_mode = run.get("mode")
    if run_mode != role:
        raise ValueError(
            f"{label}: field 'run.mode' must be {role!r} for config_role {role!r}; "
            f"got {run_mode!r}"
        )

    if "allow_long_training_local" in run and run["allow_long_training_local"] is not False:
        raise ValueError(
            f"{label}: field 'run.allow_long_training_local' must be false"
        )
    _validate_batch_fields(run, label)
    model = _required_mapping(config, "model", label)
    _validate_model_fields(model, label)
    optimizer = _required_mapping(config, "optimizer", label)
    _validate_optimizer_seed_fields(optimizer, label)
    _validate_optimizer_schedule_fields(optimizer, label)
    loss = _required_mapping(config, "loss", label)
    _validate_loss_fields(loss, label)
    train_metrics_schedule = run.get("train_metrics_schedule")
    if train_metrics_schedule is not None and train_metrics_schedule not in TRAIN_METRICS_SCHEDULES:
        raise ValueError(
            f"{label}: field 'run.train_metrics_schedule' must be one of "
            f"{sorted(TRAIN_METRICS_SCHEDULES)}, got {train_metrics_schedule!r}"
        )
    grad_norm_report_every = run.get("grad_norm_report_every")
    if grad_norm_report_every is not None:
        if isinstance(grad_norm_report_every, bool) or not isinstance(grad_norm_report_every, int):
            raise ValueError(f"{label}: field 'run.grad_norm_report_every' must be an int or null")
        if grad_norm_report_every < 0:
            raise ValueError(f"{label}: field 'run.grad_norm_report_every' must be >= 0")

    export = _required_mapping(config, "export", label)
    dataset = _required_mapping(config, "dataset", label)
    graph = config.get("graph")
    if graph is not None:
        if not isinstance(graph, Mapping):
            raise ValueError(f"{label}: field 'graph' must be a mapping")
        _validate_graph_fields(graph, label)

    boundary_mask_fallback = dataset.get("boundary_mask_fallback")
    if boundary_mask_fallback is not None and not isinstance(boundary_mask_fallback, bool):
        raise ValueError(
            f"{label}: field 'dataset.boundary_mask_fallback' must be a bool or null"
        )
    normalization_profile = dataset.get("normalization_profile")
    if normalization_profile is not None and normalization_profile not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"{label}: field 'dataset.normalization_profile' must be one of "
            f"{sorted(NORMALIZATION_PROFILES)}, got {normalization_profile!r}"
        )
    condition_feature_transform = dataset.get("condition_feature_transform")
    if condition_feature_transform is not None:
        if condition_feature_transform not in CONDITION_FEATURE_TRANSFORMS:
            raise ValueError(
                f"{label}: field 'dataset.condition_feature_transform' must be "
                f"one of {sorted(CONDITION_FEATURE_TRANSFORMS)}, got "
                f"{condition_feature_transform!r}"
            )
        if (
            normalization_profile in {None, "legacy_zscore"}
            and condition_feature_transform != CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE
        ):
            raise ValueError(
                f"{label}: legacy_zscore requires "
                "dataset.condition_feature_transform="
                f"{CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE!r}"
            )
        if (
            normalization_profile == "semantic_normalization_v1"
            and condition_feature_transform == CONDITION_FEATURE_TRANSFORM_LEGACY_ZSCORE
        ):
            raise ValueError(
                f"{label}: semantic_normalization_v1 requires a semantic "
                "dataset.condition_feature_transform"
            )
    split_map_path = dataset.get("split_map_path")
    if split_map_path is not None:
        if not isinstance(split_map_path, str) or not split_map_path:
            raise ValueError(f"{label}: field 'dataset.split_map_path' must be a non-empty string or null")
        resolved = _resolve_path(split_map_path, config_path=config_path)
        if not resolved.exists():
            raise ValueError(
                f"{label}: field 'dataset.split_map_path' points to a missing file: "
                f"{split_map_path!r} resolved as {resolved}"
            )

    output_dir = export.get("output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, str):
            raise ValueError(f"{label}: field 'export.output_dir' must be a string")
        if not _is_output_relative_path(output_dir):
            raise ValueError(
                f"{label}: field 'export.output_dir' must be under output/, "
                f"got {output_dir!r}"
            )
    prediction_split = export.get("prediction_split")
    if prediction_split is not None and prediction_split not in PREDICTION_SPLITS:
        raise ValueError(
            f"{label}: field 'export.prediction_split' must be one of "
            f"{sorted(PREDICTION_SPLITS)}, got {prediction_split!r}"
        )

    if role == "controlled":
        baseline_reference = config.get("baseline_reference")
        if isinstance(baseline_reference, Mapping) and baseline_reference.get("path"):
            reference_path = baseline_reference["path"]
            if not isinstance(reference_path, str):
                raise ValueError(
                    f"{label}: field 'baseline_reference.path' must be a string"
                )
            resolved = _resolve_path(reference_path, config_path=config_path)
            if not resolved.exists():
                raise ValueError(
                    f"{label}: field 'baseline_reference.path' points to a "
                    f"missing file: {reference_path!r} resolved as {resolved}"
                )


def _validate_batch_fields(run: Mapping[str, Any], label: str) -> None:
    for field in BATCH_SIZE_FIELDS:
        if field not in run:
            continue
        value = run[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}: field 'run.{field}' must be an int or null")
        if value <= 0:
            raise ValueError(f"{label}: field 'run.{field}' must be a positive int or null")

    for field in BATCH_BOOL_FIELDS:
        if field in run and not isinstance(run[field], bool):
            raise ValueError(f"{label}: field 'run.{field}' must be a bool")

    batch_plan = run.get("batch_plan")
    if batch_plan is not None and batch_plan not in BATCH_PLANS:
        raise ValueError(
            f"{label}: field 'run.batch_plan' must be one of "
            f"{sorted(BATCH_PLANS)}, got {batch_plan!r}"
        )
    batch_build_seed = run.get("batch_build_seed")
    if batch_build_seed is not None:
        if isinstance(batch_build_seed, bool) or not isinstance(batch_build_seed, int):
            raise ValueError(f"{label}: field 'run.batch_build_seed' must be an int or null")
        if batch_build_seed < 0:
            raise ValueError(f"{label}: field 'run.batch_build_seed' must be >= 0")
    if batch_plan == "sample_shuffle":
        batch_size = run.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"{label}: field 'run.batch_size' must be a positive int when "
                "run.batch_plan is 'sample_shuffle'"
            )
    init_mode = run.get("init_mode")
    if init_mode is not None and init_mode not in INIT_MODES:
        raise ValueError(
            f"{label}: field 'run.init_mode' must be one of "
            f"{sorted(INIT_MODES)}, got {init_mode!r}"
        )
    init_checkpoint = run.get("init_checkpoint")
    if init_checkpoint is not None:
        if not isinstance(init_checkpoint, str) or not init_checkpoint:
            raise ValueError(
                f"{label}: field 'run.init_checkpoint' must be a non-empty string or null"
            )
    checkpoint_load_strict = run.get("checkpoint_load_strict")
    if checkpoint_load_strict is not None and not isinstance(checkpoint_load_strict, bool):
        raise ValueError(f"{label}: field 'run.checkpoint_load_strict' must be a bool or null")
    partial_load_policy = run.get("partial_load_policy")
    if partial_load_policy is not None and partial_load_policy not in PARTIAL_LOAD_POLICIES:
        raise ValueError(
            f"{label}: field 'run.partial_load_policy' must be one of "
            f"{sorted(PARTIAL_LOAD_POLICIES)}, got {partial_load_policy!r}"
        )
    final_probe_eval = run.get("final_probe_eval_after_training")
    if final_probe_eval is not None and not isinstance(final_probe_eval, bool):
        raise ValueError(f"{label}: field 'run.final_probe_eval_after_training' must be a bool or null")
    final_probe_kind = run.get("final_probe_checkpoint_kind")
    if final_probe_kind is not None and final_probe_kind not in FINAL_PROBE_CHECKPOINT_KINDS:
        raise ValueError(
            f"{label}: field 'run.final_probe_checkpoint_kind' must be one of "
            f"{sorted(FINAL_PROBE_CHECKPOINT_KINDS)}, got {final_probe_kind!r}"
        )
    final_probe_output_dir = run.get("final_probe_output_dir")
    if final_probe_output_dir is not None:
        if not isinstance(final_probe_output_dir, str):
            raise ValueError(f"{label}: field 'run.final_probe_output_dir' must be a string or null")
        if not _is_output_relative_path(final_probe_output_dir):
            raise ValueError(
                f"{label}: field 'run.final_probe_output_dir' must be under output/, "
                f"got {final_probe_output_dir!r}"
            )
    for field in ("final_probe_subset", "final_probe_provenance"):
        value = run.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{label}: field 'run.{field}' must be a non-empty string or null")
    final_probe_batch_size = run.get("final_probe_batch_size")
    if final_probe_batch_size is not None:
        if isinstance(final_probe_batch_size, bool) or not isinstance(final_probe_batch_size, int):
            raise ValueError(f"{label}: field 'run.final_probe_batch_size' must be an int or null")
        if final_probe_batch_size < 0:
            raise ValueError(f"{label}: field 'run.final_probe_batch_size' must be >= 0")
    post_training_diagnostics = run.get("post_training_diagnostics")
    if post_training_diagnostics is not None and not isinstance(post_training_diagnostics, bool):
        raise ValueError(f"{label}: field 'run.post_training_diagnostics' must be a bool or null")
    post_training_diagnostics_output_dir = run.get("post_training_diagnostics_output_dir")
    if post_training_diagnostics_output_dir is not None:
        if not isinstance(post_training_diagnostics_output_dir, str):
            raise ValueError(
                f"{label}: field 'run.post_training_diagnostics_output_dir' must be a string or null"
            )
        if not _is_output_relative_path(post_training_diagnostics_output_dir):
            raise ValueError(
                f"{label}: field 'run.post_training_diagnostics_output_dir' must be under output/, "
                f"got {post_training_diagnostics_output_dir!r}"
            )
    sample_weight_policy = run.get("sample_weight_policy")
    if sample_weight_policy is not None and sample_weight_policy not in SAMPLE_WEIGHT_POLICIES:
        raise ValueError(
            f"{label}: field 'run.sample_weight_policy' must be one of "
            f"{sorted(SAMPLE_WEIGHT_POLICIES)}, got {sample_weight_policy!r}"
        )
    sample_weight_json = run.get("sample_weight_json")
    if sample_weight_json is not None and (not isinstance(sample_weight_json, str) or not sample_weight_json):
        raise ValueError(f"{label}: field 'run.sample_weight_json' must be a non-empty string or null")
    sample_weight_default = run.get("sample_weight_default")
    if sample_weight_default is not None:
        if isinstance(sample_weight_default, bool) or not isinstance(sample_weight_default, (int, float)):
            raise ValueError(f"{label}: field 'run.sample_weight_default' must be numeric or null")
        if float(sample_weight_default) < 0.0:
            raise ValueError(f"{label}: field 'run.sample_weight_default' must be >= 0")
    sample_weight_normalize = run.get("sample_weight_normalize")
    if sample_weight_normalize is not None and not isinstance(sample_weight_normalize, bool):
        raise ValueError(f"{label}: field 'run.sample_weight_normalize' must be a bool or null")
    if sample_weight_policy == "hard_sample_list" and not sample_weight_json:
        raise ValueError(
            f"{label}: field 'run.sample_weight_json' is required when "
            "run.sample_weight_policy is 'hard_sample_list'"
        )


def _validate_optimizer_seed_fields(optimizer: Mapping[str, Any], label: str) -> None:
    for field in ("seed", "model_seed", "batch_order_seed", "graph_seed"):
        if field not in optimizer:
            continue
        value = optimizer[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}: field 'optimizer.{field}' must be an int or null")
        if value < 0:
            raise ValueError(f"{label}: field 'optimizer.{field}' must be >= 0")


def _validate_model_fields(model: Mapping[str, Any], label: str) -> None:
    p_edge_masking = model.get("p_edge_masking")
    if p_edge_masking is not None:
        if isinstance(p_edge_masking, bool) or not isinstance(p_edge_masking, (int, float)):
            raise ValueError(
                f"{label}: field 'model.p_edge_masking' must be numeric or null"
            )
        if float(p_edge_masking) < 0.0 or float(p_edge_masking) >= 1.0:
            raise ValueError(
                f"{label}: field 'model.p_edge_masking' must satisfy 0 <= value < 1"
            )
    mode = model.get("decoder_bypass_mode")
    features = model.get("decoder_bypass_features")
    source = model.get("decoder_bypass_feature_source")
    init = model.get("decoder_bypass_init")
    if mode is not None and mode not in DECODER_BYPASS_MODES:
        raise ValueError(
            f"{label}: field 'model.decoder_bypass_mode' must be one of "
            f"{sorted(DECODER_BYPASS_MODES)}, got {mode!r}"
        )
    if features is not None and features not in DECODER_BYPASS_FEATURES:
        raise ValueError(
            f"{label}: field 'model.decoder_bypass_features' must be one of "
            f"{sorted(DECODER_BYPASS_FEATURES)}, got {features!r}"
        )
    if source is not None and source not in DECODER_BYPASS_FEATURE_SOURCES:
        raise ValueError(
            f"{label}: field 'model.decoder_bypass_feature_source' must be one of "
            f"{sorted(DECODER_BYPASS_FEATURE_SOURCES)}, got {source!r}"
        )
    if init is not None and init not in DECODER_BYPASS_INITS:
        raise ValueError(
            f"{label}: field 'model.decoder_bypass_init' must be one of "
            f"{sorted(DECODER_BYPASS_INITS)}, got {init!r}"
        )
    for field in ("decoder_bypass_hidden_size", "decoder_bypass_layers"):
        value = model.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label}: field 'model.{field}' must be an int >= 1")
    residual_scale = model.get("decoder_bypass_residual_scale")
    if residual_scale is not None:
        if isinstance(residual_scale, bool) or not isinstance(residual_scale, (int, float)):
            raise ValueError(
                f"{label}: field 'model.decoder_bypass_residual_scale' must be numeric"
            )
        if float(residual_scale) < 0.0:
            raise ValueError(
                f"{label}: field 'model.decoder_bypass_residual_scale' must be >= 0"
            )
    if mode in {None, "none"}:
        if features not in {None, "none"}:
            raise ValueError(
                f"{label}: model.decoder_bypass_mode='none' requires "
                "model.decoder_bypass_features='none'"
            )
    elif features != "full_condition":
        raise ValueError(
            f"{label}: model.decoder_bypass_mode='post_decoder_residual' requires "
            "model.decoder_bypass_features='full_condition'"
        )


def _validate_loss_fields(loss: Mapping[str, Any], label: str) -> None:
    mode = loss.get("mode")
    if mode is not None and mode not in LOSS_MODES:
        raise ValueError(
            f"{label}: field 'loss.mode' must be one of {sorted(LOSS_MODES)}, "
            f"got {mode!r}"
        )

    for field in ("background_quantile", "hotspot_quantile", "strong_q_quantile"):
        if field not in loss or loss[field] is None:
            continue
        value = loss[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: field 'loss.{field}' must be numeric or null")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"{label}: field 'loss.{field}' must be in [0, 1]")

    weight_fields = (
        "background_weight",
        "hotspot_weight",
        "strong_q_weight",
        "background_l1_weight",
        "background_bias_weight",
        "background_over_weight",
        "background_relative_weight",
        "pseudo_negative_weight",
    )
    for field in weight_fields:
        if field not in loss or loss[field] is None:
            continue
        value = loss[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: field 'loss.{field}' must be numeric or null")
        if float(value) < 0.0:
            raise ValueError(f"{label}: field 'loss.{field}' must be >= 0")

    if mode == "hotspot_strong_q":
        required = (
            "hotspot_quantile",
            "hotspot_weight",
            "strong_q_quantile",
            "strong_q_weight",
        )
        missing = [field for field in required if loss.get(field) is None]
        if missing:
            raise ValueError(
                f"{label}: loss.mode='hotspot_strong_q' requires "
                f"{', '.join('loss.' + field for field in missing)}"
            )


def _validate_optimizer_schedule_fields(optimizer: Mapping[str, Any], label: str) -> None:
    lr_schedule = optimizer.get("lr_schedule")
    if lr_schedule is not None and lr_schedule not in LR_SCHEDULES:
        raise ValueError(
            f"{label}: field 'optimizer.lr_schedule' must be one of "
            f"{sorted(LR_SCHEDULES)}, got {lr_schedule!r}"
        )

    for field in (
        "lr",
        "min_lr",
        "second_stage_lr",
        "lr_init",
        "lr_peak",
        "lr_base",
        "lr_lowr",
        "weight_decay",
        "gradient_clip_norm",
    ):
        if field not in optimizer or optimizer[field] is None:
            continue
        value = optimizer[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: field 'optimizer.{field}' must be numeric or null")
        if float(value) < 0.0:
            raise ValueError(f"{label}: field 'optimizer.{field}' must be >= 0")

    for field in ("warmup_epochs", "second_stage_epoch"):
        if field not in optimizer or optimizer[field] is None:
            continue
        value = optimizer[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}: field 'optimizer.{field}' must be an int or null")
        if value < 0:
            raise ValueError(f"{label}: field 'optimizer.{field}' must be >= 0")

    for field in ("pct_start", "pct_final"):
        if field not in optimizer or optimizer[field] is None:
            continue
        value = optimizer[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: field 'optimizer.{field}' must be numeric or null")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"{label}: field 'optimizer.{field}' must be in [0, 1]")

    if lr_schedule == "upstream_onecycle":
        required = ("lr_init", "lr_peak", "lr_base", "lr_lowr", "pct_start", "pct_final")
        missing = [field for field in required if optimizer.get(field) is None]
        if missing:
            raise ValueError(
                f"{label}: optimizer.lr_schedule='upstream_onecycle' requires "
                f"{', '.join('optimizer.' + field for field in missing)}"
            )
        pct_start = float(optimizer["pct_start"])
        pct_final = float(optimizer["pct_final"])
        if pct_start <= 0.0:
            raise ValueError(f"{label}: field 'optimizer.pct_start' must be > 0")
        if pct_final >= 1.0:
            raise ValueError(f"{label}: field 'optimizer.pct_final' must be < 1")
        if pct_start + pct_final >= 1.0:
            raise ValueError(
                f"{label}: optimizer.pct_start + optimizer.pct_final must be < 1"
            )


def _validate_graph_fields(graph: Mapping[str, Any], label: str) -> None:
    radius_policy = graph.get("radius_policy")
    if radius_policy is not None and radius_policy not in RADIUS_POLICIES:
        raise ValueError(
            f"{label}: field 'graph.radius_policy' must be one of "
            f"{sorted(RADIUS_POLICIES)}, got {radius_policy!r}"
        )

    coverage_repair_policy = graph.get("coverage_repair_policy")
    if (
        coverage_repair_policy is not None
        and coverage_repair_policy not in COVERAGE_REPAIR_POLICIES
    ):
        raise ValueError(
            f"{label}: field 'graph.coverage_repair_policy' must be one of "
            f"{sorted(COVERAGE_REPAIR_POLICIES)}, got {coverage_repair_policy!r}"
        )

    for field in ("repair_p2r", "repair_r2p"):
        if field in graph and not isinstance(graph[field], bool):
            raise ValueError(f"{label}: field 'graph.{field}' must be a bool")

    min_physical_coverage = graph.get("min_physical_coverage")
    if min_physical_coverage is not None:
        if isinstance(min_physical_coverage, bool) or not isinstance(min_physical_coverage, int):
            raise ValueError(
                f"{label}: field 'graph.min_physical_coverage' must be an int or null"
            )
        if min_physical_coverage < 1:
            raise ValueError(
                f"{label}: field 'graph.min_physical_coverage' must be >= 1"
            )


def _validate_baseline_reference(config: Mapping[str, Any], *, label: str) -> None:
    baseline_root = config.get("baseline_reference")
    if baseline_root is not None and not isinstance(baseline_root, Mapping):
        raise ValueError(f"{label}: field 'baseline_reference' must be a mapping")

    name = _baseline_value(config, "name")
    if name in (_MISSING, None, ""):
        raise ValueError(
            f"{label}: missing required baseline reference field 'name'"
        )

    dataset = _baseline_value(config, "dataset")
    if dataset is _MISSING:
        raise ValueError(
            f"{label}: missing required baseline reference field 'dataset'"
        )
    if not isinstance(dataset, Mapping):
        raise ValueError(
            f"{label}: baseline reference field 'dataset' must be a mapping"
        )

    best_epoch = _baseline_value(config, "best_epoch")
    if best_epoch in (_MISSING, None):
        raise ValueError(
            f"{label}: missing required baseline reference field 'best_epoch'"
        )

    metrics = _baseline_value(config, "metrics")
    if metrics is _MISSING:
        raise ValueError(
            f"{label}: missing required baseline reference field 'metrics'"
        )
    if metrics is not None and not isinstance(metrics, Mapping):
        raise ValueError(
            f"{label}: baseline reference field 'metrics' must be a mapping or null"
        )


def _validate_schema_version(schema_version: Any, role: str, label: str) -> None:
    if role == "baseline_reference":
        allowed = {CONFIG_SCHEMA_VERSION, REFERENCE_SCHEMA_VERSION}
        if schema_version not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(
                f"{label}: field 'schema_version' must be one of {allowed_text} "
                f"for baseline_reference, got {schema_version!r}"
            )
        return

    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{label}: field 'schema_version' must be {CONFIG_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )


def _required_field(config: Mapping[str, Any], field: str, label: str) -> Any:
    if field not in config:
        raise ValueError(f"{label}: missing required field '{field}'")
    return config[field]


def _required_mapping(
    config: Mapping[str, Any], field: str, label: str
) -> Mapping[str, Any]:
    value = _required_field(config, field, label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: field '{field}' must be a mapping")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _baseline_value(config: Mapping[str, Any], field: str) -> Any:
    if field in config:
        return config[field]

    baseline_root = config.get("baseline_reference")
    if isinstance(baseline_root, Mapping) and field in baseline_root:
        return baseline_root[field]

    if field == "best_epoch":
        training = config.get("training")
        if isinstance(training, Mapping) and "best_epoch" in training:
            return training["best_epoch"]
        if isinstance(baseline_root, Mapping):
            nested_training = baseline_root.get("training")
            if isinstance(nested_training, Mapping) and "best_epoch" in nested_training:
                return nested_training["best_epoch"]

    return _MISSING


def _resolve_path(
    path_value: str,
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    candidates.append(Path.cwd() / path)
    if config_path is not None:
        config_parent = Path(config_path).parent
        candidates.append(config_parent / path)
        for ancestor in config_parent.parents:
            candidates.append(ancestor / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _is_output_relative_path(path_value: str) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return False
    return bool(path.parts) and path.parts[0] == "output" and ".." not in path.parts


def _summarize_diagnostics(diagnostics: Mapping[str, Any]) -> list[str]:
    enabled: list[str] = []
    for field in (
        "run_baseline_comparison",
        "run_error_bins",
        "run_condition_diagnostics",
        "run_summary",
    ):
        if diagnostics.get(field):
            enabled.append(field)

    has_enabled_diagnostic = bool(enabled)

    field_shape_metrics = diagnostics.get("field_shape_metrics")
    if field_shape_metrics:
        enabled.append(f"field_shape_metrics={list(field_shape_metrics)}")

    p_quantiles = diagnostics.get("p_quantiles")
    if p_quantiles and has_enabled_diagnostic:
        enabled.append(f"p_quantiles={list(p_quantiles)}")

    return enabled or ["none"]


def _config_label(config_path: str | Path | None) -> str:
    if config_path is None:
        return "<config>"
    return str(config_path)
