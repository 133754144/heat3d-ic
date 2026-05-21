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

    command = [python_executable, TRAINING_SCRIPT]
    _append_option(command, "--subset", dataset.get("subset_path"))
    _append_option(command, "--epochs", run.get("epochs"))
    _append_option(command, "--node-latent-size", model.get("node_latent_size"))
    _append_option(command, "--edge-latent-size", model.get("edge_latent_size"))
    _append_option(command, "--processor-steps", model.get("processor_steps"))
    _append_option(command, "--mlp-hidden-layers", model.get("mlp_hidden_layers"))
    _append_option(command, "--batch-size", run.get("batch_size"))
    _append_option(command, "--validation-batch-size", run.get("validation_batch_size"))
    _append_option(command, "--prediction-batch-size", run.get("prediction_batch_size"))
    if run.get("shuffle_train_batches") is True:
        command.append("--shuffle-train-batches")
    if run.get("drop_last") is True:
        command.append("--drop-last")
    _append_option(command, "--optimizer", _runner_optimizer_name(optimizer.get("name")))
    _append_option(command, "--lr", optimizer.get("lr"))
    _append_option(command, "--lr-schedule", optimizer.get("lr_schedule"))
    _append_option(command, "--warmup-epochs", optimizer.get("warmup_epochs"))
    _append_option(command, "--min-lr", optimizer.get("min_lr"))
    _append_option(
        command, "--second-stage-epoch", optimizer.get("second_stage_epoch")
    )
    _append_option(command, "--second-stage-lr", optimizer.get("second_stage_lr"))
    _append_option(command, "--gradient-clip-norm", optimizer.get("gradient_clip_norm"))
    _append_option(command, "--weight-decay", optimizer.get("weight_decay"))
    _append_option(command, "--seed", optimizer.get("seed"))
    _append_option(command, "--output-dir", export.get("output_dir"))

    if export.get("save_final_predictions") is True:
        command.append("--save-predictions")
    if export.get("save_best_predictions") is True:
        command.append("--save-best-predictions")
    _append_option(command, "--best-predictions-name", export.get("best_predictions_name"))
    _append_option(command, "--report-every", run.get("report_every"))
    _append_option(command, "--log-mode", run.get("log_mode"))
    command.append("--progress-log" if run.get("progress_log", True) else "--no-progress-log")
    _append_option(command, "--progress-detail", run.get("progress_detail"))
    _append_option(command, "--selection-metric", export.get("selection_metric"))

    _append_option(command, "--loss-mode", loss.get("mode"))
    _append_option(command, "--background-quantile", loss.get("background_quantile"))
    _append_option(command, "--hotspot-quantile", loss.get("hotspot_quantile"))
    _append_option(command, "--background-weight", loss.get("background_weight"))
    _append_option(command, "--hotspot-weight", loss.get("hotspot_weight"))
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

    plan: dict[str, Any] = {
        "config_name": _config_name(config),
        "config_role": role,
        "training_command": build_training_command(
            config, python_executable=python_executable
        ),
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
        ("model.node_latent_size", "training --node-latent-size"),
        ("model.edge_latent_size", "training --edge-latent-size"),
        ("model.processor_steps", "training --processor-steps"),
        ("model.mlp_hidden_layers", "training --mlp-hidden-layers"),
        ("run.epochs", "training --epochs"),
        ("run.report_every", "training --report-every"),
        ("run.log_mode", "training --log-mode"),
        ("run.progress_log", "training --progress-log/--no-progress-log"),
        ("run.progress_detail", "training --progress-detail"),
        ("run.batch_size", "planned training --batch-size"),
        ("run.validation_batch_size", "planned training --validation-batch-size"),
        ("run.prediction_batch_size", "planned training --prediction-batch-size"),
        ("run.shuffle_train_batches", "planned training --shuffle-train-batches"),
        ("run.drop_last", "planned training --drop-last"),
        ("optimizer.name", "training --optimizer"),
        ("optimizer.lr", "training --lr"),
        ("optimizer.lr_schedule", "training --lr-schedule"),
        ("optimizer.warmup_epochs", "training --warmup-epochs"),
        ("optimizer.min_lr", "training --min-lr"),
        ("optimizer.second_stage_epoch", "training --second-stage-epoch"),
        ("optimizer.second_stage_lr", "training --second-stage-lr"),
        ("optimizer.gradient_clip_norm", "training --gradient-clip-norm"),
        ("optimizer.weight_decay", "training --weight-decay"),
        ("optimizer.seed", "training --seed"),
        ("loss.mode", "training --loss-mode"),
        ("loss.background_quantile", "training --background-quantile"),
        ("loss.hotspot_quantile", "training --hotspot-quantile"),
        ("loss.background_weight", "training --background-weight"),
        ("loss.hotspot_weight", "training --hotspot-weight"),
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
        ("export.save_final_predictions", "training --save-predictions"),
        ("export.save_best_predictions", "training --save-best-predictions"),
        ("export.best_predictions_name", "training --best-predictions-name"),
        ("export.selection_metric", "training --selection-metric"),
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
    if _has_any_batch_cli_field(config):
        warnings.append(
            "batch CLI is dry-run only until runner implements it; generated "
            "batch flags are planned command-interface fields and must not be "
            "executed against the current runner."
        )
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


def _has_any_batch_cli_field(config: Mapping[str, Any]) -> bool:
    for field in (
        "run.batch_size",
        "run.validation_batch_size",
        "run.prediction_batch_size",
        "run.shuffle_train_batches",
        "run.drop_last",
    ):
        if _get_dotted(config, field) is not None:
            return True
    return False


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
