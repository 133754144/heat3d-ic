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
        "optimizer_name": optimizer.get("name"),
        "optimizer_lr": optimizer.get("lr"),
        "optimizer_seed": optimizer.get("seed"),
        "model_seed": optimizer.get("model_seed"),
        "batch_order_seed": optimizer.get("batch_order_seed"),
        "graph_seed": optimizer.get("graph_seed"),
        "loss_mode": loss.get("mode"),
        "run_mode": run.get("mode"),
        "run_epochs": run.get("epochs"),
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
    _validate_optimizer_seed_fields(_required_mapping(config, "optimizer", label), label)
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
