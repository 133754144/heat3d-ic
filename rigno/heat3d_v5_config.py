"""V5 clean-first YAML resolution and runner-command adaptation.

V5 plans intentionally sit outside the V4 registry.  This adapter resolves a
tracked V5 plan plus its inherited V4 base, turns one named ablation into an
actual runner configuration, and preserves V5's no-training guardrails.  It
does not import JAX, load data, or execute a command.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rigno.heat3d_v2_config import validate_v2_config
from rigno.heat3d_v2_runner_command import build_training_command
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES


V5_PLAN_PREFIX = "heat3d_v5_"
PREPARE_ONLY_ROLE = "prepare_only_no_training"
LOCAL_BYPASS_FEATURES = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
)


class V5ConfigError(ValueError):
    """Raised when a V5 plan cannot safely map to a runnable command."""


def load_v5_plan(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the source plan and fully inherited V4/V5 effective mapping."""

    config_path = Path(path).resolve()
    source = _load_yaml(config_path)
    schema = str(source.get("schema_version") or "")
    if not schema.startswith(V5_PLAN_PREFIX):
        raise V5ConfigError(f"{config_path}: expected a V5 schema, found {schema!r}")
    extends = source.get("extends")
    if not isinstance(extends, str) or not extends:
        raise V5ConfigError(f"{config_path}: V5 plan requires non-empty extends")
    base = _resolve_inherited_yaml((config_path.parent / extends).resolve(), stack=(config_path,))
    overlay = {key: value for key, value in source.items() if key != "extends"}
    return source, _deep_merge(base, overlay)


def build_v5_runner_plan(
    path: str | Path,
    *,
    variant: str,
    python_executable: str = "python",
) -> dict[str, Any]:
    """Build a non-executing legacy-runner command for one V5 ablation."""

    source, effective = load_v5_plan(path)
    matrix = effective.get("ablation_matrix")
    if not isinstance(matrix, Mapping) or variant not in matrix:
        available = sorted(str(name) for name in matrix) if isinstance(matrix, Mapping) else []
        raise V5ConfigError(f"unknown V5 ablation variant {variant!r}; available={available}")
    variant_config = matrix[variant]
    if not isinstance(variant_config, Mapping):
        raise V5ConfigError(f"V5 ablation variant {variant!r} must be a mapping")
    runner_config = _runner_config_for_variant(effective, variant_config)
    validate_v2_config(runner_config, config_path=path)
    command = build_training_command(runner_config, python_executable=python_executable)
    return {
        "schema_version": "heat3d_v5_runner_plan_v1",
        "source_plan": str(Path(path)),
        "source_plan_sha256": _sha256(Path(path)),
        "variant": variant,
        "training_allowed": bool(variant_config.get("training_allowed", False)),
        "run_config": runner_config,
        "command": command,
        "guardrails": {
            "fit_roles": list(_as_sequence(effective.get("dataset", {}).get("fit_roles"))),
            "selection_roles": list(_as_sequence(effective.get("dataset", {}).get("selection_roles"))),
            "report_only_roles": list(_as_sequence(effective.get("dataset", {}).get("report_only_roles"))),
            "hard_roles_must_not_select": bool(
                _mapping(effective.get("selection_and_reporting")).get("hard_roles_must_not_select", False)
            ),
            "no_git_pull_before_all_v5_tasks_complete": bool(
                _mapping(effective.get("guardrails")).get("no_git_pull_before_all_v5_tasks_complete", False)
            ),
        },
    }


def _runner_config_for_variant(
    effective: Mapping[str, Any],
    variant_config: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _mapping(effective.get("dataset"))
    if tuple(_as_sequence(dataset.get("fit_roles"))) != ("train",):
        raise V5ConfigError("V5 runner requires dataset.fit_roles: [train]")
    if tuple(_as_sequence(dataset.get("selection_roles"))) != ("valid_iid",):
        raise V5ConfigError("V5 runner requires dataset.selection_roles: [valid_iid]")
    model_plan = _mapping(effective.get("model"))
    bypass_plan = _mapping(model_plan.get("decoder_bypass"))
    runner = copy.deepcopy(dict(effective))
    runner["schema_version"] = "heat3d_v2_config_draft_v0"
    runner["config_role"] = "controlled"
    runner.pop("extends", None)
    runner.pop("frozen_reference", None)
    runner.pop("ablation_matrix", None)
    runner.pop("selection_and_reporting", None)
    runner.pop("guardrails", None)
    model = dict(_mapping(runner.get("model")))
    model.pop("backbone", None)
    model.pop("native_output_mode", None)
    model.pop("decoder_bypass", None)
    feature_mode = str(variant_config.get("decoder_bypass_feature_mode") or "")
    if feature_mode not in {"none", "full_condition", "explicit_local_condition"}:
        raise V5ConfigError(f"unsupported V5 decoder bypass mode: {feature_mode!r}")
    if feature_mode == "none":
        model["decoder_bypass_mode"] = "none"
        model["decoder_bypass_features"] = "none"
        model.pop("decoder_bypass_local_feature_names", None)
    else:
        model["decoder_bypass_mode"] = str(bypass_plan.get("mode") or "post_decoder_residual")
        model["decoder_bypass_features"] = feature_mode
        model["decoder_bypass_feature_source"] = str(
            bypass_plan.get("feature_source") or "normalized_c"
        )
        model["decoder_bypass_init"] = str(bypass_plan.get("initialization") or "zero_residual")
        model["decoder_bypass_residual_scale"] = float(bypass_plan.get("residual_scale", 1.0))
        if feature_mode == "explicit_local_condition":
            local_names = tuple(
                str(name)
                for name in _as_sequence(bypass_plan.get("retained_node_local_feature_names"))
            )
            if local_names != LOCAL_BYPASS_FEATURES:
                raise V5ConfigError(
                    "V5 local bypass schema drifted from frozen node-local audit allowlist"
                )
            model["decoder_bypass_local_feature_names"] = list(local_names)
        else:
            model.pop("decoder_bypass_local_feature_names", None)
    global_mode = str(variant_config.get("global_context_mode") or model_plan.get("global_context_mode") or "none")
    model["global_context_mode"] = global_mode
    if global_mode == "film":
        model["global_context_feature_names"] = list(GLOBAL_CONTEXT_FEATURES)
        model["global_context_feature_dim"] = len(GLOBAL_CONTEXT_FEATURES)
        model["film_target"] = "rnodes_processed"
        model["film_init"] = "identity"
        model["film_hidden_size"] = 64
    elif global_mode == "none":
        model["global_context_feature_names"] = []
        model["global_context_feature_dim"] = 0
        model["film_target"] = "rnodes_processed"
        model["film_init"] = "identity"
        model["film_hidden_size"] = 64
    else:
        raise V5ConfigError(f"unsupported V5 global_context_mode: {global_mode!r}")
    runner["model"] = model
    run = dict(_mapping(runner.get("run")))
    if global_mode == "film":
        # The V4 final-probe helper lacks a physical-context reconstruction path;
        # the dedicated V5 runner owns that future report surface.
        run["final_probe_eval_after_training"] = False
    runner["run"] = run
    return runner


def _resolve_inherited_yaml(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        raise V5ConfigError(f"inheritance cycle: {' -> '.join(str(item) for item in (*stack, path))}")
    payload = _load_yaml(path)
    schema = str(payload.get("schema_version") or "")
    if schema == "heat3d_v4_inherited_config_v0":
        extends = payload.get("extends")
        if not isinstance(extends, str) or not extends:
            raise V5ConfigError(f"{path}: V4 inherited config requires extends")
        base = _resolve_inherited_yaml((path.parent / extends).resolve(), stack=(*stack, path))
        overrides = payload.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            raise V5ConfigError(f"{path}: V4 inherited overrides must be a mapping")
        return _deep_merge(base, overrides)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V5ConfigError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise V5ConfigError(f"{path}: YAML root must be a mapping")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise V5ConfigError(f"expected a sequence, got {value!r}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
