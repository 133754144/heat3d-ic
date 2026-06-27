#!/usr/bin/env python3
"""Check the authoritative Heat3D V4 run registry.

The JSON registry is the only source of truth. It resolves one baseline plus
per-run overrides into CSV mirror rows and generated YAML.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue.
    raise SystemExit("PyYAML is required for V4 registry checks.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


REGISTRY_SCHEMA_VERSION = "heat3d_v4_run_registry_v0"
INHERITED_SCHEMA_VERSION = "heat3d_v4_inherited_config_v0"
METRICS_CONTRACT_SCHEMA_VERSION = "heat3d_v4_metrics_contract_v0"
DEFAULT_METRICS_PROFILE = "v4_metrics_v0"
DEFAULT_METRICS_CONTRACT = "configs/heat3d_v4/metrics_v0.json"
DEFAULT_SELECTION_METRIC = "valid_base_mse"
NORMALIZATION_PROFILE_LEGACY_ZSCORE = "legacy_zscore"
NORMALIZATION_PROFILE_SEMANTIC_V1 = "semantic_normalization_v1"
NORMALIZATION_PROFILES = {
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
}
RUNNER_FAMILY_LEGACY_V1 = "v1_controlled_legacy_runner"
RUNNER_FAMILY_V4_SEMANTIC = "v4_controlled_semantic_wrapper"
TARGET_MODE_NORMALIZED_DELTAT = "normalized_deltaT"
BRIDGE_POLICY_ZERO_DELTA_U = "zero_delta_u_bridge"
COORD_POLICY_TRAIN_MINMAX_UNIT_BOX = "train_minmax_to_unit_box"
NODE_COORDINATE_ENCODING_RAW = "raw"
NODE_COORDINATE_ENCODING_RAW_PLUS_FOURIER = "raw_plus_fourier"
NODE_COORDINATE_ENCODINGS = {
    NODE_COORDINATE_ENCODING_RAW,
    NODE_COORDINATE_ENCODING_RAW_PLUS_FOURIER,
}
CONDITION_TRANSFORM_LEGACY_ZSCORE = "legacy_zscore_all_condition_features"
CONDITION_TRANSFORM_SEMANTIC_V1 = (
    "semantic_v1_logk_signedlog1p_q_binary_bcflags_independent_bc_scalars"
)
CONDITION_TRANSFORM_SEMANTIC_BC_ONLY = (
    "semantic_v1_bc_flags_binary_passthrough_only"
)
CONDITION_TRANSFORM_SEMANTIC_Q_ONLY = "semantic_v1_q_signedlog1p_only"
CONDITION_TRANSFORM_SEMANTIC_K_ONLY = "semantic_v1_k_log_only"
CONDITION_TRANSFORMS = {
    CONDITION_TRANSFORM_LEGACY_ZSCORE,
    CONDITION_TRANSFORM_SEMANTIC_V1,
    CONDITION_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_TRANSFORM_SEMANTIC_Q_ONLY,
    CONDITION_TRANSFORM_SEMANTIC_K_ONLY,
}
SEMANTIC_CONDITION_TRANSFORMS = {
    CONDITION_TRANSFORM_SEMANTIC_V1,
    CONDITION_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_TRANSFORM_SEMANTIC_Q_ONLY,
    CONDITION_TRANSFORM_SEMANTIC_K_ONLY,
}
TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF = (
    "deltaT_norm_to_K_plus_T_ref"
)
FEATURE_MANIFEST_HASH_PLANNED = "planned"
DECODER_BYPASS_MODE_NONE = "none"
DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL = "post_decoder_residual"
DECODER_BYPASS_MODES = {
    DECODER_BYPASS_MODE_NONE,
    DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL,
}
DECODER_BYPASS_FEATURES_NONE = "none"
DECODER_BYPASS_FEATURES_FULL_CONDITION = "full_condition"
DECODER_BYPASS_FEATURES = {
    DECODER_BYPASS_FEATURES_NONE,
    DECODER_BYPASS_FEATURES_FULL_CONDITION,
}
DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C = "normalized_c"
DECODER_BYPASS_FEATURE_SOURCES = {DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C}
DECODER_BYPASS_INIT_ZERO_RESIDUAL = "zero_residual"
DECODER_BYPASS_INITS = {DECODER_BYPASS_INIT_ZERO_RESIDUAL}
DEFAULT_REGISTRY = Path("configs/heat3d_v4/v4_run_registry.json")
CONFIG_FIELDNAMES = (
    "config_id",
    "phase",
    "status",
    "base_yaml",
    "generated_yaml",
    "task",
    "split_map_path",
    "runner_family",
    "target_mode",
    "bridge_policy",
    "normalization_profile",
    "coord_policy",
    "condition_feature_transform",
    "node_coordinate_encoding",
    "node_coordinate_freqs",
    "decoder_bypass_mode",
    "decoder_bypass_features",
    "decoder_bypass_feature_source",
    "decoder_bypass_hidden_size",
    "decoder_bypass_layers",
    "decoder_bypass_init",
    "decoder_bypass_residual_scale",
    "target_recovery_policy",
    "feature_manifest_hash",
    "model_capacity",
    "node_latent_size",
    "edge_latent_size",
    "processor_steps",
    "mlp_hidden_layers",
    "batch_size",
    "validation_batch_size",
    "prediction_batch_size",
    "batch_plan",
    "optimizer",
    "lr",
    "model_seed",
    "batch_order_seed",
    "graph_seed",
    "seed",
    "multi_seed",
    "batch_build_seed",
    "lr_schedule",
    "warmup_epochs",
    "min_lr",
    "weight_decay",
    "epochs",
    "graph_radius_policy",
    "coverage_repair_policy",
    "loss_mode",
    "selection_metric",
    "metrics_profile",
    "metrics_contract",
    "output_dir",
    "run_name",
    "log_path",
    "final_probe_output_dir",
    "post_training_diagnostics_output_dir",
    "save_final_predictions",
    "save_best_predictions",
    "final_probe_eval_after_training",
    "post_training_diagnostics",
    "run_baseline_comparison",
    "run_error_bins",
    "run_condition_diagnostics",
    "run_summary_diagnostics",
    "run_field_shape_diagnostics",
    "dry_run_required",
    "launch_policy",
    "notes",
)
RESULT_FIELDNAMES = (
    "result_status",
    "result_source",
    "result_updated_at",
    "result_commit",
    "result_run_dir",
    "result_log_path",
    "result_loss_summary",
    "result_params_best",
    "result_params_final",
    "result_best_epoch",
    "result_best_valid_base_mse",
    "result_best_mse",
    "result_best_rmse",
    "result_best_mae",
    "result_final_valid_base_mse",
    "result_final_mse",
    "result_final_rmse",
    "result_final_mae",
    "result_best_raw_deltaT_mse",
    "result_best_raw_deltaT_rmse",
    "result_best_raw_deltaT_mae",
    "result_final_raw_deltaT_mse",
    "result_final_raw_deltaT_rmse",
    "result_final_raw_deltaT_mae",
    "result_best_valid_iid",
    "result_final_valid_iid",
    "result_best_stress",
    "result_final_stress",
    "result_corr_iid",
    "result_corr_stress",
    "result_amp",
    "result_amp_stress",
    "result_field_variance_iid",
    "result_field_variance_stress",
    "result_valid_iid_topk",
    "result_valid_stress_topk",
    "result_zrmse",
    "result_top5_rmse",
    "result_top10_rmse",
    "result_strong_q_rmse",
    "result_p95_abs",
    "result_p99_abs",
    "result_peak_abs",
    "result_peak_rel",
    "result_hotspot_mae",
    "result_bin0_bias",
    "result_bin0_over",
    "result_le005_bias",
    "result_le005_over",
    "result_final_probe_rmse",
    "result_final_probe_relrmse",
    "result_final_probe_tmax_error",
    "result_probe_p02_rmse",
    "result_probe_p03_rmse",
    "result_probe_p09_rmse",
    "result_final_probe_status",
    "result_post_training_diagnostics_status",
    "result_notes",
)
CSV_FIELDNAMES = CONFIG_FIELDNAMES + RESULT_FIELDNAMES
UNIQUE_RESOLVED_FIELDS = (
    "config_id",
    "generated_yaml",
    "output_dir",
    "run_name",
    "log_path",
    "final_probe_output_dir",
    "post_training_diagnostics_output_dir",
)
EXPECTED_V4_BASELINE = {
    "config_id": "V4_baseline",
    "phase": "v4",
    "status": "registered",
    "base_yaml": "configs/heat3d_v4/V4_base.yaml",
    "generated_yaml": "configs/heat3d_v4/generated/V4_baseline.yaml",
    "task": "coords+k(x)+q(x)+BC->T(x)",
    "split_map_path": "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json",
    "runner_family": RUNNER_FAMILY_LEGACY_V1,
    "target_mode": TARGET_MODE_NORMALIZED_DELTAT,
    "bridge_policy": BRIDGE_POLICY_ZERO_DELTA_U,
    "normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    "coord_policy": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    "condition_feature_transform": CONDITION_TRANSFORM_LEGACY_ZSCORE,
    "node_coordinate_encoding": NODE_COORDINATE_ENCODING_RAW,
    "node_coordinate_freqs": "4",
    "decoder_bypass_mode": DECODER_BYPASS_MODE_NONE,
    "decoder_bypass_features": DECODER_BYPASS_FEATURES_NONE,
    "decoder_bypass_feature_source": DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
    "decoder_bypass_hidden_size": "64",
    "decoder_bypass_layers": "2",
    "decoder_bypass_init": DECODER_BYPASS_INIT_ZERO_RESIDUAL,
    "decoder_bypass_residual_scale": "1.0",
    "target_recovery_policy": TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF,
    "feature_manifest_hash": FEATURE_MANIFEST_HASH_PLANNED,
    "model_capacity": "96/96/s6/m2",
    "node_latent_size": "96",
    "edge_latent_size": "96",
    "processor_steps": "6",
    "mlp_hidden_layers": "2",
    "batch_size": "88",
    "validation_batch_size": "88",
    "prediction_batch_size": "88",
    "batch_plan": "sample_shuffle",
    "optimizer": "adamw",
    "lr": "0.0005",
    "model_seed": "0",
    "batch_order_seed": "0",
    "graph_seed": "0",
    "seed": "0",
    "multi_seed": "[]",
    "batch_build_seed": "0",
    "lr_schedule": "warmup_cosine",
    "warmup_epochs": "10",
    "min_lr": "0.00005",
    "weight_decay": "0.0001",
    "epochs": "600",
    "graph_radius_policy": "discrete_physical_coverage",
    "coverage_repair_policy": "none",
    "loss_mode": "mse",
    "selection_metric": DEFAULT_SELECTION_METRIC,
    "metrics_profile": DEFAULT_METRICS_PROFILE,
    "metrics_contract": DEFAULT_METRICS_CONTRACT,
    "save_final_predictions": "true",
    "save_best_predictions": "true",
    "final_probe_eval_after_training": "true",
    "post_training_diagnostics": "true",
    "run_baseline_comparison": "true",
    "run_error_bins": "true",
    "run_condition_diagnostics": "true",
    "run_summary_diagnostics": "true",
    "run_field_shape_diagnostics": "true",
    "dry_run_required": "true",
    "launch_policy": "explicit_user_instruction_only",
}


def main() -> int:
    args = _parse_args()
    rows = check_registry(_repo_path(args.registry), emit_warnings=True)
    print(f"checked authoritative registry: {args.registry}")
    print(f"checked registry rows: {len(rows)}")
    for row in rows:
        print(f"  {row['config_id']}: {row['generated_yaml']}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    return parser.parse_args()


def check_registry(
    registry_path: Path | str = DEFAULT_REGISTRY, *, emit_warnings: bool = False
) -> list[dict[str, str]]:
    registry_path = _repo_path(registry_path)
    registry = load_registry(registry_path)
    rows = registry_rows(registry)
    _check_metrics_contracts(rows)
    _check_csv_mirror(registry, rows)
    for row in rows:
        _check_generated_yaml(row)
        if row["config_id"] == "V4_baseline":
            _check_v4_baseline(row)
    if not any(row["config_id"] == "V4_baseline" for row in rows):
        raise AssertionError("registry must contain V4_baseline")
    if emit_warnings:
        for warning in runner_control_warnings(rows):
            print(f"WARNING: {warning}")
    return rows


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    path = _repo_path(path)
    with path.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, dict):
        raise ValueError(f"{path}: registry must be a JSON object")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    if registry.get("registry_role") != "authoritative":
        raise ValueError(f"{path}: registry_role must be 'authoritative'")
    if not isinstance(registry.get("csv_mirror_path"), str):
        raise ValueError(f"{path}: csv_mirror_path must be a string")
    if not isinstance(registry.get("baseline"), dict):
        raise ValueError(f"{path}: baseline must be an object")
    if not isinstance(registry.get("runs"), dict) or not registry["runs"]:
        raise ValueError(f"{path}: runs must be a non-empty object")
    return registry


def registry_rows(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    baseline = _normalize_resolved_row(
        registry.get("baseline", {}), context="registry baseline"
    )
    if baseline["config_id"] != "V4_baseline":
        raise ValueError("registry baseline config_id must be 'V4_baseline'")
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("registry runs must be a mapping")
    for key, raw_run in runs.items():
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"registry run {key!r} must be a mapping")
        extra = sorted(set(raw_run) - {"config_id", "overrides"})
        if extra:
            raise ValueError(
                f"registry run {key!r} must contain only config_id and overrides; "
                f"unsupported fields: {', '.join(extra)}"
            )
        config_id = _stringify(raw_run.get("config_id", key))
        if key != config_id:
            raise ValueError(f"registry key {key!r} must match config_id {config_id!r}")
        overrides = raw_run.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"registry run {key!r} overrides must be a mapping")
        row = _resolve_registry_row(baseline, config_id, overrides)
        config_id = row["config_id"]
        if config_id in seen:
            raise ValueError(f"duplicate config_id {config_id!r}")
        seen.add(config_id)
        rows.append(row)
    _check_unique_resolved_fields(rows)
    return rows


def write_csv_mirror(path: Path, rows: list[dict[str, str]]) -> None:
    existing_results = _read_existing_result_fields(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(CSV_FIELDNAMES), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row.update(existing_results.get(row["config_id"], {}))
            for field in RESULT_FIELDNAMES:
                output_row.setdefault(field, "")
            writer.writerow(output_row)


def build_inherited_yaml(row: Mapping[str, str]) -> dict[str, Any]:
    base_path = _repo_path(row["base_yaml"])
    generated_path = _repo_path(row["generated_yaml"])
    base_config = load_v2_config(base_path)
    desired = _desired_config_from_row(row)
    overrides = _diff_mapping(base_config, desired)
    return {
        "schema_version": INHERITED_SCHEMA_VERSION,
        "config_id": row["config_id"],
        "extends": _relative_path(base_path, generated_path.parent),
        "overrides": overrides,
    }


def resolve_inherited_yaml(
    inherited: Mapping[str, Any], generated_path: Path
) -> dict[str, Any]:
    if inherited.get("schema_version") != INHERITED_SCHEMA_VERSION:
        raise ValueError("inherited YAML has wrong schema_version")
    extends = inherited.get("extends")
    if not isinstance(extends, str) or not extends:
        raise ValueError("inherited YAML requires a non-empty extends field")
    base_path = (generated_path.parent / extends).resolve()
    base_config = load_v2_config(base_path)
    overrides = inherited.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("inherited YAML overrides field must be a mapping")
    return _deep_merge(base_config, overrides)


def runner_control_warnings(rows: list[dict[str, str]]) -> list[str]:
    baseline = next((row for row in rows if row["config_id"] == "V4_baseline"), None)
    if baseline is None:
        return []
    baseline_config = _resolved_config_from_row(baseline)
    baseline_unmapped = _runner_unmapped_field_reasons(baseline_config)
    warnings: list[str] = []
    for row in rows:
        if row["config_id"] == "V4_baseline":
            continue
        config = _resolved_config_from_row(row)
        unmapped = _runner_unmapped_field_reasons(config)
        for field in sorted(set(baseline_unmapped) | set(unmapped)):
            actual = _get_dotted(config, field)
            expected = _get_dotted(baseline_config, field)
            if actual != expected:
                reason = unmapped.get(field) or baseline_unmapped.get(field)
                warnings.append(
                    f"{row['config_id']}: {field} differs from V4_baseline "
                    f"({expected!r} -> {actual!r}); V2 runner reports this "
                    f"field as unmapped: {reason}"
                )
    return warnings


def _resolved_config_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    return resolve_inherited_yaml(
        build_inherited_yaml(row), _repo_path(row["generated_yaml"])
    )


def _runner_unmapped_field_reasons(config: Mapping[str, Any]) -> dict[str, str]:
    plan = build_v2_command_plan(config)
    reasons: dict[str, str] = {}
    for entry in plan.get("unmapped_fields", []):
        field = entry.get("field")
        reason = entry.get("reason")
        if isinstance(field, str) and isinstance(reason, str):
            reasons[field] = reason
    return reasons


def _resolve_registry_row(
    baseline: Mapping[str, str], config_id: str, overrides: Mapping[str, Any]
) -> dict[str, str]:
    if "config_id" in overrides:
        raise ValueError(
            f"registry run {config_id!r} overrides must not contain config_id; "
            "use the run key instead"
        )
    extra = sorted(set(overrides) - set(CONFIG_FIELDNAMES))
    if extra:
        raise ValueError(
            f"registry run {config_id!r} overrides unsupported fields: "
            f"{', '.join(extra)}"
        )
    raw_row: dict[str, Any] = dict(baseline)
    raw_row.update(overrides)
    raw_row["config_id"] = config_id
    return _normalize_resolved_row(
        raw_row, context=f"resolved registry run {config_id}"
    )


def _normalize_resolved_row(
    raw_row: Mapping[str, Any], *, context: str
) -> dict[str, str]:
    missing = [field for field in CONFIG_FIELDNAMES if field not in raw_row]
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")
    extra = sorted(set(raw_row) - set(CONFIG_FIELDNAMES))
    if extra:
        raise ValueError(f"{context} has unsupported fields: {', '.join(extra)}")
    row = {field: _stringify(raw_row[field]) for field in CONFIG_FIELDNAMES}
    if not row["config_id"]:
        raise ValueError(f"{context} has empty config_id")
    if row["normalization_profile"] not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"{context} normalization_profile must be one of "
            f"{sorted(NORMALIZATION_PROFILES)}, got {row['normalization_profile']!r}"
        )
    _check_provenance_fields(row, context=context)
    _check_node_coordinate_fields(row, context=context)
    _check_decoder_bypass_fields(row, context=context)
    return row


def _check_node_coordinate_fields(row: Mapping[str, str], *, context: str) -> None:
    encoding = row["node_coordinate_encoding"]
    if encoding not in NODE_COORDINATE_ENCODINGS:
        raise ValueError(
            f"{context} node_coordinate_encoding must be one of "
            f"{sorted(NODE_COORDINATE_ENCODINGS)}, got {encoding!r}"
        )
    freqs = _positive_int(row, "node_coordinate_freqs", context)
    if encoding == NODE_COORDINATE_ENCODING_RAW and freqs != 4:
        raise ValueError(
            f"{context} node_coordinate_freqs must remain 4 for raw baseline, "
            f"got {freqs!r}"
        )


def _check_provenance_fields(row: Mapping[str, str], *, context: str) -> None:
    common_expected = {
        "target_mode": TARGET_MODE_NORMALIZED_DELTAT,
        "bridge_policy": BRIDGE_POLICY_ZERO_DELTA_U,
        "coord_policy": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
        "target_recovery_policy": TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF,
    }
    profile_expected = {
        NORMALIZATION_PROFILE_LEGACY_ZSCORE: {
            "runner_family": RUNNER_FAMILY_LEGACY_V1,
        },
        NORMALIZATION_PROFILE_SEMANTIC_V1: {
            "runner_family": RUNNER_FAMILY_V4_SEMANTIC,
        },
    }
    for field, expected in {
        **common_expected,
        **profile_expected[row["normalization_profile"]],
    }.items():
        if row[field] != expected:
            raise ValueError(
                f"{context} {field} must be {expected!r} for "
                f"normalization_profile={row['normalization_profile']!r}, "
                f"got {row[field]!r}"
            )
    transform = row["condition_feature_transform"]
    if transform not in CONDITION_TRANSFORMS:
        raise ValueError(
            f"{context} condition_feature_transform must be one of "
            f"{sorted(CONDITION_TRANSFORMS)}, got {transform!r}"
        )
    if row["normalization_profile"] == NORMALIZATION_PROFILE_LEGACY_ZSCORE:
        if transform != CONDITION_TRANSFORM_LEGACY_ZSCORE:
            raise ValueError(
                f"{context} legacy_zscore requires condition_feature_transform="
                f"{CONDITION_TRANSFORM_LEGACY_ZSCORE!r}, got {transform!r}"
            )
    elif transform not in SEMANTIC_CONDITION_TRANSFORMS:
        raise ValueError(
            f"{context} semantic_normalization_v1 requires a semantic "
            f"condition_feature_transform, got {transform!r}"
        )
    feature_manifest_hash = row["feature_manifest_hash"]
    if feature_manifest_hash not in {"", FEATURE_MANIFEST_HASH_PLANNED}:
        if len(feature_manifest_hash) < 8:
            raise ValueError(
                f"{context} feature_manifest_hash must be blank, "
                f"{FEATURE_MANIFEST_HASH_PLANNED!r}, or a real hash-like value"
            )


def _check_decoder_bypass_fields(row: Mapping[str, str], *, context: str) -> None:
    mode = row["decoder_bypass_mode"]
    features = row["decoder_bypass_features"]
    source = row["decoder_bypass_feature_source"]
    init = row["decoder_bypass_init"]
    if mode not in DECODER_BYPASS_MODES:
        raise ValueError(
            f"{context} decoder_bypass_mode must be one of "
            f"{sorted(DECODER_BYPASS_MODES)}, got {mode!r}"
        )
    if features not in DECODER_BYPASS_FEATURES:
        raise ValueError(
            f"{context} decoder_bypass_features must be one of "
            f"{sorted(DECODER_BYPASS_FEATURES)}, got {features!r}"
        )
    if source not in DECODER_BYPASS_FEATURE_SOURCES:
        raise ValueError(
            f"{context} decoder_bypass_feature_source must be one of "
            f"{sorted(DECODER_BYPASS_FEATURE_SOURCES)}, got {source!r}"
        )
    if init not in DECODER_BYPASS_INITS:
        raise ValueError(
            f"{context} decoder_bypass_init must be one of "
            f"{sorted(DECODER_BYPASS_INITS)}, got {init!r}"
        )
    hidden_size = _positive_int(row, "decoder_bypass_hidden_size", context)
    layers = _positive_int(row, "decoder_bypass_layers", context)
    _ = (hidden_size, layers)
    try:
        residual_scale = float(row["decoder_bypass_residual_scale"])
    except ValueError as exc:
        raise ValueError(
            f"{context} decoder_bypass_residual_scale must be numeric"
        ) from exc
    if residual_scale < 0.0:
        raise ValueError(
            f"{context} decoder_bypass_residual_scale must be >= 0"
        )
    if mode == DECODER_BYPASS_MODE_NONE:
        if features != DECODER_BYPASS_FEATURES_NONE:
            raise ValueError(
                f"{context} decoder_bypass_mode='none' requires "
                "decoder_bypass_features='none'"
            )
        return
    if features != DECODER_BYPASS_FEATURES_FULL_CONDITION:
        raise ValueError(
            f"{context} decoder_bypass_mode='post_decoder_residual' requires "
            "decoder_bypass_features='full_condition'"
        )


def _positive_int(row: Mapping[str, str], field: str, context: str) -> int:
    try:
        value = int(row[field])
    except ValueError as exc:
        raise ValueError(f"{context} {field} must be an int") from exc
    if value < 1:
        raise ValueError(f"{context} {field} must be >= 1")
    return value


def _check_unique_resolved_fields(rows: list[dict[str, str]]) -> None:
    for field in UNIQUE_RESOLVED_FIELDS:
        seen: dict[str, str] = {}
        for row in rows:
            value = row[field]
            prior = seen.get(value)
            if prior is not None:
                raise ValueError(
                    f"registry conflict: {field} {value!r} is used by "
                    f"{prior!r} and {row['config_id']!r}"
                )
            seen[value] = row["config_id"]


def _check_csv_mirror(
    registry: Mapping[str, Any], rows: list[dict[str, str]]
) -> None:
    mirror_path = _repo_path(str(registry["csv_mirror_path"]))
    with mirror_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDNAMES:
            raise AssertionError(f"{mirror_path}: CSV mirror header mismatch")
        csv_rows = list(reader)
    csv_config_rows = [
        {field: row.get(field, "") for field in CONFIG_FIELDNAMES}
        for row in csv_rows
    ]
    if csv_config_rows != rows:
        raise AssertionError(
            f"{mirror_path}: CSV configuration fields differ from "
            "authoritative JSON"
        )
    for row in csv_rows:
        missing_result_fields = [
            field for field in RESULT_FIELDNAMES if field not in row
        ]
        if missing_result_fields:
            raise AssertionError(
                f"{mirror_path}: CSV result fields missing: "
                f"{', '.join(missing_result_fields)}"
            )


def _read_existing_result_fields(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    results: dict[str, dict[str, str]] = {}
    for row in rows:
        config_id = row.get("config_id")
        if not config_id:
            continue
        results[config_id] = {
            field: row.get(field, "") for field in RESULT_FIELDNAMES
        }
    return results


def _check_generated_yaml(row: Mapping[str, str]) -> None:
    expected = build_inherited_yaml(row)
    generated_path = _repo_path(row["generated_yaml"])
    with generated_path.open("r", encoding="utf-8") as file:
        actual = yaml.safe_load(file)
    if actual != expected:
        raise AssertionError(
            f"{row['generated_yaml']}: generated YAML is not reproducible "
            "from authoritative JSON"
        )
    resolved = resolve_inherited_yaml(actual, generated_path)
    validate_v2_config(resolved, config_path=_repo_path(row["base_yaml"]))
    _assert_registry_matches_resolved(row, resolved)


def _check_v4_baseline(row: Mapping[str, str]) -> None:
    for field, expected in EXPECTED_V4_BASELINE.items():
        actual = row.get(field)
        if actual != expected:
            raise AssertionError(
                f"V4_baseline {field} expected {expected!r}, got {actual!r}"
            )
    base = load_v2_config(_repo_path(row["base_yaml"]))
    resolved = resolve_inherited_yaml(
        build_inherited_yaml(row), _repo_path(row["generated_yaml"])
    )
    for config, label in ((base, "V4_base"), (resolved, "V4_baseline")):
        checks = {
            "model.node_latent_size": 96,
            "model.edge_latent_size": 96,
            "model.processor_steps": 6,
            "model.mlp_hidden_layers": 2,
            "model.decoder_bypass_mode": DECODER_BYPASS_MODE_NONE,
            "model.decoder_bypass_features": DECODER_BYPASS_FEATURES_NONE,
            "model.decoder_bypass_feature_source": DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
            "model.decoder_bypass_hidden_size": 64,
            "model.decoder_bypass_layers": 2,
            "model.decoder_bypass_init": DECODER_BYPASS_INIT_ZERO_RESIDUAL,
            "model.decoder_bypass_residual_scale": 1.0,
            "dataset.split_map_path": "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json",
            "graph.node_coordinate_encoding": NODE_COORDINATE_ENCODING_RAW,
            "graph.node_coordinate_freqs": 4,
            "run.batch_size": 88,
            "run.batch_plan": "sample_shuffle",
            "optimizer.name": "adamw",
            "optimizer.lr": 0.0005,
            "optimizer.model_seed": 0,
            "optimizer.batch_order_seed": 0,
            "optimizer.graph_seed": 0,
            "optimizer.seed": 0,
            "optimizer.multi_seed": [],
            "optimizer.lr_schedule": "warmup_cosine",
            "optimizer.warmup_epochs": 10,
            "optimizer.min_lr": 0.00005,
            "optimizer.weight_decay": 0.0001,
            "run.epochs": 600,
            "run.batch_build_seed": 0,
            "graph.radius_policy": "discrete_physical_coverage",
            "graph.coverage_repair_policy": "none",
            "loss.mode": "mse",
            "dataset.normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
            "dataset.condition_feature_transform": CONDITION_TRANSFORM_LEGACY_ZSCORE,
            "export.selection_metric": DEFAULT_SELECTION_METRIC,
            "metadata.runner_family": RUNNER_FAMILY_LEGACY_V1,
            "metadata.target_mode": TARGET_MODE_NORMALIZED_DELTAT,
            "metadata.bridge_policy": BRIDGE_POLICY_ZERO_DELTA_U,
            "metadata.normalization_profile": NORMALIZATION_PROFILE_LEGACY_ZSCORE,
            "metadata.coord_policy": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
            "metadata.condition_feature_transform": CONDITION_TRANSFORM_LEGACY_ZSCORE,
            "metadata.node_coordinate_encoding": NODE_COORDINATE_ENCODING_RAW,
            "metadata.node_coordinate_freqs": 4,
            "metadata.decoder_bypass_mode": DECODER_BYPASS_MODE_NONE,
            "metadata.decoder_bypass_features": DECODER_BYPASS_FEATURES_NONE,
            "metadata.decoder_bypass_feature_source": DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
            "metadata.target_recovery_policy": TARGET_RECOVERY_POLICY_DELTAT_NORM_TO_K_PLUS_T_REF,
            "metadata.feature_manifest_hash": FEATURE_MANIFEST_HASH_PLANNED,
            "metadata.metrics_profile": DEFAULT_METRICS_PROFILE,
            "metadata.metrics_contract": DEFAULT_METRICS_CONTRACT,
            "metadata.selection_metric": DEFAULT_SELECTION_METRIC,
        }
        for dotted, expected in checks.items():
            actual = _get_dotted(config, dotted)
            if actual != expected:
                raise AssertionError(
                    f"{label} {dotted} expected {expected!r}, got {actual!r}"
                )


def _desired_config_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    config_id = row["config_id"]
    return {
        "description": (
            f"Heat3D V4 registry config {config_id}; standard task "
            f"{row['task']}; generated from {Path(row['base_yaml']).name}."
        ),
        "model": {
            "node_latent_size": _int(row, "node_latent_size"),
            "edge_latent_size": _int(row, "edge_latent_size"),
            "processor_steps": _int(row, "processor_steps"),
            "mlp_hidden_layers": _int(row, "mlp_hidden_layers"),
            "decoder_bypass_mode": row["decoder_bypass_mode"],
            "decoder_bypass_features": row["decoder_bypass_features"],
            "decoder_bypass_feature_source": row["decoder_bypass_feature_source"],
            "decoder_bypass_hidden_size": _int(row, "decoder_bypass_hidden_size"),
            "decoder_bypass_layers": _int(row, "decoder_bypass_layers"),
            "decoder_bypass_init": row["decoder_bypass_init"],
            "decoder_bypass_residual_scale": _float(
                row, "decoder_bypass_residual_scale"
            ),
        },
        "dataset": {
            "split_map_path": row["split_map_path"],
            "normalization_profile": row["normalization_profile"],
            "condition_feature_transform": row["condition_feature_transform"],
        },
        "optimizer": {
            "name": row["optimizer"],
            "lr": _float(row, "lr"),
            "model_seed": _int(row, "model_seed"),
            "batch_order_seed": _int(row, "batch_order_seed"),
            "graph_seed": _int(row, "graph_seed"),
            "seed": _int(row, "seed"),
            "multi_seed": _json_list(row, "multi_seed"),
            "lr_schedule": row["lr_schedule"],
            "warmup_epochs": _int(row, "warmup_epochs"),
            "min_lr": _float(row, "min_lr"),
            "weight_decay": _float(row, "weight_decay"),
        },
        "loss": {"mode": row["loss_mode"]},
        "run": {
            "epochs": _int(row, "epochs"),
            "batch_size": _int(row, "batch_size"),
            "validation_batch_size": _int(row, "validation_batch_size"),
            "prediction_batch_size": _int(row, "prediction_batch_size"),
            "batch_plan": row["batch_plan"],
            "batch_build_seed": _int(row, "batch_build_seed"),
            "final_probe_eval_after_training": _bool(
                row, "final_probe_eval_after_training"
            ),
            "final_probe_output_dir": row["final_probe_output_dir"],
            "post_training_diagnostics": _bool(row, "post_training_diagnostics"),
            "post_training_diagnostics_output_dir": row[
                "post_training_diagnostics_output_dir"
            ],
        },
        "graph": {
            "node_coordinate_encoding": row["node_coordinate_encoding"],
            "node_coordinate_freqs": _int(row, "node_coordinate_freqs"),
            "radius_policy": row["graph_radius_policy"],
            "coverage_repair_policy": row["coverage_repair_policy"],
        },
        "export": {
            "output_dir": row["output_dir"],
            "run_name": row["run_name"],
            "selection_metric": row["selection_metric"],
            "save_final_predictions": _bool(row, "save_final_predictions"),
            "save_best_predictions": _bool(row, "save_best_predictions"),
        },
        "diagnostics": {
            "run_baseline_comparison": _bool(row, "run_baseline_comparison"),
            "run_error_bins": _bool(row, "run_error_bins"),
            "run_condition_diagnostics": _bool(row, "run_condition_diagnostics"),
            "run_summary": _bool(row, "run_summary_diagnostics"),
            "run_field_shape_diagnostics": _bool(
                row, "run_field_shape_diagnostics"
            ),
        },
        "metadata": {
            "registry_config_id": config_id,
            "registry_status": row["status"],
            "standard_task": row["task"],
            "variant_label": config_id,
            "model_label": (
                f"latent{row['node_latent_size']}-edge{row['edge_latent_size']}"
                f"-s{row['processor_steps']}-mlp{row['mlp_hidden_layers']}"
            ),
            "batch_policy_label": f"B{row['batch_size']}_{row['batch_plan']}",
            "optimizer_label": (
                f"{row['optimizer']}_{row['lr_schedule']}_lr{row['lr']}"
                f"_minlr{row['min_lr']}"
            ),
            "graph_policy_label": (
                f"{row['graph_radius_policy']}_repair_{row['coverage_repair_policy']}"
            ),
            "loss_label": f"plain_{row['loss_mode']}",
            "metrics_profile": row["metrics_profile"],
            "metrics_contract": row["metrics_contract"],
            "selection_metric": row["selection_metric"],
            "runner_family": row["runner_family"],
            "target_mode": row["target_mode"],
            "bridge_policy": row["bridge_policy"],
            "normalization_profile": row["normalization_profile"],
            "coord_policy": row["coord_policy"],
            "condition_feature_transform": row["condition_feature_transform"],
            "node_coordinate_encoding": row["node_coordinate_encoding"],
            "node_coordinate_freqs": _int(row, "node_coordinate_freqs"),
            "decoder_bypass_mode": row["decoder_bypass_mode"],
            "decoder_bypass_features": row["decoder_bypass_features"],
            "decoder_bypass_feature_source": row["decoder_bypass_feature_source"],
            "target_recovery_policy": row["target_recovery_policy"],
            "feature_manifest_hash": row["feature_manifest_hash"],
            "launch_policy": row["launch_policy"],
            "log_path": row["log_path"],
            "notes": row["notes"],
        },
    }


def _assert_registry_matches_resolved(
    row: Mapping[str, str], config: Mapping[str, Any]
) -> None:
    checks = {
        "model.node_latent_size": _int(row, "node_latent_size"),
        "model.edge_latent_size": _int(row, "edge_latent_size"),
        "model.processor_steps": _int(row, "processor_steps"),
        "model.mlp_hidden_layers": _int(row, "mlp_hidden_layers"),
        "model.decoder_bypass_mode": row["decoder_bypass_mode"],
        "model.decoder_bypass_features": row["decoder_bypass_features"],
        "model.decoder_bypass_feature_source": row["decoder_bypass_feature_source"],
        "model.decoder_bypass_hidden_size": _int(row, "decoder_bypass_hidden_size"),
        "model.decoder_bypass_layers": _int(row, "decoder_bypass_layers"),
        "model.decoder_bypass_init": row["decoder_bypass_init"],
        "model.decoder_bypass_residual_scale": _float(
            row, "decoder_bypass_residual_scale"
        ),
        "run.batch_size": _int(row, "batch_size"),
        "run.validation_batch_size": _int(row, "validation_batch_size"),
        "run.prediction_batch_size": _int(row, "prediction_batch_size"),
        "run.batch_plan": row["batch_plan"],
        "run.batch_build_seed": _int(row, "batch_build_seed"),
        "run.epochs": _int(row, "epochs"),
        "run.final_probe_eval_after_training": _bool(
            row, "final_probe_eval_after_training"
        ),
        "run.post_training_diagnostics": _bool(row, "post_training_diagnostics"),
        "optimizer.name": row["optimizer"],
        "optimizer.lr": _float(row, "lr"),
        "optimizer.model_seed": _int(row, "model_seed"),
        "optimizer.batch_order_seed": _int(row, "batch_order_seed"),
        "optimizer.graph_seed": _int(row, "graph_seed"),
        "optimizer.seed": _int(row, "seed"),
        "optimizer.multi_seed": _json_list(row, "multi_seed"),
        "optimizer.lr_schedule": row["lr_schedule"],
        "optimizer.warmup_epochs": _int(row, "warmup_epochs"),
        "optimizer.min_lr": _float(row, "min_lr"),
        "optimizer.weight_decay": _float(row, "weight_decay"),
        "graph.node_coordinate_encoding": row["node_coordinate_encoding"],
        "graph.node_coordinate_freqs": _int(row, "node_coordinate_freqs"),
        "graph.radius_policy": row["graph_radius_policy"],
        "graph.coverage_repair_policy": row["coverage_repair_policy"],
        "dataset.split_map_path": row["split_map_path"],
        "dataset.normalization_profile": row["normalization_profile"],
        "dataset.condition_feature_transform": row["condition_feature_transform"],
        "loss.mode": row["loss_mode"],
        "export.selection_metric": row["selection_metric"],
        "export.output_dir": row["output_dir"],
        "export.run_name": row["run_name"],
        "export.save_final_predictions": _bool(row, "save_final_predictions"),
        "export.save_best_predictions": _bool(row, "save_best_predictions"),
        "diagnostics.run_baseline_comparison": _bool(row, "run_baseline_comparison"),
        "diagnostics.run_error_bins": _bool(row, "run_error_bins"),
        "diagnostics.run_condition_diagnostics": _bool(
            row, "run_condition_diagnostics"
        ),
        "diagnostics.run_summary": _bool(row, "run_summary_diagnostics"),
        "diagnostics.run_field_shape_diagnostics": _bool(
            row, "run_field_shape_diagnostics"
        ),
        "metadata.metrics_profile": row["metrics_profile"],
        "metadata.metrics_contract": row["metrics_contract"],
        "metadata.selection_metric": row["selection_metric"],
        "metadata.runner_family": row["runner_family"],
        "metadata.target_mode": row["target_mode"],
        "metadata.bridge_policy": row["bridge_policy"],
        "metadata.normalization_profile": row["normalization_profile"],
        "metadata.coord_policy": row["coord_policy"],
        "metadata.condition_feature_transform": row["condition_feature_transform"],
        "metadata.node_coordinate_encoding": row["node_coordinate_encoding"],
        "metadata.node_coordinate_freqs": _int(row, "node_coordinate_freqs"),
        "metadata.decoder_bypass_mode": row["decoder_bypass_mode"],
        "metadata.decoder_bypass_features": row["decoder_bypass_features"],
        "metadata.decoder_bypass_feature_source": row["decoder_bypass_feature_source"],
        "metadata.target_recovery_policy": row["target_recovery_policy"],
        "metadata.feature_manifest_hash": row["feature_manifest_hash"],
    }
    for dotted, expected in checks.items():
        actual = _get_dotted(config, dotted)
        if actual != expected:
            raise AssertionError(
                f"{row['config_id']}: {dotted} expected {expected!r}, got {actual!r}"
            )


def _check_metrics_contracts(rows: list[dict[str, str]]) -> None:
    contracts: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        contract_path_text = row["metrics_contract"]
        if not contract_path_text:
            raise ValueError(f"{row['config_id']}: metrics_contract must be set")
        contract = contracts.get(contract_path_text)
        if contract is None:
            contract = _load_metrics_contract(contract_path_text)
            _check_metrics_contract(contract, contract_path_text)
            contracts[contract_path_text] = contract
        profile = _required_string(
            contract, "metrics_profile", f"{contract_path_text}: metrics contract"
        )
        if row["metrics_profile"] != profile:
            raise AssertionError(
                f"{row['config_id']}: metrics_profile {row['metrics_profile']!r} "
                f"does not match {contract_path_text} profile {profile!r}"
            )
        allowed = _allowed_selection_metrics(contract, contract_path_text)
        if row["selection_metric"] not in allowed:
            raise AssertionError(
                f"{row['config_id']}: selection_metric {row['selection_metric']!r} "
                f"is not allowed by {contract_path_text}: {sorted(allowed)}"
            )
        if (
            row["config_id"] == "V4_baseline"
            and row["selection_metric"] != DEFAULT_SELECTION_METRIC
        ):
            raise AssertionError(
                f"V4_baseline selection_metric must be {DEFAULT_SELECTION_METRIC!r}"
            )


def _load_metrics_contract(path_text: str) -> dict[str, Any]:
    path = _repo_path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"metrics contract not found: {path_text}")
    with path.open("r", encoding="utf-8") as file:
        contract = json.load(file)
    if not isinstance(contract, dict):
        raise ValueError(f"{path_text}: metrics contract must be a JSON object")
    return contract


def _check_metrics_contract(contract: Mapping[str, Any], path_text: str) -> None:
    context = f"{path_text}: metrics contract"
    schema = _required_string(contract, "schema_version", context)
    if schema != METRICS_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path_text}: schema_version must be "
            f"{METRICS_CONTRACT_SCHEMA_VERSION!r}, got {schema!r}"
        )
    profile = _required_string(contract, "metrics_profile", context)
    if profile != DEFAULT_METRICS_PROFILE:
        raise ValueError(
            f"{path_text}: metrics_profile must be {DEFAULT_METRICS_PROFILE!r}"
        )
    default_selection = _required_string(
        contract, "default_checkpoint_selection_metric", context
    )
    if default_selection != DEFAULT_SELECTION_METRIC:
        raise ValueError(
            f"{path_text}: default checkpoint selection metric must be "
            f"{DEFAULT_SELECTION_METRIC!r}"
        )
    allowed = _allowed_selection_metrics(contract, path_text)
    if default_selection not in allowed:
        raise ValueError(
            f"{path_text}: default selection metric {default_selection!r} is "
            "not listed in checkpoint_selection.allowed_selection_metrics"
        )
    metric_names = _contract_metric_names(contract, path_text)
    if default_selection not in metric_names:
        raise ValueError(
            f"{path_text}: default selection metric {default_selection!r} is "
            "not listed in metric_groups"
        )
    aggregation = contract.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError(f"{path_text}: aggregation must be an object")
    if aggregation.get("per_sample_first") is not True:
        raise ValueError(f"{path_text}: aggregation.per_sample_first must be true")
    summaries = aggregation.get("split_group_summaries")
    if not isinstance(summaries, list) or not {"mean", "median", "std"}.issubset(
        set(summaries)
    ):
        raise ValueError(
            f"{path_text}: aggregation.split_group_summaries must include "
            "mean, median, and std"
        )


def _allowed_selection_metrics(
    contract: Mapping[str, Any], path_text: str
) -> set[str]:
    selection = contract.get("checkpoint_selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{path_text}: checkpoint_selection must be an object")
    allowed = selection.get("allowed_selection_metrics")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError(
            f"{path_text}: checkpoint_selection.allowed_selection_metrics "
            "must be a non-empty list"
        )
    result: set[str] = set()
    for item in allowed:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{path_text}: allowed selection metrics must be non-empty strings"
            )
        if item in result:
            raise ValueError(f"{path_text}: duplicate allowed selection metric {item!r}")
        result.add(item)
    return result


def _contract_metric_names(contract: Mapping[str, Any], path_text: str) -> set[str]:
    groups = contract.get("metric_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{path_text}: metric_groups must be a non-empty list")
    seen: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError(f"{path_text}: each metric group must be an object")
        group_name = _required_string(group, "group", f"{path_text}: metric group")
        metrics = group.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(
                f"{path_text}: metric group {group_name!r} must list metrics"
            )
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                raise ValueError(
                    f"{path_text}: metric names in group {group_name!r} must "
                    "be non-empty strings"
                )
            prior_group = seen.get(metric)
            if prior_group is not None:
                raise ValueError(
                    f"{path_text}: metric {metric!r} is listed in both "
                    f"{prior_group!r} and {group_name!r}"
                )
            seen[metric] = group_name
    return set(seen)


def _required_string(
    payload: Mapping[str, Any], key: str, context: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def _diff_mapping(base: Mapping[str, Any], desired: Mapping[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, desired_value in desired.items():
        base_value = base.get(key)
        if isinstance(base_value, Mapping) and isinstance(desired_value, Mapping):
            nested = _diff_mapping(base_value, desired_value)
            if nested:
                diff[key] = nested
        elif base_value != desired_value:
            diff[key] = desired_value
    return diff


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)


def write_generated_yaml(row: Mapping[str, str]) -> Path:
    path = _repo_path(row["generated_yaml"])
    _write_yaml(path, build_inherited_yaml(row))
    return path


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _int(row: Mapping[str, str], key: str) -> int:
    return int(row[key])


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _bool(row: Mapping[str, str], key: str) -> bool:
    value = row[key].strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{key} must be true or false, got {row[key]!r}")


def _json_list(row: Mapping[str, str], key: str) -> list[Any]:
    value = json.loads(row[key])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return value


def _get_dotted(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
