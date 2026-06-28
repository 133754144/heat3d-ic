#!/usr/bin/env python3
"""Check that extracted Heat3D v1 legacy helpers preserve old formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v1_normalization import (  # noqa: E402
    BC_FLAG_FEATURES,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
    SEMANTIC_LOG_EPS,
    TRANSFORM_BINARY_PASSTHROUGH,
    TRANSFORM_LOG_K_ZSCORE,
    TRANSFORM_SIGNED_LOG1P_Q_ZSCORE,
    legacy_train_only_stats,
    normalize_condition,
    normalize_coords,
    normalize_target_delta,
    normalized_delta_to_raw,
    recover_raw_condition,
    recover_temperature_from_normalized_delta,
    semantic_normalization_v1_train_only_stats,
    training_normalization_stats,
)
from rigno.heat3d_v1_training_semantics import (  # noqa: E402
    BOUNDARY_DISTANCE_FEATURES,
    COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC,
    EXTENT_BROADCAST_FEATURES,
    EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST,
    INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT,
    build_configured_zero_delta_bridge,
    build_legacy_zero_delta_bridge,
    legacy_training_semantics_manifest,
)


DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)
REFERENCE_BRIDGE_POLICY = "zero_delta_u_bridge"
REFERENCE_EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pre-extraction legacy formulas against the new stable "
            "Heat3D v1 training-semantics helpers. This script does not train "
            "or write artifacts."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--max-train-examples", type=int, default=16)
    parser.add_argument("--max-check-examples", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    if path.name == "samples":
        return path
    samples = path / "samples"
    return samples if samples.exists() else path


def _reference_bridge(example: Any) -> Any:
    return example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy=REFERENCE_BRIDGE_POLICY
    )


def _reference_safe_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(array, axis=0, keepdims=True)
    std = np.std(array, axis=0, keepdims=True)
    return mean, np.where(std < REFERENCE_EPS, 1.0, std)


def _reference_train_only_stats(examples: list[Any]) -> dict[str, Any]:
    c_values = []
    delta_values = []
    coord_values = []
    feature_names = None
    for example in examples:
        bridge = _reference_bridge(example)
        names = bridge.condition_feature_names
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Relative condition feature-name mismatch in train split")

        c_values.append(np.asarray(bridge.legacy_inputs.c).reshape(-1, len(names)))
        delta_values.append(np.asarray(bridge.target_delta_u).reshape(-1, 1))
        coord_values.append(np.asarray(bridge.legacy_inputs.x_inp).reshape(-1, 3))

    c_all = np.concatenate(c_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    c_mean, c_std = _reference_safe_stats(c_all)
    delta_mean, delta_std = _reference_safe_stats(delta_all)
    coord_min = np.min(coord_all, axis=0, keepdims=True)
    coord_max = np.max(coord_all, axis=0, keepdims=True)
    coord_span = np.where((coord_max - coord_min) < REFERENCE_EPS, 1.0, coord_max - coord_min)
    return {
        "feature_names": tuple(feature_names or ()),
        "condition_mean": c_mean.reshape(1, 1, 1, -1),
        "condition_std": c_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
    }


def _reference_normalize_coords(coords: Any, stats: dict[str, Any]) -> Any:
    return 2.0 * ((coords - stats["coord_min"]) / stats["coord_span"]) - 1.0


def _reference_normalize_condition(raw_c: Any, stats: dict[str, Any]) -> Any:
    return (raw_c - stats["condition_mean"]) / stats["condition_std"]


def _reference_recover_raw_condition(normalized_c: Any, stats: dict[str, Any]) -> Any:
    return normalized_c * stats["condition_std"] + stats["condition_mean"]


def _reference_normalize_target_delta(target_delta: Any, stats: dict[str, Any]) -> Any:
    return (target_delta - stats["target_delta_mean"]) / stats["target_delta_std"]


def _reference_normalized_delta_to_raw(pred_normalized: Any, stats: dict[str, Any]) -> Any:
    return pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]


def _reference_recover_temperature(pred_normalized: Any, t_ref: Any, stats: dict[str, Any]) -> Any:
    return t_ref + _reference_normalized_delta_to_raw(pred_normalized, stats)


def _reference_semantic_transform_condition_np(
    raw_c: np.ndarray, transforms: tuple[str, ...]
) -> np.ndarray:
    columns = []
    for index, transform in enumerate(transforms):
        values = raw_c[:, index : index + 1]
        if transform == TRANSFORM_LOG_K_ZSCORE:
            columns.append(np.log(np.maximum(values, SEMANTIC_LOG_EPS)))
        elif transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
            columns.append(np.sign(values) * np.log1p(np.abs(values)))
        else:
            columns.append(values)
    return np.concatenate(columns, axis=-1)


def _reference_semantic_inverse_condition_np(
    transformed_c: np.ndarray, transforms: tuple[str, ...]
) -> np.ndarray:
    columns = []
    for index, transform in enumerate(transforms):
        values = transformed_c[..., index : index + 1]
        if transform == TRANSFORM_LOG_K_ZSCORE:
            columns.append(np.exp(values))
        elif transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
            columns.append(np.sign(values) * np.expm1(np.abs(values)))
        else:
            columns.append(values)
    return np.concatenate(columns, axis=-1)


def _reference_normalize_semantic_condition(raw_c: Any, stats: dict[str, Any]) -> np.ndarray:
    transforms = tuple(stats["condition_feature_transforms"])
    raw_array = jnp.asarray(raw_c)
    mean = stats["condition_mean"]
    std = stats["condition_std"]
    columns = []
    for index, transform in enumerate(transforms):
        raw_column = raw_array[..., index : index + 1]
        transformed = _reference_semantic_transform_column_jnp(raw_column, transform)
        columns.append(
            (transformed - mean[..., index : index + 1])
            / std[..., index : index + 1]
        )
    return jnp.concatenate(columns, axis=-1)


def _reference_recover_semantic_condition(
    normalized_c: Any, stats: dict[str, Any]
) -> np.ndarray:
    transforms = tuple(stats["condition_feature_transforms"])
    normalized_array = jnp.asarray(normalized_c)
    mean = stats["condition_mean"]
    std = stats["condition_std"]
    columns = []
    for index, transform in enumerate(transforms):
        transformed = normalized_array[..., index : index + 1]
        transformed = transformed * std[..., index : index + 1]
        transformed = transformed + mean[..., index : index + 1]
        columns.append(_reference_semantic_inverse_transform_column_jnp(transformed, transform))
    return jnp.concatenate(columns, axis=-1)


def _reference_semantic_transform_column_jnp(raw_column: Any, transform: str) -> Any:
    if transform == TRANSFORM_LOG_K_ZSCORE:
        return jnp.log(jnp.maximum(raw_column, SEMANTIC_LOG_EPS))
    if transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
        return jnp.sign(raw_column) * jnp.log1p(jnp.abs(raw_column))
    return raw_column


def _reference_semantic_inverse_transform_column_jnp(
    transformed_column: Any,
    transform: str,
) -> Any:
    if transform == TRANSFORM_LOG_K_ZSCORE:
        return jnp.exp(transformed_column)
    if transform == TRANSFORM_SIGNED_LOG1P_Q_ZSCORE:
        return jnp.sign(transformed_column) * jnp.expm1(jnp.abs(transformed_column))
    return transformed_column


def _reference_semantic_train_only_stats(
    examples: list[Any], condition_feature_transform: str
) -> dict[str, Any]:
    helper_probe = semantic_normalization_v1_train_only_stats(
        examples,
        condition_feature_transform=condition_feature_transform,
    )
    transforms = tuple(helper_probe["condition_feature_transforms"])
    c_values = []
    transformed_values = []
    delta_values = []
    coord_values = []
    for example in examples:
        bridge = _reference_bridge(example)
        raw_c = np.asarray(bridge.legacy_inputs.c, dtype=np.float64).reshape(
            -1, len(transforms)
        )
        c_values.append(raw_c)
        transformed_values.append(
            _reference_semantic_transform_condition_np(raw_c, transforms)
        )
        delta_values.append(np.asarray(bridge.target_delta_u).reshape(-1, 1))
        coord_values.append(np.asarray(bridge.legacy_inputs.x_inp).reshape(-1, 3))

    transformed_all = np.concatenate(transformed_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    condition_mean, condition_std = _reference_safe_stats(transformed_all)
    passthrough = np.asarray(
        [transform == TRANSFORM_BINARY_PASSTHROUGH for transform in transforms],
        dtype=bool,
    ).reshape(1, -1)
    condition_mean = np.where(passthrough, 0.0, condition_mean)
    condition_std = np.where(passthrough, 1.0, condition_std)
    delta_mean, delta_std = _reference_safe_stats(delta_all)
    coord_min = np.min(coord_all, axis=0, keepdims=True)
    coord_max = np.max(coord_all, axis=0, keepdims=True)
    coord_span = np.where(
        (coord_max - coord_min) < REFERENCE_EPS,
        1.0,
        coord_max - coord_min,
    )
    return {
        "feature_names": tuple(helper_probe["feature_names"]),
        "condition_feature_transforms": transforms,
        "condition_mean": condition_mean.reshape(1, 1, 1, -1),
        "condition_std": condition_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
    }


def _max_abs_diff(left: Any, right: Any) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype.kind in {"U", "S", "O"} or right_array.dtype.kind in {"U", "S", "O"}:
        return 0.0 if tuple(left_array.reshape(-1)) == tuple(right_array.reshape(-1)) else float("inf")
    if left_array.size == 0 and right_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def _split_examples(examples: list[Any]) -> tuple[list[Any], list[Any]]:
    train = [example for example in examples if example.meta.get("split") == "train"]
    non_train = [example for example in examples if example.meta.get("split") != "train"]
    if not train:
        midpoint = max(1, len(examples) // 2)
        train = examples[:midpoint]
        non_train = examples[midpoint:]
    check = train[:]
    check.extend(non_train)
    return train, check


def main() -> int:
    args = parse_args()
    sample_root = _sample_root(args.subset).resolve()
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    examples = sorted(dataset.samples, key=lambda item: item.sample_id)
    train_examples, check_examples = _split_examples(examples)
    train_examples = train_examples[: max(1, args.max_train_examples)]
    check_examples = check_examples[: max(1, args.max_check_examples)]
    if not train_examples or not check_examples:
        raise ValueError(f"No usable examples found in {sample_root}")

    reference_stats = _reference_train_only_stats(train_examples)
    helper_stats = legacy_train_only_stats(train_examples)

    diffs: dict[str, float] = {}
    for key in (
        "condition_mean",
        "condition_std",
        "target_delta_mean",
        "target_delta_std",
        "coord_min",
        "coord_span",
    ):
        diffs[f"stats.{key}"] = _max_abs_diff(reference_stats[key], helper_stats[key])
    diffs["stats.feature_names"] = _max_abs_diff(
        np.asarray(reference_stats["feature_names"], dtype=object),
        np.asarray(helper_stats["feature_names"], dtype=object),
    )

    for example in check_examples:
        reference_bridge = _reference_bridge(example)
        helper_bridge = build_legacy_zero_delta_bridge(example)
        prefix = f"sample.{example.sample_id}"
        diffs[f"{prefix}.u"] = _max_abs_diff(reference_bridge.legacy_inputs.u, helper_bridge.legacy_inputs.u)
        diffs[f"{prefix}.raw_c"] = _max_abs_diff(
            reference_bridge.legacy_inputs.c,
            helper_bridge.legacy_inputs.c,
        )
        diffs[f"{prefix}.raw_coords"] = _max_abs_diff(
            reference_bridge.legacy_inputs.x_inp,
            helper_bridge.legacy_inputs.x_inp,
        )
        diffs[f"{prefix}.target_delta"] = _max_abs_diff(
            reference_bridge.target_delta_u,
            helper_bridge.target_delta_u,
        )

        raw_c = np.asarray(helper_bridge.legacy_inputs.c)
        raw_coords = np.asarray(helper_bridge.legacy_inputs.x_inp)
        target_delta = np.asarray(helper_bridge.target_delta_u)
        target_norm_ref = _reference_normalize_target_delta(target_delta, reference_stats)
        target_norm_helper = normalize_target_delta(target_delta, helper_stats)
        diffs[f"{prefix}.normalized_c"] = _max_abs_diff(
            _reference_normalize_condition(raw_c, reference_stats),
            normalize_condition(raw_c, helper_stats),
        )
        diffs[f"{prefix}.raw_c_recovery"] = _max_abs_diff(
            _reference_recover_raw_condition(
                _reference_normalize_condition(raw_c, reference_stats),
                reference_stats,
            ),
            recover_raw_condition(normalize_condition(raw_c, helper_stats), helper_stats),
        )
        diffs[f"{prefix}.normalized_target"] = _max_abs_diff(target_norm_ref, target_norm_helper)
        diffs[f"{prefix}.normalized_coords"] = _max_abs_diff(
            _reference_normalize_coords(raw_coords, reference_stats),
            normalize_coords(raw_coords, helper_stats),
        )
        probe_pred_norm = target_norm_helper + np.asarray(0.125, dtype=target_norm_helper.dtype)
        diffs[f"{prefix}.raw_delta_recovery"] = _max_abs_diff(
            _reference_normalized_delta_to_raw(probe_pred_norm, reference_stats),
            normalized_delta_to_raw(probe_pred_norm, helper_stats),
        )
        diffs[f"{prefix}.temperature_recovery"] = _max_abs_diff(
            _reference_recover_temperature(probe_pred_norm, helper_bridge.t_ref, reference_stats),
            recover_temperature_from_normalized_delta(probe_pred_norm, helper_bridge.t_ref, helper_stats),
        )

    semantic_transforms = (
        CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
        CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY,
        CONDITION_FEATURE_TRANSFORM_SEMANTIC_Q_ONLY,
        CONDITION_FEATURE_TRANSFORM_SEMANTIC_K_ONLY,
    )
    semantic_checked = []
    for condition_transform in semantic_transforms:
        label = condition_transform.replace("semantic_v1_", "semantic.")
        reference_semantic_stats = _reference_semantic_train_only_stats(
            train_examples,
            condition_transform,
        )
        helper_semantic_stats = semantic_normalization_v1_train_only_stats(
            train_examples,
            condition_feature_transform=condition_transform,
        )
        semantic_checked.append(condition_transform)
        for key in (
            "condition_mean",
            "condition_std",
            "target_delta_mean",
            "target_delta_std",
            "coord_min",
            "coord_span",
        ):
            diffs[f"{label}.stats.{key}"] = _max_abs_diff(
                reference_semantic_stats[key],
                helper_semantic_stats[key],
            )
        for key in ("feature_names", "condition_feature_transforms"):
            diffs[f"{label}.stats.{key}"] = _max_abs_diff(
                np.asarray(reference_semantic_stats[key], dtype=object),
                np.asarray(helper_semantic_stats[key], dtype=object),
            )

        names = tuple(helper_semantic_stats["feature_names"])
        bc_indices = [index for index, name in enumerate(names) if name in BC_FLAG_FEATURES]
        non_bc_indices = [index for index in range(len(names)) if index not in bc_indices]
        for example in check_examples:
            bridge = build_legacy_zero_delta_bridge(example)
            raw_c = np.asarray(bridge.legacy_inputs.c)
            prefix = f"{label}.sample.{example.sample_id}"
            reference_norm = _reference_normalize_semantic_condition(
                raw_c,
                reference_semantic_stats,
            )
            helper_norm = normalize_condition(raw_c, helper_semantic_stats)
            diffs[f"{prefix}.normalized_c"] = _max_abs_diff(reference_norm, helper_norm)
            diffs[f"{prefix}.raw_c_recovery"] = _max_abs_diff(
                _reference_recover_semantic_condition(helper_norm, helper_semantic_stats),
                recover_raw_condition(helper_norm, helper_semantic_stats),
            )
            if condition_transform == CONDITION_FEATURE_TRANSFORM_SEMANTIC_BC_ONLY:
                legacy_norm = normalize_condition(raw_c, helper_stats)
                diffs[f"{prefix}.bc_only.non_bc_matches_legacy"] = _max_abs_diff(
                    np.asarray(helper_norm)[..., non_bc_indices],
                    np.asarray(legacy_norm)[..., non_bc_indices],
                )
                bc_raw = np.asarray(raw_c)[..., bc_indices]
                bc_norm = np.asarray(helper_norm)[..., bc_indices]
                diffs[f"{prefix}.bc_only.bc_flags_passthrough"] = _max_abs_diff(
                    bc_norm,
                    bc_raw,
                )
                diffs[f"{prefix}.bc_only.bc_flags_binary"] = float(
                    np.max(np.minimum(np.abs(bc_raw), np.abs(bc_raw - 1.0)))
                )

    p2_checks = _check_p2_feature_and_coord_policies(train_examples, check_examples)

    max_abs_diff = max(diffs.values(), default=0.0)
    payload = {
        "script": Path(__file__).name,
        "non_execution": "no training, no evaluation, no artifact writes",
        "subset": str(sample_root),
        "train_examples": [example.sample_id for example in train_examples],
        "checked_examples": [example.sample_id for example in check_examples],
        "semantics": legacy_training_semantics_manifest(),
        "semantic_condition_transforms_checked": semantic_checked,
        "p2_checks": p2_checks,
        "max_abs_diff": max_abs_diff,
        "passed": bool(max_abs_diff <= args.atol),
        "diffs": diffs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def _check_p2_feature_and_coord_policies(
    train_examples: list[Any],
    check_examples: list[Any],
) -> dict[str, Any]:
    boundary_schema_samples: dict[str, Any] = {}
    extent_schema_samples: dict[str, Any] = {}
    for example in check_examples:
        replacement_bridge = build_configured_zero_delta_bridge(
            example,
            input_feature_schema=INPUT_FEATURE_SCHEMA_BOUNDARY_DISTANCE_REPLACEMENT,
        )
        names = tuple(replacement_bridge.condition_feature_names)
        lingering_flags = [name for name in BC_FLAG_FEATURES if name in names]
        missing_distances = [
            name for name in BOUNDARY_DISTANCE_FEATURES if name not in names
        ]
        if lingering_flags or missing_distances:
            raise ValueError(
                "boundary_distance_replacement produced invalid feature names: "
                f"flags={lingering_flags} missing_distances={missing_distances} names={names}"
            )
        distances = np.asarray(replacement_bridge.legacy_inputs.c)[
            ...,
            [names.index(name) for name in BOUNDARY_DISTANCE_FEATURES],
        ]
        if np.min(distances) < -REFERENCE_EPS or np.max(distances) > 1.0 + REFERENCE_EPS:
            raise ValueError(
                "boundary distance features must be in [0, 1], got "
                f"min={float(np.min(distances))} max={float(np.max(distances))}"
            )
        for name in (
            "top_h",
            "top_T_inf_minus_T_ref",
            "bottom_T_fixed_minus_T_ref",
        ):
            if name not in names:
                raise ValueError(f"boundary schema dropped required BC scalar {name!r}")
        boundary_schema_samples[example.sample_id] = {
            "feature_count": len(names),
            "distance_min": float(np.min(distances)),
            "distance_max": float(np.max(distances)),
        }

        extent_bridge = build_configured_zero_delta_bridge(
            example,
            extent_feature_policy=EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST,
        )
        extent_names = tuple(extent_bridge.condition_feature_names)
        missing_extent = [
            name for name in EXTENT_BROADCAST_FEATURES if name not in extent_names
        ]
        if missing_extent:
            raise ValueError(f"extent broadcast missing features: {missing_extent}")
        extent_schema_samples[example.sample_id] = {
            "feature_count": len(extent_names),
            "extent_feature_names": list(EXTENT_BROADCAST_FEATURES),
        }

    isotropic_stats = training_normalization_stats(
        train_examples,
        coord_policy=COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC,
        extent_feature_policy=EXTENT_FEATURE_POLICY_LOG_EXTENT_BROADCAST,
    )
    isotropic_span_max = []
    for example in check_examples:
        bridge = build_configured_zero_delta_bridge(example)
        coords = np.asarray(normalize_coords(bridge.legacy_inputs.x_inp, isotropic_stats))
        spans = np.max(coords.reshape(-1, 3), axis=0) - np.min(
            coords.reshape(-1, 3),
            axis=0,
        )
        isotropic_span_max.append(float(np.max(spans)))
    max_span_error = max(abs(value - 2.0) for value in isotropic_span_max)
    if max_span_error > 1.0e-5:
        raise ValueError(
            "sample_local_isotropic should scale the longest sample axis to span 2; "
            f"max_span_error={max_span_error}"
        )
    return {
        "boundary_distance_replacement_samples": boundary_schema_samples,
        "log_extent_broadcast_samples": extent_schema_samples,
        "sample_local_isotropic_max_span_values": isotropic_span_max,
        "sample_local_isotropic_max_span_error": max_span_error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
