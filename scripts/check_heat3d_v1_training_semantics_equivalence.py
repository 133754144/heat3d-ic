#!/usr/bin/env python3
"""Check that extracted Heat3D v1 legacy helpers preserve old formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v1_normalization import (  # noqa: E402
    legacy_train_only_stats,
    normalize_condition,
    normalize_coords,
    normalize_target_delta,
    normalized_delta_to_raw,
    recover_raw_condition,
    recover_temperature_from_normalized_delta,
)
from rigno.heat3d_v1_training_semantics import (  # noqa: E402
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
    parser.add_argument("--atol", type=float, default=1.0e-12)
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

    max_abs_diff = max(diffs.values(), default=0.0)
    payload = {
        "script": Path(__file__).name,
        "non_execution": "no training, no evaluation, no artifact writes",
        "subset": str(sample_root),
        "train_examples": [example.sample_id for example in train_examples],
        "checked_examples": [example.sample_id for example in check_examples],
        "semantics": legacy_training_semantics_manifest(),
        "max_abs_diff": max_abs_diff,
        "passed": bool(max_abs_diff <= args.atol),
        "diffs": diffs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
