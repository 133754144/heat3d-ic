#!/usr/bin/env python3
"""Smoke-check Heat3D v2 deterministic e5 configs without running training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CONFIG_A = Path(
    "configs/heat3d_v2/"
    "frozen_v1_e005_adamw_m1_B192_base_mse_lr3e4_stratified_determinism_seed0.yaml"
)
CONFIG_B = Path(
    "configs/heat3d_v2/"
    "frozen_v1_e005_adamw_m1_B192_base_mse_lr3e4_stratified_determinism_repeat_seed0.yaml"
)


def main() -> int:
    config_a = load_v2_config(REPO_ROOT / CONFIG_A)
    config_b = load_v2_config(REPO_ROOT / CONFIG_B)
    command_a = build_training_command(config_a, python_executable="python")
    command_b = build_training_command(config_b, python_executable="python")

    for label, config, command in (
        ("determinism", config_a, command_a),
        ("determinism_repeat", config_b, command_b),
    ):
        split_map_path = REPO_ROOT / config["dataset"]["split_map_path"]
        if not split_map_path.is_file():
            raise AssertionError(f"{label}: split_map_path missing: {split_map_path}")
        _assert_equal(label, "run.epochs", config["run"]["epochs"], 5)
        _assert_equal(label, "run.batch_size", config["run"]["batch_size"], 192)
        _assert_equal(label, "run.validation_batch_size", config["run"]["validation_batch_size"], 192)
        _assert_equal(label, "run.prediction_batch_size", config["run"]["prediction_batch_size"], 192)
        _assert_equal(label, "optimizer.seed", config["optimizer"]["seed"], 0)
        _assert_equal(label, "optimizer.name", config["optimizer"]["name"], "adamw")
        _assert_equal(label, "loss.mode", config["loss"]["mode"], "mse")
        _assert_equal(label, "run.train_metrics_schedule", config["run"]["train_metrics_schedule"], "final_only")

        _assert_option(command, "--epochs", "5")
        _assert_option(command, "--batch-size", "192")
        _assert_option(command, "--validation-batch-size", "192")
        _assert_option(command, "--prediction-batch-size", "192")
        _assert_option(command, "--seed", "0")
        _assert_option(command, "--optimizer", "adamw")
        _assert_option(command, "--lr", "0.0003")
        _assert_option(command, "--loss-mode", "mse")
        _assert_option(command, "--train-metrics-schedule", "final_only")
        if "--shuffle-train-batches" not in command:
            raise AssertionError(f"{label}: expected --shuffle-train-batches")
        if "--save-predictions" not in command or "--save-best-predictions" not in command:
            raise AssertionError(f"{label}: expected final/best prediction export flags")

    normalized_a = _normalize_for_repeat_compare(config_a)
    normalized_b = _normalize_for_repeat_compare(config_b)
    if normalized_a != normalized_b:
        raise AssertionError("deterministic smoke configs differ beyond description/run_name/output_dir")

    if config_a["export"]["run_name"] == config_b["export"]["run_name"]:
        raise AssertionError("deterministic smoke configs must use distinct run_name")
    if config_a["export"]["output_dir"] == config_b["export"]["output_dir"]:
        raise AssertionError("deterministic smoke configs must use distinct output_dir")

    print("determinism: e5 B192 command smoke ok")
    print("determinism_repeat: e5 B192 command smoke ok")
    print("Heat3D v2 determinism smoke configs passed.")
    return 0


def _normalize_for_repeat_compare(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("description", None)
    normalized["export"].pop("run_name", None)
    normalized["export"].pop("output_dir", None)
    return normalized


def _assert_equal(label: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {field} expected {expected!r}, got {actual!r}")


def _assert_option(command: list[str], flag: str, expected: str) -> None:
    if flag not in command:
        raise AssertionError(f"missing command flag {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise AssertionError(f"flag {flag} is missing a value")
    actual = command[index + 1]
    if str(actual) != str(expected):
        raise AssertionError(f"{flag}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
