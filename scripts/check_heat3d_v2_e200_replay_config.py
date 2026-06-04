#!/usr/bin/env python3
"""Smoke-check the Heat3D v2 e200 replay config without running training."""

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


OLD_CONFIG = Path(
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed0.yaml"
)
REPLAY_CONFIG = Path(
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_replay_seed0.yaml"
)
EXPECTED_RUN_NAME = "m1_B192_base_mse_lr3e4_e200_stratified_replay_seed0"
EXPECTED_OUTPUT_DIR = "output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_e200_stratified_replay_seed0"


def main() -> int:
    old_config = load_v2_config(REPO_ROOT / OLD_CONFIG)
    replay_config = load_v2_config(REPO_ROOT / REPLAY_CONFIG)
    command = build_training_command(replay_config, python_executable="python")

    normalized_old = _normalize_for_replay_compare(old_config)
    normalized_replay = _normalize_for_replay_compare(replay_config)
    if normalized_old != normalized_replay:
        raise AssertionError("replay config differs from old e200 beyond description/run_name/output_dir")

    split_map_path = REPO_ROOT / replay_config["dataset"]["split_map_path"]
    if not split_map_path.is_file():
        raise AssertionError(f"split_map_path missing: {split_map_path}")

    _assert_equal("run.epochs", replay_config["run"]["epochs"], 200)
    _assert_equal("run.batch_size", replay_config["run"]["batch_size"], 192)
    _assert_equal("run.validation_batch_size", replay_config["run"]["validation_batch_size"], 192)
    _assert_equal("run.prediction_batch_size", replay_config["run"]["prediction_batch_size"], 192)
    _assert_equal("optimizer.seed", replay_config["optimizer"]["seed"], 0)
    _assert_equal("optimizer.name", replay_config["optimizer"]["name"], "adamw")
    _assert_equal("loss.mode", replay_config["loss"]["mode"], "mse")
    _assert_equal("export.run_name", replay_config["export"]["run_name"], EXPECTED_RUN_NAME)
    _assert_equal("export.output_dir", replay_config["export"]["output_dir"], EXPECTED_OUTPUT_DIR)

    _assert_option(command, "--epochs", "200")
    _assert_option(command, "--batch-size", "192")
    _assert_option(command, "--validation-batch-size", "192")
    _assert_option(command, "--prediction-batch-size", "192")
    _assert_option(command, "--seed", "0")
    _assert_option(command, "--optimizer", "adamw")
    _assert_option(command, "--lr", "0.0003")
    _assert_option(command, "--loss-mode", "mse")
    _assert_option(command, "--output-dir", EXPECTED_OUTPUT_DIR)
    if "--shuffle-train-batches" not in command:
        raise AssertionError("expected --shuffle-train-batches")
    if "--save-predictions" not in command or "--save-best-predictions" not in command:
        raise AssertionError("expected final/best prediction export flags")

    print("Heat3D v2 e200 replay config smoke passed.")
    return 0


def _normalize_for_replay_compare(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("description", None)
    normalized["export"].pop("run_name", None)
    normalized["export"].pop("output_dir", None)
    return normalized


def _assert_equal(field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{field} expected {expected!r}, got {actual!r}")


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
