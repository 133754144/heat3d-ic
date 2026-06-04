"""Smoke-check Heat3D v2 M1 LR schedule ablation config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CONFIG_DIR = REPO_DIR / "configs" / "heat3d_v2"
BASE_CONFIG = CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml"
SCHEDULE_CONFIG = (
    CONFIG_DIR
    / "frozen_v1_e050_adamw_m1_batch_lr3e4_decay_e5_to1e4_seed0.yaml"
)
ALLOWED_DIFF_PATHS = (
    ("description",),
    ("optimizer", "lr_schedule"),
    ("optimizer", "second_stage_epoch"),
    ("optimizer", "second_stage_lr"),
    ("export", "output_dir"),
    ("export", "run_name"),
)


def main() -> int:
    base = load_v2_config(BASE_CONFIG)
    config = load_v2_config(SCHEDULE_CONFIG)
    _check_expected_config(config)
    _check_only_expected_diffs(base, config)
    _check_command(config)
    _check_lr_for_epoch()
    print("Heat3D v2 M1 LR schedule config smoke passed.")
    return 0


def _check_expected_config(config: dict) -> None:
    if config["optimizer"]["lr"] != 3.0e-4:
        raise AssertionError("optimizer.lr must remain 3.0e-4")
    if config["optimizer"]["lr_schedule"] != "second_stage":
        raise AssertionError("optimizer.lr_schedule must be second_stage")
    if config["optimizer"]["second_stage_epoch"] != 5:
        raise AssertionError("optimizer.second_stage_epoch must be 5")
    if config["optimizer"]["second_stage_lr"] != 1.0e-4:
        raise AssertionError("optimizer.second_stage_lr must be 1.0e-4")
    if config["run"].get("train_metrics_schedule") != "half_and_final":
        raise AssertionError("run.train_metrics_schedule must be half_and_final")
    if config["run"].get("grad_norm_report_every") != 10:
        raise AssertionError("run.grad_norm_report_every must be 10")


def _check_only_expected_diffs(base: dict, config: dict) -> None:
    base_copy = deepcopy(base)
    config_copy = deepcopy(config)
    for path in ALLOWED_DIFF_PATHS:
        _delete_path(base_copy, path)
        _delete_path(config_copy, path)
    if base_copy != config_copy:
        raise AssertionError("schedule config differs from lr=3e-4 base outside allowed fields")


def _delete_path(mapping: dict, path: tuple[str, ...]) -> None:
    current = mapping
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(path[-1], None)


def _check_command(config: dict) -> None:
    command = build_training_command(config, python_executable="python")
    _assert_float_option(command, "--lr", 3.0e-4)
    _assert_option(command, "--lr-schedule", "second_stage")
    _assert_option(command, "--second-stage-epoch", "5")
    _assert_float_option(command, "--second-stage-lr", 1.0e-4)
    _assert_option(command, "--epochs", "50")
    _assert_option(command, "--optimizer", "adamw")
    _assert_option(command, "--batch-size", "4")
    _assert_option(command, "--train-metrics-schedule", "half_and_final")
    _assert_option(command, "--grad-norm-report-every", "10")
    if "--save-predictions" not in command:
        raise AssertionError("training command must include --save-predictions")
    if "--save-best-predictions" not in command:
        raise AssertionError("training command must include --save-best-predictions")


def _check_lr_for_epoch() -> None:
    lr_config = {
        "lr": 3.0e-4,
        "lr_schedule": "second_stage",
        "warmup_epochs": 0,
        "min_lr": 1.0e-5,
        "second_stage_epoch": 5,
        "second_stage_lr": 1.0e-4,
    }
    expected = {
        1: 3.0e-4,
        4: 3.0e-4,
        5: 1.0e-4,
        50: 1.0e-4,
    }
    for epoch, expected_lr in expected.items():
        actual = runner._lr_for_epoch(epoch, 50, lr_config)
        if abs(actual - expected_lr) > 1e-12:
            raise AssertionError(
                f"epoch {epoch}: expected lr {expected_lr}, got {actual}"
            )


def _assert_option(command: list[str], option: str, expected: str) -> None:
    actual = _option_value(command, option)
    if actual != expected:
        raise AssertionError(f"{option}: expected {expected!r}, got {actual!r}")


def _assert_float_option(command: list[str], option: str, expected: float) -> None:
    actual = float(_option_value(command, option))
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{option}: expected {expected}, got {actual}")


def _option_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise AssertionError(f"missing command option {option}") from exc
    try:
        return command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"command option {option} is missing a value") from exc


if __name__ == "__main__":
    raise SystemExit(main())
