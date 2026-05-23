"""Smoke-check Heat3D v2 M1 lower-lr ablation configs.

This check is read-only. It validates the YAML configs, dry-runs command
construction, and confirms only the intended fields differ from the lr=1e-3
M1 e50 baseline.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CONFIG_DIR = REPO_DIR / "configs" / "heat3d_v2"
BASELINE_CONFIG = CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_e50.yaml"
CASES = (
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml",
        "description": "M1 AdamW e50 seed0 lower-lr ablation: lr=3e-4.",
        "lr": 3.0e-4,
        "output_dir": "output/heat3d_v2_runs/m1_batch_e50_lr3e4_seed0",
        "run_name": "m1_batch_e50_lr3e4_seed0",
    },
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr1e4_seed0.yaml",
        "description": "M1 AdamW e50 seed0 lower-lr ablation: lr=1e-4.",
        "lr": 1.0e-4,
        "output_dir": "output/heat3d_v2_runs/m1_batch_e50_lr1e4_seed0",
        "run_name": "m1_batch_e50_lr1e4_seed0",
    },
)
ALLOWED_DIFF_PATHS = (
    ("description",),
    ("optimizer", "lr"),
    ("export", "output_dir"),
    ("export", "run_name"),
    ("run", "train_metrics_schedule"),
    ("run", "grad_norm_report_every"),
)


def main() -> int:
    baseline = load_v2_config(BASELINE_CONFIG)
    for case in CASES:
        config = load_v2_config(case["path"])
        _check_expected_values(config, case)
        _check_only_expected_diffs(baseline, config)
        _check_training_command(config, case)

    print("Heat3D v2 M1 lower-lr configs smoke passed.")
    return 0


def _check_expected_values(config: dict, case: dict) -> None:
    if config["description"] != case["description"]:
        raise AssertionError(f"{case['path']}: unexpected description")
    if float(config["optimizer"]["lr"]) != case["lr"]:
        raise AssertionError(f"{case['path']}: unexpected optimizer.lr")
    if config["export"]["output_dir"] != case["output_dir"]:
        raise AssertionError(f"{case['path']}: unexpected export.output_dir")
    if config["export"]["run_name"] != case["run_name"]:
        raise AssertionError(f"{case['path']}: unexpected export.run_name")
    if config["run"].get("train_metrics_schedule") != "half_and_final":
        raise AssertionError(f"{case['path']}: train_metrics_schedule must be explicit")
    if config["run"].get("grad_norm_report_every") != 10:
        raise AssertionError(f"{case['path']}: grad_norm_report_every must be 10")


def _check_only_expected_diffs(baseline: dict, config: dict) -> None:
    baseline_copy = deepcopy(baseline)
    config_copy = deepcopy(config)
    for path in ALLOWED_DIFF_PATHS:
        _delete_path(baseline_copy, path)
        _delete_path(config_copy, path)
    if baseline_copy != config_copy:
        raise AssertionError("lower-lr config differs from baseline outside allowed fields")


def _delete_path(mapping: dict, path: tuple[str, ...]) -> None:
    current = mapping
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(path[-1], None)


def _check_training_command(config: dict, case: dict) -> None:
    command = build_training_command(config, python_executable="python")
    _assert_option(command, "--epochs", "50")
    _assert_option(command, "--optimizer", "adamw")
    _assert_option(command, "--batch-size", "4")
    _assert_option(command, "--train-metrics-schedule", "half_and_final")
    _assert_option(command, "--grad-norm-report-every", "10")
    lr_value = _option_value(command, "--lr")
    if abs(float(lr_value) - case["lr"]) > 1e-12:
        raise AssertionError(f"{case['path']}: command has unexpected lr {lr_value!r}")
    if "--save-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must save final predictions")
    if "--save-best-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must save best predictions")


def _assert_option(command: list[str], option: str, expected: str) -> None:
    actual = _option_value(command, option)
    if actual != expected:
        raise AssertionError(f"{option}: expected {expected!r}, got {actual!r}")


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
