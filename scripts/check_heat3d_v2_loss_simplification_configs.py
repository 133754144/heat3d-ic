"""Dry-run smoke for Heat3D v2 M1 loss simplification configs."""

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
BASE_CONFIG = CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml"
CASES = (
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr3e4_base_mse_seed0.yaml",
        "loss_mode": "mse",
        "hotspot_weight": 0.0,
        "output_dir": "output/heat3d_v2_runs/m1_batch_e50_lr3e4_base_mse_seed0",
        "run_name": "m1_batch_e50_lr3e4_base_mse_seed0",
    },
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_batch_lr3e4_base_mse_hotspot_seed0.yaml",
        "loss_mode": "background_hotspot",
        "hotspot_weight": 0.02,
        "output_dir": "output/heat3d_v2_runs/m1_batch_e50_lr3e4_base_mse_hotspot_seed0",
        "run_name": "m1_batch_e50_lr3e4_base_mse_hotspot_seed0",
    },
)
ALLOWED_DIFF_PATHS = (
    ("description",),
    ("loss", "mode"),
    ("loss", "background_weight"),
    ("loss", "hotspot_weight"),
    ("loss", "background_l1_weight"),
    ("loss", "background_bias_weight"),
    ("loss", "background_over_weight"),
    ("loss", "background_relative_weight"),
    ("loss", "pseudo_negative_weight"),
    ("export", "output_dir"),
    ("export", "run_name"),
)
ZERO_WEIGHT_FIELDS = (
    "background_l1_weight",
    "background_bias_weight",
    "background_over_weight",
    "background_relative_weight",
    "pseudo_negative_weight",
)


def main() -> int:
    base = load_v2_config(BASE_CONFIG)
    for case in CASES:
        config = load_v2_config(case["path"])
        _check_expected_values(config, case)
        _check_only_expected_diffs(base, config)
        _check_command(config, case)
    print("Heat3D v2 loss simplification config smoke passed.")
    return 0


def _check_expected_values(config: dict, case: dict) -> None:
    if config["optimizer"]["lr"] != 3.0e-4:
        raise AssertionError(f"{case['path']}: lr must remain 3e-4")
    if config["run"]["batch_size"] != 4:
        raise AssertionError(f"{case['path']}: batch_size must remain 4")
    if config["run"]["epochs"] != 50:
        raise AssertionError(f"{case['path']}: epochs must remain 50")
    loss = config["loss"]
    if loss["mode"] != case["loss_mode"]:
        raise AssertionError(f"{case['path']}: unexpected loss mode")
    if float(loss["hotspot_weight"]) != case["hotspot_weight"]:
        raise AssertionError(f"{case['path']}: unexpected hotspot_weight")
    if float(loss["background_weight"]) != 0.0:
        raise AssertionError(f"{case['path']}: background_weight must be zero")
    for field in ZERO_WEIGHT_FIELDS:
        if float(loss[field]) != 0.0:
            raise AssertionError(f"{case['path']}: {field} must be zero")
    if config["export"]["output_dir"] != case["output_dir"]:
        raise AssertionError(f"{case['path']}: unexpected output_dir")
    if config["export"]["run_name"] != case["run_name"]:
        raise AssertionError(f"{case['path']}: unexpected run_name")


def _check_only_expected_diffs(base: dict, config: dict) -> None:
    base_copy = deepcopy(base)
    config_copy = deepcopy(config)
    for path in ALLOWED_DIFF_PATHS:
        _delete_path(base_copy, path)
        _delete_path(config_copy, path)
    if base_copy != config_copy:
        raise AssertionError("loss simplification config differs from lr=3e-4 base outside allowed fields")


def _check_command(config: dict, case: dict) -> None:
    command = build_training_command(config, python_executable="python")
    _assert_option(command, "--loss-mode", case["loss_mode"])
    _assert_float_option(command, "--background-weight", 0.0)
    _assert_float_option(command, "--hotspot-weight", case["hotspot_weight"])
    _assert_float_option(command, "--background-l1-weight", 0.0)
    _assert_float_option(command, "--background-bias-weight", 0.0)
    _assert_float_option(command, "--background-over-weight", 0.0)
    _assert_float_option(command, "--background-relative-weight", 0.0)
    _assert_float_option(command, "--pseudo-negative-weight", 0.0)
    _assert_option(command, "--batch-size", "4")
    _assert_option(command, "--epochs", "50")
    _assert_option(command, "--optimizer", "adamw")
    _assert_float_option(command, "--lr", 3.0e-4)
    _assert_option(command, "--output-dir", case["output_dir"])
    if "--save-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must include --save-predictions")
    if "--save-best-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must include --save-best-predictions")


def _delete_path(mapping: dict, path: tuple[str, ...]) -> None:
    current = mapping
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(path[-1], None)


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
