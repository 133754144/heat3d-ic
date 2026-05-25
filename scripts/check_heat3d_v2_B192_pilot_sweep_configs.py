"""Dry-run smoke for Heat3D v2 B192 pilot sweep configs."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CONFIG_DIR = REPO_DIR / "configs" / "heat3d_v2"
CASES = (
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_B192_base_mse_lr1e4_seed0.yaml",
        "lr": 1.0e-4,
        "loss_mode": "mse",
        "output_dir": "output/heat3d_v2_runs/m1_B192_base_mse_lr1e4_seed0",
        "run_name": "m1_B192_base_mse_lr1e4_seed0",
    },
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_B192_base_mse_lr3e5_seed0.yaml",
        "lr": 3.0e-5,
        "loss_mode": "mse",
        "output_dir": "output/heat3d_v2_runs/m1_B192_base_mse_lr3e5_seed0",
        "run_name": "m1_B192_base_mse_lr3e5_seed0",
    },
    {
        "path": CONFIG_DIR / "frozen_v1_e050_adamw_m1_B192_full_lr1e4_seed0.yaml",
        "lr": 1.0e-4,
        "loss_mode": "background_pseudo_negative",
        "output_dir": "output/heat3d_v2_runs/m1_B192_full_lr1e4_seed0",
        "run_name": "m1_B192_full_lr1e4_seed0",
    },
)


def main() -> int:
    for case in CASES:
        config = load_v2_config(case["path"])
        _check_config(config, case)
        _check_command(build_training_command(config, python_executable="python"), case)
    print("Heat3D v2 B192 pilot sweep config smoke passed.")
    return 0


def _check_config(config: dict, case: dict) -> None:
    run = config["run"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    export = config["export"]
    if run["batch_size"] != 192:
        raise AssertionError(f"{case['path']}: batch_size must be 192")
    if run["validation_batch_size"] != 192:
        raise AssertionError(f"{case['path']}: validation_batch_size must be 192")
    if run["prediction_batch_size"] != 192:
        raise AssertionError(f"{case['path']}: prediction_batch_size must be 192")
    if run["epochs"] != 50:
        raise AssertionError(f"{case['path']}: epochs must be 50")
    if optimizer["name"] != "adamw":
        raise AssertionError(f"{case['path']}: optimizer must be adamw")
    if abs(float(optimizer["lr"]) - case["lr"]) > 1e-12:
        raise AssertionError(f"{case['path']}: unexpected lr")
    if float(optimizer["weight_decay"]) != 1.0e-4:
        raise AssertionError(f"{case['path']}: weight_decay must remain 1e-4")
    if float(optimizer["gradient_clip_norm"]) != 1.0:
        raise AssertionError(f"{case['path']}: gradient_clip_norm must remain 1.0")
    if loss["mode"] != case["loss_mode"]:
        raise AssertionError(f"{case['path']}: unexpected loss mode")
    if export["output_dir"] != case["output_dir"]:
        raise AssertionError(f"{case['path']}: unexpected output_dir")
    if export["run_name"] != case["run_name"]:
        raise AssertionError(f"{case['path']}: unexpected run_name")
    if export["selection_metric"] != "valid_loss":
        raise AssertionError(f"{case['path']}: selection_metric must be valid_loss")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{case['path']}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{case['path']}: save_best_predictions must be true")


def _check_command(command: list[str], case: dict) -> None:
    _assert_option(command, "--batch-size", "192")
    _assert_option(command, "--validation-batch-size", "192")
    _assert_option(command, "--prediction-batch-size", "192")
    _assert_option(command, "--epochs", "50")
    _assert_option(command, "--optimizer", "adamw")
    _assert_float_option(command, "--lr", case["lr"])
    _assert_option(command, "--loss-mode", case["loss_mode"])
    _assert_option(command, "--output-dir", case["output_dir"])
    _assert_option(command, "--selection-metric", "valid_loss")
    if "--save-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must include --save-predictions")
    if "--save-best-predictions" not in command:
        raise AssertionError(f"{case['path']}: command must include --save-best-predictions")


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
