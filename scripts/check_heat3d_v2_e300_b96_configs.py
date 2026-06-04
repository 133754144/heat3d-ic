#!/usr/bin/env python3
"""Smoke-check the next Heat3D v2 e300 and B96 control configs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CONFIG_SPECS = {
    "M1 B192 e300": {
        "path": Path(
            "configs/heat3d_v2/"
            "frozen_v1_e300_adamw_m1_B192_base_mse_lr3e4_stratified_seed0.yaml"
        ),
        "epochs": 300,
        "batch_size": 192,
        "run_name": "m1_B192_base_mse_lr3e4_e300_stratified_seed0",
        "output_dir": "output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_e300_stratified_seed0",
    },
    "M1 B96 e100": {
        "path": Path(
            "configs/heat3d_v2/"
            "frozen_v1_e100_adamw_m1_B96_base_mse_lr3e4_stratified_seed0.yaml"
        ),
        "epochs": 100,
        "batch_size": 96,
        "run_name": "m1_B96_base_mse_lr3e4_e100_stratified_seed0",
        "output_dir": "output/heat3d_v2_runs/m1_B96_base_mse_lr3e4_e100_stratified_seed0",
    },
}


def main() -> int:
    seen_names: set[str] = set()
    seen_outputs: set[str] = set()

    for label, spec in CONFIG_SPECS.items():
        config = load_v2_config(REPO_ROOT / spec["path"])
        command = build_training_command(config, python_executable="python")

        dataset = config["dataset"]
        model = config["model"]
        optimizer = config["optimizer"]
        loss = config["loss"]
        run = config["run"]
        export = config["export"]

        split_map_path = REPO_ROOT / dataset["split_map_path"]
        if not split_map_path.is_file():
            raise AssertionError(f"{label}: split_map_path missing: {split_map_path}")

        _assert_equal(label, "dataset.name", dataset["name"], "medium1024_gapA_full1024_v2")
        _assert_equal(label, "split_source", dataset["split_source"], "split_map")
        _assert_equal(label, "loss.mode", loss["mode"], "mse")
        _assert_equal(label, "optimizer.name", optimizer["name"], "adamw")
        _assert_float(label, "optimizer.lr", optimizer["lr"], 3.0e-4)
        _assert_float(label, "optimizer.weight_decay", optimizer["weight_decay"], 1.0e-4)
        _assert_float(label, "optimizer.gradient_clip_norm", optimizer["gradient_clip_norm"], 1.0)
        _assert_equal(label, "optimizer.lr_schedule", optimizer["lr_schedule"], "constant")
        _assert_equal(label, "run.epochs", run["epochs"], spec["epochs"])
        _assert_equal(label, "run.batch_size", run["batch_size"], spec["batch_size"])
        _assert_equal(label, "run.validation_batch_size", run["validation_batch_size"], spec["batch_size"])
        _assert_equal(label, "run.prediction_batch_size", run["prediction_batch_size"], spec["batch_size"])
        _assert_equal(label, "export.selection_metric", export["selection_metric"], "valid_loss")
        _assert_equal(label, "export.run_name", export["run_name"], spec["run_name"])
        _assert_equal(label, "export.output_dir", export["output_dir"], spec["output_dir"])
        _assert_model(label, model)

        _assert_option(command, "--split-map", dataset["split_map_path"])
        _assert_option(command, "--epochs", spec["epochs"])
        _assert_option(command, "--batch-size", spec["batch_size"])
        _assert_option(command, "--validation-batch-size", spec["batch_size"])
        _assert_option(command, "--prediction-batch-size", spec["batch_size"])
        _assert_option(command, "--optimizer", "adamw")
        _assert_option(command, "--loss-mode", "mse")
        _assert_option(command, "--output-dir", spec["output_dir"])
        _assert_option(command, "--selection-metric", "valid_loss")
        if "--save-predictions" not in command or "--save-best-predictions" not in command:
            raise AssertionError(f"{label}: final/best prediction export flags missing")

        if export["run_name"] in seen_names:
            raise AssertionError(f"{label}: duplicate run_name {export['run_name']}")
        if export["output_dir"] in seen_outputs:
            raise AssertionError(f"{label}: duplicate output_dir {export['output_dir']}")
        seen_names.add(export["run_name"])
        seen_outputs.add(export["output_dir"])

        print(
            f"{label}: epochs={run['epochs']} batch={run['batch_size']} "
            f"run_name={export['run_name']}"
        )

    print("Heat3D v2 e300/B96 config smoke passed.")
    return 0


def _assert_model(label: str, model: dict[str, Any]) -> None:
    expected = {
        "architecture": "RIGNO",
        "node_latent_size": 64,
        "edge_latent_size": 64,
        "processor_steps": 4,
        "mlp_hidden_layers": 2,
    }
    for field, expected_value in expected.items():
        _assert_equal(label, f"model.{field}", model[field], expected_value)


def _assert_option(command: list[str], flag: str, expected: Any) -> None:
    if flag not in command:
        raise AssertionError(f"missing command flag {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise AssertionError(f"flag {flag} is missing a value")
    actual = command[index + 1]
    if str(actual) != str(expected):
        raise AssertionError(f"{flag}: expected {expected!r}, got {actual!r}")


def _assert_equal(label: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {field} expected {expected!r}, got {actual!r}")


def _assert_float(label: str, field: str, actual: Any, expected: float, tolerance: float = 1.0e-12) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: {field} expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
