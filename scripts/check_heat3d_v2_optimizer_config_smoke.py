#!/usr/bin/env python3
"""Smoke-check Heat3D v2 optimizer ablation configs map to dry-run commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


ABLATIONS = {
    "A1": {
        "path": Path("configs/heat3d_v2/frozen_v1_e050_adam_lr1e3_seed0.yaml"),
        "optimizer": "adam",
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "output_dir": "output/heat3d_v2_runs/adam_lr1e3_seed0",
    },
    "A2": {
        "path": Path("configs/heat3d_v2/frozen_v1_e050_adamw_lr1e3_wd1e4_seed0.yaml"),
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "output_dir": "output/heat3d_v2_runs/adamw_lr1e3_wd1e4_seed0",
    },
    "A3": {
        "path": Path("configs/heat3d_v2/frozen_v1_e050_adamw_lr3e4_wd1e4_seed0.yaml"),
        "optimizer": "adamw",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "output_dir": "output/heat3d_v2_runs/adamw_lr3e4_wd1e4_seed0",
    },
}

BASELINE_PATH = Path("configs/heat3d_v2/frozen_v1_best_e050_seed0.yaml")
REQUIRED_DIAGNOSTIC_KINDS = {
    "baseline_comparison",
    "error_bins",
    "run_summary",
    "condition_diagnostics",
    "field_shape_diagnostics",
}


def main() -> int:
    baseline = load_v2_config(REPO_ROOT / BASELINE_PATH)
    baseline_dataset = baseline["dataset"]
    baseline_model = baseline["model"]
    baseline_loss = baseline["loss"]
    baseline_run = baseline["run"]
    expected_dataset_name = baseline_dataset["name"]

    for label, spec in ABLATIONS.items():
        config = load_v2_config(REPO_ROOT / spec["path"])
        _assert_same("dataset", config["dataset"], baseline_dataset)
        _assert_same("model", config["model"], baseline_model)
        _assert_same("loss", config["loss"], baseline_loss)
        if config["run"]["epochs"] != 50 or config["run"]["mode"] != "controlled":
            raise AssertionError(f"{label}: expected controlled e50 run")
        if config["run"]["report_every"] != baseline_run["report_every"]:
            raise AssertionError(f"{label}: report_every drifted from baseline")
        if config["export"]["selection_metric"] != "valid_loss":
            raise AssertionError(f"{label}: selection metric must be valid_loss")

        plan = build_v2_command_plan(config, python_executable="python3")
        command = plan["training_command"]
        _assert_option(command, "--optimizer", spec["optimizer"])
        _assert_float_option(command, "--lr", spec["lr"])
        _assert_float_option(command, "--weight-decay", spec["weight_decay"])
        _assert_float_option(command, "--gradient-clip-norm", spec["gradient_clip_norm"])
        _assert_option(command, "--lr-schedule", "constant")
        _assert_option(command, "--seed", "0")
        _assert_option(command, "--epochs", "50")
        _assert_option(command, "--selection-metric", "valid_loss")
        _assert_option(command, "--output-dir", spec["output_dir"])
        _assert_option(command, "--loss-mode", baseline_loss["mode"])
        _assert_option(command, "--pseudo-negative-loss-type", "relative_l1")
        _assert_float_option(command, "--pseudo-negative-weight", 0.10)
        _assert_float_option(command, "--background-relative-weight", 0.10)
        if "--save-predictions" not in command or "--save-best-predictions" not in command:
            raise AssertionError(f"{label}: expected final and best prediction export")

        _assert_final_best_diagnostics(label, plan)
        print(
            f"{label}: dataset={expected_dataset_name} optimizer={spec['optimizer']} "
            f"lr={spec['lr']} weight_decay={spec['weight_decay']} "
            f"gradient_clip_norm={spec['gradient_clip_norm']}"
        )

    print("Heat3D v2 optimizer config smoke passed.")
    return 0


def _assert_same(name: str, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual != expected:
        raise AssertionError(f"{name} differs from strict A0 baseline")


def _assert_final_best_diagnostics(label: str, plan: dict[str, Any]) -> None:
    groups: dict[str, set[str]] = {}
    for entry in plan["diagnostics_commands"]:
        groups.setdefault(entry["prediction_label"], set()).add(entry["kind"])
    for prediction_label in ("final", "best"):
        kinds = groups.get(prediction_label)
        if kinds != REQUIRED_DIAGNOSTIC_KINDS:
            raise AssertionError(f"{label}: missing {prediction_label} diagnostics {kinds}")


def _assert_option(command: list[str], flag: str, expected: Any) -> None:
    actual = _option_value(command, flag)
    if actual != str(expected):
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


def _assert_float_option(command: list[str], flag: str, expected: float) -> None:
    actual = _option_value(command, flag)
    if abs(float(actual) - expected) > 1e-12:
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


def _option_value(command: list[str], flag: str) -> str:
    if flag not in command:
        raise AssertionError(f"command missing {flag}")
    index = command.index(flag)
    try:
        return command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"command flag {flag} has no value") from exc


if __name__ == "__main__":
    raise SystemExit(main())
