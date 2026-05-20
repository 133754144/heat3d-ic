#!/usr/bin/env python3
"""Check the dry-run plan for a frozen-v1-equivalent Heat3D v2 runbook."""

from __future__ import annotations

import copy
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


CONTROLLED_CONFIG = Path("configs/heat3d_v2/medium1024_gapA_controlled.yaml")
REFERENCE_CONFIG = Path("configs/heat3d_v2/frozen_v1_reference.yaml")
REPRODUCTION_RUN_NAME = "frozen_v1_equivalent_seed0"
REPRODUCTION_OUTPUT_DIR = f"output/heat3d_v2_runs/{REPRODUCTION_RUN_NAME}"
REQUIRED_DIAGNOSTIC_KINDS = {
    "baseline_comparison",
    "error_bins",
    "run_summary",
    "condition_diagnostics",
}


def main() -> int:
    controlled = load_v2_config(REPO_ROOT / CONTROLLED_CONFIG)
    reference = load_v2_config(REPO_ROOT / REFERENCE_CONFIG)
    reproduction_config = _build_frozen_v1_equivalent_config(controlled, reference)
    plan = build_v2_command_plan(reproduction_config, python_executable="python3")

    _assert_training_command(plan["training_command"], reference)
    _assert_diagnostics_plan(plan)
    _assert_prediction_paths(plan)

    _print_plan_summary(plan, reference)
    print("Heat3D v2 frozen-v1 reproduction plan check passed.")
    return 0


def _build_frozen_v1_equivalent_config(
    controlled: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(controlled)
    training = _mapping(reference, "training")
    reference_dataset = _mapping(reference, "dataset")

    config["description"] = (
        "In-memory frozen-v1-equivalent dry-run plan; source YAML is not modified."
    )

    dataset = config.setdefault("dataset", {})
    for field in ("name", "subset_path", "manifest_path"):
        if reference_dataset.get(field) is not None:
            dataset[field] = reference_dataset[field]

    optimizer = config.setdefault("optimizer", {})
    optimizer["name"] = training.get("optimizer") or "manual_full_batch_gradient_descent"
    optimizer["lr"] = training.get("lr")
    optimizer["lr_schedule"] = training.get("lr_schedule") or "constant"
    optimizer["warmup_epochs"] = 0
    optimizer["second_stage_epoch"] = None
    optimizer["second_stage_lr"] = None
    optimizer["weight_decay"] = None
    optimizer["gradient_clip_norm"] = None
    optimizer["multi_seed"] = []

    loss = config.setdefault("loss", {})
    if training.get("loss_mode") is not None:
        loss["mode"] = training["loss_mode"]
    if training.get("pseudo_negative_loss_type") is not None:
        loss["pseudo_negative_loss_type"] = training["pseudo_negative_loss_type"]
    if training.get("pseudo_negative_weight") is not None:
        loss["pseudo_negative_weight"] = training["pseudo_negative_weight"]
    if training.get("background_relative_weight") is not None:
        loss["background_relative_weight"] = training["background_relative_weight"]

    export = config.setdefault("export", {})
    export["run_name"] = REPRODUCTION_RUN_NAME
    export["output_dir"] = REPRODUCTION_OUTPUT_DIR
    export["save_final_predictions"] = True
    export["final_predictions_name"] = "predictions.npz"
    export["save_best_predictions"] = True
    export["best_predictions_name"] = "best_predictions.npz"
    export["selection_metric"] = export.get("selection_metric") or "valid_loss"

    diagnostics = config.setdefault("diagnostics", {})
    diagnostics["prediction_labels"] = ["final", "best"]
    diagnostics["run_baseline_comparison"] = True
    diagnostics["run_error_bins"] = True
    diagnostics["run_summary"] = True
    diagnostics["run_condition_diagnostics"] = True
    return config


def _assert_training_command(command: list[str], reference: dict[str, Any]) -> None:
    training = _mapping(reference, "training")
    subset = _option_value(command, "--subset")
    if "medium1024_gapA_full1024_v2" not in subset:
        raise AssertionError(f"training command has wrong subset: {subset}")

    _assert_option(command, "--loss-mode", training["loss_mode"])
    _assert_option(
        command,
        "--pseudo-negative-loss-type",
        training["pseudo_negative_loss_type"],
    )
    _assert_float_option(
        command,
        "--pseudo-negative-weight",
        float(training["pseudo_negative_weight"]),
    )
    _assert_float_option(
        command,
        "--background-relative-weight",
        float(training["background_relative_weight"]),
    )
    _assert_float_option(command, "--lr", float(training["lr"]))
    _assert_option(command, "--lr-schedule", training["lr_schedule"])

    for flag in ("--save-predictions", "--save-best-predictions"):
        if flag not in command:
            raise AssertionError(f"training command missing {flag}")
    _assert_option(command, "--best-predictions-name", "best_predictions.npz")


def _assert_diagnostics_plan(plan: dict[str, Any]) -> None:
    groups: dict[str, set[str]] = {}
    for entry in plan["diagnostics_commands"]:
        groups.setdefault(entry["prediction_label"], set()).add(entry["kind"])

    for label in ("final", "best"):
        kinds = groups.get(label)
        if kinds != REQUIRED_DIAGNOSTIC_KINDS:
            raise AssertionError(f"missing {label} diagnostics commands: {kinds}")

    expected_count = len(REQUIRED_DIAGNOSTIC_KINDS) * 2
    actual_count = len(plan["diagnostics_commands"])
    if actual_count != expected_count:
        raise AssertionError(
            f"expected {expected_count} diagnostics commands, got {actual_count}"
        )


def _assert_prediction_paths(plan: dict[str, Any]) -> None:
    expected_prediction = {
        "final": f"{REPRODUCTION_OUTPUT_DIR}/predictions.npz",
        "best": f"{REPRODUCTION_OUTPUT_DIR}/best_predictions.npz",
    }

    for entry in plan["diagnostics_commands"]:
        label = entry["prediction_label"]
        command = entry["command"]

        prediction_path = _optional_value(command, "--trained-predictions")
        if prediction_path is not None and prediction_path != expected_prediction[label]:
            raise AssertionError(
                f"{label} {entry['kind']} uses mismatched predictions path: "
                f"{prediction_path}"
            )

        prediction_label = _optional_value(command, "--prediction-label")
        if prediction_label is not None and prediction_label != label:
            raise AssertionError(
                f"{label} {entry['kind']} has mismatched prediction label: "
                f"{prediction_label}"
            )

        for flag in (
            "--output-json",
            "--output-md",
            "--baseline-comparison-json",
            "--error-bins-json",
        ):
            value = _optional_value(command, flag)
            if value is not None and f"_{label}." not in value:
                raise AssertionError(
                    f"{label} {entry['kind']} has mismatched {flag}: {value}"
                )


def _print_plan_summary(plan: dict[str, Any], reference: dict[str, Any]) -> None:
    training = _mapping(reference, "training")
    print(f"controlled config: {CONTROLLED_CONFIG}")
    print(f"reference config: {REFERENCE_CONFIG}")
    print(
        "frozen reference: "
        f"loss={training['loss_mode']} "
        f"pseudo_negative={training['pseudo_negative_loss_type']} "
        f"weight={training['pseudo_negative_weight']} "
        f"background_relative={training['background_relative_weight']} "
        f"lr={training['lr']} "
        f"schedule={training['lr_schedule']} "
        f"best_epoch={training['best_epoch']}"
    )
    print("training command:")
    print(f"  {shlex.join(plan['training_command'])}")
    print("prediction paths:")
    print(f"  final: {REPRODUCTION_OUTPUT_DIR}/predictions.npz")
    print(f"  best: {REPRODUCTION_OUTPUT_DIR}/best_predictions.npz")
    print("diagnostics command order:")
    for entry in plan["diagnostics_commands"]:
        print(
            "  "
            f"{entry['prediction_label']}:{entry['kind']}: "
            f"{shlex.join(entry['command'])}"
        )
    print(f"mapped fields: {len(plan['mapped_fields'])}")
    print(f"unmapped fields: {len(plan['unmapped_fields'])}")
    print(f"warnings: {len(plan['warnings'])}")
    print("No commands were executed.")


def _mapping(config: dict[str, Any], field: str) -> dict[str, Any]:
    value = config.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping field {field!r}")
    return value


def _assert_option(command: list[str], flag: str, expected: Any) -> None:
    actual = _option_value(command, flag)
    if actual != str(expected):
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


def _assert_float_option(command: list[str], flag: str, expected: float) -> None:
    actual = _option_value(command, flag)
    if abs(float(actual) - expected) > 1e-12:
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


def _option_value(command: list[str], flag: str) -> str:
    value = _optional_value(command, flag)
    if value is None:
        raise AssertionError(f"command missing {flag}")
    return value


def _optional_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    try:
        return command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"command flag {flag} has no value") from exc


if __name__ == "__main__":
    raise SystemExit(main())
