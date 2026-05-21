#!/usr/bin/env python3
"""Smoke-check Heat3D v2 batch dry-run configs map to planned commands."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


BATCH_CONFIGS = {
    "M1-lite": {
        "path": Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1lite_batch_dryrun.yaml"),
        "batch_size": 16,
        "micro_batch_size": 4,
        "validation_batch_size": 16,
        "prediction_batch_size": 16,
        "model": {
            "node_latent_size": 32,
            "edge_latent_size": 32,
            "processor_steps": 3,
            "mlp_hidden_layers": 2,
        },
    },
    "M1": {
        "path": Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_dryrun.yaml"),
        "batch_size": 4,
        "micro_batch_size": 2,
        "validation_batch_size": 4,
        "prediction_batch_size": 4,
        "model": {
            "node_latent_size": 64,
            "edge_latent_size": 64,
            "processor_steps": 4,
            "mlp_hidden_layers": 2,
        },
    },
}

REQUIRED_WARNING_SNIPPETS = (
    "run.micro_batch_size is a future gradient-accumulation field",
)


def main() -> int:
    for label, spec in BATCH_CONFIGS.items():
        config = load_v2_config(REPO_ROOT / spec["path"])
        plan = build_v2_command_plan(config, python_executable="python3")
        command = plan["training_command"]

        _assert_model_command(command, spec["model"])
        _assert_option(command, "--optimizer", "adamw")
        _assert_float_option(command, "--lr", 1.0e-3)
        _assert_float_option(command, "--weight-decay", 1.0e-4)
        _assert_float_option(command, "--gradient-clip-norm", 1.0)
        _assert_option(command, "--epochs", "50")
        _assert_option(command, "--batch-size", spec["batch_size"])
        _assert_option(command, "--validation-batch-size", spec["validation_batch_size"])
        _assert_option(command, "--prediction-batch-size", spec["prediction_batch_size"])
        if "--shuffle-train-batches" not in command:
            raise AssertionError(f"{label}: expected --shuffle-train-batches")
        if "--micro-batch-size" in command:
            raise AssertionError(f"{label}: micro_batch_size must not be passed to runner")

        _assert_mapped(plan, "run.batch_size")
        _assert_mapped(plan, "run.validation_batch_size")
        _assert_mapped(plan, "run.prediction_batch_size")
        _assert_mapped(plan, "run.shuffle_train_batches")
        _assert_mapped(plan, "run.drop_last")
        _assert_unmapped(plan, "run.micro_batch_size")
        _assert_warnings(plan)

        print(
            f"{label}: batch={spec['batch_size']} "
            f"micro={spec['micro_batch_size']} "
            f"valid={spec['validation_batch_size']} "
            f"pred={spec['prediction_batch_size']}"
        )

    print("Heat3D v2 batch config smoke passed.")
    return 0


def _assert_model_command(command: list[str], expected_model: dict[str, int]) -> None:
    field_to_flag = {
        "node_latent_size": "--node-latent-size",
        "edge_latent_size": "--edge-latent-size",
        "processor_steps": "--processor-steps",
        "mlp_hidden_layers": "--mlp-hidden-layers",
    }
    for field, flag in field_to_flag.items():
        _assert_option(command, flag, expected_model[field])


def _assert_mapped(plan: dict[str, Any], field: str) -> None:
    mapped = {item["field"] for item in plan["mapped_fields"]}
    if field not in mapped:
        raise AssertionError(f"expected mapped field {field}")


def _assert_unmapped(plan: dict[str, Any], field: str) -> None:
    unmapped = {item["field"] for item in plan["unmapped_fields"]}
    if field not in unmapped:
        raise AssertionError(f"expected unmapped field {field}")


def _assert_warnings(plan: dict[str, Any]) -> None:
    warnings = "\n".join(plan["warnings"])
    for snippet in REQUIRED_WARNING_SNIPPETS:
        if snippet not in warnings:
            raise AssertionError(f"expected warning snippet {snippet!r}")


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
