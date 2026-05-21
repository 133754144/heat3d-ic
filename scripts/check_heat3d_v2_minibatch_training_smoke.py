#!/usr/bin/env python3
"""Smoke-check Heat3D v2 mini-batch runner helpers without real training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _batch_config_from_args,
    _batch_config_payload,
    _chunk_examples,
    _epoch_train_groups,
    _validate_batch_config,
)


CONFIG_PATHS = (
    Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1lite_batch_smoke_e1.yaml"),
    Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1lite_batch_smoke_e3.yaml"),
    Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_smoke_e1.yaml"),
    Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_e50.yaml"),
)


def main() -> int:
    _check_batch_config_helpers()
    _check_batch_splitting_helpers()
    _check_batch_command_configs()
    print("Heat3D v2 mini-batch training smoke passed.")
    return 0


def _check_batch_config_helpers() -> None:
    legacy = _batch_config_from_args(
        SimpleNamespace(
            batch_size=0,
            validation_batch_size=0,
            prediction_batch_size=0,
            shuffle_train_batches=False,
            drop_last=False,
        )
    )
    _validate_batch_config(legacy)
    legacy_payload = _batch_config_payload(legacy)
    if legacy_payload["batching_mode"] != "legacy_full_batch":
        raise AssertionError("legacy batch_size=0 must preserve full-batch mode")

    mini = _batch_config_from_args(
        SimpleNamespace(
            batch_size=4,
            validation_batch_size=8,
            prediction_batch_size=16,
            shuffle_train_batches=True,
            drop_last=False,
        )
    )
    _validate_batch_config(mini)
    mini_payload = _batch_config_payload(mini)
    if mini_payload["batching_mode"] != "mini_batch":
        raise AssertionError("positive batch_size must enable mini_batch mode")
    for field in (
        "batch_size",
        "validation_batch_size",
        "prediction_batch_size",
        "shuffle_train_batches",
        "drop_last",
    ):
        if field not in mini_payload:
            raise AssertionError(f"run_config payload missing {field}")


def _check_batch_splitting_helpers() -> None:
    examples = list(range(10))
    legacy_chunks = _chunk_examples(examples, batch_size=None, drop_last=False)
    if legacy_chunks != [examples]:
        raise AssertionError("legacy chunking must keep one full group")

    chunks = _chunk_examples(examples, batch_size=4, drop_last=False)
    if [len(chunk) for chunk in chunks] != [4, 4, 2]:
        raise AssertionError("mini-batch chunking must keep the remainder by default")

    dropped = _chunk_examples(examples, batch_size=4, drop_last=True)
    if [len(chunk) for chunk in dropped] != [4, 4]:
        raise AssertionError("drop_last must drop incomplete train batch")

    groups = [{"name": str(index)} for index in range(8)]
    shuffled = _epoch_train_groups(groups, epoch=1, seed=0, shuffle=True)
    if sorted(item["name"] for item in shuffled) != [str(index) for index in range(8)]:
        raise AssertionError("shuffled train groups must preserve all batches")
    if _epoch_train_groups(groups, epoch=1, seed=0, shuffle=False) is not groups:
        raise AssertionError("non-shuffled train groups should reuse the original list")


def _check_batch_command_configs() -> None:
    for relative_path in CONFIG_PATHS:
        config = load_v2_config(REPO_ROOT / relative_path)
        plan = build_v2_command_plan(config, python_executable="python3")
        command = plan["training_command"]
        run = config["run"]
        _assert_option(command, "--batch-size", run["batch_size"])
        _assert_option(command, "--validation-batch-size", run["validation_batch_size"])
        _assert_option(command, "--prediction-batch-size", run["prediction_batch_size"])
        if run.get("shuffle_train_batches") and "--shuffle-train-batches" not in command:
            raise AssertionError(f"{relative_path}: missing --shuffle-train-batches")
        if run.get("drop_last") and "--drop-last" not in command:
            raise AssertionError(f"{relative_path}: missing --drop-last")
        if "--micro-batch-size" in command:
            raise AssertionError(f"{relative_path}: micro-batch CLI must not be emitted")


def _assert_option(command: list[str], flag: str, expected: object) -> None:
    if flag not in command:
        raise AssertionError(f"command missing {flag}")
    index = command.index(flag)
    try:
        actual = command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"command flag {flag} has no value") from exc
    if actual != str(expected):
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
