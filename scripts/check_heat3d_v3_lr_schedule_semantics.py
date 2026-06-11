#!/usr/bin/env python3
"""Regression checks for Heat3D v3 learning-rate schedule semantics.

This is a pure schedule check. It does not load data, build graphs, initialize
models, or start training.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _lr_for_epoch,
    _optax_learning_rate_schedule,
)


def _as_float(value) -> float:
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def _assert_close(actual: float, expected: float, label: str, tol: float = 1e-10) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _lr_at_step(config: dict, step: int, epochs: int = 1200) -> float:
    schedule = _optax_learning_rate_schedule(epochs, config)
    if callable(schedule):
        return _as_float(schedule(step))
    return _as_float(schedule)


def _base_config(schedule: str) -> dict:
    return {
        "lr": 1e-3,
        "lr_schedule": schedule,
        "warmup_epochs": 10,
        "min_lr": 1e-6,
        "second_stage_epoch": 400,
        "second_stage_lr": 3e-4,
        "updates_per_epoch": 8,
        "lr_init": 1e-5,
        "lr_peak": 2e-4,
        "lr_base": 1e-5,
        "lr_lowr": 1e-6,
        "pct_start": 0.02,
        "pct_final": 0.10,
    }


def _check_epoch_based_stage_schedule(schedule_name: str) -> None:
    config = _base_config(schedule_name)
    for step in (0, 399, 400, 3199):
        _assert_close(
            _lr_at_step(config, step),
            1e-3,
            f"{schedule_name} step {step}",
        )
    for step in (3200, 3201, 9599):
        _assert_close(
            _lr_at_step(config, step),
            3e-4,
            f"{schedule_name} step {step}",
        )
    _assert_close(_lr_for_epoch(400, 1200, config), 1e-3, f"{schedule_name} epoch 400")
    _assert_close(_lr_for_epoch(401, 1200, config), 3e-4, f"{schedule_name} epoch 401")


def _check_constant() -> None:
    config = _base_config("constant")
    for step in (0, 399, 3200, 9599):
        _assert_close(_lr_at_step(config, step), 1e-3, f"constant step {step}")
    _assert_close(_lr_for_epoch(401, 1200, config), 1e-3, "constant epoch 401")


def _check_warmup_cosine_epoch_semantics() -> None:
    config = _base_config("warmup_cosine")
    epoch1 = 1e-6 + 0.1 * (1e-3 - 1e-6)
    epoch2 = 1e-6 + 0.2 * (1e-3 - 1e-6)
    _assert_close(_lr_at_step(config, 0), epoch1, "warmup_cosine step 0")
    _assert_close(_lr_at_step(config, 7), epoch1, "warmup_cosine step 7")
    _assert_close(_lr_at_step(config, 8), epoch2, "warmup_cosine step 8")
    _assert_close(_lr_for_epoch(1, 1200, config), epoch1, "warmup_cosine epoch 1")
    _assert_close(_lr_for_epoch(2, 1200, config), epoch2, "warmup_cosine epoch 2")


def _check_upstream_onecycle_continuous() -> None:
    config = _base_config("upstream_onecycle")
    _assert_close(_lr_at_step(config, 0), 1e-5, "upstream_onecycle step 0")
    final_step = 1200 * 8 - 1
    _assert_close(_lr_at_step(config, final_step), 1e-6, "upstream_onecycle final step")
    if not (_lr_at_step(config, 1) > _lr_at_step(config, 0)):
        raise AssertionError("upstream_onecycle should increase after step 0")


def main() -> int:
    _check_epoch_based_stage_schedule("two_stage")
    _check_epoch_based_stage_schedule("second_stage")
    _check_constant()
    _check_warmup_cosine_epoch_semantics()
    _check_upstream_onecycle_continuous()
    print("lr schedule semantics smoke ok")
    print("two_stage/second_stage: steps 0,399,400,3199 use 1e-3; step 3200+ uses 3e-4")
    print("constant, warmup_cosine, upstream_onecycle checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
