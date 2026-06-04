"""Smoke-check Heat3D v2 full train metrics schedule plumbing."""

from __future__ import annotations

import json
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


PROFILE_CONFIG = (
    REPO_DIR
    / "configs"
    / "heat3d_v2"
    / "frozen_v1_e005_adamw_m1_batch_profile_trainmetrics_half_final.yaml"
)


def main() -> int:
    expected = {
        ("half_and_final", 1): [1],
        ("half_and_final", 5): [3, 5],
        ("half_and_final", 50): [25, 50],
        ("final_only", 5): [5],
        ("none", 5): [],
        ("every_epoch", 5): [1, 2, 3, 4, 5],
    }
    for (schedule, epochs), expected_epochs in expected.items():
        actual = runner.train_metrics_epochs(schedule, epochs)
        if actual != expected_epochs:
            raise AssertionError(
                f"{schedule} epochs={epochs}: expected {expected_epochs}, got {actual}"
            )

    payload = {
        "train_metrics_schedule": "half_and_final",
        "train_metrics_epochs": runner.train_metrics_epochs("half_and_final", 5),
        "per_epoch": [
            {"epoch_index": 1, "train_metrics_computed": False, "train_metrics_time": 0.0},
            {"epoch_index": 3, "train_metrics_computed": True, "train_metrics_time": 110.0},
        ],
    }
    json.loads(json.dumps(payload, sort_keys=True))

    config = load_v2_config(PROFILE_CONFIG)
    command = build_training_command(config, python_executable="python")
    if "--train-metrics-schedule" not in command:
        raise AssertionError("command builder did not emit --train-metrics-schedule")
    index = command.index("--train-metrics-schedule")
    if command[index + 1] != "half_and_final":
        raise AssertionError("command builder emitted the wrong train metrics schedule")
    if "--profile-timing" not in command:
        raise AssertionError("command builder did not emit --profile-timing")
    if "--profile-timing-json" not in command:
        raise AssertionError("command builder did not emit --profile-timing-json")

    print("Heat3D v2 train metrics schedule smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
