"""Smoke-check Heat3D v2 nonessential training-time optimizations."""

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
    / "frozen_v1_e005_adamw_m1_batch_profile_p1p2p3_timeopt.yaml"
)


def main() -> int:
    _check_grad_norm_schedule()
    _check_decisions()
    _check_command_builder()
    _check_json_payload()
    print("Heat3D v2 training time optimization smoke passed.")
    return 0


def _reported(every: int, batch_count: int) -> list[int]:
    return [
        batch_index
        for batch_index in range(1, batch_count + 1)
        if runner.should_report_grad_norm(every, batch_index)
    ]


def _check_grad_norm_schedule() -> None:
    if _reported(1, 3) != [1, 2, 3]:
        raise AssertionError("grad_norm_report_every=1 should report every batch")
    if _reported(10, 20) != [10, 20]:
        raise AssertionError("grad_norm_report_every=10 should report batches 10 and 20")
    if _reported(0, 20) != []:
        raise AssertionError("grad_norm_report_every=0 should skip all batches")


def _check_decisions() -> None:
    if not runner.should_reuse_final_metrics(True):
        raise AssertionError("computed final epoch metrics should be reused")
    if runner.should_reuse_final_metrics(False):
        raise AssertionError("missing final epoch metrics should fall back to compute")
    if runner.should_build_final_predictions(False):
        raise AssertionError("save_predictions=False should skip final prediction export")
    if not runner.should_build_final_predictions(True):
        raise AssertionError("save_predictions=True should build final predictions")


def _check_command_builder() -> None:
    config = load_v2_config(PROFILE_CONFIG)
    command = build_training_command(config, python_executable="python")
    if "--grad-norm-report-every" not in command:
        raise AssertionError("command builder did not emit --grad-norm-report-every")
    index = command.index("--grad-norm-report-every")
    if command[index + 1] != "10":
        raise AssertionError("command builder emitted the wrong grad norm report frequency")
    if "--train-metrics-schedule" not in command:
        raise AssertionError("command builder did not preserve train metrics schedule")
    if "--save-predictions" in command:
        raise AssertionError("timeopt config must not save final predictions")
    if "--save-best-predictions" in command:
        raise AssertionError("timeopt config must not save best predictions")


def _check_json_payload() -> None:
    payload = {
        "grad_norm_report_every": 10,
        "grad_norm_reported_batch_count": 2,
        "grad_norm_skipped_batch_count": 18,
        "final_metrics_reused": True,
        "final_metrics_reuse_source": "last_epoch_full_metrics",
        "final_prediction_export_skipped": True,
        "final_prediction_export_skip_reason": "save_predictions_false",
        "per_batch": [
            {"batch_index": 1, "grad_norm_reported": False, "grad_norm_time": 0.0},
            {"batch_index": 10, "grad_norm_reported": True, "grad_norm_time": 0.2},
        ],
    }
    json.loads(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
