#!/usr/bin/env python3
"""Smoke-check Heat3D v1 medium training progress logging helpers."""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def _parse(argv: list[str]):
    original = sys.argv[:]
    try:
        sys.argv = ["run_heat3d_v1_medium_controlled_training_export.py", *argv]
        return runner.parse_args()
    finally:
        sys.argv = original


def main() -> int:
    default_args = _parse([])
    quiet_args = _parse(["--log-mode", "quiet"])
    disabled_args = _parse(["--no-progress-log"])
    enabled_args = _parse(["--progress-log", "--log-mode", "compact"])

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runner._progress(True, "startup", "loading dataset from fake_subset ...", time.perf_counter() - 0.01)
        runner._progress(False, "startup", "this should not print")
        runner._print_epoch_progress(
            {
                "epoch": 1,
                "lr": 1.0e-3,
                "train_loss": 1.0,
                "valid_loss": 2.0,
                "valid_base_mse": 3.0,
                "valid_bg_signed_bias": 4.0,
                "valid_background_relative_abs": 5.0,
                "valid_hotspot_raw_mae": 6.0,
                "valid_raw_deltaT_mse": 7.0,
                "current_background_relative_weight": 0.05,
                "current_hotspot_weight": 0.02,
            },
            epochs=2,
            log_mode="compact",
        )

    output = buffer.getvalue()
    checks = {
        "default_progress_enabled": runner._progress_enabled(default_args),
        "quiet_progress_disabled": not runner._progress_enabled(quiet_args),
        "explicit_progress_disabled": not runner._progress_enabled(disabled_args),
        "explicit_progress_enabled": runner._progress_enabled(enabled_args),
        "startup_line_printed": "[startup] loading dataset from fake_subset ..." in output,
        "elapsed_printed": "elapsed=" in output,
        "disabled_line_suppressed": "this should not print" not in output,
        "compact_epoch_printed": "epoch 001/002" in output and "valid_bg_bias=" in output,
    }
    ok = all(checks.values())
    print("Heat3D v1 medium training progress logging smoke")
    print(f"checks: {checks}")
    print(f"progress_smoke_ok: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
