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
import run_heat3d_v4_controlled_training as v4_runner  # noqa: E402


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
    verbose_args = _parse(["--progress-detail", "verbose"])
    off_args = _parse(["--progress-detail", "off"])
    best_args = _parse(
        [
            "--selection-metric",
            "valid_raw_deltaT_mse",
            "--save-best-predictions",
            "--best-predictions-name",
            "best_predictions_smoke.npz",
        ]
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runner._progress(True, "startup", "loading dataset from fake_subset ...", time.perf_counter() - 0.01)
        runner._progress(False, "startup", "this should not print")
        runner._print_epoch_progress(
            {
                "epoch": 1,
                "lr": 1.0e-3,
                "train_loss": 1.0,
                "train_base_mse": float("nan"),
                "epoch_mean_train_batch_base_mse": 1.25,
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

    v4_buffer = io.StringIO()
    v4_runner._install_profile_hooks(
        v4_runner.DEFAULT_NORMALIZATION_PROFILE,
        v4_runner.DEFAULT_CONDITION_FEATURE_TRANSFORM,
        v4_runner.DEFAULT_INPUT_FEATURE_SCHEMA,
        v4_runner.DEFAULT_COORD_POLICY,
        v4_runner.DEFAULT_EXTENT_FEATURE_POLICY,
    )
    with contextlib.redirect_stdout(v4_buffer):
        runner._print_epoch_progress(
            {
                "epoch": 1,
                "lr": 1.0e-3,
                "train_loss": 1.0,
                "train_base_mse": 1.25,
                "valid_loss": 2.0,
                "valid_base_mse": 3.0,
                "valid_raw_rmse_K": 2.0,
                "valid_rel_rmse_v4_pct": 25.0,
                "best_epoch": 1,
                "best_valid_iid_loss": 2.0,
                "best_valid_iid_base_mse": 4.0,
            },
            epochs=2,
            log_mode="compact",
        )

    best_payload = runner._best_selection_payload(
        {
            "selection_metric": "valid_raw_deltaT_mse",
            "best_record": {
                "epoch": 2,
                "valid_loss": 1.5,
                "valid_raw_deltaT_mse": 1.25,
                "valid_base_mse": 1.1,
            },
            "final_epoch": 3,
            "final_valid_loss": 2.0,
            "valid_metrics": {"raw_delta_mse": 1.9},
            "final_valid_loss_components": {"base_mse": 1.8},
        },
        best_predictions_path=Path("best_predictions_smoke.npz"),
        best_predictions_saved=True,
    )
    combined_metrics = runner._combine_metric_payloads(
        [
            (
                1,
                {
                    "raw_delta_mse": 4.0,
                    "recovered_temperature_mse": 4.0,
                    "mean_abs_true_deltaT": 2.0,
                    "mean_square_true_deltaT": 100.0,
                    "finite_ok": True,
                    "shape_ok": True,
                },
            )
        ]
    )

    output = buffer.getvalue()
    checks = {
        "default_progress_enabled": runner._progress_enabled(default_args),
        "quiet_progress_disabled": not runner._progress_enabled(quiet_args),
        "explicit_progress_disabled": not runner._progress_enabled(disabled_args),
        "explicit_progress_enabled": runner._progress_enabled(enabled_args),
        "default_progress_detail_basic": default_args.progress_detail == "basic",
        "verbose_progress_detail_enabled": runner._verbose_progress_enabled(verbose_args),
        "progress_detail_off_disabled": not runner._progress_detail_enabled(off_args),
        "progress_checkpoints_full1024": runner._progress_checkpoints(1024) == {256, 512, 768, 1024},
        "selection_metric_parsed": best_args.selection_metric == "valid_raw_deltaT_mse",
        "save_best_predictions_parsed": best_args.save_best_predictions,
        "best_predictions_name_parsed": best_args.best_predictions_name == "best_predictions_smoke.npz",
        "best_payload_epoch": best_payload["best_epoch"] == 2,
        "best_payload_saved": best_payload["best_predictions_saved"],
        "relative_rmse_uses_true_rms_denominator": (
            runner._deltaT_error_pct(4.0, 100.0) == 20.0
            and not (runner._deltaT_error_pct(4.0, 100.0) < 20.0)
        ),
        "combined_metrics_recompute_relative_rmse": (
            combined_metrics["rel_rmse_v4_pct"] == 20.0
        ),
        "startup_line_printed": "[startup] loading dataset from fake_subset ..." in output,
        "elapsed_printed": "elapsed=" in output,
        "disabled_line_suppressed": "this should not print" not in output,
        "compact_report_uses_online_batch_base_mse_fallback": (
            "train=1.25 valid=3" in output
        ),
        "legacy_compact_stress_placeholders_preserved": (
            "epoch 1/2" in output
            and "stress=skipped" in output
            and "stress_raw_rmse_K=skipped" in output
        ),
        "v4_compact_stress_placeholders_suppressed": (
            "stress=" not in v4_buffer.getvalue()
            and "stress_raw_rmse_K=" not in v4_buffer.getvalue()
        ),
        "compact_report_uses_base_mse_not_total_loss": (
            "train=1.25 valid=3" in v4_buffer.getvalue()
            and "best=e1/4" in v4_buffer.getvalue()
            and "train=1 valid=2" not in v4_buffer.getvalue()
            and "best=e1/2" not in v4_buffer.getvalue()
        ),
    }
    ok = all(checks.values())
    print("Heat3D v1 medium training progress logging smoke")
    print(f"checks: {checks}")
    print(f"progress_smoke_ok: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
