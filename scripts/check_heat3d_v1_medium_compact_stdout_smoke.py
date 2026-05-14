#!/usr/bin/env python3
"""Smoke-check compact/full/quiet stdout modes for Heat3D v1 diagnostics."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPTS_DIR.parents[0]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_heat3d_v1_medium_error_bins as error_bins  # noqa: E402
import compare_heat3d_v1_medium_baselines as compare  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary_row(predictor: str, split: str | None = None) -> dict:
    value = 1.0 if predictor == "zero_delta" else 0.8
    row = {
        "predictor": predictor,
        "sample_count": 2,
        "row_count": 2,
        "mean_recovered_T_rmse": value,
        "mean_recovered_T_mae": value / 2.0,
        "mean_DeltaT_rmse": value,
        "mean_max_abs_error": value * 2.0,
        "mean_p95_abs_error": value * 1.5,
        "mean_peak_T_error": value,
        "mean_hotspot_coord_error": value,
    }
    if split is not None:
        row["split"] = split
    return row


def _fake_baseline() -> dict:
    condition_rows = {
        key: [
            {**_summary_row("zero_delta"), key: "case_a"},
            {**_summary_row("trained_prediction"), key: "case_a"},
        ]
        for key in ("source_pattern_tag", "k_region_mode", "k_field_mode", "stack_template", "bc_category")
    }
    return {
        "trained_comparison_status": "computed",
        "per_sample": [],
        "overall": [_summary_row("zero_delta"), _summary_row("trained_prediction")],
        "split_summary": [
            _summary_row("zero_delta", "train"),
            _summary_row("trained_prediction", "train"),
        ],
        "condition_summary": condition_rows,
    }


def _fake_loss_summary() -> dict:
    return {
        "status_ok": True,
        "grad_finite": True,
        "train_losses": [2.0, 1.0],
        "valid_losses": [2.5, 1.2],
        "loss_mode": "background_l1_relative",
        "lr_schedule": "constant",
        "loss_weight_schedule": "constant",
        "selection_metric": "valid_loss",
        "best_epoch": 1,
        "best_valid_loss": 1.2,
        "best_valid_raw_deltaT_mse": 0.9,
        "best_valid_base_mse": 0.8,
        "final_epoch": 1,
        "final_valid_loss": 1.2,
        "best_predictions_saved": True,
        "best_predictions_path": "best_predictions.npz",
        "train_metrics": {"raw_delta_mse": 1.0, "recovered_temperature_mse": 1.0},
        "valid_metrics": {"raw_delta_mse": 0.9, "recovered_temperature_mse": 0.9},
        "final_train_loss_components": {"background_relative_abs": 0.1},
        "final_valid_loss_components": {"background_relative_abs": 0.2},
        "epoch_history": [{"epoch": 1, "train_loss": 1.0, "valid_loss": 1.2}],
    }


def _write_fake_diversity_subset(root: Path) -> None:
    samples = root / "samples"
    for index in range(2):
        sample = samples / f"sample_{index:03d}"
        sample.mkdir(parents=True, exist_ok=True)
        _write_json(
            sample / "metadata.json",
            {
                "sample_id": sample.name,
                "split": "train",
                "source_pattern_tag": "centered_single_hotspot",
                "k_region_mode": "layerwise_isotropic_k",
                "k_field_mode": "iso1",
                "stack_template": "baseline_4_layer",
                "bc_category": "nominal_top_h",
            },
        )
        coords = np.arange(6, dtype=np.float64).reshape(2, 3)
        np.save(sample / "coords.npy", coords)
        np.save(sample / "k_field.npy", np.full((2, 1), 1.0 + index))
        np.save(sample / "q_field.npy", np.full((2, 1), 0.1 + index))
        np.save(sample / "temperature.npy", np.full((2, 1), 300.0 + index))


def _capture_prints(fn, *args) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args)
    return buffer.getvalue()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_stdout_smoke_") as tmp:
        tmpdir = Path(tmp)
        run_dir = tmpdir / "run"
        run_dir.mkdir()
        _write_json(run_dir / "loss_summary.json", _fake_loss_summary())
        _write_json(run_dir / "baseline_comparison.json", _fake_baseline())

        for mode in ("compact", "full", "quiet"):
            run_summary = _run(
                [
                    sys.executable,
                    "scripts/analyze_heat3d_v1_medium_run_summary.py",
                    "--run-dir",
                    str(run_dir),
                    "--stdout-mode",
                    mode,
                ]
            )
            if run_summary.returncode != 0:
                print(run_summary.stdout)
                print(run_summary.stderr)
                return 1

        subset = tmpdir / "subset"
        _write_fake_diversity_subset(subset)
        for mode in ("compact", "full", "quiet"):
            diversity = _run(
                [
                    sys.executable,
                    "scripts/analyze_heat3d_v1_medium1024_gapA_diversity.py",
                    "--subset",
                    str(subset),
                    "--output-json",
                    str(tmpdir / f"diversity_{mode}.json"),
                    "--output-md",
                    str(tmpdir / f"diversity_{mode}.md"),
                    "--stdout-mode",
                    mode,
                ]
            )
            if diversity.returncode != 0:
                print(diversity.stdout)
                print(diversity.stderr)
                return 1

        compare_outputs = [
            _capture_prints(compare._print_report, _fake_baseline(), Path("fake_subset"), None, mode)
            for mode in ("compact", "full", "quiet")
        ]
        fake_error_payload = {
            "inputs": {"subset": "fake_subset", "trained_predictions": "predictions.npz"},
            "outputs": {"json": str(tmpdir / "error_bins.json"), "markdown": str(tmpdir / "error_bins.md")},
            "sample_count": 2,
            "point_count": 4,
            "overall": {
                "bins": [
                    {
                        "bin_name": f"bin_{index}",
                        "relative_rmse_change": -0.1 + index * 0.01,
                        "relative_mae_change": -0.05,
                        "trained_signed_bias": 0.01,
                        "trained_overprediction_ratio": 0.5,
                    }
                    for index in range(5)
                ]
            },
            "interpretation": {
                "likely_background_overprediction": True,
                "likely_hotspot_region_improvement": True,
                "likely_hotspot_learning_with_background_bias": True,
            },
        }
        error_outputs = [
            _capture_prints(error_bins._print_stdout_summary, fake_error_payload, mode)
            for mode in ("compact", "full", "quiet")
        ]

        checks = {
            "run_analysis_json": (run_dir / "run_analysis.json").is_file(),
            "run_analysis_md": (run_dir / "run_analysis.md").is_file(),
            "diversity_json": (tmpdir / "diversity_compact.json").is_file(),
            "diversity_md": (tmpdir / "diversity_compact.md").is_file(),
            "compare_compact": "overall:" in compare_outputs[0],
            "compare_full": "overall summary" in compare_outputs[1],
            "compare_quiet": "baseline_comparison_written" in compare_outputs[2],
            "error_compact": "bin_0" in error_outputs[0],
            "error_quiet": "error_bins_written" in error_outputs[2],
        }
        ok = all(checks.values())
        print("Heat3D v1 medium compact stdout smoke")
        print(f"checks: {checks}")
        print(f"compact_stdout_smoke_ok: {ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
