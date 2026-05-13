#!/usr/bin/env python3
"""Smoke test for Heat3D v1 medium multi-seed summary tooling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_DIR = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = REPO_DIR / "scripts" / "summarize_heat3d_v1_medium_multiseed_runs.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _summary_row(predictor: str, scale: float) -> dict:
    return {
        "predictor": predictor,
        "sample_count": 3,
        "row_count": 3,
        "mean_recovered_T_rmse": 0.10 * scale,
        "mean_recovered_T_mae": 0.07 * scale,
        "mean_DeltaT_rmse": 0.10 * scale,
        "mean_max_abs_error": 0.40 * scale,
        "mean_p95_abs_error": 0.18 * scale,
        "mean_peak_T_error": 0.35 * scale,
        "mean_hotspot_coord_error": 0.006 * scale,
    }


def _split_row(split: str, predictor: str, scale: float) -> dict:
    row = _summary_row(predictor, scale)
    row["split"] = split
    return row


def _baseline(scale: float) -> dict:
    zero_scale = 1.0
    return {
        "diagnostic_scope": "fake smoke comparison; not benchmark",
        "trained_comparison_status": "computed",
        "overall": [
            _summary_row("trained_prediction", scale),
            _summary_row("zero_delta", zero_scale),
        ],
        "split_summary": [
            _split_row(split, "trained_prediction", scale * split_factor)
            for split, split_factor in (
                ("train", 0.95),
                ("valid", 1.05),
                ("test_id", 1.08),
                ("test_ood_bc_candidate", 1.15),
                ("test_ood_stack_candidate", 1.22),
            )
        ]
        + [
            _split_row(split, "zero_delta", zero_scale)
            for split in (
                "train",
                "valid",
                "test_id",
                "test_ood_bc_candidate",
                "test_ood_stack_candidate",
            )
        ],
        "condition_summary": {},
    }


def _bin(name: str, bias: float, over: float, under: float, rmse_change: float, mae_change: float) -> dict:
    index = int(name.split("_")[1])
    return {
        "bin_index": index,
        "bin_name": name,
        "lower": float(index),
        "upper": float(index + 1),
        "point_count": 100,
        "sample_count": 3,
        "DeltaT_min": float(index),
        "DeltaT_max": float(index + 1),
        "DeltaT_mean": float(index) + 0.5,
        "zero_delta_rmse": 0.1,
        "zero_delta_mae": 0.08,
        "trained_rmse": 0.1 * (1.0 + rmse_change),
        "trained_mae": 0.08 * (1.0 + mae_change),
        "trained_signed_bias": bias,
        "zero_signed_bias": -0.02,
        "trained_overprediction_ratio": over,
        "trained_underprediction_ratio": under,
        "relative_rmse_change": rmse_change,
        "relative_mae_change": mae_change,
    }


def _error_bins(seed: int) -> dict:
    offset = 0.01 * seed
    return {
        "diagnostic_scope": "fake smoke error bins; not benchmark",
        "overall": {
            "bins": [
                _bin("bin_0", 0.030 + offset, 0.70 + offset, 0.20, 0.10 + offset, 0.12 + offset),
                _bin("bin_1", 0.020 + offset, 0.55 + offset, 0.30, -0.03, -0.05 + offset),
                _bin("bin_2", 0.000, 0.50, 0.50, -0.08, -0.07),
                _bin("bin_3", -0.015 - offset, 0.35, 0.65 + offset, -0.04, 0.02 + offset),
                _bin("bin_4", -0.025 - offset, 0.25, 0.75 + offset, -0.02, 0.04 + offset),
            ],
        },
        "interpretation": {
            "likely_background_overprediction": True,
            "likely_hotspot_region_improvement": True,
            "likely_hotspot_learning_with_background_bias": True,
        },
    }


def _loss_summary(seed: int) -> dict:
    return {
        "loss_mode": "background_l1_relative",
        "lr": 1e-2,
        "epochs": 300,
        "loss_weight_schedule": "constant",
        "valid_metrics": {
            "raw_delta_mse": 0.004 + seed * 0.001,
        },
        "final_valid_loss_components": {
            "bg_signed_bias": 0.02 + seed * 0.01,
            "background_relative_abs": 1.2 + seed * 0.2,
            "hotspot_raw_mae": 0.04 + seed * 0.005,
        },
    }


def _make_run(root: Path, seed: int, scale: float, omit_optional: bool = False) -> Path:
    run_dir = root / f"medium256_fake_seed{seed}"
    _write_json(run_dir / "baseline_comparison.json", _baseline(scale))
    _write_json(run_dir / "error_bins.json", _error_bins(seed))
    loss_summary = _loss_summary(seed)
    if omit_optional:
        loss_summary["final_valid_loss_components"].pop("background_relative_abs")
    _write_json(run_dir / "loss_summary.json", loss_summary)
    if not omit_optional:
        _write_json(
            run_dir / "run_config.json",
            {
                "loss_mode": "background_l1_relative",
                "lr": 1e-2,
                "epochs": 300,
                "loss_weight_schedule": "constant",
            },
        )
    return run_dir


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run0 = _make_run(root, 0, 0.72)
        run1 = _make_run(root, 1, 0.88, omit_optional=True)
        run2 = _make_run(root, 2, 0.81)
        output_dir = root / "summary"
        output_json = output_dir / "multiseed_summary.json"
        output_md = output_dir / "multiseed_summary.md"
        subprocess.run(
            [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--run-dir",
                str(run0),
                "--run-dir",
                str(run1),
                "--run-dir",
                str(run2),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ],
            check=True,
        )
        if not output_json.is_file() or not output_md.is_file():
            raise AssertionError("Expected JSON and Markdown summary outputs")
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        rmse_stats = payload["overall_summary"]["mean_T_rmse"]
        for key in ("mean", "std", "best_run", "worst_run", "median_value"):
            if key not in rmse_stats or rmse_stats[key] is None:
                raise AssertionError(f"Missing summary key {key}: {rmse_stats}")
        if rmse_stats["best_run"] != "seed0":
            raise AssertionError(f"Expected seed0 best run, found {rmse_stats['best_run']}")
        md = output_md.read_text(encoding="utf-8")
        for expected in ("Heat3D v1 Medium Multi-Seed Summary", "Overall Multi-Seed Summary", "Error-Bin Summary"):
            if expected not in md:
                raise AssertionError(f"Missing Markdown section: {expected}")
        print("Heat3D v1 medium multi-seed summary smoke")
        print(f"  output_json: {output_json}")
        print(f"  output_md: {output_md}")
        print(f"  best_run_mean_T_rmse: {rmse_stats['best_run']}")
        print("  smoke_ok: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
