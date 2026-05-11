#!/usr/bin/env python3
"""Smoke test for Heat3D v1 medium run analysis tooling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "scripts" / "analyze_heat3d_v1_medium_run_summary.py"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary_row(
    predictor: str,
    *,
    rmse: float,
    mae: float,
    peak: float,
    hotspot: float,
    group_key: str | None = None,
    group_value: str | None = None,
) -> dict:
    row = {
        "predictor": predictor,
        "sample_count": 2,
        "row_count": 2,
        "mean_recovered_T_rmse": rmse,
        "mean_recovered_T_mae": mae,
        "mean_DeltaT_rmse": rmse,
        "mean_max_abs_error": max(rmse, peak),
        "mean_p95_abs_error": mae * 1.5,
        "mean_peak_T_error": peak,
        "mean_hotspot_coord_error": hotspot,
    }
    if group_key is not None:
        row[group_key] = group_value
    return row


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_run_analysis_smoke_") as tmp:
        run_dir = Path(tmp)
        _write_json(
            run_dir / "loss_summary.json",
            {
                "status_ok": True,
                "grad_finite": True,
                "train_losses": [1.0, 0.9, 0.8],
                "valid_losses": [1.1, 1.0, 0.95],
                "train_metrics": {
                    "raw_delta_mse": 0.01,
                    "recovered_temperature_mse": 0.01,
                },
                "valid_metrics": {
                    "raw_delta_mse": 0.02,
                    "recovered_temperature_mse": 0.02,
                },
                "epoch_history": [
                    {"epoch": 1, "train_loss": 0.9, "valid_loss": 1.0},
                    {"epoch": 2, "train_loss": 0.8, "valid_loss": 0.95},
                ],
            },
        )
        _write_json(
            run_dir / "baseline_comparison.json",
            {
                "diagnostic_scope": "mock baseline comparison",
                "trained_comparison_status": "computed",
                "overall": [
                    _summary_row("zero_delta", rmse=1.0, mae=0.50, peak=4.0, hotspot=0.10),
                    _summary_row("trained_prediction", rmse=1.2, mae=0.60, peak=2.0, hotspot=0.05),
                ],
                "split_summary": [
                    _summary_row("zero_delta", rmse=1.0, mae=0.50, peak=4.0, hotspot=0.10, group_key="split", group_value="valid"),
                    _summary_row("trained_prediction", rmse=1.2, mae=0.60, peak=2.0, hotspot=0.05, group_key="split", group_value="valid"),
                ],
                "condition_summary": {
                    key: [
                        _summary_row("zero_delta", rmse=1.0, mae=0.50, peak=4.0, hotspot=0.10, group_key=key, group_value=f"{key}_a"),
                        _summary_row("trained_prediction", rmse=0.8, mae=0.45, peak=2.0, hotspot=0.04, group_key=key, group_value=f"{key}_a"),
                        _summary_row("zero_delta", rmse=1.0, mae=0.50, peak=4.0, hotspot=0.10, group_key=key, group_value=f"{key}_b"),
                        _summary_row("trained_prediction", rmse=1.3, mae=0.70, peak=3.5, hotspot=0.12, group_key=key, group_value=f"{key}_b"),
                    ]
                    for key in (
                        "source_pattern_tag",
                        "k_region_mode",
                        "k_field_mode",
                        "stack_template",
                        "bc_category",
                    )
                },
            },
        )

        subprocess.run(
            [sys.executable, str(ANALYZER), "--run-dir", str(run_dir)],
            cwd=REPO_ROOT,
            check=True,
        )

        output_json = run_dir / "run_analysis.json"
        output_md = run_dir / "run_analysis.md"
        if not output_json.is_file() or not output_md.is_file():
            raise AssertionError("analysis outputs were not generated")
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        flag = payload["baseline_comparison"]["overall_status"]["likely_hotspot_learning_with_background_bias"]
        if flag is not True:
            raise AssertionError("expected likely_hotspot_learning_with_background_bias to be true")
        md_text = output_md.read_text(encoding="utf-8")
        if "likely_hotspot_learning_with_background_bias = true" not in md_text:
            raise AssertionError("markdown did not include hotspot/background-bias interpretation")

    print("Heat3D v1 medium run analysis smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
