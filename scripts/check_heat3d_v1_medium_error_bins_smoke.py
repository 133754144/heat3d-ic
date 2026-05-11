#!/usr/bin/env python3
"""Smoke test for Heat3D v1 medium error-bin diagnostics tooling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "scripts" / "analyze_heat3d_v1_medium_error_bins.py"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sample_meta(sample_id: str, split: str, source: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "boundary_params": {
            "bottom": {"fixed_temperature_K": 300.0},
            "top": {"ambient_temperature_K": 300.0, "h_W_m2K": 1000.0},
        },
        "stack": {"stack_template": "baseline_4_layer"},
        "generation_config": {
            "sample_plan": {
                "source_pattern_tag": source,
                "k_region_mode": "layerwise_isotropic_k",
                "k_field_mode": "iso1",
                "stack_template": "baseline_4_layer",
                "bc_category": "nominal_top_h",
            }
        },
    }


def _write_sample(samples_dir: Path, sample_id: str, split: str, source: str, delta_t: np.ndarray) -> np.ndarray:
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True)
    coords = np.column_stack(
        [
            np.linspace(0.0, 0.01, delta_t.size),
            np.zeros(delta_t.size),
            np.zeros(delta_t.size),
        ]
    )
    true_temperature = (300.0 + delta_t).reshape(-1, 1)
    low_mask = delta_t <= 0.5
    pred_delta = delta_t.copy()
    pred_delta[low_mask] = delta_t[low_mask] + 0.5
    pred_delta[~low_mask] = delta_t[~low_mask] * 0.1
    pred_temperature = (300.0 + pred_delta).reshape(-1, 1)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "temperature.npy", true_temperature)
    _write_json(sample_dir / "sample_meta.json", _sample_meta(sample_id, split, source))
    return pred_temperature


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_error_bins_smoke_") as tmp:
        root = Path(tmp)
        subset = root / "subset"
        samples_dir = subset / "samples"
        run_dir = root / "run"
        samples_dir.mkdir(parents=True)
        run_dir.mkdir(parents=True)

        delta_t = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        predictions = {
            "medium_000": _write_sample(samples_dir, "medium_000", "train", "centered_single_hotspot", delta_t),
            "medium_001": _write_sample(samples_dir, "medium_001", "valid", "shifted_single_hotspot", delta_t),
        }
        predictions_path = run_dir / "predictions.npz"
        np.savez_compressed(predictions_path, **predictions)

        output_json = run_dir / "error_bins.json"
        output_md = run_dir / "error_bins.md"
        subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "--subset",
                str(subset),
                "--trained-predictions",
                str(predictions_path),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        if not output_json.is_file() or not output_md.is_file():
            raise AssertionError("error-bin analysis outputs were not generated")
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        interpretation = payload["interpretation"]
        expected_flags = (
            "likely_background_overprediction",
            "likely_hotspot_region_improvement",
            "likely_hotspot_learning_with_background_bias",
        )
        for flag in expected_flags:
            if interpretation.get(flag) is not True:
                raise AssertionError(f"expected {flag}=true, found {interpretation.get(flag)}")
        md_text = output_md.read_text(encoding="utf-8")
        if "likely_hotspot_learning_with_background_bias: `True`" not in md_text:
            raise AssertionError("markdown did not include hotspot/background-bias interpretation")

    print("Heat3D v1 medium error bins smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
