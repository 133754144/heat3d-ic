#!/usr/bin/env python3
"""Smoke-check Heat3D v1 medium condition-wise diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_sample(root: Path, index: int, split: str, source: str, k_region: str, bc: str, q_power: float) -> tuple[str, np.ndarray]:
    sample_id = f"sample_{index:03d}"
    sample = root / "samples" / sample_id
    sample.mkdir(parents=True, exist_ok=True)
    coords = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    true_temp = np.asarray([[300.0], [300.02 + 0.01 * index], [300.15 + 0.02 * index], [300.35 + 0.03 * index]])
    # Deliberately overpredict low-DeltaT points while improving high-DeltaT points.
    pred_temp = true_temp.copy()
    pred_temp[0:2] += 0.04
    pred_temp[2:] -= 0.03
    np.save(sample / "coords.npy", coords)
    np.save(sample / "k_field.npy", np.full((4, 1), 1.0 + 0.1 * index))
    np.save(sample / "q_field.npy", np.full((4, 1), q_power / 4.0))
    np.save(sample / "temperature.npy", true_temp)
    meta = {
        "sample_id": sample_id,
        "split": split,
        "source_pattern_tag": source,
        "k_region_mode": k_region,
        "k_field_mode": "iso1" if index % 2 == 0 else "diag3",
        "bc_category": bc,
        "boundary_params": {"bottom": {"fixed_temperature_K": 300.0}},
        "generation_config": {
            "sample_plan": {
                "source_pattern_tag": source,
                "k_region_mode": k_region,
                "k_field_mode": "iso1" if index % 2 == 0 else "diag3",
                "bc_category": bc,
            }
        },
    }
    _write_json(sample / "sample_meta.json", meta)
    _write_json(
        sample / "metadata.json",
        {
            **{key: meta[key] for key in ("sample_id", "split", "source_pattern_tag", "k_region_mode", "k_field_mode", "bc_category")},
            "integrated_power_W": q_power,
        },
    )
    return sample_id, pred_temp


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_condition_diag_smoke_") as tmp:
        tmpdir = Path(tmp)
        subset = tmpdir / "subset"
        predictions = {}
        specs = [
            ("train", "low_power_near_zero_background_cases", "high_contrast_interface_k", "nominal_top_h", 0.1),
            ("train", "high_dynamic_range_power_cases", "layerwise_isotropic_k", "high_top_h", 1.0),
            ("valid", "centered_single_hotspot", "low_k_barrier_or_TIM_variation", "low_top_h", 0.4),
            ("test_id", "multi_block_power", "blockwise_isotropic_k", "nominal_top_h", 0.7),
        ]
        for index, spec in enumerate(specs):
            sample_id, pred = _write_sample(subset, index, *spec)
            predictions[sample_id] = pred
        pred_path = tmpdir / "predictions.npz"
        np.savez_compressed(pred_path, **predictions)
        output_json = tmpdir / "condition_diagnostics.json"
        output_md = tmpdir / "condition_diagnostics.md"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_heat3d_v1_medium_condition_diagnostics.py",
                "--subset",
                str(subset),
                "--trained-predictions",
                str(pred_path),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
                "--prediction-label",
                "smoke_final",
                "--stdout-mode",
                "compact",
            ],
            cwd=REPO_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return 1
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        checks = {
            "json_written": output_json.is_file(),
            "md_written": output_md.is_file(),
            "label_recorded": payload["prediction_label"] == "smoke_final",
            "group_keys_present": set(payload["condition_groups"]) >= {"split", "source_category", "k_region_mode", "bc_category", "k_mode", "q_power_range"},
            "bin0_present": payload["overall"]["bin_summary"]["bin_0"]["present"] if "present" in payload["overall"]["bin_summary"]["bin_0"] else True,
            "top_background_groups": bool(payload["top_background_bias_groups"]),
            "compact_stdout": "bin_0:" in result.stdout,
        }
        ok = all(checks.values())
        print("Heat3D v1 medium condition diagnostics smoke")
        print(f"checks: {checks}")
        print(f"condition_diagnostics_smoke_ok: {ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
