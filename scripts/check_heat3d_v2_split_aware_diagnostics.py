#!/usr/bin/env python3
"""Smoke-check Heat3D v2 split-aware diagnostics on a tiny synthetic subset."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.analyze_heat3d_v2_split_aware_diagnostics import analyze_split_aware_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_sample(root: Path, sample_id: str, split: str, scale: float) -> np.ndarray:
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    delta = scale * np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10], dtype=np.float64)
    temperature = 300.0 + delta
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "temperature.npy", temperature.reshape(-1, 1))
    np.save(sample_dir / "q_field.npy", np.ones((8, 1), dtype=np.float64) * scale)
    _write_json(
        sample_dir / "sample_meta.json",
        {
            "sample_id": sample_id,
            "split": "old_valid",
            "source_pattern_tag": "tiny_source",
            "power_scale_category": "nominal",
            "bc_category": "nominal_top_h",
            "k_field_mode": "diag3",
            "k_region_mode": "layerwise_isotropic_k",
            "stack": {"stack_template": "tiny_stack"},
            "boundary_params": {"bottom": {"fixed_temperature_K": 300.0}},
        },
    )
    _write_json(
        sample_dir / "metadata.json",
        {
            "sample_id": sample_id,
            "split": split,
            "integrated_power_W": float(scale),
            "bottom_T_fixed_K": 300.0,
        },
    )
    return temperature.reshape(-1, 1)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_split_diag_") as tmp:
        work = Path(tmp)
        subset = work / "subset"
        predictions = {}
        split_map = {"sample_splits": {}}
        for index, split in enumerate(("valid_iid", "valid_iid", "valid_stress", "valid_stress")):
            sample_id = f"sample_{index:04d}"
            true_temperature = _make_sample(subset, sample_id, split, scale=1.0 + index)
            predictions[sample_id] = true_temperature + 0.001 * (index + 1)
            split_map["sample_splits"][sample_id] = split

        pred_path = work / "predictions.npz"
        np.savez_compressed(pred_path, **predictions)
        split_map_path = work / "split_map.json"
        _write_json(split_map_path, split_map)

        out_json = work / "valid_iid.json"
        out_md = work / "valid_iid.md"
        slice_dir = work / "slices"
        payload = analyze_split_aware_diagnostics(
            subset=subset,
            trained_predictions=pred_path,
            split_map=split_map_path,
            split="valid_iid",
            prediction_label="best",
            output_json=out_json,
            output_md=out_md,
            slice_output_dir=slice_dir,
            top_k=2,
            max_slice_samples=2,
        )
        if payload["sample_count"] != 2:
            raise AssertionError(f"expected 2 valid_iid samples, got {payload['sample_count']}")
        if "field_variance_ratio" not in payload["overall"]:
            raise AssertionError("missing field-shape metric")
        if "background_diagnostics" not in payload:
            raise AssertionError("missing background diagnostics")
        if "deltaT_bin_errors" not in payload or "bin_0" not in payload["deltaT_bin_errors"]:
            raise AssertionError("missing raw DeltaT bin_0 diagnostics")
        bin0 = payload["deltaT_bin_errors"]["bin_0"]
        for key in ("mae", "signed_bias", "over_ratio", "under_ratio"):
            if key not in bin0:
                raise AssertionError(f"missing bin_0 field: {key}")
        low_bin = payload["low_deltaT_bin_errors"]["le_0p05"]
        if "underprediction_ratio" not in low_bin:
            raise AssertionError("missing low-DeltaT underprediction ratio")
        if not payload["slice_exports"]:
            raise AssertionError("slice metadata was not exported")
        if not out_json.exists() or not out_md.exists():
            raise AssertionError("diagnostics outputs were not written")
    print("Heat3D v2 split-aware diagnostics smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
