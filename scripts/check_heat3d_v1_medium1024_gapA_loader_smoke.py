#!/usr/bin/env python3
"""Smoke-check native supervised loading for the medium1024 Gap-A stage."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402


GAP_A_STAGE = "physics_label_medium1024_gapA_generation_candidate"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_fake_sample(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.002],
            [0.01, 0.01, 0.002],
        ],
        dtype=np.float64,
    )
    n_points = coords.shape[0]
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "layer_id.npy", np.zeros((n_points, 1), dtype=np.int32))
    np.save(sample_dir / "region_id.npy", np.zeros((n_points, 1), dtype=np.int32))
    np.save(sample_dir / "material_id.npy", np.ones((n_points, 1), dtype=np.int32))
    np.save(sample_dir / "k_field.npy", np.full((n_points, 1), 22.0, dtype=np.float64))
    np.save(sample_dir / "q_field.npy", np.array([[0.0], [1.0e7], [0.0], [2.0e7]], dtype=np.float64))
    np.save(sample_dir / "temperature.npy", np.array([[300.0], [300.1], [300.0], [300.2]], dtype=np.float64))
    _write_json(
        sample_dir / "sample_meta.json",
        {
            "schema_version": "physics_label_medium1024_gapA_v0",
            "subset_name": "v1_multilayer_bc_eq_physics_label_medium1024_gapA_loader_smoke",
            "sample_id": sample_dir.name,
            "split": "train",
            "stage": GAP_A_STAGE,
            "boundary_regions": [
                {"name": "bottom", "point_indices": [0, 1]},
                {"name": "top", "point_indices": [2, 3]},
            ],
            "boundary_params": {
                "top": {"h_W_m2K": 1000.0, "ambient_temperature_K": 300.0},
                "bottom": {"fixed_temperature_K": 300.0},
            },
            "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
            "generation_config": {
                "dataset_name": "v1_multilayer_bc_eq_physics_label_medium1024_gapA",
                "sample_plan": {
                    "source_pattern_tag": "low_power_near_zero_background_cases",
                    "k_region_mode": "high_contrast_interface_k",
                    "k_field_mode": "iso1",
                    "stack_template": "baseline_4_layer",
                    "bc_category": "nominal_top_h",
                },
            },
            "units": {"coords": "m", "k_field": "W/m/K", "q_field": "W/m^3", "temperature": "K"},
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_gapA_loader_") as tmp:
        samples_dir = Path(tmp) / "samples"
        sample_dir = samples_dir / "medium_gapA_loader_0000"
        _write_fake_sample(sample_dir)
        dataset = Heat3DV1NativeSupervisedDataset(samples_dir, k_encoding_mode="diag3")
        if len(dataset) != 1:
            raise AssertionError(f"expected one loaded sample, found {len(dataset)}")
        example = dataset[0]
        conditions = example.condition.condition_features
        target = example.target.target_u
        checks = {
            "sample_id_ok": example.sample_id == "medium_gapA_loader_0000",
            "stage_ok": example.meta.get("stage") == GAP_A_STAGE,
            "condition_finite": bool(np.all(np.isfinite(conditions))),
            "target_finite": bool(np.all(np.isfinite(target))),
            "coords_shape_ok": example.condition.coords.shape == (4, 3),
            "target_shape_ok": target.shape == (4, 1),
            "k_encoding_ok": example.condition.k_encoding_mode == "diag3",
        }
        ok = all(checks.values())
        print("Heat3D v1 medium1024 Gap-A loader smoke")
        print(f"stage: {GAP_A_STAGE}")
        print(f"sample_count: {len(dataset)}")
        print(f"condition_feature_names: {example.condition.condition_feature_names}")
        print(f"condition_shape: {conditions.shape}")
        print(f"target_shape: {target.shape}")
        print(f"checks: {checks}")
        print(f"medium1024_gapA_loader_smoke_ok: {ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
