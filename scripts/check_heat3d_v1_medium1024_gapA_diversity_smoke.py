#!/usr/bin/env python3
"""Smoke-check medium1024 Gap-A diversity diagnostics on a fake subset."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_heat3d_v1_medium1024_gapA_diversity import analyze_subset, write_markdown  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _base_metadata(sample_id: str, overrides: dict) -> dict:
    metadata = {
        "sample_id": sample_id,
        "split": "train",
        "source_pattern_tag": "centered_single_hotspot",
        "k_region_mode": "layerwise_isotropic_k",
        "k_field_mode": "iso1",
        "stack_template": "baseline_4_layer",
        "bc_category": "nominal_top_h",
        "power_scale_category": "nominal_power",
        "k_contrast_category": "nominal_contrast",
        "barrier_k_category": "nominal_barrier_k",
    }
    metadata.update(overrides)
    return metadata


def _write_sample(sample_dir: Path, metadata: dict, q_field: np.ndarray, k_field: np.ndarray, temperature: np.ndarray) -> None:
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
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "q_field.npy", q_field.astype(np.float64))
    np.save(sample_dir / "k_field.npy", k_field.astype(np.float64))
    np.save(sample_dir / "temperature.npy", temperature.astype(np.float64))
    _write_json(sample_dir / "metadata.json", metadata)


def _combo_tuple(item: dict) -> tuple[str, ...]:
    combo = item["combo"]
    return (
        combo["split"],
        combo["source_pattern_tag"],
        combo["k_region_mode"],
        combo["k_field_mode"],
        combo["stack_template"],
        combo["bc_category"],
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_gapA_diversity_") as tmp:
        subset = Path(tmp) / "subset"
        samples = subset / "samples"
        q_a = np.array([[0.0], [1.0e6], [0.0], [2.0e6]])
        q_b = np.array([[0.0], [2.0e6], [0.0], [3.0e6]])
        q_c = np.array([[0.0], [4.0e6], [0.0], [1.0e6]])
        k_a = np.full((4, 1), 22.0)
        k_b = np.array([[8.0], [12.0], [30.0], [45.0]])
        t_a = np.array([[300.0], [300.1], [300.0], [300.2]])
        t_b = np.array([[300.0], [300.2], [300.0], [300.3]])
        t_c = np.array([[300.0], [300.4], [300.1], [300.5]])

        sample_specs = [
            (
                "fake_0000",
                _base_metadata("fake_0000", {"source_pattern_tag": "low_power_near_zero_background_cases", "power_scale_category": "low_power"}),
                q_a,
                k_a,
                t_a,
            ),
            (
                "fake_0001",
                _base_metadata("fake_0001", {"source_pattern_tag": "low_power_near_zero_background_cases", "power_scale_category": "low_power"}),
                q_b,
                k_a,
                t_b,
            ),
            (
                "fake_0002",
                _base_metadata("fake_0002", {"source_pattern_tag": "high_dynamic_range_power_cases", "power_scale_category": "high_dynamic_range"}),
                q_a,
                k_a,
                t_a,
            ),
            (
                "fake_0003",
                _base_metadata(
                    "fake_0003",
                    {
                        "split": "valid",
                        "k_region_mode": "high_contrast_interface_k",
                        "k_contrast_category": "high_contrast",
                    },
                ),
                q_c,
                k_b,
                t_c,
            ),
            (
                "fake_0004",
                _base_metadata(
                    "fake_0004",
                    {
                        "split": "test_ood_bc_candidate",
                        "k_region_mode": "low_k_barrier_or_TIM_variation",
                        "bc_category": "very_low_top_h_candidate",
                        "barrier_k_category": "low_k",
                    },
                ),
                q_b,
                k_b,
                t_b,
            ),
            (
                "fake_0005",
                _base_metadata(
                    "fake_0005",
                    {
                        "split": "test_ood_combined_candidate",
                        "bc_category": "very_high_top_h_candidate",
                        "stack_template": "held_out_interposer_like_candidate",
                    },
                ),
                q_c,
                k_a,
                t_c,
            ),
        ]

        for sample_id, metadata, q_field, k_field, temperature in sample_specs:
            _write_sample(samples / sample_id, metadata, q_field, k_field, temperature)

        result = analyze_subset(subset, top_n=30)
        output_json = Path(tmp) / "diversity.json"
        output_md = Path(tmp) / "diversity.md"
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_markdown(output_md, result)

        flags = result["diagnostic_flags"]
        first_combo = (
            "train",
            "low_power_near_zero_background_cases",
            "layerwise_isotropic_k",
            "iso1",
            "baseline_4_layer",
            "nominal_top_h",
        )
        matching_combo = next(
            item for item in result["per_combo_diversity"] if _combo_tuple(item) == first_combo
        )
        checks = {
            "json_written": output_json.is_file(),
            "markdown_written": output_md.is_file(),
            "sample_count_ok": result["sample_count"] == 6,
            "same_combo_q_diverse": matching_combo["sample_count"] == 2 and matching_combo["unique_q_hash_count"] == 2,
            "q_duplicates_flag": flags["likely_true_q_duplicates"],
            "k_duplicates_flag": flags["likely_true_k_duplicates"],
            "temperature_duplicates_flag": flags["likely_true_temperature_duplicates"],
            "training_smoke_ready": flags["diversity_ready_for_training_smoke"],
            "formal_benchmark_false": not flags["diversity_ready_for_formal_benchmark"],
            "markdown_title_ok": "Heat3D v1 Medium1024 Gap-A Diversity Diagnostics" in output_md.read_text(encoding="utf-8"),
        }
        ok = all(checks.values())
        print("Heat3D v1 medium1024 Gap-A diversity smoke")
        print(f"checks: {checks}")
        print(f"diagnostic_flags: {flags}")
        print(f"medium1024_gapA_diversity_smoke_ok: {ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
