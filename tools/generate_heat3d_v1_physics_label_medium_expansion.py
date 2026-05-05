#!/usr/bin/env python3
"""Generate the Heat3D v1 24-sample medium expansion smoke subset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_reference_solver_v2 import solve_reference_temperature_v2  # noqa: E402
from rigno.heat3d_v1_region_sources import assign_q_field_volume_fraction  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium_expansion_manifest.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_expansion_v2"
)
PROTECTED_SUBSET_NAMES = {
    "v1_multilayer_bc_eq_demo",
    "v1_multilayer_bc_eq_supervised_smoke",
    "v1_multilayer_bc_eq_supervised_small",
    "v1_multilayer_bc_eq_physics_label_small_v2",
    "v1_multilayer_bc_eq_physics_label_medium_pilot_v2",
}
DOMAIN_BOUNDS = {"x": (0.0, 0.01), "y": (0.0, 0.01), "z": (0.0, 0.002)}
RESOLUTION_MAP = {"medium_expansion_mid": (8, 8, 6)}
Q_DENSITY_MAP = {"low": 0.5e8, "nominal": 1.0e8, "high": 1.5e8}
BC_CATEGORY_MAP = {
    "nominal_top_h": {"bottom_K": 300.0, "top_K": 300.0, "h_W_m2K": 1000.0},
    "low_top_h": {"bottom_K": 300.0, "top_K": 300.0, "h_W_m2K": 500.0},
    "high_top_h": {"bottom_K": 300.0, "top_K": 300.0, "h_W_m2K": 1500.0},
    "held_out_top_h_candidate": {"bottom_K": 300.0, "top_K": 300.0, "h_W_m2K": 2000.0},
}
STACK_TEMPLATES = {
    "baseline_4_layer": [
        {"name": "substrate", "z": (0.0, 0.0004), "k_iso": 90.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.0004, 0.0010), "k_iso": 22.0, "material_id": 2},
        {"name": "tim_equivalent", "z": (0.0010, 0.0013), "k_iso": 5.0, "material_id": 3},
        {"name": "heat_spreader_equivalent", "z": (0.0013, 0.0020), "k_iso": 140.0, "material_id": 4},
    ],
    "compact_3_layer": [
        {"name": "substrate", "z": (0.0, 0.00045), "k_iso": 80.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.00045, 0.00115), "k_iso": 20.0, "material_id": 2},
        {"name": "heat_spreader_equivalent", "z": (0.00115, 0.0020), "k_iso": 120.0, "material_id": 4},
    ],
    "dual_active_4_layer": [
        {"name": "substrate", "z": (0.0, 0.00035), "k_iso": 90.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.00035, 0.00085), "k_iso": 22.0, "material_id": 2},
        {"name": "interposer_equivalent", "z": (0.00085, 0.00135), "k_iso": 45.0, "material_id": 5},
        {"name": "active_die_1", "z": (0.00135, 0.0020), "k_iso": 18.0, "material_id": 6},
    ],
    "interposer_like_4_layer": [
        {"name": "substrate", "z": (0.0, 0.00035), "k_iso": 85.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.00035, 0.0009), "k_iso": 22.0, "material_id": 2},
        {"name": "interposer_like_equivalent", "z": (0.0009, 0.00145), "k_iso": 55.0, "material_id": 7},
        {"name": "heat_spreader_equivalent", "z": (0.00145, 0.0020), "k_iso": 140.0, "material_id": 4},
    ],
    "held_out_interposer_like_candidate": [
        {"name": "substrate", "z": (0.0, 0.0003), "k_iso": 85.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.0003, 0.00075), "k_iso": 22.0, "material_id": 2},
        {"name": "interposer_like_equivalent", "z": (0.00075, 0.00145), "k_iso": 60.0, "material_id": 7},
        {"name": "heat_spreader_equivalent", "z": (0.00145, 0.0020), "k_iso": 140.0, "material_id": 4},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 24-sample medium expansion physics-label smoke subset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _grid(shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xs = np.linspace(*DOMAIN_BOUNDS["x"], shape[0], dtype=np.float64)
    ys = np.linspace(*DOMAIN_BOUNDS["y"], shape[1], dtype=np.float64)
    zs = np.linspace(*DOMAIN_BOUNDS["z"], shape[2], dtype=np.float64)
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    return coords, {"x": xs, "y": ys, "z": zs}


def _layer_for_z(layers: list[dict[str, Any]], z: float) -> tuple[int, dict[str, Any]]:
    for idx, layer in enumerate(layers):
        lo, hi = layer["z"]
        if (idx == len(layers) - 1 and lo <= z <= hi) or (idx < len(layers) - 1 and lo <= z < hi):
            return idx, layer
    raise ValueError(f"z coordinate outside stack template: {z}")


def _layer_arrays(coords: np.ndarray, layers: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layer_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    region_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    material_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    for idx, point in enumerate(coords):
        layer_idx, layer = _layer_for_z(layers, float(point[2]))
        layer_id[idx, 0] = layer_idx
        region_id[idx, 0] = layer_idx
        material_id[idx, 0] = int(layer["material_id"])
    return layer_id, region_id, material_id, [str(layer["name"]) for layer in layers]


def _k_field(coords: np.ndarray, layers: list[dict[str, Any]], sample: dict[str, Any]) -> np.ndarray:
    k_mode = sample["k_field_mode"]
    region_mode = sample["k_region_mode"]
    if k_mode == "diag3":
        k = np.empty((coords.shape[0], 3), dtype=np.float64)
        for idx, point in enumerate(coords):
            _, layer = _layer_for_z(layers, float(point[2]))
            base = float(layer["k_iso"])
            k[idx] = [base * 1.20, base * 0.90, base * 0.55]
        return k
    if k_mode != "iso1":
        raise ValueError(f"unsupported k_field_mode: {k_mode}")
    k = np.empty((coords.shape[0], 1), dtype=np.float64)
    for idx, point in enumerate(coords):
        _, layer = _layer_for_z(layers, float(point[2]))
        value = float(layer["k_iso"])
        if region_mode == "blockwise_isotropic_k" and "active_die" in layer["name"]:
            value *= 0.75 if point[0] < 0.005 else 1.25
        elif region_mode == "interposer_equivalent_k" and "interposer" in layer["name"]:
            value *= 1.10
        k[idx, 0] = value
    return k


def _source_box(region: dict[str, Any], layers: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    layer = next((item for item in layers if item["name"] == region["layer"]), None)
    if layer is None:
        raise ValueError(f"source layer not found: {region['layer']}")
    x0, x1 = DOMAIN_BOUNDS["x"]
    y0, y1 = DOMAIN_BOUNDS["y"]
    cx = x0 + float(region["center_xy_fraction"][0]) * (x1 - x0)
    cy = y0 + float(region["center_xy_fraction"][1]) * (y1 - y0)
    sx = float(region["size_xy_fraction"][0]) * (x1 - x0)
    sy = float(region["size_xy_fraction"][1]) * (y1 - y0)
    z0, z1 = layer["z"]
    z_span = z1 - z0
    return {
        "x": (max(x0, cx - 0.5 * sx), min(x1, cx + 0.5 * sx)),
        "y": (max(y0, cy - 0.5 * sy), min(y1, cy + 0.5 * sy)),
        "z": (z0 + 0.20 * z_span, z0 + 0.80 * z_span),
    }


def _resolved_sources(sample: dict[str, Any], layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for region in sample["source_regions"]:
        q_category = region["q_scale_category"]
        sources.append({
            "region_id": region["region_id"],
            "layer": region["layer"],
            "source_box_m": _source_box(region, layers),
            "q_density_W_m3": Q_DENSITY_MAP[q_category],
            "q_scale_category": q_category,
        })
    return sources


def _sample_meta(
    manifest: dict[str, Any],
    manifest_path: Path,
    sample: dict[str, Any],
    layers: list[dict[str, Any]],
    grid_shape: tuple[int, int, int],
    layer_names: list[str],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    bc = BC_CATEGORY_MAP[sample["bc_category"]]
    subset_name = str(manifest.get("output_subset_name", "v1_multilayer_bc_eq_physics_label_medium_expansion_v2"))
    sample_stage = str(manifest.get("sample_stage", "physics_label_medium_expansion_smoke"))
    description = str(
        manifest.get(
            "sample_description",
            "24-sample medium expansion smoke with region-first volume-fraction source assignment.",
        )
    )
    non_claim_flags = dict(manifest.get("non_claim_flags", {}))
    if not non_claim_flags:
        non_claim_flags = {
            "not_formal_benchmark": True,
            "not_high_fidelity_solver": True,
            "not_model_performance_evidence": True,
            "not_ood_generalization_evidence": True,
        }
    return {
        "schema_version": str(manifest.get("schema_version", "physics_label_medium_expansion_v2")),
        "subset_name": subset_name,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "stage": sample_stage,
        "description": description,
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": bc["h_W_m2K"], "ambient_temperature_K": bc["top_K"]},
            "bottom": {"fixed_temperature_K": bc["bottom_K"]},
            "sides": {"adiabatic": True},
        },
        "interfaces": [{"type": "perfect_contact", "note": "rectilinear equivalent-layer expansion smoke"}],
        "stack": {
            "stack_template": sample["stack_template"],
            "layers": [
                {
                    "layer_id": idx,
                    "name": layer["name"],
                    "z_min_m": layer["z"][0],
                    "z_max_m": layer["z"][1],
                    "material_id": layer["material_id"],
                    "k_iso_W_mK": layer["k_iso"],
                }
                for idx, layer in enumerate(layers)
            ],
            "layer_names": layer_names,
        },
        "generation_config": {
            "source_manifest": str(manifest_path),
            "manifest_version": manifest.get("manifest_version"),
            "dataset_name": manifest.get("dataset_name"),
            "sample_plan": sample,
            "grid_shape": list(grid_shape),
            "domain_bounds_m": DOMAIN_BOUNDS,
            "region_first": True,
            "source_assignment": "volume_fraction",
            "q_policy": "fixed_density",
            "reference_solver": "heat3d_v1_reference_solver_v2",
            **non_claim_flags,
        },
        "source_diagnostics": source_summary,
        "validation": {
            "temperature_required": True,
            "label_meta_required": True,
            "source_power_consistency_required": True,
            "label_diagnostics_required": True,
        },
        "units": {"coords": "m", "k_field": "W/m/K", "q_field": "W/m^3", "temperature": "K"},
    }


def _validate_output_path(path: Path, overwrite: bool) -> Path:
    output_subset = path.resolve()
    if output_subset.name in PROTECTED_SUBSET_NAMES:
        raise ValueError(f"refusing to write protected subset: {output_subset.name}")
    try:
        output_subset.relative_to(REPO_ROOT / "data")
    except ValueError as exc:
        raise ValueError(f"output subset must be under ignored data/: {output_subset}") from exc
    if output_subset.exists() and not overwrite:
        raise FileExistsError(f"output subset exists: {output_subset}; use --overwrite")
    return output_subset


def _select_samples(manifest: dict[str, Any], sample_ids: list[str] | None) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")
    selected = [sample for sample in samples if isinstance(sample, dict)]
    if sample_ids is None:
        return selected
    requested = set(sample_ids)
    filtered = [sample for sample in selected if sample["sample_id"] in requested]
    missing = sorted(requested - {sample["sample_id"] for sample in filtered})
    if missing:
        raise ValueError(f"requested sample ids missing from manifest: {missing}")
    return filtered


def _write_sample(
    samples_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    grid_shape = RESOLUTION_MAP[sample["resolution_category"]]
    layers = STACK_TEMPLATES[sample["stack_template"]]
    coords, axes = _grid(grid_shape)
    layer_id, region_id, material_id, layer_names = _layer_arrays(coords, layers)
    k_field = _k_field(coords, layers, sample)
    sources = _resolved_sources(sample, layers)
    q_field, source_summary = assign_q_field_volume_fraction(axes["x"], axes["y"], axes["z"], sources)

    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "layer_id.npy", layer_id)
    np.save(sample_dir / "region_id.npy", region_id)
    np.save(sample_dir / "material_id.npy", material_id)
    np.save(sample_dir / "k_field.npy", k_field)
    np.save(sample_dir / "q_field.npy", q_field)
    _write_json(sample_dir / "sample_meta.json", _sample_meta(
        manifest, manifest_path, sample, layers, grid_shape, layer_names, source_summary
    ))

    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    label_meta = dict(label_meta)
    label_role = str(manifest.get("label_role", "physics_label_medium_expansion_smoke"))
    non_claim_flags = dict(manifest.get("non_claim_flags", {}))
    if not non_claim_flags:
        non_claim_flags = {
            "not_formal_benchmark": True,
            "not_high_fidelity_solver": True,
            "not_model_performance_evidence": True,
            "not_ood_generalization_evidence": True,
        }
    label_meta.update({
        "sample_id": sample_id,
        "label_role": label_role,
        "source_assignment": "volume_fraction",
        "q_policy": "fixed_density",
        "source_diagnostics": source_summary,
        **non_claim_flags,
    })
    np.save(sample_dir / "temperature.npy", temperature)
    _write_json(sample_dir / "label_meta.json", label_meta)
    return {
        "sample_id": sample_id,
        "split": sample["split"],
        "source_pattern_tag": sample["source_pattern_tag"],
        "k_region_mode": sample["k_region_mode"],
        "stack_template": sample["stack_template"],
        "bc_category": sample["bc_category"],
        "k_shape": list(k_field.shape),
        "source_missed": source_summary["source_missed"],
        "integrated_q_power": source_summary["integrated_q_power"],
        "integrated_q_power_relative_error": source_summary["integrated_q_power_relative_error"],
        "active_source_volume_discrete": source_summary["active_source_volume_discrete"],
        "T_min": float(np.min(temperature)),
        "T_max": float(np.max(temperature)),
        "convergence_flag": label_meta["convergence_flag"],
        "residual_norm": label_meta["residual_norm"],
        "bottom_dirichlet_error": label_meta["bottom_dirichlet_error"],
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    samples = _select_samples(manifest, args.sample_ids)
    output_subset = _validate_output_path(args.output_subset, overwrite=args.overwrite)
    print("Heat3D v1 physics-label medium expansion generator")
    print(f"manifest: {manifest_path}")
    print(f"output_subset: {output_subset}")
    print(f"selected_sample_count: {len(samples)}")
    print(f"split_counts: {dict(Counter(sample['split'] for sample in samples))}")
    print("scope: 24-sample medium expansion smoke / benchmark-candidate planning diagnostic only")
    print("source_assignment: volume_fraction")
    print("q_policy: fixed_density")
    if not args.write:
        print("write_enabled: False")
        print("no_data_written: True")
        return 0
    if output_subset.exists() and args.overwrite:
        shutil.rmtree(output_subset)
    samples_dir = output_subset / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    summaries = [_write_sample(samples_dir, manifest, manifest_path, sample) for sample in samples]
    print("write_enabled: True")
    print(f"wrote_sample_count: {len(summaries)}")
    for summary in summaries:
        print(
            "- "
            f"{summary['sample_id']} split={summary['split']} "
            f"source={summary['source_pattern_tag']} stack={summary['stack_template']} "
            f"k={summary['k_region_mode']} bc={summary['bc_category']} "
            f"k_shape={summary['k_shape']} source_missed={summary['source_missed']} "
            f"active_volume={summary['active_source_volume_discrete']:.6e} "
            f"integrated_power={summary['integrated_q_power']:.6e} "
            f"power_rel_error={summary['integrated_q_power_relative_error']:.6e} "
            f"T_range=[{summary['T_min']:.6f}, {summary['T_max']:.6f}] "
            f"converged={summary['convergence_flag']} "
            f"residual_norm={summary['residual_norm']:.6e} "
            f"bottom_error={summary['bottom_dirichlet_error']:.6e}"
        )
    print("temperature_written: True")
    print("label_meta_written: True")
    print("formal_64_sample_dataset_generated: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
