#!/usr/bin/env python3
"""Generate the Heat3D v1 physics-label medium pilot smoke subset.

This generator reads the planning draft manifest and writes an 8-sample
region-first / volume-fraction pilot subset under ignored data/. It is a
physics-label smoke path only; it does not create a formal benchmark or a
high-fidelity dataset.
"""

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


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v1_physics_label_medium_manifest_draft.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_pilot_v2"
)
PROTECTED_SUBSET_NAMES = {
    "v1_multilayer_bc_eq_demo",
    "v1_multilayer_bc_eq_supervised_smoke",
    "v1_multilayer_bc_eq_supervised_small",
    "v1_multilayer_bc_eq_physics_label_small_v2",
}
DOMAIN_BOUNDS = {"x": (0.0, 0.01), "y": (0.0, 0.01), "z": (0.0, 0.002)}
RESOLUTION_MAP = {"pilot_mid": (8, 8, 6)}
Q_DENSITY_MAP = {"low": 0.5e8, "nominal": 1.0e8, "high": 1.5e8}
BC_CATEGORY_MAP = {
    "baseline_300K_nominal_top_h": {
        "bottom_fixed_temperature_K": 300.0,
        "top_ambient_temperature_K": 300.0,
        "top_h_W_m2K": 1000.0,
        "parameter_status": "provisional_engineering_assumption",
    },
    "baseline_300K_held_out_top_h": {
        "bottom_fixed_temperature_K": 300.0,
        "top_ambient_temperature_K": 300.0,
        "top_h_W_m2K": 2000.0,
        "parameter_status": "requires_user_confirmation",
        "role": "held-out HTC smoke candidate only",
    },
}
STACK_TEMPLATES = {
    "baseline_4_layer": [
        {"name": "substrate", "z": (0.0, 0.0004), "k_iso": 90.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.0004, 0.0010), "k_iso": 22.0, "material_id": 2},
        {"name": "tim_equivalent", "z": (0.0010, 0.0013), "k_iso": 5.0, "material_id": 3},
        {"name": "heat_spreader_equivalent", "z": (0.0013, 0.0020), "k_iso": 140.0, "material_id": 4},
    ],
    "dual_active_4_layer": [
        {"name": "substrate", "z": (0.0, 0.00035), "k_iso": 90.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.00035, 0.00085), "k_iso": 22.0, "material_id": 2},
        {"name": "interposer_equivalent", "z": (0.00085, 0.00135), "k_iso": 45.0, "material_id": 5},
        {"name": "active_die_1", "z": (0.00135, 0.0020), "k_iso": 18.0, "material_id": 6},
    ],
    "heldout_interposer_4_layer": [
        {"name": "substrate", "z": (0.0, 0.0003), "k_iso": 85.0, "material_id": 1},
        {"name": "active_die_0", "z": (0.0003, 0.00075), "k_iso": 22.0, "material_id": 2},
        {"name": "interposer_like_equivalent", "z": (0.00075, 0.00145), "k_iso": 55.0, "material_id": 7},
        {"name": "heat_spreader_equivalent", "z": (0.00145, 0.0020), "k_iso": 140.0, "material_id": 4},
    ],
}
SOURCE_REL_TOL = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Heat3D v1 medium pilot physics-label smoke samples."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--write", action="store_true", help="Write the ignored pilot subset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing pilot subset.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _axis(bounds: tuple[float, float], count: int) -> np.ndarray:
    return np.linspace(bounds[0], bounds[1], count, dtype=np.float64)


def _grid(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xs = _axis(DOMAIN_BOUNDS["x"], grid_shape[0])
    ys = _axis(DOMAIN_BOUNDS["y"], grid_shape[1])
    zs = _axis(DOMAIN_BOUNDS["z"], grid_shape[2])
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    return coords, {"x": xs, "y": ys, "z": zs}


def _layer_for_z(layers: list[dict[str, Any]], z: float) -> tuple[int, dict[str, Any]]:
    for idx, layer in enumerate(layers):
        z_min, z_max = layer["z"]
        if idx == len(layers) - 1:
            inside = z >= z_min and z <= z_max
        else:
            inside = z >= z_min and z < z_max
        if inside:
            return idx, layer
    raise ValueError(f"z coordinate outside stack template: {z}")


def _layer_arrays(coords: np.ndarray, layers: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layer_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    region_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    material_id = np.empty((coords.shape[0], 1), dtype=np.int32)
    layer_names: list[str] = []
    for idx, point in enumerate(coords):
        layer_idx, layer = _layer_for_z(layers, float(point[2]))
        layer_id[idx, 0] = layer_idx
        region_id[idx, 0] = layer_idx
        material_id[idx, 0] = int(layer["material_id"])
    for layer in layers:
        layer_names.append(str(layer["name"]))
    return layer_id, region_id, material_id, layer_names


def _k_field(coords: np.ndarray, layers: list[dict[str, Any]], k_region_mode: str) -> np.ndarray:
    if k_region_mode == "diagonal_anisotropic_diagnostic":
        k = np.empty((coords.shape[0], 3), dtype=np.float64)
        for idx, point in enumerate(coords):
            _, layer = _layer_for_z(layers, float(point[2]))
            base = float(layer["k_iso"])
            k[idx, 0] = base * 1.20
            k[idx, 1] = base * 0.90
            k[idx, 2] = base * 0.55
        return k

    if k_region_mode not in {"isotropic_equivalent", "blockwise_equivalent"}:
        raise ValueError(f"unsupported k_region_mode: {k_region_mode}")

    k = np.empty((coords.shape[0], 1), dtype=np.float64)
    for idx, point in enumerate(coords):
        _, layer = _layer_for_z(layers, float(point[2]))
        value = float(layer["k_iso"])
        if k_region_mode == "blockwise_equivalent" and layer["name"] == "active_die_0":
            value *= 0.75 if point[0] < 0.5 * sum(DOMAIN_BOUNDS["x"]) else 1.25
        k[idx, 0] = value
    return k


def _source_box_from_region(region: dict[str, Any], layers: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    layer_name = region["layer"]
    layer = next((item for item in layers if item["name"] == layer_name), None)
    if layer is None:
        raise ValueError(f"source layer {layer_name!r} not found in stack template")

    x0, x1 = DOMAIN_BOUNDS["x"]
    y0, y1 = DOMAIN_BOUNDS["y"]
    x_width = x1 - x0
    y_width = y1 - y0
    cx_frac, cy_frac = region["center_xy_fraction"]
    sx_frac, sy_frac = region["size_xy_fraction"]
    cx = x0 + float(cx_frac) * x_width
    cy = y0 + float(cy_frac) * y_width
    sx = float(sx_frac) * x_width
    sy = float(sy_frac) * y_width
    z_min, z_max = layer["z"]
    z_span = z_max - z_min
    return {
        "x": (max(x0, cx - 0.5 * sx), min(x1, cx + 0.5 * sx)),
        "y": (max(y0, cy - 0.5 * sy), min(y1, cy + 0.5 * sy)),
        "z": (z_min + 0.20 * z_span, z_min + 0.80 * z_span),
    }


def _resolved_source_regions(sample: dict[str, Any], layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    for region in sample["source_regions"]:
        q_category = region["q_scale_category"]
        if q_category not in Q_DENSITY_MAP:
            raise ValueError(f"unsupported q_scale_category: {q_category}")
        resolved.append({
            "region_id": region["region_id"],
            "layer": region["layer"],
            "source_box_m": _source_box_from_region(region, layers),
            "q_density_W_m3": Q_DENSITY_MAP[q_category],
            "q_scale_category": q_category,
            "parameter_status": sample.get("parameter_status"),
        })
    return resolved


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
    return {
        "schema_version": "physics_label_medium_pilot_v2",
        "subset_name": "v1_multilayer_bc_eq_physics_label_medium_pilot_v2",
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "stage": "physics_label_medium_pilot_smoke",
        "description": "Region-first volume-fraction physics-label medium pilot smoke sample.",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {
                "h_W_m2K": bc["top_h_W_m2K"],
                "ambient_temperature_K": bc["top_ambient_temperature_K"],
            },
            "bottom": {"fixed_temperature_K": bc["bottom_fixed_temperature_K"]},
            "sides": {"adiabatic": True},
        },
        "interfaces": [{"type": "perfect_contact", "note": "rectilinear equivalent-layer pilot"}],
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
            "not_formal_benchmark": True,
            "not_high_fidelity_solver": True,
            "not_model_performance_evidence": True,
            "not_ood_generalization_evidence": True,
        },
        "source_diagnostics": source_summary,
        "validation": {
            "temperature_required": True,
            "label_meta_required": True,
            "label_diagnostics_required": True,
            "source_power_consistency_required": True,
        },
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _validate_output_path(output_subset: Path, overwrite: bool) -> Path:
    output_subset = output_subset.resolve()
    if output_subset.name in PROTECTED_SUBSET_NAMES:
        raise ValueError(f"refusing to write protected subset: {output_subset.name}")
    try:
        output_subset.relative_to(REPO_ROOT / "data")
    except ValueError as exc:
        raise ValueError(f"output subset must be under ignored data/: {output_subset}") from exc
    if output_subset.exists() and not overwrite:
        raise FileExistsError(f"output subset already exists: {output_subset}; use --overwrite")
    return output_subset


def _select_samples(manifest: dict[str, Any], sample_ids: list[str] | None) -> list[dict[str, Any]]:
    samples = manifest.get("pilot_samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest pilot_samples must be a list")
    if sample_ids is None:
        return [sample for sample in samples if isinstance(sample, dict)]
    requested = set(sample_ids)
    selected = [sample for sample in samples if isinstance(sample, dict) and sample.get("sample_id") in requested]
    found = {sample.get("sample_id") for sample in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"requested pilot sample ids not found: {missing}")
    return selected


def _write_sample(
    target_samples: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    grid_shape = RESOLUTION_MAP[sample["resolution_category"]]
    layers = STACK_TEMPLATES[sample["stack_template"]]
    coords, axes = _grid(grid_shape)
    layer_id, region_id, material_id, layer_names = _layer_arrays(coords, layers)
    k_field = _k_field(coords, layers, sample["k_region_mode"])
    resolved_sources = _resolved_source_regions(sample, layers)
    q_field, source_summary = assign_q_field_volume_fraction(
        axes["x"],
        axes["y"],
        axes["z"],
        resolved_sources,
    )

    sample_dir = target_samples / sample_id
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "layer_id.npy", layer_id)
    np.save(sample_dir / "region_id.npy", region_id)
    np.save(sample_dir / "material_id.npy", material_id)
    np.save(sample_dir / "k_field.npy", k_field)
    np.save(sample_dir / "q_field.npy", q_field)
    meta = _sample_meta(
        manifest=manifest,
        manifest_path=manifest_path,
        sample=sample,
        layers=layers,
        grid_shape=grid_shape,
        layer_names=layer_names,
        source_summary=source_summary,
    )
    _write_json(sample_dir / "sample_meta.json", meta)

    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    label_meta = dict(label_meta)
    label_meta.update({
        "sample_id": sample_id,
        "label_role": "physics_label_medium_pilot_smoke",
        "source_assignment": "volume_fraction",
        "q_policy": "fixed_density",
        "source_diagnostics": source_summary,
        "not_formal_benchmark": True,
        "not_high_fidelity_solver": True,
        "not_model_performance_evidence": True,
        "not_ood_generalization_evidence": True,
    })
    np.save(sample_dir / "temperature.npy", temperature)
    _write_json(sample_dir / "label_meta.json", label_meta)

    return {
        "sample_id": sample_id,
        "split": sample["split"],
        "purpose_tag": sample["purpose_tag"],
        "k_shape": list(k_field.shape),
        "node_count": int(coords.shape[0]),
        "source_missed": source_summary["source_missed"],
        "active_source_volume_discrete": source_summary["active_source_volume_discrete"],
        "integrated_q_power": source_summary["integrated_q_power"],
        "integrated_q_power_relative_error": source_summary["integrated_q_power_relative_error"],
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "convergence_flag": label_meta["convergence_flag"],
        "residual_norm": label_meta["residual_norm"],
        "bottom_dirichlet_error": label_meta["bottom_dirichlet_error"],
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    selected = _select_samples(manifest, args.sample_ids)
    output_subset = _validate_output_path(args.output_subset, overwrite=args.overwrite)
    split_counts = Counter(sample["split"] for sample in selected)

    print("Heat3D v1 physics-label medium pilot generator")
    print(f"manifest: {manifest_path}")
    print(f"output_subset: {output_subset}")
    print(f"selected_sample_ids: {[sample['sample_id'] for sample in selected]}")
    print(f"split_counts: {dict(split_counts)}")
    print("scope: medium pilot / physics-label smoke / benchmark-candidate pilot only")
    print("source_assignment: volume_fraction")
    print("q_policy: fixed_density")

    if not args.write:
        print("write_enabled: False")
        print("no_data_written: True")
        return 0

    if output_subset.exists() and args.overwrite:
        shutil.rmtree(output_subset)
    target_samples = output_subset / "samples"
    target_samples.mkdir(parents=True, exist_ok=False)
    summaries = [_write_sample(target_samples, manifest, manifest_path, sample) for sample in selected]

    print("write_enabled: True")
    print(f"wrote_sample_count: {len(summaries)}")
    for summary in summaries:
        print(
            "- "
            f"{summary['sample_id']} split={summary['split']} purpose={summary['purpose_tag']} "
            f"k_shape={summary['k_shape']} nodes={summary['node_count']} "
            f"source_missed={summary['source_missed']} "
            f"active_volume={summary['active_source_volume_discrete']:.6e} "
            f"integrated_power={summary['integrated_q_power']:.6e} "
            f"power_rel_error={summary['integrated_q_power_relative_error']:.6e} "
            f"T_range=[{summary['temperature_min']:.6f}, {summary['temperature_max']:.6f}] "
            f"converged={summary['convergence_flag']} "
            f"residual_norm={summary['residual_norm']:.6e} "
            f"bottom_error={summary['bottom_dirichlet_error']:.6e}"
        )
    print("temperature_written: True")
    print("label_meta_written: True")
    print("old_supervised_small_overwritten: False")
    print("formal_64_sample_dataset_generated: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
