#!/usr/bin/env python3
"""Generate Heat3D v3 final-target probe v0 samples.

This writes a small ignored diagnostic subset under data/. It does not train a
model and does not create a formal benchmark. The v0 generator intentionally
keeps model inputs pure-physics: coords, k_field, q_field, and BC metadata.
Layer/region/material ids are bookkeeping metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v3_final_target_probe_manifest_v0.json"
DEFAULT_OUTPUT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v3_final_target_probe_v0"
)
DOMAIN_BOUNDS = {"x": (0.0, 0.01), "y": (0.0, 0.01), "z": (0.0, 0.002)}
BC_TOP_H = {
    "nominal_top_h": 1000.0,
    "low_top_h": 450.0,
    "high_top_h": 1800.0,
    "very_high_top_h_candidate": 3400.0,
}
BASE_Q = {
    "trace": 0.04e8,
    "low": 0.20e8,
    "nominal": 0.75e8,
    "nominal_high": 1.10e8,
    "high": 1.55e8,
    "high_dynamic_range": 1.90e8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subset", type=Path, default=DEFAULT_OUTPUT_SUBSET)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        choices=(1024, 4096),
        help="Generate one manifest-supported resolution. 4096 is opt-in.",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_output_path(path: Path, overwrite: bool) -> Path:
    output_subset = path.resolve()
    try:
        output_subset.relative_to(REPO_ROOT / "data")
    except ValueError as exc:
        raise ValueError(f"output subset must be under ignored data/: {output_subset}") from exc
    if output_subset.exists() and not overwrite:
        raise FileExistsError(f"output subset exists: {output_subset}; use --overwrite")
    return output_subset


def _grid(shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    xs = np.linspace(*DOMAIN_BOUNDS["x"], shape[0], dtype=np.float64)
    ys = np.linspace(*DOMAIN_BOUNDS["y"], shape[1], dtype=np.float64)
    zs = np.linspace(*DOMAIN_BOUNDS["z"], shape[2], dtype=np.float64)
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    fx = (coords[:, 0] - DOMAIN_BOUNDS["x"][0]) / (DOMAIN_BOUNDS["x"][1] - DOMAIN_BOUNDS["x"][0])
    fy = (coords[:, 1] - DOMAIN_BOUNDS["y"][0]) / (DOMAIN_BOUNDS["y"][1] - DOMAIN_BOUNDS["y"][0])
    fz = (coords[:, 2] - DOMAIN_BOUNDS["z"][0]) / (DOMAIN_BOUNDS["z"][1] - DOMAIN_BOUNDS["z"][0])
    fractions = {"x": fx, "y": fy, "z": fz}
    axes = {"x": xs, "y": ys, "z": zs}
    return coords, fractions, axes


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _box_mask(frac: dict[str, np.ndarray], center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    return (
        (np.abs(frac["x"] - center[0]) <= 0.5 * size[0])
        & (np.abs(frac["y"] - center[1]) <= 0.5 * size[1])
        & (np.abs(frac["z"] - center[2]) <= 0.5 * size[2])
    )


def _ellipsoid_mask(
    frac: dict[str, np.ndarray],
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
) -> np.ndarray:
    value = (
        ((frac["x"] - center[0]) / radius[0]) ** 2
        + ((frac["y"] - center[1]) / radius[1]) ** 2
        + ((frac["z"] - center[2]) / radius[2]) ** 2
    )
    return value <= 1.0


def _cylinder_z_mask(frac: dict[str, np.ndarray], center_xy: tuple[float, float], radius: float) -> np.ndarray:
    return ((frac["x"] - center_xy[0]) ** 2 + (frac["y"] - center_xy[1]) ** 2) <= radius**2


def _random_block_background(
    frac: dict[str, np.ndarray],
    seed: int,
    *,
    count: int,
    low: float,
    high: float,
    base: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = _rng(seed)
    k = np.full(frac["x"].shape, base, dtype=np.float64)
    region = np.zeros(frac["x"].shape, dtype=np.int32)
    blocks: list[dict[str, Any]] = []
    for idx in range(count):
        center = tuple(float(v) for v in rng.uniform([0.15, 0.15, 0.15], [0.85, 0.85, 0.85]))
        size = tuple(float(v) for v in rng.uniform([0.16, 0.16, 0.25], [0.42, 0.42, 0.90]))
        high_k = bool(idx % 2 == 0)
        value = float(rng.uniform(95.0, high)) if high_k else float(rng.uniform(low, 9.0))
        mask = _box_mask(frac, center, size)
        k[mask] = value
        region[mask] = idx + 1
        blocks.append({
            "id": idx + 1,
            "center_fraction": list(center),
            "size_fraction": list(size),
            "k_W_mK": value,
            "class": "high_k_block" if high_k else "low_k_block",
        })
    return k, region, blocks


def _apply_k_scene(
    probe: dict[str, Any],
    frac: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(probe["semantic_scene_seed"])
    kind = str(probe["scene_kind"])
    base = 32.0
    k, region, blocks = _random_block_background(frac, seed, count=7, low=1.8, high=170.0, base=base)
    notes: list[dict[str, Any]] = [{"kind": "random_block_background", "blocks": blocks}]

    if kind == "sparse_high_k_bridges":
        for idx, center_xy in enumerate(((0.30, 0.32), (0.72, 0.62), (0.54, 0.78))):
            mask = _cylinder_z_mask(frac, center_xy, radius=0.065 + 0.01 * idx)
            k[mask] = 230.0 - 20.0 * idx
            region[mask] = 20 + idx
        notes.append({"kind": "sparse_high_k_bridges", "count": 3})
    elif kind == "dense_low_k_barriers_around_source":
        barriers = [
            ((0.50, 0.34, 0.55), (0.62, 0.10, 0.90)),
            ((0.50, 0.66, 0.55), (0.62, 0.10, 0.90)),
            ((0.34, 0.50, 0.55), (0.10, 0.62, 0.90)),
            ((0.66, 0.50, 0.55), (0.10, 0.62, 0.90)),
        ]
        for idx, (center, size) in enumerate(barriers):
            mask = _box_mask(frac, center, size)
            k[mask] = 1.25 + 0.25 * idx
            region[mask] = 30 + idx
        notes.append({"kind": "low_k_barrier_ring", "count": len(barriers)})
    elif kind == "mixed_block_sizes_high_contrast_interfaces":
        nested = [
            ((0.50, 0.50, 0.50), (0.72, 0.72, 0.90), 4.0, 40),
            ((0.50, 0.50, 0.50), (0.38, 0.38, 0.80), 165.0, 41),
            ((0.50, 0.50, 0.50), (0.16, 0.16, 0.60), 2.5, 42),
        ]
        for center, size, value, region_id in nested:
            mask = _box_mask(frac, center, size)
            k[mask] = value
            region[mask] = region_id
        notes.append({"kind": "nested_multiscale_interface", "count": len(nested)})
    elif kind == "tsv_like_vertical_high_k_path":
        mask = _cylinder_z_mask(frac, (0.58, 0.45), radius=0.075)
        k[mask] = 260.0
        region[mask] = 50
        shell = _cylinder_z_mask(frac, (0.58, 0.45), radius=0.145) & ~mask
        k[shell] = np.minimum(k[shell], 7.0)
        region[shell] = 51
        notes.append({"kind": "tsv_like_vertical_high_k_path", "center_xy_fraction": [0.58, 0.45]})
    elif kind == "localized_diag3_anisotropic_patch":
        patch = _box_mask(frac, (0.48, 0.52, 0.55), (0.42, 0.34, 0.70))
        k_diag = np.repeat(k.reshape(-1, 1), repeats=3, axis=1)
        k_diag[patch, 0] *= 3.2
        k_diag[patch, 1] *= 0.65
        k_diag[patch, 2] *= 0.22
        region[patch] = 60
        notes.append({
            "kind": "localized_diag3_anisotropic_patch",
            "diag_ratio_patch": [3.2, 0.65, 0.22],
            "full_tensor_k_status": "schema_solver_gap_not_generated",
        })
        return k_diag, region, _materials_for_region(region, k_diag), notes

    return k.reshape(-1, 1), region, _materials_for_region(region, k), notes


def _materials_for_region(region: np.ndarray, k_values: np.ndarray) -> list[dict[str, Any]]:
    materials = []
    flat = k_values if k_values.ndim == 1 else k_values[:, 0]
    for region_id in sorted(int(value) for value in np.unique(region)):
        mask = region == region_id
        materials.append({
            "id": region_id,
            "name": f"probe_region_{region_id}",
            "role": "metadata_only_region_material",
            "k_representative_W_mK": float(np.median(flat[mask])),
        })
    return materials


def _add_q_box(q: np.ndarray, frac: dict[str, np.ndarray], center: tuple[float, float, float], size: tuple[float, float, float], value: float) -> None:
    q[_box_mask(frac, center, size)] += value


def _add_q_ellipsoid(q: np.ndarray, frac: dict[str, np.ndarray], center: tuple[float, float, float], radius: tuple[float, float, float], value: float) -> None:
    q[_ellipsoid_mask(frac, center, radius)] += value


def _apply_q_scene(probe: dict[str, Any], frac: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source = str(probe["source_category"])
    q_range = str(probe["q_power_range"])
    base = BASE_Q.get(q_range, BASE_Q["nominal"])
    q = np.zeros(frac["x"].shape, dtype=np.float64)
    sources: list[dict[str, Any]] = []

    def box(name: str, center: tuple[float, float, float], size: tuple[float, float, float], scale: float) -> None:
        value = base * scale
        _add_q_box(q, frac, center, size, value)
        sources.append({"name": name, "shape": "box", "center_fraction": list(center), "size_fraction": list(size), "q_density_W_m3": value})

    def ellipsoid(name: str, center: tuple[float, float, float], radius: tuple[float, float, float], scale: float) -> None:
        value = base * scale
        _add_q_ellipsoid(q, frac, center, radius, value)
        sources.append({"name": name, "shape": "ellipsoid", "center_fraction": list(center), "radius_fraction": list(radius), "q_density_W_m3": value})

    if source == "multi_block_power":
        box("src_a", (0.30, 0.30, 0.50), (0.18, 0.18, 0.55), 0.9)
        box("src_b", (0.70, 0.44, 0.50), (0.16, 0.20, 0.55), 1.1)
        box("src_c", (0.48, 0.72, 0.50), (0.20, 0.16, 0.55), 0.7)
    elif source == "compact_hotspot_with_weak_background":
        q[:] += BASE_Q["trace"]
        ellipsoid("src_hdr_hotspot", (0.40, 0.40, 0.50), (0.13, 0.13, 0.34), 1.0)
        sources.append({"name": "weak_background", "shape": "global", "q_density_W_m3": BASE_Q["trace"]})
    elif source == "contained_hotspot":
        ellipsoid("src_contained", (0.50, 0.50, 0.50), (0.15, 0.15, 0.42), 1.0)
    elif source == "multi_blob_power":
        ellipsoid("blob_a", (0.24, 0.30, 0.30), (0.12, 0.16, 0.30), 0.7)
        ellipsoid("blob_b", (0.58, 0.48, 0.62), (0.18, 0.12, 0.34), 1.1)
        ellipsoid("blob_c", (0.76, 0.72, 0.50), (0.10, 0.14, 0.28), 0.8)
    elif source == "elongated_power":
        q[:] += BASE_Q["trace"] * 0.5
        box("elongated_strip", (0.50, 0.50, 0.52), (0.70, 0.13, 0.55), 1.0)
        sources.append({"name": "weak_background", "shape": "global", "q_density_W_m3": BASE_Q["trace"] * 0.5})
    elif source == "via_adjacent_hotspot":
        ellipsoid("via_adjacent", (0.50, 0.43, 0.38), (0.12, 0.12, 0.30), 1.0)
    elif source == "active_hotspot_motif":
        box("active_a", (0.35, 0.42, 0.48), (0.16, 0.16, 0.35), 1.0)
        box("active_b", (0.65, 0.58, 0.48), (0.16, 0.16, 0.35), 0.55)
        q += BASE_Q["trace"] * _box_mask(frac, (0.50, 0.50, 0.48), (0.62, 0.44, 0.40))
    elif source == "patch_adjacent_hotspot":
        ellipsoid("patch_adjacent", (0.44, 0.48, 0.50), (0.14, 0.12, 0.32), 1.0)
    elif source == "compact_hotspot":
        ellipsoid("compact", (0.52, 0.48, 0.55), (0.13, 0.13, 0.36), 1.0)
    else:
        raise ValueError(f"unsupported source_category: {source}")

    return q.reshape(-1, 1), sources


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _sample_id(probe: dict[str, Any], resolution: int) -> str:
    return f"v3_probe_{probe['probe_id']}_r{resolution}"


def _top_h_for_probe(probe: dict[str, Any]) -> float:
    return BC_TOP_H.get(str(probe["bc_category"]), BC_TOP_H["nominal_top_h"])


def _sample_meta(
    manifest: dict[str, Any],
    manifest_path: Path,
    probe: dict[str, Any],
    resolution: int,
    grid_shape: tuple[int, int, int],
    materials: list[dict[str, Any]],
    k_notes: list[dict[str, Any]],
    q_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_id = _sample_id(probe, resolution)
    top_h = _top_h_for_probe(probe)
    notes = str(probe["generator_capability_notes"])
    return {
        "schema_version": "heat3d_v3_final_target_probe_sample_v0",
        "subset_name": manifest["output_subset_name"],
        "sample_id": sample_id,
        "stage": manifest["sample_stage"],
        "split": "test_ood_final_probe",
        "probe_id": probe["probe_id"],
        "probe_family": probe["probe_family"],
        "intended_stressor": probe["intended_stressor"],
        "resolution": resolution,
        "paired_scene_key": f"{probe['probe_id']}_seed{probe['semantic_scene_seed']}",
        "paired_4096_status": "deferred_v0" if resolution == 1024 else "generated_v0",
        "semantic_scene_seed": probe["semantic_scene_seed"],
        "k_mode": probe["k_mode"],
        "k_field_mode": probe["k_mode"],
        "k_region_mode": probe["k_region_mode"],
        "source_category": probe["source_category"],
        "source_pattern_tag": probe["source_category"],
        "q_power_range": probe["q_power_range"],
        "power_scale_category": probe["q_power_range"],
        "bc_category": probe["bc_category"],
        "stack_template": "random_block_first_probe",
        "label_status": "physics_label_generated",
        "generator_capability_notes": notes,
        "model_input_policy": "pure_physics_coords_k_q_bc_only",
        "layer_region_material_input_status": "metadata_only_not_default_model_input",
        "domain": {
            "bounds_m": manifest["domain_bounds_m"],
            "grid_shape": list(grid_shape),
            "point_count": int(np.prod(grid_shape)),
        },
        "layers": [
            {
                "id": 0,
                "name": "full_volume_probe_domain",
                "z_min_m": DOMAIN_BOUNDS["z"][0],
                "z_max_m": DOMAIN_BOUNDS["z"][1],
                "metadata_only": True,
            }
        ],
        "regions": [
            {
                "id": item["id"],
                "name": item["name"],
                "layer_id": 0,
                "material_id": item["id"],
                "metadata_only": True,
            }
            for item in materials
        ],
        "materials": materials,
        "boundary_regions": [
            {"name": "top", "surface": "z_max"},
            {"name": "bottom", "surface": "z_min"},
            {"name": "sides", "surface": "x_or_y_minmax"},
        ],
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": top_h, "ambient_temperature_K": 300.0},
            "bottom": {"fixed_temperature_K": 300.0},
            "sides": {"heat_flux_W_m2": 0.0},
        },
        "interfaces": [],
        "generation_config": {
            "source_manifest": str(manifest_path),
            "manifest_version": manifest["manifest_version"],
            "dataset_name": manifest["dataset_name"],
            "probe_plan": probe,
            "grid_shape": list(grid_shape),
            "k_scene_notes": k_notes,
            "q_sources": q_sources,
            "reference_solver": manifest["generation_policy"]["reference_solver"],
            "not_formal_benchmark": True,
            "not_high_fidelity_solver": True,
            "not_model_performance_evidence": True,
            "not_publication_ready_dataset": True,
        },
        "units": {
            "coords": "m",
            "k_field": "W/(m*K)",
            "q_field": "W/m^3",
            "temperature": "K",
            "htc": "W/(m^2*K)",
        },
        "validation": {
            "temperature_required": True,
            "label_meta_required": True,
            "solver_residual_required": True,
            "duplicate_q_k_T_hash_forbidden_within_subset": True,
        },
        "parameter_sources": {
            "literature_backed": [],
            "provisional_engineering_assumption": [
                "probe geometry",
                "probe material conductivity values",
                "probe power density values",
                "probe top Robin HTC values",
            ],
            "requires_user_confirmation": [],
        },
    }


def _metadata_json(
    meta: dict[str, Any],
    label_meta: dict[str, Any],
    coords: np.ndarray,
    k_field: np.ndarray,
    q_field: np.ndarray,
    temperature: np.ndarray,
) -> dict[str, Any]:
    return {
        "metadata_schema_version": "heat3d_v3_final_target_probe_metadata_v0",
        "sample_id": meta["sample_id"],
        "split": meta["split"],
        "probe_id": meta["probe_id"],
        "probe_family": meta["probe_family"],
        "intended_stressor": meta["intended_stressor"],
        "resolution": meta["resolution"],
        "paired_scene_key": meta["paired_scene_key"],
        "paired_4096_status": meta["paired_4096_status"],
        "semantic_scene_seed": meta["semantic_scene_seed"],
        "k_mode": meta["k_mode"],
        "k_field_mode": meta["k_field_mode"],
        "k_region_mode": meta["k_region_mode"],
        "source_category": meta["source_category"],
        "source_pattern_tag": meta["source_pattern_tag"],
        "q_power_range": meta["q_power_range"],
        "power_scale_category": meta["power_scale_category"],
        "bc_category": meta["bc_category"],
        "stack_template": meta["stack_template"],
        "label_status": meta["label_status"],
        "generator_capability_notes": meta["generator_capability_notes"],
        "model_input_policy": meta["model_input_policy"],
        "top_h_W_m2K": meta["boundary_params"]["top"]["h_W_m2K"],
        "point_count": int(coords.shape[0]),
        "k_shape": list(k_field.shape),
        "q_nonzero_count": int(np.count_nonzero(q_field)),
        "q_integral_proxy_sum": float(np.sum(q_field)),
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "deltaT_max_K": float(np.max(temperature - 300.0)),
        "convergence_flag": bool(label_meta.get("convergence_flag")),
        "residual_norm": float(label_meta.get("residual_norm", float("nan"))),
        "bottom_dirichlet_error": float(label_meta.get("bottom_dirichlet_error", float("nan"))),
        "q_hash": _array_hash(q_field),
        "k_hash": _array_hash(k_field),
        "temperature_hash": _array_hash(temperature),
    }


def _write_sample(
    samples_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    probe: dict[str, Any],
    resolution: int,
    grid_shape: tuple[int, int, int],
) -> dict[str, Any]:
    sample_id = _sample_id(probe, resolution)
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True)

    coords, frac, _axes = _grid(grid_shape)
    k_field, region_id, materials, k_notes = _apply_k_scene(probe, frac)
    q_field, q_sources = _apply_q_scene(probe, frac)
    layer_id = np.zeros((coords.shape[0],), dtype=np.int32)
    material_id = region_id.astype(np.int32)

    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "layer_id.npy", layer_id)
    np.save(sample_dir / "region_id.npy", region_id.astype(np.int32))
    np.save(sample_dir / "material_id.npy", material_id)
    np.save(sample_dir / "k_field.npy", k_field.astype(np.float64))
    np.save(sample_dir / "q_field.npy", q_field.astype(np.float64))

    meta = _sample_meta(manifest, manifest_path, probe, resolution, grid_shape, materials, k_notes, q_sources)
    _write_json(sample_dir / "sample_meta.json", meta)
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    label_meta = dict(label_meta)
    label_meta.update({
        "sample_id": sample_id,
        "probe_id": probe["probe_id"],
        "label_role": manifest["label_role"],
        "label_status": "physics_label_generated",
        "not_formal_benchmark": True,
        "not_high_fidelity_solver": True,
    })
    np.save(sample_dir / "temperature.npy", temperature)
    _write_json(sample_dir / "label_meta.json", label_meta)
    metadata = _metadata_json(meta, label_meta, coords, k_field, q_field, temperature)
    _write_json(sample_dir / "metadata.json", metadata)
    return {
        "sample_id": sample_id,
        "probe_id": probe["probe_id"],
        "resolution": resolution,
        "k_shape": list(k_field.shape),
        "q_nonzero_count": int(np.count_nonzero(q_field)),
        "T_min": float(np.min(temperature)),
        "T_max": float(np.max(temperature)),
        "residual_norm": float(label_meta["residual_norm"]),
        "bottom_dirichlet_error": float(label_meta["bottom_dirichlet_error"]),
        "convergence_flag": bool(label_meta["convergence_flag"]),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    resolution_config = manifest.get("resolutions", {}).get(str(args.resolution))
    if not isinstance(resolution_config, dict):
        raise ValueError(f"resolution {args.resolution} missing from manifest")
    if args.resolution == 4096 and resolution_config.get("status") != "enabled_v0":
        raise ValueError("4096 generation is intentionally deferred in v0; edit manifest only after stability gate")
    grid_shape = tuple(int(value) for value in resolution_config["grid_shape"])
    expected_points = int(np.prod(grid_shape))
    if expected_points != args.resolution:
        raise ValueError(f"grid shape {grid_shape} does not match resolution {args.resolution}")

    output_subset = _validate_output_path(args.output_subset, overwrite=args.overwrite)
    print("Heat3D v3 final-target probe generator")
    print(f"manifest: {manifest_path}")
    print(f"output_subset: {output_subset}")
    print(f"resolution: {args.resolution}")
    print(f"grid_shape: {grid_shape}")
    print(f"sample_count: {len(manifest.get('probes', []))}")
    print("scope: diagnostic final-target probe only; not a formal benchmark")
    if not args.write:
        print("write_enabled: False")
        print("no_data_written: True")
        return 0

    if output_subset.exists() and args.overwrite:
        shutil.rmtree(output_subset)
    samples_dir = output_subset / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_subset / "manifest_snapshot.json", manifest)
    summaries = [
        _write_sample(samples_dir, manifest, manifest_path, probe, args.resolution, grid_shape)
        for probe in manifest.get("probes", [])
    ]
    _write_json(output_subset / "generation_summary.json", {
        "manifest": str(manifest_path),
        "output_subset": str(output_subset),
        "resolution": args.resolution,
        "grid_shape": list(grid_shape),
        "sample_count": len(summaries),
        "summaries": summaries,
        "generated_data_committed_to_git": False,
    })
    print("write_enabled: True")
    print(f"wrote_sample_count: {len(summaries)}")
    for row in summaries:
        print(
            "- "
            f"{row['sample_id']} probe={row['probe_id']} k_shape={row['k_shape']} "
            f"q_nonzero={row['q_nonzero_count']} T=[{row['T_min']:.6f}, {row['T_max']:.6f}] "
            f"converged={row['convergence_flag']} residual={row['residual_norm']:.3e} "
            f"bottom_error={row['bottom_dirichlet_error']:.3e}"
        )
    print("formal_benchmark_generated: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
