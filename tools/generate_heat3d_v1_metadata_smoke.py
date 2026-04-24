"""Generate Heat3D v1 metadata-only smoke samples.

This tool writes small local smoke samples under the ignored data/ directory.
It does not run a solver and intentionally does not create temperature.npy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_schema import SCHEMA_VERSION, SUBSET_NAME, default_v1_samples_dir


FOOTPRINT_M = (0.010, 0.010)
NX = 4
NY = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Heat3D v1 metadata-only smoke samples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_v1_samples_dir(REPO_DIR),
        help="Directory where sample_xxx folders will be written.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional list of sample ids to generate. Defaults to the full configured set.",
    )
    return parser.parse_args()


def _stack_templates() -> dict[str, list[dict[str, Any]]]:
    return {
        "baseline_4_layer": [
            {"id": 0, "name": "substrate_equiv", "thickness_m": 0.0010, "base_k": 12.0},
            {"id": 1, "name": "active_die_0", "thickness_m": 0.00020, "base_k": 120.0},
            {"id": 2, "name": "tim_equiv", "thickness_m": 0.00008, "base_k": 5.0},
            {"id": 3, "name": "heatsink_equiv", "thickness_m": 0.0015, "base_k": 180.0},
        ],
        "compact_3_layer": [
            {"id": 0, "name": "substrate_equiv", "thickness_m": 0.0010, "base_k": 10.0},
            {"id": 1, "name": "active_die_0", "thickness_m": 0.00025, "base_k": 100.0},
            {"id": 2, "name": "heatsink_equiv", "thickness_m": 0.0015, "base_k": 160.0},
        ],
        "dual_active_4_layer": [
            {"id": 0, "name": "substrate_equiv", "thickness_m": 0.0010, "base_k": 12.0},
            {"id": 1, "name": "active_die_0", "thickness_m": 0.00018, "base_k": 115.0},
            {"id": 2, "name": "active_die_1", "thickness_m": 0.00018, "base_k": 110.0},
            {"id": 3, "name": "heatsink_equiv", "thickness_m": 0.0015, "base_k": 175.0},
        ],
        "heldout_interposer_4_layer": [
            {"id": 0, "name": "substrate_equiv", "thickness_m": 0.0010, "base_k": 12.0},
            {"id": 1, "name": "interposer_equiv", "thickness_m": 0.00035, "base_k": 35.0},
            {"id": 2, "name": "active_die_0", "thickness_m": 0.00020, "base_k": 110.0},
            {"id": 3, "name": "heatsink_equiv", "thickness_m": 0.0015, "base_k": 170.0},
        ],
    }


def _sample_configs() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "sample_000",
            "split": "train",
            "template": "baseline_4_layer",
            "heat_layers": ["active_die_0"],
            "blockwise_k": False,
            "description": "4-layer baseline metadata-only sample.",
        },
        {
            "sample_id": "sample_001",
            "split": "train",
            "template": "compact_3_layer",
            "heat_layers": ["active_die_0"],
            "blockwise_k": True,
            "description": "3-layer compact stack with block-wise equivalent conductivity.",
        },
        {
            "sample_id": "sample_002",
            "split": "valid",
            "template": "baseline_4_layer",
            "heat_layers": ["active_die_0"],
            "blockwise_k": True,
            "description": "4-layer validation sample with block-wise equivalent conductivity and a passive tim_equiv layer.",
        },
        {
            "sample_id": "sample_003",
            "split": "test_id",
            "template": "dual_active_4_layer",
            "heat_layers": ["active_die_0", "active_die_1"],
            "blockwise_k": True,
            "description": "Same-family 4-layer stack with two active heat-source layers.",
        },
        {
            "sample_id": "sample_004",
            "split": "test_ood_stack",
            "template": "heldout_interposer_4_layer",
            "heat_layers": ["active_die_0"],
            "blockwise_k": True,
            "description": "Held-out stack template with an interposer-equivalent layer.",
        },
        {
            "sample_id": "sample_005",
            "split": "valid",
            "template": "baseline_4_layer",
            "heat_layers": ["active_die_0"],
            "blockwise_k": True,
            "k_field_mode": "diag3_diagnostic",
            "diagnostic_only": True,
            "description": (
                "Diagonal anisotropic diagnostic metadata-only smoke sample. "
                "This sample exists only to validate real (N,3) k_field support "
                "through schema, validator, inspect, and loader paths."
            ),
        },
    ]


def _points_for_layers(layers: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    xs = np.linspace(0.0, FOOTPRINT_M[0], NX)
    ys = np.linspace(0.0, FOOTPRINT_M[1], NY)
    coords = []
    layer_ids = []
    z_ranges = []

    z0 = 0.0
    for layer in layers:
        z1 = z0 + float(layer["thickness_m"])
        z_ranges.append((z0, z1))
        # Two z planes per layer are enough for metadata and boundary checks.
        for z in (z0, z1):
            for y in ys:
                for x in xs:
                    coords.append((x, y, z))
                    layer_ids.append(layer["id"])
        z0 = z1

    return np.asarray(coords, dtype=np.float64), np.asarray(layer_ids, dtype=np.int64), z_ranges


def _region_and_material_fields(
    coords: np.ndarray,
    layer_ids: np.ndarray,
    layers: list[dict[str, Any]],
    blockwise_k: bool,
    k_field_mode: str = "iso1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    region_id = np.zeros(coords.shape[0], dtype=np.int64)
    material_id = np.zeros(coords.shape[0], dtype=np.int64)
    k_channels = 3 if k_field_mode == "diag3_diagnostic" else 1
    k_field = np.zeros((coords.shape[0], k_channels), dtype=np.float64)
    regions = []
    materials = []

    material_counter = 0
    region_counter = 0
    for layer in layers:
        layer_mask = layer_ids == layer["id"]
        base_k = float(layer["base_k"])

        left_mask = layer_mask & (coords[:, 0] <= FOOTPRINT_M[0] * 0.5)
        right_mask = layer_mask & ~left_mask
        blocks = [("left_block", left_mask, base_k)]
        if blockwise_k and "active_die" in layer["name"]:
            blocks.append(("right_hot_block", right_mask, base_k * 0.72))
        else:
            blocks[0] = ("full_layer", layer_mask, base_k)

        for block_name, mask, k_value in blocks:
            if not np.any(mask):
                continue
            region_id[mask] = region_counter
            material_id[mask] = material_counter
            if k_field_mode == "diag3_diagnostic":
                if layer["name"] == "substrate_equiv":
                    diag_k = [12.0, 11.0, 7.0]
                elif layer["name"] == "active_die_0" and block_name == "left_block":
                    diag_k = [130.0, 100.0, 75.0]
                elif layer["name"] == "active_die_0" and block_name == "right_hot_block":
                    diag_k = [118.0, 88.0, 62.0]
                elif layer["name"] == "tim_equiv":
                    diag_k = [5.0, 4.5, 2.0]
                elif layer["name"] == "heatsink_equiv":
                    diag_k = [180.0, 176.0, 210.0]
                else:
                    diag_k = [k_value, k_value, k_value]
                k_field[mask] = np.asarray(diag_k, dtype=np.float64)
                material_entry = {
                    "id": material_counter,
                    "name": f"{layer['name']}_{block_name}_material",
                    "model": "diagonal_anisotropic_equivalent",
                    "thermal_conductivity_diag_W_mK": diag_k,
                    "source_tag": "provisional_engineering_assumption",
                }
            else:
                k_field[mask, 0] = k_value
                material_entry = {
                    "id": material_counter,
                    "name": f"{layer['name']}_{block_name}_material",
                    "model": "isotropic_equivalent",
                    "thermal_conductivity_W_mK": k_value,
                    "source_tag": "provisional_engineering_assumption",
                }

            materials.append(material_entry)
            regions.append({
                "id": region_counter,
                "name": f"{layer['name']}_{block_name}",
                "layer_id": layer["id"],
                "material_id": material_counter,
                "selector": {
                    "type": "x_half_block" if blockwise_k and "active_die" in layer["name"] else "full_layer",
                    "x_range_m": (
                        [0.0, FOOTPRINT_M[0]]
                        if block_name == "full_layer"
                        else [0.0, FOOTPRINT_M[0] * 0.5]
                        if "left" in block_name
                        else [FOOTPRINT_M[0] * 0.5, FOOTPRINT_M[0]]
                    ),
                },
            })
            material_counter += 1
            region_counter += 1

    return region_id, material_id, k_field, regions, materials


def _q_field(
    coords: np.ndarray,
    layer_ids: np.ndarray,
    layers: list[dict[str, Any]],
    heat_layers: list[str],
) -> np.ndarray:
    q = np.zeros((coords.shape[0], 1), dtype=np.float64)
    heat_layer_ids = {layer["id"] for layer in layers if layer["name"] in heat_layers}
    center_x = FOOTPRINT_M[0] * 0.5
    center_y = FOOTPRINT_M[1] * 0.5
    radius = FOOTPRINT_M[0] * 0.28

    radial = (coords[:, 0] - center_x) ** 2 + (coords[:, 1] - center_y) ** 2
    source_mask = np.isin(layer_ids, list(heat_layer_ids)) & (radial <= radius ** 2)
    q[source_mask, 0] = 1.0e8
    return q


def _boundary_indices(coords: np.ndarray) -> dict[str, list[int]]:
    tol = 1.0e-12
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())

    top = np.flatnonzero(np.isclose(coords[:, 2], z_max, atol=tol)).tolist()
    bottom = np.flatnonzero(np.isclose(coords[:, 2], z_min, atol=tol)).tolist()
    sides = np.flatnonzero(
        np.isclose(coords[:, 0], x_min, atol=tol)
        | np.isclose(coords[:, 0], x_max, atol=tol)
        | np.isclose(coords[:, 1], y_min, atol=tol)
        | np.isclose(coords[:, 1], y_max, atol=tol)
    ).tolist()
    return {"top": top, "bottom": bottom, "sides": sides}


def _meta(
    config: dict[str, Any],
    layers: list[dict[str, Any]],
    z_ranges: list[tuple[float, float]],
    regions: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    coords: np.ndarray,
) -> dict[str, Any]:
    boundary_indices = _boundary_indices(coords)
    layer_meta = []
    for layer, z_range in zip(layers, z_ranges):
        layer_meta.append({
            "id": layer["id"],
            "name": layer["name"],
            "z_range_m": [z_range[0], z_range[1]],
            "thickness_m": layer["thickness_m"],
            "equivalent_model": "layerwise_or_blockwise_effective_conductivity",
            "source_tag": "provisional_engineering_assumption",
        })

    interfaces = []
    for lower, upper in zip(layer_meta[:-1], layer_meta[1:]):
        interfaces.append({
            "name": f"{lower['name']}__to__{upper['name']}",
            "type": "perfect_contact",
            "lower_layer_id": lower["id"],
            "upper_layer_id": upper["id"],
            "z_m": lower["z_range_m"][1],
            "normal": [0.0, 0.0, 1.0],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "subset_name": SUBSET_NAME,
        "sample_id": config["sample_id"],
        "stage": "metadata_only",
        "split": config["split"],
        "description": config["description"],
        "domain": {
            "geometry": "regular_layered_rectangular_stack",
            "coordinate_system": "cartesian",
            "footprint_m": list(FOOTPRINT_M),
            "height_m": float(coords[:, 2].max() - coords[:, 2].min()),
            "point_representation": "sampled_nodes",
            "stack_template": config["template"],
        },
        "layers": layer_meta,
        "regions": regions,
        "materials": materials,
        "boundary_regions": [
            {
                "name": "top",
                "surface": "z_max",
                "point_indices": boundary_indices["top"],
            },
            {
                "name": "bottom",
                "surface": "z_min",
                "point_indices": boundary_indices["bottom"],
            },
            {
                "name": "sides",
                "surface": "x_or_y_minmax",
                "point_indices": boundary_indices["sides"],
            },
        ],
        "boundary_types": {
            "top": "Robin",
            "bottom": "Dirichlet",
            "sides": "adiabatic",
        },
        "boundary_params": {
            "top": {
                "h_W_m2K": 2000.0,
                "ambient_temperature_K": 300.0,
                "source_tag": "provisional_engineering_assumption",
            },
            "bottom": {
                "fixed_temperature_K": 300.0,
                "source_tag": "provisional_engineering_assumption",
            },
            "sides": {
                "heat_flux_W_m2": 0.0,
                "source_tag": "provisional_engineering_assumption",
            },
        },
        "interfaces": interfaces,
        "generation_config": {
            "generator": "tools/generate_heat3d_v1_metadata_smoke.py",
            "solver": "not_run",
            "temperature_field": "not_generated_metadata_only",
            "q_field_unit": "volumetric_heat_generation_W_m3",
            "q_field_conversion": "none; provisional volumetric values written directly",
            "heat_layers": config["heat_layers"],
            "default_input_mode": "pure_physics",
            "optional_auxiliary_features": ["layer_id", "region_id", "material_id"],
            "k_field_shape": "(N,3)" if config.get("k_field_mode") == "diag3_diagnostic" else "(N,1)",
            "reserved_future_k_shapes": ["(N,3)", "(N,6)"],
            "reserved_ood_bc": "held_out_top_robin_htc_range",
            "diagnostic_only": bool(config.get("diagnostic_only", False)),
        },
        "units": {
            "coords": "m",
            "k_field": "W/(m*K)",
            "q_field": "W/m^3",
            "temperature": "K",
            "thickness": "m",
            "htc": "W/(m^2*K)",
        },
        "validation": {
            "expected_stage": "metadata_only",
            "temperature_required": False,
            "required_arrays": [
                "coords.npy",
                "layer_id.npy",
                "region_id.npy",
                "material_id.npy",
                "k_field.npy",
                "q_field.npy",
                "sample_meta.json",
            ],
            "solver_stage_requires": ["temperature.npy"],
        },
        "parameter_sources": {
            "literature_backed": [
                "top_robin_bottom_dirichlet_side_adiabatic_bc_pattern",
                "perfect_contact_interface_assumption",
                "equivalent_layer_abstraction_for_fine_structures",
            ],
            "provisional_engineering_assumption": [
                "footprint_m",
                "layer_thickness_m",
                "thermal_conductivity_W_mK",
                "q_field_values_W_m3",
                "top_h_W_m2K",
                "ambient_temperature_K",
                "bottom_fixed_temperature_K",
            ],
            "requires_user_confirmation": [
                "final_material_property_ranges",
                "final_power_density_or_volumetric_heat_generation_ranges",
                "final_robin_htc_ranges",
                "final_layer_stack_templates",
            ],
        },
    }


def write_sample(output_dir: Path, config: dict[str, Any]) -> Path:
    layers = _stack_templates()[config["template"]]
    coords, layer_ids, z_ranges = _points_for_layers(layers)
    region_id, material_id, k_field, regions, materials = _region_and_material_fields(
        coords, layer_ids, layers, config["blockwise_k"], config.get("k_field_mode", "iso1")
    )
    q_field = _q_field(coords, layer_ids, layers, config["heat_layers"])
    meta = _meta(config, layers, z_ranges, regions, materials, coords)

    sample_dir = output_dir / config["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "layer_id.npy", layer_ids)
    np.save(sample_dir / "region_id.npy", region_id)
    np.save(sample_dir / "material_id.npy", material_id)
    np.save(sample_dir / "k_field.npy", k_field)
    np.save(sample_dir / "q_field.npy", q_field)
    with (sample_dir / "sample_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    return sample_dir


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = _sample_configs()
    if args.sample_ids is not None:
        wanted = set(args.sample_ids)
        configs = [config for config in configs if config["sample_id"] in wanted]
        missing = sorted(wanted - {config["sample_id"] for config in configs})
        if missing:
            raise ValueError(f"Unknown sample ids requested: {missing}")

    written = []
    for config in configs:
        written.append(write_sample(args.output_dir, config))

    print(f"Wrote {len(written)} metadata-only sample(s) to {args.output_dir}")
    for sample_dir in written:
        print(f"  {sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
