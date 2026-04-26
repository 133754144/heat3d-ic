"""Generate Heat3D v1 metadata-only smoke samples.

This tool writes small local smoke samples under the ignored data/ directory.
It does not run a solver and intentionally does not create temperature.npy.
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


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_schema import SCHEMA_VERSION, SUBSET_NAME, default_v1_samples_dir
from rigno.heat3d_v1_manifest_resolver import load_manifest, resolve_manifest


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional supervised-small manifest for dry-run or explicit "
            "metadata-only writing."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve a manifest generation plan without writing data.",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write manifest-driven metadata-only samples. Must be used with --manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing manifest target subset.",
    )
    return parser.parse_args()


def _manifest_output_subset_dir(dataset_name: str) -> Path:
    return REPO_DIR / "data" / "heat3d-thermal-simulation" / "subsets" / dataset_name


def _print_manifest_dry_run(manifest_path: Path) -> int:
    manifest = load_manifest(manifest_path)
    dataset_name = str(manifest.get("dataset_name"))
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")

    resolved = resolve_manifest(manifest)
    resolved_samples = resolved["resolved_samples"]
    errors = resolved["errors"]
    sample_lookup = {
        sample.get("sample_id"): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    split_counts = dict(Counter(sample.get("split") for sample in samples))
    target_subset_dir = _manifest_output_subset_dir(dataset_name)
    protected_subset_collision = dataset_name in {
        "v1_multilayer_bc_eq_demo",
        "v1_multilayer_bc_eq_supervised_smoke",
    }
    under_ignored_data_dir_expected = "data" in target_subset_dir.relative_to(REPO_DIR).parts

    print("Heat3D v1 metadata manifest dry-run")
    print(f"manifest_path: {manifest_path.resolve()}")
    print(f"dataset_name: {dataset_name}")
    print(f"sample_count: {len(samples)}")
    print(f"resolved_sample_count: {len(resolved_samples)}")
    print(f"split_counts: {split_counts}")
    print(f"target_output_subset_path: {target_subset_dir}")
    print(f"no_data_written: True")
    print(f"no_temperature_written: True")
    print(f"protected_subset_collision: {protected_subset_collision}")
    print(f"output_under_ignored_data_dir_expected: {under_ignored_data_dir_expected}")
    print()

    if errors:
        print("resolver_errors")
        for error in errors:
            print(f"  {error}")
        print("dry_run_pass: False")
        return 1

    print("resolved generation plan")
    for resolved_sample in resolved_samples:
        sample_id = resolved_sample["sample_id"]
        original = sample_lookup.get(sample_id, {})
        stack = resolved_sample["stack_template"]
        source = resolved_sample["heat_source_pattern"]
        q_scale = resolved_sample["q_scale"]
        top_h = resolved_sample["top_h"]
        bc = resolved_sample["bc_baseline"]

        print(f"- sample_id: {sample_id}")
        print(f"  split: {resolved_sample['split']}")
        print(
            "  stack_template: "
            f"{stack['template_name']} / resolved_stack_variant={stack['variant']}"
        )
        print(f"  k_field_shape: {resolved_sample['k_field_shape']}")
        print(f"  anisotropy_type: {resolved_sample['anisotropy_type']}")
        print(
            "  heat_source_pattern: "
            f"{source['pattern_name']} / source_blocks={source['source_blocks']}"
        )
        print(
            "  q_scale: "
            f"{q_scale['category']} / multiplier={q_scale['multiplier_to_current_smoke_nominal']} "
            f"/ resolved_q_value_W_m3={q_scale['resolved_value_W_m3']} "
            f"/ source={q_scale['resolved_value_source']}"
        )
        print(
            "  bc_baseline: "
            f"{bc['category']} / bottom_T_fixed={bc['bottom_fixed_temperature_K']} "
            f"/ top_T_inf={bc['top_ambient_temperature_K']}"
        )
        print(
            "  top_h: "
            f"{top_h['category']} / multiplier={top_h['multiplier_to_current_smoke_nominal']} "
            f"/ resolved_h_value_W_m2K={top_h['resolved_value_W_m2K']} "
            f"/ source={top_h['resolved_value_source']}"
        )
        print(f"  expected_purpose: {original.get('expected_purpose')}")
        print(f"  parameter_status: {resolved_sample['parameter_status']}")
        print(f"  ood_role: {resolved_sample['ood_role']}")
    print()

    safe_no_write_plan = under_ignored_data_dir_expected and not protected_subset_collision
    print(f"dry_run_pass: {safe_no_write_plan}")
    return 0 if safe_no_write_plan else 1


def _manifest_samples_dir(dataset_name: str) -> Path:
    return _manifest_output_subset_dir(dataset_name) / "samples"


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
        "interposer_like_4_layer": [
            {"id": 0, "name": "substrate_equiv", "thickness_m": 0.0010, "base_k": 12.0},
            {"id": 1, "name": "interposer_equiv", "thickness_m": 0.00030, "base_k": 40.0},
            {"id": 2, "name": "active_die_0", "thickness_m": 0.00020, "base_k": 112.0},
            {"id": 3, "name": "heatsink_equiv", "thickness_m": 0.0015, "base_k": 172.0},
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


def _q_field_from_source_blocks(
    coords: np.ndarray,
    layer_ids: np.ndarray,
    layers: list[dict[str, Any]],
    source_blocks: list[dict[str, Any]],
    q_value: float,
) -> tuple[np.ndarray, list[str]]:
    q = np.zeros((coords.shape[0], 1), dtype=np.float64)
    layer_name_to_id = {layer["name"]: layer["id"] for layer in layers}
    heat_layers = []

    for block in source_blocks:
        layer_name = block["layer"]
        if layer_name not in layer_name_to_id:
            raise ValueError(f"source block references missing layer {layer_name!r}")
        heat_layers.append(layer_name)
        center_x, center_y = block["center_xy_fraction"]
        size_x, size_y = block["size_xy_fraction"]
        x0 = max(0.0, (center_x - size_x * 0.5) * FOOTPRINT_M[0])
        x1 = min(FOOTPRINT_M[0], (center_x + size_x * 0.5) * FOOTPRINT_M[0])
        y0 = max(0.0, (center_y - size_y * 0.5) * FOOTPRINT_M[1])
        y1 = min(FOOTPRINT_M[1], (center_y + size_y * 0.5) * FOOTPRINT_M[1])

        source_mask = (
            (layer_ids == layer_name_to_id[layer_name])
            & (coords[:, 0] >= x0)
            & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] <= y1)
        )
        if not np.any(source_mask):
            target_layer_mask = layer_ids == layer_name_to_id[layer_name]
            target_indices = np.flatnonzero(target_layer_mask)
            if target_indices.size == 0:
                raise ValueError(f"source block layer {layer_name!r} has no points")
            center_xy = np.asarray([center_x * FOOTPRINT_M[0], center_y * FOOTPRINT_M[1]])
            distances = np.sum((coords[target_indices, :2] - center_xy) ** 2, axis=1)
            source_mask[target_indices[np.argmin(distances)]] = True
        q[source_mask, 0] = np.maximum(q[source_mask, 0], q_value)

    return q, sorted(set(heat_layers))


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
        "subset_name": config.get("subset_name", SUBSET_NAME),
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
            "resolved_stack_variant": config.get("resolved_stack_variant"),
            "stack_role": config.get("stack_role"),
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
                "h_W_m2K": float(config.get("top_h_W_m2K", 2000.0)),
                "ambient_temperature_K": float(config.get("top_ambient_temperature_K", 300.0)),
                "source_tag": "provisional_engineering_assumption",
            },
            "bottom": {
                "fixed_temperature_K": float(config.get("bottom_fixed_temperature_K", 300.0)),
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
            "source_manifest": config.get("source_manifest"),
            "manifest_version": config.get("manifest_version"),
            "scaffold_base_commit": config.get("scaffold_base_commit"),
            "manifest_dataset_name": config.get("manifest_dataset_name"),
            "default_input_mode": "pure_physics",
            "optional_auxiliary_features": ["layer_id", "region_id", "material_id"],
            "k_field_shape": config.get(
                "k_field_shape",
                "(N,3)" if config.get("k_field_mode") == "diag3_diagnostic" else "(N,1)",
            ),
            "anisotropy_type": config.get("anisotropy_type"),
            "heat_source_pattern": config.get("heat_source_pattern"),
            "resolved_source_blocks": config.get("resolved_source_blocks"),
            "q_scale_category": config.get("q_scale_category"),
            "q_multiplier": config.get("q_multiplier"),
            "resolved_q_value_W_m3": config.get("resolved_q_value_W_m3"),
            "resolved_q_value_source": config.get("resolved_q_value_source"),
            "bc_baseline_category": config.get("bc_baseline_category"),
            "top_h_category": config.get("top_h_category"),
            "top_h_multiplier": config.get("top_h_multiplier"),
            "resolved_h_value_W_m2K": config.get("resolved_h_value_W_m2K"),
            "resolved_h_value_source": config.get("resolved_h_value_source"),
            "parameter_status": config.get("parameter_status"),
            "ood_role": config.get("ood_role"),
            "expected_purpose": config.get("expected_purpose"),
            "non_claims": config.get("non_claims"),
            "sample_scope": config.get("sample_scope"),
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


def _manifest_config(
    manifest_path: Path,
    manifest: dict[str, Any],
    resolved_sample: dict[str, Any],
    original_sample: dict[str, Any],
) -> dict[str, Any]:
    stack = resolved_sample["stack_template"]
    source = resolved_sample["heat_source_pattern"]
    q_scale = resolved_sample["q_scale"]
    top_h = resolved_sample["top_h"]
    bc = resolved_sample["bc_baseline"]
    k_field_shape = resolved_sample["k_field_shape"]
    anisotropy_type = str(resolved_sample["anisotropy_type"])
    return {
        "sample_id": resolved_sample["sample_id"],
        "split": resolved_sample["split"],
        "template": stack["template_name"],
        "heat_layers": sorted({block["layer"] for block in source["source_blocks"]}),
        "blockwise_k": "blockwise" in anisotropy_type or k_field_shape == "(N,3)",
        "k_field_mode": "diag3_diagnostic" if k_field_shape == "(N,3)" else "iso1",
        "diagnostic_only": k_field_shape == "(N,3)",
        "description": (
            f"{original_sample.get('expected_purpose', 'manifest-driven metadata-only sample')}. "
            "Manifest-driven metadata-only smoke sample; no temperature.npy is generated."
        ),
        "subset_name": manifest["dataset_name"],
        "source_manifest": str(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "scaffold_base_commit": manifest.get("scaffold_base_commit"),
        "manifest_dataset_name": manifest.get("dataset_name"),
        "resolved_stack_variant": stack["variant"],
        "stack_role": stack["role"],
        "k_field_shape": k_field_shape,
        "anisotropy_type": anisotropy_type,
        "heat_source_pattern": source["pattern_name"],
        "resolved_source_blocks": source["source_blocks"],
        "q_scale_category": q_scale["category"],
        "q_multiplier": q_scale["multiplier_to_current_smoke_nominal"],
        "resolved_q_value_W_m3": q_scale["resolved_value_W_m3"],
        "resolved_q_value_source": q_scale["resolved_value_source"],
        "bc_baseline_category": bc["category"],
        "bottom_fixed_temperature_K": bc["bottom_fixed_temperature_K"],
        "top_ambient_temperature_K": bc["top_ambient_temperature_K"],
        "top_h_category": top_h["category"],
        "top_h_multiplier": top_h["multiplier_to_current_smoke_nominal"],
        "top_h_W_m2K": top_h["resolved_value_W_m2K"],
        "resolved_h_value_W_m2K": top_h["resolved_value_W_m2K"],
        "resolved_h_value_source": top_h["resolved_value_source"],
        "parameter_status": resolved_sample["parameter_status"],
        "ood_role": resolved_sample["ood_role"],
        "expected_purpose": original_sample.get("expected_purpose"),
        "non_claims": manifest.get("non_claims", []),
        "sample_scope": "small_supervised_metadata_only_smoke",
    }


def write_manifest_sample(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    resolved_sample: dict[str, Any],
    original_sample: dict[str, Any],
) -> Path:
    config = _manifest_config(manifest_path, manifest, resolved_sample, original_sample)
    layers = _stack_templates()[config["template"]]
    coords, layer_ids, z_ranges = _points_for_layers(layers)
    region_id, material_id, k_field, regions, materials = _region_and_material_fields(
        coords,
        layer_ids,
        layers,
        config["blockwise_k"],
        config.get("k_field_mode", "iso1"),
    )
    q_field, heat_layers = _q_field_from_source_blocks(
        coords,
        layer_ids,
        layers,
        config["resolved_source_blocks"],
        float(config["resolved_q_value_W_m3"]),
    )
    config["heat_layers"] = heat_layers
    meta = _meta(config, layers, z_ranges, regions, materials, coords)

    sample_dir = output_dir / str(config["sample_id"])
    sample_dir.mkdir(parents=True, exist_ok=False)
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


def write_manifest_metadata(manifest_path: Path, overwrite: bool = False) -> list[Path]:
    manifest = load_manifest(manifest_path)
    dataset_name = str(manifest.get("dataset_name"))
    if dataset_name in {"v1_multilayer_bc_eq_demo", "v1_multilayer_bc_eq_supervised_smoke"}:
        raise ValueError(f"Refusing to write protected subset {dataset_name!r}")

    resolved = resolve_manifest(manifest)
    if resolved["errors"]:
        raise ValueError(f"Manifest resolver errors: {resolved['errors']}")

    subset_dir = _manifest_output_subset_dir(dataset_name)
    samples_dir = subset_dir / "samples"
    if subset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target subset already exists: {subset_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(subset_dir)
    samples_dir.mkdir(parents=True, exist_ok=False)

    sample_lookup = {
        sample.get("sample_id"): sample
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict)
    }
    written = []
    for resolved_sample in resolved["resolved_samples"]:
        sample_id = resolved_sample["sample_id"]
        written.append(
            write_manifest_sample(
                samples_dir,
                manifest_path,
                manifest,
                resolved_sample,
                sample_lookup.get(sample_id, {}),
            )
        )
    return written


def main() -> int:
    args = parse_args()

    if args.manifest is not None:
        if args.dry_run and args.write_metadata:
            raise ValueError("--dry-run and --write-metadata are mutually exclusive")
        if args.dry_run:
            return _print_manifest_dry_run(args.manifest)
        if not args.write_metadata:
            raise ValueError("--manifest requires either --dry-run or --write-metadata")
        written = write_manifest_metadata(args.manifest, overwrite=args.overwrite)
        print(f"Wrote {len(written)} manifest metadata-only sample(s)")
        for sample_dir in written:
            print(f"  {sample_dir}")
        print("temperature.npy written: False")
        return 0

    if args.dry_run:
        raise ValueError("--dry-run currently requires --manifest")
    if args.write_metadata:
        raise ValueError("--write-metadata requires --manifest")

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
