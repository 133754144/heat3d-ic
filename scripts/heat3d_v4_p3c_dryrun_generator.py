#!/usr/bin/env python3
"""Dry-run scene generator for the V4 P3c random-block contract.

This module is intentionally in-memory only. It reads the P3c parameter
registry, validates the executable contract, and creates dry scene manifests.
It does not write datasets, call solvers, export artifacts, or touch
data/output/checkpoints/logs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs/heat3d_v4/p3c_parameter_registry.json"
REQUIRED_TOP_SECTIONS = (
    "generation_policy",
    "source_refs",
    "parameters",
    "geometry",
    "deltaT_distribution",
    "cooling_regimes",
    "production_mix",
    "q_source_policy",
    "background_k_policy",
    "k_overlap_policy",
    "q_overlap_policy",
    "power_calibration_policy",
)
REQUIRED_PARAMETER_SECTIONS = ("k", "q", "BC", "contact")
REQUIRED_SOURCE_FIELDS = ("id", "title", "authors", "year", "venue", "url_or_doi", "notes")
REQUIRED_Q_FIELDS = (
    "source_volume_fraction",
    "integrated_power_target",
    "DeltaT_target_bin",
)
REQUIRED_DELTAT_AUDIT_FIELDS = (
    "deltaT_peak_K",
    "deltaT_p95_K",
    "deltaT_bin",
    "q_rescale_factor",
    "reject_reason",
)
FINAL_PROBE_ROLE = "reference_diagnostic_only_not_pass_fail"
PRODUCTION_CONTACT_MODEL = "R_contact=0_perfect_contact"
PENDING_DELTAT_BIN = "pending_until_solve"
SMOKE16_SAMPLE_COUNT = 16
SMOKE16_SEED = 4301
SMOKE16_DATASET_DIR = "data/heat3d_v4_p3c_smoke16_v3"
SMOKE16_OUTPUT_DIR = "output/heat3d_v4_p3c_smoke16_v3"
SEMANTIC_DOMAIN = (16.0, 16.0, 4.0)
Q_SOURCE_POLICY = "semantic_boundary_inset_5pct_solver_safe_deposition"
Q_POWER_INTEGRATION_POLICY = "solver_control_volume_weighted"
SEMANTIC_BOUNDARY_INSET_FRACTION = 0.05
SOLVER_SAFE_DEPOSITION_MASK = "exclude_bottom_top_side_boundary_nodes"
DEFAULT_MATERIAL_CLAIM_THRESHOLD = 0.2
PLANNED_SAMPLE_FILES = (
    "coords.npy",
    "layer_id.npy",
    "region_id.npy",
    "material_id.npy",
    "k_field.npy",
    "q_field.npy",
    "bc_features.npy",
    "sample_meta.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    validate_registry(registry)
    return registry


def _by_name(entries: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for entry in entries:
        if entry.get("name") == name:
            return entry
    raise ValueError(f"missing registry entry: {name}")


def _semantic_inset_domain() -> dict[str, list[float]]:
    return {
        "x": [
            SEMANTIC_DOMAIN[0] * SEMANTIC_BOUNDARY_INSET_FRACTION,
            SEMANTIC_DOMAIN[0] * (1.0 - SEMANTIC_BOUNDARY_INSET_FRACTION),
        ],
        "y": [
            SEMANTIC_DOMAIN[1] * SEMANTIC_BOUNDARY_INSET_FRACTION,
            SEMANTIC_DOMAIN[1] * (1.0 - SEMANTIC_BOUNDARY_INSET_FRACTION),
        ],
        "z": [
            SEMANTIC_DOMAIN[2] * SEMANTIC_BOUNDARY_INSET_FRACTION,
            SEMANTIC_DOMAIN[2] * (1.0 - SEMANTIC_BOUNDARY_INSET_FRACTION),
        ],
    }


def validate_registry(registry: dict[str, Any]) -> None:
    for section in REQUIRED_TOP_SECTIONS:
        _require(section in registry, f"missing top-level section: {section}")

    parameters = registry["parameters"]
    _require(isinstance(parameters, dict), "parameters must be an object")
    for section in REQUIRED_PARAMETER_SECTIONS:
        _require(section in parameters, f"missing parameters.{section}")
        _require(isinstance(parameters[section], list), f"parameters.{section} must be a list")
        _require(parameters[section], f"parameters.{section} must not be empty")

    source_refs = registry["source_refs"]
    _require(isinstance(source_refs, list) and source_refs, "source_refs must be a non-empty list")
    source_ids = set()
    for source in source_refs:
        for field in REQUIRED_SOURCE_FIELDS:
            _require(field in source, f"source_ref missing field {field}: {source}")
        source_ids.add(source["id"])
    _require("SRC-BSPDN-2025" in source_ids, "BSPDN 2025 anchor is required")
    _require("SRC-3DICE4-2025" in source_ids, "3D-ICE 4.0 anchor is required")
    _require("SRC-HBM-MEAS-2023" in source_ids, "HBM measurement anchor is required")

    policy = registry["generation_policy"]
    _require(policy.get("stress_split") == "disabled", "stress split must be disabled")
    _require(policy.get("splits") == ["train", "test"], "P3c must use train/test splits only")
    _require(policy.get("final_probe_role") == FINAL_PROBE_ROLE, "final_probe must be reference only")
    _require(
        policy.get("production_contact_model") == PRODUCTION_CONTACT_MODEL,
        "production contact model must be R_contact=0_perfect_contact",
    )
    q_source_policy = registry["q_source_policy"]
    _require(q_source_policy.get("name") == Q_SOURCE_POLICY, "bad q_source_policy")
    _require(
        list(q_source_policy.get("semantic_domain_xyz", [])) == list(SEMANTIC_DOMAIN),
        "semantic domain must be [16,16,4]",
    )
    _require(
        float(q_source_policy.get("semantic_boundary_inset_fraction"))
        == SEMANTIC_BOUNDARY_INSET_FRACTION,
        "bad semantic boundary inset fraction",
    )
    _require(
        q_source_policy.get("solver_safe_deposition_mask") == SOLVER_SAFE_DEPOSITION_MASK,
        "bad solver-safe deposition mask",
    )
    expected_inset = _semantic_inset_domain()
    _require(
        q_source_policy.get("semantic_inset_domain_xyz") == expected_inset,
        "bad semantic inset domain",
    )
    for field in (
        "q_total_target_power_W",
        "q_integral_from_array_W",
        "q_total_power_error_W",
        "q_power_on_bottom_W",
        "q_power_on_top_W",
        "q_power_on_xmin_W",
        "q_power_on_xmax_W",
        "q_power_on_ymin_W",
        "q_power_on_ymax_W",
        "q_power_on_side_W",
        "q_power_on_boundary_W",
        "q_power_on_bottom_fraction",
        "q_power_on_top_fraction",
        "q_power_on_side_fraction",
        "q_source_boundary_violation_count",
        "q_source_side_boundary_violation_count",
        "q_active_z_min",
        "q_active_z_max",
        "q_power_integration_policy",
        "semantic_boundary_inset_fraction",
        "semantic_inset_domain_xyz",
        "solver_safe_deposition_mask",
        "q_deposited_on_boundary_node_count",
    ):
        _require(
            field in q_source_policy.get("required_audit_fields", []),
            f"q_source_policy missing audit field: {field}",
        )

    for entry in parameters["k"]:
        for field in ("literature_anchor", "sampling_envelope", "rationale"):
            _require(field in entry, f"k entry {entry.get('name')} missing {field}")
        _require(entry.get("source_ref"), f"k entry {entry.get('name')} missing source_ref")

    for entry in parameters["q"]:
        for field in REQUIRED_Q_FIELDS:
            _require(field in entry, f"q entry {entry.get('name')} missing {field}")
        for field in ("range", "default", "source_ref", "source_type", "rationale"):
            _require(field in entry, f"q entry {entry.get('name')} missing {field}")
        _require(entry["DeltaT_target_bin"], f"q entry {entry.get('name')} missing DeltaT target")

    contact_entries = parameters["contact"]
    production_contact = _by_name(contact_entries, "production_contact_resistance")
    _require(production_contact.get("default") == 0.0, "production contact default must be 0")
    _require(production_contact.get("used_in_v4_production") is True, "R=0 contact must be production")
    finite_contact = _by_name(contact_entries, "finite_contact_resistance_deferred")
    _require(finite_contact.get("used_in_v4_production") is False, "finite contact must be deferred")

    delta = registry["deltaT_distribution"]
    for field in REQUIRED_DELTAT_AUDIT_FIELDS:
        _require(field in delta.get("audit_fields", []), f"missing DeltaT audit field: {field}")
    bin_names = {entry.get("name") for entry in delta.get("bins", [])}
    for name in ("reject_low", "low", "nominal", "hard", "reject_high"):
        _require(name in bin_names, f"missing DeltaT bin: {name}")

    cooling_names = {entry.get("name") for entry in registry["cooling_regimes"]}
    for name in ("weak_effective_air", "nominal_package", "strong_forced_or_effective_heatsink"):
        _require(name in cooling_names, f"missing cooling regime: {name}")

    production_mix = registry["production_mix"]
    diag3_target = _by_name(production_mix, "diag3_target_fraction")
    _require(float(diag3_target.get("default")) == 0.2, "diag3 target fraction must be 0.20")

    background_policy = registry["background_k_policy"]
    _require(
        background_policy.get("default_family") == "effective_stack_medium_k",
        "default background family must be effective_stack_medium_k",
    )
    allowed_backgrounds = set(background_policy.get("allowed_families", []))
    _require(
        allowed_backgrounds == {"effective_stack_medium_k", "silicon_like", "hbm_like_anisotropic_k"},
        "background allowed_families mismatch",
    )
    _require(
        background_policy.get("low_k_dielectric_underfill_policy")
        == "minority_background_or_block_only_not_default_background",
        "low-k background policy mismatch",
    )
    for family in background_policy.get("families", []):
        for field in ("source_ref", "source_type", "rationale", "metadata_tag"):
            _require(field in family, f"background family missing {field}: {family.get('name')}")
    _require(
        "non_default_low_k_reference" in background_policy,
        "background policy must document low-k non-default reference",
    )

    k_overlap_policy = registry["k_overlap_policy"]
    _require(k_overlap_policy.get("name") == "deterministic_priority_override", "bad k overlap policy")
    _require(
        k_overlap_policy.get("projection") == "continuous_semantic_bbox_overlap",
        "k overlap projection must be continuous semantic bbox overlap",
    )
    _require(
        0.0 < float(k_overlap_policy.get("material_claim_threshold", 0.0)) <= 1.0,
        "k material_claim_threshold must be within (0,1]",
    )
    _require(
        k_overlap_policy.get("forbidden_default_merge") == "arithmetic_mean",
        "k arithmetic mean must be forbidden as generator default",
    )
    q_overlap_policy = registry["q_overlap_policy"]
    _require(q_overlap_policy.get("name") == "sum_volumetric_sources", "bad q overlap policy")
    _require(q_overlap_policy.get("cell_merge") == "sum", "q overlap must sum per cell")
    _require(
        q_overlap_policy.get("projection") == "continuous_semantic_bbox_overlap_fraction",
        "q overlap projection must use continuous semantic bbox overlap fractions",
    )
    _require(q_overlap_policy.get("q_source_policy") == Q_SOURCE_POLICY, "bad q source policy link")
    _require(
        q_overlap_policy.get("forbidden_default_merge") == "max_pooling",
        "q max pooling must be forbidden as generator merge",
    )
    power_policy = registry["power_calibration_policy"]
    _require(
        power_policy.get("name") == "calibrate_q_density_from_realized_volume_and_integrated_power_target",
        "bad power calibration policy",
    )
    _require(
        power_policy.get("integration") == Q_POWER_INTEGRATION_POLICY,
        "power calibration must use solver control-volume weights",
    )
    for field in (
        "target_power_W",
        "realized_volume_m3",
        "calibrated_q_density_W_m3",
        "realized_power_W",
        "power_error_W",
        "power_integration_policy",
    ):
        _require(
            field in power_policy.get("required_metadata_fields", []),
            f"missing power calibration metadata field: {field}",
        )


def _rng_uniform(rng: random.Random, bounds: dict[str, Any], *, log_space: bool = False) -> float:
    lo = float(bounds["min"])
    hi = float(bounds["max"])
    if log_space:
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


def _rng_int(rng: random.Random, bounds: dict[str, Any]) -> int:
    return rng.randint(int(bounds["min"]), int(bounds["max"]))


def _default_geometry(registry: dict[str, Any]) -> dict[str, Any]:
    geometry_entries = registry["geometry"]
    return {
        "domain_xy_mm": float(_by_name(geometry_entries, "domain_xy_mm")["default"]),
        "domain_z_mm": float(_by_name(geometry_entries, "domain_z_mm")["default"]),
        "grid_shape": list(_by_name(geometry_entries, "grid_shape_candidates")["default"]),
    }


def _node_index(i: int, j: int, k: int, grid_shape: list[int]) -> int:
    _, ny, nz = [int(v) for v in grid_shape]
    return (i * ny + j) * nz + k


def _block_node_indices(block: dict[str, Any], grid_shape: list[int]) -> list[int]:
    start_i, start_j, start_k = [int(v) for v in block["start_ijk"]]
    extent_i, extent_j, extent_k = [int(v) for v in block["extent_ijk"]]
    indices: list[int] = []
    for i in range(start_i, start_i + extent_i):
        for j in range(start_j, start_j + extent_j):
            for k in range(start_k, start_k + extent_k):
                indices.append(_node_index(i, j, k, grid_shape))
    return indices


def _node_ijk(index: int, grid_shape: list[int]) -> tuple[int, int, int]:
    _, ny, nz = [int(v) for v in grid_shape]
    i = int(index) // (ny * nz)
    remainder = int(index) % (ny * nz)
    j = remainder // nz
    k = remainder % nz
    return i, j, k


def _boundary_node_masks(grid_shape: list[int]) -> dict[str, np.ndarray]:
    nx, ny, nz = [int(v) for v in grid_shape]
    node_count = nx * ny * nz
    masks = {
        "bottom": np.zeros((node_count,), dtype=bool),
        "top": np.zeros((node_count,), dtype=bool),
        "xmin": np.zeros((node_count,), dtype=bool),
        "xmax": np.zeros((node_count,), dtype=bool),
        "ymin": np.zeros((node_count,), dtype=bool),
        "ymax": np.zeros((node_count,), dtype=bool),
    }
    for idx in range(node_count):
        i, j, k = _node_ijk(idx, grid_shape)
        masks["bottom"][idx] = k == 0
        masks["top"][idx] = k == nz - 1
        masks["xmin"][idx] = i == 0
        masks["xmax"][idx] = i == nx - 1
        masks["ymin"][idx] = j == 0
        masks["ymax"][idx] = j == ny - 1
    masks["side"] = masks["xmin"] | masks["xmax"] | masks["ymin"] | masks["ymax"]
    masks["boundary"] = masks["bottom"] | masks["top"] | masks["side"]
    masks["solver_safe"] = ~masks["boundary"]
    return masks


def _control_volume_overlap_fraction(
    bbox: dict[str, float],
    *,
    i: int,
    j: int,
    k: int,
) -> float:
    x_overlap = max(0.0, min(float(bbox["x_max"]), i + 1.0) - max(float(bbox["x_min"]), float(i)))
    y_overlap = max(0.0, min(float(bbox["y_max"]), j + 1.0) - max(float(bbox["y_min"]), float(j)))
    z_overlap = max(0.0, min(float(bbox["z_max"]), k + 1.0) - max(float(bbox["z_min"]), float(k)))
    return float(x_overlap * y_overlap * z_overlap)


def _block_overlap_fractions(block: dict[str, Any], grid_shape: list[int]) -> np.ndarray:
    bbox = block["continuous_bbox"]
    nx, ny, nz = [int(v) for v in grid_shape]
    overlaps = np.zeros((nx * ny * nz,), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                overlaps[_node_index(i, j, k, grid_shape)] = _control_volume_overlap_fraction(
                    bbox,
                    i=i,
                    j=j,
                    k=k,
                )
    return overlaps


def _grid_axes(domain: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = [int(v) for v in domain["grid_shape"]]
    x_max = float(domain["domain_xy_mm"]) * 1.0e-3
    y_max = float(domain["domain_xy_mm"]) * 1.0e-3
    z_max = float(domain["domain_z_mm"]) * 1.0e-3
    xs = np.linspace(0.0, x_max, nx, dtype=np.float64)
    ys = np.linspace(0.0, y_max, ny, dtype=np.float64)
    zs = np.linspace(0.0, z_max, nz, dtype=np.float64)
    return xs, ys, zs


def _grid_coords(domain: dict[str, Any]) -> np.ndarray:
    xs, ys, zs = _grid_axes(domain)
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def _control_widths(axis: np.ndarray) -> np.ndarray:
    widths = np.zeros_like(axis, dtype=np.float64)
    if axis.size == 1:
        widths[0] = 1.0
        return widths
    widths[0] = 0.5 * (axis[1] - axis[0])
    widths[-1] = 0.5 * (axis[-1] - axis[-2])
    if axis.size > 2:
        widths[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return widths


def control_volume_weights_for_domain(domain: dict[str, Any]) -> np.ndarray:
    """Return solver-equivalent control-volume weights in generator node order."""

    grid_shape = [int(v) for v in domain["grid_shape"]]
    nx, ny, nz = grid_shape
    xs, ys, zs = _grid_axes(domain)
    dx = _control_widths(xs)
    dy = _control_widths(ys)
    dz = _control_widths(zs)
    weights = np.zeros((nx * ny * nz,), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = _node_index(i, j, k, grid_shape)
                weights[idx] = float(dx[i] * dy[j] * dz[k])
    return weights


def _k_entry(registry: dict[str, Any], name: str) -> dict[str, Any]:
    return _by_name(registry["parameters"]["k"], name)


def _background_k(registry: dict[str, Any], *, diag3: bool) -> tuple[str, np.ndarray, dict[str, Any]]:
    family = registry["background_k_policy"]["default_family"]
    entry = _k_entry(registry, family)
    value = float(entry["default"])
    if diag3:
        background_value = np.array([value, value, value], dtype=np.float64)
        serial_value: Any = {"kx": value, "ky": value, "kz": value}
    else:
        background_value = np.array([value], dtype=np.float64)
        serial_value = value
    return family, background_value, {
        "background_k_family": family,
        "background_k_value": serial_value,
        "background_k_metadata_tag": f"background_k_family={family}",
    }


def _block_k_value(block: dict[str, Any], *, diag3: bool) -> np.ndarray:
    value = block["k_value"]
    if "k" in value:
        scalar = float(value["k"])
        if diag3:
            return np.array([scalar, scalar, scalar], dtype=np.float64)
        return np.array([scalar], dtype=np.float64)
    diag_value = np.array(
        [float(value["kx"]), float(value["ky"]), float(value["kz"])],
        dtype=np.float64,
    )
    if diag3:
        return diag_value
    return np.array([float(np.mean(diag_value))], dtype=np.float64)


def _bc_features(domain: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    nx, ny, nz = [int(v) for v in domain["grid_shape"]]
    flags = np.zeros((nx * ny * nz, 4), dtype=np.float64)
    counts = {"top": 0, "bottom": 0, "side": 0, "interior": 0}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = _node_index(i, j, k, domain["grid_shape"])
                if k == nz - 1:
                    channel = 0
                    counts["top"] += 1
                elif k == 0:
                    channel = 1
                    counts["bottom"] += 1
                elif i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    channel = 2
                    counts["side"] += 1
                else:
                    channel = 3
                    counts["interior"] += 1
                flags[idx, channel] = 1.0
    return flags, counts


def _semantic_id_arrays(
    *,
    grid_shape: list[int],
    final_winner: list[str],
    k_blocks: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nx, ny, nz = [int(v) for v in grid_shape]
    node_count = nx * ny * nz
    material_keys = ["background"] + [block["block_id"] for block in k_blocks]
    material_id_by_key = {key: index for index, key in enumerate(material_keys)}
    layer_id = np.zeros((node_count,), dtype=np.int64)
    region_id = np.zeros((node_count,), dtype=np.int64)
    material_id = np.zeros((node_count,), dtype=np.int64)
    for idx, winner in enumerate(final_winner):
        _, _, k = _node_ijk(idx, grid_shape)
        material_index = material_id_by_key.get(winner, 0)
        layer_id[idx] = k
        region_id[idx] = material_index
        material_id[idx] = material_index

    layers = [
        {"id": int(k), "name": f"z_layer_{k}", "z_index": int(k)}
        for k in range(nz)
    ]
    regions = [
        {
            "id": int(material_id_by_key[key]),
            "name": key,
            "layer_id": 0,
            "material_id": int(material_id_by_key[key]),
        }
        for key in material_keys
    ]
    materials = [
        {
            "id": int(material_id_by_key[key]),
            "name": key,
            "source": "background_k" if key == "background" else "k_block",
        }
        for key in material_keys
    ]
    return layer_id, region_id, material_id, layers, regions, materials


def _boundary_regions(grid_shape: list[int], boundary_masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    top = np.nonzero(boundary_masks["top"])[0].astype(int).tolist()
    bottom = np.nonzero(boundary_masks["bottom"])[0].astype(int).tolist()
    side_only = boundary_masks["side"] & ~boundary_masks["top"] & ~boundary_masks["bottom"]
    sides = np.nonzero(side_only)[0].astype(int).tolist()
    return [
        {"name": "top", "point_indices": top},
        {"name": "bottom", "point_indices": bottom},
        {"name": "sides", "point_indices": sides},
    ]


def _solver_boundary_contract(scene: dict[str, Any]) -> dict[str, Any]:
    bc = scene["BC"]
    top_ambient = float(bc["top_ambient_temperature_K"])
    bottom_fixed = float(bc["bottom_dirichlet_temperature_K"])
    return {
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {
                "type": "robin",
                "h_W_m2K": float(bc["top_h_W_m2K"]),
                "T_inf_K": top_ambient,
                "ambient_temperature_K": top_ambient,
            },
            "bottom": {
                "type": "dirichlet",
                "T_fixed_K": bottom_fixed,
                "fixed_temperature_K": bottom_fixed,
            },
            "side": {
                "type": "adiabatic",
            },
        },
    }


def _solver_interface_contract(scene: dict[str, Any]) -> list[dict[str, Any]]:
    domain_z_m = float(scene["domain"]["domain_z_mm"]) * 1.0e-3
    return [
        {
            "id": "p3c_v0_perfect_contact",
            "type": "perfect_contact",
            "adjacent_layer_ids": [0, 1],
            "z_position_m": domain_z_m * 0.5,
            "R_contact_m2K_W": 0.0,
            "contact_model": PRODUCTION_CONTACT_MODEL,
        }
    ]


def _project_block(
    *,
    grid_shape: list[int],
    xy_fraction: float,
    z_fraction: float,
    rng: random.Random,
    placement_policy: str = "full_domain",
    material_claim_threshold: float | None = None,
) -> dict[str, Any]:
    nx, ny, nz = [int(v) for v in grid_shape]
    total_cells = nx * ny * nz
    if (float(nx), float(ny), float(nz)) != SEMANTIC_DOMAIN:
        raise ValueError(f"P3c semantic projection expects grid {SEMANTIC_DOMAIN}, got {grid_shape}")
    allowed = {
        "x": [0.0, float(nx)],
        "y": [0.0, float(ny)],
        "z": [0.0, float(nz)],
    }
    if placement_policy == Q_SOURCE_POLICY:
        allowed = _semantic_inset_domain()
    elif placement_policy != "full_domain":
        raise ValueError(f"unsupported semantic placement policy: {placement_policy}")

    x_span = allowed["x"][1] - allowed["x"][0]
    y_span = allowed["y"][1] - allowed["y"][0]
    z_span = allowed["z"][1] - allowed["z"][0]
    if x_span <= 0.0 or y_span <= 0.0 or z_span <= 0.0:
        raise ValueError(f"invalid semantic domain for policy {placement_policy}")
    if z_span <= 0.0:
        raise ValueError(f"invalid semantic z span for policy {placement_policy}")

    side_fraction = math.sqrt(max(float(xy_fraction), 0.0))
    requested_lengths = [
        min(x_span, max(1.0e-9, x_span * side_fraction)),
        min(y_span, max(1.0e-9, y_span * side_fraction)),
        min(z_span, max(1.0e-9, z_span * max(float(z_fraction), 0.0))),
    ]
    requested_dims_floor = [math.floor(length) for length in requested_lengths]
    starts = [
        rng.uniform(allowed["x"][0], allowed["x"][1] - requested_lengths[0]),
        rng.uniform(allowed["y"][0], allowed["y"][1] - requested_lengths[1]),
        rng.uniform(allowed["z"][0], allowed["z"][1] - requested_lengths[2]),
    ]
    bbox = {
        "x_min": starts[0],
        "x_max": starts[0] + requested_lengths[0],
        "y_min": starts[1],
        "y_max": starts[1] + requested_lengths[1],
        "z_min": starts[2],
        "z_max": starts[2] + requested_lengths[2],
    }
    overlaps = _block_overlap_fractions({"continuous_bbox": bbox}, grid_shape)
    if material_claim_threshold is None:
        active_mask = overlaps > 0.0
    else:
        active_mask = overlaps >= float(material_claim_threshold)
    realized_cell_count = int(np.count_nonzero(active_mask))
    if realized_cell_count <= 0:
        active_mask = overlaps > 0.0
        realized_cell_count = int(np.count_nonzero(active_mask))
    overlap_fraction_sum = float(np.sum(overlaps))
    adjusted = any(dim <= 0 for dim in requested_dims_floor)
    approx_start = [
        max(0, min(int(math.floor(bbox["x_min"])), nx - 1)),
        max(0, min(int(math.floor(bbox["y_min"])), ny - 1)),
        max(0, min(int(math.floor(bbox["z_min"])), nz - 1)),
    ]
    approx_extent = [
        max(1, min(nx - approx_start[0], int(math.ceil(bbox["x_max"])) - approx_start[0])),
        max(1, min(ny - approx_start[1], int(math.ceil(bbox["y_max"])) - approx_start[1])),
        max(1, min(nz - approx_start[2], int(math.ceil(bbox["z_max"])) - approx_start[2])),
    ]
    return {
        "requested_fraction": float(xy_fraction) * float(z_fraction),
        "requested_xy_fraction": float(xy_fraction),
        "requested_z_fraction": float(z_fraction),
        "requested_dims_floor": requested_dims_floor,
        "requested_lengths_semantic": requested_lengths,
        "continuous_bbox": bbox,
        "semantic_domain": {
            "x": [0.0, float(nx)],
            "y": [0.0, float(ny)],
            "z": [0.0, float(nz)],
        },
        "placement_policy": placement_policy,
        "semantic_boundary_inset_fraction": (
            SEMANTIC_BOUNDARY_INSET_FRACTION if placement_policy == Q_SOURCE_POLICY else 0.0
        ),
        "semantic_inset_domain_xyz": _semantic_inset_domain() if placement_policy == Q_SOURCE_POLICY else None,
        "start_ijk": approx_start,
        "extent_ijk": approx_extent,
        "realized_fraction": overlap_fraction_sum / total_cells,
        "realized_cell_count": realized_cell_count,
        "overlap_fraction_sum": overlap_fraction_sum,
        "max_overlap_fraction": float(np.max(overlaps)) if overlaps.size else 0.0,
        "projection_status": "resampled_min_one_cell" if adjusted else "realized",
        "reject_reason": "projected_zero_cells_resampled_to_one_cell" if adjusted else None,
    }


def project_block_preview(
    *,
    grid_shape: list[int],
    xy_fraction: float,
    z_fraction: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Project a dry-run block for checker-only boundary tests."""

    return _project_block(
        grid_shape=grid_shape,
        xy_fraction=xy_fraction,
        z_fraction=z_fraction,
        rng=random.Random(seed),
    )


def _sample_fraction(
    rng: random.Random,
    entry: dict[str, Any],
    *,
    fallback_min: float,
    fallback_max: float,
) -> float:
    bounds = entry.get("range")
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        return _rng_uniform(rng, bounds)
    return rng.uniform(fallback_min, fallback_max)


def _material_blocks(
    registry: dict[str, Any],
    *,
    grid_shape: list[int],
    diag3_mode: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    geometry = registry["geometry"]
    count = _rng_int(rng, _by_name(geometry, "material_block_count")["range"])
    xy_entry = _by_name(geometry, "material_block_xy_fraction")
    z_entry = _by_name(geometry, "material_block_z_fraction")
    k_entries = registry["parameters"]["k"]
    scalar_entries = [
        entry
        for entry in k_entries
        if entry["name"] not in {"hbm_like_anisotropic_k", "diag3_anisotropy_ratio"}
    ]
    diag3_ratio = _by_name(k_entries, "diag3_anisotropy_ratio")
    hbm_entry = _by_name(k_entries, "hbm_like_anisotropic_k")
    claim_threshold = float(
        registry["k_overlap_policy"].get("material_claim_threshold", DEFAULT_MATERIAL_CLAIM_THRESHOLD)
    )
    blocks: list[dict[str, Any]] = []
    for block_index in range(count):
        xy_fraction = _sample_fraction(rng, xy_entry, fallback_min=0.05, fallback_max=0.6)
        z_fraction = _sample_fraction(rng, z_entry, fallback_min=0.25, fallback_max=1.0)
        projection = _project_block(
            grid_shape=grid_shape,
            xy_fraction=xy_fraction,
            z_fraction=z_fraction,
            rng=rng,
            placement_policy="full_domain",
            material_claim_threshold=claim_threshold,
        )
        if diag3_mode == "hbm_like_strong" and block_index == 0:
            value = dict(hbm_entry["default"])
            k_family = hbm_entry["name"]
            metadata_tag = hbm_entry["metadata_tag"]
            hbm_like_strong = True
        else:
            scalar_entry = scalar_entries[(block_index + rng.randrange(len(scalar_entries))) % len(scalar_entries)]
            scalar_k = _rng_uniform(rng, scalar_entry["range"], log_space=True)
            k_family = scalar_entry["name"]
            metadata_tag = scalar_entry["metadata_tag"]
            hbm_like_strong = False
            if diag3_mode == "mild":
                ratio = _rng_uniform(rng, diag3_ratio["range"])
                value = {
                    "kx": scalar_k,
                    "ky": scalar_k * ratio,
                    "kz": scalar_k / ratio,
                }
                metadata_tag = f"{metadata_tag};k_mode=diag3"
            else:
                value = {"k": scalar_k}
        blocks.append(
            {
                "block_id": f"m{block_index:02d}",
                "k_family": k_family,
                "k_value": value,
                "diag3_mode": diag3_mode,
                "hbm_like_strong_anisotropy": hbm_like_strong,
                "metadata_tag": metadata_tag,
                **projection,
            }
        )
    return blocks


def _q_blocks(
    registry: dict[str, Any],
    *,
    grid_shape: list[int],
    q_entry: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    source_count = q_entry.get("sampling", {}).get("source_count")
    if isinstance(source_count, dict):
        count = _rng_int(rng, source_count)
    else:
        count = 1
    bounds = q_entry["source_volume_fraction"]
    q_blocks = []
    for block_index in range(count):
        volume_fraction = _rng_uniform(rng, bounds)
        z_fraction = min(1.0, max(0.25, math.sqrt(volume_fraction)))
        xy_fraction = min(0.95, max(0.0001, volume_fraction / z_fraction))
        projection = _project_block(
            grid_shape=grid_shape,
            xy_fraction=xy_fraction,
            z_fraction=z_fraction,
            rng=rng,
            placement_policy=Q_SOURCE_POLICY,
        )
        power_target = _rng_uniform(rng, q_entry["integrated_power_target"])
        q_density = _rng_uniform(rng, q_entry["range"], log_space=True)
        q_blocks.append(
            {
                "block_id": f"q{block_index:02d}",
                "q_family": q_entry["name"],
                "q_density_W_m3": q_density,
                "source_volume_fraction": volume_fraction,
                "integrated_power_target_W": power_target,
                "DeltaT_target_bin": q_entry["DeltaT_target_bin"],
                "metadata_tag": q_entry["metadata_tag"],
                **projection,
            }
        )
    return q_blocks


def _choose_diag3_modes(sample_count: int, target_fraction: float) -> list[str]:
    diag3_count = int(round(sample_count * target_fraction))
    strong_count = max(1, int(round(diag3_count * 0.2))) if diag3_count >= 3 else 0
    modes = ["scalar"] * sample_count
    for idx in range(diag3_count):
        modes[idx] = "hbm_like_strong" if idx < strong_count else "mild"
    return modes


def generate_dryrun_batch(
    registry: dict[str, Any],
    *,
    sample_count: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    validate_registry(registry)
    seed = int(seed if seed is not None else registry["generation_policy"]["random_split_seed"])
    rng = random.Random(seed)
    geometry = _default_geometry(registry)
    grid_shape = geometry["grid_shape"]
    node_count = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
    q_entries = registry["parameters"]["q"]
    cooling_regimes = registry["cooling_regimes"]
    diag3_target = float(_by_name(registry["production_mix"], "diag3_target_fraction")["default"])
    diag3_modes = _choose_diag3_modes(sample_count, diag3_target)

    scenes: list[dict[str, Any]] = []
    for scene_index in range(sample_count):
        mode = diag3_modes[scene_index]
        q_entry = q_entries[scene_index % len(q_entries)]
        cooling = cooling_regimes[scene_index % len(cooling_regimes)]
        top_h = _rng_uniform(rng, cooling["range"], log_space=True)
        material_blocks = _material_blocks(registry, grid_shape=grid_shape, diag3_mode=mode, rng=rng)
        q_blocks = _q_blocks(registry, grid_shape=grid_shape, q_entry=q_entry, rng=rng)
        q_cells = sum(block["realized_cell_count"] for block in q_blocks)
        scenes.append(
            {
                "scene_id": f"p3c_dry_{scene_index:04d}",
                "seed": seed,
                "sample_index": scene_index,
                "domain": {
                    "domain_xy_mm": geometry["domain_xy_mm"],
                    "domain_z_mm": geometry["domain_z_mm"],
                    "grid_shape": grid_shape,
                    "node_count": node_count,
                    "semantic_domain_xyz": list(SEMANTIC_DOMAIN),
                },
                "semantic_projection": {
                    "mode": "continuous_bbox_to_physical_control_volume_overlap",
                    "semantic_domain_xyz": list(SEMANTIC_DOMAIN),
                    "physical_grid_shape": grid_shape,
                    "physical_control_volume_count": node_count,
                    "material_claim_threshold": float(
                        registry["k_overlap_policy"].get(
                            "material_claim_threshold",
                            DEFAULT_MATERIAL_CLAIM_THRESHOLD,
                        )
                    ),
                },
                "k": {
                    "mode": "diag3" if mode != "scalar" else "scalar",
                    "diag3_policy": mode,
                    "blocks": material_blocks,
                },
                "q": {
                    "family": q_entry["name"],
                    "blocks": q_blocks,
                    "q_source_policy": Q_SOURCE_POLICY,
                    "semantic_boundary_inset_fraction": SEMANTIC_BOUNDARY_INSET_FRACTION,
                    "semantic_inset_domain_xyz": _semantic_inset_domain(),
                    "solver_safe_deposition_mask": SOLVER_SAFE_DEPOSITION_MASK,
                    "DeltaT_target_bin": q_entry["DeltaT_target_bin"],
                    "q_rescale_factor": 1.0,
                },
                "BC": {
                    "cooling_regime": cooling["name"],
                    "top_h_W_m2K": top_h,
                    "top_ambient_temperature_K": 300.0,
                    "bottom_dirichlet_temperature_K": 300.0,
                    "side_boundary_model": "adiabatic",
                    "bc_flag_channels": ["top", "bottom", "side", "interior"],
                    "metadata_tag": cooling["metadata_tag"],
                },
                "contact": {
                    "contact_model": PRODUCTION_CONTACT_MODEL,
                    "R_contact_m2K_W": 0.0,
                    "finite_contact_resistance_status": "implemented_deferred_not_v4_dataset_default",
                },
                "deltaT_qc": {
                    "deltaT_peak_K": None,
                    "deltaT_p95_K": None,
                    "deltaT_bin": PENDING_DELTAT_BIN,
                    "q_rescale_factor": 1.0,
                    "reject_reason": None,
                },
                "array_preview": {
                    "k_shape": [node_count, 3 if mode != "scalar" else 1],
                    "q_shape": [node_count, 1],
                    "bc_shape": [node_count, 4],
                    "q_nonzero_cell_count_upper_bound": q_cells,
                    "material_block_count": len(material_blocks),
                    "q_block_count": len(q_blocks),
                },
                "artifact_writes": False,
            }
        )

    diag3_count = sum(1 for scene in scenes if scene["k"]["mode"] == "diag3")
    hbm_count = sum(1 for scene in scenes if scene["k"]["diag3_policy"] == "hbm_like_strong")
    mild_count = sum(1 for scene in scenes if scene["k"]["diag3_policy"] == "mild")
    all_blocks = [
        block
        for scene in scenes
        for group in (scene["k"]["blocks"], scene["q"]["blocks"])
        for block in group
    ]
    summary = {
        "sample_count": sample_count,
        "seed": seed,
        "diag3_target_fraction": diag3_target,
        "diag3_count": diag3_count,
        "diag3_fraction": diag3_count / sample_count if sample_count else 0.0,
        "mild_diag3_count": mild_count,
        "hbm_like_strong_diag3_count": hbm_count,
        "q_family_counts": dict(Counter(scene["q"]["family"] for scene in scenes)),
        "cooling_regime_counts": dict(Counter(scene["BC"]["cooling_regime"] for scene in scenes)),
        "projection_resample_count": sum(
            1 for block in all_blocks if block["projection_status"] == "resampled_min_one_cell"
        ),
        "projection_reject_count": sum(1 for block in all_blocks if block["projection_status"] == "rejected"),
        "artifact_writes": False,
    }
    return {
        "schema_version": "heat3d_v4_p3c_generator_dryrun_v3",
        "registry_schema_version": registry.get("schema_version"),
        "final_probe_role": registry["generation_policy"]["final_probe_role"],
        "stress_split": registry["generation_policy"]["stress_split"],
        "artifact_writes": False,
        "summary": summary,
        "scenes": scenes,
    }


def materialize_scene_arrays(
    scene: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Materialize a dry scene into in-memory arrays without solving or writing."""

    validate_registry(registry)
    scene = deepcopy(scene)
    domain = scene["domain"]
    grid_shape = [int(v) for v in domain["grid_shape"]]
    node_count = int(domain["node_count"])
    coords = _grid_coords(domain)
    if coords.shape != (node_count, 3):
        raise ValueError(f"coords shape mismatch: {coords.shape} vs {(node_count, 3)}")

    is_diag3 = scene["k"]["mode"] == "diag3"
    background_family, background_value, background_meta = _background_k(registry, diag3=is_diag3)
    k_width = 3 if is_diag3 else 1
    k_field = np.repeat(background_value.reshape(1, k_width), node_count, axis=0)
    covered_by_blocks: list[list[str]] = [[] for _ in range(node_count)]
    winning_block_id: list[str | None] = [None for _ in range(node_count)]
    winning_block_overlap_fraction = np.zeros((node_count,), dtype=np.float64)
    material_claim_threshold = float(
        registry["k_overlap_policy"].get("material_claim_threshold", DEFAULT_MATERIAL_CLAIM_THRESHOLD)
    )

    for block in scene["k"]["blocks"]:
        overlaps = _block_overlap_fractions(block, grid_shape)
        indices = np.nonzero(overlaps >= material_claim_threshold)[0].astype(np.int64)
        block_value = _block_k_value(block, diag3=is_diag3)
        for idx in indices:
            covered_by_blocks[idx].append(block["block_id"])
            winning_block_id[idx] = block["block_id"]
            winning_block_overlap_fraction[idx] = float(overlaps[idx])
            k_field[idx, :] = block_value
        block["covered_cell_count"] = int(indices.size)
        block["claimed_cell_count"] = int(indices.size)
        block["material_claim_threshold"] = material_claim_threshold
        block["overlap_fraction_sum"] = float(np.sum(overlaps))
        block["max_overlap_fraction"] = float(np.max(overlaps)) if overlaps.size else 0.0

    q_field = np.zeros((node_count, 1), dtype=np.float64)
    q_contributors: list[list[str]] = [[] for _ in range(node_count)]
    q_contributor_overlaps: list[list[float]] = [[] for _ in range(node_count)]
    control_volume_weights = control_volume_weights_for_domain(domain)
    boundary_masks = _boundary_node_masks(grid_shape)
    solver_safe_mask = boundary_masks["solver_safe"]
    q_block_metadata = []
    for block in scene["q"]["blocks"]:
        overlaps = _block_overlap_fractions(block, grid_shape)
        deposited_overlaps = np.where(solver_safe_mask, overlaps, 0.0)
        blocked_overlap_fraction_sum = float(np.sum(overlaps) - np.sum(deposited_overlaps))
        indices = np.nonzero(deposited_overlaps > 0.0)[0].astype(np.int64)
        realized_volume = float(np.sum(deposited_overlaps * control_volume_weights))
        if realized_volume <= 0.0:
            raise ValueError(f"q block has non-positive realized volume: {block['block_id']}")
        target_power = float(block["integrated_power_target_W"])
        calibrated_q_density = target_power / realized_volume
        for idx in indices:
            overlap_fraction = float(deposited_overlaps[idx])
            q_field[idx, 0] += calibrated_q_density * overlap_fraction
            q_contributors[idx].append(block["block_id"])
            q_contributor_overlaps[idx].append(overlap_fraction)
        realized_power = calibrated_q_density * realized_volume
        power_error = realized_power - target_power
        block_meta = {
            "block_id": block["block_id"],
            "q_family": block["q_family"],
            "target_power_W": target_power,
            "realized_volume_m3": realized_volume,
            "calibrated_q_density_W_m3": calibrated_q_density,
            "realized_power_W": realized_power,
            "power_error_W": power_error,
            "power_integration_policy": Q_POWER_INTEGRATION_POLICY,
            "realized_cell_count": int(len(indices)),
            "overlap_fraction_sum": float(np.sum(overlaps)),
            "deposited_overlap_fraction_sum": float(np.sum(deposited_overlaps)),
            "blocked_boundary_overlap_fraction_sum": blocked_overlap_fraction_sum,
            "max_overlap_fraction": float(np.max(overlaps)) if overlaps.size else 0.0,
            "q_source_policy": Q_SOURCE_POLICY,
            "semantic_boundary_inset_fraction": SEMANTIC_BOUNDARY_INSET_FRACTION,
            "solver_safe_deposition_mask": SOLVER_SAFE_DEPOSITION_MASK,
            "DeltaT_target_bin": block["DeltaT_target_bin"],
            "metadata_tag": block["metadata_tag"],
        }
        block.update(block_meta)
        q_block_metadata.append(block_meta)

    bc_features, bc_counts = _bc_features(domain)
    uncovered = [idx for idx, winner in enumerate(winning_block_id) if winner is None]
    final_winner = [
        winner if winner is not None else "background"
        for winner in winning_block_id
    ]
    q_flat = q_field.reshape(-1)
    q_active_z_indices = []
    _, _, nz = grid_shape
    for idx in range(node_count):
        _, _, k = _node_ijk(idx, grid_shape)
        if q_flat[idx] > 0.0:
            q_active_z_indices.append(k)
    q_total_target_power = float(sum(block["target_power_W"] for block in q_block_metadata))
    q_weighted_power = q_flat * control_volume_weights
    q_integral_from_array = float(np.sum(q_weighted_power))
    q_power_on_bottom = float(np.sum(q_weighted_power[boundary_masks["bottom"]]))
    q_power_on_top = float(np.sum(q_weighted_power[boundary_masks["top"]]))
    q_power_on_xmin = float(np.sum(q_weighted_power[boundary_masks["xmin"]]))
    q_power_on_xmax = float(np.sum(q_weighted_power[boundary_masks["xmax"]]))
    q_power_on_ymin = float(np.sum(q_weighted_power[boundary_masks["ymin"]]))
    q_power_on_ymax = float(np.sum(q_weighted_power[boundary_masks["ymax"]]))
    q_power_on_side = float(np.sum(q_weighted_power[boundary_masks["side"]]))
    q_power_on_boundary = float(np.sum(q_weighted_power[boundary_masks["boundary"]]))
    q_boundary_violation_count = int(np.count_nonzero(q_flat[boundary_masks["boundary"]] > 0.0))
    q_side_boundary_violation_count = int(np.count_nonzero(q_flat[boundary_masks["side"]] > 0.0))
    q_active_z_min = float(min(q_active_z_indices)) if q_active_z_indices else None
    q_active_z_max = float(max(q_active_z_indices) + 1) if q_active_z_indices else None
    q_power_audit = {
        "q_total_target_power_W": q_total_target_power,
        "q_integral_from_array_W": q_integral_from_array,
        "q_total_power_error_W": q_integral_from_array - q_total_target_power,
        "q_power_on_bottom_W": q_power_on_bottom,
        "q_power_on_top_W": q_power_on_top,
        "q_power_on_xmin_W": q_power_on_xmin,
        "q_power_on_xmax_W": q_power_on_xmax,
        "q_power_on_ymin_W": q_power_on_ymin,
        "q_power_on_ymax_W": q_power_on_ymax,
        "q_power_on_side_W": q_power_on_side,
        "q_power_on_boundary_W": q_power_on_boundary,
        "q_power_on_bottom_fraction": (
            q_power_on_bottom / q_integral_from_array if q_integral_from_array > 0.0 else 0.0
        ),
        "q_power_on_top_fraction": (
            q_power_on_top / q_integral_from_array if q_integral_from_array > 0.0 else 0.0
        ),
        "q_power_on_side_fraction": (
            q_power_on_side / q_integral_from_array if q_integral_from_array > 0.0 else 0.0
        ),
        "q_source_boundary_violation_count": q_boundary_violation_count,
        "q_source_side_boundary_violation_count": q_side_boundary_violation_count,
        "q_active_z_min": q_active_z_min,
        "q_active_z_max": q_active_z_max,
        "q_power_integration_policy": Q_POWER_INTEGRATION_POLICY,
        "control_volume_weight_sum_m3": float(np.sum(control_volume_weights)),
        "solver_safe_control_volume_sum_m3": float(np.sum(control_volume_weights[solver_safe_mask])),
        "semantic_boundary_inset_fraction": SEMANTIC_BOUNDARY_INSET_FRACTION,
        "semantic_inset_domain_xyz": _semantic_inset_domain(),
        "solver_safe_deposition_mask": SOLVER_SAFE_DEPOSITION_MASK,
        "q_deposited_on_boundary_node_count": q_boundary_violation_count,
        "q_source_policy": Q_SOURCE_POLICY,
    }
    layer_id, region_id, material_id, layers, regions, materials = _semantic_id_arrays(
        grid_shape=grid_shape,
        final_winner=final_winner,
        k_blocks=scene["k"]["blocks"],
    )
    sample_meta = {
        "schema_version": "heat3d_v4_p3c_array_preflight_v3",
        "scene_id": scene["scene_id"],
        "seed": scene["seed"],
        "subset_name": "heat3d_v4_p3c_random_block",
        "stage": "physics_label_medium1024_gapA_generation_candidate",
        "split": "unassigned",
        "array_preflight_only": True,
        "artifact_writes": False,
        "solver_called": False,
        "domain": scene["domain"],
        "layers": layers,
        "regions": regions,
        "materials": materials,
        "boundary_regions": _boundary_regions(grid_shape, boundary_masks),
        "generation_config": {
            "generator": "heat3d_v4_p3c_dryrun_generator",
            "q_source_policy": Q_SOURCE_POLICY,
            "q_power_integration_policy": Q_POWER_INTEGRATION_POLICY,
        },
        "validation": {
            "solver_label_pending": True,
            "array_preflight_passed": True,
        },
        "parameter_sources": {
            "literature_backed": True,
            "provisional_engineering_assumption": True,
            "requires_user_confirmation": False,
        },
        "generation_policy": {
            "final_probe_role": registry["generation_policy"]["final_probe_role"],
            "stress_split": registry["generation_policy"]["stress_split"],
        },
        "background_k": {
            **background_meta,
            "allowed_families": list(registry["background_k_policy"]["allowed_families"]),
            "node_count": node_count,
            "uncovered_node_count": len(uncovered),
        },
        "k_overlap_policy": registry["k_overlap_policy"]["name"],
        "semantic_projection": {
            "semantic_domain_xyz": list(SEMANTIC_DOMAIN),
            "physical_grid_shape": grid_shape,
            "physical_control_volume_count": node_count,
            "material_claim_threshold": material_claim_threshold,
            "q_source_policy": Q_SOURCE_POLICY,
            "semantic_boundary_inset_fraction": SEMANTIC_BOUNDARY_INSET_FRACTION,
            "semantic_inset_domain_xyz": _semantic_inset_domain(),
            "solver_safe_deposition_mask": SOLVER_SAFE_DEPOSITION_MASK,
            "q_power_integration_policy": Q_POWER_INTEGRATION_POLICY,
        },
        "k_node_metadata": {
            "covered_by_blocks": covered_by_blocks,
            "winning_block_id": final_winner,
            "winning_block_overlap_fraction": winning_block_overlap_fraction.tolist(),
        },
        "q_overlap_policy": registry["q_overlap_policy"]["name"],
        "q_node_metadata": {
            "contributing_q_blocks": q_contributors,
            "contributing_q_overlap_fractions": q_contributor_overlaps,
        },
        "power_calibration_policy": registry["power_calibration_policy"]["name"],
        "q_block_metadata": q_block_metadata,
        "q_power_audit": q_power_audit,
        "bc_feature_names": ["is_top", "is_bottom", "is_side", "is_interior"],
        "bc_counts": bc_counts,
        **_solver_boundary_contract(scene),
        "interfaces": _solver_interface_contract(scene),
        "contact": scene["contact"],
        "deltaT_qc": scene["deltaT_qc"],
        "k_shape_policy": "diag3_[N,3]" if is_diag3 else "scalar_[N,1]",
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
            "bc_features": "one_hot_boundary_flags",
        },
    }
    return {
        "coords": coords,
        "layer_id": layer_id,
        "region_id": region_id,
        "material_id": material_id,
        "k_field": k_field,
        "q_field": q_field,
        "bc_features": bc_features,
        "sample_meta": sample_meta,
        "scene": scene,
    }


def build_smoke16_write_plan(
    registry: dict[str, Any],
    *,
    sample_count: int = SMOKE16_SAMPLE_COUNT,
    seed: int = SMOKE16_SEED,
    dataset_dir: str = SMOKE16_DATASET_DIR,
    output_dir: str = SMOKE16_OUTPUT_DIR,
) -> dict[str, Any]:
    """Return a no-write dataset plan for the P3c smoke dataset."""

    batch = generate_dryrun_batch(registry, sample_count=sample_count, seed=seed)
    samples = []
    for index, scene in enumerate(batch["scenes"]):
        sample_id = f"sample_{index:03d}"
        samples.append(
            {
                "sample_id": sample_id,
                "scene_id": scene["scene_id"],
                "sample_dir": f"{dataset_dir}/{sample_id}",
                "planned_files": list(PLANNED_SAMPLE_FILES),
                "label_file_after_solver": "temperature.npy",
                "k_mode": scene["k"]["mode"],
                "diag3_policy": scene["k"]["diag3_policy"],
                "q_family": scene["q"]["family"],
                "cooling_regime": scene["BC"]["cooling_regime"],
            }
        )
    return {
        "schema_version": "heat3d_v4_p3c_smoke16_write_plan_v3",
        "sample_count": sample_count,
        "seed": seed,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "artifact_writes": False,
        "solver_called": False,
        "root_dataset_files": ["manifest.json"],
        "root_output_files": ["audit_summary.json"],
        "sample_schema": {
            "required_files": list(PLANNED_SAMPLE_FILES),
            "label_file_after_solver": "temperature.npy",
        },
        "coverage": {
            "k_modes": sorted({sample["k_mode"] for sample in samples}),
            "diag3_policies": sorted({sample["diag3_policy"] for sample in samples}),
            "q_families": sorted({sample["q_family"] for sample in samples}),
            "cooling_regimes": sorted({sample["cooling_regime"] for sample in samples}),
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preview-limit", type=int, default=3)
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    batch = generate_dryrun_batch(registry, sample_count=args.samples, seed=args.seed)
    preview_limit = max(0, int(args.preview_limit))
    output = {
        "schema_version": batch["schema_version"],
        "registry_schema_version": batch["registry_schema_version"],
        "artifact_writes": batch["artifact_writes"],
        "final_probe_role": batch["final_probe_role"],
        "stress_split": batch["stress_split"],
        "summary": batch["summary"],
        "scene_preview": batch["scenes"][:preview_limit],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
