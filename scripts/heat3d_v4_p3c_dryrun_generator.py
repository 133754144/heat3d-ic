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
from pathlib import Path
from typing import Any


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


def _project_block(
    *,
    grid_shape: list[int],
    xy_fraction: float,
    z_fraction: float,
    rng: random.Random,
) -> dict[str, Any]:
    nx, ny, nz = [int(v) for v in grid_shape]
    total_cells = nx * ny * nz
    side_fraction = math.sqrt(max(float(xy_fraction), 0.0))
    requested_dims = [
        math.floor(nx * side_fraction),
        math.floor(ny * side_fraction),
        math.floor(nz * max(float(z_fraction), 0.0)),
    ]
    realized_dims = [
        min(nx, max(1, requested_dims[0])),
        min(ny, max(1, requested_dims[1])),
        min(nz, max(1, requested_dims[2])),
    ]
    starts = [
        rng.randint(0, nx - realized_dims[0]),
        rng.randint(0, ny - realized_dims[1]),
        rng.randint(0, nz - realized_dims[2]),
    ]
    realized_cell_count = realized_dims[0] * realized_dims[1] * realized_dims[2]
    adjusted = any(dim <= 0 for dim in requested_dims)
    return {
        "requested_fraction": float(xy_fraction) * float(z_fraction),
        "requested_xy_fraction": float(xy_fraction),
        "requested_z_fraction": float(z_fraction),
        "requested_dims_floor": requested_dims,
        "start_ijk": starts,
        "extent_ijk": realized_dims,
        "realized_fraction": realized_cell_count / total_cells,
        "realized_cell_count": realized_cell_count,
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
    blocks: list[dict[str, Any]] = []
    for block_index in range(count):
        xy_fraction = _sample_fraction(rng, xy_entry, fallback_min=0.05, fallback_max=0.6)
        z_fraction = _sample_fraction(rng, z_entry, fallback_min=0.25, fallback_max=1.0)
        projection = _project_block(
            grid_shape=grid_shape,
            xy_fraction=xy_fraction,
            z_fraction=z_fraction,
            rng=rng,
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
    strong_count = max(1, int(round(diag3_count * 0.2))) if diag3_count >= 5 else 0
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
                },
                "k": {
                    "mode": "diag3" if mode != "scalar" else "scalar",
                    "diag3_policy": mode,
                    "blocks": material_blocks,
                },
                "q": {
                    "family": q_entry["name"],
                    "blocks": q_blocks,
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
        "schema_version": "heat3d_v4_p3c_generator_dryrun_v0",
        "registry_schema_version": registry.get("schema_version"),
        "final_probe_role": registry["generation_policy"]["final_probe_role"],
        "stress_split": registry["generation_policy"]["stress_split"],
        "artifact_writes": False,
        "summary": summary,
        "scenes": scenes,
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
