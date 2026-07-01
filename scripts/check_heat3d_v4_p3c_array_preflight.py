#!/usr/bin/env python3
"""Check P3c real-array preflight without writing data or calling the solver."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heat3d_v4_p3c_dryrun_generator import (  # noqa: E402
    FINAL_PROBE_ROLE,
    PENDING_DELTAT_BIN,
    PLANNED_SAMPLE_FILES,
    PRODUCTION_CONTACT_MODEL,
    Q_ACTIVE_Z_MAX,
    Q_ACTIVE_Z_MIN,
    Q_SOURCE_Z_POLICY,
    SEMANTIC_DOMAIN,
    SMOKE16_SAMPLE_COUNT,
    SMOKE16_SEED,
    build_smoke16_write_plan,
    generate_dryrun_batch,
    load_registry,
    materialize_scene_arrays,
)


REGISTRY = REPO_ROOT / "configs/heat3d_v4/p3c_parameter_registry.json"
SAMPLE_COUNT = 50
SEED = 4301
POWER_TOL = 1.0e-12
FORBIDDEN_ARTIFACT_KEYS = {
    "data_path",
    "output_path",
    "checkpoint_path",
    "log_path",
    "artifact_path",
}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _first_scene(batch: dict[str, Any], *, mode: str) -> dict[str, Any]:
    for scene in batch["scenes"]:
        if scene["k"]["mode"] == mode:
            return scene
    raise AssertionError(f"missing scene mode: {mode}")


def _block_indices(block: dict[str, Any], grid_shape: list[int]) -> list[int]:
    _, ny, nz = [int(v) for v in grid_shape]
    start_i, start_j, start_k = [int(v) for v in block["start_ijk"]]
    extent_i, extent_j, extent_k = [int(v) for v in block["extent_ijk"]]
    indices: list[int] = []
    for i in range(start_i, start_i + extent_i):
        for j in range(start_j, start_j + extent_j):
            for k in range(start_k, start_k + extent_k):
                indices.append((i * ny + j) * nz + k)
    return indices


def _check_registry_policies(registry: dict[str, Any]) -> None:
    for section in (
        "background_k_policy",
        "k_overlap_policy",
        "q_overlap_policy",
        "power_calibration_policy",
        "q_source_z_policy",
    ):
        _expect(section in registry, f"missing registry policy: {section}")
    background = registry["background_k_policy"]
    _expect(background["default_family"] == "effective_stack_medium_k", "bad default background")
    _expect(
        background["allowed_families"]
        == ["effective_stack_medium_k", "silicon_like", "hbm_like_anisotropic_k"],
        "background allowed families mismatch",
    )
    for family in background["families"]:
        for field in ("source_ref", "source_type", "rationale", "metadata_tag"):
            _expect(field in family, f"background family missing {field}")
        _expect("reference_values_W_mK" in family, "background family missing reference values")
    _expect(
        background["low_k_dielectric_underfill_policy"]
        == "minority_background_or_block_only_not_default_background",
        "low-k background policy mismatch",
    )
    _expect(registry["k_overlap_policy"]["name"] == "deterministic_priority_override", "bad k policy")
    _expect(
        registry["k_overlap_policy"]["projection"] == "continuous_semantic_bbox_overlap",
        "bad k projection policy",
    )
    _expect(
        0.0 < float(registry["k_overlap_policy"]["material_claim_threshold"]) <= 1.0,
        "bad material claim threshold",
    )
    _expect(registry["q_overlap_policy"]["name"] == "sum_volumetric_sources", "bad q policy")
    _expect(
        registry["q_overlap_policy"]["projection"] == "continuous_semantic_bbox_overlap_fraction",
        "bad q projection policy",
    )
    _expect(registry["q_overlap_policy"]["q_source_z_policy"] == Q_SOURCE_Z_POLICY, "bad q source link")
    q_source = registry["q_source_z_policy"]
    _expect(q_source["name"] == Q_SOURCE_Z_POLICY, "bad q source z policy")
    _expect(q_source["semantic_domain_xyz"] == list(SEMANTIC_DOMAIN), "bad semantic domain")
    _expect(q_source["active_z_min"] == Q_ACTIVE_Z_MIN, "bad active z min")
    _expect(q_source["active_z_max"] == Q_ACTIVE_Z_MAX, "bad active z max")
    _expect(
        registry["power_calibration_policy"]["name"]
        == "calibrate_q_density_from_realized_volume_and_integrated_power_target",
        "bad power calibration policy",
    )
    _expect(
        registry["generation_policy"]["final_probe_role"] == FINAL_PROBE_ROLE,
        "final_probe must remain reference-only",
    )


def _check_array_bundle(bundle: dict[str, Any], *, expected_k_width: int) -> None:
    coords = bundle["coords"]
    k_field = bundle["k_field"]
    q_field = bundle["q_field"]
    bc_features = bundle["bc_features"]
    meta = bundle["sample_meta"]
    node_count = coords.shape[0]
    _expect(coords.shape == (node_count, 3), "coords shape mismatch")
    _expect(k_field.shape == (node_count, expected_k_width), "k_field shape mismatch")
    _expect(q_field.shape == (node_count, 1), "q_field shape mismatch")
    _expect(bc_features.shape == (node_count, 4), "bc_features shape mismatch")
    _expect(np.all(np.isfinite(coords)), "coords contain NaN/Inf")
    _expect(np.all(np.isfinite(k_field)), "k_field contains NaN/Inf")
    _expect(np.all(np.isfinite(q_field)), "q_field contains NaN/Inf")
    _expect(np.all(np.isfinite(bc_features)), "bc_features contain NaN/Inf")
    _expect(meta["artifact_writes"] is False, "array preflight must not write artifacts")
    _expect(meta["solver_called"] is False, "array preflight must not call solver")
    _expect(meta["contact"]["contact_model"] == PRODUCTION_CONTACT_MODEL, "contact model must be R=0")
    _expect(meta["contact"]["R_contact_m2K_W"] == 0.0, "R_contact must be zero")
    _expect(meta["deltaT_qc"]["deltaT_bin"] == PENDING_DELTAT_BIN, "DeltaT must stay pending")
    projection = meta["semantic_projection"]
    _expect(projection["semantic_domain_xyz"] == list(SEMANTIC_DOMAIN), "semantic domain mismatch")
    _expect(projection["physical_grid_shape"] == bundle["scene"]["domain"]["grid_shape"], "grid shape mismatch")
    _expect(projection["physical_control_volume_count"] == 1024, "physical grid must remain 16x16x4")
    _expect(projection["q_source_z_policy"] == Q_SOURCE_Z_POLICY, "q source z policy mismatch")
    _expect(projection["q_active_z_range"] == [Q_ACTIVE_Z_MIN, Q_ACTIVE_Z_MAX], "q active z range mismatch")

    background = meta["background_k"]
    _expect(background["background_k_family"] in background["allowed_families"], "bad background family")
    _expect(background["background_k_family"] != "low_k_dielectric_underfill", "low-k cannot be default")
    _expect(background["uncovered_node_count"] > 0, "expected uncovered nodes using background k")
    winners = meta["k_node_metadata"]["winning_block_id"]
    winner_overlaps = meta["k_node_metadata"]["winning_block_overlap_fraction"]
    _expect(len(winner_overlaps) == node_count, "winning overlap metadata length mismatch")
    background_indices = [idx for idx, winner in enumerate(winners) if winner == "background"]
    _expect(background_indices, "no background-owned nodes found")
    bg_value = background["background_k_value"]
    if expected_k_width == 1:
        expected_bg = np.array([float(bg_value)])
    else:
        expected_bg = np.array([float(bg_value["kx"]), float(bg_value["ky"]), float(bg_value["kz"])])
    _expect(np.allclose(k_field[background_indices[0]], expected_bg), "background k value mismatch")

    flag_sums = bc_features.sum(axis=1)
    _expect(np.allclose(flag_sums, np.ones_like(flag_sums)), "BC flags must be one-hot")
    counts = meta["bc_counts"]
    _expect(sum(counts.values()) == node_count, "BC counts do not sum to node count")
    grid_shape = bundle["scene"]["domain"]["grid_shape"]
    nx, ny, _ = [int(v) for v in grid_shape]
    _expect(counts["top"] == nx * ny, "top flag count mismatch")
    _expect(counts["bottom"] == nx * ny, "bottom flag count mismatch")
    _expect(counts["side"] > 0 and counts["interior"] > 0, "side/interior flags missing")

    for block_meta in meta["q_block_metadata"]:
        _expect(block_meta["realized_volume_m3"] > 0.0, "q block realized volume must be positive")
        _expect(block_meta["calibrated_q_density_W_m3"] > 0.0, "q density must be positive")
        _expect(abs(block_meta["power_error_W"]) <= POWER_TOL, "q power calibration error too large")
        _expect(block_meta["q_source_z_policy"] == Q_SOURCE_Z_POLICY, "q block source policy mismatch")

    q_audit = meta["q_power_audit"]
    for field in (
        "q_total_target_power_W",
        "q_integral_from_array_W",
        "q_total_power_error_W",
        "q_power_on_bottom_W",
        "q_power_on_top_W",
        "q_power_on_boundary_W",
        "q_power_on_bottom_fraction",
        "q_power_on_top_fraction",
        "q_source_boundary_violation_count",
        "q_active_z_min",
        "q_active_z_max",
    ):
        _expect(field in q_audit, f"q audit missing {field}")
    _expect(abs(q_audit["q_total_power_error_W"]) <= POWER_TOL, "q total power error too large")
    _expect(q_audit["q_power_on_bottom_W"] == 0.0, "q power touched bottom Dirichlet boundary")
    _expect(q_audit["q_power_on_top_W"] == 0.0, "q power touched top Robin boundary")
    _expect(q_audit["q_power_on_boundary_W"] == 0.0, "q power touched z boundary")
    _expect(q_audit["q_power_on_bottom_fraction"] == 0.0, "bottom q fraction must be zero")
    _expect(q_audit["q_power_on_top_fraction"] == 0.0, "top q fraction must be zero")
    _expect(q_audit["q_source_boundary_violation_count"] == 0, "q boundary violation count must be zero")
    _expect(q_audit["q_active_z_min"] >= Q_ACTIVE_Z_MIN, "q active z below interior range")
    _expect(q_audit["q_active_z_max"] <= Q_ACTIVE_Z_MAX, "q active z above interior range")
    domain = bundle["scene"]["domain"]
    xy_m = float(domain["domain_xy_mm"]) * 1.0e-3
    z_m = float(domain["domain_z_mm"]) * 1.0e-3
    node_volume = xy_m * xy_m * z_m / node_count
    _expect(np.isclose(np.sum(q_field) * node_volume, q_audit["q_integral_from_array_W"]), "q array integral mismatch")

    q_grid = q_field.reshape(bundle["scene"]["domain"]["grid_shape"])
    _expect(np.all(q_grid[:, :, 0] == 0.0), "bottom q layer must be zero")
    _expect(np.all(q_grid[:, :, -1] == 0.0), "top q layer must be zero")


def _check_boundary_bridge(bundle: dict[str, Any]) -> None:
    scene_bc = bundle["scene"]["BC"]
    meta = bundle["sample_meta"]
    _expect(
        meta["boundary_types"] == {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "solver boundary_types contract mismatch",
    )
    params = meta["boundary_params"]
    top = params["top"]
    bottom = params["bottom"]
    side = params["side"]
    _expect(top["type"] == "robin", "top boundary type mismatch")
    _expect(bottom["type"] == "dirichlet", "bottom boundary type mismatch")
    _expect(side["type"] == "adiabatic", "side boundary type mismatch")
    _expect(np.isclose(top["h_W_m2K"], scene_bc["top_h_W_m2K"]), "top h bridge mismatch")
    _expect(np.isclose(top["T_inf_K"], scene_bc["top_ambient_temperature_K"]), "top T_inf bridge mismatch")
    _expect(
        np.isclose(top["ambient_temperature_K"], scene_bc["top_ambient_temperature_K"]),
        "legacy top ambient bridge mismatch",
    )
    _expect(
        np.isclose(bottom["T_fixed_K"], scene_bc["bottom_dirichlet_temperature_K"]),
        "bottom T_fixed bridge mismatch",
    )
    _expect(
        np.isclose(bottom["fixed_temperature_K"], scene_bc["bottom_dirichlet_temperature_K"]),
        "legacy bottom fixed bridge mismatch",
    )

    # Simulate both the requested bridge contract and the current solver access pattern.
    _ = meta["boundary_params"]["top"]["h_W_m2K"]
    _ = meta["boundary_params"]["top"]["T_inf_K"]
    _ = meta["boundary_params"]["top"]["ambient_temperature_K"]
    _ = meta["boundary_params"]["bottom"]["T_fixed_K"]
    _ = meta["boundary_params"]["bottom"]["fixed_temperature_K"]
    _ = meta["boundary_params"]["side"]["type"]
    interfaces = meta["interfaces"]
    _expect(isinstance(interfaces, list) and interfaces, "perfect-contact interface missing")
    _expect(interfaces[0]["type"] == "perfect_contact", "interface must stay perfect_contact")
    _expect(interfaces[0]["R_contact_m2K_W"] == 0.0, "P3c production interface must keep R=0")


def _check_k_overlap_override(scene: dict[str, Any], registry: dict[str, Any]) -> None:
    overlap_scene = deepcopy(scene)
    first_block = deepcopy(overlap_scene["k"]["blocks"][0])
    override = deepcopy(first_block)
    override["block_id"] = "m_override"
    override["k_family"] = "silicon_like"
    override["k_value"] = {"k": 77.0}
    override["metadata_tag"] = "k_class=override_test"
    overlap_scene["k"]["blocks"].append(override)
    bundle = materialize_scene_arrays(overlap_scene, registry)
    meta = bundle["sample_meta"]
    claimed_indices = [
        idx
        for idx, winner in enumerate(meta["k_node_metadata"]["winning_block_id"])
        if winner == "m_override"
    ]
    _expect(claimed_indices, "override did not claim any control volume")
    test_idx = claimed_indices[0]
    _expect(meta["k_overlap_policy"] == "deterministic_priority_override", "bad k overlap policy")
    _expect("m_override" in meta["k_node_metadata"]["covered_by_blocks"][test_idx], "override not recorded")
    _expect(
        meta["k_node_metadata"]["winning_block_overlap_fraction"][test_idx]
        >= meta["semantic_projection"]["material_claim_threshold"],
        "override overlap below claim threshold",
    )
    _expect(float(bundle["k_field"][test_idx, 0]) == 77.0, "k override value mismatch")


def _check_q_overlap_sum(scene: dict[str, Any], registry: dict[str, Any]) -> None:
    overlap_scene = deepcopy(scene)
    base = deepcopy(overlap_scene["q"]["blocks"][0])
    q1 = deepcopy(base)
    q2 = deepcopy(base)
    q1["block_id"] = "q_overlap_a"
    q2["block_id"] = "q_overlap_b"
    q1["integrated_power_target_W"] = 1.0
    q2["integrated_power_target_W"] = 2.0
    overlap_scene["q"]["blocks"] = [q1, q2]
    bundle = materialize_scene_arrays(overlap_scene, registry)
    contributors_by_node = bundle["sample_meta"]["q_node_metadata"]["contributing_q_blocks"]
    overlap_fractions_by_node = bundle["sample_meta"]["q_node_metadata"]["contributing_q_overlap_fractions"]
    matching_indices = [
        idx
        for idx, contributors in enumerate(contributors_by_node)
        if contributors == ["q_overlap_a", "q_overlap_b"]
    ]
    _expect(matching_indices, "q overlap test found no shared projected cell")
    test_idx = matching_indices[0]
    q_meta = {entry["block_id"]: entry for entry in bundle["sample_meta"]["q_block_metadata"]}
    q_a = q_meta["q_overlap_a"]["calibrated_q_density_W_m3"]
    q_b = q_meta["q_overlap_b"]["calibrated_q_density_W_m3"]
    q_value = float(bundle["q_field"][test_idx, 0])
    frac_a, frac_b = overlap_fractions_by_node[test_idx]
    expected_value = q_a * frac_a + q_b * frac_b
    _expect(bundle["sample_meta"]["q_overlap_policy"] == "sum_volumetric_sources", "bad q overlap policy")
    _expect(np.isclose(q_value, expected_value), "q overlap must sum overlap-weighted densities")
    _expect(q_value > max(q_a * frac_a, q_b * frac_b), "q overlap behaved like max pooling")
    contributors = bundle["sample_meta"]["q_node_metadata"]["contributing_q_blocks"][test_idx]
    _expect(contributors == ["q_overlap_a", "q_overlap_b"], "q contributors metadata mismatch")


def _check_no_artifact_keys(value: Any) -> None:
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                _expect(key not in FORBIDDEN_ARTIFACT_KEYS, f"forbidden artifact key emitted: {key}")
                stack.append(nested)
        elif isinstance(current, list):
            stack.extend(current)


def _check_smoke16_write_plan(registry: dict[str, Any]) -> None:
    plan = build_smoke16_write_plan(registry, sample_count=SMOKE16_SAMPLE_COUNT, seed=SMOKE16_SEED)
    _expect(plan["artifact_writes"] is False, "write plan must not write artifacts")
    _expect(plan["solver_called"] is False, "write plan must not call solver")
    _expect(plan["sample_count"] == SMOKE16_SAMPLE_COUNT, "write plan sample count mismatch")
    _expect(plan["sample_schema"]["required_files"] == list(PLANNED_SAMPLE_FILES), "sample schema mismatch")
    _expect("manifest.json" in plan["root_dataset_files"], "manifest plan missing")
    _expect("audit_summary.json" in plan["root_output_files"], "audit summary plan missing")

    q_families = {entry["name"] for entry in registry["parameters"]["q"]}
    cooling_regimes = {entry["name"] for entry in registry["cooling_regimes"]}
    coverage = plan["coverage"]
    _expect("scalar" in coverage["k_modes"], "write plan missing scalar k mode")
    _expect("mild" in coverage["diag3_policies"], "write plan missing mild diag3")
    _expect("hbm_like_strong" in coverage["diag3_policies"], "write plan missing HBM-like diag3")
    _expect(q_families.issubset(set(coverage["q_families"])), "write plan missing q family coverage")
    _expect(
        cooling_regimes.issubset(set(coverage["cooling_regimes"])),
        "write plan missing cooling regime coverage",
    )

    batch = generate_dryrun_batch(registry, sample_count=SMOKE16_SAMPLE_COUNT, seed=SMOKE16_SEED)
    for scene in batch["scenes"]:
        bundle = materialize_scene_arrays(scene, registry)
        _check_boundary_bridge(bundle)
        _expect(bundle["sample_meta"]["artifact_writes"] is False, "write-plan materialization wrote artifact")
        _expect(bundle["sample_meta"]["solver_called"] is False, "write-plan materialization called solver")


def main() -> int:
    print("Heat3D V4 P3c array preflight check")
    print("scope: in-memory coords/k/q/BC arrays only; no solver, no dataset, no artifact writes")
    registry = load_registry(REGISTRY)
    _check_registry_policies(registry)
    batch = generate_dryrun_batch(registry, sample_count=SAMPLE_COUNT, seed=SEED)
    scalar_scene = _first_scene(batch, mode="scalar")
    diag3_scene = _first_scene(batch, mode="diag3")
    scalar_bundle = materialize_scene_arrays(scalar_scene, registry)
    diag3_bundle = materialize_scene_arrays(diag3_scene, registry)
    _check_array_bundle(scalar_bundle, expected_k_width=1)
    _check_array_bundle(diag3_bundle, expected_k_width=3)
    _check_boundary_bridge(scalar_bundle)
    _check_boundary_bridge(diag3_bundle)
    _check_k_overlap_override(scalar_scene, registry)
    _check_q_overlap_sum(scalar_scene, registry)
    _check_no_artifact_keys(scalar_bundle["sample_meta"])
    _check_no_artifact_keys(diag3_bundle["sample_meta"])
    _check_smoke16_write_plan(registry)
    print(
        "- "
        f"scalar_k_shape={scalar_bundle['k_field'].shape} "
        f"diag3_k_shape={diag3_bundle['k_field'].shape} "
        f"q_shape={scalar_bundle['q_field'].shape} "
        f"bc_shape={scalar_bundle['bc_features'].shape} "
        f"smoke16_write_plan_samples={SMOKE16_SAMPLE_COUNT} "
        f"background={scalar_bundle['sample_meta']['background_k']['background_k_family']} "
        f"final_probe_role={batch['final_probe_role']}"
    )
    print("p3c_array_preflight_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
