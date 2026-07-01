#!/usr/bin/env python3
"""Check the V4 P3c dry-run generator contract without writing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heat3d_v4_p3c_dryrun_generator import (  # noqa: E402
    FINAL_PROBE_ROLE,
    PENDING_DELTAT_BIN,
    PRODUCTION_CONTACT_MODEL,
    Q_ACTIVE_Z_MAX,
    Q_ACTIVE_Z_MIN,
    Q_SOURCE_Z_POLICY,
    SEMANTIC_DOMAIN,
    generate_dryrun_batch,
    load_registry,
    project_block_preview,
)


REGISTRY = REPO_ROOT / "configs/heat3d_v4/p3c_parameter_registry.json"
SAMPLE_COUNT = 50
SEED = 4301


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _walk_blocks(batch: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for scene in batch["scenes"]:
        blocks.extend(scene["k"]["blocks"])
        blocks.extend(scene["q"]["blocks"])
    return blocks


def _check_schema_and_policy(batch: dict[str, Any]) -> None:
    _expect(batch["final_probe_role"] == FINAL_PROBE_ROLE, "final_probe must be reference-only")
    _expect(batch["stress_split"] == "disabled", "stress split must be disabled")
    _expect(batch["artifact_writes"] is False, "dry-run must not write artifacts")
    _expect(batch["summary"]["artifact_writes"] is False, "summary must report no artifact writes")


def _check_diag3(batch: dict[str, Any]) -> None:
    summary = batch["summary"]
    target = float(summary["diag3_target_fraction"])
    realized = float(summary["diag3_fraction"])
    tolerance = 1.0 / float(summary["sample_count"])
    _expect(abs(realized - target) <= tolerance, f"diag3 fraction mismatch: {realized} vs {target}")
    _expect(summary["mild_diag3_count"] > summary["hbm_like_strong_diag3_count"], "mild diag3 must dominate")
    _expect(
        summary["hbm_like_strong_diag3_count"] <= max(1, summary["diag3_count"] // 4),
        "HBM-like strong anisotropy must be a small diag3 subset",
    )

    hbm_scenes = [scene for scene in batch["scenes"] if scene["k"]["diag3_policy"] == "hbm_like_strong"]
    _expect(hbm_scenes, "dry-run should include at least one HBM-like strong anisotropy scene")
    for scene in hbm_scenes:
        hbm_blocks = [block for block in scene["k"]["blocks"] if block["hbm_like_strong_anisotropy"]]
        _expect(hbm_blocks, "HBM-like diag3 scene missing strong-anisotropy block")
        for block in hbm_blocks:
            _expect("hbm_like_diag3" in block["metadata_tag"], "HBM-like block missing metadata tag")


def _check_block_projection(batch: dict[str, Any]) -> None:
    blocks = _walk_blocks(batch)
    _expect(blocks, "dry-run produced no blocks")
    for block in blocks:
        for field in (
            "requested_fraction",
            "realized_fraction",
            "realized_cell_count",
            "continuous_bbox",
            "overlap_fraction_sum",
        ):
            _expect(field in block, f"block missing projection field: {field}")
        _expect(block["realized_cell_count"] >= 1, "block projected to zero cells")
        _expect(block["realized_fraction"] > 0.0, "block realized fraction must be positive")
        bbox = block["continuous_bbox"]
        _expect(0.0 <= bbox["x_min"] < bbox["x_max"] <= SEMANTIC_DOMAIN[0], "bad x semantic bbox")
        _expect(0.0 <= bbox["y_min"] < bbox["y_max"] <= SEMANTIC_DOMAIN[1], "bad y semantic bbox")
        _expect(0.0 <= bbox["z_min"] < bbox["z_max"] <= SEMANTIC_DOMAIN[2], "bad z semantic bbox")
        if block["projection_status"] != "realized":
            _expect(block["projection_status"] in {"resampled_min_one_cell", "rejected"}, "bad block status")
            _expect(block["reject_reason"], "invalid/resampled block must carry reject_reason")

    tiny_projection = project_block_preview(
        grid_shape=[16, 16, 4],
        xy_fraction=1.0e-8,
        z_fraction=1.0e-8,
        seed=SEED,
    )
    _expect(tiny_projection["realized_cell_count"] >= 1, "tiny block must be resampled to >=1 cell")
    _expect(tiny_projection["projection_status"] == "resampled_min_one_cell", "tiny block resample missing")
    _expect(tiny_projection["reject_reason"], "tiny resample must carry reject_reason")
    _expect(batch["summary"]["projection_reject_count"] == 0, "dry-run should resample, not reject, v0 blocks")


def _check_q_and_delta_t(batch: dict[str, Any]) -> None:
    q_families = set()
    for scene in batch["scenes"]:
        _expect(
            scene["semantic_projection"]["mode"] == "continuous_bbox_to_physical_control_volume_overlap",
            "semantic projection mode mismatch",
        )
        q_families.add(scene["q"]["family"])
        _expect(scene["q"]["q_source_z_policy"] == Q_SOURCE_Z_POLICY, "q source z policy mismatch")
        _expect(scene["q"]["q_active_z_range"] == [Q_ACTIVE_Z_MIN, Q_ACTIVE_Z_MAX], "q active z range mismatch")
        _expect(scene["deltaT_qc"]["deltaT_bin"] == PENDING_DELTAT_BIN, "DeltaT bin must stay pending")
        _expect(scene["deltaT_qc"]["q_rescale_factor"] == 1.0, "q rescale factor must be 1.0")
        _expect(scene["deltaT_qc"]["reject_reason"] is None, "dry-run must not solve/reject by DeltaT")
        for block in scene["q"]["blocks"]:
            for field in (
                "source_volume_fraction",
                "integrated_power_target_W",
                "DeltaT_target_bin",
                "q_density_W_m3",
            ):
                _expect(field in block, f"q block missing {field}")
            _expect(block["source_volume_fraction"] > 0.0, "q source volume fraction must be positive")
            _expect(block["integrated_power_target_W"] > 0.0, "q integrated power must be positive")
            _expect(block["DeltaT_target_bin"], "q block missing DeltaT target bin")
            _expect(block["z_policy"] == Q_SOURCE_Z_POLICY, "q block z policy mismatch")
            bbox = block["continuous_bbox"]
            _expect(bbox["z_min"] >= Q_ACTIVE_Z_MIN, "q bbox touches bottom boundary domain")
            _expect(bbox["z_max"] <= Q_ACTIVE_Z_MAX, "q bbox touches top boundary domain")
    _expect(len(q_families) >= 7, "all q families should be exercised in the dry-run batch")


def _check_contact_and_bc(batch: dict[str, Any]) -> None:
    cooling_regimes = set()
    for scene in batch["scenes"]:
        _expect(scene["contact"]["contact_model"] == PRODUCTION_CONTACT_MODEL, "contact model must be R=0")
        _expect(scene["contact"]["R_contact_m2K_W"] == 0.0, "contact resistance must be zero")
        _expect(
            scene["contact"]["finite_contact_resistance_status"]
            == "implemented_deferred_not_v4_dataset_default",
            "finite contact status mismatch",
        )
        _expect(scene["BC"]["bc_flag_channels"] == ["top", "bottom", "side", "interior"], "BC flags mismatch")
        cooling_regimes.add(scene["BC"]["cooling_regime"])
    _expect(
        cooling_regimes
        == {"weak_effective_air", "nominal_package", "strong_forced_or_effective_heatsink"},
        f"cooling regime coverage mismatch: {cooling_regimes}",
    )


def _check_no_artifact_contract(batch: dict[str, Any]) -> None:
    forbidden_keys = {"data_path", "output_path", "checkpoint_path", "log_path", "artifact_path"}
    stack: list[Any] = [batch]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, nested in value.items():
                _expect(key not in forbidden_keys, f"dry-run emitted forbidden artifact key: {key}")
                stack.append(nested)
        elif isinstance(value, list):
            stack.extend(value)


def main() -> int:
    print("Heat3D V4 P3c dry-run generator check")
    print("scope: registry + in-memory dry scenes only; no solver, no dataset, no artifact writes")
    registry = load_registry(REGISTRY)
    batch = generate_dryrun_batch(registry, sample_count=SAMPLE_COUNT, seed=SEED)
    _check_schema_and_policy(batch)
    _check_diag3(batch)
    _check_block_projection(batch)
    _check_q_and_delta_t(batch)
    _check_contact_and_bc(batch)
    _check_no_artifact_contract(batch)
    summary = batch["summary"]
    print(
        "- "
        f"samples={summary['sample_count']} "
        f"seed={summary['seed']} "
        f"diag3_fraction={summary['diag3_fraction']:.3f} "
        f"mild_diag3={summary['mild_diag3_count']} "
        f"hbm_like_strong={summary['hbm_like_strong_diag3_count']} "
        f"q_families={len(summary['q_family_counts'])} "
        f"projection_resamples={summary['projection_resample_count']} "
        f"artifact_writes={summary['artifact_writes']}"
    )
    print("p3c_generator_dryrun_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
