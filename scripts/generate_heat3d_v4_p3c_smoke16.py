#!/usr/bin/env python3
"""Generate and solve the local V4 P3c 16-sample smoke dataset.

This script is intentionally limited to the P3c smoke dataset path. It writes
only the user-scoped smoke dataset and audit directories, calls the V4
reference solver for labels, and never starts model training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heat3d_v4_p3c_dryrun_generator import (  # noqa: E402
    DEFAULT_REGISTRY,
    PLANNED_SAMPLE_FILES,
    SMOKE16_DATASET_DIR,
    SMOKE16_OUTPUT_DIR,
    SMOKE16_SAMPLE_COUNT,
    SMOKE16_SEED,
    build_smoke16_write_plan,
    generate_dryrun_batch,
    load_registry,
    materialize_scene_arrays,
)
from rigno.heat3d_v4_reference_solver import (  # noqa: E402
    SolverOptions,
    extract_problem_from_arrays,
    solve_temperature_from_problem,
)


DELTA_T_BINS = (
    ("reject_low", None, 0.02),
    ("low", 0.02, 0.2),
    ("nominal", 0.2, 2.0),
    ("hard", 2.0, 8.0),
    ("review_high", 8.0, 15.0),
    ("reject_high", 15.0, None),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _ensure_clean_target(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if force:
        shutil.rmtree(path)
        return
    raise FileExistsError(f"target already exists; pass --force to replace: {path}")


def _node_volume_m3(scene: dict[str, Any]) -> float:
    domain = scene["domain"]
    xy_m = float(domain["domain_xy_mm"]) * 1.0e-3
    z_m = float(domain["domain_z_mm"]) * 1.0e-3
    return xy_m * xy_m * z_m / float(domain["node_count"])


def _delta_t_bin(delta_t_peak: float) -> tuple[str, str | None]:
    for name, low, high in DELTA_T_BINS:
        if low is not None and delta_t_peak < low:
            continue
        if high is not None and delta_t_peak >= high:
            continue
        reason = None if name in {"low", "nominal", "hard"} else f"deltaT_peak_bin={name}"
        return name, reason
    return "reject_high", "deltaT_peak_bin=reject_high"


def _finite_ok(*arrays: np.ndarray) -> bool:
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _low_k_overlap_fraction(bundle: dict[str, Any]) -> float:
    q_active = bundle["q_field"].reshape(-1) > 0.0
    active_count = int(np.count_nonzero(q_active))
    if active_count == 0:
        return 0.0
    block_family = {
        block["block_id"]: block["k_family"]
        for block in bundle["scene"]["k"]["blocks"]
    }
    winners = bundle["sample_meta"]["k_node_metadata"]["winning_block_id"]
    low_k_count = 0
    for index, active in enumerate(q_active):
        if active and block_family.get(winners[index]) == "low_k_dielectric_underfill":
            low_k_count += 1
    return low_k_count / float(active_count)


def _triage_delta_t(sample: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    high_delta_t = float(sample["DeltaT_peak_K"]) > 15.0
    audit_passed = (
        sample["solver_status"] == "solved"
        and sample["nan_inf_ok"]
        and abs(float(sample["q_total_power_error_W"])) <= 1.0e-10
        and float(sample["q_power_on_boundary_W"]) == 0.0
        and int(sample["q_source_boundary_violation_count"]) == 0
        and int(sample["q_source_side_boundary_violation_count"]) == 0
        and int(sample["q_deposited_on_boundary_node_count"]) == 0
    )
    low_k_fraction = _low_k_overlap_fraction(bundle)
    hotspot_like = "hotspot" in sample["q_family"]
    if not high_delta_t:
        return {
            "high_deltaT_triage": "not_high_deltaT",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": None,
            "dataset_action": "keep_for_pilot",
        }
    if not audit_passed:
        return {
            "high_deltaT_triage": "reject_policy_or_solver_violation",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": None,
            "dataset_action": "reject",
        }
    if hotspot_like and low_k_fraction >= 0.5:
        return {
            "high_deltaT_triage": "physical_low_k_enclosed_compact_hotspot",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": "low_k_enclosed_compact_hotspot",
            "dataset_action": "keep_for_pilot",
        }
    return {
        "high_deltaT_triage": "audit_passed_high_deltaT_unclassified",
        "low_k_q_overlap_fraction": low_k_fraction,
        "physical_keep_reason": None,
        "dataset_action": "review_for_pilot",
    }


def _sample_audit(
    *,
    sample_id: str,
    bundle: dict[str, Any],
    temperature: np.ndarray,
    solve_meta: dict[str, Any],
) -> dict[str, Any]:
    scene = bundle["scene"]
    meta = bundle["sample_meta"]
    bottom_t = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
    delta_t = temperature.reshape(-1) - bottom_t
    delta_t_peak = float(np.max(delta_t))
    delta_t_p95 = float(np.percentile(delta_t, 95))
    delta_t_bin, reject_reason = _delta_t_bin(delta_t_peak)
    q_meta = meta["q_block_metadata"]
    q_total_realized = float(sum(block["realized_power_W"] for block in q_meta))
    q_integral_from_array = float(np.sum(bundle["q_field"].reshape(-1)) * _node_volume_m3(scene))
    q_power_audit = meta["q_power_audit"]
    solution_audit = solve_meta["solution_audit"]
    nan_inf_ok = _finite_ok(
        bundle["coords"],
        bundle["k_field"],
        bundle["q_field"],
        bundle["bc_features"],
        temperature,
    )
    sample = {
        "sample_id": sample_id,
        "scene_id": scene["scene_id"],
        "solver_status": solution_audit["status"],
        "residual_norm": solution_audit["residual_norm"],
        "energy_balance_residual": solution_audit["energy_balance_residual"],
        "bottom_dirichlet_error": solution_audit["bottom_dirichlet_error"],
        "DeltaT_peak_K": delta_t_peak,
        "DeltaT_p95_K": delta_t_p95,
        "DeltaT_bin": delta_t_bin,
        "q_total_target_power_W": q_power_audit["q_total_target_power_W"],
        "q_total_realized_power_W": q_total_realized,
        "q_integral_from_array_W": q_integral_from_array,
        "q_total_power_error_W": q_power_audit["q_total_power_error_W"],
        "q_power_on_bottom_W": q_power_audit["q_power_on_bottom_W"],
        "q_power_on_top_W": q_power_audit["q_power_on_top_W"],
        "q_power_on_xmin_W": q_power_audit["q_power_on_xmin_W"],
        "q_power_on_xmax_W": q_power_audit["q_power_on_xmax_W"],
        "q_power_on_ymin_W": q_power_audit["q_power_on_ymin_W"],
        "q_power_on_ymax_W": q_power_audit["q_power_on_ymax_W"],
        "q_power_on_side_W": q_power_audit["q_power_on_side_W"],
        "q_power_on_boundary_W": q_power_audit["q_power_on_boundary_W"],
        "q_power_on_bottom_fraction": q_power_audit["q_power_on_bottom_fraction"],
        "q_power_on_top_fraction": q_power_audit["q_power_on_top_fraction"],
        "q_power_on_side_fraction": q_power_audit["q_power_on_side_fraction"],
        "q_source_boundary_violation_count": q_power_audit["q_source_boundary_violation_count"],
        "q_source_side_boundary_violation_count": q_power_audit["q_source_side_boundary_violation_count"],
        "q_active_z_min": q_power_audit["q_active_z_min"],
        "q_active_z_max": q_power_audit["q_active_z_max"],
        "semantic_boundary_inset_fraction": q_power_audit["semantic_boundary_inset_fraction"],
        "semantic_inset_domain_xyz": q_power_audit["semantic_inset_domain_xyz"],
        "solver_safe_deposition_mask": q_power_audit["solver_safe_deposition_mask"],
        "q_deposited_on_boundary_node_count": q_power_audit["q_deposited_on_boundary_node_count"],
        "q_max_after_sum_W_m3": float(np.max(bundle["q_field"])),
        "background_k_family": meta["background_k"]["background_k_family"],
        "background_k_value": meta["background_k"]["background_k_value"],
        "material_block_count": len(scene["k"]["blocks"]),
        "k_mode": scene["k"]["mode"],
        "diag3_policy": scene["k"]["diag3_policy"],
        "q_family": scene["q"]["family"],
        "q_block_count": len(scene["q"]["blocks"]),
        "cooling_regime": scene["BC"]["cooling_regime"],
        "top_h_W_m2K": scene["BC"]["top_h_W_m2K"],
        "contact_model": meta["contact"]["contact_model"],
        "nan_inf_ok": nan_inf_ok,
        "reject_or_review_reason": reject_reason,
        "operator_checksum": solution_audit["operator_checksum"],
    }
    sample.update(_triage_delta_t(sample, bundle))
    return sample


def _summary(samples: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = len(samples)
    total = pass_count + len(failures)
    finite_energy = [
        abs(float(sample["energy_balance_residual"]))
        for sample in samples
        if np.isfinite(float(sample["energy_balance_residual"]))
    ]
    bottom_errors = [
        abs(float(sample["bottom_dirichlet_error"]))
        for sample in samples
        if np.isfinite(float(sample["bottom_dirichlet_error"]))
    ]
    q_boundary_power = [
        abs(float(sample["q_power_on_boundary_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_power_on_boundary_W"]))
    ]
    q_side_power = [
        abs(float(sample["q_power_on_side_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_power_on_side_W"]))
    ]
    q_power_errors = [
        abs(float(sample["q_total_power_error_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_total_power_error_W"]))
    ]
    dataset_actions = sorted({sample["dataset_action"] for sample in samples})
    high_triage = sorted({sample["high_deltaT_triage"] for sample in samples})
    return {
        "schema_version": "heat3d_v4_p3c_smoke16_audit_v2",
        "sample_count": total,
        "pass_count": pass_count,
        "failure_count": len(failures),
        "solver_pass_rate": pass_count / total if total else 0.0,
        "max_abs_energy_balance_residual": max(finite_energy) if finite_energy else None,
        "max_bottom_dirichlet_error": max(bottom_errors) if bottom_errors else None,
        "max_abs_q_total_power_error_W": max(q_power_errors) if q_power_errors else None,
        "max_q_power_on_boundary_W": max(q_boundary_power) if q_boundary_power else None,
        "max_q_power_on_side_W": max(q_side_power) if q_side_power else None,
        "q_source_boundary_violation_count": sum(
            int(sample["q_source_boundary_violation_count"]) for sample in samples
        ),
        "q_source_side_boundary_violation_count": sum(
            int(sample["q_source_side_boundary_violation_count"]) for sample in samples
        ),
        "q_deposited_on_boundary_node_count": sum(
            int(sample["q_deposited_on_boundary_node_count"]) for sample in samples
        ),
        "high_deltaT_count": sum(1 for sample in samples if float(sample["DeltaT_peak_K"]) > 15.0),
        "dataset_action_counts": {
            name: sum(1 for sample in samples if sample["dataset_action"] == name)
            for name in dataset_actions
        },
        "high_deltaT_triage_counts": {
            name: sum(1 for sample in samples if sample["high_deltaT_triage"] == name)
            for name in high_triage
        },
        "DeltaT_bin_counts": {
            name: sum(1 for sample in samples if sample["DeltaT_bin"] == name)
            for name, _, _ in DELTA_T_BINS
        },
        "k_mode_counts": {
            name: sum(1 for sample in samples if sample["k_mode"] == name)
            for name in sorted({sample["k_mode"] for sample in samples})
        },
        "diag3_policy_counts": {
            name: sum(1 for sample in samples if sample["diag3_policy"] == name)
            for name in sorted({sample["diag3_policy"] for sample in samples})
        },
        "q_family_counts": {
            name: sum(1 for sample in samples if sample["q_family"] == name)
            for name in sorted({sample["q_family"] for sample in samples})
        },
        "cooling_regime_counts": {
            name: sum(1 for sample in samples if sample["cooling_regime"] == name)
            for name in sorted({sample["cooling_regime"] for sample in samples})
        },
        "nan_inf_ok": all(sample["nan_inf_ok"] for sample in samples) and not failures,
        "samples": samples,
        "failures": failures,
    }


def generate_smoke16(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    dataset_dir: Path = REPO_ROOT / SMOKE16_DATASET_DIR,
    output_dir: Path = REPO_ROOT / SMOKE16_OUTPUT_DIR,
    sample_count: int = SMOKE16_SAMPLE_COUNT,
    seed: int = SMOKE16_SEED,
    force: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    _ensure_clean_target(dataset_dir, force=force)
    _ensure_clean_target(output_dir, force=force)
    dataset_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    write_plan = build_smoke16_write_plan(registry, sample_count=sample_count, seed=seed)
    batch = generate_dryrun_batch(registry, sample_count=sample_count, seed=seed)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest_samples = []
    solver_options = SolverOptions(solver_mode="perfect_contact", matrix_backend="sparse_csr")

    for index, scene in enumerate(batch["scenes"]):
        sample_id = f"sample_{index:03d}"
        sample_dir = dataset_dir / sample_id
        sample_dir.mkdir()
        bundle = materialize_scene_arrays(scene, registry)
        meta = dict(bundle["sample_meta"])
        meta.update(
            {
                "array_preflight_only": False,
                "artifact_writes": True,
                "solver_called": False,
                "dataset_id": dataset_dir.name,
                "sample_id": sample_id,
            }
        )
        np.save(sample_dir / "coords.npy", bundle["coords"])
        np.save(sample_dir / "k_field.npy", bundle["k_field"])
        np.save(sample_dir / "q_field.npy", bundle["q_field"])
        np.save(sample_dir / "bc_features.npy", bundle["bc_features"])

        try:
            problem = extract_problem_from_arrays(
                coords=bundle["coords"],
                k_field=bundle["k_field"],
                q_field=bundle["q_field"],
                sample_meta=meta,
                sample_dir=sample_dir,
            )
            temperature, solve_meta = solve_temperature_from_problem(problem, solver_options)
            meta["solver_called"] = True
            meta["label_solver"] = {
                "solver_family": solve_meta["solver_family"],
                "solver_mode": solve_meta["solver_mode"],
                "matrix_backend": solve_meta["matrix_backend"],
                "operator_checksum": solve_meta["solution_audit"]["operator_checksum"],
            }
            np.save(sample_dir / "temperature.npy", temperature)
            _write_json(sample_dir / "sample_meta.json", meta)
            sample_audit = _sample_audit(
                sample_id=sample_id,
                bundle={**bundle, "sample_meta": meta},
                temperature=temperature,
                solve_meta=solve_meta,
            )
            samples.append(sample_audit)
            manifest_samples.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": sample_id,
                    "files": [*PLANNED_SAMPLE_FILES, "temperature.npy"],
                    "DeltaT_bin": sample_audit["DeltaT_bin"],
                    "q_family": sample_audit["q_family"],
                    "cooling_regime": sample_audit["cooling_regime"],
                    "k_mode": sample_audit["k_mode"],
                    "diag3_policy": sample_audit["diag3_policy"],
                    "high_deltaT_triage": sample_audit["high_deltaT_triage"],
                    "physical_keep_reason": sample_audit["physical_keep_reason"],
                    "dataset_action": sample_audit["dataset_action"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            _write_json(sample_dir / "sample_meta.json", meta)
            failure = {
                "sample_id": sample_id,
                "scene_id": scene["scene_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "input_summary": {
                    "k_mode": scene["k"]["mode"],
                    "diag3_policy": scene["k"]["diag3_policy"],
                    "q_family": scene["q"]["family"],
                    "cooling_regime": scene["BC"]["cooling_regime"],
                    "top_h_W_m2K": scene["BC"]["top_h_W_m2K"],
                    "q_block_count": len(scene["q"]["blocks"]),
                },
            }
            failures.append(failure)
            break

    manifest = {
        "schema_version": "heat3d_v4_p3c_smoke16_manifest_v2",
        "dataset_id": dataset_dir.name,
        "sample_count_requested": sample_count,
        "sample_count_written": len(manifest_samples),
        "seed": seed,
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "write_plan": write_plan,
        "sample_schema": {
            "required_files": [*PLANNED_SAMPLE_FILES, "temperature.npy"],
        },
        "samples": manifest_samples,
    }
    audit = _summary(samples, failures)
    _write_json(dataset_dir / "manifest.json", manifest)
    _write_json(output_dir / "audit_summary.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / SMOKE16_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / SMOKE16_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=SMOKE16_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=SMOKE16_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    audit = generate_smoke16(
        registry_path=args.registry,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sample_count=args.samples,
        seed=args.seed,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset_dir),
                "output": str(args.output_dir),
                "sample_count": audit["sample_count"],
                "solver_pass_rate": audit["solver_pass_rate"],
                "failure_count": audit["failure_count"],
                "DeltaT_bin_counts": audit["DeltaT_bin_counts"],
                "max_abs_energy_balance_residual": audit["max_abs_energy_balance_residual"],
                "max_bottom_dirichlet_error": audit["max_bottom_dirichlet_error"],
                "max_q_power_on_boundary_W": audit["max_q_power_on_boundary_W"],
                "max_q_power_on_side_W": audit["max_q_power_on_side_W"],
                "q_source_boundary_violation_count": audit["q_source_boundary_violation_count"],
                "q_source_side_boundary_violation_count": audit["q_source_side_boundary_violation_count"],
                "q_deposited_on_boundary_node_count": audit["q_deposited_on_boundary_node_count"],
                "high_deltaT_count": audit["high_deltaT_count"],
                "dataset_action_counts": audit["dataset_action_counts"],
                "high_deltaT_triage_counts": audit["high_deltaT_triage_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
