#!/usr/bin/env python3
"""Verification smoke for the Heat3D v1 reference solver v2 minimal path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_reference_solver_v2 import solve_reference_temperature_v2  # noqa: E402


DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)
DEFAULT_SAMPLE_IDS = ("sample_000", "sample_005", "sample_008")
BOTTOM_TOL_K = 1e-6
ZERO_Q_TOL_K = 1e-8
BASELINE_SHIFT_K = 50.0
BASELINE_SHIFT_TOL_K = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reference solver v2 verification smoke on selected v1 samples."
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET,
        help="Subset root or samples directory containing sample_xxx directories.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Sample ids to verify.",
    )
    return parser.parse_args()


def _samples_root(path: Path) -> Path:
    if path.name == "samples":
        return path
    return path / "samples"


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "sample_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(sample_dir: Path, meta: dict) -> None:
    (sample_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _bottom_error_from_coords(sample_dir: Path, temperature: np.ndarray) -> float:
    coords = np.load(sample_dir / "coords.npy")
    meta = _load_meta(sample_dir)
    bottom_t = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
    bottom_mask = np.isclose(coords[:, 2], float(np.min(coords[:, 2])))
    return float(np.max(np.abs(temperature[bottom_mask, 0] - bottom_t)))


def _copy_sample_to_temp(sample_dir: Path, tmp_path: Path, suffix: str) -> Path:
    target = tmp_path / f"{sample_dir.name}_{suffix}"
    shutil.copytree(sample_dir, target)
    return target


def _t_ref_from_meta(sample_dir: Path) -> float:
    meta = _load_meta(sample_dir)
    return float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])


def _set_q_zero(sample_dir: Path) -> None:
    q_path = sample_dir / "q_field.npy"
    q = np.load(q_path)
    np.save(q_path, np.zeros_like(q))


def _shift_baseline(sample_dir: Path, shift_k: float) -> None:
    meta = _load_meta(sample_dir)
    meta["boundary_params"]["bottom"]["fixed_temperature_K"] = (
        float(meta["boundary_params"]["bottom"]["fixed_temperature_K"]) + shift_k
    )
    meta["boundary_params"]["top"]["ambient_temperature_K"] = (
        float(meta["boundary_params"]["top"]["ambient_temperature_K"]) + shift_k
    )
    _write_meta(sample_dir, meta)


def _basic_solver_checks(sample_dir: Path, temperature: np.ndarray, label_meta: dict) -> tuple[bool, dict]:
    finite_temperature = bool(np.all(np.isfinite(temperature)))
    bottom_error = _bottom_error_from_coords(sample_dir, temperature)
    ok = (
        temperature.ndim == 2
        and temperature.shape[1] == 1
        and finite_temperature
        and bool(label_meta["convergence_flag"])
        and np.isfinite(float(label_meta["residual_norm"]))
        and bottom_error <= BOTTOM_TOL_K
    )
    return ok, {
        "shape": list(temperature.shape),
        "finite": finite_temperature,
        "T_min": float(np.min(temperature)),
        "T_max": float(np.max(temperature)),
        "T_mean": float(np.mean(temperature)),
        "supported_k_mode": label_meta["supported_k_mode"],
        "convergence_flag": bool(label_meta["convergence_flag"]),
        "residual_norm": float(label_meta["residual_norm"]),
        "bottom_error": bottom_error,
        "warnings": label_meta["warnings"],
    }


def _print_basic_summary(sample_id: str, summary: dict, sample_ok: bool) -> None:
    print(f"\n{sample_id}")
    print(f"  temperature shape: {tuple(summary['shape'])}")
    print(f"  finite: {summary['finite']}")
    print(
        "  T min/max/mean: "
        f"{summary['T_min']:.6f}, {summary['T_max']:.6f}, {summary['T_mean']:.6f}"
    )
    print(f"  supported_k_mode: {summary['supported_k_mode']}")
    print(f"  convergence_flag: {summary['convergence_flag']}")
    print(f"  residual_norm: {summary['residual_norm']:.6e}")
    print(f"  bottom_dirichlet_error: {summary['bottom_error']:.6e}")
    print(f"  warnings: {summary['warnings']}")
    print(f"  sample_ok: {sample_ok}")


def _run_regular_case(sample_dir: Path) -> tuple[bool, dict]:
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    sample_ok, summary = _basic_solver_checks(sample_dir, temperature, label_meta)
    summary.update({
        "case": "regular_sample",
        "top_robin_status": label_meta["top_robin_status"]["status"],
        "side_adiabatic_status": label_meta["side_adiabatic_status"]["status"],
        "interface_status": label_meta["interface_status"]["status"],
        "energy_balance_status": label_meta["energy_balance_status"]["status"],
    })
    _print_basic_summary(sample_dir.name, summary, sample_ok)
    print(f"  top_robin_status: {summary['top_robin_status']}")
    print(f"  side_adiabatic_status: {summary['side_adiabatic_status']}")
    print(f"  interface_status: {summary['interface_status']}")
    print(f"  energy_balance_status: {summary['energy_balance_status']}")
    return sample_ok, summary


def _run_zero_q_case(source_sample_dir: Path, tmp_path: Path) -> tuple[bool, dict]:
    sample_dir = _copy_sample_to_temp(source_sample_dir, tmp_path, "zero_q")
    _set_q_zero(sample_dir)
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    sample_ok, summary = _basic_solver_checks(sample_dir, temperature, label_meta)
    t_ref = _t_ref_from_meta(sample_dir)
    delta = temperature - t_ref
    max_abs_delta = float(np.max(np.abs(delta)))
    zero_q_ok = sample_ok and max_abs_delta <= ZERO_Q_TOL_K
    summary.update({
        "case": "zero_q",
        "T_ref": t_ref,
        "max_abs_deltaT": max_abs_delta,
        "zero_q_tol_K": ZERO_Q_TOL_K,
        "case_ok": zero_q_ok,
    })
    print(f"\nzero_q case from {source_sample_dir.name}")
    print(f"  T_ref: {t_ref:.6f}")
    print(f"  max_abs_deltaT: {max_abs_delta:.6e}")
    print(f"  convergence_flag: {label_meta['convergence_flag']}")
    print(f"  residual_norm: {label_meta['residual_norm']:.6e}")
    print(f"  bottom_dirichlet_error: {summary['bottom_error']:.6e}")
    print(f"  zero_q_case_ok: {zero_q_ok}")
    return zero_q_ok, summary


def _run_baseline_shift_case(source_sample_dir: Path, tmp_path: Path) -> tuple[bool, dict]:
    base_dir = _copy_sample_to_temp(source_sample_dir, tmp_path, "baseline_base")
    shifted_dir = _copy_sample_to_temp(source_sample_dir, tmp_path, "baseline_shifted")
    _shift_baseline(shifted_dir, BASELINE_SHIFT_K)

    base_temperature, base_label_meta = solve_reference_temperature_v2(base_dir)
    shifted_temperature, shifted_label_meta = solve_reference_temperature_v2(shifted_dir)
    base_ok, base_summary = _basic_solver_checks(base_dir, base_temperature, base_label_meta)
    shifted_ok, shifted_summary = _basic_solver_checks(shifted_dir, shifted_temperature, shifted_label_meta)

    base_t_ref = _t_ref_from_meta(base_dir)
    shifted_t_ref = _t_ref_from_meta(shifted_dir)
    shift_error = float(np.max(np.abs((shifted_temperature - base_temperature) - BASELINE_SHIFT_K)))
    delta_error = float(np.max(np.abs((shifted_temperature - shifted_t_ref) - (base_temperature - base_t_ref))))
    case_ok = base_ok and shifted_ok and shift_error <= BASELINE_SHIFT_TOL_K and delta_error <= BASELINE_SHIFT_TOL_K
    summary = {
        "case": "baseline_shift",
        "base_T_ref": base_t_ref,
        "shifted_T_ref": shifted_t_ref,
        "expected_shift_K": BASELINE_SHIFT_K,
        "max_abs_shift_error_K": shift_error,
        "max_abs_deltaT_error_K": delta_error,
        "base_residual_norm": base_summary["residual_norm"],
        "shifted_residual_norm": shifted_summary["residual_norm"],
        "case_ok": case_ok,
    }
    print(f"\nbaseline_shift case from {source_sample_dir.name}")
    print(f"  base_T_ref: {base_t_ref:.6f}")
    print(f"  shifted_T_ref: {shifted_t_ref:.6f}")
    print(f"  expected_shift_K: {BASELINE_SHIFT_K:.6f}")
    print(f"  max_abs_shift_error_K: {shift_error:.6e}")
    print(f"  max_abs_deltaT_error_K: {delta_error:.6e}")
    print(f"  base_residual_norm: {base_summary['residual_norm']:.6e}")
    print(f"  shifted_residual_norm: {shifted_summary['residual_norm']:.6e}")
    print(f"  baseline_shift_case_ok: {case_ok}")
    return case_ok, summary


def main() -> int:
    args = parse_args()
    root = _samples_root(args.subset)
    if not root.is_dir():
        print(f"ERROR: sample root does not exist: {root}")
        print("No data were generated. Restore ignored v1 supervised small data before running this smoke.")
        return 1

    failed = False
    summaries: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_solver_v2_") as tmp:
        tmp_path = Path(tmp)
        for sample_id in args.sample_ids:
            sample_dir = root / sample_id
            if not sample_dir.is_dir():
                print(f"ERROR: missing sample directory: {sample_dir}")
                failed = True
                continue

            sample_ok, summary = _run_regular_case(sample_dir)
            failed = failed or not sample_ok
            summary["sample_id"] = sample_id
            summary["sample_ok"] = sample_ok
            summaries.append(summary)

        zero_q_ok, zero_q_summary = _run_zero_q_case(root / "sample_000", tmp_path)
        baseline_shift_ok, baseline_shift_summary = _run_baseline_shift_case(root / "sample_000", tmp_path)
        failed = failed or not zero_q_ok or not baseline_shift_ok
        summaries.extend([zero_q_summary, baseline_shift_summary])

        print("\nsummary")
        print(f"  subset: {args.subset}")
        print(f"  sample_ids: {list(args.sample_ids)}")
        print(f"  regular_sample_count: {len(args.sample_ids)}")
        print(f"  zero_q_case_ok: {zero_q_ok}")
        print(f"  baseline_shift_case_ok: {baseline_shift_ok}")
        print(
            "  diagonal_anisotropic_case_present: "
            f"{any(summary.get('supported_k_mode') == 'diag3' for summary in summaries)}"
        )
        print(f"  temporary_output_dir: {tmp_path}")
        print("  temporary outputs are removed after this smoke")
        print(f"  all_cases_ok: {not failed}")
        print("  scope: minimal research reference path; not high-fidelity validation")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
