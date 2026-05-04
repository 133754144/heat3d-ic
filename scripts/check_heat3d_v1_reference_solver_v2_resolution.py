#!/usr/bin/env python3
"""Resolution / node-count smoke for Heat3D v1 reference solver v2.

This script creates temporary controlled rectilinear samples and runs the v2
research reference solver on multiple node counts. It is a diagnostic smoke
only, not a grid-convergence study and not high-fidelity validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_reference_solver_v2 import BOTTOM_TOL_K, RESIDUAL_TOL, solve_reference_temperature_v2  # noqa: E402


RESOLUTION_CASES = {
    "coarse": (4, 4, 4),
    "mid": (6, 6, 5),
    "fine": (8, 8, 6),
}
K_MODES = ("isotropic", "diag3")
DELTA_T_UPPER_BOUND_K = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run node-count smoke diagnostics for Heat3D v1 reference solver v2. "
            "No formal grid-convergence claim is made."
        )
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["coarse", "mid", "fine"],
        choices=sorted(RESOLUTION_CASES),
        help="Resolution labels to run.",
    )
    parser.add_argument(
        "--k-modes",
        nargs="*",
        default=list(K_MODES),
        choices=list(K_MODES),
        help="Conductivity modes to test.",
    )
    return parser.parse_args()


def _make_coords(grid_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid_shape
    xs = np.linspace(0.0, 0.01, nx)
    ys = np.linspace(0.0, 0.01, ny)
    zs = np.linspace(0.0, 0.002, nz)
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    return coords


def _make_k_field(coords: np.ndarray, k_mode: str) -> np.ndarray:
    n = coords.shape[0]
    if k_mode == "isotropic":
        return np.full((n, 1), 12.0, dtype=np.float64)
    if k_mode == "diag3":
        k = np.empty((n, 3), dtype=np.float64)
        k[:, 0] = 14.0
        k[:, 1] = 10.0
        k[:, 2] = 6.0
        return k
    raise ValueError(f"unsupported k_mode: {k_mode}")


def _make_q_field(coords: np.ndarray) -> np.ndarray:
    q = np.zeros((coords.shape[0], 1), dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    active = (
        (x >= 0.003)
        & (x <= 0.007)
        & (y >= 0.003)
        & (y <= 0.007)
        & (z >= 0.0004)
        & (z <= 0.0016)
    )
    if not np.any(active):
        x_mid = np.isclose(x, x[np.argmin(np.abs(x - 0.005))])
        y_mid = np.isclose(y, y[np.argmin(np.abs(y - 0.005))])
        z_mid = np.isclose(z, z[np.argmin(np.abs(z - 0.001))])
        active = x_mid & y_mid & z_mid
    q[active, 0] = 1.0e8
    return q


def _make_meta(grid_shape: tuple[int, int, int], k_mode: str, resolution: str) -> dict[str, Any]:
    return {
        "schema_version": "solver_v2_resolution_smoke",
        "subset_name": "temporary_reference_solver_v2_resolution_smoke",
        "sample_id": f"{k_mode}_{resolution}",
        "stage": "temporary_solver_verification_smoke",
        "split": "diagnostic_only",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": 1000.0, "ambient_temperature_K": 300.0},
            "bottom": {"fixed_temperature_K": 300.0},
            "sides": {"adiabatic": True},
        },
        "interfaces": [{"type": "perfect_contact", "note": "single rectilinear stack smoke abstraction"}],
        "generation_config": {
            "resolution_label": resolution,
            "grid_shape": list(grid_shape),
            "k_mode": k_mode,
            "temporary_sample": True,
            "not_formal_dataset": True,
            "not_grid_convergence_study": True,
        },
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _write_sample(root: Path, resolution: str, k_mode: str) -> Path:
    grid_shape = RESOLUTION_CASES[resolution]
    sample_dir = root / f"{k_mode}_{resolution}"
    sample_dir.mkdir(parents=True)
    coords = _make_coords(grid_shape)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "k_field.npy", _make_k_field(coords, k_mode))
    np.save(sample_dir / "q_field.npy", _make_q_field(coords))
    (sample_dir / "sample_meta.json").write_text(
        json.dumps(_make_meta(grid_shape, k_mode, resolution), indent=2) + "\n",
        encoding="utf-8",
    )
    return sample_dir


def _peak_coord(coords: np.ndarray, temperature: np.ndarray) -> list[float]:
    peak_index = int(np.argmax(temperature.reshape(-1)))
    return [float(value) for value in coords[peak_index]]


def _run_case(sample_dir: Path, resolution: str, k_mode: str) -> tuple[bool, dict[str, Any]]:
    coords = np.load(sample_dir / "coords.npy")
    t_ref = 300.0
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    delta_t = temperature - t_ref
    finite = bool(np.all(np.isfinite(temperature)))
    residual_norm = float(label_meta["residual_norm"])
    bottom_error = float(label_meta["bottom_dirichlet_error"])
    delta_min = float(np.min(delta_t))
    delta_max = float(np.max(delta_t))
    case_ok = (
        finite
        and bool(label_meta["convergence_flag"])
        and residual_norm <= RESIDUAL_TOL
        and bottom_error <= BOTTOM_TOL_K
        and delta_min >= -1.0e-8
        and delta_max <= DELTA_T_UPPER_BOUND_K
    )
    summary = {
        "resolution": resolution,
        "k_mode": k_mode,
        "grid_shape": label_meta["assembly"]["grid_shape"],
        "node_count": int(label_meta["assembly"]["node_count"]),
        "T_min": float(np.min(temperature)),
        "T_max": float(np.max(temperature)),
        "T_mean": float(np.mean(temperature)),
        "DeltaT_min": delta_min,
        "DeltaT_max": delta_max,
        "DeltaT_mean": float(np.mean(delta_t)),
        "peak_T": float(np.max(temperature)),
        "peak_coord": _peak_coord(coords, temperature),
        "residual_norm": residual_norm,
        "convergence_flag": bool(label_meta["convergence_flag"]),
        "bottom_dirichlet_error": bottom_error,
        "warnings": label_meta["warnings"],
        "case_ok": case_ok,
    }
    return case_ok, summary


def _print_case(summary: dict[str, Any]) -> None:
    print(
        f"{summary['k_mode']} {summary['resolution']}: "
        f"grid_shape={summary['grid_shape']} "
        f"node_count={summary['node_count']} "
        f"T_range=[{summary['T_min']:.6f}, {summary['T_max']:.6f}] "
        f"DeltaT_range=[{summary['DeltaT_min']:.6e}, {summary['DeltaT_max']:.6e}] "
        f"peak_T={summary['peak_T']:.6f} "
        f"peak_coord={summary['peak_coord']} "
        f"residual_norm={summary['residual_norm']:.6e} "
        f"bottom_error={summary['bottom_dirichlet_error']:.6e} "
        f"converged={summary['convergence_flag']} "
        f"warnings={summary['warnings']} "
        f"case_ok={summary['case_ok']}"
    )


def _node_counts_increase(summaries: list[dict[str, Any]], resolutions: list[str]) -> bool:
    by_mode: dict[str, list[int]] = {}
    order = {resolution: idx for idx, resolution in enumerate(resolutions)}
    for summary in summaries:
        by_mode.setdefault(summary["k_mode"], []).append(
            (order[summary["resolution"]], summary["node_count"])
        )
    for values in by_mode.values():
        counts = [count for _, count in sorted(values)]
        if any(later <= earlier for earlier, later in zip(counts, counts[1:])):
            return False
    return True


def main() -> int:
    args = parse_args()
    print("Heat3D v1 reference solver v2 resolution smoke")
    print("scope: node-count smoke / research reference diagnostics only; not grid convergence")
    print(f"resolutions: {args.resolutions}")
    print(f"k_modes: {args.k_modes}")

    summaries: list[dict[str, Any]] = []
    failed = False
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_solver_v2_resolution_") as tmp:
        root = Path(tmp)
        for k_mode in args.k_modes:
            for resolution in args.resolutions:
                sample_dir = _write_sample(root, resolution=resolution, k_mode=k_mode)
                case_ok, summary = _run_case(sample_dir, resolution=resolution, k_mode=k_mode)
                failed = failed or not case_ok
                summaries.append(summary)
                _print_case(summary)

        node_count_ok = _node_counts_increase(summaries, args.resolutions)
        failed = failed or not node_count_ok
        print("\nsummary")
        print(f"  temporary_sample_root: {root}")
        print("  temporary samples are removed after this smoke")
        print(f"  case_count: {len(summaries)}")
        print(f"  node_count_increases_with_resolution: {node_count_ok}")
        print(f"  residual_tolerance: {RESIDUAL_TOL}")
        print(f"  bottom_tolerance_K: {BOTTOM_TOL_K}")
        print(f"  deltaT_upper_bound_K: {DELTA_T_UPPER_BOUND_K}")
        print("  flux_energy_pde_diagnostics: not_computed / requires_numerical_operator")
        print(f"  resolution_smoke_ok: {not failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
