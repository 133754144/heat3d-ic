#!/usr/bin/env python3
"""Source-power consistency smoke for Heat3D v1 reference solver v2.

This script builds temporary controlled samples and checks that source power is
normalized consistently across node counts. It is a source-power consistency
smoke only; it is not a formal energy-balance check or grid-convergence study.
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


DOMAIN_BOUNDS = {
    "x": (0.0, 0.01),
    "y": (0.0, 0.01),
    "z": (0.0, 0.002),
}
SOURCE_BOX = {
    "x": (0.003, 0.007),
    "y": (0.003, 0.007),
    "z": (0.0005, 0.0015),
}
RESOLUTION_CASES = {
    "coarse": (4, 4, 4),
    "mid": (6, 6, 5),
    "fine": (8, 8, 6),
}
K_MODES = ("isotropic", "diag3")
Q_AMPLITUDE_W_M3 = 1.0e8
TOP_H_W_M2K = 1000.0
T_REF_K = 300.0
POWER_REL_TOL = 1.0e-10
DELTA_T_UPPER_BOUND_K = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run source-power consistency smoke for Heat3D v1 reference solver v2. "
            "No formal energy-balance or grid-convergence claim is made."
        )
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["coarse", "mid", "fine"],
        choices=sorted(RESOLUTION_CASES),
    )
    parser.add_argument(
        "--k-modes",
        nargs="*",
        default=list(K_MODES),
        choices=list(K_MODES),
    )
    return parser.parse_args()


def _axis(bounds: tuple[float, float], count: int) -> np.ndarray:
    return np.linspace(bounds[0], bounds[1], count, dtype=np.float64)


def _cell_bounds(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty_like(axis, dtype=np.float64)
    upper = np.empty_like(axis, dtype=np.float64)
    lower[0] = axis[0]
    upper[-1] = axis[-1]
    if axis.size == 1:
        upper[0] = axis[0] + 1.0
        return lower, upper
    mids = 0.5 * (axis[:-1] + axis[1:])
    upper[:-1] = mids
    lower[1:] = mids
    return lower, upper


def _control_widths(axis: np.ndarray) -> np.ndarray:
    lower, upper = _cell_bounds(axis)
    return upper - lower


def _overlap_lengths(axis: np.ndarray, source_bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = _cell_bounds(axis)
    src_min, src_max = source_bounds
    return np.maximum(0.0, np.minimum(upper, src_max) - np.maximum(lower, src_min))


def _make_grid(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xs = _axis(DOMAIN_BOUNDS["x"], grid_shape[0])
    ys = _axis(DOMAIN_BOUNDS["y"], grid_shape[1])
    zs = _axis(DOMAIN_BOUNDS["z"], grid_shape[2])
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    axes = {"x": xs, "y": ys, "z": zs}
    return coords, axes


def _cell_volumes(axes: dict[str, np.ndarray]) -> np.ndarray:
    wx = _control_widths(axes["x"])
    wy = _control_widths(axes["y"])
    wz = _control_widths(axes["z"])
    return np.array([dx * dy * dz for dx in wx for dy in wy for dz in wz], dtype=np.float64).reshape(-1, 1)


def _source_overlap_volumes(axes: dict[str, np.ndarray]) -> np.ndarray:
    ox = _overlap_lengths(axes["x"], SOURCE_BOX["x"])
    oy = _overlap_lengths(axes["y"], SOURCE_BOX["y"])
    oz = _overlap_lengths(axes["z"], SOURCE_BOX["z"])
    return np.array([dx * dy * dz for dx in ox for dy in oy for dz in oz], dtype=np.float64).reshape(-1, 1)


def _make_q_field(cell_volumes: np.ndarray, overlap_volumes: np.ndarray) -> np.ndarray:
    fractions = np.divide(
        overlap_volumes,
        cell_volumes,
        out=np.zeros_like(overlap_volumes),
        where=cell_volumes > 0.0,
    )
    return Q_AMPLITUDE_W_M3 * fractions


def _make_k_field(coords: np.ndarray, k_mode: str) -> np.ndarray:
    if k_mode == "isotropic":
        return np.full((coords.shape[0], 1), 12.0, dtype=np.float64)
    if k_mode == "diag3":
        k = np.empty((coords.shape[0], 3), dtype=np.float64)
        k[:, 0] = 14.0
        k[:, 1] = 10.0
        k[:, 2] = 6.0
        return k
    raise ValueError(f"unsupported k_mode: {k_mode}")


def _make_meta(grid_shape: tuple[int, int, int], k_mode: str, resolution: str) -> dict[str, Any]:
    return {
        "schema_version": "solver_v2_source_power_smoke",
        "subset_name": "temporary_reference_solver_v2_source_power_smoke",
        "sample_id": f"{k_mode}_{resolution}_source_power",
        "stage": "temporary_solver_source_power_smoke",
        "split": "diagnostic_only",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": TOP_H_W_M2K, "ambient_temperature_K": T_REF_K},
            "bottom": {"fixed_temperature_K": T_REF_K},
            "sides": {"adiabatic": True},
        },
        "interfaces": [{"type": "perfect_contact", "note": "single rectilinear stack smoke abstraction"}],
        "generation_config": {
            "resolution_label": resolution,
            "grid_shape": list(grid_shape),
            "k_mode": k_mode,
            "source_projection": "control_volume_overlap_fraction",
            "source_box_m": SOURCE_BOX,
            "q_amplitude_W_m3": Q_AMPLITUDE_W_M3,
            "temporary_sample": True,
            "not_formal_dataset": True,
            "not_formal_energy_balance": True,
            "not_grid_convergence_study": True,
        },
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _write_sample(root: Path, resolution: str, k_mode: str) -> tuple[Path, dict[str, float]]:
    grid_shape = RESOLUTION_CASES[resolution]
    coords, axes = _make_grid(grid_shape)
    cell_volumes = _cell_volumes(axes)
    overlap_volumes = _source_overlap_volumes(axes)
    q_field = _make_q_field(cell_volumes, overlap_volumes)
    sample_dir = root / f"{k_mode}_{resolution}_source_power"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "k_field.npy", _make_k_field(coords, k_mode))
    np.save(sample_dir / "q_field.npy", q_field)
    (sample_dir / "sample_meta.json").write_text(
        json.dumps(_make_meta(grid_shape, k_mode, resolution), indent=2) + "\n",
        encoding="utf-8",
    )
    active_volume = float(np.sum(overlap_volumes))
    integrated_power = float(np.sum(q_field * cell_volumes))
    active_nodes = int(np.count_nonzero(q_field[:, 0] > 0.0))
    expected_volume = (
        (SOURCE_BOX["x"][1] - SOURCE_BOX["x"][0])
        * (SOURCE_BOX["y"][1] - SOURCE_BOX["y"][0])
        * (SOURCE_BOX["z"][1] - SOURCE_BOX["z"][0])
    )
    return sample_dir, {
        "active_source_node_count": active_nodes,
        "active_source_physical_volume_proxy": active_volume,
        "expected_source_volume": expected_volume,
        "integrated_q_power": integrated_power,
        "expected_integrated_q_power": Q_AMPLITUDE_W_M3 * expected_volume,
    }


def _top_robin_heat_removal_proxy(coords: np.ndarray, temperature: np.ndarray) -> float:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    z_max = float(np.max(coords[:, 2]))
    wx = _control_widths(xs)
    wy = _control_widths(ys)
    area_by_xy = {(float(x), float(y)): float(dx * dy) for x, dx in zip(xs, wx) for y, dy in zip(ys, wy)}
    total = 0.0
    for point, temp in zip(coords, temperature[:, 0]):
        if np.isclose(point[2], z_max):
            total += TOP_H_W_M2K * (float(temp) - T_REF_K) * area_by_xy[(float(point[0]), float(point[1]))]
    return float(total)


def _peak_coord(coords: np.ndarray, temperature: np.ndarray) -> list[float]:
    peak_index = int(np.argmax(temperature.reshape(-1)))
    return [float(value) for value in coords[peak_index]]


def _run_case(sample_dir: Path, source_summary: dict[str, float], resolution: str, k_mode: str) -> tuple[bool, dict[str, Any]]:
    coords = np.load(sample_dir / "coords.npy")
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    delta_t = temperature - T_REF_K
    top_proxy = _top_robin_heat_removal_proxy(coords, temperature)
    residual_norm = float(label_meta["residual_norm"])
    bottom_error = float(label_meta["bottom_dirichlet_error"])
    finite = bool(np.all(np.isfinite(temperature)))
    delta_min = float(np.min(delta_t))
    delta_max = float(np.max(delta_t))
    case_ok = (
        finite
        and bool(label_meta["convergence_flag"])
        and residual_norm <= RESIDUAL_TOL
        and bottom_error <= BOTTOM_TOL_K
        and delta_min >= -1.0e-8
        and delta_max <= DELTA_T_UPPER_BOUND_K
        and np.isfinite(top_proxy)
    )
    summary: dict[str, Any] = {
        "resolution": resolution,
        "k_mode": k_mode,
        "grid_shape": label_meta["assembly"]["grid_shape"],
        "node_count": int(label_meta["assembly"]["node_count"]),
        "q_amplitude_W_m3": Q_AMPLITUDE_W_M3,
        "active_source_node_count": source_summary["active_source_node_count"],
        "active_source_physical_volume_proxy": source_summary["active_source_physical_volume_proxy"],
        "expected_source_volume": source_summary["expected_source_volume"],
        "integrated_q_power": source_summary["integrated_q_power"],
        "expected_integrated_q_power": source_summary["expected_integrated_q_power"],
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
        "top_robin_heat_removal_proxy_W": top_proxy,
        "warnings": label_meta["warnings"],
        "case_ok": case_ok,
    }
    return case_ok, summary


def _print_case(summary: dict[str, Any]) -> None:
    print(
        f"{summary['k_mode']} {summary['resolution']}: "
        f"grid_shape={summary['grid_shape']} "
        f"node_count={summary['node_count']} "
        f"q_amplitude={summary['q_amplitude_W_m3']:.6e} "
        f"active_nodes={summary['active_source_node_count']} "
        f"active_volume={summary['active_source_physical_volume_proxy']:.6e} "
        f"integrated_q_power={summary['integrated_q_power']:.6e} "
        f"T_range=[{summary['T_min']:.6f}, {summary['T_max']:.6f}] "
        f"DeltaT_range=[{summary['DeltaT_min']:.6e}, {summary['DeltaT_max']:.6e}] "
        f"peak_T={summary['peak_T']:.6f} "
        f"peak_coord={summary['peak_coord']} "
        f"residual_norm={summary['residual_norm']:.6e} "
        f"bottom_error={summary['bottom_dirichlet_error']:.6e} "
        f"top_robin_proxy_W={summary['top_robin_heat_removal_proxy_W']:.6e} "
        f"converged={summary['convergence_flag']} "
        f"warnings={summary['warnings']} "
        f"case_ok={summary['case_ok']}"
    )


def _node_counts_increase(summaries: list[dict[str, Any]], resolutions: list[str]) -> bool:
    order = {resolution: idx for idx, resolution in enumerate(resolutions)}
    by_mode: dict[str, list[tuple[int, int]]] = {}
    for summary in summaries:
        by_mode.setdefault(summary["k_mode"], []).append((order[summary["resolution"]], summary["node_count"]))
    for values in by_mode.values():
        counts = [count for _, count in sorted(values)]
        if any(later <= earlier for earlier, later in zip(counts, counts[1:])):
            return False
    return True


def _power_consistency_ok(summaries: list[dict[str, Any]]) -> tuple[bool, dict[str, float]]:
    by_mode: dict[str, list[float]] = {}
    for summary in summaries:
        by_mode.setdefault(summary["k_mode"], []).append(float(summary["integrated_q_power"]))
    max_rel_by_mode = {}
    ok = True
    for mode, powers in by_mode.items():
        ref = max(abs(powers[0]), 1.0e-30)
        max_rel = max(abs(power - powers[0]) / ref for power in powers)
        max_rel_by_mode[mode] = float(max_rel)
        ok = ok and max_rel <= POWER_REL_TOL
    return ok, max_rel_by_mode


def _source_volume_ok(summaries: list[dict[str, Any]]) -> tuple[bool, dict[str, float]]:
    by_mode: dict[str, list[float]] = {}
    for summary in summaries:
        by_mode.setdefault(summary["k_mode"], []).append(float(summary["active_source_physical_volume_proxy"]))
    max_rel_by_mode = {}
    ok = True
    for mode, volumes in by_mode.items():
        ref = max(abs(volumes[0]), 1.0e-30)
        max_rel = max(abs(volume - volumes[0]) / ref for volume in volumes)
        max_rel_by_mode[mode] = float(max_rel)
        ok = ok and max_rel <= POWER_REL_TOL
    return ok, max_rel_by_mode


def main() -> int:
    args = parse_args()
    print("Heat3D v1 reference solver v2 source-power consistency smoke")
    print("scope: source-power consistency / resolution diagnostic only; not formal energy balance")
    print(f"resolutions: {args.resolutions}")
    print(f"k_modes: {args.k_modes}")
    print(f"source_box_m: {SOURCE_BOX}")
    print(f"q_amplitude_W_m3: {Q_AMPLITUDE_W_M3:.6e}")

    summaries: list[dict[str, Any]] = []
    failed = False
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_solver_v2_source_power_") as tmp:
        root = Path(tmp)
        for k_mode in args.k_modes:
            for resolution in args.resolutions:
                sample_dir, source_summary = _write_sample(root, resolution=resolution, k_mode=k_mode)
                case_ok, summary = _run_case(
                    sample_dir,
                    source_summary=source_summary,
                    resolution=resolution,
                    k_mode=k_mode,
                )
                failed = failed or not case_ok
                summaries.append(summary)
                _print_case(summary)

        node_count_ok = _node_counts_increase(summaries, args.resolutions)
        power_ok, power_rel_by_mode = _power_consistency_ok(summaries)
        volume_ok, volume_rel_by_mode = _source_volume_ok(summaries)
        failed = failed or not node_count_ok or not power_ok or not volume_ok

        print("\nsummary")
        print(f"  temporary_sample_root: {root}")
        print("  temporary samples are removed after this smoke")
        print(f"  case_count: {len(summaries)}")
        print(f"  node_count_increases_with_resolution: {node_count_ok}")
        print(f"  source_volume_consistency_ok: {volume_ok}")
        print(f"  source_volume_max_rel_diff_by_mode: {volume_rel_by_mode}")
        print(f"  integrated_q_power_consistency_ok: {power_ok}")
        print(f"  integrated_q_power_max_rel_diff_by_mode: {power_rel_by_mode}")
        print(f"  power_relative_tolerance: {POWER_REL_TOL}")
        print(f"  residual_tolerance: {RESIDUAL_TOL}")
        print(f"  bottom_tolerance_K: {BOTTOM_TOL_K}")
        print("  top_robin_heat_removal_proxy: computed")
        print("  bottom_flux_global_energy_balance: not_computed / requires_numerical_operator")
        print(f"  source_power_smoke_ok: {not failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
