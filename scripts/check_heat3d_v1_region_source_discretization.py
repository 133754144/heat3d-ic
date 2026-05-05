#!/usr/bin/env python3
"""Region-source discretization smoke for Heat3D v1 reference solver v2.

This script compares point/center source assignment with volume-fraction
assignment on temporary controlled samples. It is a source-assignment
diagnostic only; it is not a formal grid-convergence study, not a benchmark,
and not model-performance evidence.
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
# Deliberately not aligned with the coarse grid node locations. This exposes
# the risk that point/center assignment can miss a physically present source.
SOURCE_BOX = {
    "x": (0.0039, 0.0061),
    "y": (0.0039, 0.0061),
    "z": (0.00075, 0.00125),
}
RESOLUTION_CASES = {
    "coarse": (4, 4, 4),
    "mid": (6, 6, 5),
    "fine": (8, 8, 6),
}
ASSIGNMENT_METHODS = ("center_in_box", "volume_fraction")
Q_DENSITY_W_M3 = 1.0e8
TOP_H_W_M2K = 1000.0
T_REF_K = 300.0
POWER_REL_TOL = 1.0e-10
DELTA_T_UPPER_BOUND_K = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run region-first source discretization smoke for Heat3D v1 solver v2. "
            "No formal grid-convergence or benchmark claim is made."
        )
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["coarse", "mid", "fine"],
        choices=sorted(RESOLUTION_CASES),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(ASSIGNMENT_METHODS),
        choices=list(ASSIGNMENT_METHODS),
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
    return coords, {"x": xs, "y": ys, "z": zs}


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


def _target_source_volume() -> float:
    return float(
        (SOURCE_BOX["x"][1] - SOURCE_BOX["x"][0])
        * (SOURCE_BOX["y"][1] - SOURCE_BOX["y"][0])
        * (SOURCE_BOX["z"][1] - SOURCE_BOX["z"][0])
    )


def _make_q_field(
    coords: np.ndarray,
    axes: dict[str, np.ndarray],
    cell_volumes: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_volume = _target_source_volume()
    target_power = Q_DENSITY_W_M3 * target_volume
    q = np.zeros((coords.shape[0], 1), dtype=np.float64)

    if method == "center_in_box":
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        active = (
            (x >= SOURCE_BOX["x"][0])
            & (x <= SOURCE_BOX["x"][1])
            & (y >= SOURCE_BOX["y"][0])
            & (y <= SOURCE_BOX["y"][1])
            & (z >= SOURCE_BOX["z"][0])
            & (z <= SOURCE_BOX["z"][1])
        )
        q[active, 0] = Q_DENSITY_W_M3
        active_volume = float(np.sum(cell_volumes[active, 0]))
    elif method == "volume_fraction":
        overlap_volumes = _source_overlap_volumes(axes)
        fractions = np.divide(
            overlap_volumes,
            cell_volumes,
            out=np.zeros_like(overlap_volumes),
            where=cell_volumes > 0.0,
        )
        q = Q_DENSITY_W_M3 * fractions
        active_volume = float(np.sum(overlap_volumes))
    else:
        raise ValueError(f"unsupported assignment method: {method}")

    integrated_power = float(np.sum(q * cell_volumes))
    active_count = int(np.count_nonzero(q[:, 0] > 0.0))
    source_missed = active_count == 0 or integrated_power <= 0.0
    volume_rel_error = abs(active_volume - target_volume) / max(abs(target_volume), 1.0e-30)
    power_rel_error = abs(integrated_power - target_power) / max(abs(target_power), 1.0e-30)
    return q, {
        "active_source_cell_count": active_count,
        "active_source_volume_proxy": active_volume,
        "target_source_volume": target_volume,
        "source_volume_relative_error": float(volume_rel_error),
        "integrated_q_power": integrated_power,
        "target_integrated_q_power": target_power,
        "integrated_q_power_relative_error": float(power_rel_error),
        "source_missed": bool(source_missed),
    }


def _make_k_field(coords: np.ndarray) -> np.ndarray:
    return np.full((coords.shape[0], 1), 12.0, dtype=np.float64)


def _make_meta(
    grid_shape: tuple[int, int, int],
    resolution: str,
    method: str,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "solver_v2_region_source_discretization_smoke",
        "subset_name": "temporary_region_source_discretization_smoke",
        "sample_id": f"{method}_{resolution}_region_source",
        "stage": "temporary_region_source_discretization_smoke",
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
            "k_mode": "isotropic",
            "source_assignment_method": method,
            "source_box_m": SOURCE_BOX,
            "q_density_W_m3": Q_DENSITY_W_M3,
            "source_summary": source_summary,
            "temporary_sample": True,
            "not_formal_dataset": True,
            "not_grid_convergence_study": True,
            "not_model_performance_evidence": True,
        },
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _write_sample(root: Path, resolution: str, method: str) -> tuple[Path, dict[str, Any]]:
    grid_shape = RESOLUTION_CASES[resolution]
    coords, axes = _make_grid(grid_shape)
    cell_volumes = _cell_volumes(axes)
    q_field, source_summary = _make_q_field(coords, axes, cell_volumes, method)

    sample_dir = root / f"{method}_{resolution}_region_source"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coords.npy", coords)
    np.save(sample_dir / "k_field.npy", _make_k_field(coords))
    np.save(sample_dir / "q_field.npy", q_field)
    (sample_dir / "sample_meta.json").write_text(
        json.dumps(_make_meta(grid_shape, resolution, method, source_summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return sample_dir, source_summary


def _peak_coord(coords: np.ndarray, temperature: np.ndarray) -> list[float]:
    peak_index = int(np.argmax(temperature.reshape(-1)))
    return [float(value) for value in coords[peak_index]]


def _run_case(sample_dir: Path, source_summary: dict[str, Any], resolution: str, method: str) -> tuple[bool, dict[str, Any]]:
    coords = np.load(sample_dir / "coords.npy")
    temperature, label_meta = solve_reference_temperature_v2(sample_dir)
    delta_t = temperature - T_REF_K
    residual_norm = float(label_meta["residual_norm"])
    bottom_error = float(label_meta["bottom_dirichlet_error"])
    delta_min = float(np.min(delta_t))
    delta_max = float(np.max(delta_t))
    finite = bool(np.all(np.isfinite(temperature)))
    solver_ok = (
        finite
        and bool(label_meta["convergence_flag"])
        and residual_norm <= RESIDUAL_TOL
        and bottom_error <= BOTTOM_TOL_K
        and delta_min >= -1.0e-8
        and delta_max <= DELTA_T_UPPER_BOUND_K
    )
    summary: dict[str, Any] = {
        "method": method,
        "resolution": resolution,
        "grid_shape": label_meta["assembly"]["grid_shape"],
        "node_count": int(label_meta["assembly"]["node_count"]),
        "q_density_W_m3": Q_DENSITY_W_M3,
        **source_summary,
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
        "solver_ok": solver_ok,
    }
    return solver_ok, summary


def _print_case(summary: dict[str, Any]) -> None:
    print(
        f"{summary['method']} {summary['resolution']}: "
        f"grid_shape={summary['grid_shape']} "
        f"node_count={summary['node_count']} "
        f"q_density={summary['q_density_W_m3']:.6e} "
        f"active_cells={summary['active_source_cell_count']} "
        f"active_volume={summary['active_source_volume_proxy']:.6e} "
        f"target_volume={summary['target_source_volume']:.6e} "
        f"volume_rel_error={summary['source_volume_relative_error']:.6e} "
        f"integrated_power={summary['integrated_q_power']:.6e} "
        f"power_rel_error={summary['integrated_q_power_relative_error']:.6e} "
        f"source_missed={summary['source_missed']} "
        f"T_range=[{summary['T_min']:.6f}, {summary['T_max']:.6f}] "
        f"DeltaT_range=[{summary['DeltaT_min']:.6e}, {summary['DeltaT_max']:.6e}] "
        f"peak_T={summary['peak_T']:.6f} "
        f"peak_coord={summary['peak_coord']} "
        f"residual_norm={summary['residual_norm']:.6e} "
        f"bottom_error={summary['bottom_dirichlet_error']:.6e} "
        f"converged={summary['convergence_flag']} "
        f"solver_ok={summary['solver_ok']}"
    )


def _node_counts_increase(summaries: list[dict[str, Any]], resolutions: list[str], method: str) -> bool:
    order = {resolution: idx for idx, resolution in enumerate(resolutions)}
    values = [
        (order[summary["resolution"]], int(summary["node_count"]))
        for summary in summaries
        if summary["method"] == method
    ]
    counts = [count for _, count in sorted(values)]
    return all(later > earlier for earlier, later in zip(counts, counts[1:]))


def _volume_fraction_consistency_ok(summaries: list[dict[str, Any]]) -> bool:
    vf = [summary for summary in summaries if summary["method"] == "volume_fraction"]
    return all(
        (not summary["source_missed"])
        and summary["source_volume_relative_error"] <= POWER_REL_TOL
        and summary["integrated_q_power_relative_error"] <= POWER_REL_TOL
        and summary["solver_ok"]
        for summary in vf
    )


def _center_in_box_warning_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    center = [summary for summary in summaries if summary["method"] == "center_in_box"]
    missed = [summary["resolution"] for summary in center if summary["source_missed"]]
    max_power_rel_error = max(
        (float(summary["integrated_q_power_relative_error"]) for summary in center),
        default=0.0,
    )
    return {
        "source_missed_resolutions": missed,
        "max_integrated_q_power_relative_error": max_power_rel_error,
        "expected_diagnostic_warning": bool(missed or max_power_rel_error > 0.25),
    }


def main() -> int:
    args = parse_args()
    print("Heat3D v1 region-source discretization smoke")
    print("scope: source-assignment diagnostic / research reference diagnostics only")
    print("not formal grid convergence, not benchmark, not model-performance evidence")
    print(f"resolutions: {args.resolutions}")
    print(f"methods: {args.methods}")
    print(f"source_box_m: {SOURCE_BOX}")
    print(f"q_density_W_m3: {Q_DENSITY_W_M3:.6e}")

    summaries: list[dict[str, Any]] = []
    failed = False
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_region_source_discretization_") as tmp:
        root = Path(tmp)
        for method in args.methods:
            for resolution in args.resolutions:
                sample_dir, source_summary = _write_sample(root, resolution=resolution, method=method)
                solver_ok, summary = _run_case(
                    sample_dir,
                    source_summary=source_summary,
                    resolution=resolution,
                    method=method,
                )
                failed = failed or not solver_ok
                summaries.append(summary)
                _print_case(summary)

        node_count_ok_by_method = {
            method: _node_counts_increase(summaries, args.resolutions, method)
            for method in args.methods
        }
        volume_fraction_ok = (
            _volume_fraction_consistency_ok(summaries)
            if "volume_fraction" in args.methods
            else True
        )
        center_warning = (
            _center_in_box_warning_summary(summaries)
            if "center_in_box" in args.methods
            else {
                "source_missed_resolutions": [],
                "max_integrated_q_power_relative_error": 0.0,
                "expected_diagnostic_warning": False,
            }
        )
        failed = failed or not all(node_count_ok_by_method.values()) or not volume_fraction_ok

        print("\nsummary")
        print(f"  temporary_sample_root: {root}")
        print("  temporary samples are removed after this smoke")
        print(f"  case_count: {len(summaries)}")
        print(f"  node_count_increases_by_method: {node_count_ok_by_method}")
        print(f"  volume_fraction_consistency_ok: {volume_fraction_ok}")
        print(f"  center_in_box_warning_summary: {center_warning}")
        print(f"  power_relative_tolerance: {POWER_REL_TOL}")
        print(f"  residual_tolerance: {RESIDUAL_TOL}")
        print(f"  bottom_tolerance_K: {BOTTOM_TOL_K}")
        print("  flux_energy_pde_diagnostics: not_computed / requires_numerical_operator")
        print(f"  region_source_discretization_smoke_ok: {not failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
