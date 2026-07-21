#!/usr/bin/env python3
"""Read-only V5 physics-distribution audit and bounded counterfactual solver.

The default operation reads the frozen P5 sample arrays, metadata, manifest,
and split map.  It never mutates the dataset.  Optional counterfactuals solve
only explicitly named samples and write only the requested JSON report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_SCHEMA = "heat3d_v6_p0_v5_physics_distribution_v1"
ROLE_ORDER = (
    "train",
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
METRIC_UNITS = {
    "k_min_W_mK": "W/(m K)",
    "k_max_W_mK": "W/(m K)",
    "kz_harmonic_W_mK": "W/(m K)",
    "nonzero_q_min_W_m3": "W/m^3",
    "nonzero_q_median_W_m3": "W/m^3",
    "nonzero_q_max_W_m3": "W/m^3",
    "source_volume_m3": "m^3",
    "source_volume_fraction": "1",
    "total_power_W": "W",
    "face_power_density_W_m2": "W/m^2",
    "top_h_W_m2K": "W/(m^2 K)",
    "deltaT_peak_K": "K",
    "deltaT_cv_mean_K": "K",
    "deltaT_cv_rms_K": "K",
    "Rth_peak_K_W": "K/W",
    "top_heat_flux_fraction": "1",
    "bottom_heat_flux_fraction": "1",
    "energy_balance_relative_error": "1",
    "power_calibration_error_W": "W",
    "power_calibration_relative_error": "1",
    "q_rescale_factor": "1",
    "q_clipping_detected": "bool",
}


class AuditError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _widths(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise AuditError("grid axes must be finite, 1-D, and strictly increasing")
    result = np.empty_like(axis)
    result[0] = 0.5 * (axis[1] - axis[0])
    result[-1] = 0.5 * (axis[-1] - axis[-2])
    if axis.size > 2:
        result[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return result


def _grid_contract(coords: np.ndarray) -> dict[str, Any]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise AuditError(f"invalid coords shape/content: {coords.shape}")
    axes: list[np.ndarray] = []
    inverse: list[np.ndarray] = []
    for dim in range(3):
        axis, indices = np.unique(coords[:, dim], return_inverse=True)
        axes.append(axis)
        inverse.append(indices)
    shape = tuple(int(axis.size) for axis in axes)
    if int(np.prod(shape)) != coords.shape[0]:
        raise AuditError(f"coords are not a complete rectilinear grid: {shape}")
    grid = np.full(shape, -1, dtype=np.int64)
    for node, ijk in enumerate(zip(*inverse)):
        if grid[ijk] != -1:
            raise AuditError("duplicate coordinate")
        grid[ijk] = node
    if np.any(grid < 0):
        raise AuditError("incomplete rectilinear grid")
    widths = tuple(_widths(axis) for axis in axes)
    weights = widths[0][inverse[0]] * widths[1][inverse[1]] * widths[2][inverse[2]]
    return {"axes": axes, "widths": widths, "shape": shape, "grid": grid, "weights": weights}


def _expand_k(k_field: np.ndarray, node_count: int) -> tuple[np.ndarray, int]:
    k_field = np.asarray(k_field, dtype=np.float64)
    if k_field.shape == (node_count, 1):
        return np.repeat(k_field, 3, axis=1), 1
    if k_field.shape == (node_count, 3):
        return k_field, 3
    raise AuditError(f"k_field must have shape [{node_count},1] or [{node_count},3]")


def _boundary(meta: Mapping[str, Any]) -> tuple[float, float, float]:
    params = meta.get("boundary_params")
    if not isinstance(params, Mapping):
        raise AuditError("missing boundary_params")
    top = params.get("top")
    bottom = params.get("bottom")
    if not isinstance(top, Mapping) or not isinstance(bottom, Mapping):
        raise AuditError("missing top/bottom boundary parameters")
    top_h = float(top["h_W_m2K"])
    top_t = float(top.get("T_inf_K", top.get("ambient_temperature_K")))
    bottom_t = float(bottom.get("T_fixed_K", bottom.get("fixed_temperature_K")))
    return top_h, top_t, bottom_t


def _boundary_fluxes(
    coords: np.ndarray,
    k_diag: np.ndarray,
    temperature: np.ndarray,
    top_h: float,
    top_t: float,
    grid_info: Mapping[str, Any],
) -> tuple[float, float]:
    axes = grid_info["axes"]
    widths = grid_info["widths"]
    grid = grid_info["grid"]
    xs, ys, zs = axes
    dx, dy, _ = widths
    top_flux = 0.0
    bottom_flux = 0.0
    for ix in range(xs.size):
        for iy in range(ys.size):
            area = float(dx[ix] * dy[iy])
            top_idx = int(grid[ix, iy, zs.size - 1])
            bottom_idx = int(grid[ix, iy, 0])
            above_idx = int(grid[ix, iy, 1])
            top_flux += top_h * area * (float(temperature[top_idx]) - top_t)
            k0 = float(k_diag[bottom_idx, 2])
            k1 = float(k_diag[above_idx, 2])
            harmonic = 2.0 * k0 * k1 / (k0 + k1)
            conductance = harmonic * area / float(zs[1] - zs[0])
            bottom_flux += conductance * (
                float(temperature[above_idx]) - float(temperature[bottom_idx])
            )
    return float(top_flux), float(bottom_flux)


def _sample_record(
    sample_dir: Path,
    role: str,
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    meta = _read_json(sample_dir / "sample_meta.json")
    coords = np.asarray(np.load(sample_dir / "coords.npy", mmap_mode="r"), dtype=np.float64)
    q = np.asarray(np.load(sample_dir / "q_field.npy", mmap_mode="r"), dtype=np.float64).reshape(-1)
    k_raw = np.asarray(np.load(sample_dir / "k_field.npy", mmap_mode="r"), dtype=np.float64)
    temperature = np.asarray(
        np.load(sample_dir / "temperature.npy", mmap_mode="r"), dtype=np.float64
    ).reshape(-1)
    if not all(np.all(np.isfinite(array)) for array in (coords, q, k_raw, temperature)):
        raise AuditError(f"{sample_dir.name}: non-finite array")
    grid_info = _grid_contract(coords)
    weights = np.asarray(grid_info["weights"], dtype=np.float64)
    k_diag, k_width = _expand_k(k_raw, coords.shape[0])
    if np.any(k_diag <= 0.0) or np.any(q < 0.0):
        raise AuditError(f"{sample_dir.name}: invalid k or q")
    top_h, top_t, bottom_t = _boundary(meta)
    delta_t = temperature - bottom_t
    total_volume = float(np.sum(weights))
    total_power = float(np.dot(q, weights))
    q_positive = q > 0.0
    q_values = q[q_positive]
    source_volume = float(np.sum(weights[q_positive]))
    xs, ys, _ = grid_info["axes"]
    footprint_area = float((xs[-1] - xs[0]) * (ys[-1] - ys[0]))
    q_audit = meta.get("q_power_audit") or {}
    target_power = float(q_audit.get("q_total_target_power_W", total_power))
    power_error = total_power - target_power
    delta_t_mean = float(np.dot(delta_t, weights) / total_volume)
    delta_t_rms = float(math.sqrt(np.dot(delta_t * delta_t, weights) / total_volume))
    delta_t_peak = float(np.max(delta_t))
    kz_harmonic = float(total_volume / np.dot(weights, 1.0 / k_diag[:, 2]))
    top_flux, bottom_flux = _boundary_fluxes(
        coords, k_diag, temperature, top_h, top_t, grid_info
    )
    scale = max(abs(total_power), 1.0e-30)
    delta_t_qc = meta.get("deltaT_qc") or {}
    rescale = float(delta_t_qc.get("q_rescale_factor", 1.0))
    clip_keys = (
        "q_clip_min_W_m3",
        "q_clip_max_W_m3",
        "q_clipped_node_count",
        "q_clipping_fraction",
    )
    clipping_declared = any(key in meta or key in q_audit for key in clip_keys)
    clipping_detected = bool(clipping_declared or not np.isclose(rescale, 1.0))
    source_family = str(manifest_row.get("q_family") or "unreported")
    bc_regime = str(manifest_row.get("cooling_regime") or "unreported")
    return {
        "sample_id": sample_dir.name,
        "role": role,
        "source_family": source_family,
        "bc_regime": bc_regime,
        "k_mode": str(manifest_row.get("k_mode") or f"width_{k_width}"),
        "deltaT_bin": str(manifest_row.get("DeltaT_bin") or "unreported"),
        "qc_class": str(manifest_row.get("qc_class") or "unreported"),
        "grid_shape": list(grid_info["shape"]),
        "metrics": {
            "k_min_W_mK": float(np.min(k_diag)),
            "k_max_W_mK": float(np.max(k_diag)),
            "kz_harmonic_W_mK": kz_harmonic,
            "nonzero_q_min_W_m3": float(np.min(q_values)) if q_values.size else 0.0,
            "nonzero_q_median_W_m3": float(np.median(q_values)) if q_values.size else 0.0,
            "nonzero_q_max_W_m3": float(np.max(q_values)) if q_values.size else 0.0,
            "source_volume_m3": source_volume,
            "source_volume_fraction": source_volume / total_volume,
            "total_power_W": total_power,
            "face_power_density_W_m2": total_power / footprint_area,
            "top_h_W_m2K": top_h,
            "ambient_temperature_K": top_t,
            "bottom_temperature_K": bottom_t,
            "deltaT_peak_K": delta_t_peak,
            "deltaT_cv_mean_K": delta_t_mean,
            "deltaT_cv_rms_K": delta_t_rms,
            "Rth_peak_K_W": delta_t_peak / scale,
            "top_heat_flux_W": top_flux,
            "bottom_heat_flux_W": bottom_flux,
            "top_heat_flux_fraction": top_flux / scale,
            "bottom_heat_flux_fraction": bottom_flux / scale,
            "energy_balance_relative_error": (total_power - top_flux - bottom_flux) / scale,
            "power_calibration_error_W": power_error,
            "power_calibration_relative_error": power_error / scale,
            "q_rescale_factor": rescale,
            "q_clipping_detected": clipping_detected,
        },
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("count", "min", "p05", "p25", "median", "mean", "p75", "p95", "max")}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _summarize_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_names = [name for name, unit in METRIC_UNITS.items() if unit != "bool"]
    return {
        "sample_count": len(records),
        "metrics": {
            name: _summary(float(record["metrics"][name]) for record in records)
            for name in numeric_names
        },
        "q_clipping_detected_count": sum(
            bool(record["metrics"]["q_clipping_detected"]) for record in records
        ),
        "grid_shapes": dict(
            sorted(Counter("x".join(map(str, record["grid_shape"])) for record in records).items())
        ),
    }


def _grouped(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        label = " | ".join(str(record[key]) for key in keys)
        groups[label].append(record)
    return {label: _summarize_group(group) for label, group in sorted(groups.items())}


def _solve_fvm(
    *,
    coords: np.ndarray,
    k_diag: np.ndarray,
    q: np.ndarray,
    top_h: float,
    top_t: float,
    bottom_mode: str,
    bottom_t: float,
    bottom_h: float | None = None,
    contact_resistance_m2K_W: float = 0.0,
    contact_lower_z_index: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    info = _grid_contract(coords)
    axes = info["axes"]
    widths = info["widths"]
    grid = info["grid"]
    xs, ys, zs = axes
    dx, dy, dz = widths
    n = coords.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n, dtype=np.float64)

    def add(i: int, j: int, value: float) -> None:
        rows.append(i)
        cols.append(j)
        data.append(float(value))

    def face_g(i: int, j: int, axis: int, area: float, distance: float, crossing: bool) -> float:
        ki = float(k_diag[i, axis])
        kj = float(k_diag[j, axis])
        if crossing and contact_resistance_m2K_W > 0.0:
            return area / (
                0.5 * distance / ki
                + contact_resistance_m2K_W
                + 0.5 * distance / kj
            )
        harmonic = 2.0 * ki * kj / (ki + kj)
        return harmonic * area / distance

    for ix in range(xs.size):
        for iy in range(ys.size):
            for iz in range(zs.size):
                i = int(grid[ix, iy, iz])
                if iz == 0 and bottom_mode == "dirichlet":
                    add(i, i, 1.0)
                    rhs[i] = bottom_t
                    continue
                diag = 0.0
                volume = float(dx[ix] * dy[iy] * dz[iz])
                row_rhs = float(q[i]) * volume
                if iz == zs.size - 1:
                    robin = top_h * float(dx[ix] * dy[iy])
                    diag += robin
                    row_rhs += robin * top_t
                if iz == 0 and bottom_mode == "robin":
                    if bottom_h is None or bottom_h <= 0.0:
                        raise AuditError("bottom Robin requires positive bottom_h")
                    robin = bottom_h * float(dx[ix] * dy[iy])
                    diag += robin
                    row_rhs += robin * bottom_t
                neighbors = (
                    (ix - 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(xs[ix] - xs[ix - 1]) if ix else 0.0, False),
                    (ix + 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(xs[ix + 1] - xs[ix]) if ix + 1 < xs.size else 0.0, False),
                    (ix, iy - 1, iz, 1, float(dx[ix] * dz[iz]), float(ys[iy] - ys[iy - 1]) if iy else 0.0, False),
                    (ix, iy + 1, iz, 1, float(dx[ix] * dz[iz]), float(ys[iy + 1] - ys[iy]) if iy + 1 < ys.size else 0.0, False),
                    (ix, iy, iz - 1, 2, float(dx[ix] * dy[iy]), float(zs[iz] - zs[iz - 1]) if iz else 0.0, contact_lower_z_index is not None and iz - 1 == contact_lower_z_index),
                    (ix, iy, iz + 1, 2, float(dx[ix] * dy[iy]), float(zs[iz + 1] - zs[iz]) if iz + 1 < zs.size else 0.0, contact_lower_z_index is not None and iz == contact_lower_z_index),
                )
                for jx, jy, jz, axis, area, distance, crossing in neighbors:
                    if not (0 <= jx < xs.size and 0 <= jy < ys.size and 0 <= jz < zs.size):
                        continue
                    j = int(grid[jx, jy, jz])
                    g = face_g(i, j, axis, area, distance, crossing)
                    diag += g
                    add(i, j, -g)
                add(i, i, diag)
                rhs[i] = row_rhs
    matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    matrix.sum_duplicates()
    temperature = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    residual = float(np.linalg.norm(matrix.dot(temperature) - rhs) / max(np.linalg.norm(rhs), 1.0))
    if not np.all(np.isfinite(temperature)) or not math.isfinite(residual):
        raise AuditError("counterfactual solve returned non-finite values")
    top_flux, bottom_flux = _boundary_fluxes(coords, k_diag, temperature, top_h, top_t, info)
    if bottom_mode == "robin":
        bottom_flux = 0.0
        for ix in range(xs.size):
            for iy in range(ys.size):
                i = int(grid[ix, iy, 0])
                bottom_flux += float(bottom_h) * float(dx[ix] * dy[iy]) * (
                    float(temperature[i]) - bottom_t
                )
    elif bottom_mode == "adiabatic":
        bottom_flux = 0.0
    return temperature, {"linear_residual": residual, "top_flux_W": top_flux, "bottom_flux_W": bottom_flux}


def _case_metrics(
    *,
    temperature: np.ndarray,
    q: np.ndarray,
    coords: np.ndarray,
    bottom_t: float,
    solver_audit: Mapping[str, float],
) -> dict[str, float]:
    info = _grid_contract(coords)
    weights = np.asarray(info["weights"], dtype=np.float64)
    volume = float(np.sum(weights))
    power = float(np.dot(q, weights))
    delta_t = temperature - bottom_t
    scale = max(abs(power), 1.0e-30)
    return {
        "total_power_W": power,
        "source_volume_m3": float(np.sum(weights[q > 0.0])),
        "deltaT_peak_K": float(np.max(delta_t)),
        "deltaT_cv_mean_K": float(np.dot(delta_t, weights) / volume),
        "deltaT_cv_rms_K": float(math.sqrt(np.dot(delta_t * delta_t, weights) / volume)),
        "top_heat_flux_fraction": float(solver_audit["top_flux_W"] / scale),
        "bottom_heat_flux_fraction": float(solver_audit["bottom_flux_W"] / scale),
        "energy_balance_relative_error": float(
            (power - solver_audit["top_flux_W"] - solver_audit["bottom_flux_W"]) / scale
        ),
        "linear_residual": float(solver_audit["linear_residual"]),
    }


def _resample_nearest(
    coords: np.ndarray,
    values: np.ndarray,
    new_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    info = _grid_contract(coords)
    old_axes = info["axes"]
    old_grid = info["grid"]
    new_axes = [np.linspace(axis[0], axis[-1], int(size)) for axis, size in zip(old_axes, new_shape)]
    mesh = np.meshgrid(*new_axes, indexing="ij")
    new_coords = np.column_stack([part.reshape(-1) for part in mesh])
    nearest = [
        np.abs(old_axis[:, None] - new_axis[None, :]).argmin(axis=0)
        for old_axis, new_axis in zip(old_axes, new_axes)
    ]
    old_values = np.asarray(values)
    trailing = old_values.shape[1:]
    new_values = np.empty(tuple(map(int, new_shape)) + trailing, dtype=np.float64)
    for ix, ox in enumerate(nearest[0]):
        for iy, oy in enumerate(nearest[1]):
            for iz, oz in enumerate(nearest[2]):
                new_values[ix, iy, iz] = old_values[int(old_grid[ox, oy, oz])]
    return new_coords, new_values.reshape((new_coords.shape[0],) + trailing)


def _counterfactual(sample_dir: Path, refined_shape: Sequence[int]) -> dict[str, Any]:
    from scipy.ndimage import binary_dilation

    meta = _read_json(sample_dir / "sample_meta.json")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_raw = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float64).reshape(-1)
    stored_temperature = np.asarray(np.load(sample_dir / "temperature.npy"), dtype=np.float64).reshape(-1)
    k_diag, _ = _expand_k(k_raw, coords.shape[0])
    top_h, top_t, bottom_t = _boundary(meta)
    info = _grid_contract(coords)
    weights = np.asarray(info["weights"], dtype=np.float64)
    original_power = float(np.dot(q, weights))
    mid_face = max(0, int(info["shape"][2] // 2) - 1)

    def solve_case(
        name: str,
        *,
        case_coords: np.ndarray = coords,
        case_k: np.ndarray = k_diag,
        case_q: np.ndarray = q,
        case_top_h: float = top_h,
        bottom_mode: str = "dirichlet",
        bottom_h: float | None = None,
        contact_r: float = 0.0,
        contact_face: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        temperature, solver_audit = _solve_fvm(
            coords=case_coords,
            k_diag=case_k,
            q=case_q,
            top_h=case_top_h,
            top_t=top_t,
            bottom_mode=bottom_mode,
            bottom_t=bottom_t,
            bottom_h=bottom_h,
            contact_resistance_m2K_W=contact_r,
            contact_lower_z_index=contact_face,
        )
        return name, _case_metrics(
            temperature=temperature,
            q=case_q,
            coords=case_coords,
            bottom_t=bottom_t,
            solver_audit=solver_audit,
        ) | {"node_count": int(case_coords.shape[0])}

    cases = dict(
        [
            solve_case("baseline_replay"),
            solve_case("power_x0.5", case_q=q * 0.5),
            solve_case("power_x2", case_q=q * 2.0),
            solve_case("top_h_x0.5", case_top_h=top_h * 0.5),
            solve_case("top_h_x2", case_top_h=top_h * 2.0),
            solve_case("bottom_robin_h1000", bottom_mode="robin", bottom_h=1000.0),
            solve_case("bottom_adiabatic", bottom_mode="adiabatic"),
            solve_case(
                "contact_R_5e-6_m2K_W",
                contact_r=5.0e-6,
                contact_face=mid_face,
            ),
            solve_case(
                "contact_R_1e-5_m2K_W",
                contact_r=1.0e-5,
                contact_face=mid_face,
            ),
        ]
    )

    mask = (q > 0.0).reshape(info["shape"])
    interior = np.ones(info["shape"], dtype=bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    interior[:, :, [0, -1]] = False
    expanded = binary_dilation(mask, iterations=1) & interior
    if np.any(expanded):
        q_expanded = np.zeros_like(q)
        expanded_flat = expanded.reshape(-1)
        q_expanded[expanded_flat] = original_power / float(np.sum(weights[expanded_flat]))
        name, metrics = solve_case("larger_source_volume_fixed_power", case_q=q_expanded)
        cases[name] = metrics

    new_coords, new_k = _resample_nearest(coords, k_diag, refined_shape)
    _, new_q_2d = _resample_nearest(coords, q[:, None], refined_shape)
    new_q = new_q_2d.reshape(-1)
    new_weights = np.asarray(_grid_contract(new_coords)["weights"], dtype=np.float64)
    projected_power = float(np.dot(new_q, new_weights))
    if projected_power <= 0.0:
        raise AuditError("refined q projection has zero power")
    new_q *= original_power / projected_power
    new_contact_face = max(0, int(refined_shape[2] // 2) - 1)
    name, metrics = solve_case(
        "refined_nearest_projection_fixed_power",
        case_coords=new_coords,
        case_k=new_k,
        case_q=new_q,
        contact_face=new_contact_face,
    )
    cases[name] = metrics

    replay_temperature, _ = _solve_fvm(
        coords=coords,
        k_diag=k_diag,
        q=q,
        top_h=top_h,
        top_t=top_t,
        bottom_mode="dirichlet",
        bottom_t=bottom_t,
    )
    return {
        "sample_id": sample_dir.name,
        "original_grid_shape": list(info["shape"]),
        "refined_grid_shape": list(map(int, refined_shape)),
        "top_h_W_m2K": top_h,
        "bottom_T_K": bottom_t,
        "baseline_replay_max_abs_error_K": float(np.max(np.abs(replay_temperature - stored_temperature))),
        "cases": cases,
        "interpretation_limits": [
            "The refined case is nearest-neighbor field projection with fixed integrated power, not a generator-native mesh convergence study.",
            "Finite contact resistance is inserted on the middle z face in the audit-only FVM; V5 production samples use perfect contact.",
            "Bottom Robin/adiabatic cases change the audit-only boundary operator and do not alter the production solver.",
        ],
    }


def _self_check() -> dict[str, Any]:
    axes = [np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 0.5e-3, 4)]
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.column_stack([part.reshape(-1) for part in mesh])
    k = np.full((coords.shape[0], 3), 10.0)
    q = np.zeros(coords.shape[0])
    info = _grid_contract(coords)
    mask = np.ones(info["shape"], dtype=bool)
    mask[[0, -1], :, :] = False
    mask[:, [0, -1], :] = False
    mask[:, :, [0, -1]] = False
    q[mask.reshape(-1)] = 1.0e8
    t1, a1 = _solve_fvm(
        coords=coords, k_diag=k, q=q, top_h=1000.0, top_t=300.0,
        bottom_mode="dirichlet", bottom_t=300.0,
    )
    t2, a2 = _solve_fvm(
        coords=coords, k_diag=k, q=2.0 * q, top_h=1000.0, top_t=300.0,
        bottom_mode="dirichlet", bottom_t=300.0,
    )
    linearity_error = float(np.max(np.abs((t2 - 300.0) - 2.0 * (t1 - 300.0))))
    m1 = _case_metrics(temperature=t1, q=q, coords=coords, bottom_t=300.0, solver_audit=a1)
    return {
        "passed": bool(
            linearity_error < 1.0e-9
            and abs(m1["energy_balance_relative_error"]) < 1.0e-9
            and m1["linear_residual"] < 1.0e-10
        ),
        "power_linearity_max_abs_error_K": linearity_error,
        "energy_balance_relative_error": m1["energy_balance_relative_error"],
        "linear_residual": m1["linear_residual"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset is None or args.split_map is None:
        raise AuditError("--dataset and --split-map are required outside --self-check-only")
    dataset = args.dataset.resolve()
    split_map = args.split_map.resolve()
    if not dataset.is_dir() or not split_map.is_file():
        raise AuditError("dataset or split map is missing")
    split_payload = _read_json(split_map)
    assignments = {str(k): str(v) for k, v in split_payload["sample_splits"].items()}
    sample_dirs = {path.name for path in dataset.glob("sample_*") if path.is_dir()}
    if sample_dirs != set(assignments):
        raise AuditError("dataset sample directories and split map IDs differ")
    manifest_path = dataset / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_rows = {str(row["sample_id"]): row for row in manifest["samples"]}
    if set(manifest_rows) != set(assignments):
        raise AuditError("manifest and split map IDs differ")
    counts = Counter(assignments.values())
    if args.dry_run:
        return {
            "audit_schema": AUDIT_SCHEMA,
            "mode": "dry_run",
            "read_only": True,
            "sample_count": len(assignments),
            "role_counts": dict(sorted(counts.items())),
            "planned_array_reads": 0,
            "planned_solver_calls": 0,
            "planned_dataset_writes": 0,
        }
    records = [
        _sample_record(dataset / sample_id, assignments[sample_id], manifest_rows[sample_id])
        for sample_id in sorted(assignments)
    ]
    counterfactuals = [
        _counterfactual(dataset / sample_id, args.refined_grid_shape)
        for sample_id in args.counterfactual_sample_id
    ]
    payload: dict[str, Any] = {
        "audit_schema": AUDIT_SCHEMA,
        "mode": "read_only_audit_with_bounded_counterfactuals" if counterfactuals else "read_only_audit",
        "dataset": {
            "path": str(dataset),
            "dataset_id": split_payload.get("dataset_id"),
            "sample_count": len(records),
            "role_counts": dict(sorted(counts.items())),
            "manifest_sha256": _sha256(manifest_path),
            "split_map_sha256": _sha256(split_map),
        },
        "units": METRIC_UNITS,
        "summaries": {
            "all": _summarize_group(records),
            "by_split": _grouped(records, ["role"]),
            "by_source": _grouped(records, ["source_family"]),
            "by_bc": _grouped(records, ["bc_regime"]),
            "by_split_source_bc": _grouped(records, ["role", "source_family", "bc_regime"]),
        },
        "category_counts": {
            "source_family": dict(sorted(Counter(record["source_family"] for record in records).items())),
            "bc_regime": dict(sorted(Counter(record["bc_regime"] for record in records).items())),
            "k_mode": dict(sorted(Counter(record["k_mode"] for record in records).items())),
            "deltaT_bin": dict(sorted(Counter(record["deltaT_bin"] for record in records).items())),
            "qc_class": dict(sorted(Counter(record["qc_class"] for record in records).items())),
        },
        "integrity": {
            "dataset_sample_ids_equal_manifest": True,
            "dataset_sample_ids_equal_split_map": True,
            "all_metrics_finite": True,
            "q_clipping_detected_count": sum(record["metrics"]["q_clipping_detected"] for record in records),
            "max_abs_power_calibration_error_W": max(abs(record["metrics"]["power_calibration_error_W"]) for record in records),
            "max_abs_energy_balance_relative_error": max(abs(record["metrics"]["energy_balance_relative_error"]) for record in records),
        },
        "counterfactuals": counterfactuals,
        "self_check": _self_check(),
        "guardrails": {
            "training_runs": 0,
            "model_inference_runs": 0,
            "full_dataset_generation_runs": 0,
            "dataset_writes": 0,
            "permitted_write": "explicit audit JSON only",
        },
    }
    if args.include_sample_records:
        payload["sample_records"] = records
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-sample-records", action="store_true")
    parser.add_argument("--counterfactual-sample-id", action="append", default=[])
    parser.add_argument("--refined-grid-shape", nargs=3, type=int, default=(24, 24, 8))
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check_only:
        payload = {"self_check": _self_check()}
    else:
        payload = run(args)
    if args.output_json:
        _write_json(args.output_json, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("self_check", {"passed": True}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
