"""Minimal Heat3D v1 reference solver v2 for physics-label pipeline smoke.

This module implements a restricted research reference path for steady 3D heat
conduction on regular multilayer rectangular stacks. It is not a high-fidelity
solver, not a commercial FEM replacement, and not a formal benchmark label
generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SOLVER_NAME = "heat3d_v1_reference_solver_v2"
SOLVER_VERSION = "v2_minimal_fv_smoke_0"
DISCRETIZATION_TYPE = "conservative_finite_difference_finite_volume_style"
RESIDUAL_TOL = 1e-8
BOTTOM_TOL_K = 1e-6


def _load_sample(sample_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sample_path = Path(sample_dir)
    coords = np.load(sample_path / "coords.npy")
    k_field = np.load(sample_path / "k_field.npy")
    q_field = np.load(sample_path / "q_field.npy")
    meta = json.loads((sample_path / "sample_meta.json").read_text())
    return coords, k_field, q_field, meta


def _validate_supported_problem(meta: dict[str, Any], k_field: np.ndarray) -> list[str]:
    warnings: list[str] = []
    boundary_types = meta.get("boundary_types", {})
    expected = {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"}
    if boundary_types != expected:
        raise ValueError(
            "reference solver v2 minimal path supports only top Robin / "
            "bottom Dirichlet / sides adiabatic"
        )

    interfaces = meta.get("interfaces", [])
    if any(interface.get("type") != "perfect_contact" for interface in interfaces):
        raise ValueError("reference solver v2 minimal path supports only perfect_contact interfaces")

    if k_field.ndim != 2 or k_field.shape[1] not in (1, 3):
        raise ValueError(
            f"reference solver v2 minimal path supports only k_field shapes (N,1) and (N,3), found {k_field.shape}"
        )
    if k_field.shape[1] == 1:
        warnings.append("isotropic (N,1) conductivity expanded to diagonal (N,3)")
    return warnings


def _expand_k(k_field: np.ndarray) -> tuple[np.ndarray, str]:
    if k_field.shape[1] == 1:
        return np.repeat(k_field.astype(np.float64), repeats=3, axis=1), "isotropic_expanded_to_diag3"
    if k_field.shape[1] == 3:
        return k_field.astype(np.float64), "diag3"
    raise ValueError(f"unsupported k_field shape for solver v2: {k_field.shape}")


def _merge_duplicate_points(
    coords: np.ndarray,
    k_field: np.ndarray,
    q_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    n_unique = unique_coords.shape[0]

    k_acc = np.zeros((n_unique, k_field.shape[1]), dtype=np.float64)
    q_acc = np.zeros((n_unique, 1), dtype=np.float64)
    counts = np.zeros((n_unique, 1), dtype=np.float64)

    for original_idx, unique_idx in enumerate(inverse):
        k_acc[unique_idx] += k_field[original_idx]
        q_acc[unique_idx, 0] = max(q_acc[unique_idx, 0], q_field[original_idx, 0])
        counts[unique_idx, 0] += 1.0

    merged_k = k_acc / counts
    merged_q = q_acc
    merge_meta = {
        "original_node_count": int(coords.shape[0]),
        "unique_node_count": int(n_unique),
        "merged_duplicate_count": int(coords.shape[0] - n_unique),
        "q_merge_policy": "max_preserves_active_source_when_duplicate_interface_nodes_exist",
        "k_merge_policy": "arithmetic_mean_on_duplicate_coordinates_before_face_harmonic_means",
    }
    return unique_coords, inverse, merged_k, merged_q, merge_meta


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


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Conductivity must be positive for harmonic mean, got {a}, {b}")
    return 2.0 * a * b / (a + b)


def _grid_mapping(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    zs = np.unique(coords[:, 2])
    grid = -np.ones((xs.size, ys.size, zs.size), dtype=np.int64)
    lookup = {tuple(point): idx for idx, point in enumerate(coords)}
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                key = (x, y, z)
                if key not in lookup:
                    raise ValueError("Coordinates do not form a complete rectilinear grid after merging")
                grid[ix, iy, iz] = lookup[key]
    return xs, ys, zs, grid


def _assemble_system(
    coords: np.ndarray,
    k_diag: np.ndarray,
    q_field: np.ndarray,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs, ys, zs, grid = _grid_mapping(coords)
    dx_cv = _control_widths(xs)
    dy_cv = _control_widths(ys)
    dz_cv = _control_widths(zs)

    n = coords.shape[0]
    a = np.zeros((n, n), dtype=np.float64)
    b = np.zeros((n,), dtype=np.float64)

    boundary_params = meta["boundary_params"]
    h_top = float(boundary_params["top"]["h_W_m2K"])
    t_inf = float(boundary_params["top"]["ambient_temperature_K"])
    t_bottom = float(boundary_params["bottom"]["fixed_temperature_K"])

    def conductance(idx_i: int, idx_j: int, axis: int, area: float, distance: float) -> float:
        k_i = float(k_diag[idx_i, axis])
        k_j = float(k_diag[idx_j, axis])
        return _harmonic_mean(k_i, k_j) * area / distance

    for ix in range(xs.size):
        for iy in range(ys.size):
            for iz in range(zs.size):
                idx = int(grid[ix, iy, iz])

                if iz == 0:
                    a[idx, idx] = 1.0
                    b[idx] = t_bottom
                    continue

                volume = dx_cv[ix] * dy_cv[iy] * dz_cv[iz]
                rhs = float(q_field[idx, 0]) * volume
                diag = 0.0

                if iz == zs.size - 1:
                    area_top = dx_cv[ix] * dy_cv[iy]
                    diag += h_top * area_top
                    rhs += h_top * area_top * t_inf

                if ix > 0:
                    left = int(grid[ix - 1, iy, iz])
                    g = conductance(idx, left, axis=0, area=dy_cv[iy] * dz_cv[iz], distance=xs[ix] - xs[ix - 1])
                    diag += g
                    a[idx, left] -= g
                if ix < xs.size - 1:
                    right = int(grid[ix + 1, iy, iz])
                    g = conductance(idx, right, axis=0, area=dy_cv[iy] * dz_cv[iz], distance=xs[ix + 1] - xs[ix])
                    diag += g
                    a[idx, right] -= g

                if iy > 0:
                    back = int(grid[ix, iy - 1, iz])
                    g = conductance(idx, back, axis=1, area=dx_cv[ix] * dz_cv[iz], distance=ys[iy] - ys[iy - 1])
                    diag += g
                    a[idx, back] -= g
                if iy < ys.size - 1:
                    front = int(grid[ix, iy + 1, iz])
                    g = conductance(idx, front, axis=1, area=dx_cv[ix] * dz_cv[iz], distance=ys[iy + 1] - ys[iy])
                    diag += g
                    a[idx, front] -= g

                if iz > 0:
                    below = int(grid[ix, iy, iz - 1])
                    g = conductance(idx, below, axis=2, area=dx_cv[ix] * dy_cv[iy], distance=zs[iz] - zs[iz - 1])
                    diag += g
                    a[idx, below] -= g
                if iz < zs.size - 1:
                    above = int(grid[ix, iy, iz + 1])
                    g = conductance(idx, above, axis=2, area=dx_cv[ix] * dy_cv[iy], distance=zs[iz + 1] - zs[iz])
                    diag += g
                    a[idx, above] -= g

                a[idx, idx] = diag
                b[idx] = rhs

    assembly_meta = {
        "grid_shape": [int(xs.size), int(ys.size), int(zs.size)],
        "node_count": int(n),
        "top_robin_h_W_m2K": h_top,
        "top_robin_T_inf_K": t_inf,
        "bottom_dirichlet_T_K": t_bottom,
        "side_boundary_policy": "adiabatic_natural_zero_flux",
        "face_conductivity_policy": "harmonic_mean_between_neighboring_nodes",
        "linear_system_shape": [int(a.shape[0]), int(a.shape[1])],
    }
    return a, b, assembly_meta


def _bottom_dirichlet_error(
    coords: np.ndarray,
    temperature: np.ndarray,
    bottom_t: float,
) -> float:
    z_min = float(np.min(coords[:, 2]))
    bottom_mask = np.isclose(coords[:, 2], z_min)
    if not np.any(bottom_mask):
        return float("inf")
    return float(np.max(np.abs(temperature[bottom_mask, 0] - bottom_t)))


def solve_reference_temperature_v2(sample_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the current restricted steady problem and return temperature + metadata."""

    coords, k_field, q_field, meta = _load_sample(sample_dir)
    warnings = _validate_supported_problem(meta, k_field)
    k_diag, supported_k_mode = _expand_k(k_field)
    unique_coords, inverse, merged_k, merged_q, merge_meta = _merge_duplicate_points(coords, k_diag, q_field)
    a, b, assembly_meta = _assemble_system(unique_coords, merged_k, merged_q, meta)

    try:
        temperature_unique = np.linalg.solve(a, b)
        solve_error = None
    except np.linalg.LinAlgError as exc:
        temperature_unique = np.full((a.shape[0],), np.nan, dtype=np.float64)
        solve_error = str(exc)

    # Use np.dot rather than the matmul operator here because some Accelerate /
    # NumPy builds emit spurious floating-point warnings for this dense smoke
    # matrix even when all inputs and outputs are finite.
    residual = np.dot(a, temperature_unique) - b
    b_norm = float(max(np.linalg.norm(b), 1.0))
    residual_norm = float(np.linalg.norm(residual) / b_norm)
    temperature_full = temperature_unique[inverse].reshape(-1, 1).astype(np.float64)
    bottom_t = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
    bottom_error = _bottom_dirichlet_error(coords, temperature_full, bottom_t)
    finite_temperature = bool(np.all(np.isfinite(temperature_full)))
    convergence_flag = bool(
        solve_error is None
        and finite_temperature
        and np.isfinite(residual_norm)
        and residual_norm <= RESIDUAL_TOL
        and bottom_error <= BOTTOM_TOL_K
    )

    if solve_error is not None:
        warnings.append(f"linear solve failed: {solve_error}")
    if not finite_temperature:
        warnings.append("temperature contains NaN or Inf")
    if bottom_error > BOTTOM_TOL_K:
        warnings.append(f"bottom Dirichlet error exceeds tolerance: {bottom_error}")

    label_meta: dict[str, Any] = {
        "solver_name": SOLVER_NAME,
        "solver_version": SOLVER_VERSION,
        "solver_role": "minimal_research_reference_path",
        "not_high_fidelity": True,
        "discretization_type": DISCRETIZATION_TYPE,
        "supported_k_mode": supported_k_mode,
        "convergence_flag": convergence_flag,
        "residual_norm": residual_norm,
        "residual_tolerance": RESIDUAL_TOL,
        "bottom_dirichlet_error": bottom_error,
        "bottom_dirichlet_tolerance_K": BOTTOM_TOL_K,
        "top_robin_status": {
            "status": "included_in_linear_system",
            "diagnostic_scope": "residual_proxy_only",
        },
        "side_adiabatic_status": {
            "status": "natural_zero_flux_boundary_in_discretization",
            "diagnostic_scope": "no_explicit_flux_violation_metric_yet",
        },
        "interface_status": {
            "status": "perfect_contact_supported_for_current_rectilinear_grid",
            "diagnostic_scope": "shared_nodes_or_conservative_face_treatment",
        },
        "energy_balance_status": {
            "status": "not_computed",
            "reason": "global energy balance proxy deferred to later label diagnostics integration",
        },
        "pde_residual_status": {
            "status": "linear_system_residual_proxy_only",
            "reason": "continuous PDE residual is not computed",
        },
        "assembly": assembly_meta,
        "duplicate_merge": merge_meta,
        "warnings": warnings,
    }
    return temperature_full, label_meta
