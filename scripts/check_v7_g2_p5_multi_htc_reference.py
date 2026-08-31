#!/usr/bin/env python3
"""Independent analytical-vs-3D gate for the DeepOHeat multi-HTC PDE."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyamg
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres


K = 0.2
AMBIENT_U = 0.2
SOURCE_START = 0.25
SOURCE_END = 0.30
DOMAIN_Z = 0.55
PASS_THRESHOLDS = {
    "field_rmse_u_max": 3.0e-3,
    "relative_l2_u_max": 3.0e-3,
    "peak_absolute_error_u_max": 8.0e-3,
    "xy_nonuniformity_peak_to_peak_u_max": 1.0e-8,
    "relative_linear_residual_max": 1.0e-8,
}


def analytical_coefficients(beta_top: float, beta_bottom: float) -> np.ndarray:
    za, zb, length = SOURCE_START, SOURCE_END, DOMAIN_Z
    # Unknowns are a1,b1,a2,b2,a3,b3 for linear/quadratic/linear regions.
    matrix = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    matrix[0] = [za, 1, -za, -1, 0, 0]; rhs[0] = -(za**2) / (2 * K)
    matrix[1] = [1, 0, -1, 0, 0, 0]; rhs[1] = -za / K
    matrix[2] = [0, 0, zb, 1, -zb, -1]; rhs[2] = (zb**2) / (2 * K)
    matrix[3] = [0, 0, 1, 0, -1, 0]; rhs[3] = zb / K
    matrix[4] = [-beta_bottom, 1, 0, 0, 0, 0]; rhs[4] = AMBIENT_U
    matrix[5] = [0, 0, 0, 0, length + beta_top, 1]; rhs[5] = AMBIENT_U
    return np.linalg.solve(matrix, rhs)


def analytical_u(z: np.ndarray, beta_top: float, beta_bottom: float) -> np.ndarray:
    a1, b1, a2, b2, a3, b3 = analytical_coefficients(beta_top, beta_bottom)
    z = np.asarray(z, dtype=np.float64)
    return np.where(
        z < SOURCE_START,
        a1 * z + b1,
        np.where(z <= SOURCE_END, -(z**2) / (2 * K) + a2 * z + b2, a3 * z + b3),
    )


def source_control_volume_average(z: np.ndarray) -> np.ndarray:
    dz = float(z[1] - z[0])
    left = np.maximum(z - dz / 2, 0.0)
    right = np.minimum(z + dz / 2, DOMAIN_Z)
    overlap = np.maximum(0.0, np.minimum(right, SOURCE_END) - np.maximum(left, SOURCE_START))
    widths = right - left
    return overlap / widths


def build_matrix(beta_top: float, beta_bottom: float, n: int = 51) -> sparse.csr_matrix:
    nx = ny = nz = n; dx = dy = 1.0 / (n - 1); dz = DOMAIN_Z / (n - 1); nxy = nx * ny
    rows: list[np.ndarray] = []; cols: list[np.ndarray] = []; values: list[np.ndarray] = []
    ij = np.arange(nxy, dtype=np.int64); ix = np.tile(np.arange(nx), ny); iy = np.repeat(np.arange(ny), nx)
    def add(row: np.ndarray, col: np.ndarray, value: np.ndarray) -> None:
        rows.append(np.asarray(row, dtype=np.int64)); cols.append(np.asarray(col, dtype=np.int64)); values.append(np.asarray(value, dtype=np.float64))
    for iz in range(nz):
        row = iz * nxy + ij
        if iz == 0:
            add(row, row, np.full(nxy, 1 + beta_bottom / dz)); add(row, row + nxy, np.full(nxy, -beta_bottom / dz)); continue
        if iz == nz - 1:
            add(row, row, np.full(nxy, 1 + beta_top / dz)); add(row, row - nxy, np.full(nxy, -beta_top / dz)); continue
        diag = np.full(nxy, -2 * K / dz**2)
        for coordinate, stride, spacing in ((ix, 1, dx), (iy, nx, dy)):
            low, high = coordinate == 0, coordinate == n - 1; middle = ~(low | high)
            diag[low | high] += -K / spacing**2; diag[middle] += -2 * K / spacing**2
            add(row[low], row[low] + stride, np.full(low.sum(), K / spacing**2))
            add(row[high], row[high] - stride, np.full(high.sum(), K / spacing**2))
            add(row[middle], row[middle] - stride, np.full(middle.sum(), K / spacing**2))
            add(row[middle], row[middle] + stride, np.full(middle.sum(), K / spacing**2))
        add(row, row - nxy, np.full(nxy, K / dz**2)); add(row, row + nxy, np.full(nxy, K / dz**2)); add(row, row, diag)
    result = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n**3, n**3)).tocsr()
    result.sum_duplicates(); return result


def solve_3d(beta_top: float, beta_bottom: float) -> tuple[np.ndarray, dict[str, Any]]:
    n = 51; z = np.linspace(0.0, DOMAIN_Z, n); nxy = n * n
    matrix = build_matrix(beta_top, beta_bottom, n)
    rhs = np.zeros(n**3, dtype=np.float64); qz = source_control_volume_average(z)
    for iz in range(n):
        rhs[iz*nxy:(iz+1)*nxy] = AMBIENT_U if iz in (0, n-1) else -qz[iz]
    setup = time.perf_counter(); ml = pyamg.smoothed_aggregation_solver(matrix, max_levels=10, max_coarse=100)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x: ml.solve(x, tol=1e-1, maxiter=1), dtype=np.float64)
    setup_seconds = time.perf_counter() - setup; residuals: list[float] = []; started = time.perf_counter()
    solution, info = gmres(matrix, rhs, M=preconditioner, rtol=1e-10, atol=0.0, restart=150, maxiter=150, callback=lambda x: residuals.append(float(x)), callback_type="pr_norm")
    wall = time.perf_counter() - started
    relative_residual = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
    field = solution.reshape((n, n, n)).transpose(1, 2, 0)
    return field, {"gmres_info": int(info), "iterations": len(residuals), "relative_linear_residual": relative_residual, "matrix_nnz": int(matrix.nnz), "amg_setup_seconds": setup_seconds, "solve_wall_seconds": wall}


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    cases = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.1)]
    rows = []
    for beta_top, beta_bottom in cases:
        field, solve = solve_3d(beta_top, beta_bottom)
        reference_z = analytical_u(np.linspace(0.0, DOMAIN_Z, 51), beta_top, beta_bottom)
        reference = np.broadcast_to(reference_z, field.shape)
        error = field - reference
        metrics = {
            "field_rmse_u": float(np.sqrt(np.mean(error**2))),
            "relative_l2_u": float(np.linalg.norm(error.ravel()) / np.linalg.norm(reference.ravel())),
            "peak_absolute_error_u": float(np.max(np.abs(error))),
            "xy_nonuniformity_peak_to_peak_u": float(np.max(np.ptp(field, axis=(0, 1)))),
        }
        passed = solve["gmres_info"] == 0 and solve["relative_linear_residual"] <= PASS_THRESHOLDS["relative_linear_residual_max"] and all(metrics[key] <= PASS_THRESHOLDS[f"{key}_max"] for key in metrics)
        rows.append({"beta_top": beta_top, "beta_bottom": beta_bottom, "h_top": K/beta_top, "h_bottom": K/beta_bottom, "status": "PASS" if passed else "FAIL", "solve": solve, "metrics": metrics})
    overall = all(row["status"] == "PASS" for row in rows)
    receipt = {
        "schema_version": "heat3d_v7_g2_p5_multi_htc_analytical_gate_v1", "status": "PASS" if overall else "FAIL_CLOSED_NO_FORMAL_LABELS",
        "physics": {"equations": {"outside": "u''=0", "source": "0.2*u''+1=0"}, "source_z": [SOURCE_START, SOURCE_END], "side": "adiabatic", "bottom": "u-0.2-beta_bottom*u_z=0", "top": "u-0.2+beta_top*u_z=0", "parameter": "beta=k_Robin; physical h=conductivity/beta", "conductivity": K},
        "independence": "closed_form_six_coefficient_piecewise_solution_vs_51x51x51_sparse_finite_volume_difference_system",
        "thresholds_frozen_before_gate": PASS_THRESHOLDS, "cases": rows,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pyamg": pyamg.__version__, "device": "Mac_CPU", "peak_rss_bytes": peak_rss_bytes()},
        "formal_labels_generated": 0,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(receipt, indent=2, sort_keys=True)); return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
