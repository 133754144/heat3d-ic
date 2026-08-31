#!/usr/bin/env python3
"""Fail-closed fidelity gate for the DeepOHeat-v1 volumetric reference solver.

The matrix and right-hand side reproduce the finite-difference system in the
official ``hybrid_solver.ipynb`` at xlyu0127/DeepOHeat-v1 commit
3ef3d9c41666a56b5940b39a61166ccaa5aaedb2.  This script is evaluation-only:
it accepts only the released volumetric *test* input/reference arrays, solves
one to three selected cases, and writes a compact JSON receipt under /tmp or a
caller-selected documentation path.  It never opens P1i data and never creates
training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

try:
    import pyamg
except ImportError as exc:  # pragma: no cover - exercised by CLI environment
    raise SystemExit(
        "pyamg is required; install it into an isolated /tmp dependency path"
    ) from exc


UPSTREAM_SHA = "3ef3d9c41666a56b5940b39a61166ccaa5aaedb2"
OFFICIAL_FS_TEST_SHA256 = "4aa308609313d9faf861ad9eb18833e58d03cc727fd05a4433e78ba8b4137d9a"
OFFICIAL_U_TEST_SHA256 = "9f912419e8d00d87c1782a2642cc13717898c98eeff4452b03dc701ad251a07f"

# Frozen before examining solver-vs-reference accuracy.  These are deliberately
# stricter than a plotting-level match while allowing small differences between
# the official GPU/CuPy notebook and this CPU/SciPy solve.
PASS_THRESHOLDS = {
    "relative_l2_u_max": 5.0e-3,
    "rmse_deltaT_K_max": 5.0e-2,
    "peak_absolute_error_deltaT_K_max": 2.5e-1,
    "temperature_range_endpoint_error_K_max": 2.5e-1,
    "relative_linear_residual_max": 1.0e-8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


class OfficialVolumetricFDSolver:
    """CPU realization of the official DeepOHeat-v1 hybrid-solver matrix."""

    def __init__(self, nx: int = 101, ny: int = 101, nz: int = 56) -> None:
        if (nx, ny, nz) != (101, 101, 56):
            raise ValueError("formal fidelity gate requires the official 101x101x56 mesh")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self.dz = 0.55 / (nz - 1)
        build_start = time.perf_counter()
        self.A = self._build_matrix()
        self.matrix_build_seconds = time.perf_counter() - build_start
        amg_start = time.perf_counter()
        self.ml = pyamg.smoothed_aggregation_solver(
            self.A,
            max_levels=10,
            max_coarse=200,
            aggregate="standard",
            strength="symmetric",
            smooth=("jacobi", {"omega": 4 / 3.0}),
            presmoother=None,
            postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
            keep=False,
        )
        self.amg_setup_seconds = time.perf_counter() - amg_start
        self.M = LinearOperator(self.A.shape, matvec=self._amg_vcycle, dtype=np.float64)

    @staticmethod
    def _k_at_z(z: float) -> float:
        if abs(z - 0.1) < 1.0e-10:
            return 2 * 0.1 * 2.0 / (0.1 + 2.0)
        if z < 0.1:
            return 2.0
        return 0.1

    def _build_matrix(self) -> sparse.csr_matrix:
        nx, ny, nz = self.nx, self.ny, self.nz
        nxy = nx * ny
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        values: list[np.ndarray] = []
        i_grid = np.tile(np.arange(nx, dtype=np.int64), ny)
        j_grid = np.repeat(np.arange(ny, dtype=np.int64), nx)
        layer_offset = np.arange(nxy, dtype=np.int64)

        def add(row: np.ndarray, col: np.ndarray, value: np.ndarray) -> None:
            rows.append(np.asarray(row, dtype=np.int64))
            cols.append(np.asarray(col, dtype=np.int64))
            values.append(np.asarray(value, dtype=np.float64))

        for k_index in range(nz):
            base = k_index * nxy
            row = base + layer_offset
            if k_index == 0:
                add(row, row, np.full(nxy, 1.0 + 40.0 / self.dz))
                add(row, row + nxy, np.full(nxy, -40.0 / self.dz))
                continue
            if k_index == nz - 1:
                add(row, row, np.full(nxy, 1.0 + 2.0 / self.dz))
                add(row, row - nxy, np.full(nxy, -2.0 / self.dz))
                continue

            k_value = self._k_at_z(k_index * self.dz)
            diag = np.full(nxy, -2.0 * k_value / self.dz**2)

            left = i_grid == 0
            right = i_grid == nx - 1
            middle_x = ~(left | right)
            diag[left | right] += -k_value / self.dx**2
            diag[middle_x] += -2.0 * k_value / self.dx**2
            add(row[left], row[left] + 1, np.full(left.sum(), k_value / self.dx**2))
            add(row[right], row[right] - 1, np.full(right.sum(), k_value / self.dx**2))
            add(row[middle_x], row[middle_x] - 1, np.full(middle_x.sum(), k_value / self.dx**2))
            add(row[middle_x], row[middle_x] + 1, np.full(middle_x.sum(), k_value / self.dx**2))

            front = j_grid == 0
            back = j_grid == ny - 1
            middle_y = ~(front | back)
            diag[front | back] += -k_value / self.dy**2
            diag[middle_y] += -2.0 * k_value / self.dy**2
            add(row[front], row[front] + nx, np.full(front.sum(), k_value / self.dy**2))
            add(row[back], row[back] - nx, np.full(back.sum(), k_value / self.dy**2))
            add(row[middle_y], row[middle_y] - nx, np.full(middle_y.sum(), k_value / self.dy**2))
            add(row[middle_y], row[middle_y] + nx, np.full(middle_y.sum(), k_value / self.dy**2))

            add(row, row - nxy, np.full(nxy, k_value / self.dz**2))
            add(row, row + nxy, np.full(nxy, k_value / self.dz**2))
            add(row, row, diag)

        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nx * ny * nz, nx * ny * nz),
            dtype=np.float64,
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    def build_rhs(self, q_v: np.ndarray) -> np.ndarray:
        if q_v.shape != (self.nx, self.ny):
            raise ValueError(f"expected q shape {(self.nx, self.ny)}, got {q_v.shape}")
        nxy = self.nx * self.ny
        rhs = np.zeros(self.nx * self.ny * self.nz, dtype=np.float64)
        flat = np.asarray(q_v, dtype=np.float64).ravel(order="C")
        for k_index in range(self.nz):
            start = k_index * nxy
            z = k_index * self.dz
            if k_index in (0, self.nz - 1):
                rhs[start : start + nxy] = 0.2
            elif abs(z - 0.1) < 1.0e-10 or abs(z - 0.15) < 1.0e-10:
                rhs[start : start + nxy] = -flat
            elif 0.1 < z < 0.15:
                rhs[start : start + nxy] = -2.0 * flat
        return rhs

    def _amg_vcycle(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(self.ml.solve(vector, tol=1.0e-1, maxiter=1), dtype=np.float64)

    def solve(self, q_v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        rhs = self.build_rhs(q_v)
        iteration_residuals: list[float] = []
        start = time.perf_counter()
        solution, info = gmres(
            self.A,
            rhs,
            M=self.M,
            rtol=1.0e-10,
            atol=0.0,
            restart=200,
            maxiter=200,
            callback=lambda value: iteration_residuals.append(float(value)),
            callback_type="pr_norm",
        )
        wall = time.perf_counter() - start
        residual = np.linalg.norm(rhs - self.A @ solution) / np.linalg.norm(rhs)
        # This is the exact reshape/transpose used by the official notebook.
        field = solution.reshape((self.nz, self.nx, self.ny)).transpose((1, 2, 0))
        return field, {
            "gmres_info": int(info),
            "iterations": len(iteration_residuals),
            "final_preconditioned_residual_callback": (
                iteration_residuals[-1] if iteration_residuals else None
            ),
            "relative_linear_residual": float(residual),
            "wall_seconds": wall,
        }


def spatial_error_pattern(error_delta_t: np.ndarray) -> dict[str, Any]:
    layer_rmse = np.sqrt(np.mean(np.square(error_delta_t), axis=(0, 1)))
    abs_error = np.abs(error_delta_t)
    peak_index = np.unravel_index(int(np.argmax(abs_error)), abs_error.shape)
    return {
        "layer_rmse_K": layer_rmse.tolist(),
        "max_layer_rmse_K": float(layer_rmse.max()),
        "max_layer_rmse_z_index": int(np.argmax(layer_rmse)),
        "bottom_region_rmse_K_z0_to_9": float(np.sqrt(np.mean(error_delta_t[:, :, :10] ** 2))),
        "active_region_rmse_K_z10_to_15": float(np.sqrt(np.mean(error_delta_t[:, :, 10:16] ** 2))),
        "top_region_rmse_K_z16_to_55": float(np.sqrt(np.mean(error_delta_t[:, :, 16:] ** 2))),
        "peak_absolute_error_index_xyz": [int(value) for value in peak_index],
    }


def compare_case(pred_u: np.ndarray, reference_u: np.ndarray) -> dict[str, Any]:
    pred_u = np.asarray(pred_u, dtype=np.float64)
    reference_u = np.asarray(reference_u, dtype=np.float64)
    error_u = pred_u - reference_u
    error_delta_t = 25.0 * error_u
    pred_delta_t = 25.0 * pred_u
    ref_delta_t = 25.0 * reference_u
    return {
        "field_rmse_u": float(np.sqrt(np.mean(error_u**2))),
        "field_rmse_deltaT_K": float(np.sqrt(np.mean(error_delta_t**2))),
        "relative_l2_u": float(np.linalg.norm(error_u.ravel()) / np.linalg.norm(reference_u.ravel())),
        "peak_absolute_error_u": float(np.max(np.abs(error_u))),
        "peak_absolute_error_deltaT_K": float(np.max(np.abs(error_delta_t))),
        "reference_u_range": [float(reference_u.min()), float(reference_u.max())],
        "prediction_u_range": [float(pred_u.min()), float(pred_u.max())],
        "reference_temperature_K_range": [float(293.15 + ref_delta_t.min()), float(293.15 + ref_delta_t.max())],
        "prediction_temperature_K_range": [float(293.15 + pred_delta_t.min()), float(293.15 + pred_delta_t.max())],
        "temperature_range_endpoint_error_K": float(
            max(
                abs(pred_delta_t.min() - ref_delta_t.min()),
                abs(pred_delta_t.max() - ref_delta_t.max()),
            )
        ),
        "spatial_error_pattern": spatial_error_pattern(error_delta_t),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-test", type=Path, required=True)
    parser.add_argument("--u-test", type=Path, required=True)
    parser.add_argument("--case-index", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = args.case_index
    if not 1 <= len(indices) <= 3 or len(set(indices)) != len(indices):
        raise ValueError("provide one to three unique case indices")
    if any(index < 0 or index >= 100 for index in indices):
        raise ValueError("case indices must be within the official 100-test release")
    if sha256(args.fs_test) != OFFICIAL_FS_TEST_SHA256:
        raise ValueError("official fs_test_volume.npy SHA mismatch")
    if sha256(args.u_test) != OFFICIAL_U_TEST_SHA256:
        raise ValueError("official u_test_volume.npy SHA mismatch")

    fs_test = np.load(args.fs_test, mmap_mode="r", allow_pickle=False)
    u_test = np.load(args.u_test, mmap_mode="r", allow_pickle=False)
    solver = OfficialVolumetricFDSolver()
    cases: list[dict[str, Any]] = []
    for index in indices:
        prediction, solve = solver.solve(np.asarray(fs_test[index], dtype=np.float64))
        comparison = compare_case(prediction, np.asarray(u_test[index], dtype=np.float64))
        case_pass = (
            solve["gmres_info"] == 0
            and solve["relative_linear_residual"] <= PASS_THRESHOLDS["relative_linear_residual_max"]
            and comparison["relative_l2_u"] <= PASS_THRESHOLDS["relative_l2_u_max"]
            and comparison["field_rmse_deltaT_K"] <= PASS_THRESHOLDS["rmse_deltaT_K_max"]
            and comparison["peak_absolute_error_deltaT_K"] <= PASS_THRESHOLDS["peak_absolute_error_deltaT_K_max"]
            and comparison["temperature_range_endpoint_error_K"] <= PASS_THRESHOLDS["temperature_range_endpoint_error_K_max"]
        )
        cases.append({"case_index": index, "status": "PASS" if case_pass else "FAIL", "solve": solve, "comparison": comparison})

    overall = all(case["status"] == "PASS" for case in cases)
    receipt = {
        "schema_version": "heat3d_v7_g2_p4_deepoheat_v1_solver_fidelity_v1",
        "status": "PASS" if overall else "FAIL_CLOSED_NO_FORMAL_TRAIN_LABELS",
        "upstream": f"xlyu0127/DeepOHeat-v1@{UPSTREAM_SHA}",
        "official_files": {
            "fs_test_volume_sha256": OFFICIAL_FS_TEST_SHA256,
            "u_test_volume_sha256": OFFICIAL_U_TEST_SHA256,
        },
        "solver": {
            "implementation": "official_hybrid_solver_notebook_matrix_CPU_port",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyamg": pyamg.__version__,
            "device": "Mac_CPU",
            "mesh": [101, 101, 56],
            "spacing": [solver.dx, solver.dy, solver.dz],
            "matrix_shape": list(solver.A.shape),
            "matrix_nnz": int(solver.A.nnz),
            "matrix_build_seconds": solver.matrix_build_seconds,
            "amg_setup_seconds": solver.amg_setup_seconds,
            "linear_tolerance": {"rtol": 1.0e-10, "atol": 0.0, "restart": 200, "maxiter": 200},
            "peak_process_rss_bytes": peak_rss_bytes(),
            "output_values_per_case": 101 * 101 * 56,
            "output_bytes_per_case_float64": 101 * 101 * 56 * 8,
        },
        "physics": {
            "geometry": "unit x/y, z in [0,0.55]",
            "piecewise_k": "k=2 below z=0.1; harmonic 2*0.1*2/(0.1+2) at z=0.1; k=0.1 above",
            "volumetric_q": "q at z=0.1 and 0.15, 2q for 0.1<z<0.15, zero elsewhere",
            "top_robin": "u_top - 0.2 + 2*du_dz_top = 0",
            "bottom_robin": "u_bottom - 0.2 - 40*du_dz_bottom = 0",
            "side_bc": "first-order zero-normal-gradient adiabatic",
            "temperature_scaling": "T_K=293.15+25*u; with Robin ambient u=0.2, Heat3D T_ref=298.15 K and deltaT_K=25*(u-0.2)",
        },
        "pass_thresholds_frozen_before_accuracy": PASS_THRESHOLDS,
        "cases": cases,
        "formal_train_labels_generated": False,
        "p1i_test_or_sealed_access": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
