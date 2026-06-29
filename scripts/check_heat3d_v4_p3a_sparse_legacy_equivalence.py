#!/usr/bin/env python3
"""Check V4 P3a perfect-contact dense/sparse legacy equivalence.

Default scope is an in-memory synthetic tiny gate. With --sample-root, the
checker also reads existing samples and compares V4 sparse output against the
unchanged dense v2 reference solver. It does not write artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_reference_solver_v2 import solve_reference_temperature_v2  # noqa: E402
from rigno.heat3d_v4_reference_solver import (  # noqa: E402
    SolverOptions,
    build_operator,
    extract_problem_from_arrays,
    load_problem_from_sample,
    solve_operator,
    solve_temperature_from_problem,
)


A_TOL = 1e-12
RHS_TOL = 1e-12
T_TOL = 1e-10
DELTA_T_TOL = 1e-10
BOTTOM_TOL = 1e-12
DEFAULT_SAMPLE_IDS = ("sample_000", "sample_005", "sample_008")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Heat3D V4 P3a dense/sparse legacy equivalence checks."
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=None,
        help="Optional existing subset root or samples directory for read-only legacy-v2 comparison.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Sample ids to check when --sample-root is provided.",
    )
    return parser.parse_args()


def _meta() -> dict[str, Any]:
    return {
        "schema_version": "v4_p3a_sparse_equivalence_synthetic",
        "sample_id": "synthetic_sparse_equivalence",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": 1250.0, "ambient_temperature_K": 297.0},
            "bottom": {"fixed_temperature_K": 293.0},
            "sides": {"adiabatic": True},
        },
        "interfaces": [
            {
                "id": "mid_z_perfect_contact",
                "type": "perfect_contact",
                "adjacent_layer_ids": [0, 1],
                "z_position_m": 0.001,
            }
        ],
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _synthetic_problem():
    xs = np.array([0.0, 0.004, 0.01], dtype=np.float64)
    ys = np.array([0.0, 0.006, 0.011], dtype=np.float64)
    zs = np.array([0.0, 0.0007, 0.0017], dtype=np.float64)
    coords = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    k_field = np.empty((coords.shape[0], 3), dtype=np.float64)
    q_field = np.zeros((coords.shape[0], 1), dtype=np.float64)
    for idx, (_, _, z) in enumerate(coords):
        k_field[idx] = np.array([8.0 + 1500.0 * z, 12.0 + 1000.0 * z, 18.0 + 500.0 * z])
        q_field[idx, 0] = 2.0e5 if z > 0.0 else 0.0

    center = np.array([[0.004, 0.006, 0.0007]], dtype=np.float64)
    center_idx = int(np.where(np.all(np.isclose(coords, center), axis=1))[0][0])
    coords = np.vstack([coords, center])
    k_field = np.vstack([k_field, np.array([[10.0, 16.0, 21.0]], dtype=np.float64)])
    q_field = np.vstack([q_field, np.array([[5.0e5]], dtype=np.float64)])
    q_field[center_idx, 0] = 2.5e5

    return extract_problem_from_arrays(
        coords=coords,
        k_field=k_field,
        q_field=q_field,
        sample_meta=_meta(),
    )


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value)))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_synthetic_gate() -> dict[str, float | str]:
    problem = _synthetic_problem()
    dense = build_operator(problem, SolverOptions(matrix_backend="dense"))
    sparse = build_operator(problem, SolverOptions(matrix_backend="sparse_csr"))

    dense_a = np.asarray(dense.matrix, dtype=np.float64)
    sparse_a = sparse.matrix.toarray()
    max_a_diff = _max_abs(dense_a - sparse_a)
    max_rhs_diff = _max_abs(dense.rhs - sparse.rhs)

    dense_t, dense_audit = solve_operator(dense)
    sparse_t, sparse_audit = solve_operator(sparse)
    bottom_t = problem.boundary.bottom_T_fixed_K
    max_t_diff = _max_abs(dense_t - sparse_t)
    max_delta_t_diff = _max_abs((dense_t - bottom_t) - (sparse_t - bottom_t))
    bottom_error = _max_abs(sparse_t[problem.boundary.bottom_node_indices] - bottom_t)

    _expect(max_a_diff <= A_TOL, f"dense A vs sparse.toarray maxdiff too large: {max_a_diff}")
    _expect(max_rhs_diff <= RHS_TOL, f"RHS maxdiff too large: {max_rhs_diff}")
    _expect(max_t_diff <= T_TOL, f"T maxdiff too large: {max_t_diff}")
    _expect(max_delta_t_diff <= DELTA_T_TOL, f"DeltaT maxdiff too large: {max_delta_t_diff}")
    _expect(bottom_error <= BOTTOM_TOL, f"bottom Dirichlet not exact enough: {bottom_error}")
    _expect(
        dense_audit.residual_norm is not None and np.isfinite(dense_audit.residual_norm),
        "dense residual not finite",
    )
    _expect(
        sparse_audit.residual_norm is not None and np.isfinite(sparse_audit.residual_norm),
        "sparse residual not finite",
    )
    return {
        "case": "synthetic_tiny",
        "dense_sparse_A_maxdiff": max_a_diff,
        "dense_sparse_RHS_maxdiff": max_rhs_diff,
        "dense_sparse_T_maxdiff": max_t_diff,
        "dense_sparse_DeltaT_maxdiff": max_delta_t_diff,
        "bottom_dirichlet_error": bottom_error,
        "sparse_residual_norm": float(sparse_audit.residual_norm),
        "dense_residual_norm": float(dense_audit.residual_norm),
        "operator_nnz": float(sparse.meta.nnz or -1),
    }


def _samples_root(path: Path) -> Path:
    if path.name == "samples":
        return path
    return path / "samples"


def _run_sample_gate(sample_dir: Path) -> dict[str, float | str]:
    problem = load_problem_from_sample(sample_dir)
    sparse_temperature, sparse_meta = solve_temperature_from_problem(
        problem,
        SolverOptions(matrix_backend="sparse_csr"),
    )
    legacy_temperature, legacy_meta = solve_reference_temperature_v2(sample_dir)
    bottom_t = problem.boundary.bottom_T_fixed_K

    max_t_diff = _max_abs(sparse_temperature - legacy_temperature)
    max_delta_t_diff = _max_abs(
        (sparse_temperature - bottom_t) - (legacy_temperature - bottom_t)
    )
    bottom_error = float(sparse_meta["solution_audit"]["bottom_dirichlet_error"])
    residual_norm = float(sparse_meta["solution_audit"]["residual_norm"])

    _expect(max_t_diff <= T_TOL, f"{sample_dir.name} legacy v2 T maxdiff too large: {max_t_diff}")
    _expect(
        max_delta_t_diff <= DELTA_T_TOL,
        f"{sample_dir.name} legacy v2 DeltaT maxdiff too large: {max_delta_t_diff}",
    )
    _expect(bottom_error <= BOTTOM_TOL, f"{sample_dir.name} bottom Dirichlet error: {bottom_error}")
    _expect(np.isfinite(residual_norm), f"{sample_dir.name} sparse residual not finite")

    return {
        "case": sample_dir.name,
        "legacy_v2_T_maxdiff": max_t_diff,
        "legacy_v2_DeltaT_maxdiff": max_delta_t_diff,
        "bottom_dirichlet_error": bottom_error,
        "sparse_residual_norm": residual_norm,
        "legacy_residual_norm": float(legacy_meta["residual_norm"]),
        "operator_nnz": float(sparse_meta["operator"]["nnz"]),
    }


def _print_summary(summary: dict[str, float | str]) -> None:
    fields = [f"case={summary['case']}"]
    for key, value in summary.items():
        if key == "case":
            continue
        if isinstance(value, float):
            if key == "operator_nnz":
                fields.append(f"{key}={int(value)}")
            else:
                fields.append(f"{key}={value:.6e}")
        else:
            fields.append(f"{key}={value}")
    print("- " + " ".join(fields))


def main() -> int:
    print("Heat3D V4 P3a sparse legacy equivalence check")
    print("scope: perfect_contact / R_contact=0 only; no generator, no training, no artifact writes")
    failed = False
    try:
        _print_summary(_run_synthetic_gate())
    except Exception as exc:
        failed = True
        print(f"ERROR synthetic_tiny: {exc}")

    args = parse_args()
    if args.sample_root is None:
        print("legacy_v2_sample_compare: skipped (no --sample-root)")
    else:
        root = _samples_root(args.sample_root)
        for sample_id in args.sample_ids:
            sample_dir = root / sample_id
            if not sample_dir.is_dir():
                failed = True
                print(f"ERROR missing sample directory: {sample_dir}")
                continue
            try:
                _print_summary(_run_sample_gate(sample_dir))
            except Exception as exc:
                failed = True
                print(f"ERROR {sample_id}: {exc}")

    print("artifact_writes: false")
    print(f"p3a_sparse_legacy_equivalence_ok: {not failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
