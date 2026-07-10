#!/usr/bin/env python3
"""Check V4 P3a problem extraction without solving or writing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v4_reference_solver import (  # noqa: E402
    NODE_ORDERING,
    SolverOptions,
    extract_problem_from_arrays,
    operator_meta_for_problem,
)


def _coords() -> np.ndarray:
    xs = np.array([0.0, 0.005, 0.01], dtype=np.float64)
    ys = np.array([0.0, 0.005, 0.01], dtype=np.float64)
    zs = np.array([0.0, 0.001, 0.002], dtype=np.float64)
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def _meta() -> dict[str, Any]:
    return {
        "schema_version": "v4_p3a_problem_extraction_synthetic",
        "sample_id": "synthetic_problem_extraction",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": 1000.0, "ambient_temperature_K": 300.0},
            "bottom": {"fixed_temperature_K": 300.0},
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


def _synthetic_case(
    k_mode: str,
    *,
    duplicate_q_values: tuple[float, float] = (5.0, 11.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coords = _coords()
    center = np.array([[0.005, 0.005, 0.001]], dtype=np.float64)
    center_idx = int(np.where(np.all(np.isclose(coords, center), axis=1))[0][0])
    coords = np.vstack([coords, center])

    q_field = np.zeros((coords.shape[0], 1), dtype=np.float64)
    q_field[center_idx, 0] = duplicate_q_values[0]
    q_field[-1, 0] = duplicate_q_values[1]

    if k_mode == "isotropic":
        k_field = np.full((coords.shape[0], 1), 10.0, dtype=np.float64)
        k_field[center_idx, 0] = 10.0
        k_field[-1, 0] = 30.0
    elif k_mode == "diag3":
        k_field = np.empty((coords.shape[0], 3), dtype=np.float64)
        k_field[:] = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        k_field[center_idx] = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        k_field[-1] = np.array([4.0, 8.0, 10.0], dtype=np.float64)
    else:
        raise ValueError(f"unsupported synthetic k_mode: {k_mode}")

    return coords, k_field, q_field, _meta()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_case(
    k_mode: str,
    *,
    duplicate_q_values: tuple[float, float] = (5.0, 11.0),
    expected_duplicate_q: float = 11.0,
) -> dict[str, Any]:
    coords, k_field, q_field, meta = _synthetic_case(
        k_mode,
        duplicate_q_values=duplicate_q_values,
    )
    problem = extract_problem_from_arrays(
        coords=coords,
        k_field=k_field,
        q_field=q_field,
        sample_meta=meta,
    )
    operator_meta = operator_meta_for_problem(problem, SolverOptions())
    grid = problem.grid_mapping.grid
    center_idx = int(grid[1, 1, 1])

    _expect(problem.grid_spec.grid_shape == (3, 3, 3), "grid shape mismatch")
    _expect(problem.grid_spec.node_count == 27, "unique node count mismatch")
    _expect(problem.grid_mapping.original_to_unique.shape == (28,), "original mapping mismatch")
    _expect(problem.grid_mapping.original_to_unique[-1] == center_idx, "duplicate inverse mismatch")
    _expect(problem.grid_mapping.duplicate_counts[center_idx] == 2, "duplicate count mismatch")
    _expect(problem.grid_mapping.node_ordering == NODE_ORDERING, "node ordering label mismatch")
    _expect(int(grid[0, 0, 0]) == 0, "grid origin ordering mismatch")
    _expect(int(grid[2, 2, 2]) == 26, "grid end ordering mismatch")

    boundary = problem.boundary
    _expect(boundary.top_node_indices.size == 9, "top face count mismatch")
    _expect(boundary.bottom_node_indices.size == 9, "bottom face count mismatch")
    _expect(boundary.side_node_indices.size == 24, "side face count mismatch")
    _expect(np.array_equal(boundary.interior_node_indices, np.array([center_idx])), "interior mismatch")

    if k_mode == "isotropic":
        _expect(problem.supported_k_mode == "isotropic_expanded_to_diag3", "isotropic mode mismatch")
        _expect(np.allclose(problem.k_diag[center_idx], np.array([20.0, 20.0, 20.0])), "iso k merge")
    else:
        _expect(problem.supported_k_mode == "diag3", "diag3 mode mismatch")
        _expect(np.allclose(problem.k_diag[center_idx], np.array([3.0, 6.0, 8.0])), "diag3 k merge")

    _expect(
        float(problem.q_field[center_idx, 0]) == expected_duplicate_q,
        "q max pooling mismatch",
    )
    _expect(problem.duplicate_merge["merged_duplicate_count"] == 1, "merge metadata mismatch")
    _expect(problem.duplicate_merge["duplicate_unique_indices"] == [center_idx], "duplicate index metadata")

    _expect(len(problem.interfaces) == 1, "interface count mismatch")
    interface = problem.interfaces[0]
    _expect(interface.interface_type == "perfect_contact", "interface type mismatch")
    _expect(interface.adjacent_layer_ids == (0, 1), "interface adjacent layers mismatch")
    _expect(interface.contact_resistance_m2K_W is None, "unexpected contact resistance")
    _expect(np.array_equal(interface.duplicate_unique_indices, np.array([center_idx])), "interface duplicate map")

    _expect(operator_meta.matrix_backend == "not_assembled", "operator backend should be skeleton only")
    _expect(operator_meta.nnz is None, "operator nnz should be unknown before assembly")
    _expect(operator_meta.node_count == problem.grid_spec.node_count, "operator node count mismatch")

    return {
        "k_mode": k_mode,
        "supported_k_mode": problem.supported_k_mode,
        "grid_shape": problem.grid_spec.grid_shape,
        "node_count": problem.grid_spec.node_count,
        "top_nodes": int(boundary.top_node_indices.size),
        "bottom_nodes": int(boundary.bottom_node_indices.size),
        "side_nodes": int(boundary.side_node_indices.size),
        "interior_nodes": int(boundary.interior_node_indices.size),
        "merged_duplicate_count": problem.duplicate_merge["merged_duplicate_count"],
        "duplicate_q": float(problem.q_field[center_idx, 0]),
        "operator_backend": operator_meta.matrix_backend,
    }


def main() -> int:
    print("Heat3D V4 P3a problem extraction check")
    print("scope: interface skeleton only; no sparse solve, no contact solve, no artifact writes")
    summaries = [
        _check_case("isotropic"),
        _check_case("diag3"),
        _check_case(
            "isotropic",
            duplicate_q_values=(-5.0, -2.0),
            expected_duplicate_q=-2.0,
        ),
    ]
    for summary in summaries:
        print(
            "- "
            f"k_mode={summary['k_mode']} "
            f"supported={summary['supported_k_mode']} "
            f"grid_shape={summary['grid_shape']} "
            f"node_count={summary['node_count']} "
            f"top/bottom/side/interior="
            f"{summary['top_nodes']}/{summary['bottom_nodes']}/"
            f"{summary['side_nodes']}/{summary['interior_nodes']} "
            f"duplicates={summary['merged_duplicate_count']} "
            f"duplicate_q={summary['duplicate_q']} "
            f"operator_backend={summary['operator_backend']}"
        )
    print("artifact_writes: false")
    print("p3a_problem_extraction_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
