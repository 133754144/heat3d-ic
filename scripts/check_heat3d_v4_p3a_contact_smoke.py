#!/usr/bin/env python3
"""Smoke check for V4 P3a contact-resistance sparse mode.

The check is synthetic and in-memory only. It does not write data, output,
temperature labels, checkpoints, or logs.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v4_reference_solver import (  # noqa: E402
    SolverOptions,
    extract_problem_from_arrays,
    solve_temperature_from_problem,
)


T_TOL = 1e-10
BOTTOM_TOL = 1e-12
ENERGY_TOL_W = 1e-9
R_VALUES = (0.0, 1.0e-7, 5.0e-7, 1.0e-6)
PRODUCTION_CONTACT_MODEL = "R_contact=0_perfect_contact"
FINITE_CONTACT_STATUS = "experimental_smoke_deferred_not_v4_production"
REQUIRED_AUDIT_KEYS = (
    "residual_norm",
    "bottom_dirichlet_error",
    "source_power_total",
    "top_robin_flux_total",
    "bottom_flux_total",
    "energy_balance_residual",
    "operator_checksum",
    "solver_mode",
    "matrix_backend",
)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _coords() -> np.ndarray:
    xs = np.array([0.0, 0.01], dtype=np.float64)
    ys = np.array([0.0, 0.01], dtype=np.float64)
    z_sequence = np.array([0.0, 0.001, 0.001, 0.002], dtype=np.float64)
    return np.array([[x, y, z] for x in xs for y in ys for z in z_sequence], dtype=np.float64)


def _meta(interface_type: str, r_contact: float | None = None) -> dict[str, Any]:
    interface: dict[str, Any] = {
        "id": "mid_contact",
        "type": interface_type,
        "adjacent_layer_ids": [0, 1],
        "z_position_m": 0.001,
    }
    if r_contact is not None:
        interface["R_contact_m2K_W"] = r_contact
    return {
        "schema_version": "v4_p3a_contact_smoke_synthetic",
        "sample_id": "synthetic_contact_smoke",
        "boundary_types": {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"},
        "boundary_params": {
            "top": {"h_W_m2K": 250.0, "ambient_temperature_K": 300.0},
            "bottom": {"fixed_temperature_K": 300.0},
            "sides": {"adiabatic": True},
        },
        "interfaces": [interface],
        "units": {
            "coords": "m",
            "k_field": "W/m/K",
            "q_field": "W/m^3",
            "temperature": "K",
        },
    }


def _fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = _coords()
    k_field = np.full((coords.shape[0], 3), np.array([10.0, 10.0, 20.0]), dtype=np.float64)
    q_field = np.zeros((coords.shape[0], 1), dtype=np.float64)
    q_field[np.isclose(coords[:, 2], 0.002), 0] = 8.0e7
    return coords, k_field, q_field


def _perfect_problem():
    coords, k_field, q_field = _fields()
    return extract_problem_from_arrays(
        coords=coords,
        k_field=k_field,
        q_field=q_field,
        sample_meta=_meta("perfect_contact"),
        duplicate_policy="merge",
    )


def _contact_problem(r_contact: float):
    coords, k_field, q_field = _fields()
    return extract_problem_from_arrays(
        coords=coords,
        k_field=k_field,
        q_field=q_field,
        sample_meta=_meta("contact_resistance", r_contact),
        duplicate_policy="preserve",
    )


def _solve_contact(r_contact: float):
    problem = _contact_problem(r_contact)
    return solve_temperature_from_problem(
        problem,
        SolverOptions(solver_mode="contact_resistance", matrix_backend="sparse_csr"),
    )


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value)))


def _contact_jump(meta: dict[str, Any]) -> float:
    interfaces = meta["contact_interfaces"]
    _expect(len(interfaces) == 1, "expected one contact interface metadata entry")
    entry = interfaces[0]
    for key in (
        "interface_id",
        "R_contact_m2K_W",
        "face_count",
        "flux_lower_to_upper_W",
        "temperature_jump_upper_minus_lower_K",
        "effective_conductance_W_K",
    ):
        _expect(key in entry, f"missing contact metadata key: {key}")
    _expect(entry["interface_id"] == "mid_contact", "interface id mismatch")
    _expect(int(entry["face_count"]) == 4, "contact face count mismatch")
    _expect(np.isfinite(float(entry["flux_lower_to_upper_W"])), "contact flux not finite")
    _expect(
        np.isfinite(float(entry["temperature_jump_upper_minus_lower_K"]["max_abs"])),
        "contact jump not finite",
    )
    _expect(
        float(entry["effective_conductance_W_K"]["mean"]) > 0.0,
        "effective conductance must be positive",
    )
    return float(entry["temperature_jump_upper_minus_lower_K"]["max_abs"])


def _check_solution_audit(
    meta: dict[str, Any],
    *,
    expected_solver_mode: str,
    finite_contact_expected: bool,
) -> None:
    audit = meta["solution_audit"]
    for key in REQUIRED_AUDIT_KEYS:
        _expect(key in audit, f"missing solution_audit key: {key}")
    for key in (
        "residual_norm",
        "bottom_dirichlet_error",
        "source_power_total",
        "top_robin_flux_total",
        "bottom_flux_total",
        "energy_balance_residual",
    ):
        _expect(np.isfinite(float(audit[key])), f"non-finite audit field: {key}")
    _expect(
        abs(float(audit["energy_balance_residual"])) <= ENERGY_TOL_W,
        f"energy balance residual too large: {audit['energy_balance_residual']}",
    )
    _expect(str(audit["operator_checksum"]), "operator checksum is empty")
    _expect(audit["solver_mode"] == expected_solver_mode, "audit solver_mode mismatch")
    _expect(audit["matrix_backend"] == "sparse_csr", "audit matrix_backend mismatch")
    _expect(
        audit["v4_production_contact_model"] == PRODUCTION_CONTACT_MODEL,
        "production contact model must stay R_contact=0",
    )
    expected_status = FINITE_CONTACT_STATUS if finite_contact_expected else "not_enabled"
    _expect(
        audit["finite_contact_resistance_status"] == expected_status,
        "finite contact status mismatch",
    )


def _run_gate() -> dict[str, float]:
    perfect_problem = _perfect_problem()
    perfect_t, perfect_meta = solve_temperature_from_problem(
        perfect_problem,
        SolverOptions(solver_mode="perfect_contact", matrix_backend="sparse_csr"),
    )
    _check_solution_audit(
        perfect_meta,
        expected_solver_mode="perfect_contact",
        finite_contact_expected=False,
    )
    contact_t0, contact_meta0 = _solve_contact(0.0)
    r0_diff = _max_abs(contact_t0 - perfect_t)
    bottom_error = float(contact_meta0["solution_audit"]["bottom_dirichlet_error"])
    residual = float(contact_meta0["solution_audit"]["residual_norm"])
    _check_solution_audit(
        contact_meta0,
        expected_solver_mode="contact_resistance",
        finite_contact_expected=False,
    )

    _expect(r0_diff <= T_TOL, f"R=0 contact vs perfect_contact maxdiff too large: {r0_diff}")
    _expect(bottom_error <= BOTTOM_TOL, f"bottom Dirichlet error too large: {bottom_error}")
    _expect(np.isfinite(residual), "contact residual not finite")
    _expect(np.all(np.isfinite(contact_t0)), "contact temperature contains NaN/Inf")
    _expect(np.isfinite(float(perfect_meta["solution_audit"]["residual_norm"])), "perfect residual not finite")

    jumps = []
    for r_contact in R_VALUES:
        contact_t, contact_meta = _solve_contact(r_contact)
        _check_solution_audit(
            contact_meta,
            expected_solver_mode="contact_resistance",
            finite_contact_expected=r_contact > 0.0,
        )
        _expect(np.all(np.isfinite(contact_t)), f"contact temperature has NaN/Inf for R={r_contact}")
        _expect(
            np.isfinite(float(contact_meta["solution_audit"]["residual_norm"])),
            f"contact residual not finite for R={r_contact}",
        )
        _expect(
            float(contact_meta["solution_audit"]["bottom_dirichlet_error"]) <= BOTTOM_TOL,
            f"bottom Dirichlet error too large for R={r_contact}",
        )
        jumps.append(_contact_jump(contact_meta))

    for before, after in zip(jumps, jumps[1:]):
        _expect(after >= before - 1.0e-12, f"contact jump not monotonic: {jumps}")

    return {
        "r0_contact_vs_perfect_T_maxdiff": r0_diff,
        "r0_bottom_dirichlet_error": bottom_error,
        "r0_residual_norm": residual,
        "jump_R0": jumps[0],
        "jump_R1e_7": jumps[1],
        "jump_R5e_7": jumps[2],
        "jump_R1e_6": jumps[3],
        "r0_energy_balance_residual": float(contact_meta0["solution_audit"]["energy_balance_residual"]),
    }


def main() -> int:
    print("Heat3D V4 P3a contact-resistance smoke check")
    print("scope: synthetic in-memory only; sparse contact mode; no artifact writes")
    try:
        summary = _run_gate()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("artifact_writes: false")
        print("p3a_contact_smoke_ok: false")
        return 1
    print(
        "- "
        f"R0_contact_vs_perfect_T_maxdiff={summary['r0_contact_vs_perfect_T_maxdiff']:.6e} "
        f"R0_bottom_error={summary['r0_bottom_dirichlet_error']:.6e} "
        f"R0_residual={summary['r0_residual_norm']:.6e} "
        f"R0_energy_residual={summary['r0_energy_balance_residual']:.6e} "
        f"jump_R0/R1e-7/R5e-7/R1e-6="
        f"{summary['jump_R0']:.6e}/"
        f"{summary['jump_R1e_7']:.6e}/"
        f"{summary['jump_R5e_7']:.6e}/"
        f"{summary['jump_R1e_6']:.6e}"
    )
    print("artifact_writes: false")
    print("p3a_contact_smoke_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
