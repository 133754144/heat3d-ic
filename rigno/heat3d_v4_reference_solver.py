"""Heat3D V4 reference-solver problem contract.

This module is the P3a interface layer for the sparse-equivalent solver
refactor. It extracts the current dense-v2 problem semantics into stable
dataclasses and, for P3a-2, can assemble/solve the perfect-contact
legacy-equivalent operator without changing the legacy dense solver output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROBLEM_CONTRACT_VERSION = "heat3d_v4_p3a_problem_contract_0"
SOLVER_FAMILY = "heat3d_v4_reference_solver"
LEGACY_EQUIVALENCE_TARGET = "rigno.heat3d_v1_reference_solver_v2"
SUPPORTED_BOUNDARY_TYPES = {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"}
K_MERGE_POLICY = "arithmetic_mean_on_duplicate_coordinates_before_face_harmonic_means"
Q_MERGE_POLICY = "max_preserves_active_source_when_duplicate_interface_nodes_exist"
NODE_ORDERING = "np_unique_axis0_lexicographic"
CONTACT_NODE_ORDERING = "lexicographic_xyz_with_duplicate_interface_order_preserved"
CONTACT_DUPLICATE_POLICY = "preserve_contact_duplicates_without_k_q_merge"


@dataclass(frozen=True)
class GridSpec:
    """Rectilinear grid axes and control-volume widths."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    grid_shape: tuple[int, int, int]
    node_count: int
    coordinate_unit: str = "m"
    coordinate_system: str = "rectilinear_xyz"


@dataclass(frozen=True)
class GridMapping:
    """Stable mapping between original sample nodes and merged grid nodes."""

    grid: np.ndarray
    original_to_unique: np.ndarray
    unique_to_first_original: np.ndarray
    duplicate_counts: np.ndarray
    node_ordering: str = NODE_ORDERING
    complete_rectilinear_grid: bool = True


@dataclass(frozen=True)
class BoundarySpec:
    """Current dense-v2 boundary-condition contract."""

    boundary_types: dict[str, str]
    top_h_W_m2K: float
    top_T_inf_K: float
    bottom_T_fixed_K: float
    side_policy: str
    top_node_indices: np.ndarray
    bottom_node_indices: np.ndarray
    side_node_indices: np.ndarray
    interior_node_indices: np.ndarray


@dataclass(frozen=True)
class InterfaceRecord:
    """Interface metadata prepared for future contact-resistance support."""

    interface_id: str
    interface_type: str
    adjacent_layer_ids: tuple[int, int] | None
    z_position_m: float | None
    contact_resistance_m2K_W: float | None
    duplicate_unique_indices: np.ndarray
    contact_pair_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64)
    )
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorMeta:
    """Operator contract metadata for unassembled or assembled operators."""

    contract_version: str
    solver_family: str
    solver_mode: str
    matrix_backend: str
    sparse_format: str | None
    node_ordering: str
    grid_shape: tuple[int, int, int]
    node_count: int
    nnz: int | None
    operator_checksum: str | None
    legacy_equivalence_target: str


@dataclass(frozen=True)
class AssembledOperator:
    """Assembled linear operator plus the exact triplets used to build it."""

    matrix: Any
    rhs: np.ndarray
    row: np.ndarray
    col: np.ndarray
    data: np.ndarray
    meta: OperatorMeta
    assembly: dict[str, Any]


@dataclass(frozen=True)
class SolutionAudit:
    """Placeholder audit contract for later sparse solve results."""

    residual_norm: float | None = None
    energy_balance_residual: float | None = None
    top_robin_flux_residual: float | None = None
    side_adiabatic_flux_residual: float | None = None
    interface_flux_mismatch: float | None = None
    contact_temperature_jump_residual: float | None = None
    status: str = "not_solved"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SolverOptions:
    """P3a options accepted by the future sparse solver path."""

    solver_mode: str = "legacy_equivalent"
    matrix_backend: str = "not_assembled"
    sparse_format: str | None = None
    residual_tolerance: float = 1e-8
    equivalence_tolerance: float = 1e-10
    contact_enabled: bool = False


@dataclass(frozen=True)
class Heat3DProblem:
    """Extracted Heat3D steady-problem inputs under the dense-v2 semantics."""

    contract_version: str
    sample_dir: str | None
    sample_meta: dict[str, Any]
    coords_original: np.ndarray
    k_field_original: np.ndarray
    q_field_original: np.ndarray
    coords: np.ndarray
    k_diag: np.ndarray
    q_field: np.ndarray
    supported_k_mode: str
    grid_spec: GridSpec
    grid_mapping: GridMapping
    boundary: BoundarySpec
    interfaces: tuple[InterfaceRecord, ...]
    duplicate_merge: dict[str, Any]
    warnings: tuple[str, ...]


def load_problem_from_sample(sample_dir: str | Path) -> Heat3DProblem:
    """Load an existing sample directory and extract the V4 problem contract."""

    sample_path = Path(sample_dir)
    coords = np.load(sample_path / "coords.npy")
    k_field = np.load(sample_path / "k_field.npy")
    q_field = np.load(sample_path / "q_field.npy")
    meta = json.loads((sample_path / "sample_meta.json").read_text())
    return extract_problem_from_arrays(
        coords=coords,
        k_field=k_field,
        q_field=q_field,
        sample_meta=meta,
        sample_dir=sample_path,
    )


def extract_problem_from_arrays(
    *,
    coords: np.ndarray,
    k_field: np.ndarray,
    q_field: np.ndarray,
    sample_meta: dict[str, Any],
    sample_dir: str | Path | None = None,
    duplicate_policy: str = "merge",
) -> Heat3DProblem:
    """Extract a problem contract without solving or writing artifacts."""

    coords = _as_float_array(coords, "coords")
    k_field = _as_float_array(k_field, "k_field")
    q_field = _as_float_array(q_field, "q_field")
    _validate_shapes(coords, k_field, q_field)
    if duplicate_policy not in {"merge", "preserve"}:
        raise ValueError(f"unsupported duplicate_policy: {duplicate_policy}")
    warnings = _validate_supported_problem(sample_meta, k_field, duplicate_policy)

    k_diag, supported_k_mode = _expand_k(k_field)
    if duplicate_policy == "merge":
        merged = _merge_duplicate_points(coords, k_diag, q_field)
        grid_spec, grid_mapping = _grid_contract(
            merged["coords"],
            original_to_unique=merged["inverse"],
            unique_to_first_original=merged["unique_to_first_original"],
            duplicate_counts=merged["duplicate_counts"],
        )
    else:
        merged = _preserve_duplicate_points(coords, k_diag, q_field)
        grid_spec, grid_mapping = _contact_grid_contract(
            merged["coords"],
            original_to_unique=merged["inverse"],
            unique_to_first_original=merged["unique_to_first_original"],
            duplicate_counts=merged["duplicate_counts"],
        )
    boundary = _boundary_contract(sample_meta, merged["coords"])
    interfaces = _interface_records(
        sample_meta,
        grid_mapping,
        contact_pair_indices=merged["metadata"].get("contact_pair_indices", []),
        contact_pair_z=merged["metadata"].get("contact_pair_z_m", []),
    )

    sample_dir_text = str(Path(sample_dir)) if sample_dir is not None else None
    return Heat3DProblem(
        contract_version=PROBLEM_CONTRACT_VERSION,
        sample_dir=sample_dir_text,
        sample_meta=dict(sample_meta),
        coords_original=coords,
        k_field_original=k_field,
        q_field_original=q_field,
        coords=merged["coords"],
        k_diag=merged["k_diag"],
        q_field=merged["q_field"],
        supported_k_mode=supported_k_mode,
        grid_spec=grid_spec,
        grid_mapping=grid_mapping,
        boundary=boundary,
        interfaces=tuple(interfaces),
        duplicate_merge=merged["metadata"],
        warnings=tuple(warnings),
    )


def operator_meta_for_problem(
    problem: Heat3DProblem,
    options: SolverOptions | None = None,
) -> OperatorMeta:
    """Return the P3a operator contract metadata without assembling an operator."""

    options = options or SolverOptions()
    return OperatorMeta(
        contract_version=problem.contract_version,
        solver_family=SOLVER_FAMILY,
        solver_mode=options.solver_mode,
        matrix_backend=options.matrix_backend,
        sparse_format=options.sparse_format,
        node_ordering=problem.grid_mapping.node_ordering,
        grid_shape=problem.grid_spec.grid_shape,
        node_count=problem.grid_spec.node_count,
        nnz=None,
        operator_checksum=None,
        legacy_equivalence_target=LEGACY_EQUIVALENCE_TARGET,
    )


def build_operator(
    problem: Heat3DProblem,
    options: SolverOptions | None = None,
    *,
    matrix_backend: str | None = None,
) -> AssembledOperator:
    """Build the current perfect-contact linear operator.

    Dense and sparse backends share the same row/col/data/RHS assembly. This is
    intentionally limited to the dense-v2 physics: q*control-volume source,
    harmonic face conductance, top Robin, bottom Dirichlet row replacement, and
    natural side adiabatic boundaries.
    """

    options = options or SolverOptions()
    backend = matrix_backend or options.matrix_backend
    if backend == "not_assembled":
        backend = "sparse_csr"
    if backend not in {"dense", "sparse_csr"}:
        raise ValueError(f"unsupported matrix_backend for P3a-2: {backend}")
    _validate_assembly_options(problem, options)

    row, col, data, rhs, assembly_meta = _assemble_triplets(problem, options)
    n = problem.grid_spec.node_count
    if backend == "dense":
        matrix = np.zeros((n, n), dtype=np.float64)
        np.add.at(matrix, (row, col), data)
        sparse_format = None
        nnz = int(np.count_nonzero(matrix))
    else:
        from scipy.sparse import csr_matrix

        matrix = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)
        matrix.sum_duplicates()
        sparse_format = "csr"
        nnz = int(matrix.nnz)

    checksum = _operator_checksum(row, col, data, rhs, n)
    meta = OperatorMeta(
        contract_version=problem.contract_version,
        solver_family=SOLVER_FAMILY,
        solver_mode=options.solver_mode,
        matrix_backend=backend,
        sparse_format=sparse_format,
        node_ordering=problem.grid_mapping.node_ordering,
        grid_shape=problem.grid_spec.grid_shape,
        node_count=problem.grid_spec.node_count,
        nnz=nnz,
        operator_checksum=checksum,
        legacy_equivalence_target=LEGACY_EQUIVALENCE_TARGET,
    )
    assembly = dict(assembly_meta)
    assembly.update(
        {
            "matrix_backend": backend,
            "sparse_format": sparse_format,
            "operator_checksum": checksum,
            "nnz": nnz,
        }
    )
    return AssembledOperator(
        matrix=matrix,
        rhs=rhs,
        row=row,
        col=col,
        data=data,
        meta=meta,
        assembly=assembly,
    )


def solve_operator(operator: AssembledOperator) -> tuple[np.ndarray, SolutionAudit]:
    """Solve an assembled operator and return unique-node temperature."""

    try:
        if operator.meta.matrix_backend == "dense":
            temperature = np.linalg.solve(operator.matrix, operator.rhs)
        elif operator.meta.matrix_backend == "sparse_csr":
            from scipy.sparse.linalg import spsolve

            temperature = spsolve(operator.matrix, operator.rhs)
        else:
            raise ValueError(f"unsupported matrix_backend: {operator.meta.matrix_backend}")
        solve_error = None
    except Exception as exc:  # pragma: no cover - exercised by future failure tests.
        temperature = np.full((operator.rhs.shape[0],), np.nan, dtype=np.float64)
        solve_error = str(exc)

    temperature = np.asarray(temperature, dtype=np.float64).reshape(-1)
    residual = operator.matrix.dot(temperature) - operator.rhs
    residual_norm = float(np.linalg.norm(residual) / max(float(np.linalg.norm(operator.rhs)), 1.0))
    finite = bool(np.all(np.isfinite(temperature)) and np.isfinite(residual_norm))
    warnings: list[str] = []
    status = "solved" if solve_error is None and finite else "solve_failed"
    if solve_error is not None:
        warnings.append(f"linear solve failed: {solve_error}")
    if not finite:
        warnings.append("temperature or residual contains NaN or Inf")
    return temperature, SolutionAudit(
        residual_norm=residual_norm,
        status=status,
        warnings=tuple(warnings),
    )


def solve_temperature_from_problem(
    problem: Heat3DProblem,
    options: SolverOptions | None = None,
    *,
    matrix_backend: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a problem and map unique-node temperature back to original nodes."""

    options = options or SolverOptions()
    if options.solver_mode == "contact_resistance" and _all_contact_resistances_zero(problem):
        return _solve_zero_resistance_contact_limit(
            problem,
            options,
            matrix_backend=matrix_backend,
        )

    operator = build_operator(problem, options=options, matrix_backend=matrix_backend)
    temperature_unique, audit = solve_operator(operator)
    temperature_full = temperature_unique[problem.grid_mapping.original_to_unique].reshape(-1, 1)
    bottom_error = _bottom_dirichlet_error(
        problem.coords_original,
        temperature_full,
        problem.boundary.bottom_T_fixed_K,
    )
    meta = {
        "solver_family": SOLVER_FAMILY,
        "contract_version": problem.contract_version,
        "solver_mode": operator.meta.solver_mode,
        "matrix_backend": operator.meta.matrix_backend,
        "sparse_format": operator.meta.sparse_format,
        "legacy_equivalence_target": LEGACY_EQUIVALENCE_TARGET,
        "operator": operator.assembly,
        "solution_audit": {
            "status": audit.status,
            "residual_norm": audit.residual_norm,
            "bottom_dirichlet_error": bottom_error,
            "warnings": list(audit.warnings),
        },
        "contact_interfaces": _contact_solution_metadata(problem, operator, temperature_unique),
        "duplicate_merge": dict(problem.duplicate_merge),
    }
    return temperature_full.astype(np.float64), meta


def _solve_zero_resistance_contact_limit(
    problem: Heat3DProblem,
    options: SolverOptions,
    *,
    matrix_backend: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the exact R_contact=0 limit by equivalent perfect-contact contraction."""

    contact_operator = build_operator(problem, options=options, matrix_backend=matrix_backend)
    perfect_meta = _perfect_contact_meta(problem.sample_meta)
    perfect_problem = extract_problem_from_arrays(
        coords=problem.coords_original,
        k_field=problem.k_field_original,
        q_field=problem.q_field_original,
        sample_meta=perfect_meta,
        duplicate_policy="merge",
    )
    perfect_temperature, perfect_solve_meta = solve_temperature_from_problem(
        perfect_problem,
        SolverOptions(solver_mode="perfect_contact", matrix_backend=contact_operator.meta.matrix_backend),
    )
    temperature_contact_order = perfect_temperature[
        problem.grid_mapping.unique_to_first_original,
        0,
    ]
    bottom_error = _bottom_dirichlet_error(
        problem.coords_original,
        perfect_temperature,
        problem.boundary.bottom_T_fixed_K,
    )
    meta = {
        "solver_family": SOLVER_FAMILY,
        "contract_version": problem.contract_version,
        "solver_mode": "contact_resistance",
        "contact_zero_resistance_limit": "perfect_contact_contraction",
        "matrix_backend": contact_operator.meta.matrix_backend,
        "sparse_format": contact_operator.meta.sparse_format,
        "legacy_equivalence_target": LEGACY_EQUIVALENCE_TARGET,
        "operator": contact_operator.assembly,
        "solution_audit": {
            "status": perfect_solve_meta["solution_audit"]["status"],
            "residual_norm": perfect_solve_meta["solution_audit"]["residual_norm"],
            "bottom_dirichlet_error": bottom_error,
            "warnings": list(perfect_solve_meta["solution_audit"].get("warnings", [])),
        },
        "contact_interfaces": _contact_solution_metadata(
            problem,
            contact_operator,
            temperature_contact_order,
        ),
        "duplicate_merge": dict(problem.duplicate_merge),
    }
    return perfect_temperature.astype(np.float64), meta


def _all_contact_resistances_zero(problem: Heat3DProblem) -> bool:
    contact_interfaces = [
        interface for interface in problem.interfaces if interface.contact_pair_indices.size
    ]
    if not contact_interfaces:
        return False
    return all(float(interface.contact_resistance_m2K_W or 0.0) == 0.0 for interface in contact_interfaces)


def _perfect_contact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    perfect_meta = json.loads(json.dumps(meta))
    perfect_interfaces = []
    for index, raw in enumerate(perfect_meta.get("interfaces", [])):
        converted = dict(raw)
        converted["type"] = "perfect_contact"
        converted.pop("R_contact_m2K_W", None)
        converted.pop("contact_resistance_m2K_W", None)
        converted.setdefault("id", raw.get("id") or f"interface_{index}")
        perfect_interfaces.append(converted)
    perfect_meta["interfaces"] = perfect_interfaces
    return perfect_meta


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _validate_shapes(coords: np.ndarray, k_field: np.ndarray, q_field: np.ndarray) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N,3), found {coords.shape}")
    if k_field.ndim != 2 or k_field.shape[1] not in (1, 3):
        raise ValueError(f"k_field must have shape (N,1) or (N,3), found {k_field.shape}")
    if q_field.ndim != 2 or q_field.shape[1] != 1:
        raise ValueError(f"q_field must have shape (N,1), found {q_field.shape}")
    if k_field.shape[0] != coords.shape[0] or q_field.shape[0] != coords.shape[0]:
        raise ValueError("coords, k_field, and q_field must have the same node count")


def _validate_supported_problem(
    meta: dict[str, Any],
    k_field: np.ndarray,
    duplicate_policy: str,
) -> list[str]:
    warnings: list[str] = []
    boundary_types = meta.get("boundary_types", {})
    if boundary_types != SUPPORTED_BOUNDARY_TYPES:
        raise ValueError(
            "P3a problem extraction currently supports only top Robin / "
            "bottom Dirichlet / sides adiabatic"
        )
    interfaces = meta.get("interfaces", [])
    if not isinstance(interfaces, list):
        raise ValueError("sample_meta.interfaces must be a list")
    allowed_interface_types = (
        {"perfect_contact"}
        if duplicate_policy == "merge"
        else {"perfect_contact", "contact_resistance"}
    )
    if any(interface.get("type") not in allowed_interface_types for interface in interfaces):
        raise ValueError(
            f"P3a extraction with duplicate_policy={duplicate_policy} supports only "
            f"{sorted(allowed_interface_types)} interfaces"
        )
    if k_field.shape[1] == 1:
        warnings.append("isotropic (N,1) conductivity expanded to diagonal (N,3)")
    return warnings


def _expand_k(k_field: np.ndarray) -> tuple[np.ndarray, str]:
    if k_field.shape[1] == 1:
        return np.repeat(k_field.astype(np.float64), repeats=3, axis=1), "isotropic_expanded_to_diag3"
    if k_field.shape[1] == 3:
        return k_field.astype(np.float64), "diag3"
    raise ValueError(f"unsupported k_field shape: {k_field.shape}")


def _merge_duplicate_points(
    coords: np.ndarray,
    k_diag: np.ndarray,
    q_field: np.ndarray,
) -> dict[str, Any]:
    unique_coords, first_indices, inverse, counts = np.unique(
        coords,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    n_unique = unique_coords.shape[0]
    k_acc = np.zeros((n_unique, k_diag.shape[1]), dtype=np.float64)
    q_acc = np.full((n_unique, 1), -np.inf, dtype=np.float64)
    count_acc = np.zeros((n_unique, 1), dtype=np.float64)

    for original_idx, unique_idx in enumerate(inverse):
        k_acc[unique_idx] += k_diag[original_idx]
        q_acc[unique_idx, 0] = max(q_acc[unique_idx, 0], q_field[original_idx, 0])
        count_acc[unique_idx, 0] += 1.0

    duplicate_unique_indices = np.nonzero(counts > 1)[0].astype(np.int64)
    metadata = {
        "original_node_count": int(coords.shape[0]),
        "unique_node_count": int(n_unique),
        "merged_duplicate_count": int(coords.shape[0] - n_unique),
        "duplicate_unique_indices": duplicate_unique_indices.tolist(),
        "k_merge_policy": K_MERGE_POLICY,
        "q_merge_policy": Q_MERGE_POLICY,
        "node_ordering": NODE_ORDERING,
    }
    return {
        "coords": unique_coords.astype(np.float64),
        "inverse": inverse.astype(np.int64),
        "unique_to_first_original": first_indices.astype(np.int64),
        "duplicate_counts": counts.astype(np.int64),
        "k_diag": k_acc / count_acc,
        "q_field": q_acc,
        "metadata": metadata,
    }


def _preserve_duplicate_points(
    coords: np.ndarray,
    k_diag: np.ndarray,
    q_field: np.ndarray,
) -> dict[str, Any]:
    original_order = np.arange(coords.shape[0], dtype=np.int64)
    order = np.lexsort((original_order, coords[:, 2], coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order].astype(np.float64)
    sorted_k = k_diag[order].astype(np.float64)
    sorted_q = q_field[order].astype(np.float64)
    inverse = np.empty((coords.shape[0],), dtype=np.int64)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    contact_pairs, contact_pair_z = _contact_duplicate_pairs(sorted_coords)
    unique_coord_count = int(np.unique(coords, axis=0).shape[0])
    metadata = {
        "original_node_count": int(coords.shape[0]),
        "unique_node_count": int(coords.shape[0]),
        "preserved_node_count": int(coords.shape[0]),
        "merged_duplicate_count": 0,
        "preserved_duplicate_count": int(coords.shape[0] - unique_coord_count),
        "duplicate_unique_indices": sorted(set(int(v) for pair in contact_pairs for v in pair)),
        "contact_pair_indices": contact_pairs,
        "contact_pair_z_m": contact_pair_z,
        "k_merge_policy": CONTACT_DUPLICATE_POLICY,
        "q_merge_policy": CONTACT_DUPLICATE_POLICY,
        "node_ordering": CONTACT_NODE_ORDERING,
        "duplicate_policy": "preserve_contact_duplicates",
    }
    return {
        "coords": sorted_coords,
        "inverse": inverse,
        "unique_to_first_original": order.astype(np.int64),
        "duplicate_counts": np.ones((coords.shape[0],), dtype=np.int64),
        "k_diag": sorted_k,
        "q_field": sorted_q,
        "metadata": metadata,
    }


def _contact_duplicate_pairs(coords: np.ndarray) -> tuple[list[list[int]], list[float]]:
    groups: dict[tuple[float, float, float], list[int]] = {}
    for idx, point in enumerate(coords):
        groups.setdefault(tuple(float(v) for v in point), []).append(idx)

    pairs: list[list[int]] = []
    pair_z: list[float] = []
    for key, indices in groups.items():
        if len(indices) == 1:
            continue
        if len(indices) != 2:
            raise ValueError(
                "P3a contact mode currently supports exactly two duplicate nodes "
                f"per interface coordinate, found {len(indices)} at {key}"
            )
        pairs.append([int(indices[0]), int(indices[1])])
        pair_z.append(float(key[2]))
    return pairs, pair_z


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


def _control_widths_allow_repeated_axis(axis: np.ndarray) -> np.ndarray:
    widths = np.zeros_like(axis, dtype=np.float64)
    if axis.size == 1:
        widths[0] = 1.0
        return widths
    for idx in range(axis.size):
        prev_gap = 0.0
        next_gap = 0.0
        if idx > 0:
            gap = float(axis[idx] - axis[idx - 1])
            if gap > 0.0:
                prev_gap = gap
        if idx < axis.size - 1:
            gap = float(axis[idx + 1] - axis[idx])
            if gap > 0.0:
                next_gap = gap
        widths[idx] = 0.5 * (prev_gap + next_gap)
        if widths[idx] <= 0.0:
            raise ValueError("repeated-axis control width is zero; contact grid is unsupported")
    return widths


def _grid_contract(
    coords: np.ndarray,
    *,
    original_to_unique: np.ndarray,
    unique_to_first_original: np.ndarray,
    duplicate_counts: np.ndarray,
) -> tuple[GridSpec, GridMapping]:
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

    unique_coords = np.unique(coords, axis=0)
    if unique_coords.shape[0] != coords.shape[0] or not np.array_equal(unique_coords, coords):
        raise ValueError("Merged coordinates must already be unique and in node order")

    spec = GridSpec(
        x=xs.astype(np.float64),
        y=ys.astype(np.float64),
        z=zs.astype(np.float64),
        dx=_control_widths(xs),
        dy=_control_widths(ys),
        dz=_control_widths(zs),
        grid_shape=(int(xs.size), int(ys.size), int(zs.size)),
        node_count=int(coords.shape[0]),
    )
    mapping = GridMapping(
        grid=grid,
        original_to_unique=original_to_unique.astype(np.int64),
        unique_to_first_original=unique_to_first_original.astype(np.int64),
        duplicate_counts=duplicate_counts.astype(np.int64),
    )
    return spec, mapping


def _contact_grid_contract(
    coords: np.ndarray,
    *,
    original_to_unique: np.ndarray,
    unique_to_first_original: np.ndarray,
    duplicate_counts: np.ndarray,
) -> tuple[GridSpec, GridMapping]:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    grid_rows: list[np.ndarray] = []
    reference_z: np.ndarray | None = None
    for x in xs:
        for y in ys:
            mask = np.isclose(coords[:, 0], x) & np.isclose(coords[:, 1], y)
            indices = np.nonzero(mask)[0]
            indices = indices[np.lexsort((indices, coords[indices, 2]))]
            if reference_z is None:
                reference_z = coords[indices, 2].astype(np.float64)
            elif indices.size != reference_z.size or not np.allclose(coords[indices, 2], reference_z):
                raise ValueError("Contact-mode coordinates must share the same z sequence for every x/y column")
            grid_rows.append(indices.astype(np.int64))
    if reference_z is None:
        raise ValueError("empty contact grid")

    grid = -np.ones((xs.size, ys.size, reference_z.size), dtype=np.int64)
    row_index = 0
    for ix in range(xs.size):
        for iy in range(ys.size):
            grid[ix, iy, :] = grid_rows[row_index]
            row_index += 1

    spec = GridSpec(
        x=xs.astype(np.float64),
        y=ys.astype(np.float64),
        z=reference_z.astype(np.float64),
        dx=_control_widths(xs),
        dy=_control_widths(ys),
        dz=_control_widths_allow_repeated_axis(reference_z),
        grid_shape=(int(xs.size), int(ys.size), int(reference_z.size)),
        node_count=int(coords.shape[0]),
    )
    mapping = GridMapping(
        grid=grid,
        original_to_unique=original_to_unique.astype(np.int64),
        unique_to_first_original=unique_to_first_original.astype(np.int64),
        duplicate_counts=duplicate_counts.astype(np.int64),
        node_ordering=CONTACT_NODE_ORDERING,
    )
    return spec, mapping


def _boundary_contract(meta: dict[str, Any], coords: np.ndarray) -> BoundarySpec:
    params = meta["boundary_params"]
    top = params["top"]
    bottom = params["bottom"]
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))
    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))

    top_mask = np.isclose(coords[:, 2], z_max)
    bottom_mask = np.isclose(coords[:, 2], z_min)
    side_mask = (
        np.isclose(coords[:, 0], x_min)
        | np.isclose(coords[:, 0], x_max)
        | np.isclose(coords[:, 1], y_min)
        | np.isclose(coords[:, 1], y_max)
    )
    interior_mask = ~(top_mask | bottom_mask | side_mask)
    return BoundarySpec(
        boundary_types=dict(meta.get("boundary_types", {})),
        top_h_W_m2K=float(top["h_W_m2K"]),
        top_T_inf_K=float(top["ambient_temperature_K"]),
        bottom_T_fixed_K=float(bottom["fixed_temperature_K"]),
        side_policy="adiabatic_natural_zero_flux",
        top_node_indices=np.nonzero(top_mask)[0].astype(np.int64),
        bottom_node_indices=np.nonzero(bottom_mask)[0].astype(np.int64),
        side_node_indices=np.nonzero(side_mask)[0].astype(np.int64),
        interior_node_indices=np.nonzero(interior_mask)[0].astype(np.int64),
    )


def _interface_records(
    meta: dict[str, Any],
    mapping: GridMapping,
    *,
    contact_pair_indices: list[list[int]] | None = None,
    contact_pair_z: list[float] | None = None,
) -> list[InterfaceRecord]:
    duplicate_unique_indices = np.nonzero(mapping.duplicate_counts > 1)[0].astype(np.int64)
    contact_pairs = np.asarray(contact_pair_indices or [], dtype=np.int64).reshape(-1, 2)
    contact_z = np.asarray(contact_pair_z or [], dtype=np.float64).reshape(-1)
    records: list[InterfaceRecord] = []
    for index, raw in enumerate(meta.get("interfaces", [])):
        adjacent = _adjacent_layer_ids(raw)
        z_position = _optional_float(
            raw.get("z_m", raw.get("z_position_m", raw.get("z_position")))
        )
        selected_pairs = _select_contact_pairs_for_interface(contact_pairs, contact_z, z_position)
        selected_nodes = (
            np.unique(selected_pairs.reshape(-1)).astype(np.int64)
            if selected_pairs.size
            else duplicate_unique_indices
        )
        records.append(
            InterfaceRecord(
                interface_id=str(raw.get("id") or raw.get("name") or f"interface_{index}"),
                interface_type=str(raw.get("type")),
                adjacent_layer_ids=adjacent,
                z_position_m=z_position,
                contact_resistance_m2K_W=_optional_float(
                    raw.get("R_contact_m2K_W", raw.get("contact_resistance_m2K_W"))
                ),
                duplicate_unique_indices=selected_nodes,
                contact_pair_indices=selected_pairs,
                raw_metadata=dict(raw),
            )
        )
    return records


def _select_contact_pairs_for_interface(
    contact_pairs: np.ndarray,
    contact_z: np.ndarray,
    z_position: float | None,
) -> np.ndarray:
    if contact_pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if z_position is None:
        return contact_pairs.astype(np.int64)
    mask = np.isclose(contact_z, z_position)
    return contact_pairs[mask].astype(np.int64)


def _adjacent_layer_ids(raw: dict[str, Any]) -> tuple[int, int] | None:
    value = raw.get("adjacent_layer_ids", raw.get("layers"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    lower = raw.get("lower_layer_id")
    upper = raw.get("upper_layer_id")
    if lower is not None and upper is not None:
        return int(lower), int(upper)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate_assembly_options(problem: Heat3DProblem, options: SolverOptions) -> None:
    if options.solver_mode not in {"legacy_equivalent", "perfect_contact", "contact_resistance"}:
        raise ValueError(
            "P3a supports only legacy_equivalent/perfect_contact/contact_resistance, "
            f"got {options.solver_mode}"
        )
    if options.contact_enabled and options.solver_mode != "contact_resistance":
        raise ValueError("contact_enabled is only valid with solver_mode=contact_resistance")
    if options.solver_mode != "contact_resistance":
        for interface in problem.interfaces:
            if interface.interface_type != "perfect_contact":
                raise ValueError("Perfect-contact assembly supports only perfect_contact interfaces")
            if interface.contact_resistance_m2K_W not in (None, 0.0):
                raise ValueError("Perfect-contact assembly supports only R_contact=0")
        return

    if problem.duplicate_merge.get("duplicate_policy") != "preserve_contact_duplicates":
        raise ValueError("contact_resistance assembly requires duplicate_policy='preserve'")
    contact_interfaces = [interface for interface in problem.interfaces if interface.contact_pair_indices.size]
    if not contact_interfaces:
        raise ValueError("contact_resistance assembly requires preserved duplicate contact pairs")
    for interface in contact_interfaces:
        if interface.interface_type not in {"perfect_contact", "contact_resistance"}:
            raise ValueError(f"unsupported contact interface type: {interface.interface_type}")
        resistance = interface.contact_resistance_m2K_W
        if resistance is None or resistance < 0.0:
            raise ValueError("contact_resistance mode requires non-negative R_contact_m2K_W")


def _assemble_triplets(
    problem: Heat3DProblem,
    options: SolverOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    grid = problem.grid_mapping.grid
    spec = problem.grid_spec
    n = spec.node_count
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros((n,), dtype=np.float64)

    def add(row: int, col: int, value: float) -> None:
        rows.append(int(row))
        cols.append(int(col))
        values.append(float(value))

    def conductance(idx_i: int, idx_j: int, axis: int, area: float, distance: float) -> float:
        k_i = float(problem.k_diag[idx_i, axis])
        k_j = float(problem.k_diag[idx_j, axis])
        return _harmonic_mean(k_i, k_j) * area / distance

    xs, ys, zs = spec.x, spec.y, spec.z
    dx_cv, dy_cv, dz_cv = spec.dx, spec.dy, spec.dz
    h_top = problem.boundary.top_h_W_m2K
    t_inf = problem.boundary.top_T_inf_K
    t_bottom = problem.boundary.bottom_T_fixed_K
    contact_lookup, contact_faces = _contact_face_lookup(problem) if options.solver_mode == "contact_resistance" else ({}, [])

    def z_conductance(idx_i: int, idx_j: int, iz_i: int, iz_j: int, area: float, distance: float) -> float:
        if distance > 0.0:
            return conductance(idx_i, idx_j, axis=2, area=area, distance=distance)
        key = tuple(sorted((int(idx_i), int(idx_j))))
        if key not in contact_lookup:
            raise ValueError("zero-distance z neighbor is not registered as a contact face")
        return float(contact_lookup[key]["effective_conductance_W_K"])

    for ix in range(xs.size):
        for iy in range(ys.size):
            for iz in range(zs.size):
                idx = int(grid[ix, iy, iz])
                if iz == 0:
                    add(idx, idx, 1.0)
                    rhs[idx] = t_bottom
                    continue

                volume = float(dx_cv[ix] * dy_cv[iy] * dz_cv[iz])
                row_rhs = float(problem.q_field[idx, 0]) * volume
                diag = 0.0

                if iz == zs.size - 1:
                    area_top = float(dx_cv[ix] * dy_cv[iy])
                    robin = h_top * area_top
                    diag += robin
                    row_rhs += robin * t_inf

                if ix > 0:
                    neighbor = int(grid[ix - 1, iy, iz])
                    g = conductance(
                        idx,
                        neighbor,
                        axis=0,
                        area=float(dy_cv[iy] * dz_cv[iz]),
                        distance=float(xs[ix] - xs[ix - 1]),
                    )
                    diag += g
                    add(idx, neighbor, -g)
                if ix < xs.size - 1:
                    neighbor = int(grid[ix + 1, iy, iz])
                    g = conductance(
                        idx,
                        neighbor,
                        axis=0,
                        area=float(dy_cv[iy] * dz_cv[iz]),
                        distance=float(xs[ix + 1] - xs[ix]),
                    )
                    diag += g
                    add(idx, neighbor, -g)

                if iy > 0:
                    neighbor = int(grid[ix, iy - 1, iz])
                    g = conductance(
                        idx,
                        neighbor,
                        axis=1,
                        area=float(dx_cv[ix] * dz_cv[iz]),
                        distance=float(ys[iy] - ys[iy - 1]),
                    )
                    diag += g
                    add(idx, neighbor, -g)
                if iy < ys.size - 1:
                    neighbor = int(grid[ix, iy + 1, iz])
                    g = conductance(
                        idx,
                        neighbor,
                        axis=1,
                        area=float(dx_cv[ix] * dz_cv[iz]),
                        distance=float(ys[iy + 1] - ys[iy]),
                    )
                    diag += g
                    add(idx, neighbor, -g)

                if iz > 0:
                    neighbor = int(grid[ix, iy, iz - 1])
                    g = z_conductance(
                        idx,
                        neighbor,
                        area=float(dx_cv[ix] * dy_cv[iy]),
                        distance=float(zs[iz] - zs[iz - 1]),
                        iz_i=iz,
                        iz_j=iz - 1,
                    )
                    diag += g
                    add(idx, neighbor, -g)
                if iz < zs.size - 1:
                    neighbor = int(grid[ix, iy, iz + 1])
                    g = z_conductance(
                        idx,
                        neighbor,
                        area=float(dx_cv[ix] * dy_cv[iy]),
                        distance=float(zs[iz + 1] - zs[iz]),
                        iz_i=iz,
                        iz_j=iz + 1,
                    )
                    diag += g
                    add(idx, neighbor, -g)

                add(idx, idx, diag)
                rhs[idx] = row_rhs

    row = np.asarray(rows, dtype=np.int64)
    col = np.asarray(cols, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    assembly_meta = {
        "grid_shape": [int(v) for v in spec.grid_shape],
        "node_count": int(n),
        "top_robin_h_W_m2K": h_top,
        "top_robin_T_inf_K": t_inf,
        "bottom_dirichlet_T_K": t_bottom,
        "side_boundary_policy": "adiabatic_natural_zero_flux",
        "source_policy": "q_times_control_volume",
        "face_conductivity_policy": "harmonic_mean_between_neighboring_nodes",
        "bottom_dirichlet_policy": "row_replacement_T_equals_bottom",
        "solver_mode": options.solver_mode,
        "contact_model": (
            "series_half_cell_R_contact_half_cell"
            if options.solver_mode == "contact_resistance"
            else "not_enabled"
        ),
        "contact_faces": contact_faces,
        "triplet_count": int(data.size),
        "linear_system_shape": [int(n), int(n)],
    }
    return row, col, data, rhs, assembly_meta


def _contact_face_lookup(problem: Heat3DProblem) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    positions = _node_grid_positions(problem.grid_mapping.grid)
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    faces: list[dict[str, Any]] = []
    for interface in problem.interfaces:
        for pair in interface.contact_pair_indices:
            lower, upper = _ordered_contact_pair(problem, positions, int(pair[0]), int(pair[1]))
            ix, iy, lower_iz = positions[lower]
            _, _, upper_iz = positions[upper]
            lower_half = 0.5 * _positive_gap_before(problem.grid_spec.z, lower_iz)
            upper_half = 0.5 * _positive_gap_after(problem.grid_spec.z, upper_iz)
            area = float(problem.grid_spec.dx[ix] * problem.grid_spec.dy[iy])
            k_lower = float(problem.k_diag[lower, 2])
            k_upper = float(problem.k_diag[upper, 2])
            resistance = float(interface.contact_resistance_m2K_W or 0.0)
            denominator = lower_half / k_lower + resistance + upper_half / k_upper
            if denominator <= 0.0:
                raise ValueError("contact face series resistance must be positive")
            effective = area / denominator
            face = {
                "interface_id": interface.interface_id,
                "lower_node": int(lower),
                "upper_node": int(upper),
                "z_position_m": float(problem.coords[lower, 2]),
                "area_m2": area,
                "R_contact_m2K_W": resistance,
                "lower_half_thickness_m": lower_half,
                "upper_half_thickness_m": upper_half,
                "lower_kz_W_mK": k_lower,
                "upper_kz_W_mK": k_upper,
                "effective_conductance_W_K": effective,
            }
            key = tuple(sorted((int(lower), int(upper))))
            lookup[key] = face
            faces.append(face)
    return lookup, faces


def _node_grid_positions(grid: np.ndarray) -> dict[int, tuple[int, int, int]]:
    positions: dict[int, tuple[int, int, int]] = {}
    for ix in range(grid.shape[0]):
        for iy in range(grid.shape[1]):
            for iz in range(grid.shape[2]):
                positions[int(grid[ix, iy, iz])] = (int(ix), int(iy), int(iz))
    return positions


def _ordered_contact_pair(
    problem: Heat3DProblem,
    positions: dict[int, tuple[int, int, int]],
    first: int,
    second: int,
) -> tuple[int, int]:
    if first not in positions or second not in positions:
        raise ValueError("contact pair node is missing from grid")
    first_pos = positions[first]
    second_pos = positions[second]
    if first_pos[0] != second_pos[0] or first_pos[1] != second_pos[1]:
        raise ValueError("contact pair nodes must share x/y grid indices")
    if abs(first_pos[2] - second_pos[2]) != 1:
        raise ValueError("contact pair nodes must be adjacent in computational z")
    if not np.allclose(problem.coords[first], problem.coords[second]):
        raise ValueError("contact pair nodes must share the same physical coordinate")
    return (first, second) if first_pos[2] < second_pos[2] else (second, first)


def _positive_gap_before(axis: np.ndarray, index: int) -> float:
    for cursor in range(index - 1, -1, -1):
        gap = float(axis[index] - axis[cursor])
        if gap > 0.0:
            return gap
    raise ValueError("contact face lower side has no positive adjacent thickness")


def _positive_gap_after(axis: np.ndarray, index: int) -> float:
    for cursor in range(index + 1, axis.size):
        gap = float(axis[cursor] - axis[index])
        if gap > 0.0:
            return gap
    raise ValueError("contact face upper side has no positive adjacent thickness")


def _contact_solution_metadata(
    problem: Heat3DProblem,
    operator: AssembledOperator,
    temperature_unique: np.ndarray,
) -> list[dict[str, Any]]:
    del problem
    faces = operator.assembly.get("contact_faces", [])
    grouped: dict[str, dict[str, Any]] = {}
    for face in faces:
        interface_id = str(face["interface_id"])
        lower = int(face["lower_node"])
        upper = int(face["upper_node"])
        conductance = float(face["effective_conductance_W_K"])
        jump = float(temperature_unique[upper] - temperature_unique[lower])
        flux = float(conductance * (temperature_unique[lower] - temperature_unique[upper]))
        entry = grouped.setdefault(
            interface_id,
            {
                "interface_id": interface_id,
                "R_contact_m2K_W": float(face["R_contact_m2K_W"]),
                "face_count": 0,
                "flux_lower_to_upper_W": 0.0,
                "max_abs_face_flux_W": 0.0,
                "temperature_jump_upper_minus_lower_K": {
                    "mean": 0.0,
                    "max_abs": 0.0,
                },
                "effective_conductance_W_K": {
                    "sum": 0.0,
                    "mean": 0.0,
                },
            },
        )
        entry["face_count"] += 1
        entry["flux_lower_to_upper_W"] += flux
        entry["max_abs_face_flux_W"] = max(entry["max_abs_face_flux_W"], abs(flux))
        entry["temperature_jump_upper_minus_lower_K"]["mean"] += jump
        entry["temperature_jump_upper_minus_lower_K"]["max_abs"] = max(
            entry["temperature_jump_upper_minus_lower_K"]["max_abs"],
            abs(jump),
        )
        entry["effective_conductance_W_K"]["sum"] += conductance

    results: list[dict[str, Any]] = []
    for entry in grouped.values():
        face_count = int(entry["face_count"])
        entry["temperature_jump_upper_minus_lower_K"]["mean"] /= face_count
        entry["effective_conductance_W_K"]["mean"] = (
            entry["effective_conductance_W_K"]["sum"] / face_count
        )
        results.append(entry)
    return sorted(results, key=lambda item: str(item["interface_id"]))


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Conductivity must be positive for harmonic mean, got {a}, {b}")
    return 2.0 * a * b / (a + b)


def _operator_checksum(
    row: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    rhs: np.ndarray,
    n: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([n, n], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(row).tobytes())
    digest.update(np.ascontiguousarray(col).tobytes())
    digest.update(np.ascontiguousarray(data).tobytes())
    digest.update(np.ascontiguousarray(rhs).tobytes())
    return digest.hexdigest()


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
