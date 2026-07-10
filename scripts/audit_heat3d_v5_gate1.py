#!/usr/bin/env python3
"""Build and verify the read-only V5 Gate 1 physics-scale closeout.

This script reads frozen P5 arrays and metadata.  It never imports the model or
the reference label solver, and its normal mode writes only the explicitly
requested CSV table, JSON summary, and Markdown closeout.  ``--dry-run`` and
``--verify-summary`` write nothing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_ID = "V5-Gate-1"
SCHEMA_VERSION = "heat3d_v5_gate1_closeout_v1"
P0_AUDIT_SHA256 = "cf231c690884b18d6cd331887caa2c0411e01bbe2928f432d6f1b983dfea9c4e"
ROLE_ORDER = (
    "train",
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
CALIBRATION_ROLE = "train"
SELECTION_ROLE = "valid_iid"
OOD_ROLE = "hard_challenge_valid"
TEST_ROLES = ("test_iid", "hard_challenge_test")
CLEAN_ROLES = ("train", "valid_iid", "test_iid")
HARD_ROLES = ("hard_train_holdout", "hard_challenge_valid", "hard_challenge_test")
CANDIDATES = (
    "constant",
    "power_only",
    "q_rms_lz2_over_kz",
    "legacy_p_array_r_series",
    "source_centroid_two_path",
    "z_collapsed_1d",
)
PHYSICS_CANDIDATES = CANDIDATES[1:]
INPUT_ARRAYS = ("coords", "k_field", "q_field", "bc_features")
FULL_ARRAYS = INPUT_ARRAYS + ("temperature",)
EPS = 1.0e-30
BOTTOM_LABEL_TOL_K = 1.0e-8
BOOTSTRAP_SEED = 1701
BOOTSTRAP_RESAMPLES = 2000


class AuditError(RuntimeError):
    """Raised for a Gate 1 contract, schema, or reconstruction violation."""


STRING_COLUMNS = {
    "sample_id",
    "role",
    "driver_category",
    "input_fingerprint",
    "full_fingerprint",
    "provenance_source_id",
    "grid_shape",
    "selected_or_best_physics_candidate",
}
INT_COLUMNS = {
    "bottom_node_count",
    "bottom_q_nonzero_count",
    "bottom_mask_matches_bc",
    "bottom_mask_matches_metadata",
    "bottom_temperature_label_pass",
    "is_clean_role",
    "is_hard_role",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{name} must be finite, got {value!r}")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{name} must be an object")
    return value


def _role_key(role: str) -> tuple[int, str]:
    try:
        return (ROLE_ORDER.index(role), role)
    except ValueError:
        return (len(ROLE_ORDER), role)


def _sample_dirs(dataset: Path) -> set[str]:
    return {
        path.name
        for path in dataset.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    }


def _load_inputs(
    dataset: Path,
    split_map: Path,
    contract_path: Path | None,
) -> tuple[dict[str, str], list[str], dict[str, Any], dict[str, Any] | None]:
    if not dataset.is_dir():
        raise AuditError(f"dataset directory does not exist: {dataset}")
    if not split_map.is_file():
        raise AuditError(f"split map does not exist: {split_map}")
    split_payload = _read_json(split_map)
    raw_assignments = _mapping(split_payload.get("sample_splits"), "sample_splits")
    assignments = {str(sample_id): str(role) for sample_id, role in raw_assignments.items()}
    if not assignments:
        raise AuditError("sample_splits is empty")
    on_disk = _sample_dirs(dataset)
    missing = sorted(set(assignments) - on_disk)
    extra = sorted(on_disk - set(assignments))
    if missing:
        raise AuditError(f"split samples missing from dataset: {missing[:8]}")
    if extra:
        raise AuditError(f"dataset samples absent from split map: {extra[:8]}")
    counts = Counter(assignments.values())
    declared = split_payload.get("actual_counts")
    if declared is not None:
        normalized = {str(key): int(value) for key, value in _mapping(declared, "actual_counts").items()}
        if normalized != dict(counts):
            raise AuditError(f"actual_counts mismatch: {normalized} != {dict(counts)}")
    roles = sorted(counts, key=_role_key)

    contract: dict[str, Any] | None = None
    if contract_path is not None:
        contract = _read_json(contract_path)
        dataset_contract = _mapping(contract.get("dataset_contract"), "dataset_contract")
        expected_id = dataset_contract.get("dataset_id")
        if expected_id is not None and expected_id != split_payload.get("dataset_id"):
            raise AuditError("contract dataset_id disagrees with split map")
        expected_counts = dataset_contract.get("role_counts")
        if expected_counts is not None:
            normalized_expected = {
                str(key): int(value)
                for key, value in _mapping(expected_counts, "dataset_contract.role_counts").items()
            }
            if normalized_expected != dict(counts):
                raise AuditError("contract role_counts disagree with split map")
        expected_total = dataset_contract.get("total_sample_count")
        if expected_total is not None and int(expected_total) != len(assignments):
            raise AuditError("contract total_sample_count disagrees with split map")
        protocol = _mapping(contract.get("calibration_and_selection"), "calibration_and_selection")
        if protocol.get("fit_role") != CALIBRATION_ROLE:
            raise AuditError("contract fit role must remain train")
        if protocol.get("selection_role") != SELECTION_ROLE:
            raise AuditError("contract selection role must remain valid_iid")
        if protocol.get("ood_inspection_role") != OOD_ROLE:
            raise AuditError("contract OOD role must remain hard_challenge_valid")
        if tuple(protocol.get("test_roles", [])) != TEST_ROLES:
            raise AuditError("contract test roles changed")
    return assignments, roles, split_payload, contract


def _control_widths(axis: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise AuditError(f"{name} needs at least two coordinates")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise AuditError(f"{name} coordinates must be finite and increasing")
    widths = np.empty_like(axis)
    widths[0] = 0.5 * (axis[1] - axis[0])
    widths[-1] = 0.5 * (axis[-1] - axis[-2])
    if axis.size > 2:
        widths[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return widths


def _control_volumes(coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[int]]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise AuditError("coords must be finite [N,3]")
    axes: list[np.ndarray] = []
    inverse: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    for dim, name in enumerate(("x", "y", "z")):
        axis, index = np.unique(coords[:, dim], return_inverse=True)
        axes.append(axis)
        inverse.append(index)
        widths.append(_control_widths(axis, name))
    shape = [int(axis.size) for axis in axes]
    if int(np.prod(shape)) != coords.shape[0]:
        raise AuditError(f"coordinates are not a full rectilinear grid: {shape}")
    flattened = np.ravel_multi_index(tuple(inverse), tuple(shape))
    if np.unique(flattened).size != coords.shape[0]:
        raise AuditError("duplicate coordinates are unsupported by Gate 1 P5 audit")
    volumes = widths[0][inverse[0]] * widths[1][inverse[1]] * widths[2][inverse[2]]
    if np.any(volumes <= 0.0) or not np.all(np.isfinite(volumes)):
        raise AuditError("invalid control-volume weights")
    return volumes, axes, inverse, shape


def _weighted_harmonic(values: np.ndarray, weights: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.shape != weights.shape or np.any(values <= 0.0):
        raise AuditError(f"invalid harmonic input for {name}")
    return float(weights.sum() / np.dot(weights, 1.0 / values))


def _fingerprint(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _load_arrays(sample_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in FULL_ARRAYS:
        path = sample_dir / f"{name}.npy"
        if not path.is_file():
            raise AuditError(f"missing required array {path}")
        try:
            arrays[name] = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise AuditError(f"cannot read {path}: {exc}") from exc
    return arrays


def _temperature_values(meta: Mapping[str, Any]) -> tuple[float, float]:
    params = _mapping(meta.get("boundary_params"), "boundary_params")
    bottom = _mapping(params.get("bottom"), "boundary_params.bottom")
    top = _mapping(params.get("top"), "boundary_params.top")
    bottom_value = bottom.get("fixed_temperature_K", bottom.get("T_fixed_K"))
    top_value = top.get("ambient_temperature_K", top.get("T_inf_K"))
    return (
        _finite_float(bottom_value, "bottom fixed temperature"),
        _finite_float(top_value, "top ambient temperature"),
    )


def _bottom_metadata_indices(meta: Mapping[str, Any]) -> set[int]:
    regions = meta.get("boundary_regions")
    if not isinstance(regions, list):
        raise AuditError("sample_meta.boundary_regions must be a list")
    for region in regions:
        if isinstance(region, Mapping) and str(region.get("name")) == "bottom":
            indices = region.get("point_indices")
            if not isinstance(indices, list):
                raise AuditError("bottom boundary metadata has no point_indices")
            return {int(index) for index in indices}
    raise AuditError("bottom boundary metadata is missing")


def _driver_category(p_operator: float, bc_offset: float) -> str:
    power_tol = max(1.0e-14, abs(p_operator) * 1.0e-12)
    offset_tol = 1.0e-10
    source_active = abs(p_operator) > power_tol
    bc_active = abs(bc_offset) > offset_tol
    if source_active and not bc_active:
        return "source_driven"
    if not source_active and bc_active:
        return "bc_driven"
    if source_active and bc_active:
        return "mixed_source_bc"
    return "zero_drive"


def _two_path_proxy(
    *,
    p_operator: float,
    source_z: float | None,
    axes: Sequence[np.ndarray],
    volumes: np.ndarray,
    k_z: np.ndarray,
    top_h: float,
    area: float,
    coords: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if source_z is None or p_operator <= EPS:
        return None, None, None
    z = coords[:, 2]
    z_min = float(axes[2][0])
    z_max = float(axes[2][-1])
    bottom_mask = z <= source_z + 1.0e-15
    top_mask = z >= source_z - 1.0e-15
    k_bottom = _weighted_harmonic(k_z[bottom_mask], volumes[bottom_mask], "centroid bottom kz")
    k_top = _weighted_harmonic(k_z[top_mask], volumes[top_mask], "centroid top kz")
    r_bottom = max(source_z - z_min, 0.0) / (area * k_bottom)
    r_top = (z_max - source_z) / (area * k_top) + 1.0 / (top_h * area)
    if r_bottom <= 0.0:
        r_equivalent = 0.0
    else:
        r_equivalent = 1.0 / (1.0 / r_bottom + 1.0 / r_top)
    return p_operator * r_equivalent, r_bottom, r_top


def _z_collapsed_1d_proxy(
    *,
    axes: Sequence[np.ndarray],
    inverse: Sequence[np.ndarray],
    volumes: np.ndarray,
    q_operator: np.ndarray,
    k_z: np.ndarray,
    top_h: float,
    top_area: float,
    bc_offset: float,
) -> float:
    """Return the CV-RMS of a z-only resistance network relative to T_bottom.

    This is a deterministic collapsed proxy. It is not a call to the Heat3D
    reference label solver and never reads temperature labels.
    """

    z_axis = axes[2]
    z_widths = _control_widths(z_axis, "z")
    z_indices = inverse[2]
    count = z_axis.size
    layer_volumes = np.zeros(count, dtype=np.float64)
    layer_power = np.zeros(count, dtype=np.float64)
    layer_k = np.zeros(count, dtype=np.float64)
    for index in range(count):
        mask = z_indices == index
        layer_volumes[index] = float(volumes[mask].sum())
        layer_power[index] = float(np.dot(q_operator[mask], volumes[mask]))
        layer_k[index] = _weighted_harmonic(k_z[mask], volumes[mask], f"z layer {index} kz")
    if np.any(layer_volumes <= 0.0):
        raise AuditError("z-collapsed proxy found an empty layer")

    matrix = np.zeros((count, count), dtype=np.float64)
    rhs = np.zeros(count, dtype=np.float64)
    matrix[0, 0] = 1.0
    for index in range(1, count):
        distance = float(z_axis[index] - z_axis[index - 1])
        conductance = top_area / (
            0.5 * z_widths[index - 1] / layer_k[index - 1]
            + 0.5 * z_widths[index] / layer_k[index]
        )
        matrix[index, index] += conductance
        matrix[index, index - 1] -= conductance
        matrix[index - 1, index] -= conductance
        matrix[index - 1, index - 1] += conductance
        if distance <= 0.0:
            raise AuditError("z axis must increase")
    # Reimpose the bottom Dirichlet row after adding adjacent conductance.
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = 0.0
    for index in range(1, count):
        rhs[index] += layer_power[index]
    robin = top_h * top_area
    matrix[-1, -1] += robin
    rhs[-1] += robin * bc_offset
    try:
        delta = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise AuditError(f"z-collapsed 1D proxy matrix is singular: {exc}") from exc
    total_volume = float(layer_volumes.sum())
    return float(math.sqrt(np.dot(layer_volumes, delta * delta) / total_volume))


def _sample_row(sample_dir: Path, role: str) -> dict[str, Any]:
    meta = _read_json(sample_dir / "sample_meta.json")
    arrays = _load_arrays(sample_dir)
    coords = np.asarray(arrays["coords"], dtype=np.float64)
    q = np.asarray(arrays["q_field"], dtype=np.float64).reshape(-1)
    temperature = np.asarray(arrays["temperature"], dtype=np.float64).reshape(-1)
    k_field = np.asarray(arrays["k_field"], dtype=np.float64)
    bc = np.asarray(arrays["bc_features"], dtype=np.float64)
    node_count = coords.shape[0]
    if q.shape != (node_count,) or temperature.shape != (node_count,) or bc.shape[0] != node_count:
        raise AuditError(f"{sample_dir}: array node counts disagree")
    if k_field.shape == (node_count, 1):
        k_diag = np.repeat(k_field, 3, axis=1)
    elif k_field.shape == (node_count, 3):
        k_diag = k_field
    else:
        raise AuditError(f"{sample_dir}: k_field must have [N,1] or [N,3]")
    if not all(np.all(np.isfinite(value)) for value in (coords, q, temperature, k_diag, bc)):
        raise AuditError(f"{sample_dir}: non-finite array values")
    if np.any(k_diag <= 0.0):
        raise AuditError(f"{sample_dir}: non-positive conductivity")

    volumes, axes, inverse, shape = _control_volumes(coords)
    total_volume = float(volumes.sum())
    z_min = float(axes[2][0])
    bottom_mask = np.isclose(coords[:, 2], z_min)
    names = meta.get("bc_feature_names")
    if not isinstance(names, list) or "is_bottom" not in names:
        raise AuditError(f"{sample_dir}: bc_feature_names must include is_bottom")
    bottom_feature_index = names.index("is_bottom")
    bc_bottom_mask = np.isclose(bc[:, bottom_feature_index], 1.0)
    bottom_indices = set(np.nonzero(bottom_mask)[0].astype(int))
    bottom_metadata_matches = bottom_indices == _bottom_metadata_indices(meta)
    bottom_bc_matches = bool(np.array_equal(bottom_mask, bc_bottom_mask))
    if not bottom_bc_matches or not bottom_metadata_matches:
        raise AuditError(f"{sample_dir}: bottom mask does not match BC contract")

    t_bottom, t_inf = _temperature_values(meta)
    bottom_label_error = float(np.max(np.abs(temperature[bottom_mask] - t_bottom)))
    bottom_label_pass = bottom_label_error <= BOTTOM_LABEL_TOL_K
    if not bottom_label_pass:
        raise AuditError(f"{sample_dir}: bottom temperature label violates Dirichlet value")
    delta_t = temperature - t_bottom
    s_y = float(math.sqrt(np.dot(volumes, delta_t * delta_t) / total_volume))
    if s_y <= EPS:
        raise AuditError(f"{sample_dir}: target CV RMS DeltaT must be positive for log calibration")

    p_array = float(np.dot(q, volumes))
    p_bottom = float(np.dot(q[bottom_mask], volumes[bottom_mask]))
    q_operator = q.copy()
    q_operator[bottom_mask] = 0.0
    p_operator = float(np.dot(q_operator, volumes))
    p_identity_error = p_array - p_bottom - p_operator
    if abs(p_identity_error) > max(1.0e-14, abs(p_array) * 1.0e-11):
        raise AuditError(f"{sample_dir}: P_array != P_bottom + P_operator")
    bottom_q_nonzero = int(np.count_nonzero(np.abs(q[bottom_mask]) > EPS))
    bottom_q_abs_max = float(np.max(np.abs(q[bottom_mask])))
    bc_offset = t_inf - t_bottom

    lengths = [float(axis[-1] - axis[0]) for axis in axes]
    top_area = lengths[0] * lengths[1]
    if top_area <= 0.0 or lengths[2] <= 0.0:
        raise AuditError(f"{sample_dir}: invalid physical extents")
    kx_h = _weighted_harmonic(k_diag[:, 0], volumes, "kx")
    ky_h = _weighted_harmonic(k_diag[:, 1], volumes, "ky")
    kz_h = _weighted_harmonic(k_diag[:, 2], volumes, "kz")
    anisotropy = math.sqrt(kx_h * ky_h) / kz_h
    params = _mapping(meta.get("boundary_params"), "boundary_params")
    top = _mapping(params.get("top"), "boundary_params.top")
    top_h = _finite_float(top.get("h_W_m2K"), "top h")
    if top_h <= 0.0:
        raise AuditError(f"{sample_dir}: top h must be positive")

    q_rms_operator = float(math.sqrt(np.dot(q_operator * q_operator, volumes) / total_volume))
    active_q_mask = np.abs(q_operator) > EPS
    q_active_fraction = float(volumes[active_q_mask].sum() / total_volume)
    q_mean_operator = p_operator / total_volume
    q_concentration = q_rms_operator / abs(q_mean_operator) if abs(q_mean_operator) > EPS else None
    q_positive = np.maximum(q_operator, 0.0)
    p_positive = float(np.dot(q_positive, volumes))
    source_z: float | None = None
    source_z_normalized: float | None = None
    if p_positive > EPS:
        source_z = float(np.dot(coords[:, 2], q_positive * volumes) / p_positive)
        source_z_normalized = (source_z - z_min) / lengths[2]

    r_top = 1.0 / (top_h * top_area)
    r_z = lengths[2] / (top_area * kz_h)
    r_series = r_top + r_z
    raw_centroid, r_centroid_bottom, r_centroid_top = _two_path_proxy(
        p_operator=p_operator,
        source_z=source_z,
        axes=axes,
        volumes=volumes,
        k_z=k_diag[:, 2],
        top_h=top_h,
        area=top_area,
        coords=coords,
    )
    raw_1d = _z_collapsed_1d_proxy(
        axes=axes,
        inverse=inverse,
        volumes=volumes,
        q_operator=q_operator,
        k_z=k_diag[:, 2],
        top_h=top_h,
        top_area=top_area,
        bc_offset=bc_offset,
    )
    raw_candidates: dict[str, float | None] = {
        "constant": 1.0,
        "power_only": p_operator if p_operator > EPS else None,
        "q_rms_lz2_over_kz": q_rms_operator * lengths[2] ** 2 / kz_h if q_rms_operator > EPS else None,
        "legacy_p_array_r_series": p_array * r_series if p_array > EPS else None,
        "source_centroid_two_path": raw_centroid,
        "z_collapsed_1d": raw_1d if raw_1d > EPS else None,
    }
    provenance = meta.get("p5_provenance")
    provenance_id = ""
    if isinstance(provenance, Mapping) and provenance.get("source_sample_id") is not None:
        provenance_id = str(provenance["source_sample_id"])

    row: dict[str, Any] = {
        "sample_id": sample_dir.name,
        "role": role,
        "driver_category": _driver_category(p_operator, bc_offset),
        "input_fingerprint": _fingerprint(arrays, INPUT_ARRAYS),
        "full_fingerprint": _fingerprint(arrays, FULL_ARRAYS),
        "provenance_source_id": provenance_id,
        "grid_shape": "x".join(map(str, shape)),
        "bottom_node_count": int(bottom_mask.sum()),
        "bottom_q_nonzero_count": bottom_q_nonzero,
        "bottom_mask_matches_bc": int(bottom_bc_matches),
        "bottom_mask_matches_metadata": int(bottom_metadata_matches),
        "bottom_temperature_label_pass": int(bottom_label_pass),
        "is_clean_role": int(role in CLEAN_ROLES),
        "is_hard_role": int(role in HARD_ROLES),
        "P_array_W": p_array,
        "P_bottom_W": p_bottom,
        "P_operator_W": p_operator,
        "P_identity_error_W": p_identity_error,
        "bottom_q_abs_max_W_m3": bottom_q_abs_max,
        "bottom_temperature_label_max_abs_error_K": bottom_label_error,
        "T_bottom_K": t_bottom,
        "T_inf_K": t_inf,
        "T_inf_minus_T_bottom_K": bc_offset,
        "cv_weight_sum_m3": total_volume,
        "Lx_m": lengths[0],
        "Ly_m": lengths[1],
        "Lz_m": lengths[2],
        "top_area_m2": top_area,
        "s_y_cv_rms_deltaT_K": s_y,
        "target_cv_mean_deltaT_K": float(np.dot(volumes, delta_t) / total_volume),
        "target_max_deltaT_K": float(delta_t.max()),
        "q_operator_cv_rms_W_m3": q_rms_operator,
        "q_active_cv_fraction": q_active_fraction,
        "q_rms_to_mean_concentration": q_concentration,
        "source_z_centroid_m": source_z,
        "source_z_centroid_normalized": source_z_normalized,
        "harmonic_kx_W_mK": kx_h,
        "harmonic_ky_W_mK": ky_h,
        "harmonic_kz_W_mK": kz_h,
        "anisotropy_xy_over_z": anisotropy,
        "top_h_W_m2K": top_h,
        "R_top_K_W": r_top,
        "R_z_K_W": r_z,
        "R_series_K_W": r_series,
        "R_centroid_bottom_K_W": r_centroid_bottom,
        "R_centroid_top_K_W": r_centroid_top,
    }
    for candidate, raw in raw_candidates.items():
        row[f"raw_{candidate}_K"] = raw
        row[f"pred_{candidate}_K"] = None
        row[f"log_residual_{candidate}"] = None
    row["selected_or_best_physics_candidate"] = ""
    row["selected_or_best_physics_prediction_K"] = None
    row["selected_or_best_physics_log_residual"] = None
    return row


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(_rank_average(x), _rank_average(y))[0, 1])


def _has_resolvable_spread(values: np.ndarray) -> bool:
    if values.size < 2:
        return False
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(np.ptp(values)) > 1.0e-12 * scale


def _evaluation_metrics(rows: Sequence[Mapping[str, Any]], candidate: str) -> dict[str, Any]:
    target = np.asarray([float(row["s_y_cv_rms_deltaT_K"]) for row in rows], dtype=np.float64)
    prediction = np.asarray(
        [float(row[f"pred_{candidate}_K"]) if row[f"pred_{candidate}_K"] is not None else np.nan for row in rows],
        dtype=np.float64,
    )
    valid = np.isfinite(target) & np.isfinite(prediction) & (target > EPS) & (prediction > EPS)
    if not np.all(valid):
        return {"available": False, "sample_count": int(valid.sum()), "reason": "non-positive or missing prediction"}
    log_target = np.log(target)
    log_prediction = np.log(prediction)
    error = log_prediction - log_target
    total = float(np.square(log_target - log_target.mean()).sum())
    slope: float | None = None
    if _has_resolvable_spread(log_prediction):
        slope = float(np.polyfit(log_prediction, log_target, deg=1)[0])
    ratio = prediction / target
    return {
        "available": True,
        "sample_count": int(target.size),
        "log_RMSE": float(math.sqrt(np.mean(error * error))),
        "log_MAE": float(np.mean(np.abs(error))),
        "Spearman_rho": _spearman(log_prediction, log_target),
        "log_R2": float(1.0 - np.square(error).sum() / total) if total > 0.0 else None,
        "log_slope": slope,
        "ratio_quantiles": {
            "q05": float(np.quantile(ratio, 0.05)),
            "q25": float(np.quantile(ratio, 0.25)),
            "q50": float(np.quantile(ratio, 0.50)),
            "q75": float(np.quantile(ratio, 0.75)),
            "q95": float(np.quantile(ratio, 0.95)),
        },
        "factor_2_accuracy": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    }


def _fit_calibrations(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    train_rows = [row for row in rows if row["role"] == CALIBRATION_ROLE]
    if not train_rows:
        raise AuditError("calibration requires train rows")
    calibrations: dict[str, dict[str, Any]] = {}
    log_target = np.log(np.asarray([float(row["s_y_cv_rms_deltaT_K"]) for row in train_rows], dtype=np.float64))
    for candidate in CANDIDATES:
        raw = np.asarray(
            [float(row[f"raw_{candidate}_K"]) if row[f"raw_{candidate}_K"] is not None else np.nan for row in train_rows],
            dtype=np.float64,
        )
        if candidate == "constant":
            intercept = float(log_target.mean())
            slope = 0.0
        else:
            if not np.all(np.isfinite(raw) & (raw > EPS)):
                raise AuditError(f"train calibration raw proxy is invalid: {candidate}")
            slope, intercept = np.polyfit(np.log(raw), log_target, deg=1)
            slope = float(slope)
            intercept = float(intercept)
        calibrations[candidate] = {
            "fit_role": CALIBRATION_ROLE,
            "fit_sample_count": len(train_rows),
            "intercept": intercept,
            "slope": slope,
            "form": "log(s_y)=intercept+slope*log(raw); constant uses slope=0",
        }
    return calibrations


def _apply_calibrations(rows: Sequence[dict[str, Any]], calibrations: Mapping[str, Mapping[str, Any]]) -> None:
    for row in rows:
        target = float(row["s_y_cv_rms_deltaT_K"])
        for candidate in CANDIDATES:
            raw = row[f"raw_{candidate}_K"]
            calibration = calibrations[candidate]
            prediction: float | None = None
            if candidate == "constant":
                prediction = float(math.exp(float(calibration["intercept"])))
            elif raw is not None and float(raw) > EPS:
                log_prediction = float(calibration["intercept"]) + float(calibration["slope"]) * math.log(float(raw))
                if log_prediction < 700.0:
                    prediction = float(math.exp(log_prediction))
            row[f"pred_{candidate}_K"] = prediction
            row[f"log_residual_{candidate}"] = (
                float(math.log(prediction / target)) if prediction is not None and target > EPS else None
            )


def _paired_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str = "constant",
) -> dict[str, Any]:
    target = np.asarray([float(row["s_y_cv_rms_deltaT_K"]) for row in rows], dtype=np.float64)
    pred_candidate = np.asarray([float(row[f"pred_{candidate}_K"]) for row in rows], dtype=np.float64)
    pred_reference = np.asarray([float(row[f"pred_{reference}_K"]) for row in rows], dtype=np.float64)
    if np.any(target <= EPS) or np.any(pred_candidate <= EPS) or np.any(pred_reference <= EPS):
        raise AuditError("paired bootstrap needs positive calibrated predictions")
    log_error_candidate = np.log(pred_candidate / target)
    log_error_reference = np.log(pred_reference / target)
    rng = np.random.default_rng(BOOTSTRAP_SEED + CANDIDATES.index(candidate))
    deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        draw = rng.integers(0, target.size, size=target.size)
        candidate_rmse = math.sqrt(float(np.mean(log_error_candidate[draw] ** 2)))
        reference_rmse = math.sqrt(float(np.mean(log_error_reference[draw] ** 2)))
        deltas[index] = candidate_rmse - reference_rmse
    return {
        "metric": "candidate_log_RMSE_minus_constant_log_RMSE",
        "paired_role": SELECTION_ROLE,
        "seed": BOOTSTRAP_SEED + CANDIDATES.index(candidate),
        "resamples": BOOTSTRAP_RESAMPLES,
        "point_estimate": float(math.sqrt(np.mean(log_error_candidate ** 2)) - math.sqrt(np.mean(log_error_reference ** 2))),
        "ci95": {
            "low": float(np.quantile(deltas, 0.025)),
            "median": float(np.quantile(deltas, 0.5)),
            "high": float(np.quantile(deltas, 0.975)),
        },
    }


def _select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["role"] == SELECTION_ROLE]
    if not valid_rows:
        raise AuditError("selection requires valid_iid rows")
    metrics = {candidate: _evaluation_metrics(valid_rows, candidate) for candidate in CANDIDATES}
    available = [candidate for candidate in PHYSICS_CANDIDATES if metrics[candidate]["available"]]
    if not available:
        raise AuditError("no physical proxy is available on valid_iid")
    best_physics = min(available, key=lambda candidate: float(metrics[candidate]["log_RMSE"]))
    bootstrap = {candidate: _paired_bootstrap_delta(valid_rows, candidate) for candidate in available}
    ci_high = float(bootstrap[best_physics]["ci95"]["high"])
    accepted = ci_high < 0.0
    return {
        "fit_role": CALIBRATION_ROLE,
        "selection_role": SELECTION_ROLE,
        "ood_role": OOD_ROLE,
        "test_roles_report_only": list(TEST_ROLES),
        "primary_metric": "valid_iid log_RMSE",
        "constant_metrics_valid_iid": metrics["constant"],
        "physical_metrics_valid_iid": {candidate: metrics[candidate] for candidate in available},
        "best_physics_candidate": best_physics,
        "paired_bootstrap_vs_constant": bootstrap,
        "decision_rule": "accept only when best physical candidate CI95 upper endpoint is below zero",
        "decision": "select_deterministic_physics_base" if accepted else "reject_all_single_proxy_require_later_global_scale_learning",
        "selected_candidate": best_physics if accepted else None,
        "residual_analysis_candidate": best_physics,
        "test_roles_used_for_selection": False,
    }


def _duplicate_groups(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field, ""))
        if key:
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, members in groups.items():
        roles = sorted({str(member["role"]) for member in members}, key=_role_key)
        if len(roles) > 1:
            output.append(
                {
                    "key": key,
                    "roles": roles,
                    "samples": [
                        {"sample_id": str(member["sample_id"]), "role": str(member["role"])}
                        for member in sorted(members, key=lambda member: str(member["sample_id"]))
                    ],
                }
            )
    return sorted(output, key=lambda item: (item["roles"], item["key"]))


def _duplicate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_groups = _duplicate_groups(rows, "input_fingerprint")
    full_groups = _duplicate_groups(rows, "full_fingerprint")
    provenance_groups = _duplicate_groups(rows, "provenance_source_id")
    ids = [str(row["sample_id"]) for row in rows]
    return {
        "unique_sample_ids": len(ids) == len(set(ids)),
        "cross_role_model_input_duplicate_groups": {"group_count": len(input_groups), "groups": input_groups},
        "cross_role_full_sample_duplicate_groups": {"group_count": len(full_groups), "groups": full_groups},
        "cross_role_provenance_duplicate_groups": {"group_count": len(provenance_groups), "groups": provenance_groups},
        "pass": len(ids) == len(set(ids)) and not (input_groups or full_groups or provenance_groups),
    }


def _operator_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "bottom_mask_matches_bc_count": int(sum(int(row["bottom_mask_matches_bc"]) for row in rows)),
        "bottom_mask_matches_metadata_count": int(sum(int(row["bottom_mask_matches_metadata"]) for row in rows)),
        "bottom_temperature_label_pass_count": int(sum(int(row["bottom_temperature_label_pass"]) for row in rows)),
        "P_array_W": _summary(row["P_array_W"] for row in rows),
        "P_bottom_W": _summary(row["P_bottom_W"] for row in rows),
        "P_operator_W": _summary(row["P_operator_W"] for row in rows),
        "P_identity_error_W": _summary(row["P_identity_error_W"] for row in rows),
        "bottom_q_nonzero_count": _summary(row["bottom_q_nonzero_count"] for row in rows),
        "bottom_temperature_label_max_abs_error_K": _summary(
            row["bottom_temperature_label_max_abs_error_K"] for row in rows
        ),
        "T_inf_minus_T_bottom_K": _summary(row["T_inf_minus_T_bottom_K"] for row in rows),
        "driver_categories": dict(sorted(Counter(str(row["driver_category"]) for row in rows).items())),
    }


def _feature_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "s_y_cv_rms_deltaT_K",
        "target_cv_mean_deltaT_K",
        "target_max_deltaT_K",
        "q_active_cv_fraction",
        "q_rms_to_mean_concentration",
        "source_z_centroid_normalized",
        "harmonic_kz_W_mK",
        "anisotropy_xy_over_z",
        "top_h_W_m2K",
        "T_inf_minus_T_bottom_K",
    )
    return {key: _summary(row[key] for row in rows) for key in keys}


def _split_summaries(rows: Sequence[Mapping[str, Any]], roles: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_groups = [(role, [row for row in rows if row["role"] == role]) for role in roles]
    all_groups.append(("all_samples", list(rows)))
    for role, group in all_groups:
        candidate_metrics = {candidate: _evaluation_metrics(group, candidate) for candidate in CANDIDATES}
        result[role] = {
            "sample_count": len(group),
            "operator_semantics": _operator_summary(group),
            "target_and_features": _feature_summary(group),
            "candidate_metrics": candidate_metrics,
        }
    return result


def _residual_relation(x_values: Sequence[float | None], residual_values: Sequence[float | None]) -> dict[str, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, residual_values)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return {"sample_count": len(pairs), "Spearman_rho": None, "slope": None, "R2": None}
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if not _has_resolvable_spread(x) or not _has_resolvable_spread(y):
        return {"sample_count": len(pairs), "Spearman_rho": _spearman(x, y), "slope": None, "R2": None}
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    total = float(np.square(y - y.mean()).sum())
    return {
        "sample_count": len(pairs),
        "Spearman_rho": _spearman(x, y),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(1.0 - np.square(y - prediction).sum() / total) if total > 0.0 else None,
    }


def _residual_analysis(rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> dict[str, Any]:
    candidate = str(decision["residual_analysis_candidate"])
    residual_key = f"log_residual_{candidate}"
    feature_keys = (
        "q_active_cv_fraction",
        "q_rms_to_mean_concentration",
        "source_z_centroid_normalized",
        "harmonic_kz_W_mK",
        "anisotropy_xy_over_z",
        "top_h_W_m2K",
        "T_inf_minus_T_bottom_K",
    )
    residuals = [row[residual_key] for row in rows]
    role_rows = {role: [row for row in rows if row["role"] == role] for role in sorted({str(row["role"]) for row in rows}, key=_role_key)}
    role_summary = {
        role: _summary(row[residual_key] for row in group)
        for role, group in role_rows.items()
    }
    clean_rows = [row for row in rows if int(row["is_clean_role"]) == 1]
    hard_rows = [row for row in rows if int(row["is_hard_role"]) == 1]
    return {
        "candidate": candidate,
        "selection_decision": decision["decision"],
        "scope": "all roles are descriptive post-selection reporting; test roles were not used for fit or selection",
        "numeric_relations_all_samples": {
            key: _residual_relation([row[key] for row in rows], residuals)
            for key in feature_keys
        },
        "residual_by_role": role_summary,
        "residual_by_clean_hard_group": {
            "clean": _summary(row[residual_key] for row in clean_rows),
            "hard": _summary(row[residual_key] for row in hard_rows),
        },
    }


def _table_columns() -> list[str]:
    base = [
        "sample_id", "role", "driver_category", "input_fingerprint", "full_fingerprint", "provenance_source_id", "grid_shape",
        "bottom_node_count", "bottom_q_nonzero_count", "bottom_mask_matches_bc", "bottom_mask_matches_metadata", "bottom_temperature_label_pass", "is_clean_role", "is_hard_role",
        "P_array_W", "P_bottom_W", "P_operator_W", "P_identity_error_W", "bottom_q_abs_max_W_m3", "bottom_temperature_label_max_abs_error_K", "T_bottom_K", "T_inf_K", "T_inf_minus_T_bottom_K",
        "cv_weight_sum_m3", "Lx_m", "Ly_m", "Lz_m", "top_area_m2", "s_y_cv_rms_deltaT_K", "target_cv_mean_deltaT_K", "target_max_deltaT_K",
        "q_operator_cv_rms_W_m3", "q_active_cv_fraction", "q_rms_to_mean_concentration", "source_z_centroid_m", "source_z_centroid_normalized",
        "harmonic_kx_W_mK", "harmonic_ky_W_mK", "harmonic_kz_W_mK", "anisotropy_xy_over_z", "top_h_W_m2K", "R_top_K_W", "R_z_K_W", "R_series_K_W", "R_centroid_bottom_K_W", "R_centroid_top_K_W",
    ]
    for candidate in CANDIDATES:
        base.extend((f"raw_{candidate}_K", f"pred_{candidate}_K", f"log_residual_{candidate}"))
    base.extend(("selected_or_best_physics_candidate", "selected_or_best_physics_prediction_K", "selected_or_best_physics_log_residual"))
    return base


TABLE_COLUMNS = _table_columns()


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["sample_id"])):
            encoded: dict[str, str] = {}
            for column in TABLE_COLUMNS:
                value = row.get(column)
                if value is None:
                    encoded[column] = ""
                elif column in STRING_COLUMNS:
                    encoded[column] = str(value)
                elif column in INT_COLUMNS:
                    encoded[column] = str(int(value))
                else:
                    encoded[column] = format(float(value), ".17g")
            writer.writerow(encoded)


def _read_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"per-sample table does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TABLE_COLUMNS:
            raise AuditError("per-sample table columns do not match Gate 1 schema")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for column in TABLE_COLUMNS:
                value = raw[column]
                if value == "":
                    row[column] = None
                elif column in STRING_COLUMNS:
                    row[column] = value
                elif column in INT_COLUMNS:
                    row[column] = int(value)
                else:
                    row[column] = float(value)
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_reconstructed(rows: Sequence[Mapping[str, Any]], roles: Sequence[str]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "role_counts": {role: int(sum(row["role"] == role for row in rows)) for role in roles},
        "duplicate_leakage": _duplicate_summary(rows),
        "split_summaries": _split_summaries(rows, roles),
    }


def _assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AuditError(f"summary reconstruction keys differ at {path}")
        for key in actual:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AuditError(f"summary reconstruction list length differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)) and not isinstance(actual, bool) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise AuditError(f"summary reconstruction differs at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AuditError(f"summary reconstruction differs at {path}: {actual!r} != {expected!r}")


def _verify_summary(table: Path, summary_path: Path) -> dict[str, Any]:
    rows = _read_table(table)
    payload = _read_json(summary_path)
    roles = [str(role) for role in _mapping(payload.get("dataset"), "dataset").get("roles", [])]
    if not roles:
        raise AuditError("summary has no roles")
    table_meta = _mapping(payload.get("per_sample_table"), "per_sample_table")
    if _sha256(table) != table_meta.get("sha256"):
        raise AuditError("per-sample table SHA256 does not match summary")
    reconstructed = _build_reconstructed(rows, roles)
    _assert_close(reconstructed, payload.get("reconstructed_from_table"), "reconstructed_from_table")
    calibrations = _fit_calibrations([dict(row) for row in rows])
    _assert_close(calibrations, payload.get("calibrations"), "calibrations")
    copied_rows = [dict(row) for row in rows]
    _apply_calibrations(copied_rows, calibrations)
    for original, rebuilt in zip(rows, copied_rows):
        for candidate in CANDIDATES:
            _assert_close(rebuilt[f"pred_{candidate}_K"], original[f"pred_{candidate}_K"], f"prediction.{candidate}")
            _assert_close(rebuilt[f"log_residual_{candidate}"], original[f"log_residual_{candidate}"], f"residual.{candidate}")
    decision = _select_candidate(copied_rows)
    _assert_close(decision, payload.get("selection"), "selection")
    residual = _residual_analysis(copied_rows, decision)
    _assert_close(residual, payload.get("residual_analysis"), "residual_analysis")
    return {"audit_id": AUDIT_ID, "verification": "passed", "row_count": len(rows), "table_sha256": _sha256(table)}


def _output_paths(dataset: Path, paths: Sequence[Path | None], overwrite: bool) -> tuple[Path, Path, Path]:
    if any(path is None for path in paths):
        raise AuditError("normal audit requires --output-table, --output-json, and --output-md")
    resolved = tuple(path.resolve() for path in paths if path is not None)
    if len(set(resolved)) != 3:
        raise AuditError("Gate 1 output paths must be distinct")
    dataset_path = dataset.resolve()
    for path in resolved:
        try:
            path.relative_to(dataset_path)
        except ValueError:
            pass
        else:
            raise AuditError(f"output path must not be inside dataset: {path}")
        if path.exists() and not overwrite:
            raise AuditError(f"output exists; use --overwrite: {path}")
    return resolved  # type: ignore[return-value]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0 or (1.0e-3 <= abs(number) < 1.0e4):
        return f"{number:.{digits}f}"
    return f"{number:.{digits}e}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    dataset = _mapping(payload["dataset"], "dataset")
    selection = _mapping(payload["selection"], "selection")
    reconstructed = _mapping(payload["reconstructed_from_table"], "reconstructed_from_table")
    summaries = _mapping(reconstructed["split_summaries"], "split_summaries")
    selected = selection.get("selected_candidate") or selection["best_physics_candidate"]
    lines = [
        "# V5 Gate 1 Closeout: Operator-Consistent Physics Scale",
        "",
        "## Scope",
        "",
        f"- Dataset: `{dataset['dataset_id']}`; samples: `{dataset['sample_count']}`.",
        "- Read-only audit: no model/loss/config/data change, no training, and no reference-label solver call.",
        "- P0 remains intact; its historical effective source power is preserved here as `P_array`.",
        "",
        "## Frozen Operator Semantics",
        "",
        "`P_array = sum_all(q*CV)`, `P_bottom = sum_bottom(q*CV)`, and `P_operator = sum_non_bottom(q*CV)`. The latter matches V4 bottom-row replacement: bottom Dirichlet rows receive `T_bottom`, not `q*CV`.",
        "",
            "| role | n | P_array mean W | P_bottom mean W | P_operator mean W | BC offset range K | bottom label max error K | driver categories |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for role in [*dataset["roles"], "all_samples"]:
        semantic = _mapping(_mapping(summaries[role], "role")["operator_semantics"], "operator semantics")
        lines.append(
            "| {role} | {n} | {pa} | {pb} | {po} | {bc} | {bottom} | {drivers} |".format(
                role=role,
                n=_mapping(summaries[role], "role")["sample_count"],
                pa=_fmt(_mapping(semantic["P_array_W"], "pa")["mean"]),
                pb=_fmt(_mapping(semantic["P_bottom_W"], "pb")["mean"]),
                po=_fmt(_mapping(semantic["P_operator_W"], "po")["mean"]),
                bc="[{low}, {high}]".format(
                    low=_fmt(_mapping(semantic["T_inf_minus_T_bottom_K"], "bc")["min"]),
                    high=_fmt(_mapping(semantic["T_inf_minus_T_bottom_K"], "bc")["max"]),
                ),
                bottom=_fmt(_mapping(semantic["bottom_temperature_label_max_abs_error_K"], "bottom")["max"]),
                drivers=", ".join(f"{key}:{value}" for key, value in semantic["driver_categories"].items()),
            )
        )
    lines.extend(
        [
            "",
            "## Calibration And Selection",
            "",
            "All calibrations fit only `train`; candidate selection uses only `valid_iid`; `hard_challenge_valid` is OOD inspection; test roles are report-only after selection.",
            f"- Best physical valid candidate: `{selection['best_physics_candidate']}`.",
            f"- Decision: `{selection['decision']}`.",
            f"- Selected deterministic base: `{selection.get('selected_candidate') or 'none'}`.",
            "",
            "| candidate | valid log-RMSE | valid log-MAE | valid Spearman | valid factor-2 | paired delta CI95 vs constant | hard-challenge-valid log-RMSE | test_iid log-RMSE |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    valid_metrics = _mapping(selection["physical_metrics_valid_iid"], "physical metrics")
    bootstrap = _mapping(selection["paired_bootstrap_vs_constant"], "bootstrap")
    for candidate in CANDIDATES:
        if candidate == "constant":
            valid = _mapping(selection["constant_metrics_valid_iid"], "constant metrics")
            ci = "reference"
        else:
            valid = _mapping(valid_metrics[candidate], "candidate metrics")
            ci_values = _mapping(_mapping(bootstrap[candidate], "bootstrap")["ci95"], "ci values")
            ci = f"[{_fmt(ci_values['low'])}, {_fmt(ci_values['high'])}]"
        hard = _mapping(_mapping(summaries[OOD_ROLE], "ood")["candidate_metrics"], "ood metrics")[candidate]
        test = _mapping(_mapping(summaries["test_iid"], "test")["candidate_metrics"], "test metrics")[candidate]
        lines.append(
            "| {candidate} | {rmse} | {mae} | {rho} | {factor2} | {ci} | {hard} | {test} |".format(
                candidate=candidate,
                rmse=_fmt(valid.get("log_RMSE")),
                mae=_fmt(valid.get("log_MAE")),
                rho=_fmt(valid.get("Spearman_rho")),
                factor2=_fmt(valid.get("factor_2_accuracy")),
                ci=ci,
                hard=_fmt(_mapping(hard, "hard").get("log_RMSE")),
                test=_fmt(_mapping(test, "test").get("log_RMSE")),
            )
        )
    lines.extend(
        [
            "",
            "### Valid-IID Calibration Detail",
            "",
            "| candidate | valid log-R2 | valid log slope | ratio q05 | ratio q50 | ratio q95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in CANDIDATES:
        if candidate == "constant":
            valid = _mapping(selection["constant_metrics_valid_iid"], "constant metrics")
        else:
            valid = _mapping(valid_metrics[candidate], "candidate metrics")
        ratios = _mapping(valid["ratio_quantiles"], "ratio quantiles")
        lines.append(
            "| {candidate} | {r2} | {slope} | {q05} | {q50} | {q95} |".format(
                candidate=candidate,
                r2=_fmt(valid.get("log_R2")),
                slope=_fmt(valid.get("log_slope")),
                q05=_fmt(ratios.get("q05")),
                q50=_fmt(ratios.get("q50")),
                q95=_fmt(ratios.get("q95")),
            )
        )
    residual = _mapping(payload["residual_analysis"], "residual analysis")
    lines.extend(
        [
            "",
            "## Residual Analysis",
            "",
            f"Residuals are `log(pred/s_y)` for `{selected}`. Test-role rows appear only in this post-selection descriptive report.",
            "",
            "| feature | n | Spearman rho | slope | R2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for feature, relation in _mapping(residual["numeric_relations_all_samples"], "relations").items():
        relation_map = _mapping(relation, "relation")
        lines.append(
            f"| {feature} | {relation_map['sample_count']} | {_fmt(relation_map.get('Spearman_rho'))} | {_fmt(relation_map.get('slope'))} | {_fmt(relation_map.get('R2'))} |"
        )
    clean_hard = _mapping(residual["residual_by_clean_hard_group"], "clean hard residual")
    lines.extend(
        [
            "",
            "| role family | n | residual mean | residual median | residual std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in ("clean", "hard"):
        stats = _mapping(clean_hard[family], "family residual")
        lines.append(
            "| {family} | {count} | {mean} | {median} | {std} |".format(
                family=family,
                count=stats["count"],
                mean=_fmt(stats["mean"]),
                median=_fmt(stats["median"]),
                std=_fmt(stats["std"]),
            )
        )
    leakage = _mapping(reconstructed["duplicate_leakage"], "leakage")
    lines.extend(
        [
            "",
            "## Integrity And Reproducibility",
            "",
            f"- Per-sample table rows: `{reconstructed['row_count']}`; SHA256: `{_mapping(payload['per_sample_table'], 'table')['sha256']}`.",
            f"- Cross-role model-input/full-sample/provenance duplicate groups: `{_mapping(leakage['cross_role_model_input_duplicate_groups'], 'input')['group_count']}` / `{_mapping(leakage['cross_role_full_sample_duplicate_groups'], 'full')['group_count']}` / `{_mapping(leakage['cross_role_provenance_duplicate_groups'], 'provenance')['group_count']}`.",
            "- `--verify-summary` regenerates split summaries, calibration, predictions, selection, and residual analysis from the CSV only.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_md(payload: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))


def _run_audit(
    dataset: Path,
    split_map: Path,
    contract_path: Path | None,
    output_table: Path,
    output_json: Path,
    output_md: Path,
    table_label: str | None,
) -> dict[str, Any]:
    assignments, roles, split_payload, contract = _load_inputs(dataset, split_map, contract_path)
    rows = [_sample_row(dataset / sample_id, assignments[sample_id]) for sample_id in sorted(assignments)]
    calibrations = _fit_calibrations(rows)
    _apply_calibrations(rows, calibrations)
    selection = _select_candidate(rows)
    chosen = str(selection["selected_candidate"] or selection["best_physics_candidate"])
    for row in rows:
        row["selected_or_best_physics_candidate"] = chosen
        row["selected_or_best_physics_prediction_K"] = row[f"pred_{chosen}_K"]
        row["selected_or_best_physics_log_residual"] = row[f"log_residual_{chosen}"]
    _write_table(rows, output_table)
    reconstructed = _build_reconstructed(rows, roles)
    residual = _residual_analysis(rows, selection)
    payload = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only",
        "contract_id": contract.get("contract_id") if contract else None,
        "p0_backward_traceability": {
            "audit_sha256": P0_AUDIT_SHA256,
            "historical_effective_source_power_term": "P_array_W",
            "p0_files_modified": False,
        },
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "dataset_path": dataset.as_posix(),
            "split_map_path": split_map.as_posix(),
            "sample_count": len(rows),
            "roles": roles,
            "role_counts": {role: int(sum(row["role"] == role for row in rows)) for role in roles},
        },
        "operator_semantics": {
            "P_array_W": "sum_all_nodes(q*CV_volume)",
            "P_bottom_W": "sum_bottom_Dirichlet_nodes(q*CV_volume)",
            "P_operator_W": "sum_non_bottom_nodes(q*CV_volume)",
            "bottom_row_policy": "V4 _assemble_triplets replaces bottom rows with T=T_bottom before source RHS deposition",
            "source_total_policy": "V4 _source_power_total iterates iz=1..end",
            "target": "s_y = CV-weighted RMS(temperature-T_bottom)",
        },
        "read_only_guardrails": {
            "sample_array_writes": 0,
            "sample_metadata_writes": 0,
            "reference_solver_calls": 0,
            "model_calls": 0,
            "training_runs": 0,
            "permitted_writes": ["explicit Gate 1 CSV", "explicit Gate 1 JSON", "explicit Gate 1 Markdown"],
        },
        "calibrations": calibrations,
        "selection": selection,
        "residual_analysis": residual,
        "per_sample_table": {
            "path": table_label or output_table.as_posix(),
            "sha256": _sha256(output_table),
            "row_count": len(rows),
            "columns": TABLE_COLUMNS,
        },
        "reconstructed_from_table": reconstructed,
        "gate1_closeout": {
            "decision": selection["decision"],
            "selected_deterministic_physics_base": selection["selected_candidate"],
            "next_gate_authorized": False,
            "models_or_training_changed": False,
        },
    }
    _write_json_md(payload, output_json, output_md)
    return payload


def _dry_run(
    dataset: Path,
    split_map: Path,
    contract_path: Path | None,
) -> dict[str, Any]:
    assignments, roles, split_payload, contract = _load_inputs(dataset, split_map, contract_path)
    return {
        "audit_id": AUDIT_ID,
        "mode": "dry_run",
        "read_only": True,
        "contract_id": contract.get("contract_id") if contract else None,
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "path": dataset.as_posix(),
            "sample_count": len(assignments),
            "role_counts": {role: int(sum(value == role for value in assignments.values())) for role in roles},
        },
        "planned_reads": {"sample_directories": len(assignments), "arrays": 0, "sample_metadata": 0},
        "planned_writes": [],
        "reference_solver_calls": 0,
        "model_calls": 0,
        "training_runs": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--table-label", help="canonical repository label for the generated CSV table")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-summary", action="store_true")
    parser.add_argument("--table", type=Path, help="CSV table used with --verify-summary")
    parser.add_argument("--summary-json", type=Path, help="JSON summary used with --verify-summary")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_summary:
            if args.table is None or args.summary_json is None:
                raise AuditError("--verify-summary requires --table and --summary-json")
            print(json.dumps(_verify_summary(args.table, args.summary_json), indent=2, sort_keys=True))
            return 0
        if args.dataset is None or args.split_map is None:
            raise AuditError("audit requires --dataset and --split-map")
        if args.dry_run:
            print(json.dumps(_dry_run(args.dataset, args.split_map, args.contract), indent=2, sort_keys=True))
            return 0
        table, summary_json, summary_md = _output_paths(
            args.dataset,
            (args.output_table, args.output_json, args.output_md),
            args.overwrite,
        )
        _run_audit(
            args.dataset,
            args.split_map,
            args.contract,
            table,
            summary_json,
            summary_md,
            args.table_label,
        )
    except AuditError as exc:
        print(f"Gate 1 audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
