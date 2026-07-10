#!/usr/bin/env python3
"""Read-only V5-P0-1 physics-scale audit for the frozen P5 dataset.

The script never changes samples, split maps, model code, or training state.
Outside ``--dry-run``, its only writes are the explicitly requested JSON and
Markdown reports, and it rejects report paths inside the dataset directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


AUDIT_ID = "V5-P0-1"
INPUT_ARRAY_NAMES = ("coords", "k_field", "q_field", "bc_features")
FULL_ARRAY_NAMES = INPUT_ARRAY_NAMES + ("temperature",)
P5_ROLE_ORDER = (
    "train",
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)


class AuditError(RuntimeError):
    """Raised when an input violates the P0 data contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{field} must be finite, got {value!r}")
    return result


def _nested_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{field} must be an object")
    return value


def _role_sort_key(role: str) -> tuple[int, str]:
    try:
        return (P5_ROLE_ORDER.index(role), role)
    except ValueError:
        return (len(P5_ROLE_ORDER), role)


def _sample_directories(dataset: Path) -> set[str]:
    return {
        path.name
        for path in dataset.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    }


def _load_split_contract(
    dataset: Path,
    split_map: Path,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    if not dataset.is_dir():
        raise AuditError(f"dataset directory does not exist: {dataset}")
    if not split_map.is_file():
        raise AuditError(f"split map does not exist: {split_map}")

    payload = _read_json(split_map)
    raw_assignments = payload.get("sample_splits")
    if not isinstance(raw_assignments, Mapping) or not raw_assignments:
        raise AuditError(f"{split_map}: sample_splits must be a non-empty object")
    assignments = {str(sample_id): str(role) for sample_id, role in raw_assignments.items()}
    if len(assignments) != len(raw_assignments):
        raise AuditError(f"{split_map}: duplicate sample IDs after string normalization")

    samples_on_disk = _sample_directories(dataset)
    assigned_ids = set(assignments)
    missing_dirs = sorted(assigned_ids - samples_on_disk)
    unassigned_dirs = sorted(samples_on_disk - assigned_ids)
    if missing_dirs:
        raise AuditError(f"split-map samples missing from dataset: {missing_dirs[:8]}")
    if unassigned_dirs:
        raise AuditError(f"dataset samples absent from split map: {unassigned_dirs[:8]}")

    actual_counts = Counter(assignments.values())
    declared_counts = payload.get("actual_counts")
    if declared_counts is not None:
        declared = _nested_mapping(declared_counts, "actual_counts")
        normalized_declared = {str(role): int(count) for role, count in declared.items()}
        if normalized_declared != dict(actual_counts):
            raise AuditError(
                "split-map actual_counts disagree with sample_splits: "
                f"declared={normalized_declared}, actual={dict(actual_counts)}"
            )

    roles = sorted(actual_counts, key=_role_sort_key)
    return assignments, roles, payload


def _validate_optional_contract(
    contract_path: Path | None,
    split_payload: Mapping[str, Any],
    roles: Sequence[str],
) -> dict[str, Any] | None:
    if contract_path is None:
        return None
    if not contract_path.is_file():
        raise AuditError(f"contract does not exist: {contract_path}")
    contract = _read_json(contract_path)
    dataset_contract = _nested_mapping(contract.get("dataset_contract"), "dataset_contract")
    expected_dataset_id = dataset_contract.get("dataset_id")
    actual_dataset_id = split_payload.get("dataset_id")
    if expected_dataset_id is not None and expected_dataset_id != actual_dataset_id:
        raise AuditError(
            "contract dataset_id disagrees with split map: "
            f"{expected_dataset_id!r} != {actual_dataset_id!r}"
        )
    required_roles = dataset_contract.get("required_split_roles")
    if required_roles is not None:
        if not isinstance(required_roles, list):
            raise AuditError("dataset_contract.required_split_roles must be a list")
        if set(map(str, required_roles)) != set(roles):
            raise AuditError(
                "contract required_split_roles disagree with split map: "
                f"{required_roles!r} != {list(roles)!r}"
            )
    actual_counts = Counter(str(role) for role in _nested_mapping(split_payload.get("sample_splits"), "sample_splits").values())
    expected_counts = dataset_contract.get("expected_role_counts")
    if expected_counts is not None:
        if not isinstance(expected_counts, Mapping):
            raise AuditError("dataset_contract.expected_role_counts must be an object")
        normalized_expected = {str(role): int(count) for role, count in expected_counts.items()}
        if normalized_expected != dict(actual_counts):
            raise AuditError(
                "contract expected_role_counts disagree with split map: "
                f"{normalized_expected} != {dict(actual_counts)}"
            )
    expected_total = dataset_contract.get("expected_total_sample_count")
    if expected_total is not None and int(expected_total) != sum(actual_counts.values()):
        raise AuditError(
            "contract expected_total_sample_count disagrees with split map: "
            f"{expected_total} != {sum(actual_counts.values())}"
        )
    audit_contract = _nested_mapping(contract.get("audit_contract"), "audit_contract")
    if audit_contract.get("mode") != "read_only":
        raise AuditError("V5 P0 audit contract must specify read_only mode")
    return contract


def _control_widths(axis: np.ndarray, field: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise AuditError(f"{field} needs at least two unique coordinates")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise AuditError(f"{field} coordinates must be finite and strictly increasing")
    widths = np.empty_like(axis)
    widths[0] = 0.5 * (axis[1] - axis[0])
    widths[-1] = 0.5 * (axis[-1] - axis[-2])
    if axis.size > 2:
        widths[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return widths


def _control_volume_weights(coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise AuditError(f"coords must have shape [N, 3], got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise AuditError("coords contain NaN or Inf")

    axes: list[np.ndarray] = []
    inverse_indices: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    for dimension, label in enumerate(("x", "y", "z")):
        axis, inverse = np.unique(coords[:, dimension], return_inverse=True)
        axes.append(axis)
        inverse_indices.append(inverse)
        widths.append(_control_widths(axis, label))

    grid_shape = [int(axis.size) for axis in axes]
    expected_nodes = int(np.prod(grid_shape))
    if expected_nodes != coords.shape[0]:
        raise AuditError(
            "coords are not a complete rectilinear grid: "
            f"grid={grid_shape}, nodes={coords.shape[0]}"
        )
    flattened = np.ravel_multi_index(tuple(inverse_indices), tuple(grid_shape))
    if np.unique(flattened).size != coords.shape[0]:
        raise AuditError("coords contain duplicate grid positions")

    weights = (
        widths[0][inverse_indices[0]]
        * widths[1][inverse_indices[1]]
        * widths[2][inverse_indices[2]]
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise AuditError("control-volume weights must be finite and positive")
    return weights, axes, grid_shape


def _array_fingerprint(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _load_arrays(sample_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in FULL_ARRAY_NAMES:
        path = sample_dir / f"{name}.npy"
        if not path.is_file():
            raise AuditError(f"missing required array: {path}")
        try:
            arrays[name] = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise AuditError(f"cannot read {path}: {exc}") from exc
    return arrays


def _reference_temperature(meta: Mapping[str, Any]) -> float:
    boundary_params = _nested_mapping(meta.get("boundary_params"), "boundary_params")
    bottom = _nested_mapping(boundary_params.get("bottom"), "boundary_params.bottom")
    for key in ("T_fixed_K", "fixed_temperature_K"):
        if key in bottom:
            return _finite_float(bottom[key], f"boundary_params.bottom.{key}")
    raise AuditError("boundary_params.bottom needs T_fixed_K or fixed_temperature_K")


def _sample_metrics(sample_dir: Path, role: str) -> dict[str, Any]:
    meta = _read_json(sample_dir / "sample_meta.json")
    arrays = _load_arrays(sample_dir)
    coords = np.asarray(arrays["coords"], dtype=np.float64)
    k_field = np.asarray(arrays["k_field"], dtype=np.float64)
    q_field = np.asarray(arrays["q_field"], dtype=np.float64).reshape(-1)
    bc_features = np.asarray(arrays["bc_features"], dtype=np.float64)
    temperature = np.asarray(arrays["temperature"], dtype=np.float64).reshape(-1)

    node_count = coords.shape[0]
    if k_field.shape == (node_count, 1):
        k_for_metrics = np.repeat(k_field, 3, axis=1)
        k_field_width = 1
    elif k_field.shape == (node_count, 3):
        k_for_metrics = k_field
        k_field_width = 3
    else:
        raise AuditError(
            f"{sample_dir}: k_field must have shape [{node_count}, 1] or [{node_count}, 3]"
        )
    if q_field.shape != (node_count,):
        raise AuditError(f"{sample_dir}: q_field node count mismatch")
    if temperature.shape != (node_count,):
        raise AuditError(f"{sample_dir}: temperature node count mismatch")
    if bc_features.shape[0] != node_count:
        raise AuditError(f"{sample_dir}: bc_features node count mismatch")
    if not all(np.all(np.isfinite(value)) for value in (k_field, q_field, bc_features, temperature)):
        raise AuditError(f"{sample_dir}: arrays contain NaN or Inf")
    if np.any(k_for_metrics <= 0.0):
        raise AuditError(f"{sample_dir}: k_field must be positive")

    weights, axes, grid_shape = _control_volume_weights(coords)
    volume = float(weights.sum())
    q_audit = _nested_mapping(meta.get("q_power_audit"), "q_power_audit")
    recorded_volume = _finite_float(
        q_audit.get("control_volume_weight_sum_m3"),
        "q_power_audit.control_volume_weight_sum_m3",
    )
    volume_error = abs(volume - recorded_volume)
    if volume_error > max(1.0e-18, abs(recorded_volume) * 1.0e-10):
        raise AuditError(
            f"{sample_dir}: control-volume sum mismatch: {volume} != {recorded_volume}"
        )

    effective_power = float(np.dot(q_field, weights))
    recorded_integral = _finite_float(
        q_audit.get("q_integral_from_array_W"),
        "q_power_audit.q_integral_from_array_W",
    )
    recorded_target = _finite_float(
        q_audit.get("q_total_target_power_W"),
        "q_power_audit.q_total_target_power_W",
    )
    integral_error = effective_power - recorded_integral
    target_error = effective_power - recorded_target
    power_tolerance = max(1.0e-12, abs(recorded_target) * 1.0e-10)
    if max(abs(integral_error), abs(target_error)) > power_tolerance:
        raise AuditError(
            f"{sample_dir}: q power mismatch, effective={effective_power}, "
            f"recorded={recorded_integral}, target={recorded_target}"
        )

    reference_temperature = _reference_temperature(meta)
    delta_t = temperature - reference_temperature
    if not np.all(np.isfinite(delta_t)):
        raise AuditError(f"{sample_dir}: DeltaT contains NaN or Inf")
    cv_mean = float(np.dot(delta_t, weights) / volume)
    cv_rms = float(math.sqrt(np.dot(delta_t * delta_t, weights) / volume))
    delta_t_max = float(delta_t.max())

    kz_harmonic = float(volume / np.dot(weights, 1.0 / k_for_metrics[:, 2]))
    q_cv_rms = float(math.sqrt(np.dot(q_field * q_field, weights) / volume))
    positive_q_volume_fraction = float(weights[q_field > 0.0].sum() / volume)
    lengths = [float(axis[-1] - axis[0]) for axis in axes]
    top_area = lengths[0] * lengths[1]
    if top_area <= 0.0 or lengths[2] <= 0.0:
        raise AuditError(f"{sample_dir}: invalid physical extents")

    boundary_params = _nested_mapping(meta.get("boundary_params"), "boundary_params")
    top = _nested_mapping(boundary_params.get("top"), "boundary_params.top")
    top_h = _finite_float(top.get("h_W_m2K"), "boundary_params.top.h_W_m2K")
    if top_h <= 0.0:
        raise AuditError(f"{sample_dir}: top_h must be positive")
    top_robin_resistance = 1.0 / (top_h * top_area)
    z_conduction_resistance = lengths[2] / (top_area * kz_harmonic)
    series_resistance = top_robin_resistance + z_conduction_resistance
    top_robin_delta_t_proxy = effective_power * top_robin_resistance
    series_delta_t_proxy = effective_power * series_resistance
    target_to_series_ratio = (
        cv_mean / series_delta_t_proxy if abs(series_delta_t_proxy) > 1.0e-18 else None
    )

    provenance = meta.get("p5_provenance")
    provenance_source_id: str | None = None
    if isinstance(provenance, Mapping) and provenance.get("source_sample_id") is not None:
        provenance_source_id = str(provenance["source_sample_id"])

    return {
        "sample_id": sample_dir.name,
        "role": role,
        "input_fingerprint": _array_fingerprint(arrays, INPUT_ARRAY_NAMES),
        "full_fingerprint": _array_fingerprint(arrays, FULL_ARRAY_NAMES),
        "provenance_source_id": provenance_source_id,
        "grid_shape": grid_shape,
        "k_field_width": k_field_width,
        "metrics": {
            "cv_weight_sum_m3": volume,
            "cv_weight_min_m3": float(weights.min()),
            "cv_weight_max_m3": float(weights.max()),
            "metadata_cv_weight_sum_abs_error_m3": volume_error,
            "effective_source_power_W": effective_power,
            "metadata_q_integral_error_W": integral_error,
            "metadata_q_target_error_W": target_error,
            "target_deltaT_cv_rms_K": cv_rms,
            "target_deltaT_cv_mean_K": cv_mean,
            "target_deltaT_max_K": delta_t_max,
            "q_cv_rms_W_m3": q_cv_rms,
            "positive_q_cv_volume_fraction": positive_q_volume_fraction,
            "harmonic_kz_W_mK": kz_harmonic,
            "top_h_W_m2K": top_h,
            "top_robin_resistance_K_W": top_robin_resistance,
            "z_conduction_resistance_proxy_K_W": z_conduction_resistance,
            "series_thermal_resistance_proxy_K_W": series_resistance,
            "top_robin_deltaT_proxy_K": top_robin_delta_t_proxy,
            "series_deltaT_proxy_K": series_delta_t_proxy,
            "target_cv_mean_to_series_proxy_ratio": target_to_series_ratio,
        },
    }


def _numeric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    array = np.asarray(finite_values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def _linear_relation(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise AuditError("linear relation inputs must be equal one-dimensional vectors")
    result: dict[str, Any] = {
        "sample_count": int(x.size),
        "pearson_r": None,
        "slope": None,
        "intercept": None,
        "r_squared": None,
    }
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return result
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    residual = float(np.square(y - predicted).sum())
    total = float(np.square(y - y.mean()).sum())
    result.update(
        {
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(1.0 - residual / total) if total > 0.0 else None,
        }
    )
    return result


def _multiple_linear_relation(
    predictors: Mapping[str, Sequence[float]],
    target: Sequence[float],
) -> dict[str, Any]:
    names = list(predictors)
    columns = [np.asarray(predictors[name], dtype=np.float64) for name in names]
    y = np.asarray(target, dtype=np.float64)
    if not columns or any(column.shape != y.shape for column in columns):
        raise AuditError("multiple linear relation vectors must share a shape")
    result: dict[str, Any] = {
        "sample_count": int(y.size),
        "predictors": names,
        "intercept": None,
        "coefficients": None,
        "r_squared": None,
    }
    if y.size <= len(columns) or np.ptp(y) == 0.0:
        return result
    design = np.column_stack([np.ones_like(y), *columns])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        return result
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    residual = float(np.square(y - predicted).sum())
    total = float(np.square(y - y.mean()).sum())
    result.update(
        {
            "intercept": float(coefficients[0]),
            "coefficients": {name: float(value) for name, value in zip(names, coefficients[1:])},
            "r_squared": float(1.0 - residual / total) if total > 0.0 else None,
        }
    )
    return result


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise AuditError("cannot summarize an empty split")
    metrics = [record["metrics"] for record in records]

    def summary(name: str) -> dict[str, Any]:
        return _numeric_summary([metric[name] for metric in metrics])

    power = [float(metric["effective_source_power_W"]) for metric in metrics]
    top_h = [float(metric["top_h_W_m2K"]) for metric in metrics]
    target_mean = [float(metric["target_deltaT_cv_mean_K"]) for metric in metrics]
    top_proxy = [float(metric["top_robin_deltaT_proxy_K"]) for metric in metrics]
    grid_shapes = Counter("x".join(map(str, record["grid_shape"])) for record in records)
    k_field_widths = Counter(str(record["k_field_width"]) for record in records)
    return {
        "sample_count": len(records),
        "control_volume_weights": {
            "grid_shapes": dict(sorted(grid_shapes.items())),
            "k_field_width_counts": dict(sorted(k_field_widths.items())),
            "cv_weight_sum_m3": summary("cv_weight_sum_m3"),
            "cv_weight_min_m3": summary("cv_weight_min_m3"),
            "cv_weight_max_m3": summary("cv_weight_max_m3"),
            "metadata_cv_weight_sum_abs_error_m3": summary("metadata_cv_weight_sum_abs_error_m3"),
        },
        "effective_source_power": {
            "effective_source_power_W": summary("effective_source_power_W"),
            "metadata_q_integral_error_W": summary("metadata_q_integral_error_W"),
            "metadata_q_target_error_W": summary("metadata_q_target_error_W"),
        },
        "target_statistics": {
            "target_deltaT_cv_rms_K": summary("target_deltaT_cv_rms_K"),
            "target_deltaT_cv_mean_K": summary("target_deltaT_cv_mean_K"),
            "target_deltaT_max_K": summary("target_deltaT_max_K"),
        },
        "physics_scale_proxies": {
            "q_cv_rms_W_m3": summary("q_cv_rms_W_m3"),
            "positive_q_cv_volume_fraction": summary("positive_q_cv_volume_fraction"),
            "harmonic_kz_W_mK": summary("harmonic_kz_W_mK"),
            "top_h_W_m2K": summary("top_h_W_m2K"),
            "top_robin_resistance_K_W": summary("top_robin_resistance_K_W"),
            "z_conduction_resistance_proxy_K_W": summary("z_conduction_resistance_proxy_K_W"),
            "series_thermal_resistance_proxy_K_W": summary("series_thermal_resistance_proxy_K_W"),
            "top_robin_deltaT_proxy_K": summary("top_robin_deltaT_proxy_K"),
            "series_deltaT_proxy_K": summary("series_deltaT_proxy_K"),
            "target_cv_mean_to_series_proxy_ratio": summary("target_cv_mean_to_series_proxy_ratio"),
        },
        "q_bc_linear_relations": {
            "effective_source_power_vs_top_h": _linear_relation(power, top_h),
            "target_cv_mean_vs_effective_source_power": _linear_relation(power, target_mean),
            "target_cv_mean_vs_top_robin_deltaT_proxy": _linear_relation(top_proxy, target_mean),
            "target_cv_mean_vs_q_bc_predictors": _multiple_linear_relation(
                {
                    "effective_source_power_W": power,
                    "top_robin_deltaT_proxy_K": top_proxy,
                },
                target_mean,
            ),
        },
    }


def _cross_role_duplicate_groups(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is not None and value != "":
            groups[str(value)].append(record)
    findings: list[dict[str, Any]] = []
    for value, members in groups.items():
        roles = sorted({str(member["role"]) for member in members}, key=_role_sort_key)
        if len(roles) > 1:
            findings.append(
                {
                    "key": value,
                    "roles": roles,
                    "samples": [
                        {"sample_id": str(member["sample_id"]), "role": str(member["role"])}
                        for member in sorted(members, key=lambda member: str(member["sample_id"]))
                    ],
                }
            )
    return sorted(findings, key=lambda finding: (finding["roles"], finding["key"]))


def _duplicate_leakage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_input = _cross_role_duplicate_groups(records, "input_fingerprint")
    full_sample = _cross_role_duplicate_groups(records, "full_fingerprint")
    provenance = _cross_role_duplicate_groups(records, "provenance_source_id")
    return {
        "method": {
            "model_input_fingerprint": "SHA256 over coords/k/q/BC arrays including dtype and shape",
            "full_sample_fingerprint": "SHA256 over coords/k/q/BC/temperature arrays including dtype and shape",
            "provenance_key": "sample_meta.p5_provenance.source_sample_id",
            "geometry_note": "coords alone are intentionally shared by the fixed P5 grid and are not treated as leakage.",
        },
        "cross_role_model_input_duplicate_groups": {
            "group_count": len(model_input),
            "groups": model_input,
        },
        "cross_role_full_sample_duplicate_groups": {
            "group_count": len(full_sample),
            "groups": full_sample,
        },
        "cross_role_provenance_duplicate_groups": {
            "group_count": len(provenance),
            "groups": provenance,
        },
        "pass": not (model_input or full_sample or provenance),
    }


def _relative_or_text(path: Path) -> str:
    return path.as_posix()


def _dry_run_payload(
    dataset: Path,
    split_map: Path,
    assignments: Mapping[str, str],
    roles: Sequence[str],
    split_payload: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts = Counter(assignments.values())
    return {
        "audit_id": AUDIT_ID,
        "mode": "dry_run",
        "read_only": True,
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "path": _relative_or_text(dataset),
            "sample_count": len(assignments),
        },
        "split_map": {
            "path": _relative_or_text(split_map),
            "role_counts": {role: int(counts[role]) for role in roles},
        },
        "contract_id": contract.get("contract_id") if contract else None,
        "planned_reads": {
            "sample_directories": len(assignments),
            "array_payloads": 0,
            "sample_metadata_payloads": 0,
        },
        "planned_writes": [],
        "solver_calls": 0,
        "training_runs": 0,
    }


def run_audit(
    dataset: Path,
    split_map: Path,
    contract: Mapping[str, Any] | None,
    assignments: Mapping[str, str],
    roles: Sequence[str],
    split_payload: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sample_id in sorted(assignments):
        records.append(_sample_metrics(dataset / sample_id, assignments[sample_id]))

    grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    for record in records:
        grouped[str(record["role"])].append(record)
    summaries = {role: _summarize_records(grouped[role]) for role in roles}
    summaries["all_samples"] = _summarize_records(records)
    leakage = _duplicate_leakage(records)
    counts = Counter(assignments.values())

    return {
        "audit_id": AUDIT_ID,
        "audit_schema_version": "heat3d_v5_p0_1_physics_scale_audit_v1",
        "mode": "read_only",
        "contract_id": contract.get("contract_id") if contract else None,
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "path": _relative_or_text(dataset),
            "split_map_path": _relative_or_text(split_map),
            "sample_count": len(records),
            "roles": list(roles),
            "role_counts": {role: int(counts[role]) for role in roles},
        },
        "read_only_guardrails": {
            "sample_array_writes": 0,
            "sample_metadata_writes": 0,
            "solver_calls": 0,
            "training_runs": 0,
            "permitted_writes": ["explicit audit JSON", "explicit audit Markdown"],
        },
        "integrity": {
            "all_split_map_samples_present": True,
            "all_dataset_samples_assigned_once": True,
            "declared_role_counts_match": True,
            "control_volume_and_power_metadata_validated_for_all_samples": True,
        },
        "split_summaries": summaries,
        "duplicate_leakage": leakage,
        "audit_pass": bool(leakage["pass"]),
        "interpretation_limit": "All q/BC linear relations are descriptive statistics across heterogeneous P5 scenes, not causal response laws or model metrics.",
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0 or (1.0e-3 <= abs(number) < 1.0e4):
        return f"{number:.{digits}f}"
    return f"{number:.{digits}e}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    dataset = _nested_mapping(payload["dataset"], "dataset")
    summaries = _nested_mapping(payload["split_summaries"], "split_summaries")
    roles = [str(role) for role in dataset["roles"]]
    leakage = _nested_mapping(payload["duplicate_leakage"], "duplicate_leakage")
    lines = [
        "# V5-P0-1 P5 Physics-Scale Read-Only Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{dataset['dataset_id']}` (`{dataset['path']}`).",
        f"- Split map: `{dataset['split_map_path']}`.",
        f"- Samples audited: `{dataset['sample_count']}` across `{', '.join(roles)}`.",
        "- The audit read arrays and metadata only; it did not write samples, call a solver, modify a model, or train/evaluate.",
        "",
        "DeltaT is `temperature.npy - sample_meta.boundary_params.bottom.T_fixed_K`. CV RMS and CV mean use rectilinear control-volume weights inferred from `coords.npy`; target max is the nodewise DeltaT maximum.",
        "",
        "## Per-Role Target And Power Summary",
        "",
        "| role | n | CV volume mean m3 | effective power mean W | target CV-RMS mean K | target CV-mean mean K | target max max K |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in [*roles, "all_samples"]:
        summary = _nested_mapping(summaries[role], f"split_summaries.{role}")
        cv = _nested_mapping(summary["control_volume_weights"], "control_volume_weights")
        power = _nested_mapping(summary["effective_source_power"], "effective_source_power")
        target = _nested_mapping(summary["target_statistics"], "target_statistics")
        lines.append(
            "| {role} | {n} | {volume} | {power} | {rms} | {mean} | {maximum} |".format(
                role=role,
                n=summary["sample_count"],
                volume=_format(_nested_mapping(cv["cv_weight_sum_m3"], "volume")["mean"]),
                power=_format(_nested_mapping(power["effective_source_power_W"], "power")["mean"]),
                rms=_format(_nested_mapping(target["target_deltaT_cv_rms_K"], "rms")["mean"]),
                mean=_format(_nested_mapping(target["target_deltaT_cv_mean_K"], "mean")["mean"]),
                maximum=_format(_nested_mapping(target["target_deltaT_max_K"], "maximum")["max"]),
            )
        )

    lines.extend(
        [
            "",
            "## Physics-Scale Proxies",
            "",
            "`R_top = 1 / (top_h * top area)` and `R_z = depth / (top area * CV-harmonic kz)`. `R_series = R_top + R_z`; its DeltaT proxy is effective source power times `R_series`.",
            "",
            "| role | harmonic kz median W/m/K | top_h median W/m2/K | R_series median K/W | target mean / series proxy median |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role in [*roles, "all_samples"]:
        proxy = _nested_mapping(summaries[role]["physics_scale_proxies"], "physics_scale_proxies")
        lines.append(
            "| {role} | {kz} | {h} | {resistance} | {ratio} |".format(
                role=role,
                kz=_format(_nested_mapping(proxy["harmonic_kz_W_mK"], "kz")["median"]),
                h=_format(_nested_mapping(proxy["top_h_W_m2K"], "top_h")["median"]),
                resistance=_format(
                    _nested_mapping(proxy["series_thermal_resistance_proxy_K_W"], "series resistance")["median"]
                ),
                ratio=_format(
                    _nested_mapping(proxy["target_cv_mean_to_series_proxy_ratio"], "target ratio")["median"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## q/BC Linear Relations",
            "",
            "The table reports descriptive Pearson r or R2. The combined predictor is the two-column least-squares fit using effective source power and the top-Robin DeltaT proxy; it is not a causal law because conductivity and source geometry also vary.",
            "",
            "| role | power vs top_h r | target mean vs power R2 | target mean vs top-Robin proxy R2 | combined q/BC R2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role in [*roles, "all_samples"]:
        relations = _nested_mapping(summaries[role]["q_bc_linear_relations"], "q_bc_linear_relations")
        lines.append(
            "| {role} | {power_h} | {target_power} | {target_top} | {combined} |".format(
                role=role,
                power_h=_format(_nested_mapping(relations["effective_source_power_vs_top_h"], "power_vs_h")["pearson_r"]),
                target_power=_format(
                    _nested_mapping(relations["target_cv_mean_vs_effective_source_power"], "target_vs_power")["r_squared"]
                ),
                target_top=_format(
                    _nested_mapping(relations["target_cv_mean_vs_top_robin_deltaT_proxy"], "target_vs_top")["r_squared"]
                ),
                combined=_format(
                    _nested_mapping(relations["target_cv_mean_vs_q_bc_predictors"], "combined")["r_squared"]
                ),
            )
        )

    input_groups = _nested_mapping(
        leakage["cross_role_model_input_duplicate_groups"], "input duplicate groups"
    )
    full_groups = _nested_mapping(
        leakage["cross_role_full_sample_duplicate_groups"], "full duplicate groups"
    )
    provenance_groups = _nested_mapping(
        leakage["cross_role_provenance_duplicate_groups"], "provenance duplicate groups"
    )
    lines.extend(
        [
            "",
            "## Split Duplicate Leakage",
            "",
            f"- Cross-role model-input duplicate groups: `{input_groups['group_count']}`.",
            f"- Cross-role full-sample duplicate groups: `{full_groups['group_count']}`.",
            f"- Cross-role P5 provenance duplicate groups: `{provenance_groups['group_count']}`.",
            f"- Audit pass: `{payload['audit_pass']}`.",
            "- Shared fixed-grid coordinates alone are expected and are not considered leakage; fingerprints include q/k/BC, and full fingerprints also include temperature.",
            "",
            "## Interpretation Limit",
            "",
            str(payload["interpretation_limit"]),
            "",
        ]
    )
    return "\n".join(lines)


def _ensure_report_paths(
    dataset: Path,
    output_json: Path | None,
    output_md: Path | None,
    overwrite: bool,
) -> tuple[Path, Path]:
    if output_json is None or output_md is None:
        raise AuditError("normal audit requires both --output-json and --output-md")
    json_path = output_json.resolve()
    md_path = output_md.resolve()
    dataset_path = dataset.resolve()
    if json_path == md_path:
        raise AuditError("JSON and Markdown output paths must differ")
    for path in (json_path, md_path):
        try:
            path.relative_to(dataset_path)
        except ValueError:
            pass
        else:
            raise AuditError(f"report path must not be inside dataset: {path}")
        if path.exists() and not overwrite:
            raise AuditError(f"report already exists; use --overwrite: {path}")
    return json_path, md_path


def _write_reports(payload: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assignments, roles, split_payload = _load_split_contract(args.dataset, args.split_map)
        contract = _validate_optional_contract(args.contract, split_payload, roles)
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_payload(
                        args.dataset,
                        args.split_map,
                        assignments,
                        roles,
                        split_payload,
                        contract,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        json_path, md_path = _ensure_report_paths(
            args.dataset,
            args.output_json,
            args.output_md,
            args.overwrite,
        )
        payload = run_audit(
            args.dataset,
            args.split_map,
            contract,
            assignments,
            roles,
            split_payload,
        )
        _write_reports(payload, json_path, md_path)
    except AuditError as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
