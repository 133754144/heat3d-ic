#!/usr/bin/env python3
"""Read-only V5 Gate 3 boundary-general shape/scale oracle diagnostics.

The audit consumes frozen P5 arrays, metadata, Gate 1 scale records, and
already-exported raw-temperature V4 checkpoints.  It never trains, changes a
model, calls the reference solver, or writes dataset/checkpoint/log files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_ID = "V5-Gate-3-boundary-general-shape-scale-oracle"
SCHEMA_VERSION = "heat3d_v5_gate3_boundary_general_v1"
GATE1_FINAL_TABLE_SHA256 = "79b7f79c32ac5c3da100e27ebafeeea25cb185088687785c6140f0359bde7de9"
ROLE_ORDER = (
    "train",
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
CLEAN_ROLES = ("train", "valid_iid", "test_iid")
HARD_ROLES = ("hard_train_holdout", "hard_challenge_valid", "hard_challenge_test")
CHECKPOINTS = ("best", "final")
ORACLE_VARIANTS = (
    "original",
    "predicted_shape_true_scale",
    "true_shape_predicted_scale",
    "boundary_projected_original",
)
TARGET_ARRAYS = ("coords", "k_field", "q_field", "bc_features", "temperature")
EPS_K = 1.0e-12
DIRICHLET_TOL_K = 1.0e-8
RECONSTRUCTION_TOL_K = 1.0e-9
SHAPE_RMS_TOL = 1.0e-9
LOW_K_QUANTILE = 0.20
HOTSPOT_QUANTILE = 0.95


class AuditError(RuntimeError):
    """Raised when an immutable Gate 3 contract is violated."""


@dataclass(frozen=True)
class BoundaryRegion:
    """One explicitly described boundary region, independent of location."""

    region_id: str
    boundary_type: str
    mask: np.ndarray
    mask_source: str
    prescribed_value_K: float | None
    metadata_name: str | None
    bc_feature_name: str | None
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class BoundaryContract:
    """Resolved region-wise BC interface for one sample."""

    regions: tuple[BoundaryRegion, ...]
    dirichlet_mask: np.ndarray
    prescribed_temperature_K: np.ndarray
    reference_region_id: str
    reference_temperature_K: float
    coordinate_fallback_used: bool


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{name} must be an object")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{name} must be finite, got {value!r}")
    return result


def _role_key(role: str) -> tuple[int, str]:
    try:
        return (ROLE_ORDER.index(role), role)
    except ValueError:
        return (len(ROLE_ORDER), role)


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
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


def _sample_dirs(dataset: Path) -> set[str]:
    if not dataset.is_dir():
        raise AuditError(f"dataset directory does not exist: {dataset}")
    return {path.name for path in dataset.iterdir() if path.is_dir() and path.name.startswith("sample_")}


def _load_inputs(
    dataset: Path,
    split_map: Path,
    contract_path: Path | None,
) -> tuple[dict[str, str], list[str], dict[str, Any], dict[str, Any] | None]:
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
    if missing or extra:
        raise AuditError(f"split/dataset mismatch missing={missing[:5]} extra={extra[:5]}")
    counts = Counter(assignments.values())
    declared = split_payload.get("actual_counts")
    if declared is not None:
        expected = {str(key): int(value) for key, value in _mapping(declared, "actual_counts").items()}
        if expected != dict(counts):
            raise AuditError("split actual_counts disagrees with sample_splits")
    roles = sorted(counts, key=_role_key)

    contract: dict[str, Any] | None = None
    if contract_path is not None:
        contract = _read_json(contract_path)
        dataset_contract = _mapping(contract.get("dataset_contract"), "dataset_contract")
        if dataset_contract.get("dataset_id") != split_payload.get("dataset_id"):
            raise AuditError("contract dataset_id disagrees with split map")
        expected_count = int(dataset_contract.get("total_sample_count", -1))
        if expected_count != len(assignments):
            raise AuditError("contract total_sample_count disagrees with split map")
        expected_roles = {str(key): int(value) for key, value in _mapping(dataset_contract.get("role_counts"), "role_counts").items()}
        if expected_roles != dict(counts):
            raise AuditError("contract role_counts disagrees with split map")
    return assignments, roles, split_payload, contract


def _control_widths(axis: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise AuditError(f"{name} axis needs at least two coordinates")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise AuditError(f"{name} axis must be finite and strictly increasing")
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
        raise AuditError(f"coords are not a full rectilinear grid: {shape}")
    flattened = np.ravel_multi_index(tuple(inverse), tuple(shape))
    if np.unique(flattened).size != coords.shape[0]:
        raise AuditError("duplicate coordinates are unsupported")
    volumes = widths[0][inverse[0]] * widths[1][inverse[1]] * widths[2][inverse[2]]
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise AuditError("invalid control-volume weights")
    return volumes, axes, inverse, shape


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise AuditError("non-positive metric weight total")
    return float(math.sqrt(np.dot(weights, values * values) / total))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.asarray(weights, dtype=np.float64).sum())
    if total <= 0.0:
        raise AuditError("non-positive metric weight total")
    return float(np.dot(np.asarray(values, dtype=np.float64), weights) / total)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise AuditError("weighted quantile inputs are invalid")
    if not 0.0 <= quantile <= 1.0 or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise AuditError("weighted quantile parameters are invalid")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


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
    if x.size < 2 or y.size != x.size:
        return None
    rank_x = _rank_average(x)
    rank_y = _rank_average(y)
    dx = rank_x - rank_x.mean()
    dy = rank_y - rank_y.mean()
    denom = math.sqrt(float(np.dot(dx, dx) * np.dot(dy, dy)))
    return None if denom <= 0.0 else float(np.dot(dx, dy) / denom)


def _fingerprint(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrays(sample_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in TARGET_ARRAYS:
        path = sample_dir / f"{name}.npy"
        if not path.is_file():
            raise AuditError(f"missing required array {path}")
        try:
            arrays[name] = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise AuditError(f"cannot load {path}: {exc}") from exc
    return arrays


def _normalize_region_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _name_variants(value: str) -> set[str]:
    normalized = _normalize_region_name(value)
    variants = {normalized}
    if normalized.endswith("s") and len(normalized) > 1:
        variants.add(normalized[:-1])
    elif normalized:
        variants.add(normalized + "s")
    return variants


def _value_from_keys(params: Mapping[str, Any], keys: Sequence[str], label: str) -> float:
    for key in keys:
        if key in params:
            return _finite_float(params[key], label)
    raise AuditError(f"{label} missing one of {list(keys)}")


def _coordinate_fallback_mask(
    *,
    params: Mapping[str, Any],
    coords: np.ndarray,
    region_id: str,
) -> np.ndarray:
    fallback = _mapping(params.get("coordinate_fallback"), f"boundary_params.{region_id}.coordinate_fallback")
    axis_name = str(fallback.get("axis", "")).lower()
    if axis_name not in {"x", "y", "z"}:
        raise AuditError(f"{region_id}: coordinate fallback axis must be x/y/z")
    extreme = str(fallback.get("extremum", "")).lower()
    if extreme not in {"min", "max"}:
        raise AuditError(f"{region_id}: coordinate fallback extremum must be min/max")
    dimension = {"x": 0, "y": 1, "z": 2}[axis_name]
    values = np.asarray(coords[:, dimension], dtype=np.float64)
    target = float(values.min() if extreme == "min" else values.max())
    tolerance = _finite_float(fallback.get("tolerance", 1.0e-12), f"{region_id} fallback tolerance")
    mask = np.isclose(values, target, rtol=0.0, atol=tolerance)
    if not np.any(mask):
        raise AuditError(f"{region_id}: explicit coordinate fallback selected no nodes")
    return mask


def _resolve_boundary_contract(
    *,
    meta: Mapping[str, Any],
    bc_features: np.ndarray,
    coords: np.ndarray,
    reference_region_id: str,
    allow_coordinate_fallback: bool,
) -> BoundaryContract:
    """Read region type/masks/values without using a fixed geometric boundary.

    Coordinate inference is disabled by default.  When explicitly enabled, a
    region must declare its own axis/extremum fallback in metadata; there is no
    implicit "lowest z is Dirichlet" rule anywhere in this interface.
    """

    params_by_region = _mapping(meta.get("boundary_params"), "sample_meta.boundary_params")
    raw_regions = meta.get("boundary_regions")
    if not isinstance(raw_regions, list):
        raise AuditError("sample_meta.boundary_regions must be a list")
    feature_names = meta.get("bc_feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise AuditError("sample_meta.bc_feature_names must be a string list")
    bc = np.asarray(bc_features, dtype=np.float64)
    node_count = coords.shape[0]
    if bc.shape != (node_count, len(feature_names)):
        raise AuditError("bc_features shape does not agree with bc_feature_names")

    metadata_by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in raw_regions:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise AuditError("each boundary_regions entry needs a string name")
        for variant in _name_variants(str(item["name"])):
            metadata_by_variant[variant].append(item)
    feature_index = {str(name): index for index, name in enumerate(feature_names)}

    regions: list[BoundaryRegion] = []
    fallback_used = False
    for raw_region_id, raw_params in params_by_region.items():
        region_id = str(raw_region_id)
        params = _mapping(raw_params, f"boundary_params.{region_id}")
        boundary_type = str(params.get("boundary_type", params.get("type", ""))).lower()
        if boundary_type == "mixed":
            raise AuditError(
                f"{region_id}: composite type=mixed must be expanded into explicit per-region "
                "dirichlet/robin/neumann/adiabatic entries before Gate 3"
            )
        if boundary_type not in {"dirichlet", "robin", "neumann", "adiabatic"}:
            raise AuditError(f"{region_id}: unsupported or unknown boundary type {boundary_type!r}")

        metadata_candidates: list[Mapping[str, Any]] = []
        for variant in _name_variants(region_id):
            metadata_candidates.extend(metadata_by_variant.get(variant, []))
        unique_metadata = {id(item): item for item in metadata_candidates}
        if len(unique_metadata) > 1:
            raise AuditError(f"{region_id}: boundary metadata name is ambiguous")
        metadata_region = next(iter(unique_metadata.values()), None)
        metadata_mask: np.ndarray | None = None
        metadata_name: str | None = None
        if metadata_region is not None:
            indices = metadata_region.get("point_indices")
            if not isinstance(indices, list):
                raise AuditError(f"{region_id}: boundary metadata has no point_indices list")
            index_array = np.asarray(indices, dtype=np.int64)
            if index_array.ndim != 1 or index_array.size == 0 or np.any(index_array < 0) or np.any(index_array >= node_count):
                raise AuditError(f"{region_id}: boundary metadata point indices are invalid")
            if np.unique(index_array).size != index_array.size:
                raise AuditError(f"{region_id}: boundary metadata point indices are duplicated")
            metadata_mask = np.zeros(node_count, dtype=bool)
            metadata_mask[index_array] = True
            metadata_name = str(metadata_region["name"])

        explicit_feature = params.get("mask_feature_name", params.get("bc_feature_name"))
        if explicit_feature is not None:
            matched_features = [str(explicit_feature)] if str(explicit_feature) in feature_index else []
        else:
            variants = _name_variants(region_id)
            if metadata_name is not None:
                variants.update(_name_variants(metadata_name))
            matched_features = [
                name
                for name in feature_index
                if name.startswith("is_") and _normalize_region_name(name[3:]) in variants
            ]
        if len(matched_features) > 1:
            raise AuditError(f"{region_id}: BC feature match is ambiguous: {matched_features}")
        feature_mask: np.ndarray | None = None
        feature_name: str | None = None
        if matched_features:
            feature_name = matched_features[0]
            values = bc[:, feature_index[feature_name]]
            if not np.all(np.isfinite(values)) or not np.all(np.isclose(values, 0.0) | np.isclose(values, 1.0)):
                raise AuditError(f"{region_id}: BC feature {feature_name} must be finite binary")
            feature_mask = np.isclose(values, 1.0)

        if metadata_mask is not None and feature_mask is not None:
            if not np.array_equal(metadata_mask, feature_mask):
                raise AuditError(f"{region_id}: metadata and BC feature masks disagree")
            mask = metadata_mask
            mask_source = "metadata_and_bc_feature"
        elif metadata_mask is not None or feature_mask is not None:
            raise AuditError(f"{region_id}: Gate 3 requires both metadata and BC feature masks")
        elif allow_coordinate_fallback:
            mask = _coordinate_fallback_mask(params=params, coords=coords, region_id=region_id)
            mask_source = "explicit_coordinate_fallback"
            fallback_used = True
        else:
            raise AuditError(
                f"{region_id}: no matching metadata/BC mask; coordinate inference is disabled unless "
                "--allow-coordinate-fallback is explicitly supplied with coordinate_fallback metadata"
            )

        prescribed: float | None = None
        if boundary_type == "dirichlet":
            prescribed = _value_from_keys(
                params,
                ("fixed_temperature_K", "T_fixed_K", "prescribed_value_K", "value_K"),
                f"{region_id} prescribed Dirichlet temperature",
            )
        elif boundary_type == "robin":
            h = _value_from_keys(params, ("h_W_m2K",), f"{region_id} Robin h")
            if h <= 0.0:
                raise AuditError(f"{region_id}: Robin h must be positive")
            _value_from_keys(params, ("ambient_temperature_K", "T_inf_K"), f"{region_id} Robin ambient temperature")
        elif boundary_type == "neumann":
            _value_from_keys(params, ("flux_W_m2", "heat_flux_W_m2", "q_n_W_m2"), f"{region_id} Neumann flux")

        regions.append(
            BoundaryRegion(
                region_id=region_id,
                boundary_type=boundary_type,
                mask=mask,
                mask_source=mask_source,
                prescribed_value_K=prescribed,
                metadata_name=metadata_name,
                bc_feature_name=feature_name,
                parameters=params,
            )
        )

    if not regions:
        raise AuditError("boundary_params is empty")
    for first_index, first in enumerate(regions):
        for second in regions[first_index + 1 :]:
            if np.any(first.mask & second.mask) and not (
                first.boundary_type == "dirichlet" and second.boundary_type == "dirichlet"
            ):
                raise AuditError(
                    f"overlapping boundary regions {first.region_id!r}/{second.region_id!r} need an "
                    "explicit disjoint mixed-BC expansion before Gate 3"
                )
    dirichlet_regions = [region for region in regions if region.boundary_type == "dirichlet"]
    if not dirichlet_regions:
        raise AuditError("no Dirichlet region is available for a shape/scale reference")
    lookup = {region.region_id: region for region in regions}
    reference = lookup.get(reference_region_id)
    if reference is None or reference.boundary_type != "dirichlet" or reference.prescribed_value_K is None:
        raise AuditError(f"reference Dirichlet region {reference_region_id!r} is absent or not Dirichlet")

    dirichlet_mask = np.zeros(node_count, dtype=bool)
    prescribed_values = np.full(node_count, np.nan, dtype=np.float64)
    for region in dirichlet_regions:
        assert region.prescribed_value_K is not None
        overlap = dirichlet_mask & region.mask
        if np.any(overlap) and not np.allclose(
            prescribed_values[overlap], region.prescribed_value_K, rtol=0.0, atol=DIRICHLET_TOL_K
        ):
            raise AuditError("overlapping Dirichlet regions prescribe incompatible temperatures")
        dirichlet_mask |= region.mask
        prescribed_values[region.mask] = region.prescribed_value_K
    if not np.any(dirichlet_mask):
        raise AuditError("Dirichlet mask is empty")
    return BoundaryContract(
        regions=tuple(regions),
        dirichlet_mask=dirichlet_mask,
        prescribed_temperature_K=prescribed_values,
        reference_region_id=reference.region_id,
        reference_temperature_K=float(reference.prescribed_value_K),
        coordinate_fallback_used=fallback_used,
    )


def _boundary_signature(boundary: BoundaryContract) -> str:
    return ";".join(
        f"{region.region_id}:{region.boundary_type}:{int(region.mask.sum())}" for region in boundary.regions
    )


def _boundary_project_raw(raw_temperature_K: np.ndarray, boundary: BoundaryContract) -> np.ndarray:
    raw = np.asarray(raw_temperature_K, dtype=np.float64).reshape(-1)
    if raw.shape != boundary.dirichlet_mask.shape:
        raise AuditError("boundary projection node count mismatch")
    projected = raw.copy()
    projected[boundary.dirichlet_mask] = boundary.prescribed_temperature_K[boundary.dirichlet_mask]
    return projected


def _shape_scale_decompose(raw_temperature_K: np.ndarray, weights: np.ndarray, reference_K: float) -> tuple[np.ndarray, float, np.ndarray]:
    raw = np.asarray(raw_temperature_K, dtype=np.float64).reshape(-1)
    delta = raw - float(reference_K)
    scale = _weighted_rms(delta, weights)
    shape = delta / (scale + EPS_K)
    return delta, scale, shape


def _shape_scale_reconstruct(shape: np.ndarray, scale_K: float, reference_K: float) -> np.ndarray:
    return np.asarray(shape, dtype=np.float64) * (float(scale_K) + EPS_K) + float(reference_K)


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    centered_x = x - _weighted_mean(x, weights)
    centered_y = y - _weighted_mean(y, weights)
    denom = math.sqrt(float(np.dot(weights, centered_x * centered_x) * np.dot(weights, centered_y * centered_y)))
    return None if denom <= 0.0 else float(np.dot(weights, centered_x * centered_y) / denom)


METRIC_COLUMNS = (
    "cv_rmse_K",
    "cv_mae_K",
    "spatial_correlation",
    "amplitude_ratio",
    "cv_rms_ratio",
    "hotspot_cv_weight_fraction",
    "hotspot_cv_rmse_K",
    "hotspot_cv_mae_K",
    "hotspot_mean_bias_K",
    "hotspot_peak_abs_error_K",
    "hotspot_pred_to_true_mean_ratio",
)


def _sample_metrics(
    *,
    predicted_raw_K: np.ndarray,
    target_raw_K: np.ndarray,
    reference_K: float,
    volumes: np.ndarray,
    hotspot_mask: np.ndarray,
) -> dict[str, float | None]:
    predicted_delta = np.asarray(predicted_raw_K, dtype=np.float64).reshape(-1) - reference_K
    target_delta = np.asarray(target_raw_K, dtype=np.float64).reshape(-1) - reference_K
    error = predicted_delta - target_delta
    total = float(volumes.sum())
    target_rms = _weighted_rms(target_delta, volumes)
    predicted_rms = _weighted_rms(predicted_delta, volumes)
    denom = float(np.dot(volumes, target_delta * target_delta))
    amplitude = None if denom <= 0.0 else float(np.dot(volumes, predicted_delta * target_delta) / denom)
    hot_weights = volumes[hotspot_mask]
    hot_error = error[hotspot_mask]
    hot_target = target_delta[hotspot_mask]
    hot_prediction = predicted_delta[hotspot_mask]
    hot_total = float(hot_weights.sum())
    hot_target_mean = _weighted_mean(hot_target, hot_weights)
    hot_prediction_mean = _weighted_mean(hot_prediction, hot_weights)
    return {
        "cv_rmse_K": _weighted_rms(error, volumes),
        "cv_mae_K": float(np.dot(volumes, np.abs(error)) / total),
        "spatial_correlation": _weighted_correlation(predicted_delta, target_delta, volumes),
        "amplitude_ratio": amplitude,
        "cv_rms_ratio": None if target_rms <= 0.0 else predicted_rms / target_rms,
        "hotspot_cv_weight_fraction": hot_total / total,
        "hotspot_cv_rmse_K": _weighted_rms(hot_error, hot_weights),
        "hotspot_cv_mae_K": float(np.dot(hot_weights, np.abs(hot_error)) / hot_total),
        "hotspot_mean_bias_K": _weighted_mean(hot_error, hot_weights),
        "hotspot_peak_abs_error_K": float(np.max(np.abs(hot_error))),
        "hotspot_pred_to_true_mean_ratio": None if abs(hot_target_mean) <= EPS_K else hot_prediction_mean / hot_target_mean,
    }


def _local_lateral_diagnostics(
    *,
    q_field: np.ndarray,
    k_z: np.ndarray,
    volumes: np.ndarray,
    coords: np.ndarray,
    z_inverse: np.ndarray,
    dirichlet_mask: np.ndarray,
) -> dict[str, float | None]:
    q = np.asarray(q_field, dtype=np.float64).reshape(-1).copy()
    q[dirichlet_mask] = 0.0
    q_positive = np.maximum(q, 0.0)
    q_weights = q_positive * volumes
    source_power = float(q_weights.sum())
    total_volume = float(volumes.sum())
    low_threshold = _weighted_quantile(k_z, volumes, LOW_K_QUANTILE)
    result: dict[str, float | None] = {
        "source_power_positive_W": source_power,
        "low_k_q20_threshold_W_mK": low_threshold,
        "q_weighted_local_kz_W_mK": None,
        "q_weighted_inverse_kz_mK_W": None,
        "q_low_k_overlap_fraction": None,
        "source_layer_kz_heterogeneity_cv": None,
        "source_concentration": None,
        "source_z_centroid_m": None,
        "source_z_centroid_normalized": None,
    }
    if source_power <= EPS_K:
        return result
    result["q_weighted_local_kz_W_mK"] = float(np.dot(q_weights, k_z) / source_power)
    result["q_weighted_inverse_kz_mK_W"] = float(np.dot(q_weights, 1.0 / k_z) / source_power)
    result["q_low_k_overlap_fraction"] = float(q_weights[k_z <= low_threshold].sum() / source_power)
    q_rms = _weighted_rms(q_positive, volumes)
    q_mean = source_power / total_volume
    result["source_concentration"] = None if q_mean <= EPS_K else q_rms / q_mean
    z = np.asarray(coords[:, 2], dtype=np.float64)
    z_centroid = float(np.dot(q_weights, z) / source_power)
    z_extent = float(z.max() - z.min())
    result["source_z_centroid_m"] = z_centroid
    result["source_z_centroid_normalized"] = None if z_extent <= 0.0 else (z_centroid - float(z.min())) / z_extent

    weighted_cv_sum = 0.0
    layer_power_sum = 0.0
    for layer in np.unique(z_inverse):
        mask = z_inverse == layer
        layer_power = float(q_weights[mask].sum())
        if layer_power <= EPS_K:
            continue
        layer_k = k_z[mask]
        layer_volume = volumes[mask]
        mean_k = _weighted_mean(layer_k, layer_volume)
        std_k = math.sqrt(_weighted_mean((layer_k - mean_k) ** 2, layer_volume))
        weighted_cv_sum += layer_power * std_k / mean_k
        layer_power_sum += layer_power
    result["source_layer_kz_heterogeneity_cv"] = (
        None if layer_power_sum <= EPS_K else weighted_cv_sum / layer_power_sum
    )
    return result


def _load_gate1_table(path: Path, expected_ids: set[str]) -> tuple[dict[str, dict[str, float]], str]:
    if not path.is_file():
        raise AuditError(f"Gate 1 final table does not exist: {path}")
    required = {
        "sample_id",
        "s_y_cv_rms_deltaT_K",
        "pred_z_collapsed_1d_operator_K",
        "log_residual_z_collapsed_1d_operator",
    }
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError("Gate 1 table is missing required corrected-operator columns")
        for raw in reader:
            sample_id = str(raw["sample_id"])
            if sample_id in rows:
                raise AuditError(f"Gate 1 table duplicates {sample_id}")
            try:
                rows[sample_id] = {
                    "scale_K": float(raw["s_y_cv_rms_deltaT_K"]),
                    "prediction_K": float(raw["pred_z_collapsed_1d_operator_K"]),
                    "log_residual": float(raw["log_residual_z_collapsed_1d_operator"]),
                }
            except (TypeError, ValueError) as exc:
                raise AuditError(f"Gate 1 table has invalid corrected-operator values for {sample_id}") from exc
    if set(rows) != expected_ids:
        raise AuditError("Gate 1 table sample IDs do not match the frozen P5 split")
    return rows, _sha256(path)


def _load_prediction_archives(
    paths: Sequence[Path],
    expected_ids: set[str],
    checkpoint: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not paths:
        raise AuditError(f"{checkpoint}: no frozen prediction archives were supplied")
    merged: dict[str, np.ndarray] = {}
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise AuditError(f"{checkpoint}: prediction archive does not exist: {path}")
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AuditError(f"{checkpoint}: cannot read prediction archive {path}: {exc}") from exc
        sample_ids = list(archive.files)
        if not sample_ids:
            raise AuditError(f"{checkpoint}: prediction archive is empty: {path}")
        for sample_id in sample_ids:
            if sample_id not in expected_ids:
                raise AuditError(f"{checkpoint}: prediction archive has unexpected sample {sample_id}")
            if sample_id in merged:
                raise AuditError(f"{checkpoint}: duplicate prediction for {sample_id} across archives")
            value = np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise AuditError(f"{checkpoint}: invalid prediction for {sample_id}")
            merged[sample_id] = value
        artifacts.append({"path": path.as_posix(), "sha256": _sha256(path), "sample_count": len(sample_ids)})
    if set(merged) != expected_ids:
        missing = sorted(expected_ids - set(merged))
        raise AuditError(f"{checkpoint}: frozen predictions do not cover all P5 samples; missing={missing[:8]}")
    return merged, artifacts


def _prediction_metric_columns() -> list[str]:
    columns: list[str] = []
    for checkpoint in CHECKPOINTS:
        columns.extend(
            (
                f"{checkpoint}_prediction_available",
                f"{checkpoint}_prediction_shape_reconstruction_max_abs_error_K",
                f"{checkpoint}_predicted_scale_K",
                f"{checkpoint}_predicted_shape_cv_rms",
                f"{checkpoint}_predicted_scale_ratio",
                f"{checkpoint}_predicted_scale_log_ratio",
                f"{checkpoint}_original_dirichlet_max_abs_error_K",
                f"{checkpoint}_boundary_projection_dirichlet_max_abs_error_K",
                f"{checkpoint}_boundary_projection_non_dirichlet_max_abs_change_K",
            )
        )
        for variant in ORACLE_VARIANTS:
            columns.extend(f"{checkpoint}_{variant}_{metric}" for metric in METRIC_COLUMNS)
    return columns


LATERAL_COLUMNS = (
    "source_power_positive_W",
    "low_k_q20_threshold_W_mK",
    "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W",
    "q_low_k_overlap_fraction",
    "source_layer_kz_heterogeneity_cv",
    "source_concentration",
    "source_z_centroid_m",
    "source_z_centroid_normalized",
)


def _table_columns() -> list[str]:
    return [
        "sample_id",
        "role",
        "is_clean_role",
        "is_hard_role",
        "input_fingerprint",
        "full_fingerprint",
        "provenance_source_id",
        "grid_shape",
        "boundary_signature",
        "reference_dirichlet_region",
        "dirichlet_mask_source",
        "coordinate_fallback_used",
        "dirichlet_region_count",
        "dirichlet_node_count",
        "reference_temperature_K",
        "cv_weight_sum_m3",
        "target_is_nonzero",
        "target_scale_cv_rms_K",
        "target_cv_mean_deltaT_K",
        "target_max_deltaT_K",
        "target_shape_cv_rms",
        "target_shape_cv_rms_abs_error",
        "target_raw_reconstruction_max_abs_error_K",
        "target_delta_reconstruction_max_abs_error_K",
        "target_projected_dirichlet_max_abs_error_K",
        "target_projection_non_dirichlet_max_abs_change_K",
        "target_dirichlet_label_max_abs_error_K",
        "target_decomposition_pass",
        "gate1_operator_scale_K",
        "gate1_operator_prediction_K",
        "gate1_operator_scale_log_residual",
        *LATERAL_COLUMNS,
        *_prediction_metric_columns(),
    ]


TABLE_COLUMNS = _table_columns()
STRING_COLUMNS = {
    "sample_id",
    "role",
    "input_fingerprint",
    "full_fingerprint",
    "provenance_source_id",
    "grid_shape",
    "boundary_signature",
    "reference_dirichlet_region",
    "dirichlet_mask_source",
}
INT_COLUMNS = {
    "is_clean_role",
    "is_hard_role",
    "coordinate_fallback_used",
    "dirichlet_region_count",
    "dirichlet_node_count",
    "target_is_nonzero",
    "target_decomposition_pass",
    *(f"{checkpoint}_prediction_available" for checkpoint in CHECKPOINTS),
}


def _sample_row(
    *,
    sample_dir: Path,
    role: str,
    gate1: Mapping[str, float],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    reference_region_id: str,
    allow_coordinate_fallback: bool,
) -> dict[str, Any]:
    meta = _read_json(sample_dir / "sample_meta.json")
    arrays = _load_arrays(sample_dir)
    coords = np.asarray(arrays["coords"], dtype=np.float64)
    q = np.asarray(arrays["q_field"], dtype=np.float64).reshape(-1)
    target_raw = np.asarray(arrays["temperature"], dtype=np.float64).reshape(-1)
    k_field = np.asarray(arrays["k_field"], dtype=np.float64)
    bc_features = np.asarray(arrays["bc_features"], dtype=np.float64)
    node_count = coords.shape[0]
    if q.shape != (node_count,) or target_raw.shape != (node_count,):
        raise AuditError(f"{sample_dir}: q/temperature node count mismatch")
    if k_field.shape == (node_count, 1):
        k_diag = np.repeat(k_field, 3, axis=1)
    elif k_field.shape == (node_count, 3):
        k_diag = k_field
    else:
        raise AuditError(f"{sample_dir}: k_field must be [N,1] or [N,3]")
    if not all(np.all(np.isfinite(value)) for value in (coords, q, target_raw, k_diag, bc_features)):
        raise AuditError(f"{sample_dir}: non-finite arrays")
    if np.any(k_diag <= 0.0):
        raise AuditError(f"{sample_dir}: non-positive conductivity")
    volumes, _axes, inverse, shape = _control_volumes(coords)
    boundary = _resolve_boundary_contract(
        meta=meta,
        bc_features=bc_features,
        coords=coords,
        reference_region_id=reference_region_id,
        allow_coordinate_fallback=allow_coordinate_fallback,
    )
    target_delta, target_scale, target_shape = _shape_scale_decompose(
        target_raw, volumes, boundary.reference_temperature_K
    )
    target_raw_reconstruction = _shape_scale_reconstruct(target_shape, target_scale, boundary.reference_temperature_K)
    target_delta_reconstruction = target_raw_reconstruction - boundary.reference_temperature_K
    target_projected = _boundary_project_raw(target_raw_reconstruction, boundary)
    target_dirichlet_error = float(
        np.max(np.abs(target_raw[boundary.dirichlet_mask] - boundary.prescribed_temperature_K[boundary.dirichlet_mask]))
    )
    target_projected_dirichlet_error = float(
        np.max(
            np.abs(
                target_projected[boundary.dirichlet_mask]
                - boundary.prescribed_temperature_K[boundary.dirichlet_mask]
            )
        )
    )
    non_dirichlet = ~boundary.dirichlet_mask
    target_projection_non_dirichlet_change = float(
        np.max(np.abs(target_projected[non_dirichlet] - target_raw_reconstruction[non_dirichlet]))
    ) if np.any(non_dirichlet) else 0.0
    target_shape_rms = _weighted_rms(target_shape, volumes)
    target_is_nonzero = target_scale > EPS_K
    target_shape_error = abs(target_shape_rms - 1.0) if target_is_nonzero else abs(target_shape_rms)
    target_raw_recon_error = float(np.max(np.abs(target_raw_reconstruction - target_raw)))
    target_delta_recon_error = float(np.max(np.abs(target_delta_reconstruction - target_delta)))
    target_pass = (
        target_dirichlet_error <= DIRICHLET_TOL_K
        and target_projected_dirichlet_error <= DIRICHLET_TOL_K
        and target_projection_non_dirichlet_change <= RECONSTRUCTION_TOL_K
        and target_raw_recon_error <= RECONSTRUCTION_TOL_K
        and target_delta_recon_error <= RECONSTRUCTION_TOL_K
        and target_shape_error <= SHAPE_RMS_TOL
    )
    gate1_scale = float(gate1["scale_K"])
    if not math.isclose(gate1_scale, target_scale, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise AuditError(f"{sample_dir}: Gate 1 target scale differs from Gate 3 decomposition")
    provenance = meta.get("p5_provenance")
    provenance_id = ""
    if isinstance(provenance, Mapping) and provenance.get("source_sample_id") is not None:
        provenance_id = str(provenance["source_sample_id"])
    lateral = _local_lateral_diagnostics(
        q_field=q,
        k_z=k_diag[:, 2],
        volumes=volumes,
        coords=coords,
        z_inverse=inverse[2],
        dirichlet_mask=boundary.dirichlet_mask,
    )
    dirichlet_sources = sorted({region.mask_source for region in boundary.regions if region.boundary_type == "dirichlet"})
    row: dict[str, Any] = {
        "sample_id": sample_dir.name,
        "role": role,
        "is_clean_role": int(role in CLEAN_ROLES),
        "is_hard_role": int(role in HARD_ROLES),
        "input_fingerprint": _fingerprint(arrays, TARGET_ARRAYS[:-1]),
        "full_fingerprint": _fingerprint(arrays, TARGET_ARRAYS),
        "provenance_source_id": provenance_id,
        "grid_shape": "x".join(map(str, shape)),
        "boundary_signature": _boundary_signature(boundary),
        "reference_dirichlet_region": boundary.reference_region_id,
        "dirichlet_mask_source": "+".join(dirichlet_sources),
        "coordinate_fallback_used": int(boundary.coordinate_fallback_used),
        "dirichlet_region_count": len([r for r in boundary.regions if r.boundary_type == "dirichlet"]),
        "dirichlet_node_count": int(boundary.dirichlet_mask.sum()),
        "reference_temperature_K": boundary.reference_temperature_K,
        "cv_weight_sum_m3": float(volumes.sum()),
        "target_is_nonzero": int(target_is_nonzero),
        "target_scale_cv_rms_K": target_scale,
        "target_cv_mean_deltaT_K": _weighted_mean(target_delta, volumes),
        "target_max_deltaT_K": float(target_delta.max()),
        "target_shape_cv_rms": target_shape_rms,
        "target_shape_cv_rms_abs_error": target_shape_error,
        "target_raw_reconstruction_max_abs_error_K": target_raw_recon_error,
        "target_delta_reconstruction_max_abs_error_K": target_delta_recon_error,
        "target_projected_dirichlet_max_abs_error_K": target_projected_dirichlet_error,
        "target_projection_non_dirichlet_max_abs_change_K": target_projection_non_dirichlet_change,
        "target_dirichlet_label_max_abs_error_K": target_dirichlet_error,
        "target_decomposition_pass": int(target_pass),
        "gate1_operator_scale_K": gate1_scale,
        "gate1_operator_prediction_K": float(gate1["prediction_K"]),
        "gate1_operator_scale_log_residual": float(gate1["log_residual"]),
        **lateral,
    }
    hotspot_threshold = _weighted_quantile(target_delta, volumes, HOTSPOT_QUANTILE)
    hotspot_mask = target_delta >= hotspot_threshold
    if not np.any(hotspot_mask):
        raise AuditError(f"{sample_dir}: hotspot mask is empty")
    for checkpoint in CHECKPOINTS:
        prediction_raw = np.asarray(predictions[checkpoint][sample_dir.name], dtype=np.float64).reshape(-1)
        if prediction_raw.shape != target_raw.shape:
            raise AuditError(f"{sample_dir}: {checkpoint} prediction node count mismatch")
        prediction_delta, predicted_scale, predicted_shape = _shape_scale_decompose(
            prediction_raw, volumes, boundary.reference_temperature_K
        )
        prediction_reconstruction = _shape_scale_reconstruct(
            predicted_shape, predicted_scale, boundary.reference_temperature_K
        )
        projected_original = _boundary_project_raw(prediction_raw, boundary)
        predicted_shape_true_scale = _shape_scale_reconstruct(
            predicted_shape, target_scale, boundary.reference_temperature_K
        )
        true_shape_predicted_scale = _shape_scale_reconstruct(
            target_shape, predicted_scale, boundary.reference_temperature_K
        )
        variants = {
            "original": prediction_raw,
            "predicted_shape_true_scale": predicted_shape_true_scale,
            "true_shape_predicted_scale": true_shape_predicted_scale,
            "boundary_projected_original": projected_original,
        }
        projection_dirichlet_error = float(
            np.max(
                np.abs(
                    projected_original[boundary.dirichlet_mask]
                    - boundary.prescribed_temperature_K[boundary.dirichlet_mask]
                )
            )
        )
        projection_non_dirichlet_change = float(
            np.max(np.abs(projected_original[non_dirichlet] - prediction_raw[non_dirichlet]))
        ) if np.any(non_dirichlet) else 0.0
        row.update(
            {
                f"{checkpoint}_prediction_available": 1,
                f"{checkpoint}_prediction_shape_reconstruction_max_abs_error_K": float(
                    np.max(np.abs(prediction_reconstruction - prediction_raw))
                ),
                f"{checkpoint}_predicted_scale_K": predicted_scale,
                f"{checkpoint}_predicted_shape_cv_rms": _weighted_rms(predicted_shape, volumes),
                f"{checkpoint}_predicted_scale_ratio": None if target_scale <= EPS_K else predicted_scale / target_scale,
                f"{checkpoint}_predicted_scale_log_ratio": None if target_scale <= EPS_K or predicted_scale <= EPS_K else math.log(predicted_scale / target_scale),
                f"{checkpoint}_original_dirichlet_max_abs_error_K": float(
                    np.max(
                        np.abs(
                            prediction_raw[boundary.dirichlet_mask]
                            - boundary.prescribed_temperature_K[boundary.dirichlet_mask]
                        )
                    )
                ),
                f"{checkpoint}_boundary_projection_dirichlet_max_abs_error_K": projection_dirichlet_error,
                f"{checkpoint}_boundary_projection_non_dirichlet_max_abs_change_K": projection_non_dirichlet_change,
            }
        )
        for variant, raw in variants.items():
            for metric, value in _sample_metrics(
                predicted_raw_K=raw,
                target_raw_K=target_raw,
                reference_K=boundary.reference_temperature_K,
                volumes=volumes,
                hotspot_mask=hotspot_mask,
            ).items():
                row[f"{checkpoint}_{variant}_{metric}"] = value
    if set(row) != set(TABLE_COLUMNS):
        missing = sorted(set(TABLE_COLUMNS) - set(row))
        extra = sorted(set(row) - set(TABLE_COLUMNS))
        raise AuditError(f"{sample_dir}: internal CSV schema mismatch missing={missing} extra={extra}")
    return row


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["sample_id"])):
            encoded: dict[str, str] = {}
            for column in TABLE_COLUMNS:
                value = row[column]
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
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TABLE_COLUMNS:
            raise AuditError("per-sample table columns do not match the Gate 3 schema")
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


def _duplicate_groups(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field) or "")
        if value:
            groups[value].append(row)
    output: list[dict[str, Any]] = []
    for value, members in groups.items():
        roles = sorted({str(member["role"]) for member in members}, key=_role_key)
        if len(roles) > 1:
            output.append(
                {
                    "key": value,
                    "roles": roles,
                    "samples": [
                        {"sample_id": str(member["sample_id"]), "role": str(member["role"])}
                        for member in sorted(members, key=lambda item: str(item["sample_id"]))
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


def _metric_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    return {metric: _summary(row[f"{prefix}_{metric}"] for row in rows) for metric in METRIC_COLUMNS}


def _oracle_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        result[checkpoint] = {
            variant: _metric_summary(rows, f"{checkpoint}_{variant}") for variant in ORACLE_VARIANTS
        }
        result[checkpoint]["scale_diagnostics"] = {
            "predicted_scale_K": _summary(row[f"{checkpoint}_predicted_scale_K"] for row in rows),
            "predicted_scale_ratio": _summary(row[f"{checkpoint}_predicted_scale_ratio"] for row in rows),
            "predicted_scale_log_ratio": _summary(row[f"{checkpoint}_predicted_scale_log_ratio"] for row in rows),
            "original_dirichlet_max_abs_error_K": _summary(
                row[f"{checkpoint}_original_dirichlet_max_abs_error_K"] for row in rows
            ),
        }
    return result


def _relation(x_values: Sequence[float | None], y_values: Sequence[float | None]) -> dict[str, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return {"sample_count": len(pairs), "Spearman_rho": None, "slope": None, "intercept": None, "R2": None}
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(x) <= 1.0e-15 or np.ptp(y) <= 1.0e-15:
        return {"sample_count": len(pairs), "Spearman_rho": _spearman(x, y), "slope": None, "intercept": None, "R2": None}
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    total = float(np.square(y - y.mean()).sum())
    return {
        "sample_count": len(pairs),
        "Spearman_rho": _spearman(x, y),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": None if total <= 0.0 else float(1.0 - np.square(y - predicted).sum() / total),
    }


def _lateral_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "feature_distributions": {key: _summary(row[key] for row in rows) for key in LATERAL_COLUMNS},
        "relations_to_gate1_operator_scale_log_residual": {
            key: _relation(
                [row[key] for row in rows],
                [row["gate1_operator_scale_log_residual"] for row in rows],
            )
            for key in LATERAL_COLUMNS
            if key not in {"source_power_positive_W", "low_k_q20_threshold_W_mK"}
        },
    }


def _group_rows(rows: Sequence[Mapping[str, Any]], roles: Sequence[str]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {role: [row for row in rows if row["role"] == role] for role in roles}
    groups["clean"] = [row for row in rows if int(row["is_clean_role"]) == 1]
    groups["hard"] = [row for row in rows if int(row["is_hard_role"]) == 1]
    groups["all_samples"] = list(rows)
    return groups


def _hard_failure_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hard_rows = [row for row in rows if int(row["is_hard_role"]) == 1]
    result: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        def mean(variant: str) -> float | None:
            return _summary(row[f"{checkpoint}_{variant}_cv_rmse_K"] for row in hard_rows)["mean"]

        original = mean("original")
        shape_only = mean("predicted_shape_true_scale")
        scale_only = mean("true_shape_predicted_scale")
        projected = mean("boundary_projected_original")
        if shape_only is None or scale_only is None:
            direction = "insufficient_hard_predictions"
        elif scale_only > shape_only:
            direction = "prioritize_scale_path_diagnostics_before_any_Gate_4_model_change"
        elif shape_only > scale_only:
            direction = "prioritize_shape_or_lateral_spreading_diagnostics_before_any_Gate_4_model_change"
        else:
            direction = "scale_and_shape_are_tied_in_this_oracle_diagnostic"
        result[checkpoint] = {
            "sample_count": len(hard_rows),
            "end_to_end_original_cv_rmse_K": original,
            "shape_only_oracle_cv_rmse_K": shape_only,
            "scale_only_oracle_cv_rmse_K": scale_only,
            "boundary_projected_original_cv_rmse_K": projected,
            "interpretation": (
                "oracle components are non-additive counterfactual diagnostics: predicted-shape+true-scale "
                "isolates shape error, while true-shape+predicted-scale isolates scale error"
            ),
            "gate4_direction": direction,
        }
    return result


def _target_decomposition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures = Counter(str(row["boundary_signature"]) for row in rows)
    return {
        "sample_count": len(rows),
        "nonzero_target_count": int(sum(int(row["target_is_nonzero"]) for row in rows)),
        "decomposition_pass_count": int(sum(int(row["target_decomposition_pass"]) for row in rows)),
        "target_scale_cv_rms_K": _summary(row["target_scale_cv_rms_K"] for row in rows),
        "target_shape_cv_rms_abs_error": _summary(row["target_shape_cv_rms_abs_error"] for row in rows),
        "target_raw_reconstruction_max_abs_error_K": _summary(row["target_raw_reconstruction_max_abs_error_K"] for row in rows),
        "target_projected_dirichlet_max_abs_error_K": _summary(row["target_projected_dirichlet_max_abs_error_K"] for row in rows),
        "target_projection_non_dirichlet_max_abs_change_K": _summary(
            row["target_projection_non_dirichlet_max_abs_change_K"] for row in rows
        ),
        "target_dirichlet_label_max_abs_error_K": _summary(
            row["target_dirichlet_label_max_abs_error_K"] for row in rows
        ),
        "coordinate_fallback_used_count": int(sum(int(row["coordinate_fallback_used"]) for row in rows)),
        "boundary_signatures": dict(sorted(signatures.items())),
    }


def _build_reconstructed(rows: Sequence[Mapping[str, Any]], roles: Sequence[str]) -> dict[str, Any]:
    groups = _group_rows(rows, roles)
    return {
        "row_count": len(rows),
        "role_counts": {role: len(groups[role]) for role in roles},
        "target_decomposition": _target_decomposition_summary(rows),
        "oracle_diagnostics": {name: _oracle_summary(group) for name, group in groups.items()},
        "hard_failure_decomposition": _hard_failure_decomposition(rows),
        "lateral_spreading": {name: _lateral_summary(group) for name, group in groups.items()},
        "duplicate_leakage": _duplicate_summary(rows),
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
            raise AuditError(f"summary reconstruction list lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None or not math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise AuditError(f"summary reconstruction values differ at {path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AuditError(f"summary reconstruction values differ at {path}: {actual!r} != {expected!r}")


def _verify_summary(table: Path, summary_path: Path) -> dict[str, Any]:
    payload = _read_json(summary_path)
    rows = _read_table(table)
    roles = sorted({str(row["role"]) for row in rows}, key=_role_key)
    reconstructed = _build_reconstructed(rows, roles)
    expected = _mapping(payload.get("reconstructed_from_table"), "reconstructed_from_table")
    _assert_close(reconstructed, expected)
    table_info = _mapping(payload.get("per_sample_table"), "per_sample_table")
    if table_info.get("sha256") != _sha256(table) or int(table_info.get("row_count", -1)) != len(rows):
        raise AuditError("per-sample table checksum or row count mismatch")
    if table_info.get("columns") != TABLE_COLUMNS:
        raise AuditError("per-sample table column manifest mismatch")
    return {"audit_id": payload.get("audit_id"), "row_count": len(rows), "table_sha256": _sha256(table), "verification": "passed"}


def _output_paths(paths: Sequence[Path | None], overwrite: bool) -> tuple[Path, Path, Path]:
    if len(paths) != 3 or any(path is None for path in paths):
        raise AuditError("audit requires --output-table, --output-json, and --output-md")
    table, summary_json, summary_md = (Path(path) for path in paths if path is not None)
    if len({table.resolve(), summary_json.resolve(), summary_md.resolve()}) != 3:
        raise AuditError("output paths must be distinct")
    if not overwrite:
        existing = [path for path in (table, summary_json, summary_md) if path.exists()]
        if existing:
            raise AuditError(f"refusing to overwrite existing output(s): {existing}")
    return table, summary_json, summary_md


def _contract_runtime(contract: Mapping[str, Any] | None) -> tuple[str, bool, dict[str, str]]:
    if contract is None:
        raise AuditError("Gate 3 requires a machine-readable contract")
    semantic = _mapping(contract.get("current_v5_semantics"), "current_v5_semantics")
    reference = str(semantic.get("reference_dirichlet_region", ""))
    if not reference:
        raise AuditError("contract must name a reference Dirichlet region")
    fallback = bool(_mapping(contract.get("boundary_interface"), "boundary_interface").get("allow_coordinate_fallback", False))
    if fallback:
        raise AuditError("frozen V5 contract must not enable coordinate fallback")
    raw_expected = _mapping(semantic.get("expected_region_types"), "current_v5_semantics.expected_region_types")
    expected = {str(region): str(boundary_type).lower() for region, boundary_type in raw_expected.items()}
    if not expected or expected.get(reference) != "dirichlet":
        raise AuditError("contract expected_region_types must include the reference Dirichlet region")
    return reference, fallback, expected


def _validate_current_v5_rows(
    rows: Sequence[Mapping[str, Any]],
    reference_region_id: str,
    expected_region_types: Mapping[str, str],
) -> None:
    for row in rows:
        if row["reference_dirichlet_region"] != reference_region_id:
            raise AuditError(f"{row['sample_id']}: reference region drifted from frozen contract")
        if int(row["dirichlet_region_count"]) != 1:
            raise AuditError(f"{row['sample_id']}: current V5 must instantiate exactly one Dirichlet region")
        actual_region_types: dict[str, str] = {}
        for entry in str(row["boundary_signature"]).split(";"):
            parts = entry.split(":")
            if len(parts) != 3:
                raise AuditError(f"{row['sample_id']}: malformed boundary signature")
            actual_region_types[parts[0]] = parts[1]
        if actual_region_types != dict(expected_region_types):
            raise AuditError(
                f"{row['sample_id']}: frozen V5 boundary types differ from contract "
                f"{actual_region_types} != {dict(expected_region_types)}"
            )
        if int(row["coordinate_fallback_used"]) != 0:
            raise AuditError(f"{row['sample_id']}: frozen V5 must not use coordinate fallback")
        if int(row["target_decomposition_pass"]) != 1:
            raise AuditError(f"{row['sample_id']}: target shape/scale reconstruction invariant failed")
        for checkpoint in CHECKPOINTS:
            if int(row[f"{checkpoint}_prediction_available"]) != 1:
                raise AuditError(f"{row['sample_id']}: missing {checkpoint} frozen prediction")
            if float(row[f"{checkpoint}_prediction_shape_reconstruction_max_abs_error_K"]) > RECONSTRUCTION_TOL_K:
                raise AuditError(f"{row['sample_id']}: {checkpoint} shape/scale reconstruction invariant failed")
            if float(row[f"{checkpoint}_boundary_projection_dirichlet_max_abs_error_K"]) > DIRICHLET_TOL_K:
                raise AuditError(f"{row['sample_id']}: {checkpoint} boundary projection invariant failed")
            if float(row[f"{checkpoint}_boundary_projection_non_dirichlet_max_abs_change_K"]) > RECONSTRUCTION_TOL_K:
                raise AuditError(f"{row['sample_id']}: {checkpoint} projection changed non-Dirichlet nodes")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    reconstructed = _mapping(payload["reconstructed_from_table"], "reconstructed")
    target = _mapping(reconstructed["target_decomposition"], "target")
    oracle = _mapping(reconstructed["oracle_diagnostics"], "oracle")
    hard = _mapping(reconstructed["hard_failure_decomposition"], "hard")
    lateral = _mapping(reconstructed["lateral_spreading"], "lateral")
    table = _mapping(payload["per_sample_table"], "table")
    lines = [
        "# V5 Gate 3 Boundary-General Shape–Scale Oracle Closeout",
        "",
        "## Scope And Frozen Semantics",
        "",
        "- Frozen V5 semantics: bottom Dirichlet, top Robin, sides adiabatic; `DeltaT = T - T_bottom`.",
        "- `scale` is CV-weighted RMS(`DeltaT`); `shape = DeltaT / (scale + eps)`.",
        "- The implementation reads region type, masks, and prescribed values from metadata plus BC features. It does not infer a Dirichlet location from z coordinates in this V5 audit.",
        "- This is a read-only diagnostic of frozen V4P5_02 best/final raw-temperature predictions. No model, loss, configuration, data, or training changed.",
        "",
        "## Boundary And Reconstruction Invariants",
        "",
        f"- Samples decomposed: `{target['sample_count']}`; nonzero target scales: `{target['nonzero_target_count']}`; all invariant passes: `{target['decomposition_pass_count']}`.",
        f"- Target reconstruction max abs error: `{_fmt(_mapping(target['target_raw_reconstruction_max_abs_error_K'], 'recon')['max'], 12)}` K; Dirichlet projection max error: `{_fmt(_mapping(target['target_projected_dirichlet_max_abs_error_K'], 'dirichlet')['max'], 12)}` K.",
        f"- Coordinate fallback used: `{target['coordinate_fallback_used_count']}`. Current boundary signature(s): `{target['boundary_signatures']}`.",
        "- Projection occurs in raw physical temperature space: it sets only Dirichlet nodes to their prescribed values and leaves non-Dirichlet values unchanged.",
        "",
        "## Sample-First CV-Weighted Oracle Metrics",
        "",
        "`predicted_shape_true_scale` is the shape-only oracle; `true_shape_predicted_scale` is the scale-only oracle. Amplitude ratio is the weighted projection onto the target DeltaT, while CV-RMS ratio is the uncentered field RMS ratio.",
        "",
    ]
    for group in ("clean", "hard", *ROLE_ORDER):
        if group not in oracle:
            continue
        group_metrics = _mapping(oracle[group], f"oracle.{group}")
        lines.extend([f"### {group}", "", "| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for checkpoint in CHECKPOINTS:
            result = _mapping(group_metrics[checkpoint], f"{group}.{checkpoint}")
            def metric(variant: str, name: str) -> Any:
                return _mapping(_mapping(result[variant], variant)[name], name)["mean"]
            lines.append(
                "| " + " | ".join(
                    (
                        checkpoint,
                        _fmt(metric("original", "cv_rmse_K")),
                        _fmt(metric("predicted_shape_true_scale", "cv_rmse_K")),
                        _fmt(metric("true_shape_predicted_scale", "cv_rmse_K")),
                        _fmt(metric("boundary_projected_original", "cv_rmse_K")),
                        _fmt(metric("original", "spatial_correlation")),
                        _fmt(metric("original", "amplitude_ratio")),
                        _fmt(metric("original", "cv_rms_ratio")),
                        _fmt(metric("original", "hotspot_cv_rmse_K")),
                    )
                ) + " |"
            )
        lines.append("")
    lines.extend(["## Hard-Failure Decomposition And Gate 4 Direction", ""])
    for checkpoint in CHECKPOINTS:
        result = _mapping(hard[checkpoint], checkpoint)
        lines.extend(
            (
                f"- `{checkpoint}`: original / shape-only / scale-only / boundary-projected hard RMSE = `{_fmt(result['end_to_end_original_cv_rmse_K'])}` / `{_fmt(result['shape_only_oracle_cv_rmse_K'])}` / `{_fmt(result['scale_only_oracle_cv_rmse_K'])}` / `{_fmt(result['boundary_projected_original_cv_rmse_K'])}` K.",
                f"  Gate 4 direction: `{result['gate4_direction']}`.",
            )
        )
    lines.extend(["", "These counterfactual components are non-additive diagnostic evidence; they do not themselves establish a causal mechanism.", "", "## Lateral-Spreading Mechanism Evidence", "", "The following relations use the frozen Gate 1 corrected `z_collapsed_1d_operator` scale log residual. They are descriptive associations, not causal claims.", "", "| group | feature | n | Spearman rho | slope | R2 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for group in ("all_samples", "clean", "hard", *ROLE_ORDER):
        if group not in lateral:
            continue
        relations = _mapping(_mapping(lateral[group], group)["relations_to_gate1_operator_scale_log_residual"], group)
        for feature, relation_raw in relations.items():
            relation = _mapping(relation_raw, feature)
            lines.append(
                f"| {group} | {feature} | {relation['sample_count']} | {_fmt(relation['Spearman_rho'])} | {_fmt(relation['slope'])} | {_fmt(relation['R2'])} |"
            )
    leakage = _mapping(reconstructed["duplicate_leakage"], "leakage")
    lines.extend(
        (
            "",
            "## Integrity And Reproducibility",
            "",
            f"- Per-sample CSV: `{table['row_count']}` rows; SHA256 `{table['sha256']}`.",
            f"- Cross-role input/full/provenance duplicate groups: `{_mapping(leakage['cross_role_model_input_duplicate_groups'], 'input')['group_count']}` / `{_mapping(leakage['cross_role_full_sample_duplicate_groups'], 'full')['group_count']}` / `{_mapping(leakage['cross_role_provenance_duplicate_groups'], 'provenance')['group_count']}`.",
            "- `--verify-summary` independently reconstructs all target, oracle, hard-failure, lateral, and leakage summaries from the CSV only.",
            "- Test roles are frozen descriptive reports only; no Gate 3 threshold, formula, or method was selected from them.",
            "",
        )
    )
    return "\n".join(lines)


def _write_json_md(payload: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def _run_audit(
    *,
    dataset: Path,
    split_map: Path,
    contract_path: Path,
    gate1_table: Path,
    prediction_paths: Mapping[str, Sequence[Path]],
    output_table: Path,
    output_json: Path,
    output_md: Path,
    table_label: str | None,
) -> dict[str, Any]:
    assignments, roles, split_payload, contract = _load_inputs(dataset, split_map, contract_path)
    assert contract is not None
    reference_region_id, allow_coordinate_fallback, expected_region_types = _contract_runtime(contract)
    expected_ids = set(assignments)
    gate1, gate1_hash = _load_gate1_table(gate1_table, expected_ids)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_artifacts: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in CHECKPOINTS:
        predictions[checkpoint], prediction_artifacts[checkpoint] = _load_prediction_archives(
            prediction_paths[checkpoint], expected_ids, checkpoint
        )
    rows = [
        _sample_row(
            sample_dir=dataset / sample_id,
            role=assignments[sample_id],
            gate1=gate1[sample_id],
            predictions=predictions,
            reference_region_id=reference_region_id,
            allow_coordinate_fallback=allow_coordinate_fallback,
        )
        for sample_id in sorted(assignments)
    ]
    _validate_current_v5_rows(rows, reference_region_id, expected_region_types)
    _write_table(rows, output_table)
    reconstructed = _build_reconstructed(rows, roles)
    payload = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_frozen_prediction_diagnostics",
        "contract_id": contract.get("contract_id"),
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "dataset_path": dataset.as_posix(),
            "split_map_path": split_map.as_posix(),
            "sample_count": len(rows),
            "roles": roles,
            "role_counts": {role: int(sum(row["role"] == role for row in rows)) for role in roles},
        },
        "current_v5_semantics": {
            "bottom_dirichlet": True,
            "top_robin": True,
            "sides_adiabatic": True,
            "deltaT": "temperature_K - prescribed value of contract reference Dirichlet region",
            "current_reference_region": reference_region_id,
            "scale": "CV-weighted RMS(DeltaT)",
            "shape": "DeltaT/(scale+eps)",
            "eps_K": EPS_K,
        },
        "boundary_interface": {
            "mask_inputs": "sample_meta.boundary_regions.point_indices cross-checked against sample_meta.bc_feature_names/bc_features",
            "supported_region_types": ["dirichlet", "robin", "neumann", "adiabatic"],
            "mixed_contract": "future mixed BC is represented by multiple explicit region entries; literal type=mixed must be expanded and otherwise errors",
            "coordinate_fallback": "disabled for frozen V5; if enabled, requires an explicit region-local coordinate_fallback mapping",
            "current_data_coverage": "only current bottom Dirichlet/top Robin/sides adiabatic is instantiated; this audit does not claim arbitrary-BC generalization",
            "projection": "raw physical temperature only; assign Dirichlet values and do not alter non-Dirichlet nodes",
        },
        "gate1_scale_source": {
            "table": gate1_table.as_posix(),
            "sha256": gate1_hash,
            "candidate": "z_collapsed_1d_operator",
            "residual": "log(pred_z_collapsed_1d_operator_K / target_scale_cv_rms_K)",
        },
        "frozen_prediction_archives": prediction_artifacts,
        "read_only_guardrails": {
            "model_parameter_changes": 0,
            "training_runs": 0,
            "reference_solver_calls": 0,
            "dataset_writes": 0,
            "permitted_writes": ["explicit Gate 3 CSV", "explicit Gate 3 JSON", "explicit Gate 3 Markdown"],
        },
        "per_sample_table": {
            "path": table_label or output_table.as_posix(),
            "sha256": _sha256(output_table),
            "row_count": len(rows),
            "columns": TABLE_COLUMNS,
        },
        "reconstructed_from_table": reconstructed,
        "method_selection": {
            "test_roles_used_for_threshold_formula_or_method_selection": False,
            "policy": "all oracle and lateral reports are descriptive; no learned scale branch or Gate 4 model decision is implemented here",
        },
    }
    _write_json_md(payload, output_json, output_md)
    return payload


def _dry_run(
    *,
    dataset: Path,
    split_map: Path,
    contract_path: Path,
    gate1_table: Path,
    prediction_paths: Mapping[str, Sequence[Path]],
) -> dict[str, Any]:
    assignments, roles, split_payload, contract = _load_inputs(dataset, split_map, contract_path)
    reference_region_id, _fallback, _expected_region_types = _contract_runtime(contract)
    expected_ids = set(assignments)
    _gate1, gate1_hash = _load_gate1_table(gate1_table, expected_ids)
    archive_counts: dict[str, int] = {}
    for checkpoint in CHECKPOINTS:
        merged, _artifacts = _load_prediction_archives(prediction_paths[checkpoint], expected_ids, checkpoint)
        archive_counts[checkpoint] = len(merged)
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
        "reference_dirichlet_region": reference_region_id,
        "gate1_table_sha256": gate1_hash,
        "prediction_coverage": archive_counts,
        "planned_reads": {"sample_directories": len(assignments), "sample_arrays": 0, "sample_metadata": 0},
        "planned_writes": [],
        "model_parameter_changes": 0,
        "training_runs": 0,
        "reference_solver_calls": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--gate1-table", type=Path)
    parser.add_argument("--best-predictions", type=Path, action="append", default=[])
    parser.add_argument("--final-predictions", type=Path, action="append", default=[])
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--table-label", help="canonical repository label for the generated CSV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-summary", action="store_true")
    parser.add_argument("--table", type=Path, help="CSV table used with --verify-summary")
    parser.add_argument("--summary-json", type=Path, help="summary JSON used with --verify-summary")
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
        required = (args.dataset, args.split_map, args.contract, args.gate1_table)
        if any(value is None for value in required):
            raise AuditError("audit requires --dataset, --split-map, --contract, and --gate1-table")
        prediction_paths = {"best": list(args.best_predictions), "final": list(args.final_predictions)}
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run(
                        dataset=args.dataset,
                        split_map=args.split_map,
                        contract_path=args.contract,
                        gate1_table=args.gate1_table,
                        prediction_paths=prediction_paths,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        table, summary_json, summary_md = _output_paths(
            (args.output_table, args.output_json, args.output_md), args.overwrite
        )
        _run_audit(
            dataset=args.dataset,
            split_map=args.split_map,
            contract_path=args.contract,
            gate1_table=args.gate1_table,
            prediction_paths=prediction_paths,
            output_table=table,
            output_json=summary_json,
            output_md=summary_md,
            table_label=args.table_label,
        )
    except AuditError as exc:
        print(f"Gate 3 audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
