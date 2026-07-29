"""Frozen 24D inference-only global context for V6 dual-Robin P1g."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


EPS = 1.0e-12
SCHEMA_VERSION = "heat3d_v6_dual_robin_global_physics_context_v1"
GLOBAL_CONTEXT_FEATURES_V6 = (
    "log_s_phys_K",
    "P_operator_W",
    "log_P_operator_W",
    "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W",
    "q_low_k_overlap_fraction",
    "source_concentration",
    "source_z_centroid_normalized",
    "source_layer_kz_heterogeneity_cv",
    "harmonic_kx_W_mK",
    "harmonic_ky_W_mK",
    "harmonic_kz_W_mK",
    "anisotropy_xy_over_z",
    "log_Lx_m",
    "log_Ly_m",
    "log_Lz_m",
    "log_top_area_m2",
    "log_top_h_W_m2K",
    "log_bottom_h_W_m2K",
    "top_T_inf_K",
    "bottom_T_inf_K",
    "bc_top_cv_fraction",
    "bc_bottom_cv_fraction",
    "bc_side_cv_fraction",
)


def validate_v6_global_context_schema(
    feature_names: Sequence[str] = GLOBAL_CONTEXT_FEATURES_V6,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if names != GLOBAL_CONTEXT_FEATURES_V6:
        raise ValueError("V6 global context drifted from the frozen 24D schema")
    forbidden = ("target", "label", "oracle", "prediction", "error")
    if any(any(token in name.lower() for token in forbidden) for name in names):
        raise ValueError("V6 global context contains a label-derived feature token")
    return names


def global_context_from_v6_inputs(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    reference_temperature_K: float,
    top_T_inf_K: float,
    bottom_T_inf_K: float,
    operator_point_weights: np.ndarray,
    package_total_power_W: float,
    package_extents_m: Sequence[float],
) -> dict[str, float]:
    validate_v6_global_context_schema()
    points = np.asarray(coords, dtype=np.float64)
    condition = np.asarray(raw_condition, dtype=np.float64)
    names = tuple(condition_feature_names)
    weights = np.asarray(operator_point_weights, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("V6 coords must be finite [N,3]")
    if condition.shape != (points.shape[0], len(names)) or not np.all(np.isfinite(condition)):
        raise ValueError("V6 raw_condition shape/finite invariant failed")
    if weights.shape != (points.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("V6 operator-point weights must be positive [N]")
    weights = weights / np.sum(weights)
    values = {name: condition[:, index] for index, name in enumerate(names)}
    required = (
        "k_x", "k_y", "k_z", "q", "is_top", "is_bottom", "is_side",
        "top_h", "bottom_h", "top_T_inf_minus_T_ref",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"V6 global context missing fields: {missing}")
    kx, ky, kz = values["k_x"], values["k_y"], values["k_z"]
    q = np.maximum(values["q"], 0.0)
    if any(np.any(array <= 0.0) for array in (kx, ky, kz)):
        raise ValueError("V6 conductivity must be positive")
    top_h = _broadcast(values["top_h"], "top_h")
    bottom_h = _broadcast(values["bottom_h"], "bottom_h")
    if min(top_h, bottom_h) <= 0.0:
        raise ValueError("V6 Robin h must be positive")
    if not all(math.isfinite(float(v)) for v in (reference_temperature_K, top_T_inf_K, bottom_T_inf_K)):
        raise ValueError("V6 ambient/reference temperatures must be finite")
    expected_offset = float(top_T_inf_K) - float(reference_temperature_K)
    if not math.isclose(
        _broadcast(values["top_T_inf_minus_T_ref"], "top_T_inf_minus_T_ref"),
        expected_offset,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError("V6 top ambient offset is inconsistent with prescribed metadata")

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = np.asarray(package_extents_m, dtype=np.float64)
    if np.any(spans <= 0.0):
        raise ValueError("V6 coordinate extents must be positive")
    lx, ly, lz = (float(v) for v in spans)
    top_area = lx * ly
    # P1g q is a projected volumetric field.  Its exact package power is
    # prescribed metadata, but the context may only consume inference inputs;
    # recover an operator estimate using the bounding-box volume.
    package_volume = lx * ly * lz
    point_volumes = weights * package_volume
    source_weights_raw = q * point_volumes
    raw_power = float(np.sum(source_weights_raw))
    p_operator = float(package_total_power_W)
    if p_operator <= EPS:
        raise ValueError("V6 global context requires positive source power")
    if raw_power <= EPS:
        raise ValueError("V6 projected q field requires positive support")
    source_weights = source_weights_raw * (p_operator / raw_power)
    q_mean = p_operator / package_volume
    q_rms = math.sqrt(float(np.sum(np.square(q) * point_volumes) / package_volume))
    source_concentration = q_rms / max(q_mean, EPS)
    source_z = float(np.sum(points[:, 2] * source_weights) / p_operator)
    source_z_normalized = (source_z - mins[2]) / lz
    q_local_kz = float(np.sum(source_weights * kz) / p_operator)
    q_inverse_kz = float(np.sum(source_weights / kz) / p_operator)
    low_k = _weighted_quantile(kz, weights, 0.25)
    q_low_overlap = float(np.sum(source_weights[kz <= low_k]) / p_operator)
    harmonic_kx = _weighted_harmonic(kx, weights)
    harmonic_ky = _weighted_harmonic(ky, weights)
    harmonic_kz = _weighted_harmonic(kz, weights)
    anisotropy = math.sqrt(harmonic_kx * harmonic_ky) / harmonic_kz
    layer_heterogeneity = _source_z_band_heterogeneity(points[:, 2], kz, source_weights)

    # Two Robin branches in parallel.  This is an inference-only scale proxy,
    # not a target-derived estimate and not a replacement for the solver.
    z_top = maxs[2] - source_z
    z_bottom = source_z - mins[2]
    r_top = z_top / max(harmonic_kz * top_area, EPS) + 1.0 / (top_h * top_area)
    r_bottom = z_bottom / max(harmonic_kz * top_area, EPS) + 1.0 / (bottom_h * top_area)
    g_top, g_bottom = 1.0 / r_top, 1.0 / r_bottom
    delta_proxy = (
        p_operator
        + g_top * (float(top_T_inf_K) - float(reference_temperature_K))
        + g_bottom * (float(bottom_T_inf_K) - float(reference_temperature_K))
    ) / (g_top + g_bottom)
    s_phys = max(abs(delta_proxy), EPS)

    total_weight = float(np.sum(weights))
    context = {
        "log_s_phys_K": math.log(s_phys),
        "P_operator_W": p_operator,
        "log_P_operator_W": math.log(p_operator),
        "q_weighted_local_kz_W_mK": q_local_kz,
        "q_weighted_inverse_kz_mK_W": q_inverse_kz,
        "q_low_k_overlap_fraction": q_low_overlap,
        "source_concentration": source_concentration,
        "source_z_centroid_normalized": source_z_normalized,
        "source_layer_kz_heterogeneity_cv": layer_heterogeneity,
        "harmonic_kx_W_mK": harmonic_kx,
        "harmonic_ky_W_mK": harmonic_ky,
        "harmonic_kz_W_mK": harmonic_kz,
        "anisotropy_xy_over_z": anisotropy,
        "log_Lx_m": math.log(lx),
        "log_Ly_m": math.log(ly),
        "log_Lz_m": math.log(lz),
        "log_top_area_m2": math.log(top_area),
        "log_top_h_W_m2K": math.log(top_h),
        "log_bottom_h_W_m2K": math.log(bottom_h),
        "top_T_inf_K": float(top_T_inf_K),
        "bottom_T_inf_K": float(bottom_T_inf_K),
        "bc_top_cv_fraction": float(np.sum(weights[values["is_top"] > 0.5]) / total_weight),
        "bc_bottom_cv_fraction": float(np.sum(weights[values["is_bottom"] > 0.5]) / total_weight),
        "bc_side_cv_fraction": float(np.sum(weights[values["is_side"] > 0.5]) / total_weight),
    }
    vector = np.asarray([context[name] for name in GLOBAL_CONTEXT_FEATURES_V6])
    if not np.all(np.isfinite(vector)):
        raise ValueError("V6 global context contains non-finite values")
    return context


def fit_train_only_v6_standardizer(
    contexts: Sequence[Mapping[str, Any]], *, fit_sample_ids: Sequence[str]
) -> dict[str, Any]:
    ids = tuple(str(value) for value in fit_sample_ids)
    if not contexts or len(contexts) != len(ids) or len(ids) != len(set(ids)):
        raise ValueError("V6 train-only context fit requires unique matching sample IDs")
    matrix = np.vstack([_vector(row) for row in contexts])
    mean = matrix.mean(axis=0)
    raw_std = matrix.std(axis=0)
    std = np.where(raw_std > EPS, raw_std, 1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(GLOBAL_CONTEXT_FEATURES_V6),
        "fit_population": "train_only",
        "fit_sample_count": len(ids),
        "fit_sample_ids_sha256": hashlib.sha256(
            json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "zero_variance_feature_names": [
            GLOBAL_CONTEXT_FEATURES_V6[index] for index in np.flatnonzero(raw_std <= EPS)
        ],
        "target_or_label_derived_inputs": False,
    }


def standardize_v6_contexts(
    contexts: Sequence[Mapping[str, Any]], standardizer: Mapping[str, Any]
) -> np.ndarray:
    if tuple(standardizer.get("feature_names") or ()) != GLOBAL_CONTEXT_FEATURES_V6:
        raise ValueError("V6 standardizer schema mismatch")
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    matrix = np.vstack([_vector(row) for row in contexts])
    result = (matrix - mean) / std
    if not np.all(np.isfinite(result)):
        raise ValueError("V6 standardized context contains non-finite values")
    return result.astype(np.float32)


def _vector(row: Mapping[str, Any]) -> np.ndarray:
    if set(row) != set(GLOBAL_CONTEXT_FEATURES_V6):
        raise ValueError("V6 context row schema mismatch")
    return np.asarray([float(row[name]) for name in GLOBAL_CONTEXT_FEATURES_V6])


def _broadcast(values: np.ndarray, name: str) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(array))
    if not np.allclose(array, mean, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(f"{name} must be sample-global broadcast")
    return mean


def _weighted_harmonic(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights) / np.sum(weights / values))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order[min(index, values.size - 1)]])


def _source_z_band_heterogeneity(
    z: np.ndarray, kz: np.ndarray, source_weights: np.ndarray
) -> float:
    if float(np.sum(source_weights)) <= EPS:
        return 0.0
    edges = np.linspace(float(np.min(z)), float(np.max(z)), 10)
    indices = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, 7)
    values: list[float] = []
    weights: list[float] = []
    for band in range(8):
        mask = indices == band
        power = float(np.sum(source_weights[mask]))
        if power > EPS:
            values.append(float(np.sum(source_weights[mask] * kz[mask]) / power))
            weights.append(power)
    if not values:
        return 0.0
    w = np.asarray(weights)
    v = np.asarray(values)
    mean = float(np.sum(w * v) / np.sum(w))
    variance = float(np.sum(w * np.square(v - mean)) / np.sum(w))
    return math.sqrt(max(variance, 0.0)) / max(abs(mean), EPS)
