"""Inference-only global physics context for V5 Global FiLM.

The context is assembled from one sample's raw ``coords/k/q/BC`` inputs and
control-volume weights.  It intentionally has no target-temperature, target
shape, target-scale, residual, or oracle argument.  Train-only standardization
is a separate explicit step so a caller cannot accidentally fit it on valid,
test, or hard roles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_v5_metrics import control_volume_weights


EPS = 1.0e-12
SCHEMA_VERSION = "heat3d_v5_global_physics_context_v1"
GLOBAL_CONTEXT_FEATURES = (
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
    "T_bottom_K",
    "T_inf_K",
    "T_inf_minus_T_bottom_K",
    "bc_top_cv_fraction",
    "bc_bottom_cv_fraction",
    "bc_side_cv_fraction",
)
FORBIDDEN_INPUT_TOKENS = (
    "target",
    "label",
    "oracle",
    "residual",
    "prediction",
    "error",
)
FEATURE_PROVENANCE = {
    "log_s_phys_K": "coords+k_z+q+bottom_mask+top_h+BC_temperature_offset via z_collapsed_1d_operator",
    "P_operator_W": "q+control_volume+bottom_BC_mask",
    "log_P_operator_W": "P_operator_W",
    "q_weighted_local_kz_W_mK": "q+k_z+control_volume",
    "q_weighted_inverse_kz_mK_W": "q+k_z+control_volume",
    "q_low_k_overlap_fraction": "q+k_z+control_volume",
    "source_concentration": "q+control_volume",
    "source_z_centroid_normalized": "coords+q+control_volume",
    "source_layer_kz_heterogeneity_cv": "coords+k_z+q+control_volume",
    "harmonic_kx_W_mK": "k_x+control_volume",
    "harmonic_ky_W_mK": "k_y+control_volume",
    "harmonic_kz_W_mK": "k_z+control_volume",
    "anisotropy_xy_over_z": "harmonic conductivity features",
    "log_Lx_m": "coords",
    "log_Ly_m": "coords",
    "log_Lz_m": "coords",
    "log_top_area_m2": "coords",
    "log_top_h_W_m2K": "top_h BC feature",
    "T_bottom_K": "reference Dirichlet temperature plus bottom BC feature",
    "T_inf_K": "reference Dirichlet temperature plus top ambient BC feature",
    "T_inf_minus_T_bottom_K": "top/bottom BC features",
    "bc_top_cv_fraction": "top BC mask+control_volume",
    "bc_bottom_cv_fraction": "bottom BC mask+control_volume",
    "bc_side_cv_fraction": "side BC mask+control_volume",
}


class GlobalContextError(ValueError):
    """Raised when an inference-only global-context invariant is violated."""


def validate_global_context_schema(feature_names: Sequence[str] = GLOBAL_CONTEXT_FEATURES) -> tuple[str, ...]:
    """Validate the fixed feature schema and its non-label provenance."""

    names = tuple(str(name) for name in feature_names)
    if names != GLOBAL_CONTEXT_FEATURES:
        raise GlobalContextError("global context feature schema drifted from the frozen V5 allowlist")
    if len(names) != len(set(names)):
        raise GlobalContextError("global context feature names must be unique")
    for name in names:
        normalized = name.lower()
        if any(token in normalized for token in FORBIDDEN_INPUT_TOKENS):
            raise GlobalContextError(f"label-derived token appears in global context feature {name!r}")
        if name not in FEATURE_PROVENANCE:
            raise GlobalContextError(f"global context feature lacks provenance: {name}")
    return names


def global_context_from_raw_condition(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    reference_temperature_K: float,
) -> dict[str, float]:
    """Build one V5 context row strictly from raw physical model inputs.

    ``raw_condition`` must be the *recovered raw* condition view used by the
    V4 model, with k/q/BC fields and optional broadcast geometry fields.  The
    reference temperature is a prescribed Dirichlet BC, never a target label.
    """

    validate_global_context_schema()
    points = _coords(coords)
    condition = np.asarray(raw_condition, dtype=np.float64)
    names = tuple(str(name) for name in condition_feature_names)
    if condition.ndim != 2 or condition.shape[0] != points.shape[0] or condition.shape[1] != len(names):
        raise GlobalContextError(
            "raw_condition must have shape [node_count, len(condition_feature_names)]"
        )
    if not np.all(np.isfinite(condition)):
        raise GlobalContextError("raw_condition contains non-finite values")
    if not math.isfinite(float(reference_temperature_K)):
        raise GlobalContextError("reference_temperature_K must be finite")

    values = {name: condition[:, index] for index, name in enumerate(names)}
    required = (
        "k_x",
        "k_y",
        "k_z",
        "q",
        "is_top",
        "is_bottom",
        "is_side",
        "top_h",
        "top_T_inf_minus_T_ref",
        "bottom_T_fixed_minus_T_ref",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise GlobalContextError(f"raw condition misses required global-context fields: {missing}")

    volumes = control_volume_weights(points)
    axes, inverse = _rectilinear_axes(points)
    kx = values["k_x"]
    ky = values["k_y"]
    kz = values["k_z"]
    q = values["q"]
    _positive(kx, "k_x")
    _positive(ky, "k_y")
    _positive(kz, "k_z")
    top_h = _broadcast_scalar(values["top_h"], "top_h")
    if top_h <= 0.0:
        raise GlobalContextError("top_h must be positive")
    top_offset = _broadcast_scalar(values["top_T_inf_minus_T_ref"], "top_T_inf_minus_T_ref")
    bottom_offset = _broadcast_scalar(values["bottom_T_fixed_minus_T_ref"], "bottom_T_fixed_minus_T_ref")
    t_bottom = float(reference_temperature_K) + bottom_offset
    t_inf = float(reference_temperature_K) + top_offset
    bc_offset = t_inf - t_bottom
    bottom_mask = values["is_bottom"] > 0.5
    top_mask = values["is_top"] > 0.5
    side_mask = values["is_side"] > 0.5
    if not np.any(bottom_mask) or not np.any(top_mask):
        raise GlobalContextError("BC masks must identify both top and bottom nodes")

    q_operator = np.asarray(q, dtype=np.float64).copy()
    q_operator[bottom_mask] = 0.0
    p_operator = float(np.dot(q_operator, volumes))
    q_positive = np.maximum(q_operator, 0.0)
    p_positive = float(np.dot(q_positive, volumes))
    if p_operator <= EPS or p_positive <= EPS:
        raise GlobalContextError("P5 global context requires positive non-bottom source power")
    s_phys = z_collapsed_1d_operator(
        axes=axes,
        inverse=inverse,
        volumes=volumes,
        q_operator=q_operator,
        k_z=kz,
        top_h=top_h,
        bc_offset=bc_offset,
    )
    if s_phys <= EPS:
        raise GlobalContextError("z_collapsed_1d_operator must be positive")

    source_weights = q_positive * volumes
    q_weighted_local_kz = float(np.dot(source_weights, kz) / p_positive)
    q_weighted_inverse_kz = float(np.dot(source_weights, 1.0 / kz) / p_positive)
    low_k_threshold = _weighted_quantile(kz, volumes, 0.25)
    q_low_k_overlap = float(np.sum(source_weights[kz <= low_k_threshold]) / p_positive)
    total_volume = float(np.sum(volumes))
    q_rms = math.sqrt(float(np.dot(q_operator * q_operator, volumes) / total_volume))
    q_mean = p_operator / total_volume
    source_concentration = q_rms / max(abs(q_mean), EPS)
    z_axis = axes[2]
    lz = float(z_axis[-1] - z_axis[0])
    source_z = float(np.dot(points[:, 2], source_weights) / p_positive)
    source_z_normalized = (source_z - float(z_axis[0])) / lz
    layer_kz_heterogeneity = _source_layer_kz_heterogeneity(
        kz=kz,
        q_positive=q_positive,
        volumes=volumes,
        z_inverse=inverse[2],
        layer_count=z_axis.size,
    )
    harmonic_kx = _weighted_harmonic(kx, volumes, "k_x")
    harmonic_ky = _weighted_harmonic(ky, volumes, "k_y")
    harmonic_kz = _weighted_harmonic(kz, volumes, "k_z")
    anisotropy = math.sqrt(harmonic_kx * harmonic_ky) / harmonic_kz
    lx = float(axes[0][-1] - axes[0][0])
    ly = float(axes[1][-1] - axes[1][0])
    top_area = lx * ly
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0 or top_area <= 0.0:
        raise GlobalContextError("physical coordinate extents must be positive")

    context = {
        "log_s_phys_K": math.log(s_phys),
        "P_operator_W": p_operator,
        "log_P_operator_W": math.log(p_operator),
        "q_weighted_local_kz_W_mK": q_weighted_local_kz,
        "q_weighted_inverse_kz_mK_W": q_weighted_inverse_kz,
        "q_low_k_overlap_fraction": q_low_k_overlap,
        "source_concentration": source_concentration,
        "source_z_centroid_normalized": source_z_normalized,
        "source_layer_kz_heterogeneity_cv": layer_kz_heterogeneity,
        "harmonic_kx_W_mK": harmonic_kx,
        "harmonic_ky_W_mK": harmonic_ky,
        "harmonic_kz_W_mK": harmonic_kz,
        "anisotropy_xy_over_z": anisotropy,
        "log_Lx_m": math.log(lx),
        "log_Ly_m": math.log(ly),
        "log_Lz_m": math.log(lz),
        "log_top_area_m2": math.log(top_area),
        "log_top_h_W_m2K": math.log(top_h),
        "T_bottom_K": t_bottom,
        "T_inf_K": t_inf,
        "T_inf_minus_T_bottom_K": bc_offset,
        "bc_top_cv_fraction": float(np.sum(volumes[top_mask]) / total_volume),
        "bc_bottom_cv_fraction": float(np.sum(volumes[bottom_mask]) / total_volume),
        "bc_side_cv_fraction": float(np.sum(volumes[side_mask]) / total_volume),
    }
    _validate_context_values(context)
    return context


def context_vector(context: Mapping[str, Any]) -> np.ndarray:
    """Return the fixed-order V5 global context vector."""

    validate_global_context_schema()
    keys = set(context)
    expected = set(GLOBAL_CONTEXT_FEATURES)
    if keys != expected:
        raise GlobalContextError(
            f"global context keys differ from schema: missing={sorted(expected - keys)} extra={sorted(keys - expected)}"
        )
    vector = np.asarray([float(context[name]) for name in GLOBAL_CONTEXT_FEATURES], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise GlobalContextError("global context vector contains non-finite values")
    return vector


def fit_train_only_standardizer(
    contexts: Sequence[Mapping[str, Any]],
    *,
    fit_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit a finite global-context standardizer from train samples only."""

    if not contexts:
        raise GlobalContextError("at least one train context is required")
    sample_ids = tuple(str(sample_id) for sample_id in fit_sample_ids)
    if len(sample_ids) != len(contexts) or len(set(sample_ids)) != len(sample_ids):
        raise GlobalContextError("fit_sample_ids must be unique and match train context count")
    matrix = np.vstack([context_vector(context) for context in contexts])
    mean = matrix.mean(axis=0)
    raw_std = matrix.std(axis=0)
    std = np.where(raw_std > EPS, raw_std, 1.0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(GLOBAL_CONTEXT_FEATURES),
        "fit_population": "train_only",
        "fit_sample_count": len(sample_ids),
        "fit_sample_ids_sha256": _sample_ids_hash(sample_ids),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "zero_variance_feature_names": [
            GLOBAL_CONTEXT_FEATURES[index] for index in np.flatnonzero(raw_std <= EPS)
        ],
        "input_provenance": FEATURE_PROVENANCE,
        "target_or_label_derived_inputs": False,
    }
    return payload


def standardize_contexts(
    contexts: Sequence[Mapping[str, Any]],
    standardizer: Mapping[str, Any],
) -> np.ndarray:
    """Apply a persisted train-only standardizer without refitting it."""

    if tuple(standardizer.get("feature_names") or ()) != GLOBAL_CONTEXT_FEATURES:
        raise GlobalContextError("standardizer feature schema does not match V5 global context")
    mean = np.asarray(standardizer.get("mean"), dtype=np.float64)
    std = np.asarray(standardizer.get("std"), dtype=np.float64)
    if mean.shape != (len(GLOBAL_CONTEXT_FEATURES),) or std.shape != mean.shape:
        raise GlobalContextError("global-context standardizer dimensions are invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise GlobalContextError("global-context standardizer must be finite and positive")
    matrix = np.vstack([context_vector(context) for context in contexts])
    standardized = (matrix - mean[None, :]) / std[None, :]
    if not np.all(np.isfinite(standardized)):
        raise GlobalContextError("standardized global context contains non-finite values")
    return standardized.astype(np.float32)


def batch_global_context_from_raw_condition(
    *,
    coords_per_sample: Sequence[np.ndarray],
    raw_conditions_per_sample: Sequence[np.ndarray],
    condition_feature_names: Sequence[str],
    reference_temperatures_K: Sequence[float],
) -> list[dict[str, float]]:
    """Build context rows for a batch without accessing any target field."""

    lengths = {
        len(coords_per_sample),
        len(raw_conditions_per_sample),
        len(reference_temperatures_K),
    }
    if len(lengths) != 1:
        raise GlobalContextError("batch context inputs must have the same sample count")
    return [
        global_context_from_raw_condition(
            coords=coords,
            raw_condition=condition,
            condition_feature_names=condition_feature_names,
            reference_temperature_K=float(t_ref),
        )
        for coords, condition, t_ref in zip(
            coords_per_sample, raw_conditions_per_sample, reference_temperatures_K
        )
    ]


def z_collapsed_1d_operator(
    *,
    axes: Sequence[np.ndarray],
    inverse: Sequence[np.ndarray],
    volumes: np.ndarray,
    q_operator: np.ndarray,
    k_z: np.ndarray,
    top_h: float,
    bc_offset: float,
) -> float:
    """Exact Gate-1 `z_collapsed_1d_operator` scalar without labels."""

    x_axis, y_axis, z_axis = [np.asarray(axis, dtype=np.float64) for axis in axes]
    ix_values, iy_values, iz_values = [np.asarray(value, dtype=np.int64) for value in inverse]
    shape = (int(x_axis.size), int(y_axis.size), int(z_axis.size))
    node_count = int(np.asarray(volumes).size)
    if q_operator.shape != (node_count,) or k_z.shape != (node_count,):
        raise GlobalContextError("operator arrays must match the control-volume node count")
    if int(np.prod(shape)) != node_count:
        raise GlobalContextError("z-collapsed operator requires a complete rectilinear grid")
    grid = -np.ones(shape, dtype=np.int64)
    grid[ix_values, iy_values, iz_values] = np.arange(node_count, dtype=np.int64)
    if np.any(grid < 0):
        raise GlobalContextError("z-collapsed operator grid mapping is incomplete")
    dx_cv = _control_widths(x_axis, "x")
    dy_cv = _control_widths(y_axis, "y")
    layer_volumes = np.zeros(shape[2], dtype=np.float64)
    layer_power = np.zeros(shape[2], dtype=np.float64)
    for iz in range(shape[2]):
        nodes = grid[:, :, iz].reshape(-1)
        layer_volumes[iz] = float(np.sum(volumes[nodes]))
        layer_power[iz] = float(np.dot(q_operator[nodes], volumes[nodes]))
    if np.any(layer_volumes <= 0.0):
        raise GlobalContextError("z-collapsed operator found an empty layer")
    matrix = np.zeros((shape[2], shape[2]), dtype=np.float64)
    rhs = np.zeros(shape[2], dtype=np.float64)
    for iz in range(shape[2] - 1):
        distance = float(z_axis[iz + 1] - z_axis[iz])
        if distance <= 0.0:
            raise GlobalContextError("z coordinates must increase")
        total_conductance = 0.0
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                lower = int(grid[ix, iy, iz])
                upper = int(grid[ix, iy, iz + 1])
                harmonic_k = 2.0 * float(k_z[lower]) * float(k_z[upper]) / (
                    float(k_z[lower]) + float(k_z[upper])
                )
                total_conductance += harmonic_k * float(dx_cv[ix] * dy_cv[iy]) / distance
        matrix[iz, iz] += total_conductance
        matrix[iz, iz + 1] -= total_conductance
        matrix[iz + 1, iz] -= total_conductance
        matrix[iz + 1, iz + 1] += total_conductance
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = 0.0
    for iz in range(1, shape[2]):
        rhs[iz] += layer_power[iz]
    top_area = float(np.sum(np.outer(dx_cv, dy_cv)))
    robin = float(top_h) * top_area
    matrix[-1, -1] += robin
    rhs[-1] += robin * float(bc_offset)
    try:
        delta = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise GlobalContextError(f"z-collapsed operator matrix is singular: {exc}") from exc
    scale = float(math.sqrt(np.dot(layer_volumes, delta * delta) / float(np.sum(layer_volumes))))
    if not math.isfinite(scale):
        raise GlobalContextError("z-collapsed operator returned a non-finite scale")
    return scale


def _coords(coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise GlobalContextError("coords must be finite with shape [N, 3]")
    return points


def _rectilinear_axes(coords: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    axes: list[np.ndarray] = []
    inverse: list[np.ndarray] = []
    for axis in range(3):
        values, indices = np.unique(coords[:, axis], return_inverse=True)
        if values.size < 2:
            raise GlobalContextError("every physical coordinate axis needs at least two values")
        axes.append(values)
        inverse.append(indices)
    if int(np.prod([axis.size for axis in axes])) != coords.shape[0]:
        raise GlobalContextError("coords must be a complete rectilinear grid")
    return axes, inverse


def _control_widths(axis: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise GlobalContextError(f"{label} axis must be strictly increasing")
    result = np.empty_like(values)
    result[0] = 0.5 * (values[1] - values[0])
    result[-1] = 0.5 * (values[-1] - values[-2])
    if values.size > 2:
        result[1:-1] = 0.5 * (values[2:] - values[:-2])
    if np.any(result <= 0.0):
        raise GlobalContextError(f"{label} control widths must be positive")
    return result


def _broadcast_scalar(values: np.ndarray, name: str) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise GlobalContextError(f"{name} must be non-empty and finite")
    mean = float(np.mean(array))
    tolerance = 1.0e-10 * max(1.0, abs(mean))
    if float(np.max(np.abs(array - mean))) > tolerance:
        raise GlobalContextError(f"{name} must be a sample-global broadcast field")
    return mean


def _positive(values: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise GlobalContextError(f"{name} must be finite and positive")


def _weighted_harmonic(values: np.ndarray, weights: np.ndarray, name: str) -> float:
    _positive(values, name)
    denominator = float(np.dot(weights, 1.0 / values))
    if denominator <= EPS:
        raise GlobalContextError(f"{name} harmonic denominator must be positive")
    return float(np.sum(weights) / denominator)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise GlobalContextError("weighted quantile must be in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _source_layer_kz_heterogeneity(
    *,
    kz: np.ndarray,
    q_positive: np.ndarray,
    volumes: np.ndarray,
    z_inverse: np.ndarray,
    layer_count: int,
) -> float:
    layer_power = []
    layer_k = []
    for layer in range(int(layer_count)):
        mask = z_inverse == layer
        layer_power.append(float(np.dot(q_positive[mask], volumes[mask])))
        layer_k.append(_weighted_harmonic(kz[mask], volumes[mask], f"k_z_layer_{layer}"))
    weights = np.asarray(layer_power, dtype=np.float64)
    values = np.asarray(layer_k, dtype=np.float64)
    if float(np.sum(weights)) <= EPS:
        return 0.0
    mean = float(np.dot(values, weights) / np.sum(weights))
    variance = float(np.dot(np.square(values - mean), weights) / np.sum(weights))
    return float(math.sqrt(max(variance, 0.0)) / max(abs(mean), EPS))


def _validate_context_values(context: Mapping[str, float]) -> None:
    if tuple(context) != GLOBAL_CONTEXT_FEATURES:
        raise GlobalContextError("global context was not constructed in fixed schema order")
    values = np.asarray([context[name] for name in GLOBAL_CONTEXT_FEATURES], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = [name for name in GLOBAL_CONTEXT_FEATURES if not math.isfinite(float(context[name]))]
        raise GlobalContextError(f"global context contains non-finite values: {bad}")


def _sample_ids_hash(sample_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(sample_ids), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
