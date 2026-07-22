"""Inference-only V5 scale-context and source/volume aggregation features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_v5_metrics import control_volume_weights


SCALE_CONTEXT_MODES = ("none", "xy_raw_global")
SCALE_DEEPSETS_MODES = ("none", "source_volume_residual")
XY_SCALE_CONTEXT_FEATURES = (
    "q_weighted_kx",
    "q_weighted_ky",
    "q_weighted_inverse_kx",
    "q_weighted_inverse_ky",
    "source_x_centroid_normalized",
    "source_y_centroid_normalized",
    "source_x_spread_normalized",
    "source_y_spread_normalized",
    "q_low_kx_overlap",
    "q_low_ky_overlap",
)

EPS = 1.0e-12


def validate_scale_context_schema(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Require the frozen, raw-input-only XY scale schema."""

    names = tuple(str(name) for name in feature_names)
    if names != XY_SCALE_CONTEXT_FEATURES:
        raise ValueError(
            "scale context schema must exactly match the frozen XY feature order; "
            f"got={names} expected={XY_SCALE_CONTEXT_FEATURES}"
        )
    return names


def xy_scale_context_from_raw_condition(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
) -> dict[str, float]:
    """Build ten sample-global XY features without temperature labels."""

    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(raw_condition, dtype=np.float64)
    names = tuple(str(name) for name in condition_feature_names)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,3], got {points.shape}")
    if values.ndim != 2 or values.shape[0] != points.shape[0]:
        raise ValueError(
            "raw_condition must have shape [N,F] aligned with coords, "
            f"got {values.shape} for {points.shape}"
        )
    required = ("k_x", "k_y", "q")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"XY scale context lacks raw condition features: {missing}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("XY scale context inputs must be finite")

    index = {name: names.index(name) for name in required}
    kx = np.maximum(values[:, index["k_x"]], EPS)
    ky = np.maximum(values[:, index["k_y"]], EPS)
    q = np.maximum(values[:, index["q"]], 0.0)
    volumes = np.asarray(control_volume_weights(points), dtype=np.float64)
    source_power = q * volumes
    total_power = float(np.sum(source_power))
    source_denominator = max(total_power, EPS)

    result: dict[str, float] = {}
    for axis_name, conductivity in (("x", kx), ("y", ky)):
        result[f"q_weighted_k{axis_name}"] = float(
            np.sum(source_power * conductivity) / source_denominator
        )
        result[f"q_weighted_inverse_k{axis_name}"] = float(
            np.sum(source_power / conductivity) / source_denominator
        )

    for axis, axis_name in ((0, "x"), (1, "y")):
        coordinate = points[:, axis]
        lower = float(np.min(coordinate))
        extent = max(float(np.max(coordinate) - lower), EPS)
        normalized = (coordinate - lower) / extent
        centroid = float(np.sum(source_power * normalized) / source_denominator)
        variance = float(
            np.sum(source_power * np.square(normalized - centroid)) / source_denominator
        )
        result[f"source_{axis_name}_centroid_normalized"] = centroid
        result[f"source_{axis_name}_spread_normalized"] = float(np.sqrt(max(variance, 0.0)))

    for axis_name, conductivity in (("x", kx), ("y", ky)):
        threshold = float(np.quantile(conductivity, 0.25))
        result[f"q_low_k{axis_name}_overlap"] = float(
            np.sum(source_power[conductivity <= threshold]) / source_denominator
        )

    if total_power <= EPS:
        for name in XY_SCALE_CONTEXT_FEATURES:
            if name.startswith("source_") or name.startswith("q_"):
                result[name] = 0.0
    if tuple(result) != XY_SCALE_CONTEXT_FEATURES:
        result = {name: float(result[name]) for name in XY_SCALE_CONTEXT_FEATURES}
    if not np.all(np.isfinite(list(result.values()))):
        raise ValueError("XY scale context produced non-finite features")
    return result


def fit_train_only_scale_context_standardizer(
    rows: Sequence[Mapping[str, float]],
    *,
    fit_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit mean/std on train rows only and retain sample IDs as provenance."""

    if not rows or len(rows) != len(fit_sample_ids):
        raise ValueError("scale context standardizer requires aligned nonempty train rows")
    matrix = np.asarray(
        [[float(row[name]) for name in XY_SCALE_CONTEXT_FEATURES] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scale context train matrix contains non-finite values")
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    safe_std = np.where(std > EPS, std, 1.0)
    return {
        "feature_names": list(XY_SCALE_CONTEXT_FEATURES),
        "mean": mean.tolist(),
        "std": safe_std.tolist(),
        "constant_feature_names": [
            name
            for name, value in zip(XY_SCALE_CONTEXT_FEATURES, std, strict=True)
            if value <= EPS
        ],
        "fit_roles": ["train"],
        "fit_population": "train_only",
        "fit_sample_ids": [str(sample_id) for sample_id in fit_sample_ids],
        "target_or_label_derived_inputs": False,
    }


def standardize_scale_contexts(
    rows: Sequence[Mapping[str, float]],
    standardizer: Mapping[str, Any],
) -> np.ndarray:
    """Encode rows using a frozen train-only standardizer."""

    validate_scale_context_schema(standardizer.get("feature_names", ()))
    matrix = np.asarray(
        [[float(row[name]) for name in XY_SCALE_CONTEXT_FEATURES] for row in rows],
        dtype=np.float64,
    )
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    encoded = (matrix - mean) / std
    if encoded.shape != (len(rows), len(XY_SCALE_CONTEXT_FEATURES)):
        raise ValueError(f"unexpected scale context shape {encoded.shape}")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("standardized scale context contains non-finite values")
    return encoded.astype(np.float32)


def regional_source_volume_weights_from_raw(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    p2r_edge_indices: np.ndarray,
    rnode_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conservative source/volume-aware latent DeepSets weights."""

    source, volume, _ = _regional_source_volume_partition(
        coords=coords,
        raw_condition=raw_condition,
        condition_feature_names=condition_feature_names,
        p2r_edge_indices=p2r_edge_indices,
        rnode_count=rnode_count,
    )
    return source, volume


def p2r_partition_of_unity_audit(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    p2r_edge_indices: np.ndarray,
    rnode_count: int,
) -> dict[str, Any]:
    """Audit degree-normalized P2R partition and regional conservation."""

    _, _, audit = _regional_source_volume_partition(
        coords=coords,
        raw_condition=raw_condition,
        condition_feature_names=condition_feature_names,
        p2r_edge_indices=p2r_edge_indices,
        rnode_count=rnode_count,
    )
    return audit


def _regional_source_volume_partition(
    *,
    coords: np.ndarray,
    raw_condition: np.ndarray,
    condition_feature_names: Sequence[str],
    p2r_edge_indices: np.ndarray,
    rnode_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split each physical node's source and volume across its valid P2R degree."""

    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(raw_condition, dtype=np.float64)
    names = tuple(str(name) for name in condition_feature_names)
    if "q" not in names:
        raise ValueError("regional DeepSets weights require raw q")
    q = np.maximum(values[:, names.index("q")], 0.0)
    volumes = np.asarray(control_volume_weights(points), dtype=np.float64)
    edges = np.asarray(p2r_edge_indices, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"p2r_edge_indices must have shape [E,2], got {edges.shape}")
    physical, regional = edges[:, 0], edges[:, 1]
    valid = (
        (physical >= 0)
        & (physical < points.shape[0])
        & (regional >= 0)
        & (regional < int(rnode_count))
    )
    if not np.any(valid):
        raise ValueError("regional DeepSets weights found no non-dummy P2R edges")
    physical = physical[valid]
    regional = regional[valid]
    degree = np.bincount(physical, minlength=points.shape[0]).astype(np.int64)
    zero_degree = np.flatnonzero(degree == 0)
    if zero_degree.size:
        raise ValueError(
            "regional DeepSets P2R partition has zero-degree physical nodes: "
            f"count={zero_degree.size} first={zero_degree[:8].tolist()}"
        )
    edge_partition = 1.0 / degree[physical].astype(np.float64)
    physical_partition_sum = np.bincount(
        physical,
        weights=edge_partition,
        minlength=points.shape[0],
    )
    source = np.zeros(int(rnode_count), dtype=np.float64)
    volume = np.zeros(int(rnode_count), dtype=np.float64)
    np.add.at(
        source,
        regional,
        q[physical] * volumes[physical] * edge_partition,
    )
    np.add.at(volume, regional, volumes[physical] * edge_partition)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(volume)):
        raise ValueError("regional DeepSets weights are non-finite")
    if np.sum(volume) <= EPS:
        raise ValueError("regional DeepSets volume weights sum to zero")
    physical_source_total = float(np.sum(q * volumes))
    physical_volume_total = float(np.sum(volumes))
    regional_source_total = float(np.sum(source))
    regional_volume_total = float(np.sum(volume))
    partition_error = float(np.max(np.abs(physical_partition_sum - 1.0)))
    source_conserved = bool(
        np.isclose(
            regional_source_total,
            physical_source_total,
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )
    volume_conserved = bool(
        np.isclose(
            regional_volume_total,
            physical_volume_total,
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )
    if partition_error > 1.0e-12 or not source_conserved or not volume_conserved:
        raise ValueError(
            "regional DeepSets P2R partition failed conservation: "
            f"partition_error={partition_error:.3e} "
            f"source_conserved={source_conserved} "
            f"volume_conserved={volume_conserved}"
        )
    return source, volume, {
        "physical_node_count": int(points.shape[0]),
        "valid_edge_count": int(physical.size),
        "zero_degree_node_count": int(zero_degree.size),
        "minimum_degree": int(np.min(degree)),
        "maximum_degree": int(np.max(degree)),
        "maximum_partition_of_unity_error": partition_error,
        "physical_source_total": physical_source_total,
        "regional_source_total": regional_source_total,
        "source_conservation_absolute_error": float(
            abs(regional_source_total - physical_source_total)
        ),
        "physical_volume_total": physical_volume_total,
        "regional_volume_total": regional_volume_total,
        "volume_conservation_absolute_error": float(
            abs(regional_volume_total - physical_volume_total)
        ),
        "source_conserved": source_conserved,
        "volume_conserved": volume_conserved,
    }
