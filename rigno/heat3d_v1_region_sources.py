"""Region-source helpers for Heat3D v1 physics-label smoke generation.

These helpers project axis-aligned physical source boxes onto rectilinear
control volumes. They are research pipeline utilities, not a general geometry
engine and not a high-fidelity source model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cell_bounds(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return control-volume lower/upper bounds for sorted grid centers."""

    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("axis must be a non-empty 1D array")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("axis values must be strictly increasing")

    lower = np.empty_like(values, dtype=np.float64)
    upper = np.empty_like(values, dtype=np.float64)
    lower[0] = values[0]
    upper[-1] = values[-1]
    if values.size == 1:
        upper[0] = values[0] + 1.0
        return lower, upper

    mids = 0.5 * (values[:-1] + values[1:])
    upper[:-1] = mids
    lower[1:] = mids
    return lower, upper


def control_widths(axis: np.ndarray) -> np.ndarray:
    """Return rectilinear control-volume widths for sorted grid centers."""

    lower, upper = cell_bounds(axis)
    return upper - lower


def cell_volumes_from_axes(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Return flattened control-volume volumes in x-major/y-major/z-major order."""

    wx = control_widths(xs)
    wy = control_widths(ys)
    wz = control_widths(zs)
    return np.array([dx * dy * dz for dx in wx for dy in wy for dz in wz], dtype=np.float64).reshape(-1, 1)


def _overlap_lengths(axis: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = cell_bounds(axis)
    lo, hi = bounds
    if hi <= lo:
        raise ValueError(f"invalid source bounds: {bounds}")
    return np.maximum(0.0, np.minimum(upper, hi) - np.maximum(lower, lo))


def box_overlap_volumes(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    source_box: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return overlap volume between each control volume and a source box."""

    ox = _overlap_lengths(xs, source_box["x"])
    oy = _overlap_lengths(ys, source_box["y"])
    oz = _overlap_lengths(zs, source_box["z"])
    return np.array([dx * dy * dz for dx in ox for dy in oy for dz in oz], dtype=np.float64).reshape(-1, 1)


def source_box_volume(source_box: dict[str, tuple[float, float]]) -> float:
    """Return physical volume for an axis-aligned source box."""

    return float(
        (source_box["x"][1] - source_box["x"][0])
        * (source_box["y"][1] - source_box["y"][0])
        * (source_box["z"][1] - source_box["z"][0])
    )


def assign_q_field_volume_fraction(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    source_regions: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign q by volume fraction for one or more axis-aligned source boxes.

    Each source region must include:
    - ``source_box_m`` with x/y/z tuple bounds
    - ``q_density_W_m3``
    """

    cell_volumes = cell_volumes_from_axes(xs, ys, zs)
    q_field = np.zeros_like(cell_volumes, dtype=np.float64)
    source_summaries: list[dict[str, Any]] = []

    for source in source_regions:
        source_box = source["source_box_m"]
        q_density = float(source["q_density_W_m3"])
        overlap_volumes = box_overlap_volumes(xs, ys, zs, source_box)
        fractions = np.divide(
            overlap_volumes,
            cell_volumes,
            out=np.zeros_like(overlap_volumes),
            where=cell_volumes > 0.0,
        )
        q_field += q_density * fractions

        target_volume = source_box_volume(source_box)
        active_volume = float(np.sum(overlap_volumes))
        integrated_power = float(np.sum(q_density * overlap_volumes))
        target_power = q_density * target_volume
        active_count = int(np.count_nonzero(overlap_volumes[:, 0] > 0.0))
        source_missed = active_count == 0 or active_volume <= 0.0 or integrated_power <= 0.0
        source_summaries.append({
            "region_id": source.get("region_id"),
            "layer": source.get("layer"),
            "source_box_m": {
                axis: [float(source_box[axis][0]), float(source_box[axis][1])]
                for axis in ("x", "y", "z")
            },
            "q_density_W_m3": q_density,
            "source_region_volume_target": target_volume,
            "active_source_volume_discrete": active_volume,
            "integrated_q_power": integrated_power,
            "target_integrated_q_power": target_power,
            "active_source_cell_count": active_count,
            "source_volume_relative_error": float(
                abs(active_volume - target_volume) / max(abs(target_volume), 1.0e-30)
            ),
            "integrated_q_power_relative_error": float(
                abs(integrated_power - target_power) / max(abs(target_power), 1.0e-30)
            ),
            "source_missed": bool(source_missed),
        })

    total_target_volume = float(sum(item["source_region_volume_target"] for item in source_summaries))
    total_active_volume = float(sum(item["active_source_volume_discrete"] for item in source_summaries))
    total_target_power = float(sum(item["target_integrated_q_power"] for item in source_summaries))
    total_integrated_power = float(np.sum(q_field * cell_volumes))
    active_source_cell_count = int(np.count_nonzero(q_field[:, 0] > 0.0))
    summary = {
        "source_assignment": "volume_fraction",
        "q_policy": "fixed_density",
        "source_regions": source_summaries,
        "source_region_count": len(source_summaries),
        "source_region_volume_target": total_target_volume,
        "active_source_volume_discrete": total_active_volume,
        "integrated_q_power": total_integrated_power,
        "target_integrated_q_power": total_target_power,
        "active_source_cell_count": active_source_cell_count,
        "source_volume_relative_error": float(
            abs(total_active_volume - total_target_volume) / max(abs(total_target_volume), 1.0e-30)
        ),
        "integrated_q_power_relative_error": float(
            abs(total_integrated_power - total_target_power) / max(abs(total_target_power), 1.0e-30)
        ),
        "source_missed": bool(any(item["source_missed"] for item in source_summaries)),
    }
    return q_field, summary
