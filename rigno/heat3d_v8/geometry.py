"""Geometry primitives that do not assume a full cuboid active domain."""

from __future__ import annotations

import numpy as np


def validate_domain_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape != shape:
        raise ValueError(f"domain mask shape {values.shape} != {shape}")
    if not np.any(values):
        raise ValueError("domain mask is empty")
    return values


def masked_cells(
    *,
    coords_m: np.ndarray,
    control_volume_m3: np.ndarray,
    valid_cell_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid cells and their original flat indices for any topology."""

    coords = np.asarray(coords_m, dtype=np.float64)
    volume = np.asarray(control_volume_m3, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid_cell_mask, dtype=bool).reshape(-1)
    if coords.shape != (len(volume), 3) or mask.shape != volume.shape:
        raise ValueError("coordinates, volumes and domain mask are misaligned")
    if np.any(~np.isfinite(coords)) or np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("geometry arrays must be finite with positive volume")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("domain mask is empty")
    return coords[indices], volume[indices], indices
