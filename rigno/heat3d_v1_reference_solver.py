"""Minimal reference steady solver for v1 smoke samples only.

This module is intentionally narrow:
- regular layered rectangular stacks only
- top Robin / bottom Dirichlet / sides adiabatic
- perfect-contact interfaces handled approximately through merged duplicate nodes
- k_field supports (N,1) isotropic and (N,3) diagonal anisotropic

It is not a general-purpose solver and must not be described as a formal
high-fidelity production data generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_sample(sample_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
  sample_path = Path(sample_dir)
  coords = np.load(sample_path / "coords.npy")
  k_field = np.load(sample_path / "k_field.npy")
  q_field = np.load(sample_path / "q_field.npy")
  meta = json.loads((sample_path / "sample_meta.json").read_text())
  return coords, k_field, q_field, meta


def _validate_supported_problem(meta: dict[str, Any], k_field: np.ndarray) -> None:
  boundary_types = meta.get("boundary_types", {})
  if boundary_types != {"top": "Robin", "bottom": "Dirichlet", "sides": "adiabatic"}:
    raise ValueError(
      "Minimal reference solver supports only top Robin / bottom Dirichlet / sides adiabatic"
    )

  interfaces = meta.get("interfaces", [])
  if any(interface.get("type") != "perfect_contact" for interface in interfaces):
    raise ValueError("Minimal reference solver supports only perfect_contact interfaces")

  if k_field.ndim != 2 or k_field.shape[1] not in (1, 3):
    raise ValueError(
      f"Minimal reference solver supports only k_field shapes (N,1) and (N,3), found {k_field.shape}"
    )


def _merge_duplicate_points(
  coords: np.ndarray,
  k_field: np.ndarray,
  q_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
  n_unique = unique_coords.shape[0]

  k_acc = np.zeros((n_unique, k_field.shape[1]), dtype=np.float64)
  q_acc = np.zeros((n_unique, 1), dtype=np.float64)
  counts = np.zeros((n_unique, 1), dtype=np.float64)

  for idx_unique, idx_original in enumerate(inverse):
    k_acc[idx_original] += k_field[idx_unique]
    q_acc[idx_original, 0] = max(q_acc[idx_original, 0], q_field[idx_unique, 0])
    counts[idx_original, 0] += 1.0

  merged_k = k_acc / counts
  merged_q = q_acc
  return unique_coords, inverse, merged_k, merged_q


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


def _harmonic_mean(a: float, b: float) -> float:
  if a <= 0 or b <= 0:
    return max(a, b)
  return 2.0 * a * b / (a + b)


def _grid_mapping(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
          raise ValueError("Merged coordinates do not form a complete rectilinear grid")
        grid[ix, iy, iz] = lookup[key]
  return xs, ys, zs, grid


def solve_reference_temperature(sample_dir: str | Path) -> np.ndarray:
  """Returns a temperature field on the original sample nodes."""

  coords, k_field, q_field, meta = _load_sample(sample_dir)
  _validate_supported_problem(meta, k_field)

  if k_field.shape[1] == 1:
    k_field = np.repeat(k_field, repeats=3, axis=1)

  unique_coords, inverse, merged_k, merged_q = _merge_duplicate_points(coords, k_field, q_field)
  xs, ys, zs, grid = _grid_mapping(unique_coords)
  dx_cv = _control_widths(xs)
  dy_cv = _control_widths(ys)
  dz_cv = _control_widths(zs)

  n = unique_coords.shape[0]
  a = np.zeros((n, n), dtype=np.float64)
  b = np.zeros((n,), dtype=np.float64)

  boundary_params = meta["boundary_params"]
  h_top = float(boundary_params["top"]["h_W_m2K"])
  t_inf = float(boundary_params["top"]["ambient_temperature_K"])
  t_bottom = float(boundary_params["bottom"]["fixed_temperature_K"])

  def conductance(idx_i: int, idx_j: int, axis: int, area: float, distance: float) -> float:
    k_i = float(merged_k[idx_i, axis])
    k_j = float(merged_k[idx_j, axis])
    return _harmonic_mean(k_i, k_j) * area / distance

  for ix in range(xs.size):
    for iy in range(ys.size):
      for iz in range(zs.size):
        idx = int(grid[ix, iy, iz])

        if iz == 0:
          a[idx, idx] = 1.0
          b[idx] = t_bottom
          continue

        volume = dx_cv[ix] * dy_cv[iy] * dz_cv[iz]
        rhs = float(merged_q[idx, 0]) * volume
        diag = 0.0

        if iz == zs.size - 1:
          area_top = dx_cv[ix] * dy_cv[iy]
          diag += h_top * area_top
          rhs += h_top * area_top * t_inf

        if ix > 0:
          left = int(grid[ix - 1, iy, iz])
          g = conductance(idx, left, axis=0, area=dy_cv[iy] * dz_cv[iz], distance=xs[ix] - xs[ix - 1])
          diag += g
          a[idx, left] -= g
        if ix < xs.size - 1:
          right = int(grid[ix + 1, iy, iz])
          g = conductance(idx, right, axis=0, area=dy_cv[iy] * dz_cv[iz], distance=xs[ix + 1] - xs[ix])
          diag += g
          a[idx, right] -= g

        if iy > 0:
          back = int(grid[ix, iy - 1, iz])
          g = conductance(idx, back, axis=1, area=dx_cv[ix] * dz_cv[iz], distance=ys[iy] - ys[iy - 1])
          diag += g
          a[idx, back] -= g
        if iy < ys.size - 1:
          front = int(grid[ix, iy + 1, iz])
          g = conductance(idx, front, axis=1, area=dx_cv[ix] * dz_cv[iz], distance=ys[iy + 1] - ys[iy])
          diag += g
          a[idx, front] -= g

        if iz > 0:
          below = int(grid[ix, iy, iz - 1])
          g = conductance(idx, below, axis=2, area=dx_cv[ix] * dy_cv[iy], distance=zs[iz] - zs[iz - 1])
          diag += g
          a[idx, below] -= g
        if iz < zs.size - 1:
          above = int(grid[ix, iy, iz + 1])
          g = conductance(idx, above, axis=2, area=dx_cv[ix] * dy_cv[iy], distance=zs[iz + 1] - zs[iz])
          diag += g
          a[idx, above] -= g

        a[idx, idx] = diag
        b[idx] = rhs

  temperature_unique = np.linalg.solve(a, b)
  temperature_full = temperature_unique[inverse].reshape(-1, 1)
  return temperature_full.astype(np.float64)
