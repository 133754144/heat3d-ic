"""Small diagnostic metrics for Heat3D v1 validation smoke.

These functions are intentionally simple array metrics. They do not implement
physics-aware residuals or boundary/interface consistency checks.
"""

from __future__ import annotations

import numpy as np


def _as_1d(array) -> np.ndarray:
  values = np.asarray(array, dtype=np.float64)
  return values.reshape(-1)


def mse(pred, target) -> float:
  pred_values = _as_1d(pred)
  target_values = _as_1d(target)
  _require_same_shape(pred_values, target_values)
  return float(np.mean(np.square(pred_values - target_values)))


def rmse(pred, target) -> float:
  return float(np.sqrt(mse(pred, target)))


def mae(pred, target) -> float:
  pred_values = _as_1d(pred)
  target_values = _as_1d(target)
  _require_same_shape(pred_values, target_values)
  return float(np.mean(np.abs(pred_values - target_values)))


def max_abs_error(pred, target) -> float:
  pred_values = _as_1d(pred)
  target_values = _as_1d(target)
  _require_same_shape(pred_values, target_values)
  return float(np.max(np.abs(pred_values - target_values)))


def hotspot_index(temperature) -> int:
  values = _as_1d(temperature)
  if values.size == 0:
    raise ValueError("temperature must contain at least one value")
  return int(np.argmax(values))


def peak_T_abs_error(pred_temperature, target_temperature) -> float:
  pred_values = _as_1d(pred_temperature)
  target_values = _as_1d(target_temperature)
  if pred_values.size == 0 or target_values.size == 0:
    raise ValueError("temperature arrays must contain at least one value")
  return float(abs(np.max(pred_values) - np.max(target_values)))


def hotspot_coord_distance(pred_temperature, target_temperature, coords) -> float:
  coords_array = np.asarray(coords, dtype=np.float64)
  if coords_array.ndim != 2 or coords_array.shape[1] != 3:
    raise ValueError(f"coords must have shape (N,3), found {coords_array.shape}")
  pred_index = hotspot_index(pred_temperature)
  target_index = hotspot_index(target_temperature)
  if pred_index >= coords_array.shape[0] or target_index >= coords_array.shape[0]:
    raise ValueError("hotspot index exceeds coords length")
  return float(np.linalg.norm(coords_array[pred_index] - coords_array[target_index]))


def top_k_hotspot_overlap(pred_temperature, target_temperature, k: int = 5) -> float:
  if k < 1:
    raise ValueError("k must be >= 1")
  pred_values = _as_1d(pred_temperature)
  target_values = _as_1d(target_temperature)
  _require_same_shape(pred_values, target_values)
  if pred_values.size == 0:
    raise ValueError("temperature arrays must contain at least one value")
  actual_k = min(k, pred_values.size)
  pred_top = set(np.argpartition(pred_values, -actual_k)[-actual_k:].tolist())
  target_top = set(np.argpartition(target_values, -actual_k)[-actual_k:].tolist())
  return float(len(pred_top & target_top) / actual_k)


def _require_same_shape(pred: np.ndarray, target: np.ndarray) -> None:
  if pred.shape != target.shape:
    raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}")
