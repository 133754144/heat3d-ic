"""Utilities for training and evaluating steady 3D heat RIGNO models."""

from __future__ import annotations

from pathlib import Path
import json
import pickle
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rigno.models.operator import Inputs
from rigno.utils import normalize, unnormalize


def split_dataset_indices(
  num_samples: int,
  n_train: int,
  n_valid: int,
  n_test: int,
  seed: int = 0,
) -> dict[str, np.ndarray]:
  """Creates deterministic train/valid/test splits."""

  if (n_train + n_valid + n_test) > num_samples:
    raise ValueError("Requested split is larger than dataset size.")

  rng = np.random.default_rng(seed)
  permutation = rng.permutation(num_samples)
  return {
    "train": permutation[:n_train],
    "valid": permutation[n_train:(n_train + n_valid)],
    "test": permutation[(n_train + n_valid):(n_train + n_valid + n_test)],
  }


def compute_heat3d_stats(dataset, train_indices: Sequence[int]) -> dict:
  """Computes normalization stats from the training subset only."""

  u_train, x_train, c_train = dataset.get_batch(list(train_indices))

  x_min = jnp.min(x_train[:, :1], axis=(0, 1, 2), keepdims=True)
  x_max = jnp.max(x_train[:, :1], axis=(0, 1, 2), keepdims=True)

  return {
    "u": {
      "mean": jnp.mean(u_train, axis=(0, 1, 2), keepdims=True),
      "std": jnp.std(u_train, axis=(0, 1, 2), keepdims=True),
    },
    "c": {
      "mean": jnp.mean(c_train, axis=(0, 1, 2), keepdims=True),
      "std": jnp.std(c_train, axis=(0, 1, 2), keepdims=True),
    },
    "x": {"min": x_min, "max": x_max},
    "t": {"min": None, "max": None},
    "res": {"mean": None, "std": None},
    "der": {"mean": None, "std": None},
  }


def split_coefficients(coefficients: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Splits multi-channel coefficients into operator `u` and optional `c` parts."""

  if coefficients.shape[-1] < 1:
    raise ValueError("Expected at least one coefficient channel.")
  u_inp = coefficients[..., :1]
  c_inp = coefficients[..., 1:] if coefficients.shape[-1] > 1 else None
  return u_inp, c_inp


def get_batch_inputs(dataset, batch_indices: Sequence[int]):
  """Returns target/output plus model inputs for a given batch."""

  u_tgt, x, coefficients, g = dataset.get_batch(list(batch_indices), return_graphs=True)
  u_inp, c_inp = split_coefficients(coefficients)
  return u_tgt, u_inp, c_inp, x, g


def iterate_batch_indices(
  indices: Sequence[int],
  batch_size: int,
  shuffle: bool = False,
  seed: int = 0,
) -> Iterable[np.ndarray]:
  """Yields index mini-batches."""

  indices = np.asarray(indices)
  if shuffle:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(indices)

  for start in range(0, len(indices), batch_size):
    yield indices[start:(start + batch_size)]


class HeatSteadyOutputStepper:
  """Output stepper specialized for coefficient-to-temperature steady problems."""

  def __init__(self, operator):
    self._apply_operator = operator.apply

  def normalize_inputs(self, stats, inputs: Inputs) -> Inputs:
    coeff_mean = stats["c"]["mean"]
    coeff_std = stats["c"]["std"]

    u_channels = inputs.u.shape[-1]
    u_nrm = normalize(
      inputs.u,
      shift=coeff_mean[..., :u_channels],
      scale=coeff_std[..., :u_channels],
    )

    c_nrm = None
    if inputs.c is not None:
      start = u_channels
      stop = start + inputs.c.shape[-1]
      c_nrm = normalize(
        inputs.c,
        shift=coeff_mean[..., start:stop],
        scale=coeff_std[..., start:stop],
      )

    x_inp_nrm = 2 * ((inputs.x_inp - stats["x"]["min"]) / (stats["x"]["max"] - stats["x"]["min"])) - 1
    x_out_nrm = 2 * ((inputs.x_out - stats["x"]["min"]) / (stats["x"]["max"] - stats["x"]["min"])) - 1

    return Inputs(
      u=u_nrm,
      c=c_nrm,
      x_inp=x_inp_nrm,
      x_out=x_out_nrm,
      t=None,
      tau=None,
    )

  def apply(self, variables, stats, inputs: Inputs, **kwargs):
    inputs_nrm = self.normalize_inputs(stats, inputs)
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )
    return unnormalize(
      u_prd_nrm,
      mean=stats["u"]["mean"],
      std=stats["u"]["std"],
    )

  def get_loss_inputs(self, variables, stats, u_tgt, inputs: Inputs, **kwargs):
    inputs_nrm = self.normalize_inputs(stats, inputs)
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )
    u_tgt_nrm = normalize(
      u_tgt,
      shift=stats["u"]["mean"],
      scale=stats["u"]["std"],
    )
    return u_tgt_nrm, u_prd_nrm


def _lp_norm(arr: jnp.ndarray, p: int) -> jnp.ndarray:
  pow_abs = jnp.power(jnp.abs(arr), p)
  abs_pow_sum = jnp.sum(pow_abs, axis=(1, 2, 3))
  return jnp.power(abs_pow_sum, 1.0 / p)


def _rel_lp_error_norm(u_tgt: jnp.ndarray, u_prd: jnp.ndarray, p: int) -> jnp.ndarray:
  err_norm = _lp_norm(u_prd - u_tgt, p=p)
  tgt_norm = _lp_norm(u_tgt, p=p)
  rel = err_norm / (tgt_norm + 1e-10)
  return rel


def prediction_metrics(u_tgt: jnp.ndarray, u_prd: jnp.ndarray) -> dict[str, float]:
  """Aggregates basic regression metrics on raw predictions."""

  err = u_prd - u_tgt
  mse = float(jnp.mean(jnp.square(err)))
  rmse = float(jnp.sqrt(mse))
  mae = float(jnp.mean(jnp.abs(err)))
  max_abs_error = float(jnp.max(jnp.abs(err)))
  rel_l1 = float(jnp.mean(_rel_lp_error_norm(u_tgt, u_prd, p=1)))
  rel_l2 = float(jnp.mean(_rel_lp_error_norm(u_tgt, u_prd, p=2)))

  target_range = float(jnp.max(u_tgt) - jnp.min(u_tgt))
  nrmse = (rmse / target_range) if target_range > 0 else 0.0

  sse = float(jnp.sum(jnp.square(err)))
  target_mean = jnp.mean(u_tgt)
  sst = float(jnp.sum(jnp.square(u_tgt - target_mean)))
  r2 = (1.0 - sse / sst) if sst > 0 else 0.0

  return {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "max_abs_error": max_abs_error,
    "rel_l1": rel_l1,
    "rel_l2": rel_l2,
    "nrmse": nrmse,
    "r2": r2,
  }


def per_sample_relative_l1_error(
  u_tgt: np.ndarray | jnp.ndarray,
  u_prd: np.ndarray | jnp.ndarray,
) -> np.ndarray:
  """
  Returns the author-style per-sample Relative L1 error.

  In `example.py`, the paper computes:
    rel_lp_error_mean(gtr, prd, p=1)

  For the steady 3D heat task here, each sample has shape
  [time=1, space, var=1], so that quantity reduces to:

    sum(|u_pred - u_true|) / sum(|u_true|)

  over the whole spatial field for each sample.
  """

  return np.asarray(_rel_lp_error_norm(jnp.asarray(u_tgt), jnp.asarray(u_prd), p=1))


def median_relative_l1_error(
  u_tgt: np.ndarray | jnp.ndarray,
  u_prd: np.ndarray | jnp.ndarray,
) -> float:
  """Returns the median of the author-style per-sample Relative L1 error."""

  return float(np.median(per_sample_relative_l1_error(u_tgt, u_prd)))


def per_sample_prediction_metrics(
  u_tgt: np.ndarray,
  u_prd: np.ndarray,
  x: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
  """Computes sample-wise accuracy metrics for steady 3D heat fields."""

  err = u_prd - u_tgt
  mse = np.mean(np.square(err), axis=(1, 2, 3))
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(err), axis=(1, 2, 3))
  max_abs_error = np.max(np.abs(err), axis=(1, 2, 3))

  rel_l1 = per_sample_relative_l1_error(u_tgt, u_prd)
  rel_l2 = np.asarray(_rel_lp_error_norm(jnp.asarray(u_tgt), jnp.asarray(u_prd), p=2))

  target_mean = np.mean(u_tgt, axis=(1, 2, 3), keepdims=True)
  sse = np.sum(np.square(err), axis=(1, 2, 3))
  sst = np.sum(np.square(u_tgt - target_mean), axis=(1, 2, 3))
  r2 = np.where(sst > 0, 1.0 - sse / sst, 0.0)

  mean_temp_abs_error = np.abs(
    np.mean(u_prd, axis=(1, 2, 3)) - np.mean(u_tgt, axis=(1, 2, 3))
  )
  peak_tgt = np.max(u_tgt, axis=(1, 2, 3))
  peak_prd = np.max(u_prd, axis=(1, 2, 3))
  peak_temp_abs_error = np.abs(peak_prd - peak_tgt)
  peak_temp_rel_error = peak_temp_abs_error / np.maximum(np.abs(peak_tgt), 1e-10)

  metrics = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "max_abs_error": max_abs_error,
    "rel_l1": rel_l1,
    "rel_l2": rel_l2,
    "r2": r2,
    "mean_temp_abs_error": mean_temp_abs_error,
    "peak_temp_abs_error": peak_temp_abs_error,
    "peak_temp_rel_error": peak_temp_rel_error,
  }

  if x is not None:
    tgt_peak_idx = np.argmax(u_tgt[:, 0, :, 0], axis=1)
    prd_peak_idx = np.argmax(u_prd[:, 0, :, 0], axis=1)
    hotspot_distance = np.linalg.norm(
      x[np.arange(len(x)), 0, tgt_peak_idx, :] - x[np.arange(len(x)), 0, prd_peak_idx, :],
      axis=1,
    )
    metrics["hotspot_l2_distance"] = hotspot_distance

  return metrics


def summarize_metric_array(values: np.ndarray) -> dict[str, float]:
  """Returns descriptive statistics for one sample-wise metric array."""

  return {
    "mean": float(np.mean(values)),
    "median": float(np.median(values)),
    "std": float(np.std(values)),
    "min": float(np.min(values)),
    "max": float(np.max(values)),
    "p90": float(np.percentile(values, 90)),
    "p95": float(np.percentile(values, 95)),
  }


def save_checkpoint(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "wb") as file:
    pickle.dump(payload, file)


def load_checkpoint(path: Path) -> dict:
  with open(path, "rb") as file:
    return pickle.load(file)


def save_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as file:
    json.dump(payload, file, indent=2)
