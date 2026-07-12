#!/usr/bin/env python3
"""Deterministic native V5 shape--scale decomposition/loss fixture."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    native_shape_scale_losses,
    normalize_shape,
    project_raw_dirichlet,
    reconstruct_shape_scale,
    target_shape_scale,
)
from rigno.models.rigno import RIGNO  # noqa: E402


LOSS_WEIGHTS = {
    "shape_cv": 1.0,
    "log_scale": 1.0,
    "relative_field": 1.0,
    "raw_absolute": 1.0,
}


def main() -> int:
    target = jnp.asarray(
        [
            [[[1.0], [2.0], [3.0], [4.0]]],
            [[[2.0], [1.0], [4.0], [3.0]]],
        ],
        dtype=jnp.float32,
    )
    volumes = jnp.asarray(
        [
            [[1.0, 2.0, 1.0, 3.0]],
            [[2.0, 1.0, 3.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    target_scale, target_shape = target_shape_scale(target, volumes)
    reconstructed = reconstruct_shape_scale(target_scale, target_shape)
    recon_error = float(jnp.max(jnp.abs(reconstructed - target)))
    if recon_error > 1.0e-6:
        raise AssertionError(f"target decomposition/reconstruction drift={recon_error}")
    shape_rms = jnp.sqrt(
        jnp.sum(jnp.square(target_shape) * volumes[:, :, :, None], axis=2)
        / jnp.sum(volumes[:, :, :, None], axis=2)
    )
    if not np.allclose(np.asarray(shape_rms), 1.0, rtol=0.0, atol=1.0e-6):
        raise AssertionError("target shape is not unit CV-RMS")

    psi = target * 3.7
    psi_scale, phi_hat = normalize_shape(psi, volumes)
    if not np.allclose(np.asarray(phi_hat), np.asarray(target_shape), rtol=0.0, atol=1.0e-6):
        raise AssertionError("native psi normalization changed target shape")
    prediction = {
        "psi": psi,
        "phi_hat": phi_hat,
        "s_hat": target_scale,
        "deltaT_hat": target,
    }
    losses = native_shape_scale_losses(
        prediction,
        target_deltaT=target,
        control_volumes=volumes,
        loss_weights=LOSS_WEIGHTS,
    )
    if float(losses["total_loss"]) > 1.0e-10 or not bool(losses["s_hat_positive"]):
        raise AssertionError("perfect native reconstruction did not yield finite zero losses")

    reference = jnp.full_like(target, 300.0)
    prescribed = jnp.full_like(target, 301.25)
    bottom_mask = jnp.asarray(
        [
            [[[1.0], [0.0], [0.0], [0.0]]],
            [[[0.0], [1.0], [0.0], [0.0]]],
        ],
        dtype=jnp.float32,
    )
    projected = project_raw_dirichlet(reference + target, bottom_mask, prescribed)
    if not np.allclose(
        np.asarray(projected)[np.asarray(bottom_mask, dtype=bool)],
        np.asarray(prescribed)[np.asarray(bottom_mask, dtype=bool)],
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("raw Dirichlet projection failed")

    def _loss_for_grad(raw_psi):
        _, shape = normalize_shape(raw_psi, volumes)
        value = native_shape_scale_losses(
            {"phi_hat": shape, "s_hat": target_scale, "deltaT_hat": target},
            target_deltaT=target,
            control_volumes=volumes,
            loss_weights=LOSS_WEIGHTS,
        )["total_loss"]
        return value

    grad = jax.grad(_loss_for_grad)(psi)
    if not np.all(np.isfinite(np.asarray(grad))):
        raise AssertionError("native shape-scale gradient is non-finite")
    signature = set(inspect.signature(RIGNO.predict_native_shape_scale).parameters)
    forbidden = {"target", "target_deltaT", "target_shape", "target_scale", "label", "residual"}
    if signature.intersection(forbidden):
        raise AssertionError("native model prediction API exposes a target-derived input")

    print(
        json.dumps(
            {
                "status": "passed",
                "target_reconstruction_max_abs_error": recon_error,
                "shape_cv_rms_min": float(jnp.min(shape_rms)),
                "shape_cv_rms_max": float(jnp.max(shape_rms)),
                "psi_cv_rms_mean": float(jnp.mean(psi_scale)),
                "gradient_finite": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
