"""Differentiable native V5 shape--scale field semantics and losses.

The module is deliberately split from data loading: physical context is an
inference input, while target decomposition exists only in training/evaluation
loss code.  A native prediction is ``DeltaT_hat = s_hat * phi_hat`` before
raw-temperature Dirichlet projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


EPS = 1.0e-12
LOSS_SCHEMA_VERSION = "heat3d_v5_native_shape_scale_loss_v1"
REQUIRED_LOSS_WEIGHTS = (
    "shape_cv",
    "log_scale",
    "relative_field",
    "raw_absolute",
)


class ShapeScaleError(ValueError):
    """Raised when native shape--scale tensors do not share a field layout."""


def field_layout(value: Any, *, batch_size: int | None = None, node_count: int | None = None, name: str) -> jnp.ndarray:
    """Coerce a field to the model layout ``[B,1,N,1]`` without target inference."""

    array = jnp.asarray(value)
    if array.ndim == 4 and array.shape[1] == 1 and array.shape[3] == 1:
        return array
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, :, :, None]
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, None, :, :]
    if array.ndim == 2:
        return array[:, None, :, None]
    if array.ndim == 1:
        if batch_size is None:
            return array[None, None, :, None]
        if node_count is None or array.shape[0] != node_count:
            raise ShapeScaleError(f"{name} rank-1 input must have N={node_count}, found {array.shape}")
        return jnp.broadcast_to(array[None, None, :, None], (batch_size, 1, node_count, 1))
    raise ShapeScaleError(f"{name} must have shape [B,1,N,1], [B,N], or [N], found {array.shape}")


def sample_scalar_layout(value: Any, prediction: jnp.ndarray, *, name: str) -> jnp.ndarray:
    """Coerce one scalar per sample to ``[B,1,1,1]``."""

    array = jnp.asarray(value, dtype=prediction.dtype)
    if array.ndim == 1 and array.shape[0] == prediction.shape[0]:
        return array[:, None, None, None]
    if array.ndim == 2 and array.shape == (prediction.shape[0], 1):
        return array[:, :, None, None]
    if array.ndim == 4 and array.shape == (prediction.shape[0], 1, 1, 1):
        return array
    raise ShapeScaleError(f"{name} must be one scalar per sample, found {array.shape}")


def control_volume_layout(control_volumes: Any, prediction: jnp.ndarray) -> jnp.ndarray:
    """Coerce physical CV weights to the prediction field layout."""

    weights = field_layout(
        control_volumes,
        batch_size=prediction.shape[0],
        node_count=prediction.shape[2],
        name="control_volumes",
    ).astype(prediction.dtype)
    if weights.shape != prediction.shape:
        raise ShapeScaleError(
            f"control_volumes shape {weights.shape} must match prediction {prediction.shape}"
        )
    return weights


def cv_rms(field: Any, control_volumes: Any, *, eps: float = EPS) -> jnp.ndarray:
    """Return per-sample CV-RMS with output shape ``[B,1,1,1]``."""

    values = field_layout(field, name="field")
    weights = control_volume_layout(control_volumes, values)
    denominator = jnp.sum(weights, axis=2, keepdims=True)
    return jnp.sqrt(
        jnp.sum(jnp.square(values) * weights, axis=2, keepdims=True)
        / jnp.maximum(denominator, eps)
    )


def normalize_shape(psi: Any, control_volumes: Any, *, eps: float = EPS) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize an unnormalized decoder field to positive unit CV-RMS shape."""

    values = field_layout(psi, name="psi")
    scale = cv_rms(values, control_volumes, eps=eps)
    return scale, values / jnp.maximum(scale, eps)


def target_shape_scale(target_deltaT: Any, control_volumes: Any, *, eps: float = EPS) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose target DeltaT only for loss/evaluation, never model input."""

    scale, shape = normalize_shape(target_deltaT, control_volumes, eps=eps)
    return scale, shape


def reconstruct_shape_scale(scale: Any, shape: Any) -> jnp.ndarray:
    """Reconstruct a field from a positive per-sample scale and shape."""

    values = field_layout(shape, name="shape")
    return sample_scalar_layout(scale, values, name="scale") * values


def project_raw_dirichlet(
    raw_temperature: Any,
    dirichlet_mask: Any,
    prescribed_temperature: Any,
) -> jnp.ndarray:
    """Project known Dirichlet temperature nodes after raw reconstruction."""

    raw = field_layout(raw_temperature, name="raw_temperature")
    mask = field_layout(
        dirichlet_mask,
        batch_size=raw.shape[0],
        node_count=raw.shape[2],
        name="dirichlet_mask",
    ) > 0.5
    prescribed = field_layout(
        prescribed_temperature,
        batch_size=raw.shape[0],
        node_count=raw.shape[2],
        name="prescribed_temperature",
    ).astype(raw.dtype)
    return jnp.where(mask, prescribed, raw)


def native_shape_scale_losses(
    prediction: Mapping[str, Any],
    *,
    target_deltaT: Any,
    control_volumes: Any,
    loss_weights: Mapping[str, float],
    eps: float = EPS,
) -> dict[str, jnp.ndarray]:
    """Compute the four required V5 native losses as per-sample CV losses.

    ``prediction`` must come from ``RIGNO.predict_native_shape_scale``. Target
    shape/scale is constructed solely here, after model inference.
    """

    _validate_loss_weights(loss_weights)
    if "phi_hat" not in prediction or "s_hat" not in prediction or "deltaT_hat" not in prediction:
        raise ShapeScaleError("native prediction must contain phi_hat, s_hat, and deltaT_hat")
    phi_hat = field_layout(prediction["phi_hat"], name="phi_hat")
    target = field_layout(
        target_deltaT,
        batch_size=phi_hat.shape[0],
        node_count=phi_hat.shape[2],
        name="target_deltaT",
    ).astype(phi_hat.dtype)
    weights = control_volume_layout(control_volumes, phi_hat)
    target_scale, target_shape = target_shape_scale(target, weights, eps=eps)
    s_hat = sample_scalar_layout(prediction["s_hat"], phi_hat, name="s_hat")
    delta_hat = field_layout(
        prediction["deltaT_hat"],
        batch_size=phi_hat.shape[0],
        node_count=phi_hat.shape[2],
        name="deltaT_hat",
    )
    volume_sum = jnp.sum(weights, axis=2, keepdims=True)
    shape_cv_mse = jnp.sum(jnp.square(phi_hat - target_shape) * weights, axis=2, keepdims=True) / jnp.maximum(volume_sum, eps)
    raw_cv_mse = jnp.sum(jnp.square(delta_hat - target) * weights, axis=2, keepdims=True) / jnp.maximum(volume_sum, eps)
    relative_field_mse = raw_cv_mse / jnp.maximum(jnp.square(target_scale), eps)
    log_scale_mse = jnp.square(jnp.log(jnp.maximum(s_hat, eps)) - jnp.log(jnp.maximum(target_scale, eps)))
    components = {
        "shape_cv_loss": jnp.mean(shape_cv_mse),
        "log_scale_loss": jnp.mean(log_scale_mse),
        "relative_field_loss": jnp.mean(relative_field_mse),
        "raw_absolute_field_loss": jnp.mean(raw_cv_mse),
    }
    total = (
        float(loss_weights["shape_cv"]) * components["shape_cv_loss"]
        + float(loss_weights["log_scale"]) * components["log_scale_loss"]
        + float(loss_weights["relative_field"]) * components["relative_field_loss"]
        + float(loss_weights["raw_absolute"]) * components["raw_absolute_field_loss"]
    )
    return {
        **components,
        "total_loss": total,
        "target_scale": target_scale,
        "target_shape": target_shape,
        "s_hat_positive": jnp.all(s_hat > 0.0),
    }


def _validate_loss_weights(loss_weights: Mapping[str, float]) -> None:
    missing = [name for name in REQUIRED_LOSS_WEIGHTS if name not in loss_weights]
    if missing:
        raise ShapeScaleError(f"native loss weights missing {missing}")
    for name in REQUIRED_LOSS_WEIGHTS:
        value = float(loss_weights[name])
        if value < 0.0:
            raise ShapeScaleError(f"native loss weight {name} must be >= 0")
