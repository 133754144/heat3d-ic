"""Differentiable native V5 shape--scale field semantics and losses.

The module is deliberately split from data loading: physical context is an
inference input, while target decomposition exists only in training/evaluation
loss code.  A native prediction is ``DeltaT_hat = s_hat * phi_hat`` before
raw-temperature Dirichlet projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jax import tree_util
import jax.numpy as jnp

from rigno.heat3d_v5_scale_pooling import SCALE_POOLING_MODES


EPS = 1.0e-12
LOSS_SCHEMA_VERSION = "heat3d_v5_native_shape_scale_loss_v1"
REQUIRED_LOSS_WEIGHTS = (
    "shape_cv",
    "log_scale",
    "relative_field",
    "raw_absolute",
)
BRANCH_MODES = ("scale_only", "shape_only", "joint")
SCALE_HEAD_MODES = ("physics_only", "physics_plus_pooled_latent")


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


def free_field(field: Any, dirichlet_mask: Any | None) -> jnp.ndarray:
    """Zero Dirichlet nodes before any native shape/scale decomposition."""

    values = field_layout(field, name="field")
    if dirichlet_mask is None:
        return values
    mask = field_layout(
        dirichlet_mask,
        batch_size=values.shape[0],
        node_count=values.shape[2],
        name="dirichlet_mask",
    ).astype(values.dtype)
    return (1.0 - jnp.clip(mask, 0.0, 1.0)) * values


def normalize_shape(
    psi: Any,
    control_volumes: Any,
    *,
    dirichlet_mask: Any | None = None,
    eps: float = EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mask Dirichlet nodes, then normalize the free field to unit CV-RMS."""

    values = free_field(psi, dirichlet_mask)
    scale = cv_rms(values, control_volumes, eps=eps)
    return scale, values / jnp.maximum(scale, eps)


def target_shape_scale(
    target_deltaT: Any,
    control_volumes: Any,
    *,
    dirichlet_mask: Any | None = None,
    eps: float = EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose target DeltaT only for loss/evaluation, never model input."""

    scale, shape = normalize_shape(
        target_deltaT, control_volumes, dirichlet_mask=dirichlet_mask, eps=eps
    )
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
    dirichlet_mask: Any | None = None,
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
    target_scale, target_shape = target_shape_scale(
        target, weights, dirichlet_mask=dirichlet_mask, eps=eps
    )
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


def native_shape_scale_diagnostics(
    prediction: Mapping[str, Any],
    *,
    target_deltaT: Any,
    control_volumes: Any,
    dirichlet_mask: Any | None = None,
    s_phys: Any,
    top_k: int = 5,
    hotspot_quantile: float = 0.9,
    eps: float = EPS,
) -> dict[str, Any]:
    """Return joint and oracle fields plus decomposed Gate-5 diagnostics."""

    phi_hat = field_layout(prediction["phi_hat"], name="phi_hat")
    target = field_layout(
        target_deltaT,
        batch_size=phi_hat.shape[0],
        node_count=phi_hat.shape[2],
        name="target_deltaT",
    ).astype(phi_hat.dtype)
    weights = control_volume_layout(control_volumes, phi_hat)
    s_true, phi_true = target_shape_scale(
        target, weights, dirichlet_mask=dirichlet_mask, eps=eps
    )
    s_hat = sample_scalar_layout(prediction["s_hat"], phi_hat, name="s_hat")
    s_phys_value = sample_scalar_layout(s_phys, phi_hat, name="s_phys")
    fields = {
        "joint": s_hat * phi_hat,
        "oracle_scale": s_true * phi_hat,
        "oracle_shape": s_hat * phi_true,
        "physics_scale": s_phys_value * phi_hat,
    }
    target_free = free_field(target, dirichlet_mask)
    volume_sum = jnp.sum(weights, axis=2, keepdims=True)

    def cv_mse(value):
        return jnp.sum(jnp.square(value) * weights, axis=2, keepdims=True) / jnp.maximum(
            volume_sum, eps
        )

    def field_metrics(field):
        error = field - target_free
        relative = jnp.sqrt(cv_mse(error)) / jnp.maximum(s_true, eps)
        amplitude = cv_rms(field, weights, eps=eps) / jnp.maximum(s_true, eps)
        flat_field = field.reshape((field.shape[0], -1))
        flat_target = target_free.reshape((target_free.shape[0], -1))
        flat_weights = weights.reshape((weights.shape[0], -1))
        weight_sum = jnp.sum(flat_weights, axis=1, keepdims=True)
        mean_field = jnp.sum(flat_field * flat_weights, axis=1, keepdims=True) / jnp.maximum(weight_sum, eps)
        mean_target = jnp.sum(flat_target * flat_weights, axis=1, keepdims=True) / jnp.maximum(weight_sum, eps)
        centered_field = flat_field - mean_field
        centered_target = flat_target - mean_target
        covariance = jnp.sum(centered_field * centered_target * flat_weights, axis=1)
        variance = jnp.sqrt(
            jnp.sum(jnp.square(centered_field) * flat_weights, axis=1)
            * jnp.sum(jnp.square(centered_target) * flat_weights, axis=1)
        )
        correlation = covariance / jnp.maximum(variance, eps)
        threshold = jnp.quantile(flat_target, hotspot_quantile, axis=1, keepdims=True)
        hotspot = flat_target >= threshold
        hotspot_rmse = jnp.sqrt(
            jnp.sum(jnp.square(flat_field - flat_target) * flat_weights * hotspot, axis=1)
            / jnp.maximum(jnp.sum(flat_weights * hotspot, axis=1), eps)
        )
        k = min(max(int(top_k), 1), flat_target.shape[1])
        indices = jnp.argsort(flat_target, axis=1)[:, -k:]
        top_error = jnp.take_along_axis(flat_field - flat_target, indices, axis=1)
        top_weight = jnp.take_along_axis(flat_weights, indices, axis=1)
        topk_rmse = jnp.sqrt(
            jnp.sum(jnp.square(top_error) * top_weight, axis=1)
            / jnp.maximum(jnp.sum(top_weight, axis=1), eps)
        )
        return {
            "relative_rmse": jnp.mean(relative),
            "amplitude_ratio": jnp.mean(amplitude),
            "spatial_correlation": jnp.mean(correlation),
            "hotspot_rmse": jnp.mean(hotspot_rmse),
            "topk_rmse": jnp.mean(topk_rmse),
        }

    scale_error = jnp.mean(
        jnp.abs(jnp.log(jnp.maximum(s_hat, eps)) - jnp.log(jnp.maximum(s_true, eps)))
    )
    shape_error = jnp.mean(jnp.sqrt(cv_mse(phi_hat - phi_true)))
    return {
        "fields": fields,
        "scale_log_abs_error": scale_error,
        "shape_cv_rmse": shape_error,
        "metrics": {name: field_metrics(field) for name, field in fields.items()},
        "s_true": s_true,
        "phi_true": phi_true,
    }


def parameter_group(path: Any) -> str:
    """Map a JAX tree path to Gate-5 backbone/shape-decoder/scale-head groups."""

    names = tuple(str(getattr(item, "key", getattr(item, "name", item))) for item in path)
    joined = "/".join(names)
    if any(
        token in joined
        for token in ("global_scale_", "latent_attention", "qk_attention")
    ):
        return "scale_head"
    if "decoder" in joined:
        return "shape_decoder"
    return "backbone"


def native_gradient_group_norms(gradients: Any) -> dict[str, jnp.ndarray]:
    """Return L2 norms for the Gate-5 backbone, shape, and scale branches."""

    squared = {
        "backbone": jnp.asarray(0.0),
        "shape_decoder": jnp.asarray(0.0),
        "scale_head": jnp.asarray(0.0),
    }
    for path, value in tree_util.tree_flatten_with_path(gradients)[0]:
        group = parameter_group(path)
        squared[group] = squared[group] + jnp.sum(jnp.square(value))
    return {name: jnp.sqrt(value) for name, value in squared.items()}


def apply_scale_head_lr_multiplier(updates: Any, multiplier: float) -> Any:
    """Scale only native scale-head optimizer updates.

    The default multiplier is exactly one and returns the input tree without a
    transformation, preserving the established N3/V13 optimizer path.  This
    acts after Optax (or manual-GD) produces updates, so its semantics are an
    explicit scale-head learning-rate multiplier rather than a loss reweight.
    """

    value = float(multiplier)
    if value <= 0.0:
        raise ShapeScaleError("scale_head_lr_multiplier must be > 0")
    if value == 1.0:
        return updates
    path_leaves, treedef = tree_util.tree_flatten_with_path(updates)
    leaves = [
        leaf * value if parameter_group(path) == "scale_head" else leaf
        for path, leaf in path_leaves
    ]
    return tree_util.tree_unflatten(treedef, leaves)


def mask_branch_gradients(gradients: Any, branch_mode: str) -> Any:
    """Zero gradients outside the selected native branch training contract."""

    if branch_mode not in BRANCH_MODES:
        raise ShapeScaleError(f"unsupported branch mode {branch_mode!r}")
    allowed = {
        "scale_only": {"scale_head"},
        "shape_only": {"backbone", "shape_decoder"},
        "joint": {"backbone", "shape_decoder", "scale_head"},
    }[branch_mode]
    return tree_util.tree_map_with_path(
        lambda path, value: value if parameter_group(path) in allowed else jnp.zeros_like(value),
        gradients,
    )


def _validate_loss_weights(loss_weights: Mapping[str, float]) -> None:
    missing = [name for name in REQUIRED_LOSS_WEIGHTS if name not in loss_weights]
    if missing:
        raise ShapeScaleError(f"native loss weights missing {missing}")
    for name in REQUIRED_LOSS_WEIGHTS:
        value = float(loss_weights[name])
        if value < 0.0:
            raise ShapeScaleError(f"native loss weight {name} must be >= 0")
