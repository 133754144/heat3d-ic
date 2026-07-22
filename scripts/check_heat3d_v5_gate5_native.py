#!/usr/bin/env python3
"""Deterministic Gate-5 native shape--scale branch and diagnostic smoke."""

from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    BRANCH_MODES,
    SCALE_HEAD_MODES,
    cv_rms,
    mask_branch_gradients,
    native_shape_scale_diagnostics,
    native_shape_scale_losses,
    normalize_shape,
    parameter_group,
    target_shape_scale,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
from rigno.models.operator import Inputs  # noqa: E402


LOSS_WEIGHTS = {"shape_cv": 1.0, "log_scale": 1.0, "relative_field": 1.0, "raw_absolute": 1.0}
LOSS_NAMES = ("shape_cv_loss", "log_scale_loss", "relative_field_loss", "raw_absolute_field_loss")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _fixture(batch: int, nodes: int = 17) -> dict:
    sample = jnp.arange(batch, dtype=jnp.float32)[:, None, None, None]
    node = jnp.linspace(0.0, 1.0, nodes, dtype=jnp.float32)[None, None, :, None]
    inputs = 0.4 + node + 0.01 * sample
    volumes = (0.8 + 0.4 * node) * jnp.ones((batch, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.zeros((batch, 1, nodes, 1), dtype=jnp.float32).at[:, :, 0, :].set(1.0)
    target = (1.0 + 0.15 * sample) * jnp.sin(jnp.pi * node)
    physics = jnp.concatenate(
        [0.2 + 0.01 * sample[:, 0, 0, 0:1], 0.5 + 0.02 * sample[:, 0, 0, 0:1], jnp.ones((batch, 1))],
        axis=1,
    )
    s_phys = 0.9 * target_shape_scale(target, volumes, dirichlet_mask=mask)[0]
    return {"inputs": inputs, "volumes": volumes, "mask": mask, "target": target, "physics": physics, "s_phys": s_phys}


def _params() -> dict:
    return {
        "encoder": {"kernel": jnp.asarray(0.85, dtype=jnp.float32)},
        "decoder": {
            "kernel": jnp.asarray(1.1, dtype=jnp.float32),
            "bias": jnp.asarray(0.03, dtype=jnp.float32),
        },
        "global_scale_hidden": {
            "kernel": jnp.asarray([0.06, -0.04, 0.03, 0.02], dtype=jnp.float32),
            "bias": jnp.asarray(0.0, dtype=jnp.float32),
        },
        "global_scale_output": {"kernel": jnp.asarray(0.7, dtype=jnp.float32)},
    }


def _forward(params: dict, batch: dict, scale_head_mode: str) -> dict:
    latent = batch["inputs"] * params["encoder"]["kernel"]
    psi = latent * params["decoder"]["kernel"] + params["decoder"]["bias"]
    _, phi = normalize_shape(psi, batch["volumes"], dirichlet_mask=batch["mask"])
    pooled = jnp.mean(latent, axis=2).reshape((latent.shape[0], 1))
    features = batch["physics"]
    if scale_head_mode == "physics_plus_pooled_latent":
        features = jnp.concatenate([features, pooled], axis=1)
    width = features.shape[1]
    hidden = jnp.tanh(
        features @ params["global_scale_hidden"]["kernel"][:width, None]
        + params["global_scale_hidden"]["bias"]
    )
    residual = hidden * params["global_scale_output"]["kernel"]
    log_s = jnp.log(batch["s_phys"].reshape((latent.shape[0], 1))) + residual
    s_hat = jnp.exp(log_s)[:, :, None, None]
    return {
        "psi": psi,
        "phi_hat": phi,
        "s_hat": s_hat,
        "log_s_hat": log_s[:, :, None, None],
        "deltaT_hat_unprojected": s_hat * phi,
        "deltaT_hat": s_hat * phi,
        "pooled_rnodes": pooled if scale_head_mode == "physics_plus_pooled_latent" else jnp.zeros((latent.shape[0], 0)),
    }


def _losses(params: dict, batch: dict, mode: str) -> dict:
    return native_shape_scale_losses(
        _forward(params, batch, mode),
        target_deltaT=batch["target"],
        control_volumes=batch["volumes"],
        dirichlet_mask=batch["mask"],
        loss_weights=LOSS_WEIGHTS,
    )


def _norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves)))


def _group_norms(gradients: dict) -> dict[str, float]:
    return {
        "backbone": _norm({"encoder": gradients["encoder"]}),
        "shape_decoder": _norm({"decoder": gradients["decoder"]}),
        "scale_head": _norm({
            "global_scale_hidden": gradients["global_scale_hidden"],
            "global_scale_output": gradients["global_scale_output"],
        }),
    }


def _path_group_norms(tree) -> dict[str, float]:
    sums = {"backbone": 0.0, "shape_decoder": 0.0, "scale_head": 0.0}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        sums[parameter_group(path)] += float(jnp.sum(jnp.square(value)))
    return {name: float(np.sqrt(value)) for name, value in sums.items()}


def _actual_rigno_smoke(
    batch_size: int, scale_head_mode: str, branch_mode: str = "joint"
) -> dict:
    axis = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    builder = Heat3DGraphBuilder(rmesh_levels=1, subsample_factor=2)
    metadata = builder.build_metadata(coords, key=jax.random.PRNGKey(11))
    metadata = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, repeats=batch_size, axis=0), metadata
    )
    graphs = builder.build_graphs(metadata)
    normalized_coords = jnp.asarray(2.0 * coords - 1.0)[None, None, :, :]
    normalized_coords = jnp.repeat(normalized_coords, repeats=batch_size, axis=0)
    node = jnp.linspace(0.0, 1.0, coords.shape[0])[None, None, :, None]
    condition = jnp.concatenate(
        [node + 0.1 * index for index in range(8)], axis=-1
    )
    condition = jnp.repeat(condition, repeats=batch_size, axis=0)
    inputs = Inputs(
        u=jnp.zeros((batch_size, 1, coords.shape[0], 1), dtype=jnp.float32),
        c=condition,
        x_inp=normalized_coords,
        x_out=normalized_coords,
        t=None,
        tau=None,
    )
    volumes = jnp.ones((batch_size, coords.shape[0]), dtype=jnp.float32)
    mask_row = (coords[:, 2] == coords[:, 2].min()).astype(np.float32)
    mask = jnp.repeat(jnp.asarray(mask_row)[None, :], repeats=batch_size, axis=0)
    context = jnp.zeros((batch_size, len(GLOBAL_CONTEXT_FEATURES)), dtype=jnp.float32)
    log_s_phys = jnp.log(jnp.linspace(1.0, 1.2, batch_size, dtype=jnp.float32))
    reference = jnp.full((batch_size, coords.shape[0]), 300.0, dtype=jnp.float32)
    prescribed = reference
    target = (1.0 - mask[:, None, :, None]) * (
        0.5 + node * jnp.linspace(1.0, 1.3, batch_size)[:, None, None, None]
    )
    model = RIGNO(
        num_outputs=1,
        node_latent_size=8,
        edge_latent_size=8,
        processor_steps=1,
        mlp_hidden_layers=1,
        concatenate_t=False,
        concatenate_tau=False,
        conditioned_normalization=False,
        p_edge_masking=0.0,
        decoder_bypass_mode="post_decoder_residual",
        decoder_bypass_features="explicit_local_condition",
        decoder_bypass_feature_indices=tuple(range(8)),
        decoder_bypass_feature_names=tuple(f"local_{index}" for index in range(8)),
        decoder_bypass_local_feature_names=tuple(f"local_{index}" for index in range(8)),
        decoder_bypass_num_features=8,
        decoder_bypass_output_space="native_psi",
        decoder_bypass_hidden_size=8,
        decoder_bypass_layers=1,
        decoder_bypass_init="zero_residual",
        global_context_mode="none",
        global_context_feature_dim=len(GLOBAL_CONTEXT_FEATURES),
        global_context_feature_names=tuple(GLOBAL_CONTEXT_FEATURES),
        native_output_mode="native_shape_scale",
        native_branch_mode=branch_mode,
        scale_head_mode=scale_head_mode,
        scale_pooling="mean",
        scale_head_hidden_size=8,
    )

    def apply(current_params):
        return model.apply(
            {"params": current_params},
            inputs=inputs,
            graphs=graphs,
            control_volumes=volumes,
            log_s_phys=log_s_phys,
            reference_temperature=reference,
            dirichlet_mask=mask,
            prescribed_temperature=prescribed,
            global_context=context,
            method=model.predict_native_shape_scale,
        )

    variables = model.init(
        jax.random.PRNGKey(17),
        inputs=inputs,
        graphs=graphs,
        control_volumes=volumes,
        log_s_phys=log_s_phys,
        reference_temperature=reference,
        dirichlet_mask=mask,
        prescribed_temperature=prescribed,
        global_context=context,
        method=model.predict_native_shape_scale,
    )
    params = variables["params"]
    prediction = jax.jit(apply)(params)
    legacy_call_output = jax.jit(
        lambda current_params: model.apply(
            {"params": current_params},
            inputs=inputs,
            graphs=graphs,
            global_context=context,
        )
    )(params)
    if legacy_call_output.shape != prediction["psi"].shape:
        raise AssertionError("refactored RIGNO call path changed output layout")
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as handle:
        handle.write(serialization.to_bytes(params))
        handle.flush()
        restored_params = serialization.from_bytes(params, Path(handle.name).read_bytes())
    restored_prediction = jax.jit(apply)(restored_params)
    reload_error = float(
        jnp.max(jnp.abs(prediction["deltaT_hat"] - restored_prediction["deltaT_hat"]))
    )
    if reload_error != 0.0:
        raise AssertionError("actual RIGNO save/reload prediction drift")

    def loss(current_params):
        return native_shape_scale_losses(
            apply(current_params),
            target_deltaT=target,
            control_volumes=volumes,
            dirichlet_mask=mask,
            loss_weights=LOSS_WEIGHTS,
        )["total_loss"]

    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    masked_gradients = mask_branch_gradients(gradients, branch_mode)
    masked_norms = _path_group_norms(masked_gradients)
    if branch_mode == "scale_only" and (
        masked_norms["backbone"] != 0.0 or masked_norms["shape_decoder"] != 0.0
    ):
        raise AssertionError("actual RIGNO scale_only gradient mask leaked")
    if branch_mode == "shape_only" and masked_norms["scale_head"] != 0.0:
        raise AssertionError("actual RIGNO shape_only gradient mask leaked")
    shape_rms = cv_rms(prediction["phi_hat"], volumes)
    field_rms = cv_rms(prediction["deltaT_hat_unprojected"], volumes)
    if not np.allclose(np.asarray(shape_rms), 1.0, atol=2e-5, rtol=0.0):
        raise AssertionError("actual RIGNO shape normalization drift")
    if not np.allclose(np.asarray(field_rms), np.asarray(prediction["s_hat"]), atol=2e-5, rtol=0.0):
        raise AssertionError("actual RIGNO field scale drift")
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)):
        raise AssertionError("actual RIGNO gradient is non-finite")
    return {
        "batch_size": batch_size,
        "scale_head_mode": scale_head_mode,
        "branch_mode": branch_mode,
        "prediction_shape": list(prediction["deltaT_hat"].shape),
        "loss": float(value),
        "finite_gradient": True,
        "positive_scale": bool(jnp.all(prediction["s_hat"] > 0.0)),
        "pooled_latent_width": int(prediction["pooled_rnodes"].shape[-1]),
        "save_reload_max_abs_error": reload_error,
        "legacy_call_shape": list(legacy_call_output.shape),
        "masked_gradient_norms": masked_norms,
    }


def main() -> int:
    args = _args()
    params = _params()
    results = {}
    gradient_report = {}
    for batch_size in (1, 28):
        batch = _fixture(batch_size)
        results[f"B{batch_size}"] = {}
        for mode in SCALE_HEAD_MODES:
            forward_jit = jax.jit(lambda current: _forward(current, batch, mode))
            prediction = forward_jit(params)
            losses = jax.jit(lambda current: _losses(current, batch, mode))(params)
            shape_rms = cv_rms(prediction["phi_hat"], batch["volumes"])
            field_rms = cv_rms(prediction["deltaT_hat_unprojected"], batch["volumes"])
            if not np.allclose(np.asarray(shape_rms), 1.0, atol=2e-6, rtol=0.0):
                raise AssertionError(f"B{batch_size}/{mode}: shape CV-RMS drift")
            if not np.allclose(np.asarray(field_rms), np.asarray(prediction["s_hat"]), atol=2e-6, rtol=0.0):
                raise AssertionError(f"B{batch_size}/{mode}: field CV-RMS != s_hat")
            if not bool(jnp.all(prediction["s_hat"] > 0.0)):
                raise AssertionError("scale must remain positive")
            if not all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves((prediction, losses))):
                raise AssertionError("forward/loss contains non-finite values")
            total_value, total_grad = jax.jit(jax.value_and_grad(lambda current: _losses(current, batch, mode)["total_loss"]))(params)
            if not np.isfinite(float(total_value)) or not all(
                bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(total_grad)
            ):
                raise AssertionError("backward contains non-finite values")
            diagnostics = native_shape_scale_diagnostics(
                prediction,
                target_deltaT=batch["target"],
                control_volumes=batch["volumes"],
                dirichlet_mask=batch["mask"],
                s_phys=batch["s_phys"],
            )
            results[f"B{batch_size}"][mode] = {
                "shape_cv_rms_min": float(jnp.min(shape_rms)),
                "shape_cv_rms_max": float(jnp.max(shape_rms)),
                "field_scale_max_abs_error": float(jnp.max(jnp.abs(field_rms - prediction["s_hat"]))),
                "positive_scale": True,
                "jit": True,
                "finite_backward": True,
                "diagnostics": {
                    "scale_log_abs_error": float(diagnostics["scale_log_abs_error"]),
                    "shape_cv_rmse": float(diagnostics["shape_cv_rmse"]),
                    "joint_relative_rmse": float(diagnostics["metrics"]["joint"]["relative_rmse"]),
                    "oracle_scale_relative_rmse": float(diagnostics["metrics"]["oracle_scale"]["relative_rmse"]),
                    "oracle_shape_relative_rmse": float(diagnostics["metrics"]["oracle_shape"]["relative_rmse"]),
                    "physics_scale_relative_rmse": float(diagnostics["metrics"]["physics_scale"]["relative_rmse"]),
                    "amplitude_ratio": float(diagnostics["metrics"]["joint"]["amplitude_ratio"]),
                    "correlation": float(diagnostics["metrics"]["joint"]["spatial_correlation"]),
                    "hotspot_rmse": float(diagnostics["metrics"]["joint"]["hotspot_rmse"]),
                    "top5_rmse": float(diagnostics["metrics"]["joint"]["topk_rmse"]),
                },
            }
        if batch_size == 28:
            mode = "physics_plus_pooled_latent"
            for loss_name in LOSS_NAMES:
                gradients = jax.grad(lambda current, key=loss_name: _losses(current, batch, mode)[key])(params)
                gradient_report[loss_name] = _group_norms(gradients)
            raw_gradients = jax.grad(lambda current: _losses(current, batch, mode)["total_loss"])(params)
            for branch_mode in BRANCH_MODES:
                masked = mask_branch_gradients(raw_gradients, branch_mode)
                norms = _group_norms(masked)
                if branch_mode == "scale_only" and (norms["backbone"] != 0.0 or norms["shape_decoder"] != 0.0):
                    raise AssertionError("scale_only leaked gradients")
                if branch_mode == "shape_only" and norms["scale_head"] != 0.0:
                    raise AssertionError("shape_only leaked gradients")
                gradient_report[f"branch_{branch_mode}"] = norms

    batch = _fixture(28)
    before = _forward(params, batch, "physics_plus_pooled_latent")
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as handle:
        handle.write(serialization.to_bytes(params))
        handle.flush()
        restored = serialization.from_bytes(params, Path(handle.name).read_bytes())
    after = _forward(restored, batch, "physics_plus_pooled_latent")
    reload_error = float(jnp.max(jnp.abs(before["deltaT_hat"] - after["deltaT_hat"])))
    if reload_error != 0.0:
        raise AssertionError("save/reload prediction drift")
    signature = set(inspect.signature(RIGNO.predict_native_shape_scale).parameters)
    forbidden = {"target", "target_deltaT", "target_shape", "target_scale", "label", "residual"}
    if signature & forbidden:
        raise AssertionError(f"target leakage in prediction signature: {signature & forbidden}")
    payload = {
        "schema_version": "heat3d_v5_gate5_native_smoke_v1",
        "status": "passed",
        "batches": results,
        "loss_gradient_norms": gradient_report,
        "branch_modes": list(BRANCH_MODES),
        "scale_head_modes": list(SCALE_HEAD_MODES),
        "save_reload_max_abs_error": reload_error,
        "target_or_label_derived_inference_inputs": False,
        "training_runs": 0,
        "formal_output_writes": 0,
        "actual_rigno": {
            "B1_joint": _actual_rigno_smoke(1, "physics_only", "joint"),
            "B1_scale_only": _actual_rigno_smoke(1, "physics_plus_pooled_latent", "scale_only"),
            "B1_shape_only": _actual_rigno_smoke(1, "physics_plus_pooled_latent", "shape_only"),
            "B28_joint": _actual_rigno_smoke(28, "physics_plus_pooled_latent", "joint"),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        resolved = args.output_json.resolve()
        if any(part in {"data", "output", "checkpoints", "logs"} for part in resolved.parts):
            raise ValueError("Gate-5 smoke output must not use formal run/output paths")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
