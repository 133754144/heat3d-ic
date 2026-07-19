#!/usr/bin/env python3
"""No-update Gate 6Q single-batch forward/backward and memory smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES  # noqa: E402
from rigno.heat3d_v5_scale_context import XY_SCALE_CONTEXT_FEATURES  # noqa: E402
from rigno.heat3d_v5_scale_pooling import qk_region_feature_names  # noqa: E402
from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    native_gradient_group_norms,
    native_shape_scale_losses,
    parameter_group,
)
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    return resolved


def _grid(spec: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in spec.split(","))
    if len(values) != 3 or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("grid must be three integers >=2, e.g. 8,8,16")
    return values


def _fixture(batch_size: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    axes = [np.linspace(0.0, 1.0, count, dtype=np.float32) for count in grid]
    coords = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    builder = Heat3DGraphBuilder(rmesh_levels=2, subsample_factor=2)
    metadata = builder.build_metadata(coords, key=jax.random.PRNGKey(0))
    metadata = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, repeats=batch_size, axis=0), metadata
    )
    graphs = builder.build_graphs(metadata)
    normalized = jnp.asarray(2.0 * coords - 1.0)[None, None, :, :]
    normalized = jnp.repeat(normalized, repeats=batch_size, axis=0)
    node = jnp.linspace(0.0, 1.0, coords.shape[0])[None, None, :, None]
    condition = jnp.concatenate(
        [node + 0.01 * index for index in range(8)], axis=-1
    )
    condition = jnp.repeat(condition, repeats=batch_size, axis=0)
    inputs = Inputs(
        u=jnp.zeros((batch_size, 1, coords.shape[0], 1), dtype=jnp.float32),
        c=condition,
        x_inp=normalized,
        x_out=normalized,
        t=jnp.zeros((batch_size, 1), dtype=jnp.float32),
        tau=jnp.ones((batch_size, 1), dtype=jnp.float32),
    )
    volumes = jnp.ones((batch_size, coords.shape[0]), dtype=jnp.float32)
    bottom = (coords[:, 2] == np.min(coords[:, 2])).astype(np.float32)
    mask = jnp.repeat(jnp.asarray(bottom)[None, :], repeats=batch_size, axis=0)
    reference = jnp.full((batch_size, coords.shape[0]), 300.0, dtype=jnp.float32)
    context = jnp.linspace(
        -0.2,
        0.2,
        batch_size * len(GLOBAL_CONTEXT_FEATURES),
        dtype=jnp.float32,
    ).reshape(batch_size, len(GLOBAL_CONTEXT_FEATURES))
    rnode_count = int(metadata.x_rnodes.shape[1] - 1)
    qk_width = len(qk_region_feature_names("sparse_safe_v2"))
    qk = jnp.linspace(
        -0.3,
        0.4,
        batch_size * rnode_count * qk_width,
        dtype=jnp.float32,
    ).reshape(batch_size, rnode_count, qk_width)
    scale_context = jnp.linspace(
        -0.5,
        0.5,
        batch_size * len(XY_SCALE_CONTEXT_FEATURES),
        dtype=jnp.float32,
    ).reshape(batch_size, len(XY_SCALE_CONTEXT_FEATURES))
    source_weights = jnp.linspace(
        0.0, 1.0, rnode_count, dtype=jnp.float32
    )[None, :]
    source_weights = jnp.repeat(source_weights, repeats=batch_size, axis=0)
    volume_weights = jnp.ones((batch_size, rnode_count), dtype=jnp.float32)
    target_node = jnp.sin(jnp.pi * node)
    target = (1.0 + 0.05 * jnp.arange(batch_size)[:, None, None, None]) * target_node
    target = target * (1.0 - mask[:, None, :, None])
    return {
        "inputs": inputs,
        "graphs": graphs,
        "volumes": volumes,
        "mask": mask,
        "reference": reference,
        "context": context,
        "qk": qk,
        "scale_context": scale_context,
        "source_weights": source_weights,
        "volume_weights": volume_weights,
        "target": target,
        "log_s_phys": jnp.zeros((batch_size,), dtype=jnp.float32),
        "node_count": int(coords.shape[0]),
        "rnode_count": rnode_count,
    }


def _model(config: dict[str, Any]) -> RIGNO:
    model = config["model"]
    local_names = tuple(model["decoder_bypass_local_feature_names"])
    scale_context_names = tuple(model.get("scale_context_feature_names") or ())
    return RIGNO(
        num_outputs=1,
        node_latent_size=int(model["node_latent_size"]),
        edge_latent_size=int(model["edge_latent_size"]),
        processor_steps=int(model["processor_steps"]),
        mlp_hidden_layers=int(model["mlp_hidden_layers"]),
        p_edge_masking=float(model["p_edge_masking"]),
        edge_masking_scope=str(model["edge_masking_scope"]),
        decoder_bypass_mode=str(model["decoder_bypass_mode"]),
        decoder_bypass_features=str(model["decoder_bypass_features"]),
        decoder_bypass_feature_indices=tuple(range(len(local_names))),
        decoder_bypass_feature_names=local_names,
        decoder_bypass_local_feature_names=local_names,
        decoder_bypass_num_features=len(local_names),
        decoder_bypass_output_space=str(model["decoder_bypass_output_space"]),
        decoder_bypass_hidden_size=int(model["decoder_bypass_hidden_size"]),
        decoder_bypass_layers=int(model["decoder_bypass_layers"]),
        decoder_bypass_init=str(model["decoder_bypass_init"]),
        decoder_bypass_residual_scale=float(model["decoder_bypass_residual_scale"]),
        global_context_mode=str(model["global_context_mode"]),
        global_context_feature_dim=len(model["global_context_feature_names"]),
        global_context_feature_names=tuple(model["global_context_feature_names"]),
        film_target=str(model["film_target"]),
        film_init=str(model["film_init"]),
        film_hidden_size=int(model["film_hidden_size"]),
        native_output_mode=str(model["native_output_mode"]),
        native_branch_mode=str(model["native_branch_mode"]),
        scale_head_mode=str(model["scale_head_mode"]),
        scale_pooling=str(model["scale_pooling"]),
        scale_head_hidden_size=int(model["scale_head_hidden_size"]),
        scale_head_depth=int(model.get("scale_head_depth", 1)),
        pooled_latent_stop_gradient=bool(model["pooled_latent_stop_gradient"]),
        shape_attention_mode=str(model.get("shape_attention_mode", "none")),
        scale_attention_mode=str(model.get("scale_attention_mode", "none")),
        regional_attention_hidden_size=int(
            model.get("regional_attention_hidden_size", 64)
        ),
        qk_region_feature_version=str(
            model.get("qk_region_feature_version", "bugged_v1")
        ),
        scale_context_mode=str(model.get("scale_context_mode", "none")),
        scale_context_feature_dim=len(scale_context_names),
        scale_context_feature_names=scale_context_names,
        scale_deepsets_mode=str(model.get("scale_deepsets_mode", "none")),
        scale_deepsets_hidden_size=int(
            model.get("scale_deepsets_hidden_size", 64)
        ),
    )


def run_smoke(
    config_path: Path,
    *,
    batch_size: int,
    grid: tuple[int, int, int],
) -> dict[str, Any]:
    config = _resolved(config_path)
    fixture = _fixture(batch_size, grid)
    model = _model(config)
    scale_enabled = config["model"].get("scale_context_mode", "none") != "none"
    deepsets_enabled = (
        config["model"].get("scale_deepsets_mode", "none") != "none"
    )
    kwargs = {
        "inputs": fixture["inputs"],
        "graphs": fixture["graphs"],
        "control_volumes": fixture["volumes"],
        "log_s_phys": fixture["log_s_phys"],
        "reference_temperature": fixture["reference"],
        "dirichlet_mask": fixture["mask"],
        "prescribed_temperature": fixture["reference"],
        "global_context": fixture["context"],
        "qk_region_features": fixture["qk"],
        "scale_context": fixture["scale_context"] if scale_enabled else None,
        "scale_region_source_weights": (
            fixture["source_weights"] if deepsets_enabled else None
        ),
        "scale_region_volume_weights": (
            fixture["volume_weights"] if deepsets_enabled else None
        ),
        "key": jax.random.PRNGKey(1),
        "method": model.predict_native_shape_scale,
    }
    params = model.init(jax.random.PRNGKey(0), **kwargs)["params"]
    initial_prediction = model.apply({"params": params}, **kwargs)
    prediction_checksums = {
        name: float(jnp.sum(initial_prediction[name]))
        for name in ("phi_hat", "s_hat", "deltaT_hat")
    }
    loss = config["loss"]

    def objective(current_params):
        prediction = model.apply({"params": current_params}, **kwargs)
        return native_shape_scale_losses(
            prediction,
            target_deltaT=fixture["target"],
            control_volumes=fixture["volumes"],
            dirichlet_mask=fixture["mask"],
            loss_weights={
                "shape_cv": float(loss["native_shape_cv_weight"]),
                "log_scale": float(loss["native_log_scale_weight"]),
                "relative_field": float(loss["native_relative_field_weight"]),
                "raw_absolute": float(loss["native_raw_field_weight"]),
            },
            raw_loss_mode=str(
                loss.get("native_raw_loss_mode", "per_sample_cv_mse")
            ),
            raw_train_target_energy_per_point=float(
                jnp.mean(jnp.square(fixture["target"]))
            ),
            log_scale_weight_mode=str(
                loss.get("native_log_scale_weight_mode", "uniform")
            ),
            log_scale_train_true_scale_sq_mean=1.0,
            log_scale_weight_clip=(
                float(loss.get("native_log_scale_weight_clip_min", 0.25)),
                float(loss.get("native_log_scale_weight_clip_max", 4.0)),
            ),
        )["total_loss"]

    value, gradients = jax.value_and_grad(objective)(params)
    jax.block_until_ready(value)
    gradient_norms = {
        name: float(norm)
        for name, norm in native_gradient_group_norms(gradients).items()
    }
    leaves = jax.tree_util.tree_leaves(params)
    parameter_count_by_group = {
        "backbone": 0,
        "shape_decoder": 0,
        "scale_head": 0,
    }
    deepsets_output_max_abs = None
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        parameter_count_by_group[parameter_group(path)] += int(np.asarray(leaf).size)
        path_text = "/".join(
            str(getattr(item, "key", getattr(item, "name", item)))
            for item in path
        )
        if "scale_deepsets_output" in path_text:
            output_value = float(np.max(np.abs(np.asarray(leaf))))
            deepsets_output_max_abs = max(
                output_value, deepsets_output_max_abs or 0.0
            )
    finite = bool(np.isfinite(float(value))) and all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in [*leaves, *jax.tree_util.tree_leaves(gradients)]
    )
    memory_stats = jax.devices()[0].memory_stats() or {}
    return {
        "config_id": config["config_id"],
        "batch_size": batch_size,
        "node_count": fixture["node_count"],
        "rnode_count": fixture["rnode_count"],
        "parameter_count": int(sum(np.asarray(leaf).size for leaf in leaves)),
        "parameter_count_by_trainable_group": parameter_count_by_group,
        "native_trainable_scope": config["optimizer"].get(
            "native_trainable_scope", "branch"
        ),
        "native_branch_mode": config["model"]["native_branch_mode"],
        "all_parameters_trainable": (
            config["optimizer"].get("native_trainable_scope", "branch")
            == "branch"
            and config["model"]["native_branch_mode"] == "joint"
        ),
        "deepsets_output_init_max_abs": deepsets_output_max_abs,
        "initial_prediction_checksums": prediction_checksums,
        "loss": float(value),
        "gradient_norms": gradient_norms,
        "finite_forward_backward": finite,
        "peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
        "bytes_in_use": memory_stats.get("bytes_in_use"),
        "optimizer_update_applied": False,
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--grid", type=_grid, default=(8, 8, 16))
    args = parser.parse_args()
    report = run_smoke(
        args.config.resolve(), batch_size=args.batch_size, grid=args.grid
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["finite_forward_backward"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
