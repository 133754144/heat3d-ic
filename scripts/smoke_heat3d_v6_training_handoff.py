#!/usr/bin/env python3
"""Run one real canonical V6 3xB8 -> B24 update without saving state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v1_training_semantics import build_configured_zero_delta_bridge  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def _resolved(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = resolve_inherited_yaml(payload, path)
    value["config_id"] = payload["config_id"]
    return value


def _runner_args(config: dict):
    command = build_training_command(config)
    values = list(command[2:])
    wrapper_flags = {
        "--normalization-profile",
        "--condition-feature-transform",
        "--input-feature-schema",
        "--coord-policy",
        "--extent-feature-policy",
    }
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", ["smoke_heat3d_v6_training_handoff.py", *cleaned]):
        return runner.parse_args()


def smoke(config_path: Path) -> dict:
    config = _resolved(config_path)
    args = _runner_args(config)
    dataset = Heat3DV6DualRobinDataset(
        ROOT / config["dataset"]["subset_path"],
        ROOT / config["dataset"]["manifest_path"],
    )
    index = dataset.sample_index_by_id()
    train_examples = [dataset[index[sample_id]] for sample_id in dataset.split_ids["train"]]
    runner._bridge_for = lambda example: build_configured_zero_delta_bridge(
        example,
        input_feature_schema=config["dataset"]["input_feature_schema"],
        coord_policy=config["dataset"]["coord_policy"],
        extent_feature_policy=config["dataset"]["extent_feature_policy"],
    )
    stats = training_normalization_stats(
        train_examples,
        normalization_profile=config["dataset"]["normalization_profile"],
        condition_feature_transform=config["dataset"]["condition_feature_transform"],
        input_feature_schema=config["dataset"]["input_feature_schema"],
        coord_policy=config["dataset"]["coord_policy"],
        extent_feature_policy=config["dataset"]["extent_feature_policy"],
    )
    model_config = runner._resolve_decoder_bypass_model_config(
        runner._model_config_from_args(args), stats
    )
    runner._validate_model_config(model_config)
    graph_config = runner._graph_config_from_args(args)
    builder = Heat3DGraphBuilder(**graph_config)
    order = np.random.default_rng(0).permutation(len(train_examples))[:24]
    batch_examples = [train_examples[int(index)] for index in order]
    groups = [
        runner._make_batch_group_with_seed(
            f"v6_real_B24_micro_{micro_index + 1}_B8",
            batch_examples[micro_index * 8 : (micro_index + 1) * 8],
            stats,
            builder,
            graph_seed=0,
        )
        for micro_index in range(3)
    ]
    examples_by_id = {example.sample_id: example for example in batch_examples}
    global_lookup, context_payload = runner._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=batch_examples,
    )
    if global_lookup:
        runner._attach_global_context_to_groups(
            groups,
            global_lookup,
            expected_feature_dim=int(model_config["global_context_feature_dim"]),
        )
    native = model_config.get("native_output_mode") == "native_shape_scale"
    if native:
        runner._attach_native_physics_to_groups(groups, examples_by_id)
        if model_config.get("scale_attention_mode", "none") != "none":
            runner._attach_qk_region_features_to_groups(
                groups,
                examples_by_id,
                feature_version=str(model_config["qk_region_feature_version"]),
            )
    loss_config = runner._loss_config_from_args(args)
    if native:
        runner._fit_native_loss_train_references(loss_config, train_examples)
    model = RIGNO(**model_config)
    params = runner._model_init(
        model, jax.random.PRNGKey(0), groups[0], groups[0]["inputs"]
    )["params"]

    key = jax.random.PRNGKey(1) if float(model_config.get("p_edge_masking", 0.0)) > 0 else None

    weighted_loss = jnp.asarray(0.0)
    weighted_gradients = jax.tree_util.tree_map(jnp.zeros_like, params)
    for group in groups:
        def objective(current_params):
            return runner._loss_components(
                model, current_params, [group], stats, loss_config, key=key
            )["total_loss"]

        micro_loss, micro_gradients = jax.value_and_grad(objective)(params)
        weighted_loss += micro_loss * 8
        weighted_gradients = jax.tree_util.tree_map(
            lambda total, value: total + value * 8,
            weighted_gradients,
            micro_gradients,
        )
    loss = weighted_loss / 24.0
    gradients = jax.tree_util.tree_map(
        lambda value: value / 24.0, weighted_gradients
    )
    jax.block_until_ready(loss)
    import optax

    optimizer = optax.adamw(
        learning_rate=float(config["optimizer"].get("lr_peak", config["optimizer"]["lr"])),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    state = optimizer.init(params)
    updates, _ = optimizer.update(gradients, state, params)
    updated = optax.apply_updates(params, updates)
    leaves = [
        *jax.tree_util.tree_leaves(params),
        *jax.tree_util.tree_leaves(gradients),
        *jax.tree_util.tree_leaves(updated),
    ]
    finite = bool(np.isfinite(float(loss))) and all(
        np.all(np.isfinite(np.asarray(value))) for value in leaves
    )
    gradient_norm = float(
        np.sqrt(
            sum(float(np.sum(np.square(np.asarray(value)))) for value in jax.tree_util.tree_leaves(gradients))
        )
    )
    parameter_count = int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(params)))
    memory = jax.devices()[0].memory_stats() or {}
    return {
        "config_id": config["config_id"],
        "canonical_dataset_id": config["dataset"]["name"],
        "configured_batch_size": 24,
        "micro_batch_size": 8,
        "micro_batch_count": 3,
        "realized_effective_batch_size": len(batch_examples),
        "distinct_geometry_group_count": len(
            {example.meta["group_id"] for example in batch_examples}
        ),
        "node_count": 1024,
        "loss": float(loss),
        "global_gradient_norm": gradient_norm,
        "parameter_count": parameter_count,
        "finite_forward_backward": finite,
        "adamw_update_finite": finite,
        "global_context_fit_population": context_payload.get("standardizer", {}).get("fit_population"),
        "global_context_fit_sample_count": context_payload.get("standardizer", {}).get("fit_sample_count"),
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
        "checkpoint_saved": False,
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = smoke(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["finite_forward_backward"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
