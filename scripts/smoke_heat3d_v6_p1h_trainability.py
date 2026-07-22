#!/usr/bin/env python3
"""Run one real train-only P1h B24 update and checkpoint reload on local JAX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from unittest.mock import patch

import jax
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

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
    value["config_id"] = "V6_P1H_B24_trainability_smoke"
    return value


def _runner_args(config: dict):
    command = build_training_command(config)
    values = list(command[2:])
    wrapper_flags = {
        "--normalization-profile", "--condition-feature-transform",
        "--input-feature-schema", "--coord-policy", "--extent-feature-policy",
    }
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", ["smoke_heat3d_v6_p1h_trainability.py", *cleaned]):
        return runner.parse_args()


def _tree_max_abs(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return float("inf")
    return max(
        (float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(left_leaves, right_leaves, strict=True)),
        default=0.0,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke(config_path: Path, dataset_root: Path, manifest_path: Path) -> dict:
    config = _resolved(config_path)
    config["dataset"].update({
        "name": "heat3d_v6_p1h_shared_support1024_v0",
        "subset_path": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split_map_path": None,
    })
    config["run"].update({"batch_size": 24, "micro_batch_size": 24})
    args = _runner_args(config)
    dataset = Heat3DV6DualRobinDataset(
        dataset_root, manifest_path, include_roles={"train", "valid"}
    )
    if dataset.materialized_roles != {"train", "valid"} or len(dataset) != 896:
        raise RuntimeError("trainability smoke materialized a forbidden role")
    index_by_id = dataset.sample_index_by_id()
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in dataset.split_ids["train"]]
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
    batch_examples = [train_examples[int(raw_index)] for raw_index in order]
    groups = [runner._make_batch_group_with_seed(
        "v6_p1h_real_B24", batch_examples, stats, builder, graph_seed=0
    )]
    examples_by_id = {example.sample_id: example for example in batch_examples}
    global_lookup, context_payload = runner._prepare_global_context_lookup(
        model_config, train_examples=train_examples, required_examples=batch_examples
    )
    if global_lookup:
        runner._attach_global_context_to_groups(
            groups, global_lookup,
            expected_feature_dim=int(model_config["global_context_feature_dim"]),
        )
    native = model_config.get("native_output_mode") == "native_shape_scale"
    if native:
        runner._attach_native_physics_to_groups(groups, examples_by_id)
        if model_config.get("scale_attention_mode", "none") != "none":
            runner._attach_qk_region_features_to_groups(
                groups, examples_by_id,
                feature_version=str(model_config["qk_region_feature_version"]),
            )
    loss_config = runner._loss_config_from_args(args)
    if native:
        runner._fit_native_loss_train_references(loss_config, train_examples)
    model = RIGNO(**model_config)
    params = runner._model_init(
        model, jax.random.PRNGKey(0), groups[0], groups[0]["inputs"]
    )["params"]
    edge_key = jax.random.PRNGKey(1) if float(model_config.get("p_edge_masking", 0.0)) > 0 else None

    def objective(current_params):
        return runner._loss_components(
            model, current_params, groups, stats, loss_config, key=edge_key
        )["total_loss"]

    loss_before, gradients = jax.value_and_grad(objective)(params)
    jax.block_until_ready(loss_before)
    import optax

    optimizer = optax.adamw(
        learning_rate=float(config["optimizer"].get("lr_peak", config["optimizer"]["lr"])),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    state = optimizer.init(params)
    updates, _ = optimizer.update(gradients, state, params)
    updated = optax.apply_updates(params, updates)
    loss_after = objective(updated)
    jax.block_until_ready(loss_after)
    with tempfile.TemporaryDirectory(prefix="v6_p1h_checkpoint_smoke_") as raw_dir:
        checkpoint = Path(raw_dir) / "params_after_one_update.pkl"
        with checkpoint.open("wb") as handle:
            pickle.dump(jax.device_get(updated), handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_sha = _sha256(checkpoint)
        with checkpoint.open("rb") as handle:
            reloaded = pickle.load(handle)
        reload_parameter_error = _tree_max_abs(updated, reloaded)
        reload_loss = objective(reloaded)
        jax.block_until_ready(reload_loss)
        reload_loss_error = abs(float(reload_loss) - float(loss_after))

    leaves = [
        *jax.tree_util.tree_leaves(params), *jax.tree_util.tree_leaves(gradients),
        *jax.tree_util.tree_leaves(updated),
    ]
    finite = all(np.all(np.isfinite(np.asarray(value))) for value in leaves)
    gradient_norm = float(np.sqrt(sum(
        float(np.sum(np.square(np.asarray(value))))
        for value in jax.tree_util.tree_leaves(gradients)
    )))
    coordinate_hashes = {
        hashlib.sha256(np.ascontiguousarray(example.condition.coords).tobytes()).hexdigest()
        for example in batch_examples
    }
    report = {
        "schema_version": "heat3d_v6_p1h_trainability_smoke_v1",
        "status": "passed" if finite and reload_parameter_error == 0.0 and reload_loss_error == 0.0 else "failed",
        "dataset_id": config["dataset"]["name"],
        "baseline_config": str(config_path.relative_to(ROOT)),
        "materialized_roles": sorted(dataset.materialized_roles),
        "test_samples_materialized": 0,
        "train_sample_count": len(train_examples),
        "normalization_fit_population": "train_only",
        "normalization_fit_sample_count": len(train_examples),
        "batch_size": len(batch_examples),
        "micro_batch_size": 24,
        "node_count": 1024,
        "distinct_coordinate_hashes_in_batch": len(coordinate_hashes),
        "loss_before": float(loss_before),
        "loss_after_one_update": float(loss_after),
        "gradient_norm": gradient_norm,
        "finite_forward_backward_update": finite,
        "checkpoint_reload_parameter_max_abs_error": reload_parameter_error,
        "checkpoint_reload_loss_abs_error": reload_loss_error,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_persisted": False,
        "global_context_fit_population": context_payload.get("standardizer", {}).get("fit_population"),
        "global_context_fit_sample_count": context_payload.get("standardizer", {}).get("fit_sample_count"),
        "training_started": False,
        "model_inference_runs_on_test": 0,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/heat3d_v6/V6_02_V5best.yaml")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    args = parser.parse_args()
    report = smoke(args.config.resolve(), args.dataset.resolve(), args.manifest.resolve())
    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
