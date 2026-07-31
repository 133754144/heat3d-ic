#!/usr/bin/env python3
"""One-update P1i varying-support smoke with explicit holdout closure."""

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
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
from rigno.models.rigno import RIGNO  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def _resolved(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = resolve_inherited_yaml(payload, path)
    value["config_id"] = payload["config_id"]
    return value


def _runner_args(config: dict):
    values = list(build_training_command(config)[2:])
    wrapper_flags = {"--normalization-profile", "--condition-feature-transform",
                     "--input-feature-schema", "--coord-policy", "--extent-feature-policy"}
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", ["smoke_heat3d_v6_p1i_training.py", *cleaned]):
        return runner.parse_args()


def _tree_max_abs(left, right) -> float:
    return max((float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(jax.tree_util.tree_leaves(left),
                                jax.tree_util.tree_leaves(right), strict=True)), default=0.0)


def _tree_hash(value) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def smoke(config_path: Path, dataset_root: Path, manifest_path: Path, batch_size: int) -> dict:
    config = _resolved(config_path)
    args = _runner_args(config)
    dataset = Heat3DV6DualRobinDataset(
        dataset_root, manifest_path, include_roles={"train", "valid_iid"}
    )
    if dataset.manifest.get("dataset_id") != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise RuntimeError("smoke received the wrong dataset family")
    if dataset.materialized_roles != {"train", "valid_iid"} or len(dataset) != 896:
        raise RuntimeError("smoke materialized a forbidden holdout role")
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
    builder = runner.RunSharedSupportGraphBuilder(
        Heat3DGraphBuilder(**runner._graph_config_from_args(args))
    )
    order = np.random.default_rng(int(args.batch_build_seed)).permutation(len(train_examples))
    examples = [train_examples[int(raw)] for raw in order[:batch_size]]
    group = runner._make_batch_group_with_seed(
        f"p1i_varying_support_B{batch_size}", examples, stats, builder,
        graph_seed=int(args.graph_seed),
    )
    groups = [group]
    by_id = {example.sample_id: example for example in examples}
    context_lookup, context_payload = runner._prepare_global_context_lookup(
        model_config, train_examples=train_examples, required_examples=examples
    )
    runner._attach_global_context_to_groups(
        groups, context_lookup,
        expected_feature_dim=int(model_config["global_context_feature_dim"]),
    )
    runner._attach_native_physics_to_groups(groups, by_id)
    if model_config.get("scale_attention_mode", "none") != "none":
        runner._attach_qk_region_features_to_groups(
            groups, by_id, feature_version=str(model_config["qk_region_feature_version"])
        )
    loss_config = runner._loss_config_from_args(args)
    runner._fit_native_loss_train_references(loss_config, train_examples)
    model = RIGNO(**model_config)
    params = runner._model_init(model, jax.random.PRNGKey(0), group, group["inputs"])["params"]
    edge_key = jax.random.PRNGKey(1)

    def objective(current):
        return runner._loss_components(model, current, groups, stats, loss_config, key=edge_key)["total_loss"]

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
    with tempfile.TemporaryDirectory(prefix="p1i-checkpoint-smoke-") as raw:
        checkpoint = Path(raw) / "params.pkl"
        with checkpoint.open("wb") as handle:
            pickle.dump(jax.device_get(updated), handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with checkpoint.open("rb") as handle:
            reloaded = pickle.load(handle)
        reload_parameter_error = _tree_max_abs(updated, reloaded)
        reload_loss_error = abs(float(objective(reloaded)) - float(loss_after))

    finite = all(np.all(np.isfinite(np.asarray(leaf))) for leaf in [
        *jax.tree_util.tree_leaves(params), *jax.tree_util.tree_leaves(gradients),
        *jax.tree_util.tree_leaves(updated)])
    coord_hashes = [hashlib.sha256(np.ascontiguousarray(e.condition.coords).tobytes()).hexdigest()
                    for e in examples]
    wrong_reuse_rejected = (
        len(set(coord_hashes)) == batch_size
        and int(builder.audit["varying_support_fallback_calls"]) == batch_size - 1
        and bool(group["shared_metadata"]) is False
    )
    memory = jax.devices()[0].memory_stats() or {}
    standardizer = context_payload.get("standardizer", {})
    # Parameter bytes are the strict checkpoint invariant.  A replayed GPU
    # reduction can differ by a few float32 ulps even with identical parameters,
    # so use an explicit relative numerical tolerance for the scalar loss.
    reload_loss_tolerance = max(1.0e-5, 1.0e-5 * abs(float(loss_after)))
    passed = finite and reload_parameter_error == 0.0 and reload_loss_error <= reload_loss_tolerance and wrong_reuse_rejected
    return {
        "schema_version": "heat3d_v6_p1i_training_smoke_v1",
        "status": "passed" if passed else "failed",
        "config_id": config["config_id"], "dataset_id": dataset.manifest["dataset_id"],
        "archive_source": str(dataset_root), "materialized_roles": sorted(dataset.materialized_roles),
        "test_samples_materialized": 0, "sealed_samples_materialized": 0,
        "train_sample_count": len(train_examples), "batch_size": batch_size,
        "batch_sample_ids": [example.sample_id for example in examples],
        "batch_order_sha256": hashlib.sha256(json.dumps([e.sample_id for e in examples]).encode()).hexdigest(),
        "node_count": 1024, "distinct_coordinate_hashes": len(set(coord_hashes)),
        "graph_metadata_sha256": _tree_hash(group["metadata"]),
        "graph_builder_audit": builder.audit, "wrong_shared_graph_reuse_rejected": wrong_reuse_rejected,
        "loss_before": float(loss_before), "loss_after_one_update": float(loss_after),
        "finite_forward_backward_update": finite,
        "checkpoint_reload_parameter_max_abs_error": reload_parameter_error,
        "checkpoint_reload_loss_abs_error": reload_loss_error,
        "checkpoint_reload_loss_tolerance": reload_loss_tolerance,
        "ephemeral_checkpoint_sha256": checkpoint_sha,
        "global_context_fit_population": standardizer.get("fit_population"),
        "global_context_fit_sample_count": standardizer.get("fit_sample_count"),
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
        "peak_pool_bytes": memory.get("peak_pool_bytes"),
        "training_started": False, "test_or_sealed_inference_runs": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.config.resolve(), args.dataset_root.resolve(), args.manifest.resolve(), args.batch_size)
    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
