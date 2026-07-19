#!/usr/bin/env python3
"""Run one real train-only B28 forward/backward/AdamW update without artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import jax
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    Heat3DV1NativeSupervisedDataset,
)
from rigno.heat3d_v1_supervised import (  # noqa: E402
    PHYSICS_LABEL_SUPERVISED_STAGES,
    Heat3DV1SupervisedDataset,
)
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_scale_context import (  # noqa: E402
    p2r_partition_of_unity_audit,
)
from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    native_gradient_group_norms,
    parameter_group,
)
from scripts import run_heat3d_v4_controlled_training as v4_wrapper  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


runner = v4_wrapper.legacy_runner


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    return resolved


def _args_from_config(config: dict[str, Any]) -> argparse.Namespace:
    command = build_training_command(config, python_executable="python")
    argv = [command[1], *command[2:]]
    profile = v4_wrapper._pop_v4_profile_args(
        argv,
        default_profile=v4_wrapper.DEFAULT_NORMALIZATION_PROFILE,
        default_condition_feature_transform=(
            v4_wrapper.DEFAULT_CONDITION_FEATURE_TRANSFORM
        ),
        default_input_feature_schema=v4_wrapper.DEFAULT_INPUT_FEATURE_SCHEMA,
        default_coord_policy=v4_wrapper.DEFAULT_COORD_POLICY,
        default_extent_feature_policy=v4_wrapper.DEFAULT_EXTENT_FEATURE_POLICY,
    )
    v4_wrapper._install_profile_hooks(*profile)
    original_argv = sys.argv
    try:
        sys.argv = argv
        return runner.parse_args()
    finally:
        sys.argv = original_argv


def _train_only_examples(
    sample_root: Path,
    train_ids: list[str],
) -> list[Any]:
    """Load labels only for IDs assigned to train by the frozen split map."""

    legacy = Heat3DV1SupervisedDataset.__new__(Heat3DV1SupervisedDataset)
    legacy.datadir = sample_root
    legacy.sample_dirs = [sample_root / sample_id for sample_id in train_ids]
    legacy.input_mode = "pure_physics"
    legacy.k_encoding_mode = "diag3"
    legacy.allowed_stages = tuple(PHYSICS_LABEL_SUPERVISED_STAGES)
    legacy.boundary_mask_fallback = True
    legacy.samples = [
        legacy._load_sample(sample_dir) for sample_dir in legacy.sample_dirs
    ]

    native = Heat3DV1NativeSupervisedDataset.__new__(
        Heat3DV1NativeSupervisedDataset
    )
    native._legacy_dataset = legacy
    native.k_encoding_mode = "diag3"
    native.boundary_mask_fallback = True
    native.samples = [native._to_native(sample) for sample in legacy.samples]
    return native.samples


def _tree_finite(tree: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_l2(tree: Any) -> float:
    total = sum(
        float(np.sum(np.square(np.asarray(leaf, dtype=np.float64))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return float(np.sqrt(total))


def _parameter_counts(params: Any) -> tuple[int, dict[str, int]]:
    groups = {"backbone": 0, "shape_decoder": 0, "scale_head": 0}
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        size = int(np.asarray(leaf).size)
        total += size
        groups[parameter_group(path)] += size
    return total, groups


def _git_payload() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("real-data smoke requires a clean training checkout")
    return {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "clean": True,
    }


def run(config_path: Path) -> dict[str, Any]:
    git = _git_payload()
    fraction = os.environ.get(
        "MEM_FRACTION", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    )
    if fraction is None or not np.isclose(float(fraction), 0.85):
        raise RuntimeError("set MEM_FRACTION=0.85 for the real-data smoke")

    config = _resolved(config_path)
    if config["run"].get("init_checkpoint") is not None:
        raise RuntimeError("Gate 6Q smoke requires random initialization")
    if config["run"]["batch_size"] != 28:
        raise RuntimeError("Gate 6Q real smoke requires train batch size 28")
    if config["export"]["prediction_split"] != "valid_iid":
        raise RuntimeError("unexpected prediction split contract")

    args = _args_from_config(config)
    sample_root = runner._sample_root(args.subset)
    split_map = Path(args.split_map)
    split_ids, split_source, primary_valid, stress_valid = (
        runner._resolve_training_splits(sample_root, split_map)
    )
    train_ids = list(split_ids["train"])
    if len(train_ids) != 672:
        raise RuntimeError(f"expected 672 train samples, found {len(train_ids)}")
    forbidden = {
        sample_id
        for role, sample_ids in split_ids.items()
        if role != "train"
        for sample_id in sample_ids
    }
    if set(train_ids).intersection(forbidden):
        raise RuntimeError("split map assigns overlapping train/forbidden IDs")
    train_examples = _train_only_examples(sample_root, train_ids)

    loss_config = runner._loss_config_from_args(args)
    lr_config = runner._lr_config_from_args(args)
    optimizer_config = runner._optimizer_config_from_args(args)
    seed_config = runner._seed_config_from_args(args)
    model_config = runner._model_config_from_args(args)
    batch_config = runner._batch_config_from_args(args)
    graph_config = runner._graph_config_from_args(args)
    stats = runner._train_only_stats(train_examples)
    model_config = runner._resolve_decoder_bypass_model_config(
        model_config, stats
    )
    runner._fit_native_loss_train_references(loss_config, train_examples)

    rng = np.random.default_rng(int(batch_config["batch_build_seed"]))
    selected_indices = rng.permutation(len(train_examples))[:28]
    batch_examples = [train_examples[int(index)] for index in selected_indices]
    builder = Heat3DGraphBuilder(**graph_config)
    group = runner._make_batch_group_with_seed(
        "gate6q_real_train_B28",
        batch_examples,
        stats,
        builder,
        graph_seed=seed_config["graph_seed"],
    )
    by_id = {example.sample_id: example for example in batch_examples}
    global_lookup, global_payload = runner._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=[],
    )
    global_standardizer = global_payload.get("standardizer", {})
    if (
        global_standardizer.get("fit_population") != "train_only"
        or global_standardizer.get("fit_sample_count") != len(train_examples)
    ):
        raise RuntimeError("global-context standardizer was not fit on train only")
    runner._attach_global_context_to_groups(
        [group],
        global_lookup,
        expected_feature_dim=model_config["global_context_feature_dim"],
    )
    scale_lookup, scale_payload = runner._prepare_scale_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=[],
    )
    runner._attach_scale_context_to_groups(
        [group],
        scale_lookup,
        expected_feature_dim=model_config["scale_context_feature_dim"],
    )
    runner._attach_native_physics_to_groups([group], by_id)
    if (
        model_config.get("scale_pooling") == "qk_gated"
        or model_config.get("shape_attention_mode") != "none"
        or model_config.get("scale_attention_mode") != "none"
    ):
        runner._attach_qk_region_features_to_groups(
            [group],
            by_id,
            feature_version=model_config["qk_region_feature_version"],
        )
    partition_audit = None
    if model_config.get("scale_deepsets_mode", "none") != "none":
        runner._attach_scale_deepsets_weights_to_groups([group], by_id)
        first = batch_examples[0]
        relative = first.get_relative_bc_feature_view()
        metadata = group["metadata"]
        partition_audit = p2r_partition_of_unity_audit(
            coords=np.asarray(first.condition.coords, dtype=np.float64),
            raw_condition=np.asarray(
                relative.condition_features, dtype=np.float64
            ),
            condition_feature_names=tuple(relative.condition_feature_names),
            p2r_edge_indices=np.asarray(metadata.p2r_edge_indices)[0],
            rnode_count=int(np.asarray(metadata.x_rnodes).shape[1] - 1),
        )

    model = runner.GraphNeuralOperator(**model_config)
    params = runner._model_init(
        model,
        jax.random.PRNGKey(seed_config["model_seed"]),
        group,
        group["inputs"],
    )["params"]
    initial_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
    updates_per_epoch = int(np.ceil(len(train_examples) / 28))
    lr_config["updates_per_epoch"] = updates_per_epoch
    optax_state = runner._build_optax_state(
        params,
        epochs=600,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
    )
    edge_key = runner._training_edge_masking_key(
        model_config,
        model_seed=seed_config["model_seed"],
        epoch=1,
        batch_index=1,
    )
    active_loss_config = runner._loss_config_for_epoch(loss_config, 1)

    def objective(current_params):
        components = runner._loss_components(
            model,
            current_params,
            [group],
            stats,
            active_loss_config,
            key=edge_key,
        )
        return components["total_loss"], components

    started = time.perf_counter()
    (loss_value, components), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(params)
    updates, next_state = optax_state["tx"].update(
        gradients, optax_state["state"], params
    )
    updates = runner._apply_native_update_controls(
        updates,
        native_enabled=True,
        model_config=model_config,
        optimizer_config=optimizer_config,
    )
    params = optax_state["apply_updates"](params, updates)
    runner._block_until_ready_tree(params)
    elapsed = time.perf_counter() - started

    delta = jax.tree_util.tree_map(
        lambda after, before: after - before, params, initial_params
    )
    total_params, grouped_params = _parameter_counts(params)
    memory = jax.devices()[0].memory_stats() or {}
    sample_hash = hashlib.sha256(
        "\n".join(group["sample_ids"]).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "heat3d_v5_gate6q_real_train_update_smoke_v1",
        "config_id": config["config_id"],
        "git": git,
        "device": str(jax.devices()[0]),
        "memory_fraction": float(fraction),
        "batch_size": 28,
        "node_count": int(group["inputs"].x_inp.shape[2]),
        "train_split_count": len(train_ids),
        "accessed_roles": ["train"],
        "forbidden_roles_accessed": [],
        "split_source": split_source,
        "primary_validation_role_not_loaded": primary_valid,
        "stress_validation_role_not_loaded": stress_valid,
        "sample_id_hash": sample_hash,
        "random_initialization": True,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "output_artifacts_written": 0,
        "optimizer": "adamw",
        "optimizer_update_applied": True,
        "update_nonzero": _tree_l2(delta) > 0.0,
        "loss": float(loss_value),
        "loss_components": {
            name: float(value)
            for name, value in components.items()
            if np.asarray(value).ndim == 0
        },
        "gradient_norms": {
            name: float(value)
            for name, value in native_gradient_group_norms(gradients).items()
        },
        "gradient_global_norm": _tree_l2(gradients),
        "update_global_norm": _tree_l2(delta),
        "finite_loss": bool(np.isfinite(float(loss_value))),
        "finite_gradients": _tree_finite(gradients),
        "finite_updated_parameters": _tree_finite(params),
        "parameter_count": total_params,
        "parameter_count_by_trainable_group": grouped_params,
        "native_loss_train_references": {
            "raw_train_target_energy_per_point": loss_config.get(
                "native_raw_train_target_energy_per_point"
            ),
            "log_scale_train_true_scale_sq_mean": loss_config.get(
                "native_log_scale_train_true_scale_sq_mean"
            ),
            "log_scale_weight_diagnostics": loss_config.get(
                "native_log_scale_weight_diagnostics"
            ),
        },
        "global_context_fit_roles": ["train"],
        "global_context_fit_sample_count": global_standardizer.get(
            "fit_sample_count"
        ),
        "scale_context_fit_roles": scale_payload.get("fit_roles"),
        "p2r_partition_of_unity": partition_audit,
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
        "bytes_in_use": memory.get("bytes_in_use"),
        "elapsed_seconds": elapsed,
        "training_started": False,
        "formal_e600_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = (
        report["optimizer_update_applied"]
        and report["update_nonzero"]
        and report["finite_loss"]
        and report["finite_gradients"]
        and report["finite_updated_parameters"]
        and report["node_count"] == 1024
        and report["forbidden_roles_accessed"] == []
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
