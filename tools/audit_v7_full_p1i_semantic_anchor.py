#!/usr/bin/env python3
"""Audit the frozen V6 Full P1i path against the V7 library path.

This is deliberately a compatibility-audit tool.  It uses the frozen V1--V6
runner helper as an oracle, but it does not belong to the V7 production graph.
Only explicitly selected train and valid_iid samples are materialized; no
metrics, test_iid samples, sealed samples, solver, or formal experiment are
run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import (
    TrainingBatch,
    TrainingDependencies,
    V7FormalTrainer,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_init_full,
    loss_fn_full,
)
from rigno.heat3d_training.p1i import (
    attach_input_contexts,
    attach_native_physics,
    attach_qk_features,
    build_p1i_batches,
    fit_native_loss_references,
)
from rigno.heat3d_v1_normalization import legacy_train_only_stats
from rigno.heat3d_v1_training_semantics import COORD_POLICY_TRAIN_MINMAX_UNIT_BOX
from rigno.heat3d_v5_shape_scale import native_shape_scale_losses
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset
from rigno.models.rigno import RIGNO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json"
HISTORICAL_COMMIT = "8a812619ab0112b4ecfc37ef18189f731180059d"
HISTORICAL_RUNNER = "scripts/run_heat3d_v1_medium_controlled_training_export.py"
HISTORICAL_RUNNER_BLOB = "c16b22d80a8f721c31264c0a22a30acdc6f53a31"
HISTORICAL_MODEL_BLOB = "c434da5b60ef66f1a8621a7e513a22ee5fdc174a"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--valid-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_diff(left: Any, right: Any) -> dict[str, Any]:
    left_leaves, left_def = tree.tree_flatten(left)
    right_leaves, right_def = tree.tree_flatten(right)
    if left_def != right_def or len(left_leaves) != len(right_leaves):
        return {
            "exact": False,
            "shapes_match": False,
            "max_abs": None,
            "rmse": None,
            "reason": "tree_structure_mismatch",
            "leaf_counts": [len(left_leaves), len(right_leaves)],
        }
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    shapes_match = True
    exact = True
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if left_array.shape != right_array.shape:
            shapes_match = False
            exact = False
            continue
        if left_array.dtype.kind in "OUS" or right_array.dtype.kind in "OUS":
            equal = np.array_equal(left_array, right_array)
            exact = exact and bool(equal)
            continue
        difference = left_array.astype(np.float64) - right_array.astype(np.float64)
        local_max = float(np.max(np.abs(difference), initial=0.0))
        max_abs = max(max_abs, local_max)
        sum_sq += float(np.sum(np.square(difference)))
        count += int(difference.size)
        exact = exact and local_max == 0.0
    return {
        "exact": bool(exact and shapes_match),
        "shapes_match": shapes_match,
        "max_abs": max_abs,
        "rmse": float(np.sqrt(sum_sq / max(count, 1))),
    }


def _tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    leaves, definition = tree.tree_flatten(value)
    digest.update(str(definition).encode("utf-8"))
    for leaf in leaves:
        if leaf is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _scalar_diff(left: Any, right: Any) -> dict[str, Any]:
    left_value = float(np.asarray(left))
    right_value = float(np.asarray(right))
    difference = abs(left_value - right_value)
    return {
        "left": left_value,
        "right": right_value,
        "abs": difference,
        "exact": difference == 0.0,
    }


def _resolved_model_config(model_config: Mapping[str, Any], feature_names: Sequence[str]) -> dict[str, Any]:
    resolved = dict(model_config)
    resolved.pop("architecture", None)
    resolved["global_context_feature_names"] = tuple(
        resolved.get("global_context_feature_names") or ()
    )
    local_names = tuple(resolved.get("decoder_bypass_local_feature_names") or ())
    resolved["decoder_bypass_local_feature_names"] = local_names
    resolved["decoder_bypass_feature_names"] = local_names
    resolved["decoder_bypass_feature_indices"] = tuple(
        tuple(feature_names).index(name) for name in local_names
    )
    resolved["decoder_bypass_num_features"] = len(
        resolved["decoder_bypass_feature_indices"]
    )
    return resolved


def _selected_examples(subset: Path, manifest: Path) -> tuple[list[Any], list[Any]]:
    dataset = Heat3DV6DualRobinDataset(
        subset,
        manifest,
        include_roles={"train", "valid_iid"},
    )
    train = [
        sample
        for sample in dataset.samples
        if sample.meta["v6_adapter"]["manifest_split_role"] == "train"
    ]
    valid = [
        sample
        for sample in dataset.samples
        if sample.meta["v6_adapter"]["manifest_split_role"] == "valid_iid"
    ]
    if len(train) < 24 or len(valid) < 32:
        raise ValueError(f"insufficient train/valid_iid fixture: {len(train)}/{len(valid)}")
    return train, valid


def _attach_legacy_features(
    legacy_module: Any,
    groups: list[dict[str, Any]],
    train_examples: list[Any],
    required_examples: list[Any],
    model_config: Mapping[str, Any],
) -> None:
    context, _payload = legacy_module._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=required_examples,
    )
    legacy_module._attach_global_context_to_groups(
        groups,
        context,
        expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
    )
    by_id = {example.sample_id: example for example in [*train_examples, *required_examples]}
    legacy_module._attach_native_physics_to_groups(groups, by_id)
    legacy_module._attach_qk_region_features_to_groups(
        groups,
        by_id,
        feature_version=str(model_config.get("qk_region_feature_version", "sparse_safe_v2")),
    )


def _stable_components(
    prediction: Mapping[str, Any],
    group: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    physics = group["native_physics"]
    weights = {
        "shape_cv": float(loss_config["native_shape_cv_weight"]),
        "log_scale": float(loss_config["native_log_scale_weight"]),
        "relative_field": float(loss_config["native_relative_field_weight"]),
        "raw_absolute": float(loss_config["native_raw_field_weight"]),
    }
    components = native_shape_scale_losses(
        prediction,
        target_deltaT=group["target_delta_raw"],
        control_volumes=physics["control_volumes"],
        dirichlet_mask=physics["dirichlet_mask"],
        loss_weights=weights,
        raw_loss_mode=str(loss_config["native_raw_loss_mode"]),
        raw_train_target_energy_per_point=float(
            loss_config["native_raw_train_target_energy_per_point"]
        ),
        log_scale_weight_mode=str(loss_config["native_log_scale_weight_mode"]),
        log_scale_train_true_scale_sq_mean=float(
            loss_config["native_log_scale_train_true_scale_sq_mean"]
        ),
        log_scale_weight_clip=(
            float(loss_config["native_log_scale_weight_clip_min"]),
            float(loss_config["native_log_scale_weight_clip_max"]),
        ),
    )
    # The legacy wrapper and ``loss_fn_full`` both aggregate a group by
    # multiplying each component by its sample count and dividing by the
    # batch count.  Reproduce that frozen boundary here instead of comparing
    # the raw single-group total to the aggregated legacy scalar.  This is a
    # comparison-path correction only; the loss definition is unchanged.
    sample_count = int(group["target_delta_raw"].shape[0])
    return {
        **components,
        "shape_cv_loss": components["shape_cv_loss"] * sample_count / max(sample_count, 1),
        "log_scale_loss": components["log_scale_loss"] * sample_count / max(sample_count, 1),
        "relative_field_loss": components["relative_field_loss"] * sample_count / max(sample_count, 1),
        "raw_absolute_field_loss": components["raw_absolute_field_loss"] * sample_count / max(sample_count, 1),
        "total_loss": components["total_loss"] * sample_count / max(sample_count, 1),
    }


def _component_diffs(old_components: Mapping[str, Any], new_components: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "shape_cv_loss",
        "log_scale_loss",
        "relative_field_loss",
        "raw_absolute_field_loss",
        "total_loss",
    )
    result: dict[str, Any] = {}
    for name in names:
        if name not in old_components or name not in new_components:
            result[name] = {"exact": False, "reason": "component_missing"}
        else:
            result[name] = _tree_diff(old_components[name], new_components[name])
    return result


def _group_comparison(old_group: Mapping[str, Any], new_group: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "inputs",
        "target_normalized",
        "target_delta_raw",
        "target_temperature",
        "t_ref",
        "metadata",
        "graphs",
        "global_context",
        "native_physics",
        "qk_region_features",
    )
    comparison = {field: _tree_diff(old_group[field], new_group[field]) for field in fields}
    comparison["sample_ids"] = {
        "exact": tuple(old_group["sample_ids"]) == tuple(new_group["sample_ids"]),
        "old_hash": hashlib.sha256(json.dumps(list(old_group["sample_ids"]), separators=(",", ":")).encode()).hexdigest(),
        "new_hash": hashlib.sha256(json.dumps(list(new_group["sample_ids"]), separators=(",", ":")).encode()).hexdigest(),
    }
    comparison["feature_names"] = {
        "exact": tuple(old_group["feature_names"]) == tuple(new_group["feature_names"]),
    }
    comparison["hashes"] = {
        "support_inputs": _tree_hash(new_group["inputs"]),
        "graph_metadata": _tree_hash(new_group["metadata"]),
        "graph_tensors": _tree_hash(new_group["graphs"]),
        "global_context": _tree_hash(new_group["global_context"]),
        "native_physics": _tree_hash(new_group["native_physics"]),
        "qk_region_features": _tree_hash(new_group["qk_region_features"]),
    }
    return comparison


def _all_exact(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "exact" in value:
            return value["exact"] is True
        return all(_all_exact(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_exact(child) for child in value)
    # Counts, hashes, paths and other receipt metadata are not comparison
    # failures; comparison equality is represented by a sibling ``exact``
    # field in the structured diff record.
    return True


def main() -> int:
    args = _args()
    if args.train_count != 24 or args.valid_count != 32:
        raise ValueError("the semantic anchor fixture is frozen at train=24 and valid_iid=32")
    if args.steps < 1 or args.steps > 10:
        raise ValueError("semantic anchor steps must be in [1, 10]")
    jax.config.update("jax_platform_name", "cpu")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train, valid = _selected_examples(args.subset, args.manifest)
    train_all = list(train)
    train_order = np.random.default_rng(args.seed).permutation(len(train_all))
    train_fixture = [train_all[int(index)] for index in train_order[: args.train_count]]
    valid_fixture = list(valid[: args.valid_count])
    stats = legacy_train_only_stats(
        train_all,
        coord_policy=COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    )
    model_config = _resolved_model_config(config["model"], stats["feature_names"])
    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(train_all, loss_config))
    graph_config = dict(config["graph"])

    # This compatibility import is intentionally outside the production graph.
    from scripts import run_heat3d_v1_medium_controlled_training_export as legacy

    old_train_builder = Heat3DGraphBuilder(**graph_config)
    old_valid_builder = Heat3DGraphBuilder(**graph_config)
    old_train_group = legacy._make_batch_group_with_seed(
        "semantic_train",
        train_fixture,
        stats,
        old_train_builder,
        graph_seed=args.seed,
    )
    old_valid_group = legacy._make_batch_group_with_seed(
        "semantic_valid_iid",
        valid_fixture,
        stats,
        old_valid_builder,
        graph_seed=args.seed,
    )
    _attach_legacy_features(
        legacy,
        [old_train_group, old_valid_group],
        train_all,
        [*train_fixture, *valid_fixture],
        config["model"],
    )

    stable_train_builder = Heat3DGraphBuilder(**graph_config)
    stable_valid_builder = Heat3DGraphBuilder(**graph_config)
    stable_train_batch = build_p1i_batches(
        train_fixture,
        stats,
        stable_train_builder,
        label="semantic_train",
        batch_size=args.train_count,
        graph_seed=args.seed,
        batch_build_seed=None,
    )[0]
    stable_valid_batch = build_p1i_batches(
        valid_fixture,
        stats,
        stable_valid_builder,
        label="semantic_valid_iid",
        batch_size=args.valid_count,
        graph_seed=args.seed,
        batch_build_seed=None,
    )[0]
    stable_context = attach_input_contexts(
        [stable_train_batch, stable_valid_batch],
        train_all,
        [*train_fixture, *valid_fixture],
        config["model"],
    )
    examples_by_id = {example.sample_id: example for example in [*train_all, *valid_fixture]}
    attach_native_physics(
        [stable_train_batch, stable_valid_batch],
        examples_by_id,
        context_by_id=stable_context["raw_context_by_id"],
    )
    attach_qk_features(
        [stable_train_batch, stable_valid_batch],
        examples_by_id,
        feature_version=str(config["model"]["qk_region_feature_version"]),
    )
    new_train_group = stable_train_batch.groups[0]
    new_valid_group = stable_valid_batch.groups[0]

    old_model = RIGNO(**model_config)
    new_model = RIGNO(**model_config)
    init_key = jax.random.PRNGKey(args.seed)
    old_params = legacy._model_init(
        old_model,
        init_key,
        old_train_group,
        old_train_group["inputs"],
    )["params"]
    new_params = model_init_full(new_model, init_key, stable_train_batch)["params"]
    initial_old_params = old_params
    initial_new_params = new_params

    optimizer_config = dict(config["optimizer"])
    old_lr_config = dict(optimizer_config)
    old_lr_config["updates_per_epoch"] = 32
    old_optax = legacy._build_optax_state(
        old_params,
        epochs=200,
        lr_config=old_lr_config,
        optimizer_config=optimizer_config,
    )
    new_optimizer = make_p1i_optimizer(
        optimizer_config,
        epochs=200,
        updates_per_epoch=32,
    )
    dependencies = TrainingDependencies(
        data_source={"dataset_id": config["dataset"]["dataset_id"], "roles": ["train", "valid_iid"]},
        feature_transform="v7_explicit_p1i_feature_assembly",
        normalization=stats,
        graph_builder=stable_train_builder,
        model=new_model,
        model_apply=lambda params, batch, rng: model_apply_full(new_model, params, batch, rng),
        loss_fn=lambda predictions, batch: loss_fn_full(predictions, batch, loss_config),
        optimizer=new_optimizer,
        batch_iterator=lambda batches: batches,
        validation_fn=lambda params, batch: loss_fn_full(
            model_apply_full(new_model, params, batch), batch, loss_config
        ),
        checkpoint_writer=lambda path, payload: None,
        metrics_fn=lambda params, batch: {},
        gradient_transform=make_gradient_transform(model_config, optimizer_config),
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=False)
    new_state = trainer.initialize(new_params)
    old_optimizer_state = old_optax["state"]

    prepared = _group_comparison(old_train_group, new_train_group)
    initial_old_prediction = legacy._model_apply(
        old_model,
        old_params,
        old_train_group,
        key=jax.random.fold_in(jax.random.PRNGKey(args.seed), 0),
    )
    initial_new_prediction = model_apply_full(
        new_model,
        new_params,
        stable_train_batch,
        # model_apply_full folds the run key by group index; pass the same
        # unfurled run key used by the legacy single-group call below.
        rng=jax.random.PRNGKey(args.seed),
    )[0]
    initial_old_components = legacy._native_loss_components(
        old_model,
        old_params,
        [old_train_group],
        stats,
        loss_config,
        key=jax.random.PRNGKey(args.seed),
    )
    initial_new_components = _stable_components(
        initial_new_prediction,
        new_train_group,
        loss_config,
    )
    steps: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        step_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), step)

        old_prediction_before_update = legacy._model_apply(
            old_model,
            old_params,
            old_train_group,
            # The legacy native loss path folds the run key by group index;
            # mirror that one-group boundary before comparing model outputs.
            key=jax.random.fold_in(step_key, 0),
        )
        old_components_before_update = legacy._native_loss_components(
            old_model,
            old_params,
            [old_train_group],
            stats,
            loss_config,
            key=step_key,
        )

        def old_total(current_params: Any) -> Any:
            return legacy._native_loss_components(
                old_model,
                current_params,
                [old_train_group],
                stats,
                loss_config,
                key=step_key,
            )["total_loss"]

        old_loss, old_raw_gradients = jax.value_and_grad(old_total)(old_params)
        old_gradients = legacy.mask_native_trainable_scope(
            old_raw_gradients,
            branch_mode=model_config["native_branch_mode"],
            trainable_scope=str(optimizer_config.get("native_trainable_scope", "branch")),
        )
        old_updates, old_optimizer_state = old_optax["tx"].update(
            old_gradients,
            old_optimizer_state,
            old_params,
        )
        old_updates = legacy._apply_native_update_controls(
            old_updates,
            native_enabled=True,
            model_config=model_config,
            optimizer_config=optimizer_config,
        )
        old_params = optax.apply_updates(old_params, old_updates)
        new_result = trainer.step(new_state, stable_train_batch, rng=step_key)
        new_state = new_result.state
        step_record = {
            "step": step,
            "loss": _scalar_diff(old_loss, new_result.loss),
            "gradients": _tree_diff(old_gradients, new_result.gradients),
            "updates": _tree_diff(old_updates, new_result.updates),
            "params": _tree_diff(old_params, new_state.params),
            "optimizer_state": _tree_diff(old_optimizer_state, new_state.optimizer_state),
            "prediction_before_update": _tree_diff(
                old_prediction_before_update,
                new_result.prediction[0],
            ),
            "loss_components_before_update": _component_diffs(
                old_components_before_update,
                _stable_components(new_result.prediction[0], new_train_group, loss_config),
            ),
        }
        steps.append(step_record)

    final_old_valid_prediction = legacy._model_apply(
        old_model,
        old_params,
        old_valid_group,
        key=None,
    )
    final_new_valid_prediction = model_apply_full(
        new_model,
        new_state.params,
        stable_valid_batch,
        rng=None,
    )
    validation = [
        _tree_diff(final_old_valid_prediction, final_new_valid_prediction[0])
    ]

    # The historical helper source is byte-identical to the recorded V6
    # execution commit.  The model module has only the explicit physics-only
    # branch added after that commit; Full uses the default learned-residual
    # branch, so this audit records the source delta instead of hiding it.
    current_runner_blob = _git("hash-object", HISTORICAL_RUNNER)
    current_model_blob = _git("hash-object", "rigno/models/rigno.py")
    result = {
        "schema_version": "heat3d_v7_full_p1i_semantic_anchor_receipt_v1",
        "status": "pending",
        "execution_role": "compatibility_audit",
        "sample_role": "train_plus_valid_fixture",
        "labels_read": True,
        "metrics_executed": False,
        "model_selection": False,
        "scientific_evidence_eligible": False,
        "test_iid_labels_read": False,
        "sealed_labels_read": False,
        "solver_executed": False,
        "formal_g1_started": False,
        "backend": jax.default_backend(),
        "device": [str(device) for device in jax.devices()],
        "historical": {
            "execution_commit": HISTORICAL_COMMIT,
            "runner_file": HISTORICAL_RUNNER,
            "runner_blob_at_execution_commit": HISTORICAL_RUNNER_BLOB,
            "runner_blob_current": current_runner_blob,
            "runner_source_reconciled": current_runner_blob == HISTORICAL_RUNNER_BLOB,
            "model_blob_at_execution_commit": HISTORICAL_MODEL_BLOB,
            "model_blob_current": current_model_blob,
            "model_source_delta_scope": "explicit physics-only branch only; Full default learned_residual behavior is the compared scope",
        },
        "v7": {
            "code_commit": _git("rev-parse", "HEAD"),
            "runtime": "rigno.heat3d_training + rigno.models.rigno.RIGNO",
            "entrypoint": "scripts/run_heat3d_v7_formal_p1i_training.py",
        },
        "dataset": {
            "dataset_id": config["dataset"]["dataset_id"],
            "manifest_sha256": config["dataset"]["manifest_sha256"],
            "full_field_archive_sha256": config["dataset"]["full_field_archive_sha256"],
            "train_population": len(train_all),
            "valid_iid_population": len(valid),
            "test_iid_sample_dirs_loaded": False,
            "sealed_sample_dirs_loaded": False,
        },
        "fixture": {
            "seed": args.seed,
            "batch_build_seed": args.seed,
            "graph_seed": args.seed,
            "train_sample_ids": [str(example.sample_id) for example in train_fixture],
            "valid_iid_sample_ids": [str(example.sample_id) for example in valid_fixture],
            "train_batch_size": len(train_fixture),
            "valid_iid_batch_size": len(valid_fixture),
            "steps": args.steps,
            "optimizer": "adamw",
            "schedule": "warmup_cosine; epochs=200; updates_per_epoch=32",
        },
        "comparison": {
            "prepared_inputs_and_features": prepared,
            "initial_params": _tree_diff(
                initial_old_params,
                initial_new_params,
            ),
            "initial_prediction": _tree_diff(initial_old_prediction, initial_new_prediction),
            "initial_loss_components": _component_diffs(initial_old_components, initial_new_components),
            "initial_optimizer_state": _tree_diff(
                old_optax["tx"].init(initial_old_params),
                new_optimizer.init(initial_new_params),
            ),
            "steps": steps,
            "validation_prediction": validation,
            "validation_hashes": {
                "legacy": [_tree_hash(value) for value in final_old_valid_prediction],
                "v7": [_tree_hash(value) for value in final_new_valid_prediction],
            },
        },
        "policy": {
            "same_cpu_backend_is_semantic_oracle": True,
            "exact_required_for_comparable_arrays": True,
            "new_tolerance_created": False,
            "training_is_compatibility_only": True,
            "publication_evidence_written": False,
        },
    }
    exact_sections = [
        prepared,
        result["comparison"]["initial_params"],
        result["comparison"]["initial_prediction"],
        result["comparison"]["initial_loss_components"],
        result["comparison"]["initial_optimizer_state"],
        steps,
        validation,
    ]
    passed = all(_all_exact(section) for section in exact_sections)
    result["pass"] = passed
    result["status"] = "PASS" if passed else "FAIL_CLOSED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "steps": args.steps}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
