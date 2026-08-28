#!/usr/bin/env python3
"""Qualify V7 fixed-batch JIT executable reuse against the stable trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import (
    TrainingBatch,
    TrainingDependencies,
    V7FormalTrainer,
    build_v1_training_batches,
    build_v1_training_stats,
    load_selected_v1_examples,
    make_optimizer,
)
from rigno.models.rigno import RIGNO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = ROOT / "data" / "heat3d-thermal-simulation" / "subsets" / "v1_multilayer_bc_eq_supervised_small"
DEFAULT_MANIFEST = ROOT / "configs" / "heat3d_v1_supervised_small_manifest.json"
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 2,
    "node_latent_size": 16,
    "edge_latent_size": 16,
    "mlp_hidden_layers": 1,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}
MAX_ABS_FLOAT32_COMPILER_ROUNDOFF = 2.0e-6
MAX_LOSS_ABS_FLOAT32_COMPILER_ROUNDOFF = 2.0e-7


def _tree_diff(left: Any, right: Any) -> dict[str, Any]:
    left_leaves, right_leaves = tree.tree_leaves(left), tree.tree_leaves(right)
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    shapes_match = len(left_leaves) == len(right_leaves)
    for a, b in zip(left_leaves, right_leaves, strict=False):
        aa, bb = np.asarray(a), np.asarray(b)
        shapes_match = shapes_match and aa.shape == bb.shape
        if aa.shape != bb.shape:
            continue
        diff = aa.astype(np.float64) - bb.astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(diff), initial=0.0)))
        sum_sq += float(np.sum(np.square(diff)))
        count += int(diff.size)
    return {
        "max_abs": max_abs,
        "rmse": float(np.sqrt(sum_sq / max(count, 1))),
        "shapes_match": shapes_match,
        "exact": shapes_match and max_abs == 0.0,
    }


def _apply(model: RIGNO, params: Any, batch: TrainingBatch) -> tuple[Any, ...]:
    return tuple(
        model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        for group in batch.groups
    )


def _loss(predictions: tuple[Any, ...], batch: TrainingBatch) -> Any:
    weighted = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for prediction, group in zip(predictions, batch.groups, strict=True):
        target = group["target_normalized"]
        n = int(target.shape[0])
        weighted = weighted + jnp.mean(jnp.square(prediction - target)) * n
        count += n
    return weighted / max(count, 1)


def _dependencies(model: RIGNO, batch: TrainingBatch, builder: Any, stats: Any, *, jit_cache: bool) -> V7FormalTrainer:
    dependencies = TrainingDependencies(
        data_source="selected_train_valid_fixture",
        feature_transform="rigno.heat3d_training.prepare",
        normalization=stats,
        graph_builder=builder,
        model=model,
        model_apply=lambda params, current_batch, rng: _apply(model, params, current_batch),
        loss_fn=_loss,
        optimizer=make_optimizer("manual_gd", learning_rate=1.0e-5),
        batch_iterator=lambda batches: batches,
        validation_fn=lambda params, current_batch: _apply(model, params, current_batch),
        checkpoint_writer=lambda path, payload: None,
        metrics_fn=lambda params, current_batch: {},
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=jit_cache)
    return trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify V7 JIT reuse against eager stable trainer.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--valid-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    jax.config.update("jax_platform_name", "cpu")
    selected = load_selected_v1_examples(
        args.subset,
        args.manifest,
        train_count=args.train_count,
        valid_count=args.valid_count,
    )
    stats = build_v1_training_stats(selected["train"])
    builder = Heat3DGraphBuilder(radius_policy="legacy_kdtree_mean4", coverage_repair_policy="none")
    train_batch = build_v1_training_batches(
        selected["train"], stats, builder, batch_prefix="optimization_train"
    )[0]
    valid_batch = build_v1_training_batches(
        selected["valid"], stats, builder, batch_prefix="optimization_valid"
    )[0]
    stable_model = RIGNO(**MODEL_CONFIG)
    optimized_model = RIGNO(**MODEL_CONFIG)
    initial = stable_model.init(
        jax.random.PRNGKey(0),
        inputs=train_batch.groups[0]["inputs"],
        graphs=train_batch.groups[0]["graphs"],
    )["params"]
    stable = _dependencies(stable_model, train_batch, builder, stats, jit_cache=False)
    optimized = _dependencies(optimized_model, train_batch, builder, stats, jit_cache=True)
    stable_state = stable.initialize(initial)
    optimized_state = optimized.initialize(initial)
    stable_rows = []
    optimized_rows = []
    key = jax.random.PRNGKey(1)
    for step in range(args.steps):
        key, step_key = jax.random.split(key)
        stable_result = stable.step(stable_state, train_batch, step_key)
        optimized_result = optimized.step(optimized_state, train_batch, step_key)
        stable_state, optimized_state = stable_result.state, optimized_result.state
        stable_rows.append(stable_result)
        optimized_rows.append(optimized_result)
    rows = []
    for stable_result, optimized_result in zip(stable_rows, optimized_rows, strict=True):
        rows.append({
            "step": stable_result.state.step,
            "loss": {"abs": abs(float(stable_result.loss) - float(optimized_result.loss)), "exact": float(stable_result.loss) == float(optimized_result.loss)},
            "prediction": _tree_diff(stable_result.prediction, optimized_result.prediction),
            "gradients": _tree_diff(stable_result.gradients, optimized_result.gradients),
            "updates": _tree_diff(stable_result.updates, optimized_result.updates),
            "params": _tree_diff(stable_result.state.params, optimized_result.state.params),
        })
    stable_valid = stable.validate(stable_state, valid_batch)
    optimized_valid = optimized.validate(optimized_state, valid_batch)
    valid_diff = _tree_diff(stable_valid, optimized_valid)
    result = {
        "schema_version": "heat3d_v7_1T_training_optimization_equivalence_v1",
        "optimization": "fixed_batch_jit_executable_reuse",
        "backend": jax.default_backend(),
        "sample_ids": {role: [example.sample_id for example in rows] for role, rows in selected.items()},
        "steps": int(args.steps),
        "scientific_evidence_eligible": False,
        "test_or_sealed_accessed": False,
        "stable_runtime": "V7FormalTrainer(jit_cache=false)",
        "optimized_runtime": "V7FormalTrainer(jit_cache=true)",
        "stable_compile_count": stable.compile_count,
        "optimized_compile_count": optimized.compile_count,
        "equivalence_policy": {
            "prepared_and_state_shapes": "exact",
            "numeric": "CPU deterministic float32 compiler-roundoff envelope only",
            "max_abs_allowed_non_loss": MAX_ABS_FLOAT32_COMPILER_ROUNDOFF,
            "max_abs_allowed_loss": MAX_LOSS_ABS_FLOAT32_COMPILER_ROUNDOFF,
            "rationale": "JAX eager and compiled CPU lowerings preserve semantics but may reassociate float32 reductions by a few ULPs",
        },
        "step_equivalence": rows,
        "validation_equivalence": valid_diff,
        "pass": all(
            row["loss"]["abs"] <= MAX_LOSS_ABS_FLOAT32_COMPILER_ROUNDOFF
            and row["prediction"]["shapes_match"]
            and row["prediction"]["max_abs"] <= MAX_ABS_FLOAT32_COMPILER_ROUNDOFF
            and row["gradients"]["shapes_match"]
            and row["gradients"]["max_abs"] <= MAX_ABS_FLOAT32_COMPILER_ROUNDOFF
            and row["updates"]["shapes_match"]
            and row["updates"]["max_abs"] <= MAX_ABS_FLOAT32_COMPILER_ROUNDOFF
            and row["params"]["shapes_match"]
            and row["params"]["max_abs"] <= MAX_ABS_FLOAT32_COMPILER_ROUNDOFF
            for row in rows
        ) and valid_diff["shapes_match"] and valid_diff["max_abs"] <= MAX_ABS_FLOAT32_COMPILER_ROUNDOFF,
    }
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
