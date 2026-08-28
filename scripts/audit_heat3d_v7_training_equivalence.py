#!/usr/bin/env python3
"""Compare the historical small training semantics with the V7 trainer.

This is a compatibility audit entrypoint.  It imports the historical checker
only as an oracle and loads only explicitly selected train/valid samples via
the V7 scoped loader; it never enumerates or opens test/sealed samples.
"""

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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy/V7 training-step equivalence on train/valid only.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--valid-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _tree_diff(left: Any, right: Any) -> dict[str, Any]:
    left_leaves, right_leaves = tree.tree_leaves(left), tree.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return {"leaf_count_mismatch": [len(left_leaves), len(right_leaves)], "exact": False}
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    shapes_match = True
    for a, b in zip(left_leaves, right_leaves, strict=True):
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


def _legacy_replay(model: RIGNO, batch_groups: list[dict[str, Any]], steps: int, seed: int, lr: float):
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=batch_groups[0]["inputs"],
        graphs=batch_groups[0]["graphs"],
    )["params"]
    records = [{"step": 0, "params": params, "prediction": _apply(model, params, TrainingBatch("legacy", (), tuple(batch_groups)))}]
    for step in range(1, steps + 1):
        batch = TrainingBatch("legacy", (), tuple(batch_groups))
        def loss_fn(current_params):
            return _loss(_apply(model, current_params, batch), batch)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates = tree.tree_map(lambda gradient: -lr * gradient, gradients)
        params = tree.tree_map(lambda parameter, update: parameter + update, params, updates)
        records.append({
            "step": step,
            "loss": loss,
            "gradients": gradients,
            "updates": updates,
            "params": params,
            "prediction": _apply(model, params, batch),
        })
    return records


def main() -> int:
    args = _args()
    jax.config.update("jax_platform_name", "cpu")
    selected = load_selected_v1_examples(
        args.subset,
        args.manifest,
        train_count=args.train_count,
        valid_count=args.valid_count,
    )
    stats = build_v1_training_stats(selected["train"])
    legacy_builder = Heat3DGraphBuilder(radius_policy="legacy_kdtree_mean4", coverage_repair_policy="none")
    legacy_groups = []
    # This is the historical helper invoked with already-scoped examples; no
    # historical dataset constructor is called, so test samples are untouched.
    from scripts import check_heat3d_v1_small_train_valid_smoke as legacy
    legacy_groups = legacy._make_groups(selected["train"], stats, legacy_builder)
    legacy_valid_groups = legacy._make_groups(selected["valid"], stats, legacy_builder)

    stable_builder = Heat3DGraphBuilder(radius_policy="legacy_kdtree_mean4", coverage_repair_policy="none")
    stable_train_batch = build_v1_training_batches(
        selected["train"], stats, stable_builder, batch_prefix="equivalence_train"
    )[0]
    stable_valid_batch = build_v1_training_batches(
        selected["valid"], stats, stable_builder, batch_prefix="equivalence_valid"
    )[0]
    model_legacy = RIGNO(**legacy.MODEL_CONFIG)
    model_v7 = RIGNO(**legacy.MODEL_CONFIG)
    v7_dependencies = TrainingDependencies(
        data_source=selected,
        feature_transform="rigno.heat3d_training.prepare",
        normalization=stats,
        graph_builder=stable_builder,
        model=model_v7,
        model_apply=lambda params, batch, rng: _apply(model_v7, params, batch),
        loss_fn=_loss,
        optimizer=make_optimizer("manual_gd", learning_rate=args.lr),
        batch_iterator=lambda batches: batches,
        validation_fn=lambda params, batch: _apply(model_v7, params, batch),
        checkpoint_writer=lambda path, payload: None,
        metrics_fn=lambda params, batch: {},
    )
    trainer = V7FormalTrainer(v7_dependencies, jit_cache=False)
    v7_state = trainer.initialize(model_v7.init(
        jax.random.PRNGKey(args.seed),
        inputs=stable_train_batch.groups[0]["inputs"],
        graphs=stable_train_batch.groups[0]["graphs"],
    )["params"])
    legacy_records = _legacy_replay(model_legacy, legacy_groups, args.steps, args.seed, args.lr)
    v7_records = [{
        "step": 0,
        "params": v7_state.params,
        "prediction": _apply(model_v7, v7_state.params, stable_train_batch),
    }]
    key = jax.random.PRNGKey(args.seed + 1)
    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        result = trainer.step(v7_state, stable_train_batch, step_key)
        v7_state = result.state
        v7_records.append({
            "step": step,
            "loss": result.loss,
            "gradients": result.gradients,
            "updates": result.updates,
            "params": result.state.params,
            "prediction": result.prediction,
        })

    prepared = []
    for old, new in zip(legacy_groups, stable_train_batch.groups, strict=True):
        prepared.append({
            "sample_ids": list(old["sample_ids"]),
            "inputs": _tree_diff(old["inputs"], new["inputs"]),
            "graphs": _tree_diff(old["graphs"], new["graphs"]),
            "target_normalized": _tree_diff(old["target_normalized"], new["target_normalized"]),
        })
    steps = []
    for old, new in zip(legacy_records, v7_records, strict=True):
        row = {"step": int(old["step"]), "params": _tree_diff(old["params"], new["params"]), "prediction": _tree_diff(old["prediction"], new["prediction"])}
        for name in ("loss", "gradients", "updates"):
            if name in old and name in new:
                row[name] = _tree_diff(old[name], new[name]) if name != "loss" else {"abs": abs(float(old[name]) - float(new[name])), "exact": float(old[name]) == float(new[name])}
        steps.append(row)
    validation = []
    legacy_valid_prediction = _apply(model_legacy, legacy_records[-1]["params"], TrainingBatch("legacy_valid", (), tuple(legacy_valid_groups)))
    v7_valid_prediction = _apply(model_v7, v7_records[-1]["params"], stable_valid_batch)
    for old, new in zip(legacy_valid_prediction, v7_valid_prediction, strict=True):
        validation.append(_tree_diff(old, new))
    result = {
        "schema_version": "heat3d_v7_1T_training_equivalence_v1",
        "execution_role": "compatibility_audit",
        "sample_role": "train_plus_valid_fixture",
        "labels_read": True,
        "metrics_executed": False,
        "model_selection": False,
        "scientific_evidence_eligible": False,
        "test_or_sealed_accessed": False,
        "backend": jax.default_backend(),
        "steps": int(args.steps),
        "sample_ids": {role: [example.sample_id for example in rows] for role, rows in selected.items()},
        "prepared_inputs": prepared,
        "training_steps": steps,
        "validation_prediction": validation,
        "pass": all(
            row["inputs"]["exact"] and row["graphs"]["exact"] and row["target_normalized"]["exact"]
            for row in prepared
        ) and all(
            row["params"]["exact"] and row["prediction"]["exact"] and row.get("loss", {"exact": True})["exact"]
            and row.get("gradients", {"exact": True})["exact"] and row.get("updates", {"exact": True})["exact"]
            for row in steps
        ) and all(row["exact"] for row in validation),
    }
    text = json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
