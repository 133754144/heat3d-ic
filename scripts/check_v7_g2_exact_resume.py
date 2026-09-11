#!/usr/bin/env python3
"""Framework-level exact-resume qualification on a deterministic toy batch.

This is an execution-contract test, not a model or accuracy experiment.  It
compares a continuous sequence of optimizer updates with a process-boundary
checkpoint/reload sequence using identical parameters, keys, batch order and
metadata.  Real Heat3D/GINO/Transolver runners can reuse the same checkpoint
schema after their own immutable data/config hashes are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rigno.heat3d_training import (
    ManualGradientDescent,
    TrainingBatch,
    TrainingDependencies,
    V7FormalTrainer,
    atomic_latest_checkpoint,
    load_latest_checkpoint,
    tree_max_abs_difference,
)


def _digest_tree(value) -> str:
    leaves = jax.tree_util.tree_leaves(value)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def _trainer() -> tuple[V7FormalTrainer, TrainingBatch]:
    batch = TrainingBatch(
        batch_id="exact_resume_fixture",
        sample_ids=("fixture",),
        groups=({"x": jnp.asarray([1.5, -0.5]), "y": jnp.asarray([0.25, -0.75])},),
    )

    def apply(params, current, _rng):
        return params["w"] * current.groups[0]["x"] + params["b"]

    def loss(prediction, current):
        return jnp.mean(jnp.square(prediction - current.groups[0]["y"]))

    deps = TrainingDependencies(
        data_source="deterministic_fixture",
        feature_transform="identity",
        normalization="none",
        graph_builder="none",
        model="quadratic_fixture",
        model_apply=apply,
        loss_fn=loss,
        optimizer=ManualGradientDescent(0.05),
        batch_iterator=lambda value: value,
        validation_fn=lambda params, current: loss(apply(params, current, None), current),
        checkpoint_writer=lambda _path, _payload: None,
        metrics_fn=lambda _params, _current: {"finite": True},
    )
    return V7FormalTrainer(deps, jit_cache=True), batch


def _run(trainer, state, batch, keys, start_step=0):
    losses = []
    for offset, key in enumerate(keys, start=1):
        result = trainer.step_slim(state, batch, key)
        state = result.state
        state = state.__class__(
            params=jax.block_until_ready(state.params),
            optimizer_state=jax.block_until_ready(state.optimizer_state),
            step=state.step,
        )
        losses.append(float(result.loss))
        if state.step != start_step + offset:
            raise AssertionError("update count drifted")
    return state, losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--split", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.split < args.steps:
        parser.error("split must be between 1 and steps-1")

    keys = [jax.random.PRNGKey(100 + index) for index in range(args.steps)]
    params = {"w": jnp.asarray(0.8), "b": jnp.asarray(-0.2)}

    continuous_trainer, batch = _trainer()
    continuous_state = continuous_trainer.initialize(params)
    continuous_state, continuous_losses = _run(
        continuous_trainer, continuous_state, batch, keys
    )

    resumed_trainer, resumed_batch = _trainer()
    resumed_state = resumed_trainer.initialize(params)
    resumed_state, prefix_losses = _run(
        resumed_trainer, resumed_state, resumed_batch, keys[: args.split]
    )
    with tempfile.TemporaryDirectory(prefix="g2_exact_resume_") as temporary:
        checkpoint_path = Path(temporary) / "latest.pkl"
        receipt = atomic_latest_checkpoint(
            checkpoint_path,
            state=resumed_state,
            metadata={
                "epoch": args.split,
                "global_update_count": args.split,
                "best_metric": 0.0,
                "best_epoch": 1,
                "runner_sha": "fixture-runner",
                "config_sha": "fixture-config",
                "data_sha": "fixture-data",
            },
            rng_state={"next_key_index": args.split},
            batch_state={"batch_ids": [batch.batch_id], "order": list(range(args.steps))},
            scheduler_state={"name": "none"},
        )
        loaded = load_latest_checkpoint(checkpoint_path)
        resumed_state, suffix_losses = _run(
            resumed_trainer,
            loaded["state_object"],
            resumed_batch,
            keys[args.split :],
            start_step=args.split,
        )

    params_diff = tree_max_abs_difference(continuous_state.params, resumed_state.params)
    optimizer_diff = tree_max_abs_difference(
        continuous_state.optimizer_state, resumed_state.optimizer_state
    )
    losses_match = all(
        np.float64(a).tobytes() == np.float64(b).tobytes()
        for a, b in zip(continuous_losses, prefix_losses + suffix_losses, strict=True)
    )
    payload = {
        "schema_version": "g2_exact_resume_qualification_v1",
        "status": (
            "EXACT_RESUME_EQUIVALENCE_PASS"
            if params_diff == 0.0 and optimizer_diff == 0.0 and losses_match
            else "EXACT_RESUME_EQUIVALENCE_FAIL_CLOSED"
        ),
        "steps": args.steps,
        "split": args.split,
        "continuous_update_count": continuous_state.step,
        "resumed_update_count": resumed_state.step,
        "params_max_abs_difference": params_diff,
        "optimizer_state_max_abs_difference": optimizer_diff,
        "loss_trajectory_bitwise_equal": losses_match,
        "continuous_params_tree_sha256": _digest_tree(continuous_state.params),
        "resumed_params_tree_sha256": _digest_tree(resumed_state.params),
        "checkpoint_receipt": receipt,
        "test_or_sealed_access": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    return 0 if payload["status"] == "EXACT_RESUME_EQUIVALENCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
