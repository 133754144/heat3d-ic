#!/usr/bin/env python3
"""Short real-Heat3D exact-resume qualification.

The test uses the actual Heat3D RIGNO model and one frozen B24 train batch for
two deterministic optimizer updates.  It compares a continuous path with a
split path that writes an atomic, provenance-complete checkpoint after update
one and reloads it before update two.  This is an execution gate only: no
accuracy metric and no forbidden split are opened, and artifacts remain under
``/tmp``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_diff(left, right) -> float:
    leaves_left, treedef_left = jax.tree_util.tree_flatten(left)
    leaves_right, treedef_right = jax.tree_util.tree_flatten(right)
    if treedef_left != treedef_right or len(leaves_left) != len(leaves_right):
        return float("inf")
    errors = []
    for a, b in zip(leaves_left, leaves_right, strict=True):
        array_a = np.asarray(a)
        array_b = np.asarray(b)
        if array_a.size:
            errors.append(float(np.max(np.abs(array_a - array_b))))
    return max(errors, default=0.0)


def loss_bytes(values: list[float]) -> list[str]:
    return [np.asarray(value, dtype=np.float64).tobytes().hex() for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in (args.checkpoint, args.output):
        if not str(path).startswith(("/tmp/", "/private/tmp/")):
            raise ValueError("real resume artifacts must remain under /tmp")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: real Heat3D resume gate requires JAX CUDA")

    import importlib.util

    script_path = ROOT / "scripts" / "profile_v7_g2_heat3d_epoch.py"
    spec = importlib.util.spec_from_file_location("g2_e2_epoch", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load epoch preparation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = module._prepare(args)
    from rigno.heat3d_training import V7FormalTrainer, block_until_ready
    from rigno.heat3d_training.resume import atomic_latest_checkpoint, load_latest_checkpoint

    batch = prepared["train_batches"][0]
    initial_params = prepared["state"].params
    dependencies = prepared["trainer"].dependencies
    key_base = jax.random.PRNGKey(args.seed)
    keys = [jax.random.fold_in(key_base, index) for index in (1, 2)]

    continuous_trainer = V7FormalTrainer(dependencies, jit_cache=True)
    continuous_state = continuous_trainer.initialize(initial_params)
    continuous_losses = []
    for key in keys:
        result = continuous_trainer.step(continuous_state, batch, key)
        continuous_state = result.state
        block_until_ready((continuous_state.params, continuous_state.optimizer_state, result.loss))
        continuous_losses.append(float(np.asarray(result.loss)))

    split_trainer = V7FormalTrainer(dependencies, jit_cache=True)
    split_state = split_trainer.initialize(initial_params)
    first = split_trainer.step(split_state, batch, keys[0])
    split_state = first.state
    block_until_ready((split_state.params, split_state.optimizer_state, first.loss))
    runner_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_sha = sha256(args.heat3d_config)
    data_sha = sha256(args.labels_root / "label_generation_receipt.json")
    checkpoint = atomic_latest_checkpoint(
        args.checkpoint,
        state=split_state,
        metadata={
            "epoch": 1,
            "global_update_count": 1,
            "best_metric": float("inf"),
            "best_epoch": None,
            "runner_sha": runner_sha,
            "config_sha": config_sha,
            "data_sha": data_sha,
            "seed": args.seed,
            "test_access": False,
            "qualification_only": True,
        },
        rng_state={"jax_key_index": 1},
        batch_state={"batch_id": batch.batch_id, "next_key_index": 2},
        scheduler_state={"schedule": "embedded_in_optax", "epoch": 1},
    )
    loaded = load_latest_checkpoint(
        args.checkpoint,
        expected_runner_sha=runner_sha,
        expected_config_sha=config_sha,
        expected_data_sha=data_sha,
    )
    reload_state = loaded["state_object"]
    checkpoint_state_param_diff = tree_diff(split_state.params, reload_state.params)
    checkpoint_state_opt_diff = tree_diff(split_state.optimizer_state, reload_state.optimizer_state)
    second = split_trainer.step(reload_state, batch, keys[1])
    resumed_state = second.state
    block_until_ready((resumed_state.params, resumed_state.optimizer_state, second.loss))
    resumed_losses = [float(np.asarray(first.loss)), float(np.asarray(second.loss))]

    mismatch_rejected = False
    try:
        load_latest_checkpoint(args.checkpoint, expected_runner_sha=runner_sha, expected_config_sha="mismatch", expected_data_sha=data_sha)
    except ValueError:
        mismatch_rejected = True
    params_diff = tree_diff(continuous_state.params, resumed_state.params)
    optimizer_diff = tree_diff(continuous_state.optimizer_state, resumed_state.optimizer_state)
    losses_equal = loss_bytes(continuous_losses) == loss_bytes(resumed_losses)
    status = "EXACT_RESUME_EQUIVALENCE_PASS" if params_diff == 0.0 and optimizer_diff == 0.0 and losses_equal and checkpoint_state_param_diff == 0.0 and checkpoint_state_opt_diff == 0.0 and mismatch_rejected else "EXACT_RESUME_EQUIVALENCE_FAIL_CLOSED"
    payload = {
        "schema_version": "heat3d_v7_g2_e2_real_heat3d_resume_v1",
        "status": status,
        "model": "actual_Heat3D_RIGNO_frozen_contract",
        "fixture": {"train_batch_id": batch.batch_id, "batch_size": 24, "optimizer_updates": 2, "continuous_vs_split": True},
        "continuous": {"loss_bytes": loss_bytes(continuous_losses), "final_step": continuous_state.step},
        "resumed": {"loss_bytes": loss_bytes(resumed_losses), "final_step": resumed_state.step},
        "differences": {"params_max_abs": params_diff, "optimizer_state_max_abs": optimizer_diff, "checkpoint_param_reload_max_abs": checkpoint_state_param_diff, "checkpoint_optimizer_reload_max_abs": checkpoint_state_opt_diff, "loss_trajectory_bitwise_equal": losses_equal, "provenance_mismatch_rejected": mismatch_rejected},
        "checkpoint": {**checkpoint, "runner_sha": runner_sha, "config_sha": config_sha, "data_sha": data_sha},
        "accuracy_used": False,
        "test_or_sealed_access": False,
        "publication_resume_claim_allowed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if status == "EXACT_RESUME_EQUIVALENCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
