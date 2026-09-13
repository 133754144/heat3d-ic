#!/usr/bin/env python3
"""Bounded deterministic-XLA A/B for one frozen Heat3D B24 batch.

Each invocation runs one compile/warmup call followed by three synchronized
steady-state updates.  The only changed variable is the process-level
``--xla_gpu_deterministic_ops`` flag; model, batch, seed, objective, optimizer,
and normalization remain frozen.  No accuracy or test/sealed artifact is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: deterministic A/B requires devbox CUDA")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the frozen train input pool is accepted")
    if any(token in str(path).lower() for path in (args.fs_train, args.labels_root, args.normalization) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    module = load_script("profile_v7_g2_heat3d_epoch.py")
    # The opt-in is used solely to permit the explicit false-flag diagnostic;
    # it is never set by the formal runner or by the frozen baseline profile.
    setattr(args, "allow_nondeterministic_xla", True)
    prepared = module._prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    batch = prepared["train_batches"][0]
    block = prepared["block_until_ready"]
    base_key = jax.random.PRNGKey(args.seed)
    key = jax.random.fold_in(base_key, 1)
    compile_started = time.perf_counter()
    first = trainer.step(state, batch, key)
    state = first.state
    block((state.params, state.optimizer_state, first.loss))
    compile_seconds = time.perf_counter() - compile_started
    warm_rows = []
    losses = [float(np.asarray(first.loss))]
    for step_index in range(2, 5):
        started = time.perf_counter()
        result = trainer.step(state, batch, key)
        state = result.state
        block((state.params, state.optimizer_state, result.loss))
        seconds = time.perf_counter() - started
        value = float(np.asarray(result.loss))
        losses.append(value)
        warm_rows.append({"step": step_index, "wall_seconds": seconds, "loss": value, "loss_finite": bool(np.isfinite(value))})
    memory = jax.devices()[0].memory_stats() or {}
    payload = {
        "schema_version": "heat3d_v7_g2_e2_deterministic_xla_ab_v1",
        "status": "PASS_FINITE",
        "deterministic_flag": "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""),
        "scope": "one_frozen_B24_batch_one_compile_plus_three_postwarmup_steps",
        "scientific_contract": {
            "batch_size": 24,
            "seed": int(args.seed),
            "same_initial_state_and_key": True,
            "model_loss_optimizer_normalization_changed": False,
            "accuracy_used": False,
            "test_or_sealed_access": False,
            "formal_training_started": False,
        },
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "execution": {
            "compile_plus_first_step_seconds": compile_seconds,
            "postwarmup_steps": warm_rows,
            "median_postwarmup_step_seconds": float(np.median([row["wall_seconds"] for row in warm_rows])),
            "p95_postwarmup_step_seconds": float(np.percentile([row["wall_seconds"] for row in warm_rows], 95)),
            "samples_per_second_postwarmup": float(24.0 / np.median([row["wall_seconds"] for row in warm_rows])),
            "compile_count": int(trainer.compile_count),
            "host_scalar_extractions": 3,
            "explicit_sync_calls": 4,
        },
        "trajectory_probe": {
            "losses_are_finite_only_sanity": True,
            "loss_sequence": losses,
            "final_params_tree_sha256": tree_hash(state.params),
            "final_optimizer_state_tree_sha256": tree_hash(state.optimizer_state),
        },
        "resource": {
            "device": str(jax.devices()[0]),
            "memory_stats": {str(k): int(v) for k, v in memory.items() if isinstance(v, (int, np.integer))},
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runner_sha256": hashlib.sha256((ROOT / "rigno/heat3d_training/core.py").read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "deterministic_flag": receipt["deterministic_flag"], "compile_seconds": receipt["execution"]["compile_plus_first_step_seconds"], "median_step_seconds": receipt["execution"]["median_postwarmup_step_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
