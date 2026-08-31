#!/usr/bin/env python3
"""Test-isolated DeepOHeat-v1 CUDA preflight and formal training.

Only the official 100,000-function training pool is an accepted data input.
The runner installs an audit guard that rejects either official test filename,
does not import the evaluation routine, and serializes only the final model for
formal training. ``preflight`` performs one full-mesh, batch-50 update without
writing a model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = "3ef3d9c41666a56b5940b39a61166ccaa5aaedb2"
FS_TRAIN_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
FORBIDDEN_TEST_FILES = {"fs_test_volume.npy", "u_test_volume.npy"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def finite_tree(value: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(value)
    )


def install_test_file_guard() -> None:
    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(candidate)).name
            if name in FORBIDDEN_TEST_FILES:
                raise PermissionError(
                    f"FAIL-CLOSED: formal training attempted to open sealed {name}"
                )

    sys.addaudithook(audit)


def accelerator_receipt() -> dict[str, Any]:
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise SystemExit(
            "FAIL-CLOSED: DeepOHeat-v1 preflight/formal training requires JAX CUDA"
        )
    stats = devices[0].memory_stats() or {}
    return {
        "device": str(devices[0]),
        "platform": devices[0].platform,
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_in_use": stats.get("bytes_in_use"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract-check", "preflight", "train"), required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--fs-train", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    upstream = args.upstream_root.resolve()
    if git_sha(upstream) != UPSTREAM_SHA:
        raise ValueError("DeepOHeat-v1 upstream SHA mismatch")
    if args.mode == "contract-check":
        print(json.dumps({
            "status": "PASS_TRAIN_ONLY_CONTRACT_NO_DATA_NO_TRAINING",
            "upstream_sha": UPSTREAM_SHA,
            "accepted_data": "fs_train_volume.npy only",
            "forbidden_test_files": sorted(FORBIDDEN_TEST_FILES),
            "formal_checkpoint": "final only",
        }, indent=2, sort_keys=True))
        return 0
    if args.fs_train is None or args.output_dir is None:
        parser.error("preflight/train require --fs-train and --output-dir")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the official fs_train_volume.npy filename is accepted")
    if sha256(args.fs_train) != FS_TRAIN_SHA256:
        raise ValueError("official fs_train_volume.npy SHA mismatch")

    install_test_file_guard()
    device_before = accelerator_receipt()
    sys.path.insert(0, str(upstream))
    from heat_volumetric import (  # type: ignore
        apply_model_deepoheat_st,
        deepoheat_st_train_generator,
    )
    from models import DeepOHeat_v1  # type: ignore
    from train import update as upstream_update  # type: ignore

    fs_train = jnp.asarray(
        np.asarray(np.load(args.fs_train, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    ).reshape(-1, 101**2)
    if fs_train.shape != (100000, 101**2):
        raise ValueError(f"official training-pool shape mismatch: {fs_train.shape}")
    key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key, 2)
    model = DeepOHeat_v1(
        dim=3, branch_dim=101**2, field_dim=1,
        branch_depth=8, branch_hidden=256,
        trunk_depth=3, trunk_hidden=64, rank=128, key=model_key,
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    parameter_count = sum(
        int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(params)
    )
    optimizer = optax.adam(optax.exponential_decay(1.0e-3, 1000, 0.9))
    opt_state = optimizer.init(params)
    key, train_key = jax.random.split(key)
    iterations = 1 if args.mode == "preflight" else 100000
    started = time.perf_counter()
    first_step_seconds = None
    last_loss = None
    for iteration in range(iterations):
        train_key, sample_key = jax.random.split(train_key)
        inputs = deepoheat_st_train_generator(fs_train, 50, 101, sample_key)
        step_started = time.perf_counter()
        loss, gradients = apply_model_deepoheat_st(model, *inputs)
        model, opt_state = upstream_update(gradients, optimizer, opt_state, model)
        jax.block_until_ready((loss, model))
        if first_step_seconds is None:
            first_step_seconds = time.perf_counter() - step_started
        if not np.isfinite(float(loss)) or not finite_tree(gradients):
            raise RuntimeError(f"FAIL-CLOSED: nonfinite loss/gradient at iteration {iteration + 1}")
        last_loss = float(loss)
        if iteration % 100 == 0:
            print(f"iteration={iteration + 1}/{iterations} physics_loss={last_loss:.9g}", flush=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.mode == "train":
        checkpoint = output / "DeepOHeat_v1_final.eqx"
        eqx.tree_serialise_leaves(checkpoint, model)
    device_after = accelerator_receipt()
    receipt = {
        "schema_version": "heat3d_v7_g2_p6_deepoheat_v1_formal_v1",
        "status": "PASS_CUDA_FULL_PHYSICS_PREFLIGHT" if args.mode == "preflight" else "COMPLETE_FORMAL_TRAIN_FINAL_CHECKPOINT_FROZEN",
        "mode": args.mode,
        "upstream": f"xlyu0127/DeepOHeat-v1@{UPSTREAM_SHA}",
        "training_data": {"file": "fs_train_volume.npy", "sha256": FS_TRAIN_SHA256, "functions": 100000},
        "test_isolation": {
            "training_test_files_loaded": False,
            "forbidden_filenames": sorted(FORBIDDEN_TEST_FILES),
            "official_100_test_status": "sealed",
            "evaluation_imported_or_executed": False,
        },
        "physics": {"mesh": [101, 101, 56], "batch_functions": 50, "PDE_BC_or_mesh_changed": False},
        "optimizer": {"name": "Optax Adam", "lr": 0.001, "schedule": "exponential_decay_0.9_per_1000"},
        "iterations": iterations,
        "seed": 42,
        "parameter_count": parameter_count,
        "last_physics_loss": last_loss,
        "resource": {
            "first_train_step_wall_seconds": first_step_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "before": device_before,
            "after": device_after,
        },
        "environment": {
            "python": platform.python_version(), "jax": jax.__version__,
            "equinox": eqx.__version__, "optax": optax.__version__,
        },
        "checkpoint": None if checkpoint is None else {
            "policy": "final_only_no_valid_or_test_selection",
            "file": checkpoint.name,
            "sha256": sha256(checkpoint),
        },
    }
    receipt_path = output / "preflight_receipt.json" if args.mode == "preflight" else output / "formal_training_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
