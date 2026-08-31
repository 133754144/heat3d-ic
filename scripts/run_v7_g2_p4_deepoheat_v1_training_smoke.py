#!/usr/bin/env python3
"""One-step upstream-faithful DeepOHeat-v1 volumetric training-path smoke.

The function batch is reduced to one official training function, but the
101x101x56 spatial mesh, PDE, piecewise conductivity/source terms, Robin/side
BCs, model architecture, physics loss, Adam optimizer, and exponential LR
schedule remain exactly those in upstream ``heat_volumetric.py``. All outputs
must live under /tmp. This is not convergence or reproduction evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


UPSTREAM_SHA = "3ef3d9c41666a56b5940b39a61166ccaa5aaedb2"
SOURCE_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def finite_tree(value: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(value))


def tree_l2(value: Any) -> float:
    return float(
        np.sqrt(
            sum(
                float(np.sum(np.square(np.asarray(leaf, dtype=np.float64))))
                for leaf in jax.tree_util.tree_leaves(value)
                if np.asarray(leaf).size
            )
        )
    )


def tree_max_abs(left: Any, right: Any) -> float:
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    if left_def != right_def:
        return float("inf")
    differences = []
    for a, b in zip(left_leaves, right_leaves, strict=True):
        av, bv = np.asarray(a), np.asarray(b)
        if av.shape != bv.shape:
            return float("inf")
        if av.size:
            differences.append(float(np.max(np.abs(av - bv))))
    return max(differences, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=97)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--resume-postprocess",
        action="store_true",
        help="Verify an already-created one-step checkpoint without another optimizer step.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not str(output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("DeepOHeat-v1 smoke artifacts must remain under /tmp")
    output.mkdir(parents=True, exist_ok=True)
    root = args.upstream_root.resolve()
    if sha256(args.fs_train) != SOURCE_SHA256:
        raise ValueError("official fs_train_volume.npy SHA mismatch")
    sys.path.insert(0, str(root))
    from heat_volumetric import apply_model_deepoheat_st  # type: ignore
    from models import DeepOHeat_v1  # type: ignore
    from train import update as upstream_update  # type: ignore

    source = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    function = jnp.asarray(np.asarray(source[args.source_index], dtype=np.float32).reshape(1, -1))
    x = jnp.linspace(0, 1, 101).reshape(-1, 1)
    y = jnp.linspace(0, 1, 101).reshape(-1, 1)
    z = jnp.linspace(0, 0.55, 56).reshape(-1, 1)
    key = jax.random.PRNGKey(42)
    # Current Equinox/Optax cannot align optimizer state with the extra
    # ``eqx.filter_jit(model)`` wrapper used in the 2025 script. The upstream
    # loss is itself filter-jitted, so keeping the plain module changes no
    # model/loss numerics and restores the parameter/gradient pytree contract.
    model = DeepOHeat_v1(
        dim=3,
        branch_dim=101**2,
        field_dim=1,
        branch_depth=8,
        branch_hidden=256,
        trunk_depth=3,
        trunk_hidden=64,
        rank=128,
        key=key,
    )
    # Integer ChebyKAN ``arange`` buffers are arrays but not differentiable.
    # New Optax requires optimizer state to match the inexact gradient tree;
    # filtering them out is a dependency compatibility fix, not a trainable
    # parameter change.
    params = eqx.filter(model, eqx.is_inexact_array)
    parameter_count = sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(params))
    schedule = optax.exponential_decay(1.0e-3, 1000, 0.9)
    optimizer = optax.adam(schedule)
    checkpoint_path = output / "DeepOHeat_v1_one_step.eqx"
    template = DeepOHeat_v1(
        dim=3,
        branch_dim=101**2,
        field_dim=1,
        branch_depth=8,
        branch_hidden=256,
        trunk_depth=3,
        trunk_hidden=64,
        rank=128,
        key=key,
    )
    if args.resume_postprocess:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "--resume-postprocess requires the existing one-step checkpoint"
            )
        # The earlier invocation completed and serialised one optimizer step;
        # only rebuild loss/gradient and reload evidence here. No optimizer is
        # initialized or applied in this path.
        updated_model = eqx.tree_deserialise_leaves(checkpoint_path, template)
        start = time.perf_counter()
        loss, gradients = apply_model_deepoheat_st(updated_model, x, y, z, function)
        jax.block_until_ready((loss, gradients))
        step_seconds = time.perf_counter() - start
        updated_opt_state = ()
    else:
        opt_state = optimizer.init(params)
        start = time.perf_counter()
        loss, gradients = apply_model_deepoheat_st(model, x, y, z, function)
        updated_model, updated_opt_state = upstream_update(
            gradients, optimizer, opt_state, model
        )
        jax.block_until_ready((loss, updated_model))
        step_seconds = time.perf_counter() - start
        eqx.tree_serialise_leaves(checkpoint_path, updated_model)
    updated_params = eqx.filter(updated_model, eqx.is_inexact_array)
    updates = jax.tree_util.tree_map(
        lambda before, after: None if before is None else after - before,
        params,
        updated_params,
        is_leaf=lambda value: value is None,
    )
    reloaded_model = eqx.tree_deserialise_leaves(checkpoint_path, template)
    reload_error = tree_max_abs(
        updated_params, eqx.filter(reloaded_model, eqx.is_inexact_array)
    )
    receipt = {
        "schema_version": "heat3d_v7_g2_p4_deepoheat_v1_original_training_path_smoke_v1",
        "status": (
            "PASS_ONE_FUNCTION_ONE_FULL_MESH_PHYSICS_STEP"
            if np.isfinite(float(loss))
            and finite_tree(gradients)
            and finite_tree(updated_opt_state)
            and tree_l2(updates) > 0.0
            and reload_error == 0.0
            else "FAIL"
        ),
        "upstream": f"xlyu0127/DeepOHeat-v1@{UPSTREAM_SHA}",
        "source": {
            "file_sha256": SOURCE_SHA256,
            "index": args.source_index,
            "function_batch": 1,
        },
        "spatial_physics": {
            "mesh": [101, 101, 56],
            "mesh_reduced": False,
            "PDE_or_BC_changed": False,
            "loss": "upstream apply_model_deepoheat_st full PDE plus top/bottom Robin plus side adiabatic",
        },
        "model": {
            "architecture": "DeepOHeat_v1 separable operator with ChebyKAN trunks",
            "branch": "8 layers width 256 over 101^2 power function",
            "trunk": "3 layers width 64 rank 128 per axis",
            "parameter_count": parameter_count,
        },
        "optimizer": {
            "name": "optax_adam",
            "lr": 0.001,
            "schedule": "exponential_decay_transition_steps_1000_decay_rate_0.9",
            "optimizer_step_count": 1,
        },
        "evidence": {
            "loss_finite": bool(np.isfinite(float(loss))),
            "gradient_finite": finite_tree(gradients),
            "gradient_l2": tree_l2(gradients),
            "update_l2": tree_l2(updates),
            "step_wall_seconds_including_first_JIT": step_seconds,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_reload_max_abs_difference": reload_error,
            "postprocess_only_invocation": args.resume_postprocess,
            "optimizer_step_not_repeated": args.resume_postprocess,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "optax": optax.__version__,
            "device": str(jax.devices()[0]),
            "compatibility_patch": "omit redundant outer eqx.filter_jit model wrapper and initialize Optax on inexact/trainable arrays only; upstream loss, module, PDE/BC and Adam semantics are unchanged",
        },
        "epochs": 0,
        "pipeline_steps": 1,
        "convergence_or_reproduction_accuracy_claim": False,
        "formal_or_long_training_started": False,
        "p1i_test_or_sealed_access": False,
        "temporary_artifact_root": str(output),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
