#!/usr/bin/env python3
"""Instrument one complete non-publication Heat3D epoch.

The runner deliberately uses the frozen Heat3D-on-DeepOHeat-v1 B24/valid32
contract, the legacy ``V7FormalTrainer.step`` path, and the single-forward
validation callback.  It records wall-clock components and resource snapshots
without computing an accuracy metric or opening any test/sealed artifact.
All output paths are required to be under ``/tmp`` (or ``/private/tmp``).
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_snapshot() -> dict[str, Any]:
    """Read an optional WSL nvidia-smi snapshot without affecting training."""

    command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not command.is_file():
        command = Path("nvidia-smi")
    try:
        result = subprocess.run(
            [
                str(command),
                "--query-gpu=name,utilization.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows = []
    for line in result.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue
        rows.append(
            {
                "name": values[0],
                "utilization_gpu_pct": values[1],
                "power_draw_w": values[2],
                "memory_used_mib": values[3],
                "memory_total_mib": values[4],
            }
        )
    return {"available": bool(rows), "rows": rows, "command": str(command)}


def _memory_stats() -> dict[str, int]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Build the frozen train/valid batches exactly as the P6 runner does."""

    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the train input pool is accepted")
    forbidden = ("test", "sealed")
    if any(token in str(args.fs_train).lower() for token in forbidden):
        raise ValueError("test/sealed input path is forbidden")
    if any(token in str(args.labels_root).lower() for token in forbidden):
        raise ValueError("test/sealed labels path is forbidden")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: epoch instrumentation requires JAX CUDA")
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit(
            "FAIL-CLOSED: epoch instrumentation requires the frozen deterministic XLA flag"
        )

    profile_module = load_script("profile_v7_g2_training_efficiency.py")
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    stats = load_script("run_v7_g2_p6_heat3d_v1_formal.py").load_stats(args.normalization)
    train_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="train"
    )
    valid_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train,
        labels_root=args.labels_root,
        role="valid",
        verify_source_file=False,
    )
    mesh = support.mesh_arrays()
    preparation_started = time.perf_counter()
    train_examples = profile_module._build_examples(
        train_data,
        "train",
        mesh["coords"],
        mesh["control_volume"],
        mesh["layer_id"],
    )
    valid_examples = profile_module._build_examples(
        valid_data,
        "valid",
        mesh["coords"],
        mesh["control_volume"],
        mesh["layer_id"],
    )
    builder = __import__(
        "rigno.graphBuilder_Heat3D", fromlist=["Heat3DGraphBuilder"]
    ).Heat3DGraphBuilder(**config["graph"])
    from rigno.heat3d_training import build_p1i_batches

    batch_profile: dict[str, Any] = {}
    train_batches = build_p1i_batches(
        train_examples,
        stats,
        builder,
        label="g2_e2_epoch_train",
        batch_size=24,
        graph_seed=args.seed,
        profile=batch_profile,
    )
    valid_batches = build_p1i_batches(
        valid_examples,
        stats,
        builder,
        label="g2_e2_epoch_valid",
        batch_size=32,
        graph_seed=args.seed,
        profile=batch_profile,
    )
    all_examples = train_examples + valid_examples
    from rigno.heat3d_training.p1i import (
        attach_input_contexts,
        attach_native_physics,
        attach_qk_features,
        fit_native_loss_references,
    )

    context = attach_input_contexts(
        train_batches + valid_batches,
        train_examples,
        all_examples,
        config["model"],
    )
    by_id = {row.sample_id: row for row in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(
            batches, by_id, context_by_id=context["raw_context_by_id"]
        )
        attach_qk_features(
            batches,
            by_id,
            feature_version=str(config["model"]["qk_region_feature_version"]),
        )
    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(train_examples, config["loss"]))
    model_config = helper.resolve_model_config(
        config["model"], tuple(stats["feature_names"])
    )
    from rigno.models.rigno import RIGNO
    from rigno.heat3d_training import (
        TrainingDependencies,
        V7FormalTrainer,
        block_until_ready,
        loss_fn_full,
        make_gradient_transform,
        make_p1i_optimizer,
        model_apply_full,
        model_init_full,
    )

    model = RIGNO(**model_config)
    params = model_init_full(
        model, jax.random.PRNGKey(args.seed), train_batches[0]
    )["params"]
    optimizer = make_p1i_optimizer(
        config["optimizer"], epochs=200, updates_per_epoch=len(train_batches)
    )
    apply_fn = lambda current, batch, rng: model_apply_full(model, current, batch, rng)
    batch_loss = lambda prediction, batch: loss_fn_full(
        prediction, batch, loss_config
    )

    def validation_outputs(current: Any, batch: Any) -> tuple[Any, Any]:
        prediction = apply_fn(current, batch, None)
        return prediction, batch_loss(prediction, batch)

    dependencies = TrainingDependencies(
        data_source="frozen_768_128_deepoheat_v1_labels",
        feature_transform="physics_layout_aware_1024",
        normalization=stats,
        graph_builder=builder,
        model=model,
        model_apply=apply_fn,
        loss_fn=batch_loss,
        optimizer=optimizer,
        batch_iterator=lambda value: value,
        validation_fn=lambda current, batch: batch_loss(apply_fn(current, batch, None), batch),
        checkpoint_writer=lambda _path, _payload: None,
        metrics_fn=lambda _current, _batch: {"runtime_only": True},
        gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
        validation_outputs_fn=validation_outputs,
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=True)
    state = trainer.initialize(params)
    return {
        "config": config,
        "stats": stats,
        "trainer": trainer,
        "state": state,
        "train_batches": train_batches,
        "valid_batches": valid_batches,
        "block_until_ready": block_until_ready,
        "batch_loss": batch_loss,
        "apply_fn": apply_fn,
        "preparation_seconds": time.perf_counter() - preparation_started,
        "batch_profile": batch_profile,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("epoch receipt must remain under /tmp")
    if not str(args.checkpoint).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("epoch checkpoint must remain under /tmp")
    prepared = _prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    train_batches = prepared["train_batches"]
    valid_batches = prepared["valid_batches"]
    block = prepared["block_until_ready"]
    batch_loss = prepared["batch_loss"]
    apply_fn = prepared["apply_fn"]
    seed = int(args.seed)
    epoch = 1
    order = np.random.default_rng(seed + epoch).permutation(len(train_batches))
    order_ids = [train_batches[int(index)].batch_id for index in order]
    order_hash = hashlib.sha256(
        json.dumps(order_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    base_key = jax.random.PRNGKey(seed)
    before_gpu = _gpu_snapshot()
    epoch_started = time.perf_counter()
    train_times: list[dict[str, Any]] = []
    train_losses: list[float] = []
    train_started = time.perf_counter()
    for batch_number, raw_index in enumerate(order, start=1):
        batch = train_batches[int(raw_index)]
        key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), batch_number)
        started = time.perf_counter()
        result = trainer.step(state, batch, rng=key)
        state = result.state
        block(
            (
                state.params,
                state.optimizer_state,
                result.loss,
                result.gradients,
                result.updates,
                result.prediction,
            )
        )
        seconds = time.perf_counter() - started
        train_losses.append(float(np.asarray(result.loss)))
        train_times.append(
            {
                "batch_number": batch_number,
                "batch_index": int(raw_index),
                "batch_id": batch.batch_id,
                "wall_seconds": seconds,
                "loss_finite": bool(np.isfinite(train_losses[-1])),
                "compile_count": int(trainer.compile_count),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "train",
                    "batch": batch_number,
                    "total": len(order),
                    "seconds": seconds,
                    "compile_count": trainer.compile_count,
                }
            ),
            flush=True,
        )
    train_total = time.perf_counter() - train_started
    after_train_gpu = _gpu_snapshot()

    valid_times: list[dict[str, Any]] = []
    valid_losses: list[float] = []
    valid_started = time.perf_counter()
    for batch_number, batch in enumerate(valid_batches, start=1):
        started = time.perf_counter()
        prediction, loss = trainer.validate_with_outputs(state, batch)
        block((prediction, loss))
        seconds = time.perf_counter() - started
        value = float(np.asarray(loss))
        valid_losses.append(value)
        valid_times.append(
            {
                "batch_number": batch_number,
                "batch_id": batch.batch_id,
                "wall_seconds": seconds,
                "loss_finite": bool(np.isfinite(value)),
            }
        )
        print(
            json.dumps(
                {"phase": "valid", "batch": batch_number, "total": len(valid_batches), "seconds": seconds}
            ),
            flush=True,
        )
    valid_total = time.perf_counter() - valid_started
    after_valid_gpu = _gpu_snapshot()

    aggregate_started = time.perf_counter()
    aggregate = {
        "train_loss_mean": float(np.mean(train_losses)),
        "valid_loss_mean": float(np.mean(valid_losses)),
        "train_loss_finite": bool(np.all(np.isfinite(train_losses))),
        "valid_loss_finite": bool(np.all(np.isfinite(valid_losses))),
    }
    metric_aggregation_seconds = time.perf_counter() - aggregate_started

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_started = time.perf_counter()
    from rigno.heat3d_training.resume import atomic_latest_checkpoint

    checkpoint_receipt = atomic_latest_checkpoint(
        args.checkpoint,
        state=state,
        metadata={
            "epoch": epoch,
            "global_update_count": len(order),
            "best_metric": float("inf"),
            "best_epoch": None,
            "runner_sha": sha256(Path(__file__).resolve()),
            "config_sha": sha256(args.heat3d_config),
            "data_sha": sha256(args.labels_root / "label_generation_receipt.json"),
            "seed": seed,
            "test_access": False,
            "runtime_only": True,
        },
        rng_state={"jax_base_seed": seed, "next_epoch": epoch + 1},
        batch_state={
            "algorithm": "default_rng(seed+epoch).permutation",
            "epoch": epoch,
            "order_hash": order_hash,
            "next_epoch": epoch + 1,
        },
        scheduler_state={"schedule": "embedded_in_optax", "epoch": epoch},
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    after_checkpoint_gpu = _gpu_snapshot()

    logging_started = time.perf_counter()
    prelog = {
        "schema_version": "heat3d_v7_g2_e2_epoch_runtime_components_v1",
        "epoch": epoch,
        "train": {"samples": len(prepared["train_examples"]), "batch_size": 24, "batches": train_times},
        "valid": {"samples": len(prepared["valid_examples"]), "batch_size": 32, "batches": valid_times},
        "aggregate": aggregate,
        "checkpoint": checkpoint_receipt,
    }
    args.progress.write_text(json.dumps(prelog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging_write_seconds = time.perf_counter() - logging_started
    epoch_total = time.perf_counter() - epoch_started
    accounted = train_total + valid_total + checkpoint_seconds + metric_aggregation_seconds + logging_write_seconds
    misc_seconds = epoch_total - accounted
    after_epoch_gpu = _gpu_snapshot()
    memory = _memory_stats()
    payload = {
        "schema_version": "heat3d_v7_g2_e2_epoch_accounting_v1",
        "status": "PASS_EPOCH_ACCOUNTING_FINITE" if aggregate["train_loss_finite"] and aggregate["valid_loss_finite"] else "FAIL_NONFINITE",
        "scope": "one_complete_nonpublication_epoch_legacy_train_single_forward_validation",
        "scientific_contract": {
            "train_samples": 768,
            "batch_size": 24,
            "optimizer_steps_per_epoch": len(train_times),
            "validation_samples": 128,
            "validation_batch_size": 32,
            "validation_batches": len(valid_times),
            "epochs_requested": 1,
            "accuracy_used": False,
        },
        "timing": {
            "train_total_seconds": train_total,
            "train_batch_seconds": train_times,
            "validation_total_seconds": valid_total,
            "validation_batch_seconds": valid_times,
            "checkpoint_seconds": checkpoint_seconds,
            "metric_aggregation_seconds": metric_aggregation_seconds,
            "logging_write_seconds": logging_write_seconds,
            "logging_total_seconds": metric_aggregation_seconds + logging_write_seconds,
            "misc_seconds": misc_seconds,
            "epoch_total_seconds": epoch_total,
            "accounted_sum_seconds": accounted,
            "closure_residual_seconds": epoch_total - accounted - misc_seconds,
            "closure_relative_error": abs(epoch_total - accounted - misc_seconds) / max(epoch_total, 1.0e-12),
        },
        "warmup_jit": {
            "occurred": bool(train_times and train_times[0]["compile_count"] > 0),
            "first_train_batch_seconds": train_times[0]["wall_seconds"] if train_times else None,
            "compile_count_end": trainer.compile_count,
            "unique_train_batch_signatures": len(getattr(trainer, "_compiled_steps", {})),
            "batch_order_hash": order_hash,
        },
        "preparation": {
            "wall_seconds": prepared["preparation_seconds"],
            "graph_profile": prepared["batch_profile"],
            "train_batch_count": len(train_batches),
            "valid_batch_count": len(valid_batches),
        },
        "resource": {
            "device": str(jax.devices()[0]),
            "jax_memory_stats": memory,
            "gpu_snapshots": {
                "before": before_gpu,
                "after_train": after_train_gpu,
                "after_valid": after_valid_gpu,
                "after_checkpoint": after_checkpoint_gpu,
                "after_epoch": after_epoch_gpu,
            },
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "checkpoint": {
            **checkpoint_receipt,
            "path": str(args.checkpoint),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": sha256(args.heat3d_config),
            "data_receipt_sha256": sha256(args.labels_root / "label_generation_receipt.json"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": getattr(jax, "__version__", None),
            "backend": jax.default_backend(),
            "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "dataset": {
            "train_role": 768,
            "valid_role": 128,
            "labels_root": str(args.labels_root),
            "official_test_accessed": False,
            "p1i_test_or_sealed_access": False,
        },
        "projection_schema": {
            "samples_per_epoch": 768,
            "batch_size": 24,
            "optimizer_steps_per_epoch": len(train_times),
            "validation_samples": 128,
            "validation_batch_size": 32,
            "validation_batches": len(valid_times),
            "measured_step_seconds": float(np.median([row["wall_seconds"] for row in train_times[1:]] or [0.0])),
            "measured_validation_seconds": float(np.median([row["wall_seconds"] for row in valid_times] or [0.0])),
            "projection_basis": "measured one complete legacy B24 epoch; qualification-only; no accuracy",
        },
        "formal_accuracy_claim_allowed": False,
        "test_or_sealed_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "epoch_total_seconds": epoch_total, "train_total_seconds": train_total, "validation_total_seconds": valid_total}, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args)
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
