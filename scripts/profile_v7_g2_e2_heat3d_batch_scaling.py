#!/usr/bin/env python3
"""Three-epoch runtime-only Heat3D batch-scaling diagnostic.

The diagnostic is deliberately non-publication: it reads only the frozen
768-train/128-valid_iid population, computes loss finiteness (not accuracy),
and writes all artifacts under ``/tmp``.  ``control`` mirrors the current
runner's per-step synchronization and host diagnostics.  ``compiled`` uses
the same ``V7FormalTrainer.step`` JIT executable and objective, but avoids
returning diagnostics to the host until the step boundary; this measures the
execution-stack overhead without introducing a new numerical update path.

Batch sizes are restricted to divisors of 768.  Validation defaults to B32,
which divides 128.  The optimizer retains the frozen 200-epoch schedule;
only the natural updates-per-epoch value changes with the diagnostic batch
size.  No checkpoint-selection metric is evaluated.
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
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLOWED_BATCHES = (24, 48, 96, 128)


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


def gpu_snapshot() -> dict[str, Any]:
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
        if len(values) == 5:
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


def jax_memory_stats() -> dict[str, int]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def tree_finite_host(value: Any) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree_util.tree_leaves(value))


def tree_finite_device(value: Any) -> Any:
    flags = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(value)]
    result = jnp.asarray(True)
    for flag in flags:
        result = jnp.logical_and(result, flag)
    return result


def tree_l2_norm_host(value: Any) -> float:
    leaves = [np.asarray(leaf, dtype=np.float64) for leaf in jax.tree_util.tree_leaves(value)]
    return float(np.sqrt(sum(float(np.sum(np.square(leaf))) for leaf in leaves)))


def digest_sample_order(batches: list[Any], order: np.ndarray) -> str:
    sample_ids = []
    for index in order:
        sample_ids.extend(str(value) for value in batches[int(index)].sample_ids)
    return hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def require_paths(args: argparse.Namespace) -> None:
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only fs_train_volume.npy is accepted")
    forbidden = ("test", "sealed")
    for path in (args.fs_train, args.labels_root, args.normalization):
        if any(token in str(path).lower() for token in forbidden):
            raise ValueError("test/sealed path is forbidden")
    for path in (args.output, args.checkpoint, args.progress):
        if not str(path).startswith(("/tmp/", "/private/tmp/")):
            raise ValueError("diagnostic artifacts must remain under /tmp")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic artifact: {path}")
    if args.batch_size not in ALLOWED_BATCHES:
        raise ValueError(f"batch_size must be one of {ALLOWED_BATCHES}")
    if 768 % args.batch_size or 128 % args.valid_batch_size:
        raise ValueError("train/valid batch sizes must divide frozen populations")
    if args.epochs != 3:
        raise ValueError("this diagnostic is fixed to exactly three epochs")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: diagnostic requires JAX CUDA")
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit("FAIL-CLOSED: diagnostic requires deterministic XLA flag")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    module = load_script("profile_v7_g2_heat3d_epoch.py")
    # _prepare defaults remain B24/valid32 for the historical receipt; these
    # attributes activate only its bounded batch-scaling grouping path.
    setattr(args, "train_batch_size", args.batch_size)
    setattr(args, "valid_batch_size", args.valid_batch_size)
    return module._prepare(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_paths(args)
    prepared = prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    train_batches = prepared["train_batches"]
    valid_batches = prepared["valid_batches"]
    block_until_ready = prepared["block_until_ready"]
    seed = int(args.seed)
    base_key = jax.random.PRNGKey(seed)
    compile_seen: set[str] = set()
    all_step_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, int]] = []
    warmup_batch_ids: list[str] = []
    latest_checkpoint = None
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_batches))
        if len(order) != 768 // args.batch_size:
            raise AssertionError("unexpected number of complete train batches")
        epoch_started = time.perf_counter()
        before_gpu = gpu_snapshot()
        gpu_rows.append({"epoch": epoch, "phase": "before", **before_gpu})
        train_rows: list[dict[str, Any]] = []
        train_losses: list[float] = []
        train_started = time.perf_counter()
        for batch_number, raw_index in enumerate(order, start=1):
            batch = train_batches[int(raw_index)]
            first_for_batch = batch.batch_id not in compile_seen
            if first_for_batch:
                compile_seen.add(batch.batch_id)
                warmup_batch_ids.append(batch.batch_id)
            key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), batch_number)
            started = time.perf_counter()
            dispatch_started = time.perf_counter()
            result = trainer.step(state, batch, rng=key)
            dispatch_seconds = time.perf_counter() - dispatch_started
            state = result.state
            if args.mode == "control":
                sync_started = time.perf_counter()
                block_until_ready(
                    (
                        state.params,
                        state.optimizer_state,
                        result.loss,
                        result.gradients,
                        result.updates,
                        result.prediction,
                    )
                )
                sync_seconds = time.perf_counter() - sync_started
                host_started = time.perf_counter()
                loss_value = float(np.asarray(result.loss))
                grad_finite = tree_finite_host(result.gradients)
                update_finite = tree_finite_host(result.updates)
                param_finite = tree_finite_host(state.params)
                grad_norm = tree_l2_norm_host(result.gradients)
                update_norm = tree_l2_norm_host(result.updates)
                param_norm = tree_l2_norm_host(state.params)
                host_seconds = time.perf_counter() - host_started
                host_scalar_extractions = 4
            else:
                grad_finite_device = tree_finite_device(result.gradients)
                sync_started = time.perf_counter()
                block_until_ready(
                    (state.params, state.optimizer_state, result.loss, grad_finite_device)
                )
                sync_seconds = time.perf_counter() - sync_started
                host_started = time.perf_counter()
                loss_value = float(np.asarray(result.loss))
                grad_finite = bool(np.asarray(grad_finite_device))
                update_finite = None
                param_finite = None
                grad_norm = update_norm = param_norm = None
                host_seconds = time.perf_counter() - host_started
                host_scalar_extractions = 2
            step_seconds = time.perf_counter() - started
            train_losses.append(loss_value)
            train_rows.append(
                {
                    "epoch": epoch,
                    "batch_number": batch_number,
                    "batch_id": batch.batch_id,
                    "first_seen_compile_phase": first_for_batch,
                    "dispatch_seconds": dispatch_seconds,
                    "explicit_sync_seconds": sync_seconds,
                    "host_postprocess_seconds": host_seconds,
                    "wall_seconds": step_seconds,
                    "loss_finite": bool(np.isfinite(loss_value)),
                    "grad_finite": grad_finite,
                    "update_finite": update_finite,
                    "param_finite": param_finite,
                    "grad_norm": grad_norm,
                    "update_norm": update_norm,
                    "param_norm": param_norm,
                    "explicit_sync_calls": 1,
                    "host_scalar_extractions": host_scalar_extractions,
                    "compile_count": int(trainer.compile_count),
                }
            )
            all_step_rows.append(train_rows[-1])
        train_total = time.perf_counter() - train_started
        after_train_gpu = gpu_snapshot()
        gpu_rows.append({"epoch": epoch, "phase": "after_train", **after_train_gpu})

        valid_rows: list[dict[str, Any]] = []
        valid_losses: list[float] = []
        valid_started = time.perf_counter()
        for batch_number, batch in enumerate(valid_batches, start=1):
            started = time.perf_counter()
            prediction, loss = trainer.validate_with_outputs(state, batch)
            block_until_ready((prediction, loss))
            seconds = time.perf_counter() - started
            value = float(np.asarray(loss))
            valid_losses.append(value)
            valid_rows.append(
                {
                    "batch_number": batch_number,
                    "batch_id": batch.batch_id,
                    "wall_seconds": seconds,
                    "loss_finite": bool(np.isfinite(value)),
                    "explicit_sync_calls": 1,
                    "host_scalar_extractions": 1,
                }
            )
        valid_total = time.perf_counter() - valid_started
        after_valid_gpu = gpu_snapshot()
        gpu_rows.append({"epoch": epoch, "phase": "after_valid", **after_valid_gpu})

        aggregate_started = time.perf_counter()
        train_loss_mean = float(np.mean(train_losses))
        valid_loss_mean = float(np.mean(valid_losses))
        finite = bool(
            np.all(np.isfinite(train_losses))
            and np.all(np.isfinite(valid_losses))
            and all(row["grad_finite"] for row in train_rows)
            and all(row["update_finite"] is not False for row in train_rows)
            and all(row["param_finite"] is not False for row in train_rows)
        )
        aggregation_seconds = time.perf_counter() - aggregate_started

        checkpoint_started = time.perf_counter()
        from rigno.heat3d_training.resume import atomic_latest_checkpoint

        checkpoint_receipt = atomic_latest_checkpoint(
            args.checkpoint,
            state=state,
            metadata={
                "epoch": epoch,
                "global_update_count": epoch * len(train_batches),
                "best_metric": float("inf"),
                "best_epoch": None,
                "runner_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "config_sha": sha256(args.heat3d_config),
                "data_sha": sha256(args.labels_root / "label_generation_receipt.json"),
                "seed": seed,
                "test_access": False,
                "qualification_only": True,
                "diagnostic_mode": args.mode,
                "batch_size": args.batch_size,
            },
            rng_state={"jax_base_seed": seed, "next_epoch": epoch + 1},
            batch_state={
                "algorithm": "default_rng(seed+epoch).permutation",
                "next_epoch": epoch + 1,
                "batch_size": args.batch_size,
            },
            scheduler_state={
                "schedule": "frozen_200_epoch_optimizer_schedule",
                "epoch": epoch,
                "updates_per_epoch": len(train_batches),
            },
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        logging_started = time.perf_counter()
        args.progress.write_text(
            json.dumps(
                {
                    "schema_version": "g2_e2_runtime_batch_progress_v1",
                    "status": "RUNNING",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "mode": args.mode,
                    "batch_size": args.batch_size,
                    "test_or_sealed_access": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logging_seconds = time.perf_counter() - logging_started
        epoch_seconds = time.perf_counter() - epoch_started
        mem = jax_memory_stats()
        memory_rows.append({"epoch": epoch, **mem})
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_batches": len(train_batches),
                "train_samples": len(train_batches) * args.batch_size,
                "validation_batches": len(valid_batches),
                "validation_samples": len(valid_batches) * args.valid_batch_size,
                "train_total_seconds": train_total,
                "validation_total_seconds": valid_total,
                "checkpoint_seconds": checkpoint_seconds,
                "metric_aggregation_seconds": aggregation_seconds,
                "logging_seconds": logging_seconds,
                "epoch_wall_seconds": epoch_seconds,
                "train_loss_mean": train_loss_mean,
                "valid_loss_mean": valid_loss_mean,
                "loss_finite": finite,
                "grad_finite": all(row["grad_finite"] for row in train_rows),
                "warmup_compile_batches": sum(
                    1 for row in train_rows if row["first_seen_compile_phase"]
                ),
                "sample_order_sha256": digest_sample_order(train_batches, order),
                "checkpoint": checkpoint_receipt,
                "validation_rows": valid_rows,
            }
        )
        latest_checkpoint = checkpoint_receipt

    train_steps = [row["wall_seconds"] for row in all_step_rows]
    steady_steps = [
        row["wall_seconds"]
        for row in all_step_rows
        if not row["first_seen_compile_phase"]
    ]
    compile_phase_steps = [
        row["wall_seconds"]
        for row in all_step_rows
        if row["first_seen_compile_phase"]
    ]
    util_values = []
    for row in gpu_rows:
        for gpu in row.get("rows", []):
            try:
                util_values.append(float(gpu["utilization_gpu_pct"]))
            except (KeyError, TypeError, ValueError):
                pass
    memory_used = []
    for row in gpu_rows:
        for gpu in row.get("rows", []):
            try:
                memory_used.append(int(gpu["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                pass
    total_train_seconds = float(sum(row["train_total_seconds"] for row in epoch_rows))
    total_epoch_seconds = float(sum(row["epoch_wall_seconds"] for row in epoch_rows))
    total_samples = args.epochs * 768
    steady_train_seconds = float(
        sum(row["train_total_seconds"] for row in epoch_rows[1:])
    )
    explicit_sync_calls = sum(int(row["explicit_sync_calls"]) for row in all_step_rows)
    host_scalar_extractions = sum(
        int(row["host_scalar_extractions"]) for row in all_step_rows
    )
    validation_sync_calls = sum(
        len(row["validation_rows"]) for row in epoch_rows
    )
    payload = {
        "schema_version": "heat3d_v7_g2_e2_batch_scaling_runtime_v1",
        "status": "PASS_RUNTIME_FINITE" if all(row["loss_finite"] for row in epoch_rows) else "FAIL_NONFINITE",
        "scope": "three_epoch_runtime_diagnostic_only",
        "mode": args.mode,
        "scientific_contract": {
            "train_samples_per_epoch": 768,
            "batch_size": args.batch_size,
            "batches_per_epoch": len(train_batches),
            "optimizer_steps_per_epoch": len(train_batches),
            "validation_samples": 128,
            "validation_batch_size": args.valid_batch_size,
            "validation_batches": len(valid_batches),
            "optimizer_schedule_horizon_epochs": 200,
            "architecture_loss_optimizer_normalization_changed": False,
            "remainder_batch": False,
            "accuracy_used": False,
            "checkpoint_selection_used": False,
            "test_or_sealed_access": False,
        },
        "preparation": {
            "seconds": prepared["preparation_seconds"],
            "batch_profile": prepared["batch_profile"],
            "train_examples": len(prepared["train_examples"]),
            "valid_examples": len(prepared["valid_examples"]),
        },
        "execution": {
            "train_step_jit": "V7FormalTrainer.step -> jax.jit(_step_impl); value_and_grad + gradient transform + optax update + apply_updates in one executable",
            "mode_detail": (
                "control: current runner full output block + host diagnostics"
                if args.mode == "control"
                else "compiled: same full numerical step; only params/optimizer/loss/device grad-finite are synchronized; host norms deferred/omitted"
            ),
            "compile_count": int(trainer.compile_count),
            "unique_batch_signatures": len(compile_seen),
            "warmup_batch_ids": warmup_batch_ids,
            "explicit_sync_calls_train": explicit_sync_calls,
            "explicit_sync_calls_validation": validation_sync_calls,
            "host_scalar_extractions_train": host_scalar_extractions,
            "host_scalar_extractions_validation": validation_sync_calls,
        },
        "epoch_rows": epoch_rows,
        "step_summary": {
            "all_step_count": len(train_steps),
            "compile_phase_step_count": len(train_steps) - len(steady_steps),
            "post_warmup_step_count": len(steady_steps),
            "compile_phase_wall_seconds": float(sum(compile_phase_steps)),
            "median_post_warmup_seconds": percentile(steady_steps, 50),
            "p95_post_warmup_seconds": percentile(steady_steps, 95),
            "median_all_seconds": percentile(train_steps, 50),
            "p95_all_seconds": percentile(train_steps, 95),
            "dispatch_seconds_total": float(sum(row["dispatch_seconds"] for row in all_step_rows)),
            "explicit_sync_seconds_total": float(sum(row["explicit_sync_seconds"] for row in all_step_rows)),
            "host_postprocess_seconds_total": float(sum(row["host_postprocess_seconds"] for row in all_step_rows)),
        },
        "throughput": {
            "all_three_epoch_samples_per_second": total_samples / max(total_train_seconds, 1.0e-12),
            "post_warmup_samples_per_second": (2 * 768) / max(steady_train_seconds, 1.0e-12),
            "epoch_wall_seconds_median": percentile([row["epoch_wall_seconds"] for row in epoch_rows], 50),
            "validation_seconds_median": percentile([row["validation_total_seconds"] for row in epoch_rows], 50),
        },
        "time_split": {
            "forward_loss_backward_grad_optimizer_update": "inside one JIT; individual kernel components not separately observable without profiler",
            "dispatch_seconds_total": float(sum(row["dispatch_seconds"] for row in all_step_rows)),
            "explicit_sync_plus_host_postprocess_seconds_total": float(
                sum(row["explicit_sync_seconds"] + row["host_postprocess_seconds"] for row in all_step_rows)
            ),
            "checkpoint_seconds_total": float(sum(row["checkpoint_seconds"] for row in epoch_rows)),
            "logging_seconds_total": float(sum(row["logging_seconds"] for row in epoch_rows)),
            "other_epoch_seconds_total": max(
                total_epoch_seconds
                - sum(row["train_total_seconds"] + row["validation_total_seconds"] + row["checkpoint_seconds"] + row["metric_aggregation_seconds"] + row["logging_seconds"] for row in epoch_rows),
                0.0,
            ),
        },
        "resources": {
            "gpu_snapshots": gpu_rows,
            "gpu_utilization_pct_summary": {
                "count": len(util_values),
                "min": min(util_values) if util_values else None,
                "median": percentile(util_values, 50),
                "max": max(util_values) if util_values else None,
            },
            "gpu_memory_used_mib_summary": {
                "max": max(memory_used) if memory_used else None,
            },
            "jax_memory_stats_by_epoch": memory_rows,
            "peak_jax_bytes_in_use": max(
                (row.get("peak_bytes_in_use", 0) for row in memory_rows), default=0
            ),
            "peak_jax_bytes_reserved": max(
                (row.get("peak_bytes_reserved", 0) for row in memory_rows), default=0
            ),
        },
        "checkpoint": {
            "latest_per_epoch": True,
            "last_receipt": latest_checkpoint,
            "selection": "none; runtime diagnostic only",
        },
        "hard_boundaries": {
            "g1_modified": False,
            "test_iid_or_sealed_access": False,
            "deepoheat_official_test_access": False,
            "formal_training_started": False,
            "multi_htc_started": False,
            "therm_fm_downloaded": False,
            "science_config_changed": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "repo_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "formal_accuracy_claim_allowed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "compiled"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": payload["mode"],
                "batch_size": payload["scientific_contract"]["batch_size"],
                "batches_per_epoch": payload["scientific_contract"]["batches_per_epoch"],
                "median_post_warmup_seconds": payload["step_summary"]["median_post_warmup_seconds"],
                "samples_per_second": payload["throughput"]["post_warmup_samples_per_second"],
                "peak_jax_bytes_in_use": payload["resources"]["peak_jax_bytes_in_use"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS_RUNTIME_FINITE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
