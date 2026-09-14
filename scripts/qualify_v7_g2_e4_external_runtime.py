#!/usr/bin/env python3
"""Bounded full-role GINO/Transolver runtime rehearsal.

This entry point is deliberately separate from the formal launcher.  It uses
the exact frozen P1i train/valid roles, batch/order, normalization, objective,
optimizer and scheduler semantics, but limits execution to a small number of
complete epochs and writes only temporary runtime receipts.  It never opens a
test or sealed role and never makes an accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATS_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
GINO_UPSTREAM = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
TRANSOLVER_UPSTREAM = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


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
        values = [item.strip() for item in line.split(",")]
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


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model not in {"GINO", "Transolver"}:
        raise ValueError("unsupported model")
    if args.epochs < 1 or args.epochs > 3:
        raise ValueError("runtime rehearsal is limited to 1--3 epochs")
    if not str(args.output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("runtime receipt must remain under /tmp")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: runtime rehearsal requires CUDA")
    if any(token in str(path).lower() for path in (args.dataset_root, args.dataset_manifest, args.statistics) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")

    external = load_module("g2_e4_runtime_external", ROOT / "scripts/run_v7_g2_p1i_external_formal.py")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    expected_upstream = GINO_UPSTREAM if args.model == "GINO" else TRANSOLVER_UPSTREAM
    if git_head(args.upstream_root) != expected_upstream:
        raise ValueError("pinned upstream commit mismatch")
    if sha256(args.dataset_manifest) != DATASET_SHA:
        raise ValueError("frozen dataset manifest SHA mismatch")
    stats = external.load_stats(args.statistics)
    train_source = external.P1iRoleDataset(args.dataset_root, args.dataset_manifest, "train")
    valid_source = external.P1iRoleDataset(args.dataset_root, args.dataset_manifest, "valid_iid")
    train = external.VerifiedRAMRoleDataset(train_source)
    valid = external.VerifiedRAMRoleDataset(valid_source)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if args.model == "GINO":
        model = external.build_gino(0.15, 0.033, use_open3d=True, use_torch_scatter=True).to(device)
        if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
            raise RuntimeError("GINO did not use Open3D FixedRadiusSearch")
        if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
            raise RuntimeError("GINO did not use torch-scatter reduction")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        grid = external.latent_queries(32).to(device)
        expected_epochs = 301
    else:
        model = external.build_transolver(args.upstream_root.resolve()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        grid = None
        expected_epochs = 500

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epoch_rows: list[dict[str, Any]] = []
    first_step_seconds: float | None = None
    all_step_seconds: list[float] = []
    all_valid_seconds: list[float] = []
    overall_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # This is the formal runner's epoch-indexed order, with no remainder
        # batch and no change to seed/batch semantics.
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed * 100000 + epoch - 1)).tolist()
        model.train()
        train_losses: list[float] = []
        step_seconds: list[float] = []
        train_started = time.perf_counter()
        for index in order:
            row = train[index]
            coords, features, _target, target_n, local = external.normalize(row, stats, device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction_n = external.predict(args.model, model, coords, features, grid)
            if args.model == "GINO":
                loss = external.relative_l2(prediction_n, target_n)
            else:
                prediction = prediction_n * local["target_std"] + local["target_mean"]
                target = target_n * local["target_std"] + local["target_mean"]
                loss = external.relative_l2(prediction, target)
            if not finite_tensor(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            if args.model == "Transolver":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if first_step_seconds is None:
                first_step_seconds = elapsed
            step_seconds.append(elapsed)
            all_step_seconds.append(elapsed)
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_total = time.perf_counter() - train_started

        model.eval()
        valid_losses: list[float] = []
        valid_seconds: list[float] = []
        valid_started = time.perf_counter()
        with torch.no_grad():
            for index in range(len(valid)):
                row = valid[index]
                coords, features, _target, target_n, local = external.normalize(row, stats, device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction_n = external.predict(args.model, model, coords, features, grid)
                if args.model == "GINO":
                    loss = external.relative_l2(prediction_n, target_n)
                else:
                    prediction = prediction_n * local["target_std"] + local["target_mean"]
                    target = target_n * local["target_std"] + local["target_mean"]
                    loss = external.relative_l2(prediction, target)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not finite_tensor(loss):
                    raise FloatingPointError("non-finite validation loss")
                valid_seconds.append(elapsed)
                all_valid_seconds.append(elapsed)
                valid_losses.append(float(loss.detach().cpu()))
        valid_total = time.perf_counter() - valid_started

        checkpoint_started = time.perf_counter()
        checkpoint_path = args.output.parent / f"{args.output.stem}_{args.model.lower()}_seed{args.seed}_epoch{epoch}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "seed": args.seed, "test_or_sealed_access": False}, checkpoint_path)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        epoch_total = train_total + valid_total + checkpoint_seconds
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_samples": len(train),
                "batch_size": 1,
                "optimizer_steps": len(order),
                "train_seconds": train_total,
                "valid_samples": len(valid),
                "validation_batch_size": 1,
                "validation_batches": len(valid),
                "valid_seconds": valid_total,
                "checkpoint_seconds": checkpoint_seconds,
                "epoch_seconds_including_checkpoint": epoch_total,
                "step_median_seconds": float(np.median(step_seconds)),
                "step_p95_seconds": float(np.percentile(step_seconds, 95)),
                "step_first_seconds": float(step_seconds[0]),
                "valid_forward_median_seconds": float(np.median(valid_seconds)),
                "valid_forward_p95_seconds": float(np.percentile(valid_seconds, 95)),
                "train_objective_mean_finite_sanity": float(np.mean(train_losses)),
                "valid_objective_mean_finite_sanity": float(np.mean(valid_losses)),
                "loss_finite": bool(np.all(np.isfinite(train_losses)) and np.all(np.isfinite(valid_losses))),
                "gpu_snapshot": gpu_snapshot(),
            }
        )
        print(json.dumps({"model": args.model, "seed": args.seed, "epoch": epoch, "epoch_seconds": epoch_total, "step_median_seconds": float(np.median(step_seconds))}, sort_keys=True), flush=True)

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    median_epoch = float(np.median([row["epoch_seconds_including_checkpoint"] for row in epoch_rows]))
    payload = {
        "schema_version": "heat3d_v7_g2_e4_external_runtime_v1",
        "status": "PASS_RUNTIME_FINITE" if all(row["loss_finite"] for row in epoch_rows) else "FAIL_NONFINITE",
        "model": args.model,
        "seed": args.seed,
        "epochs_completed": args.epochs,
        "scope": "devbox-only bounded complete train/valid runtime rehearsal; no formal training",
        "data_pipeline": {"cache_mode": "verified_ram", "train_startup_seconds": train.startup_seconds, "valid_startup_seconds": valid.startup_seconds, "manifest_sha256": DATASET_SHA, "statistics_sha256": STATS_SHA, "byte_identical_verified": True},
        "scientific_contract": {"train_samples": 768, "valid_iid_samples": 128, "batch_size": 1, "model_architecture_unchanged": True, "loss_optimizer_schedule_unchanged": True, "test_or_sealed_access": False, "accuracy_used_for_decision": False},
        "upstream": {"commit": expected_upstream, "path": str(args.upstream_root)},
        "architecture": {"parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "gino_radius": [0.15, 0.033] if args.model == "GINO" else None, "latent_grid": [32, 32, 32] if args.model == "GINO" else None},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "repo_sha": git_head(ROOT), "xla_flags": os.environ.get("XLA_FLAGS")},
        "timing": {"first_train_step_seconds_compile_inclusive": first_step_seconds, "post_warmup_step_median_seconds": float(np.median(all_step_seconds[1:] or all_step_seconds)), "post_warmup_step_p95_seconds": float(np.percentile(all_step_seconds[1:] or all_step_seconds, 95)), "samples_per_second_post_warmup": float(1.0 / np.median(all_step_seconds[1:] or all_step_seconds)), "validation_forward_median_seconds": float(np.median(all_valid_seconds)), "validation_forward_p95_seconds": float(np.percentile(all_valid_seconds, 95)), "epoch_median_seconds_including_checkpoint": median_epoch, "projection_basis": "median bounded epoch including measured validation and checkpoint; runtime-only"},
        "epoch_rows": epoch_rows,
        "projection": {"expected_formal_epochs": expected_epochs, "single_seed_hours": float(expected_epochs * median_epoch / 3600.0), "three_seed_hours": float(3 * expected_epochs * median_epoch / 3600.0), "steps_per_epoch_explicit": 768, "validation_samples_explicit": 128},
        "resource": {"peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved, "gpu_snapshots": [row["gpu_snapshot"] for row in epoch_rows]},
        "formal_accuracy_claim_allowed": False,
        "test_or_sealed_access": False,
        "wall_seconds_total": time.perf_counter() - overall_started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("GINO", "Transolver"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "model": payload["model"], "projection": payload["projection"]}, sort_keys=True), flush=True)
    return 0 if payload["status"] == "PASS_RUNTIME_FINITE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
