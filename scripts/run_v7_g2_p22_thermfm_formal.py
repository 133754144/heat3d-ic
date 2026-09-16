#!/usr/bin/env python3
"""Valid-only Therm-FM P1i formal runner for the frozen P22 contract.

The runner intentionally does not construct the upstream test dataset.  It
uses the split-preserving native-65 adapter and reads only train/valid rows from
the frozen P1i archive.  A caller must provide a new, non-existing output
directory; existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


EPOCHS = 200
BATCH_SIZE = 40
VAL_BATCH_SIZE = 40
LEARNING_RATE = 5e-5
EMBEDDING_RECOVERY_LR = 5e-4
WEIGHT_DECAY = 1e-6
MAX_GRAD_NORM = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("v7_g2_p22_thermfm_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import adapter {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_model(therm_repo: Path, poseidon_dir: Path, device: torch.device):
    sys.path.insert(0, str(therm_repo))
    from scOT.model import ScOT, ScOTConfig  # type: ignore

    cfg = ScOTConfig(
        image_size=65,
        patch_size=4,
        num_channels=741,
        num_out_channels=57,
        embed_dim=48,
        depths=[4, 4, 4, 4],
        num_heads=[3, 6, 12, 24],
        skip_connections=[2, 2, 2, 0],
        window_size=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=2,
        channel_slice_list_normalized_loss=[0, 57],
        residual_model="convnext",
        use_conditioning=False,
        learn_residual=False,
    )
    model = ScOT.from_pretrained(
        str(poseidon_dir),
        config=cfg,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    # Upstream --replace_embedding_recovery semantics.  The pinned
    # Transformers version leaves shape-mismatched ConvTranspose2d weights
    # uninitialized, so initialize only those replacement modules with scOT's
    # own initializer (no backbone/loss change).
    for module in (
        model.embeddings.patch_embeddings.projection,
        model.patch_recovery.projection,
        model.patch_recovery.mixup,
    ):
        model._init_weights(module)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(
            model.patch_recovery.projection.weight, std=cfg.initializer_range
        )
        if model.patch_recovery.projection.bias is not None:
            torch.nn.init.zeros_(model.patch_recovery.projection.bias)
    return model


def build_optimizer(model):
    from transformers.trainer import get_parameter_names  # type: ignore
    from scOT.model import ConditionalLayerNorm, LayerNorm  # type: ignore

    decay_names = set(
        name
        for name in get_parameter_names(
            model,
            [torch.nn.LayerNorm, LayerNorm, ConditionalLayerNorm],
        )
        if "bias" not in name
    )
    standard, no_decay, embeddings = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embeddings" in name or "patch_recovery" in name:
            embeddings.append(param)
        elif name in decay_names:
            standard.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": standard, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "lr": LEARNING_RATE, "weight_decay": 0.0},
            {
                "params": embeddings,
                "lr": EMBEDDING_RECOVERY_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def batch_to_device(batch: dict[str, Any], device: torch.device):
    return batch["pixel_values"].to(device, non_blocking=True), batch["labels"].to(
        device, non_blocking=True
    )


def normalized_validation(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    finite = True
    with torch.no_grad():
        for batch in loader:
            x, y = batch_to_device(batch, device)
            out = model(pixel_values=x, labels=y)
            loss = out.loss
            finite = finite and bool(torch.isfinite(loss).item())
            n = int(x.shape[0])
            total += float(loss.detach().cpu()) * n
            count += n
    return total / max(count, 1), finite


def denorm_fullfield_metrics(model, loader, device, dataset):
    """Compute valid-only physical-space metrics without retaining predictions."""
    model.eval()
    out_mean = dataset.output_mean.astype(np.float64)
    out_std = dataset.output_std.astype(np.float64)
    sum_sq_err = 0.0
    sum_sq_target = 0.0
    sum_abs_err = 0.0
    total_points = 0
    sample_relative: list[float] = []
    peak_abs: list[float] = []
    finite = True
    with torch.no_grad():
        for batch in loader:
            x, y = batch_to_device(batch, device)
            prediction = model(pixel_values=x).output.detach().cpu().numpy().astype(np.float64)
            target = y.detach().cpu().numpy().astype(np.float64)
            prediction = prediction * out_std + out_mean
            target = target * out_std + out_mean
            error = prediction - target
            finite = finite and bool(np.isfinite(prediction).all() and np.isfinite(target).all())
            for idx in range(prediction.shape[0]):
                ef = error[idx].reshape(-1)
                tf = target[idx].reshape(-1)
                sample_relative.append(
                    float(np.linalg.norm(ef) / max(np.linalg.norm(tf), 1e-12) * 100.0)
                )
                peak_abs.append(float(np.max(np.abs(ef))))
            sum_sq_err += float(np.square(error).sum())
            sum_sq_target += float(np.square(target).sum())
            sum_abs_err += float(np.abs(error).sum())
            total_points += int(error.size)
    mse = sum_sq_err / max(total_points, 1)
    global_relative = (sum_sq_err / max(sum_sq_target, 1e-24)) ** 0.5 * 100.0
    return {
        "sample_first_relative_rmse_pct": float(np.mean(sample_relative)),
        "point_global_relative_rmse_pct": float(global_relative),
        "rmse_K": float(np.sqrt(mse)),
        "mae_K": float(sum_abs_err / max(total_points, 1)),
        "peak_abs_error_K_mean": float(np.mean(peak_abs)),
        "peak_error_rmse_K": float(np.sqrt(np.mean(np.square(peak_abs)))),
        "valid_samples": len(sample_relative),
        "nodes_per_sample": int(total_points // max(len(sample_relative), 1)),
        "finite": finite,
    }


def state_equal(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    if lhs.keys() != rhs.keys():
        return False
    for key in lhs:
        a, b = lhs[key], rhs[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a.cpu(), b.cpu()):
                return False
        elif isinstance(a, dict) and isinstance(b, dict):
            if not state_equal(a, b):
                return False
        elif a != b:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--therm-repo", type=Path, required=True)
    parser.add_argument("--poseidon-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing formal output: {args.output_dir}")
    required = (
        args.adapter,
        args.therm_repo,
        args.poseidon_dir,
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
        args.stats,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED: Therm-FM formal requires CUDA")
    device = torch.device("cuda")

    adapter = load_adapter(args.adapter)
    source = adapter.P1iThermFMSource(
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
    )
    train_ids = source.ids_by_role["train"]
    valid_ids = source.ids_by_role["valid_iid"]
    if len(train_ids) != 768 or len(valid_ids) != 128:
        raise RuntimeError(f"unexpected split sizes {len(train_ids)} / {len(valid_ids)}")
    train_ds = adapter.P1iThermFMDataset(source, train_ids, args.stats)
    valid_ds = adapter.P1iThermFMDataset(source, valid_ids, args.stats)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    updates_per_epoch = len(train_loader)
    total_updates = EPOCHS * updates_per_epoch

    load_start = time.perf_counter()
    model = initialize_model(args.therm_repo, args.poseidon_dir, device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    load_seconds = time.perf_counter() - load_start
    parameter_count = int(sum(param.numel() for param in model.parameters()))
    args.output_dir.mkdir(parents=True)
    runner_sha = sha256_file(Path(__file__))
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    run_config = {
        "schema_version": "heat3d_v7_g2_p22_thermfm_formal_run_config_v1",
        "status": "RUNNING_VALID_ONLY",
        "seed": args.seed,
        "repo_commit_sha": repo_commit,
        "runner_sha256": runner_sha,
        "adapter_sha256": sha256_file(args.adapter),
        "thermfm_repo_commit_sha": "1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1",
        "poseidon_revision": "93adcbf10f75b45ac3bca3939cc3d3f239e1e663",
        "poseidon_config_sha256": sha256_file(args.poseidon_dir / "config.json"),
        "poseidon_weights_sha256": sha256_file(args.poseidon_dir / "pytorch_model.bin"),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "full_field_archive_sha256": sha256_file(args.full_field_archive),
        "stats_sha256": sha256_file(args.stats),
        "train_ids_sha256": sha256_ids(train_ids),
        "valid_ids_sha256": sha256_ids(valid_ids),
        "train_count": len(train_ids),
        "valid_count": len(valid_ids),
        "test_iid_read": False,
        "sealed_read": False,
        "official100_read": False,
        "native_grid": [65, 65, 57],
        "input_channels": 741,
        "output_channels": 57,
        "parameter_count": parameter_count,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_batch_size": VAL_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "learning_rate_embedding_recovery": EMBEDDING_RECOVERY_LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "scheduler": {"name": "cosine", "warmup_ratio": 0.0, "total_updates": total_updates},
        "loss": "upstream ScOT normalized p=2, channel slice [0,57]",
        "checkpoint_selection": "minimum valid normalized loss, lower is better",
        "output_dir": str(args.output_dir),
        "load_seconds": load_seconds,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    history_path = args.output_dir / "history.jsonl"
    best_loss = float("inf")
    best_epoch = None
    best_path = args.output_dir / "best_checkpoint.pt"
    final_path = args.output_dir / "final_checkpoint.pt"
    epoch_records: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_count = 0
        finite = True
        step_seconds: list[float] = []
        for batch in train_loader:
            x, y = batch_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model(pixel_values=x, labels=y)
            loss = out.loss
            finite = finite and bool(torch.isfinite(loss).item())
            if not finite:
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_start)
            n = int(x.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * n
            train_count += n

        valid_loss, valid_finite = normalized_validation(model, valid_loader, device)
        if not valid_finite:
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
        epoch_record = {
            "epoch": epoch,
            "global_updates": epoch * updates_per_epoch,
            "train_samples": train_count,
            "train_batches": updates_per_epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "valid_normalized_loss": valid_loss,
            "finite": finite and valid_finite,
            "median_step_seconds": float(np.median(step_seconds)),
            "p95_step_seconds": float(np.percentile(step_seconds, 95)),
            "epoch_seconds": time.perf_counter() - epoch_start,
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
        }
        epoch_records.append(epoch_record)
        with history_path.open("a") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_updates": epoch * updates_per_epoch,
            "best_metric": min(best_loss, valid_loss),
            "best_epoch": epoch if valid_loss < best_loss else best_epoch,
            "seed": args.seed,
            "repo_commit_sha": repo_commit,
            "runner_sha256": runner_sha,
            "adapter_sha256": sha256_file(args.adapter),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "stats_sha256": sha256_file(args.stats),
        }
        atomic_torch_save(checkpoint_payload, args.output_dir / "latest_epoch.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            checkpoint_payload["best_metric"] = best_loss
            checkpoint_payload["best_epoch"] = best_epoch
            atomic_torch_save(checkpoint_payload, best_path)
        print(json.dumps(epoch_record), flush=True)

    final_payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": EPOCHS,
        "global_updates": total_updates,
        "best_metric": best_loss,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "repo_commit_sha": repo_commit,
        "runner_sha256": runner_sha,
        "adapter_sha256": sha256_file(args.adapter),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "stats_sha256": sha256_file(args.stats),
    }
    atomic_torch_save(final_payload, final_path)
    checkpoint_reload_state_equal = False
    loaded = torch.load(final_path, map_location=device, weights_only=False)
    reload_model = initialize_model(args.therm_repo, args.poseidon_dir, device)
    reload_model.load_state_dict(loaded["model"], strict=True)
    checkpoint_reload_state_equal = state_equal(reload_model.state_dict(), loaded["model"])
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    # Full-field metrics are valid-only and evaluated only after training; no
    # test/sealed loader or row is constructed.
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    best_model = initialize_model(args.therm_repo, args.poseidon_dir, device)
    best_model.load_state_dict(best_state["model"], strict=True)
    best_metrics = denorm_fullfield_metrics(best_model, valid_loader, device, valid_ds)
    final_metrics = denorm_fullfield_metrics(reload_model, valid_loader, device, valid_ds)
    completion = {
        "schema_version": "heat3d_v7_g2_p22_thermfm_formal_completion_receipt_v1",
        "status": "COMPLETE_VALID_ONLY",
        "seed": args.seed,
        "repo_commit_sha": repo_commit,
        "runner_sha256": runner_sha,
        "config_sha256": sha256_file(args.output_dir / "run_config.json"),
        "dataset_split_sha256": sha256_file(args.split_manifest),
        "normalization_sha256": sha256_file(args.stats),
        "upstream_commit_sha": "1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1",
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "best_epoch": best_epoch,
        "best_valid_normalized_loss": best_loss,
        "best_fullfield_metrics_valid_only": best_metrics,
        "final_fullfield_metrics_valid_only": final_metrics,
        "epochs": EPOCHS,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "training_walltime_seconds": time.perf_counter() - overall_start,
        "peak_memory_allocated_bytes": peak_alloc,
        "peak_memory_reserved_bytes": peak_reserved,
        "checkpoint_reload_state_equal": checkpoint_reload_state_equal,
        "test_iid_read": False,
        "sealed_read": False,
        "official100_read": False,
        "evaluation_unlock": "not performed; valid-only completion",
    }
    (args.output_dir / "completion_receipt.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
