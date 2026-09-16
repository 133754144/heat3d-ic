#!/usr/bin/env python3
"""Bounded native-65 Therm-FM qualification on real P1i targets.

This runner is intentionally not the formal trainer.  It uses only the frozen
train/valid roles, performs a few optimizer updates, and writes a small receipt
while all checkpoint/model outputs remain in a caller-supplied temporary path.
The official Therm-FM test loader is not used because it would construct a test
dataset and would violate the V7 sealed-data contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


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

    # This is the upstream --replace_embedding_recovery intent.  Transformers
    # 4.57 does not initialize shape-mismatched ConvTranspose2d modules, so the
    # replacement modules receive scOT's own initializer and a zero bias.
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
    return model, cfg


def build_optimizer(model, torch_module):
    # Keep the official Therm-FM two-rate parameter-group semantics without
    # importing Trainer (which would instantiate the upstream test dataset).
    from transformers.trainer import get_parameter_names  # type: ignore
    from scOT.model import ConditionalLayerNorm, LayerNorm  # type: ignore

    decay_names = get_parameter_names(
        model,
        [torch_module.nn.LayerNorm, LayerNorm, ConditionalLayerNorm],
    )
    decay_names = {name for name in decay_names if "bias" not in name}
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
    groups = [
        {"params": standard, "lr": 5e-5, "weight_decay": 1e-6},
        {"params": no_decay, "lr": 5e-5, "weight_decay": 0.0},
        {"params": embeddings, "lr": 5e-4, "weight_decay": 1e-6},
    ]
    return torch.optim.AdamW(groups, lr=5e-5, weight_decay=1e-6)


def metric_payload(pred_norm: torch.Tensor, target_norm: torch.Tensor, dataset, source, sid: str):
    pred = pred_norm.detach().cpu().numpy().astype(np.float64)
    target = target_norm.detach().cpu().numpy().astype(np.float64)
    mean = dataset.output_mean.astype(np.float64)
    std = dataset.output_std.astype(np.float64)
    pred = pred * std + mean
    target = target * std + mean
    err = pred - target
    flat_pred = pred.reshape(-1)
    flat_target = target.reshape(-1)
    flat_err = err.reshape(-1)
    denom = np.linalg.norm(flat_target)
    sample_rel = float(np.linalg.norm(flat_err) / max(denom, 1e-12) * 100.0)
    point_rel = float(
        np.sqrt(np.mean(flat_err**2))
        / max(np.sqrt(np.mean(flat_target**2)), 1e-12)
        * 100.0
    )
    return {
        "sample_id": sid,
        "nodes": int(flat_pred.size),
        "sample_first_relative_rmse_pct": sample_rel,
        "point_global_relative_rmse_pct": point_rel,
        "rmse_K": float(np.sqrt(np.mean(flat_err**2))),
        "mae_K": float(np.mean(np.abs(flat_err))),
        "peak_abs_error_K": float(np.max(np.abs(flat_err))),
        "pred_range_K": [float(pred.min()), float(pred.max())],
        "target_range_K": [float(target.min()), float(target.max())],
        "finite": bool(np.isfinite(pred).all() and np.isfinite(target).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--therm-repo", type=Path, required=True)
    parser.add_argument("--poseidon-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=3)
    args = parser.parse_args()

    if args.checkpoint_out.exists():
        raise SystemExit(f"refusing to overwrite existing qualification checkpoint: {args.checkpoint_out}")
    for required in (
        args.adapter,
        args.therm_repo,
        args.poseidon_dir,
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
        args.stats,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(args.adapter)
    source = adapter.P1iThermFMSource(
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
    )
    train_ids = source.ids_by_role["train"]
    valid_ids = source.ids_by_role["valid_iid"]
    if len(train_ids) != 768 or len(valid_ids) != 128:
        raise RuntimeError(f"unexpected authorized split sizes: {len(train_ids)}, {len(valid_ids)}")
    dataset_cls = adapter.P1iThermFMDataset
    train_ds = dataset_cls(source, train_ids[:2], args.stats)
    valid_ds = dataset_cls(source, valid_ids[:1], args.stats)
    train_batch = [train_ds[i] for i in range(len(train_ds))]
    valid_item = valid_ds[0]
    x = torch.stack([row["pixel_values"] for row in train_batch]).to(device)
    y = torch.stack([row["labels"] for row in train_batch]).to(device)
    xv = valid_item["pixel_values"].unsqueeze(0).to(device)
    yv = valid_item["labels"].unsqueeze(0).to(device)
    if tuple(x.shape[1:]) != (741, 65, 65) or tuple(y.shape[1:]) != (57, 65, 65):
        raise RuntimeError(f"native shape mismatch: {tuple(x.shape)}, {tuple(y.shape)}")

    load_start = time.perf_counter()
    model, cfg = initialize_model(args.therm_repo, args.poseidon_dir, device)
    load_seconds = time.perf_counter() - load_start
    optimizer = build_optimizer(model, torch)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    forward_seconds: list[float] = []
    update_seconds: list[float] = []
    finite_flags: list[bool] = []
    for _ in range(args.train_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(pixel_values=x, labels=y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds.append(time.perf_counter() - start)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        update_seconds.append(time.perf_counter() - start)
        losses.append(float(loss.detach().cpu()))
        finite_flags.append(bool(torch.isfinite(loss).item()))

    model.eval()
    with torch.no_grad():
        valid_out = model(pixel_values=xv)
    valid_metrics = metric_payload(valid_out.output[0], yv[0], valid_ds, source, valid_ids[0])
    output_shape = list(valid_out.output.shape)
    model_finite = bool(
        torch.isfinite(valid_out.output).all().item()
        and all(torch.isfinite(p).all().item() for p in model.parameters())
    )

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "train_steps": args.train_steps,
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "stats_sha256": sha256_file(args.stats),
        },
        args.checkpoint_out,
    )
    checkpoint = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
    reloaded, _ = initialize_model(args.therm_repo, args.poseidon_dir, device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reload_equal = all(
        torch.equal(value, checkpoint["model"][name])
        for name, value in reloaded.state_dict().items()
    )
    reload_optimizer = build_optimizer(reloaded, torch)
    reload_optimizer.load_state_dict(checkpoint["optimizer"])
    reload_optimizer_equal = all(
        torch.equal(a, b)
        for a, b in zip(optimizer.state_dict()["state"].values(), reload_optimizer.state_dict()["state"].values())
        if torch.is_tensor(a) and torch.is_tensor(b)
    )
    if device.type == "cuda":
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = peak_reserved = None

    receipt = {
        "schema_version": "heat3d_v7_g2_p22_thermfm_real_target_qualification_v1",
        "status": "PASS_T4_REAL_TARGET_NATIVE65" if (model_finite and reload_equal and reload_optimizer_equal and valid_metrics["finite"]) else "NEEDS_AMENDMENT",
        "scope": "bounded one-seed qualification; not formal accuracy evidence",
        "seed": args.seed,
        "train_fixture_ids": list(train_ds.ids),
        "valid_fixture_ids": [valid_ids[0]],
        "authorized_train_count": len(train_ids),
        "authorized_valid_count": len(valid_ids),
        "train_ids_sha256": sha256_ids(train_ids),
        "valid_ids_sha256": sha256_ids(valid_ids),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "adapter_path": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "full_field_archive_sha256": sha256_file(args.full_field_archive),
        "stats_path": str(args.stats),
        "stats_sha256": sha256_file(args.stats),
        "thermfm_repo_commit_sha": os.environ.get("THERMFM_REPO_COMMIT_SHA", "unknown"),
        "poseidon_revision": os.environ.get("POSEIDON_REVISION", "unknown"),
        "poseidon_config_sha256": sha256_file(args.poseidon_dir / "config.json"),
        "poseidon_weights_sha256": sha256_file(args.poseidon_dir / "pytorch_model.bin"),
        "native_input_shape": list(x.shape),
        "native_target_shape": list(y.shape),
        "native_output_shape": output_shape,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "external_padding": False,
        "external_resampling": False,
        "padding_pixels_in_loss": 0,
        "target": "real deltaT_K",
        "losses": losses,
        "loss_decrease": bool(losses[-1] <= losses[0]) if losses else False,
        "loss_trend": "decreasing_or_flat" if losses and losses[-1] <= losses[0] else "finite_but_not_decreasing_in_short_probe",
        "finite_flags": finite_flags,
        "forward_seconds": forward_seconds,
        "optimizer_update_seconds": update_seconds,
        "valid_metrics": valid_metrics,
        "model_finite": model_finite,
        "checkpoint_path": str(args.checkpoint_out),
        "checkpoint_sha256": sha256_file(args.checkpoint_out),
        "checkpoint_reload_state_equal": reload_equal,
        "checkpoint_reload_optimizer_equal": reload_optimizer_equal,
        "load_seconds": load_seconds,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_allocated_bytes": peak_alloc,
        "peak_memory_reserved_bytes": peak_reserved,
        "test_iid_accessed": False,
        "sealed_accessed": False,
        "deepoheat_official100_accessed": False,
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
