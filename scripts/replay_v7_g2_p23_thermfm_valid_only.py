#!/usr/bin/env python3
"""Replay a frozen Therm-FM checkpoint on the P23 valid-only manifest.

No optimizer step is performed.  The source is constructed with the
``valid_iid`` role only and predictions are written to a new, non-existing
output path.  The script never discovers or iterates over test/sealed cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


GRID = (65, 65, 57)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
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
        str(poseidon_dir), config=cfg, ignore_mismatched_sizes=True, local_files_only=True
    ).to(device)
    # Preserve P22 replacement semantics before loading the frozen state.
    for module in (
        model.embeddings.patch_embeddings.projection,
        model.patch_recovery.projection,
        model.patch_recovery.mixup,
    ):
        model._init_weights(module)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(model.patch_recovery.projection.weight, std=cfg.initializer_range)
        if model.patch_recovery.projection.bias is not None:
            torch.nn.init.zeros_(model.patch_recovery.projection.bias)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--therm-repo", type=Path, required=True)
    parser.add_argument("--poseidon-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--valid-case-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    required = (
        args.adapter,
        args.therm_repo,
        args.poseidon_dir,
        args.split_manifest,
        args.valid_case_manifest,
        args.sample_root,
        args.full_field_archive,
        args.stats,
        args.checkpoint,
    )
    if any(not path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    if args.output.exists() or args.receipt.exists():
        raise SystemExit("refusing to overwrite existing replay artifact")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {checkpoint_sha} != {args.checkpoint_sha256}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED: P23 Therm-FM replay requires CUDA")
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    case_payload = json.loads(args.valid_case_manifest.read_text(encoding="utf-8"))
    cases = case_payload["cases"]
    if case_payload.get("role") != "valid_iid" or len(cases) != 128:
        raise SystemExit("valid-only case manifest contract failed")
    valid_ids = tuple(str(case["sample_id"]) for case in cases)
    if any(case.get("split_role") != "valid_iid" for case in cases):
        raise SystemExit("case manifest includes non-valid role")

    adapter = load_module(args.adapter)
    source = adapter.P1iThermFMSource(
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
        roles=("valid_iid",),
    )
    if tuple(source.ids_by_role["valid_iid"]) != tuple(sorted(valid_ids)):
        raise SystemExit("valid IDs drifted from frozen split manifest")
    dataset = adapter.P1iThermFMDataset(source, tuple(sorted(valid_ids)), args.stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    model = initialize_model(args.therm_repo, args.poseidon_dir, device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("checkpoint does not contain model state")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    finite = True
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            if x.shape[1:] != (GRID[2] * 13, GRID[0], GRID[1]):
                raise RuntimeError(f"unexpected input shape {tuple(x.shape)}")
            output = model(pixel_values=x).output
            if output.shape[1:] != (GRID[2], GRID[0], GRID[1]):
                raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
            torch.cuda.synchronize()
            # The common evaluator consumes physical deltaT_K.  The frozen
            # Therm-FM checkpoint predicts train-normalized channels, so
            # denormalize with the already-frozen train-only statistics before
            # writing the valid-only archive.  This is deterministic post-
            # processing, not a new adapter or learned reconstruction.
            values = output.detach().cpu().numpy().astype(np.float32, copy=False)
            values = values * dataset.output_std[None, ...] + dataset.output_mean[None, ...]
            finite = finite and bool(np.isfinite(values).all())
            for sid, value in zip(batch["sample_id"], values, strict=True):
                predictions[str(sid)] = np.array(value, copy=True)
    elapsed = time.perf_counter() - start
    if set(predictions) != set(valid_ids) or len(predictions) != 128:
        raise RuntimeError("replay did not produce exactly the authorized 128 valid cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **predictions)
    receipt = {
        "schema_version": "heat3d_v7_g2_p23_thermfm_valid_only_replay_v1",
        "status": "PASS_VALID_ONLY_REPLAY" if finite else "FAIL_NONFINITE",
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_sha,
        "valid_case_manifest_sha256": sha256_file(args.valid_case_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "normalization_sha256": sha256_file(args.stats),
        "authorized_role": "valid_iid",
        "valid_count": len(predictions),
        "grid": list(GRID),
        "layout": "z_x_y float32 arrays; evaluator transposes to canonical x,y,z",
        "value_kind": "deltaT_K",
        "batch_size": args.batch_size,
        "replay_seconds": elapsed,
        "finite": finite,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
        "training_or_optimizer_step": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
