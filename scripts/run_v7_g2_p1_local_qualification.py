#!/usr/bin/env python3
"""Run the bounded G2-P1 local qualification gates.

This runner is deliberately limited to four frozen P1i train samples, two
``valid_iid`` samples, one epoch, CPU, and temporary output.  It never accepts
test/sealed roles and does not implement a formal G2 training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_g2.adapters import _load_official_transolver_model, prediction_sha256
from rigno.heat3d_g2.p1i import DATASET_ID, MANIFEST_SHA256


TRAIN_IDS = ("v6p1if1_0000", "v6p1if1_0002", "v6p1if1_0004", "v6p1if1_0005")
VALID_IDS = ("v6p1if1_0003", "v6p1if1_0009")
REQUIRED_FILES = (
    "coords.npy",
    "k_field.npy",
    "q_field.npy",
    "bc_features.npy",
    "deltaT.npy",
    "control_volume.npy",
    "layer_id.npy",
    "sample_meta.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("frozen P1i manifest SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("frozen P1i dataset ID mismatch")
    rows = {str(row["sample_id"]): row for row in payload["samples"]}
    return payload, rows


def sample_dir(root: Path, sample_id: str) -> Path:
    direct = root / "samples" / sample_id
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"missing frozen sample directory: {direct}")


def load_sample(
    root: Path,
    rows: dict[str, dict[str, Any]],
    sample_id: str,
    expected_role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    row = rows[sample_id]
    role = str(row["split_role"])
    if role != expected_role or role not in {"train", "valid_iid"}:
        raise ValueError(f"role boundary violation for {sample_id}: {role}")
    directory = sample_dir(root, sample_id)
    for name in REQUIRED_FILES:
        path = directory / name
        expected = row["file_sha256"][name]
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen file SHA mismatch: {sample_id}/{name}")
    if (directory / "temperature.npy").exists():
        raise ValueError("qualification cache must omit temperature.npy")

    coords = np.asarray(np.load(directory / "coords.npy"), dtype=np.float32)
    k_field = np.asarray(np.load(directory / "k_field.npy"), dtype=np.float32)
    q_field = np.asarray(np.load(directory / "q_field.npy"), dtype=np.float32).reshape(-1, 1)
    bc = np.asarray(np.load(directory / "bc_features.npy"), dtype=np.float32)
    target = np.asarray(np.load(directory / "deltaT.npy"), dtype=np.float32).reshape(-1, 1)
    metadata = json.loads((directory / "sample_meta.json").read_text(encoding="utf-8"))
    if coords.shape != (1024, 3) or k_field.shape != (1024, 3):
        raise ValueError(f"unexpected P1i geometry or conductivity shape: {sample_id}")
    if q_field.shape != (1024, 1) or target.shape != (1024, 1):
        raise ValueError(f"unexpected P1i field shape: {sample_id}")
    if bc.shape == (1024, 4):
        bc = np.column_stack(
            (
                bc,
                np.full(1024, metadata["top_h_W_m2K"], dtype=np.float32),
                np.full(1024, metadata["bottom_h_W_m2K"], dtype=np.float32),
                np.zeros(1024, dtype=np.float32),
            )
        )
    if bc.shape != (1024, 7):
        raise ValueError(f"unexpected P1i boundary feature shape: {sample_id}")
    features = np.concatenate((k_field, q_field, bc), axis=-1)
    return coords, features, target, {"sample_id": sample_id, "role": role}


def load_qualification_data(root: Path, manifest: Path) -> dict[str, Any]:
    _payload, rows = load_manifest(manifest)
    train = [load_sample(root, rows, sid, "train") for sid in TRAIN_IDS]
    valid = [load_sample(root, rows, sid, "valid_iid") for sid in VALID_IDS]
    all_ids = set(TRAIN_IDS + VALID_IDS)
    if any("test" in sid.lower() or "sealed" in sid.lower() for sid in all_ids):
        raise ValueError("closed role identifier reached qualification runner")
    return {
        "train": train,
        "valid": valid,
        "file_count_verified": len(all_ids) * len(REQUIRED_FILES),
    }


def normalizers(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    coords = torch.from_numpy(np.stack([row[0] for row in data["train"]]))
    features = torch.from_numpy(np.stack([row[1] for row in data["train"]]))
    targets = torch.from_numpy(np.stack([row[2] for row in data["train"]]))
    coord_min = coords.amin(dim=(0, 1), keepdim=True)
    coord_max = coords.amax(dim=(0, 1), keepdim=True)
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    feature_std = features.std(dim=(0, 1), keepdim=True)
    feature_std = torch.where(feature_std > 0, feature_std, torch.ones_like(feature_std))
    # Formal GINO and Transolver use the same train-only, per-channel global
    # statistics. This is invariant to point ordering and remains defined for
    # variable point sets. In particular, no node-index target statistics are
    # fitted even when a qualification subset happens to share node counts.
    gino_feature_mean = feature_mean
    gino_feature_std = feature_std
    trans_feature_mean = feature_mean
    trans_feature_std = feature_std
    gino_y_mean = targets.mean(dim=(0, 1), keepdim=True)
    gino_y_std = targets.std(dim=(0, 1), keepdim=True) + 1e-7
    # Transolver UnitTransformer reduces both sample and point dimensions.
    trans_y_mean = targets.mean(dim=(0, 1), keepdim=True)
    trans_y_std = targets.std(dim=(0, 1), keepdim=True) + 1e-8
    return {
        "coord_min": coord_min,
        "coord_span": torch.clamp(coord_max - coord_min, min=1e-12),
        "gino_feature_mean": gino_feature_mean,
        "gino_feature_std": gino_feature_std,
        "trans_feature_mean": trans_feature_mean,
        "trans_feature_std": trans_feature_std,
        "gino_y_mean": gino_y_mean,
        "gino_y_std": gino_y_std,
        "trans_y_mean": trans_y_mean,
        "trans_y_std": trans_y_std,
    }


def tensor_row(row: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value).unsqueeze(0) for value in row[:3])


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((pred - truth).reshape(pred.shape[0], -1), dim=1).div(
        torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1)
    ).sum()


def unit_coords(coords: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (coords - stats["coord_min"]) / stats["coord_span"]


def latent_queries(resolution: int = 32) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)


def build_gino() -> torch.nn.Module:
    from neuralop.models import GINO

    return GINO(
        in_channels=11,
        out_channels=1,
        latent_feature_channels=None,
        gno_coord_dim=3,
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        in_gno_pos_embed_type="nerf",
        out_gno_pos_embed_type="nerf",
        gno_embed_channels=16,
        # With a pointwise input function, the upstream linear input integral
        # requires its FNO input width to equal the function channel count.
        fno_in_channels=11,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_use_channel_mlp=True,
        fno_norm="instance_norm",
        fno_ada_in_features=32,
        fno_factorization="tucker",
        fno_rank=0.4,
        fno_channel_mlp_expansion=1.0,
        fno_resolution_scaling_factor=1,
        # Optional acceleration packages are unavailable on this Mac.  The
        # upstream pure-PyTorch paths retain radius-neighbor/mean-sum semantics.
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    )


def run_gino(data: dict[str, Any], stats: dict[str, torch.Tensor], output: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    model = build_gino()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    grid = latent_queries(32)
    start = time.monotonic()
    model.train()
    train_total = 0.0
    for row in data["train"]:
        coords, features, target = tensor_row(row)
        coords = unit_coords(coords, stats)
        features = (features - stats["gino_feature_mean"]) / stats["gino_feature_std"]
        target_n = (target - stats["gino_y_mean"]) / stats["gino_y_std"]
        optimizer.zero_grad(set_to_none=True)
        pred_n = model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
        loss = relative_l2(pred_n, target_n)
        if not torch.isfinite(loss):
            raise FloatingPointError("GINO produced a non-finite training loss")
        loss.backward()
        optimizer.step()
        train_total += float(loss.detach())
    scheduler.step()
    model.eval()
    valid_total = 0.0
    first_prediction = None
    with torch.no_grad():
        for row in data["valid"]:
            coords, features, target = tensor_row(row)
            coords = unit_coords(coords, stats)
            features = (features - stats["gino_feature_mean"]) / stats["gino_feature_std"]
            pred_n = model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
            pred = pred_n * stats["gino_y_std"] + stats["gino_y_mean"]
            valid_total += float(relative_l2(pred, target))
            first_prediction = pred if first_prediction is None else first_prediction
    checkpoint = output / "gino_one_epoch.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}, checkpoint)
    reloaded = build_gino()
    # GINO's state dict carries activation metadata (torch._C._nn.gelu), which
    # PyTorch's weights-only unpickler rejects.  This file was created locally
    # moments above; no third-party checkpoint is deserialized here.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["model"])
    reloaded.eval()
    coords, features, _target = tensor_row(data["valid"][0])
    coords = unit_coords(coords, stats)
    features = (features - stats["gino_feature_mean"]) / stats["gino_feature_std"]
    with torch.no_grad():
        pred_n = reloaded(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
        reloaded_prediction = pred_n * stats["gino_y_std"] + stats["gino_y_mean"]
    reload_equal = torch.equal(first_prediction, reloaded_prediction)
    return {
        "status": "PASS_1_EPOCH_TRAIN_VALID_CHECKPOINT_RELOAD" if reload_equal else "FAIL_RELOAD_DRIFT",
        "parameter_count": parameter_count,
        "train_samples": len(data["train"]),
        "valid_samples": len(data["valid"]),
        "train_relative_l2_mean": train_total / len(data["train"]),
        "valid_relative_l2_mean": valid_total / len(data["valid"]),
        "prediction_sha256": prediction_sha256(first_prediction),
        "reload_prediction_sha256": prediction_sha256(reloaded_prediction),
        "reload_bitwise_equal": reload_equal,
        "checkpoint_sha256": sha256(checkpoint),
        "wall_seconds": time.monotonic() - start,
    }


def build_transolver(upstream: Path) -> torch.nn.Module:
    model_class = _load_official_transolver_model(upstream)
    return model_class(
        space_dim=3,
        n_layers=8,
        n_hidden=128,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=11,
        out_dim=1,
        slice_num=64,
        ref=8,
        unified_pos=False,
    )


def run_transolver(
    data: dict[str, Any], stats: dict[str, torch.Tensor], upstream: Path, output: Path
) -> dict[str, Any]:
    torch.manual_seed(0)
    model = build_transolver(upstream)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    start = time.monotonic()
    model.train()
    train_total = 0.0
    for row in data["train"]:
        coords, features, target = tensor_row(row)
        coords = unit_coords(coords, stats)
        features = (features - stats["trans_feature_mean"]) / stats["trans_feature_std"]
        target_n = (target - stats["trans_y_mean"]) / stats["trans_y_std"]
        optimizer.zero_grad(set_to_none=True)
        pred_n = model(coords, features)
        pred = pred_n * stats["trans_y_std"] + stats["trans_y_mean"]
        decoded_target = target_n * stats["trans_y_std"] + stats["trans_y_mean"]
        loss = relative_l2(pred, decoded_target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Transolver produced a non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        train_total += float(loss.detach())
    scheduler.step()
    model.eval()
    valid_total = 0.0
    first_prediction = None
    with torch.no_grad():
        for row in data["valid"]:
            coords, features, target = tensor_row(row)
            coords = unit_coords(coords, stats)
            features = (features - stats["trans_feature_mean"]) / stats["trans_feature_std"]
            pred_n = model(coords, features)
            pred = pred_n * stats["trans_y_std"] + stats["trans_y_mean"]
            valid_total += float(relative_l2(pred, target))
            first_prediction = pred if first_prediction is None else first_prediction
    checkpoint = output / "transolver_one_epoch.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}, checkpoint)
    reloaded = build_transolver(upstream)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model"])
    reloaded.eval()
    coords, features, _target = tensor_row(data["valid"][0])
    coords = unit_coords(coords, stats)
    features = (features - stats["trans_feature_mean"]) / stats["trans_feature_std"]
    with torch.no_grad():
        pred_n = reloaded(coords, features)
        reloaded_prediction = pred_n * stats["trans_y_std"] + stats["trans_y_mean"]
    reload_equal = torch.equal(first_prediction, reloaded_prediction)
    return {
        "status": "PASS_1_EPOCH_TRAIN_VALID_CHECKPOINT_RELOAD" if reload_equal else "FAIL_RELOAD_DRIFT",
        "parameter_count": parameter_count,
        "train_samples": len(data["train"]),
        "valid_samples": len(data["valid"]),
        "train_relative_l2_mean": train_total / len(data["train"]),
        "valid_relative_l2_mean": valid_total / len(data["valid"]),
        "prediction_sha256": prediction_sha256(first_prediction),
        "reload_prediction_sha256": prediction_sha256(reloaded_prediction),
        "reload_bitwise_equal": reload_equal,
        "checkpoint_sha256": sha256(checkpoint),
        "wall_seconds": time.monotonic() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gino-root", type=Path, required=True)
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("GINO", "Transolver"), default=("GINO", "Transolver"))
    args = parser.parse_args()
    if not str(args.output_dir.resolve()).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("qualification checkpoints must be written under /tmp")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    data = load_qualification_data(args.subset.resolve(), args.manifest.resolve())
    stats = normalizers(data)
    receipt: dict[str, Any] = {
        "schema_version": "heat3d_v7_g2_p1_local_qualification_v1",
        "status": "PASS",
        "execution_role": "local_one_epoch_qualification_nonpublication",
        "formal_g2": False,
        "host": "local_mac_cpu",
        "epochs": 1,
        "dataset": {
            "dataset_id": DATASET_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "archive_revision": "7b3af69e2164ad06d1c079fbde4d6cbd50183c9a",
            "train_ids": list(TRAIN_IDS),
            "valid_iid_ids": list(VALID_IDS),
            "verified_file_count": data["file_count_verified"],
            "temperature_npy_present": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
        "information": {
            "coordinates": "train-bounds affine map to unit cube",
            "features": ["kx", "ky", "kz", "q", "is_top", "is_bottom", "is_side", "is_interior", "top_h", "bottom_h", "top_T_inf_minus_T_ref"],
            "learned_adapter": False,
            "extra_physical_information": False,
            "GINO_feature_normalization": "train-only global per-channel z-score over sample and point dimensions; point-order invariant",
            "Transolver_feature_normalization": "train-only UnitTransformer semantics over sample and point dimensions",
        },
        "models": {},
    }
    if "GINO" in args.models:
        sys.path.insert(0, str(args.gino_root.resolve()))
        receipt["models"]["GINO"] = run_gino(data, stats, args.output_dir)
    if "Transolver" in args.models:
        receipt["models"]["Transolver"] = run_transolver(
            data, stats, args.transolver_root.resolve(), args.output_dir
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
