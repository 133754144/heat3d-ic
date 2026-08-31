#!/usr/bin/env python3
"""Frozen P1i GINO/Transolver runner prepared for post-G1 remote execution.

The runner refuses test/sealed roles, consumes exactly coordinates plus eleven
physical features, and fits no statistics. ``contract-check`` performs no data
access or training. ``preflight`` is one train step plus one valid forward;
``train`` uses the immutable epoch/seed/checkpoint contracts. Remote execution
still requires separate authorization after G1 completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample  # noqa: E402
from scripts.run_v7_g2_p1_local_qualification import (  # noqa: E402
    build_gino,
    build_transolver,
    latent_queries,
    relative_l2,
)

DATASET_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATS_PAYLOAD_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
FEATURES = (
    "kx", "ky", "kz", "q", "is_top", "is_bottom", "is_side", "is_interior",
    "top_h", "bottom_h", "top_T_inf_minus_T_ref",
)
REQUIRED = (
    "coords.npy", "k_field.npy", "q_field.npy", "bc_features.npy", "deltaT.npy",
    "control_volume.npy", "layer_id.npy", "sample_meta.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_stats(path: Path) -> dict[str, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = json_sha256(payload)
    if claimed != actual or actual != STATS_PAYLOAD_SHA or payload["fit_role"] != "train_only":
        raise ValueError("frozen train-only normalization payload mismatch")
    s = payload["statistics"]
    result = {
        "coord_min": torch.tensor(s["coordinate_min"], dtype=torch.float32).view(1, 1, 3),
        "coord_max": torch.tensor(s["coordinate_max"], dtype=torch.float32).view(1, 1, 3),
        "feature_mean": torch.tensor(s["feature_mean"], dtype=torch.float32).view(1, 1, 11),
        "feature_std": torch.tensor(s["feature_std"], dtype=torch.float32).view(1, 1, 11),
        "target_mean": torch.tensor(s["target_mean"], dtype=torch.float32).view(1, 1, 1),
        "target_std": torch.tensor(s["target_std"], dtype=torch.float32).view(1, 1, 1),
    }
    result["feature_std"] = torch.where(
        result["feature_std"] > 0, result["feature_std"], torch.ones_like(result["feature_std"])
    )
    return result


class P1iRoleDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, manifest: Path, role: str):
        if role not in {"train", "valid_iid"}:
            raise ValueError("formal external runner permits only train/valid_iid")
        if sha256(manifest) != DATASET_SHA:
            raise ValueError("frozen P1i manifest SHA mismatch")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.rows = [row for row in payload["samples"] if row["split_role"] == role]
        expected = 768 if role == "train" else 128
        if len(self.rows) != expected:
            raise ValueError(f"{role} count mismatch: {len(self.rows)}")
        self.root, self.role = root, role

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if row["split_role"] != self.role:
            raise ValueError("role drift")
        directory = self.root / "samples" / row["sample_id"]
        for name in REQUIRED:
            path = directory / name
            if not path.is_file() or sha256(path) != row["file_sha256"][name]:
                raise ValueError(f"frozen file mismatch: {row['sample_id']}/{name}")
        meta = json.loads((directory / "sample_meta.json").read_text())
        coords = np.asarray(np.load(directory / "coords.npy"), dtype=np.float32)
        k = np.asarray(np.load(directory / "k_field.npy"), dtype=np.float32)
        q = np.asarray(np.load(directory / "q_field.npy"), dtype=np.float32).reshape(-1, 1)
        bc = np.asarray(np.load(directory / "bc_features.npy"), dtype=np.float32)
        if bc.shape == (1024, 4):
            bc = np.column_stack((bc, np.full(1024, meta["top_h_W_m2K"], np.float32),
                                  np.full(1024, meta["bottom_h_W_m2K"], np.float32),
                                  np.zeros(1024, np.float32)))
        features = np.concatenate((k, q, bc), axis=-1)
        if coords.shape != (1024, 3) or features.shape != (1024, 11):
            raise ValueError("formal information-budget shape mismatch")
        return {
            "sample_id": row["sample_id"],
            "coords": torch.from_numpy(coords),
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(np.asarray(np.load(directory / "deltaT.npy"), dtype=np.float32).reshape(-1, 1)),
            "q": np.asarray(q).reshape(-1),
            "control_volume": np.asarray(np.load(directory / "control_volume.npy"), dtype=np.float64).reshape(-1),
            "layer_id": np.asarray(np.load(directory / "layer_id.npy"), dtype=np.int32).reshape(-1),
        }


def normalize(row: dict[str, Any], stats: dict[str, torch.Tensor], device: torch.device):
    coords = row["coords"].unsqueeze(0).to(device)
    features = row["features"].unsqueeze(0).to(device)
    target = row["target"].unsqueeze(0).to(device)
    local = {key: value.to(device) for key, value in stats.items()}
    coords_n = (coords - local["coord_min"]) / torch.clamp(local["coord_max"] - local["coord_min"], 1e-12)
    features_n = (features - local["feature_mean"]) / local["feature_std"]
    target_n = (target - local["target_mean"]) / local["target_std"]
    return coords_n, features_n, target, target_n, local


def predict(model_name: str, model, coords, features, grid):
    if model_name == "GINO":
        return model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
    return model(coords, features)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("GINO", "Transolver"), required=True)
    parser.add_argument("--mode", choices=("contract-check", "preflight", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--statistics", type=Path, default=REPO / "docs/v7_g2_p3_p1i_train_statistics.json")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend-qualification-receipt", type=Path)
    args = parser.parse_args()
    launch_path = REPO / "configs/heat3d_v7" / f"g2_{args.model.lower()}_formal_launch_manifest.json"
    launch = json.loads(launch_path.read_text())
    expected_epochs = 301 if args.model == "GINO" else 500
    if launch["budget"]["epochs"] != expected_epochs or args.seed not in launch["budget"]["seeds"]:
        raise ValueError("launch manifest budget mismatch")
    if args.model == "Transolver" and "decode_prediction_and_target" not in launch["objective"]:
        raise ValueError("Transolver objective is not decoded physical-space relative L2")
    if args.model == "GINO" and launch["architecture"]["input_radius"] != 0.15:
        raise ValueError("GINO formal radius mismatch")
    if args.mode == "contract-check":
        print(json.dumps({"status": "PASS_CONTRACT_CHECK_NO_DATA_NO_TRAINING", "model": args.model,
                          "seed": args.seed, "repo_sha": git_sha(), "launch_manifest_sha256": sha256(launch_path)}, indent=2))
        return 0
    if None in (args.dataset_root, args.dataset_manifest, args.upstream_root, args.output_dir):
        parser.error("preflight/train require dataset-root, dataset-manifest, upstream-root, output-dir")

    if not torch.cuda.is_available():
        raise SystemExit(
            f"FAIL-CLOSED: {args.model} formal preflight/training requires CUDA; "
            "CPU fallback is forbidden"
        )
    if args.model == "GINO" and args.mode == "train":
        if args.backend_qualification_receipt is None:
            parser.error("GINO train requires --backend-qualification-receipt")
        backend_receipt = json.loads(args.backend_qualification_receipt.read_text(encoding="utf-8"))
        if backend_receipt.get("status") != "PASS_OPTIMIZED_BACKEND_QUALIFIED":
            raise ValueError("GINO optimized backend qualification did not PASS")
        if backend_receipt.get("scientific_config_unchanged") != {
            "r_in": 0.15, "r_out": 0.033, "latent_grid": [32, 32, 32]
        }:
            raise ValueError("GINO backend receipt scientific config mismatch")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    stats = load_stats(args.statistics)
    train = P1iRoleDataset(args.dataset_root, args.dataset_manifest, "train")
    valid = P1iRoleDataset(args.dataset_root, args.dataset_manifest, "valid_iid")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    if args.model == "GINO":
        model = build_gino(0.15, 0.033, use_open3d=True, use_torch_scatter=True).to(device)
        if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
            raise RuntimeError("formal GINO silently fell back from Open3D")
        if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
            raise RuntimeError("formal GINO silently fell back from torch-scatter")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        grid = latent_queries(32).to(device)
    else:
        model = build_transolver(args.upstream_root.resolve()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        grid = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.mode == "preflight" else expected_epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best = float("inf"); history = []
    first_train_step_seconds = None
    first_valid_forward_seconds = None
    for epoch in range(epochs):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed * 100000 + epoch)).tolist()
        if args.mode == "preflight": order = order[:1]
        model.train(); train_sum = 0.0
        for index in order:
            row = train[index]
            coords, features, target, target_n, local = normalize(row, stats, device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda": torch.cuda.synchronize()
            step_started = time.perf_counter()
            pred_n = predict(args.model, model, coords, features, grid)
            loss = relative_l2(pred_n, target_n) if args.model == "GINO" else relative_l2(
                pred_n * local["target_std"] + local["target_mean"],
                target_n * local["target_std"] + local["target_mean"],
            )
            loss.backward()
            if args.model == "Transolver": torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if device.type == "cuda": torch.cuda.synchronize()
            if first_train_step_seconds is None: first_train_step_seconds = time.perf_counter() - step_started
            train_sum += float(loss.detach())
        scheduler.step()
        model.eval(); samples = []; reload_probe_prediction = None
        valid_indices = range(1) if args.mode == "preflight" else range(len(valid))
        with torch.no_grad():
            for index in valid_indices:
                row = valid[index]
                coords, features, target, _target_n, local = normalize(row, stats, device)
                if device.type == "cuda": torch.cuda.synchronize()
                valid_started = time.perf_counter()
                pred_n = predict(args.model, model, coords, features, grid)
                if device.type == "cuda": torch.cuda.synchronize()
                if first_valid_forward_seconds is None: first_valid_forward_seconds = time.perf_counter() - valid_started
                pred = (pred_n * local["target_std"] + local["target_mean"]).cpu().numpy().reshape(-1)
                if reload_probe_prediction is None:
                    reload_probe_prediction = torch.from_numpy(pred.copy())
                samples.append(EvaluationSample(row["sample_id"], pred, target.cpu().numpy().reshape(-1),
                                                row["control_volume"], row["coords"].numpy(), row["layer_id"], row["q"]))
        evaluation = EvaluationCore().evaluate(samples)
        selection = evaluation["metrics"]["sample_first_relative_rmse_pct"]
        history.append({"epoch": epoch + 1, "train_objective_mean": train_sum / len(order),
                        "valid_metrics": evaluation["metrics"]})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "seed": args.seed,
                 "repo_sha": git_sha(), "upstream": launch["upstream"],
                 "launch_manifest": launch, "launch_manifest_sha256": sha256(launch_path),
                 "normalization": {key: value.cpu() for key, value in stats.items()},
                 "normalization_payload_sha256": STATS_PAYLOAD_SHA,
                 "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
                 "torch_random_state": torch.get_rng_state(),
                 "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "reload_probe_valid_sample_id": samples[0].sample_id,
                 "reload_probe_prediction": reload_probe_prediction,
                 "backend_qualification_receipt_sha256": (
                     sha256(args.backend_qualification_receipt)
                     if args.model == "GINO" and args.backend_qualification_receipt is not None
                     else None
                 )}
        if args.mode == "train" and selection < best:
            best = selection; torch.save(state, args.output_dir / "best_valid_iid.pt")
        if args.mode == "train" and epoch + 1 == epochs:
            torch.save(state, args.output_dir / "final_epoch.pt")
    reload_checks = {}
    if args.mode == "train":
        for checkpoint_name in ("best_valid_iid.pt", "final_epoch.pt"):
            payload = torch.load(args.output_dir / checkpoint_name, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            model.eval()
            row = valid[0]
            coords, features, _target, _target_n, local = normalize(row, stats, device)
            with torch.no_grad():
                pred_n = predict(args.model, model, coords, features, grid)
                prediction = (pred_n * local["target_std"] + local["target_mean"]).cpu().reshape(-1)
            exact = torch.equal(prediction, payload["reload_probe_prediction"].cpu())
            if not exact:
                raise RuntimeError(f"checkpoint reload prediction drift: {checkpoint_name}")
            reload_checks[checkpoint_name] = {
                "prediction_bitwise_equal": True,
                "epoch": int(payload["epoch"]),
                "valid_sample_id": payload["reload_probe_valid_sample_id"],
            }
    receipt = {"status": "PASS_RESOURCE_PREFLIGHT" if args.mode == "preflight" else "COMPLETE_FORMAL_TRAIN",
               "model": args.model, "seed": args.seed, "mode": args.mode, "device": str(device),
               "resource": {"gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
                            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
                            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None,
                            "first_train_step_wall_seconds": first_train_step_seconds,
                            "first_valid_forward_wall_seconds": first_valid_forward_seconds},
               "epochs": epochs, "history": history, "test_or_sealed_access": False,
               "checkpoint_reload_checks": reload_checks,
               "formal_accuracy_claim_allowed": args.mode == "train"}
    (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
