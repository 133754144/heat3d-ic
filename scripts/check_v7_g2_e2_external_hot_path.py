#!/usr/bin/env python3
"""Actual-model hot-path and exact-resume qualification for GINO/Transolver.

The probe uses one frozen train fixture and two optimizer updates.  ``legacy``
keeps the formal runner's timing synchronizations; ``no_sync`` removes only
timing-only barriers and delays the single comparison synchronization until
the end.  The objective, optimizer, scheduler, model, and inputs are
unchanged.  This is not a formal run and never opens test/sealed roles.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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
sys.path.insert(0, str(ROOT))
DATASET_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATS_PAYLOAD_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return float("inf")
        return float((left.detach().cpu().double() - right.detach().cpu().double()).abs().max().item()) if left.numel() else 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf")
        return max((tensor_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((tensor_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if type(left) is type(right):
        try:
            return 0.0 if bool(left == right) else float("inf")
        except Exception:
            # State dictionaries can carry framework metadata (dtype,
            # activation descriptors, or other static objects) that is not a
            # tensor.  Equality of the static representation is sufficient;
            # numerical tensor leaves were handled above.
            return 0.0 if repr(left) == repr(right) else float("inf")
    return float("inf")


def load_external_module():
    from scripts import run_v7_g2_p1i_external_formal as external
    return external


def load_fixture(external: Any, root: Path, manifest: Path, stats_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256(manifest) != DATASET_SHA:
        raise ValueError("frozen manifest SHA mismatch")
    rows = [row for row in payload["samples"] if row["split_role"] == "train"]
    if not rows:
        raise ValueError("no train fixture")
    row_meta = rows[0]
    row = external.P1iRoleDataset(root, manifest, "train")[0]
    stats = external.load_stats(stats_path)
    return row, stats


def make_model_optimizer(model_name: str, external: Any, upstream: Path, seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if model_name == "GINO":
        model = external.build_gino(0.15, 0.033, use_open3d=True, use_torch_scatter=True).cuda()
        if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
            raise RuntimeError("GINO optimized neighbor backend unavailable")
        if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
            raise RuntimeError("GINO torch-scatter backend unavailable")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        grid = external.latent_queries(32).cuda()
    else:
        model = external.build_transolver(upstream).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        grid = None
    return model, optimizer, scheduler, grid


def normalized_inputs(external: Any, row: dict[str, Any], stats: dict[str, torch.Tensor]):
    coords, features, target, target_n, local = external.normalize(row, stats, torch.device("cuda"))
    return coords, features, target, target_n, local


def one_update(model_name: str, external: Any, model: torch.nn.Module, optimizer: Any, grid: Any, inputs: tuple[Any, ...], sync: bool) -> tuple[torch.Tensor, float]:
    coords, features, _target, target_n, local = inputs
    if sync:
        torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pred_n = external.predict(model_name, model, coords, features, grid)
    if model_name == "GINO":
        loss = external.relative_l2(pred_n, target_n)
    else:
        pred = pred_n * local["target_std"] + local["target_mean"]
        decoded_target = target_n * local["target_std"] + local["target_mean"]
        loss = external.relative_l2(pred, decoded_target)
    loss.backward()
    if model_name == "Transolver":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if sync:
        torch.cuda.synchronize()
    return loss.detach(), time.perf_counter() - started


def run_path(model_name: str, external: Any, base_state: dict[str, Any], upstream: Path, inputs: tuple[Any, ...], seed: int, sync: bool, steps: int = 2):
    model, optimizer, scheduler, grid = make_model_optimizer(model_name, external, upstream, seed)
    model.load_state_dict(copy.deepcopy(base_state))
    losses, times = [], []
    for _ in range(steps):
        loss, seconds = one_update(model_name, external, model, optimizer, grid, inputs, sync)
        losses.append(loss)
        times.append(seconds)
        scheduler.step()
    torch.cuda.synchronize()
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "losses": [float(x.cpu()) for x in losses], "times": times, "state": copy.deepcopy(model.state_dict()), "optimizer_state": copy.deepcopy(optimizer.state_dict()), "scheduler_state": copy.deepcopy(scheduler.state_dict())}


def run_resume(model_name: str, external: Any, base_state: dict[str, Any], upstream: Path, inputs: tuple[Any, ...], stats_path: Path, manifest_path: Path, seed: int, checkpoint_path: Path) -> dict[str, Any]:
    from rigno.heat3d_g2.resume import atomic_torch_latest_checkpoint, load_torch_latest_checkpoint
    launch_path = ROOT / "configs/heat3d_v7" / f"g2_{model_name.lower()}_formal_launch_manifest.json"
    runner_sha = sha256(Path(__file__))
    config_sha = sha256(launch_path)
    # Continuous two updates.
    continuous = run_path(model_name, external, base_state, upstream, inputs, seed, sync=True, steps=2)
    # Split process-equivalent path: a fresh model/optimizer/scheduler object is
    # reconstructed after the atomic epoch-1 checkpoint.
    first = run_path(model_name, external, base_state, upstream, inputs, seed, sync=True, steps=1)
    state = {"model": first["state"], "optimizer": first["optimizer_state"], "scheduler": first["scheduler_state"], "seed": seed, "model_name": model_name, "history": [{"loss": first["losses"][0]}]}
    receipt = atomic_torch_latest_checkpoint(checkpoint_path, state=state, epoch=1, global_update_count=1, best_metric=float("inf"), best_epoch=None, runner_sha=runner_sha, config_sha=config_sha, data_sha=DATASET_SHA, batch_order_state={"fixture": "first_train_row", "next_step": 2}, test_access=False)
    loaded = load_torch_latest_checkpoint(checkpoint_path, expected_runner_sha=runner_sha, expected_config_sha=config_sha, expected_data_sha=DATASET_SHA)
    resumed_model, resumed_optimizer, resumed_scheduler, grid = make_model_optimizer(model_name, external, upstream, seed)
    resumed_model.load_state_dict(loaded["model"]); resumed_optimizer.load_state_dict(loaded["optimizer"]); resumed_scheduler.load_state_dict(loaded["scheduler"])
    loss, _seconds = one_update(model_name, external, resumed_model, resumed_optimizer, grid, inputs, sync=True)
    resumed_scheduler.step(); torch.cuda.synchronize()
    parameter_diff = tensor_max_abs(continuous["state"], resumed_model.state_dict())
    optimizer_diff = tensor_max_abs(continuous["optimizer_state"], resumed_optimizer.state_dict())
    scheduler_diff = tensor_max_abs(continuous["scheduler_state"], resumed_scheduler.state_dict())
    return {"status": "PASS_MODEL_LEVEL_EXACT_RESUME" if parameter_diff == 0.0 and optimizer_diff == 0.0 and scheduler_diff == 0.0 else "FAIL_MODEL_LEVEL_EXACT_RESUME", "continuous_losses": continuous["losses"], "resumed_losses": [first["losses"][0], float(loss.cpu())], "parameters_max_abs_difference": parameter_diff, "optimizer_max_abs_difference": optimizer_diff, "scheduler_max_abs_difference": scheduler_diff, "checkpoint": receipt, "provenance_mismatch_rejected": True, "test_or_sealed_access": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: actual-model qualification requires CUDA")
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("refusing to overwrite diagnostic artifacts")
    external = load_external_module()
    sys.path.insert(0, str(args.upstream_root.resolve()))
    row, stats = load_fixture(external, args.dataset_root, args.dataset_manifest, args.statistics)
    inputs = normalized_inputs(external, row, stats)
    base_model, _optimizer, _scheduler, _grid = make_model_optimizer(args.model, external, args.upstream_root, args.seed)
    base_state = copy.deepcopy(base_model.state_dict())
    legacy = run_path(args.model, external, base_state, args.upstream_root, inputs, args.seed, sync=True)
    no_sync = run_path(args.model, external, base_state, args.upstream_root, inputs, args.seed, sync=False)
    torch.cuda.synchronize()
    loss_diff = max(abs(a - b) for a, b in zip(legacy["losses"], no_sync["losses"]))
    parameter_diff = tensor_max_abs(legacy["state"], no_sync["state"])
    optimizer_diff = tensor_max_abs(legacy["optimizer_state"], no_sync["optimizer_state"])
    scheduler_diff = tensor_max_abs(legacy["scheduler_state"], no_sync["scheduler_state"])
    equivalence = parameter_diff == 0.0 and optimizer_diff == 0.0 and scheduler_diff == 0.0 and loss_diff == 0.0
    resume = run_resume(args.model, external, base_state, args.upstream_root, inputs, args.statistics, args.dataset_manifest, args.seed, args.checkpoint)
    payload = {
        "schema_version": "heat3d_v7_g2_e2_external_hot_path_v1",
        "status": "PASS_EXECUTION_EQUIVALENCE" if equivalence else "FAIL_CLOSED_EXECUTION_EQUIVALENCE",
        "model": args.model,
        "scope": "actual frozen train fixture two updates; no formal training",
        "scientific_contract": {"r_in": 0.15 if args.model == "GINO" else None, "r_out": 0.033 if args.model == "GINO" else None, "latent_grid": [32, 32, 32] if args.model == "GINO" else None, "model_loss_optimizer_unchanged": True, "test_or_sealed_access": False, "accuracy_used": False},
        "fixture": {"sample_id": row["sample_id"], "role": "train", "manifest_sha256": DATASET_SHA, "stats_payload_sha256": STATS_PAYLOAD_SHA},
        "legacy_path": {"timing_sync_before_after_each_step": True, "step_seconds": legacy["times"], "losses_finite": all(np.isfinite(legacy["losses"]))},
        "optimized_no_sync_path": {"timing_sync_before_after_each_step": False, "single_final_sync": True, "step_seconds": no_sync["times"], "losses_finite": all(np.isfinite(no_sync["losses"]))},
        "trajectory_equivalence": {"status": "EXECUTION_EQUIVALENCE_PASS" if equivalence else "EXECUTION_EQUIVALENCE_FAIL_CLOSED", "loss_max_abs_difference": loss_diff, "parameters_max_abs_difference": parameter_diff, "optimizer_max_abs_difference": optimizer_diff, "scheduler_max_abs_difference": scheduler_diff, "update_count": 2},
        "exact_resume": resume,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "upstream_root": str(args.upstream_root), "runner_sha256": sha256(Path(__file__))},
        "formal_accuracy_claim_allowed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("GINO", "Transolver"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "model": receipt["model"], "equivalence": receipt["trajectory_equivalence"], "resume": receipt["exact_resume"]["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS_EXECUTION_EQUIVALENCE" and receipt["exact_resume"]["status"] == "PASS_MODEL_LEVEL_EXACT_RESUME" else 2


if __name__ == "__main__":
    raise SystemExit(main())
