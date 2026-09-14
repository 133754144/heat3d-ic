#!/usr/bin/env python3
"""Bounded 50-update GINO reproducibility and process-resume qualification.

Open3D FixedRadiusSearch plus torch-scatter is the authoritative upstream
backend.  This probe uses only one frozen train fixture for updates and one
valid_iid fixture for fixed-probe predictions.  It records absolute and
RMS-normalized update drift, launches every comparison in a fresh process,
and performs a real save/exit/resume pair.  No formal training or test access
is possible through this entry point.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import pickle
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
UPSTREAM_COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
R_IN, R_OUT, LATENT = 0.15, 0.033, 32
STEPS = 50
SPLIT_STEPS = 25
PROBE_LIMIT = 16384


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


def tree_arrays(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy().copy()
    if isinstance(value, dict):
        return {key: tree_arrays(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        converted = [tree_arrays(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return copy.deepcopy(value)


def tree_probe(value: Any, limit: int = PROBE_LIMIT) -> dict[str, Any]:
    leaves: list[np.ndarray] = []
    remaining = int(limit)

    def visit(item: Any) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        if isinstance(item, np.ndarray):
            flat = item.reshape(-1)
            take = min(len(flat), remaining)
            leaves.append(np.asarray(flat[:take], dtype=np.float64).copy())
            remaining -= take
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {"leaf_values": leaves, "budget": int(limit), "used": int(limit - remaining)}


def probe_finite(probe: dict[str, Any]) -> bool:
    """Check each bounded probe leaf without coercing ragged arrays."""

    return all(
        bool(np.all(np.isfinite(np.asarray(leaf, dtype=np.float64))))
        for leaf in probe.get("leaf_values", [])
    )


def tree_sub(left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return np.asarray([np.inf], dtype=np.float64)
        return np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return np.asarray([np.inf], dtype=np.float64)
        return {key: tree_sub(left[key], right[key]) for key in sorted(left)}
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return np.asarray([np.inf], dtype=np.float64)
        values = [tree_sub(a, b) for a, b in zip(left, right)]
        return tuple(values) if isinstance(left, tuple) else values
    return 0.0 if left == right else np.asarray([np.inf], dtype=np.float64)


def tensor_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        digest.update(str(array.dtype).encode()); digest.update(repr(tuple(array.shape)).encode()); digest.update(array.numpy().tobytes(order="C")); return digest.hexdigest()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode()); digest.update(tensor_hash(value[key]).encode())
        return digest.hexdigest()
    digest.update(repr(value).encode()); return digest.hexdigest()


def tree_relative(left: Any, right: Any) -> dict[str, float]:
    left_values: list[np.ndarray] = []
    right_values: list[np.ndarray] = []

    def visit(a: Any, b: Any) -> None:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape:
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            left_values.append(np.asarray(a, dtype=np.float64).reshape(-1)); right_values.append(np.asarray(b, dtype=np.float64).reshape(-1)); return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            for key in sorted(a): visit(a[key], b[key])
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            for x, y in zip(a, b): visit(x, y)
            return
        if a != b:
            left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0]))

    visit(left, right)
    if not left_values:
        return {"relative_l2": 0.0, "max_abs": 0.0, "rms_normalized": 0.0}
    diff = np.concatenate([a - b for a, b in zip(left_values, right_values)])
    base = np.concatenate(right_values)
    return {
        "relative_l2": float(np.linalg.norm(diff) / max(float(np.linalg.norm(base)), 1.0e-12)),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_normalized": float(np.sqrt(np.mean(diff * diff)) / max(float(np.sqrt(np.mean(base * base))), 1.0e-12)) if diff.size else 0.0,
    }


def load_inputs(args: argparse.Namespace, external: Any, fixture_module: Any) -> dict[str, tuple[Any, ...]]:
    inputs, _ = load_module("g2_e4_e3_loader", ROOT / "scripts/qualify_v7_g2_e3_gino_reproducibility.py").load_inputs(args, external, fixture_module)
    return inputs


def build_model(external: Any, seed: int) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.Tensor]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = external.build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).cuda()
    if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
        raise RuntimeError("Open3D backend unavailable")
    if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
        raise RuntimeError("torch-scatter backend unavailable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    return model, optimizer, scheduler, external.latent_queries(LATENT).cuda()


def run_updates(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, grid: torch.Tensor, external: Any, train_input: tuple[Any, ...], valid_input: tuple[Any, ...], seed: int, start_step: int, end_step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_coords, train_features, train_target, train_target_n, train_meta = train_input
    valid_coords, valid_features, _valid_target, _valid_target_n, valid_meta = valid_input
    del train_target, train_meta
    model.train()
    previous = tree_arrays(model.state_dict())
    rows: list[dict[str, Any]] = []
    for step_index in range(start_step, end_step):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); started = time.perf_counter()
        prediction = model(input_geom=train_coords, latent_queries=grid, output_queries=train_coords, x=train_features)
        loss = external.relative_l2(prediction, train_target_n)
        loss.backward(); optimizer.step(); scheduler.step()
        torch.cuda.synchronize(); wall = time.perf_counter() - started
        current = tree_arrays(model.state_dict())
        # The probe is the parameter increment, not a new learned quantity.
        update_probe = tree_probe(tree_sub(current, previous))
        current_probe = tree_probe(current)
        loss_value = float(loss.detach().cpu())
        with torch.no_grad():
            valid_prediction = model(input_geom=valid_coords, latent_queries=grid, output_queries=valid_coords, x=valid_features)
            valid_prediction = valid_prediction.detach().cpu().numpy().copy()
        prediction_probe = tree_probe(prediction.detach().cpu().numpy())
        rows.append({"step": step_index + 1, "wall_seconds": wall, "loss": loss_value, "loss_finite": bool(np.isfinite(loss_value)), "prediction_probe": prediction_probe, "valid_prediction_probe": tree_probe(valid_prediction), "parameter_probe": current_probe, "update_probe": update_probe, "parameter_hash": tensor_hash(current), "finite": bool(np.isfinite(loss_value) and probe_finite(current_probe) and probe_finite(update_probe) and probe_finite(prediction_probe))})
        previous = current
    final = {"model": tree_arrays(model.state_dict()), "optimizer": tree_arrays(optimizer.state_dict()), "scheduler": tree_arrays(scheduler.state_dict()), "step": end_step}
    return rows, final


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: GINO production qualification requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.phase != "resume":
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    external = load_module("g2_e4_external", ROOT / "scripts/run_v7_g2_p1_local_qualification.py")
    fixture_module = load_module("g2_e4_fixture", ROOT / "scripts/check_v7_g2_p9_gino_repeatability.py")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    inputs = load_inputs(args, external, fixture_module)
    train_input, valid_input = inputs["train"], inputs["valid_iid"]
    model, optimizer, scheduler, grid = build_model(external, args.seed)
    initial_state = copy.deepcopy(model.state_dict())
    runner_sha = sha256(Path(__file__).resolve())
    if args.phase == "resume":
        if args.resume_from is None or not args.resume_from.is_file():
            raise ValueError("resume checkpoint missing")
        payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        for key, expected in (("runner_sha", runner_sha), ("dataset_sha", DATASET_SHA), ("statistics_sha", STATS_SHA), ("upstream_commit", UPSTREAM_COMMIT)):
            if str(payload.get(key)) != str(expected):
                raise ValueError(f"resume provenance mismatch: {key}")
        if payload.get("test_or_sealed_access") is not False or int(payload.get("seed", args.seed)) != args.seed:
            raise ValueError("resume seed/test contract mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"]); end_step = STEPS
        rows, final = run_updates(model, optimizer, scheduler, grid, external, train_input, valid_input, args.seed, start_step, end_step)
    else:
        end_step = SPLIT_STEPS if args.phase == "split-save" else STEPS
        rows, final = run_updates(model, optimizer, scheduler, grid, external, train_input, valid_input, args.seed, 0, end_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.phase == "split-save":
        checkpoint = args.output_dir / "mid_checkpoint.pt"
        if checkpoint.exists(): raise FileExistsError("refusing to overwrite mid checkpoint")
        # Keep the resume artifact in native torch state-dict form.  The
        # NumPy snapshot in ``final`` is intentionally reserved for compact
        # trace comparison and is not a valid ``load_state_dict`` input.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": SPLIT_STEPS, "seed": args.seed, "runner_sha": runner_sha, "dataset_sha": DATASET_SHA, "statistics_sha": STATS_SHA, "upstream_commit": UPSTREAM_COMMIT, "test_or_sealed_access": False}, checkpoint)
    final_state_path = args.output_dir / "final_state.pt"
    if args.phase != "split-save":
        if final_state_path.exists(): raise FileExistsError("refusing to overwrite final state")
        torch.save(final, final_state_path)
    receipt = {
        "schema_version": "heat3d_v7_g2_e4_gino_production_child_v1", "status": "PASS_FINITE" if all(row["finite"] for row in rows) else "FAIL_NONFINITE", "phase": args.phase, "seed": args.seed, "steps_completed": end_step,
        "rows": [{key: value for key, value in row.items() if key not in {"prediction_probe", "valid_prediction_probe", "parameter_probe", "update_probe"}} for row in rows],
        "losses": [row["loss"] for row in rows], "median_step_seconds": float(np.median([row["wall_seconds"] for row in rows])) if rows else None, "p95_step_seconds": float(np.percentile([row["wall_seconds"] for row in rows], 95)) if rows else None,
        "checkpoint": {"path": str(checkpoint) if checkpoint else None, "sha256": sha256(checkpoint) if checkpoint else None}, "final_state_sha256": sha256(final_state_path) if final_state_path.exists() else None,
        "runner_sha256": runner_sha, "dataset_sha": DATASET_SHA, "statistics_sha": STATS_SHA, "upstream_commit": UPSTREAM_COMMIT, "backend": "Open3D_FixedRadiusSearch_plus_torch_scatter", "test_or_sealed_access": False,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "open3d": __import__("importlib.metadata", fromlist=["version"]).version("open3d"), "torch_scatter": __import__("importlib.metadata", fromlist=["version"]).version("torch-scatter")},
    }
    receipt_path = args.output_dir / f"receipt_{args.phase}.json"
    if receipt_path.exists(): raise FileExistsError("refusing to overwrite child receipt")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / f"trace_{args.phase}.pkl").open("wb") as stream:
        pickle.dump({"rows": rows, "final": final}, stream, protocol=5)
    print(json.dumps({"status": receipt["status"], "phase": args.phase, "seed": args.seed, "steps": end_step, "median_step_seconds": receipt["median_step_seconds"]}, sort_keys=True), flush=True)
    return receipt


def compare_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for a, b in zip(left, right):
        metrics = {}
        for key in ("prediction_probe", "parameter_probe", "update_probe"):
            metrics[key] = tree_relative(a[key], b[key])
        metrics["loss_relative_difference"] = abs(float(a["loss"]) - float(b["loss"])) / max(abs(float(a["loss"])), 1.0e-12)
        rows.append({"step": a["step"], **metrics})
    return {"steps": rows}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()): raise FileExistsError(f"refusing to overwrite {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Child traces contain optimizer/state metadata whose pickle provenance
    # references the pinned upstream package.  Make that immutable checkout
    # importable before unpickling; this does not alter model execution.
    sys.path.insert(0, str(args.upstream_root.resolve()))
    common = ["--dataset-root", str(args.dataset_root), "--dataset-manifest", str(args.dataset_manifest), "--statistics", str(args.statistics), "--upstream-root", str(args.upstream_root)]
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    specs = [("same_seed_0", 0, "continuous", args.work_dir / "same_0"), ("same_seed_0_repeat1", 0, "continuous", args.work_dir / "same_1"), ("same_seed_0_repeat2", 0, "continuous", args.work_dir / "same_2"), ("inter_seed_0", 0, "continuous", args.work_dir / "inter_0"), ("inter_seed_1", 1, "continuous", args.work_dir / "inter_1"), ("inter_seed_2", 2, "continuous", args.work_dir / "inter_2")]
    receipts: dict[str, Any] = {}; traces: dict[str, Any] = {}
    for name, seed, phase, directory in specs:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--phase", phase, *common, "--output-dir", str(directory), "--seed", str(seed)]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        if result.returncode != 0: raise RuntimeError(f"GINO {name} failed: {result.stdout[-1000:]} {result.stderr[-2000:]}")
        receipts[name] = json.loads((directory / f"receipt_{phase}.json").read_text()); traces[name] = pickle.loads((directory / f"trace_{phase}.pkl").read_bytes())
    resume_dir = args.work_dir / "resume_seed0"; save_command = [sys.executable, str(Path(__file__).resolve()), "--child", "--phase", "split-save", *common, "--output-dir", str(resume_dir), "--seed", "0"]
    result = subprocess.run(save_command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0: raise RuntimeError(f"GINO split-save failed: {result.stdout[-1000:]} {result.stderr[-2000:]}")
    mid = resume_dir / "mid_checkpoint.pt"; resume_command = [sys.executable, str(Path(__file__).resolve()), "--child", "--phase", "resume", *common, "--output-dir", str(resume_dir), "--resume-from", str(mid), "--seed", "0"]
    result = subprocess.run(resume_command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0: raise RuntimeError(f"GINO resume failed: {result.stdout[-1000:]} {result.stderr[-2000:]}")
    receipts["split_save"] = json.loads((resume_dir / "receipt_split-save.json").read_text()); receipts["resume"] = json.loads((resume_dir / "receipt_resume.json").read_text()); traces["split_save"] = pickle.loads((resume_dir / "trace_split-save.pkl").read_bytes()); traces["resume"] = pickle.loads((resume_dir / "trace_resume.pkl").read_bytes())
    same_pairs = [compare_rows(traces["same_seed_0"]["rows"], traces[name]["rows"]) for name in ("same_seed_0_repeat1", "same_seed_0_repeat2")]
    inter_pairs = [compare_rows(traces["inter_seed_0"]["rows"], traces[name]["rows"]) for name in ("inter_seed_1", "inter_seed_2")]
    resume_pair = compare_rows(traces["same_seed_0"]["rows"][SPLIT_STEPS:], traces["resume"]["rows"])
    def median_metric(pairs: list[dict[str, Any]], key: str, sub: str | None = None) -> float:
        values = []
        for pair in pairs:
            for row in pair["steps"]:
                value = row[key] if sub is None else row[key][sub]
                values.append(float(value))
        return float(np.median(values)) if values else float("nan")
    paths = {"loss": ("loss_relative_difference", None), "prediction": ("prediction_probe", "rms_normalized"), "parameters": ("parameter_probe", "rms_normalized"), "updates_rms": ("update_probe", "rms_normalized"), "updates_abs": ("update_probe", "max_abs")}
    same = {name: median_metric(same_pairs, key, sub) for name, (key, sub) in paths.items()}
    inter = {name: median_metric(inter_pairs, key, sub) for name, (key, sub) in paths.items()}
    ratios = {name: float(same[name] / max(inter[name], 1.0e-12)) for name in same}
    resume = {name: median_metric([resume_pair], key, sub) for name, (key, sub) in paths.items()}
    resume_vs_same = {name: float(resume[name] / max(same[name], 1.0e-12)) for name in resume}
    continuous_state = torch.load(args.work_dir / "same_0" / "final_state.pt", map_location="cpu", weights_only=False)
    resumed_state = torch.load(resume_dir / "final_state.pt", map_location="cpu", weights_only=False)
    resume_final_relative = {"model": tree_relative(continuous_state["model"], resumed_state["model"]), "optimizer": tree_relative(continuous_state["optimizer"], resumed_state["optimizer"]), "scheduler": tree_relative(continuous_state["scheduler"], resumed_state["scheduler"])}
    finite = all(receipt["status"] == "PASS_FINITE" for receipt in receipts.values()) and all(receipt["status"] == "PASS_FINITE" for receipt in (receipts["split_save"], receipts["resume"]))
    resume_within_noise = all(resume[name] <= max(same[name], 1.0e-12) for name in ("loss", "prediction", "parameters", "updates_rms"))
    stable = finite and all(np.isfinite(list(same.values()))) and all(np.isfinite(list(inter.values()))) and all(ratios[name] < 0.1 for name in ("loss", "prediction", "parameters", "updates_rms")) and all(np.isfinite(list(resume.values()))) and resume_within_noise and all(np.isfinite(list(item["relative_l2"] for item in resume_final_relative.values())))
    payload = {"schema_version": "heat3d_v7_g2_e4_gino_production_aggregate_v1", "status": "GINO_AUTHOR_SEMANTICS_QUALIFIED" if stable else "GINO_NUMERICAL_STABILITY_FAIL_CLOSED", "scope": "devbox-only authoritative Open3D+torch-scatter; fresh process 50-update repeats plus real 25-update save/exit/resume", "authoritative_backend": "Open3D_FixedRadiusSearch_plus_torch_scatter", "children": {name: {"status": value["status"], "seed": value["seed"], "median_step_seconds": value["median_step_seconds"], "p95_step_seconds": value["p95_step_seconds"], "final_state_sha256": value.get("final_state_sha256")} for name, value in receipts.items()}, "same_seed_median": same, "inter_seed_median": inter, "same_over_inter_ratio": ratios, "resume_median": resume, "resume_over_same_seed_noise": resume_vs_same, "resume_checkpoint_restore": {"mid_checkpoint_sha256": sha256(mid), "save_steps": SPLIT_STEPS, "resume_steps": STEPS - SPLIT_STEPS, "final_state_compared": True, "final_state_relative": resume_final_relative, "within_same_seed_noise": resume_within_noise}, "decision": {"finite": finite, "same_seed_noise_target_lt_0.1": all(ratios[name] < 0.1 for name in ("loss", "prediction", "parameters", "updates_rms")), "absolute_and_rms_update_reported": True, "resume_within_same_seed_noise": resume_within_noise, "no_accuracy_used": True, "formal_training_started": False, "test_or_sealed_access": False, "classification": "GINO_AUTHOR_SEMANTICS_QUALIFIED" if stable else "GINO_NUMERICAL_STABILITY_FAIL_CLOSED"}, "environment": receipts["same_seed_0"]["environment"], "scientific_contract": {"dataset_sha256": DATASET_SHA, "statistics_sha256": STATS_SHA, "upstream_commit": UPSTREAM_COMMIT, "r_in": R_IN, "r_out": R_OUT, "latent_grid": [LATENT] * 3, "steps": STEPS, "architecture_changed": False, "information_budget_changed": False}}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--child", action="store_true"); parser.add_argument("--phase", choices=("continuous", "split-save", "resume")); parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--dataset-manifest", type=Path, required=True); parser.add_argument("--statistics", type=Path, required=True); parser.add_argument("--upstream-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path); parser.add_argument("--resume-from", type=Path); parser.add_argument("--work-dir", type=Path, default=Path("/tmp/g2_e4_gino_production")); parser.add_argument("--output", type=Path, default=Path("/tmp/g2_e4_gino_production.json")); parser.add_argument("--seed", type=int, default=0); args = parser.parse_args()
    # The GINO input loader uses the explicit dataset/manifest/statistics names;
    # the unused Heat3D arguments are not accepted in child mode.
    if args.child:
        if args.phase is None or args.output_dir is None: parser.error("child requires --phase and --output-dir")
        receipt = run_child(args); return 0 if receipt["status"] == "PASS_FINITE" else 2
    receipt = aggregate(args); print(json.dumps({"status": receipt["status"], "same_over_inter_ratio": receipt["same_over_inter_ratio"]}, sort_keys=True)); return 0 if receipt["status"] == "GINO_AUTHOR_SEMANTICS_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
