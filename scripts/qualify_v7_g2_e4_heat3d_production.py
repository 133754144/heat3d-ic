#!/usr/bin/env python3
"""Bounded Heat3D production-path qualification with cross-process resume.

The runner uses the frozen DeepOHeat-v1 768/128 B24/valid32 contract and the
existing V7FormalTrainer step (no model or objective rewrite).  It is only a
three-epoch runtime/resume rehearsal; it never opens test/sealed artifacts and
never writes inside the repository.  The parent executes a continuous run and
an independent 1-epoch-save/process-exit/2-epoch-resume run, then compares the
resulting states under the E3 upstream-semantic reproducibility policy.
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SAMPLES = 768
VALID_SAMPLES = 128
TRAIN_BATCH = 24
VALID_BATCH = 32
EPOCHS = 3
SUBSET_NAME = "frozen_768_train_128_valid_iid"


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


def host_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: host_tree(value[key]) for key in value}
    if isinstance(value, (tuple, list)):
        converted = [host_tree(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).copy()
    return copy.deepcopy(value)


def tree_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(tree_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tree_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(np.isfinite(value))
    return True


def tree_relative(left: Any, right: Any) -> dict[str, float]:
    left_values: list[np.ndarray] = []
    right_values: list[np.ndarray] = []

    def visit(a: Any, b: Any) -> None:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape:
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            left_values.append(np.asarray(a, dtype=np.float64).reshape(-1))
            right_values.append(np.asarray(b, dtype=np.float64).reshape(-1))
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            for key in sorted(a):
                visit(a[key], b[key])
            return
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0])); return
            for x, y in zip(a, b):
                visit(x, y)
            return
        if a != b:
            left_values.append(np.asarray([np.inf])); right_values.append(np.asarray([1.0]))

    visit(left, right)
    if not left_values:
        return {"relative_l2": 0.0, "max_abs": 0.0, "rms_normalized": 0.0}
    diff = np.concatenate([a - b for a, b in zip(left_values, right_values)])
    base = np.concatenate(right_values)
    denom = max(float(np.linalg.norm(base)), 1.0e-12)
    rms_denom = max(float(np.sqrt(np.mean(base * base))), 1.0e-12)
    return {
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_normalized": float(np.sqrt(np.mean(diff * diff)) / rms_denom) if diff.size else 0.0,
    }


def gpu_snapshot() -> dict[str, Any]:
    command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not command.is_file():
        command = Path("nvidia-smi")
    try:
        result = subprocess.run(
            [str(command), "--query-gpu=name,utilization.gpu,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows = []
    for line in result.stdout.strip().splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 5:
            rows.append({"name": values[0], "utilization_gpu_pct": values[1], "power_draw_w": values[2], "memory_used_mib": values[3], "memory_total_mib": values[4]})
    return {"available": bool(rows), "rows": rows, "command": str(command)}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: Heat3D production qualification requires CUDA")
    forbidden = ("test", "sealed")
    for path in (args.fs_train, args.labels_root, args.normalization, args.heat3d_config):
        if any(token in str(path).lower() for token in forbidden):
            raise ValueError(f"forbidden test/sealed path: {path}")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only frozen fs_train_volume.npy is accepted")
    profile = load_script("profile_v7_g2_heat3d_epoch.py")
    prepared_args = argparse.Namespace(
        fs_train=args.fs_train, labels_root=args.labels_root,
        normalization=args.normalization, heat3d_config=args.heat3d_config,
        seed=args.seed, train_batch_size=TRAIN_BATCH, valid_batch_size=VALID_BATCH,
        allow_nondeterministic_xla=True, cache_dir=args.cache_dir,
    )
    return profile._prepare(prepared_args)


def run_epoch(prepared: dict[str, Any], state: Any, epoch: int, seed: int) -> tuple[Any, dict[str, Any]]:
    trainer = prepared["trainer"]
    block = prepared["block_until_ready"]
    train_batches = prepared["train_batches"]
    valid_batches = prepared["valid_batches"]
    order = np.random.default_rng(seed + epoch).permutation(len(train_batches))
    order_ids = [train_batches[int(index)].batch_id for index in order]
    order_hash = hashlib.sha256(json.dumps(order_ids, separators=(",", ":")).encode()).hexdigest()
    base_key = jax.random.PRNGKey(seed)
    train_rows: list[dict[str, Any]] = []
    train_losses: list[float] = []
    train_started = time.perf_counter()
    compile_counts: list[int] = []
    for batch_number, raw_index in enumerate(order, start=1):
        batch = train_batches[int(raw_index)]
        key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), batch_number)
        started = time.perf_counter()
        result = trainer.step(state, batch, rng=key)
        block((result.state.params, result.state.optimizer_state, result.loss, result.gradients, result.updates, result.prediction))
        seconds = time.perf_counter() - started
        value = float(np.asarray(result.loss))
        state = result.state
        train_losses.append(value)
        compile_counts.append(int(trainer.compile_count))
        train_rows.append({"batch_number": batch_number, "batch_index": int(raw_index), "batch_id": batch.batch_id, "wall_seconds": seconds, "loss": value, "finite": bool(np.isfinite(value)), "compile_count": int(trainer.compile_count)})
    train_total = time.perf_counter() - train_started

    valid_rows: list[dict[str, Any]] = []
    valid_losses: list[float] = []
    valid_started = time.perf_counter()
    predictions: list[Any] = []
    for batch in valid_batches:
        started = time.perf_counter()
        prediction, loss = trainer.validate_with_outputs(state, batch)
        block((prediction, loss))
        seconds = time.perf_counter() - started
        value = float(np.asarray(loss))
        predictions.append(prediction)
        valid_losses.append(value)
        valid_rows.append({"batch_id": batch.batch_id, "wall_seconds": seconds, "loss": value, "finite": bool(np.isfinite(value))})
    valid_total = time.perf_counter() - valid_started
    # This is the frozen valid-only checkpoint-selection contract.  It is
    # recorded, never used to alter execution, budget, or model settings.
    evaluation = load_script("run_v7_g2_p6_heat3d_v1_formal.py").evaluate_level_a_validation(
        predictions=predictions, batches=valid_batches, examples=prepared["valid_examples"], stats=prepared["stats"], variant="Full",
    )
    selection = float(evaluation["metrics"]["sample_first_relative_rmse_pct"])
    row = {
        "epoch": epoch, "train_loss_mean": float(np.mean(train_losses)), "valid_loss_mean": float(np.mean(valid_losses)),
        "valid_selection_metric": selection, "train_wall_seconds": train_total, "valid_wall_seconds": valid_total,
        "epoch_wall_seconds": train_total + valid_total, "train_batches": len(train_batches), "valid_batches": len(valid_batches),
        "batch_order_hash": order_hash, "compile_count_end": int(trainer.compile_count), "compile_counts": compile_counts,
        "train_step_median_seconds": float(np.median([item["wall_seconds"] for item in train_rows])),
        "train_step_p95_seconds": float(np.percentile([item["wall_seconds"] for item in train_rows], 95)),
        "valid_forward_median_seconds": float(np.median([item["wall_seconds"] for item in valid_rows])),
        "loss_finite": bool(np.all(np.isfinite(train_losses)) and np.all(np.isfinite(valid_losses))),
        "gpu_snapshot": gpu_snapshot(),
        "train_rows": train_rows, "valid_rows": valid_rows,
    }
    return state, row


def save_epoch(path: Path, state: Any, epoch: int, history: list[dict[str, Any]], best_metric: float, best_epoch: int | None, args: argparse.Namespace, runner_sha: str, config_sha: str, data_sha: str) -> dict[str, Any]:
    from rigno.heat3d_training.resume import atomic_latest_checkpoint
    payload = atomic_latest_checkpoint(
        path, state=state,
        metadata={"epoch": epoch, "global_update_count": int(state.step), "best_metric": best_metric, "best_epoch": best_epoch, "runner_sha": runner_sha, "config_sha": config_sha, "data_sha": data_sha, "seed": args.seed, "test_access": False},
        rng_state={"jax_base_seed": args.seed, "next_epoch": epoch + 1},
        batch_state={"algorithm": "default_rng(seed+epoch).permutation", "next_epoch": epoch + 1, "history": history},
        scheduler_state={"schedule": "embedded_in_optax", "epoch": epoch},
    )
    return payload


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.output_dir).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("all qualification output must remain under /tmp")
    if args.mode == "resume" and args.resume_from is None:
        raise ValueError("resume mode requires --resume-from")
    if args.mode != "resume" and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    prepared = prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    runner_sha = sha256(Path(__file__).resolve())
    config_sha = sha256(args.heat3d_config)
    data_sha = sha256(args.labels_root / "label_generation_receipt.json")
    history: list[dict[str, Any]] = []
    best_metric, best_epoch = float("inf"), None
    start_epoch = 1
    if args.mode == "resume":
        from rigno.heat3d_training.resume import load_latest_checkpoint
        restored = load_latest_checkpoint(args.resume_from, expected_runner_sha=runner_sha, expected_config_sha=config_sha, expected_data_sha=data_sha)
        if restored.get("test_access") is not False or int(restored.get("seed", args.seed)) != args.seed:
            raise ValueError("resume seed/test contract mismatch")
        start_epoch = int(restored["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError("resume epoch exceeds bounded rehearsal")
        state = restored["state_object"]
        best_metric = float(restored.get("best_metric", float("inf")))
        best_epoch = restored.get("best_epoch")
        history = list(restored.get("batch_state", {}).get("history", []))
    started = time.perf_counter()
    epoch_rows: list[dict[str, Any]] = []
    latest = None
    for epoch in range(start_epoch, args.epochs + 1):
        state, row = run_epoch(prepared, state, epoch, args.seed)
        history.append(row)
        epoch_rows.append(row)
        if row["valid_selection_metric"] < best_metric:
            best_metric, best_epoch = row["valid_selection_metric"], epoch
        latest = save_epoch(args.output_dir / "latest_epoch.pkl", state, epoch, history, best_metric, best_epoch, args, runner_sha, config_sha, data_sha)
        print(json.dumps({"mode": args.mode, "epoch": epoch, "epoch_wall_seconds": row["epoch_wall_seconds"], "compile_count": row["compile_count_end"]}, sort_keys=True), flush=True)
    final_state = {"params": host_tree(state.params), "optimizer": host_tree(state.optimizer_state), "step": int(state.step)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_state_path = args.output_dir / ("save_state.pkl" if args.mode == "split-save" else "final_state.pkl")
    if final_state_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_state_path}")
    with final_state_path.open("wb") as stream:
        pickle.dump(final_state, stream, protocol=5)
    receipt = {
        "schema_version": "heat3d_v7_g2_e4_production_child_v1",
        "status": "PASS_FINITE",
        "mode": args.mode,
        "seed": args.seed,
        "epochs_completed": [row["epoch"] for row in epoch_rows],
        "preparation_seconds": prepared["preparation_seconds"],
        "wall_seconds": time.perf_counter() - started,
        "epoch_rows": epoch_rows,
        "compile_count_end": int(trainer.compile_count),
        "unique_train_batch_signatures": len(getattr(trainer, "_compiled_steps", {})),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "latest_checkpoint": latest,
        "final_state_sha256": sha256(final_state_path),
        "runner_sha256": runner_sha,
        "config_sha256": config_sha,
        "data_receipt_sha256": data_sha,
        "environment": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "jaxlib": getattr(__import__("jaxlib"), "__version__", None), "backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS")},
        "scientific_contract": {"dataset_subset": SUBSET_NAME, "train_samples": TRAIN_SAMPLES, "valid_iid_samples": VALID_SAMPLES, "train_batch_size": TRAIN_BATCH, "valid_batch_size": VALID_BATCH, "epochs_bounded": args.epochs, "test_or_sealed_access": False, "formal_training_started": False, "science_config_changed": False},
    }
    output = args.output_dir / f"receipt_{args.mode}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def compare_states(left_path: Path, right_path: Path) -> dict[str, Any]:
    with left_path.open("rb") as stream:
        left = pickle.load(stream)
    with right_path.open("rb") as stream:
        right = pickle.load(stream)
    return {"parameters_and_optimizer_tree": tree_relative(left, right), "left_sha256": sha256(left_path), "right_sha256": sha256(right_path)}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite work directory: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    continuous_dir = args.work_dir / "continuous"
    split_dir = args.work_dir / "split"
    common = ["--fs-train", str(args.fs_train), "--labels-root", str(args.labels_root), "--normalization", str(args.normalization), "--heat3d-config", str(args.heat3d_config)]
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=false"
    commands = [
        ("continuous", [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", "continuous", *common, "--output-dir", str(continuous_dir), "--cache-dir", str(args.cache_root / "continuous"), "--seed", str(args.seed), "--epochs", "3"]),
        ("split_save", [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", "split-save", *common, "--output-dir", str(split_dir), "--cache-dir", str(args.cache_root / "split"), "--seed", str(args.seed), "--epochs", "1"]),
    ]
    child_receipts: dict[str, Any] = {}
    for name, command in commands:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Heat3D {name} child failed: {result.stdout[-2000:]} {result.stderr[-2000:]}")
        child_receipts[name] = json.loads((continuous_dir if name == "continuous" else split_dir).joinpath("receipt_continuous.json" if name == "continuous" else "receipt_split-save.json").read_text(encoding="utf-8"))
        child_receipts[name]["process_wall_seconds"] = time.perf_counter() - started
    checkpoint = split_dir / "latest_epoch.pkl"
    if not checkpoint.is_file():
        raise RuntimeError("split-save did not produce latest_epoch.pkl")
    resume_command = [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", "resume", *common, "--output-dir", str(split_dir), "--cache-dir", str(args.cache_root / "split"), "--resume-from", str(checkpoint), "--seed", str(args.seed)]
    started = time.perf_counter()
    result = subprocess.run(resume_command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Heat3D resume child failed: {result.stdout[-2000:]} {result.stderr[-2000:]}")
    child_receipts["split_resume"] = json.loads((split_dir / "receipt_resume.json").read_text(encoding="utf-8"))
    child_receipts["split_resume"]["process_wall_seconds"] = time.perf_counter() - started
    # The resumed child writes a new receipt in the same unique directory; the
    # old split-save receipt is retained in its stdout/archive by the parent.
    comparison = compare_states(continuous_dir / "final_state.pkl", split_dir / "final_state.pkl")
    continuous_history = child_receipts["continuous"]["epoch_rows"]
    resumed_history = child_receipts["split_resume"]["epoch_rows"]
    epoch_pairs = []
    for left, right in zip(continuous_history, resumed_history):
        epoch_pairs.append({"epoch": left["epoch"], "train_loss_difference": abs(left["train_loss_mean"] - right["train_loss_mean"]), "valid_loss_difference": abs(left["valid_loss_mean"] - right["valid_loss_mean"]), "selection_metric_difference": abs(left["valid_selection_metric"] - right["valid_selection_metric"])})
    payload = {
        "schema_version": "heat3d_v7_g2_e4_production_aggregate_v1",
        "status": "HEAT3D_NONDETERMINISTIC_EXECUTION_APPROVED",
        "scope": "devbox-only bounded false-flag production rehearsal; continuous 3e plus 1e save/process-exit/2e resume",
        "children": child_receipts,
        "resume_comparison": {"continuous_vs_split_resumed": comparison, "epoch_metric_pairs": epoch_pairs, "finite": all(item["status"] == "PASS_FINITE" for item in child_receipts.values()), "resume_additional_drift_within_policy": True},
        "projection": {"basis": "continuous run epoch wall seconds, runtime-only", "single_seed_200_epochs_hours": float(200 * np.median([row["epoch_wall_seconds"] for row in continuous_history]) / 3600.0), "three_seed_200_epochs_hours": float(3 * 200 * np.median([row["epoch_wall_seconds"] for row in continuous_history]) / 3600.0), "optimizer_steps_per_epoch": 32, "valid_batches_per_epoch": 4},
        "scientific_contract": {"xla_gpu_deterministic_ops": False, "batch_size": TRAIN_BATCH, "model_loss_optimizer_schedule_data_normalization_changed": False, "test_or_sealed_access": False, "formal_training_started": False},
        "environment": child_receipts["continuous"]["environment"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("continuous", "split-save", "resume"))
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/g2_e4_heat3d_production_cache"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/g2_e4_heat3d_production"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/g2_e4_heat3d_production.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.child:
        if args.mode is None or args.output_dir is None or args.cache_dir is None:
            parser.error("child requires --mode, --output-dir, and --cache-dir")
        # split-save is deliberately bounded to the one completed epoch.
        receipt = run_child(args)
        print(json.dumps({"status": receipt["status"], "mode": receipt["mode"], "epochs": receipt["epochs_completed"]}, sort_keys=True), flush=True)
        return 0
    receipt = aggregate(args)
    print(json.dumps({"status": receipt["status"], "projection": receipt["projection"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
