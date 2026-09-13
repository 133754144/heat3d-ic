#!/usr/bin/env python3
"""Bounded fresh-process Heat3D deterministic-XLA qualification.

This is an engineering/reproducibility probe, not a training run.  Each child
prepares the frozen DeepOHeat-v1 train/valid cache, performs one synchronized
compile/first update and five synchronized post-warm B24 updates, then exits.
The parent launches independent processes for deterministic=true/false and
bounded seeds.  No test/sealed artifact is accepted or opened.
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
POSTWARM_STEPS = 5
TRAIN_BATCH_SIZE = 24
DATASET_SUBSET = "frozen_768_train_128_valid_iid"


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
    """Materialize JAX leaves after the explicit synchronization boundary."""
    if isinstance(value, dict):
        return {key: host_tree(value[key]) for key in value}
    if isinstance(value, list):
        return [host_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(host_tree(item) for item in value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).copy()
    return copy.deepcopy(value)


def tree_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode())
            digest.update(repr(tuple(array.shape)).encode())
            digest.update(array.tobytes(order="C"))
            return
        digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def tree_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(tree_finite(value[key]) for key in value)
    if isinstance(value, (list, tuple)):
        return all(tree_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(np.isfinite(value))
    return True


def tree_relative(left: Any, right: Any) -> dict[str, float]:
    try:
        left_leaves, right_leaves = [], []

        def flatten(item: Any, out: list[Any]) -> None:
            if isinstance(item, dict):
                for key in sorted(item):
                    flatten(item[key], out)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    flatten(child, out)
            elif isinstance(item, np.ndarray):
                out.append(np.asarray(item, dtype=np.float64).reshape(-1))
            elif isinstance(item, (float, int, np.floating, np.integer)):
                out.append(np.asarray([item], dtype=np.float64))

        flatten(left, left_leaves)
        flatten(right, right_leaves)
        if len(left_leaves) != len(right_leaves):
            raise ValueError("pytree leaf count mismatch")
        if not left_leaves:
            return {"relative_l2": 0.0, "max_abs": 0.0, "rms_normalized": 0.0}
        if any(a.shape != b.shape for a, b in zip(left_leaves, right_leaves)):
            raise ValueError("pytree leaf shape mismatch")
        diff = np.concatenate([a - b for a, b in zip(left_leaves, right_leaves)])
        base = np.concatenate(right_leaves)
        denom = max(float(np.linalg.norm(base)), 1.0e-12)
        rms_denom = max(float(np.sqrt(np.mean(base * base))), 1.0e-12)
        return {
            "relative_l2": float(np.linalg.norm(diff) / denom),
            "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "rms_normalized": float(np.sqrt(np.mean(diff * diff)) / rms_denom) if diff.size else 0.0,
        }
    except (TypeError, ValueError):
        return {"relative_l2": float("inf"), "max_abs": float("inf"), "rms_normalized": float("inf")}


def array_relative(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return tree_relative(np.asarray(left), np.asarray(right))


def gpu_snapshot() -> dict[str, Any]:
    command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not command.is_file():
        command = Path("nvidia-smi")
    try:
        result = subprocess.run(
            [str(command), "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows = []
    for line in result.stdout.strip().splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 4:
            rows.append({"utilization_gpu_pct": values[0], "power_draw_w": values[1], "memory_used_mib": values[2], "memory_total_mib": values[3]})
    return {"available": bool(rows), "rows": rows, "command": str(command)}


def configure_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    from jax.experimental.compilation_cache import compilation_cache
    compilation_cache.set_cache_dir(str(cache_dir))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: Heat3D qualification requires devbox CUDA")
    for path in (args.fs_train, args.labels_root, args.normalization, args.heat3d_config):
        if any(token in str(path).lower() for token in ("test", "sealed")):
            raise ValueError(f"forbidden test/sealed path: {path}")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the frozen fs_train_volume.npy is accepted")
    configure_cache(args.cache_dir)
    profile = load_script("profile_v7_g2_heat3d_epoch.py")
    prepared_args = argparse.Namespace(
        fs_train=args.fs_train,
        labels_root=args.labels_root,
        normalization=args.normalization,
        heat3d_config=args.heat3d_config,
        seed=args.seed,
        train_batch_size=TRAIN_BATCH_SIZE,
        valid_batch_size=32,
        allow_nondeterministic_xla=True,
    )
    prepared = profile._prepare(prepared_args)
    return prepared


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.trace.exists():
        raise FileExistsError("refusing to overwrite child receipt/trace")
    prepared = prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    batch = prepared["train_batches"][0]
    block = prepared["block_until_ready"]
    base_key = jax.random.PRNGKey(args.seed)
    initial_params = host_tree(state.params)
    initial_optimizer = host_tree(state.optimizer_state)
    initial_state = {"params": initial_params, "optimizer": initial_optimizer}
    compile_started = time.perf_counter()
    first_key = jax.random.fold_in(base_key, 1)
    first = trainer.step(state, batch, first_key)
    block((first.state.params, first.state.optimizer_state, first.loss, first.gradients, first.updates, first.prediction))
    compile_first_seconds = time.perf_counter() - compile_started
    state = first.state
    postwarm: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for index in range(POSTWARM_STEPS):
        key = jax.random.fold_in(base_key, index + 2)
        started = time.perf_counter()
        result = trainer.step(state, batch, key)
        block((result.state.params, result.state.optimizer_state, result.loss, result.gradients, result.updates, result.prediction))
        wall = time.perf_counter() - started
        state = result.state
        params = host_tree(result.state.params)
        optimizer = host_tree(result.state.optimizer_state)
        gradients = host_tree(result.gradients)
        updates = host_tree(result.updates)
        prediction = host_tree(result.prediction)
        loss = float(np.asarray(result.loss))
        postwarm.append({
            "step": index + 2,
            "wall_seconds": wall,
            "loss": loss,
            "loss_finite": bool(np.isfinite(loss)),
            "grad_finite": tree_finite(gradients),
            "params_finite": tree_finite(params),
            "optimizer_finite": tree_finite(optimizer),
            "prediction_finite": tree_finite(prediction),
            "params_hash": tree_hash(params),
            "optimizer_hash": tree_hash(optimizer),
        })
        snapshots.append({"step": index + 2, "params": params, "optimizer": optimizer, "gradients": gradients, "updates": updates, "prediction": prediction, "loss": loss})
        if index == POSTWARM_STEPS - 1:
            final_prediction = prediction
        print(json.dumps({"phase": "postwarm", "step": index + 2, "seconds": wall}, sort_keys=True), flush=True)
    final_params = host_tree(state.params)
    final_optimizer = host_tree(state.optimizer_state)
    trace_payload = {
        "seed": int(args.seed),
        "initial": initial_state,
        "first": {"loss": float(np.asarray(first.loss)), "params": host_tree(first.state.params), "optimizer": host_tree(first.state.optimizer_state), "gradients": host_tree(first.gradients), "updates": host_tree(first.updates), "prediction": host_tree(first.prediction)},
        "postwarm": snapshots,
        "final": {"params": final_params, "optimizer": final_optimizer, "prediction": final_prediction},
    }
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("wb") as stream:
        pickle.dump(trace_payload, stream, protocol=5)
    memory = jax.devices()[0].memory_stats() or {}
    finite = all(item["loss_finite"] and item["grad_finite"] and item["params_finite"] and item["optimizer_finite"] and item["prediction_finite"] for item in postwarm)
    payload = {
        "schema_version": "heat3d_v7_g2_e3_heat3d_determinism_child_v1",
        "status": "PASS_FINITE" if finite else "FAIL_NONFINITE",
        "process_role": "fresh_child",
        "seed": int(args.seed),
        "deterministic_flag": "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""),
        "scientific_contract": {"dataset_subset": DATASET_SUBSET, "batch_size": TRAIN_BATCH_SIZE, "postwarm_steps": POSTWARM_STEPS, "model_loss_optimizer_normalization_changed": False, "accuracy_used_for_decision": False, "test_or_sealed_access": False, "formal_training_started": False},
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "execution": {"compile_plus_first_step_seconds": compile_first_seconds, "postwarm": postwarm, "median_postwarm_step_seconds": float(np.median([x["wall_seconds"] for x in postwarm])), "p95_postwarm_step_seconds": float(np.percentile([x["wall_seconds"] for x in postwarm], 95)), "samples_per_second": float(TRAIN_BATCH_SIZE / np.median([x["wall_seconds"] for x in postwarm])), "compile_count": int(trainer.compile_count), "explicit_full_result_block": True, "gpu_snapshot": gpu_snapshot()},
        "numerical": {"initial_params_tree_sha256": tree_hash(initial_params), "initial_optimizer_tree_sha256": tree_hash(initial_optimizer), "final_params_tree_sha256": tree_hash(final_params), "final_optimizer_tree_sha256": tree_hash(final_optimizer), "losses": [x["loss"] for x in postwarm], "finite": finite},
        "resource": {"device": str(jax.devices()[0]), "memory_stats": {str(k): int(v) for k, v in memory.items() if isinstance(v, (int, np.integer))}, "peak_memory_source": "jax.devices()[0].memory_stats", "xla_flags": os.environ.get("XLA_FLAGS")},
        "environment": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "jaxlib": getattr(__import__("jaxlib"), "__version__", None), "backend": jax.default_backend(), "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner_sha256": sha256(ROOT / "rigno/heat3d_training/core.py")},
        "hard_boundaries": {"test_or_sealed_access": False, "formal_training_started": False, "science_config_changed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def pairwise_trace(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for lrow, rrow in zip(left["postwarm"], right["postwarm"]):
        rows.append({"step": lrow["step"], "loss_relative_difference": float(abs(lrow["loss"] - rrow["loss"]) / max(abs(lrow["loss"]), 1.0e-12)), "gradients": tree_relative(lrow["gradients"], rrow["gradients"]), "updates": tree_relative(lrow["updates"], rrow["updates"]), "parameters": tree_relative(lrow["params"], rrow["params"]), "optimizer": tree_relative(lrow["optimizer"], rrow["optimizer"]), "prediction": tree_relative(lrow["prediction"], rrow["prediction"])})
    return {"steps": rows}


def median_path(items: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for item in items:
        for row in item["steps"]:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
    return float(np.median(values)) if values else float("nan")


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate receipt: {args.output}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, bool, int]] = []
    specs.extend(("deterministic_true_same_seed", True, 0) for _ in range(3))
    specs.extend(("deterministic_false_same_seed", False, 0) for _ in range(3))
    specs.extend(("deterministic_false_inter_seed", False, seed) for seed in (1, 2))
    children: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, (group, deterministic, seed) in enumerate(specs):
        child_output = args.work_dir / f"child_{index:02d}_{group}_seed{seed}.json"
        child_trace = args.work_dir / f"child_{index:02d}_{group}_seed{seed}.pkl"
        child_log = args.work_dir / f"child_{index:02d}.log"
        xla = "--xla_gpu_deterministic_ops=true" if deterministic else "--xla_gpu_deterministic_ops=false"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--fs-train", str(args.fs_train), "--labels-root", str(args.labels_root), "--normalization", str(args.normalization), "--heat3d-config", str(args.heat3d_config), "--output", str(child_output), "--trace", str(child_trace), "--seed", str(seed), "--cache-dir", str(args.cache_root / ("true" if deterministic else "false"))]
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["XLA_FLAGS"] = xla
        started = time.perf_counter()
        with child_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        wall = time.perf_counter() - started
        if result.returncode != 0 or not child_output.is_file() or not child_trace.is_file():
            failure = {"group": group, "deterministic": deterministic, "seed": seed, "returncode": result.returncode, "log": str(child_log), "wall_seconds": wall}
            payload = {"schema_version": "heat3d_v7_g2_e3_heat3d_determinism_aggregate_v1", "status": "HEAT3D_NUMERICAL_STABILITY_FAIL_CLOSED", "failure": failure, "children": children, "test_or_sealed_access": False, "formal_training_started": False}
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return payload
        child = json.loads(child_output.read_text(encoding="utf-8")); child["group"] = group; child["deterministic"] = deterministic; child["process_wall_seconds"] = wall; child["raw_sha256"] = sha256(child_output); children.append(child)
        with child_trace.open("rb") as stream:
            traces.append(pickle.load(stream))
        print(json.dumps({"child": index, "group": group, "seed": seed, "wall_seconds": wall}, sort_keys=True), flush=True)
    true_traces = traces[0:3]; false_traces = traces[3:6]; inter_traces = traces[6:8]
    true_same = [pairwise_trace(true_traces[0], item) for item in true_traces[1:]]
    false_same = [pairwise_trace(false_traces[0], item) for item in false_traces[1:]]
    false_inter = [pairwise_trace(false_traces[0], item) for item in inter_traces]
    paths = {"loss": ("loss_relative_difference",), "gradients": ("gradients", "relative_l2"), "updates": ("updates", "relative_l2"), "parameters": ("parameters", "relative_l2"), "optimizer": ("optimizer", "relative_l2"), "prediction": ("prediction", "relative_l2")}
    same_rel = {key: median_path(false_same, path) for key, path in paths.items()}
    inter_rel = {key: median_path(false_inter, path) for key, path in paths.items()}
    ratios = {key: float(same_rel[key] / max(inter_rel[key], 1.0e-12)) for key in same_rel}
    true_rel = {key: median_path(true_same, path) for key, path in paths.items()}
    cross_true_false = pairwise_trace(true_traces[0], false_traces[0])
    false_times = [float(x["execution"]["median_postwarm_step_seconds"]) for x in children if not x["deterministic"]]
    true_times = [float(x["execution"]["median_postwarm_step_seconds"]) for x in children if x["deterministic"]]
    stable = all(child["status"] == "PASS_FINITE" for child in children) and all(np.isfinite(list(same_rel.values()))) and all(ratios[key] < 0.1 for key in ratios)
    performance = bool(false_times and true_times and np.median(false_times) < np.median(true_times))
    status = "HEAT3D_NONDETERMINISTIC_GPU_AMENDMENT_READY_FOR_APPROVAL" if stable and performance else "HEAT3D_NUMERICAL_STABILITY_FAIL_CLOSED"
    payload = {
        "schema_version": "heat3d_v7_g2_e3_heat3d_determinism_aggregate_v1",
        "status": status,
        "scope": "eight fresh processes: true x3 seed0, false x3 seed0, false seeds1/2; one frozen B24 batch; one compile/first plus five synchronized postwarm updates",
        "children": [{"group": child["group"], "deterministic": child["deterministic"], "seed": child["seed"], "status": child["status"], "raw_sha256": child["raw_sha256"], "process_wall_seconds": child["process_wall_seconds"], "median_postwarm_step_seconds": child["execution"]["median_postwarm_step_seconds"], "p95_postwarm_step_seconds": child["execution"]["p95_postwarm_step_seconds"], "samples_per_second": child["execution"]["samples_per_second"]} for child in children],
        "runtime": {"deterministic_true_median_seconds": float(np.median(true_times)), "deterministic_true_p95_of_child_medians_seconds": float(np.percentile(true_times, 95)), "deterministic_false_median_seconds": float(np.median(false_times)), "deterministic_false_p95_of_child_medians_seconds": float(np.percentile(false_times, 95)), "false_over_true_speedup": float(np.median(true_times) / max(np.median(false_times), 1.0e-12)), "true_children_compile_plus_first_seconds": [child["execution"]["compile_plus_first_step_seconds"] for child in children if child["deterministic"]], "false_children_compile_plus_first_seconds": [child["execution"]["compile_plus_first_step_seconds"] for child in children if not child["deterministic"]]},
        "numerical_drift": {"false_same_seed_median_relative": same_rel, "false_inter_seed_median_relative": inter_rel, "false_same_over_inter_ratio": ratios, "true_same_seed_median_relative": true_rel, "true_vs_false_same_initial_state": cross_true_false, "finite_and_no_systematic_drift_rule": "all bounded probe leaves finite and same-seed noise <10% of inter-seed variability; no accuracy read"},
        "decision": {"same_seed_target_ratio_lt_0.1": True, "same_seed_noise_target_met": stable, "stable_performance_advantage": performance, "formal_config_changed": False, "formal_config_amended": False, "candidate_only": status.endswith("READY_FOR_APPROVAL"), "formal_training_started": False, "test_or_sealed_access": False},
        "environment": children[0]["environment"],
        "hard_boundaries": {"p1i_test_iid_accessed": False, "p1i_sealed_accessed": False, "deepoheat_official_test_accessed": False, "formal_training_started": False, "science_config_changed": False},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/g2_e3_heat3d_determinism_cache"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/g2_e3_heat3d_determinism"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.child:
        if args.trace is None:
            parser.error("--child requires --trace")
        payload = run_child(args)
        print(json.dumps({"status": payload["status"], "deterministic": payload["deterministic_flag"], "median_step": payload["execution"]["median_postwarm_step_seconds"]}, sort_keys=True), flush=True)
        return 0 if payload["status"] == "PASS_FINITE" else 2
    payload = aggregate(args)
    print(json.dumps({"status": payload["status"], "children": len(payload.get("children", []))}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
