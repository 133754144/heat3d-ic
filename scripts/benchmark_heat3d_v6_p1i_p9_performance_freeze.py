#!/usr/bin/env python3
"""P9 publication-safe persistent neural/FVM performance freeze."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import gc
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median_seconds": float(np.median(array)), "mean_seconds": float(np.mean(array)), "std_seconds": float(np.std(array)), "p95_seconds": float(np.quantile(array, 0.95))}


def tree_sha(tree: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest.update(str(treedef).encode())
    for array_like in leaves:
        array = np.ascontiguousarray(np.asarray(array_like))
        digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def complete_hash(row: dict[str, Any]) -> dict[str, str]:
    scopes = {
        "selected_indices": row["selected"], "selected_cv": row["selected_cv"],
        "reconstruction_indices": row["indices"], "reconstruction_weights": row["map_weights"],
        "anchor_group": row["anchor"], "query_group": row["query"],
        "anchor_inputs": row["anchor"]["inputs"], "query_inputs": row["query"]["inputs"],
        "anchor_graph": row["anchor"]["graphs"], "query_graph": row["query"]["graphs"],
    }
    for name in ("native_physics", "global_context", "scale_context", "qk_region_features"):
        if name in row["anchor"]: scopes[f"anchor_{name}"] = row["anchor"][name]
        if name in row["query"]: scopes[f"query_{name}"] = row["query"][name]
    return {key: tree_sha(value) for key, value in scopes.items()}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("neural", "fvm"), required=True)
    for name in ("protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields", "run_dir", "native_padding_result", "query_padding_result", "output"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True); p.add_argument("--checkpoint-epoch", type=int, default=559)
    return p.parse_args()


def serialized_state(state: dict[str, Any]) -> dict[str, str]:
    args = state["args"]
    result = {key: str(getattr(args, key)) for key in ("protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields", "run_dir", "native_padding_result", "query_padding_result", "checkpoint_sha256", "checkpoint_epoch")}
    result["sample_count"] = "32"; result["_anchor_targets_json"] = json.dumps(state["anchor_targets"]); result["_query_targets_json"] = json.dumps(state["query_targets"])
    return result


def prepare_runner(state: dict[str, Any], backend: str) -> tuple[Callable[[list[int]], list[dict[str, Any]]], Callable[[], None], float]:
    if backend == "serial":
        p8.prepare_case(state, 0)
        return lambda indices: [p8.prepare_case(state, index) for index in indices], lambda: None, 0.0
    workers = int("".join(filter(str.isdigit, backend)))
    if backend.startswith("thread"):
        pool = ThreadPoolExecutor(max_workers=workers); list(pool.map(lambda _: p8.prepare_case(state, 0), range(workers)))
        return lambda indices: list(pool.map(lambda index: p8.prepare_case(state, index), indices)), lambda: pool.shutdown(), 0.0
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    started = time.perf_counter(); pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=p8.init_prepare_worker, initargs=(serialized_state(state),))
    ready = {future.result() for future in [pool.submit(p8.worker_ready) for _ in range(workers * 4)]}
    if len(ready) != workers: raise RuntimeError(f"{backend}: worker readiness failed")
    list(pool.map(p8.prepare_worker, [0] * workers))
    return lambda indices: list(pool.map(p8.prepare_worker, indices)), lambda: pool.shutdown(), time.perf_counter() - started


def neural(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text()); state = p8.runtime_state(args); state["args"] = args; p8.ensure_edge_envelope(state)
    orders = [np.random.default_rng(seed).permutation(32).tolist() for seed in protocol["randomized_order_seeds"]]
    reference_hashes = None; backend_rows = []
    for backend in protocol["neural"]["persistent_preprocessing_backends"]:
        run, close, startup = prepare_runner(state, backend); started = time.perf_counter(); rows = run(orders[0]); elapsed = time.perf_counter() - started; close()
        hashes = {row["sample_id"]: complete_hash(row) for row in rows}
        reference_hashes = hashes if reference_hashes is None else reference_hashes
        exact = hashes == reference_hashes
        backend_rows.append({"backend": backend, "startup_and_untimed_warmup_seconds": startup, "steady_wall_seconds": elapsed, "samples_per_second": 32 / elapsed, "full_payload_exact_vs_serial": exact, "hashes": hashes})
        if not exact: raise RuntimeError(f"{backend}: complete prepared payload exact gate failed")
        del rows
        jax.clear_caches()
        gc.collect()
    winner = max(backend_rows, key=lambda row: row["samples_per_second"])["backend"]

    runtime = state["runtime"]; model = GraphNeuralOperator(**runtime["model_config"]); params = highn.runner._device_params(runtime["checkpoint"]["params"]); gpu = jax.devices("gpu")[0]
    @jax.jit
    def forward(params, anchor, query, weights, indices, map_weights):
        anchor_result = highn.runner._model_apply(model, params, anchor); raw = anchor_result["raw_temperature"][:, 0, :, 0]; scale = anchor_result["s_hat"].reshape((raw.shape[0], -1))[:, 0]
        query_value = highn.runner._model_apply(model, params, query)["raw_temperature"][:, 0, :, 0] - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights, axis=1, keepdims=True); query_scale = jnp.sqrt(jnp.sum(normalized * query_value * query_value, axis=1)); support = query_value / query_scale[:, None] * scale[:, None]
        gathered = support[jnp.arange(support.shape[0])[:, None, None], indices]; return jnp.sum(gathered * map_weights.astype(support.dtype), axis=2)
    def host_batch(rows): return (p8.stack([r["anchor"] for r in rows]), p8.stack([r["query"] for r in rows]), np.concatenate([r["weights"] for r in rows]), np.concatenate([r["indices"] for r in rows]), np.concatenate([r["map_weights"] for r in rows]))
    run, close, winner_startup = prepare_runner(state, winner)
    warm_rows = run([0] * 32)
    resident = {}
    for size in (1, 16, 32):
        host = host_batch(warm_rows[:size]); device = jax.device_put(host, gpu); p8.block(device); prediction = forward(params, *device); p8.block(prediction); values = []
        for _ in range(protocol["neural"]["resident_repeats"]):
            started = time.perf_counter(); prediction = forward(params, *device); p8.block(prediction); values.append(time.perf_counter() - started)
        resident[str(size)] = stats(values)
        del host, device, prediction
        gc.collect()
    del warm_rows
    gc.collect()
    repeat_rows = []; b1_values = []
    for repeat_index, order in enumerate(orders):
        b1_elapsed = []
        for index in order:
            started = time.perf_counter(); row = run([index])[0]; host = host_batch([row]); device = jax.device_put(host, gpu); p8.block(device); prediction = forward(params, *device); p8.block(prediction); b1_elapsed.append(time.perf_counter() - started)
        b1_values.extend(b1_elapsed)
        batch16 = []
        for start in (0, 16):
            tick = time.perf_counter(); rows = run(order[start:start + 16]); host = host_batch(rows); device = jax.device_put(host, gpu); p8.block(device); prediction = forward(params, *device); p8.block(prediction); batch16.append(time.perf_counter() - tick)
        tick = time.perf_counter(); rows = run(order); host = host_batch(rows); device = jax.device_put(host, gpu); p8.block(device); prediction = forward(params, *device); p8.block(prediction); batch32 = time.perf_counter() - tick
        repeat_rows.append({"repeat": repeat_index, "order_seed": protocol["randomized_order_seeds"][repeat_index], "order": order, "fresh_b1": stats(b1_elapsed), "two_B16_wall_seconds": float(sum(batch16)), "B16_individual_seconds": batch16, "B32_wall_seconds": batch32, "two_B16_samples_per_second": 32 / sum(batch16), "B32_samples_per_second": 32 / batch32, "marginal_added_case_seconds": (batch32 - float(np.median(batch16))) / 16.0})
    close()
    result = {"schema_version": "heat3d_v6_p1i_p9_neural_v1", "status": "passed", "winner": winner, "backend_comparison": backend_rows, "complete_payload_hash_exact": True, "winner_startup_and_warmup_seconds": winner_startup, "fresh_b1": stats(b1_values), "resident_inference": resident, "repeat_rows": repeat_rows, "full_valid32_two_B16": stats([row["two_B16_wall_seconds"] for row in repeat_rows]), "full_valid32_B32": stats([row["B32_wall_seconds"] for row in repeat_rows]), "marginal_added_case": stats([row["marginal_added_case_seconds"] for row in repeat_rows]), "peak_vram_bytes": int(candidate.publication._device_memory().get("peak_bytes_in_use", 0)), "protocol_sha256": p8.sha256(args.protocol), "role_contract": protocol["role_contract"]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"status": "passed", "winner": winner, "B32_samples_s": 32 / result["full_valid32_B32"]["median_seconds"]})); return 0


def fvm(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text()); orders = [np.random.default_rng(seed).permutation(32).tolist() for seed in protocol["randomized_order_seeds"]]; serialized = {key: str(getattr(args, key)) for key in ("dataset_root", "manifest", "full_fields")}; output = []
    for count in protocol["fvm"]["persistent_process_counts"]:
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        started = time.perf_counter(); pool = ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn"), initializer=p8.init_fvm_worker, initargs=(serialized,)); ready = {future.result() for future in [pool.submit(p8.worker_ready) for _ in range(count * 4)]}
        if len(ready) != count: raise RuntimeError("FVM worker readiness failed")
        list(pool.map(p8.fvm_worker, [0] * count)); startup = time.perf_counter() - started; repeats = []
        for repeat_index, order in enumerate(orders):
            tick = time.perf_counter(); measurements = list(pool.map(p8.fvm_worker, order)); wall = time.perf_counter() - tick
            repeats.append({"repeat": repeat_index, "order_seed": protocol["randomized_order_seeds"][repeat_index], "order": order, "steady_wall_seconds": wall, "samples_per_second": 32 / wall, "measurements": measurements})
        pool.shutdown(); output.append({"process_count": count, "startup_and_untimed_warmup_seconds": startup, "repeats": repeats, "steady_wall": stats([row["steady_wall_seconds"] for row in repeats]), "throughput": stats([row["samples_per_second"] for row in repeats])})
    saturation = max(output, key=lambda row: row["throughput"]["median_seconds"])
    result = {"schema_version": "heat3d_v6_p1i_p9_fvm_v1", "status": "passed", "rows": output, "saturation_process_count": saturation["process_count"], "saturation_samples_per_second": saturation["throughput"]["median_seconds"], "protocol_sha256": p8.sha256(args.protocol), "role_contract": {**protocol["role_contract"], "accessed_roles": ["valid_iid"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"status": "passed", "saturation": saturation["process_count"], "samples_s": result["saturation_samples_per_second"]})); return 0


def main() -> int:
    args = parse(); return neural(args) if args.mode == "neural" else fvm(args)


if __name__ == "__main__": raise SystemExit(main())
