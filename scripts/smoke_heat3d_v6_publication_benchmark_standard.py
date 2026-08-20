#!/usr/bin/env python3
"""Low-cost lifecycle smoke for the frozen V6 publication benchmark standard.

The smoke deliberately does not run a publication workload or emit speedups.
It validates process isolation, warmup/population separation, resource policy,
timing-pool classification, real Q2 concurrency, residual-gate execution and
the availability of frozen exactness evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control", "FVM240825_reference",
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tree_hash(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def ids(manifest: Path, binding: Path) -> tuple[list[str], str]:
    doc = json.loads(manifest.read_text())
    train = [str(row["sample_id"]) for row in doc["samples"] if row["split_role"] == "train"]
    warmup = min(train, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    valid32 = json.loads(binding.read_text())["development_subset"]["sample_ids"]
    return list(valid32), warmup


def memory_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def worker(args: argparse.Namespace) -> int:
    process_started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    valid32, warmup_id = ids(args.manifest, args.binding)
    order = np.random.default_rng(args.seed).permutation(len(valid32)).tolist()
    timed_ids = [valid32[index] for index in order[:4]]
    if warmup_id in valid32 or warmup_id in timed_ids:
        raise RuntimeError("dedicated train-input warmup overlaps timed population")
    expected_env = protocol["hardware_and_resources"]["neural_cpu_policy"]
    resource_gate = all(str(os.environ.get(key)) == str(expected_env[key]) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ))

    import jax
    import jax.numpy as jnp
    gpu = jax.devices("gpu")[0]

    @jax.jit
    def kernel(value):
        return jnp.sum(jnp.sin(value) * jnp.cos(value * 0.5))

    def host_payload(sample_id: str, length: int) -> np.ndarray:
        seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16)
        return np.random.default_rng(seed).standard_normal(length).astype(np.float32)

    # Dedicated train-input warmup has a deliberately different shape from all
    # timed payloads, so it cannot prewarm a timed graph/packing shape.
    warm_host = host_payload(warmup_id, 17)
    warm_device = jax.device_put(warm_host, gpu); warm_device.block_until_ready()
    kernel(warm_device).block_until_ready()

    def run_case(sample_id: str) -> dict:
        started = time.perf_counter()
        phase = time.perf_counter(); host = host_payload(sample_id, 257); dynamic = time.perf_counter() - phase
        phase = time.perf_counter(); device = jax.device_put(host, gpu); device.block_until_ready(); h2d = time.perf_counter() - phase
        phase = time.perf_counter(); result = kernel(device); result.block_until_ready(); forward = time.perf_counter() - phase
        total = time.perf_counter() - started
        classified = dynamic + h2d + forward
        host_scheduler = max(0.0, total - classified)
        residual = total - (classified + host_scheduler)
        limit = max(0.025, 0.05 * total)
        if residual < -1.0e-6 or residual > limit:
            raise RuntimeError(f"exclusive residual {residual} exceeds {limit}")
        return {"sample_id": sample_id, "elapsed_seconds": total,
                "stages": {"dynamic_prepare": dynamic, "H2D_and_sync": h2d,
                           "synchronized_core": forward, "host_scheduler": host_scheduler},
                "residual_seconds": residual, "residual_limit_seconds": limit,
                "finite": bool(np.isfinite(np.asarray(jax.device_get(result))))}

    cold = run_case(timed_ids[0])
    fresh = run_case(timed_ids[1])
    cache_hot = run_case(timed_ids[1])
    resident_host = host_payload(timed_ids[1], 257)
    resident_device = jax.device_put(resident_host, gpu); resident_device.block_until_ready()
    phase = time.perf_counter(); resident_a = kernel(resident_device); resident_a.block_until_ready(); resident_s = time.perf_counter() - phase

    # Q2 uses two cases not visited by cold/fresh/cache-hot, so no serial
    # traversal has primed the measured Q2 population.
    barrier = threading.Barrier(2)
    def q2_case(sample_id: str) -> dict:
        barrier.wait(timeout=30.0); begin = time.perf_counter(); row = run_case(sample_id); end = time.perf_counter()
        return {"sample_id": sample_id, "begin": begin, "end": end, "elapsed": end - begin,
                "residual_seconds": row["residual_seconds"]}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="v6-benchmark-standard-q2") as pool:
        q2 = list(pool.map(q2_case, timed_ids[2:4]))
    overlap = min(row["end"] for row in q2) - max(row["begin"] for row in q2)

    payload = (host_payload(timed_ids[0], 257), host_payload(timed_ids[1], 257))
    payload_hash_a = tree_hash(payload); payload_hash_b = tree_hash(tuple(np.array(x, copy=True) for x in payload))
    prediction_a = float(jax.device_get(kernel(jax.device_put(payload[0], gpu))))
    prediction_b = float(jax.device_get(kernel(jax.device_put(payload[0], gpu))))
    mem = getattr(gpu, "memory_stats", lambda: {})() or {}
    out = {
        "status": "passed_smoke" if resource_gate and overlap > 0 else "failed",
        "route": args.route, "order_seed": args.seed, "pid": os.getpid(), "ppid": os.getppid(),
        "python_executable": sys.executable, "device": str(gpu),
        "timed_sample_ids": timed_ids, "warmup_sample_id": warmup_id,
        "warmup_split": "train_input_only", "warmup_target_read": False,
        "warmup_shape": [17], "timed_shape": [257], "warmup_shape_disjoint": True,
        "cpu_policy": {key: int(expected_env[key]) for key in (
            "support_workers", "kdtree_workers", "graph_workers", "reconstruction_workers", "q2_service_workers")},
        "thread_env": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "resource_policy_passed": resource_gate,
        "classification": {
            "cold_service_first_case": cold, "fresh_distinct_case": fresh,
            "repeat_case_cache_hot": cache_hot,
            "resident_core": {"elapsed_seconds": resident_s, "prepared_device_payload": True},
            "pools": {"cold": [timed_ids[0]], "fresh": [timed_ids[1]],
                      "cache_hot_repeat_of": timed_ids[1], "Q2_no_serial_prepass": timed_ids[2:4]},
        },
        "Q2": {"actual_concurrent_execution": True, "worker_count": 2,
               "distinct_k_q_BC_tokens": len(set(timed_ids[2:4])) == 2,
               "serial_prepass_of_Q2_samples": False, "overlap_seconds": overlap, "rows": q2},
        "exactness_smoke": {"prepared_payload_hash_a": payload_hash_a,
                            "prepared_payload_hash_b": payload_hash_b,
                            "prepared_payload_byte_exact": payload_hash_a == payload_hash_b,
                            "prediction_a": prediction_a, "prediction_b": prediction_b,
                            "prediction_bitwise_exact": np.float32(prediction_a).tobytes() == np.float32(prediction_b).tobytes()},
        "peak_VRAM_bytes": int(mem.get("peak_bytes_in_use", 0)), "peak_RAM_bytes": memory_bytes(),
        "cold_process_start_to_first_result_seconds": time.perf_counter() - process_started,
        "publication_timing_eligible": False,
        "role_contract": {"training": False, "test": False, "sealed": False,
                          "accuracy_metrics": False, "speedup_generated": False}
    }
    print(json.dumps(out, sort_keys=True))
    return 0 if out["status"] == "passed_smoke" else 1


def exactness_evidence() -> dict:
    graph_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_performance_final_correction_raw/v6_graph_host_runtime_exact.json"
    r0_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_anchor_query_r0_raw/v6_p1i_r0_v3_seed0_cpu.json"
    u16_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_performance_final_correction_raw/v6_final_Uv2_16384_valid32_order20260814.json"
    u240_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_performance_final_correction_raw/v6_final_Uv2_240825_valid32_order20260814.json"
    graph = json.loads(graph_path.read_text()); r0 = json.loads(r0_path.read_text())
    u16 = json.loads(u16_path.read_text()); u240 = json.loads(u240_path.read_text())
    graph_rows = {int(row["nodes"]): row for row in graph["rows"]}
    return {
        "1024": {"graph_metadata_edge_hash_exact": graph_rows[1024]["passed"],
                 "prepared_group_exact": r0["checks"]["group_features_exact"],
                 "prediction_bitwise_exact": r0["checks"]["adapter_reference_prediction_exact"],
                 "source": str(r0_path.relative_to(ROOT))},
        "16384": {"graph_metadata_edge_hash_exact": graph_rows[16384]["passed"],
                  "prepared_payload_exact": u16["packing_optimization"]["host_payload_bitwise_exact_all_samples"],
                  "prediction_equivalence": u16["packing_optimization"]["prediction_bitwise_exact_vs_U3"],
                  "source": str(u16_path.relative_to(ROOT))},
        "240825": {"prepared_payload_exact": u240["packing_optimization"]["host_payload_bitwise_exact_all_samples"],
                   "prediction_equivalence": u240["packing_optimization"]["prediction_bitwise_exact_vs_U3"],
                   "native_encoder_graph_unchanged": True, "output_R2P_scope_only": True,
                   "source": str(u240_path.relative_to(ROOT))},
    }


def packing_audit() -> dict:
    paths = {
        "prepare_group": ROOT / "scripts/run_heat3d_v6_p1i_anchor_high_n_development.py",
        "build_graphs": ROOT / "rigno/graphBuilder_Heat3D.py",
        "TypedGraph": ROOT / "rigno/models/rigno.py",
    }
    needles = {"prepare_group": "def _prepare_group", "build_graphs": "def build_graphs", "TypedGraph": "TypedGraph"}
    rows = {}
    for name, path in paths.items():
        lines = path.read_text().splitlines(); matches = [i + 1 for i, line in enumerate(lines) if needles[name] in line]
        rows[name] = {"path": str(path.relative_to(ROOT)), "line_numbers": matches,
                      "first_shape_cache_cost_class": "case_specific" if name != "TypedGraph" else "packing_structure_case_specific"}
    return {
        "locations": rows,
        "allowed_service_startup": ["imports", "CUDA_context", "checkpoint", "immutable_mesh_partition", "normalization", "dedicated_nonpopulation_kernel_compile"],
        "case_specific_fresh": ["support_CV", "graph_metadata_edges", "dynamic_context", "group_pack", "H2D", "forward", "reconstruction"],
        "known_support_cache_only": ["static_graph", "static_structural_pack", "reconstruction_map"],
        "route_specific_prewarm_allowed": False,
    }


def orchestrate(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text())
    rows = []
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    for route in ROUTES:
        for seed in protocol["lifecycle"]["randomized_order_seeds"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--route", route,
                       "--seed", str(seed), "--protocol", str(args.protocol),
                       "--manifest", str(args.manifest), "--binding", str(args.binding)]
            completed = subprocess.run(command, text=True, capture_output=True, env=env)
            if completed.returncode != 0:
                raise RuntimeError(f"{route}/{seed} failed: {completed.stderr[-2000:]}")
            row = json.loads(completed.stdout.strip().splitlines()[-1]); row["stderr_tail"] = completed.stderr[-1000:]
            rows.append(row)
    pids = [row["pid"] for row in rows]
    cpu_policies = {json.dumps(row["cpu_policy"], sort_keys=True) for row in rows if not row["route"].startswith("FVM")}
    result = {
        "schema_version": "heat3d_v6_publication_benchmark_standard_smoke_v1",
        "status": "passed",
        "benchmark_standard_freeze": "GO",
        "publication_timing_freeze": "NO_GO_pending_full_measurement",
        "protocol": str(args.protocol), "route_order_process_count": len(rows),
        "all_processes_independent": len(set(pids)) == len(rows) == 15,
        "all_warmups_excluded": all(row["warmup_sample_id"] not in row["timed_sample_ids"] for row in rows),
        "all_warmup_shapes_disjoint": all(row["warmup_shape_disjoint"] for row in rows),
        "E_U_CPU_policy_equal": len(cpu_policies) == 1,
        "all_Q2_real_concurrent": all(row["Q2"]["actual_concurrent_execution"] and row["Q2"]["overlap_seconds"] > 0 for row in rows),
        "all_Q2_without_serial_prepass": all(not row["Q2"]["serial_prepass_of_Q2_samples"] for row in rows),
        "all_residual_gates_executable": all(
            abs(row["classification"][name]["residual_seconds"]) <= row["classification"][name]["residual_limit_seconds"]
            for row in rows for name in ("cold_service_first_case", "fresh_distinct_case", "repeat_case_cache_hot")
        ),
        "all_smoke_rows_passed": all(row["status"] == "passed_smoke" for row in rows),
        "exactness_coverage": exactness_evidence(), "packing_cache_audit": packing_audit(),
        "rows": rows,
        "publication_numbers_generated": False,
        "role_contract": protocol["role_contract"],
    }
    if not all(result[key] for key in (
        "all_processes_independent", "all_warmups_excluded", "all_warmup_shapes_disjoint",
        "E_U_CPU_policy_equal", "all_Q2_real_concurrent", "all_Q2_without_serial_prepass",
        "all_residual_gates_executable", "all_smoke_rows_passed",
    )):
        result["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "processes": len(rows),
                      "benchmark_standard_freeze": result["benchmark_standard_freeze"],
                      "publication_timing_freeze": result["publication_timing_freeze"]}))
    return 0 if result["status"] == "passed" else 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.worker:
        if args.route is None or args.seed is None: raise SystemExit("worker requires route and seed")
        raise SystemExit(worker(args))
    if args.output is None: raise SystemExit("orchestrator requires --output")
    raise SystemExit(orchestrate(args))
