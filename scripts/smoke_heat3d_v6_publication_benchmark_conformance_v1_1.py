#!/usr/bin/env python3
"""Real-route, low-cost conformance smoke for benchmark protocol v1.1.

This runner deliberately emits no publication latency or speedup table.  Each
route/order/service-mode cell runs in its own Python process.  The FVM service
uses in-memory physics payloads and a persistent P1/P2 worker pool; neural
routes delegate to their frozen real production implementations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
    "FVM240825_reference",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_FVM_STATE: dict[str, Any] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _hwm_bytes(pid: int) -> int:
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return 0
    for line in path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return 0


def _init_fvm(serialized: dict[str, str]) -> None:
    os.environ.update(THREAD_ENV)
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    import benchmark_heat3d_v6_inference_qualification as qualification

    data = qualification.FamilyData(
        family="p1i",
        dataset_root=Path(serialized["dataset_root"]),
        manifest_path=Path(serialized["manifest"]),
        full_fields_path=Path(serialized["full_fields"]),
        randomblock_config=None,
    )
    rows = data.selected_rows(32)
    mesh = qualification.prior.core.build_mesh(data.physics(rows[0]))
    if not np.array_equal(mesh["coords"], data.full_shared()["coords"]):
        raise RuntimeError("FVM mesh/full-field coordinate drift")
    global _FVM_STATE
    _FVM_STATE = {"qualification": qualification, "mesh": mesh}


def _fvm_ready() -> int:
    return os.getpid()


def _fvm_case(payload: dict[str, Any]) -> dict[str, Any]:
    q = _FVM_STATE["qualification"]
    started = time.perf_counter()
    phase = time.perf_counter()
    k = np.asarray(payload["k_xyz"], dtype=np.float64)
    source = np.asarray(payload["q_W_m3"], dtype=np.float64)
    decode = time.perf_counter() - phase
    phase = time.perf_counter()
    system = q.prior._assemble(
        _FVM_STATE["mesh"], k, source,
        float(payload["top_h"]), float(payload["bottom_h"]),
    )
    assembly = time.perf_counter() - phase
    phase = time.perf_counter()
    temperature = q.prior._solve(*system)
    solve = time.perf_counter() - phase
    total = time.perf_counter() - started
    residual = total - (decode + assembly + solve)
    limit = max(0.025, 0.05 * total)
    if residual < -1.0e-6 or residual > limit:
        raise RuntimeError(f"FVM residual {residual} exceeds {limit}")
    if not np.all(np.isfinite(temperature)):
        raise RuntimeError("nonfinite FVM output")
    return {
        "sample_id": payload["sample_id"], "worker_pid": os.getpid(),
        "service_seconds": total,
        "stages": {"payload_decode": decode, "assembly": assembly, "solve": solve},
        "residual_seconds": residual, "residual_limit_seconds": limit,
    }


def fvm_worker(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text())
    binding = json.loads(args.binding.read_text())
    import benchmark_heat3d_v6_inference_qualification as qualification

    data = qualification.FamilyData(
        family="p1i", dataset_root=args.dataset_root,
        manifest_path=args.manifest, full_fields_path=args.full_fields,
        randomblock_config=None,
    )
    rows = data.selected_rows(32)
    valid_ids = [str(row["sample_id"]) for row in rows]
    if valid_ids != binding["development_subset"]["sample_ids"]:
        raise RuntimeError("FVM valid32 order differs from frozen binding")
    order = np.random.default_rng(args.order_seed).permutation(32)[:args.sample_count].tolist()
    ordered_rows = [rows[index] for index in order]
    payloads = []
    for row in ordered_rows:
        example, _ = data.load_example(row)
        k, source = data.full_kq(row)
        payloads.append({
            "sample_id": str(row["sample_id"]),
            "k_xyz": np.asarray(k, dtype=np.float64),
            "q_W_m3": np.asarray(source, dtype=np.float64),
            "top_h": float(example.condition.condition_features[0, 8]),
            "bottom_h": float(example.condition.condition_features[0, 9]),
        })
    process_count = 1 if args.service_mode == "serial" else 2
    serialized = {
        "dataset_root": str(args.dataset_root), "manifest": str(args.manifest),
        "full_fields": str(args.full_fields),
    }
    pool = ProcessPoolExecutor(
        max_workers=process_count, mp_context=mp.get_context("spawn"),
        initializer=_init_fvm, initargs=(serialized,),
    )
    worker_pids = sorted({future.result() for future in [
        pool.submit(_fvm_ready) for _ in range(process_count * 3)]})
    if len(worker_pids) != process_count:
        raise RuntimeError("FVM persistent worker count drift")
    rows_out: list[dict[str, Any]] = []
    q2 = None
    if args.service_mode == "serial":
        for payload in payloads:
            submitted = time.perf_counter()
            row = pool.submit(_fvm_case, payload).result()
            completed = time.perf_counter()
            row["submit_to_result_seconds"] = completed - submitted
            rows_out.append(row)
    else:
        began = time.perf_counter()
        pending: dict[Any, tuple[int, float]] = {}
        cursor = 0
        completions = []
        while cursor < min(2, len(payloads)):
            submitted = time.perf_counter()
            pending[pool.submit(_fvm_case, payloads[cursor])] = (cursor, submitted)
            cursor += 1
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index, submitted = pending.pop(future)
                row = future.result(); completed = time.perf_counter()
                row.update(
                    input_index=index,
                    submit_offset_seconds=submitted - began,
                    completion_offset_seconds=completed - began,
                    submit_to_result_seconds=completed - submitted,
                )
                rows_out.append(row); completions.append(completed - began)
                if cursor < len(payloads):
                    new_submitted = time.perf_counter()
                    pending[pool.submit(_fvm_case, payloads[cursor])] = (cursor, new_submitted)
                    cursor += 1
        rows_out.sort(key=lambda row: row["completion_offset_seconds"])
        ordered_completion = [row["completion_offset_seconds"] for row in rows_out]
        inter = np.diff(np.asarray([0.0] + ordered_completion)).tolist()
        q2 = {
            "actual_concurrent_execution": True, "queue_depth": 2,
            "worker_count": 2, "serial_prepass_of_Q2_samples": False,
            "submit_to_result": stats([row["submit_to_result_seconds"] for row in rows_out]),
            "inter_completion": stats(inter),
            "samples_per_second": len(rows_out) / ordered_completion[-1],
        }
    aggregate_peak = _hwm_bytes(os.getpid()) + sum(_hwm_bytes(pid) for pid in worker_pids)
    pool.shutdown()
    result = {
        "schema_version": "heat3d_v6_publication_fvm_service_smoke_v1_1",
        "status": "passed_smoke", "route": args.route,
        "service_mode": args.service_mode, "process_id": os.getpid(),
        "worker_pids": worker_pids, "worker_count": process_count,
        "order_seed": args.order_seed,
        "ordered_sample_ids": [payload["sample_id"] for payload in payloads],
        "rows": rows_out, "Q2": q2,
        "warmup": {"kind": "not_applicable_no_JIT", "timed_case_prepass": False},
        "classification": {
            "cold_service_first_case": "first_row",
            "fresh_distinct_case": "all_rows",
            "repeat_case_cache_hot": "not_measured_in_conformance_smoke",
            "resident_core": "not_measured_in_conformance_smoke",
        },
        "aggregate_service_worker_peak_RAM_bytes": aggregate_peak,
        "thread_env": THREAD_ENV,
        "publication_timing_eligible": False,
        "role_contract": protocol["role_contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "pid": os.getpid()}))
    return 0


def route_command(args: argparse.Namespace, route: str, seed: int, mode: str, output: Path) -> list[str]:
    common = [
        "--protocol", str(args.protocol), "--binding", str(args.binding),
        "--artifact-root", str(args.artifact_root), "--dataset-root", str(args.dataset_root),
        "--manifest", str(args.manifest), "--full-fields", str(args.full_fields),
        "--run-dir", str(args.run_dir), "--native-padding-result", str(args.native_padding_result),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--sample-count", str(args.sample_count), "--output", str(output),
    ]
    if route.startswith("E"):
        padding = args.e16384_padding_result if route.startswith("E16384") else args.e240825_padding_result
        return [
            sys.executable, str(ROOT / "scripts/benchmark_heat3d_v6_p1i_final_e_service.py"),
            *common, "--query-padding-result", str(padding), "--route", route,
            "--order-seed", str(seed), "--service-mode", mode,
            "--resident-repeats", "2", "--standard-v1-1-smoke",
        ]
    if route.startswith("U"):
        resolution = 16384 if "16384" in route else 240825
        qualification_path = args.u16384_qualification if resolution == 16384 else args.u240825_qualification
        command = [
            sys.executable, str(ROOT / "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py"),
            *common, "--query-padding-result", str(qualification_path),
            "--resolution", str(resolution), "--asymmetric-mode", "u_v2",
            "--population-mode", "frozen_valid32", "--timing-only",
            "--qualification-result", str(qualification_path), "--timing-regression-audit",
            "--order-seed", str(seed), "--repeats", "2", "--batch-sizes", "1",
            "--standard-v1-1-smoke", "--true-concurrent-depth", "1" if mode == "serial" else "2",
        ]
        if mode == "Q2":
            command.append("--concurrent-only")
        return command
    return [
        sys.executable, str(Path(__file__).resolve()), "--fvm-worker",
        "--protocol", str(args.protocol), "--binding", str(args.binding),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--route", route,
        "--order-seed", str(seed), "--service-mode", mode,
        "--sample-count", str(args.sample_count), "--output", str(output),
    ]


def ordered_ids(row: dict[str, Any], seed: int) -> list[str]:
    value = row["ordered_sample_ids"]
    if isinstance(value, dict):
        return list(value[str(seed)])
    return list(value)


def exactness_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r0_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_anchor_query_r0_raw/v6_p1i_r0_v3_seed0_cpu.json"
    r0 = json.loads(r0_path.read_text())
    graph_sample = r0["graph_equivalence"]["samples"][0]
    r0_direct = {
        "metadata_candidate_sha256": graph_sample["metadata"]["adapter_sha256"],
        "metadata_reference_sha256": graph_sample["metadata"]["reference_sha256"],
        "edge_candidate_sha256": graph_sample["adapter_real_edge_sha256"],
        "edge_reference_sha256": graph_sample["reference_real_edge_sha256"],
    }
    if (
        r0_direct["metadata_candidate_sha256"] != r0_direct["metadata_reference_sha256"]
        or r0_direct["edge_candidate_sha256"] != r0_direct["edge_reference_sha256"]
    ):
        raise RuntimeError("1024 candidate/reference metadata or edge hashes differ")
    resolution_rows: dict[str, list[dict[str, Any]]] = {"16384": [], "240825": []}
    for row in rows:
        if row["route"].startswith("E"):
            evidence = row.get("exactness_provenance")
            if evidence is None:
                raise RuntimeError(f"{row['route']}: absent direct exactness provenance")
            if evidence["graph_candidate_hashes"] != evidence["graph_reference_hashes"]:
                raise RuntimeError(f"{row['route']}: graph candidate/reference hash mismatch")
            if evidence["candidate_prepared_payload_sha256"] != evidence["reference_prepared_payload_sha256"]:
                raise RuntimeError(f"{row['route']}: payload candidate/reference hash mismatch")
            resolution_rows[str(row["resolution"])].append(evidence)
        elif row["route"].startswith("U"):
            if row["concurrent_only"]:
                continue
            audits = [sample["packing_audit"] for sample in row["samples"]
                      if sample["packing_audit"]["graph_exactness_audit_executed"]]
            if len(audits) != 1:
                raise RuntimeError(f"{row['route']}: direct graph exactness audit count drift")
            audit = audits[0]
            if audit["graph_candidate_hashes"] != audit["graph_reference_hashes"]:
                raise RuntimeError(f"{row['route']}: graph replay hash mismatch")
            if audit["candidate_payload_sha256"] != audit["reference_payload_sha256"]:
                raise RuntimeError(f"{row['route']}: prepared payload hash mismatch")
            resolution_rows[str(row["resolution"])].append(audit)
    return {
        "status": "passed", "1024": r0_direct,
        "16384_direct_audit_count": len(resolution_rows["16384"]),
        "240825_direct_audit_count": len(resolution_rows["240825"]),
        "prewritten_boolean_used_without_hash_comparison": False,
        "r0_source": str(r0_path.relative_to(ROOT)),
    }


def orchestrate(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_before_real_route_conformance_smoke":
        raise RuntimeError("protocol v1.1 is not frozen before conformance smoke")
    if args.sample_count < 2 or args.sample_count > 4:
        raise RuntimeError("conformance smoke is limited to 2-4 valid32 inputs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    process_records = []
    failure = None
    env = os.environ.copy(); env.update(THREAD_ENV)
    for seed in protocol["randomized_order_seeds"]:
        for mode in protocol["lifecycle"]["service_modes"]:
            for route in ROUTES:
                output = args.output_root / f"{route}_seed{seed}_{mode}.json"
                log = args.output_root / f"{route}_seed{seed}_{mode}.log"
                command = route_command(args, route, seed, mode, output)
                started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True, env=env)
                log.write_text(completed.stdout + completed.stderr)
                record = {
                    "route": route, "seed": seed, "service_mode": mode,
                    "returncode": completed.returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "command": command, "output": str(output), "log": str(log),
                }
                process_records.append(record)
                if completed.returncode != 0 or not output.exists():
                    failure = {**record, "stderr_tail": completed.stderr[-4000:]}
                    break
                row = json.loads(output.read_text())
                row["route"] = route
                row["service_mode"] = mode
                row["order_seed"] = seed
                row["artifact_path"] = str(output)
                row["artifact_sha256"] = file_sha256(output)
                rows.append(row)
            if failure:
                break
        if failure:
            break
    result: dict[str, Any] = {
        "schema_version": "heat3d_v6_publication_benchmark_conformance_v1_1",
        "status": "failed_fail_closed" if failure else "checking",
        "protocol_sha256": file_sha256(args.protocol),
        "sample_count": args.sample_count, "process_records": process_records,
        "rows": rows, "failure": failure,
        "publication_numbers_generated": False,
        "role_contract": protocol["role_contract"],
    }
    if failure is None:
        pids = [row["process_id"] for row in rows]
        if len(rows) != 30 or len(set(pids)) != 30:
            raise RuntimeError("30 independent route/seed/mode process lifecycle failed")
        for seed in protocol["randomized_order_seeds"]:
            expected = None
            for row in rows:
                if (row.get("order_seed") == seed or seed in row.get("order_seeds", [])):
                    ids = ordered_ids(row, seed)
                    expected = ids if expected is None else expected
                    if ids != expected:
                        raise RuntimeError(f"seed {seed}: cross-route ordered sample IDs differ")
        neural_rows = [row for row in rows if not row["route"].startswith("FVM")]
        if any(row["warmup"]["source_split"] != "train" or row["warmup"]["target_read"]
               for row in neural_rows):
            raise RuntimeError("neural warmup is not train-input/target-free")
        if any(row["warmup"]["timed_graph_or_packing_prebuilt"] for row in neural_rows):
            raise RuntimeError("timed graph/packing was prewarmed")
        if any(row.get("service_mode") == "Q2" and row["route"].startswith("E")
               and row["serial_orders"] for row in rows):
            raise RuntimeError("E Q2 process traversed serial population")
        if any(row.get("service_mode") == "Q2" and row["route"].startswith("U")
               and not row["concurrent_only"] for row in rows):
            raise RuntimeError("U Q2 process traversed serial population")
        exactness = exactness_gate(rows)
        result.update({
            "status": "passed",
            "benchmark_protocol_v1_1": "GO",
            "benchmark_implementation_freeze": "GO",
            "publication_timing_freeze": "NO_GO_pending_full_valid32",
            "independent_process_count": 30,
            "independent_process_lifecycle": True,
            "same_seed_cross_route_order_exact": True,
            "E_U_CPU_resource_policy_equal": True,
            "Q2_without_serial_prepass": True,
            "cold_fresh_cache_hot_classification_separate": True,
            "real_route_smoke_only": True,
            "exactness_provenance": exactness,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(rows), "failure": failure}))
    return 0 if result["status"] == "passed" else 1


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fvm-worker", action="store_true")
    for name in ("protocol", "binding", "dataset_root", "manifest", "full_fields", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--native-padding-result", type=Path)
    parser.add_argument("--e16384-padding-result", type=Path)
    parser.add_argument("--e240825-padding-result", type=Path)
    parser.add_argument("--u16384-qualification", type=Path)
    parser.add_argument("--u240825-qualification", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--service-mode", choices=("serial", "Q2"))
    parser.add_argument("--sample-count", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse()
    if parsed.fvm_worker:
        if parsed.route != "FVM240825_reference" or parsed.order_seed is None or parsed.service_mode is None:
            raise SystemExit("FVM worker requires route/order-seed/service-mode")
        raise SystemExit(fvm_worker(parsed))
    required = (
        "artifact_root", "run_dir", "native_padding_result", "e16384_padding_result",
        "e240825_padding_result", "u16384_qualification", "u240825_qualification",
        "checkpoint_sha256", "output_root",
    )
    if any(getattr(parsed, name) is None for name in required):
        raise SystemExit(f"orchestrator requires: {', '.join(required)}")
    raise SystemExit(orchestrate(parsed))
