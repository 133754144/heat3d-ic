#!/usr/bin/env python3
"""Real-route, low-cost conformance smoke for benchmark protocol v1.1.

This runner deliberately emits no publication latency or speedup table.  Each
route/order/service-mode cell runs in its own Python process.  The FVM service
uses in-memory physics payloads and either an in-process persistent P1 service
or a persistent P2 worker pool; neural routes delegate to their frozen real
production implementations.
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

from heat3d_v6_publication_lifecycle_schema import (
    provenance as lifecycle_provenance,
    q2_metrics,
    serial_metrics,
    timing_stats,
    validate_cell,
)

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
    return timing_stats(values)


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
    _FVM_STATE = {"qualification": qualification, "mesh": mesh, "systems": {}}


def _fvm_ready() -> int:
    # Keep readiness tasks alive briefly so a P2 pool must schedule both
    # persistent workers; this is untimed service startup, not a case prepass.
    time.sleep(0.1)
    return os.getpid()


def _fvm_case(
    payload: dict[str, Any], use_cached_system: bool = False,
    cache_system: bool = False,
) -> dict[str, Any]:
    q = _FVM_STATE["qualification"]
    started = time.perf_counter()
    phase = time.perf_counter()
    k = np.asarray(payload["k_xyz"], dtype=np.float64)
    source = np.asarray(payload["q_W_m3"], dtype=np.float64)
    decode = time.perf_counter() - phase
    phase = time.perf_counter()
    if use_cached_system:
        system = _FVM_STATE["systems"][payload["sample_id"]]
        assembly = 0.0
        prepared_system_lookup = time.perf_counter() - phase
    else:
        system = q.prior._assemble(
            _FVM_STATE["mesh"], k, source,
            float(payload["top_h"]), float(payload["bottom_h"]),
        )
        if cache_system:
            _FVM_STATE["systems"][payload["sample_id"]] = system
        assembly = time.perf_counter() - phase
        prepared_system_lookup = 0.0
    phase = time.perf_counter()
    temperature = q.prior._solve(*system)
    solve = time.perf_counter() - phase
    total = time.perf_counter() - started
    residual = total - (decode + assembly + prepared_system_lookup + solve)
    limit = max(0.025, 0.05 * total)
    if residual < -1.0e-6 or residual > limit:
        raise RuntimeError(f"FVM residual {residual} exceeds {limit}")
    if not np.all(np.isfinite(temperature)):
        raise RuntimeError("nonfinite FVM output")
    return {
        "sample_id": payload["sample_id"], "worker_pid": os.getpid(),
        "service_seconds": total,
        "stages": {"payload_decode": decode, "assembly": assembly,
                   "prepared_system_lookup": prepared_system_lookup, "solve": solve},
        "residual_seconds": residual, "residual_limit_seconds": limit,
    }


def fvm_worker(args: argparse.Namespace) -> int:
    os.environ.update(THREAD_ENV)
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
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
        meta = json.loads((data.sample_dir(row) / "sample_meta.json").read_text())
        k, source = data.full_kq(row)
        payloads.append({
            "sample_id": str(row["sample_id"]),
            "k_xyz": np.asarray(k, dtype=np.float64),
            "q_W_m3": np.asarray(source, dtype=np.float64),
            "top_h": float(meta["top_h_W_m2K"]),
            "bottom_h": float(meta["bottom_h_W_m2K"]),
        })
    serialized = {
        "dataset_root": str(args.dataset_root), "manifest": str(args.manifest),
        "full_fields": str(args.full_fields),
    }
    rows_out: list[dict[str, Any]] = []
    cache_hot_rows: list[dict[str, Any]] = []
    resident_rows: list[dict[str, Any]] = []
    q2 = None
    pool = None
    if args.service_mode == "serial":
        # The FVM worker command is itself the independent lifecycle/service.
        # Fresh/Q1 therefore execute directly in this process: no ProcessPool,
        # IPC, serialization, or child scheduling belongs to the timing span.
        _init_fvm(serialized)
        process_count = 1
        worker_pids = [os.getpid()]
        for payload in payloads:
            submitted = time.perf_counter()
            row = _fvm_case(payload, False, args.formal_measurement)
            completed = time.perf_counter()
            row["submit_to_result_seconds"] = completed - submitted
            rows_out.append(row)
        if args.formal_measurement:
            repeated = payloads[0]
            for _ in range(args.cache_hot_repeats):
                submitted = time.perf_counter()
                row = _fvm_case(repeated, True)
                row["submit_to_result_seconds"] = time.perf_counter() - submitted
                cache_hot_rows.append(row)
            # Resident is an independent prepared-system solve-only pool.  It
            # must not alias or fall back to the cache-hot measurements.
            for _ in range(args.cache_hot_repeats):
                resident_rows.append(_fvm_case(repeated, True))
    else:
        process_count = 2
        pool = ProcessPoolExecutor(
            max_workers=process_count, mp_context=mp.get_context("spawn"),
            initializer=_init_fvm, initargs=(serialized,),
        )
        startup_worker_pids = sorted({future.result() for future in [
            pool.submit(_fvm_ready) for _ in range(process_count * 3)]})
        if len(startup_worker_pids) != process_count:
            raise RuntimeError("FVM persistent worker count drift")
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
        # Case-bearing worker PIDs, not readiness-task scheduling, are the
        # authoritative P2 participation evidence.
        worker_pids = sorted({int(row["worker_pid"]) for row in rows_out})
        if len(worker_pids) != process_count:
            raise RuntimeError(
                f"FVM P2 case worker participation drift: {worker_pids}")
        ordered_completion = [row["completion_offset_seconds"] for row in rows_out]
        inter = np.diff(np.asarray([0.0] + ordered_completion)).tolist()
        q2 = {
            "actual_concurrent_execution": True, "queue_depth": 2,
            "worker_count": 2, "serial_prepass_of_Q2_samples": False,
            "submit_to_result": stats([row["submit_to_result_seconds"] for row in rows_out]),
            "inter_completion": stats(inter),
            "samples_per_second": len(rows_out) / ordered_completion[-1],
            "B16_wall_seconds": (
                ordered_completion[15] if len(ordered_completion) >= 16 else None),
            "B32_wall_seconds": (
                ordered_completion[31] if len(ordered_completion) >= 32 else None),
            "true_B16_to_B32_marginal_seconds": (
                (ordered_completion[31] - ordered_completion[15]) / 16.0
                if len(ordered_completion) >= 32 else None),
        }
    if args.service_mode == "serial":
        memory_value = _hwm_bytes(os.getpid())
        memory_field = "service_process_HWM_bytes"
        memory_semantics = "single_in_process_persistent_P1_service_VmHWM"
    else:
        memory_value = _hwm_bytes(os.getpid()) + sum(_hwm_bytes(pid) for pid in worker_pids)
        memory_field = "summed_process_HWM_upper_bound_bytes"
        memory_semantics = "sum_of_parent_and_worker_VmHWM_not_simultaneous_RSS"
        assert pool is not None
        pool.shutdown()
    lifecycle_metrics = None
    if args.formal_measurement and args.service_mode == "serial":
        lifecycle_metrics = serial_metrics(
            cold_seconds=float(rows_out[0]["submit_to_result_seconds"]),
            fresh_q1=stats([row["submit_to_result_seconds"] for row in rows_out]),
            cache_hot=stats([row["submit_to_result_seconds"] for row in cache_hot_rows]),
            resident=stats([row["stages"]["solve"] for row in resident_rows]),
        )
    elif args.formal_measurement and args.service_mode == "Q2":
        if q2 is None:
            raise RuntimeError("FVM Q2 lifecycle missing result")
        lifecycle_metrics = q2_metrics(
            submit_to_result=q2["submit_to_result"],
            inter_completion=q2["inter_completion"],
            throughput_samples_per_second=q2["samples_per_second"],
            b16_to_b32_marginal_seconds=q2["true_B16_to_B32_marginal_seconds"],
        )
    result = {
        "schema_version": "heat3d_v6_publication_fvm_service_smoke_v1_1",
        "status": ("passed" if args.formal_measurement and args.sample_count == 32 else "passed_smoke"),
        "route": args.route, "sample_count": args.sample_count,
        "service_mode": args.service_mode, "process_id": os.getpid(),
        "worker_pids": worker_pids, "worker_count": process_count,
        "startup_barrier_worker_pids": (
            worker_pids if args.service_mode == "serial" else startup_worker_pids),
        "case_worker_pid_evidence": sorted({int(row["worker_pid"]) for row in rows_out}),
        "execution_model": (
            "in_process_persistent_P1_one_thread"
            if args.service_mode == "serial" else "persistent_P2_each_one_thread"),
        "IPC_used_in_fresh_Q1": False,
        "order_seed": args.order_seed,
        "ordered_sample_ids": [payload["sample_id"] for payload in payloads],
        "rows": rows_out, "Q2": q2,
        "warmup": {"kind": "not_applicable_no_JIT", "timed_case_prepass": False},
        "classification": {
            "cold_service_first_case": "first_row" if args.service_mode == "serial" else None,
            "fresh_distinct_case": "all_rows" if args.service_mode == "serial" else None,
            "repeat_case_cache_hot": (
                "separate_post_fresh_prepared_system_cache_pool"
                if args.formal_measurement and args.service_mode == "serial"
                else None),
            "resident_core": (
                "prepared_system_solve_only"
                if args.formal_measurement and args.service_mode == "serial"
                else None),
        },
        "aggregate_service_worker_peak_RAM_bytes": memory_value,
        "memory_measurement": {
            "field": memory_field,
            "value": memory_value,
            "semantics": memory_semantics,
        },
        "repeat_case_cache_hot": (
            stats([row["submit_to_result_seconds"] for row in cache_hot_rows])
            if args.formal_measurement and args.service_mode == "serial"
            else None),
        "resident_core": (
            stats([row["stages"]["solve"] for row in resident_rows])
            if args.formal_measurement and args.service_mode == "serial"
            else None),
        "lifecycle_metrics": lifecycle_metrics,
        "measurement_provenance": lifecycle_provenance(
            attempted=bool(args.formal_measurement), matrix_completed=False, generated=False),
        "thread_env": THREAD_ENV,
        "publication_timing_eligible": False,
        "role_contract": protocol["role_contract"],
    }
    if args.formal_measurement:
        validate_cell(result, formal=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    if args.formal_measurement:
        common.extend(("--golden-seal", str(args.pre_measurement_seal)))
    if route.startswith("E"):
        padding = args.e16384_padding_result if route.startswith("E16384") else args.e240825_padding_result
        command = [
            sys.executable, str(ROOT / "scripts/benchmark_heat3d_v6_p1i_final_e_service.py"),
            *common, "--query-padding-result", str(padding), "--route", route,
            "--order-seed", str(seed), "--service-mode", mode,
            "--resident-repeats", "2" if not args.formal_measurement else "20",
        ]
        command.append("--publication-v1-1" if args.formal_measurement else "--standard-v1-1-smoke")
        return command
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
            "--true-concurrent-depth", "1" if mode == "serial" else "2",
        ]
        command.append("--publication-v1-1" if args.formal_measurement else "--standard-v1-1-smoke")
        if mode == "Q2":
            command.append("--concurrent-only")
        return command
    command = [
        sys.executable, str(Path(__file__).resolve()), "--fvm-worker",
        "--protocol", str(args.protocol), "--binding", str(args.binding),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--route", route,
        "--order-seed", str(seed), "--service-mode", mode,
        "--sample-count", str(args.sample_count), "--output", str(output),
    ]
    if args.formal_measurement:
        command.extend(("--formal-measurement", "--cache-hot-repeats", "20"))
    return command


def ordered_ids(row: dict[str, Any], seed: int) -> list[str]:
    value = row["ordered_sample_ids"]
    if isinstance(value, dict):
        return list(value[str(seed)])
    return list(value)


def validate_completed_cell(
    row: dict[str, Any], *, route: str, seed: int, mode: str,
    expected_ids: list[str], formal: bool,
) -> None:
    """Fail immediately after every cell on lifecycle/order/resource drift."""
    validate_cell(row, formal=formal)
    if row.get("route") != route or row.get("service_mode") != mode:
        raise RuntimeError(f"{route}/{seed}/{mode}: route or mode identity drift")
    if ordered_ids(row, seed) != expected_ids:
        raise RuntimeError(f"{route}/{seed}/{mode}: ordered sample IDs drift")
    if formal and row.get("status") != "passed":
        raise RuntimeError(f"{route}/{seed}/{mode}: formal status is not passed")
    provenance = row.get("measurement_provenance", {})
    if formal and not provenance.get("formal_measurement_attempted"):
        raise RuntimeError(f"{route}/{seed}/{mode}: formal provenance missing")
    if mode == "Q2":
        if route.startswith("E") and row.get("serial_orders"):
            raise RuntimeError(f"{route}/{seed}: Q2 traversed serial population")
        if route.startswith("U") and not row.get("concurrent_only"):
            raise RuntimeError(f"{route}/{seed}: U Q2 is not concurrent-only")
        if route.startswith("FVM") and len(row.get("case_worker_pid_evidence", [])) != 2:
            raise RuntimeError(f"{route}/{seed}: FVM P2 did not use two case workers")
    if not route.startswith("FVM"):
        warmup = row.get("warmup", {})
        if warmup.get("source_split") != "train" or warmup.get("target_read"):
            raise RuntimeError(f"{route}/{seed}/{mode}: warmup role drift")
        if warmup.get("timed_graph_or_packing_prebuilt"):
            raise RuntimeError(f"{route}/{seed}/{mode}: timed case prewarmed")


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
    if args.formal_measurement:
        seal = json.loads(args.pre_measurement_seal.read_text())
        if (seal["pre_measurement_seal"] != "GO"
                or seal["ready_for_authoritative_valid32"] != "GO"
                or seal.get("benchmark_lifecycle_schema") != "GO"
                or seal.get("benchmark_runtime_isolation") != "GO"
                or seal["publication_timing_freeze"] != "NO_GO_ready_for_full_valid32"
                or seal["protocol_sha256"] != file_sha256(
                    ROOT / "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_protocol.json")):
            raise RuntimeError("formal measurement pre-measurement seal drift")
        for relative, expected in seal["frozen_implementation_sha256"].items():
            candidate = ROOT / relative
            if not candidate.is_file() or file_sha256(candidate) != expected:
                raise RuntimeError(f"formal implementation SHA drift: {relative}")
        if args.sample_count != 32:
            raise RuntimeError("formal measurement requires frozen valid32")
    elif args.sample_count < 2 or args.sample_count > 4:
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
                    persisted = None
                    if output.exists():
                        try:
                            persisted = json.loads(output.read_text())
                        except Exception as exc:
                            persisted = {"parse_error": f"{type(exc).__name__}: {exc}"}
                    failure = {
                        **record, "stderr_tail": completed.stderr[-4000:],
                        "inner_failure_artifact": persisted,
                        "inner_failure_artifact_sha256": (
                            file_sha256(output) if output.exists() else None),
                    }
                    break
                row = json.loads(output.read_text())
                row["route"] = route
                row["service_mode"] = mode
                row["order_seed"] = seed
                row["artifact_path"] = str(output)
                row["artifact_sha256"] = file_sha256(output)
                expected_order = np.random.default_rng(seed).permutation(32)[:args.sample_count].tolist()
                binding_ids = json.loads(args.binding.read_text())["development_subset"]["sample_ids"]
                expected_ids = [binding_ids[int(index)] for index in expected_order]
                validate_completed_cell(
                    row, route=route, seed=seed, mode=mode,
                    expected_ids=expected_ids, formal=bool(args.formal_measurement))
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
        "formal_measurement_attempted": bool(args.formal_measurement),
        "formal_matrix_completed": False,
        "publication_results_generated": False,
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
        fvm_serial = next(row for row in rows
                          if row["route"] == "FVM240825_reference"
                          and row["service_mode"] == "serial")
        if not (
            fvm_serial["execution_model"] == "in_process_persistent_P1_one_thread"
            and fvm_serial["worker_pids"] == [fvm_serial["process_id"]]
            and fvm_serial["worker_count"] == 1
            and not fvm_serial["IPC_used_in_fresh_Q1"]
        ):
            raise RuntimeError("FVM Fresh/Q1 is not an in-process persistent P1 service")
        exactness = exactness_gate(rows)
        result.update({
            "status": "passed",
            "formal_matrix_completed": bool(args.formal_measurement),
            "benchmark_protocol_v1_1": "GO",
            "benchmark_implementation_freeze": "GO",
            "publication_timing_freeze": (
                "NO_GO_pending_collector" if args.formal_measurement
                else "NO_GO_pending_full_valid32"),
            "authoritative_full_valid32": (
                "completed_hard_gates_passed" if args.formal_measurement else "not_executed_smoke"),
            "independent_process_count": 30,
            "independent_process_lifecycle": True,
            "same_seed_cross_route_order_exact": True,
            "E_U_CPU_resource_policy_equal": True,
            "Q2_without_serial_prepass": True,
            "cold_fresh_cache_hot_classification_separate": True,
            "real_route_smoke_only": not bool(args.formal_measurement),
            "measurement_role": (
                "formal_full_valid32" if args.formal_measurement else "conformance_smoke"),
            "exactness_provenance": exactness,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    parser.add_argument("--formal-measurement", action="store_true")
    parser.add_argument("--cache-hot-repeats", type=int, default=20)
    parser.add_argument("--pre-measurement-seal", type=Path)
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
    if parsed.formal_measurement and parsed.pre_measurement_seal is None:
        raise SystemExit("formal measurement requires --pre-measurement-seal")
    raise SystemExit(orchestrate(parsed))
