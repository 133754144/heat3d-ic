#!/usr/bin/env python3
"""Collect closeout provenance from existing supplemental S3 artifacts only.

This tool does not import JAX, construct graphs, or execute inference.  It hashes
the already-written runner JSON/log files and records host facts independently
from the execution-embedded runtime contract.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys


ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
)
MODES = ("fresh_new_case", "known_topology_new_physics")
EXPECTED_EXECUTION_COMMIT = "8a812619ab0112b4ecfc37ef18189f731180059d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "N/A"


def first_cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "N/A"


def windows_hardware() -> dict[str, object]:
    powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    require(powershell.is_file(), "Windows hardware evidence unavailable")
    script = (
        "$gpu=Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion;"
        "$cpu=Get-CimInstance Win32_Processor | "
        "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors;"
        "@{gpu=$gpu;cpu=$cpu}|ConvertTo-Json -Compress -Depth 4"
    )
    raw = subprocess.check_output(
        [str(powershell), "-NoProfile", "-Command", script], text=True
    ).strip()
    payload = json.loads(raw)
    gpus = payload.get("gpu", [])
    if isinstance(gpus, dict):
        gpus = [gpus]
    nvidia = [row for row in gpus if "NVIDIA" in str(row.get("Name", ""))]
    require(len(nvidia) == 1, "exactly one NVIDIA device must be observed")
    return {"nvidia_gpu": nvidia[0], "windows_cpu": payload.get("cpu")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine-role", default="devbox")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    artifact = args.artifact_root.resolve()
    s3 = artifact / "work" / "S3"
    require(s3.is_dir(), f"S3 artifact root absent: {s3}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    require(head == EXPECTED_EXECUTION_COMMIT, f"execution checkout drift: {head}")

    source_files: list[dict[str, object]] = []
    cpu_policies: set[str] = set()
    thread_envs: set[str] = set()
    process_ids: set[int] = set()
    cuda_log_count = 0
    gpu_runtime_record_count = 0
    result_count = 0
    log_count = 0
    for route in ROUTES:
        for group_index in range(8):
            group = s3 / route / f"group_{group_index}"
            require((group / "input_plan.json").is_file(), f"input plan absent: {group}")
            source_files.append({
                "kind": "input_plan",
                "path": str((group / "input_plan.json").relative_to(repo)),
                "sha256": sha256(group / "input_plan.json"),
            })
            for mode in MODES:
                path = group / ("fresh" if mode == "fresh_new_case" else "known") / f"{mode}.json"
                log = path.with_suffix(".log")
                require(path.is_file() and log.is_file(), f"runner evidence absent: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                require(data.get("route") == route, f"route drift: {path}")
                require(data.get("status") in {"passed", "passed_smoke"}, f"status drift: {path}")
                checkpoint_unchanged = data.get(
                    "checkpoint_unchanged", data.get("checkpoint_parameters_unchanged")
                )
                require(checkpoint_unchanged is True, f"checkpoint drift: {path}")
                role = data.get("role_contract", {})
                for forbidden in ("training", "test", "sealed", "accuracy_tuning"):
                    require(role.get(forbidden) is False, f"forbidden role {forbidden}: {path}")
                cpu_policies.add(json.dumps(data["cpu_policy"], sort_keys=True))
                thread_envs.add(json.dumps(data["thread_env"], sort_keys=True))
                process_ids.add(int(data["process_id"]))
                if int(data.get("peak_vram_bytes", 0)) > 0:
                    gpu_runtime_record_count += 1
                else:
                    memory = data.get("memory", {})
                    if memory.get("platform") == "gpu" and str(memory.get("device", "")).startswith("cuda:"):
                        gpu_runtime_record_count += 1
                log_text = log.read_text(encoding="utf-8", errors="replace")
                if "CUDA" in log_text or "cuda" in log_text or "CudaDevice" in log_text:
                    cuda_log_count += 1
                for kind, item in (("runner_json", path), ("runner_log", log)):
                    source_files.append({
                        "kind": kind,
                        "path": str(item.relative_to(repo)),
                        "sha256": sha256(item),
                        "size_bytes": item.stat().st_size,
                    })
                result_count += 1
                log_count += 1

    require(result_count == 64 and log_count == 64, "S3 result/log population drift")
    require(len(cpu_policies) == 1, "CPU policy differs across S3 runners")
    require(len(thread_envs) == 1, "thread environment differs across S3 runners")
    require(len(process_ids) == 64, "S3 runners did not use independent processes")
    require(gpu_runtime_record_count == 64, "GPU runtime is not evidenced in every S3 result")
    require(cuda_log_count >= 62, "CUDA log evidence population is unexpectedly incomplete")
    hardware = windows_hardware()
    mem_total_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    artifact_set_sha = hashlib.sha256(
        json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": "heat3d_v6_supplemental_environment_provenance_v1",
        "status": "complete_from_existing_artifacts_no_inference",
        "execution_commit": head,
        "machine": {
            "role": args.machine_role,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": first_cpu_model(),
            "cpu_cores": hardware["windows_cpu"],
            "gpu": hardware["nvidia_gpu"]["Name"],
            "gpu_driver": hardware["nvidia_gpu"]["DriverVersion"],
            "memory_total_kib": mem_total_kib,
        },
        "software": {
            "python": sys.version.split()[0],
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "backend": "CUDA",
            "backend_evidence": "all_64_S3_results_record_GPU_runtime; 62_of_64_logs_explicitly_name_CUDA",
        },
        "execution_embedded_contract": {
            "cpu_policy": json.loads(next(iter(cpu_policies))),
            "thread_env": json.loads(next(iter(thread_envs))),
            "worker_policy": "one_independent_process_per_route_geometry_mode; graph/kdtree/reconstruction/support workers=1",
            "independent_process_count": len(process_ids),
        },
        "evidence_scope": {
            "artifact_root": str(artifact.relative_to(repo)),
            "runner_json_count": result_count,
            "runner_log_count": log_count,
            "input_plan_count": 32,
            "cuda_evidenced_log_count": cuda_log_count,
            "gpu_runtime_record_count": gpu_runtime_record_count,
            "collection_is_closeout_only": True,
            "gpu_inference_executed_by_collector": False,
            "physics_experiment_executed_by_collector": False,
            "host_observation_time_utc": subprocess.check_output(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
            ).strip(),
            "machine_role_source": "SSH_alias_and_user_authorized_execution_host",
            "hardware_observation_source": "same_host_procfs_and_Windows_CIM_at_closeout",
            "execution_contract_source": "existing_S3_runner_JSON_and_logs",
        },
        "source_artifact_set_sha256": artifact_set_sha,
        "source_artifacts": source_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "artifact_set_sha256": artifact_set_sha,
        "runner_json_count": result_count,
        "runner_log_count": log_count,
        "hostname": payload["machine"]["hostname"],
        "gpu": payload["machine"]["gpu"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
