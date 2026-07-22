#!/usr/bin/env python3
"""Run one Heat3D config while recording GPU utilization through NVML."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlDevice:
    def __init__(self, index: int) -> None:
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Utilization),
        ]
        self.library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Memory),
        ]
        self.library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        self._check(self.library.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.library.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(index), ctypes.byref(self.handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    @staticmethod
    def _check(status: int, name: str) -> None:
        if status != 0:
            raise RuntimeError(f"{name} failed with NVML status {status}")

    def sample(self) -> dict[str, float | int]:
        utilization = _Utilization()
        memory = _Memory()
        self._check(
            self.library.nvmlDeviceGetUtilizationRates(
                self.handle, ctypes.byref(utilization)
            ),
            "nvmlDeviceGetUtilizationRates",
        )
        self._check(
            self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)),
            "nvmlDeviceGetMemoryInfo",
        )
        mib = 1024.0 * 1024.0
        return {
            "gpu_utilization_pct": int(utilization.gpu),
            "memory_utilization_pct": int(utilization.memory),
            "device_memory_used_mb": float(memory.used / mib),
            "device_memory_total_mb": float(memory.total / mib),
        }

    def close(self) -> None:
        self.library.nvmlShutdown()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--monitor-jsonl", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    for path in (args.monitor_jsonl, args.pid_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
    device = NvmlDevice(args.device_index)
    command = [
        sys.executable,
        "scripts/run_heat3d_v4_config.py",
        "--config",
        str(args.config),
    ]
    process = subprocess.Popen(command)
    args.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    def _forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    samples = 0
    try:
        with args.monitor_jsonl.open("x", encoding="utf-8") as handle:
            while process.poll() is None:
                record = {
                    "schema_version": "heat3d_gpu_nvml_monitor_v1",
                    "event": "sample",
                    "timestamp_unix": time.time(),
                    "runner_pid": process.pid,
                    "monitor_pid": os.getpid(),
                    **device.sample(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                samples += 1
                time.sleep(args.interval_seconds)
            return_code = int(process.wait())
            final = {
                "schema_version": "heat3d_gpu_nvml_monitor_v1",
                "event": "complete",
                "timestamp_unix": time.time(),
                "runner_pid": process.pid,
                "return_code": return_code,
                "sample_count": samples,
                **device.sample(),
            }
            handle.write(json.dumps(final, sort_keys=True) + "\n")
            handle.flush()
    finally:
        device.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
