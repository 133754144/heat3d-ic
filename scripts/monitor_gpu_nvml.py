#!/usr/bin/env python3
"""Read-only NVIDIA utilization monitor through NVML (no nvidia-smi)."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import time


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit_v2 failed")
    handle = ctypes.c_void_p()
    if nvml.nvmlDeviceGetHandleByIndex_v2(args.device, ctypes.byref(handle)) != 0:
        raise RuntimeError(f"NVML device {args.device} unavailable")
    try:
        while True:
            utilization = Utilization()
            memory = Memory()
            if nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)) != 0:
                raise RuntimeError("nvmlDeviceGetUtilizationRates failed")
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                raise RuntimeError("nvmlDeviceGetMemoryInfo failed")
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(
                f"{timestamp} gpu={utilization.gpu}% mem_util={utilization.memory}% "
                f"memory={memory.used / 2**30:.2f}/{memory.total / 2**30:.2f} GiB",
                flush=True,
            )
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        nvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
