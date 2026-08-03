#!/usr/bin/env python3
"""Capture the fixed qualification host without invoking GPU monitor tools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import jax
import jaxlib
import numpy
import scipy


def proc_value(path: Path, prefix: str) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = jax.devices()[0]
    payload = {
        "schema_version": "heat3d_v6_inference_qualification_environment_v1",
        "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": proc_value(Path("/proc/cpuinfo"), "model name"),
        "logical_cpu_count": os.cpu_count(),
        "memory_total": proc_value(Path("/proc/meminfo"), "MemTotal"),
        "python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "model_device": str(device), "model_device_kind": device.device_kind,
        "benchmark_threads": 1, "model_batch_size": 1, "memory_fraction": 0.85,
        "gpu_monitor_command_invoked": False,
        "test_accessed": False, "sealed_accessed": False, "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "host": payload["host"], "device": payload["model_device_kind"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
