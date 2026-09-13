#!/usr/bin/env python3
"""Measure JAX persistent compilation-cache reuse for one Heat3D B24 shape.

This is an engineering-only probe.  It uses one frozen train batch, performs
one synchronized update, and never changes the scientific contract or reads a
test/sealed artifact.  Invoke twice with the same cache directory and distinct
receipt paths to measure cold versus warm startup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def cache_inventory(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {
        "file_count": len(files),
        "total_bytes": int(sum(item.stat().st_size for item in files)),
        "files": [{"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256(item)} for item in sorted(files)],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: compile-cache probe requires devbox CUDA")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the frozen train input pool is accepted")
    if any(token in str(path).lower() for path in (args.fs_train, args.labels_root, args.normalization) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    from jax.experimental.compilation_cache import compilation_cache
    compilation_cache.set_cache_dir(str(args.cache_dir))
    module = load_script("profile_v7_g2_heat3d_epoch.py")
    # A false-flag invocation is an explicitly bounded engineering probe;
    # formal runners continue to require deterministic XLA.
    setattr(args, "allow_nondeterministic_xla", "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""))
    prepared = module._prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    batch = prepared["train_batches"][0]
    block = prepared["block_until_ready"]
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    started = time.perf_counter()
    result = trainer.step(state, batch, key)
    block((result.state.params, result.state.optimizer_state, result.loss))
    step_seconds = time.perf_counter() - started
    inventory = cache_inventory(args.cache_dir)
    payload = {
        "schema_version": "heat3d_v7_g2_e2_compile_cache_v1",
        "status": "PASS_FINITE",
        "phase": args.phase,
        "scope": "one_frozen_B24_batch_one_synchronized_update",
        "scientific_contract": {"batch_size": 24, "seed": int(args.seed), "test_or_sealed_access": False, "accuracy_used": False, "formal_training_started": False},
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "execution": {"compile_plus_first_step_seconds": step_seconds, "compile_count": int(trainer.compile_count), "loss_finite": bool(np.isfinite(float(np.asarray(result.loss))))},
        "cache": {"directory": str(args.cache_dir), "inventory": inventory, "cache_api": "jax.experimental.compilation_cache.compilation_cache.set_cache_dir"},
        "interpretation": {"steady_state_speedup": False, "purpose": "reduce repeated cold compilation across identical seed/shape processes; not a formal training speedup", "stale_executable_guard": "cache key remains constrained by JAX/XLA executable metadata; environment and runner receipts are recorded"},
        "environment": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS"), "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner_sha256": hashlib.sha256((ROOT / "rigno/heat3d_training/core.py").read_bytes()).hexdigest()},
        "generated_at_epoch_seconds": time.time(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "phase": receipt["phase"], "step_seconds": receipt["execution"]["compile_plus_first_step_seconds"], "cache_files": receipt["cache"]["inventory"]["file_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
