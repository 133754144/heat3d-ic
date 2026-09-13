#!/usr/bin/env python3
"""Bounded JAX trace for one frozen Heat3D B24 train batch.

The script warms exactly one static batch executable, traces two post-warmup
updates, and writes only a small receipt plus trace artifacts under ``/tmp``.
It never reads a test/sealed split and is not a formal training runner.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
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


def trace_files(trace_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file():
            continue
        rows.append({"path": str(path.relative_to(trace_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def parse_chrome_events(trace_dir: Path) -> dict[str, Any]:
    """Best-effort category timing; XPlane protobuf remains listed, not guessed."""
    events: list[dict[str, Any]] = []
    for path in trace_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            events.extend(payload.get("traceEvents", []))
        except (OSError, ValueError, TypeError):
            pass
    for path in trace_dir.rglob("*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            events.extend(payload.get("traceEvents", []))
        except (OSError, ValueError, TypeError):
            pass
    totals: defaultdict[str, float] = defaultdict(float)
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("dur"), (int, float)):
            continue
        name = str(event.get("name", "<unnamed>"))
        totals[name] += float(event["dur"]) / 1.0e6
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:30]
    total = float(sum(totals.values()))
    return {
        "parsed_json_event_count": len(events),
        "category_timing_available": bool(ranked),
        "top_events_wall_seconds": [
            {"name": name, "wall_seconds": seconds, "share_of_parsed_event_time": seconds / total if total else 0.0}
            for name, seconds in ranked
        ],
        "unparsed_xplane_files_are_not_interpreted": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: trace requires devbox CUDA")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the frozen train input pool is accepted")
    if any(token in str(path).lower() for path in (args.fs_train, args.labels_root, args.normalization) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")
    for path in (args.output, args.trace_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    module = load_script("profile_v7_g2_heat3d_epoch.py")
    # Permit an explicitly non-deterministic trace only as an engineering
    # profiler variant; the formal runner still requires deterministic XLA.
    setattr(args, "allow_nondeterministic_xla", "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""))
    prepared = module._prepare(args)
    trainer = prepared["trainer"]
    state = prepared["state"]
    batch = prepared["train_batches"][0]
    block = prepared["block_until_ready"]
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    started = time.perf_counter()
    warm = trainer.step(state, batch, key)
    block((warm.state.params, warm.state.optimizer_state, warm.loss))
    first_seconds = time.perf_counter() - started
    trace_status = "STARTED"
    trace_error = None
    args.trace_dir.mkdir(parents=True)
    try:
        jax.profiler.start_trace(str(args.trace_dir), create_perfetto_link=False)
        traced_rows = []
        trace_state = warm.state
        for index in (2, 3):
            trace_started = time.perf_counter()
            result = trainer.step(trace_state, batch, key)
            trace_state = result.state
            block((result.state.params, result.state.optimizer_state, result.loss))
            traced_rows.append({
                "step": index,
                "wall_seconds": time.perf_counter() - trace_started,
                "loss_finite": bool(np.isfinite(float(np.asarray(result.loss)))),
            })
        jax.profiler.stop_trace()
        trace_status = "PASS_TRACE_CAPTURED"
    except Exception as exc:  # bounded profiling must leave a receipt on failure
        trace_error = f"{type(exc).__name__}: {exc}"
        try:
            jax.profiler.stop_trace()
        except Exception:
            pass
        traced_rows = []
        trace_status = "FAIL_TRACE_CAPTURE"
    files = trace_files(args.trace_dir)
    payload = {
        "schema_version": "heat3d_v7_g2_e2_jax_trace_v1",
        "status": trace_status,
        "scope": "one_static_B24_batch_two_postwarmup_steps",
        "scientific_contract": {
            "batch_size": 24,
            "train_samples": 768,
            "loss_optimizer_architecture_changed": False,
            "accuracy_used": False,
            "test_or_sealed_access": False,
            "formal_training_started": False,
        },
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "jit": {
            "first_call_compile_plus_execution_seconds": first_seconds,
            "postwarmup_steps": traced_rows,
            "compile_count": int(trainer.compile_count),
            "static_batch_closure": True,
            "cache_key": "batch_id",
        },
        "trace": {
            "directory": str(args.trace_dir),
            "error": trace_error,
            "files": files,
            "chrome_event_summary": parse_chrome_events(args.trace_dir),
        },
        "resource": {
            "device": str(jax.devices()[0]),
            "memory_stats": {str(k): int(v) for k, v in (jax.devices()[0].memory_stats() or {}).items() if isinstance(v, (int, np.integer))},
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "interpretation": {
            "kernel_percentages": "reported only when Chrome JSON events are parseable; XPlane protobuf is not guessed",
            "source_audit": "fused value_and_grad/optimizer/update remains one JIT body; host block is an explicit measurement boundary",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runner_sha256": sha256(ROOT / "rigno/heat3d_training/core.py"),
        },
        "generated_at_epoch_seconds": time.time(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "compile_count": receipt["jit"]["compile_count"], "trace_files": len(receipt["trace"]["files"])}, sort_keys=True))
    return 0 if receipt["status"] == "PASS_TRACE_CAPTURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
