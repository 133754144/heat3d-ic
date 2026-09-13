#!/usr/bin/env python3
"""Audit the frozen Heat3D B24 batch shapes and per-batch JIT policy.

This is a bounded, geometry/runtime-only receipt.  It prepares the frozen
768-train/128-valid_iid labels, but does not execute a training update or read
any test/sealed artifact.  The purpose is to make the current closure-capture
JIT granularity explicit before considering a separate execution candidate.
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


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p50", "p90", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def edge_row_stats(metadata: Any, field_name: str, sender: int, receiver: int) -> dict[str, Any]:
    value = getattr(metadata, field_name)
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"unexpected {field_name} shape {array.shape}")
    real_counts: list[int] = []
    padded_counts: list[int] = []
    for row in array:
        padded = int(np.count_nonzero((row[:, 0] == sender) & (row[:, 1] == receiver)))
        padded_counts.append(padded)
        real_counts.append(int(row.shape[0] - padded))
    return {
        "padded_shape": [int(x) for x in array.shape],
        "real_counts": real_counts,
        "padded_counts": padded_counts,
        "real_total": int(sum(real_counts)),
        "padded_total": int(sum(padded_counts)),
        "total_slots": int(array.shape[0] * array.shape[1]),
        "padding_ratio_over_real": float(sum(padded_counts) / max(sum(real_counts), 1)),
        "padding_fraction_of_slots": float(sum(padded_counts) / max(array.shape[0] * array.shape[1], 1)),
        "real_count_summary": quantiles(real_counts),
        "padded_count_summary": quantiles(padded_counts),
    }


def batch_row(batch: Any, role: str) -> dict[str, Any]:
    group = batch.groups[0]
    metadata = group["metadata"]
    p_in = int(np.asarray(metadata.x_pnodes_inp).shape[1] - 1)
    p_out = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    r_nodes = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    return {
        "role": role,
        "batch_id": str(batch.batch_id),
        "sample_count": int(len(batch.sample_ids)),
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([str(x) for x in batch.sample_ids], separators=(",", ":")).encode()
        ).hexdigest(),
        "node_shapes": {
            "x_pnodes_inp": [int(x) for x in np.asarray(metadata.x_pnodes_inp).shape],
            "x_rnodes": [int(x) for x in np.asarray(metadata.x_rnodes).shape],
            "x_pnodes_out": [int(x) for x in np.asarray(metadata.x_pnodes_out).shape],
            "real_nodes_per_sample": {"p_in": p_in, "r": r_nodes, "p_out": p_out},
        },
        "edges": {
            "p2r": edge_row_stats(metadata, "p2r_edge_indices", p_in, r_nodes),
            "r2r": edge_row_stats(metadata, "r2r_edge_indices", r_nodes, r_nodes),
            "r2p": edge_row_stats(metadata, "r2p_edge_indices", r_nodes, p_out),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: shape audit requires devbox CUDA")
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only the frozen train input pool is accepted")
    if any(token in str(path).lower() for path in (args.fs_train, args.labels_root, args.normalization) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    module = load_script("profile_v7_g2_heat3d_epoch.py")
    prepared = module._prepare(args)
    rows = [
        *(batch_row(batch, "train") for batch in prepared["train_batches"]),
        *(batch_row(batch, "valid_iid") for batch in prepared["valid_batches"]),
    ]
    train_rows = [row for row in rows if row["role"] == "train"]
    valid_rows = [row for row in rows if row["role"] == "valid_iid"]
    payload = {
        "schema_version": "heat3d_v7_g2_e2_jit_shape_audit_v1",
        "status": "PASS_SHAPES_AND_STATIC_JIT_AUDIT",
        "scope": "frozen_768_train_128_valid_iid_B24_no_update",
        "scientific_contract": {
            "train_samples": 768,
            "train_batch_size": 24,
            "train_batches": len(train_rows),
            "valid_samples": 128,
            "valid_batch_size": 32,
            "valid_batches": len(valid_rows),
            "test_or_sealed_access": False,
            "accuracy_used": False,
        },
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "jit_policy": {
            "source": "rigno/heat3d_training/core.py:218-225",
            "closure_captures_batch": True,
            "cache_key": "batch_id",
            "observed_B24_executables_from_runtime_receipt": 32,
            "observed_B24_unique_batch_signatures": 32,
            "unique_executable_per_train_batch": True,
            "HLO_count_interpretation": "one compiled executable per distinct static batch closure; exact HLO-module count not separately exported by current runner",
            "dynamic_or_bucket_candidate_tested": False,
        },
        "train_batches": train_rows,
        "valid_batches": valid_rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runner_sha256": file_sha(ROOT / "rigno/heat3d_training/core.py"),
        },
        "formal_accuracy_claim_allowed": False,
        "test_or_sealed_access": False,
        "generated_at_epoch_seconds": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "train_batches": receipt["scientific_contract"]["train_batches"], "valid_batches": receipt["scientific_contract"]["valid_batches"], "preparation_seconds": receipt["preparation_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
