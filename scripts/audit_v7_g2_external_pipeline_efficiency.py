#!/usr/bin/env python3
"""Audit and benchmark the frozen GINO/Transolver input pipeline.

``static`` mode is safe on any machine and reports the current per-sample
hash/``np.load``/CUDA-synchronization boundaries.  ``data`` mode performs a
bounded, accuracy-blind before/after benchmark: the baseline calls the
existing frozen dataset, while the cached path verifies every selected sample
once and then serves byte-identical in-RAM objects.  No model, target metric,
test role, or sealed artifact is opened by this script.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_formal_runner():
    path = ROOT / "scripts/run_v7_g2_p1i_external_formal.py"
    spec = importlib.util.spec_from_file_location("g2_external_formal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_audit() -> dict[str, Any]:
    path = ROOT / "scripts/run_v7_g2_p1i_external_formal.py"
    source = path.read_text(encoding="utf-8")
    parsed = ast.parse(source)
    np_load_calls = sum(
        1
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "load"
    )
    sha_calls = source.count("sha256(")
    synchronize_calls = source.count("torch.cuda.synchronize()")
    return {
        "schema_version": "g2_external_pipeline_static_audit_v1",
        "status": "PASS_STATIC_PIPELINE_AUDIT",
        "baseline_contract": {
            "hashing": "per __getitem__ file SHA verification",
            "loading": "per __getitem__ np.load",
            "device_staging": "per sample tensor.to(device)",
            "timing_sync": "torch.cuda.synchronize around every train/valid step",
        },
        "source_counts": {
            "np_load_calls": np_load_calls,
            "sha256_call_sites": sha_calls,
            "cuda_synchronize_call_sites": synchronize_calls,
        },
        "science_neutral_candidate": {
            "startup_hash_once": True,
            "verified_ram_cache": True,
            "gpu_staging_optional": True,
            "content_byte_identical_required": True,
            "formal_not_started": True,
        },
        "test_or_sealed_access": False,
    }


def _equal_value(left: Any, right: Any) -> bool:
    try:
        import torch

        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return bool(torch.equal(left, right))
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return bool(np.array_equal(left, right))
    except ImportError:
        pass
    return left == right


class CachedRoleDataset:
    """Byte-identical RAM cache around the existing role-restricted dataset."""

    def __init__(self, source: Any):
        self.role = source.role
        started = time.perf_counter()
        self.rows = [copy.deepcopy(source[index]) for index in range(len(source))]
        self.startup_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _bench(dataset: Any, count: int, repeats: int) -> dict[str, Any]:
    count = min(int(count), len(dataset))
    started = time.perf_counter()
    for _ in range(repeats):
        for index in range(count):
            row = dataset[index]
            if not row.get("sample_id"):
                raise ValueError("dataset row missing sample id")
    seconds = time.perf_counter() - started
    return {
        "items": count * repeats,
        "seconds": seconds,
        "items_per_second": (count * repeats) / seconds if seconds else None,
    }


def data_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in {"train", "valid_iid"}:
        raise ValueError("only train and valid_iid roles are permitted")
    if any(token in str(args.dataset_root).lower() for token in ("test", "sealed")):
        raise ValueError("test/sealed paths are forbidden")
    runner = _load_formal_runner()
    baseline = runner.P1iRoleDataset(args.dataset_root, args.dataset_manifest, args.role)
    count = min(args.count, len(baseline))
    baseline_bench = _bench(baseline, count, args.repeats)
    cached = CachedRoleDataset(baseline)
    cached_bench = _bench(cached, count, args.repeats)
    equality_rows = []
    for index in range(count):
        original = baseline[index]
        cached_row = cached[index]
        keys = sorted(set(original) | set(cached_row))
        equal = all(_equal_value(original[key], cached_row[key]) for key in keys)
        equality_rows.append({"index": index, "sample_id": original["sample_id"], "byte_identical_values": equal})
        if not equal:
            raise RuntimeError(f"cached row changed content at index {index}")
    return {
        "schema_version": "g2_external_pipeline_data_benchmark_v1",
        "status": "PASS_BYTE_IDENTICAL_CACHE_BENCHMARK",
        "role": args.role,
        "dataset_count": len(baseline),
        "bench_count": count,
        "repeats": args.repeats,
        "baseline_per_sample_io": baseline_bench,
        "verified_ram_cache": {"startup_seconds": cached.startup_seconds, **cached_bench},
        "speedup_ratio": (
            cached_bench["items_per_second"] / baseline_bench["items_per_second"]
            if baseline_bench["items_per_second"] and cached_bench["items_per_second"]
            else None
        ),
        "equality": {"checked_rows": equality_rows, "all_byte_identical": True},
        "accuracy_read": False,
        "test_or_sealed_access": False,
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "data"), default="static")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--role", choices=("train", "valid_iid"), default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "static":
        payload = static_audit()
    else:
        if args.dataset_root is None or args.dataset_manifest is None:
            parser.error("data mode requires --dataset-root and --dataset-manifest")
        payload = data_benchmark(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
