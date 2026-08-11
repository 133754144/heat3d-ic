#!/usr/bin/env python3
"""P5-A1 exact-equivalence and timing gate for nested support ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rigno.heat3d_v6_p1i_anchor_query as anchor_query  # noqa: E402


def _reference_weighted_interleave(
    buckets: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    queues = {name: list(map(int, values)) for name, values in buckets.items()}
    consumed = {name: 0 for name in queues}
    result: list[int] = []
    while any(queues.values()):
        active = [name for name, values in queues.items() if values]
        total_weight = sum(float(weights[name]) for name in active)
        step = len(result) + 1
        chosen = max(
            active,
            key=lambda name: (
                float(weights[name]) / total_weight * step - consumed[name],
                -list(sorted(active)).index(name),
            ),
        )
        result.append(queues[chosen].pop(0))
        consumed[chosen] += 1
    return np.asarray(result, dtype=np.int64)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "mean_seconds": float(np.mean(ordered)),
        "median_seconds": float(np.median(ordered)),
        "p95_seconds": float(np.quantile(ordered, 0.95)),
        "min_seconds": float(np.min(ordered)),
        "max_seconds": float(np.max(ordered)),
    }


def _run_order(
    *,
    interleave: Any,
    sample_id: str,
    anchors: np.ndarray,
    coords: np.ndarray,
    cv: np.ndarray,
    layer: np.ndarray,
    q: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    original_interleave = anchor_query._weighted_interleave
    original_hash_order = anchor_query._hash_order
    timing = {"sha256_sort_seconds": 0.0, "weighted_interleave_seconds": 0.0}

    def timed_hash_order(*args: Any, **kwargs: Any) -> np.ndarray:
        started = time.perf_counter()
        result = original_hash_order(*args, **kwargs)
        timing["sha256_sort_seconds"] += time.perf_counter() - started
        return result

    def timed_interleave(*args: Any, **kwargs: Any) -> np.ndarray:
        started = time.perf_counter()
        result = interleave(*args, **kwargs)
        timing["weighted_interleave_seconds"] += time.perf_counter() - started
        return result

    anchor_query._hash_order = timed_hash_order
    anchor_query._weighted_interleave = timed_interleave
    started = time.perf_counter()
    try:
        order, audit = anchor_query.deterministic_nested_query_order(
            sample_id=sample_id,
            anchor_indices=anchors,
            full_coords=coords,
            full_control_volume=cv,
            full_layer_id=layer,
            full_q=q,
            layer_boundaries_m=boundaries,
        )
    finally:
        anchor_query._hash_order = original_hash_order
        anchor_query._weighted_interleave = original_interleave
    timing["full_order_seconds"] = time.perf_counter() - started
    return order, audit, timing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    preflight = json.loads(
        (args.artifact_root / "actual_data_preflight.json").read_text(encoding="utf-8")
    )
    sample_ids = [row["sample_id"] for row in preflight["samples"]]
    if len(sample_ids) != 32 or len(set(sample_ids)) != 32:
        raise RuntimeError("frozen valid32 population is not exactly 32 unique samples")
    with h5py.File(args.full_fields, "r") as archive:
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
    boundaries = np.asarray(preflight["shared"]["layer_boundaries_m"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for number, sample_id in enumerate(sample_ids, start=1):
        with np.load(args.artifact_root / "support" / "8192" / f"{sample_id}.npz") as support:
            anchors = np.asarray(support["selected_indices"][:1024], dtype=np.int64)
        with np.load(args.artifact_root / "physics" / f"{sample_id}.npz") as physics:
            q = np.asarray(physics["q_W_m3"], dtype=np.float64)
        reference, ref_audit, ref_timing = _run_order(
            interleave=_reference_weighted_interleave,
            sample_id=sample_id,
            anchors=anchors,
            coords=coords,
            cv=cv,
            layer=layer,
            q=q,
            boundaries=boundaries,
        )
        candidate, cand_audit, cand_timing = _run_order(
            interleave=anchor_query._weighted_interleave,
            sample_id=sample_id,
            anchors=anchors,
            coords=coords,
            cv=cv,
            layer=layer,
            q=q,
            boundaries=boundaries,
        )
        exact = bool(np.array_equal(reference, candidate))
        row = {
            "sample_id": sample_id,
            "selected_indices_array_equal": exact,
            "reference_order_sha256": ref_audit["order_sha256"],
            "candidate_order_sha256": cand_audit["order_sha256"],
            "selected_indices_sha256_equal": ref_audit["order_sha256"] == cand_audit["order_sha256"],
            "full_order_is_permutation": bool(len(np.unique(candidate)) == len(candidate) == len(coords)),
            "anchor_prefix_exact": bool(np.array_equal(candidate[:1024], anchors)),
            "reference_timing": ref_timing,
            "candidate_timing": cand_timing,
        }
        rows.append(row)
        print(f"[P5-A1] {number}/32 {sample_id} exact={exact}", flush=True)

    timing = {}
    for name in ("full_order_seconds", "sha256_sort_seconds", "weighted_interleave_seconds"):
        reference_values = [float(row["reference_timing"][name]) for row in rows]
        candidate_values = [float(row["candidate_timing"][name]) for row in rows]
        ref = _summary(reference_values)
        cand = _summary(candidate_values)
        timing[name] = {
            "reference": ref,
            "candidate": cand,
            "median_speedup": ref["median_seconds"] / max(cand["median_seconds"], 1.0e-30),
        }
    hard_gate = bool(all(
        row[gate]
        for row in rows
        for gate in (
            "selected_indices_array_equal",
            "selected_indices_sha256_equal",
            "full_order_is_permutation",
            "anchor_prefix_exact",
        )
    ))
    promoted = bool(
        hard_gate and timing["full_order_seconds"]["median_speedup"] > 1.0
    )
    payload = {
        "schema_version": "heat3d_v6_p1i_p5_a1_result_v1",
        "status": "passed" if hard_gate else "failed",
        "phase": protocol["phase"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "population": protocol["population"],
        "hard_gate_passed": hard_gate,
        "candidate_promoted": promoted,
        "decision": "GO_cursor_interleave" if promoted else "NO_GO_keep_reference",
        "timing": timing,
        "samples": rows,
        "role_contract": protocol["role_contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not hard_gate:
        raise RuntimeError("P5-A1 exact-equivalence gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
