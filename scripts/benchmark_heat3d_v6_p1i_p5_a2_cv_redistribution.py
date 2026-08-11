#!/usr/bin/env python3
"""P5-A2 exact CV redistribution timing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
)


def _redistribute(
    coords: np.ndarray,
    cv: np.ndarray,
    layer: np.ndarray,
    selected: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    result = np.zeros(len(selected), dtype=np.float64)
    assignment = np.full(len(coords), -1, dtype=np.int64)
    timings: dict[str, float] = {}
    for layer_id in sorted(map(int, np.unique(layer))):
        started = time.perf_counter()
        full_local = np.flatnonzero(layer == layer_id)
        support_local = np.flatnonzero(layer[selected] == layer_id)
        tree = cKDTree(coords[selected[support_local]])
        nearest = np.asarray(
            tree.query(coords[full_local], k=1, workers=workers)[1], dtype=np.int64
        )
        mapped = support_local[nearest]
        assignment[full_local] = mapped
        np.add.at(result, mapped, cv[full_local])
        timings[f"layer_{layer_id:02d}"] = time.perf_counter() - started
    return result, assignment, timings


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    sample_ids = [row["sample_id"] for row in preflight["samples"]]
    with h5py.File(args.full_fields, "r") as archive:
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for resolution, route in ((8192, "B8192"), (32768, "E32768")):
        for number, sample_id in enumerate(sample_ids, start=1):
            with np.load(args.artifact_root / "support" / str(resolution) / f"{sample_id}.npz") as payload:
                selected = np.asarray(payload["selected_indices"], dtype=np.int64)
            started = time.perf_counter()
            reference, ref_assignment, ref_layers = _redistribute(
                coords, cv, layer, selected, workers=1
            )
            reference_seconds = time.perf_counter() - started
            started = time.perf_counter()
            candidate, cand_assignment, cand_layers = _redistribute(
                coords, cv, layer, selected, workers=-1
            )
            candidate_seconds = time.perf_counter() - started
            production, production_audit = conservative_selected_control_volume(
                full_coords=coords,
                full_control_volume=cv,
                full_layer_id=layer,
                selected_indices=selected,
            )
            row = {
                "route": route,
                "resolution": resolution,
                "sample_id": sample_id,
                "selected_cv_array_equal": bool(np.array_equal(reference, candidate)),
                "production_array_equal_candidate": bool(np.array_equal(production, candidate)),
                "selected_cv_sha256_equal": array_sha256(reference) == array_sha256(candidate) == production_audit["weights_sha256"],
                "nearest_assignment_equal": bool(np.array_equal(ref_assignment, cand_assignment)),
                "volume_sum_bitwise_equal": bool(np.sum(reference) == np.sum(candidate)),
                "relative_volume_error_not_increased": bool(abs(np.sum(candidate) - np.sum(cv)) <= abs(np.sum(reference) - np.sum(cv))),
                "reference_seconds": reference_seconds,
                "candidate_seconds": candidate_seconds,
                "reference_layer_seconds": ref_layers,
                "candidate_layer_seconds": cand_layers,
            }
            rows.append(row)
            print(f"[P5-A2] {route} {number}/32 exact={row['selected_cv_array_equal']}", flush=True)
    gates = (
        "selected_cv_array_equal", "production_array_equal_candidate",
        "selected_cv_sha256_equal", "nearest_assignment_equal",
        "volume_sum_bitwise_equal", "relative_volume_error_not_increased",
    )
    hard_gate = bool(all(row[key] for row in rows for key in gates))
    summary: dict[str, Any] = {}
    for route in ("B8192", "E32768", "pooled"):
        selected_rows = rows if route == "pooled" else [row for row in rows if row["route"] == route]
        ref = _stats([float(row["reference_seconds"]) for row in selected_rows])
        cand = _stats([float(row["candidate_seconds"]) for row in selected_rows])
        layer_summary = {}
        for name in sorted(selected_rows[0]["reference_layer_seconds"]):
            layer_summary[name] = {
                "reference": _stats([row["reference_layer_seconds"][name] for row in selected_rows]),
                "candidate": _stats([row["candidate_layer_seconds"][name] for row in selected_rows]),
            }
        summary[route] = {
            "reference": ref,
            "candidate": cand,
            "median_speedup": ref["median_seconds"] / max(cand["median_seconds"], 1.0e-30),
            "layers": layer_summary,
        }
    promoted = bool(hard_gate and summary["pooled"]["median_speedup"] > 1.0)
    payload = {
        "schema_version": "heat3d_v6_p1i_p5_a2_result_v1",
        "status": "passed" if hard_gate else "failed",
        "phase": protocol["phase"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "hard_gate_passed": hard_gate,
        "candidate_promoted": promoted,
        "decision": "GO_parallel_ckdtree" if promoted else "NO_GO_keep_single_worker",
        "summary": summary,
        "samples": rows,
        "role_contract": protocol["role_contract"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not hard_gate:
        raise RuntimeError("P5-A2 exact-equivalence gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
