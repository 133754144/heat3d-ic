#!/usr/bin/env python3
"""Regression-check uncovered-only U-v2 repair against the frozen primitive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from rigno.models.rigno import RegionInteractionGraphBuilder
from scripts.probe_heat3d_v6_p1i_u1_asymmetric_query import (
    _repair_uncovered_physical_nodes_exact,
)


def digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20260817)
    rows = []
    for count, centers_count in ((1024, 256), (8192, 256), (240825, 256)):
        points = rng.uniform(-1.02, 1.02, size=(count, 3)).astype(np.float32)
        centers = rng.uniform(-1.0, 1.0, size=(centers_count, 3)).astype(np.float32)
        # Construct a sparse frozen-order edge array with deliberate uncovered
        # points, including rows that already contain a non-nearest edge.
        covered = np.arange(0, count, 3, dtype=np.int32)
        edges = np.column_stack((covered, covered % centers_count)).astype(np.int32)
        impl = RegionInteractionGraphBuilder(
            periodic=False, rmesh_levels=3, subsample_factor=4,
            overlap_factor_p2r=1.5, overlap_factor_r2p=2.0,
            node_coordinate_freqs=4, min_physical_coverage=1,
        )
        reference = impl._repair_physical_node_coverage(
            edge_indices=jax.numpy.asarray(edges), centers=centers, points=points
        )
        candidate = _repair_uncovered_physical_nodes_exact(
            edge_indices=jax.numpy.asarray(edges), centers=centers, points=points,
            min_physical_coverage=1,
        )
        equal = bool(np.array_equal(np.asarray(reference), np.asarray(candidate)))
        rows.append({
            "point_count": count, "regional_count": centers_count,
            "edge_array_equal": equal,
            "reference_sha256": digest(reference),
            "candidate_sha256": digest(candidate),
        })
    passed = all(row["edge_array_equal"] and row["reference_sha256"] == row["candidate_sha256"] for row in rows)
    result = {
        "schema_version": "heat3d_v6_p1i_u_v2_exact_repair_check_v1",
        "status": "passed" if passed else "failed",
        "rows": rows,
        "test_or_sealed_accessed": False,
        "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("uncovered-only repair changed frozen edge ordering")
    print(json.dumps({"status": "passed", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
