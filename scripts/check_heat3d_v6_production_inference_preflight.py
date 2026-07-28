#!/usr/bin/env python3
"""Check graph-backend and full-field reconstruction production invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash, metadata_hash  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402


BASE_GRAPH_CONFIG = {
    "rmesh_levels": 3,
    "subsample_factor": 4,
    "overlap_factor_p2r": 1.5,
    "overlap_factor_r2p": 2.0,
    "node_coordinate_encoding": "raw",
    "node_coordinate_freqs": 4,
    "coverage_repair_policy": "none",
    "radius_policy": "discrete_physical_coverage",
    "repair_p2r": True,
    "repair_r2p": True,
    "min_physical_coverage": 1,
    "discrete_graph_chunk_size": 256,
    "discrete_coverage_multiplier": 1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    args = parser.parse_args()
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    archive_path = args.dataset / "full_fields.h5"
    graph_rows = []
    with h5py.File(archive_path, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        layer_id = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
        for resolution in (1024, 4096):
            probe = ladder["probes"][str(resolution)]
            indices = np.asarray(probe["indices"], dtype=np.int32)
            support = np.asarray(coords[indices], dtype=np.float32)
            rows = []
            for backend in (
                "dense_reference",
                "chunked_numpy_v1",
                "sparse_kdtree_v1",
            ):
                builder = Heat3DGraphBuilder(
                    **BASE_GRAPH_CONFIG, discrete_graph_backend=backend
                )
                started = time.perf_counter()
                metadata = builder.build_metadata(
                    support, key=jax.random.PRNGKey(0)
                )
                build_seconds = time.perf_counter() - started
                rows.append(
                    {
                        "backend": backend,
                        "metadata": metadata,
                        "metadata_hash": metadata_hash(metadata),
                        "graph_hash": graph_hash(builder.build_graphs(metadata)),
                        "build_seconds": float(build_seconds),
                    }
                )
            for candidate in rows[1:]:
                if (
                    rows[0]["metadata_hash"] != candidate["metadata_hash"]
                    or rows[0]["graph_hash"] != candidate["graph_hash"]
                ):
                    raise AssertionError(
                        f"{resolution}:{candidate['backend']}: graph hash mismatch"
                    )
                for field in rows[0]["metadata"]._fields:
                    left = getattr(rows[0]["metadata"], field)
                    right = getattr(candidate["metadata"], field)
                    if left is None or right is None:
                        if left is not None or right is not None:
                            raise AssertionError(
                                f"{resolution}:{candidate['backend']}:{field}: None mismatch"
                            )
                    elif not np.array_equal(np.asarray(left), np.asarray(right)):
                        raise AssertionError(
                            f"{resolution}:{candidate['backend']}:{field}: array mismatch"
                        )
            graph_rows.append(
                {
                    "resolution": resolution,
                    "metadata_hash": rows[0]["metadata_hash"],
                    "graph_hash": rows[0]["graph_hash"],
                    "dense_build_seconds": rows[0]["build_seconds"],
                    "chunked_build_seconds": rows[1]["build_seconds"],
                    "sparse_kdtree_build_seconds": rows[2]["build_seconds"],
                }
            )
        indices = np.asarray(ladder["probes"]["1024"]["indices"], dtype=np.int32)
        mapping, audit = build_reconstruction_map(
            coords=np.asarray(coords, dtype=np.float64),
            layer_id=layer_id,
            boundaries=boundaries,
            support_indices=indices,
        )
        support_values = np.arange(len(indices), dtype=np.float64)
        reconstructed = mapping.reconstruct(support_values)
        selected_error = float(
            np.max(np.abs(reconstructed[indices] - support_values))
        )
        if selected_error > 1.0e-12:
            raise AssertionError("reconstruction is not exact at support nodes")
        if audit["target_or_split_inputs"]:
            raise AssertionError("reconstruction map uses target/split inputs")
    payload = {
        "status": "passed",
        "graph_backend_rows": graph_rows,
        "reconstruction": {
            "support_exact_max_abs_error": selected_error,
            "partition_of_unity_max_abs_error": audit[
                "partition_of_unity_max_abs_error"
            ],
            "all_domain_support_nonempty": all(
                row["support_node_count"] > 0
                for row in audit["domain_coverage"].values()
            ),
            "label_independent": audit["label_independent"],
        },
        "test_hard_accessed": False,
        "training_executed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
