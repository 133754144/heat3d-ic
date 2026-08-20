#!/usr/bin/env python3
"""Check host-packed graph metadata against a frozen builder source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.models.rigno import RegionInteractionGraphBuilder


FIELDS = (
    "x_pnodes_inp", "x_pnodes_out", "x_rnodes", "r_rnodes",
    "p2r_edge_indices", "r2r_edge_indices", "r2r_edge_domains",
    "r2p_edge_indices",
)


def _load_reference(path: Path) -> type[Any]:
    spec = importlib.util.spec_from_file_location("frozen_rigno_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RegionInteractionGraphBuilder


def _builder(cls: type[Any], factor: int) -> Any:
    return cls(
        periodic=False,
        rmesh_levels=3,
        subsample_factor=factor,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=2.0,
        node_coordinate_freqs=4,
        coverage_repair_policy="none",
        radius_policy="discrete_physical_coverage",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
        discrete_graph_backend="sparse_kdtree_v1",
        discrete_graph_chunk_size=1024,
        discrete_coverage_multiplier=1.0,
        reuse_exact_p2r_for_r2p=True,
    )


def _digest(metadata: Any) -> str:
    digest = hashlib.sha256()
    for field in FIELDS:
        value = getattr(metadata, field)
        digest.update(field.encode())
        if value is None:
            digest.update(b"none")
            continue
        array = np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--nodes", type=int, nargs="+", default=(1024, 8192, 16384))
    args = parser.parse_args()
    reference_cls = _load_reference(args.reference_source)
    rows = []
    for node_count in args.nodes:
        factor = 8 if node_count >= 8192 else 4
        rng = np.random.default_rng(20260820 + node_count)
        coords = rng.random((node_count, 3), dtype=np.float32)
        domain = np.stack((coords.min(axis=0), coords.max(axis=0)))
        key = jax.random.PRNGKey(20260712)
        reference = _builder(reference_cls, factor).build_metadata(
            x_inp=coords, x_out=coords, domain=domain, key=key)
        candidate = _builder(RegionInteractionGraphBuilder, factor).build_metadata(
            x_inp=coords, x_out=coords, domain=domain, key=key)
        exact = {}
        for field in FIELDS:
            old = getattr(reference, field)
            new = getattr(candidate, field)
            exact[field] = bool(
                old is None and new is None
                or old is not None and new is not None
                and np.array_equal(np.asarray(old), np.asarray(new))
            )
        rows.append({
            "nodes": node_count,
            "factor": factor,
            "fields_exact": exact,
            "reference_sha256": _digest(reference),
            "candidate_sha256": _digest(candidate),
            "passed": all(exact.values()) and _digest(reference) == _digest(candidate),
        })
    payload = {
        "schema_version": "heat3d_v6_graph_host_runtime_exact_v1",
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "rows": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise RuntimeError("host graph runtime changed frozen metadata bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
