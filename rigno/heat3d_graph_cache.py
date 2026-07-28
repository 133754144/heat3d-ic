"""Deterministic graph-metadata cache for fixed-support Heat3D inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rigno.models.rigno import RegionInteractionGraphMetadata


METADATA_FIELDS = RegionInteractionGraphMetadata._fields


def _hash_arrays(items: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def metadata_hash(metadata: RegionInteractionGraphMetadata) -> str:
    return _hash_arrays(
        [(field, getattr(metadata, field)) for field in METADATA_FIELDS]
    )


def graph_hash(graphs: Any) -> str:
    leaves = jax.tree_util.tree_leaves(graphs)
    return _hash_arrays([(f"leaf_{index:04d}", value) for index, value in enumerate(leaves)])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key_payload(
    *,
    support_hash: str,
    graph_config: Mapping[str, Any],
    graph_seed: int,
    commit: str,
) -> dict[str, Any]:
    if len(support_hash) != 64 or len(commit) != 40:
        raise ValueError("cache support hash/commit must be full SHA values")
    return {
        "schema_version": "heat3d_graph_cache_key_v1",
        "support_hash": support_hash,
        "graph_config": dict(sorted(graph_config.items())),
        "graph_seed": int(graph_seed),
        "commit": commit,
    }


def cache_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_metadata(path: Path, metadata: RegionInteractionGraphMetadata) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, np.ndarray] = {}
    none_fields = []
    for field in METADATA_FIELDS:
        value = getattr(metadata, field)
        if value is None:
            none_fields.append(field)
        else:
            values[field] = np.asarray(value)
    values["__none_fields_utf8"] = np.asarray(none_fields, dtype="S64")
    started = time.perf_counter()
    with path.open("wb") as handle:
        np.savez(handle, **values)
    return {
        "save_seconds": float(time.perf_counter() - started),
        "cache_file_sha256": file_sha256(path),
        "cache_file_bytes": path.stat().st_size,
    }


def load_metadata(path: Path) -> tuple[RegionInteractionGraphMetadata, dict[str, Any]]:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as payload:
        none_fields = {
            value.decode("utf-8")
            for value in np.asarray(payload["__none_fields_utf8"]).tolist()
        }
        values = {
            field: (
                None
                if field in none_fields
                else jnp.asarray(np.asarray(payload[field]))
            )
            for field in METADATA_FIELDS
        }
    elapsed = time.perf_counter() - started
    metadata = RegionInteractionGraphMetadata(**values)
    return metadata, {
        "load_seconds": float(elapsed),
        "cache_file_sha256": file_sha256(path),
        "cache_file_bytes": path.stat().st_size,
        "metadata_hash": metadata_hash(metadata),
    }
