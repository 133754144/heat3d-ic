"""Sparse, chunked V8 query metadata cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class QueryChunk:
    start: int
    stop: int
    coords: np.ndarray
    cache_key: str


class GeometryQueryCache:
    """In-memory cache keyed only by support/query geometry and builder id."""

    def __init__(self):
        self._items: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(support_coords: np.ndarray, query_coords: np.ndarray, builder_id: str) -> str:
        return hashlib.sha256(
            f"{builder_id}:{_array_hash(support_coords)}:{_array_hash(query_coords)}".encode()
        ).hexdigest()

    def get_or_build(
        self,
        *,
        support_coords: np.ndarray,
        query_coords: np.ndarray,
        builder_id: str,
        build: Callable[[], Any],
    ) -> tuple[Any, bool, str]:
        key = self.key(support_coords, query_coords, builder_id)
        if key in self._items:
            self.hits += 1
            return self._items[key], True, key
        value = build()
        self._items[key] = value
        self.misses += 1
        return value, False, key


def iter_query_chunks(
    coords: np.ndarray,
    *,
    chunk_size: int,
    support_coords: np.ndarray,
    builder_id: str,
) -> list[QueryChunk]:
    points = np.asarray(coords)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("query coordinates must have shape [N,3]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        result.append(
            QueryChunk(
                start=start,
                stop=stop,
                coords=chunk,
                cache_key=GeometryQueryCache.key(support_coords, chunk, builder_id),
            )
        )
    return result


def concatenate_query_chunks(
    chunks: list[QueryChunk],
    predictions: list[np.ndarray],
    *,
    expected_count: int,
) -> np.ndarray:
    if len(chunks) != len(predictions):
        raise ValueError("chunk/prediction count mismatch")
    cursor = 0
    output = []
    for chunk, prediction in zip(chunks, predictions):
        if chunk.start != cursor or chunk.stop <= chunk.start:
            raise ValueError("query chunk ordering is not contiguous")
        array = np.asarray(prediction)
        if array.shape[0] != chunk.stop - chunk.start:
            raise ValueError("query prediction length does not match chunk")
        output.append(array)
        cursor = chunk.stop
    if cursor != expected_count:
        raise ValueError("query chunks do not cover the expected domain")
    return np.concatenate(output, axis=0)
