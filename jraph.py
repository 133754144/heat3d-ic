"""Minimal local subset of jraph used by this repository.

This project only relies on a few type aliases plus ``segment_sum`` and
``segment_mean``. Shipping this tiny compatibility layer avoids requiring the
external ``jraph`` package in environments without internet access.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


ArrayTree = Any
NodeFeatures = Any
GNUpdateEdgeFn = Callable[..., Any]
AggregateEdgesToNodesFn = Callable[..., Any]
AggregateNodesToGlobalsFn = Callable[..., Any]
AggregateEdgesToGlobalsFn = Callable[..., Any]
InteractionUpdateEdgeFn = Callable[..., Any]
EmbedEdgeFn = Callable[..., Any]
EmbedNodeFn = Callable[..., Any]
EmbedGlobalFn = Callable[..., Any]


def segment_sum(data, segment_ids, num_segments=None):
  """Thin wrapper around JAX segment sum."""

  return jax.ops.segment_sum(data, segment_ids, num_segments)


def segment_mean(data, segment_ids, num_segments=None):
  """Computes a segment-wise mean with safe zero-count handling."""

  sums = segment_sum(data, segment_ids, num_segments)
  counts = segment_sum(
    jnp.ones_like(data[..., :1]),
    segment_ids,
    num_segments,
  )
  counts = jnp.where(counts == 0, 1, counts)
  return sums / counts
