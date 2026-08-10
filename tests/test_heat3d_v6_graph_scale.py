from __future__ import annotations

import numpy as np
import jax

from rigno.heat3d_v6_graph_scale import _nearest_reference_values
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder


def test_native_radius_assignment_is_geometry_only_and_deterministic() -> None:
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    radii = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    query = np.asarray([[0.01, 0, 0], [0.9, 0, 0], [0, 0.8, 0]], dtype=np.float32)
    first = _nearest_reference_values(query, reference, radii, chunk_size=1)
    second = _nearest_reference_values(query, reference, radii, chunk_size=3)
    np.testing.assert_array_equal(first, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_array_equal(first, second)


def test_native_radius_assignment_uses_float32_lowest_index_tie_rule() -> None:
    reference = np.asarray([[-1.0, 0, 0], [1.0, 0, 0]], dtype=np.float64)
    radii = np.asarray([0.25, 0.75], dtype=np.float64)
    query = np.asarray([[0.0, 0, 0]], dtype=np.float64)
    observed = _nearest_reference_values(query, reference, radii)
    assert observed.dtype == np.float32
    np.testing.assert_array_equal(observed, np.asarray([0.25], dtype=np.float32))


def test_exact_reverse_and_gpu_tiled_backends_preserve_graph_edges() -> None:
    rng = np.random.default_rng(20260810)
    coords = rng.random((256, 3), dtype=np.float32)
    key = jax.random.PRNGKey(17)
    common = dict(
        subsample_factor=8,
        radius_policy="discrete_physical_coverage",
        coverage_repair_policy="none",
        discrete_graph_chunk_size=64,
    )
    reference = Heat3DGraphBuilder(
        **common, discrete_graph_backend="sparse_kdtree_v1",
    ).build_metadata(coords, key=key)
    reverse = Heat3DGraphBuilder(
        **common, discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    ).build_metadata(coords, key=key)
    tiled = Heat3DGraphBuilder(
        **common, discrete_graph_backend="gpu_tiled_exact_v1",
        reuse_exact_p2r_for_r2p=True,
    ).build_metadata(coords, key=key)
    for observed in (reverse, tiled):
        np.testing.assert_array_equal(observed.p2r_edge_indices, reference.p2r_edge_indices)
        np.testing.assert_array_equal(observed.r2r_edge_indices, reference.r2r_edge_indices)
        np.testing.assert_array_equal(observed.r2r_edge_domains, reference.r2r_edge_domains)
        assert observed.r2p_edge_indices is None
        assert reference.r2p_edge_indices is None
