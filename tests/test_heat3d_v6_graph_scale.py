from __future__ import annotations

import numpy as np

from rigno.heat3d_v6_graph_scale import _nearest_reference_values


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
