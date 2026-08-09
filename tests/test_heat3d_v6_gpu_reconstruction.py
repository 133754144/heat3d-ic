from __future__ import annotations

import numpy as np

from rigno.heat3d_v6_full_field import ReconstructionMap
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map


def _fixture() -> ReconstructionMap:
    return ReconstructionMap(
        support_indices=np.asarray([3, 5, 8], dtype=np.int32),
        neighbor_local_indices=np.asarray(
            [[0, 0], [0, 1], [1, 2], [2, 2]], dtype=np.int32
        ),
        neighbor_weights=np.asarray(
            [[1.0, 0.0], [0.25, 0.75], [0.4, 0.6], [0.5, 0.5]],
            dtype=np.float64,
        ),
        domain_code=np.asarray([0, 0, 1, 1], dtype=np.int16),
        domain_names=("layer_00", "layer_01"),
    )


def test_device_reconstruction_matches_frozen_cpu_apply() -> None:
    mapping = _fixture()
    values = np.asarray([2.0, 5.0, 11.0], dtype=np.float32)
    expected = mapping.reconstruct(values)
    observed = np.asarray(to_device_reconstruction_map(mapping).reconstruct(values))
    np.testing.assert_allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)


def test_device_reconstruction_rejects_wrong_support_shape() -> None:
    mapping = to_device_reconstruction_map(_fixture())
    try:
        mapping.reconstruct(np.asarray([1.0, 2.0], dtype=np.float32))
    except ValueError as error:
        assert "support value shape" in str(error)
    else:
        raise AssertionError("wrong support shape was accepted")
