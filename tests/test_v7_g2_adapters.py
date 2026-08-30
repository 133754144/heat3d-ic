"""Unit tests for the G2-A input boundary without optional upstream packages."""

from __future__ import annotations

import unittest

import numpy as np

from rigno.heat3d_g2.inputs import P1IInputBatch, P1I_FEATURE_NAMES, unit_cube_latent_queries


class G2InputTests(unittest.TestCase):
    def _batch(self, split: str = "train") -> P1IInputBatch:
        coords = np.zeros((1, 8, 3), dtype=np.float32)
        features = np.ones((1, 8, len(P1I_FEATURE_NAMES)), dtype=np.float32)
        return P1IInputBatch.from_arrays(
            sample_ids=("synthetic_g2_0000",),
            coords=coords,
            features=features,
            split=split,
        )

    def test_input_schema_is_label_free(self) -> None:
        batch = self._batch()
        self.assertEqual(batch.point_count, 8)
        self.assertEqual(batch.feature_names, P1I_FEATURE_NAMES)
        self.assertEqual(batch.dataset_id, "synthetic_g2_native_smoke")

    def test_forbidden_split_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self._batch("test_iid")

    def test_latent_grid_is_explicit(self) -> None:
        grid = unit_cube_latent_queries(3)
        self.assertEqual(grid.shape, (1, 3, 3, 3, 3))
        self.assertEqual(float(grid.min()), 0.0)
        self.assertEqual(float(grid.max()), 1.0)

    def test_v6_example_adapter_does_not_require_target(self) -> None:
        class Condition:
            coords = np.zeros((8, 3), dtype=np.float32)
            condition_features = np.ones((8, len(P1I_FEATURE_NAMES)), dtype=np.float32)
            condition_feature_names = P1I_FEATURE_NAMES

        class Example:
            sample_id = "train_without_target_attribute"
            condition = Condition()
            meta = {"v6_adapter": {"manifest_split_role": "train"}}

        batch = P1IInputBatch.from_v6_examples([Example()])
        self.assertEqual(batch.sample_ids, ("train_without_target_attribute",))


if __name__ == "__main__":
    unittest.main()
