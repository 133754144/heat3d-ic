"""Unit tests for the G2-A input boundary without optional upstream packages."""

from __future__ import annotations

import unittest

import numpy as np

from rigno.heat3d_g2.inputs import P1IInputBatch, P1I_FEATURE_NAMES, unit_cube_latent_queries
from rigno.heat3d_g2.p1i import evaluate_valid_prediction


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

    def test_level_a_bridge_uses_explicit_prediction(self) -> None:
        from rigno.heat3d_runtime.evaluation import EvaluationSample

        truth = np.ones(8, dtype=np.float64)
        sample = EvaluationSample(
            sample_id="valid_bridge",
            prediction_deltaT_K=np.zeros(8),
            truth_deltaT_K=truth,
            control_volumes_m3=np.ones(8),
            coords=np.zeros((8, 3)),
            layer_id=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
            q_W_m3=np.ones(8),
        )
        result = evaluate_valid_prediction(sample, np.ones((1, 8, 1)))
        self.assertEqual(result["metric_schema_version"], "heat3d_v7_evaluation_core_v1_v6_definitions")
        self.assertAlmostEqual(result["metrics"]["raw_K_CV_RMSE_K"], 0.0)


if __name__ == "__main__":
    unittest.main()
