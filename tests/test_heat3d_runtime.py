"""Static/unit checks for the V7 stable inference runtime."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from rigno.heat3d_runtime.equivalence import compare_metadata, compare_named_arrays
from rigno.heat3d_runtime.features import FeatureTransform
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v1_training_semantics import build_configured_zero_delta_bridge
from rigno.heat3d_v6_dataset import V6DualRobinExample


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "rigno" / "heat3d_runtime"


class StableRuntimeStaticTests(unittest.TestCase):
    def test_runtime_has_no_script_or_runtime_mutation_imports(self) -> None:
        for path in sorted(RUNTIME_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    self.assertFalse(module == "scripts" or module.startswith("scripts."))
                    self.assertNotIn("smoke", module.lower())
                    self.assertNotIn("development", module.lower())
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("sys.path", source)
            self.assertNotIn("runner_module", source)

    def test_high_n_reference_entrypoint_uses_stable_runtime_only(self) -> None:
        for filename in ("run_heat3d_v7_high_n_reference.py", "run_heat3d_v7_u_high_n_reference.py"):
            path = ROOT / "scripts" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            self.assertIn("rigno.heat3d_runtime", imported)
            self.assertFalse(any(module.startswith("scripts") for module in imported))
            self.assertFalse(any("smoke" in module.lower() for module in imported))
            self.assertFalse(any("development" in module.lower() for module in imported))
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("sys.path", source)
            self.assertNotIn("install_checkpoint_feature_hooks", source)

    def test_u_runtime_exposes_separate_conditioning_and_query_resolutions(self) -> None:
        from rigno.heat3d_runtime.u_split import UHighNRuntime, u_v2_asymmetric_metadata

        self.assertTrue(UHighNRuntime)
        self.assertTrue(u_v2_asymmetric_metadata)
        source = (RUNTIME_ROOT / "u_split.py").read_text(encoding="utf-8")
        self.assertIn("conditioning_resolution", source)
        self.assertIn("query_resolution", source)
        self.assertIn('"direct_query": True', source)
        self.assertNotIn("reconstruction_only", source)

    def test_equivalence_reports_actual_errors_without_widening(self) -> None:
        report = compare_named_arrays(
            {"normalized_c": np.asarray([1.0, 2.0])},
            {"normalized_c": np.asarray([1.0, 2.0 + 1.0e-7])},
        )
        self.assertFalse(report.passed)
        row = report.comparisons[0]
        self.assertAlmostEqual(row.max_abs or 0.0, 1.0e-7, places=12)
        self.assertAlmostEqual(row.rmse or 0.0, 1.0e-7 / np.sqrt(2.0), places=12)

    def test_prediction_tolerance_is_explicit_and_strict(self) -> None:
        report = compare_named_arrays(
            {"prediction": np.asarray([1.0, 2.0])},
            {"prediction": np.asarray([1.0, 2.0 + 5.0e-7])},
            per_name_tolerance={"prediction": (1.0e-6, 1.0e-6)},
        )
        self.assertTrue(report.passed)

    def test_metadata_comparison_normalizes_json_sequence_types(self) -> None:
        self.assertTrue(compare_metadata({"features": ("k_x", "q")}, {"features": ["k_x", "q"]})["passed"])

    def test_explicit_transform_matches_frozen_zero_delta_inputs(self) -> None:
        coords = np.asarray(
            [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
            dtype=np.float64,
        )
        flags = np.zeros((8, 4), dtype=np.float64)
        flags[:, 3] = 1.0
        flags[[1, 3, 5, 7], 0] = 1.0
        flags[[1, 3, 5, 7], 3] = 0.0
        condition = np.column_stack(
            [
                np.full(8, 10.0),
                np.full(8, 11.0),
                np.full(8, 12.0),
                np.full(8, 2.0),
                flags,
                np.full(8, 100.0),
                np.full(8, 200.0),
                np.full(8, 0.0),
            ]
        )
        example = V6DualRobinExample(
            sample_id="unit",
            condition=V1SteadyConditionInput(
                coords=coords,
                condition_features=condition,
                condition_feature_names=(
                    "k_x", "k_y", "k_z", "q", "is_top", "is_bottom",
                    "is_side", "is_interior", "top_h", "bottom_h",
                    "top_T_inf_minus_T_ref",
                ),
                k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(target_u=np.full(8, 300.0)),
            meta={
                "v6_adapter": {
                    "reference_temperature_K": 300.0,
                    "top_T_inf_K": 300.0,
                    "bottom_T_inf_K": 300.0,
                },
                "package_total_power_W": 1.0,
                "physics": {"footprint_m": (1.0, 1.0)},
                "layers_bottom_to_top": [{"thickness_m": 1.0}],
            },
            operator_point_weights=np.ones(8),
        )
        stats = {
            "input_feature_schema": "legacy_bc_flags",
            "coord_policy": "train_minmax_to_unit_box",
            "extent_feature_policy": "none",
            "normalization_profile": "legacy_zscore",
            "feature_names": tuple(example.condition.condition_feature_names),
            "condition_mean": np.zeros((1, 1, 1, 11), dtype=np.float32),
            "condition_std": np.ones((1, 1, 1, 11), dtype=np.float32),
            "coord_min": np.zeros((1, 1, 1, 3), dtype=np.float32),
            "coord_span": np.ones((1, 1, 1, 3), dtype=np.float32),
        }
        expected = build_configured_zero_delta_bridge(example)
        observed = FeatureTransform(stats).transform(example)
        self.assertTrue(np.array_equal(np.asarray(expected.legacy_inputs.u), np.asarray(observed.inputs.u)))
        self.assertTrue(np.array_equal(np.asarray(expected.legacy_inputs.c), np.asarray(observed.inputs.c)))
        expected_coords = 2.0 * np.asarray(expected.legacy_inputs.x_inp) - 1.0
        self.assertTrue(np.array_equal(expected_coords, np.asarray(observed.inputs.x_inp)))


if __name__ == "__main__":
    unittest.main()
