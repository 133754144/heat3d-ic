"""Static/unit checks for the V7 stable inference runtime."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np

from rigno.heat3d_runtime.equivalence import compare_metadata, compare_named_arrays
from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample
from rigno.heat3d_runtime.features import FeatureTransform
from rigno.heat3d_v1_normalization import normalize_coords
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
            self.assertIn("bind_registered_route", source)
        u_source = (ROOT / "scripts" / "run_heat3d_v7_u_high_n_reference.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('else "U_v2_direct240825"', u_source)

    def test_eu_runtime_contract_names_all_resolution_roles(self) -> None:
        from rigno.heat3d_runtime.u_split import UHighNRuntime, u_v2_asymmetric_metadata

        self.assertTrue(UHighNRuntime)
        self.assertTrue(u_v2_asymmetric_metadata)
        manifest = json.loads(
            (ROOT / "configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["common"]["resolution_schema"]["required_fields"],
            [
                "anchor_context_resolution",
                "encoder_input_resolution",
                "output_query_resolution",
                "reconstruction_resolution",
            ],
        )

    def test_frozen_eu_contract_keeps_resolution_roles_explicit(self) -> None:
        manifest = json.loads(
            (ROOT / "configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["status"], "frozen_from_v6_p1i_evidence")
        self.assertFalse(manifest["common"]["validity"]["u_is_reconstruction_only"])
        expected = {
            "E16384_reconstruction": (1024, 16384, 16384, 240825),
            "E32768_direct_compatibility": (1024, 32768, 32768, 240825),
            "U_v2_16384_reconstruction": (1024, 1024, 16384, 240825),
        }
        for route_name, values in expected.items():
            route = manifest["strategies"][route_name]
            self.assertEqual(route["route_id"], route_name)
            self.assertEqual(
                tuple(route[field] for field in manifest["common"]["resolution_schema"]["required_fields"]),
                values,
            )
            self.assertTrue(route["direct_query"])
            self.assertNotIn("conditioning_resolution", route)
            self.assertNotIn("query_resolution", route)
        self.assertNotEqual(
            manifest["strategies"]["U_v2_16384_reconstruction"]["encoder_input_resolution"],
            manifest["strategies"]["U_v2_16384_reconstruction"]["output_query_resolution"],
        )

    def test_registered_route_binding_is_exact_and_fail_closed(self) -> None:
        from rigno.heat3d_runtime import bind_registered_route, load_registered_route
        from rigno.heat3d_runtime.preflight import SemanticContractError

        contract_path = ROOT / "configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json"
        e_route = load_registered_route(contract_path, "E16384_reconstruction")
        bound = bind_registered_route(
            contract_path=contract_path,
            route_id="E16384_reconstruction",
            requested_strategy="E",
            anchor_context_resolution=1024,
            encoder_input_resolution=16384,
            output_query_resolution=16384,
            reconstruction_resolution=240825,
            fixed_edge_targets=e_route["fixed_edge_targets"],
        )
        self.assertEqual(bound["route_id"], "E16384_reconstruction")

        u_route = load_registered_route(contract_path, "U_v2_16384_reconstruction")
        u_bound = bind_registered_route(
            contract_path=contract_path,
            route_id="U_v2_16384_reconstruction",
            requested_strategy="U-v2",
            anchor_context_resolution=1024,
            encoder_input_resolution=1024,
            output_query_resolution=16384,
            reconstruction_resolution=240825,
            fixed_edge_targets=u_route["fixed_edge_targets"],
        )
        self.assertEqual(u_bound["encoder_input_resolution"], 1024)

        mismatches = (
            {"route_id": "U_v2_16384_reconstruction"},
            {"requested_strategy": "U-v2"},
            {"anchor_context_resolution": 2048},
            {"encoder_input_resolution": 1024},
            {"output_query_resolution": 32768},
            {"reconstruction_resolution": 1024},
            {"fixed_edge_targets": {"p2r_edge_indices": 1}},
        )
        for override in mismatches:
            request = {
                "route_id": "E16384_reconstruction",
                "requested_strategy": "E",
                "anchor_context_resolution": 1024,
                "encoder_input_resolution": 16384,
                "output_query_resolution": 16384,
                "reconstruction_resolution": 240825,
                "fixed_edge_targets": e_route["fixed_edge_targets"],
            }
            request.update(override)
            with self.assertRaises(SemanticContractError):
                bind_registered_route(contract_path=contract_path, **request)

        with self.assertRaises(SemanticContractError):
            load_registered_route(contract_path, "U_v2_32768")
        with self.assertRaises(SemanticContractError):
            bind_registered_route(
                contract_path=contract_path,
                route_id="U_v2_direct240825",
                requested_strategy="U-v2",
                anchor_context_resolution=1024,
                encoder_input_resolution=1024,
                output_query_resolution=32768,
                reconstruction_resolution=240825,
                fixed_edge_targets={
                    "native": {}, "query": {}, "combined_model_input": {}
                },
            )

        with self.assertRaises(SemanticContractError):
            load_registered_route(contract_path, "E32768_direct_compatibility")

    def test_semantic_preflight_is_fail_closed(self) -> None:
        from rigno.heat3d_runtime.preflight import (
            SemanticContractError,
            validate_semantic_contract,
        )

        run_config = {
            "graph_config": {
                "discrete_graph_backend": "sparse_kdtree_v1",
                "coverage_repair_policy": "frozen",
                "discrete_coverage_multiplier": 1,
                "discrete_graph_chunk_size": 1,
                "min_physical_coverage": 1,
                "node_coordinate_encoding": "raw",
                "node_coordinate_freqs": [],
                "radius_policy": "frozen",
                "repair_p2r": False,
                "repair_r2p": False,
            },
            "graph_seed": 7,
            "global_context": {
                "standardizer": {"feature_names": ["x"]},
                "target_or_label_derived_inputs": False,
            },
            "scale_context": {"target_or_label_derived_inputs": False},
            "input_feature_schema": "legacy_bc_flags",
            "coord_policy": "sample_local_isotropic",
            "extent_feature_policy": "log_extent_broadcast",
            "normalization_profile": "semantic_normalization_v1",
            "condition_feature_transform": "semantic_v1",
        }
        model_config = {
            key: "none"
            for key in (
                "qk_region_feature_version",
                "decoder_bypass_mode",
                "decoder_bypass_features",
                "decoder_bypass_feature_source",
                "decoder_bypass_output_space",
                "native_output_mode",
                "global_context_mode",
                "scale_context_mode",
                "scale_pooling",
                "shape_attention_mode",
                "scale_attention_mode",
                "scale_deepsets_mode",
            )
        }
        model_config.update({"decoder_bypass_num_features": 0, "decoder_bypass_local_feature_names": []})
        model_config.update({"global_context_feature_dim": 0})
        stats = {
            "input_feature_schema": "legacy_bc_flags",
            "coord_policy": "sample_local_isotropic",
            "extent_feature_policy": "log_extent_broadcast",
            "normalization_profile": "semantic_normalization_v1",
            "feature_names": ["x"],
            "condition_feature_transforms": ["identity"],
            "condition_mean": [0.0],
            "condition_std": [1.0],
            "coord_min": [0.0],
            "coord_span": [1.0],
        }
        route = {
            "route_id": "U_v2_16384_reconstruction",
            "strategy_name": "U-v2",
            "anchor_context_resolution": 1024,
            "encoder_input_resolution": 1024,
            "output_query_resolution": 16384,
            "direct_query": True,
            "reconstruction_resolution": 240825,
            "fixed_edge_targets": {"native": {"p2r_edge_indices": 1}},
        }
        broken = dict(route)
        del broken["output_query_resolution"]
        with self.assertRaises(SemanticContractError):
            validate_semantic_contract(
                run_config=run_config,
                model_config=model_config,
                stats=stats,
                execution_role="production_inference",
                route_contract=broken,
            )
        broken_config = dict(run_config)
        del broken_config["graph_seed"]
        with self.assertRaises(SemanticContractError):
            validate_semantic_contract(
                run_config=broken_config,
                model_config=model_config,
                stats=stats,
                execution_role="production_inference",
                route_contract=route,
            )
        for mapping, field in (
            (model_config, "qk_region_feature_version"),
            (stats, "normalization_profile"),
        ):
            broken_mapping = dict(mapping)
            del broken_mapping[field]
            with self.assertRaises(SemanticContractError):
                validate_semantic_contract(
                    run_config=run_config,
                    model_config=broken_mapping if mapping is model_config else model_config,
                    stats=broken_mapping if mapping is stats else stats,
                    execution_role="production_inference",
                    route_contract=route,
                )
        broken_route = dict(route)
        broken_route["conditioning_resolution"] = 1024
        with self.assertRaises(SemanticContractError):
            validate_semantic_contract(
                run_config=run_config,
                model_config=model_config,
                stats=stats,
                execution_role="production_inference",
                route_contract=broken_route,
            )

    def test_e_high_n_graph_policy_does_not_collapse_query_resolution(self) -> None:
        from rigno.heat3d_runtime.high_n import HighNRuntime

        runtime = object.__new__(HighNRuntime)
        runtime.graph_config = {"subsample_factor": 4, "reuse_exact_p2r_for_r2p": False}
        native = runtime.graph_config_for_resolution(1024)
        high_n = runtime.graph_config_for_resolution(16384)
        self.assertEqual(native["subsample_factor"], 4)
        self.assertEqual(high_n["subsample_factor"], 64.0)
        self.assertTrue(high_n["reuse_exact_p2r_for_r2p"])

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

    def test_graph_coordinate_normalization_preserves_frozen_float64_order(self) -> None:
        coords = np.asarray(
            [[0.123456789012345, 0.23456789012345, 0.3456789012345],
             [0.523456789012345, 0.53456789012345, 0.5456789012345],
             [0.923456789012345, 0.83456789012345, 0.7456789012345]],
            dtype=np.float64,
        )
        condition = np.zeros((3, 11), dtype=np.float64)
        condition[:, 0:3] = 1.0
        condition[:, 3] = 2.0
        condition[:, 7] = 1.0
        example = V6DualRobinExample(
            sample_id="graph-coordinate-unit",
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
            target=V1SteadyTarget(target_u=np.full((3, 1), 300.0)),
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
            operator_point_weights=np.ones(3),
        )
        stats = {
            "input_feature_schema": "legacy_bc_flags",
            "coord_policy": "sample_local_isotropic",
            "extent_feature_policy": "none",
            "normalization_profile": "legacy_zscore",
            "feature_names": tuple(example.condition.condition_feature_names),
            "condition_mean": np.zeros((1, 1, 1, 11), dtype=np.float32),
            "condition_std": np.ones((1, 1, 1, 11), dtype=np.float32),
            "coord_min": np.zeros(3, dtype=np.float64),
            "coord_span": np.ones(3, dtype=np.float64),
        }
        observed = FeatureTransform(stats).transform(example)
        expected_graph = normalize_coords(coords.reshape(1, 1, 3, 3), stats)
        np.testing.assert_array_equal(
            np.asarray(observed.graph_coords), np.asarray(expected_graph).reshape(3, 3)
        )

    def test_evaluation_core_keeps_v6_sufficient_statistics(self) -> None:
        coords = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        kwargs = {
            "truth_deltaT_K": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "control_volumes_m3": np.asarray([1.0, 2.0, 1.0, 2.0]),
            "coords": coords,
            "layer_id": np.asarray([0, 0, 1, 1]),
            "q_W_m3": np.asarray([1.0, 0.0, 1.0, 0.0]),
        }
        samples = [
            EvaluationSample(
                sample_id="valid-a",
                prediction_deltaT_K=np.asarray([1.5, 1.5, 3.5, 3.5]),
                **kwargs,
            ),
            EvaluationSample(
                sample_id="valid-b",
                prediction_deltaT_K=np.asarray([0.5, 2.5, 2.5, 4.5]),
                **kwargs,
            ),
        ]
        result = EvaluationCore().evaluate(samples)
        stats = result["sufficient_statistics"]
        expected_errors = [
            np.asarray([0.5, -0.5, 0.5, -0.5]),
            np.asarray([-0.5, 0.5, -0.5, 0.5]),
        ]
        self.assertEqual(stats["point_count"], 8)
        self.assertEqual(stats["sample_cv_relative_rmse_count"], 2)
        self.assertEqual(stats["interface_error_count"], 2)
        self.assertAlmostEqual(
            stats["point_sse"], sum(float(np.sum(error * error)) for error in expected_errors)
        )
        self.assertAlmostEqual(
            result["metrics"]["point_global_relative_rmse_pct"],
            np.sqrt(stats["point_sse"] / stats["point_truth_energy"]) * 100.0,
        )
        self.assertAlmostEqual(
            result["metrics"]["interface_RMSE_K"], 0.0, places=12
        )

    def test_evaluation_core_rejects_non_valid_truth(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationCore().evaluate(
                [
                    {
                        "sample_id": "test-only",
                        "split": "test_iid",
                        "prediction_deltaT_K": [1.0],
                        "truth_deltaT_K": [1.0],
                        "control_volumes_m3": [1.0],
                        "coords": [[0.0, 0.0, 0.0]],
                        "layer_id": [0],
                        "q_W_m3": [0.0],
                    }
                ]
            )

    def test_timing_core_freezes_boundary_and_excludes_evaluation(self) -> None:
        from rigno.heat3d_runtime.timing import TimingCore

        lifecycle = TimingCore.begin(state="fresh", query_count=1)
        for stage in TimingCore.contract()["stage_order"]:
            lifecycle.record(stage, 0.0)
        record = lifecycle.complete()
        TimingCore.validate_record(record)
        self.assertFalse(record["excluded_from_latency"] == [])
        self.assertFalse(TimingCore.contract()["truth_loading_in_latency"])
        self.assertFalse(TimingCore.contract()["metrics_in_latency"])
        with self.assertRaises(ValueError):
            TimingCore.begin(state="fresh").record("metrics", 0.0)

    def test_timing_core_runs_real_stage_callbacks_without_changing_boundary(self) -> None:
        from rigno.heat3d_runtime.timing import TIMING_STAGE_ORDER, TimingCore

        seen = []
        steps = {
            stage: (lambda prior, stage=stage: seen.append((stage, tuple(prior))) or stage)
            for stage in TIMING_STAGE_ORDER
        }
        record, results = TimingCore.run(state="fresh", steps=steps)
        self.assertEqual([row["stage"] for row in record["stages"]], list(TIMING_STAGE_ORDER))
        self.assertEqual(list(results), list(TIMING_STAGE_ORDER))
        self.assertEqual([row[0] for row in seen], list(TIMING_STAGE_ORDER))
        self.assertTrue(all(row["elapsed_seconds"] >= 0.0 for row in record["stages"]))
        self.assertEqual(record["latency_boundary"], TimingCore.contract()["workload_boundary"])

    def test_formal_orchestration_binds_route_and_loads_truth_after_prediction(self) -> None:
        from rigno.heat3d_runtime import FormalEvaluationOrchestrator, PredictionOnlyRecord
        from rigno.heat3d_runtime.evaluation import METRIC_SCHEMA_VERSION

        coords = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        prediction = np.asarray([1.5, 1.5, 3.5, 3.5])
        record = PredictionOnlyRecord(
            sample_id="valid-formal",
            prediction_deltaT_K=prediction,
            coords=coords,
            control_volumes_m3=np.ones(4),
            layer_id=np.asarray([0, 0, 1, 1]),
            q_W_m3=np.asarray([1.0, 0.0, 1.0, 0.0]),
            route_id="native_1024",
            anchor_context_resolution=1024,
            encoder_input_resolution=1024,
            output_query_resolution=1024,
            reconstruction_resolution=1024,
            direct_query=True,
        ).validated()
        truth = np.asarray([1.0, 2.0, 3.0, 4.0])
        loaded = []
        receipt = {
            "route_id": "native_1024",
            "anchor_context_resolution": 1024,
            "encoder_input_resolution": 1024,
            "output_query_resolution": 1024,
            "reconstruction_resolution": 1024,
            "prediction_artifact_sha256_by_sample": {
                "valid-formal": record.prediction_artifact_sha256,
            },
            "reconstruction_contract_sha256": None,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
        }

        def truth_loader(sample_id: str) -> dict[str, object]:
            loaded.append(sample_id)
            return {"truth_deltaT_K": truth, "split": "valid_iid"}

        result = FormalEvaluationOrchestrator().run(
            [record], truth_loader=truth_loader, receipt=receipt
        )
        self.assertEqual(loaded, ["valid-formal"])
        self.assertTrue(result["prediction_only"])
        self.assertFalse(result["labels_read_by_inference"])
        self.assertEqual(result["truth_loaded_by"], "EvaluationCore")
        self.assertEqual(result["evaluation"]["metric_schema_version"], METRIC_SCHEMA_VERSION)
        with self.assertRaises(ValueError):
            FormalEvaluationOrchestrator().run(
                [record],
                truth_loader=truth_loader,
                receipt={"route_id": "native_1024"},
            )


if __name__ == "__main__":
    unittest.main()
