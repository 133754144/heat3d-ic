"""Lightweight tests for the V7 formal trainer boundary."""

from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax.numpy as jnp

from rigno.heat3d_training import (
    learning_rate_for_epoch,
    ManualGradientDescent,
    TrainingDependencies,
    V7FormalTrainer,
    model_apply_vanilla,
    model_init_vanilla,
    loss_fn_vanilla,
)


ROOT = Path(__file__).resolve().parents[1]


class V7TrainingStaticTests(unittest.TestCase):
    def test_production_entrypoint_has_no_legacy_import(self) -> None:
        path = ROOT / "scripts" / "run_heat3d_v7_formal_training.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(module.startswith("scripts") for module in imported))
        self.assertFalse(any("smoke" in module.lower() for module in imported))
        self.assertFalse(any("development" in module.lower() for module in imported))
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("monkey_patch", source)

    def test_p1i_production_entrypoint_has_no_legacy_import(self) -> None:
        path = ROOT / "scripts" / "run_heat3d_v7_formal_p1i_training.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(module.startswith("scripts") for module in imported))
        self.assertFalse(any("smoke" in module.lower() for module in imported))
        self.assertFalse(any("development" in module.lower() for module in imported))
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("monkey_patch", source)

    def test_p1i_registered_batching_contract(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json").read_text(
                encoding="utf-8"
            )
        )
        batching = config["batching"]
        self.assertEqual(batching["batch_size"], 24)
        self.assertEqual(batching["micro_batch_size"], 24)
        self.assertEqual(batching["validation_batch_size"], 32)
        self.assertEqual(batching["train_samples_per_epoch"] // batching["batch_size"], 32)
        self.assertTrue(batching["shuffle_train_batches"])
        self.assertFalse(batching["drop_last"])
        self.assertEqual(config["dataset"]["roles"]["train"], 768)
        self.assertEqual(config["dataset"]["roles"]["valid_iid"], 128)
        self.assertEqual(config["dataset"]["label_access"]["test_iid"], "forbidden")
        self.assertEqual(config["dataset"]["label_access"]["sealed"], "forbidden")

    def test_e200_budget_contract_is_explicit_and_not_formal_g1(self) -> None:
        budget = json.loads(
            (ROOT / "configs" / "heat3d_v7" / "v7_g1_budget_qualification.json").read_text(
                encoding="utf-8"
            )
        )
        contract = json.loads(
            (ROOT / "configs" / "heat3d_v7" / "v7_g1_epoch_budget_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(budget["status"], "completed_nonpublication_qualification")
        self.assertEqual(contract["proposed_budget"]["epochs"], 200)
        self.assertEqual(contract["proposed_budget"]["cosine_horizon_epochs"], 200)
        self.assertEqual(contract["proposed_budget"]["warmup_epochs"], 10)
        self.assertEqual(
            {row["variant"] for row in budget["qualification_runs"]},
            {"Full", "vanilla_RIGNO"},
        )
        self.assertTrue(all(not row["g1_formal"] for row in budget["qualification_runs"]))
        self.assertTrue(all(row["execution_started"] for row in budget["qualification_runs"]))
        self.assertEqual(contract["decision_values"]["G1_epoch_budget"], 200)
        self.assertEqual(contract["qualification"]["qualification_decision"], "PASS_e200")

    def test_g1_matrix_is_frozen_but_not_started(self) -> None:
        registry = json.loads(
            (ROOT / "configs/heat3d_v7/v7_experiment_registry.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(registry["base_formal_variant_count"], 6)
        self.assertEqual(registry["formal_variant_count"], 7)
        self.assertEqual(registry["formal_matrix"]["run_count"], 21)
        self.assertFalse(registry["formal_matrix"]["formal_execution_started"])

    def test_capacity_fairness_observation_is_registered(self) -> None:
        fairness = json.loads(
            (ROOT / "configs/heat3d_v7/v7_parameter_fairness_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(fairness["observed"]["capacity_matched_triggered"])
        self.assertAlmostEqual(fairness["observed"]["relative_gap"], 0.07448564925580436)

    def test_all_registered_variants_resolve_in_dry_run_only(self) -> None:
        registry = json.loads(
            (ROOT / "configs/heat3d_v7/v7_experiment_registry.json").read_text(
                encoding="utf-8"
            )
        )
        variant_ids = [
            row["experiment_id"]
            for row in registry["registered_runs"]
            if row["experiment_id"].startswith("V7-G1-Full-P1i")
        ]
        self.assertEqual(len(variant_ids), 7)
        for experiment_id in variant_ids:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_heat3d_v7_formal_p1i_training",
                    "--experiment-id",
                    experiment_id,
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["experiment_id"], experiment_id)
            self.assertEqual(payload["training_runs"], 0)

    def test_synchronized_seed_bundle_is_pre_registered_only(self) -> None:
        bundle = json.loads(
            (ROOT / "configs/heat3d_v7/v7_g1_seed_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(bundle["seed_set"], [0, 1, 2])
        self.assertFalse(bundle["formal_execution_guard"]["multi_seed_started"])
        self.assertEqual(
            bundle["synchronization"]["batch_build_seed"]["value"], 0
        )

    def test_e200_schedule_reaches_registered_minimum(self) -> None:
        config = json.loads(
            (ROOT / "configs/heat3d_v7/v7_g1_full_p1i.json").read_text(
                encoding="utf-8"
            )
        )
        optimizer = config["optimizer"]
        start = learning_rate_for_epoch(
            1, epochs=200, updates_per_epoch=32, config=optimizer
        )
        end = learning_rate_for_epoch(
            200, epochs=200, updates_per_epoch=32, config=optimizer
        )
        self.assertAlmostEqual(start, 9.5e-5, places=10)
        self.assertAlmostEqual(end, optimizer["min_lr"], places=10)
        self.assertNotEqual(start, end)

    def test_vanilla_training_api_is_exported(self) -> None:
        self.assertTrue(callable(model_apply_vanilla))
        self.assertTrue(callable(model_init_vanilla))
        self.assertTrue(callable(loss_fn_vanilla))

    def test_manual_gradient_descent_contract(self) -> None:
        optimizer = ManualGradientDescent(0.1)
        params = {"x": jnp.asarray([2.0, -1.0])}
        updates, state = optimizer.update({"x": jnp.asarray([3.0, 4.0])}, optimizer.init(params), params)
        self.assertEqual(state, {})
        self.assertTrue(bool(jnp.array_equal(updates["x"], jnp.asarray([-0.3, -0.4]))))

    def test_checkpoint_roundtrip_dependency_is_explicit(self) -> None:
        self.assertTrue(hasattr(V7FormalTrainer, "write_checkpoint"))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "roundtrip.pkl"

            def writer(path: Path, payload: dict) -> None:
                with path.open("wb") as stream:
                    pickle.dump(payload, stream)

            dependencies = TrainingDependencies(
                data_source=None,
                feature_transform=None,
                normalization=None,
                graph_builder=None,
                model=None,
                model_apply=lambda params, batch, rng: None,
                loss_fn=lambda prediction, batch: None,
                optimizer=ManualGradientDescent(0.1),
                batch_iterator=lambda batches: batches,
                validation_fn=lambda params, batch: None,
                checkpoint_writer=writer,
                metrics_fn=lambda params, batch: {},
            )
            trainer = V7FormalTrainer(dependencies)
            state = trainer.initialize({"x": jnp.asarray([2.0, -1.0])})
            trainer.write_checkpoint(checkpoint_path, state, {"schema": "ci-test"})

            with checkpoint_path.open("rb") as stream:
                loaded = pickle.load(stream)
            self.assertEqual(loaded["schema"], "ci-test")
            self.assertEqual(loaded["step"], state.step)
            self.assertTrue(bool(jnp.array_equal(loaded["params"]["x"], state.params["x"])))
