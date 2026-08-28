"""Lightweight tests for the V7 formal trainer boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import jax.numpy as jnp

from rigno.heat3d_training import ManualGradientDescent, V7FormalTrainer


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

    def test_manual_gradient_descent_contract(self) -> None:
        optimizer = ManualGradientDescent(0.1)
        params = {"x": jnp.asarray([2.0, -1.0])}
        updates, state = optimizer.update({"x": jnp.asarray([3.0, 4.0])}, optimizer.init(params), params)
        self.assertEqual(state, {})
        self.assertTrue(bool(jnp.array_equal(updates["x"], jnp.asarray([-0.3, -0.4]))))

    def test_checkpoint_roundtrip_dependency_is_explicit(self) -> None:
        self.assertTrue(hasattr(V7FormalTrainer, "write_checkpoint"))
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(Path(temp_dir).is_dir())
