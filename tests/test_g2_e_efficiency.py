from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rigno.heat3d_training import (
    ManualGradientDescent,
    TrainingBatch,
    TrainingDependencies,
    V7FormalTrainer,
    atomic_latest_checkpoint,
    load_latest_checkpoint,
    tree_max_abs_difference,
)


ROOT = Path(__file__).resolve().parents[1]


def _deps():
    batch = TrainingBatch(
        batch_id="slim_equivalence",
        sample_ids=("x",),
        groups=({"x": jnp.asarray([1.0, 2.0]), "y": jnp.asarray([0.0, 1.0])},),
    )

    def apply(params, current, _rng):
        return params["w"] * current.groups[0]["x"] + params["b"]

    def loss(prediction, current):
        return jnp.mean((prediction - current.groups[0]["y"]) ** 2)

    deps = TrainingDependencies(
        data_source="fixture",
        feature_transform="identity",
        normalization="none",
        graph_builder="none",
        model="fixture",
        model_apply=apply,
        loss_fn=loss,
        optimizer=ManualGradientDescent(0.01),
        batch_iterator=lambda value: value,
        validation_fn=lambda params, current: loss(apply(params, current, None), current),
        checkpoint_writer=lambda _path, _payload: None,
        metrics_fn=lambda _params, _current: {},
    )
    return V7FormalTrainer(deps, jit_cache=True), batch


def test_slim_step_matches_reference_update() -> None:
    reference, batch = _deps()
    slim, _ = _deps()
    params = {"w": jnp.asarray(0.75), "b": jnp.asarray(-0.1)}
    ref_state = reference.initialize(params)
    slim_state = slim.initialize(params)
    key = jax.random.PRNGKey(17)
    ref = reference.step(ref_state, batch, key)
    compact = slim.step_slim(slim_state, batch, key)
    assert tree_max_abs_difference(ref.state.params, compact.state.params) == 0.0
    assert tree_max_abs_difference(ref.state.optimizer_state, compact.state.optimizer_state) == 0.0
    assert np.asarray(ref.loss).tobytes() == np.asarray(compact.loss).tobytes()
    assert compact.state.step == ref.state.step == 1


def test_atomic_resume_roundtrip_preserves_required_state(tmp_path: Path) -> None:
    trainer, batch = _deps()
    state = trainer.initialize({"w": jnp.asarray(1.0), "b": jnp.asarray(0.0)})
    state = trainer.step_slim(state, batch, jax.random.PRNGKey(1)).state
    path = tmp_path / "latest.pkl"
    receipt = atomic_latest_checkpoint(
        path,
        state=state,
        metadata={
            "epoch": 1,
            "global_update_count": 1,
            "best_metric": 1.0,
            "best_epoch": 1,
            "runner_sha": "runner",
            "config_sha": "config",
            "data_sha": "data",
        },
        rng_state={"key": 1},
        batch_state={"order": [0]},
    )
    loaded = load_latest_checkpoint(path)
    assert receipt["atomic_replace"] is True
    assert loaded["schema_version"] == "g2_exact_resume_checkpoint_v1"
    assert loaded["state_object"].step == 1
    assert tree_max_abs_difference(loaded["state_object"].params, state.params) == 0.0


def test_exact_resume_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_v7_g2_exact_resume.py", "--steps", "6", "--split", "2"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXACT_RESUME_EQUIVALENCE_PASS" in result.stdout
