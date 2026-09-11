#!/usr/bin/env python3
"""One frozen Heat3D batch equivalence gate for the slim execution path.

The fixture uses one allowed ``train`` case only.  It compares the legacy
diagnostic-returning update with ``step_slim`` from identical initial params,
batch and RNG key.  It never opens valid/test/sealed labels and does not
produce an accuracy metric.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(token in str(value).lower() for value in (args.fs_train, args.labels_root) for token in ("test", "sealed")):
        raise ValueError("test/sealed paths are forbidden")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: Heat3D execution equivalence requires CUDA")

    profile = load_script("profile_v7_g2_training_efficiency.py")
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    stats = load_script("run_v7_g2_p6_heat3d_v1_formal.py").load_stats(args.normalization)
    source = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="train"
    )

    class FirstRow:
        def __len__(self):
            return 24

        def __getitem__(self, index):
            if not 0 <= index < 24:
                raise IndexError(index)
            return source[index]

    dataset = FirstRow()
    mesh = support.mesh_arrays()
    examples = profile._build_examples(dataset, "train", mesh["coords"], mesh["control_volume"], mesh["layer_id"])
    from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
    from rigno.heat3d_training import (
        TrainingDependencies,
        V7FormalTrainer,
        block_until_ready,
        build_p1i_batches,
        loss_fn_full,
        make_gradient_transform,
        make_p1i_optimizer,
        model_apply_full,
        model_init_full,
        tree_max_abs_difference,
    )
    from rigno.heat3d_training.p1i import (
        attach_input_contexts,
        attach_native_physics,
        attach_qk_features,
        fit_native_loss_references,
    )
    batches = build_p1i_batches(examples, stats, Heat3DGraphBuilder(**config["graph"]), label="g2_e_equivalence", batch_size=24, graph_seed=0)
    context = attach_input_contexts(batches, examples, examples, config["model"])
    by_id = {row.sample_id: row for row in examples}
    attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
    attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(examples, config["loss"]))
    model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
    from rigno.models.rigno import RIGNO
    model = RIGNO(**model_config)
    params = model_init_full(model, jax.random.PRNGKey(0), batches[0])["params"]
    apply_fn = lambda current, batch, rng: model_apply_full(model, current, batch, rng)
    batch_loss = lambda prediction, batch: loss_fn_full(prediction, batch, loss_config)

    def make_trainer():
        deps = TrainingDependencies(
            data_source="one_frozen_train_case",
            feature_transform="physics_layout_aware_1024",
            normalization=stats,
            graph_builder="frozen",
            model=model,
            model_apply=apply_fn,
            loss_fn=batch_loss,
            optimizer=make_p1i_optimizer(config["optimizer"], epochs=1, updates_per_epoch=1),
            batch_iterator=lambda value: value,
            validation_fn=lambda current, batch: batch_loss(apply_fn(current, batch, None), batch),
            checkpoint_writer=lambda _path, _payload: None,
            metrics_fn=lambda _current, _batch: {},
            gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
        )
        return V7FormalTrainer(deps, jit_cache=True)

    legacy_trainer = make_trainer()
    slim_trainer = make_trainer()
    legacy = legacy_trainer.step(legacy_trainer.initialize(params), batches[0], jax.random.PRNGKey(901))
    slim = slim_trainer.step_slim(slim_trainer.initialize(params), batches[0], jax.random.PRNGKey(901))
    block_until_ready((legacy.state.params, legacy.state.optimizer_state, legacy.loss))
    block_until_ready((slim.state.params, slim.state.optimizer_state, slim.loss))
    params_diff = tree_max_abs_difference(legacy.state.params, slim.state.params)
    optimizer_diff = tree_max_abs_difference(legacy.state.optimizer_state, slim.state.optimizer_state)
    loss_equal = np.asarray(legacy.loss).tobytes() == np.asarray(slim.loss).tobytes()
    payload = {
        "schema_version": "heat3d_v7_g2_e_execution_equivalence_v1",
        "status": "SCIENCE_NEUTRAL_EXECUTION_EQUIVALENCE_PASS" if params_diff == 0.0 and optimizer_diff == 0.0 and loss_equal else "SCIENCE_NEUTRAL_EXECUTION_EQUIVALENCE_FAIL_CLOSED",
        "fixture": {"role": "train", "count": 24, "sample_ids": [row.sample_id for row in examples], "batch_size": 24, "seed": 0, "rng_key": 901},
        "legacy_compile_count": legacy_trainer.compile_count,
        "slim_compile_count": len(getattr(slim_trainer, "_compiled_slim_steps", {})),
        "loss_bitwise_equal": loss_equal,
        "params_max_abs_difference": params_diff,
        "optimizer_state_max_abs_difference": optimizer_diff,
        "update_count_equal": legacy.state.step == slim.state.step,
        "test_or_sealed_access": False,
        "accuracy_used": False,
        "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
