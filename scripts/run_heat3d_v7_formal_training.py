#!/usr/bin/env python3
"""Single V7 formal-training entrypoint for readiness fixtures.

This command is intentionally small: the numerical lifecycle lives in
``rigno.heat3d_training`` and every dependency is assembled explicitly here.
The command never enumerates test or sealed samples and does not select a
model from validation results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import platform
import resource
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import (
    TrainingBatch,
    TrainingDependencies,
    V7FormalTrainer,
    build_v1_training_batches,
    build_v1_training_stats,
    load_selected_v1_examples,
    make_optimizer,
)
from rigno.models.rigno import RIGNO


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = REPO_DIR / "data" / "heat3d-thermal-simulation" / "subsets" / "v1_multilayer_bc_eq_supervised_small"
DEFAULT_MANIFEST = REPO_DIR / "configs" / "heat3d_v1_supervised_small_manifest.json"
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 2,
    "node_latent_size": 16,
    "edge_latent_size": 16,
    "mlp_hidden_layers": 1,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V7 formal training readiness fixture.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--valid-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--optimizer", choices=("manual_gd", "adam", "adamw"), default="manual_gd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--jit-cache", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _tree_global_norm(value: Any) -> Any:
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in tree.tree_leaves(value)))


def _apply_model(model: RIGNO, params: Any, batch: TrainingBatch, rng: Any) -> tuple[Any, ...]:
    del rng
    return tuple(
        model.apply(
            {"params": params},
            inputs=group["inputs"],
            graphs=group["graphs"],
        )
        for group in batch.groups
    )


def _loss(predictions: tuple[Any, ...], batch: TrainingBatch) -> Any:
    weighted = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for prediction, group in zip(predictions, batch.groups, strict=True):
        target = group["target_normalized"]
        sample_count = int(target.shape[0])
        weighted = weighted + jnp.mean(jnp.square(prediction - target)) * sample_count
        count += sample_count
    return weighted / max(count, 1)


def _metrics(model: RIGNO, params: Any, batch: TrainingBatch) -> dict[str, float]:
    predictions = _apply_model(model, params, batch, None)
    normalized_losses = []
    raw_delta_mse = []
    for prediction, group in zip(predictions, batch.groups, strict=True):
        normalized_losses.append(jnp.mean(jnp.square(prediction - group["target_normalized"])))
        raw_delta_mse.append(jnp.mean(jnp.square(prediction - group["target_normalized"])))
    return {
        "normalized_loss": float(jnp.mean(jnp.asarray(normalized_losses))),
        "raw_delta_mse_in_normalized_training_space": float(jnp.mean(jnp.asarray(raw_delta_mse))),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def _checkpoint_roundtrip(path: Path, state: Any) -> float:
    started = time.perf_counter()
    _write_checkpoint(path, {"params": state.params, "optimizer_state": state.optimizer_state, "step": state.step})
    with path.open("rb") as stream:
        loaded = pickle.load(stream)
    if int(loaded["step"]) != int(state.step):
        raise RuntimeError("checkpoint round-trip changed step")
    return time.perf_counter() - started


def _device_memory() -> dict[str, Any]:
    result = []
    for device in jax.devices():
        stats = getattr(device, "memory_stats", None)
        values = stats() if callable(stats) else None
        result.append({"device": str(device), "memory_stats": values})
    return {"devices": result}


def _host_peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return raw if platform.system() == "Darwin" else raw * 1024


def main() -> int:
    args = _parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.dry_run:
        print(json.dumps({
            "entrypoint": "scripts/run_heat3d_v7_formal_training.py",
            "library_core": "rigno.heat3d_training",
            "experiment_role": "readiness_fixture",
            "no_test_or_sealed_access": True,
            "g1_started": False,
        }, indent=2))
        return 0

    profile: dict[str, Any] = {
        "execution_role": "readiness_fixture",
        "scientific_evidence_eligible": False,
        "g1_started": False,
        "test_or_sealed_accessed": False,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }

    data_started = time.perf_counter()
    selected = load_selected_v1_examples(
        args.subset,
        args.manifest,
        train_count=args.train_count,
        valid_count=args.valid_count,
    )
    profile["data_loading_seconds"] = time.perf_counter() - data_started
    profile["sample_ids"] = {
        "train": [example.sample_id for example in selected["train"]],
        "valid": [example.sample_id for example in selected["valid"]],
    }

    stats_started = time.perf_counter()
    stats = build_v1_training_stats(selected["train"])
    builder = Heat3DGraphBuilder(
        radius_policy="legacy_kdtree_mean4",
        coverage_repair_policy="none",
        repair_p2r=True,
        repair_r2p=True,
    )
    preparation_profile: dict[str, Any] = {}
    train_batches = build_v1_training_batches(
        selected["train"], stats, builder, batch_prefix="v7_train", profile=preparation_profile
    )
    valid_batches = build_v1_training_batches(
        selected["valid"], stats, builder, batch_prefix="v7_valid", profile=preparation_profile
    )
    profile["feature_stats_and_batch_preparation_seconds"] = time.perf_counter() - stats_started
    profile.update(preparation_profile)
    profile["unique_shape_signatures"] = sorted({
        json.dumps(
            {
                "batch_id": batch.batch_id,
                "groups": [
                    {
                        "sample_count": len(group["sample_ids"]),
                        "input_shape": list(group["inputs"].c.shape),
                        "graph_leaves": [list(leaf.shape) for leaf in tree.tree_leaves(group["graphs"]) if hasattr(leaf, "shape")],
                    }
                    for group in batch.groups
                ],
            },
            sort_keys=True,
        )
        for batch in train_batches
    })
    train_batch = train_batches[0]
    valid_batch = valid_batches[0]

    model = RIGNO(**MODEL_CONFIG)
    init_started = time.perf_counter()
    params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=train_batch.groups[0]["inputs"],
        graphs=train_batch.groups[0]["graphs"],
    )["params"]
    profile["model_initialization_seconds"] = time.perf_counter() - init_started
    dependencies = TrainingDependencies(
        data_source=selected,
        feature_transform="rigno.heat3d_training.prepare",
        normalization=stats,
        graph_builder=builder,
        model=model,
        model_apply=lambda current_params, batch, rng: _apply_model(model, current_params, batch, rng),
        loss_fn=_loss,
        optimizer=make_optimizer(args.optimizer, learning_rate=args.lr, weight_decay=args.weight_decay),
        batch_iterator=lambda batches: batches,
        validation_fn=lambda current_params, batch: _metrics(model, current_params, batch),
        checkpoint_writer=_write_checkpoint,
        metrics_fn=lambda current_params, batch: _metrics(model, current_params, batch),
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=args.jit_cache)
    state = trainer.initialize(params)
    key = jax.random.PRNGKey(args.seed + 1)
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    update_norm_history: list[float] = []
    step_times: list[float] = []
    compile_warmup_seconds = 0.0
    forward_backward_seconds = 0.0
    optimizer_seconds = None
    for step_index in range(args.steps):
        key, step_key = jax.random.split(key)
        started = time.perf_counter()
        result = trainer.step(state, train_batch, step_key)
        elapsed = time.perf_counter() - started
        state = result.state
        loss_history.append(float(result.loss))
        grad_norm_history.append(float(_tree_global_norm(result.gradients)))
        update_norm_history.append(float(_tree_global_norm(result.updates)))
        step_times.append(elapsed)
        if step_index == 0 and args.jit_cache:
            compile_warmup_seconds = elapsed
        else:
            forward_backward_seconds += elapsed

    validation_started = time.perf_counter()
    valid_metrics = trainer.validate(state, valid_batch)
    profile["validation_seconds"] = time.perf_counter() - validation_started
    profile["checkpoint_io_seconds"] = 0.0
    checkpoint_path = args.checkpoint_path
    temporary_checkpoint = None
    if args.profile and checkpoint_path is None:
        temporary_checkpoint = tempfile.NamedTemporaryFile(prefix="heat3d_v7_ready_", suffix=".pkl", delete=False)
        temporary_checkpoint.close()
        checkpoint_path = Path(temporary_checkpoint.name)
    if checkpoint_path is not None:
        profile["checkpoint_io_seconds"] = _checkpoint_roundtrip(checkpoint_path, state)

    profile.update({
        "compile_warmup_seconds": compile_warmup_seconds,
        "model_forward_backward_seconds": forward_backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "optimizer_timing_status": "included_in_V7FormalTrainer_step; not separately observable inside compiled/eager closure",
        "step_wall_seconds": step_times,
        "total_step_wall_seconds": float(sum(step_times)),
        "compile_count": int(trainer.compile_count),
        "graph_rebuild_count": int(profile.get("graph_build_graphs_calls", 0)),
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
        "device_memory": _device_memory(),
        "valid_metrics": valid_metrics,
        "loss_history": loss_history,
        "gradient_norm_history": grad_norm_history,
        "update_norm_history": update_norm_history,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "model_config": MODEL_CONFIG,
        "optimizer": args.optimizer,
        "jit_cache": bool(args.jit_cache),
    })
    if temporary_checkpoint is not None:
        temporary_checkpoint_path = Path(temporary_checkpoint.name)
        temporary_checkpoint_path.unlink(missing_ok=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(json.dumps(profile, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
