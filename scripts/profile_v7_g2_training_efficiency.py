#!/usr/bin/env python3
"""Performance-only qualification for the frozen Heat3D G2 runner.

The remote mode prepares only the frozen ``train`` and ``valid`` compact
labels, executes a bounded number of warm calls, and writes a small JSON
receipt.  It never accepts a test/sealed path and never computes a publication
accuracy metric.  The isolated forward/loss/backward timings are diagnostic
components; the fused ``step`` and ``step_slim`` timings are the authoritative
throughput measurements because JAX normally fuses the former operations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p50", "p90", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _edge_padding(batches: list[Any]) -> dict[str, Any]:
    """Count real and repeated dummy edges after frozen batch padding."""

    fields = {
        "p2r": "p2r_edge_indices",
        "r2r": "r2r_edge_indices",
        "r2p": "r2p_edge_indices",
    }
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in fields}
    for batch in batches:
        group = batch.groups[0]
        metadata = group["metadata"]
        n_p_in = int(np.asarray(metadata.x_pnodes_inp).shape[1] - 1)
        n_p_out = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
        n_r = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
        dummy_endpoints = {
            "p2r": (n_p_in, n_r),
            "r2r": (n_r, n_r),
            "r2p": (n_r, n_p_out),
        }
        for name, field in fields.items():
            value = getattr(metadata, field)
            if value is None:
                continue
            array = np.asarray(value)
            for row in array:
                sender, receiver = dummy_endpoints[name]
                dummy = (row[:, 0] == sender) & (row[:, 1] == receiver)
                real = int(np.count_nonzero(~dummy))
                padded = int(np.count_nonzero(dummy))
                rows[name].append({
                    "batch_id": batch.batch_id,
                    "real_edges": real,
                    "padded_edges": padded,
                    "total_slots": real + padded,
                    "padding_over_real": float(padded / real) if real else None,
                    "inflation_total_over_real": float((real + padded) / real) if real else None,
                })
    result: dict[str, Any] = {}
    for name, values in rows.items():
        real = [float(row["real_edges"]) for row in values]
        padded = [float(row["padded_edges"]) for row in values]
        ratios = [float(row["padding_over_real"]) for row in values if row["padding_over_real"] is not None]
        result[name] = {
            "sample_count": len(values),
            "real_edges_total": int(sum(real)),
            "padded_edges_total": int(sum(padded)),
            "real_edges": quantiles(real),
            "padded_edges": quantiles(padded),
            "padding_over_real": quantiles(ratios),
            "rows": values,
        }
    return result


def _build_examples(dataset: Any, role: str, full_coords: np.ndarray, full_cv: np.ndarray, layer_id: np.ndarray) -> list[Any]:
    """Recreate the P6 input contract without touching any test artifact."""

    from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume
    from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
    from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample

    result = []
    for index in range(len(dataset)):
        compact = dataset[index]
        weights, audit = conservative_selected_control_volume(
            full_coords=full_coords,
            full_control_volume=full_cv,
            full_layer_id=layer_id,
            selected_indices=compact["support_indices"],
        )
        if audit["relative_volume_error"] > 1e-12:
            raise ValueError("support control-volume conservation failed")
        features = compact["features"].astype(np.float64)
        metadata = {
            "split": role,
            "physics": {
                "ambient_K": 298.15,
                "footprint_m": [1.0, 1.0],
                "layers_bottom_to_top": [
                    {"name": "lower", "thickness_m": 0.1, "k_W_mK": 2.0},
                    {"name": "upper", "thickness_m": 0.45, "k_W_mK": 0.1},
                ],
            },
            "package_total_power_W": float(np.dot(np.maximum(features[:, 3], 0), weights)),
            "v6_adapter": {
                "dataset_id": "deepoheat_v1_volumetric_method_native_1024",
                "manifest_split_role": role,
                "group_id": compact["sample_id"],
                "reference_temperature_K": 298.15,
                "top_T_inf_K": 298.15,
                "bottom_T_inf_K": 298.15,
                "bottom_boundary_semantics": "robin_not_dirichlet",
                "operator_point_measure": "same_layer_nearest_full_CV_partition",
                "official_source_index": compact["source_index"],
            },
        }
        result.append(V6DualRobinExample(
            sample_id=compact["sample_id"],
            condition=V1SteadyConditionInput(
                coords=compact["coords"].astype(np.float64),
                condition_features=features,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(target_u=(298.15 + compact["target_1024"]).reshape(-1, 1)),
            meta=metadata,
            operator_point_weights=weights,
        ))
    return result


def _isolated_components(apply_fn, batch_loss, params, batch, key, block) -> dict[str, Any]:
    """Measure unfused diagnostic components on one fixed batch."""

    timings: dict[str, Any] = {}
    started = time.perf_counter()
    prediction = apply_fn(params, batch, key)
    block(prediction)
    timings["forward_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    loss = batch_loss(prediction, batch)
    block(loss)
    timings["loss_seconds"] = time.perf_counter() - started

    def loss_only(current):
        return batch_loss(apply_fn(current, batch, key), batch)

    started = time.perf_counter()
    gradients = jax.grad(loss_only)(params)
    block(gradients)
    timings["backward_seconds_including_model_replay"] = time.perf_counter() - started

    timings["note"] = "isolated components are non-additive diagnostics; fused step is authoritative"
    return timings


def _optimizer_dispatch_components(trainer, state, batch, key, block) -> dict[str, Any]:
    """Measure optimizer/update and asynchronous dispatch boundaries.

    JAX normally fuses model, loss, gradient, optimizer and parameter update
    into the compiled training executable.  These timings therefore remain
    engineering diagnostics: they are not additive with the fused step and
    are never used to select an accuracy/configuration outcome.
    """

    compiled = trainer._compiled_for(batch)
    # The executable is already compiled by the preceding warm call.  Measure
    # dispatch without forcing a host/device wait, then measure the explicit
    # completion boundary separately.
    dispatch_started = time.perf_counter()
    pending = compiled(state.params, state.optimizer_state, key)
    dispatch_seconds = time.perf_counter() - dispatch_started
    sync_started = time.perf_counter()
    block(pending)
    synchronization_seconds = time.perf_counter() - sync_started

    # Isolate the optimizer update and optax parameter application on the
    # exact gradient/state trees produced by the frozen update.  This is a
    # separate diagnostic executable, not a formal training path.
    _, _, _, gradients, _updates, _prediction = pending

    def update_only(current_gradients, current_state, current_params):
        return trainer.dependencies.optimizer.update(
            current_gradients, current_state, current_params
        )

    update_compiled = jax.jit(update_only)
    update_started = time.perf_counter()
    update_first = update_compiled(gradients, state.optimizer_state, state.params)
    block(update_first)
    update_first_seconds = time.perf_counter() - update_started
    update_started = time.perf_counter()
    update_warm = update_compiled(gradients, state.optimizer_state, state.params)
    block(update_warm)
    update_warm_seconds = time.perf_counter() - update_started

    def apply_only(current_params, current_updates):
        return optax.apply_updates(current_params, current_updates)

    apply_compiled = jax.jit(apply_only)
    apply_started = time.perf_counter()
    applied_first = apply_compiled(state.params, update_first[0])
    block(applied_first)
    apply_first_seconds = time.perf_counter() - apply_started
    apply_started = time.perf_counter()
    applied_warm = apply_compiled(state.params, update_warm[0])
    block(applied_warm)
    apply_warm_seconds = time.perf_counter() - apply_started

    return {
        "dispatch_without_block_seconds": dispatch_seconds,
        "explicit_block_until_ready_seconds": synchronization_seconds,
        "optimizer_update_first_seconds": update_first_seconds,
        "optimizer_update_warm_seconds": update_warm_seconds,
        "parameter_apply_first_seconds": apply_first_seconds,
        "parameter_apply_warm_seconds": apply_warm_seconds,
        "finite": all(
            bool(np.all(np.isfinite(np.asarray(leaf))))
            for leaf in tree.tree_leaves((update_warm, applied_warm))
        ),
        "note": (
            "diagnostic executables on one frozen batch; optimizer/update and "
            "dispatch values are not additive with the fused JIT step"
        ),
    }


def run_remote(args: argparse.Namespace) -> dict[str, Any]:
    if args.fs_train.name != "fs_train_volume.npy":
        raise ValueError("only fs_train_volume.npy is accepted")
    for forbidden in ("test", "sealed"):
        if forbidden in str(args.fs_train).lower() or forbidden in str(args.labels_root).lower():
            raise ValueError("test/sealed paths are forbidden")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: remote performance profile requires JAX CUDA")

    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    # Bind the profile to the same train-only parser and SHA checks as P6.
    stats = load_script("run_v7_g2_p6_heat3d_v1_formal.py").load_stats(args.normalization)
    train_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="train"
    )
    valid_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="valid", verify_source_file=False
    )
    mesh = support.mesh_arrays()
    preparation_started = time.perf_counter()
    train_examples = _build_examples(train_data, "train", mesh["coords"], mesh["control_volume"], mesh["layer_id"])
    valid_examples = _build_examples(valid_data, "valid", mesh["coords"], mesh["control_volume"], mesh["layer_id"])
    builder = __import__("rigno.graphBuilder_Heat3D", fromlist=["Heat3DGraphBuilder"]).Heat3DGraphBuilder(**config["graph"])
    from rigno.heat3d_training import build_p1i_batches
    profile: dict[str, Any] = {}
    train_batches = build_p1i_batches(
        train_examples, stats, builder, label="g2_e_profile_train", batch_size=24, graph_seed=args.seed, profile=profile
    )
    valid_batches = build_p1i_batches(
        valid_examples, stats, builder, label="g2_e_profile_valid", batch_size=32, graph_seed=args.seed, profile=profile
    )
    all_examples = train_examples + valid_examples
    from rigno.heat3d_training.p1i import (
        attach_input_contexts,
        attach_native_physics,
        attach_qk_features,
        fit_native_loss_references,
    )
    context = attach_input_contexts(train_batches + valid_batches, train_examples, all_examples, config["model"])
    by_id = {row.sample_id: row for row in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
        attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    prep_seconds = time.perf_counter() - preparation_started

    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(train_examples, config["loss"]))
    model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
    from rigno.models.rigno import RIGNO
    from rigno.heat3d_training import (
        TrainingDependencies, V7FormalTrainer, atomic_training_checkpoint, block_until_ready,
        loss_fn_full, make_gradient_transform, make_p1i_optimizer, model_apply_full, model_init_full,
    )
    model = RIGNO(**model_config)
    params = model_init_full(model, jax.random.PRNGKey(args.seed), train_batches[0])["params"]
    optimizer = make_p1i_optimizer(config["optimizer"], epochs=200, updates_per_epoch=len(train_batches))
    apply_fn = lambda current, batch, rng: model_apply_full(model, current, batch, rng)
    batch_loss = lambda prediction, batch: loss_fn_full(prediction, batch, loss_config)
    deps = TrainingDependencies(
        data_source="frozen_768_128_deepoheat_v1_labels",
        feature_transform="physics_layout_aware_1024",
        normalization=stats,
        graph_builder=builder,
        model=model,
        model_apply=apply_fn,
        loss_fn=batch_loss,
        optimizer=optimizer,
        batch_iterator=lambda value: value,
        validation_fn=lambda current, batch: batch_loss(apply_fn(current, batch, None), batch),
        checkpoint_writer=lambda _path, _payload: None,
        metrics_fn=lambda _current, _batch: {"performance_only": True},
        gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
        validation_outputs_fn=lambda current, batch: (
            (prediction := apply_fn(current, batch, None)), batch_loss(prediction, batch)
        ),
    )
    trainer = V7FormalTrainer(deps, jit_cache=True)
    trainer_slim = V7FormalTrainer(deps, jit_cache=True)
    state = trainer.initialize(params)
    state_slim = trainer_slim.initialize(params)
    batch = train_batches[0]
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    # Compile and warm the legacy and slim paths once.  No truth or accuracy
    # is consumed; only finite scalar checks are retained.
    legacy_started = time.perf_counter()
    legacy = trainer.step(state, batch, key)
    block_until_ready((legacy.state.params, legacy.state.optimizer_state, legacy.loss))
    legacy_first = time.perf_counter() - legacy_started
    legacy_started = time.perf_counter()
    legacy_warm = trainer.step(legacy.state, batch, key)
    block_until_ready((legacy_warm.state.params, legacy_warm.state.optimizer_state, legacy_warm.loss))
    legacy_warm_seconds = time.perf_counter() - legacy_started

    slim_started = time.perf_counter()
    slim = trainer_slim.step_slim(state_slim, batch, key)
    block_until_ready((slim.state.params, slim.state.optimizer_state, slim.loss))
    slim_first = time.perf_counter() - slim_started
    slim_started = time.perf_counter()
    slim_warm = trainer_slim.step_slim(slim.state, batch, key)
    block_until_ready((slim_warm.state.params, slim_warm.state.optimizer_state, slim_warm.loss))
    slim_warm_seconds = time.perf_counter() - slim_started
    finite = bool(np.isfinite(float(np.asarray(legacy_warm.loss))) and np.isfinite(float(np.asarray(slim_warm.loss))))

    # Validation duplicate/single-forward comparison on one valid batch.
    valid_batch = valid_batches[0]
    duplicate_started = time.perf_counter()
    duplicate_prediction = apply_fn(legacy_warm.state.params, valid_batch, None)
    duplicate_prediction_2 = apply_fn(legacy_warm.state.params, valid_batch, None)
    duplicate_loss = batch_loss(duplicate_prediction_2, valid_batch)
    block_until_ready((duplicate_prediction, duplicate_prediction_2, duplicate_loss))
    duplicate_seconds = time.perf_counter() - duplicate_started
    single_started = time.perf_counter()
    single_prediction = apply_fn(legacy_warm.state.params, valid_batch, None)
    single_loss = batch_loss(single_prediction, valid_batch)
    block_until_ready((single_prediction, single_loss))
    single_seconds = time.perf_counter() - single_started

    isolated = _isolated_components(apply_fn, batch_loss, params, batch, key, block_until_ready)
    optimizer_dispatch = _optimizer_dispatch_components(
        trainer, state, batch, key, block_until_ready
    )
    with tempfile.TemporaryDirectory(prefix="g2_e_profile_") as temporary:
        checkpoint_started = time.perf_counter()
        checkpoint = atomic_training_checkpoint(Path(temporary) / "profile.pkl", state=slim_warm.state, metadata={"profile": True})
        checkpoint_seconds = time.perf_counter() - checkpoint_started

    memory = jax.devices()[0].memory_stats() or {}
    return {
        "schema_version": "heat3d_v7_g2_e_training_efficiency_profile_v1",
        "status": "PASS_PERFORMANCE_PROFILE_FINITE" if finite else "FAIL_NONFINITE",
        "scientific_config_unchanged": {
            "batch_size": int(config["batching"]["batch_size"]),
            "validation_batch_size": int(config["batching"]["validation_batch_size"]),
            "model": {key: config["model"].get(key) for key in ("node_latent_size", "edge_latent_size", "processor_steps")},
            "optimizer": config["optimizer"],
        },
        "dataset": {"train": len(train_examples), "valid": len(valid_examples), "official_test_access": False},
        "preparation": {"wall_seconds": prep_seconds, "instrumentation": profile},
        "graph_padding": _edge_padding(train_batches),
        "step_decomposition": {
            "isolated_components": isolated,
            "optimizer_update_and_dispatch": optimizer_dispatch,
            "legacy_jit_step_first_seconds": legacy_first,
            "legacy_jit_step_warm_seconds": legacy_warm_seconds,
            "slim_jit_step_first_seconds": slim_first,
            "slim_jit_step_warm_seconds": slim_warm_seconds,
            "legacy_compile_count": trainer.compile_count,
            "slim_compile_count": len(getattr(trainer_slim, "_compiled_slim_steps", {})),
            "host_device_sync_boundary": "included in block_until_ready around each measured call",
        },
        "validation_single_forward": {
            "duplicate_forward_seconds": duplicate_seconds,
            "single_forward_seconds": single_seconds,
            "speedup_ratio": duplicate_seconds / single_seconds if single_seconds else None,
            "finite": bool(np.isfinite(float(np.asarray(single_loss)))),
        },
        "checkpoint_io": {"seconds": checkpoint_seconds, **checkpoint},
        "resource": {
            "device": str(jax.devices()[0]),
            "memory_stats": {key: int(value) for key, value in memory.items() if isinstance(value, (int, np.integer))},
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "matched_g1_baseline": {"status": "DEFERRED_NOT_SAME_DATASET", "reason": "no second model execution or accuracy comparison in this bounded profile"},
        "b32_nonpublication": {"status": "NOT_RUN", "reason": "B24 frozen; B32 would require a separate shape compile and is not used to select a formal config"},
        "formal_accuracy_claim_allowed": False,
        "test_or_sealed_access": False,
        "environment": {"python": sys.version, "jax": jax.__version__, "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "remote-v1"), required=True)
    parser.add_argument("--fs-train", type=Path)
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--heat3d-config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "static":
        sources = {
            "validation_callback": "single_forward_in_p6_after_g2_e_patch",
            "slim_step_api": "V7FormalTrainer.step_slim",
            "per_batch_jit_cache": "V7FormalTrainer._compiled_for caches by batch_id",
            "gino_dataset": "P1iRoleDataset hashes and np.load per __getitem__",
            "gino_formal_synchronization": "torch.cuda.synchronize around every train/valid step",
            "scientific_semantics": "unchanged; profile is nonpublication",
        }
        payload = {"schema_version": "heat3d_v7_g2_e_static_efficiency_audit_v1", "status": "PASS_STATIC_SOURCE_AUDIT", "sources": sources, "test_or_sealed_access": False}
    else:
        required = (args.fs_train, args.labels_root, args.normalization, args.heat3d_config)
        if any(value is None for value in required):
            parser.error("remote-v1 requires --fs-train, --labels-root, --normalization and --heat3d-config")
        payload = run_remote(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
