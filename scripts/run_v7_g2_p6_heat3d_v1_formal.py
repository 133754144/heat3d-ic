#!/usr/bin/env python3
"""Formal Heat3D-on-DeepOHeat-v1 training on frozen 768/128 labels.

Checkpoint selection uses only native-1024 valid sample-first relative RMSE.
The two U domains are evaluation diagnostics and cannot influence training or
selection. Official DeepOHeat-v1 test files are neither accepted nor opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import (
    TrainingDependencies, V7FormalTrainer, atomic_training_checkpoint,
    block_until_ready, build_p1i_batches, evaluate_level_a_validation,
    loss_fn_full, make_gradient_transform, make_p1i_optimizer,
    model_apply_full, model_init_full, tree_parameter_count,
)
from rigno.heat3d_training.p1i import (
    attach_input_contexts, attach_native_physics, attach_qk_features,
    fit_native_loss_references,
)
from rigno.models.rigno import RIGNO


SUBSET_SHA = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
LABEL_RECEIPT_SHA = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
NORMALIZATION_SHA = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual or actual != NORMALIZATION_SHA or payload["valid_or_test_used_to_fit"] is not False:
        raise ValueError("frozen 768-train normalization mismatch")
    stats = dict(payload["statistics"])
    for key in ("coord_min", "coord_span", "condition_mean", "condition_std", "target_delta_mean", "target_delta_std"):
        stats[key] = np.asarray(stats[key], dtype=np.float32)
    return stats


def checkpoint_state(path: Path) -> Any:
    from rigno.heat3d_training.core import TrainingState
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return TrainingState(
        params=payload["params"], optimizer_state=payload["optimizer_state"],
        step=int(payload["step"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract-check", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--fs-train", type=Path)
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--heat3d-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "contract-check":
        print(json.dumps({
            "status": "PASS_CONTRACT_CHECK_NO_DATA_NO_TRAINING", "seed": args.seed,
            "epochs": 200, "train": 768, "valid": 128,
            "selection": "native_1024_valid_sample_first_relative_rmse_pct",
            "U_selection_influence": False, "official_test_access": False,
        }, indent=2, sort_keys=True))
        return 0
    if None in (args.fs_train, args.subset_manifest, args.labels_root, args.normalization, args.heat3d_config, args.output_dir):
        parser.error("train requires all data/config/output arguments")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: Heat3D-v1 formal training requires JAX CUDA")
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit("FAIL-CLOSED: frozen Heat3D CUDA backend requires deterministic XLA ops")
    if sha256(args.subset_manifest) != SUBSET_SHA:
        raise ValueError("subset manifest SHA mismatch")
    if sha256(args.labels_root / "label_generation_receipt.json") != LABEL_RECEIPT_SHA:
        raise ValueError("label cache receipt SHA mismatch")
    stats = load_stats(args.normalization)
    config = json.loads(args.heat3d_config.read_text())
    if config["batching"]["batch_size"] != 24 or config["batching"]["validation_batch_size"] != 32:
        raise ValueError("frozen B24/valid32 batching mismatch")

    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    loader_module = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    support_module = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    train_data = loader_module.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="train"
    )
    valid_data = loader_module.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="valid", verify_source_file=False
    )
    mesh = support_module.mesh_arrays()
    full_coords, full_cv, layer_id = mesh["coords"], mesh["control_volume"], mesh["layer_id"]
    from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume
    from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
    from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample

    def examples_from(dataset: Any, role: str) -> list[Any]:
        rows = []
        for index in range(len(dataset)):
            compact = dataset[index]
            weights, audit = conservative_selected_control_volume(
                full_coords=full_coords, full_control_volume=full_cv,
                full_layer_id=layer_id, selected_indices=compact["support_indices"],
            )
            if audit["relative_volume_error"] > 1e-12:
                raise ValueError("support control-volume conservation failed")
            features = compact["features"].astype(np.float64)
            metadata = {
                "split": role,
                "physics": {"ambient_K": 298.15, "footprint_m": [1.0, 1.0],
                    "layers_bottom_to_top": [
                        {"name": "lower", "thickness_m": 0.1, "k_W_mK": 2.0},
                        {"name": "upper", "thickness_m": 0.45, "k_W_mK": 0.1},
                    ]},
                "package_total_power_W": float(np.dot(np.maximum(features[:, 3], 0), weights)),
                "v6_adapter": {"dataset_id": "deepoheat_v1_volumetric_method_native_1024",
                    "manifest_split_role": role, "group_id": compact["sample_id"],
                    "reference_temperature_K": 298.15, "top_T_inf_K": 298.15,
                    "bottom_T_inf_K": 298.15, "bottom_boundary_semantics": "robin_not_dirichlet",
                    "operator_point_measure": "same_layer_nearest_full_CV_partition",
                    "official_source_index": compact["source_index"]},
            }
            rows.append(V6DualRobinExample(
                sample_id=compact["sample_id"],
                condition=V1SteadyConditionInput(
                    coords=compact["coords"].astype(np.float64),
                    condition_features=features,
                    condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                    k_encoding_mode="diag3",
                ),
                target=V1SteadyTarget(target_u=(298.15 + compact["target_1024"]).reshape(-1, 1)),
                meta=metadata, operator_point_weights=weights,
            ))
        return rows

    preparation_started = time.perf_counter()
    train_examples = examples_from(train_data, "train")
    valid_examples = examples_from(valid_data, "valid")
    builder = Heat3DGraphBuilder(**config["graph"])
    train_batches = build_p1i_batches(train_examples, stats, builder, label="g2_v1_train", batch_size=24, graph_seed=args.seed)
    valid_batches = build_p1i_batches(valid_examples, stats, builder, label="g2_v1_valid", batch_size=32, graph_seed=args.seed)
    all_examples = train_examples + valid_examples
    context = attach_input_contexts(train_batches + valid_batches, train_examples, all_examples, config["model"])
    by_id = {row.sample_id: row for row in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
        attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(train_examples, config["loss"]))
    model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
    model = RIGNO(**model_config)
    params = model_init_full(model, jax.random.PRNGKey(args.seed), train_batches[0])["params"]
    optimizer = make_p1i_optimizer(config["optimizer"], epochs=200, updates_per_epoch=len(train_batches))
    apply_fn = lambda current, batch, rng: model_apply_full(model, current, batch, rng)
    batch_loss = lambda prediction, batch: loss_fn_full(prediction, batch, loss_config)
    deps = TrainingDependencies(
        data_source="frozen_768_128_deepoheat_v1_labels", feature_transform="physics_layout_aware_1024",
        normalization=stats, graph_builder=builder, model=model, model_apply=apply_fn,
        loss_fn=batch_loss, optimizer=optimizer, batch_iterator=lambda value: value,
        validation_fn=lambda current, batch: batch_loss(apply_fn(current, batch, None), batch),
        checkpoint_writer=lambda path, payload: None,
        metrics_fn=lambda current, batch: {"loss": float(batch_loss(apply_fn(current, batch, None), batch))},
        gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
        validation_outputs_fn=lambda current, batch: (apply_fn(current, batch, None), batch_loss(apply_fn(current, batch, None), batch)),
    )
    trainer = V7FormalTrainer(deps, jit_cache=True)
    state = trainer.initialize(params)
    preparation_seconds = time.perf_counter() - preparation_started
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    best_metric, best_epoch = float("inf"), None
    history, update_count = [], 0
    started = time.perf_counter()
    for epoch in range(1, 201):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_batches))
        losses = []
        epoch_started = time.perf_counter()
        for batch_number, raw_index in enumerate(order, start=1):
            step_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), epoch)
            step_key = jax.random.fold_in(step_key, batch_number)
            step = trainer.step(state, train_batches[int(raw_index)], step_key)
            state = step.state; block_until_ready((state.params, state.optimizer_state, step.loss)); losses.append(float(step.loss)); update_count += 1
        predictions, valid_losses = [], []
        for batch in valid_batches:
            prediction, loss = trainer.validate_with_outputs(state, batch)
            block_until_ready((prediction, loss)); predictions.append(prediction); valid_losses.append(float(loss))
        evaluation = evaluate_level_a_validation(
            predictions=predictions, batches=valid_batches, examples=valid_examples,
            stats=stats, variant="Full",
        )
        metric = float(evaluation["metrics"]["sample_first_relative_rmse_pct"])
        if not np.isfinite(metric): raise RuntimeError("nonfinite valid selection metric")
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            atomic_training_checkpoint(output / "params_best_sample_first.pkl", state=state, metadata={
                "epoch": epoch, "seed": args.seed, "selection_metric": "native_1024_valid_sample_first_relative_rmse_pct",
                "selection_value": metric, "U_outputs_used_for_selection": False, "test_access": False,
            })
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": float(np.mean(valid_losses)),
               "native_1024_valid_sample_first_relative_rmse_pct": metric, "best_epoch": best_epoch,
               "epoch_wall_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        (output / "progress.json").write_text(json.dumps({"status":"RUNNING","seed":args.seed,"epoch":epoch,"epochs":200,"best_epoch":best_epoch,"best_metric":best_metric,"test_access":False}, indent=2)+"\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    final_report = atomic_training_checkpoint(output / "params_final.pkl", state=state, metadata={
        "epoch": 200, "seed": args.seed, "best_epoch": best_epoch, "test_access": False,
    })
    best_state = checkpoint_state(output / "params_best_sample_first.pkl")
    probe_before, _ = trainer.validate_with_outputs(best_state, valid_batches[0]); block_until_ready(probe_before)
    best_state_reload = checkpoint_state(output / "params_best_sample_first.pkl")
    probe_after, _ = trainer.validate_with_outputs(best_state_reload, valid_batches[0]); block_until_ready(probe_after)
    reload_max_abs = helper.safe_tree_difference(probe_before, probe_after)
    if reload_max_abs > 1e-5: raise RuntimeError("best checkpoint prediction reload drift")
    memory = jax.devices()[0].memory_stats() or {}
    receipt = {
        "schema_version": "heat3d_v7_g2_p6_heat3d_v1_formal_training_v1",
        "status": "COMPLETE_FORMAL_TRAIN", "seed": args.seed, "epochs": 200,
        "optimizer_update_count": update_count, "parameter_count": tree_parameter_count(state.params),
        "selection": {"domain": "native_1024", "metric": "valid_sample_first_relative_rmse_pct", "tie": "earliest", "best_epoch": best_epoch, "best_value": best_metric, "U_used": False},
        "checkpoints": {"best": {"path": "params_best_sample_first.pkl", "sha256": sha256(output / "params_best_sample_first.pkl")}, "final": {"path":"params_final.pkl", "sha256": sha256(output / "params_final.pkl"), "roundtrip": final_report}, "best_reload_prediction_max_abs": reload_max_abs},
        "resource": {"gpu": str(jax.devices()[0]), "peak_bytes_in_use": memory.get("peak_bytes_in_use"), "training_wall_seconds": time.perf_counter()-started, "preparation_wall_seconds": preparation_seconds},
        "environment": {"repo_sha": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "jax": jax.__version__, "XLA_FLAGS": os.environ.get("XLA_FLAGS")},
        "dataset": {"train":768,"valid":128,"subset_sha256":SUBSET_SHA,"label_receipt_sha256":LABEL_RECEIPT_SHA,"normalization_sha256":NORMALIZATION_SHA},
        "history": history, "test_or_sealed_access": False,
    }
    (output / "formal_training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    (output / "progress.json").write_text(json.dumps({"status":"COMPLETE","seed":args.seed,"epoch":200,"best_epoch":best_epoch,"best_metric":best_metric,"test_access":False},indent=2)+"\n")
    print(json.dumps({key:value for key,value in receipt.items() if key!="history"},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
