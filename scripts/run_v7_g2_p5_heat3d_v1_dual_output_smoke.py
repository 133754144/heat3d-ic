#!/usr/bin/env python3
"""One-step Heat3D smoke on frozen v1 support and 10201-slice reconstruction."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.tree_util as tree
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import TrainingDependencies, V7FormalTrainer, atomic_training_checkpoint, block_until_ready, build_p1i_batches, loss_fn_full, make_gradient_transform, make_p1i_optimizer, model_apply_full, model_init_full
from rigno.heat3d_training.p1i import attach_input_contexts, attach_native_physics, attach_qk_features, fit_native_loss_references
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v1_normalization import legacy_train_only_stats
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample
from rigno.heat3d_v6_full_field import build_reconstruction_map
from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume
from rigno.models.rigno import RIGNO


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name; spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[path.stem] = module; spec.loader.exec_module(module); return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def safe_tree_difference(left: Any, right: Any) -> float:
    a, da = tree.tree_flatten(left); b, db = tree.tree_flatten(right)
    if da != db or len(a) != len(b): return float("inf")
    return max((float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x, y in zip(a, b, strict=True) if np.asarray(x).size), default=0.0)


def resolve_model_config(payload: Mapping[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    result = dict(payload); result.pop("global_context_ablation", None); result.pop("architecture", None)
    result["global_context_feature_names"] = tuple(result.get("global_context_feature_names") or ())
    names = tuple(result.get("decoder_bypass_local_feature_names") or ()); result["decoder_bypass_local_feature_names"] = names; result["decoder_bypass_feature_names"] = names
    result["decoder_bypass_feature_indices"] = tuple(feature_names.index(name) for name in names); result["decoder_bypass_num_features"] = len(names)
    return result


def decode_first(payload: dict[str, Any], role: str, count: int) -> list[int]:
    row = payload["roles"][role]; values = np.frombuffer(base64.b64decode(row["indices_base64"]), dtype="<u4")
    return [int(value) for value in values[:count]]


def make_example(sample_id: str, role: str, source_index: int, arrays: dict[str, np.ndarray], target: np.ndarray, support: np.ndarray, weights: np.ndarray) -> V6DualRobinExample:
    features = np.asarray(arrays["features"])[support].astype(np.float64); coords = np.asarray(arrays["coords"])[support].astype(np.float64)
    if features.shape != (1024, 11): raise ValueError("formal support feature shape mismatch")
    meta = {"split": role, "physics": {"ambient_K": 298.15, "footprint_m": [1.0, 1.0], "layers_bottom_to_top": [{"name": "lower", "thickness_m": 0.1, "k_W_mK": 2.0}, {"name": "upper", "thickness_m": 0.45, "k_W_mK": 0.1}]}, "package_total_power_W": float(np.dot(np.maximum(features[:, 3], 0), weights)), "v6_adapter": {"dataset_id": "deepoheat_v1_volumetric_method_native_1024", "manifest_split_role": role, "group_id": sample_id, "reference_temperature_K": 298.15, "top_T_inf_K": 298.15, "bottom_T_inf_K": 298.15, "bottom_boundary_semantics": "robin_not_dirichlet", "operator_point_measure": "same_layer_nearest_full_CV_partition", "official_source_index": source_index}}
    return V6DualRobinExample(sample_id=sample_id, condition=V1SteadyConditionInput(coords=coords, condition_features=features, condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES, k_encoding_mode="diag3"), target=V1SteadyTarget(target_u=(298.15 + target).reshape(-1, 1)), meta=meta, operator_point_weights=weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--fs-train", type=Path, required=True); parser.add_argument("--subset-manifest", type=Path, required=True); parser.add_argument("--labels-root", type=Path, required=True); parser.add_argument("--heat3d-config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--resume-postprocess", action="store_true"); args = parser.parse_args()
    if not str(args.output_dir.resolve()).startswith(("/tmp/", "/private/tmp/")): raise ValueError("smoke artifacts must remain under /tmp")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset = json.loads(args.subset_manifest.read_text()); indices = {"train": decode_first(subset, "train", 6), "valid": decode_first(subset, "valid", 2)}
    converter = load_script("convert_v7_g2_semiconductor_case.py"); support_module = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False); mesh = support_module.mesh_arrays(); full_coords = mesh["coords"]; full_cv = mesh["control_volume"]; layer_id = mesh["layer_id"]
    examples: dict[str, list[V6DualRobinExample]] = {"train": [], "valid": []}; support_by_id = {}; full_truth_by_id = {}
    for role in ("train", "valid"):
        for source_index in indices[role]:
            sample_id = f"dhv1_volume_{role}_{source_index:05d}"; directory = args.labels_root / role / sample_id
            power = np.asarray(fs_train[source_index]); support = np.asarray(np.load(directory / "support_indices.npy"), dtype=np.int64)
            expected, _strata, _audit = support_module.select_support(power, source_index=source_index, role=role)
            if not np.array_equal(support, expected): raise ValueError(f"support drift: {sample_id}")
            weights, weight_audit = conservative_selected_control_volume(full_coords=full_coords, full_control_volume=full_cv, full_layer_id=layer_id, selected_indices=support)
            if weight_audit["relative_volume_error"] > 1e-12: raise ValueError("support CV conservation failed")
            target = np.asarray(np.load(directory / "deltaT_support1024_K.npy"), dtype=np.float64)
            full_truth = np.asarray(np.load(directory / "deltaT_full_K.npy", mmap_mode="r"), dtype=np.float64)
            arrays = converter.volume_v1_arrays(power); examples[role].append(make_example(sample_id, role, source_index, arrays, target, support, weights)); support_by_id[sample_id] = support; full_truth_by_id[sample_id] = full_truth
    config = json.loads(args.heat3d_config.read_text()); stats = legacy_train_only_stats(examples["train"]); builder = Heat3DGraphBuilder(**config["graph"])
    train_batches = build_p1i_batches(examples["train"], stats, builder, label="p5_train", batch_size=6, graph_seed=0); valid_batches = build_p1i_batches(examples["valid"], stats, builder, label="p5_valid", batch_size=2, graph_seed=0)
    all_examples = examples["train"] + examples["valid"]; context = attach_input_contexts(train_batches + valid_batches, examples["train"], all_examples, config["model"]); by_id = {row.sample_id: row for row in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"]); attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    loss_config = dict(config["loss"]); loss_config.update(fit_native_loss_references(examples["train"], config["loss"])); model_config = resolve_model_config(config["model"], tuple(stats["feature_names"])); model = RIGNO(**model_config)
    params = model_init_full(model, jax.random.PRNGKey(0), train_batches[0])["params"]; optimizer = make_p1i_optimizer(config["optimizer"], epochs=1, updates_per_epoch=1)
    apply_fn = lambda current, batch, rng: model_apply_full(model, current, batch, rng); batch_loss = lambda prediction, batch: loss_fn_full(prediction, batch, loss_config)
    deps = TrainingDependencies(data_source="temporary_6_train_2_valid_frozen_solver_labels", feature_transform="physics_layout_aware_1024", normalization=stats, graph_builder=builder, model=model, model_apply=apply_fn, loss_fn=batch_loss, optimizer=optimizer, batch_iterator=lambda x:x, validation_fn=lambda current,batch: batch_loss(apply_fn(current,batch,None),batch), checkpoint_writer=lambda p,v:None, metrics_fn=lambda current,batch:{"finite": True}, gradient_transform=make_gradient_transform(model_config, config["optimizer"]), validation_outputs_fn=lambda current,batch:(apply_fn(current,batch,None), batch_loss(apply_fn(current,batch,None),batch)))
    trainer = V7FormalTrainer(deps, jit_cache=True); checkpoint_path = args.output_dir / "heat3d_p5_dual_output_step1.pkl"
    if args.resume_postprocess:
        if not checkpoint_path.is_file(): raise FileNotFoundError("resume requires the completed step-1 checkpoint")
        with checkpoint_path.open("rb") as stream: completed = pickle.load(stream)
        if int(completed["step"]) != 1: raise ValueError("resume checkpoint is not step 1")
        prediction = model_apply_full(model, completed["params"], valid_batches[0], None); block_until_ready(prediction); valid_loss = loss_fn_full(prediction, valid_batches[0], loss_config)
        checkpoint = {"passed": True, "path": str(checkpoint_path), "recovery": "postprocess_only_no_repeated_optimizer_step"}
    else:
        state = trainer.initialize(params); step = trainer.step(state, train_batches[0], jax.random.PRNGKey(0)); block_until_ready(step.state.params)
        prediction, valid_loss = trainer.validate_with_outputs(step.state, valid_batches[0]); block_until_ready(prediction)
        checkpoint = atomic_training_checkpoint(checkpoint_path, state=step.state, metadata={"role": "P5_nonformal_dual_output_smoke", "epoch": 1})
    with checkpoint_path.open("rb") as stream: reloaded = pickle.load(stream)
    reloaded_prediction = model_apply_full(model, reloaded["params"], valid_batches[0], None); block_until_ready(reloaded_prediction)
    native_exact = safe_tree_difference(prediction, reloaded_prediction) == 0.0
    native_finite = all(np.all(np.isfinite(np.asarray(row["deltaT_hat"]))) for row in prediction)
    # There is one geometry group containing the ordered two-example batch.
    if len(prediction) != 1 or len(reloaded_prediction) != 1: raise RuntimeError("unexpected valid grouping for dual-output smoke")
    native_rows = np.asarray(prediction[0]["deltaT_hat"], dtype=np.float64)[:, 0, :, 0]
    reload_rows = np.asarray(reloaded_prediction[0]["deltaT_hat"], dtype=np.float64)[:, 0, :, 0]
    if native_rows.shape != (2, 1024): raise RuntimeError(f"unexpected native prediction shape {native_rows.shape}")
    u_rows = []
    for row_index, example in enumerate(examples["valid"]):
        reconstruction, audit = build_reconstruction_map(coords=full_coords, layer_id=layer_id, boundaries=np.asarray([0.0, 0.1, 0.55]), support_indices=support_by_id[example.sample_id], empty_domain_fallback="same_layer")
        native = native_rows[row_index]; native_reload = reload_rows[row_index]
        high = reconstruction.reconstruct(native).reshape(101,101,56)[:,:,15].reshape(-1); high_reload = reconstruction.reconstruct(native_reload).reshape(101,101,56)[:,:,15].reshape(-1)
        truth = full_truth_by_id[example.sample_id][:,:,15].reshape(-1)
        u_rows.append({"sample_id": example.sample_id, "output_count": int(len(high)), "finite": bool(np.all(np.isfinite(high)) and np.all(np.isfinite(truth))), "checkpoint_reload_bitwise_equal": bool(np.array_equal(high, high_reload)), "reconstruction_algorithm": audit["algorithm"]})
    status = "PASS_1_EPOCH_DUAL_OUTPUT_NONFORMAL" if checkpoint["passed"] and native_exact and native_finite and np.isfinite(float(valid_loss)) and all(row["finite"] and row["checkpoint_reload_bitwise_equal"] for row in u_rows) else "FAIL"
    receipt = {"schema_version": "heat3d_v7_g2_p5_heat3d_v1_dual_output_smoke_v1", "status": status, "scope": "one_optimizer_step_schema_normalization_native_and_reconstruction_query_checkpoint_only", "formal_accuracy_claim": False, "train_functions": 6, "valid_functions": 2, "epochs": 1, "optimizer_steps": 1, "support": {"input_count": 1024, "selector": "physics_layout_aware_source_interface_boundary_volume", "old_8x8x16_projection_used": False}, "native_output": {"count": 1024, "finite": native_finite, "reload_exact": native_exact}, "u_strategy_output": {"count": 10201, "domain": "z=0.15 source-layer top slice on 101x101 grid, not official full 571256-point comparison domain", "conditioning_count_remains": 1024, "rows": u_rows}, "validation_loss_finite": bool(np.isfinite(float(valid_loss))), "checkpoint": {**checkpoint, "sha256": sha256(checkpoint_path)}, "temporary_artifact_root": str(args.output_dir), "formal_or_long_training_started": False, "p1i_test_or_sealed_access": False}
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(receipt, indent=2, sort_keys=True)); return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
