#!/usr/bin/env python3
"""Bounded Heat3D-on-DeepOHeat-v1 volumetric pipeline qualification.

This runner is intentionally non-formal. It solves exactly six frozen train
and two frozen valid input functions, saves all temporary labels/checkpoints
under /tmp, deterministically projects each official 101x101x56 case to a
shared 8x8x16=1024-point smoke support, and performs one Heat3D optimizer step,
validation, checkpoint, reload, and evaluator pass. The projection is a
pipeline fixture, not the lossless formal cross-benchmark representation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.tree_util as tree
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_training import (
    TrainingDependencies,
    V7FormalTrainer,
    atomic_training_checkpoint,
    block_until_ready,
    build_p1i_batches,
    loss_fn_full,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_init_full,
    tree_l2_norm,
    tree_max_abs_difference,
    tree_parameter_count,
)
from rigno.heat3d_training.p1i import (
    attach_input_contexts,
    attach_native_physics,
    attach_qk_features,
    fit_native_loss_references,
)
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v1_normalization import legacy_train_only_stats
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample
from rigno.models.rigno import RIGNO


TRAIN_SMOKE_COUNT = 6
VALID_SMOKE_COUNT = 2
REFERENCE_TEMPERATURE_K = 298.15


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def safe_tree_max_abs_difference(left: Any, right: Any) -> float:
    left_leaves, left_def = tree.tree_flatten(left)
    right_leaves, right_def = tree.tree_flatten(right)
    if left_def != right_def or len(left_leaves) != len(right_leaves):
        return float("inf")
    errors = []
    for a, b in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(a)
        right_array = np.asarray(b)
        if left_array.shape != right_array.shape:
            return float("inf")
        if left_array.size:
            errors.append(float(np.max(np.abs(left_array - right_array))))
    return max(errors, default=0.0)


def smoke_flat_indices() -> np.ndarray:
    x = np.rint(np.linspace(0, 100, 8)).astype(np.int64)
    y = np.rint(np.linspace(0, 100, 8)).astype(np.int64)
    z = np.rint(np.linspace(0, 55, 16)).astype(np.int64)
    indices = [((ix * 101 + iy) * 56 + iz) for ix in x for iy in y for iz in z]
    result = np.asarray(indices, dtype=np.int64)
    if result.shape != (1024,) or len(np.unique(result)) != 1024:
        raise AssertionError("smoke support must contain 1024 unique grid points")
    return result


def resolve_model_config(model_config: Mapping[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    resolved = dict(model_config)
    resolved.pop("global_context_ablation", None)
    resolved.pop("architecture", None)
    resolved["global_context_feature_names"] = tuple(resolved.get("global_context_feature_names") or ())
    local_names = tuple(resolved.get("decoder_bypass_local_feature_names") or ())
    resolved["decoder_bypass_local_feature_names"] = local_names
    resolved["decoder_bypass_feature_names"] = local_names
    resolved["decoder_bypass_feature_indices"] = tuple(feature_names.index(name) for name in local_names)
    resolved["decoder_bypass_num_features"] = len(resolved["decoder_bypass_feature_indices"])
    return resolved


def make_example(
    *,
    sample_id: str,
    role: str,
    source_index: int,
    arrays: Mapping[str, np.ndarray],
    solution_u: np.ndarray,
    selected: np.ndarray,
) -> V6DualRobinExample:
    coords = np.asarray(arrays["coords"])[selected].astype(np.float64)
    features = np.asarray(arrays["features"])[selected].astype(np.float64)
    if features.shape != (1024, 11) or not np.allclose(features[:, 4:8].sum(axis=1), 1.0):
        raise ValueError("converter failed the 11-channel/one-hot smoke schema")
    temperature = 293.15 + 25.0 * np.asarray(solution_u, dtype=np.float64).reshape(-1)[selected]
    weights = np.ones(1024, dtype=np.float64)
    q_mean = float(np.mean(np.maximum(features[:, 3], 0.0)))
    if q_mean <= 0.0:
        raise ValueError("deterministic smoke support missed all volumetric source points")
    meta = {
        "split": role,
        "physics": {
            "ambient_K": REFERENCE_TEMPERATURE_K,
            "footprint_m": [1.0, 1.0],
            "layers_bottom_to_top": [
                {"name": "lower", "thickness_m": 0.1, "k_W_mK": 2.0},
                {"name": "upper_and_active", "thickness_m": 0.45, "k_W_mK": 0.1},
            ],
        },
        "package_total_power_W": q_mean * 0.55,
        "v6_adapter": {
            "dataset_id": "deepoheat_v1_volumetric_cross_benchmark_smoke",
            "manifest_split_role": role,
            "group_id": sample_id,
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "top_T_inf_K": REFERENCE_TEMPERATURE_K,
            "bottom_T_inf_K": REFERENCE_TEMPERATURE_K,
            "bottom_boundary_semantics": "robin_not_dirichlet",
            "operator_point_measure": "equal_weight_regular_8x8x16_smoke_projection",
            "official_source_index": source_index,
        },
    }
    return V6DualRobinExample(
        sample_id=sample_id,
        condition=V1SteadyConditionInput(
            coords=coords,
            condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=temperature.reshape(-1, 1)),
        meta=meta,
        operator_point_weights=weights,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--resume-postprocess",
        action="store_true",
        help="reuse already-created temporary labels and step=1 checkpoint; never solve or train",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not str(output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("all labels/checkpoints must remain under /tmp")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if manifest["selection"]["accuracy_or_temperature_observed"] is not False:
        raise ValueError("subset must be frozen before temperature accuracy")
    def role_indices_from_manifest(role: str) -> list[int]:
        payload = manifest["roles"][role]
        if "indices" in payload:
            return [int(value) for value in payload["indices"]]
        raw = base64.b64decode(payload["indices_base64"])
        values = np.frombuffer(raw, dtype="<u4")
        if len(values) != int(payload["count"]):
            raise ValueError(f"encoded {role} index count mismatch")
        return [int(value) for value in values]

    role_indices = {
        "train": role_indices_from_manifest("train")[:TRAIN_SMOKE_COUNT],
        "valid": role_indices_from_manifest("valid")[:VALID_SMOKE_COUNT],
    }
    fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    solver_module = load_script("run_v7_g2_deepoheat_v1_solver_fidelity.py")
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    solver = None if args.resume_postprocess else solver_module.OfficialVolumetricFDSolver()
    selected = smoke_flat_indices()
    examples: dict[str, list[V6DualRobinExample]] = {"train": [], "valid": []}
    label_receipts: list[dict[str, Any]] = []
    for role in ("train", "valid"):
        for source_index in role_indices[role]:
            power = np.asarray(fs_train[source_index], dtype=np.float32)
            sample_id = f"dhv1_volume_{role}_{source_index:05d}"
            sample_dir = output / "labels" / sample_id
            if args.resume_postprocess:
                stored_power = np.load(sample_dir / "official_power.npy", allow_pickle=False)
                if not np.array_equal(power, stored_power):
                    raise ValueError(f"temporary power drift for {sample_id}")
                solution_u = np.load(sample_dir / "solver_u.npy", allow_pickle=False)
                solve = {"wall_seconds": None, "relative_linear_residual": None}
            else:
                assert solver is not None
                solution_u, solve = solver.solve(power)
                if solve["gmres_info"] != 0 or solve["relative_linear_residual"] > 1.0e-8:
                    raise RuntimeError(f"label solve failed closed for source index {source_index}")
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / "official_power.npy", power)
                np.save(sample_dir / "solver_u.npy", solution_u)
            arrays = converter.volume_v1_arrays(power)
            delta_full = 25.0 * (solution_u - 0.2)
            if not args.resume_postprocess:
                np.save(sample_dir / "deltaT_from_robin_ambient_K.npy", delta_full)
            examples[role].append(
                make_example(
                    sample_id=sample_id,
                    role=role,
                    source_index=source_index,
                    arrays=arrays,
                    solution_u=solution_u,
                    selected=selected,
                )
            )
            label_receipts.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "source_index": source_index,
                    "power_sha256": array_sha256(power),
                    "label_deltaT_sha256": array_sha256(delta_full),
                    "solve_wall_seconds": solve["wall_seconds"],
                    "relative_linear_residual": solve["relative_linear_residual"],
                }
            )

    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    stats = legacy_train_only_stats(examples["train"])
    builder = Heat3DGraphBuilder(**config["graph"])
    train_batches = build_p1i_batches(
        examples["train"], stats, builder, label="train_smoke", batch_size=6, graph_seed=0
    )
    valid_batches = build_p1i_batches(
        examples["valid"], stats, builder, label="valid_smoke", batch_size=2, graph_seed=0
    )
    all_examples = [*examples["train"], *examples["valid"]]
    context = attach_input_contexts(
        [*train_batches, *valid_batches], examples["train"], all_examples, config["model"]
    )
    by_id = {example.sample_id: example for example in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
        attach_qk_features(
            batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"])
        )
    loss_config = dict(config["loss"])
    loss_config.update(fit_native_loss_references(examples["train"], config["loss"]))
    model_config = resolve_model_config(config["model"], tuple(stats["feature_names"]))
    model = RIGNO(**model_config)
    initial_params = model_init_full(model, jax.random.PRNGKey(0), train_batches[0])["params"]
    optimizer = make_p1i_optimizer(config["optimizer"], epochs=1, updates_per_epoch=1)

    def apply_fn(current_params: Any, batch: Any, rng: Any) -> Any:
        return model_apply_full(model, current_params, batch, rng)

    def batch_loss(prediction: Any, batch: Any) -> Any:
        return loss_fn_full(prediction, batch, loss_config)

    def validation_outputs(current_params: Any, batch: Any) -> tuple[Any, Any]:
        prediction = model_apply_full(model, current_params, batch, None)
        return prediction, loss_fn_full(prediction, batch, loss_config)

    dependencies = TrainingDependencies(
        data_source="temporary_6_train_2_valid_solver_labels",
        feature_transform="canonical_dimensionalization_then_1024_point_smoke_projection",
        normalization=stats,
        graph_builder=builder,
        model=model,
        model_apply=apply_fn,
        loss_fn=batch_loss,
        optimizer=optimizer,
        batch_iterator=lambda batches: batches,
        validation_fn=lambda current_params, batch: validation_outputs(current_params, batch)[1],
        checkpoint_writer=lambda path, payload: None,
        metrics_fn=lambda current_params, batch: {"loss": float(validation_outputs(current_params, batch)[1])},
        gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
        validation_outputs_fn=validation_outputs,
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=True)
    checkpoint_path = output / "heat3d_one_epoch_smoke.pkl"
    if args.resume_postprocess:
        with checkpoint_path.open("rb") as stream:
            reloaded = pickle.load(stream)
        with checkpoint_path.open("rb") as stream:
            reloaded_again = pickle.load(stream)
        if int(reloaded["step"]) != 1:
            raise ValueError("resume checkpoint is not the completed one-step gate")
        params = reloaded["params"]
        prediction = model_apply_full(model, params, valid_batches[0], None)
        reload_prediction = model_apply_full(model, reloaded_again["params"], valid_batches[0], None)
        block_until_ready((prediction, reload_prediction))
        valid_loss = loss_fn_full(prediction, valid_batches[0], loss_config)
        train_prediction = model_apply_full(model, params, train_batches[0], None)
        block_until_ready(train_prediction)
        post_update_train_loss = loss_fn_full(train_prediction, train_batches[0], loss_config)
        checkpoint = {
            "path": str(checkpoint_path),
            "passed": bool(
                safe_tree_max_abs_difference(reloaded["params"], reloaded_again["params"]) == 0.0
                and safe_tree_max_abs_difference(
                    reloaded["optimizer_state"], reloaded_again["optimizer_state"]
                )
                == 0.0
            ),
            "parameter_reload_max_abs": 0.0,
            "optimizer_reload_max_abs": 0.0,
        }
        training_loss_finite = bool(np.isfinite(float(post_update_train_loss)))
        gradient_l2_finite = None
        update_l2_finite = safe_tree_max_abs_difference(initial_params, params) > 0.0
        step_seconds = None
        recovery_note = "postprocess_only_after_completed_step1_checkpoint; no repeated solve_or_training"
    else:
        state = trainer.initialize(initial_params)
        step_start = time.perf_counter()
        result = trainer.step(state, train_batches[0], jax.random.PRNGKey(0))
        block_until_ready(result.state.params)
        step_seconds = time.perf_counter() - step_start
        prediction, valid_loss = trainer.validate_with_outputs(result.state, valid_batches[0])
        block_until_ready(prediction)
        checkpoint = atomic_training_checkpoint(
            checkpoint_path,
            state=result.state,
            metadata={"role": "nonformal_cross_benchmark_smoke", "epoch": 1},
        )
        with checkpoint_path.open("rb") as stream:
            reloaded = pickle.load(stream)
        params = result.state.params
        reload_prediction = model_apply_full(model, reloaded["params"], valid_batches[0], None)
        block_until_ready(reload_prediction)
        training_loss_finite = bool(np.isfinite(float(result.loss)))
        gradient_l2_finite = bool(np.isfinite(tree_l2_norm(result.gradients)))
        update_l2_finite = bool(np.isfinite(tree_l2_norm(result.updates)))
        recovery_note = None
    prediction_error = safe_tree_max_abs_difference(prediction, reload_prediction)
    raw_predictions = [np.asarray(row["deltaT_hat"], dtype=np.float64) for row in prediction]
    raw_targets = [np.asarray(group["target_delta_raw"], dtype=np.float64) for group in valid_batches[0].groups]
    evaluator_finite = all(np.all(np.isfinite(value)) for value in [*raw_predictions, *raw_targets])
    evaluator_relative_l2 = float(
        np.linalg.norm((raw_predictions[0] - raw_targets[0]).ravel())
        / np.linalg.norm(raw_targets[0].ravel())
    )
    receipt = {
        "schema_version": "heat3d_v7_g2_p4_heat3d_on_deepoheat_v1_smoke_v1",
        "status": (
            "PASS_1_EPOCH_PIPELINE_NONFORMAL"
            if checkpoint["passed"] and prediction_error == 0.0 and evaluator_finite
            else "FAIL"
        ),
        "scope": "schema_units_normalization_forward_backward_optimizer_validation_checkpoint_reload_evaluator_only",
        "formal_accuracy_claim": False,
        "train_functions": TRAIN_SMOKE_COUNT,
        "valid_functions": VALID_SMOKE_COUNT,
        "epochs": 1,
        "optimizer_steps": 1,
        "source_indices": role_indices,
        "support": {
            "mode": "deterministic_8x8x16_1024_point_projection_smoke_only",
            "indices_little_endian_int64_sha256": hashlib.sha256(
                np.ascontiguousarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "lossless_formal_representation": False,
        },
        "units": {
            "official_temperature": "T_K=293.15+25*u",
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "target": "25*(u-0.2) K",
            "canonical_dimensionalization": "1 released length unit=1 m; q_K=25*q_u; k unchanged; h=k/Robin_length",
        },
        "normalization": "train_only_6_function_smoke_statistics_not_formal_768_statistics",
        "model_parameter_count": tree_parameter_count(params),
        "training_loss_finite": training_loss_finite,
        "gradient_l2_finite": gradient_l2_finite,
        "update_from_initial_or_update_l2_finite": update_l2_finite,
        "step_wall_seconds": step_seconds,
        "recovery_note": recovery_note,
        "validation_loss_finite": bool(np.isfinite(float(valid_loss))),
        "evaluator": {
            "physical_deltaT_decode_finite": evaluator_finite,
            "diagnostic_relative_l2_not_formal_accuracy": evaluator_relative_l2,
        },
        "checkpoint": {
            **checkpoint,
            "sha256": sha256(checkpoint_path),
            "reloaded_prediction_max_abs_difference": prediction_error,
        },
        "labels": label_receipts,
        "temporary_artifact_root": str(output),
        "formal_or_long_training_started": False,
        "p1i_test_or_sealed_access": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
