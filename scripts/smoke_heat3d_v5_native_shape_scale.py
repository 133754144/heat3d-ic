#!/usr/bin/env python3
"""Read-only V5 native shape--scale RIGNO smoke on frozen P5 inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "rigno").is_dir() and (Path.cwd() / "rigno").is_dir():
    REPO_ROOT = Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import tree_util  # noqa: E402

from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    fit_train_only_standardizer,
    global_context_from_raw_condition,
    standardize_contexts,
    validate_global_context_schema,
)
from rigno.heat3d_v5_metrics import (  # noqa: E402
    control_volume_weights,
    decompose_shape_scale,
    reconstruct_shape_scale,
)
from rigno.heat3d_v5_shape_scale import native_shape_scale_losses  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _apply_checkpoint_params,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _param_leaf_items,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


SMOKE_ID = "V5-native-shape-scale-smoke"
SCHEMA_VERSION = "heat3d_v5_native_shape_scale_smoke_v1"
EXPECTED_BASELINE_ID = "V4P5_02_clean_baseline_raw_B28_e600"
EXPECTED_EPOCH = 405
LOSS_WEIGHTS = {
    "shape_cv": 1.0,
    "log_scale": 1.0,
    "relative_field": 1.0,
    "raw_absolute": 1.0,
}


class NativeSmokeError(RuntimeError):
    """Raised for native shape--scale smoke invariant violations."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument("--expected-epoch", type=int, default=EXPECTED_EPOCH)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeSmokeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeSmokeError(f"{path} must be a JSON object")
    return payload


def _ensure_output(path: Path, label: str, overwrite: bool) -> Path:
    resolved = path.resolve()
    if any(part in {"data", "output", "checkpoints", "logs"} for part in resolved.parts):
        raise NativeSmokeError(f"--{label} must not write under data/output/checkpoints/logs")
    if resolved.exists() and not overwrite:
        raise NativeSmokeError(f"--{label} exists (use --overwrite): {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_examples(
    *,
    sample_root: Path,
    sample_ids: Sequence[str],
    checkpoint_stats: Mapping[str, Any],
    boundary_mask_fallback: bool,
) -> list[Any]:
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=k_encoding_mode,
        boundary_mask_fallback=boundary_mask_fallback,
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise NativeSmokeError(f"dataset lacks sample IDs: {missing[:8]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def _raw_context(example: Any) -> dict[str, float]:
    relative = example.get_relative_bc_feature_view()
    return global_context_from_raw_condition(
        coords=np.asarray(example.condition.coords, dtype=np.float64),
        raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
        condition_feature_names=tuple(relative.condition_feature_names),
        reference_temperature_K=float(relative.t_ref_value),
    )


def _batch_physics(
    group: Mapping[str, Any],
    examples_by_id: Mapping[str, Any],
    context_rows: Mapping[str, Mapping[str, float]],
    standardizer: Mapping[str, Any],
) -> dict[str, jnp.ndarray]:
    volumes: list[np.ndarray] = []
    log_s_phys: list[float] = []
    references: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    prescribed: list[np.ndarray] = []
    ordered_contexts = []
    for sample_id in group["sample_ids"]:
        example = examples_by_id[sample_id]
        relative = example.get_relative_bc_feature_view()
        names = tuple(relative.condition_feature_names)
        values = np.asarray(relative.condition_features, dtype=np.float64)
        coords = np.asarray(example.condition.coords, dtype=np.float64)
        if "is_bottom" not in names or "bottom_T_fixed_minus_T_ref" not in names:
            raise NativeSmokeError(f"{sample_id}: raw BC features lack Dirichlet fields")
        bottom = values[:, names.index("is_bottom")] > 0.5
        if not np.any(bottom):
            raise NativeSmokeError(f"{sample_id}: no bottom Dirichlet nodes")
        bottom_offset = values[:, names.index("bottom_T_fixed_minus_T_ref")]
        t_ref = float(relative.t_ref_value)
        volumes.append(control_volume_weights(coords))
        log_s_phys.append(float(context_rows[sample_id]["log_s_phys_K"]))
        references.append(np.full(coords.shape[0], t_ref, dtype=np.float32))
        masks.append(bottom.astype(np.float32))
        prescribed.append((t_ref + bottom_offset).astype(np.float32))
        ordered_contexts.append(context_rows[sample_id])
    return {
        "control_volumes": jnp.asarray(np.stack(volumes), dtype=jnp.float32),
        "log_s_phys": jnp.asarray(log_s_phys, dtype=jnp.float32),
        "reference_temperature": jnp.asarray(np.stack(references), dtype=jnp.float32),
        "dirichlet_mask": jnp.asarray(np.stack(masks), dtype=jnp.float32),
        "prescribed_temperature": jnp.asarray(np.stack(prescribed), dtype=jnp.float32),
        "global_context": jnp.asarray(standardize_contexts(ordered_contexts, standardizer)),
    }


def _target_reconstruction_error(group: Mapping[str, Any], physics: Mapping[str, jnp.ndarray]) -> float:
    target = np.asarray(group["target_delta_raw"], dtype=np.float64)
    volumes = np.asarray(physics["control_volumes"], dtype=np.float64)
    maximum = 0.0
    for index in range(target.shape[0]):
        scale, shape = decompose_shape_scale(target[index].reshape(-1), volumes[index])
        reconstructed = reconstruct_shape_scale(scale, shape)
        maximum = max(maximum, float(np.max(np.abs(reconstructed - target[index].reshape(-1)))))
    return maximum


def _shape_cv_rms(phi_hat: Any, volumes: Any) -> np.ndarray:
    phi = np.asarray(phi_hat, dtype=np.float64)[:, 0, :, 0]
    weight = np.asarray(volumes, dtype=np.float64)
    return np.sqrt(np.sum(np.square(phi) * weight, axis=1) / np.sum(weight, axis=1))


def _tree_finite(value: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in tree_util.tree_leaves(value))


def _validate_checkpoint(path: Path, payload: Mapping[str, Any], expected_epoch: int) -> None:
    if not path.is_file():
        raise NativeSmokeError(f"checkpoint does not exist: {path}")
    if EXPECTED_BASELINE_ID not in path.as_posix():
        raise NativeSmokeError(f"checkpoint must belong to {EXPECTED_BASELINE_ID}")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise NativeSmokeError(f"checkpoint epoch differs from {expected_epoch}")


def _run(args: argparse.Namespace, output_json: Path, output_md: Path) -> dict[str, Any]:
    if args.prediction_batch_size < 1:
        raise NativeSmokeError("--prediction-batch-size must be >= 1")
    validate_global_context_schema()
    run_config = _load_json(args.run_config)
    checkpoint_payload = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, checkpoint_payload, args.expected_epoch)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise NativeSmokeError("checkpoint misses train-only normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    train_ids = list(split_ids.get("train") or ())
    valid_ids = list(split_ids.get("valid_iid") or ())
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise NativeSmokeError(f"expected train=672/valid=128, found {len(train_ids)}/{len(valid_ids)}")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    train_contexts = {example.sample_id: _raw_context(example) for example in train_examples}
    valid_contexts = {example.sample_id: _raw_context(example) for example in valid_examples}
    standardizer = fit_train_only_standardizer(
        [train_contexts[sample_id] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    if standardizer["fit_sample_count"] != 672:
        raise NativeSmokeError("native global context standardizer did not fit train only")
    examples_by_id = {example.sample_id: example for example in valid_examples}

    backbone = _resolve_decoder_bypass_model_config(dict(checkpoint_payload.get("model_config") or {}), stats)
    native_config = dict(backbone)
    native_config.update(
        {
            "decoder_bypass_mode": "none",
            "decoder_bypass_features": "none",
            "decoder_bypass_feature_indices": (),
            "decoder_bypass_feature_names": (),
            "decoder_bypass_num_features": 0,
            "global_context_mode": "none",
            "global_context_feature_dim": len(GLOBAL_CONTEXT_FEATURES),
            "global_context_feature_names": tuple(GLOBAL_CONTEXT_FEATURES),
            "film_target": "rnodes_processed",
            "film_init": "identity",
            "film_hidden_size": 64,
            "native_output_mode": "native_shape_scale",
            "shape_scale_epsilon": 1.0e-12,
            "scale_head_hidden_size": 64,
            "scale_head_init": "identity",
        }
    )
    builder = Heat3DGraphBuilder(**dict(run_config.get("graph_config") or {}))
    graph_seed = int(run_config.get("graph_seed", 0))
    groups4 = _make_groups_with_progress(
        valid_examples[:4], stats, builder, "v5_native_shape_scale_B4", False, "basic", graph_seed,
        batch_size=4, drop_last=False,
    )
    groups1 = _make_groups_with_progress(
        valid_examples[:1], stats, builder, "v5_native_shape_scale_B1", False, "basic", graph_seed,
        batch_size=1, drop_last=False,
    )
    if len(groups4) != 1 or len(groups1) != 1:
        raise NativeSmokeError("native smoke expected one B4 and one B1 group")
    group4, group1 = groups4[0], groups1[0]
    physics4 = _batch_physics(group4, examples_by_id, valid_contexts, standardizer)
    physics1 = _batch_physics(group1, examples_by_id, valid_contexts, standardizer)
    model = GraphNeuralOperator(**native_config)
    initial_params = model.init(
        jax.random.PRNGKey(20260712),
        inputs=group4["inputs"],
        graphs=group4["graphs"],
        control_volumes=physics4["control_volumes"],
        log_s_phys=physics4["log_s_phys"],
        reference_temperature=physics4["reference_temperature"],
        dirichlet_mask=physics4["dirichlet_mask"],
        prescribed_temperature=physics4["prescribed_temperature"],
        global_context=physics4["global_context"],
        method=model.predict_native_shape_scale,
    )["params"]
    params, load_info = _apply_checkpoint_params(
        initial_params,
        checkpoint_payload["params"],
        strict=False,
        partial_load_policy="encoder_processor_only",
    )
    initial_items = {path: np.asarray(value) for path, value in _param_leaf_items(initial_params)}
    loaded_items = {path: np.asarray(value) for path, value in _param_leaf_items(params)}
    frozen_items = {path: np.asarray(value) for path, value in _param_leaf_items(checkpoint_payload["params"])}
    encoder_processor_paths = [
        path for path in initial_items if path.startswith("encoder/") or path.startswith("processor/")
    ]
    if not encoder_processor_paths or any(
        path not in frozen_items or not np.array_equal(loaded_items[path], frozen_items[path])
        for path in encoder_processor_paths
    ):
        raise NativeSmokeError("native warm-start must load encoder/processor only")
    decoder_paths = [path for path in initial_items if path.startswith("decoder/")]
    if not decoder_paths or any(not np.array_equal(loaded_items[path], initial_items[path]) for path in decoder_paths):
        raise NativeSmokeError("native shape decoder must remain newly initialized")
    if not any(path.startswith("decoder/") for path in load_info["skipped_keys"]):
        raise NativeSmokeError("native shape decoder was not kept newly initialized")

    prediction4 = model.apply(
        {"params": params},
        inputs=group4["inputs"],
        graphs=group4["graphs"],
        control_volumes=physics4["control_volumes"],
        log_s_phys=physics4["log_s_phys"],
        reference_temperature=physics4["reference_temperature"],
        dirichlet_mask=physics4["dirichlet_mask"],
        prescribed_temperature=physics4["prescribed_temperature"],
        global_context=physics4["global_context"],
        method=model.predict_native_shape_scale,
    )
    prediction1 = model.apply(
        {"params": params},
        inputs=group1["inputs"],
        graphs=group1["graphs"],
        control_volumes=physics1["control_volumes"],
        log_s_phys=physics1["log_s_phys"],
        reference_temperature=physics1["reference_temperature"],
        dirichlet_mask=physics1["dirichlet_mask"],
        prescribed_temperature=physics1["prescribed_temperature"],
        global_context=physics1["global_context"],
        method=model.predict_native_shape_scale,
    )
    shape_rms = _shape_cv_rms(prediction4["phi_hat"], physics4["control_volumes"])
    if not np.allclose(shape_rms, 1.0, rtol=0.0, atol=1.0e-5):
        raise NativeSmokeError(f"native phi_hat does not have unit CV-RMS: {shape_rms}")
    if not bool(np.all(np.asarray(prediction4["s_hat"]) > 0.0)):
        raise NativeSmokeError("native s_hat is not strictly positive")
    reconstruction_error = float(np.max(np.abs(
        np.asarray(prediction4["deltaT_hat_unprojected"])
        - np.asarray(prediction4["s_hat"]) * np.asarray(prediction4["phi_hat"])
    )))
    if reconstruction_error > 1.0e-6:
        raise NativeSmokeError(f"native DeltaT=s*phi reconstruction drift {reconstruction_error:g}")
    raw_temperature = np.asarray(prediction4["raw_temperature"])
    raw_unprojected = np.asarray(prediction4["raw_temperature_unprojected"])
    mask = np.asarray(physics4["dirichlet_mask"]) > 0.5
    prescribed = np.asarray(physics4["prescribed_temperature"])
    projected_error = float(np.max(np.abs(raw_temperature[:, 0, :, 0][mask] - prescribed[mask])))
    if projected_error > 1.0e-6:
        raise NativeSmokeError(f"raw Dirichlet projection drift {projected_error:g} K")
    non_dirichlet = ~mask
    non_dirichlet_error = float(np.max(np.abs(
        raw_temperature[:, 0, :, 0][non_dirichlet] - raw_unprojected[:, 0, :, 0][non_dirichlet]
    )))
    if non_dirichlet_error > 1.0e-6:
        raise NativeSmokeError("Dirichlet projection changed non-Dirichlet nodes")
    target_reconstruction_error = _target_reconstruction_error(group4, physics4)
    if target_reconstruction_error > 1.0e-10:
        raise NativeSmokeError(f"target shape/scale reconstruction drift {target_reconstruction_error:g}")
    losses = native_shape_scale_losses(
        prediction4,
        target_deltaT=group4["target_delta_raw"],
        control_volumes=physics4["control_volumes"],
        loss_weights=LOSS_WEIGHTS,
    )

    def _loss_for_grad(current_params):
        predicted = model.apply(
            {"params": current_params},
            inputs=group4["inputs"],
            graphs=group4["graphs"],
            control_volumes=physics4["control_volumes"],
            log_s_phys=physics4["log_s_phys"],
            reference_temperature=physics4["reference_temperature"],
            dirichlet_mask=physics4["dirichlet_mask"],
            prescribed_temperature=physics4["prescribed_temperature"],
            global_context=physics4["global_context"],
            method=model.predict_native_shape_scale,
        )
        return native_shape_scale_losses(
            predicted,
            target_deltaT=group4["target_delta_raw"],
            control_volumes=physics4["control_volumes"],
            loss_weights=LOSS_WEIGHTS,
        )["total_loss"]

    gradient_loss, gradients = jax.value_and_grad(_loss_for_grad)(params)
    if not math.isfinite(float(gradient_loss)) or not _tree_finite(gradients):
        raise NativeSmokeError("native shape-scale gradient is non-finite")
    payload = {
        "smoke_id": SMOKE_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "native_shape_scale_checkpoint_partial_load_smoke",
        "baseline": {
            "config_id": EXPECTED_BASELINE_ID,
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        },
        "dataset": {
            "split_source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "hard_roles_used": [],
        },
        "architecture": {
            "native_output_mode": "native_shape_scale",
            "decoder_output": "unnormalized_psi",
            "shape_normalization": "CV_RMS(phi_hat)=1 per sample",
            "scale_formula": "log(s_hat)=log(s_phys)+residual_scale(global_context)",
            "global_context_feature_count": len(GLOBAL_CONTEXT_FEATURES),
            "global_context_mode": "none_for_film_scale_head_uses_context",
            "target_or_label_derived_inference_inputs": False,
            "warm_start_policy": "encoder_processor_only",
            "loss_weights": LOSS_WEIGHTS,
        },
        "checks": {
            "target_decomposition_reconstruction_max_abs_error_K": target_reconstruction_error,
            "predicted_shape_cv_rms": shape_rms.tolist(),
            "predicted_shape_cv_rms_pass": bool(np.allclose(shape_rms, 1.0, rtol=0.0, atol=1.0e-5)),
            "s_hat_positive": True,
            "native_reconstruction_max_abs_error_K": reconstruction_error,
            "dirichlet_projection_max_abs_error_K": projected_error,
            "non_dirichlet_projection_change_max_abs_error_K": non_dirichlet_error,
            "batch4_prediction_shape": list(np.asarray(prediction4["deltaT_hat"]).shape),
            "batch1_prediction_shape": list(np.asarray(prediction1["deltaT_hat"]).shape),
            "gradient_loss": float(gradient_loss),
            "gradient_finite": True,
            "loss_components": {key: float(value) for key, value in losses.items() if key not in {"target_scale", "target_shape", "s_hat_positive"}},
            "checkpoint_partial_load": load_info,
            "training_runs": 0,
            "checkpoint_writes": 0,
            "data_writes": 0,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _render_markdown(payload: Mapping[str, Any]) -> str:
    checks = payload["checks"]
    return "\n".join(
        [
            "# V5 Native Shape--Scale Smoke",
            "",
            "## Result",
            "",
            f"- Frozen source: `{payload['baseline']['config_id']}` epoch `{payload['baseline']['checkpoint_epoch']}`; only encoder/processor parameters were loaded.",
            "- New shape decoder emits unnormalized `psi`; `phi_hat` is normalized independently per sample by control-volume RMS.",
            "- Scale head uses inference-only global context and starts with zero residual around `log(s_phys)`.",
            f"- Target decomposition/reconstruction max error: `{checks['target_decomposition_reconstruction_max_abs_error_K']:.6g} K`.",
            f"- Native reconstruction max error before projection: `{checks['native_reconstruction_max_abs_error_K']:.6g} K`; Dirichlet projection max error: `{checks['dirichlet_projection_max_abs_error_K']:.6g} K`.",
            f"- B4 output shape: `{checks['batch4_prediction_shape']}`; B1 output shape: `{checks['batch1_prediction_shape']}`; finite gradient: `{checks['gradient_finite']}`.",
            "",
            "No target-derived value is accepted by the native prediction API. Targets are used only after inference for the four configured loss terms: shape CV, log scale, relative field, and raw absolute field.",
            "",
        ]
    )


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    run_config = _load_json(args.run_config)
    checkpoint = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, checkpoint, args.expected_epoch)
    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    splits, source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    return {
        "smoke_id": SMOKE_ID,
        "mode": "dry_run",
        "split_source": source,
        "roles": {"train": len(splits.get("train", [])), "valid_iid": len(splits.get("valid_iid", []))},
        "training_runs": 0,
        "planned_writes": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.run_config.is_file():
            raise NativeSmokeError(f"run config does not exist: {args.run_config}")
        if args.dry_run:
            print(json.dumps(_dry_run(args), indent=2, sort_keys=True))
            return 0
        if args.output_json is None or args.output_md is None:
            raise NativeSmokeError("smoke requires --output-json and --output-md")
        output_json = _ensure_output(args.output_json, "output-json", args.overwrite)
        output_md = _ensure_output(args.output_md, "output-md", args.overwrite)
        payload = _run(args, output_json, output_md)
    except (NativeSmokeError, ValueError, OSError) as exc:
        print(f"native shape-scale smoke error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"smoke_id": SMOKE_ID, "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
