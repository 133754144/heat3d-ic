"""Controlled Heat3D v1 medium training export smoke.

This runner reuses the existing v1 train/valid smoke path and writes recovered
temperature predictions to an ignored output directory for downstream
diagnostic comparison. It is not a formal training experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    MODEL_CONFIG,
    _global_norm,
    _make_groups,
    _metrics,
    _sample_root,
    _selected_steps,
    _subset_split_ids,
    _train_only_stats,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_v2"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "output" / "heat3d_v1_medium_runs" / "export_smoke_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled training export smoke for Heat3D v1 medium labels. "
            "Writes ignored predictions for diagnostic comparison only."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--loss-mode", choices=("mse", "background_hotspot", "background_l1_bias"), default="mse")
    parser.add_argument("--background-quantile", type=float, default=0.50)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--hotspot-weight", type=float, default=0.1)
    parser.add_argument("--background-l1-weight", type=float, default=1.0)
    parser.add_argument("--background-bias-weight", type=float, default=1.0)
    parser.add_argument("--background-over-weight", type=float, default=1.0)
    return parser.parse_args()


def _ensure_ignored_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (REPO_DIR / "output").resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError(f"--output-dir must be under ignored output/: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_train_valid_splits(split_ids: dict[str, list[str]]) -> None:
    train_ids = split_ids.get("train", [])
    valid_ids = split_ids.get("valid", [])
    if not train_ids or not valid_ids:
        raise ValueError(
            "Expected non-empty train and valid splits for controlled training export, "
            f"found train={len(train_ids)} valid={len(valid_ids)}"
        )


def _should_report_epoch(epoch: int, epochs: int, report_every: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % report_every == 0


def _loss_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "loss_mode": args.loss_mode,
        "background_quantile": float(args.background_quantile),
        "hotspot_quantile": float(args.hotspot_quantile),
        "background_weight": float(args.background_weight),
        "hotspot_weight": float(args.hotspot_weight),
        "background_l1_weight": float(args.background_l1_weight),
        "background_bias_weight": float(args.background_bias_weight),
        "background_over_weight": float(args.background_over_weight),
        "loss_space": (
            "base and hotspot terms use normalized_deltaT; background MSE/L1/bias/overprediction "
            "terms use raw_deltaT_K"
        ),
        "base_loss_space": "normalized_deltaT",
        "background_mask_space": "raw_deltaT_K quantile",
        "background_penalty_space": "raw_deltaT_K_squared; penalizes pred_raw_deltaT toward 0",
        "background_l1_space": "raw_deltaT_K_abs; penalizes abs(pred_raw_deltaT) in background",
        "background_signed_bias_loss_space": "raw_deltaT_K_abs_bias; penalizes abs(mean(pred_raw_deltaT - true_raw_deltaT))",
        "background_overprediction_loss_space": "raw_deltaT_K_positive_error; penalizes mean(relu(pred_raw_deltaT - true_raw_deltaT))",
        "hotspot_mask_space": "raw_deltaT_K quantile",
        "hotspot_retention_loss_space": "normalized_deltaT",
        "target_normalization": "normalized_deltaT = (raw_deltaT - train_target_delta_mean) / train_target_delta_std",
    }


def _validate_loss_config(config: dict[str, Any]) -> None:
    background_quantile = float(config["background_quantile"])
    hotspot_quantile = float(config["hotspot_quantile"])
    if not 0.0 <= background_quantile <= 1.0:
        raise ValueError("--background-quantile must be in [0, 1]")
    if not 0.0 <= hotspot_quantile <= 1.0:
        raise ValueError("--hotspot-quantile must be in [0, 1]")
    if background_quantile > hotspot_quantile:
        raise ValueError("--background-quantile must be <= --hotspot-quantile")
    if float(config["background_weight"]) < 0.0:
        raise ValueError("--background-weight must be >= 0")
    if float(config["hotspot_weight"]) < 0.0:
        raise ValueError("--hotspot-weight must be >= 0")
    if float(config["background_l1_weight"]) < 0.0:
        raise ValueError("--background-l1-weight must be >= 0")
    if float(config["background_bias_weight"]) < 0.0:
        raise ValueError("--background-bias-weight must be >= 0")
    if float(config["background_over_weight"]) < 0.0:
        raise ValueError("--background-over-weight must be >= 0")


def _masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _normalized_delta_to_raw(pred_normalized, stats: dict):
    return pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]


def _loss_components(model, params, groups: list[dict], stats: dict, loss_config: dict[str, Any]) -> dict[str, Any]:
    weighted = {
        "base_mse": 0.0,
        "background_penalty": 0.0,
        "background_l1": 0.0,
        "background_signed_bias_loss": 0.0,
        "background_overprediction_loss": 0.0,
        "hotspot_retention_loss": 0.0,
        "total_loss": 0.0,
        "bg_pred_raw_mean": 0.0,
        "bg_signed_bias": 0.0,
        "bg_abs_mean": 0.0,
        "hotspot_raw_mae": 0.0,
    }
    count = 0
    for group in groups:
        pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        target = group["target_normalized"]
        target_raw = group["target_delta_raw"]
        pred_raw_delta = _normalized_delta_to_raw(pred, stats)
        base_mse = jnp.mean(jnp.square(pred - target))
        background_threshold = jnp.quantile(target_raw, loss_config["background_quantile"])
        hotspot_threshold = jnp.quantile(target_raw, loss_config["hotspot_quantile"])
        background_mask = target_raw <= background_threshold
        hotspot_mask = target_raw >= hotspot_threshold
        background_penalty = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_l1 = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_signed_bias_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_overprediction_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        hotspot_retention_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        raw_error = pred_raw_delta - target_raw
        if loss_config["loss_mode"] == "background_hotspot":
            background_penalty = _masked_mean(jnp.square(pred_raw_delta), background_mask)
            hotspot_retention_loss = _masked_mean(jnp.square(pred - target), hotspot_mask)
            total_loss = (
                base_mse
                + loss_config["background_weight"] * background_penalty
                + loss_config["hotspot_weight"] * hotspot_retention_loss
            )
        elif loss_config["loss_mode"] == "background_l1_bias":
            background_l1 = _masked_mean(jnp.abs(pred_raw_delta), background_mask)
            background_signed_bias_loss = jnp.abs(_masked_mean(raw_error, background_mask))
            background_overprediction_loss = _masked_mean(jnp.maximum(raw_error, 0.0), background_mask)
            hotspot_retention_loss = _masked_mean(jnp.square(pred - target), hotspot_mask)
            total_loss = (
                base_mse
                + loss_config["background_l1_weight"] * background_l1
                + loss_config["background_bias_weight"] * background_signed_bias_loss
                + loss_config["background_over_weight"] * background_overprediction_loss
                + loss_config["hotspot_weight"] * hotspot_retention_loss
            )
        else:
            total_loss = base_mse
        bg_pred_raw_mean = _masked_mean(pred_raw_delta, background_mask)
        bg_signed_bias = _masked_mean(raw_error, background_mask)
        bg_abs_mean = _masked_mean(jnp.abs(raw_error), background_mask)
        hotspot_raw_mae = _masked_mean(jnp.abs(raw_error), hotspot_mask)
        n = target.shape[0]
        weighted["base_mse"] = weighted["base_mse"] + base_mse * n
        weighted["background_penalty"] = weighted["background_penalty"] + background_penalty * n
        weighted["background_l1"] = weighted["background_l1"] + background_l1 * n
        weighted["background_signed_bias_loss"] = (
            weighted["background_signed_bias_loss"] + background_signed_bias_loss * n
        )
        weighted["background_overprediction_loss"] = (
            weighted["background_overprediction_loss"] + background_overprediction_loss * n
        )
        weighted["hotspot_retention_loss"] = weighted["hotspot_retention_loss"] + hotspot_retention_loss * n
        weighted["total_loss"] = weighted["total_loss"] + total_loss * n
        weighted["bg_pred_raw_mean"] = weighted["bg_pred_raw_mean"] + bg_pred_raw_mean * n
        weighted["bg_signed_bias"] = weighted["bg_signed_bias"] + bg_signed_bias * n
        weighted["bg_abs_mean"] = weighted["bg_abs_mean"] + bg_abs_mean * n
        weighted["hotspot_raw_mae"] = weighted["hotspot_raw_mae"] + hotspot_raw_mae * n
        count += int(n)
    divisor = max(count, 1)
    return {key: value / divisor for key, value in weighted.items()}


def _loss_components_payload(components: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in components.items()}


def _epoch_history_record(
    epoch: int,
    train_components: dict[str, Any],
    valid_components: dict[str, Any],
    valid_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_components["total_loss"]),
        "valid_loss": float(valid_components["total_loss"]),
        "train_base_mse": float(train_components["base_mse"]),
        "valid_base_mse": float(valid_components["base_mse"]),
        "train_background_penalty": float(train_components["background_penalty"]),
        "valid_background_penalty": float(valid_components["background_penalty"]),
        "train_background_l1": float(train_components["background_l1"]),
        "valid_background_l1": float(valid_components["background_l1"]),
        "train_background_signed_bias_loss": float(train_components["background_signed_bias_loss"]),
        "valid_background_signed_bias_loss": float(valid_components["background_signed_bias_loss"]),
        "train_background_overprediction_loss": float(train_components["background_overprediction_loss"]),
        "valid_background_overprediction_loss": float(valid_components["background_overprediction_loss"]),
        "train_hotspot_retention_loss": float(train_components["hotspot_retention_loss"]),
        "valid_hotspot_retention_loss": float(valid_components["hotspot_retention_loss"]),
        "train_bg_pred_raw_mean": float(train_components["bg_pred_raw_mean"]),
        "valid_bg_pred_raw_mean": float(valid_components["bg_pred_raw_mean"]),
        "train_bg_signed_bias": float(train_components["bg_signed_bias"]),
        "valid_bg_signed_bias": float(valid_components["bg_signed_bias"]),
        "train_bg_abs_mean": float(train_components["bg_abs_mean"]),
        "valid_bg_abs_mean": float(valid_components["bg_abs_mean"]),
        "train_hotspot_raw_mae": float(train_components["hotspot_raw_mae"]),
        "valid_hotspot_raw_mae": float(valid_components["hotspot_raw_mae"]),
        "train_raw_deltaT_mse": float(train_metrics["raw_delta_mse"]),
        "valid_raw_deltaT_mse": float(valid_metrics["raw_delta_mse"]),
        "train_recovered_T_mse": float(train_metrics["recovered_temperature_mse"]),
        "valid_recovered_T_mse": float(valid_metrics["recovered_temperature_mse"]),
    }


def _print_epoch_progress(record: dict[str, Any], epochs: int) -> None:
    print(
        f"epoch {record['epoch']:03d}/{epochs:03d} "
        f"train_loss={record['train_loss']:.8e} "
        f"valid_loss={record['valid_loss']:.8e} "
        f"train_base_mse={record['train_base_mse']:.8e} "
        f"valid_base_mse={record['valid_base_mse']:.8e} "
        f"train_background_penalty={record['train_background_penalty']:.8e} "
        f"valid_background_penalty={record['valid_background_penalty']:.8e} "
        f"train_background_l1={record['train_background_l1']:.8e} "
        f"valid_background_l1={record['valid_background_l1']:.8e} "
        f"train_background_signed_bias_loss={record['train_background_signed_bias_loss']:.8e} "
        f"valid_background_signed_bias_loss={record['valid_background_signed_bias_loss']:.8e} "
        f"train_background_overprediction_loss={record['train_background_overprediction_loss']:.8e} "
        f"valid_background_overprediction_loss={record['valid_background_overprediction_loss']:.8e} "
        f"train_hotspot_retention_loss={record['train_hotspot_retention_loss']:.8e} "
        f"valid_hotspot_retention_loss={record['valid_hotspot_retention_loss']:.8e} "
        f"train_bg_pred_raw_mean={record['train_bg_pred_raw_mean']:.8e} "
        f"valid_bg_pred_raw_mean={record['valid_bg_pred_raw_mean']:.8e} "
        f"train_bg_signed_bias={record['train_bg_signed_bias']:.8e} "
        f"valid_bg_signed_bias={record['valid_bg_signed_bias']:.8e} "
        f"train_bg_abs_mean={record['train_bg_abs_mean']:.8e} "
        f"valid_bg_abs_mean={record['valid_bg_abs_mean']:.8e} "
        f"train_hotspot_raw_mae={record['train_hotspot_raw_mae']:.8e} "
        f"valid_hotspot_raw_mae={record['valid_hotspot_raw_mae']:.8e} "
        f"train_raw_deltaT_mse={record['train_raw_deltaT_mse']:.8e} "
        f"valid_raw_deltaT_mse={record['valid_raw_deltaT_mse']:.8e} "
        f"train_recovered_T_mse={record['train_recovered_T_mse']:.8e} "
        f"valid_recovered_T_mse={record['valid_recovered_T_mse']:.8e}"
    )


def _fit_once(
    train_groups: list[dict],
    valid_groups: list[dict],
    stats: dict,
    epochs: int,
    lr: float,
    seed: int,
    report_every: int,
    loss_config: dict[str, Any],
) -> dict:
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=train_groups[0]["inputs"],
        graphs=train_groups[0]["graphs"],
    )["params"]

    def loss_fn(current_params):
        return _loss_components(model, current_params, train_groups, stats, loss_config)["total_loss"]

    train_initial_components = _loss_components(model, params, train_groups, stats, loss_config)
    valid_initial_components = _loss_components(model, params, valid_groups, stats, loss_config)
    train_losses = [float(train_initial_components["total_loss"])]
    valid_losses = [float(valid_initial_components["total_loss"])]
    grad_norms = []
    grad_finite = True
    epoch_history = []
    for epoch in range(1, epochs + 1):
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        train_components = _loss_components(model, params, train_groups, stats, loss_config)
        valid_components = _loss_components(model, params, valid_groups, stats, loss_config)
        valid_metrics = _metrics(model, params, valid_groups, stats)
        train_losses.append(float(train_components["total_loss"]))
        valid_losses.append(float(valid_components["total_loss"]))
        if _should_report_epoch(epoch, epochs, report_every):
            train_metrics = _metrics(model, params, train_groups, stats)
            record = _epoch_history_record(epoch, train_components, valid_components, valid_metrics, train_metrics)
            epoch_history.append(record)
            _print_epoch_progress(record, epochs)

    train_metrics = _metrics(model, params, train_groups, stats)
    valid_metrics = _metrics(model, params, valid_groups, stats)
    final_train_components = _loss_components(model, params, train_groups, stats, loss_config)
    final_valid_components = _loss_components(model, params, valid_groups, stats, loss_config)
    status_ok = (
        grad_finite
        and train_metrics["finite_ok"]
        and valid_metrics["finite_ok"]
        and train_metrics["shape_ok"]
        and valid_metrics["shape_ok"]
        and bool(np.all(np.isfinite(train_losses)))
        and bool(np.all(np.isfinite(valid_losses)))
    )
    return {
        "model": model,
        "params": params,
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "valid_losses": np.asarray(valid_losses, dtype=np.float64),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "epoch_history": epoch_history,
        "final_train_loss_components": _loss_components_payload(final_train_components),
        "final_valid_loss_components": _loss_components_payload(final_valid_components),
        "grad_finite": grad_finite,
        "status_ok": status_ok,
    }


def _predict_temperatures(model, params, groups: list[dict], stats: dict) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        recovered = np.asarray(group["t_ref"] + pred_delta)
        if not np.all(np.isfinite(recovered)):
            raise ValueError(f"Non-finite recovered predictions in group {group['name']}")
        for batch_index, sample_id in enumerate(group["sample_ids"]):
            predictions[sample_id] = recovered[batch_index, 0, :, :].astype(np.float64)
    return predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _stats_payload(stats: dict) -> dict[str, Any]:
    return {
        "feature_names": list(stats["feature_names"]),
        "target_delta_mean": float(stats["target_delta_mean"].reshape(-1)[0]),
        "target_delta_std": float(stats["target_delta_std"].reshape(-1)[0]),
        "condition_mean": [float(value) for value in stats["condition_mean"].reshape(-1)],
        "condition_std": [float(value) for value in stats["condition_std"].reshape(-1)],
    }


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
        for key, value in metrics.items()
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")
    loss_config = _loss_config_from_args(args)
    _validate_loss_config(loss_config)

    output_dir = _ensure_ignored_output_dir(args.output_dir)
    sample_root = _sample_root(args.subset)
    split_ids = _subset_split_ids(sample_root)
    _require_train_valid_splits(split_ids)

    all_ids = sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    train_ids = split_ids["train"]
    valid_ids = split_ids["valid"]
    for sample_id in all_ids:
        if not (sample_root / sample_id / "temperature.npy").is_file():
            raise FileNotFoundError(f"Missing temperature.npy for {sample_id}")

    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in all_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset loader did not expose samples: {missing}")

    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]
    all_examples = [dataset[index_by_id[sample_id]] for sample_id in all_ids]

    builder = Heat3DGraphBuilder()
    stats = _train_only_stats(train_examples)
    train_groups = _make_groups(train_examples, stats, builder)
    valid_groups = _make_groups(valid_examples, stats, builder)
    all_groups = _make_groups(all_examples, stats, builder)

    split_counts = {split: len(ids) for split, ids in sorted(split_ids.items())}

    print("Heat3D v1 medium controlled training export smoke")
    print("  scope: research reference diagnostics only; not formal model performance")
    print(f"  subset: {sample_root}")
    print(f"  split counts: {split_counts}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {output_dir}")
    print(f"  save_predictions: {bool(args.save_predictions)}")
    print(f"  report every: {args.report_every}")
    print(f"  loss mode: {loss_config['loss_mode']}")
    print(f"  loss space: {loss_config['loss_space']}")
    print(
        "  loss params: "
        f"background_quantile={loss_config['background_quantile']} "
        f"hotspot_quantile={loss_config['hotspot_quantile']} "
        f"background_weight={loss_config['background_weight']} "
        f"hotspot_weight={loss_config['hotspot_weight']} "
        f"background_l1_weight={loss_config['background_l1_weight']} "
        f"background_bias_weight={loss_config['background_bias_weight']} "
        f"background_over_weight={loss_config['background_over_weight']}"
    )
    print(f"  feature mode: relative BC features, diag3 k encoding, zero_delta_u_bridge")
    print(
        "  target mode: normalized DeltaT target; normalized 0 is train mean raw DeltaT, "
        "not raw DeltaT=0"
    )

    result = _fit_once(
        train_groups,
        valid_groups,
        stats,
        args.epochs,
        args.lr,
        args.seed,
        args.report_every,
        loss_config,
    )
    predictions = _predict_temperatures(result["model"], result["params"], all_groups, stats)

    run_config = {
        "diagnostic_scope": "controlled training export smoke; not formal model performance",
        "subset": str(sample_root),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "route": "relative BC features + zero_delta_u_bridge + normalized DeltaT target",
        "output_dir": str(output_dir),
        "save_predictions": bool(args.save_predictions),
        "checkpoint_saved": False,
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "loss": loss_config,
        "split_counts": split_counts,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "ignored_candidate_ids": sorted(
            sample_id for split, ids in split_ids.items() if split not in {"train", "valid"} for sample_id in ids
        ),
    }
    _write_json(output_dir / "run_config.json", run_config)

    loss_summary = {
        "status_ok": bool(result["status_ok"]),
        "grad_finite": bool(result["grad_finite"]),
        "train_losses": [float(value) for value in result["train_losses"]],
        "valid_losses": [float(value) for value in result["valid_losses"]],
        "grad_norms": [float(value) for value in result["grad_norms"]],
        "train_metrics": _metrics_payload(result["train_metrics"]),
        "valid_metrics": _metrics_payload(result["valid_metrics"]),
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "train_loss_selected": _selected_steps(result["train_losses"], args.report_every),
        "valid_loss_selected": _selected_steps(result["valid_losses"], args.report_every),
        "grad_norm_selected": _selected_steps(result["grad_norms"], args.report_every),
        "epoch_history": result["epoch_history"],
        "loss": loss_config,
        "final_train_loss_components": result["final_train_loss_components"],
        "final_valid_loss_components": result["final_valid_loss_components"],
        "train_only_normalization": _stats_payload(stats),
    }
    _write_json(output_dir / "loss_summary.json", loss_summary)

    predictions_path = output_dir / "predictions.npz"
    if args.save_predictions:
        np.savez_compressed(predictions_path, **predictions)

    print("")
    print("summary")
    print(f"  loss mode: {loss_config['loss_mode']}")
    print(f"  train loss initial/final: {result['train_losses'][0]:.8e} -> {result['train_losses'][-1]:.8e}")
    print(f"  valid loss initial/final: {result['valid_losses'][0]:.8e} -> {result['valid_losses'][-1]:.8e}")
    print(f"  final train base MSE: {result['final_train_loss_components']['base_mse']:.8e}")
    print(f"  final valid base MSE: {result['final_valid_loss_components']['base_mse']:.8e}")
    print(f"  final train background penalty: {result['final_train_loss_components']['background_penalty']:.8e}")
    print(f"  final valid background penalty: {result['final_valid_loss_components']['background_penalty']:.8e}")
    print(f"  final train background L1: {result['final_train_loss_components']['background_l1']:.8e}")
    print(f"  final valid background L1: {result['final_valid_loss_components']['background_l1']:.8e}")
    print(
        "  final train background signed bias loss: "
        f"{result['final_train_loss_components']['background_signed_bias_loss']:.8e}"
    )
    print(
        "  final valid background signed bias loss: "
        f"{result['final_valid_loss_components']['background_signed_bias_loss']:.8e}"
    )
    print(
        "  final train background overprediction loss: "
        f"{result['final_train_loss_components']['background_overprediction_loss']:.8e}"
    )
    print(
        "  final valid background overprediction loss: "
        f"{result['final_valid_loss_components']['background_overprediction_loss']:.8e}"
    )
    print(f"  final train hotspot retention loss: {result['final_train_loss_components']['hotspot_retention_loss']:.8e}")
    print(f"  final valid hotspot retention loss: {result['final_valid_loss_components']['hotspot_retention_loss']:.8e}")
    print(f"  final train bg pred raw mean: {result['final_train_loss_components']['bg_pred_raw_mean']:.8e}")
    print(f"  final valid bg pred raw mean: {result['final_valid_loss_components']['bg_pred_raw_mean']:.8e}")
    print(f"  final train bg signed bias: {result['final_train_loss_components']['bg_signed_bias']:.8e}")
    print(f"  final valid bg signed bias: {result['final_valid_loss_components']['bg_signed_bias']:.8e}")
    print(f"  final train bg abs mean: {result['final_train_loss_components']['bg_abs_mean']:.8e}")
    print(f"  final valid bg abs mean: {result['final_valid_loss_components']['bg_abs_mean']:.8e}")
    print(f"  final train hotspot raw MAE: {result['final_train_loss_components']['hotspot_raw_mae']:.8e}")
    print(f"  final valid hotspot raw MAE: {result['final_valid_loss_components']['hotspot_raw_mae']:.8e}")
    print(f"  final train raw DeltaT MSE: {result['train_metrics']['raw_delta_mse']:.8e}")
    print(f"  final valid raw DeltaT MSE: {result['valid_metrics']['raw_delta_mse']:.8e}")
    print(f"  final train recovered temperature MSE: {result['train_metrics']['recovered_temperature_mse']:.8e}")
    print(f"  final valid recovered temperature MSE: {result['valid_metrics']['recovered_temperature_mse']:.8e}")
    print(f"  gradient finite check: {result['grad_finite']}")
    print(f"  predictions saved: {bool(args.save_predictions)}")
    print(f"  predictions path: {predictions_path if args.save_predictions else 'not_written'}")
    print(f"  prediction sample count: {len(predictions)}")
    print("  checkpoint saved: False")
    print(f"  export smoke ok: {result['status_ok']}")
    return 0 if result["status_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
