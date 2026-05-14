"""Controlled Heat3D v1 medium training export smoke.

This runner reuses the existing v1 train/valid smoke path and writes recovered
temperature predictions to an ignored output directory for downstream
diagnostic comparison. It is not a formal training experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
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
    _bridge_for,
    _global_norm,
    _make_batch_group,
    _metadata_shape_signature,
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
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine", "two_stage"), default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--second-stage-epoch", type=int, default=0)
    parser.add_argument("--second-stage-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("valid_loss", "valid_raw_deltaT_mse", "valid_base_mse"),
        default="valid_loss",
        help="Validation metric used to track the best epoch for optional best prediction export.",
    )
    parser.add_argument("--save-best-predictions", action="store_true")
    parser.add_argument("--best-predictions-name", type=str, default="best_predictions.npz")
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--log-mode", choices=("compact", "full", "quiet"), default="compact")
    parser.add_argument("--progress-log", dest="progress_log", action="store_true", default=True)
    parser.add_argument("--no-progress-log", dest="progress_log", action="store_false")
    parser.add_argument("--progress-detail", choices=("off", "basic", "verbose"), default="basic")
    parser.add_argument(
        "--loss-mode",
        choices=("mse", "background_hotspot", "background_l1_bias", "background_l1_relative"),
        default="mse",
    )
    parser.add_argument("--background-quantile", type=float, default=0.50)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--hotspot-weight", type=float, default=0.1)
    parser.add_argument("--background-l1-weight", type=float, default=1.0)
    parser.add_argument("--background-bias-weight", type=float, default=1.0)
    parser.add_argument("--background-over-weight", type=float, default=1.0)
    parser.add_argument("--background-relative-weight", type=float, default=0.0)
    parser.add_argument("--relative-floor", type=float, default=0.02)
    parser.add_argument("--relative-floor-mode", choices=("fixed", "p50", "p75"), default="fixed")
    parser.add_argument("--loss-weight-schedule", choices=("constant", "two_phase", "linear_anneal"), default="constant")
    parser.add_argument("--loss-transition-epoch", type=int, default=0)
    parser.add_argument("--background-relative-weight-start", type=float, default=None)
    parser.add_argument("--background-relative-weight-end", type=float, default=None)
    parser.add_argument("--hotspot-weight-start", type=float, default=None)
    parser.add_argument("--hotspot-weight-end", type=float, default=None)
    parser.add_argument("--background-l1-weight-start", type=float, default=None)
    parser.add_argument("--background-l1-weight-end", type=float, default=None)
    parser.add_argument("--background-bias-weight-start", type=float, default=None)
    parser.add_argument("--background-bias-weight-end", type=float, default=None)
    parser.add_argument("--background-over-weight-start", type=float, default=None)
    parser.add_argument("--background-over-weight-end", type=float, default=None)
    return parser.parse_args()


def _emit(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _progress_enabled(args: argparse.Namespace) -> bool:
    return bool(args.progress_log) and args.log_mode != "quiet"


def _format_elapsed(start_time: float | None) -> str:
    if start_time is None:
        return ""
    return f" elapsed={time.perf_counter() - start_time:.2f}s"


def _progress(enabled: bool, stage: str, message: str, start_time: float | None = None) -> None:
    if enabled:
        _emit(f"[{stage}] {message}{_format_elapsed(start_time)}")


def _progress_detail_enabled(args: argparse.Namespace) -> bool:
    return _progress_enabled(args) and args.progress_detail != "off"


def _verbose_progress_enabled(args: argparse.Namespace) -> bool:
    return _progress_enabled(args) and args.progress_detail == "verbose"


def _progress_checkpoints(total: int) -> set[int]:
    if total <= 0:
        return set()
    if total >= 768:
        step = 256
    elif total >= 256:
        step = 128
    elif total >= 64:
        step = 64
    else:
        step = total
    checkpoints = set(range(step, total + 1, step))
    checkpoints.add(total)
    return checkpoints


def _record_timing(timings: dict[str, float], key: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    timings[key] = elapsed
    return elapsed


def _timing_summary(timings: dict[str, float]) -> str:
    keys = (
        "dataset_load",
        "normalization",
        "group_build",
        "model_init",
        "initial_loss",
        "epoch_loop",
        "prediction_export",
        "prediction_save",
        "best_prediction_export",
        "best_prediction_save",
        "summary_write",
    )
    return " ".join(f"{key}={timings[key]:.2f}s" for key in keys if key in timings)


def _ensure_ignored_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (REPO_DIR / "output").resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError(f"--output-dir must be under ignored output/: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _output_filename(name: str, flag: str) -> str:
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError(f"--{flag} must be a filename under --output-dir, found {name}")
    if not name:
        raise ValueError(f"--{flag} must not be empty")
    return name


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
        "background_relative_weight": float(args.background_relative_weight),
        "relative_floor": float(args.relative_floor),
        "relative_floor_mode": args.relative_floor_mode,
        "loss_weight_schedule": args.loss_weight_schedule,
        "loss_transition_epoch": int(args.loss_transition_epoch),
        "background_relative_weight_start": args.background_relative_weight_start,
        "background_relative_weight_end": args.background_relative_weight_end,
        "hotspot_weight_start": args.hotspot_weight_start,
        "hotspot_weight_end": args.hotspot_weight_end,
        "background_l1_weight_start": args.background_l1_weight_start,
        "background_l1_weight_end": args.background_l1_weight_end,
        "background_bias_weight_start": args.background_bias_weight_start,
        "background_bias_weight_end": args.background_bias_weight_end,
        "background_over_weight_start": args.background_over_weight_start,
        "background_over_weight_end": args.background_over_weight_end,
        "loss_space": (
            "base and hotspot terms use normalized_deltaT; background MSE/L1/bias/overprediction/relative "
            "terms use raw_deltaT_K"
        ),
        "base_loss_space": "normalized_deltaT",
        "background_mask_space": "raw_deltaT_K quantile",
        "background_penalty_space": "raw_deltaT_K_squared; penalizes pred_raw_deltaT toward 0",
        "background_l1_space": "raw_deltaT_K_abs; penalizes abs(pred_raw_deltaT) in background",
        "background_signed_bias_loss_space": "raw_deltaT_K_abs_bias; penalizes abs(mean(pred_raw_deltaT - true_raw_deltaT))",
        "background_overprediction_loss_space": "raw_deltaT_K_positive_error; penalizes mean(relu(pred_raw_deltaT - true_raw_deltaT))",
        "background_relative_abs_space": "safe raw_deltaT_K relative absolute error in background",
        "background_relative_abs_denominator": (
            "max(abs(true_raw_deltaT), floor), where floor is fixed relative_floor or "
            "max(relative_floor, batch/group abs true raw DeltaT p50/p75)"
        ),
        "hotspot_mask_space": "raw_deltaT_K quantile",
        "hotspot_retention_loss_space": "normalized_deltaT",
        "target_normalization": "normalized_deltaT = (raw_deltaT - train_target_delta_mean) / train_target_delta_std",
    }


def _lr_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr": float(args.min_lr),
        "second_stage_epoch": int(args.second_stage_epoch),
        "second_stage_lr": float(args.second_stage_lr),
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
    if float(config["background_relative_weight"]) < 0.0:
        raise ValueError("--background-relative-weight must be >= 0")
    if float(config["relative_floor"]) <= 0.0:
        raise ValueError("--relative-floor must be > 0")
    if int(config["loss_transition_epoch"]) < 0:
        raise ValueError("--loss-transition-epoch must be >= 0")
    for key in (
        "background_relative_weight_start",
        "background_relative_weight_end",
        "hotspot_weight_start",
        "hotspot_weight_end",
        "background_l1_weight_start",
        "background_l1_weight_end",
        "background_bias_weight_start",
        "background_bias_weight_end",
        "background_over_weight_start",
        "background_over_weight_end",
    ):
        value = config.get(key)
        if value is not None and float(value) < 0.0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")


def _validate_lr_config(config: dict[str, Any]) -> None:
    if float(config["lr"]) < 0.0:
        raise ValueError("--lr must be >= 0")
    if int(config["warmup_epochs"]) < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if float(config["min_lr"]) < 0.0:
        raise ValueError("--min-lr must be >= 0")
    if int(config["second_stage_epoch"]) < 0:
        raise ValueError("--second-stage-epoch must be >= 0")
    if float(config["second_stage_lr"]) < 0.0:
        raise ValueError("--second-stage-lr must be >= 0")


def _lr_for_epoch(epoch: int, epochs: int, config: dict[str, Any]) -> float:
    base_lr = float(config["lr"])
    schedule = config["lr_schedule"]
    if schedule == "constant":
        return base_lr
    if schedule == "two_stage":
        second_stage_epoch = int(config["second_stage_epoch"])
        if second_stage_epoch <= 0 or epoch <= second_stage_epoch:
            return base_lr
        return float(config["second_stage_lr"])
    if schedule == "warmup_cosine":
        warmup_epochs = int(config["warmup_epochs"])
        min_lr = float(config["min_lr"])
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            progress = epoch / warmup_epochs
            return min_lr + progress * (base_lr - min_lr)
        if warmup_epochs > 0:
            decay_epochs = max(epochs - warmup_epochs, 1)
            progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
        else:
            decay_epochs = max(epochs - 1, 1)
            progress = min(max((epoch - 1) / decay_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (base_lr - min_lr)
    raise ValueError(f"Unsupported lr schedule: {schedule}")


def _loss_weight_keys() -> tuple[str, ...]:
    return (
        "background_l1_weight",
        "background_bias_weight",
        "background_over_weight",
        "background_relative_weight",
        "hotspot_weight",
    )


def _scheduled_weight(config: dict[str, Any], key: str, epoch: int) -> float:
    static = float(config[key])
    schedule = config["loss_weight_schedule"]
    if schedule == "constant":
        return static

    transition_epoch = int(config["loss_transition_epoch"])
    start_value = config.get(f"{key}_start")
    end_value = config.get(f"{key}_end")
    start = static if start_value is None else float(start_value)
    end = static if end_value is None else float(end_value)

    if schedule == "two_phase":
        if epoch <= transition_epoch:
            return start
        return end

    if schedule == "linear_anneal":
        if transition_epoch <= 0:
            return static
        if epoch >= transition_epoch:
            return end
        if transition_epoch == 1:
            return end
        progress = (epoch - 1) / (transition_epoch - 1)
        return start + progress * (end - start)

    raise ValueError(f"Unsupported loss weight schedule: {schedule}")


def _loss_config_for_epoch(config: dict[str, Any], epoch: int) -> dict[str, Any]:
    current = dict(config)
    for key in _loss_weight_keys():
        value = _scheduled_weight(config, key, epoch)
        current[key] = value
        current[f"current_{key}"] = value
    return current


def _current_weight_payload(config: dict[str, Any]) -> dict[str, float]:
    return {f"current_{key}": float(config[key]) for key in _loss_weight_keys()}


def _sequence_summary(values) -> dict[str, float | int | None]:
    floats = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not floats:
        return {"count": 0, "first": None, "last": None, "min": None, "max": None}
    return {
        "count": len(floats),
        "first": floats[0],
        "last": floats[-1],
        "min": min(floats),
        "max": max(floats),
    }


def _history_field_summary(history: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    return _sequence_summary([item.get(field) for item in history])


def _loss_weight_schedule_payload(loss_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "loss_weight_schedule",
        "loss_transition_epoch",
        "background_relative_weight_start",
        "background_relative_weight_end",
        "hotspot_weight_start",
        "hotspot_weight_end",
        "background_l1_weight_start",
        "background_l1_weight_end",
        "background_bias_weight_start",
        "background_bias_weight_end",
        "background_over_weight_start",
        "background_over_weight_end",
    ]
    return {key: loss_config.get(key) for key in keys}


def _masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _normalized_delta_to_raw(pred_normalized, stats: dict):
    return pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]


def _safe_relative_denominator(target_raw, loss_config: dict[str, Any]):
    abs_target = jnp.abs(target_raw)
    mode = loss_config["relative_floor_mode"]
    floor = jnp.asarray(loss_config["relative_floor"], dtype=target_raw.dtype)
    if mode == "fixed":
        safe_floor = floor
    elif mode == "p50":
        safe_floor = jnp.maximum(jnp.quantile(abs_target, 0.50), floor)
    elif mode == "p75":
        safe_floor = jnp.maximum(jnp.quantile(abs_target, 0.75), floor)
    else:
        raise ValueError(f"Unsupported relative floor mode: {mode}")
    return jnp.maximum(abs_target, safe_floor)


def _loss_components(model, params, groups: list[dict], stats: dict, loss_config: dict[str, Any]) -> dict[str, Any]:
    weighted = {
        "base_mse": 0.0,
        "background_penalty": 0.0,
        "background_l1": 0.0,
        "background_signed_bias_loss": 0.0,
        "background_overprediction_loss": 0.0,
        "background_relative_abs": 0.0,
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
        background_relative_abs = jnp.asarray(0.0, dtype=base_mse.dtype)
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
        elif loss_config["loss_mode"] in {"background_l1_bias", "background_l1_relative"}:
            background_l1 = _masked_mean(jnp.abs(pred_raw_delta), background_mask)
            background_signed_bias_loss = jnp.abs(_masked_mean(raw_error, background_mask))
            background_overprediction_loss = _masked_mean(jnp.maximum(raw_error, 0.0), background_mask)
            hotspot_retention_loss = _masked_mean(jnp.square(pred - target), hotspot_mask)
            if loss_config["loss_mode"] == "background_l1_relative":
                denom = _safe_relative_denominator(target_raw, loss_config)
                background_relative_abs = _masked_mean(jnp.abs(raw_error) / denom, background_mask)
            total_loss = (
                base_mse
                + loss_config["background_l1_weight"] * background_l1
                + loss_config["background_bias_weight"] * background_signed_bias_loss
                + loss_config["background_over_weight"] * background_overprediction_loss
                + loss_config["background_relative_weight"] * background_relative_abs
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
        weighted["background_relative_abs"] = weighted["background_relative_abs"] + background_relative_abs * n
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


def _copy_params(params):
    return tree.tree_map(lambda value: value.copy() if hasattr(value, "copy") else value, params)


def _best_selection_payload(
    result: dict[str, Any],
    *,
    best_predictions_path: Path | None,
    best_predictions_saved: bool,
) -> dict[str, Any]:
    best_record = result.get("best_record") or {}
    return {
        "selection_metric": result.get("selection_metric"),
        "best_epoch": best_record.get("epoch"),
        "best_valid_loss": best_record.get("valid_loss"),
        "best_valid_raw_deltaT_mse": best_record.get("valid_raw_deltaT_mse"),
        "best_valid_base_mse": best_record.get("valid_base_mse"),
        "final_epoch": result.get("final_epoch"),
        "final_valid_loss": result.get("final_valid_loss"),
        "final_valid_raw_deltaT_mse": result.get("valid_metrics", {}).get("raw_delta_mse"),
        "final_valid_base_mse": result.get("final_valid_loss_components", {}).get("base_mse"),
        "best_predictions_saved": bool(best_predictions_saved),
        "best_predictions_path": str(best_predictions_path) if best_predictions_path is not None else None,
    }


def _epoch_history_record(
    epoch: int,
    lr_epoch: float,
    current_loss_config: dict[str, Any],
    train_components: dict[str, Any],
    valid_components: dict[str, Any],
    valid_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "epoch": int(epoch),
        "lr": float(lr_epoch),
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
        "train_background_relative_abs": float(train_components["background_relative_abs"]),
        "valid_background_relative_abs": float(valid_components["background_relative_abs"]),
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
    record.update(_current_weight_payload(current_loss_config))
    return record


def _print_epoch_progress(record: dict[str, Any], epochs: int, log_mode: str) -> None:
    if log_mode == "quiet":
        return
    if log_mode == "compact":
        _emit(
            f"epoch {record['epoch']:03d}/{epochs:03d} "
            f"lr={record['lr']:.3e} "
            f"train_loss={record['train_loss']:.6e} "
            f"valid_loss={record['valid_loss']:.6e} "
            f"valid_base_mse={record['valid_base_mse']:.6e} "
            f"valid_bg_bias={record['valid_bg_signed_bias']:.6e} "
            f"valid_rel={record['valid_background_relative_abs']:.6e} "
            f"valid_hotspot_mae={record['valid_hotspot_raw_mae']:.6e} "
            f"valid_raw_deltaT_mse={record['valid_raw_deltaT_mse']:.6e} "
            f"rel_w={record['current_background_relative_weight']:.3e} "
            f"hot_w={record['current_hotspot_weight']:.3e}"
        )
        return
    _emit(
        f"epoch {record['epoch']:03d}/{epochs:03d} "
        f"lr={record['lr']:.8e} "
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
        f"train_background_relative_abs={record['train_background_relative_abs']:.8e} "
        f"valid_background_relative_abs={record['valid_background_relative_abs']:.8e} "
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
        f"valid_recovered_T_mse={record['valid_recovered_T_mse']:.8e} "
        f"current_background_l1_weight={record['current_background_l1_weight']:.8e} "
        f"current_background_bias_weight={record['current_background_bias_weight']:.8e} "
        f"current_background_over_weight={record['current_background_over_weight']:.8e} "
        f"current_background_relative_weight={record['current_background_relative_weight']:.8e} "
        f"current_hotspot_weight={record['current_hotspot_weight']:.8e}"
    )


def _fit_once(
    train_groups: list[dict],
    valid_groups: list[dict],
    stats: dict,
    epochs: int,
    lr_config: dict[str, Any],
    seed: int,
    report_every: int,
    loss_config: dict[str, Any],
    selection_metric: str,
    log_mode: str,
    progress_enabled: bool,
    timings: dict[str, float] | None = None,
) -> dict:
    timings = timings if timings is not None else {}
    init_start = time.perf_counter()
    _progress(progress_enabled, "startup", "initializing model parameters ...")
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=train_groups[0]["inputs"],
        graphs=train_groups[0]["graphs"],
    )["params"]
    _record_timing(timings, "model_init", init_start)
    _progress(progress_enabled, "startup", "model parameters initialized", init_start)

    initial_start = time.perf_counter()
    _progress(progress_enabled, "startup", "computing initial train/valid losses ...")
    initial_loss_config = _loss_config_for_epoch(loss_config, 1)
    train_initial_components = _loss_components(model, params, train_groups, stats, initial_loss_config)
    valid_initial_components = _loss_components(model, params, valid_groups, stats, initial_loss_config)
    _record_timing(timings, "initial_loss", initial_start)
    _progress(progress_enabled, "startup", "initial train/valid losses computed", initial_start)
    train_losses = [float(train_initial_components["total_loss"])]
    valid_losses = [float(valid_initial_components["total_loss"])]
    grad_norms = []
    lr_history = []
    loss_weight_history = []
    grad_finite = True
    epoch_history = []
    best_score: float | None = None
    best_record: dict[str, Any] | None = None
    best_params = None
    _progress(progress_enabled, "train", f"epoch loop start epochs={epochs} report_every={report_every}")
    epoch_loop_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        lr_epoch = _lr_for_epoch(epoch, epochs, lr_config)
        current_loss_config = _loss_config_for_epoch(loss_config, epoch)
        loss_weight_history.append({"epoch": int(epoch), **_current_weight_payload(current_loss_config)})
        lr_history.append(lr_epoch)
        should_report = _should_report_epoch(epoch, epochs, report_every)
        epoch_start = time.perf_counter()
        if should_report or epoch <= 3:
            _progress(progress_enabled, "train", f"epoch {epoch:03d}/{epochs:03d} start lr={lr_epoch:.3e}")

        def loss_fn(current_params):
            return _loss_components(model, current_params, train_groups, stats, current_loss_config)["total_loss"]

        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        params = tree.tree_map(lambda param, grad: param - lr_epoch * grad, params, grads)
        train_components = _loss_components(model, params, train_groups, stats, current_loss_config)
        valid_components = _loss_components(model, params, valid_groups, stats, current_loss_config)
        valid_metrics = _metrics(model, params, valid_groups, stats)
        train_metrics = _metrics(model, params, train_groups, stats)
        train_losses.append(float(train_components["total_loss"]))
        valid_losses.append(float(valid_components["total_loss"]))
        record = _epoch_history_record(
            epoch,
            lr_epoch,
            current_loss_config,
            train_components,
            valid_components,
            valid_metrics,
            train_metrics,
        )
        epoch_history.append(record)
        score = float(record[selection_metric])
        if best_score is None or score < best_score:
            best_score = score
            best_record = dict(record)
            best_params = _copy_params(params)
        if should_report:
            _progress(progress_enabled, "train", f"epoch {epoch:03d}/{epochs:03d} metrics computed", epoch_start)
            _print_epoch_progress(record, epochs, log_mode)
    _record_timing(timings, "epoch_loop", epoch_loop_start)

    _progress(progress_enabled, "train", "computing final train/valid metrics ...")
    final_metrics_start = time.perf_counter()
    train_metrics = _metrics(model, params, train_groups, stats)
    valid_metrics = _metrics(model, params, valid_groups, stats)
    final_loss_config = _loss_config_for_epoch(loss_config, epochs)
    final_train_components = _loss_components(model, params, train_groups, stats, final_loss_config)
    final_valid_components = _loss_components(model, params, valid_groups, stats, final_loss_config)
    _progress(progress_enabled, "train", "final train/valid metrics computed", final_metrics_start)
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
        "lr_history": np.asarray(lr_history, dtype=np.float64),
        "loss_weight_history": loss_weight_history,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "epoch_history": epoch_history,
        "selection_metric": selection_metric,
        "best_record": best_record,
        "best_params": best_params,
        "best_score": best_score,
        "final_epoch": int(epochs),
        "final_valid_loss": float(valid_losses[-1]),
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


def _print_startup_summary(
    args: argparse.Namespace,
    *,
    sample_root: Path,
    split_counts: dict[str, int],
    output_dir: Path,
    loss_config: dict[str, Any],
    lr_config: dict[str, Any],
) -> None:
    if args.log_mode == "quiet":
        return

    _emit("Heat3D v1 medium controlled training export smoke")
    _emit("  scope: research reference diagnostics only; not formal model performance")
    _emit(f"  subset: {sample_root}")
    _emit(f"  split counts: {split_counts}")
    _emit(
        "  run: "
        f"epochs={args.epochs} lr={args.lr} lr_schedule={lr_config['lr_schedule']} "
        f"seed={args.seed} report_every={args.report_every}"
    )
    _emit(
        "  output: "
        f"dir={output_dir} save_predictions={bool(args.save_predictions)} "
        f"save_best_predictions={bool(args.save_best_predictions)}"
    )
    _emit(
        "  logging: "
        f"log_mode={args.log_mode} progress_log={bool(args.progress_log)} "
        f"progress_detail={args.progress_detail}"
    )
    _emit(
        "  selection: "
        f"metric={args.selection_metric} best_predictions_name={args.best_predictions_name}"
    )
    if args.log_mode == "compact":
        _emit(
            "  loss: "
            f"mode={loss_config['loss_mode']} weight_schedule={loss_config['loss_weight_schedule']} "
            f"bg_q={loss_config['background_quantile']} hot_q={loss_config['hotspot_quantile']} "
            f"rel_w={loss_config['background_relative_weight']} hot_w={loss_config['hotspot_weight']}"
        )
    else:
        _emit(
            "  lr schedule params: "
            f"warmup_epochs={lr_config['warmup_epochs']} min_lr={lr_config['min_lr']} "
            f"second_stage_epoch={lr_config['second_stage_epoch']} "
            f"second_stage_lr={lr_config['second_stage_lr']}"
        )
        _emit(f"  loss mode: {loss_config['loss_mode']}")
        _emit(f"  loss space: {loss_config['loss_space']}")
        _emit(
            "  loss params: "
            f"background_quantile={loss_config['background_quantile']} "
            f"hotspot_quantile={loss_config['hotspot_quantile']} "
            f"background_weight={loss_config['background_weight']} "
            f"hotspot_weight={loss_config['hotspot_weight']} "
            f"background_l1_weight={loss_config['background_l1_weight']} "
            f"background_bias_weight={loss_config['background_bias_weight']} "
            f"background_over_weight={loss_config['background_over_weight']} "
            f"background_relative_weight={loss_config['background_relative_weight']} "
            f"relative_floor={loss_config['relative_floor']} "
            f"relative_floor_mode={loss_config['relative_floor_mode']}"
        )
        _emit(f"  loss weight schedule: {_loss_weight_schedule_payload(loss_config)}")
    _emit("  feature mode: relative BC features, diag3 k encoding, zero_delta_u_bridge")
    _emit(
        "  target mode: normalized DeltaT target; normalized 0 is train mean raw DeltaT, "
        "not raw DeltaT=0"
    )


def _print_final_summary(
    args: argparse.Namespace,
    *,
    result: dict[str, Any],
    loss_config: dict[str, Any],
    lr_config: dict[str, Any],
    predictions_path: Path,
    predictions_saved: bool,
    prediction_count: int,
    best_predictions_path: Path | None,
    best_predictions_saved: bool,
    best_prediction_count: int,
    timings: dict[str, float],
) -> None:
    lr_history_summary = _sequence_summary(result["lr_history"])
    relative_weight_summary = _history_field_summary(
        result["loss_weight_history"], "current_background_relative_weight"
    )
    hotspot_weight_summary = _history_field_summary(result["loss_weight_history"], "current_hotspot_weight")
    best = result.get("best_record") or {}

    _emit("")
    _emit("summary")
    _emit(
        "  final: "
        f"epoch={result['final_epoch']} valid_loss={result['final_valid_loss']:.8e} "
        f"valid_base_mse={result['final_valid_loss_components']['base_mse']:.8e} "
        f"valid_raw_deltaT_mse={result['valid_metrics']['raw_delta_mse']:.8e}"
    )
    _emit(
        "  best-valid: "
        f"metric={args.selection_metric} epoch={best.get('epoch')} "
        f"valid_loss={best.get('valid_loss'):.8e} "
        f"valid_base_mse={best.get('valid_base_mse'):.8e} "
        f"valid_raw_deltaT_mse={best.get('valid_raw_deltaT_mse'):.8e}"
    )
    _emit(
        "  predictions: "
        f"final_saved={bool(predictions_saved)} final_path={predictions_path if predictions_saved else 'not_written'} "
        f"final_count={prediction_count} best_saved={bool(best_predictions_saved)} "
        f"best_path={best_predictions_path if best_predictions_saved else 'not_written'} "
        f"best_count={best_prediction_count}"
    )
    _emit(
        "  status: "
        f"grad_finite={result['grad_finite']} checkpoint_saved=False export_smoke_ok={result['status_ok']}"
    )

    if args.log_mode == "full":
        _emit("  loss/optimization")
        _emit(f"    loss mode: {loss_config['loss_mode']}")
        _emit(f"    loss weight schedule: {loss_config['loss_weight_schedule']}")
        _emit(f"    relative weight summary: {relative_weight_summary}")
        _emit(f"    hotspot weight summary: {hotspot_weight_summary}")
        _emit(f"    lr schedule: {lr_config['lr_schedule']}")
        _emit(f"    lr history summary: {lr_history_summary}")
        _emit("  loss initial/final")
        _emit(f"    train loss initial/final: {result['train_losses'][0]:.8e} -> {result['train_losses'][-1]:.8e}")
        _emit(f"    valid loss initial/final: {result['valid_losses'][0]:.8e} -> {result['valid_losses'][-1]:.8e}")
        _emit("  final base/raw/recovered metrics")
        _emit(f"    final train base MSE: {result['final_train_loss_components']['base_mse']:.8e}")
        _emit(f"    final valid base MSE: {result['final_valid_loss_components']['base_mse']:.8e}")
        _emit(f"    final train raw DeltaT MSE: {result['train_metrics']['raw_delta_mse']:.8e}")
        _emit(f"    final valid raw DeltaT MSE: {result['valid_metrics']['raw_delta_mse']:.8e}")
        _emit(f"    final train recovered temperature MSE: {result['train_metrics']['recovered_temperature_mse']:.8e}")
        _emit(f"    final valid recovered temperature MSE: {result['valid_metrics']['recovered_temperature_mse']:.8e}")
        _emit("  final background metrics")
        _emit(f"    final train background penalty: {result['final_train_loss_components']['background_penalty']:.8e}")
        _emit(f"    final valid background penalty: {result['final_valid_loss_components']['background_penalty']:.8e}")
        _emit(f"    final train background L1: {result['final_train_loss_components']['background_l1']:.8e}")
        _emit(f"    final valid background L1: {result['final_valid_loss_components']['background_l1']:.8e}")
        _emit(
            "    final train background signed bias loss: "
            f"{result['final_train_loss_components']['background_signed_bias_loss']:.8e}"
        )
        _emit(
            "    final valid background signed bias loss: "
            f"{result['final_valid_loss_components']['background_signed_bias_loss']:.8e}"
        )
        _emit(
            "    final train background overprediction loss: "
            f"{result['final_train_loss_components']['background_overprediction_loss']:.8e}"
        )
        _emit(
            "    final valid background overprediction loss: "
            f"{result['final_valid_loss_components']['background_overprediction_loss']:.8e}"
        )
        _emit(
            "    final train background relative abs: "
            f"{result['final_train_loss_components']['background_relative_abs']:.8e}"
        )
        _emit(
            "    final valid background relative abs: "
            f"{result['final_valid_loss_components']['background_relative_abs']:.8e}"
        )
        _emit(f"    final train bg pred raw mean: {result['final_train_loss_components']['bg_pred_raw_mean']:.8e}")
        _emit(f"    final valid bg pred raw mean: {result['final_valid_loss_components']['bg_pred_raw_mean']:.8e}")
        _emit(f"    final train bg signed bias: {result['final_train_loss_components']['bg_signed_bias']:.8e}")
        _emit(f"    final valid bg signed bias: {result['final_valid_loss_components']['bg_signed_bias']:.8e}")
        _emit(f"    final train bg abs mean: {result['final_train_loss_components']['bg_abs_mean']:.8e}")
        _emit(f"    final valid bg abs mean: {result['final_valid_loss_components']['bg_abs_mean']:.8e}")
        _emit("  final hotspot metrics")
        _emit(f"    final train hotspot retention loss: {result['final_train_loss_components']['hotspot_retention_loss']:.8e}")
        _emit(f"    final valid hotspot retention loss: {result['final_valid_loss_components']['hotspot_retention_loss']:.8e}")
        _emit(f"    final train hotspot raw MAE: {result['final_train_loss_components']['hotspot_raw_mae']:.8e}")
        _emit(f"    final valid hotspot raw MAE: {result['final_valid_loss_components']['hotspot_raw_mae']:.8e}")
    else:
        _emit(
            "  optimization: "
            f"loss_mode={loss_config['loss_mode']} loss_weight_schedule={loss_config['loss_weight_schedule']} "
            f"lr_schedule={lr_config['lr_schedule']} lr_summary={lr_history_summary}"
        )
        _emit(
            "  final background/hotspot: "
            f"valid_bg_bias={result['final_valid_loss_components']['bg_signed_bias']:.8e} "
            f"valid_bg_rel={result['final_valid_loss_components']['background_relative_abs']:.8e} "
            f"valid_hotspot_mae={result['final_valid_loss_components']['hotspot_raw_mae']:.8e}"
        )

    _progress(_progress_enabled(args), "startup-summary", _timing_summary(timings))
    _progress(_progress_enabled(args), "done", "script complete")


def _make_groups_with_progress(
    examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    label: str,
    progress_enabled: bool,
    verbose_progress_enabled: bool,
) -> list[dict]:
    start = time.perf_counter()
    sample_count = len(examples)
    _progress(progress_enabled, "startup", f"group build {label}: start samples={sample_count} ...")

    grouped: dict[tuple[int, tuple[str, ...], tuple[tuple[int, ...], ...]], list] = {}
    checkpoints = _progress_checkpoints(sample_count)
    scan_start = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        bridge = _bridge_for(example)
        signature = _metadata_shape_signature(builder.build_metadata(example.condition.coords))
        key = (
            example.condition.coords.shape[0],
            bridge.condition_feature_names,
            signature,
        )
        grouped.setdefault(key, []).append(example)
        if verbose_progress_enabled and index in checkpoints:
            _progress(
                True,
                "startup",
                f"group build {label}: {index}/{sample_count} samples scanned groups={len(grouped)}",
                scan_start,
            )

    _progress(
        progress_enabled,
        "startup",
        f"group build {label}: sample scan grouped={len(grouped)}",
        scan_start,
    )

    result = []
    for group_index, ((n_points, feature_names, _signature), group_examples) in enumerate(grouped.items(), start=1):
        group_name = f"group_{group_index}_N{n_points}_F{len(feature_names)}"
        batch_start = time.perf_counter()
        _progress(
            progress_enabled,
            "startup",
            (
                f"group build {label}: group {group_index}/{len(grouped)} "
                f"{group_name} arrays+graph start samples={len(group_examples)} ..."
            ),
        )
        result.append(_make_batch_group(group_name, group_examples, stats, builder))
        _progress(
            progress_enabled,
            "startup",
            f"group build {label}: group {group_index}/{len(grouped)} arrays+graph built",
            batch_start,
        )

    _progress(progress_enabled, "startup", f"group build {label}: done groups={len(result)}", start)
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")
    _output_filename(args.best_predictions_name, "best-predictions-name")
    progress_enabled = _progress_enabled(args)
    progress_detail_enabled = _progress_detail_enabled(args)
    verbose_progress_enabled = _verbose_progress_enabled(args)
    timings: dict[str, float] = {}
    script_start = time.perf_counter()
    _progress(
        progress_enabled,
        "startup",
        (
            f"script start subset={args.subset} epochs={args.epochs} lr={args.lr} "
            f"log_mode={args.log_mode} save_predictions={bool(args.save_predictions)}"
        ),
    )
    loss_config = _loss_config_from_args(args)
    lr_config = _lr_config_from_args(args)
    _validate_loss_config(loss_config)
    _validate_lr_config(lr_config)

    output_start = time.perf_counter()
    output_dir = _ensure_ignored_output_dir(args.output_dir)
    _progress(progress_enabled, "startup", f"output dir ready: {output_dir}", output_start)

    dataset_start = time.perf_counter()
    sample_root = _sample_root(args.subset)
    _progress(progress_enabled, "startup", f"loading dataset from {sample_root} ...")
    split_ids = _subset_split_ids(sample_root)
    _require_train_valid_splits(split_ids)

    all_ids = sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    train_ids = split_ids["train"]
    valid_ids = split_ids["valid"]
    split_counts = {split: len(ids) for split, ids in sorted(split_ids.items())}
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
    _progress(
        progress_enabled,
        "startup",
        f"dataset loaded: sample_count={len(dataset)} split_counts={split_counts}",
        dataset_start,
    )
    _record_timing(timings, "dataset_load", dataset_start)

    builder = Heat3DGraphBuilder()
    norm_start = time.perf_counter()
    _progress(progress_enabled, "startup", "computing train-only target normalization ...")
    stats = _train_only_stats(train_examples)
    _progress(
        progress_enabled,
        "startup",
        (
            "target normalization done: "
            f"delta_mean={float(stats['target_delta_mean'].reshape(-1)[0]):.6e} "
            f"delta_std={float(stats['target_delta_std'].reshape(-1)[0]):.6e}"
        ),
        norm_start,
    )
    _record_timing(timings, "normalization", norm_start)
    group_start = time.perf_counter()
    _progress(progress_enabled, "startup", "building grouped JAX arrays and graphs ...")
    train_groups = _make_groups_with_progress(
        train_examples,
        stats,
        builder,
        "train",
        progress_detail_enabled,
        verbose_progress_enabled,
    )
    valid_groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid",
        progress_detail_enabled,
        verbose_progress_enabled,
    )
    all_groups = _make_groups_with_progress(
        all_examples,
        stats,
        builder,
        "all",
        progress_detail_enabled,
        verbose_progress_enabled,
    )
    _record_timing(timings, "group_build", group_start)
    _progress(
        progress_enabled,
        "startup",
        (
            "groups built: "
            f"train_groups={len(train_groups)} valid_groups={len(valid_groups)} all_groups={len(all_groups)}"
        ),
        group_start,
    )

    _print_startup_summary(
        args,
        sample_root=sample_root,
        split_counts=split_counts,
        output_dir=output_dir,
        loss_config=loss_config,
        lr_config=lr_config,
    )

    result = _fit_once(
        train_groups,
        valid_groups,
        stats,
        args.epochs,
        lr_config,
        args.seed,
        args.report_every,
        loss_config,
        args.selection_metric,
        args.log_mode,
        progress_enabled,
        timings,
    )
    prediction_start = time.perf_counter()
    _progress(progress_enabled, "export", "building recovered predictions ...")
    predictions = _predict_temperatures(result["model"], result["params"], all_groups, stats)
    _record_timing(timings, "prediction_export", prediction_start)
    _progress(progress_enabled, "export", f"prediction arrays built: key_count={len(predictions)}", prediction_start)

    predictions_path = output_dir / "predictions.npz"
    best_predictions_path = output_dir / args.best_predictions_name if args.save_best_predictions else None
    best_predictions: dict[str, np.ndarray] = {}
    best_predictions_saved = False
    best_prediction_count = 0

    save_start = time.perf_counter()
    if args.save_predictions:
        _progress(progress_enabled, "export", f"saving predictions to {predictions_path} ...")
        np.savez_compressed(predictions_path, **predictions)
        _progress(progress_enabled, "export", f"predictions saved: key_count={len(predictions)} path={predictions_path}", save_start)
    else:
        _progress(progress_enabled, "export", f"prediction save skipped: key_count={len(predictions)}", save_start)
    _record_timing(timings, "prediction_save", save_start)

    if args.save_best_predictions:
        if result.get("best_params") is None:
            raise RuntimeError("best params are unavailable; expected at least one training epoch")
        best_prediction_start = time.perf_counter()
        _progress(progress_enabled, "export", "building best-valid recovered predictions ...")
        best_predictions = _predict_temperatures(result["model"], result["best_params"], all_groups, stats)
        best_prediction_count = len(best_predictions)
        _record_timing(timings, "best_prediction_export", best_prediction_start)
        _progress(
            progress_enabled,
            "export",
            f"best-valid prediction arrays built: key_count={best_prediction_count}",
            best_prediction_start,
        )
        best_save_start = time.perf_counter()
        _progress(progress_enabled, "export", f"saving best predictions to {best_predictions_path} ...")
        np.savez_compressed(best_predictions_path, **best_predictions)
        best_predictions_saved = True
        _record_timing(timings, "best_prediction_save", best_save_start)
        _progress(
            progress_enabled,
            "export",
            f"best predictions saved: key_count={best_prediction_count} path={best_predictions_path}",
            best_save_start,
        )

    best_selection = _best_selection_payload(
        result,
        best_predictions_path=best_predictions_path,
        best_predictions_saved=best_predictions_saved,
    )

    run_config = {
        "diagnostic_scope": "controlled training export smoke; not formal model performance",
        "subset": str(sample_root),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_schedule": lr_config["lr_schedule"],
        "warmup_epochs": lr_config["warmup_epochs"],
        "min_lr": lr_config["min_lr"],
        "second_stage_epoch": lr_config["second_stage_epoch"],
        "second_stage_lr": lr_config["second_stage_lr"],
        "optimizer": "manual_full_batch_gradient_descent",
        "seed": args.seed,
        "route": "relative BC features + zero_delta_u_bridge + normalized DeltaT target",
        "output_dir": str(output_dir),
        "save_predictions": bool(args.save_predictions),
        "predictions_path": str(predictions_path) if args.save_predictions else None,
        "save_best_predictions": bool(args.save_best_predictions),
        "best_predictions_name": args.best_predictions_name,
        "log_mode": args.log_mode,
        "progress_log": bool(args.progress_log),
        "progress_detail": args.progress_detail,
        **best_selection,
        "checkpoint_saved": False,
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "background_relative_weight": loss_config["background_relative_weight"],
        "relative_floor": loss_config["relative_floor"],
        "relative_floor_mode": loss_config["relative_floor_mode"],
        **_loss_weight_schedule_payload(loss_config),
        "loss": loss_config,
        "lr_config": lr_config,
        "split_counts": split_counts,
        "timing_diagnostics": dict(timings),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "ignored_candidate_ids": sorted(
            sample_id for split, ids in split_ids.items() if split not in {"train", "valid"} for sample_id in ids
        ),
    }

    loss_summary = {
        "status_ok": bool(result["status_ok"]),
        "grad_finite": bool(result["grad_finite"]),
        "train_losses": [float(value) for value in result["train_losses"]],
        "valid_losses": [float(value) for value in result["valid_losses"]],
        "grad_norms": [float(value) for value in result["grad_norms"]],
        "lr_history": [float(value) for value in result["lr_history"]],
        "lr_history_summary": _sequence_summary(result["lr_history"]),
        "loss_weight_history": result["loss_weight_history"],
        "loss_weight_history_summary": {
            "current_background_l1_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_l1_weight"
            ),
            "current_background_bias_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_bias_weight"
            ),
            "current_background_over_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_over_weight"
            ),
            "current_background_relative_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_relative_weight"
            ),
            "current_hotspot_weight": _history_field_summary(result["loss_weight_history"], "current_hotspot_weight"),
        },
        "train_metrics": _metrics_payload(result["train_metrics"]),
        "valid_metrics": _metrics_payload(result["valid_metrics"]),
        "log_mode": args.log_mode,
        "progress_log": bool(args.progress_log),
        "progress_detail": args.progress_detail,
        **best_selection,
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "background_relative_weight": loss_config["background_relative_weight"],
        "relative_floor": loss_config["relative_floor"],
        "relative_floor_mode": loss_config["relative_floor_mode"],
        **_loss_weight_schedule_payload(loss_config),
        "lr": lr_config["lr"],
        "lr_schedule": lr_config["lr_schedule"],
        "warmup_epochs": lr_config["warmup_epochs"],
        "min_lr": lr_config["min_lr"],
        "second_stage_epoch": lr_config["second_stage_epoch"],
        "second_stage_lr": lr_config["second_stage_lr"],
        "train_loss_selected": _selected_steps(result["train_losses"], args.report_every),
        "valid_loss_selected": _selected_steps(result["valid_losses"], args.report_every),
        "grad_norm_selected": _selected_steps(result["grad_norms"], args.report_every),
        "lr_config": lr_config,
        "epoch_history": result["epoch_history"],
        "loss": loss_config,
        "final_train_loss_components": result["final_train_loss_components"],
        "final_valid_loss_components": result["final_valid_loss_components"],
        "train_only_normalization": _stats_payload(stats),
    }
    summary_write_start = time.perf_counter()
    _progress(progress_enabled, "export", "writing run_config.json and loss_summary.json ...")
    loss_summary["timing_diagnostics"] = dict(timings)
    run_config["timing_diagnostics"] = dict(timings)
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "loss_summary.json", loss_summary)
    _record_timing(timings, "summary_write", summary_write_start)
    _progress(progress_enabled, "export", "run summary files written", summary_write_start)

    _print_final_summary(
        args,
        result=result,
        loss_config=loss_config,
        lr_config=lr_config,
        predictions_path=predictions_path,
        predictions_saved=bool(args.save_predictions),
        prediction_count=len(predictions),
        best_predictions_path=best_predictions_path,
        best_predictions_saved=best_predictions_saved,
        best_prediction_count=best_prediction_count,
        timings=timings,
    )
    _progress(progress_enabled, "done", "script complete", script_start)
    return 0 if result["status_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
