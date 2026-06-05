#!/usr/bin/env python3
"""Heat3D v3 P2-redux AdamW graph-policy rerun.

This reruns small-sample graph-policy fitting with a sane optimizer/model
baseline after P3-c showed manual GD and low learning rate were confounding
prior P2 conclusions. It does not modify model, decoder, loss, objective, graph
semantics, or save checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_heat3d_v3_graph_coverage import audit_coords, summarize_records  # noqa: E402
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    _global_norm,
    _make_groups,
    _sample_root,
    _train_only_stats,
    _weighted_loss,
)
from run_heat3d_v3_p2_policy_small_training_smoke import (  # noqa: E402
    B96_ADAMW_DEFAULTS,
    POLICIES,
    _edge_totals,
    _policy_builder,
)
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v1_schema import find_sample_dirs, load_sample_meta  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "output" / "heat3d_v3_p2_adamw_rerun" / "p2_adamw_graph_policy_rerun.json"
)
TARGET_RELATIVE_RMSE = 0.20
TARGET_STRICT_RELATIVE_RMSE = 0.02
EPS = 1.0e-12
B96_MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 6,
    "node_latent_size": 128,
    "edge_latent_size": 128,
    "mlp_hidden_layers": 2,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--sample-count", type=int, choices=(1, 4, 16), default=16)
    parser.add_argument(
        "--policy",
        choices=("legacy", "nearest_repair", "discrete_radius", "all"),
        default="all",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adam", "manual_gd"),
        default=B96_ADAMW_DEFAULTS["optimizer"],
    )
    parser.add_argument("--lr", type=float, default=B96_ADAMW_DEFAULTS["lr"])
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine"),
        default=B96_ADAMW_DEFAULTS["lr_schedule"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=B96_ADAMW_DEFAULTS["warmup_epochs"])
    parser.add_argument("--min-lr", type=float, default=B96_ADAMW_DEFAULTS["min_lr"])
    parser.add_argument("--weight-decay", type=float, default=B96_ADAMW_DEFAULTS["weight_decay"])
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=B96_ADAMW_DEFAULTS["gradient_clip_norm"],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def _check_ignored(path: Path) -> None:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "check-ignore", "-q", str(relative)],
        cwd=REPO_ROOT,
        check=False,
    )
    if check.returncode != 0:
        raise ValueError(f"Refusing to write non-ignored rerun artifact: {relative}")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    _check_ignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _selected_policy_names(value: str) -> list[str]:
    if value == "all":
        return ["legacy", "nearest_repair", "discrete_radius"]
    return [value]


def _load_examples(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    sample_root = _sample_root(args.subset)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Heat3D subset sample root does not exist: {sample_root}")
    sample_ids = [
        str(load_sample_meta(sample_dir).get("sample_id", sample_dir.name))
        for sample_dir in find_sample_dirs(sample_root)
    ]
    sample_ids = sorted(sample_ids)
    if len(sample_ids) < args.sample_count:
        raise ValueError(
            f"Need at least {args.sample_count} samples, found {len(sample_ids)} in {sample_root}"
        )
    selected_ids = sample_ids[: args.sample_count]
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode=args.k_encoding_mode)
    index_by_id = dataset.sample_index_by_id()
    examples = [dataset[index_by_id[sample_id]] for sample_id in selected_ids]
    split_counts: dict[str, int] = {}
    for example in examples:
        split = str(example.meta.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1
    return examples, {
        "sample_root": str(sample_root),
        "selected_sample_ids": selected_ids,
        "original_split_counts_in_train_only_rerun": split_counts,
        "selection_note": "Selected supervised-small samples are treated as train-only fitting smoke.",
    }


def _coverage_for_examples(examples: list[Any], policy_names: list[str]) -> dict[str, Any]:
    records = []
    audit_policies = [POLICIES[name]["audit_policy"] for name in policy_names]
    for example in examples:
        records.extend(
            audit_coords(
                sample_id=example.sample_id,
                split="train_only_rerun",
                coords=np.asarray(example.condition.coords),
                seeds=[0],
                policies=audit_policies,
            )
        )
    return {
        "records": records,
        "summary": summarize_records(records),
    }


def _metrics(model: GraphNeuralOperator, params: Any, groups: list[dict], stats: dict) -> dict[str, Any]:
    finite = True
    shape_ok = True
    raw_sse = 0.0
    raw_sae = 0.0
    raw_count = 0
    target_sse = 0.0
    target_sae = 0.0
    normalized_sse = 0.0
    normalized_count = 0
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        target_normalized = group["target_normalized"]
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        target_delta = group["target_delta_raw"]
        raw_error = np.asarray(pred_delta - target_delta, dtype=np.float64)
        target_values = np.asarray(target_delta, dtype=np.float64)
        normalized_error = np.asarray(pred_normalized - target_normalized, dtype=np.float64)
        finite = (
            finite
            and bool(np.all(np.isfinite(np.asarray(pred_normalized))))
            and bool(np.all(np.isfinite(raw_error)))
        )
        shape_ok = shape_ok and pred_normalized.shape == target_normalized.shape
        raw_sse += float(np.sum(np.square(raw_error)))
        raw_sae += float(np.sum(np.abs(raw_error)))
        raw_count += int(raw_error.size)
        target_sse += float(np.sum(np.square(target_values)))
        target_sae += float(np.sum(np.abs(target_values)))
        normalized_sse += float(np.sum(np.square(normalized_error)))
        normalized_count += int(normalized_error.size)

    raw_rmse = float(np.sqrt(raw_sse / max(raw_count, 1)))
    raw_mae = raw_sae / max(raw_count, 1)
    target_rms = float(np.sqrt(target_sse / max(raw_count, 1)))
    target_abs_mean = target_sae / max(raw_count, 1)
    relative_rmse = raw_rmse / target_rms if target_rms > EPS else None
    relative_mae = raw_mae / target_abs_mean if target_abs_mean > EPS else None
    return {
        "normalized_mse": normalized_sse / max(normalized_count, 1),
        "raw_delta_rmse": raw_rmse,
        "raw_delta_mae": raw_mae,
        "target_delta_rms": target_rms,
        "target_delta_abs_mean": target_abs_mean,
        "relative_rmse": relative_rmse,
        "relative_mae": relative_mae,
        "passed_20_percent": bool(relative_rmse is not None and relative_rmse <= TARGET_RELATIVE_RMSE),
        "passed_2_percent": bool(relative_rmse is not None and relative_rmse <= TARGET_STRICT_RELATIVE_RMSE),
        "finite": finite,
        "shape_ok": shape_ok,
    }


def _grad_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "median": None, "max": None, "final": None, "all_finite": True}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
        "all_finite": bool(np.all(np.isfinite(array))),
    }


def _learning_rate_schedule(args: argparse.Namespace):
    if args.lr_schedule == "constant":
        return float(args.lr)

    def schedule(count):
        epoch = jnp.asarray(count, dtype=jnp.float32) + 1.0
        base = jnp.asarray(float(args.lr), dtype=jnp.float32)
        min_lr = jnp.asarray(float(args.min_lr), dtype=jnp.float32)
        warmup_epochs = int(args.warmup_epochs)
        if warmup_epochs > 0:
            warmup_progress = jnp.clip(epoch / float(warmup_epochs), 0.0, 1.0)
            warmup_lr = min_lr + warmup_progress * (base - min_lr)
            decay_epochs = max(args.epochs - warmup_epochs, 1)
            decay_progress = jnp.clip((epoch - float(warmup_epochs)) / float(decay_epochs), 0.0, 1.0)
            cosine_lr = min_lr + 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress)) * (base - min_lr)
            return jnp.where(epoch <= float(warmup_epochs), warmup_lr, cosine_lr)
        decay_epochs = max(args.epochs - 1, 1)
        decay_progress = jnp.clip((epoch - 1.0) / float(decay_epochs), 0.0, 1.0)
        return min_lr + 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress)) * (base - min_lr)

    return schedule


def _build_optimizer(args: argparse.Namespace, params: Any):
    if args.optimizer == "manual_gd":
        return None
    transforms = []
    if args.gradient_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(args.gradient_clip_norm)))
    lr = _learning_rate_schedule(args)
    if args.optimizer == "adam":
        if float(args.weight_decay) > 0.0:
            transforms.append(optax.add_decayed_weights(float(args.weight_decay)))
        transforms.append(optax.adam(learning_rate=lr))
    elif args.optimizer == "adamw":
        transforms.append(optax.adamw(learning_rate=lr, weight_decay=float(args.weight_decay)))
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    tx = optax.chain(*transforms)
    return {"tx": tx, "state": tx.init(params)}


def _run_policy(
    *,
    policy_name: str,
    examples: list[Any],
    args: argparse.Namespace,
    legacy_edge_totals: dict[str, int],
) -> dict[str, Any]:
    builder = _policy_builder(policy_name)
    stats = _train_only_stats(examples)
    build_start = time.perf_counter()
    groups = _make_groups(examples, stats, builder)
    graph_build_time = time.perf_counter() - build_start
    edge_totals = _edge_totals(groups)
    model = GraphNeuralOperator(**B96_MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=groups[0]["inputs"],
        graphs=groups[0]["graphs"],
    )["params"]

    def loss_fn(current_params):
        return _weighted_loss(model, current_params, groups)

    opt_state = _build_optimizer(args, params)
    losses = [float(loss_fn(params))]
    initial_metrics = _metrics(model, params, groups, stats)
    best_loss = float(losses[0])
    best_epoch = 0
    best_params = params
    grad_norms: list[float] = []
    grad_finite = True
    finite = bool(np.isfinite(losses[0]) and initial_metrics["finite"])
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        previous_loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        if opt_state is None:
            params = tree.tree_map(lambda param, grad: param - float(args.lr) * grad, params, grads)
        else:
            updates, next_state = opt_state["tx"].update(grads, opt_state["state"], params)
            opt_state["state"] = next_state
            params = optax.apply_updates(params, updates)
        loss_value = float(loss_fn(params))
        losses.append(loss_value)
        finite = finite and bool(np.isfinite(float(previous_loss)) and np.isfinite(loss_value))
        if finite and loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_params = params
        if not finite:
            break
    train_time = time.perf_counter() - train_start
    final_metrics = _metrics(model, params, groups, stats)
    best_metrics = _metrics(model, best_params, groups, stats)
    finite = bool(
        finite
        and grad_finite
        and final_metrics["finite"]
        and best_metrics["finite"]
        and final_metrics["shape_ok"]
        and best_metrics["shape_ok"]
    )
    edge_ratio_vs_legacy = {
        name: float(edge_totals[name] / legacy_edge_totals[name]) if legacy_edge_totals[name] else None
        for name in ("p2r", "r2p", "r2r")
    }
    return {
        "policy": policy_name,
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(losses) - 1),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "loss_drop": float(losses[0] - losses[-1]),
        "loss_drop_ratio": float((losses[0] - losses[-1]) / losses[0]) if abs(losses[0]) > EPS else None,
        "losses_first_last": {
            "first_10": [float(value) for value in losses[:10]],
            "last_10": [float(value) for value in losses[-10:]],
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "raw_delta_rmse": best_metrics["raw_delta_rmse"],
        "raw_delta_mae": best_metrics["raw_delta_mae"],
        "relative_rmse": best_metrics["relative_rmse"],
        "relative_mae": best_metrics["relative_mae"],
        "passed_20_percent": best_metrics["passed_20_percent"],
        "passed_2_percent": best_metrics["passed_2_percent"],
        "grad_norm": _grad_stats(grad_norms),
        "finite": finite,
        "shape_ok": best_metrics["shape_ok"],
        "group_count": len(groups),
        "graph_build_time_seconds": float(graph_build_time),
        "train_time_seconds": float(train_time),
        "train_step_time_seconds": float(train_time / max(len(losses) - 1, 1)),
        "edge_totals": edge_totals,
        "edge_ratio_vs_legacy": edge_ratio_vs_legacy,
    }


def _coverage_row(coverage: dict[str, Any], policy_name: str) -> dict[str, Any]:
    return coverage["summary"][POLICIES[policy_name]["audit_policy"]]


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.min_lr < 0:
        raise ValueError("--min-lr must be >= 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.gradient_clip_norm is not None and args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be > 0 when provided")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    policy_names = _selected_policy_names(args.policy)
    examples, dataset_metadata = _load_examples(args)
    coverage = _coverage_for_examples(examples, policy_names)
    legacy_groups = _make_groups(
        examples,
        _train_only_stats(examples),
        _policy_builder("legacy"),
    )
    legacy_edge_totals = _edge_totals(legacy_groups)

    policy_results = []
    for policy_name in policy_names:
        result = _run_policy(
            policy_name=policy_name,
            examples=examples,
            args=args,
            legacy_edge_totals=legacy_edge_totals,
        )
        policy_results.append(result)
        coverage_row = _coverage_row(coverage, policy_name)
        print(
            f"{policy_name}: best_rel_rmse={result['relative_rmse']:.6f} "
            f"best_loss={result['best_loss']:.6e}@{result['best_epoch']} "
            f"zero={coverage_row['p2r_zero_count_total']}/"
            f"{coverage_row['r2p_zero_count_total']} "
            f"edge_ratio={result['edge_ratio_vs_legacy']['p2r']:.3f}/"
            f"{result['edge_ratio_vs_legacy']['r2p']:.3f} "
            f"finite={result['finite']}"
        )

    payload = {
        "schema_version": "heat3d_v3_p2_adamw_graph_policy_rerun_v1",
        "diagnostic_scope": "P2-redux small-sample graph-policy fitting with B96-style optimizer/model",
        "config": {
            "subset": str(args.subset),
            "k_encoding_mode": args.k_encoding_mode,
            "sample_count": args.sample_count,
            "policy": args.policy,
            "policies_run": policy_names,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "epochs": args.epochs,
            "seed": args.seed,
            "model_config": B96_MODEL_CONFIG,
            "target_relative_rmse": TARGET_RELATIVE_RMSE,
            "target_strict_relative_rmse": TARGET_STRICT_RELATIVE_RMSE,
            "policies": POLICIES,
        },
        "dataset": dataset_metadata,
        "coverage": coverage,
        "legacy_edge_totals": legacy_edge_totals,
        "policy_results": policy_results,
    }
    output_path = _write_json(args.output_json, payload)
    print("Heat3D v3 P2-redux AdamW graph-policy rerun")
    print(f"  sample_count: {args.sample_count}")
    print(f"  optimizer: {args.optimizer}")
    print(f"  lr/schedule: {args.lr}/{args.lr_schedule}")
    print(f"  epochs: {args.epochs}")
    print(f"wrote={output_path}")
    print("Heat3D v3 P2-redux AdamW graph-policy rerun passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
