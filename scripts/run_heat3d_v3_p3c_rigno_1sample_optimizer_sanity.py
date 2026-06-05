#!/usr/bin/env python3
"""Heat3D v3 P3-c RIGNO 1-sample optimizer/lr sanity audit.

This script trains the existing Heat3D RIGNO path on supervised-small
sample_000 with either manual full-batch gradient descent or Optax Adam. It is
an optimizer sanity check only: no model, decoder, loss, objective, graph
semantics, or checkpoint behavior is changed.
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

from audit_heat3d_v3_p3_model_path import _metrics  # noqa: E402
from audit_heat3d_v3_p3b_rigno_trained_path import (  # noqa: E402
    _builder_for_policy,
    _load_example,
)
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _make_batch_group,
    _train_only_stats,
)
from run_heat3d_v3_p2_policy_small_training_smoke import B96_ADAMW_DEFAULTS  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "heat3d_v3_p3c" / "optimizer_sanity.json"
DEFAULT_MATRIX = (
    (B96_ADAMW_DEFAULTS["optimizer"], B96_ADAMW_DEFAULTS["lr"]),
)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--policy", choices=("legacy", "nearest_repair"), default="legacy")
    parser.add_argument("--optimizer", choices=("manual_gd", "adam", "adamw"), default=None)
    parser.add_argument("--lr", type=float, default=None)
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
        raise ValueError(f"Refusing to write non-ignored audit artifact: {relative}")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    _check_ignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _global_norm(tree_value: Any) -> float:
    leaves = tree.tree_leaves(tree_value)
    total = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(np.sqrt(total))


def _grad_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "min": None,
            "median": None,
            "max": None,
            "final": None,
            "all_finite": True,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
        "all_finite": bool(np.all(np.isfinite(array))),
    }


def _losses_summary(values: list[float]) -> dict[str, list[float]]:
    return {
        "first_10": [float(value) for value in values[:10]],
        "last_10": [float(value) for value in values[-10:]],
    }


def _metrics_for_params(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
    stats: dict,
) -> dict[str, Any]:
    pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    return _metrics(pred, group["target_normalized"], group["target_delta_raw"], stats)


def _train_setting(
    *,
    model: GraphNeuralOperator,
    initial_params: Any,
    group: dict,
    stats: dict,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    gradient_clip_norm: float | None,
    epochs: int,
) -> dict[str, Any]:
    params = initial_params

    def loss_fn(current_params):
        pred = model.apply({"params": current_params}, inputs=group["inputs"], graphs=group["graphs"])
        return jnp.mean(jnp.square(pred - group["target_normalized"]))

    if optimizer_name in {"adam", "adamw"}:
        transforms = []
        if gradient_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(float(gradient_clip_norm)))
        if optimizer_name == "adam":
            if weight_decay > 0.0:
                transforms.append(optax.add_decayed_weights(float(weight_decay)))
            transforms.append(optax.adam(learning_rate=lr))
        else:
            transforms.append(optax.adamw(learning_rate=lr, weight_decay=float(weight_decay)))
        tx = optax.chain(*transforms)
        opt_state = tx.init(params)
    else:
        tx = None
        opt_state = None

    @jax.jit
    def manual_step(current_params):
        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        next_params = tree.tree_map(lambda param, grad: param - lr * grad, current_params, grads)
        return next_params, loss_value, grads

    @jax.jit
    def adam_step(current_params, current_opt_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = tx.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss_value, grads

    initial_loss = float(loss_fn(params))
    initial_metrics = _metrics_for_params(model=model, params=params, group=group, stats=stats)
    best_loss = initial_loss
    best_epoch = 0
    best_params = params
    losses = [initial_loss]
    grad_norms: list[float] = []
    finite = bool(np.isfinite(initial_loss) and initial_metrics["finite"])
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if optimizer_name == "manual_gd":
            params, previous_loss, grads = manual_step(params)
        elif optimizer_name in {"adam", "adamw"}:
            params, opt_state, previous_loss, grads = adam_step(params, opt_state)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        loss_value = float(loss_fn(params))
        losses.append(loss_value)
        finite = finite and bool(
            np.isfinite(float(previous_loss))
            and np.isfinite(loss_value)
            and np.isfinite(grad_norm)
        )
        if finite and loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_params = params
        if not finite:
            break

    train_time = time.perf_counter() - start
    final_metrics = _metrics_for_params(model=model, params=params, group=group, stats=stats)
    best_metrics = _metrics_for_params(model=model, params=best_params, group=group, stats=stats)
    finite = finite and final_metrics["finite"] and best_metrics["finite"]
    relative_rmse = best_metrics["relative_rmse"]
    return {
        "optimizer": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(losses) - 1),
        "initial_loss": initial_loss,
        "final_loss": float(losses[-1]),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "loss_drop": float(initial_loss - losses[-1]),
        "loss_drop_ratio": (
            float((initial_loss - losses[-1]) / initial_loss)
            if abs(initial_loss) > EPS
            else None
        ),
        "losses_first_last": _losses_summary(losses),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "raw_delta_rmse": best_metrics["raw_delta_rmse"],
        "raw_delta_mae": best_metrics["raw_delta_mae"],
        "relative_rmse": relative_rmse,
        "relative_rmse_percent": (
            float(relative_rmse * 100.0) if relative_rmse is not None else None
        ),
        "passed_20_percent": bool(relative_rmse is not None and relative_rmse <= 0.20),
        "passed_2_percent": bool(relative_rmse is not None and relative_rmse <= 0.02),
        "grad_norm": _grad_summary(grad_norms),
        "finite": bool(finite),
        "train_time_seconds": float(train_time),
        "train_step_time_seconds": float(train_time / max(len(losses) - 1, 1)),
    }


def _matrix_from_args(args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.optimizer is None and args.lr is None:
        return list(DEFAULT_MATRIX)
    if args.optimizer is None or args.lr is None:
        raise ValueError("--optimizer and --lr must be provided together for a single run")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    return [(args.optimizer, args.lr)]


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    finite_results = [
        result
        for result in results
        if result["finite"] and result["relative_rmse"] is not None
    ]
    candidates = finite_results or results
    return min(
        candidates,
        key=lambda result: (
            float("inf") if result["relative_rmse"] is None else result["relative_rmse"],
            result["best_loss"],
        ),
    )


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.gradient_clip_norm is not None and args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be > 0 when provided")

    matrix = _matrix_from_args(args)
    example = _load_example(args)
    stats = _train_only_stats([example])
    builder = _builder_for_policy(args.policy)
    group = _make_batch_group("p3c_sample000", [example], stats, builder)
    model = GraphNeuralOperator(**MODEL_CONFIG)
    initial_params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )["params"]

    results = []
    for optimizer_name, lr in matrix:
        result = _train_setting(
            model=model,
            initial_params=initial_params,
            group=group,
            stats=stats,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            epochs=args.epochs,
        )
        results.append(result)
        print(
            f"{optimizer_name} lr={lr:.1e}: "
            f"loss {result['initial_loss']:.6e}->{result['final_loss']:.6e} "
            f"best={result['best_loss']:.6e}@{result['best_epoch']} "
            f"rel_rmse={result['relative_rmse_percent']:.3f}% "
            f"finite={result['finite']}"
        )

    best = _best_result(results)
    payload = {
        "schema_version": "heat3d_v3_p3c_rigno_1sample_optimizer_sanity_v1",
        "diagnostic_scope": "RIGNO one-sample optimizer/lr sanity; no checkpoint or model change",
        "config": {
            "subset": str(args.subset),
            "sample_id": args.sample_id,
            "k_encoding_mode": args.k_encoding_mode,
            "policy": args.policy,
            "epochs": args.epochs,
            "seed": args.seed,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "matrix": [
                {
                    "optimizer": opt,
                    "lr": lr,
                    "weight_decay": args.weight_decay,
                    "gradient_clip_norm": args.gradient_clip_norm,
                }
                for opt, lr in matrix
            ],
            "model_config": MODEL_CONFIG,
            "builder_config": builder.config,
        },
        "sample": {
            "sample_id": example.sample_id,
            "split": example.meta.get("split"),
            "target_name": "DeltaT",
        },
        "feature_names": list(group["feature_names"]),
        "results": results,
        "best_result": {
            "optimizer": best["optimizer"],
            "lr": best["lr"],
            "epochs_requested": best["epochs_requested"],
            "epochs_for_best_params": best["best_epoch"],
            "best_loss": best["best_loss"],
            "relative_rmse": best["relative_rmse"],
            "relative_rmse_percent": best["relative_rmse_percent"],
            "passed_20_percent": best["passed_20_percent"],
            "passed_2_percent": best["passed_2_percent"],
        },
    }
    output_path = _write_json(args.output_json, payload)

    print("Heat3D v3 P3-c RIGNO optimizer sanity")
    print(f"  sample_id: {example.sample_id}")
    print(f"  policy: {args.policy}")
    print(
        "  best: "
        f"{best['optimizer']} lr={best['lr']:.1e} "
        f"rel_rmse={best['relative_rmse_percent']:.3f}% "
        f"best_loss={best['best_loss']:.6e}@{best['best_epoch']}"
    )
    print(f"wrote={output_path}")
    print("Heat3D v3 P3-c RIGNO optimizer sanity passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
