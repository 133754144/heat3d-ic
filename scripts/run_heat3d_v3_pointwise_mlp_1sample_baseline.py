#!/usr/bin/env python3
"""Pointwise MLP 1-sample baseline for Heat3D v3 P3.

This baseline uses the same sample, temperature-rise target normalization, and
raw DeltaT metrics as the RIGNO P2 smoke, but it does not call or train RIGNO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    _bridge_for,
    _sample_root,
    _train_only_stats,
)
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "output" / "heat3d_v3_p3_model_path" / "mlp_1sample_baseline.json"
)
EPS = 1.0e-12


class PointwiseMLP(nn.Module):
    hidden_size: int
    hidden_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.swish(x)
        return nn.Dense(1)(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
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
        raise ValueError(f"Refusing to write non-ignored MLP artifact: {relative}")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    _check_ignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_example(args: argparse.Namespace) -> Any:
    sample_root = _sample_root(args.subset)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Heat3D subset sample root does not exist: {sample_root}")
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode=args.k_encoding_mode)
    index_by_id = dataset.sample_index_by_id()
    if args.sample_id not in index_by_id:
        raise ValueError(f"Sample {args.sample_id!r} not found in {sample_root}")
    return dataset[index_by_id[args.sample_id]]


def _normalize_coords(coords: np.ndarray, stats: dict) -> np.ndarray:
    return 2.0 * ((coords - stats["coord_min"]) / stats["coord_span"]) - 1.0


def _make_arrays(example: Any) -> dict[str, Any]:
    stats = _train_only_stats([example])
    bridge = _bridge_for(example)
    raw_coords = np.asarray(bridge.legacy_inputs.x_inp, dtype=np.float64)
    raw_u = np.asarray(bridge.legacy_inputs.u, dtype=np.float64)
    raw_c = np.asarray(bridge.legacy_inputs.c, dtype=np.float64)
    target_delta = np.asarray(bridge.target_delta_u, dtype=np.float64)

    coords = _normalize_coords(raw_coords, stats)
    c = (raw_c - stats["condition_mean"]) / stats["condition_std"]
    target_normalized = (
        (target_delta - stats["target_delta_mean"]) / stats["target_delta_std"]
    )
    x = np.concatenate(
        [
            coords.reshape(coords.shape[2], -1),
            raw_u.reshape(raw_u.shape[2], -1),
            c.reshape(c.shape[2], -1),
        ],
        axis=-1,
    )
    y = target_normalized.reshape(target_normalized.shape[2], -1)
    return {
        "x": jnp.asarray(x, dtype=jnp.float32),
        "y": jnp.asarray(y, dtype=jnp.float32),
        "target_delta_raw": jnp.asarray(target_delta.reshape(target_delta.shape[2], -1), dtype=jnp.float32),
        "stats": stats,
        "feature_names": (
            ("x", "y", "z", "Inputs.u")
            + tuple(bridge.condition_feature_names)
        ),
    }


def _tree_zeros_like(value: Any) -> Any:
    return tree.tree_map(jnp.zeros_like, value)


def _adam_update(params: Any, grads: Any, opt_state: dict[str, Any], lr: float) -> tuple[Any, dict[str, Any]]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    t = opt_state["t"] + 1
    m = tree.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, opt_state["m"], grads)
    v = tree.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        opt_state["v"],
        grads,
    )
    m_hat = tree.tree_map(lambda value: value / (1.0 - beta1 ** t), m)
    v_hat = tree.tree_map(lambda value: value / (1.0 - beta2 ** t), v)
    params = tree.tree_map(
        lambda param, mh, vh: param - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, {"t": t, "m": m, "v": v}


def _metrics(model: PointwiseMLP, params: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    pred_norm = model.apply({"params": params}, arrays["x"])
    target_norm = arrays["y"]
    stats = arrays["stats"]
    pred_delta = pred_norm * stats["target_delta_std"].reshape(1, 1) + stats["target_delta_mean"].reshape(1, 1)
    target_delta = arrays["target_delta_raw"]
    raw_error = np.asarray(pred_delta - target_delta, dtype=np.float64)
    target_values = np.asarray(target_delta, dtype=np.float64)
    normalized_error = np.asarray(pred_norm - target_norm, dtype=np.float64)
    raw_rmse = float(np.sqrt(np.mean(np.square(raw_error))))
    raw_mae = float(np.mean(np.abs(raw_error)))
    target_rms = float(np.sqrt(np.mean(np.square(target_values))))
    target_abs_mean = float(np.mean(np.abs(target_values)))
    relative_rmse = raw_rmse / target_rms if target_rms > EPS else None
    relative_mae = raw_mae / target_abs_mean if target_abs_mean > EPS else None
    return {
        "normalized_loss": float(np.mean(np.square(normalized_error))),
        "raw_delta_rmse": raw_rmse,
        "raw_delta_mae": raw_mae,
        "target_delta_rms": target_rms,
        "target_delta_abs_mean": target_abs_mean,
        "relative_rmse": relative_rmse,
        "relative_mae": relative_mae,
        "meets_20pct_relative_rmse": (
            bool(relative_rmse <= 0.20) if relative_rmse is not None else False
        ),
        "meets_2pct_relative_rmse": (
            bool(relative_rmse <= 0.02) if relative_rmse is not None else False
        ),
        "finite": bool(
            np.all(np.isfinite(np.asarray(pred_norm)))
            and np.all(np.isfinite(raw_error))
            and np.all(np.isfinite(normalized_error))
        ),
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.hidden_size < 1 or args.hidden_layers < 1:
        raise ValueError("--hidden-size and --hidden-layers must be >= 1")

    example = _load_example(args)
    arrays = _make_arrays(example)
    model = PointwiseMLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers)
    params = model.init(jax.random.PRNGKey(args.seed), arrays["x"])["params"]
    opt_state = {
        "t": jnp.asarray(0, dtype=jnp.int32),
        "m": _tree_zeros_like(params),
        "v": _tree_zeros_like(params),
    }

    def loss_fn(current_params):
        pred = model.apply({"params": current_params}, arrays["x"])
        return jnp.mean(jnp.square(pred - arrays["y"]))

    @jax.jit
    def train_step(current_params, current_state):
        _, grads = jax.value_and_grad(loss_fn)(current_params)
        next_params, next_state = _adam_update(current_params, grads, current_state, args.lr)
        next_loss = loss_fn(next_params)
        return next_params, next_state, next_loss

    initial_loss = float(loss_fn(params))
    best_loss = initial_loss
    best_epoch = 0
    best_params = params
    losses = [initial_loss]
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        params, opt_state, loss = train_step(params, opt_state)
        loss_value = float(loss)
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_params = params
    train_time = time.perf_counter() - start

    final_loss = float(loss_fn(params))
    final_metrics = _metrics(model, params, arrays)
    best_metrics = _metrics(model, best_params, arrays)
    payload = {
        "schema_version": "heat3d_v3_pointwise_mlp_1sample_baseline_v1",
        "diagnostic_scope": "one-sample pointwise MLP baseline; no RIGNO path",
        "config": {
            "subset": str(args.subset),
            "sample_id": args.sample_id,
            "k_encoding_mode": args.k_encoding_mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "seed": args.seed,
            "optimizer": "manual_adam",
            "features": "normalized x + Inputs.u + normalized Inputs.c",
        },
        "sample": {
            "sample_id": example.sample_id,
            "split": example.meta.get("split"),
        },
        "input": {
            "shape": list(np.asarray(arrays["x"]).shape),
            "feature_names": list(arrays["feature_names"]),
            "finite": bool(np.all(np.isfinite(np.asarray(arrays["x"])))),
        },
        "target": {
            "shape": list(np.asarray(arrays["y"]).shape),
            "finite": bool(np.all(np.isfinite(np.asarray(arrays["y"])))),
            "target_delta_rms": best_metrics["target_delta_rms"],
            "target_delta_abs_mean": best_metrics["target_delta_abs_mean"],
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_epoch": int(best_epoch),
        "loss_drop": initial_loss - final_loss,
        "loss_drop_ratio": ((initial_loss - final_loss) / initial_loss if initial_loss else None),
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "losses_first_last": {
            "first_10": [float(value) for value in losses[:10]],
            "last_10": [float(value) for value in losses[-10:]],
        },
        "train_time_seconds": float(train_time),
        "train_step_time_seconds": float(train_time / max(args.epochs, 1)),
    }
    if not final_metrics["finite"] or not best_metrics["finite"]:
        raise AssertionError("MLP baseline produced non-finite metrics")
    output_path = _write_json(args.output_json, payload)

    print("Heat3D v3 pointwise MLP 1-sample baseline")
    print(f"  sample_id: {example.sample_id}")
    print(f"  input_shape: {payload['input']['shape']}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(f"  hidden: {args.hidden_size} x {args.hidden_layers}")
    print(f"  loss {initial_loss:.6e}->{final_loss:.6e} best={best_loss:.6e}@{best_epoch}")
    print(
        "  best metrics: "
        f"rmse={best_metrics['raw_delta_rmse']:.6e} "
        f"mae={best_metrics['raw_delta_mae']:.6e} "
        f"relative_rmse={best_metrics['relative_rmse']:.6f} "
        f"<=20%={best_metrics['meets_20pct_relative_rmse']} "
        f"<=2%={best_metrics['meets_2pct_relative_rmse']}"
    )
    print(f"wrote={output_path}")
    print("Heat3D v3 pointwise MLP 1-sample baseline passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
