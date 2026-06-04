#!/usr/bin/env python3
"""Fit diagnostic pointwise baselines on fixed Heat3D memorization splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.heat3d_v2_field_shape_diagnostics import compute_field_shape_metrics


DEFAULT_SUBSET = Path(
    "data/heat3d-thermal-simulation/subsets/"
    "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_SPLITS = tuple(
    Path(f"configs/heat3d_v2/medium1024_gapA_memorization_train{count}_seed0.json")
    for count in (1, 4, 16)
)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--split-map", type=Path, action="append", dest="split_maps")
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-6)
    parser.add_argument("--mlp-hidden-size", type=int, default=128)
    parser.add_argument("--mlp-steps", type=int, default=3000)
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_false",
    )
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    return samples if samples.is_dir() else path


def _split_train_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sample_splits", payload)
    return sorted(sample_id for sample_id, split in mapping.items() if split == "train")


def _safe_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    return mean, np.where(std < EPS, 1.0, std)


def _prepare_train_arrays(dataset, sample_ids: list[str]) -> dict[str, Any]:
    index_by_id = dataset.sample_index_by_id()
    examples = [dataset[index_by_id[sample_id]] for sample_id in sample_ids]
    raw_coords: list[np.ndarray] = []
    raw_conditions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    sample_ranges: list[tuple[str, int, int]] = []
    feature_names: tuple[str, ...] | None = None
    offset = 0

    for example in examples:
        bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
            bridge_policy="zero_delta_u_bridge"
        )
        names = tuple(bridge.condition_feature_names)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("condition feature names differ within train split")
        coords = np.asarray(bridge.legacy_inputs.x_inp, dtype=np.float64).reshape(-1, 3)
        condition = np.asarray(bridge.legacy_inputs.c, dtype=np.float64).reshape(-1, len(names))
        target = np.asarray(bridge.target_delta_u, dtype=np.float64).reshape(-1, 1)
        raw_coords.append(coords)
        raw_conditions.append(condition)
        raw_targets.append(target)
        sample_ranges.append((example.sample_id, offset, offset + target.shape[0]))
        offset += target.shape[0]

    coords = np.concatenate(raw_coords, axis=0)
    conditions = np.concatenate(raw_conditions, axis=0)
    target = np.concatenate(raw_targets, axis=0)
    coord_min = np.min(coords, axis=0, keepdims=True)
    coord_span = np.max(coords, axis=0, keepdims=True) - coord_min
    coord_span = np.where(coord_span < EPS, 1.0, coord_span)
    coords_nrm = 2.0 * ((coords - coord_min) / coord_span) - 1.0
    condition_mean, condition_std = _safe_mean_std(conditions)
    condition_nrm = (conditions - condition_mean) / condition_std
    target_mean, target_std = _safe_mean_std(target)
    target_nrm = (target - target_mean) / target_std

    return {
        "x": np.concatenate([coords_nrm, condition_nrm], axis=1).astype(np.float32),
        "y": target_nrm.astype(np.float32),
        "target_raw": target,
        "target_mean": target_mean,
        "target_std": target_std,
        "sample_ranges": sample_ranges,
        "feature_names": ("coord_x", "coord_y", "coord_z", *(feature_names or ())),
    }


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return design @ weights


def _init_mlp(key, input_size: int, hidden_size: int):
    sizes = (input_size, hidden_size, hidden_size, 1)
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for layer_key, in_size, out_size in zip(keys, sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (in_size + out_size))
        params.append(
            {
                "w": jax.random.uniform(
                    layer_key, (in_size, out_size), minval=-limit, maxval=limit
                ),
                "b": jnp.zeros((out_size,), dtype=jnp.float32),
            }
        )
    return tuple(params)


def _apply_mlp(params, x):
    values = x
    for layer in params[:-1]:
        values = jax.nn.swish(values @ layer["w"] + layer["b"])
    return values @ params[-1]["w"] + params[-1]["b"]


def _fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_size: int,
    steps: int,
    learning_rate: float,
    report_every: int,
) -> tuple[np.ndarray, list[dict[str, float]], float]:
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    params = _init_mlp(jax.random.PRNGKey(0), x.shape[1], hidden_size)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state):
        def loss_fn(candidate):
            return jnp.mean(jnp.square(_apply_mlp(candidate, x_jax) - y_jax))

        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        params, state, loss = train_step(params, state)
        if step == 1 or step == steps or step % report_every == 0:
            history.append({"step": step, "normalized_mse": float(loss)})
    pred = np.asarray(_apply_mlp(params, x_jax), dtype=np.float64)
    return pred, history, time.perf_counter() - started


def _metrics(arrays: dict[str, Any], pred_nrm: np.ndarray) -> dict[str, Any]:
    pred_raw = pred_nrm * arrays["target_std"] + arrays["target_mean"]
    true_raw = arrays["target_raw"]
    error = pred_raw - true_raw
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mean_abs_true = float(np.mean(np.abs(true_raw)))
    per_sample = []
    for sample_id, start, stop in arrays["sample_ranges"]:
        shape = compute_field_shape_metrics(
            true_raw[start:stop],
            pred_raw[start:stop],
            top_k=5,
            sample_id=sample_id,
            split="train",
        )
        shape["raw_deltaT_rmse"] = float(
            np.sqrt(np.mean(np.square(error[start:stop])))
        )
        per_sample.append(shape)
    return {
        "normalized_mse": float(np.mean(np.square(pred_nrm - arrays["y"]))),
        "raw_deltaT_rmse": rmse,
        "error_pct": None if mean_abs_true <= EPS else 100.0 * rmse / mean_abs_true,
        "below_20_pct": bool(mean_abs_true > EPS and 100.0 * rmse / mean_abs_true < 20.0),
        "pred_mean": float(np.mean(pred_raw)),
        "pred_std": float(np.std(pred_raw)),
        "true_mean": float(np.mean(true_raw)),
        "true_std": float(np.std(true_raw)),
        "per_sample": per_sample,
    }


def main() -> int:
    args = parse_args()
    split_maps = args.split_maps or list(DEFAULT_SPLITS)
    dataset = Heat3DV1NativeSupervisedDataset(
        _sample_root(args.subset),
        k_encoding_mode="diag3",
        boundary_mask_fallback=args.boundary_mask_fallback,
    )
    results = []
    for split_path in split_maps:
        sample_ids = _split_train_ids(split_path)
        arrays = _prepare_train_arrays(dataset, sample_ids)
        ridge_pred = _fit_ridge(arrays["x"], arrays["y"], args.ridge_alpha)
        mlp_pred, history, elapsed = _fit_mlp(
            arrays["x"],
            arrays["y"],
            hidden_size=args.mlp_hidden_size,
            steps=args.mlp_steps,
            learning_rate=args.mlp_lr,
            report_every=args.report_every,
        )
        result = {
            "split_map": str(split_path),
            "train_sample_count": len(sample_ids),
            "point_count": int(arrays["x"].shape[0]),
            "input_feature_names": list(arrays["feature_names"]),
            "input_feature_count": int(arrays["x"].shape[1]),
            "ridge": _metrics(arrays, ridge_pred),
            "mlp": _metrics(arrays, mlp_pred),
            "mlp_history": history,
            "mlp_elapsed_s": elapsed,
        }
        results.append(result)
        print(
            f"train={len(sample_ids):2d} "
            f"ridge_err={result['ridge']['error_pct']:.2f}% "
            f"mlp_err={result['mlp']['error_pct']:.2f}% "
            f"mlp_loss={result['mlp']['normalized_mse']:.6e} "
            f"elapsed={elapsed:.2f}s"
        )

    payload = {
        "diagnostic_scope": "pointwise fitting audit; not a generalization benchmark",
        "subset": str(args.subset),
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "mlp_steps": int(args.mlp_steps),
        "mlp_lr": float(args.mlp_lr),
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Heat3D v2 pointwise baseline audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
