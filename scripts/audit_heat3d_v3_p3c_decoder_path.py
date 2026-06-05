#!/usr/bin/env python3
"""Heat3D v3 P3-c decoder/regional-path deeper audit.

The script retrains the existing RIGNO path on sample_000 using either explicit
optimizer settings or the best setting recorded by the P3-c optimizer sanity
JSON. It then audits latent scales, r2p edge-feature scales, decoder-path
ablations, and q/k/BC channel scale/correlation. It does not save parameters or
change model, decoder, loss, objective, or graph semantics.
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

from audit_heat3d_v3_p3_model_path import (  # noqa: E402
    _feature_columns,
    _metrics,
    _output_change,
)
from audit_heat3d_v3_p3b_rigno_trained_path import (  # noqa: E402
    _builder_for_policy,
    _load_example,
    _shuffle_real_nodes,
)
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _make_batch_group,
    _train_only_stats,
)
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "heat3d_v3_p3c" / "decoder_path_audit.json"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--policy", choices=("legacy", "nearest_repair"), default="legacy")
    parser.add_argument("--use-best-from", type=Path, default=None)
    parser.add_argument("--optimizer", choices=("manual_gd", "adam"), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
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


def _read_best_setting(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_best_from is None:
        if args.optimizer is None or args.lr is None or args.epochs is None:
            raise ValueError(
                "Provide --use-best-from or explicit --optimizer, --lr, and --epochs"
            )
        if args.lr <= 0 or args.epochs < 1:
            raise ValueError("Explicit --lr must be positive and --epochs must be >= 1")
        return {
            "source": "explicit_cli",
            "optimizer": args.optimizer,
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "optimizer_sanity_best": None,
        }

    with args.use_best_from.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    best = payload["best_result"]
    epochs = int(best.get("epochs_for_best_params") or best.get("epochs_requested") or 1)
    epochs = max(epochs, 1)
    return {
        "source": str(args.use_best_from),
        "optimizer": best["optimizer"],
        "lr": float(best["lr"]),
        "epochs": epochs,
        "optimizer_sanity_best": best,
    }


def _train_params(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
    optimizer_name: str,
    lr: float,
    epochs: int,
) -> dict[str, Any]:
    def loss_fn(current_params):
        pred = model.apply({"params": current_params}, inputs=group["inputs"], graphs=group["graphs"])
        return jnp.mean(jnp.square(pred - group["target_normalized"]))

    if optimizer_name == "adam":
        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(params)
    else:
        tx = None
        opt_state = None

    @jax.jit
    def manual_step(current_params):
        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        next_params = tree.tree_map(lambda param, grad: param - lr * grad, current_params, grads)
        return next_params, loss_value

    @jax.jit
    def adam_step(current_params, current_opt_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = tx.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss_value

    initial_loss = float(loss_fn(params))
    best_loss = initial_loss
    best_epoch = 0
    best_params = params
    losses = [initial_loss]
    start = time.perf_counter()
    finite = bool(np.isfinite(initial_loss))
    for epoch in range(1, epochs + 1):
        if optimizer_name == "manual_gd":
            params, _previous_loss = manual_step(params)
        elif optimizer_name == "adam":
            params, opt_state, _previous_loss = adam_step(params, opt_state)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        loss_value = float(loss_fn(params))
        losses.append(loss_value)
        finite = finite and bool(np.isfinite(loss_value))
        if finite and loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_params = params
        if not finite:
            break
    train_time = time.perf_counter() - start
    return {
        "params": best_params,
        "initial_loss": initial_loss,
        "final_loss": float(losses[-1]),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(losses) - 1),
        "finite": bool(finite),
        "train_time_seconds": float(train_time),
    }


def _decoder_forward(
    module: GraphNeuralOperator,
    inputs: Inputs,
    graphs: Any,
    decoder_ablation: str = "original",
    key: Any = None,
) -> dict[str, Any]:
    tau = None
    if inputs.c is None:
        u_inp = inputs.u
    else:
        u_inp = jnp.concatenate([inputs.u, inputs.c], axis=-1)
    pnode_features = jnp.moveaxis(
        u_inp,
        source=(0, 1, 2, 3),
        destination=(0, 3, 1, 2),
    ).squeeze(axis=3)
    dummy_pnode_features = jnp.zeros(
        shape=(pnode_features.shape[0], 1, pnode_features.shape[2]),
        dtype=pnode_features.dtype,
    )
    pnode_features = jnp.concatenate([pnode_features, dummy_pnode_features], axis=1)

    latent_rnodes, latent_pnodes = module.encoder(graphs.p2r, pnode_features, tau, key=None)
    processed_rnodes = module.processor(graphs.r2r, latent_rnodes, tau, key=None)
    if key is None:
        key = jax.random.PRNGKey(0)
    key_rnodes, key_pnodes = jax.random.split(key)

    decoder_rnodes = processed_rnodes
    decoder_pnodes = latent_pnodes
    if decoder_ablation in {"zero_rnode", "only_pnode"}:
        decoder_rnodes = jnp.zeros_like(processed_rnodes)
    elif decoder_ablation in {"zero_pnode", "only_rnode"}:
        decoder_pnodes = jnp.zeros_like(latent_pnodes)
    elif decoder_ablation == "shuffle_rnode":
        decoder_rnodes = _shuffle_real_nodes(processed_rnodes, key_rnodes)
    elif decoder_ablation == "shuffle_pnode":
        decoder_pnodes = _shuffle_real_nodes(latent_pnodes, key_pnodes)
    elif decoder_ablation != "original":
        raise ValueError(f"Unsupported decoder ablation: {decoder_ablation}")

    decoded_pnodes = module.decoder(graphs.r2p, decoder_rnodes, decoder_pnodes, tau, key=None)
    output = module._prepare_features(decoded_pnodes[:, :-1, :])
    return {
        "output": output,
        "latent_pnodes": latent_pnodes,
        "latent_rnodes": latent_rnodes,
        "processed_rnodes": processed_rnodes,
        "decoded_pnodes": decoded_pnodes,
    }


def _array_stats(value: Any, *, drop_dummy_axis1: bool = False) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    if drop_dummy_axis1 and array.ndim >= 2 and array.shape[1] > 1:
        array = array[:, :-1, ...]
    flat = array.reshape(-1)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(array))),
        "min": float(np.min(flat)),
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
        "std": float(np.std(flat)),
        "norm": float(np.linalg.norm(flat)),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "max_abs": float(np.max(np.abs(flat))),
    }


def _metrics_for_output(output: Any, group: dict, stats: dict) -> dict[str, Any]:
    return _metrics(output, group["target_normalized"], group["target_delta_raw"], stats)


def _r2p_edge_feature_stats(graphs: Any) -> dict[str, Any]:
    edge_key = graphs.r2p.edge_key_by_name("r2p")
    features = graphs.r2p.edges[edge_key].features
    return {
        "all_edges": _array_stats(features),
        "drop_last_dummy_edge": _array_stats(features, drop_dummy_axis1=True),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Correlation shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _channel_scale_and_correlation(
    *,
    group: dict,
    stats: dict,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    columns = _feature_columns(feature_names)
    c_normalized = np.asarray(group["inputs"].c, dtype=np.float64)
    c_mean = np.asarray(stats["condition_mean"], dtype=np.float64)
    c_std = np.asarray(stats["condition_std"], dtype=np.float64)
    c_raw = c_normalized * c_std + c_mean
    target_normalized = np.asarray(group["target_normalized"], dtype=np.float64)
    target_raw = np.asarray(group["target_delta_raw"], dtype=np.float64)
    rows = []
    for idx, name in enumerate(feature_names):
        normalized_values = c_normalized[..., idx]
        raw_values = c_raw[..., idx]
        rows.append(
            {
                "index": idx,
                "name": name,
                "group": (
                    "q" if idx in columns["q"] else
                    "k" if idx in columns["k"] else
                    "bc" if idx in columns["bc"] else
                    "other"
                ),
                "raw_mean": float(c_mean.reshape(-1)[idx]),
                "raw_safe_std": float(c_std.reshape(-1)[idx]),
                "raw_observed_std": float(np.std(raw_values)),
                "normalized_mean": float(np.mean(normalized_values)),
                "normalized_std": float(np.std(normalized_values)),
                "normalized_min": float(np.min(normalized_values)),
                "normalized_max": float(np.max(normalized_values)),
                "corr_with_target_normalized": _pearson(normalized_values, target_normalized),
                "corr_with_target_raw_delta": _pearson(raw_values, target_raw),
            }
        )
    return {
        "feature_columns": columns,
        "target_normalized": _array_stats(target_normalized),
        "target_raw_delta": _array_stats(target_raw),
        "channels": rows,
    }


def _decoder_ablation(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
    stats: dict,
    seed: int,
) -> dict[str, Any]:
    base = model.apply(
        {"params": params},
        group["inputs"],
        group["graphs"],
        "original",
        jax.random.PRNGKey(seed),
        method=_decoder_forward,
    )
    base_output = base["output"]
    base_metrics = _metrics_for_output(base_output, group, stats)
    rows = {}
    for offset, name in enumerate(
        (
            "original",
            "only_rnode",
            "only_pnode",
            "zero_rnode",
            "zero_pnode",
            "shuffle_rnode",
            "shuffle_pnode",
        ),
        start=1,
    ):
        path = model.apply(
            {"params": params},
            group["inputs"],
            group["graphs"],
            name,
            jax.random.PRNGKey(seed + 100 + offset),
            method=_decoder_forward,
        )
        metrics = _metrics_for_output(path["output"], group, stats)
        rows[name] = {
            "metrics": metrics,
            "output_change_vs_original": _output_change(path["output"], base_output),
            "loss_change_vs_original": metrics["normalized_loss"] - base_metrics["normalized_loss"],
            "relative_rmse_change_vs_original": (
                metrics["relative_rmse"] - base_metrics["relative_rmse"]
                if metrics["relative_rmse"] is not None and base_metrics["relative_rmse"] is not None
                else None
            ),
        }
    return {
        "base_metrics": base_metrics,
        "latent_pnodes": _array_stats(base["latent_pnodes"], drop_dummy_axis1=True),
        "latent_rnodes": _array_stats(base["latent_rnodes"], drop_dummy_axis1=True),
        "processed_rnodes": _array_stats(base["processed_rnodes"], drop_dummy_axis1=True),
        "decoded_output": _array_stats(base["output"]),
        "decoded_pnodes": _array_stats(base["decoded_pnodes"], drop_dummy_axis1=True),
        "r2p_edge_features": _r2p_edge_feature_stats(group["graphs"]),
        "ablations": rows,
    }


def _max_signal(ablations: dict[str, Any], names: tuple[str, ...]) -> float:
    values = []
    for name in names:
        row = ablations[name]
        values.append(abs(float(row["output_change_vs_original"]["rmse"])))
        rel = row.get("relative_rmse_change_vs_original")
        if rel is not None:
            values.append(abs(float(rel)))
    return max(values) if values else 0.0


def _q_scale_issue(channel_stats: dict[str, Any]) -> str:
    q_rows = [row for row in channel_stats["channels"] if row["group"] == "q"]
    if not q_rows:
        return "unclear"
    q = q_rows[0]
    if not np.isfinite(q["raw_safe_std"]) or not np.isfinite(q["normalized_std"]):
        return "true"
    if q["raw_safe_std"] <= EPS or q["normalized_std"] <= EPS:
        return "true"
    if q["normalized_std"] < 0.05 or q["normalized_std"] > 20.0:
        return "true"
    return "false"


def _judgment(
    *,
    decoder_audit: dict[str, Any],
    channel_stats: dict[str, Any],
    optimizer_best: dict[str, Any] | None,
) -> dict[str, Any]:
    ablations = decoder_audit["ablations"]
    rnode_signal = _max_signal(ablations, ("zero_rnode", "shuffle_rnode", "only_pnode"))
    pnode_signal = _max_signal(ablations, ("zero_pnode", "shuffle_pnode", "only_rnode"))
    if pnode_signal > 1.25 * rnode_signal:
        dominance = "pnode-dominant"
    elif rnode_signal > 1.25 * pnode_signal:
        dominance = "rnode-dominant"
    else:
        dominance = "mixed"
    base_rel = decoder_audit["base_metrics"]["relative_rmse"]
    passed_20 = bool(base_rel is not None and base_rel <= 0.20)
    return {
        "decoder_path": dominance,
        "rnode_signal": float(rnode_signal),
        "pnode_signal": float(pnode_signal),
        "rnode_routing_weak": (
            "true" if rnode_signal < 0.5 * pnode_signal else
            "false" if rnode_signal > 0.8 * pnode_signal else
            "unclear"
        ),
        "q_scaling_issue": _q_scale_issue(channel_stats),
        "optimizer_issue": (
            "true" if optimizer_best and optimizer_best.get("passed_20_percent") else "false"
        ),
        "capacity_or_routing_issue": "false" if passed_20 else "true",
    }


def main() -> int:
    args = parse_args()
    setting = _read_best_setting(args)
    example = _load_example(args)
    stats = _train_only_stats([example])
    builder = _builder_for_policy(args.policy)
    group = _make_batch_group("p3c_decoder_sample000", [example], stats, builder)
    model = GraphNeuralOperator(**MODEL_CONFIG)
    initial_params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )["params"]
    train_result = _train_params(
        model=model,
        params=initial_params,
        group=group,
        optimizer_name=setting["optimizer"],
        lr=setting["lr"],
        epochs=setting["epochs"],
    )
    trained_params = train_result.pop("params")
    decoder_audit = _decoder_ablation(
        model=model,
        params=trained_params,
        group=group,
        stats=stats,
        seed=args.seed,
    )
    channel_stats = _channel_scale_and_correlation(
        group=group,
        stats=stats,
        feature_names=tuple(group["feature_names"]),
    )
    payload = {
        "schema_version": "heat3d_v3_p3c_decoder_path_audit_v1",
        "diagnostic_scope": "trained RIGNO decoder/regional-path deeper audit; no model change",
        "config": {
            "subset": str(args.subset),
            "sample_id": args.sample_id,
            "k_encoding_mode": args.k_encoding_mode,
            "policy": args.policy,
            "seed": args.seed,
            "training_setting": setting,
            "model_config": MODEL_CONFIG,
            "builder_config": builder.config,
        },
        "sample": {
            "sample_id": example.sample_id,
            "split": example.meta.get("split"),
            "target_name": "DeltaT",
        },
        "feature_names": list(group["feature_names"]),
        "training": train_result,
        "decoder_path": decoder_audit,
        "channel_scale_and_correlation": channel_stats,
        "judgment": _judgment(
            decoder_audit=decoder_audit,
            channel_stats=channel_stats,
            optimizer_best=setting["optimizer_sanity_best"],
        ),
    }
    output_path = _write_json(args.output_json, payload)

    base = decoder_audit["base_metrics"]
    ablations = decoder_audit["ablations"]
    print("Heat3D v3 P3-c decoder path audit")
    print(f"  sample_id: {example.sample_id}")
    print(
        f"  setting: {setting['optimizer']} lr={setting['lr']:.1e} "
        f"epochs={setting['epochs']}"
    )
    print(
        f"  trained rel_rmse={base['relative_rmse'] * 100.0:.3f}% "
        f"loss={base['normalized_loss']:.6e}"
    )
    for name in ("only_rnode", "only_pnode", "zero_rnode", "zero_pnode", "shuffle_rnode", "shuffle_pnode"):
        row = ablations[name]
        print(
            f"  {name}: rel_rmse={row['metrics']['relative_rmse'] * 100.0:.3f}% "
            f"output_rmse_change={row['output_change_vs_original']['rmse']:.6e}"
        )
    print(f"  judgment: {payload['judgment']}")
    print(f"wrote={output_path}")
    print("Heat3D v3 P3-c decoder path audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
