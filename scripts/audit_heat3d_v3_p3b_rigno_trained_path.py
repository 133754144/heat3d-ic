#!/usr/bin/env python3
"""Heat3D v3 P3-b trained RIGNO path audit on one supervised-small sample.

This script trains the existing RIGNO path on sample_000, then audits feature
sensitivities, component gradients, processor latent changes, and decoder
rnode/pnode path dependence. It does not modify model, decoder, loss, or
objective semantics.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_heat3d_v3_p3_model_path import (  # noqa: E402
    _feature_columns,
    _grad_norms_by_component,
    _metrics,
    _output_change,
    _replace_c_columns,
)
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _make_batch_group,
    _sample_root,
    _train_only_stats,
)
from run_heat3d_v3_p2_policy_small_training_smoke import POLICIES  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "output" / "heat3d_v3_p3b_rigno_path" / "rigno_path_audit.json"
)
PATH_THRESHOLD = 1.0e-3
Q_STRONG_RELATIVE_THRESHOLD = 2.0e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--policy", choices=("legacy", "nearest_repair"), default="legacy")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
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


def _load_example(args: argparse.Namespace) -> Any:
    sample_root = _sample_root(args.subset)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Heat3D subset sample root does not exist: {sample_root}")
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode=args.k_encoding_mode)
    index_by_id = dataset.sample_index_by_id()
    if args.sample_id not in index_by_id:
        raise ValueError(f"Sample {args.sample_id!r} not found in {sample_root}")
    return dataset[index_by_id[args.sample_id]]


def _builder_for_policy(policy: str) -> Heat3DGraphBuilder:
    if policy == "legacy":
        return Heat3DGraphBuilder()
    if policy == "nearest_repair":
        return Heat3DGraphBuilder(**POLICIES["nearest_repair"]["builder_kwargs"])
    raise ValueError(f"Unsupported policy: {policy}")


def _rigno_loss(model: GraphNeuralOperator, params: Any, group: dict) -> jax.Array:
    pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    return jnp.mean(jnp.square(pred - group["target_normalized"]))


def _train(
    *,
    model: GraphNeuralOperator,
    initial_params: Any,
    group: dict,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    params = initial_params

    def loss_fn(current_params):
        return _rigno_loss(model, current_params, group)

    @jax.jit
    def train_step(current_params):
        _, grads = jax.value_and_grad(loss_fn)(current_params)
        next_params = tree.tree_map(
            lambda param, grad: param - lr * grad,
            current_params,
            grads,
        )
        return next_params, loss_fn(next_params)

    initial_loss = float(loss_fn(params))
    best_loss = initial_loss
    best_epoch = 0
    losses = [initial_loss]
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        params, loss_value = train_step(params)
        loss_float = float(loss_value)
        losses.append(loss_float)
        if loss_float < best_loss:
            best_loss = loss_float
            best_epoch = epoch
    train_time = time.perf_counter() - start
    return {
        "params": params,
        "initial_loss": initial_loss,
        "final_loss": float(losses[-1]),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "loss_drop": initial_loss - float(losses[-1]),
        "loss_drop_ratio": ((initial_loss - float(losses[-1])) / initial_loss if initial_loss else None),
        "losses_first_last": {
            "first_10": [float(value) for value in losses[:10]],
            "last_10": [float(value) for value in losses[-10:]],
        },
        "train_time_seconds": float(train_time),
        "train_step_time_seconds": float(train_time / max(epochs, 1)),
    }


def _shuffle_real_nodes(values: jax.Array, key: jax.Array) -> jax.Array:
    real = values[:, :-1, :]
    dummy = values[:, -1:, :]
    permutation = jax.random.permutation(key, real.shape[1])
    return jnp.concatenate([real[:, permutation, :], dummy], axis=1)


def _decoder_path_forward(
    module: GraphNeuralOperator,
    inputs: Inputs,
    graphs: Any,
    decoder_ablation: str = "original",
    key: Any = None,
) -> dict[str, Any]:
    """Runs RIGNO internals and applies local decoder-path ablations."""

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
    if decoder_ablation == "zero_processed_rnodes":
        decoder_rnodes = jnp.zeros_like(decoder_rnodes)
    elif decoder_ablation == "zero_latent_pnodes":
        decoder_pnodes = jnp.zeros_like(decoder_pnodes)
    elif decoder_ablation == "shuffle_processed_rnodes":
        decoder_rnodes = _shuffle_real_nodes(decoder_rnodes, key_rnodes)
    elif decoder_ablation == "shuffle_latent_pnodes":
        decoder_pnodes = _shuffle_real_nodes(decoder_pnodes, key_pnodes)
    elif decoder_ablation != "original":
        raise ValueError(f"Unsupported decoder ablation: {decoder_ablation}")

    output_pnodes = module.decoder(graphs.r2p, decoder_rnodes, decoder_pnodes, tau, key=None)
    output = module._prepare_features(output_pnodes[:, :-1, :])
    return {
        "output": output,
        "latent_rnodes": latent_rnodes,
        "processed_rnodes": processed_rnodes,
        "latent_pnodes": latent_pnodes,
    }


def _norm_stats(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "norm": float(np.linalg.norm(array.reshape(-1))),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "mean_abs": float(np.mean(np.abs(array))),
        "max_abs": float(np.max(np.abs(array))),
    }


def _rnode_change(path: dict[str, Any]) -> dict[str, Any]:
    latent = np.asarray(path["latent_rnodes"][:, :-1, :], dtype=np.float64)
    processed = np.asarray(path["processed_rnodes"][:, :-1, :], dtype=np.float64)
    delta = processed - latent
    latent_norm = float(np.linalg.norm(latent.reshape(-1)))
    stats = _norm_stats(delta)
    stats["relative_norm"] = stats["norm"] / latent_norm if latent_norm else None
    return stats


def _metrics_for_output(output: Any, group: dict, stats: dict) -> dict[str, Any]:
    return _metrics(
        output,
        group["target_normalized"],
        group["target_delta_raw"],
        stats,
    )


def _input_sensitivity(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
    stats: dict,
    columns: dict[str, list[int]],
    seed: int,
) -> dict[str, Any]:
    base_pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    base_metrics = _metrics_for_output(base_pred, group, stats)
    ablations = {
        "original": group["inputs"],
        "zero_q": _replace_c_columns(group["inputs"], columns["q"], "zero", jax.random.PRNGKey(seed + 1)),
        "shuffle_q": _replace_c_columns(group["inputs"], columns["q"], "shuffle", jax.random.PRNGKey(seed + 2)),
        "shuffle_k": _replace_c_columns(group["inputs"], columns["k"], "shuffle", jax.random.PRNGKey(seed + 3)),
        "zero_bc": _replace_c_columns(group["inputs"], columns["bc"], "zero", jax.random.PRNGKey(seed + 4)),
        "shuffle_bc": _replace_c_columns(group["inputs"], columns["bc"], "shuffle", jax.random.PRNGKey(seed + 5)),
    }
    result = {}
    for name, inputs in ablations.items():
        pred = model.apply({"params": params}, inputs=inputs, graphs=group["graphs"])
        metrics = _metrics_for_output(pred, group, stats)
        result[name] = {
            "metrics": metrics,
            "output_change_vs_original": _output_change(pred, base_pred),
            "loss_change_vs_original": metrics["normalized_loss"] - base_metrics["normalized_loss"],
            "relative_rmse_change_vs_original": (
                metrics["relative_rmse"] - base_metrics["relative_rmse"]
                if metrics["relative_rmse"] is not None and base_metrics["relative_rmse"] is not None
                else None
            ),
        }
    return result


def _decoder_path_ablation(
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
        method=_decoder_path_forward,
    )
    base_output = base["output"]
    base_metrics = _metrics_for_output(base_output, group, stats)
    result = {
        "rnode_latent_change": _rnode_change(base),
        "ablations": {},
    }
    for index, name in enumerate(
        (
            "original",
            "zero_processed_rnodes",
            "zero_latent_pnodes",
            "shuffle_processed_rnodes",
            "shuffle_latent_pnodes",
        ),
        start=1,
    ):
        path = model.apply(
            {"params": params},
            group["inputs"],
            group["graphs"],
            name,
            jax.random.PRNGKey(seed + 10 + index),
            method=_decoder_path_forward,
        )
        metrics = _metrics_for_output(path["output"], group, stats)
        result["ablations"][name] = {
            "metrics": metrics,
            "output_change_vs_original": _output_change(path["output"], base_output),
            "loss_change_vs_original": metrics["normalized_loss"] - base_metrics["normalized_loss"],
            "relative_rmse_change_vs_original": (
                metrics["relative_rmse"] - base_metrics["relative_rmse"]
                if metrics["relative_rmse"] is not None and base_metrics["relative_rmse"] is not None
                else None
            ),
        }
    return result


def _grad_norms(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
) -> dict[str, Any]:
    def loss_fn(current_params):
        return _rigno_loss(model, current_params, group)

    _, grads = jax.value_and_grad(loss_fn)(params)
    return _grad_norms_by_component(grads)


def _audit_params(
    *,
    label: str,
    model: GraphNeuralOperator,
    params: Any,
    group: dict,
    stats: dict,
    columns: dict[str, list[int]],
    seed: int,
) -> dict[str, Any]:
    pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    return {
        "label": label,
        "metrics": _metrics_for_output(pred, group, stats),
        "input_sensitivity": _input_sensitivity(
            model=model,
            params=params,
            group=group,
            stats=stats,
            columns=columns,
            seed=seed,
        ),
        "grad_norms": _grad_norms(model=model, params=params, group=group),
        "decoder_path": _decoder_path_ablation(
            model=model,
            params=params,
            group=group,
            stats=stats,
            seed=seed,
        ),
    }


def _max_path_signal(path_ablations: dict[str, Any], names: tuple[str, str]) -> float:
    values = []
    for name in names:
        row = path_ablations[name]
        values.append(abs(float(row["output_change_vs_original"]["rmse"])))
        rel = row.get("relative_rmse_change_vs_original")
        if rel is not None:
            values.append(abs(float(rel)))
    return max(values) if values else 0.0


def _depends(signal: float) -> str:
    if signal >= PATH_THRESHOLD:
        return "true"
    if signal <= PATH_THRESHOLD * 0.1:
        return "false"
    return "unclear"


def _q_sensitivity(input_sensitivity: dict[str, Any]) -> str:
    signals = []
    for name in ("zero_q", "shuffle_q"):
        row = input_sensitivity[name]
        signals.append(abs(float(row["output_change_vs_original"]["rmse"])))
        rel = row.get("relative_rmse_change_vs_original")
        if rel is not None:
            signals.append(abs(float(rel)))
    if not signals:
        return "unclear"
    if max(signals) >= Q_STRONG_RELATIVE_THRESHOLD:
        return "strong"
    return "weak"


def _judgment(trained_audit: dict[str, Any]) -> dict[str, str]:
    grad_components = trained_audit["grad_norms"]["components"]
    processor_grad = grad_components["processor"]["norm"]
    rnode_change = trained_audit["decoder_path"]["rnode_latent_change"]
    rnode_relative = rnode_change.get("relative_norm") or 0.0
    if processor_grad >= PATH_THRESHOLD and rnode_relative >= PATH_THRESHOLD:
        processor_used = "true"
    elif processor_grad <= PATH_THRESHOLD * 0.1 and rnode_relative <= PATH_THRESHOLD * 0.1:
        processor_used = "false"
    else:
        processor_used = "unclear"

    path_ablations = trained_audit["decoder_path"]["ablations"]
    rnode_signal = _max_path_signal(
        path_ablations,
        ("zero_processed_rnodes", "shuffle_processed_rnodes"),
    )
    pnode_signal = _max_path_signal(
        path_ablations,
        ("zero_latent_pnodes", "shuffle_latent_pnodes"),
    )
    return {
        "processor_used": processor_used,
        "decoder_depends_on_rnodes": _depends(rnode_signal),
        "decoder_depends_on_pnodes": _depends(pnode_signal),
        "q_sensitivity_after_training": _q_sensitivity(trained_audit["input_sensitivity"]),
        "dominant_decoder_path": (
            "pnodes" if pnode_signal > rnode_signal else "rnodes" if rnode_signal > pnode_signal else "tie"
        ),
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")

    example = _load_example(args)
    stats = _train_only_stats([example])
    builder = _builder_for_policy(args.policy)
    group = _make_batch_group("p3b_sample000", [example], stats, builder)
    feature_names = tuple(group["feature_names"])
    columns = _feature_columns(feature_names)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    initial_params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )["params"]

    train_result = _train(
        model=model,
        initial_params=initial_params,
        group=group,
        epochs=args.epochs,
        lr=args.lr,
    )
    trained_params = train_result.pop("params")
    initialized_audit = _audit_params(
        label="initialized",
        model=model,
        params=initial_params,
        group=group,
        stats=stats,
        columns=columns,
        seed=args.seed,
    )
    trained_audit = _audit_params(
        label="trained",
        model=model,
        params=trained_params,
        group=group,
        stats=stats,
        columns=columns,
        seed=args.seed + 100,
    )
    payload = {
        "schema_version": "heat3d_v3_p3b_rigno_trained_path_audit_v1",
        "diagnostic_scope": "one-sample RIGNO trained-path and decoder/regional-path audit",
        "config": {
            "subset": str(args.subset),
            "sample_id": args.sample_id,
            "k_encoding_mode": args.k_encoding_mode,
            "policy": args.policy,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "model_config": MODEL_CONFIG,
            "builder_config": builder.config,
        },
        "sample": {
            "sample_id": example.sample_id,
            "split": example.meta.get("split"),
            "target_name": "DeltaT",
        },
        "feature_names": list(feature_names),
        "feature_columns": columns,
        "training": train_result,
        "initialized": initialized_audit,
        "trained": trained_audit,
        "judgment": _judgment(trained_audit),
    }
    output_path = _write_json(args.output_json, payload)

    print("Heat3D v3 P3-b RIGNO trained-path audit")
    print(f"  sample_id: {example.sample_id}")
    print(f"  policy: {args.policy}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(
        f"  train loss {train_result['initial_loss']:.6e}->"
        f"{train_result['final_loss']:.6e} best={train_result['best_loss']:.6e}@"
        f"{train_result['best_epoch']}"
    )
    for label, audit in (("initialized", initialized_audit), ("trained", trained_audit)):
        metrics = audit["metrics"]
        rnode_change = audit["decoder_path"]["rnode_latent_change"]
        grad = audit["grad_norms"]["components"]
        zero_rnodes = audit["decoder_path"]["ablations"]["zero_processed_rnodes"]
        zero_pnodes = audit["decoder_path"]["ablations"]["zero_latent_pnodes"]
        print(
            f"  {label}: rel_rmse={metrics['relative_rmse']:.6f} "
            f"rmse={metrics['raw_delta_rmse']:.6e} mae={metrics['raw_delta_mae']:.6e} "
            f"rnode_change_rel={rnode_change['relative_norm']:.6e} "
            f"grad(enc/proc/dec/out)="
            f"{grad['encoder']['norm']:.3e}/"
            f"{grad['processor']['norm']:.3e}/"
            f"{grad['decoder']['norm']:.3e}/"
            f"{grad['output']['norm']:.3e} "
            f"zero_rnodes_rel_delta={zero_rnodes['relative_rmse_change_vs_original']:.6e} "
            f"zero_pnodes_rel_delta={zero_pnodes['relative_rmse_change_vs_original']:.6e}"
        )
    print(f"  judgment: {payload['judgment']}")
    print(f"wrote={output_path}")
    print("Heat3D v3 P3-b RIGNO trained-path audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
