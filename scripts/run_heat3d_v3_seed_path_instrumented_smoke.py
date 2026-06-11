#!/usr/bin/env python3
"""Instrument early Heat3D v3 seed paths on B88 sample_shuffle.

This is a short audit, not a benchmark. It reuses the main controlled runner's
dataset, normalization, graph, and B88 sample_shuffle group-building helpers,
then trains the unmodified RIGNO path for a small number of epochs while
recording activation, latent, gradient, and update statistics.
"""

from __future__ import annotations

import argparse
import json
import os
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


_REPO_ROOT_ENV = os.environ.get("HEAT3D_REPO_ROOT")
REPO_ROOT = Path(_REPO_ROOT_ENV).resolve() if _REPO_ROOT_ENV else Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_heat3d_v3_p3_model_path import _grad_norms_by_component  # noqa: E402
from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    DEFAULT_SPLIT_MAP,
    DEFAULT_SUBSET,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    GraphNeuralOperator,
    _epoch_train_groups,
    _fit_once,
    _global_norm,
    _group_sample_id_hash,
    _lr_for_epoch,
    _make_groups_with_progress,
    _make_sample_shuffle_groups_with_progress,
    _metadata_key,
    _normalize_coords,
    _optax_learning_rate_schedule,
    _resolve_training_splits,
    _sample_count,
    _sample_root,
    _train_only_stats,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v3_seed_path_instrumented"
EPS = 1.0e-12
COMPONENTS = ("encoder", "processor", "decoder", "output", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--split-map", type=Path, default=DEFAULT_SPLIT_MAP)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model-seeds", default="0,1,4,6")
    parser.add_argument("--checkpoint-epochs", default="0,1,2,5,10,20")
    parser.add_argument("--batch-size", type=int, default=88)
    parser.add_argument("--batch-build-seed", type=int, default=0)
    parser.add_argument("--batch-order-seed", type=int, default=0)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


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


def _array_stats(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    abs_array = np.abs(array)
    return {
        "shape": [int(dim) for dim in array.shape],
        "finite": bool(np.all(finite)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "norm": float(np.linalg.norm(array.reshape(-1))),
        "near_zero_ratio": float(np.mean(abs_array < 1.0e-8)),
        "nan_count": int(np.size(array) - np.count_nonzero(finite)),
    }


def _component_for_path(path: str) -> str:
    segments = path.split("/")
    if segments and segments[0] == "decoder" and (
        "_output_network" in path or "decoder_nodes" in path or "decoder_edges" in path
    ):
        return "output"
    if segments and segments[0] in {"encoder", "processor", "decoder"}:
        return segments[0]
    return "other"


def _tree_norms_by_component(values: Any) -> dict[str, Any]:
    totals = {name: 0.0 for name in COMPONENTS}
    leaf_counts = {name: 0 for name in COMPONENTS}
    for path, leaf in tree.tree_flatten_with_path(values)[0]:
        path_str = "/".join(str(segment.key) for segment in path)
        component = _component_for_path(path_str)
        array = np.asarray(leaf)
        totals[component] += float(np.sum(np.square(array)))
        leaf_counts[component] += 1
    components = {
        name: {
            "norm": float(np.sqrt(total)),
            "leaf_count": int(leaf_counts[name]),
        }
        for name, total in totals.items()
    }
    components["total"] = {
        "norm": float(np.sqrt(sum(totals.values()))),
        "leaf_count": int(sum(leaf_counts.values())),
    }
    return {"components": components}


def _ratio_by_component(numerator: dict[str, Any], denominator: dict[str, Any]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for name in (*COMPONENTS, "total"):
        num = float(numerator["components"][name]["norm"])
        den = float(denominator["components"][name]["norm"])
        ratios[name] = float(num / max(den, EPS))
    return ratios


def _latent_probe(module: GraphNeuralOperator, inputs, graphs):
    tau = None
    u_inp = inputs.u if inputs.c is None else jnp.concatenate([inputs.u, inputs.c], axis=-1)
    pnode_features = jnp.moveaxis(
        u_inp,
        source=(0, 1, 2, 3),
        destination=(0, 3, 1, 2),
    ).squeeze(axis=3)
    dummy = jnp.zeros(
        shape=(pnode_features.shape[0], 1, pnode_features.shape[2]),
        dtype=pnode_features.dtype,
    )
    pnode_features = jnp.concatenate([pnode_features, dummy], axis=1)
    latent_rnodes, latent_pnodes = module.encoder(graphs.p2r, pnode_features, tau, key=None)
    processed_rnodes = module.processor(graphs.r2r, latent_rnodes, tau, key=None)
    output_pnodes = module.decoder(graphs.r2p, processed_rnodes, latent_pnodes, tau, key=None)
    output = module._prepare_features(output_pnodes[:, :-1, :])
    return {
        "output": output,
        "latent_rnodes": latent_rnodes,
        "processed_rnodes": processed_rnodes,
        "latent_pnodes": latent_pnodes,
    }


def _loss_for_group(model: GraphNeuralOperator, params: Any, group: dict[str, Any]) -> jax.Array:
    pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    return jnp.mean(jnp.square(pred - group["target_normalized"]))


def _weighted_mse(model: GraphNeuralOperator, params: Any, groups: list[dict[str, Any]]) -> float:
    total = 0.0
    count = 0
    for group in groups:
        loss = float(_loss_for_group(model, params, group))
        sample_count = _sample_count(group)
        total += loss * sample_count
        count += sample_count
    return float(total / max(count, 1))


def _raw_metrics(model: GraphNeuralOperator, params: Any, groups: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, float]:
    raw_sse = 0.0
    raw_sae = 0.0
    target_sae = 0.0
    count = 0
    finite = True
    for group in groups:
        pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        pred_delta = pred * stats["target_delta_std"] + stats["target_delta_mean"]
        target_delta = group["target_delta_raw"]
        error = np.asarray(pred_delta - target_delta, dtype=np.float64)
        target = np.asarray(target_delta, dtype=np.float64)
        raw_sse += float(np.sum(np.square(error)))
        raw_sae += float(np.sum(np.abs(error)))
        target_sae += float(np.sum(np.abs(target)))
        count += int(error.size)
        finite = finite and bool(np.all(np.isfinite(error)))
    rmse = float(np.sqrt(raw_sse / max(count, 1)))
    mae = float(raw_sae / max(count, 1))
    relative_rmse = float(rmse / max(target_sae / max(count, 1), EPS))
    return {"raw_deltaT_rmse": rmse, "raw_deltaT_mae": mae, "relative_rmse": relative_rmse, "finite": finite}


def _activation_record(
    *,
    model: GraphNeuralOperator,
    params: Any,
    group: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    probe = model.apply(
        {"params": params},
        inputs=group["inputs"],
        graphs=group["graphs"],
        method=_latent_probe,
    )
    pred_delta = probe["output"] * stats["target_delta_std"] + stats["target_delta_mean"]
    target_delta = group["target_delta_raw"]
    target_std = float(np.std(np.asarray(target_delta)))
    return {
        "prediction_raw_deltaT": _array_stats(pred_delta),
        "output_normalized": _array_stats(probe["output"]),
        "output_amplitude_ratio": float(np.std(np.asarray(pred_delta)) / max(target_std, EPS)),
        "encoder_latent_rnodes": _array_stats(probe["latent_rnodes"]),
        "processor_processed_rnodes": _array_stats(probe["processed_rnodes"]),
        "processor_relative_update": float(
            np.linalg.norm(np.asarray(probe["processed_rnodes"] - probe["latent_rnodes"]).reshape(-1))
            / max(np.linalg.norm(np.asarray(probe["latent_rnodes"]).reshape(-1)), EPS)
        ),
        "decoder_latent_pnodes": _array_stats(probe["latent_pnodes"]),
    }


def _audit_checkpoint(
    *,
    epoch: int,
    model: GraphNeuralOperator,
    params: Any,
    train_groups: list[dict[str, Any]],
    valid_groups: list[dict[str, Any]],
    valid_stress_groups: list[dict[str, Any]],
    stats: dict[str, Any],
    gradient_clip_norm: float | None,
) -> dict[str, Any]:
    first_group = train_groups[0]

    def loss_fn(current_params):
        return _loss_for_group(model, current_params, first_group)

    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    grad_components = _grad_norms_by_component(grads)
    grad_global_norm = float(_global_norm(grads))
    return {
        "epoch": int(epoch),
        "first_batch_loss": float(loss_value),
        "train_loss": _weighted_mse(model, params, train_groups),
        "valid_iid_loss": _weighted_mse(model, params, valid_groups),
        "valid_stress_loss": _weighted_mse(model, params, valid_stress_groups) if valid_stress_groups else None,
        "valid_iid_raw_metrics": _raw_metrics(model, params, valid_groups, stats),
        "activation": _activation_record(model=model, params=params, group=first_group, stats=stats),
        "grad_norm": {
            "global": grad_global_norm,
            "components": grad_components["components"],
            "clip_would_trigger": (
                bool(gradient_clip_norm is not None and grad_global_norm > float(gradient_clip_norm))
            ),
        },
        "param_norm": _tree_norms_by_component(params),
    }


def _load_groups(args: argparse.Namespace) -> dict[str, Any]:
    sample_root = _sample_root(args.subset)
    split_ids, split_source, primary_validation_split, stress_validation_split = _resolve_training_splits(
        sample_root,
        args.split_map,
    )
    all_ids = sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in split_ids["train"]]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in split_ids[primary_validation_split]]
    valid_stress_examples = (
        [dataset[index_by_id[sample_id]] for sample_id in split_ids.get(stress_validation_split, [])]
        if stress_validation_split is not None
        else []
    )
    stats = _train_only_stats(train_examples)
    builder = Heat3DGraphBuilder(
        radius_policy="legacy_kdtree_mean4",
        coverage_repair_policy="nearest_rnode",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    )
    train_groups = _make_sample_shuffle_groups_with_progress(
        train_examples,
        stats,
        builder,
        "train",
        False,
        "basic",
        args.graph_seed,
        batch_size=args.batch_size,
        batch_build_seed=args.batch_build_seed,
        drop_last=False,
    )
    valid_groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        primary_validation_split,
        False,
        "basic",
        args.graph_seed,
        batch_size=args.batch_size,
        drop_last=False,
    )
    valid_stress_groups = (
        _make_groups_with_progress(
            valid_stress_examples,
            stats,
            builder,
            stress_validation_split or "valid_stress",
            False,
            "basic",
            args.graph_seed,
            batch_size=args.batch_size,
            drop_last=False,
        )
        if valid_stress_examples
        else []
    )
    return {
        "sample_root": str(sample_root),
        "split_source": split_source,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "split_counts": {split: len(ids) for split, ids in sorted(split_ids.items())},
        "all_sample_count": len(all_ids),
        "stats": stats,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "valid_stress_groups": valid_stress_groups,
    }


def _run_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    groups: dict[str, Any],
    checkpoint_epochs: set[int],
) -> dict[str, Any]:
    train_groups = groups["train_groups"]
    valid_groups = groups["valid_groups"]
    valid_stress_groups = groups["valid_stress_groups"]
    stats = groups["stats"]
    model_config = {
        "num_outputs": 1,
        "processor_steps": 6,
        "node_latent_size": 96,
        "edge_latent_size": 96,
        "mlp_hidden_layers": 2,
        "concatenate_tau": False,
        "concatenate_t": False,
        "conditioned_normalization": False,
        "cond_norm_hidden_size": 16,
        "p_edge_masking": 0.0,
    }
    model = GraphNeuralOperator(**model_config)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=train_groups[0]["inputs"],
        graphs=train_groups[0]["graphs"],
    )["params"]
    lr_config = {
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "second_stage_epoch": 0,
        "second_stage_lr": args.lr,
        "updates_per_epoch": len(train_groups),
    }
    schedule = _optax_learning_rate_schedule(args.epochs, lr_config)
    tx = optax.chain(
        optax.clip_by_global_norm(float(args.gradient_clip_norm)),
        optax.adamw(learning_rate=schedule, weight_decay=float(args.weight_decay)),
    )
    opt_state = tx.init(params)

    @jax.jit
    def train_step(current_params, current_opt_state, inputs, graphs, target):
        def loss_fn(step_params):
            pred = model.apply({"params": step_params}, inputs=inputs, graphs=graphs)
            return jnp.mean(jnp.square(pred - target))

        loss_value, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = tx.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss_value, grads, updates

    checkpoints = []
    if 0 in checkpoint_epochs:
        checkpoints.append(
            _audit_checkpoint(
                epoch=0,
                model=model,
                params=params,
                train_groups=train_groups,
                valid_groups=valid_groups,
                valid_stress_groups=valid_stress_groups,
                stats=stats,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        )

    train_history = []
    clip_trigger_count = 0
    update_to_param_ratios: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_groups = _epoch_train_groups(
            train_groups,
            epoch=epoch,
            seed=args.batch_order_seed,
            shuffle=True,
        )
        batch_losses = []
        batch_grad_norms = []
        for group in epoch_groups:
            params, opt_state, loss_value, grads, updates = train_step(
                params,
                opt_state,
                group["inputs"],
                group["graphs"],
                group["target_normalized"],
            )
            grad_norm = float(_global_norm(grads))
            batch_losses.append(float(loss_value))
            batch_grad_norms.append(grad_norm)
            if grad_norm > float(args.gradient_clip_norm):
                clip_trigger_count += 1
            param_norms = _tree_norms_by_component(params)
            update_norms = _tree_norms_by_component(updates)
            update_to_param_ratios.append(_ratio_by_component(update_norms, param_norms))
        train_history.append(
            {
                "epoch": int(epoch),
                "lr": float(_lr_for_epoch(epoch, args.epochs, lr_config)),
                "train_batch_order_hash": _group_sample_id_hash(epoch_groups),
                "mean_batch_loss": float(np.mean(batch_losses)),
                "min_batch_loss": float(np.min(batch_losses)),
                "max_batch_loss": float(np.max(batch_losses)),
                "mean_grad_norm": float(np.mean(batch_grad_norms)),
                "max_grad_norm": float(np.max(batch_grad_norms)),
            }
        )
        if epoch in checkpoint_epochs:
            checkpoint = _audit_checkpoint(
                epoch=epoch,
                model=model,
                params=params,
                train_groups=train_groups,
                valid_groups=valid_groups,
                valid_stress_groups=valid_stress_groups,
                stats=stats,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            if update_to_param_ratios:
                checkpoint["last_update_to_param_norm_ratio"] = update_to_param_ratios[-1]
            checkpoints.append(checkpoint)
    elapsed = time.perf_counter() - start
    return {
        "model_seed": int(seed),
        "epochs": int(args.epochs),
        "train_time_seconds": float(elapsed),
        "train_step_time_seconds": float(elapsed / max(args.epochs * len(train_groups), 1)),
        "clip_trigger_count": int(clip_trigger_count),
        "total_update_count": int(args.epochs * len(train_groups)),
        "clip_trigger_ratio": float(clip_trigger_count / max(args.epochs * len(train_groups), 1)),
        "train_history": train_history,
        "checkpoints": checkpoints,
    }


def _write_md(path: Path, payload: dict[str, Any]) -> Path:
    _check_ignored(path)
    lines = [
        "# Heat3D v3 Seed Path Instrumented Smoke",
        "",
        "Short B88 sample_shuffle nearest_repair audit. No e400 training was run.",
        "",
        "| seed | final valid_iid | final rel RMSE | final amplitude | final processor update | clip ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        final = run["checkpoints"][-1]
        lines.append(
            "| {seed} | {valid:.4g} | {rel:.4g} | {amp:.4g} | {proc:.4g} | {clip:.4g} |".format(
                seed=run["model_seed"],
                valid=final["valid_iid_loss"],
                rel=final["valid_iid_raw_metrics"]["relative_rmse"],
                amp=final["activation"]["output_amplitude_ratio"],
                proc=final["activation"]["processor_relative_update"],
                clip=run["clip_trigger_ratio"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    seeds = _parse_int_list(args.model_seeds)
    checkpoint_epochs = {epoch for epoch in _parse_int_list(args.checkpoint_epochs) if epoch <= args.epochs}
    checkpoint_epochs.add(0)
    output_dir = args.output_dir
    _check_ignored(output_dir / "seed_path_instrumented_smoke.json")

    groups = _load_groups(args)
    runs = [
        _run_seed(seed=seed, args=args, groups=groups, checkpoint_epochs=checkpoint_epochs)
        for seed in seeds
    ]
    payload = {
        "diagnostic_scope": "B88 sample_shuffle nearest_repair early seed path instrumentation",
        "training_semantics": "short audit only; model/decoder/loss/objective unchanged",
        "config": {
            "subset": groups["sample_root"],
            "split_map": str(args.split_map),
            "epochs": int(args.epochs),
            "model_seeds": seeds,
            "checkpoint_epochs": sorted(checkpoint_epochs),
            "batch_size": int(args.batch_size),
            "batch_build_seed": int(args.batch_build_seed),
            "batch_order_seed": int(args.batch_order_seed),
            "graph_seed": int(args.graph_seed),
            "lr": float(args.lr),
            "lr_schedule": args.lr_schedule,
            "warmup_epochs": int(args.warmup_epochs),
            "min_lr": float(args.min_lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "model": "latent96_s6_mlp2",
            "graph_policy": "nearest_repair",
        },
        "groups": {
            "split_source": groups["split_source"],
            "primary_validation_split": groups["primary_validation_split"],
            "stress_validation_split": groups["stress_validation_split"],
            "split_counts": groups["split_counts"],
            "train_group_count": len(groups["train_groups"]),
            "train_group_sample_counts": [_sample_count(group) for group in groups["train_groups"]],
            "valid_group_count": len(groups["valid_groups"]),
            "valid_stress_group_count": len(groups["valid_stress_groups"]),
        },
        "runs": runs,
    }
    json_path = _write_json(output_dir / "seed_path_instrumented_smoke.json", payload)
    md_path = _write_md(output_dir / "seed_path_instrumented_smoke.md", payload)
    print(f"wrote={json_path}")
    print(f"wrote={md_path}")
    for run in runs:
        final = run["checkpoints"][-1]
        print(
            "seed",
            run["model_seed"],
            "final_valid_iid",
            f"{final['valid_iid_loss']:.6g}",
            "relative_rmse",
            f"{final['valid_iid_raw_metrics']['relative_rmse']:.6g}",
            "amplitude_ratio",
            f"{final['activation']['output_amplitude_ratio']:.6g}",
            "processor_update",
            f"{final['activation']['processor_relative_update']:.6g}",
            "clip_ratio",
            f"{run['clip_trigger_ratio']:.6g}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
