"""One-step Heat3D v2 M1 batch-size feasibility smoke.

This script is intentionally not a training script. It loads the configured
medium1024 subset, builds one train mini-batch per requested batch size, and
tries a single forward/loss/grad/update step. It never writes output files.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any


def _repo_dir() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "rigno").is_dir() and (cwd / "scripts").is_dir():
        return cwd
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if (parent / "rigno").is_dir() and (parent / "scripts").is_dir():
            return parent
    return file_path.parents[1]


REPO_DIR = _repo_dir()
SCRIPTS_DIR = REPO_DIR / "scripts"
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import jax.tree_util as tree  # noqa: E402
import numpy as np  # noqa: E402

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v2_config import load_v2_config  # noqa: E402


DEFAULT_CONFIG = (
    REPO_DIR
    / "configs"
    / "heat3d_v2"
    / "frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml"
)
DEFAULT_BATCH_SIZES = (8, 16, 32, 64, 96, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-step Heat3D v2 M1 batch-size feasibility smoke. "
            "The script does not train epochs and does not write output."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes to test.",
    )
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=900.0,
        help="Stop before starting a new case after this elapsed time.",
    )
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="Continue testing larger batch sizes after an OOM-like failure.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the result payload as JSON to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    config = load_v2_config(args.config)
    subset = REPO_DIR / config["dataset"]["subset_path"]
    sample_root = runner._sample_root(subset)

    start = time.perf_counter()
    split_ids = runner._subset_split_ids(sample_root)
    train_ids = split_ids.get("train", [])
    if not train_ids:
        raise ValueError(f"{sample_root}: missing non-empty train split")

    dataset_start = time.perf_counter()
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=config["dataset"].get("k_encoding_mode", "diag3"),
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in train_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset loader did not expose train samples: {missing[:5]}")
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    stats = runner._train_only_stats(train_examples)
    dataset_time = time.perf_counter() - dataset_start

    builder = Heat3DGraphBuilder()
    model = runner.GraphNeuralOperator(**_model_config(config))
    params = None
    rows: list[dict[str, Any]] = []
    stop_after_oom = False

    for batch_size in batch_sizes:
        elapsed_so_far = time.perf_counter() - start
        if elapsed_so_far >= float(args.max_total_seconds):
            rows.append(_skipped_row(batch_size, "max_total_seconds_reached"))
            break
        if stop_after_oom:
            rows.append(_skipped_row(batch_size, "skipped_after_oom"))
            continue
        if batch_size > len(train_examples):
            rows.append(_failure_row(batch_size, f"batch_size exceeds train split count {len(train_examples)}"))
            continue

        row, params = _run_case(
            batch_size=batch_size,
            train_examples=train_examples,
            stats=stats,
            builder=builder,
            model=model,
            params=params,
            config=config,
        )
        rows.append(row)
        if row["oom_like_failure"] and not args.continue_after_oom:
            stop_after_oom = True
        _clear_jax_caches()
        gc.collect()

    payload = {
        "schema_version": 1,
        "diagnostic_scope": "Heat3D v2 M1 one-step batch-size feasibility smoke",
        "config": str(args.config),
        "subset": str(sample_root),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "dataset_load_time_s": float(dataset_time),
        "rows": rows,
    }
    _print_table(payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not any(row["status"] == "success" for row in rows):
        return 1
    print("Heat3D v2 batch-size feasibility smoke passed.")
    return 0


def _run_case(
    *,
    batch_size: int,
    train_examples: list[Any],
    stats: dict[str, Any],
    builder: Heat3DGraphBuilder,
    model,
    params,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    case_start = time.perf_counter()
    try:
        group_start = time.perf_counter()
        groups = runner._make_groups_with_progress(
            train_examples[:batch_size],
            stats,
            builder,
            f"B{batch_size}",
            False,
            "quiet",
            batch_size=batch_size,
            drop_last=False,
        )
        if not groups:
            raise ValueError("group build returned no groups")
        group = groups[0]
        group_build_time = time.perf_counter() - group_start

        init_time = 0.0
        if params is None:
            init_start = time.perf_counter()
            params = model.init(
                jax.random.PRNGKey(int(config["optimizer"].get("seed", 0))),
                inputs=group["inputs"],
                graphs=group["graphs"],
            )["params"]
            runner._block_until_ready_tree(params)
            init_time = time.perf_counter() - init_start

        loss_config = _loss_config(config)
        current_loss_config = runner._loss_config_for_epoch(loss_config, 1)
        lr_config = _lr_config(config)
        optimizer_config = _optimizer_config(config)

        forward_start = time.perf_counter()
        components = runner._loss_components(model, params, [group], stats, current_loss_config)
        runner._block_until_ready_tree(components)
        forward_loss_time = time.perf_counter() - forward_start

        def loss_fn(current_params):
            return runner._loss_components(model, current_params, [group], stats, current_loss_config)["total_loss"]

        grad_start = time.perf_counter()
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        runner._block_until_ready_tree((loss_value, grads))
        grad_time = time.perf_counter() - grad_start

        update_start = time.perf_counter()
        optax_state = runner._build_optax_state(
            params,
            epochs=1,
            lr_config=lr_config,
            optimizer_config=optimizer_config,
        )
        if optax_state is None:
            updated_params = tree.tree_map(
                lambda param, grad: param - float(config["optimizer"]["lr"]) * grad,
                params,
                grads,
            )
        else:
            updates, opt_state = optax_state["tx"].update(grads, optax_state["state"], params)
            optax_state["state"] = opt_state
            updated_params = optax_state["apply_updates"](params, updates)
        runner._block_until_ready_tree(updated_params)
        update_time = time.perf_counter() - update_start

        signature = runner._batch_shape_signature(group)
        return (
            {
                "batch_size": int(batch_size),
                "actual_sample_count": int(runner._sample_count(group)),
                "status": "success",
                "success": True,
                "oom_like_failure": False,
                "elapsed_s": float(time.perf_counter() - case_start),
                "group_build_time_s": float(group_build_time),
                "model_init_time_s": float(init_time),
                "forward_loss_time_s": float(forward_loss_time),
                "grad_time_s": float(grad_time),
                "update_time_s": float(update_time),
                "loss": float(loss_value),
                "total_nodes": signature.get("total_nodes"),
                "total_edges": signature.get("total_edges"),
                "target_shape": signature.get("target_shape"),
            },
            params,
        )
    except Exception as exc:  # pragma: no cover - environment dependent smoke
        message = f"{type(exc).__name__}: {exc}"
        return (_failure_row(batch_size, message, elapsed=time.perf_counter() - case_start), params)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(runner.MODEL_CONFIG)
    model_section = config["model"]
    model.update(
        {
            "node_latent_size": int(model_section["node_latent_size"]),
            "edge_latent_size": int(model_section["edge_latent_size"]),
            "processor_steps": int(model_section["processor_steps"]),
            "mlp_hidden_layers": int(model_section["mlp_hidden_layers"]),
        }
    )
    return model


def _loss_config(config: dict[str, Any]) -> dict[str, Any]:
    loss = config["loss"]
    return {
        "loss_mode": loss["mode"],
        "background_quantile": float(loss["background_quantile"]),
        "hotspot_quantile": float(loss["hotspot_quantile"]),
        "background_weight": float(loss.get("background_weight", 1.0)),
        "hotspot_weight": float(loss.get("hotspot_weight", 0.0)),
        "background_l1_weight": float(loss.get("background_l1_weight", 0.0)),
        "background_bias_weight": float(loss.get("background_bias_weight", 0.0)),
        "background_over_weight": float(loss.get("background_over_weight", 0.0)),
        "background_relative_weight": float(loss.get("background_relative_weight", 0.0)),
        "relative_floor": float(loss.get("relative_floor", 0.02)),
        "relative_floor_mode": loss.get("relative_floor_mode", "fixed"),
        "pseudo_negative_quantile": float(loss.get("pseudo_negative_quantile", 0.25)),
        "pseudo_negative_delta_threshold": loss.get("pseudo_negative_delta_threshold"),
        "pseudo_negative_weight": float(loss.get("pseudo_negative_weight", 0.0)),
        "pseudo_negative_over_margin": float(loss.get("pseudo_negative_over_margin", 0.0)),
        "pseudo_negative_min_count": int(loss.get("pseudo_negative_min_count", 1)),
        "pseudo_negative_loss_type": loss.get("pseudo_negative_loss_type", "mse"),
        "pseudo_negative_relative_floor": float(loss.get("pseudo_negative_relative_floor", 0.02)),
        "loss_weight_schedule": loss.get("weight_schedule", "constant"),
        "loss_transition_epoch": int(loss.get("transition_epoch", 0)),
        "background_relative_weight_start": loss.get("background_relative_weight_start"),
        "background_relative_weight_end": loss.get("background_relative_weight_end"),
        "hotspot_weight_start": loss.get("hotspot_weight_start"),
        "hotspot_weight_end": loss.get("hotspot_weight_end"),
        "background_l1_weight_start": loss.get("background_l1_weight_start"),
        "background_l1_weight_end": loss.get("background_l1_weight_end"),
        "background_bias_weight_start": loss.get("background_bias_weight_start"),
        "background_bias_weight_end": loss.get("background_bias_weight_end"),
        "background_over_weight_start": loss.get("background_over_weight_start"),
        "background_over_weight_end": loss.get("background_over_weight_end"),
    }


def _lr_config(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    return {
        "lr": float(optimizer["lr"]),
        "lr_schedule": optimizer.get("lr_schedule", "constant"),
        "warmup_epochs": int(optimizer.get("warmup_epochs", 0)),
        "min_lr": float(optimizer.get("min_lr", 1.0e-5)),
        "second_stage_epoch": int(optimizer.get("second_stage_epoch") or 0),
        "second_stage_lr": float(optimizer.get("second_stage_lr") or 0.0),
    }


def _optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    return {
        "optimizer": str(optimizer["name"]),
        "gradient_clip_norm": optimizer.get("gradient_clip_norm"),
        "weight_decay": float(optimizer.get("weight_decay", 0.0)),
    }


def _parse_batch_sizes(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--batch-sizes values must be positive")
        values.append(value)
    if not values:
        raise ValueError("--batch-sizes did not contain any values")
    return values


def _failure_row(batch_size: int, message: str, *, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "batch_size": int(batch_size),
        "actual_sample_count": None,
        "status": "failure",
        "success": False,
        "oom_like_failure": _is_oom_like(message),
        "elapsed_s": float(elapsed),
        "error": message,
    }


def _skipped_row(batch_size: int, reason: str) -> dict[str, Any]:
    return {
        "batch_size": int(batch_size),
        "actual_sample_count": None,
        "status": "skipped",
        "success": False,
        "oom_like_failure": False,
        "elapsed_s": 0.0,
        "error": reason,
    }


def _is_oom_like(message: str) -> bool:
    text = message.lower()
    return any(
        marker in text
        for marker in (
            "resource_exhausted",
            "out of memory",
            "oom",
            "cuda_error_out_of_memory",
            "allocation",
        )
    )


def _clear_jax_caches() -> None:
    clear = getattr(jax, "clear_caches", None)
    if clear is not None:
        clear()


def _print_table(payload: dict[str, Any]) -> None:
    print("Heat3D v2 batch-size feasibility")
    print(f"  backend: {payload['jax_default_backend']}")
    print(f"  devices: {payload['jax_devices']}")
    print(f"  dataset_load_time_s: {payload['dataset_load_time_s']:.2f}")
    print(
        "| batch_size | status | elapsed_s | group_build_s | forward_s | grad_s | update_s | nodes | edges | error |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        print(
            f"| {row['batch_size']} | {row['status']} | "
            f"{_fmt(row.get('elapsed_s'))} | {_fmt(row.get('group_build_time_s'))} | "
            f"{_fmt(row.get('forward_loss_time_s'))} | {_fmt(row.get('grad_time_s'))} | "
            f"{_fmt(row.get('update_time_s'))} | {row.get('total_nodes') or ''} | "
            f"{row.get('total_edges') or ''} | {str(row.get('error', ''))[:120]} |"
        )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
