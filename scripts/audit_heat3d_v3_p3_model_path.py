#!/usr/bin/env python3
"""Heat3D v3 P3 model-path audit for one supervised-small sample.

The audit initializes the existing RIGNO path, reports input/target contracts,
checks feature ablation sensitivity, and records one gradient pass grouped by
model component. It does not train or modify model/loss semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _bridge_for,
    _make_batch_group,
    _sample_root,
    _train_only_stats,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "output" / "heat3d_v3_p3_model_path" / "model_path_audit_sample000.json"
)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--k-encoding-mode", default="diag3")
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


def _array_stats(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "shape": None,
            "finite": None,
            "min": None,
            "mean": None,
            "max": None,
        }
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(array))),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _feature_columns(feature_names: tuple[str, ...]) -> dict[str, list[int]]:
    names = list(feature_names)
    k_cols = [idx for idx, name in enumerate(names) if name.startswith("k_")]
    q_cols = [idx for idx, name in enumerate(names) if name == "q"]
    bc_names = {
        "is_top",
        "is_bottom",
        "is_side",
        "is_interior",
        "top_h",
        "top_T_inf",
        "bottom_T_fixed",
        "top_T_inf_minus_T_ref",
        "bottom_T_fixed_minus_T_ref",
    }
    bc_cols = [
        idx
        for idx, name in enumerate(names)
        if name in bc_names or name.startswith("bc_") or "boundary" in name
    ]
    return {
        "k": k_cols,
        "q": q_cols,
        "bc": bc_cols,
    }


def _replace_c_columns(inputs: Inputs, columns: list[int], mode: str, key: jax.Array) -> Inputs:
    if inputs.c is None or not columns:
        return inputs
    c = jnp.array(inputs.c)
    if mode == "zero":
        c = c.at[..., columns].set(0.0)
    elif mode == "shuffle":
        permutation = jax.random.permutation(key, c.shape[2])
        shuffled = c[:, :, permutation, :][..., columns]
        c = c.at[..., columns].set(shuffled)
    else:
        raise ValueError(f"Unsupported ablation mode: {mode}")
    return inputs._replace(c=c)


def _metrics(pred: Any, target_normalized: Any, target_delta_raw: Any, stats: dict) -> dict[str, Any]:
    pred = jnp.asarray(pred)
    target_normalized = jnp.asarray(target_normalized)
    pred_delta = pred * stats["target_delta_std"] + stats["target_delta_mean"]
    raw_error = np.asarray(pred_delta - target_delta_raw, dtype=np.float64)
    target_delta = np.asarray(target_delta_raw, dtype=np.float64)
    normalized_error = np.asarray(pred - target_normalized, dtype=np.float64)
    raw_rmse = float(np.sqrt(np.mean(np.square(raw_error))))
    target_rms = float(np.sqrt(np.mean(np.square(target_delta))))
    return {
        "normalized_loss": float(np.mean(np.square(normalized_error))),
        "raw_delta_rmse": raw_rmse,
        "raw_delta_mae": float(np.mean(np.abs(raw_error))),
        "relative_rmse": raw_rmse / target_rms if target_rms > EPS else None,
        "finite": bool(np.all(np.isfinite(np.asarray(pred))) and np.all(np.isfinite(raw_error))),
        "shape": list(np.asarray(pred).shape),
    }


def _output_change(pred: Any, base_pred: Any) -> dict[str, float]:
    delta = np.asarray(pred - base_pred, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _path_to_string(path: tuple[Any, ...]) -> str:
    parts = []
    for item in path:
        key = getattr(item, "key", None)
        idx = getattr(item, "idx", None)
        if key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(item))
    return "/".join(parts)


def _component_for_path(path: str) -> str:
    segments = path.split("/")
    if segments and segments[0] == "decoder" and (
        "_output_network" in path or "decoder_nodes" in path or "decoder_edges" in path
    ):
        return "output"
    if segments and segments[0] in {"encoder", "processor", "decoder"}:
        return segments[0]
    return "other"


def _grad_norms_by_component(grads: Any) -> dict[str, Any]:
    totals = {name: 0.0 for name in ("encoder", "processor", "decoder", "output", "other")}
    leaf_counts = {name: 0 for name in totals}
    path_rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        path_str = _path_to_string(path)
        component = _component_for_path(path_str)
        norm_sq = float(jnp.sum(jnp.square(leaf)))
        totals[component] += norm_sq
        leaf_counts[component] += 1
        path_rows.append(
            {
                "path": path_str,
                "component": component,
                "shape": list(np.asarray(leaf).shape),
                "norm": float(np.sqrt(norm_sq)),
                "finite": bool(np.all(np.isfinite(np.asarray(leaf)))),
            }
        )
    return {
        "components": {
            name: {
                "norm": float(np.sqrt(value)),
                "leaf_count": int(leaf_counts[name]),
            }
            for name, value in totals.items()
        },
        "leaf_count": len(path_rows),
        "all_finite": all(row["finite"] for row in path_rows),
        "top_leaf_norms": sorted(path_rows, key=lambda row: row["norm"], reverse=True)[:20],
    }


def main() -> int:
    args = parse_args()
    example = _load_example(args)
    stats = _train_only_stats([example])
    builder = Heat3DGraphBuilder()
    group = _make_batch_group("p3_sample000", [example], stats, builder)
    bridge = _bridge_for(example)
    feature_names = tuple(group["feature_names"])
    columns = _feature_columns(feature_names)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(args.seed),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )["params"]

    def loss_fn(current_params):
        pred = model.apply({"params": current_params}, inputs=group["inputs"], graphs=group["graphs"])
        return jnp.mean(jnp.square(pred - group["target_normalized"]))

    base_pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    base_metrics = _metrics(
        base_pred,
        group["target_normalized"],
        group["target_delta_raw"],
        stats,
    )

    ablations = {
        "original": group["inputs"],
        "zero_q": _replace_c_columns(group["inputs"], columns["q"], "zero", jax.random.PRNGKey(args.seed + 1)),
        "shuffle_q": _replace_c_columns(group["inputs"], columns["q"], "shuffle", jax.random.PRNGKey(args.seed + 2)),
        "zero_bc": _replace_c_columns(group["inputs"], columns["bc"], "zero", jax.random.PRNGKey(args.seed + 3)),
        "shuffle_bc": _replace_c_columns(group["inputs"], columns["bc"], "shuffle", jax.random.PRNGKey(args.seed + 4)),
        "shuffle_k": _replace_c_columns(group["inputs"], columns["k"], "shuffle", jax.random.PRNGKey(args.seed + 5)),
    }
    ablation_rows = {}
    for name, inputs in ablations.items():
        pred = model.apply({"params": params}, inputs=inputs, graphs=group["graphs"])
        metrics = _metrics(pred, group["target_normalized"], group["target_delta_raw"], stats)
        ablation_rows[name] = {
            "metrics": metrics,
            "output_change_vs_original": _output_change(pred, base_pred),
            "loss_change_vs_original": metrics["normalized_loss"] - base_metrics["normalized_loss"],
            "relative_rmse_change_vs_original": (
                metrics["relative_rmse"] - base_metrics["relative_rmse"]
                if metrics["relative_rmse"] is not None and base_metrics["relative_rmse"] is not None
                else None
            ),
        }

    _, grads = jax.value_and_grad(loss_fn)(params)
    payload = {
        "schema_version": "heat3d_v3_p3_model_path_audit_v1",
        "diagnostic_scope": "one-sample initialized RIGNO model-path audit; no training",
        "config": {
            "subset": str(args.subset),
            "sample_id": args.sample_id,
            "k_encoding_mode": args.k_encoding_mode,
            "seed": args.seed,
            "model_config": MODEL_CONFIG,
            "builder_config": builder.config,
        },
        "sample": {
            "sample_id": example.sample_id,
            "split": example.meta.get("split"),
            "target_name": "DeltaT",
        },
        "arrays": {
            "inputs_u": _array_stats(group["inputs"].u),
            "inputs_c": _array_stats(group["inputs"].c),
            "inputs_x_inp": _array_stats(group["inputs"].x_inp),
            "inputs_x_out": _array_stats(group["inputs"].x_out),
            "target_normalized": _array_stats(group["target_normalized"]),
            "target_delta_raw": _array_stats(group["target_delta_raw"]),
            "legacy_bridge_u": _array_stats(bridge.legacy_inputs.u),
            "legacy_bridge_c": _array_stats(bridge.legacy_inputs.c),
        },
        "feature_names": list(feature_names),
        "feature_columns": columns,
        "baseline_metrics": base_metrics,
        "ablations": ablation_rows,
        "gradient_norms": _grad_norms_by_component(grads),
    }
    output_path = _write_json(args.output_json, payload)

    print("Heat3D v3 P3 model path audit")
    print(f"  sample_id: {example.sample_id}")
    print(f"  feature_names: {feature_names}")
    print(f"  feature_columns: {columns}")
    print(f"  baseline relative_rmse: {base_metrics['relative_rmse']:.6f}")
    for name in ("zero_q", "shuffle_q", "zero_bc", "shuffle_bc", "shuffle_k"):
        row = ablation_rows[name]
        print(
            f"  {name}: output_rmse_change={row['output_change_vs_original']['rmse']:.6e} "
            f"loss_change={row['loss_change_vs_original']:.6e} "
            f"relative_rmse_change={row['relative_rmse_change_vs_original']:.6e}"
        )
    print(f"  grad components: {payload['gradient_norms']['components']}")
    print(f"wrote={output_path}")
    print("Heat3D v3 P3 model path audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
