#!/usr/bin/env python3
"""Heat3D v3 P2-b longer 16-sample graph-policy fitting smoke.

This is a train-only controlled smoke over the 16-sample supervised-small
subset. It does not save checkpoints and does not change model, decoder, loss,
optimizer, bridge, or dataset semantics.
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

from audit_heat3d_v3_graph_coverage import audit_coords, summarize_records  # noqa: E402
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _global_norm,
    _make_groups,
    _sample_root,
    _train_only_stats,
    _weighted_loss,
)
from run_heat3d_v3_p2_policy_small_training_smoke import (  # noqa: E402
    POLICIES,
    _edge_totals,
    _policy_builder,
)
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v1_schema import find_sample_dirs, load_sample_meta  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v3_p2_policy_smoke"
TARGET_RELATIVE_ERROR = 0.20
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        raise ValueError(f"Refusing to write non-ignored smoke artifact inside repo: {relative}")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    _check_ignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    _check_ignored(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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
        "original_split_counts_in_train_only_smoke": split_counts,
        "selection_note": "All selected supervised-small samples are treated as train-only fitting smoke.",
    }


def _coverage_for_examples(examples: list[Any]) -> dict[str, Any]:
    records = []
    for example in examples:
        records.extend(
            audit_coords(
                sample_id=example.sample_id,
                split="train_only_smoke",
                coords=np.asarray(example.condition.coords),
                seeds=[0],
                policies=[policy["audit_policy"] for policy in POLICIES.values()],
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
        "meets_20pct_relative_rmse": (
            bool(relative_rmse <= TARGET_RELATIVE_ERROR) if relative_rmse is not None else False
        ),
        "relative_rmse_gap_to_20pct": (
            max(float(relative_rmse - TARGET_RELATIVE_ERROR), 0.0)
            if relative_rmse is not None
            else None
        ),
        "finite": finite,
        "shape_ok": shape_ok,
    }


def _grad_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "median": None,
            "max": None,
            "final": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
    }


def _run_policy(
    *,
    policy_name: str,
    examples: list[Any],
    epochs: int,
    lr: float,
    seed: int,
    legacy_edge_totals: dict[str, int],
) -> dict[str, Any]:
    builder = _policy_builder(policy_name)
    stats = _train_only_stats(examples)
    build_start = time.perf_counter()
    groups = _make_groups(examples, stats, builder)
    graph_build_time = time.perf_counter() - build_start
    edge_totals = _edge_totals(groups)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    first_group = groups[0]
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=first_group["inputs"],
        graphs=first_group["graphs"],
    )["params"]

    def loss_fn(current_params):
        return _weighted_loss(model, current_params, groups)

    losses = [float(loss_fn(params))]
    initial_metrics = _metrics(model, params, groups, stats)
    grad_norms: list[float] = []
    grad_finite = True
    train_start = time.perf_counter()
    for _ in range(epochs):
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        losses.append(float(loss_fn(params)))
    train_time = time.perf_counter() - train_start
    final_metrics = _metrics(model, params, groups, stats)
    finite = bool(
        grad_finite
        and initial_metrics["finite"]
        and final_metrics["finite"]
        and initial_metrics["shape_ok"]
        and final_metrics["shape_ok"]
        and np.all(np.isfinite(np.asarray(losses)))
    )
    if not finite:
        raise AssertionError(f"{policy_name}: non-finite P2-b smoke result")
    initial_loss = float(losses[0])
    final_loss = float(losses[-1])
    best_loss = float(np.min(np.asarray(losses)))
    return {
        "policy": policy_name,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_epoch": int(np.argmin(np.asarray(losses))),
        "loss_drop": initial_loss - final_loss,
        "loss_drop_ratio": (
            float((initial_loss - final_loss) / initial_loss) if initial_loss else None
        ),
        "losses": [float(value) for value in losses],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "raw_delta_rmse": final_metrics["raw_delta_rmse"],
        "raw_delta_mae": final_metrics["raw_delta_mae"],
        "relative_rmse": final_metrics["relative_rmse"],
        "relative_mae": final_metrics["relative_mae"],
        "meets_20pct_relative_rmse": final_metrics["meets_20pct_relative_rmse"],
        "relative_rmse_gap_to_20pct": final_metrics["relative_rmse_gap_to_20pct"],
        "grad_norm": _grad_stats(grad_norms),
        "grad_finite": grad_finite,
        "finite": finite,
        "shape_ok": final_metrics["shape_ok"],
        "group_count": len(groups),
        "graph_build_time_seconds": float(graph_build_time),
        "train_time_seconds": float(train_time),
        "train_step_time_seconds": float(train_time / max(epochs, 1)),
        "edge_totals": edge_totals,
        "edge_ratio_vs_legacy": {
            name: (
                float(edge_totals[name] / legacy_edge_totals[name])
                if legacy_edge_totals[name]
                else None
            )
            for name in ("p2r", "r2p", "r2r")
        },
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    rows = []
    coverage = payload["coverage"]["summary"]
    for result in payload["policy_results"]:
        ratio = result["edge_ratio_vs_legacy"]
        audit_policy = POLICIES[result["policy"]]["audit_policy"]
        coverage_row = coverage[audit_policy]
        rows.append(
            "| {policy} | {initial_loss:.6e} | {final_loss:.6e} | {best_loss:.6e} | "
            "{loss_drop:.6e} | {rmse:.6e} / {mae:.6e} | {rel:.3%} | {zero_p2r}/{zero_r2p} | "
            "{edge_ratio:.3f}/{r2p_ratio:.3f} | {meets} |".format(
                policy=result["policy"],
                initial_loss=result["initial_loss"],
                final_loss=result["final_loss"],
                best_loss=result["best_loss"],
                loss_drop=result["loss_drop"],
                rmse=result["raw_delta_rmse"],
                mae=result["raw_delta_mae"],
                rel=(result["relative_rmse"] or 0.0),
                zero_p2r=coverage_row["p2r_zero_count_total"],
                zero_r2p=coverage_row["r2p_zero_count_total"],
                edge_ratio=ratio["p2r"],
                r2p_ratio=ratio["r2p"],
                meets=result["meets_20pct_relative_rmse"],
            )
        )
    lines = [
        "# Heat3D v3 P2-b 16-sample Longer Policy Smoke",
        "",
        f"Epochs: `{payload['config']['epochs']}`. LR: `{payload['config']['lr']}`. "
        "Train-only smoke; no checkpoint.",
        "",
        "| policy | initial_loss | final_loss | best_loss | loss_drop | RMSE / MAE | relative RMSE | p2r/r2p zero | edge_ratio p2r/r2p | <=20% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(rows)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.sample_count != 16:
        raise ValueError("This P2-b script is intentionally limited to --sample-count 16")

    examples, dataset_metadata = _load_examples(args)
    coverage = _coverage_for_examples(examples)

    legacy_stats = _train_only_stats(examples)
    legacy_groups = _make_groups(examples, legacy_stats, _policy_builder("legacy"))
    legacy_edge_totals = _edge_totals(legacy_groups)

    policy_results = []
    for policy_name in POLICIES:
        policy_results.append(
            _run_policy(
                policy_name=policy_name,
                examples=examples,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                legacy_edge_totals=legacy_edge_totals,
            )
        )

    payload = {
        "schema_version": "heat3d_v3_p2_policy_16sample_longer_smoke_v1",
        "diagnostic_scope": "P2-b 16-sample train-only longer smoke; no checkpoint and no full dataset",
        "config": {
            "subset": str(args.subset),
            "k_encoding_mode": args.k_encoding_mode,
            "sample_count": args.sample_count,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "model_config": MODEL_CONFIG,
            "target_relative_error": TARGET_RELATIVE_ERROR,
            "policies": POLICIES,
        },
        "dataset": dataset_metadata,
        "coverage": coverage,
        "legacy_edge_totals": legacy_edge_totals,
        "policy_results": policy_results,
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    stem = f"p2_policy_16sample_longer_e{args.epochs}"
    json_path = _write_json(output_dir / f"{stem}.json", payload)
    md_path = _write_text(output_dir / f"{stem}.md", _markdown_summary(payload))

    print("Heat3D v3 P2-b 16-sample longer policy smoke")
    print(f"  subset: {dataset_metadata['sample_root']}")
    print(f"  selected samples: {dataset_metadata['selected_sample_ids']}")
    print(f"  original split counts: {dataset_metadata['original_split_counts_in_train_only_smoke']}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    for result in policy_results:
        ratio = result["edge_ratio_vs_legacy"]
        audit_policy = POLICIES[result["policy"]]["audit_policy"]
        coverage_row = coverage["summary"][audit_policy]
        print(
            f"  {result['policy']}: "
            f"loss {result['initial_loss']:.6e}->{result['final_loss']:.6e} "
            f"best={result['best_loss']:.6e} "
            f"drop={result['loss_drop']:.6e} "
            f"rmse={result['raw_delta_rmse']:.6e} "
            f"mae={result['raw_delta_mae']:.6e} "
            f"relative_rmse={result['relative_rmse']:.6f} "
            f"zero={coverage_row['p2r_zero_count_total']}/"
            f"{coverage_row['r2p_zero_count_total']} "
            f"edge_ratio={ratio['p2r']:.3f}/{ratio['r2p']:.3f} "
            f"<=20%={result['meets_20pct_relative_rmse']} "
            f"finite={result['finite']}"
        )
    print(f"wrote={json_path}")
    print(f"wrote={md_path}")
    print("Heat3D v3 P2-b 16-sample longer policy smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
