#!/usr/bin/env python3
"""Heat3D v3 P2-a small training smoke for graph policy comparison.

This is a constrained smoke, not a formal training experiment. It compares
legacy, nearest repair, and discrete radius on 1-sample and 4-sample
supervised-small train-only subsets.
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

from audit_heat3d_v3_graph_coverage import (  # noqa: E402
    POLICY_CURRENT,
    POLICY_DISCRETE_COVERAGE_RADIUS,
    POLICY_NEAREST_REPAIR,
    audit_coords,
    summarize_records,
)
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _global_norm,
    _load_manifest,
    _make_groups,
    _resolve_split_ids,
    _sample_root,
    _train_only_stats,
    _weighted_loss,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


POLICIES = {
    "legacy": {
        "audit_policy": POLICY_CURRENT,
        "builder_kwargs": {
            "radius_policy": "legacy_kdtree_mean4",
            "coverage_repair_policy": "none",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
    "nearest_repair": {
        "audit_policy": POLICY_NEAREST_REPAIR,
        "builder_kwargs": {
            "radius_policy": "legacy_kdtree_mean4",
            "coverage_repair_policy": "nearest_rnode",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
    "discrete_radius": {
        "audit_policy": POLICY_DISCRETE_COVERAGE_RADIUS,
        "builder_kwargs": {
            "radius_policy": "discrete_physical_coverage",
            "coverage_repair_policy": "none",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v3_p2_policy_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--sample-counts", default="1,4")
    parser.add_argument("--epochs-1-sample", type=int, default=20)
    parser.add_argument("--epochs-4-sample", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _parse_sample_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts:
        raise ValueError("--sample-counts must contain at least one positive integer")
    if any(count < 1 for count in counts):
        raise ValueError("--sample-counts must be positive")
    if max(counts) > 4:
        raise ValueError("P2-a smoke is limited to <=4 samples in this script")
    return list(dict.fromkeys(counts))


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


def _policy_builder(policy_name: str) -> Heat3DGraphBuilder:
    return Heat3DGraphBuilder(**POLICIES[policy_name]["builder_kwargs"])


def _load_train_examples(args: argparse.Namespace, needed_count: int) -> tuple[list[Any], dict[str, Any]]:
    sample_root = _sample_root(args.subset)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Heat3D subset sample root does not exist: {sample_root}")
    manifest = _load_manifest(args.manifest) if args.manifest.is_file() else {"samples": []}
    split_ids, split_source = _resolve_split_ids(manifest, sample_root)
    train_ids = split_ids.get("train", [])
    if len(train_ids) < needed_count:
        raise ValueError(
            f"Need at least {needed_count} train samples, found {len(train_ids)} in {sample_root}"
        )
    selected_ids = train_ids[:needed_count]
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode=args.k_encoding_mode)
    index_by_id = dataset.sample_index_by_id()
    examples = [dataset[index_by_id[sample_id]] for sample_id in selected_ids]
    metadata = {
        "sample_root": str(sample_root),
        "split_source": split_source,
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "selected_train_ids": selected_ids,
    }
    return examples, metadata


def _r2p_edges(metadata: Any) -> Any:
    if metadata.r2p_edge_indices is None:
        return jnp.flip(metadata.p2r_edge_indices, axis=-1)
    return metadata.r2p_edge_indices


def _count_real_edges(edges: Any, sender_count: int, receiver_count: int) -> int:
    values = np.asarray(edges)
    if values.ndim == 2:
        values = values[None, ...]
    values = values.astype(np.int64)
    real = (
        (values[..., 0] >= 0)
        & (values[..., 0] < sender_count)
        & (values[..., 1] >= 0)
        & (values[..., 1] < receiver_count)
    )
    return int(np.sum(real))


def _edge_totals(groups: list[dict]) -> dict[str, int]:
    totals = {"p2r": 0, "r2p": 0, "r2r": 0}
    for group in groups:
        metadata = group["metadata"]
        n_pnodes_inp = int(np.asarray(metadata.x_pnodes_inp).shape[1] - 1)
        n_pnodes_out = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
        n_rnodes = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
        totals["p2r"] += _count_real_edges(
            metadata.p2r_edge_indices,
            sender_count=n_pnodes_inp,
            receiver_count=n_rnodes,
        )
        totals["r2p"] += _count_real_edges(
            _r2p_edges(metadata),
            sender_count=n_rnodes,
            receiver_count=n_pnodes_out,
        )
        totals["r2r"] += _count_real_edges(
            metadata.r2r_edge_indices,
            sender_count=n_rnodes,
            receiver_count=n_rnodes,
        )
    return totals


def _metrics(model: GraphNeuralOperator, params: Any, groups: list[dict], stats: dict) -> dict[str, Any]:
    finite = True
    shape_ok = True
    raw_sse = 0.0
    raw_sae = 0.0
    raw_count = 0
    normalized_sse = 0.0
    normalized_count = 0
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        target_normalized = group["target_normalized"]
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        raw_error = np.asarray(pred_delta - group["target_delta_raw"], dtype=np.float64)
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
        normalized_sse += float(np.sum(np.square(normalized_error)))
        normalized_count += int(normalized_error.size)
    return {
        "normalized_mse": normalized_sse / max(normalized_count, 1),
        "raw_delta_rmse": float(np.sqrt(raw_sse / max(raw_count, 1))),
        "raw_delta_mae": raw_sae / max(raw_count, 1),
        "finite": finite,
        "shape_ok": shape_ok,
    }


def _coverage_for_examples(examples: list[Any]) -> dict[str, Any]:
    records = []
    for example in examples:
        records.extend(
            audit_coords(
                sample_id=example.sample_id,
                split=str(example.meta.get("split", "train")),
                coords=np.asarray(example.condition.coords),
                seeds=[0],
                policies=[policy["audit_policy"] for policy in POLICIES.values()],
            )
        )
    return {
        "records": records,
        "summary": summarize_records(records),
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

    initial_loss = float(loss_fn(params))
    initial_metrics = _metrics(model, params, groups, stats)
    losses = [initial_loss]
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

    final_loss = float(losses[-1])
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
        raise AssertionError(f"{policy_name}: training smoke produced non-finite values")
    return {
        "policy": policy_name,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": initial_loss - final_loss,
        "loss_decreased": bool(final_loss < initial_loss),
        "losses": [float(value) for value in losses],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "raw_delta_rmse": final_metrics["raw_delta_rmse"],
        "raw_delta_mae": final_metrics["raw_delta_mae"],
        "grad_norm_initial": float(grad_norms[0]) if grad_norms else None,
        "grad_norm_final": float(grad_norms[-1]) if grad_norms else None,
        "grad_norm_max": float(np.max(np.asarray(grad_norms))) if grad_norms else None,
        "grad_finite": grad_finite,
        "finite": finite,
        "shape_ok": final_metrics["shape_ok"],
        "group_count": len(groups),
        "graph_build_time_seconds": float(graph_build_time),
        "train_time_seconds": float(train_time),
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


def _epochs_for_count(args: argparse.Namespace, sample_count: int) -> int:
    if sample_count == 1:
        return args.epochs_1_sample
    if sample_count == 4:
        return args.epochs_4_sample
    return min(args.epochs_1_sample, args.epochs_4_sample)


def _markdown_summary(payload: dict[str, Any]) -> str:
    rows = []
    for sample_result in payload["sample_results"]:
        sample_count = sample_result["sample_count"]
        for result in sample_result["policy_results"]:
            ratio = result["edge_ratio_vs_legacy"]
            rows.append(
                "| {policy} | {sample_count} | {initial_loss:.6e} | "
                "{final_loss:.6e} | {loss_drop:.6e} | {rmse:.6e} / {mae:.6e} | "
                "{edge_ratio:.3f}/{r2p_ratio:.3f} | {finite} |".format(
                    policy=result["policy"],
                    sample_count=sample_count,
                    initial_loss=result["initial_loss"],
                    final_loss=result["final_loss"],
                    loss_drop=result["loss_drop"],
                    rmse=result["raw_delta_rmse"],
                    mae=result["raw_delta_mae"],
                    edge_ratio=ratio["p2r"],
                    r2p_ratio=ratio["r2p"],
                    finite=result["finite"],
                )
            )

    lines = [
        "# Heat3D v3 P2-a Policy Small Training Smoke",
        "",
        "Scope: train-only smoke on supervised-small 1-sample and 4-sample selections. "
        "No checkpoint, no full dataset, no model/decoder/loss changes.",
        "",
        "| policy | sample_count | initial_loss | final_loss | loss_drop | RMSE / MAE | edge_ratio p2r/r2p | finite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(rows)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    sample_counts = _parse_sample_counts(args.sample_counts)
    if args.epochs_1_sample < 1 or args.epochs_4_sample < 1:
        raise ValueError("epoch counts must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")

    all_examples, dataset_metadata = _load_train_examples(args, max(sample_counts))
    sample_results = []
    for sample_count in sample_counts:
        examples = all_examples[:sample_count]
        coverage = _coverage_for_examples(examples)
        legacy_builder = _policy_builder("legacy")
        legacy_stats = _train_only_stats(examples)
        legacy_groups = _make_groups(examples, legacy_stats, legacy_builder)
        legacy_edge_totals = _edge_totals(legacy_groups)
        policy_results = []
        for policy_name in POLICIES:
            policy_results.append(
                _run_policy(
                    policy_name=policy_name,
                    examples=examples,
                    epochs=_epochs_for_count(args, sample_count),
                    lr=args.lr,
                    seed=args.seed,
                    legacy_edge_totals=legacy_edge_totals,
                )
            )
        sample_results.append(
            {
                "sample_count": sample_count,
                "sample_ids": [example.sample_id for example in examples],
                "epochs": _epochs_for_count(args, sample_count),
                "legacy_edge_totals": legacy_edge_totals,
                "coverage": coverage,
                "policy_results": policy_results,
            }
        )

    payload = {
        "schema_version": "heat3d_v3_p2_policy_small_training_smoke_v1",
        "diagnostic_scope": "P2-a small train-only smoke; no checkpoint and no full dataset",
        "config": {
            "subset": str(args.subset),
            "manifest": str(args.manifest),
            "k_encoding_mode": args.k_encoding_mode,
            "sample_counts": sample_counts,
            "epochs_1_sample": args.epochs_1_sample,
            "epochs_4_sample": args.epochs_4_sample,
            "lr": args.lr,
            "seed": args.seed,
            "policies": POLICIES,
        },
        "dataset": dataset_metadata,
        "sample_results": sample_results,
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    json_path = _write_json(output_dir / "p2_policy_small_training_smoke.json", payload)
    md_path = _write_text(
        output_dir / "p2_policy_small_training_smoke.md",
        _markdown_summary(payload),
    )

    print("Heat3D v3 P2-a policy small training smoke")
    print(f"  subset: {dataset_metadata['sample_root']}")
    print(f"  selected train ids: {dataset_metadata['selected_train_ids']}")
    print(f"  lr: {args.lr}")
    for sample_result in sample_results:
        print(f"  sample_count={sample_result['sample_count']} epochs={sample_result['epochs']}")
        for result in sample_result["policy_results"]:
            ratio = result["edge_ratio_vs_legacy"]
            print(
                f"    {result['policy']}: "
                f"loss {result['initial_loss']:.6e}->{result['final_loss']:.6e} "
                f"drop={result['loss_drop']:.6e} "
                f"rmse={result['raw_delta_rmse']:.6e} "
                f"mae={result['raw_delta_mae']:.6e} "
                f"edge_ratio={ratio['p2r']:.3f}/{ratio['r2p']:.3f} "
                f"finite={result['finite']}"
            )
    print(f"wrote={json_path}")
    print(f"wrote={md_path}")
    print("Heat3D v3 P2-a policy small training smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
