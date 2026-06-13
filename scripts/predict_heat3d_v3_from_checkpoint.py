#!/usr/bin/env python3
"""Export Heat3D predictions from a saved v3 params-only checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import jax.numpy as jnp
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    DEFAULT_SUBSET,
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _device_params,
    _ensure_ignored_output_file,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _predict_temperatures,
    _resolve_training_splits,
    _sample_root,
    _stable_json_hash,
    _train_only_stats,
    _write_json,
)


PREDICTION_SPLITS = (
    "all",
    "train",
    "valid_iid",
    "valid_stress",
    "test_id",
    "test_ood_bc",
    "test_ood_stack",
    "test_ood_combined",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Heat3D v3 checkpoint prediction export."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument("--prediction-split", choices=PREDICTION_SPLITS, default="all")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def _optional_batch_size(value: int) -> int | None:
    if int(value) == 0:
        return None
    if int(value) < 0:
        raise ValueError("--batch-size must be >= 1 or 0 for one full prediction batch")
    return int(value)


def _stats_from_checkpoint_payload(
    checkpoint_stats: dict[str, Any],
    train_examples: list[Any],
) -> dict[str, Any]:
    stats = _train_only_stats(train_examples)
    checkpoint_feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    if checkpoint_feature_names and checkpoint_feature_names != tuple(stats["feature_names"]):
        raise ValueError(
            "Checkpoint feature_names do not match the selected subset train split: "
            f"checkpoint={checkpoint_feature_names} subset={tuple(stats['feature_names'])}"
        )
    stats["feature_names"] = checkpoint_feature_names or tuple(stats["feature_names"])
    stats["target_delta_mean"] = jnp.asarray(
        np.asarray(checkpoint_stats["target_delta_mean"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["target_delta_std"] = jnp.asarray(
        np.asarray(checkpoint_stats["target_delta_std"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["condition_mean"] = jnp.asarray(
        np.asarray(checkpoint_stats["condition_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    stats["condition_std"] = jnp.asarray(
        np.asarray(checkpoint_stats["condition_std"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    return stats


def _selected_prediction_ids(
    split_ids: dict[str, list[str]],
    prediction_split: str,
) -> list[str]:
    if prediction_split == "all":
        return sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    ids = split_ids.get(prediction_split, [])
    if not ids:
        raise ValueError(
            f"prediction split {prediction_split!r} is empty or missing; "
            f"available={sorted(split_ids)}"
        )
    return list(ids)


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
    batch_size = _optional_batch_size(args.batch_size)
    output_npz = _ensure_ignored_output_file(args.output_npz, "output-npz")
    output_json = _ensure_ignored_output_file(args.output_json, "output-json")

    payload = _load_params_checkpoint(args.checkpoint)
    model_config = dict(payload.get("model_config") or {})
    if not model_config:
        raise ValueError(f"{args.checkpoint}: checkpoint missing model_config")
    checkpoint_stats = dict(payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise ValueError(f"{args.checkpoint}: checkpoint missing train_only_normalization")

    sample_root = _sample_root(args.subset)
    split_ids, split_source, primary_validation_split, stress_validation_split = _resolve_training_splits(
        sample_root,
        args.split_map,
    )
    train_ids = split_ids["train"]
    prediction_ids = _selected_prediction_ids(split_ids, args.prediction_split)
    run_metadata = payload.get("run_config_metadata") if isinstance(payload.get("run_config_metadata"), dict) else {}
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode="diag3",
        boundary_mask_fallback=bool(run_metadata.get("boundary_mask_fallback", False)),
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in sorted(set(train_ids + prediction_ids)) if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset loader did not expose samples: {missing}")

    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    prediction_examples = [dataset[index_by_id[sample_id]] for sample_id in prediction_ids]
    stats = _stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    graph_config = dict(run_metadata.get("graph_config") or {})
    graph_config_source = "checkpoint_run_config_metadata" if graph_config else "builder_defaults"
    builder = Heat3DGraphBuilder(**graph_config)
    seed_config = run_metadata.get("seed_config") if isinstance(run_metadata.get("seed_config"), dict) else {}
    graph_seed = int(seed_config.get("graph_seed", 0) or 0)
    groups = _make_groups_with_progress(
        prediction_examples,
        stats,
        builder,
        args.prediction_split,
        False,
        "off",
        graph_seed,
        batch_size=batch_size,
        drop_last=False,
        profile_counts=None,
    )
    model = GraphNeuralOperator(**model_config)
    predictions = _predict_temperatures(model, _device_params(payload["params"]), groups, stats)
    np.savez_compressed(output_npz, **predictions)
    metadata = {
        "diagnostic_scope": "prediction-only checkpoint inference; no training",
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema_version": payload.get("schema_version"),
        "checkpoint_format_version": payload.get("checkpoint_format_version"),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_record": payload.get("record"),
        "checkpoint_git_commit": payload.get("git_commit"),
        "model_config_hash": payload.get("model_config_hash") or _stable_json_hash(model_config),
        "train_stats_hash": payload.get("train_stats_hash"),
        "subset": str(sample_root),
        "split_map": str(args.split_map) if args.split_map is not None else None,
        "split_source": split_source,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "prediction_split": args.prediction_split,
        "sample_count": len(prediction_ids),
        "prediction_key_count": len(predictions),
        "sample_ids": prediction_ids,
        "batch_size": batch_size,
        "group_count": len(groups),
        "group_sample_counts": [len(group["sample_ids"]) for group in groups],
        "graph_config_source": graph_config_source,
        "graph_seed": graph_seed,
        "boundary_mask_fallback": bool(run_metadata.get("boundary_mask_fallback", False)),
        "graph_config": graph_config,
        "output_npz": str(output_npz),
        "elapsed_s": time.perf_counter() - start,
    }
    _write_json(output_json, metadata)
    print(
        "prediction-only export complete: "
        f"samples={len(prediction_ids)} groups={len(groups)} "
        f"npz={output_npz} metadata={output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
