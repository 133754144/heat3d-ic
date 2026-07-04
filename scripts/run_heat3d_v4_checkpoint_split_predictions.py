#!/usr/bin/env python3
"""Read-only V4 checkpoint inference for one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _predict_temperatures,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
    _validate_model_config,
    _write_json,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


PREDICTION_SPLITS = ("valid_iid", "test_iid", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run read-only Heat3D V4 params checkpoint inference for one split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument("--prediction-split", choices=PREDICTION_SPLITS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-name", type=str, default="predictions.npz")
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--progress-detail", choices=("off", "basic"), default="basic")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_ignored_output(path: Path, label: str) -> None:
    parts = set(path.parts)
    if "data" in parts or "checkpoints" in parts or "logs" in parts:
        raise ValueError(f"--{label} must not be under data/checkpoints/logs: {path}")
    if "output" not in parts:
        raise ValueError(f"--{label} must be under ignored output/: {path}")


def _sample_ids_for_split(
    sample_root: Path,
    split_map: Path | None,
    prediction_split: str,
) -> list[str]:
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    sample_ids = list(split_ids.get(prediction_split) or [])
    if not sample_ids:
        raise ValueError(
            f"No samples resolved for prediction_split={prediction_split}; split_source={split_source}"
        )
    return sample_ids


def _load_split_examples(
    *,
    sample_root: Path,
    sample_ids: list[str],
    checkpoint_stats: dict[str, Any],
    boundary_mask_fallback: bool,
) -> list[Any]:
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=k_encoding_mode,
        boundary_mask_fallback=boundary_mask_fallback,
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"subset missing prediction split samples: {missing[:10]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def main() -> int:
    args = parse_args()
    if args.prediction_batch_size < 1:
        raise ValueError("--prediction-batch-size must be >= 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
    if not args.run_config.is_file():
        raise FileNotFoundError(f"--run-config not found: {args.run_config}")
    if "/" in args.predictions_name or args.predictions_name in {"", ".", ".."}:
        raise ValueError("--predictions-name must be a plain file name")

    _ensure_ignored_output(args.output_dir, "output-dir")
    run_config = _load_json(args.run_config)
    checkpoint_payload = _load_params_checkpoint(args.checkpoint)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise ValueError(f"{args.checkpoint}: missing train_only_normalization payload")

    sample_root = _sample_root(args.subset or Path(run_config["subset"]))
    split_map = args.split_map
    if split_map is None and run_config.get("split_map_path"):
        split_map = Path(str(run_config["split_map_path"]))
    sample_ids = _sample_ids_for_split(sample_root, split_map, args.prediction_split)

    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    prediction_examples = _load_split_examples(
        sample_root=sample_root,
        sample_ids=sample_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)

    model_config = dict(checkpoint_payload.get("model_config") or run_config.get("model_config") or {})
    if not model_config:
        raise ValueError(f"{args.checkpoint}: missing model_config payload")
    model_config = _resolve_decoder_bypass_model_config(model_config, stats)
    _validate_model_config(model_config)
    graph_config = dict(run_config.get("graph_config") or {})
    graph_seed = int(run_config.get("graph_seed", 0))

    builder = Heat3DGraphBuilder(**graph_config)
    progress = args.progress_detail != "off"
    groups = _make_groups_with_progress(
        prediction_examples,
        stats,
        builder,
        args.prediction_split,
        progress,
        args.progress_detail,
        graph_seed,
        batch_size=int(args.prediction_batch_size),
        drop_last=False,
    )
    model = GraphNeuralOperator(**model_config)
    predictions = _predict_temperatures(
        model,
        _device_params(checkpoint_payload["params"]),
        groups,
        stats,
    )
    if set(predictions) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(predictions))
        extra = sorted(set(predictions) - set(sample_ids))
        raise RuntimeError(f"prediction key mismatch missing={missing[:5]} extra={extra[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / args.predictions_name
    _ensure_ignored_output(predictions_path, "predictions-name")
    np.savez_compressed(predictions_path, **predictions)
    manifest = {
        "diagnostic_scope": "read-only checkpoint split prediction export",
        "checkpoint": str(args.checkpoint),
        "run_config": str(args.run_config),
        "subset": str(sample_root),
        "split_map": str(split_map) if split_map is not None else None,
        "prediction_split": args.prediction_split,
        "prediction_batch_size": int(args.prediction_batch_size),
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "predictions_path": str(predictions_path),
    }
    _write_json(args.output_dir / "prediction_manifest.json", manifest)
    print(
        "checkpoint_split_predictions "
        f"split={args.prediction_split} sample_count={len(sample_ids)} "
        f"predictions={predictions_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
