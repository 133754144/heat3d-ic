#!/usr/bin/env python3
"""Evaluate one completed Gate 6H run from persisted valid-only predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
from typing import Any

import numpy as np


EXPECTED_IDS = {
    "V4P5_28_gate6h_v13_stopgrad_scratch_e600",
    "V4P5_30_gate6h_v13_deep_scale_head_scratch_e600",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config-id", required=True, choices=sorted(EXPECTED_IDS))
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    stats = payload["train_only_normalization"]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "epoch": int(payload["epoch"]),
        "checkpoint_kind": str(payload["checkpoint_kind"]),
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "target_delta_mean": float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0]),
        "target_delta_std": float(np.asarray(stats["target_delta_std"]).reshape(-1)[0]),
    }


def _predictions(path: Path, ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ids):
            raise ValueError(f"{path}: prediction keys differ from valid_iid")
        fields = {
            sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            for sample_id in ids
        }
    if any(array.size != 1024 or not np.all(np.isfinite(array)) for array in fields.values()):
        raise ValueError(f"{path}: invalid prediction arrays")
    return fields


def _cv_weights(coords: np.ndarray) -> np.ndarray:
    axes = [np.unique(coords[:, index]) for index in range(3)]
    one_d = []
    for axis in axes:
        edges = np.empty(axis.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
        edges[0], edges[-1] = axis[0], axis[-1]
        one_d.append(np.maximum(np.diff(edges), 0.0))
    lookup = [
        {float(value): float(weight) for value, weight in zip(axis, weights)}
        for axis, weights in zip(axes, one_d)
    ]
    return np.asarray([
        lookup[0][float(point[0])]
        * lookup[1][float(point[1])]
        * lookup[2][float(point[2])]
        for point in coords
    ])


def _targets(data_root: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    targets = {}
    for sample_id in ids:
        sample = data_root / sample_id
        meta = json.loads((sample / "sample_meta.json").read_text(encoding="utf-8"))
        if meta.get("split") != "valid_iid":
            raise ValueError(f"{sample_id}: forbidden non-valid role")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        targets[sample_id] = {
            "bottom": bottom,
            "target": np.load(sample / "temperature.npy").astype(np.float64).reshape(-1) - bottom,
            "weights": _cv_weights(
                np.load(sample / "coords.npy").astype(np.float64).reshape(-1, 3)
            ),
        }
    return targets


def _metrics(
    fields: dict[str, np.ndarray],
    targets: dict[str, dict[str, Any]],
    ids: list[str],
    target_std: float,
) -> dict[str, float]:
    point_error_sse = 0.0
    point_true_sse = 0.0
    cv_error_sse = 0.0
    cv_volume = 0.0
    normalized_sse = 0.0
    sample_relative = []
    for sample_id in ids:
        item = targets[sample_id]
        error = fields[sample_id] - item["bottom"] - item["target"]
        true = item["target"]
        weights = item["weights"]
        point_error_sse += float(np.sum(np.square(error)))
        point_true_sse += float(np.sum(np.square(true)))
        sample_cv_error = float(np.sum(np.square(error) * weights))
        sample_cv_true = float(np.sum(np.square(true) * weights))
        cv_error_sse += sample_cv_error
        cv_volume += float(np.sum(weights))
        normalized_sse += float(np.sum(np.square(error) / (target_std * target_std)))
        sample_relative.append(100.0 * math.sqrt(sample_cv_error / sample_cv_true))
    return {
        "legacy_normalized_valid_base_mse": normalized_sse / (len(ids) * 1024),
        "point_global_relative_rmse_pct": 100.0 * math.sqrt(point_error_sse / point_true_sse),
        "sample_first_cv_relative_rmse_pct": float(np.mean(sample_relative)),
        "raw_cv_weighted_rmse_K": math.sqrt(cv_error_sse / cv_volume),
    }


def _epoch_row(summary: dict[str, Any], epoch: int) -> dict[str, Any]:
    row = next(item for item in summary["epoch_history"] if int(item["epoch"]) == epoch)
    keys = (
        "epoch",
        "valid_shape_cv_loss",
        "valid_log_scale_loss",
        "valid_relative_field_loss",
        "valid_raw_absolute_field_loss",
        "valid_loss",
        "valid_base_mse",
        "valid_rel_rmse_v4_pct",
        "valid_native_joint_amplitude_ratio",
        "valid_native_oracle_scale_relative_rmse",
        "valid_native_oracle_shape_relative_rmse",
    )
    return {key: row.get(key) for key in keys}


def main() -> int:
    args = _args()
    root = args.root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir = run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "loss_summary.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if run_dir.name != args.config_id:
        raise ValueError("config_id/run directory binding failed")
    configured_output = Path(run_config["output_dir"])
    if configured_output.name != args.config_id:
        raise ValueError("config_id/run_config output binding failed")
    if int(summary["final_epoch"]) != 600:
        raise ValueError("run is not completed e600")

    standardizer = run_config["global_context"]["standardizer"]
    if standardizer["fit_population"] != "train_only" or int(standardizer["fit_sample_count"]) != 672:
        raise ValueError("Global Context standardizer is not train-only")
    if bool(standardizer["target_or_label_derived_inputs"]):
        raise ValueError("Global Context contains target-derived inputs")

    split_path = root / run_config["split_map_path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = sorted(
        sample_id
        for sample_id, role in split["sample_splits"].items()
        if role == "valid_iid"
    )
    if len(ids) != 128:
        raise ValueError("valid_iid must have 128 samples")
    targets = _targets(root / run_config["subset"], ids)

    checkpoints = {
        "best": _checkpoint(run_dir / "params_best.pkl"),
        "final": _checkpoint(run_dir / "params_final.pkl"),
    }
    if checkpoints["best"]["epoch"] != int(summary["best_epoch"]):
        raise ValueError("best checkpoint epoch differs from loss summary")
    if checkpoints["final"]["epoch"] != 600:
        raise ValueError("final checkpoint is not epoch 600")
    if checkpoints["best"]["train_stats_hash"] != checkpoints["final"]["train_stats_hash"]:
        raise ValueError("best/final train normalization differs")

    prediction_paths = {
        "best": run_dir / "best_predictions.npz",
        "final": run_dir / "predictions.npz",
    }
    predictions = {name: _predictions(path, ids) for name, path in prediction_paths.items()}
    metrics = {
        name: _metrics(
            predictions[name],
            targets,
            ids,
            checkpoints[name]["target_delta_std"],
        )
        for name in ("best", "final")
    }
    artifacts = {
        path.name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (
            run_config_path,
            summary_path,
            run_dir / "params_best.pkl",
            run_dir / "params_final.pkl",
            prediction_paths["best"],
            prediction_paths["final"],
        )
    }
    evaluator_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    payload = {
        "schema_version": "heat3d_v5_gate6h_valid_only_true_rms_v1",
        "status": "completed",
        "config_id": args.config_id,
        "run_directory": str(run_dir),
        "training_commit": str(summary["code_version_or_git_commit"]),
        "evaluator_commit": evaluator_commit,
        "evaluator_source_sha256": _sha256(Path(__file__)),
        "scope": {
            "roles_accessed": ["valid_iid"],
            "forbidden_roles_accessed": [],
            "sealed_iid_accessed": False,
            "model_inference_run": False,
            "training_started": False,
            "sample_count": len(ids),
            "nodes_per_sample": 1024,
            "valid_sample_ids_sha256": _ids_hash(ids),
        },
        "train_only_standardizer": {
            "fit_population": standardizer["fit_population"],
            "fit_sample_count": int(standardizer["fit_sample_count"]),
            "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
            "target_or_label_derived_inputs": bool(
                standardizer["target_or_label_derived_inputs"]
            ),
        },
        "formulas": {
            "point_global_relative_rmse_pct": "100*sqrt(sum_nodes_samples(error_deltaT^2)/sum_nodes_samples(true_deltaT^2))",
            "sample_first_cv_relative_rmse_pct": "100*mean_samples(CV_RMS(error)/CV_RMS(true_deltaT))",
            "legacy_normalized_valid_base_mse": "mean(((prediction_deltaT-true_deltaT)/train_target_delta_std)^2)",
            "raw_cv_weighted_rmse_K": "sqrt(sum(error_deltaT^2*CV)/sum(CV))",
        },
        "selection": {
            "metric": str(summary["selection_metric"]),
            "best_epoch": int(summary["best_epoch"]),
            "final_epoch": 600,
        },
        "metrics": metrics,
        "epoch_25": _epoch_row(summary, 25),
        "checkpoints": checkpoints,
        "artifacts": artifacts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
