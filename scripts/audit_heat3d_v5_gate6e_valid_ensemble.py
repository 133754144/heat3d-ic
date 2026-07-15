#!/usr/bin/env python3
"""Evaluate fixed-alpha field ensembles on valid_iid only, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v5_metrics import control_volume_weights  # noqa: E402


ALPHAS = (0.25, 0.5, 0.75)
PAIRS = (
    ("n3_best_e402", "l2_best_e353"),
    ("n3_best_e402", "l2_final_e600"),
    ("l2_best_e353", "l2_final_e600"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/heat3d_v4_p5_clean_nohard_v0")
    parser.add_argument(
        "--split-map", type=Path,
        default=ROOT / "configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json",
    )
    parser.add_argument("--n3-run", type=Path, default=ROOT / "output/heat3d_v5_runs/V4P5_07_native_pooled_latent_global_film")
    parser.add_argument("--l2-run", type=Path, default=ROOT / "output/heat3d_v5_gate6c_runs/V4P5_12_gate6c_scratch_l2_shape_balanced")
    parser.add_argument(
        "--gate6e-config", type=Path,
        default=ROOT / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "configs/heat3d_v5/gate6e/valid_only_ensemble_audit.json",
    )
    parser.add_argument(
        "--output-md", type=Path, default=ROOT / "docs/v5_gate6e_valid_only_ensemble_audit.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_meta(path: Path, expected_epoch: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    epoch = int(payload["epoch"])
    if epoch != expected_epoch:
        raise ValueError(f"{path}: expected epoch {expected_epoch}, found {epoch}")
    stats = payload.get("train_only_normalization") or {}
    target_mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    target_std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    if target_std <= 0.0 or not math.isfinite(target_std):
        raise ValueError("invalid train-only target normalization")
    return {
        "epoch": epoch,
        "checkpoint_sha256": _sha256(path),
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "target_delta_mean": target_mean,
        "target_delta_std": target_std,
    }


def _predictions(path: Path, expected_ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_ids):
            raise ValueError(f"{path}: prediction keys are not exactly valid_iid")
        result = {
            sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            for sample_id in expected_ids
        }
    if any(values.size != 1024 or not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"{path}: malformed prediction field")
    return result


def _targets(data_root: Path, ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for sample_id in ids:
        sample_dir = data_root / sample_id
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        if meta.get("split") != "valid_iid":
            raise ValueError(f"{sample_id}: data metadata is not valid_iid")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        target_temperature = np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1)
        coords = np.load(sample_dir / "coords.npy").astype(np.float64).reshape(-1, 3)
        result[sample_id] = {
            "target_delta": target_temperature - bottom,
            "bottom": np.asarray(bottom),
            "control_volumes": control_volume_weights(coords),
        }
    return result


def _metrics(
    predictions: dict[str, np.ndarray],
    targets: dict[str, dict[str, np.ndarray]],
    ids: list[str],
    target_std: float,
) -> dict[str, float]:
    point_error, point_truth = 0.0, 0.0
    cv_error, cv_volume = 0.0, 0.0
    sample_relative = []
    normalized_squared_error = 0.0
    point_count = 0
    for sample_id in ids:
        target = targets[sample_id]["target_delta"]
        prediction_delta = predictions[sample_id] - float(targets[sample_id]["bottom"])
        weights = targets[sample_id]["control_volumes"]
        error = prediction_delta - target
        error_squared = np.square(error)
        point_error += float(np.sum(error_squared))
        point_truth += float(np.sum(np.square(target)))
        cv_error_i = float(np.sum(error_squared * weights))
        cv_truth_i = float(np.sum(np.square(target) * weights))
        cv_volume_i = float(np.sum(weights))
        cv_error += cv_error_i
        cv_volume += cv_volume_i
        sample_relative.append(math.sqrt(cv_error_i / cv_truth_i))
        normalized_squared_error += float(np.sum(error_squared / (target_std * target_std)))
        point_count += int(target.size)
    return {
        "valid_base_mse": normalized_squared_error / point_count,
        "point_global_relative_rmse_pct": 100.0 * math.sqrt(point_error / point_truth),
        "sample_first_cv_relative_rmse_pct": 100.0 * float(np.mean(sample_relative)),
        "raw_cv_weighted_rmse_K": math.sqrt(cv_error / cv_volume),
    }


def main() -> int:
    args = _args()
    for path in (args.output_json, args.output_md):
        if path.exists() and not args.overwrite:
            raise ValueError(f"output exists; pass --overwrite: {path}")
    split = json.loads(args.split_map.read_text(encoding="utf-8"))
    valid_ids = sorted(
        sample_id for sample_id, role in split["sample_splits"].items() if role == "valid_iid"
    )
    if len(valid_ids) != 128:
        raise ValueError("ensemble contract requires valid_iid=128")
    checkpoints = {
        "n3_best_e402": _checkpoint_meta(args.n3_run / "params_best.pkl", 402),
        "l2_best_e353": _checkpoint_meta(args.l2_run / "params_best.pkl", 353),
        "l2_final_e600": _checkpoint_meta(args.l2_run / "params_final.pkl", 600),
    }
    stats = {
        (row["target_delta_mean"], row["target_delta_std"], row["train_stats_hash"])
        for row in checkpoints.values()
    }
    if len(stats) != 1:
        raise ValueError("ensemble sources do not share train-only normalization")
    _, target_std, train_stats_hash = next(iter(stats))
    prediction_paths = {
        "n3_best_e402": args.n3_run / "best_predictions.npz",
        "l2_best_e353": args.l2_run / "best_predictions.npz",
        "l2_final_e600": args.l2_run / "predictions.npz",
    }
    prediction_fields = {
        label: _predictions(path, valid_ids) for label, path in prediction_paths.items()
    }
    targets = _targets(args.data_root, valid_ids)
    source_metrics = {
        label: _metrics(fields, targets, valid_ids, target_std)
        for label, fields in prediction_fields.items()
    }
    ensembles = []
    for left, right in PAIRS:
        for alpha in ALPHAS:
            fields = {
                sample_id: alpha * prediction_fields[left][sample_id]
                + (1.0 - alpha) * prediction_fields[right][sample_id]
                for sample_id in valid_ids
            }
            ensembles.append({
                "left": left,
                "right": right,
                "alpha": alpha,
                "formula": "prediction = alpha * left + (1 - alpha) * right",
                **_metrics(fields, targets, valid_ids, target_std),
            })
    payload = {
        "schema_version": "heat3d_v5_gate6e_valid_only_ensemble_v1",
        "status": "completed_no_training",
        "roles_accessed": ["valid_iid"],
        "forbidden_roles_accessed": [],
        "sample_count": len(valid_ids),
        "nodes_per_sample": 1024,
        "alphas": list(ALPHAS),
        "formulas": {
            "ensemble": "prediction = alpha * left + (1 - alpha) * right",
            "valid_base_mse": "mean(((prediction_deltaT-target_deltaT)/train_target_delta_std)^2)",
            "point_global_relative_rmse_pct": "100*sqrt(sum(error_deltaT^2)/sum(target_deltaT^2))",
            "sample_first_cv_relative_rmse_pct": "100*mean_samples(CV_RMS(error)/CV_RMS(target))",
            "raw_cv_weighted_rmse_K": "sqrt(sum(error_deltaT^2*CV)/sum(CV))",
        },
        "gate6e_config_frozen_before_audit": True,
        "gate6e_config_sha256": _sha256(args.gate6e_config),
        "gate6e_config_not_modified_from_ensemble": True,
        "train_stats_hash": train_stats_hash,
        "checkpoint_metadata": checkpoints,
        "prediction_artifacts": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in prediction_paths.items()
        },
        "source_metrics": source_metrics,
        "ensembles": ensembles,
        "training_started": False,
        "model_inference_run": False,
        "selection_or_tuning_use": "valid-only diagnostic; Gate 6E configuration frozen independently",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gate 6E valid-only field ensemble audit", "",
        "只访问 `valid_iid=128`；没有读取 test/hard，没有训练或模型推理。",
        "Gate 6E missing-cell 配置在本审计前已冻结，ensemble 结果未用于修改其权重。", "",
        "`prediction = alpha * left + (1-alpha) * right`。", "",
        "| left | right | alpha | valid base MSE | point-global | sample-first | raw RMSE K |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ensembles:
        lines.append(
            f"| {row['left']} | {row['right']} | {row['alpha']:.2f} | "
            f"{row['valid_base_mse']:.8f} | {row['point_global_relative_rmse_pct']:.6f}% | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f}% | {row['raw_cv_weighted_rmse_K']:.8f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "roles_accessed": ["valid_iid"], "ensembles": ensembles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
