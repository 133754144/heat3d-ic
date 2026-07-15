#!/usr/bin/env python3
"""Read-only V13 closeout from existing valid-only prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np


ALPHAS = (0.25, 0.5, 0.75)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    stats = payload["train_only_normalization"]
    record = {
        "path": str(path),
        "sha256": _sha256(path),
        "epoch": int(payload["epoch"]),
        "checkpoint_kind": str(payload["checkpoint_kind"]),
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "target_delta_mean": float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0]),
        "target_delta_std": float(np.asarray(stats["target_delta_std"]).reshape(-1)[0]),
    }
    return payload, record


def _predictions(path: Path, ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ids):
            raise ValueError(f"{path}: prediction keys differ from valid_iid")
        values = {sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1) for sample_id in ids}
    if any(array.size != 1024 or not np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError(f"{path}: invalid prediction arrays")
    return values


def _cv_weights(coords: np.ndarray) -> np.ndarray:
    axes = [np.unique(coords[:, index]) for index in range(3)]
    one_d = []
    for axis in axes:
        edges = np.empty(axis.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
        edges[0], edges[-1] = axis[0], axis[-1]
        one_d.append(np.maximum(np.diff(edges), 0.0))
    lookup = [{float(value): float(weight) for value, weight in zip(axis, weights)} for axis, weights in zip(axes, one_d)]
    return np.asarray([
        lookup[0][float(point[0])] * lookup[1][float(point[1])] * lookup[2][float(point[2])]
        for point in coords
    ])


def _targets(data_root: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for sample_id in ids:
        sample = data_root / sample_id
        meta = json.loads((sample / "sample_meta.json").read_text(encoding="utf-8"))
        if meta.get("split") != "valid_iid":
            raise ValueError(f"{sample_id}: forbidden non-valid role")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        target = np.load(sample / "temperature.npy").astype(np.float64).reshape(-1) - bottom
        coords = np.load(sample / "coords.npy").astype(np.float64).reshape(-1, 3)
        result[sample_id] = {
            "target": target,
            "bottom": bottom,
            "weights": _cv_weights(coords),
        }
    return result


def _per_sample(
    fields: dict[str, np.ndarray], targets: dict[str, dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in ids:
        item = targets[sample_id]
        prediction = fields[sample_id] - item["bottom"]
        target, weights = item["target"], item["weights"]
        error = prediction - target
        error_sse, true_sse = float(np.sum(np.square(error))), float(np.sum(np.square(target)))
        cv_error = float(np.sum(np.square(error) * weights))
        cv_true = float(np.sum(np.square(target) * weights))
        rows.append({
            "sample_id": sample_id,
            "point_sse_K2": error_sse,
            "true_point_sse_K2": true_sse,
            "sample_cv_relative_rmse_pct": 100.0 * math.sqrt(cv_error / cv_true),
            "true_cv_rms_deltaT_K": math.sqrt(cv_true / float(np.sum(weights))),
        })
    return rows


def _metrics(
    fields: dict[str, np.ndarray],
    targets: dict[str, dict[str, Any]],
    ids: list[str],
    target_std: float,
) -> dict[str, float]:
    rows = _per_sample(fields, targets, ids)
    cv_sse, cv_volume, normalized_sse = 0.0, 0.0, 0.0
    for sample_id in ids:
        item = targets[sample_id]
        error = fields[sample_id] - item["bottom"] - item["target"]
        cv_sse += float(np.sum(np.square(error) * item["weights"]))
        cv_volume += float(np.sum(item["weights"]))
        normalized_sse += float(np.sum(np.square(error) / (target_std * target_std)))
    return {
        "legacy_normalized_valid_base_mse": normalized_sse / (len(ids) * 1024),
        "point_global_relative_rmse_pct": 100.0 * math.sqrt(
            sum(row["point_sse_K2"] for row in rows) / sum(row["true_point_sse_K2"] for row in rows)
        ),
        "sample_first_cv_relative_rmse_pct": float(np.mean([row["sample_cv_relative_rmse_pct"] for row in rows])),
        "raw_cv_weighted_rmse_K": math.sqrt(cv_sse / cv_volume),
    }


def _paired(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    targets: dict[str, dict[str, Any]],
    ids: list[str],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_rows = {row["sample_id"]: row for row in _per_sample(left, targets, ids)}
    right_rows = {row["sample_id"]: row for row in _per_sample(right, targets, ids)}
    values = np.asarray([left_rows[sample_id]["true_cv_rms_deltaT_K"] for sample_id in ids])
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.searchsorted(edges[1:-1], values, side="right")
    quartiles = []
    for index in range(4):
        selected = [sample_id for sample_id, bin_index in zip(ids, bins, strict=True) if bin_index == index]
        left_sse = float(sum(left_rows[sample_id]["point_sse_K2"] for sample_id in selected))
        right_sse = float(sum(right_rows[sample_id]["point_sse_K2"] for sample_id in selected))
        quartiles.append({
            "quartile": f"Q{index + 1}",
            "sample_count": len(selected),
            f"{left_name}_point_sse_K2": left_sse,
            f"{right_name}_point_sse_K2": right_sse,
            f"{right_name}_minus_{left_name}_point_sse_K2": right_sse - left_sse,
            f"{left_name}_sample_first_rmse_pct": float(np.mean([left_rows[s]["sample_cv_relative_rmse_pct"] for s in selected])),
            f"{right_name}_sample_first_rmse_pct": float(np.mean([right_rows[s]["sample_cv_relative_rmse_pct"] for s in selected])),
        })
    return {
        "left": left_name,
        "right": right_name,
        "true_cv_rms_quartile_edges_K": edges.tolist(),
        "quartiles": quartiles,
        f"{right_name}_minus_{left_name}_total_point_sse_K2": float(sum(
            right_rows[s]["point_sse_K2"] - left_rows[s]["point_sse_K2"] for s in ids
        )),
        f"{right_name}_minus_{left_name}_mean_sample_rmse_pct_points": float(np.mean([
            right_rows[s]["sample_cv_relative_rmse_pct"] - left_rows[s]["sample_cv_relative_rmse_pct"] for s in ids
        ])),
    }


def _trajectory(summary: dict[str, Any], field: str, factor: float = 1.0) -> dict[str, Any]:
    rows = [row for row in summary["epoch_history"] if row.get(field) is not None]
    row = min(rows, key=lambda item: float(item[field]))
    return {
        "epoch": int(row["epoch"]),
        "metric": field,
        "value": factor * float(row[field]),
        "trajectory_only": True,
        "checkpoint_saved": False,
    }


def main() -> int:
    root = _args().root.resolve()
    v13_run = root / "output/heat3d_v5_gate6e_runs/V4P5_13_gate6e_scratch_branch_rebalance"
    n3_run = root / "output/heat3d_v5_runs/V4P5_07_native_pooled_latent_global_film"
    l2_run = root / "output/heat3d_v5_gate6c_runs/V4P5_12_gate6c_scratch_l2_shape_balanced"
    run_config = json.loads((v13_run / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((v13_run / "loss_summary.json").read_text(encoding="utf-8"))
    split = json.loads((root / run_config["split_map_path"]).read_text(encoding="utf-8"))
    split_roles: dict[str, list[str]] = {}
    for sample_id, role in split["sample_splits"].items():
        split_roles.setdefault(role, []).append(sample_id)
    for ids in split_roles.values():
        ids.sort()
    ids = split_roles["valid_iid"]
    if len(ids) != 128:
        raise ValueError("valid_iid must have 128 samples")

    best_payload, best_checkpoint = _checkpoint(v13_run / "params_best.pkl")
    _, final_checkpoint = _checkpoint(v13_run / "params_final.pkl")
    target_std = best_checkpoint["target_delta_std"]
    targets = _targets(root / run_config["subset"], ids)
    paths = {
        "n3_best_e402": n3_run / "best_predictions.npz",
        "l2_best_e353": l2_run / "best_predictions.npz",
        "v13_best_e318": v13_run / "best_predictions.npz",
        "v13_final_e600": v13_run / "predictions.npz",
    }
    predictions = {name: _predictions(path, ids) for name, path in paths.items()}
    source_metrics = {name: _metrics(fields, targets, ids, target_std) for name, fields in predictions.items()}

    ensemble_rows = []
    for left_name, right_name in (
        ("n3_best_e402", "v13_best_e318"),
        ("l2_best_e353", "v13_best_e318"),
    ):
        for alpha in ALPHAS:
            fields = {sample_id: alpha * predictions[left_name][sample_id] + (1.0 - alpha) * predictions[right_name][sample_id] for sample_id in ids}
            ensemble_rows.append({"left": left_name, "right": right_name, "alpha": alpha, **_metrics(fields, targets, ids, target_std)})
    equal_three = {
        sample_id: (predictions["n3_best_e402"][sample_id] + predictions["l2_best_e353"][sample_id] + predictions["v13_best_e318"][sample_id]) / 3.0
        for sample_id in ids
    }
    ensemble_rows.append({"left": "n3_best_e402+l2_best_e353+v13_best_e318", "right": "equal_weight", "alpha": 1.0 / 3.0, **_metrics(equal_three, targets, ids, target_std)})

    standardizer = run_config["global_context"]["standardizer"]
    artifacts = {}
    for name in ("run_config.json", "loss_summary.json", "params_best.pkl", "params_final.pkl", "best_predictions.npz", "predictions.npz"):
        path = v13_run / name
        artifacts[name] = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    config_path = root / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml"
    payload = {
        "schema_version": "heat3d_v5_gate6e_v13_closeout_v1",
        "status": "completed",
        "config_id": "V4P5_13_gate6e_scratch_branch_rebalance",
        "run_commit": str(summary["code_version_or_git_commit"]),
        "config_sha256": _sha256(config_path),
        "run_directory": str(v13_run),
        "split": {
            "source": run_config["split_source"],
            "counts": {role: len(values) for role, values in sorted(split_roles.items())},
            "sample_ids_sha256": {role: _ids_hash(values) for role, values in sorted(split_roles.items())},
            "roles_read_for_closeout": ["valid_iid"],
            "forbidden_roles_read": [],
        },
        "train_only_standardizer": {
            "fit_population": standardizer["fit_population"],
            "fit_sample_count": int(standardizer["fit_sample_count"]),
            "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
            "target_or_label_derived_inputs": bool(standardizer["target_or_label_derived_inputs"]),
        },
        "checkpoints": {"base_mse_best": best_checkpoint, "final": final_checkpoint},
        "selection_records": {
            "base_mse_best": {
                "epoch": int(summary["best_epoch"]),
                "metric": "legacy normalized valid_base_mse",
                "value": float(summary["best_valid_base_mse"]),
                "trajectory_only": False,
                "checkpoint_saved": True,
                "checkpoint_path": best_checkpoint["path"],
            },
            "point_global_best": _trajectory(summary, "valid_rel_rmse_v4_pct"),
            "sample_first_best": _trajectory(summary, "valid_native_joint_relative_rmse", 100.0),
            "final": {"epoch": 600, "trajectory_only": False, "checkpoint_saved": True, "checkpoint_path": final_checkpoint["path"]},
        },
        "metrics": {
            "formulas": {
                "point_global_relative_rmse_pct": "100*sqrt(sum_nodes_samples(error_deltaT^2)/sum_nodes_samples(true_deltaT^2))",
                "sample_first_cv_relative_rmse_pct": "100*mean_samples(CV_RMS(error)/CV_RMS(true_deltaT))",
                "legacy_normalized_valid_base_mse": "mean(((prediction_deltaT-true_deltaT)/train_target_delta_std)^2)",
                "raw_cv_weighted_rmse_K": "sqrt(sum(error_deltaT^2*CV)/sum(CV))",
            },
            "source_metrics": source_metrics,
        },
        "paired_valid_only": {
            "n3_vs_l2": _paired(predictions["n3_best_e402"], predictions["l2_best_e353"], targets, ids, "n3", "l2"),
            "n3_vs_v13": _paired(predictions["n3_best_e402"], predictions["v13_best_e318"], targets, ids, "n3", "v13"),
            "l2_vs_v13": _paired(predictions["l2_best_e353"], predictions["v13_best_e318"], targets, ids, "l2", "v13"),
        },
        "valid_only_ensemble": {"alphas": list(ALPHAS), "rows": ensemble_rows, "model_inference_run": False},
        "artifacts": artifacts,
        "large_artifacts_tracked": False,
        "model_inference_run": False,
        "training_started": False,
    }
    best = source_metrics["v13_best_e318"]
    final = source_metrics["v13_final_e600"]
    point = payload["selection_records"]["point_global_best"]
    sample = payload["selection_records"]["sample_first_best"]
    md = "\n".join([
        "# Gate 6E V13 closeout", "",
        "状态：`completed`。本 closeout 只读取 WSL2 已有 V13 工件和 `valid_iid=128`；未读取 test/hard/sealed，未训练或运行模型推理。", "",
        "## 运行合同", "",
        f"- run commit: `{payload['run_commit']}`",
        f"- config SHA256: `{payload['config_sha256']}`",
        f"- split: train={len(split_roles['train'])}, valid_iid={len(ids)}, test_iid={len(split_roles['test_iid'])}; nodes/sample=1024",
        f"- Global Context standardizer: `{standardizer['fit_population']}`, samples={standardizer['fit_sample_count']}", "",
        "## V13 指标", "",
        "| artifact | epoch | legacy base MSE | point-global relative RMSE | sample-first CV-relative RMSE | raw CV RMSE K |",
        "|---|---:|---:|---:|---:|---:|",
        f"| base-MSE best | 318 | {best['legacy_normalized_valid_base_mse']:.8f} | {best['point_global_relative_rmse_pct']:.6f}% | {best['sample_first_cv_relative_rmse_pct']:.6f}% | {best['raw_cv_weighted_rmse_K']:.8f} |",
        f"| final | 600 | {final['legacy_normalized_valid_base_mse']:.8f} | {final['point_global_relative_rmse_pct']:.6f}% | {final['sample_first_cv_relative_rmse_pct']:.6f}% | {final['raw_cv_weighted_rmse_K']:.8f} |", "",
        f"Point-global trajectory best 为 epoch {point['epoch']} / {point['value']:.6f}%；sample-first trajectory best 为 epoch {sample['epoch']} / {sample['value']:.6f}%。两者均未保存对应参数，因此严格标记 `trajectory_only=true`，不称为 checkpoint。", "",
        "## Paired 与 ensemble", "",
        "JSON 中记录 N3/L2/V13 的逐模型 valid-only 聚合、true-CV-RMS Q1-Q4 point SSE，以及 N3/V13、L2/V13 固定 alpha ensemble；所有比较严格区分 point-global、sample-first 和 legacy base MSE。", "",
        "大型 checkpoint、prediction 与 output 文件均未纳入 Git。", "",
    ])
    print(json.dumps({"payload": payload, "markdown": md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
