#!/usr/bin/env python3
"""Collect valid-only V6 P1i metrics from completed prediction archives.

This recovery collector is intentionally checkpoint-independent.  It is used
when training reached its final epoch and prediction export succeeded, but a
later checkpoint metadata/export step failed.  It never materializes test or
sealed samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v5_metrics import (  # noqa: E402
    compute_sample_metrics,
    summarize_metric_rows,
    validate_metric_suite,
)


class CollectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, dataset_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != dataset_id:
        raise CollectionError("dataset ID does not match the frozen manifest")
    if int(payload.get("sample_count", -1)) != 1024:
        raise CollectionError("expected the frozen 1024-sample P1i manifest")
    counts = payload.get("split_role_counts") or {}
    if counts != {"test_iid": 128, "train": 768, "valid_iid": 128}:
        raise CollectionError(f"split counts drifted: {counts}")
    return payload


def _sample_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    relative = row.get("relative_path")
    if not relative:
        raise CollectionError(f"{row.get('sample_id')}: relative_path is missing")
    path = dataset_root / str(relative)
    if not path.is_dir():
        raise CollectionError(f"sample directory is missing: {path}")
    return path


def _train_delta_stats(dataset_root: Path, rows: list[dict[str, Any]]) -> tuple[float, float]:
    count = 0
    total = 0.0
    total_squared = 0.0
    for row in rows:
        values = np.asarray(
            np.load(_sample_path(dataset_root, row) / "deltaT.npy"),
            dtype=np.float64,
        ).reshape(-1)
        if values.size != 1024 or not np.all(np.isfinite(values)):
            raise CollectionError(f"{row['sample_id']}: invalid train deltaT field")
        count += int(values.size)
        total += float(np.sum(values, dtype=np.float64))
        total_squared += float(np.sum(np.square(values), dtype=np.float64))
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if count != 768 * 1024 or not math.isfinite(std) or std <= 1.0e-8:
        raise CollectionError("invalid train-only target normalization statistics")
    return float(mean), float(std)


def _evaluate_archive(
    *,
    label: str,
    epoch: int,
    archive_path: Path,
    dataset_root: Path,
    valid_rows: list[dict[str, Any]],
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(archive_path, allow_pickle=False) as archive:
        expected_ids = [str(row["sample_id"]) for row in valid_rows]
        if set(archive.files) != set(expected_ids) or len(archive.files) != 128:
            raise CollectionError(
                f"{label}: prediction IDs are not exactly frozen valid_iid"
            )
        rows = []
        point_sse = 0.0
        point_energy = 0.0
        point_count = 0
        for manifest_row in valid_rows:
            sample_id = str(manifest_row["sample_id"])
            sample_path = _sample_path(dataset_root, manifest_row)
            target_delta = np.asarray(
                np.load(sample_path / "deltaT.npy"), dtype=np.float64
            ).reshape(-1)
            target_temperature = np.asarray(
                np.load(sample_path / "temperature.npy"), dtype=np.float64
            ).reshape(-1)
            prediction_temperature = np.asarray(
                archive[sample_id], dtype=np.float64
            ).reshape(-1)
            control_volume = np.asarray(
                np.load(sample_path / "control_volume.npy"), dtype=np.float64
            ).reshape(-1)
            q_field = np.asarray(
                np.load(sample_path / "q_field.npy"), dtype=np.float64
            ).reshape(-1)
            arrays = (
                target_delta,
                target_temperature,
                prediction_temperature,
                control_volume,
                q_field,
            )
            if any(value.size != 1024 for value in arrays):
                raise CollectionError(f"{sample_id}: field size drifted from 1024")
            if any(not np.all(np.isfinite(value)) for value in arrays):
                raise CollectionError(f"{sample_id}: non-finite evaluation field")
            reference = target_temperature - target_delta
            if float(np.ptp(reference)) > 1.0e-8:
                raise CollectionError(f"{sample_id}: reference temperature is not scalar")
            prediction_delta = prediction_temperature - float(np.mean(reference))
            error = prediction_delta - target_delta
            point_sse += float(np.sum(np.square(error), dtype=np.float64))
            point_energy += float(np.sum(np.square(target_delta), dtype=np.float64))
            point_count += int(error.size)
            metric_row = compute_sample_metrics(
                {
                    "sample_id": sample_id,
                    "split": "valid_iid",
                    "prediction_deltaT_K": prediction_delta,
                    "target_deltaT_K": target_delta,
                    "control_volumes_m3": control_volume,
                    "q_W_m3": q_field,
                    "prediction_normalized": (prediction_delta - target_mean) / target_std,
                    "target_normalized": (target_delta - target_mean) / target_std,
                }
            )
            metric_row["checkpoint_label"] = label
            metric_row["checkpoint_epoch"] = int(epoch)
            rows.append(metric_row)

    summary = summarize_metric_rows(rows)
    validate_metric_suite(summary, require_legacy=True)
    summary.update(
        {
            "checkpoint_label": label,
            "checkpoint_epoch": int(epoch),
            "prediction_archive": str(archive_path),
            "prediction_sha256": _sha256(archive_path),
            "point_global_raw_rmse_K_unweighted": math.sqrt(point_sse / point_count),
            "point_global_true_rms_K_unweighted": math.sqrt(point_energy / point_count),
            "checkpoint_available": False,
            "checkpoint_unavailable_reason": (
                "post_training_checkpoint_metadata_export_failed_before_checkpoint_write"
            ),
        }
    )
    if abs(
        summary["point_global_relative_rmse_pct"]
        - 100.0 * math.sqrt(point_sse / point_energy)
    ) > 1.0e-12:
        raise CollectionError("point-global true-RMS formula mismatch")
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "checkpoint_label",
        "checkpoint_epoch",
        "point_global_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "point_global_raw_rmse_K_unweighted",
        "raw_cv_weighted_rmse_K",
        "amplitude_ratio",
        "spatial_correlation",
        "hotspot_cv_weighted_rmse_K",
        "top5_cv_weighted_rmse_K",
        "strong_q_cv_weighted_rmse_K",
        "low_deltaT_background_bias_K",
        "low_deltaT_background_rmse_K",
        "low_deltaT_background_over_ratio",
        "shape_cv_rmse",
        "scale_log_rmse",
        "legacy_normalized_valid_base_mse",
        "prediction_sha256",
        "checkpoint_available",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V6 P1i seed0 B24 valid-only recovery closeout",
        "",
        "训练已完成 600 epochs，但 checkpoint metadata 构造阶段因未定义 `builder` 崩溃。",
        "两个 prediction archives 已完整写出，因此以下为冻结 `valid_iid` 的只读复算；",
        "test 与 sealed IID 均未访问。checkpoint 参数未落盘，不能由预测反推恢复。",
        "",
        "| archive | epoch | point-global true-RMS | sample-first CV | raw CV RMSE K | amp | corr | shape CV-RMSE | scale log-RMSE | legacy base MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["metrics"]:
        lines.append(
            "| {checkpoint_label} | {checkpoint_epoch} | "
            "{point_global_relative_rmse_pct:.6f}% | "
            "{sample_first_cv_relative_rmse_pct:.6f}% | "
            "{raw_cv_weighted_rmse_K:.6f} | {amplitude_ratio:.6f} | "
            "{spatial_correlation:.6f} | {shape_cv_rmse:.6f} | "
            "{scale_log_rmse:.6f} | {legacy_normalized_valid_base_mse:.8f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- `<20%` valid point-global gate: **{'PASS' if payload['gate']['valid_point_global_lt_20_pct'] else 'FAIL'}**。",
            "- 该结论是 valid-only 模型质量诊断，不是 checkpoint 可复现性通过。",
            "- e542 为预注册 point-global selection 对应的 `best_predictions.npz`；e600 为 final。",
            "- 训练日志中的 `raw_rmse_K` 是未加 CV 的逐点 RMSE；表中 `raw CV RMSE K` 为冻结 V5 CV 口径。",
            "- 训练日志 `best=e542/0.0031` 中 epoch 由 point-global 选择，斜杠后数值按既有日志合同显示该 epoch 的 valid base MSE。",
            "",
            "## 工件状态",
            "",
            "- predictions: saved and SHA256-bound",
            "- params checkpoints: missing because crash preceded checkpoint writes",
            "- run_config/loss_summary: missing for the same reason",
            "- retraining: not performed",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-id", default="V6_05_V5best_P1i_seed0_B24")
    parser.add_argument("--dataset-id", default="heat3d_v6_p1i_continuous_physics1024_v1")
    parser.add_argument("--training-commit", default="dfe3cf6")
    parser.add_argument("--best-epoch", type=int, required=True)
    parser.add_argument("--final-epoch", type=int, default=600)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-sample-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest, args.dataset_id)
    train_rows = [row for row in manifest["samples"] if row["split_role"] == "train"]
    valid_rows = [row for row in manifest["samples"] if row["split_role"] == "valid_iid"]
    if len(train_rows) != 768 or len(valid_rows) != 128:
        raise CollectionError("train/valid population drifted")
    target_mean, target_std = _train_delta_stats(args.dataset_root, train_rows)

    specs = [
        ("point_global_best", args.best_epoch, args.run_dir / "best_predictions.npz"),
        ("final", args.final_epoch, args.run_dir / "predictions.npz"),
    ]
    metrics = []
    per_sample = []
    for label, epoch, path in specs:
        if not path.is_file():
            raise CollectionError(f"missing prediction archive: {path}")
        summary, rows = _evaluate_archive(
            label=label,
            epoch=epoch,
            archive_path=path,
            dataset_root=args.dataset_root,
            valid_rows=valid_rows,
            target_mean=target_mean,
            target_std=target_std,
        )
        metrics.append(summary)
        per_sample.extend(rows)

    payload = {
        "schema_version": "heat3d_v6_p1i_valid_recovery_closeout_v1",
        "status": "completed_e600_export_failed_valid_predictions_evaluated",
        "config_id": args.config_id,
        "dataset_id": args.dataset_id,
        "dataset_manifest": str(args.manifest),
        "dataset_manifest_sha256": _sha256(args.manifest),
        "training_commit": args.training_commit,
        "evaluation_scope": "valid_iid_only",
        "test_access": False,
        "sealed_access": False,
        "target_normalization": {
            "fit_population": "train_only",
            "sample_count": 768,
            "target_delta_mean_K": target_mean,
            "target_delta_std_K": target_std,
        },
        "failure": {
            "stage": "checkpoint_run_metadata_before_checkpoint_write",
            "exception": "NameError: name 'builder' is not defined",
            "training_epochs_completed": 600,
            "retraining_performed": False,
        },
        "metrics": metrics,
        "gate": {
            "metric": "valid_iid point-global true-RMS relative RMSE",
            "threshold_pct": 20.0,
            "candidate_checkpoint": "point_global_best",
            "valid_point_global_lt_20_pct": bool(
                metrics[0]["point_global_relative_rmse_pct"] < 20.0
            ),
            "checkpoint_reproducibility_passed": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, metrics)
    _write_per_sample_csv(args.per_sample_csv, per_sample)
    _write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
