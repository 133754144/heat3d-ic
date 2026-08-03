#!/usr/bin/env python3
"""Evaluate all frozen P1i formal checkpoints on valid_iid support only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import collect_heat3d_v6_p1i_completed_predictions as recovery  # noqa: E402


def _tail(rows: list[dict]) -> dict:
    sse = np.asarray([row["point_error_squared_sum"] for row in rows], dtype=np.float64)
    rel = np.asarray([100.0 * row["sample_cv_relative_rmse"] for row in rows])
    order = np.argsort(sse)[::-1]
    return {
        "sample_relative_rmse_pct": {
            "p50": float(np.quantile(rel, 0.50)),
            "p90": float(np.quantile(rel, 0.90)),
            "p95": float(np.quantile(rel, 0.95)),
            "p99": float(np.quantile(rel, 0.99)),
            "max": float(np.max(rel)),
        },
        "top5_point_sse_fraction": float(np.sum(sse[order[:5]]) / np.sum(sse)),
        "top10_point_sse_fraction": float(np.sum(sse[order[:10]]) / np.sum(sse)),
        "worst_sample_ids": [rows[int(index)]["sample_id"] for index in order[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-sample-csv", type=Path, required=True)
    args = parser.parse_args()

    summary_payload = json.loads(
        (args.run_dir / "loss_summary.json").read_text(encoding="utf-8")
    )
    manifest = recovery._load_manifest(
        args.manifest, "heat3d_v6_p1i_continuous_physics1024_v1"
    )
    train_rows = [row for row in manifest["samples"] if row["split_role"] == "train"]
    valid_rows = [
        row for row in manifest["samples"] if row["split_role"] == "valid_iid"
    ]
    target_mean, target_std = recovery._train_delta_stats(args.dataset_root, train_rows)
    specs = [
        (
            "point_global_best",
            int(summary_payload["point_global_best_epoch"]),
            args.run_dir / "point_global_best_predictions.npz",
            args.run_dir / "params_best_valid_point_global.pkl",
        ),
        (
            "sample_first_best",
            int(summary_payload["sample_first_best_epoch"]),
            args.run_dir / "sample_first_best_predictions.npz",
            args.run_dir / "params_best_valid_sample_first.pkl",
        ),
        (
            "base_mse_best",
            int(summary_payload["base_mse_best_epoch"]),
            args.run_dir / "base_mse_best_predictions.npz",
            args.run_dir / "params_best_valid_base_mse.pkl",
        ),
        (
            "final",
            int(summary_payload["final_epoch"]),
            args.run_dir / "predictions.npz",
            args.run_dir / "params_final.pkl",
        ),
    ]
    metrics, per_sample = [], []
    for label, epoch, archive_path, checkpoint_path in specs:
        if not archive_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"{label}: frozen artifact is missing")
        summary, rows = recovery._evaluate_archive(
            label=label,
            epoch=epoch,
            archive_path=archive_path,
            dataset_root=args.dataset_root,
            valid_rows=valid_rows,
            target_mean=target_mean,
            target_std=target_std,
        )
        summary["checkpoint_available"] = True
        summary.pop("checkpoint_unavailable_reason", None)
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["checkpoint_sha256"] = recovery._sha256(checkpoint_path)
        summary["tail"] = _tail(rows)
        metrics.append(summary)
        per_sample.extend(rows)

    payload = {
        "schema_version": "heat3d_v6_p1i_formal_valid_support_v1",
        "status": "passed",
        "config_id": summary_payload.get("config_id")
        or args.run_dir.name,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": recovery._sha256(args.manifest),
        "accessed_roles": ["train_targets_for_frozen_normalization", "valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "primary_checkpoint": "point_global_best",
        "diagnostic_checkpoints": ["sample_first_best", "base_mse_best", "final"],
        "metrics": metrics,
        "best_to_final": recovery._paired_best_to_final(per_sample),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recovery._write_csv(args.output_csv, metrics)
    recovery._write_per_sample_csv(args.per_sample_csv, per_sample)
    print(json.dumps({"status": "passed", "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
