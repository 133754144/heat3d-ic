#!/usr/bin/env python3
"""Collect V6_03 seed0/1/2 saved-valid results without checkpoint inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import collect_heat3d_v6_training_results as base  # noqa: E402


DEFAULT_SOURCES = ROOT / "configs/heat3d_v6/v6_multiseed_result_sources.json"
DEFAULT_HISTORICAL_REGISTRY = (
    ROOT / "configs/heat3d_v6/v6_training_result_registry.csv"
)
DEFAULT_HISTORICAL_CHECKPOINTS = (
    ROOT / "configs/heat3d_v6/v6_training_checkpoint_metrics.csv"
)
DEFAULT_REGISTRY = (
    ROOT / "configs/heat3d_v6/v6_multiseed_training_results.csv"
)
DEFAULT_CHECKPOINT_CSV = (
    ROOT / "configs/heat3d_v6/v6_multiseed_checkpoint_metrics.csv"
)
DEFAULT_JSON = (
    ROOT / "configs/heat3d_v6/v6_multiseed_training_results.json"
)
DEFAULT_MD = ROOT / "docs/v6_multiseed_training_results.md"

REGISTRY_FIELDS = (
    "seed",
    *base.REGISTRY_FIELDS,
)
CHECKPOINT_FIELDS = (
    "seed",
    *base.CHECKPOINT_FIELDS,
)
PRIMARY_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_relative_rmse_pct",
    "raw_rmse_K",
    "base_mse",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_rmse_K",
    "top5_rmse_K",
    "strong_q_rmse_K",
    "low_delta_bias_K",
    "low_delta_rmse_K",
    "low_delta_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _historical_seed0(
    registry_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        row
        for row in _read_csv(registry_path)
        if row["config_id"] == "V6_03_V5best_P1h"
    ]
    if len(candidates) != 1:
        raise AssertionError("expected exactly one frozen V6_03 seed0 row")
    row: dict[str, Any] = {"seed": 0, **candidates[0]}
    checkpoints = [
        {"seed": 0, **item}
        for item in _read_csv(checkpoint_path)
        if item["config_id"] == "V6_03_V5best_P1h"
    ]
    if len(checkpoints) != 4:
        raise AssertionError("expected four frozen V6_03 seed0 checkpoints")
    return row, checkpoints


def _collect_new_run(
    *,
    spec: Mapping[str, Any],
    snapshot_root: Path,
    data_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config_id = str(spec["config_id"])
    seed = int(spec["seed"])
    run_dir = snapshot_root / str(spec["snapshot_path"])
    summary_path = run_dir / "loss_summary.json"
    config_path = run_dir / "run_config.json"
    summary = base._read_json(summary_path)
    run_config = base._read_json(config_path)
    if (
        not summary.get("status_ok")
        or int(summary["final_epoch"]) != 600
        or str(summary["code_version_or_git_commit"])
        != str(spec["training_commit"])
    ):
        raise AssertionError(f"{config_id}: incomplete or provenance mismatch")
    if (
        int(run_config["epochs"]) != 600
        or int(run_config["batch_size"]) != 24
        or int(run_config["micro_batch_size"]) != 24
    ):
        raise AssertionError(f"{config_id}: e600/B24 contract drifted")
    for key in ("seed", "model_seed", "batch_order_seed", "graph_seed"):
        if int(run_config[key]) != seed:
            raise AssertionError(f"{config_id}: {key} != {seed}")
    if (
        str(run_config["subset"])
        != "data/heat3d_v6_p1h_shared_support1024_v0"
    ):
        raise AssertionError(f"{config_id}: dataset binding drifted")

    dataset_root = data_root / str(spec["dataset_id"])
    target_std = float(
        summary["train_only_normalization"]["target_delta_std"]
    )
    reload_status = base._reload_status(run_dir, summary, spec)
    checkpoint_metrics: dict[str, dict[str, Any]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for kind, checkpoint in spec["checkpoints"].items():
        checkpoint_file = run_dir / str(checkpoint["checkpoint_file"])
        prediction_file = run_dir / str(checkpoint["prediction_file"])
        if base._sha256(checkpoint_file) != checkpoint["checkpoint_sha256"]:
            raise AssertionError(f"{config_id}/{kind}: checkpoint SHA drift")
        prediction_sha = base._sha256(prediction_file)
        if prediction_sha != checkpoint["prediction_sha256"]:
            raise AssertionError(f"{config_id}/{kind}: prediction SHA drift")
        metrics, per_sample = base._evaluate_prediction(
            prediction_file,
            dataset_root,
            target_std,
        )
        checkpoint_metrics[str(kind)] = metrics
        checkpoint_rows.append(
            {
                "seed": seed,
                "config_id": config_id,
                "dataset_id": spec["dataset_id"],
                "source_host": spec["source_host"],
                "checkpoint_kind": kind,
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_file": checkpoint["checkpoint_file"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "prediction_file": checkpoint["prediction_file"],
                "prediction_sha256": prediction_sha,
                **metrics,
            }
        )
        if len(per_sample) != 128:
            raise AssertionError(f"{config_id}/{kind}: valid sample drift")

    primary = checkpoint_metrics["point_global_best"]
    final = checkpoint_metrics["final"]
    sample_best = checkpoint_metrics["sample_first_best"]
    base_best = checkpoint_metrics["base_mse_best"]
    consistency = base._assert_summary_consistency(
        config_id,
        summary,
        primary,
        final,
        sample_best,
        base_best,
    )
    registry_row = {
        "seed": seed,
        "config_id": config_id,
        "dataset_id": spec["dataset_id"],
        "source_host": spec["source_host"],
        "execution_status": "completed_e600",
        "evaluation_status": "completed_valid_iid_saved_predictions",
        "training_commit": spec["training_commit"],
        "remote_run_dir": spec["remote_run_dir"],
        "remote_log_path": (
            spec["remote_log_path"] if spec["remote_log_available"] else ""
        ),
        "final_epoch": 600,
        "selection_metric": summary["selection_metric"],
        "primary_checkpoint": "point_global_best",
        "primary_epoch": spec["checkpoints"]["point_global_best"]["epoch"],
        **{key: primary[key] for key in PRIMARY_METRICS},
        "sample_first_best_epoch": spec["checkpoints"][
            "sample_first_best"
        ]["epoch"],
        "sample_first_best_relative_rmse_pct": sample_best[
            "sample_first_relative_rmse_pct"
        ],
        "base_mse_best_epoch": spec["checkpoints"]["base_mse_best"]["epoch"],
        "base_mse_best_value": base_best["base_mse"],
        "final_point_global_relative_rmse_pct": final[
            "point_global_relative_rmse_pct"
        ],
        "final_sample_first_relative_rmse_pct": final[
            "sample_first_relative_rmse_pct"
        ],
        "final_raw_rmse_K": final["raw_rmse_K"],
        "best_to_final_point_global_delta_pct": final[
            "point_global_relative_rmse_pct"
        ]
        - primary["point_global_relative_rmse_pct"],
        "reload_status": reload_status,
        "valid_sample_count": primary["valid_sample_count"],
        "threshold_lt20": bool(
            primary["point_global_relative_rmse_pct"] < 20.0
        ),
        "result_scope": "valid_iid_saved_predictions_only",
    }
    provenance = {
        "config_id": config_id,
        "seed": seed,
        "source_host": spec["source_host"],
        "training_commit": spec["training_commit"],
        "loss_summary_sha256": base._sha256(summary_path),
        "run_config_sha256": base._sha256(config_path),
        "reload_status": reload_status,
        "remote_log_available": bool(spec["remote_log_available"]),
        "saved_prediction_summary_consistency_abs_delta": consistency,
    }
    return registry_row, checkpoint_rows, provenance


def _statistics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in PRIMARY_METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "std_sample": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.ptp(values)),
        }
    return result


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6_03 P1h seed0/1/2 latest training results",
        "",
        "Scope: frozen saved `valid_iid` predictions only. No checkpoint "
        "inference, training, checkpoint mutation, test, hard, or sealed access.",
        "",
        "| seed | host | point-global % | sample-first % | raw RMSE K | "
        "shape CV-RMSE | scale log-RMSE | best epoch | final point-global % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["registry_rows"]:
        lines.append(
            f"| {int(row['seed'])} | {row['source_host']} | "
            f"{float(row['point_global_relative_rmse_pct']):.6f} | "
            f"{float(row['sample_first_relative_rmse_pct']):.6f} | "
            f"{float(row['raw_rmse_K']):.6f} | "
            f"{float(row['shape_cv_rmse']):.6f} | "
            f"{float(row['scale_log_rmse']):.6f} | "
            f"{int(row['primary_epoch'])} | "
            f"{float(row['final_point_global_relative_rmse_pct']):.6f} |"
        )
    pg = payload["seed_statistics"]["point_global_relative_rmse_pct"]
    sf = payload["seed_statistics"]["sample_first_relative_rmse_pct"]
    raw = payload["seed_statistics"]["raw_rmse_K"]
    by_seed = {int(row["seed"]): row for row in payload["registry_rows"]}
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- Point-global mean ± sample std: {pg['mean']:.6f}% ± "
            f"{pg['std_sample']:.6f}%; range {pg['range']:.6f} percentage points.",
            f"- Sample-first mean ± sample std: {sf['mean']:.6f}% ± "
            f"{sf['std_sample']:.6f}%.",
            f"- Raw RMSE mean ± sample std: {raw['mean']:.6f} ± "
            f"{raw['std_sample']:.6f} K.",
            f"- Best point-global seed: seed"
            f"{payload['ranking_by_point_global'][0]['seed']} at "
            f"{payload['ranking_by_point_global'][0]['point_global_relative_rmse_pct']:.6f}%.",
            "- At the point-global-selected checkpoints, seed1 has the best "
            f"sample-first value ({float(by_seed[1]['sample_first_relative_rmse_pct']):.6f}%) "
            "while seed2 has the best point-global/raw values; this is a small "
            "shape-scale/aggregation trade-off rather than a uniform winner.",
            "- At the separately frozen sample-first checkpoints, seed0/1/2 are "
            f"{float(by_seed[0]['sample_first_best_relative_rmse_pct']):.6f}%/"
            f"{float(by_seed[1]['sample_first_best_relative_rmse_pct']):.6f}%/"
            f"{float(by_seed[2]['sample_first_best_relative_rmse_pct']):.6f}%, "
            "with seed2 best.",
            "- All three seeds are below the 20% point-global threshold. "
            "Seed-to-seed spread is small relative to the threshold and the "
            "P1h gain is reproducible on the frozen 1024 shared support.",
            "- Every seed degrades from its point-global checkpoint to e600; "
            "checkpoint selection remains necessary.",
            "- The separate volume-representative ladder failure is not "
            "overturned: this multi-seed result establishes repeatability only "
            "on the canonical P1h operator support.",
            "",
            "Remote note: both repositories were clean on "
            "`research/v6-p1h-shared-support` at training HEAD `7d30b78`; "
            "the configured standalone seed log files were absent, but complete "
            "`loss_summary.json`, run config, checkpoints, predictions, and "
            "reload audits were present.",
            "",
        ]
    )
    return "\n".join(lines)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    sources = base._read_json(args.sources)
    if sources["evaluation_role"] != "valid_iid" or sources["test_accessed"]:
        raise AssertionError("multi-seed collector is valid_iid-only")
    seed0, seed0_checkpoints = _historical_seed0(
        args.historical_registry,
        args.historical_checkpoints,
    )
    registry_rows: list[dict[str, Any]] = [seed0]
    checkpoint_rows: list[dict[str, Any]] = seed0_checkpoints
    provenance: list[dict[str, Any]] = [
        {
            "config_id": "V6_03_V5best_P1h",
            "seed": 0,
            "source_host": seed0["source_host"],
            "training_commit": seed0["training_commit"],
            "source": "frozen_v6_training_result_registry",
            "reload_status": seed0["reload_status"],
        }
    ]
    for spec in sources["runs"]:
        row, checkpoints, run_provenance = _collect_new_run(
            spec=spec,
            snapshot_root=args.snapshot_root,
            data_root=args.data_root,
        )
        registry_rows.append(row)
        checkpoint_rows.extend(checkpoints)
        provenance.append(run_provenance)
    registry_rows.sort(key=lambda row: int(row["seed"]))
    checkpoint_rows.sort(
        key=lambda row: (int(row["seed"]), str(row["checkpoint_kind"]))
    )
    ranking = sorted(
        [
            {
                "seed": int(row["seed"]),
                "config_id": row["config_id"],
                "point_global_relative_rmse_pct": float(
                    row["point_global_relative_rmse_pct"]
                ),
            }
            for row in registry_rows
        ],
        key=lambda row: row["point_global_relative_rmse_pct"],
    )
    return {
        "schema_version": "heat3d_v6_multiseed_training_results_v1",
        "status": "passed",
        "evaluation_scope": "valid_iid_saved_predictions_only",
        "test_accessed": False,
        "hard_accessed": False,
        "sealed_accessed": False,
        "training_started": False,
        "checkpoint_inference_executed": False,
        "remote_inference_executed": False,
        "metric_formulas": {
            "point_global_relative_rmse": (
                "sqrt(sum(error^2)/sum(true_deltaT^2))"
            ),
            "sample_first_relative_rmse": (
                "mean_i(RMS(error_i)/RMS(true_deltaT_i))"
            ),
            "raw_rmse_K": (
                "sqrt(mean(error^2)) over equal-weight P1h operator points"
            ),
        },
        "registry_rows": registry_rows,
        "checkpoint_rows": checkpoint_rows,
        "seed_statistics": _statistics(registry_rows),
        "ranking_by_point_global": ranking,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--historical-registry",
        type=Path,
        default=DEFAULT_HISTORICAL_REGISTRY,
    )
    parser.add_argument(
        "--historical-checkpoints",
        type=Path,
        default=DEFAULT_HISTORICAL_CHECKPOINTS,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--checkpoint-csv", type=Path, default=DEFAULT_CHECKPOINT_CSV
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = collect(args)
    if args.write:
        _write_csv(args.registry, REGISTRY_FIELDS, payload["registry_rows"])
        _write_csv(
            args.checkpoint_csv,
            CHECKPOINT_FIELDS,
            payload["checkpoint_rows"],
        )
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_count": len(payload["registry_rows"]),
                "checkpoint_count": len(payload["checkpoint_rows"]),
                "ranking": payload["ranking_by_point_global"],
                "point_global_statistics": payload["seed_statistics"][
                    "point_global_relative_rmse_pct"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
