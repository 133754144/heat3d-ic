#!/usr/bin/env python3
"""Collect completed V6 e600 artifacts and evaluate saved valid_iid predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "configs/heat3d_v6/v6_training_result_sources.json"
DEFAULT_REGISTRY = ROOT / "configs/heat3d_v6/v6_training_result_registry.csv"
DEFAULT_CHECKPOINT_CSV = ROOT / "configs/heat3d_v6/v6_training_checkpoint_metrics.csv"
DEFAULT_JSON = ROOT / "configs/heat3d_v6/v6_latest_training_results.json"
DEFAULT_MD = ROOT / "docs/v6_latest_training_results.md"

REGISTRY_FIELDS = (
    "config_id",
    "dataset_id",
    "source_host",
    "execution_status",
    "evaluation_status",
    "training_commit",
    "remote_run_dir",
    "remote_log_path",
    "final_epoch",
    "selection_metric",
    "primary_checkpoint",
    "primary_epoch",
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
    "sample_first_best_epoch",
    "sample_first_best_relative_rmse_pct",
    "base_mse_best_epoch",
    "base_mse_best_value",
    "final_point_global_relative_rmse_pct",
    "final_sample_first_relative_rmse_pct",
    "final_raw_rmse_K",
    "best_to_final_point_global_delta_pct",
    "reload_status",
    "valid_sample_count",
    "threshold_lt20",
    "result_scope",
)

CHECKPOINT_FIELDS = (
    "config_id",
    "dataset_id",
    "source_host",
    "checkpoint_kind",
    "checkpoint_epoch",
    "checkpoint_file",
    "checkpoint_sha256",
    "prediction_file",
    "prediction_sha256",
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
    "scale_log_signed_bias",
    "valid_sample_count",
    "node_count",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values64))))


def _masked_rmse(errors: list[np.ndarray], masks: list[np.ndarray]) -> float:
    selected = [
        np.asarray(error, dtype=np.float64)[mask]
        for error, mask in zip(errors, masks, strict=True)
        if np.any(mask)
    ]
    if not selected:
        return float("nan")
    return _rmse(np.concatenate(selected))


def _load_truth(
    dataset_root: Path,
    sample_ids: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    truths: list[np.ndarray] = []
    q_fields: list[np.ndarray] = []
    references: list[float] = []
    for sample_id in sample_ids:
        sample_dir = dataset_root / sample_id
        meta = _read_json(sample_dir / "sample_meta.json")
        if meta["split_role"] != "valid":
            raise AssertionError(f"{sample_id}: collector may materialize valid only")
        reference = float(meta["boundary_conditions"]["bottom"]["T_inf_K"])
        temperature = np.load(sample_dir / "temperature.npy", allow_pickle=False)
        q_field = np.load(sample_dir / "q_field.npy", allow_pickle=False)
        truth = np.asarray(temperature, dtype=np.float64).reshape(-1) - reference
        q = np.asarray(q_field, dtype=np.float64).reshape(-1)
        if truth.shape != (1024,) or q.shape != (1024,):
            raise AssertionError(f"{sample_id}: expected 1024 operator nodes")
        truths.append(truth)
        q_fields.append(q)
        references.append(reference)
    return truths, q_fields, references


def _evaluate_prediction(
    prediction_path: Path,
    dataset_root: Path,
    target_std: float,
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    with np.load(prediction_path, allow_pickle=False) as archive:
        sample_ids = list(archive.files)
        if len(sample_ids) != 128 or len(set(sample_ids)) != 128:
            raise AssertionError(f"{prediction_path}: expected 128 unique valid samples")
        truths, q_fields, references = _load_truth(dataset_root, sample_ids)
        predictions = [
            np.asarray(archive[sample_id], dtype=np.float64).reshape(-1) - reference
            for sample_id, reference in zip(sample_ids, references, strict=True)
        ]

    errors = [prediction - truth for prediction, truth in zip(predictions, truths, strict=True)]
    total_sse = float(sum(np.sum(np.square(error)) for error in errors))
    total_energy = float(sum(np.sum(np.square(truth)) for truth in truths))
    point_count = len(sample_ids) * 1024
    sample_relative = np.asarray(
        [
            np.sqrt(np.mean(np.square(error)) / np.mean(np.square(truth)))
            for error, truth in zip(errors, truths, strict=True)
        ],
        dtype=np.float64,
    )
    true_scales = np.asarray([_rmse(truth) for truth in truths], dtype=np.float64)
    pred_scales = np.asarray([_rmse(prediction) for prediction in predictions], dtype=np.float64)
    shape_errors = np.asarray(
        [
            _rmse(prediction / pred_scale - truth / true_scale)
            for prediction, truth, pred_scale, true_scale in zip(
                predictions,
                truths,
                pred_scales,
                true_scales,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    log_scale_error = np.log(pred_scales / true_scales)
    correlations = []
    for prediction, truth in zip(predictions, truths, strict=True):
        correlations.append(float(np.corrcoef(prediction, truth)[0, 1]))

    hotspot_masks = [truth >= np.quantile(truth, 0.90) for truth in truths]
    top5_masks = [truth >= np.quantile(truth, 0.95) for truth in truths]
    strong_q_masks = []
    for q in q_fields:
        positive = q[q > 0.0]
        threshold = np.quantile(positive, 0.90) if positive.size else np.inf
        strong_q_masks.append((q > 0.0) & (q >= threshold))
    background_masks = [truth <= np.quantile(truth, 0.50) for truth in truths]
    background_errors = np.concatenate(
        [error[mask] for error, mask in zip(errors, background_masks, strict=True)]
    )
    background_predictions = np.concatenate(
        [prediction[mask] for prediction, mask in zip(predictions, background_masks, strict=True)]
    )
    background_truths = np.concatenate(
        [truth[mask] for truth, mask in zip(truths, background_masks, strict=True)]
    )

    metrics: dict[str, float | int] = {
        "point_global_relative_rmse_pct": float(
            100.0 * np.sqrt(total_sse / total_energy)
        ),
        "sample_first_relative_rmse_pct": 100.0 * float(np.mean(sample_relative)),
        "raw_rmse_K": float(np.sqrt(total_sse / point_count)),
        "base_mse": (total_sse / point_count) / (target_std * target_std),
        "amplitude_ratio": float(
            np.sqrt(
                sum(np.sum(np.square(prediction)) for prediction in predictions)
                / total_energy
            )
        ),
        "spatial_correlation": float(np.mean(correlations)),
        "hotspot_rmse_K": _masked_rmse(errors, hotspot_masks),
        "top5_rmse_K": _masked_rmse(errors, top5_masks),
        "strong_q_rmse_K": _masked_rmse(errors, strong_q_masks),
        "low_delta_bias_K": float(np.mean(background_errors)),
        "low_delta_rmse_K": _rmse(background_errors),
        "low_delta_over_ratio": float(np.mean(background_predictions > background_truths)),
        "shape_cv_rmse": float(np.mean(shape_errors)),
        "scale_log_rmse": _rmse(log_scale_error),
        "scale_log_signed_bias": float(np.mean(log_scale_error)),
        "valid_sample_count": len(sample_ids),
        "node_count": 1024,
    }
    per_sample = [
        {
            "sample_id": sample_id,
            "true_cv_rms_deltaT_K": float(true_scale),
            "point_sse_K2": float(np.sum(np.square(error))),
            "sample_relative_rmse_pct": float(100.0 * relative),
            "shape_cv_rmse": float(shape_error),
            "scale_log_error": float(scale_error),
        }
        for sample_id, true_scale, error, relative, shape_error, scale_error in zip(
            sample_ids,
            true_scales,
            errors,
            sample_relative,
            shape_errors,
            log_scale_error,
            strict=True,
        )
    ]
    return metrics, per_sample


def _reload_status(run_dir: Path, summary: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    recovery_name = spec.get("reload_recovery_file")
    if recovery_name:
        recovery = _read_json(run_dir / str(recovery_name))
        if recovery["status"] != "passed_recovered_post_export":
            raise AssertionError(f"{spec['config_id']}: reload recovery failed")
        return str(recovery["status"])
    audit = summary.get("checkpoint_prediction_reload_audit") or {}
    entries = audit.get("entries") or []
    if not entries or not all(bool(entry.get("passed")) for entry in entries):
        raise AssertionError(f"{spec['config_id']}: checkpoint reload audit incomplete")
    return str(audit.get("status") or "passed")


def _assert_summary_consistency(
    config_id: str,
    summary: Mapping[str, Any],
    primary: Mapping[str, Any],
    final: Mapping[str, Any],
    sample_best: Mapping[str, Any],
    base_best: Mapping[str, Any],
) -> dict[str, float]:
    expected_primary_value = summary.get("point_global_best_relative_rmse_pct")
    if expected_primary_value is None:
        expected_primary_value = summary["best_valid_iid_relative_rmse_pct_v4"]
    expected_primary = float(expected_primary_value)
    comparisons = {
        "primary_point_global_pct": (
            float(primary["point_global_relative_rmse_pct"]),
            expected_primary,
            1.0e-3,
        ),
        "primary_raw_rmse_K": (
            float(primary["raw_rmse_K"]),
            float(summary["best_valid_iid_raw_deltaT_rmse_K"]),
            1.0e-3,
        ),
        "final_point_global_pct": (
            float(final["point_global_relative_rmse_pct"]),
            float(summary["final_valid_iid_relative_rmse_pct_v4"]),
            1.0e-3,
        ),
        "base_mse_best": (
            float(base_best["base_mse"]),
            float(summary["best_valid_iid_base_mse"]),
            1.0e-4,
        ),
    }
    if summary.get("sample_first_best_relative_rmse_pct") is not None:
        comparisons["sample_first_best_pct"] = (
            float(sample_best["sample_first_relative_rmse_pct"]),
            float(summary["sample_first_best_relative_rmse_pct"]),
            1.0e-3,
        )
    deltas: dict[str, float] = {}
    for label, (actual, expected, tolerance) in comparisons.items():
        delta = abs(actual - expected)
        if delta > tolerance:
            raise AssertionError(
                f"{config_id}: {label} saved-prediction/summary mismatch "
                f"{actual} vs {expected} (tolerance {tolerance})"
            )
        deltas[label] = delta
    return deltas


def _paired_v6_03_v6_04(
    checkpoint_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    left_payload = checkpoint_payloads[
        ("V6_03_V5best_P1h", "point_global_best")
    ]
    right_payload = checkpoint_payloads[
        ("V6_04_V5best_P1h_DualAttention", "point_global_best")
    ]
    left = left_payload["per_sample"]
    right = right_payload["per_sample"]
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise AssertionError("V6_03/V6_04 valid sample IDs differ")
    ids = sorted(left_by_id)
    relative_delta = np.asarray(
        [
            right_by_id[sample_id]["sample_relative_rmse_pct"]
            - left_by_id[sample_id]["sample_relative_rmse_pct"]
            for sample_id in ids
        ],
        dtype=np.float64,
    )
    sse_delta = np.asarray(
        [
            right_by_id[sample_id]["point_sse_K2"] - left_by_id[sample_id]["point_sse_K2"]
            for sample_id in ids
        ],
        dtype=np.float64,
    )
    true_scale = np.asarray(
        [left_by_id[sample_id]["true_cv_rms_deltaT_K"] for sample_id in ids],
        dtype=np.float64,
    )
    quartile_edges = np.quantile(true_scale, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartiles = []
    for index in range(4):
        if index == 3:
            mask = (true_scale >= quartile_edges[index]) & (
                true_scale <= quartile_edges[index + 1]
            )
        else:
            mask = (true_scale >= quartile_edges[index]) & (
                true_scale < quartile_edges[index + 1]
            )
        quartiles.append(
            {
                "quartile": f"Q{index + 1}",
                "count": int(np.sum(mask)),
                "mean_sample_relative_delta_pct_point": float(np.mean(relative_delta[mask])),
                "point_sse_delta_K2": float(np.sum(sse_delta[mask])),
                "win_rate": float(np.mean(relative_delta[mask] < 0.0)),
            }
        )
    primary_metric_delta = {
        key: float(right_payload["metrics"][key] - left_payload["metrics"][key])
        for key in (
            "point_global_relative_rmse_pct",
            "sample_first_relative_rmse_pct",
            "raw_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
            "hotspot_rmse_K",
            "strong_q_rmse_K",
        )
    }
    return {
        "comparison": "V6_04_minus_V6_03_point_global_best",
        "sample_count": len(ids),
        "primary_metric_delta": primary_metric_delta,
        "sample_relative_win_rate": float(np.mean(relative_delta < 0.0)),
        "sample_relative_delta_mean_pct_point": float(np.mean(relative_delta)),
        "sample_relative_delta_median_pct_point": float(np.median(relative_delta)),
        "total_point_sse_delta_K2": float(np.sum(sse_delta)),
        "true_deltaT_quartiles": quartiles,
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: float) -> str:
    return f"{value:.6f}"


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6 latest completed training results",
        "",
        "Scope: saved `valid_iid` predictions only. No test role, checkpoint inference,",
        "training, or checkpoint mutation was performed by this collector.",
        "",
        "## Primary checkpoint comparison",
        "",
        "| config | dataset | host | epoch | point-global % | sample-first % | raw RMSE K | final point-global % |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["registry_rows"]:
        lines.append(
            "| {config_id} | {dataset_id} | {source_host} | {primary_epoch} | {pg} | "
            "{sf} | {raw} | {final} |".format(
                **row,
                pg=_format(float(row["point_global_relative_rmse_pct"])),
                sf=_format(float(row["sample_first_relative_rmse_pct"])),
                raw=_format(float(row["raw_rmse_K"])),
                final=_format(float(row["final_point_global_relative_rmse_pct"])),
            )
        )
    pair = payload["paired_v6_03_v6_04"]
    pair_delta = pair["primary_metric_delta"]
    by_id = {row["config_id"]: row for row in payload["registry_rows"]}
    v603 = by_id["V6_03_V5best_P1h"]
    v604 = by_id["V6_04_V5best_P1h_DualAttention"]
    lines.extend(
        [
            "",
            "## Primary-checkpoint diagnostics",
            "",
            "| config | amp ratio | correlation | hotspot K | strong-q K | low-DeltaT RMSE K | shape CV-RMSE | scale log-RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["registry_rows"]:
        lines.append(
            "| {config_id} | {amplitude_ratio:.6f} | {spatial_correlation:.6f} | "
            "{hotspot_rmse_K:.6f} | {strong_q_rmse_K:.6f} | "
            "{low_delta_rmse_K:.6f} | {shape_cv_rmse:.6f} | "
            "{scale_log_rmse:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "- All four runs completed e600. V6_01's training and exports completed, but its "
            "original strict max-absolute reload audit raised after export; the preserved "
            "post-export recovery audit passed without retraining or artifact mutation.",
            f"- At the point-global checkpoint, V6_04−V6_03 is "
            f"{pair_delta['point_global_relative_rmse_pct']:.6f} percentage points for "
            f"point-global and {pair_delta['raw_rmse_K']:.6f} K for raw RMSE, but "
            f"{pair_delta['sample_first_relative_rmse_pct']:+.6f} percentage points for "
            "sample-first (positive is worse).",
            f"- V6_04 also changes shape/scale error by "
            f"{pair_delta['shape_cv_rmse']:+.6f}/"
            f"{pair_delta['scale_log_rmse']:+.6f}; the point-global gain is therefore "
            "small and not a uniform shape-scale gain.",
            f"- V6_04−V6_03 paired sample-relative win rate: "
            f"{100.0 * pair['sample_relative_win_rate']:.2f}%.",
            f"- Mean/median sample-relative delta: "
            f"{pair['sample_relative_delta_mean_pct_point']:.6f}/"
            f"{pair['sample_relative_delta_median_pct_point']:.6f} percentage points.",
            f"- Total point-SSE delta: {pair['total_point_sse_delta_K2']:.6f} K² "
            "(negative favors V6_04).",
            f"- Under each run's sample-first-selected checkpoint, V6_04 is "
            f"{float(v604['sample_first_best_relative_rmse_pct']):.6f}% versus "
            f"{float(v603['sample_first_best_relative_rmse_pct']):.6f}% for V6_03; "
            "checkpoint selection materially changes the apparent conclusion.",
            "- P1g and P1h contain identical physical cases but use different operator supports; "
            "V6_02→V6_03 is therefore a representation comparison, not an identical-point metric replay.",
            "- V6_03→V6_04 is the clean shape-attention ablation because dataset, support, "
            "training contract, and seed are identical.",
            "",
            "## Metric formulas",
            "",
            "- point-global: `sqrt(sum(error^2) / sum(true_DeltaT^2))`",
            "- sample-first: mean per-sample `RMS(error) / RMS(true_DeltaT)`",
            "- raw RMSE: equal-weight RMSE over 128×1024 valid operator points",
            "- hotspot/top5: true-DeltaT per-sample top 10% / top 5%",
            "- strong-q: per-sample positive-q top decile",
            "- shape/scale: RMS-normalized field error and log RMS-scale error",
        ]
    )
    return "\n".join(lines) + "\n"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    sources = _read_json(args.sources)
    if sources["evaluation_role"] != "valid_iid" or sources["test_accessed"]:
        raise AssertionError("V6 result collector is valid_iid-only")
    registry_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    run_payloads: list[dict[str, Any]] = []

    for spec in sources["runs"]:
        config_id = str(spec["config_id"])
        run_dir = args.snapshot_root / str(spec["snapshot_path"])
        summary_path = run_dir / "loss_summary.json"
        config_path = run_dir / "run_config.json"
        summary = _read_json(summary_path)
        run_config = _read_json(config_path)
        if not summary.get("status_ok") or int(summary["final_epoch"]) != 600:
            raise AssertionError(f"{config_id}: not a completed finite e600 run")
        if str(summary["code_version_or_git_commit"]) != str(spec["training_commit"]):
            raise AssertionError(f"{config_id}: training commit mismatch")
        if int(run_config["epochs"]) != 600:
            raise AssertionError(f"{config_id}: run_config epoch drift")
        if int(run_config["batch_size"]) != 24 or int(run_config["micro_batch_size"]) != 24:
            raise AssertionError(f"{config_id}: expected B24/micro24")
        dataset_root = args.data_root / str(spec["dataset_id"])
        target_std = float(summary["train_only_normalization"]["target_delta_std"])
        reload_status = _reload_status(run_dir, summary, spec)
        checkpoint_metrics: dict[str, dict[str, Any]] = {}
        for kind, checkpoint in spec["checkpoints"].items():
            prediction_path = run_dir / str(checkpoint["prediction_file"])
            metrics, per_sample = _evaluate_prediction(
                prediction_path,
                dataset_root,
                target_std,
            )
            checkpoint_metrics[str(kind)] = metrics
            checkpoint_payloads[(config_id, str(kind))] = {
                "metrics": metrics,
                "per_sample": per_sample,
            }
            row = {
                "config_id": config_id,
                "dataset_id": spec["dataset_id"],
                "source_host": spec["source_host"],
                "checkpoint_kind": kind,
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_file": checkpoint["checkpoint_file"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "prediction_file": checkpoint["prediction_file"],
                "prediction_sha256": _sha256(prediction_path),
                **metrics,
            }
            checkpoint_rows.append(row)

        primary_kind = (
            "point_global_best"
            if "point_global_best" in checkpoint_metrics
            else "legacy_best"
        )
        primary = checkpoint_metrics[primary_kind]
        final = checkpoint_metrics["final"]
        sample_best = checkpoint_metrics.get("sample_first_best", primary)
        base_best = checkpoint_metrics.get("base_mse_best", primary)
        summary_consistency = _assert_summary_consistency(
            config_id,
            summary,
            primary,
            final,
            sample_best,
            base_best,
        )
        primary_epoch = int(spec["checkpoints"][primary_kind]["epoch"])
        registry_row = {
            "config_id": config_id,
            "dataset_id": spec["dataset_id"],
            "source_host": spec["source_host"],
            "execution_status": "completed_e600",
            "evaluation_status": "completed_valid_iid_saved_predictions",
            "training_commit": spec["training_commit"],
            "remote_run_dir": spec["remote_run_dir"],
            "remote_log_path": spec["remote_log_path"] or "",
            "final_epoch": 600,
            "selection_metric": summary["selection_metric"],
            "primary_checkpoint": primary_kind,
            "primary_epoch": primary_epoch,
            **{key: primary[key] for key in (
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
            )},
            "sample_first_best_epoch": spec["checkpoints"].get(
                "sample_first_best", spec["checkpoints"][primary_kind]
            )["epoch"],
            "sample_first_best_relative_rmse_pct": sample_best[
                "sample_first_relative_rmse_pct"
            ],
            "base_mse_best_epoch": spec["checkpoints"].get(
                "base_mse_best", spec["checkpoints"][primary_kind]
            )["epoch"],
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
        registry_rows.append(registry_row)
        run_payloads.append(
            {
                "config_id": config_id,
                "source_host": spec["source_host"],
                "dataset_id": spec["dataset_id"],
                "training_commit": spec["training_commit"],
                "run_dir": spec["remote_run_dir"],
                "loss_summary_sha256": _sha256(summary_path),
                "run_config_sha256": _sha256(config_path),
                "status": "completed_e600",
                "reload_status": reload_status,
                "saved_prediction_summary_consistency_abs_delta": summary_consistency,
                "checkpoint_metrics": checkpoint_metrics,
            }
        )

    registry_rows.sort(key=lambda row: str(row["config_id"]))
    checkpoint_rows.sort(key=lambda row: (str(row["config_id"]), str(row["checkpoint_kind"])))
    ranking = sorted(
        (
            {
                "config_id": row["config_id"],
                "point_global_relative_rmse_pct": row["point_global_relative_rmse_pct"],
            }
            for row in registry_rows
        ),
        key=lambda row: float(row["point_global_relative_rmse_pct"]),
    )
    payload = {
        "schema_version": "heat3d_v6_latest_training_results_v1",
        "status": "passed",
        "evaluation_scope": "valid_iid_saved_predictions_only",
        "test_accessed": False,
        "training_started": False,
        "checkpoint_inference_executed": False,
        "metric_formulas": {
            "point_global_relative_rmse": "sqrt(sum(error^2)/sum(true_deltaT^2))",
            "sample_first_relative_rmse": "mean_i(RMS(error_i)/RMS(true_deltaT_i))",
            "raw_rmse_K": "sqrt(mean(error^2)) over equal-weight operator points",
        },
        "runs": run_payloads,
        "registry_rows": registry_rows,
        "checkpoint_rows": checkpoint_rows,
        "ranking_by_primary_point_global": ranking,
        "paired_v6_03_v6_04": _paired_v6_03_v6_04(checkpoint_payloads),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoint-csv", type=Path, default=DEFAULT_CHECKPOINT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = collect(args)
    if args.write:
        _write_csv(args.registry, REGISTRY_FIELDS, payload["registry_rows"])
        _write_csv(args.checkpoint_csv, CHECKPOINT_FIELDS, payload["checkpoint_rows"])
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_count": len(payload["runs"]),
                "checkpoint_count": len(payload["checkpoint_rows"]),
                "ranking": payload["ranking_by_primary_point_global"],
                "paired_v6_03_v6_04": payload["paired_v6_03_v6_04"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
