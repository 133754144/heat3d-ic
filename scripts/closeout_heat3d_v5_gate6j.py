#!/usr/bin/env python3
"""Render and register the Gate 6J valid-only causal diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
EXPECTED_EVALUATOR = "3d94dacba922c4288edb09adfdb47614e370234b"
DEFAULT_REGISTRY = (
    ROOT / "configs/heat3d_v5/v5_gate6h_attention_fix_registry.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "configs/heat3d_v5/gate6j"
DEFAULT_MD = ROOT / "docs/v5_gate6j_valid_causal_diagnostic.md"
REGISTRY_FIELDS = (
    "gate6j_status",
    "gate6j_evaluator_commit",
    "gate6j_v13_checkpoint",
    "gate6j_v32_checkpoint",
    "gate6j_point_global_delta_pct_points",
    "gate6j_point_global_ci95",
    "gate6j_sample_first_delta_pct_points",
    "gate6j_sample_first_ci95",
    "gate6j_sample_first_win_rate",
    "gate6j_alpha_sensitivity_interval",
    "gate6j_recommendation",
    "gate6j_report_json",
    "gate6j_roles_accessed",
    "gate6j_training_started",
    "gate6j_test_accessed",
    "gate6j_hard_accessed",
    "gate6j_sealed_iid_accessed",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--raw-paired-csv", type=Path, required=True)
    parser.add_argument("--raw-alpha-csv", type=Path, required=True)
    parser.add_argument("--raw-strata-csv", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q975": float(np.quantile(array, 0.975)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _validate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "heat3d_v5_gate6j_valid_causal_v1":
        raise ValueError("unexpected Gate 6J schema")
    if payload.get("evaluator_commit") != EXPECTED_EVALUATOR:
        raise ValueError("Gate 6J evaluator commit drifted")
    scope = payload["scope"]
    if scope["roles_accessed"] != ["train", "valid_iid"]:
        raise ValueError("Gate 6J role scope drifted")
    for field in (
        "training_started",
        "model_parameters_modified",
        "checkpoints_modified",
        "test_accessed",
        "hard_accessed",
        "sealed_iid_accessed",
    ):
        if scope[field] is not False:
            raise ValueError(f"forbidden Gate 6J action: {field}")
    if payload["models"]["v13"]["epoch"] != 318:
        raise ValueError("V13 checkpoint epoch drifted")
    if payload["models"]["v32"]["epoch"] != 474:
        raise ValueError("V32 checkpoint epoch drifted")
    if len(payload["paired_samples"]) != 128:
        raise ValueError("paired sample count drifted")
    if payload["alpha_sweep"]["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid drifted")
    if payload["route_decision"]["recommendation"] not in {
        "residual_strength_control",
        "objective_alignment",
        "parameter_matched_control",
    }:
        raise ValueError("unknown Gate 6J route")


def _copy_csv(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not fields or not rows:
        raise ValueError(f"empty CSV: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    paired = payload["paired_samples"]
    residual = {
        "attention_residual_mean_pool_cosine": _distribution(
            [
                row["attention_residual_mean_pool_cosine"]
                for row in paired
            ]
        ),
        "attention_residual_to_mean_pool_norm_ratio": _distribution(
            [
                row["attention_residual_to_mean_pool_norm_ratio"]
                for row in paired
            ]
        ),
    }
    return {
        "schema_version": "heat3d_v5_gate6j_closeout_v1",
        "status": payload["status"],
        "evaluator_commit": payload["evaluator_commit"],
        "evaluator_source_sha256": payload["evaluator_source_sha256"],
        "metric_schema_version": payload["metric_schema_version"],
        "frozen_formula_source": payload["frozen_formula_source"],
        "scope": payload["scope"],
        "split": payload["split"],
        "normalization_and_context": payload["normalization_and_context"],
        "models": {
            name: {
                key: value
                for key, value in model.items()
                if key != "metrics"
            }
            | {"metrics": {"summary": model["metrics"]["summary"]}}
            for name, model in payload["models"].items()
        },
        "paired_bootstrap": payload["paired_bootstrap"],
        "stratification_definitions": {
            "true_cv_rms_deltaT_K": (
                "frozen target CV-RMS, evaluation-only"
            ),
            "total_power_W": "P_operator_W from q and control volumes",
            "source_occupancy_fraction": (
                "unweighted node fraction with q > 1e-12"
            ),
            "q_weighted_inverse_kz_mK_W": (
                "q and control-volume weighted inverse kz"
            ),
            "generator_condition_category": (
                "pre-solve q_block_metadata.DeltaT_target_bin; generator "
                "metadata only, not solved-temperature labels"
            ),
        },
        "stratified_paired_analysis": payload[
            "stratified_paired_analysis"
        ],
        "attention_residual_analysis": {
            "definition": payload["attention_residual_analysis"][
                "definition"
            ],
            "distributions": residual,
            "correlations": payload["attention_residual_analysis"][
                "correlations"
            ],
        },
        "alpha_sweep": {
            "alphas": payload["alpha_sweep"]["alphas"],
            "inference_only": True,
            "checkpoint_fixed": payload["alpha_sweep"][
                "checkpoint_fixed"
            ],
            "checkpoint_selection_performed": False,
            "replay": payload["alpha_sweep"]["replay"],
            "metrics": {
                alpha: {"summary": result["summary"]}
                for alpha, result in payload["alpha_sweep"][
                    "metrics"
                ].items()
            },
        },
        "route_decision": payload["route_decision"],
        "artifacts": payload["artifacts"],
        "tracked_tables": {
            "paired_samples_csv": (
                "configs/heat3d_v5/gate6j/gate6j_paired_samples.csv"
            ),
            "alpha_sweep_csv": (
                "configs/heat3d_v5/gate6j/gate6j_alpha_sweep.csv"
            ),
            "strata_csv": "configs/heat3d_v5/gate6j/gate6j_strata.csv",
        },
    }


def _ci_text(metric: Mapping[str, Any]) -> str:
    low, high = metric["bootstrap_95pct_ci"]
    return f"[{low:.6f}, {high:.6f}]"


def _markdown(payload: Mapping[str, Any]) -> str:
    models = payload["models"]
    v13 = models["v13"]["metrics"]["summary"]
    v32 = models["v32"]["metrics"]["summary"]
    bootstrap = payload["paired_bootstrap"]["metrics"]
    residual = payload["attention_residual_analysis"]
    route = payload["route_decision"]
    alpha_metrics = payload["alpha_sweep"]["metrics"]
    lines = [
        "# Gate 6J valid-only causal diagnostic",
        "",
        "Scope: existing V13 e318 and V32 e474 checkpoints, train-only "
        "normalization/context reconstruction, and `valid_iid` evaluation. "
        "No training, checkpoint/model mutation, checkpoint selection, or "
        "test/hard/sealed access occurred.",
        "",
        "## Frozen V5 metrics",
        "",
        "| model | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| V13 e318 | {v13['point_global_relative_rmse_pct']:.6f} "
            f"| {v13['sample_first_cv_relative_rmse_pct']:.6f} "
            f"| {v13['raw_cv_weighted_rmse_K']:.6f} "
            f"| {v13['shape_cv_rmse']:.6f} "
            f"| {v13['scale_log_rmse']:.6f} |"
        ),
        (
            f"| V32 e474 | {v32['point_global_relative_rmse_pct']:.6f} "
            f"| {v32['sample_first_cv_relative_rmse_pct']:.6f} "
            f"| {v32['raw_cv_weighted_rmse_K']:.6f} "
            f"| {v32['shape_cv_rmse']:.6f} "
            f"| {v32['scale_log_rmse']:.6f} |"
        ),
        "",
        "## Paired bootstrap",
        "",
        "All differences are V32 minus V13; negative is better.",
        "",
        "| metric | observed Δ | 95% CI | win rate | median per-sample Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (
        "point_global_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
        "shape_cv_rmse",
        "scale_log_rmse",
    ):
        row = bootstrap[metric]
        lines.append(
            f"| {metric} | {row['observed_aggregate_difference']:.6f} "
            f"| {_ci_text(row)} | {row['per_sample_win_rate']:.4f} "
            f"| {row['per_sample_median_difference']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The point-global gain is tail-sensitive: its CI crosses zero. "
            "Sample-first has a positive mean difference despite a negative "
            "median and a win rate above 50%, showing that a minority of "
            "regressions dominates the unweighted sample mean.",
            "",
        "## Stratified result",
        "",
        "Quartile axes are fixed from the 128 valid samples. The condition "
        "category is the pre-solve generator metadata "
        "`q_block_metadata.DeltaT_target_bin`; it is not derived from solved "
        "temperature labels.",
        "",
        ]
    )
    strata = payload["stratified_paired_analysis"]
    for feature in (
        "true_cv_rms_deltaT_K",
        "total_power_W",
        "source_occupancy_fraction",
        "q_weighted_inverse_kz_mK_W",
        "generator_condition_category",
    ):
        lines.extend(
            [
                f"### {feature}",
                "",
                "| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in strata[feature]["bins"]:
            lines.append(
                f"| {row['label']} | {row['sample_count']} "
                f"| {row['v32_minus_v13_sample_first_pct_points']:.6f} "
                f"| {row['sample_first_win_rate']:.4f} "
                f"| {row['v32_minus_v13_point_sse_K2']:.6f} |"
            )
        lines.append("")
    cosine = residual["distributions"][
        "attention_residual_mean_pool_cosine"
    ]
    ratio = residual["distributions"][
        "attention_residual_to_mean_pool_norm_ratio"
    ]
    lines.extend(
        [
            "## Attention residual",
            "",
            (
                "Residual/mean-pool cosine: "
                f"mean `{cosine['mean']:.6f}`, median `{cosine['median']:.6f}`. "
                "Norm ratio: "
                f"mean `{ratio['mean']:.6f}`, median `{ratio['median']:.6f}`."
            ),
            "",
            "Residual cosine/norm ratio has weak correlation with the V32−V13 "
            "shape, scale, and relative-error changes, while cosine is strongly "
            "correlated with true ΔT and q-weighted inverse-kz. The path is "
            "physics-responsive, but residual magnitude alone does not explain "
            "which samples regress.",
            "",
            "## Inference-only α sweep",
            "",
            "| α | point-global % | sample-first % | raw CV RMSE K | Q1 sample-first Δ pp |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    route_rows = {str(row["alpha"]): row for row in route["rows"]}
    for alpha in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        summary = alpha_metrics[alpha]["summary"]
        route_row = route_rows[alpha]
        lines.append(
            f"| {float(alpha):.2f} "
            f"| {summary['point_global_relative_rmse_pct']:.6f} "
            f"| {summary['sample_first_cv_relative_rmse_pct']:.6f} "
            f"| {summary['raw_cv_weighted_rmse_K']:.6f} "
            f"| {route_row['q1_sample_first_delta_vs_v13_pct_points']:.6f} |"
        )
    interval = route["optimal_sensitivity_interval_alpha"]
    lines.extend(
        [
            "",
            (
                "Optimal discrete sensitivity interval: "
                f"`[{interval[0]:.2f}, {interval[1]:.2f}]`. "
                "Only α=1 preserves the point-global gain; no α restores "
                "sample-first, and low-ΔT Q1 regresses for every α."
            ),
            "",
            "## Decision",
            "",
            f"Unique recommendation: **{route['recommendation']}**.",
            "",
            f"Reason: {route['reason']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _update_registry(
    path: Path, compact: Mapping[str, Any], report_path: Path
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if any(field in fields for field in REGISTRY_FIELDS):
        if not all(field in fields for field in REGISTRY_FIELDS):
            raise ValueError("partial Gate 6J registry schema")
    else:
        fields.extend(REGISTRY_FIELDS)
    matches = [row for row in rows if row["config_id"] == CONFIG_ID]
    if len(matches) != 1:
        raise ValueError("expected exactly one V32 registry row")
    row = matches[0]
    metrics = compact["paired_bootstrap"]["metrics"]
    point = metrics["point_global_relative_rmse_pct"]
    sample = metrics["sample_first_cv_relative_rmse_pct"]
    route = compact["route_decision"]
    interval = route["optimal_sensitivity_interval_alpha"]
    row.update(
        {
            "gate6j_status": "completed_valid_iid_no_training",
            "gate6j_evaluator_commit": compact["evaluator_commit"],
            "gate6j_v13_checkpoint": (
                "params_best.pkl:e318:"
                + compact["models"]["v13"]["sha256"]
            ),
            "gate6j_v32_checkpoint": (
                "params_best_valid_point_global.pkl:e474:"
                + compact["models"]["v32"]["sha256"]
            ),
            "gate6j_point_global_delta_pct_points": (
                f"{point['observed_aggregate_difference']:.12g}"
            ),
            "gate6j_point_global_ci95": "|".join(
                f"{value:.12g}" for value in point["bootstrap_95pct_ci"]
            ),
            "gate6j_sample_first_delta_pct_points": (
                f"{sample['observed_aggregate_difference']:.12g}"
            ),
            "gate6j_sample_first_ci95": "|".join(
                f"{value:.12g}" for value in sample["bootstrap_95pct_ci"]
            ),
            "gate6j_sample_first_win_rate": (
                f"{sample['per_sample_win_rate']:.12g}"
            ),
            "gate6j_alpha_sensitivity_interval": (
                f"{interval[0]:g}|{interval[1]:g}"
            ),
            "gate6j_recommendation": route["recommendation"],
            "gate6j_report_json": str(report_path.relative_to(ROOT)),
            "gate6j_roles_accessed": "train|valid_iid",
            "gate6j_training_started": "false",
            "gate6j_test_accessed": "false",
            "gate6j_hard_accessed": "false",
            "gate6j_sealed_iid_accessed": "false",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    payload = json.loads(args.raw_json.read_text(encoding="utf-8"))
    _validate(payload)
    compact = _compact(payload)
    output_json = args.output_dir / "gate6j_causal_diagnostic.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _copy_csv(
        args.raw_paired_csv,
        args.output_dir / "gate6j_paired_samples.csv",
    )
    _copy_csv(
        args.raw_alpha_csv,
        args.output_dir / "gate6j_alpha_sweep.csv",
    )
    _copy_csv(
        args.raw_strata_csv,
        args.output_dir / "gate6j_strata.csv",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(compact), encoding="utf-8")
    _update_registry(args.registry, compact, output_json)
    print(
        json.dumps(
            {
                "status": "passed",
                "output_json": str(output_json),
                "output_md": str(args.output_md),
                "recommendation": compact["route_decision"][
                    "recommendation"
                ],
                "alpha_interval": compact["route_decision"][
                    "optimal_sensitivity_interval_alpha"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
