#!/usr/bin/env python3
"""Render the frozen V32 valid-only closeout and update lifecycle fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
CHECKPOINT_ORDER = (
    "point_global_best",
    "legacy_best",
    "sample_first_best",
    "final",
)
BASELINE = {
    "label": "V13 historical noncontemporaneous",
    "point_global_relative_rmse_pct": 23.700678,
    "sample_first_cv_relative_rmse_pct": 20.316459,
    "raw_cv_weighted_rmse_K": 0.167982,
}
METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
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
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-json", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT
        / "configs/heat3d_v5/v5_gate6h_attention_fix_registry.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "configs/heat3d_v5/gate6h/v32_closeout",
    )
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    return parser.parse_args()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _summary(payload: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return payload["metrics"][checkpoint]["summary"]


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "heat3d_v5_v32_valid_only_closeout_v1":
        raise ValueError("unexpected evaluator schema")
    if payload.get("config_id") != CONFIG_ID:
        raise ValueError("config binding failed")
    scope = payload["scope"]
    if scope.get("evaluation_roles") != ["valid_iid"]:
        raise ValueError("evaluation is not valid_iid-only")
    for field in ("test_accessed", "hard_accessed", "sealed_iid_accessed"):
        if scope.get(field) is not False:
            raise ValueError(f"forbidden role accessed: {field}")
    if set(payload["metrics"]) != set(CHECKPOINT_ORDER):
        raise ValueError("four-checkpoint metric set is incomplete")
    for checkpoint in CHECKPOINT_ORDER:
        row = _summary(payload, checkpoint)
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite {checkpoint}.{metric}")


def _comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in CHECKPOINT_ORDER:
        summary = _summary(payload, checkpoint)
        metadata = payload["checkpoint_metadata"][checkpoint]
        attention = payload["attention_diagnostics"][checkpoint]
        row = {
            "checkpoint": checkpoint,
            "epoch": metadata["epoch"],
            "sha256": metadata["sha256"],
            "parameter_count": metadata["parameter_count"],
            "parameter_reload_max_abs_error": metadata[
                "parameter_reload_max_abs_error"
            ],
            "training_replay_max_abs_error_K": metadata[
                "training_replay_max_abs_error_K"
            ],
            "evaluator_replay_max_abs_error_K": attention[
                "evaluator_replay_max_abs_error_K"
            ],
        }
        row.update({metric: summary[metric] for metric in METRICS})
        for metric in (
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
        ):
            row[f"delta_vs_v13_{metric}"] = summary[metric] - BASELINE[metric]
        rows.append(row)
    return rows


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _metric_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V32 valid-only metrics",
        "",
        "- Scope: `valid_iid` only (128 samples, 1024 nodes/sample).",
        f"- Training commit: `{payload['training_commit']}`.",
        f"- Evaluator commit: `{payload['evaluator_commit']}`.",
        (
            "- Frozen formula source: "
            f"`{payload['frozen_formula_source']['commit']}`."
        ),
        (
            "- Log integrity: declared log is absent; e600 completion is supported "
            "by contiguous loss history and the final checkpoint, but log "
            "completeness cannot be verified."
        ),
        "- No test, hard, or sealed-IID role was accessed.",
        "",
        "| checkpoint | epoch | point-global % | sample-first % | raw CV RMSE K | amplitude | correlation | legacy MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINT_ORDER:
        summary = _summary(payload, checkpoint)
        epoch = payload["checkpoint_metadata"][checkpoint]["epoch"]
        lines.append(
            "| "
            + " | ".join(
                (
                    checkpoint,
                    str(epoch),
                    f"{summary['point_global_relative_rmse_pct']:.6f}",
                    f"{summary['sample_first_cv_relative_rmse_pct']:.6f}",
                    f"{summary['raw_cv_weighted_rmse_K']:.6f}",
                    f"{summary['amplitude_ratio']:.6f}",
                    f"{summary['spatial_correlation']:.6f}",
                    f"{summary['legacy_normalized_valid_base_mse']:.8f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Remaining frozen V5 metrics",
            "",
            "| checkpoint | hotspot K | top-5 K | strong-q K | low-ΔT bias K | low-ΔT RMSE K | low-ΔT over-ratio | shape CV-RMSE | scale log-RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in CHECKPOINT_ORDER:
        summary = _summary(payload, checkpoint)
        lines.append(
            "| "
            + " | ".join(
                (
                    checkpoint,
                    f"{summary['hotspot_cv_weighted_rmse_K']:.6f}",
                    f"{summary['top5_cv_weighted_rmse_K']:.6f}",
                    f"{summary['strong_q_cv_weighted_rmse_K']:.6f}",
                    f"{summary['low_deltaT_background_bias_K']:.6f}",
                    f"{summary['low_deltaT_background_rmse_K']:.6f}",
                    f"{summary['low_deltaT_background_over_ratio']:.6f}",
                    f"{summary['shape_cv_rmse']:.6f}",
                    f"{summary['scale_log_rmse']:.6f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The saved sample-first checkpoint used "
            "`valid_native_joint_relative_rmse` with ordinary raw RMSE as a "
            "tie-break only on exact equality. The correct CV metric above is "
            "post-hoc diagnostic evidence; no checkpoint was reselected.",
            "",
        ]
    )
    return "\n".join(lines)


def _attention_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V32 attention diagnostics",
        "",
        "Scope is valid_iid-only; test/hard/sealed were not accessed.",
        "",
        "| checkpoint | epoch | entropy | max weight | residual/mean-pool | classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for checkpoint in CHECKPOINT_ORDER:
        row = payload["attention_diagnostics"][checkpoint]
        lines.append(
            f"| {checkpoint} | {payload['checkpoint_metadata'][checkpoint]['epoch']} "
            f"| {row['normalized_entropy']['mean']:.6f} "
            f"| {row['maximum_weight']['mean']:.6f} "
            f"| {row['residual_to_mean_pool_l2_ratio']['mean']:.6f} "
            f"| {row['classification']} |"
        )
    primary = payload["attention_diagnostics"]["point_global_best"]
    correlations = primary[
        "weight_feature_correlations_computed_per_sample_then_aggregated"
    ]
    lines.extend(
        [
            "",
            "For the point-global checkpoint, mean per-sample Pearson/Spearman "
            "correlations are:",
            "",
            "| feature | Pearson | Spearman |",
            "|---|---:|---:|",
        ]
    )
    for feature, values in correlations.items():
        lines.append(
            f"| {feature} | {values['pearson']['mean']:.6f} "
            f"| {values['spearman']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The attention is neither uniform nor collapsed. Its residual is "
            "large relative to mean pooling, while the learned weights are "
            "negatively correlated with source occupancy/q and most strongly "
            "with inverse-kz. This supports an attention-bias/residual-strength "
            "audit before additional seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def _closeout_markdown(payload: dict[str, Any]) -> str:
    primary = _summary(payload, "point_global_best")
    final = _summary(payload, "final")
    return "\n".join(
        [
            "# V32 versus frozen V13 closeout",
            "",
            "Scope is valid_iid-only; test/hard/sealed were not accessed.",
            "",
            "This is a `historical noncontemporaneous` comparison: V13 values "
            "come from the frozen historical report, while V32 was recomputed "
            "with the frozen V5 formulas.",
            "",
            "| model/checkpoint | point-global % | sample-first % | raw CV RMSE K |",
            "|---|---:|---:|---:|",
            (
                f"| V13 historical | {BASELINE['point_global_relative_rmse_pct']:.6f} "
                f"| {BASELINE['sample_first_cv_relative_rmse_pct']:.6f} "
                f"| {BASELINE['raw_cv_weighted_rmse_K']:.6f} |"
            ),
            (
                f"| V32 point-global e474 | "
                f"{primary['point_global_relative_rmse_pct']:.6f} "
                f"| {primary['sample_first_cv_relative_rmse_pct']:.6f} "
                f"| {primary['raw_cv_weighted_rmse_K']:.6f} |"
            ),
            (
                f"| V32 final e600 | {final['point_global_relative_rmse_pct']:.6f} "
                f"| {final['sample_first_cv_relative_rmse_pct']:.6f} "
                f"| {final['raw_cv_weighted_rmse_K']:.6f} |"
            ),
            "",
            (
                "V32 e474 versus V13: point-global "
                f"{primary['point_global_relative_rmse_pct'] - BASELINE['point_global_relative_rmse_pct']:+.6f} pp, "
                "sample-first "
                f"{primary['sample_first_cv_relative_rmse_pct'] - BASELINE['sample_first_cv_relative_rmse_pct']:+.6f} pp, "
                "raw CV RMSE "
                f"{primary['raw_cv_weighted_rmse_K'] - BASELINE['raw_cv_weighted_rmse_K']:+.6f} K."
            ),
            (
                "V32 best→final: point-global "
                f"{final['point_global_relative_rmse_pct'] - primary['point_global_relative_rmse_pct']:+.6f} pp, "
                "sample-first "
                f"{final['sample_first_cv_relative_rmse_pct'] - primary['sample_first_cv_relative_rmse_pct']:+.6f} pp, "
                "raw CV RMSE "
                f"{final['raw_cv_weighted_rmse_K'] - primary['raw_cv_weighted_rmse_K']:+.6f} K."
            ),
            "",
            "Decision: **V32 is not advanced**. Point-global and raw CV RMSE "
            "improved, but sample-first regressed and the <20% valid threshold "
            "was not met. No seed1/seed2 run is authorized by this closeout.",
            "",
        ]
    )


def _options_markdown() -> str:
    return "\n".join(
        [
            "# V32 next optimization options",
            "",
            "Unique recommendation: audit and control the physics-attention "
            "residual strength before any multi-seed expansion. The preferred "
            "next experiment is a preregistered residual-strength control "
            "(for example, zero-initialized learnable residual scale) against "
            "the unchanged V32 path.",
            "",
            "Rationale: point-global improved while sample-first regressed; "
            "attention is demonstrably non-uniform and non-collapsed, but its "
            "residual norm is about three quarters of the mean-pool norm and "
            "its source/q correlations are negative. This is an attention-bias "
            "question, not evidence for more seeds or a larger model.",
            "",
            "This document does not authorize training or test/hard/sealed access.",
            "",
        ]
    )


def _update_registry(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    matches = [row for row in rows if row["config_id"] == CONFIG_ID]
    if len(matches) != 1:
        raise ValueError("expected exactly one V32 registry row")
    row = matches[0]
    row["plan_status"] = "frozen_closed"
    row["execution_status"] = "completed_e600"
    row["evaluation_status"] = "completed_valid_iid_four_checkpoint"
    row["long_training_started"] = "true"
    row["test_accessed"] = "false"
    row["hard_accessed"] = "false"
    row["sealed_iid_accessed"] = "false"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    payload = json.loads(args.evaluator_json.read_text(encoding="utf-8"))
    _validate(payload)
    rows = _comparison_rows(payload)
    _json_write(args.output_dir / "v32_valid_only_metrics.json", payload)
    attention_payload = {
        "schema_version": "heat3d_v5_v32_attention_diagnostics_v1",
        "config_id": CONFIG_ID,
        "training_commit": payload["training_commit"],
        "evaluator_commit": payload["evaluator_commit"],
        "scope": payload["scope"],
        "checkpoint_metadata": payload["checkpoint_metadata"],
        "attention_diagnostics": payload["attention_diagnostics"],
    }
    _json_write(
        args.output_dir / "v32_attention_diagnostics.json", attention_payload
    )
    _write_comparison_csv(
        args.output_dir / "v32_checkpoint_comparison.csv", rows
    )
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "v32_valid_only_metrics.md").write_text(
        _metric_markdown(payload), encoding="utf-8"
    )
    (args.docs_dir / "v32_attention_diagnostics.md").write_text(
        _attention_markdown(payload), encoding="utf-8"
    )
    (args.docs_dir / "v32_vs_v13_closeout.md").write_text(
        _closeout_markdown(payload), encoding="utf-8"
    )
    (args.docs_dir / "v32_next_optimization_options.md").write_text(
        _options_markdown(), encoding="utf-8"
    )
    _update_registry(args.registry)
    print(
        json.dumps(
            {
                "status": "passed",
                "config_id": CONFIG_ID,
                "promotion": "not_advanced",
                "unique_recommendation": "attention_residual_strength_control",
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
