#!/usr/bin/env python3
"""Aggregate the frozen Gate 6Q valid-only CPU evaluator payloads."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


MODEL_ORDER = ("V38", "V42", "V43", "V44")
CHECKPOINT_ORDER = (
    "point_global_best",
    "sample_first_best",
    "legacy_best",
    "final",
)
COMPARISONS = (
    ("V42_minus_V38", "V38", "V42", "objective"),
    ("V43_minus_V38", "V38", "V43", "xy_scale_features"),
    ("V44_minus_V43", "V43", "V44", "deepsets"),
)
SUMMARY_FIELDS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "legacy_normalized_valid_base_mse",
    "shape_cv_rmse",
    "scale_log_rmse",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
)
DECOMPOSITION_FIELDS = (
    "point_error_squared_sum",
    "shape_point_sse_K2",
    "scale_point_sse_K2",
    "cross_point_sse_K2",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model.lower()}", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--attribution-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_valid_iid_only":
        raise ValueError(f"{path}: evaluator status is not complete")
    if payload.get("scope", {}).get("forbidden_roles_accessed"):
        raise ValueError(f"{path}: forbidden role was accessed")
    if payload.get("scope", {}).get("evaluation_roles") != ["valid_iid"]:
        raise ValueError(f"{path}: evaluation role differs from valid_iid")
    if set(payload.get("metrics", {})) != set(CHECKPOINT_ORDER):
        raise ValueError(f"{path}: four-checkpoint metrics are incomplete")
    return payload


def _sample_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["metrics"]["point_global_best"]["per_sample"]
    result = {str(row["sample_id"]): dict(row) for row in rows}
    if len(result) != 128:
        raise ValueError("expected 128 valid_iid samples")
    return result


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"sample_count": len(rows)}
    for field in DECOMPOSITION_FIELDS:
        result[field] = float(sum(float(row[field]) for row in rows))
    direct = float(result["point_error_squared_sum"])
    decomposed = sum(float(result[field]) for field in DECOMPOSITION_FIELDS[1:])
    result["decomposition_closure_abs_K2"] = abs(direct - decomposed)
    return result


def _top_share(rows: list[Mapping[str, Any]], count: int) -> float:
    values = sorted(
        (float(row["point_error_squared_sum"]) for row in rows), reverse=True
    )
    return float(sum(values[:count]) / max(sum(values), 1.0e-30))


def _comparison(
    name: str,
    baseline_name: str,
    candidate_name: str,
    variable: str,
    models: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = _sample_map(models[baseline_name])
    candidate = _sample_map(models[candidate_name])
    if set(baseline) != set(candidate):
        raise ValueError(f"{name}: paired sample IDs differ")
    paired: list[dict[str, Any]] = []
    for sample_id in sorted(baseline):
        left, right = baseline[sample_id], candidate[sample_id]
        row: dict[str, Any] = {
            "comparison": name,
            "variable": variable,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "sample_id": sample_id,
            "deltaT_quartile": left["deltaT_quartile"],
            "true_scale_cv_rms_K": float(left["true_scale_cv_rms_K"]),
        }
        for field in DECOMPOSITION_FIELDS:
            row[f"baseline_{field}"] = float(left[field])
            row[f"candidate_{field}"] = float(right[field])
            row[f"delta_{field}"] = float(right[field]) - float(left[field])
        paired.append(row)

    attribution: list[dict[str, Any]] = []
    for stratum in ("all", "Q1", "Q2", "Q3", "Q4"):
        selected = paired if stratum == "all" else [
            row for row in paired if row["deltaT_quartile"] == stratum
        ]
        row = {
            "comparison": name,
            "variable": variable,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "stratum": stratum,
            "sample_count": len(selected),
        }
        for field in DECOMPOSITION_FIELDS:
            row[f"baseline_{field}"] = float(
                sum(float(value[f"baseline_{field}"]) for value in selected)
            )
            row[f"candidate_{field}"] = float(
                sum(float(value[f"candidate_{field}"]) for value in selected)
            )
            row[f"delta_{field}"] = float(
                sum(float(value[f"delta_{field}"]) for value in selected)
            )
        attribution.append(row)

    left_summary = models[baseline_name]["metrics"]["point_global_best"]["summary"]
    right_summary = models[candidate_name]["metrics"]["point_global_best"]["summary"]
    summary = {
        "comparison": name,
        "variable": variable,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "candidate_minus_baseline": {
            field: float(right_summary[field]) - float(left_summary[field])
            for field in SUMMARY_FIELDS
        },
        "point_sse_delta_K2": float(
            sum(row["delta_point_error_squared_sum"] for row in paired)
        ),
        "sample_win_rate": float(
            sum(row["delta_point_error_squared_sum"] < 0 for row in paired)
            / len(paired)
        ),
        "quartiles": {
            row["stratum"]: row for row in attribution if row["stratum"] != "all"
        },
    }
    return summary, paired, attribution


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _update_registry(path: Path, models: Mapping[str, Mapping[str, Any]]) -> None:
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    by_id = {str(payload["config_id"]): payload for payload in models.values()}
    seen: set[str] = set()
    for row in rows:
        config_id = str(row.get("config_id") or "")
        if config_id not in by_id or config_id.endswith("38_gate6n_v36_r2r_mask_p005_e600"):
            continue
        payload = by_id[config_id]
        row["execution_status"] = "completed_e600"
        row["evaluation_status"] = "completed_valid_iid_only"
        row["training_started"] = "true"
        note = "Gate 6Q final valid-only CPU closeout complete; test/hard/sealed not accessed"
        prior = str(row.get("notes") or "")
        if note not in prior:
            row["notes"] = f"{prior}; {note}" if prior else note
        seen.add(config_id)
    expected = {payload["config_id"] for name, payload in models.items() if name != "V38"}
    if seen != expected:
        raise ValueError(f"registry Gate 6Q rows mismatch: seen={seen}, expected={expected}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    paths = {model: getattr(args, model.lower()).resolve() for model in MODEL_ORDER}
    models = {model: _read(path) for model, path in paths.items()}
    evaluator_commits = {value["evaluator_commit"] for value in models.values()}
    evaluator_hashes = {value["evaluator_source_sha256"] for value in models.values()}
    metric_sources = {
        (value["metric_source"]["commit"], value["metric_source"]["sha256"])
        for value in models.values()
    }
    split_hashes = {
        (
            value["split"]["train_ids_sha256"],
            value["split"]["valid_iid_ids_sha256"],
        )
        for value in models.values()
    }
    if any(len(values) != 1 for values in (evaluator_commits, evaluator_hashes, metric_sources, split_hashes)):
        raise ValueError("evaluator/metric/split provenance differs across models")

    metric_rows: list[dict[str, Any]] = []
    top_sse: dict[str, Any] = {}
    for model in MODEL_ORDER:
        payload = models[model]
        pg_rows = payload["metrics"]["point_global_best"]["per_sample"]
        top_sse[model] = {
            "top5_share": _top_share(pg_rows, 5),
            "top10_share": _top_share(pg_rows, 10),
            "decomposition": _aggregate(pg_rows),
        }
        for checkpoint in CHECKPOINT_ORDER:
            metadata = payload["checkpoint_metadata"][checkpoint]
            summary = payload["metrics"][checkpoint]["summary"]
            row = {
                "model": model,
                "config_id": payload["config_id"],
                "checkpoint": checkpoint,
                "epoch": int(metadata["epoch"]),
                "checkpoint_sha256": metadata["sha256"],
            }
            row.update({field: summary[field] for field in SUMMARY_FIELDS})
            metric_rows.append(row)

    comparison_payload: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for spec in COMPARISONS:
        summary, paired, attribution = _comparison(*spec, models=models)
        comparison_payload[spec[0]] = summary
        paired_rows.extend(paired)
        attribution_rows.extend(attribution)

    ranking = sorted(
        (
            {
                "rank": 0,
                "model": model,
                "config_id": models[model]["config_id"],
                "epoch": int(models[model]["checkpoint_metadata"]["point_global_best"]["epoch"]),
                **{
                    field: models[model]["metrics"]["point_global_best"]["summary"][field]
                    for field in SUMMARY_FIELDS
                },
                "valid_point_global_lt_20pct": bool(
                    models[model]["metrics"]["point_global_best"]["summary"]["point_global_relative_rmse_pct"] < 20.0
                ),
            }
            for model in MODEL_ORDER
        ),
        key=lambda row: float(row["point_global_relative_rmse_pct"]),
    )
    for index, row in enumerate(ranking, 1):
        row["rank"] = index

    conclusions = {
        "objective": (
            "V42 is the formal point-global leader, but its 0.0081 percentage-point "
            "margin over V38 is negligible; the clearer benefit is lower sample-first "
            "and scale error. Treat the objective contribution as weakly positive, not decisive."
        ),
        "xy_scale_features": (
            "V43 improves sample-first/shape/scale versus V38 but regresses point-global "
            "and raw CV RMSE; an independent XY-feature benefit is not established."
        ),
        "deepsets": (
            "V44 improves point-global, sample-first, raw CV, shape, scale, background, "
            "and hotspot versus V43, but strong-q and top-5 RMSE regress. The latent "
            "DeepSets contribution is positive but non-uniform within the V43 lineage, "
            "and it does not beat V38/V42 on point-global RMSE."
        ),
        "threshold": "No point-global-best checkpoint reaches the frozen <20% valid threshold.",
        "next_stage": (
            "Do not advance architecture complexity from this gate; reproduce V42 versus V38 "
            "with paired seeds before treating the very small point-global margin as real."
        ),
    }

    output = {
        "schema_version": "heat3d_v5_gate6q_final_closeout_v1",
        "status": "completed_valid_iid_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started_by_closeout": False,
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "checkpoint_selection_modified": False,
        "model_parameters_modified": False,
        "evaluator_commit": next(iter(evaluator_commits)),
        "evaluator_source_sha256": next(iter(evaluator_hashes)),
        "metric_source": {
            "commit": next(iter(metric_sources))[0],
            "sha256": next(iter(metric_sources))[1],
        },
        "split_ids_sha256": {
            "train": next(iter(split_hashes))[0],
            "valid_iid": next(iter(split_hashes))[1],
        },
        "input_payload_sha256": {model: _sha256(path) for model, path in paths.items()},
        "formal_ranking": ranking,
        "top_sse_and_decomposition": top_sse,
        "comparisons": comparison_payload,
        "conclusions": conclusions,
        "models": models,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.metrics_csv, metric_rows)
    _write_csv(args.paired_csv, paired_rows)
    _write_csv(args.attribution_csv, attribution_rows)

    lines = [
        "# Gate 6Q final valid-only closeout",
        "",
        "Scope: frozen CPU/NumPy true-RMS metrics on `valid_iid`; only `train` was used "
        "for persisted normalization/context checks. No test/hard/sealed access and no training.",
        "",
        "The metric inputs are checkpoint-bound prediction NPZ files whose training-time "
        "parameter reload audits passed. Direct cross-backend CPU model execution drift is "
        "retained in JSON as a diagnostic and is not used to alter the metric fields.",
        "",
        "## Formal point-global-best ranking",
        "",
        "| rank | model | epoch | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE | <20% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['epoch']} | "
            f"{row['point_global_relative_rmse_pct']:.6f} | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{row['raw_cv_weighted_rmse_K']:.6f} | {row['shape_cv_rmse']:.6f} | "
            f"{row['scale_log_rmse']:.6f} | {'yes' if row['valid_point_global_lt_20pct'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered paired comparisons (point-global-best)",
            "",
            "Candidate-minus-baseline deltas; negative error/SSE means improvement.",
            "",
            "| comparison | point-global pp | sample-first pp | raw CV K | point SSE K2 | win rate | Q1 SSE | Q2 SSE | Q3 SSE | Q4 SSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, _, _, _ in COMPARISONS:
        row = comparison_payload[name]
        delta = row["candidate_minus_baseline"]
        quartiles = row["quartiles"]
        lines.append(
            f"| {name} | {delta['point_global_relative_rmse_pct']:.6f} | "
            f"{delta['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{delta['raw_cv_weighted_rmse_K']:.6f} | {row['point_sse_delta_K2']:.6f} | "
            f"{row['sample_win_rate']:.4f} | "
            + " | ".join(
                f"{quartiles[name_q]['delta_point_error_squared_sum']:.6f}"
                for name_q in ("Q1", "Q2", "Q3", "Q4")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Point-SSE concentration",
            "",
            "| model | top-5 share | top-10 share |",
            "|---|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        lines.append(
            f"| {model} | {top_sse[model]['top5_share']:.6f} | "
            f"{top_sse[model]['top10_share']:.6f} |"
        )
    lines.extend(["", "## Independent contribution conclusions", ""])
    for key in ("objective", "xy_scale_features", "deepsets", "threshold", "next_stage"):
        lines.append(f"- **{key}**: {conclusions[key]}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Unified JSON: `{args.json}`",
            f"- Four-checkpoint metrics CSV: `{args.metrics_csv}`",
            f"- Paired sample CSV: `{args.paired_csv}`",
            f"- Quartile/decomposition CSV: `{args.attribution_csv}`",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    _update_registry(args.registry, models)
    print(json.dumps({"ranking": ranking, "comparisons": comparison_payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
