#!/usr/bin/env python3
"""Corrected case bootstrap whose estimator matches the P23 common table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "sample_first_relative_rmse_pct",
    "point_global_relative_rmse_pct",
    "cv_weighted_point_global_relative_rmse_pct",
    "rmse_K",
    "mae_K",
    "peak_temperature_mae_K",
    "peak_temperature_rmse_K",
    "true_hotspot_region_rmse_K",
)
SUM_FIELDS = ("sse", "truth_sse", "cv_sse", "cv_truth_energy", "absolute_error_sum", "node_count", "hotspot_sse", "hotspot_count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def estimator(rows: list[dict[str, float]]) -> dict[str, float]:
    sums = {field: sum(float(row[field]) for row in rows) for field in SUM_FIELDS}
    return {
        "sample_first_relative_rmse_pct": float(np.mean([float(row["sample_first_relative_rmse_pct"]) for row in rows])),
        "point_global_relative_rmse_pct": float(100.0 * np.sqrt(sums["sse"] / sums["truth_sse"])),
        "cv_weighted_point_global_relative_rmse_pct": float(100.0 * np.sqrt(sums["cv_sse"] / sums["cv_truth_energy"])),
        "rmse_K": float(np.sqrt(sums["sse"] / sums["node_count"])),
        "mae_K": float(sums["absolute_error_sum"] / sums["node_count"]),
        "peak_temperature_mae_K": float(np.mean([float(row["peak_temperature_absolute_error_K"]) for row in rows])),
        "peak_temperature_rmse_K": float(np.sqrt(np.mean([float(row["peak_temperature_absolute_error_K"]) ** 2 for row in rows]))),
        "true_hotspot_region_rmse_K": float(np.sqrt(sums["hotspot_sse"] / sums["hotspot_count"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--common-eval", type=Path, required=True)
    parser.add_argument("--legacy-bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | str | int]] = []
    with args.per_case.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | str | int] = {
                "sample_id": row["sample_id"],
                "model": row["model"],
                "seed": int(row["seed"]),
            }
            for field in SUM_FIELDS + (
                "sample_first_relative_rmse_pct",
                "peak_temperature_absolute_error_K",
            ):
                parsed[field] = float(row[field])
            rows.append(parsed)
    if len(rows) != 128 * 6:
        raise ValueError(f"expected 768 per-case rows, got {len(rows)}")
    case_ids = sorted({str(row["sample_id"]) for row in rows})
    if len(case_ids) != 128:
        raise ValueError("expected 128 physical cases")
    by_model_seed_case = {
        (str(row["model"]), int(row["seed"]), str(row["sample_id"])): row for row in rows
    }
    models = ("Heat3D", "Therm-FM")
    seeds = (0, 1, 2)
    full_by_model_seed = {
        model: {seed: [by_model_seed_case[(model, seed, sid)] for sid in case_ids] for seed in seeds}
        for model in models
    }
    common = json.loads(args.common_eval.read_text(encoding="utf-8"))
    common_aggregate = common["aggregate"]
    observed_by_model = {
        model: {
            metric: float(np.mean([estimator(full_by_model_seed[model][seed])[metric] for seed in seeds]))
            for metric in METRICS
        }
        for model in models
    }
    main_table_by_model = {
        model: {metric: float(common_aggregate[model]["metrics_mean"][metric]) for metric in METRICS}
        for model in models
    }
    point_checks = {
        model: {metric: abs(observed_by_model[model][metric] - main_table_by_model[model][metric]) for metric in METRICS}
        for model in models
    }
    if max(max(values.values()) for values in point_checks.values()) > 1.0e-10:
        raise ValueError(f"corrected bootstrap point estimator disagrees with main table: {point_checks}")

    rng = np.random.default_rng(230023)
    draws = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
    bootstrap_values = {metric: [] for metric in METRICS}
    for draw in draws:
        draw_diff = {}
        for metric in METRICS:
            model_values = {}
            for model in models:
                seed_values = []
                for seed in seeds:
                    selected = [full_by_model_seed[model][seed][int(index)] for index in draw]
                    seed_values.append(estimator(selected)[metric])
                model_values[model] = float(np.mean(seed_values))
            draw_diff[metric] = model_values["Heat3D"] - model_values["Therm-FM"]
        for metric in METRICS:
            bootstrap_values[metric].append(draw_diff[metric])

    result = {
        "schema_version": "heat3d_v7_g2_p1i_corrected_paired_bootstrap_v1",
        "status": "COMPLETE_VALID_ONLY_CORRECTED_ESTIMAND",
        "comparison": "Heat3D - Therm-FM",
        "unit": "physical case resampled with replacement",
        "n_cases": 128,
        "n_training_seeds_per_model": 3,
        "resamples": 10000,
        "bootstrap_seed": 230023,
        "interval": "percentile 95 percent CI",
        "conditional_on": "frozen three-seed cohort",
        "seed_aggregation": "for each resample, compute the common-evaluator estimator separately per seed, then average the three seed estimates",
        "sufficient_statistics": list(SUM_FIELDS),
        "estimator_contract": {
            "point_global_relative_rmse_pct": "100*sqrt(sum SSE / sum truth_SSE)",
            "cv_weighted_point_global_relative_rmse_pct": "100*sqrt(sum CV_SSE / sum CV_truth_energy)",
            "rmse_K": "sqrt(sum SSE / sum node_count)",
            "mae_K": "sum absolute_error / sum node_count",
            "true_hotspot_region_rmse_K": "sqrt(sum hotspot_SSE / sum hotspot_count)",
            "sample_first_relative_rmse_pct": "mean of per-case CV-weighted relative RMSE",
            "peak_temperature_mae_K": "mean per-case absolute peak error",
            "peak_temperature_rmse_K": "sqrt(mean per-case peak error squared)",
        },
        "point_estimate_matches_main_table": point_checks,
        "main_table_difference": {
            metric: observed_by_model["Heat3D"][metric] - observed_by_model["Therm-FM"][metric]
            for metric in METRICS
        },
        "per_metric": {},
        "legacy_bootstrap_sha256": sha256_file(args.legacy_bootstrap),
        "legacy_bootstrap_status": "SUPERSEDED_ESTIMAND_AUDIT_ONLY",
        "per_case_artifact_sha256": sha256_file(args.per_case),
        "common_eval_sha256": sha256_file(args.common_eval),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    for metric, values in bootstrap_values.items():
        arr = np.asarray(values, dtype=np.float64)
        result["per_metric"][metric] = {
            "observed_mean_difference": float(result["main_table_difference"][metric]),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
