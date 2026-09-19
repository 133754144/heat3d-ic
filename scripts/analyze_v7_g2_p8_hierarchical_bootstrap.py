#!/usr/bin/env python3
"""Run the frozen P8 seed+case hierarchical bootstrap on native1024 rows.

This is a read-only analysis of an already materialized per-case artifact.  It
does not call a model or the evaluator and therefore cannot change a frozen
checkpoint or metric contract.
"""

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
SUM_FIELDS = (
    "sse",
    "truth_sse",
    "cv_sse",
    "cv_truth_energy",
    "absolute_error_sum",
    "node_count",
    "hotspot_sse",
    "hotspot_count",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def estimate(rows: list[dict[str, float]]) -> dict[str, float]:
    sums = {name: sum(float(row[name]) for row in rows) for name in SUM_FIELDS}
    return {
        "sample_first_relative_rmse_pct": float(np.mean([float(r["sample_first_relative_rmse_pct"]) for r in rows])),
        "point_global_relative_rmse_pct": float(100.0 * np.sqrt(sums["sse"] / sums["truth_sse"])),
        "cv_weighted_point_global_relative_rmse_pct": float(100.0 * np.sqrt(sums["cv_sse"] / sums["cv_truth_energy"])),
        "rmse_K": float(np.sqrt(sums["sse"] / sums["node_count"])),
        "mae_K": float(sums["absolute_error_sum"] / sums["node_count"]),
        "peak_temperature_mae_K": float(np.mean([float(r["peak_temperature_absolute_error_K"]) for r in rows])),
        "peak_temperature_rmse_K": float(np.sqrt(np.mean([float(r["peak_temperature_absolute_error_K"]) ** 2 for r in rows]))),
        "true_hotspot_region_rmse_K": float(np.sqrt(sums["hotspot_sse"] / sums["hotspot_count"])),
    }


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, object] = {"sample_id": raw["sample_id"], "model": raw["model"], "seed": int(raw["seed"])}
            for name in SUM_FIELDS + ("sample_first_relative_rmse_pct", "peak_temperature_absolute_error_K"):
                row[name] = float(raw[name])
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", type=Path, required=True)
    ap.add_argument("--main-table", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = read_rows(args.per_case)
    case_ids = sorted({str(r["sample_id"]) for r in rows})
    seeds = (0, 1, 2)
    if len(case_ids) != 128:
        raise ValueError(f"expected 128 cases, got {len(case_ids)}")
    models = ("Heat3D", "GINO", "Transolver")
    expected = {(m, s, sid) for m in models for s in seeds for sid in case_ids}
    observed = {(str(r["model"]), int(r["seed"]), str(r["sample_id"])) for r in rows}
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(f"per-case artifact identity mismatch; missing={missing}, extra={extra}")
    by = {(str(r["model"]), int(r["seed"]), str(r["sample_id"])): r for r in rows}
    table = json.loads(args.main_table.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    frozen = prereg["frozen_statistics"]["bootstrap"]
    if int(frozen["replicates"]) != 10000 or int(frozen["random_seed"]) != 20260907:
        raise ValueError("P8 preregistration does not match the requested frozen bootstrap")

    # Verify the non-resampled point estimate against the frozen aggregate.
    seed_estimates = {
        m: {s: estimate([by[(m, s, sid)] for sid in case_ids]) for s in seeds} for m in models
    }
    point = {m: {metric: float(np.mean([seed_estimates[m][s][metric] for s in seeds])) for metric in METRICS} for m in models}
    point_checks = {}
    for m in models:
        frozen_metrics = table["aggregate"][m]["metrics_mean"]
        point_checks[m] = {metric: abs(point[m][metric] - float(frozen_metrics[metric])) for metric in METRICS}
    if max(max(v.values()) for v in point_checks.values()) > 1e-10:
        raise ValueError(f"point estimate disagrees with frozen main table: {point_checks}")

    comparisons = (("Heat3D", "GINO"), ("Heat3D", "Transolver"))
    rng = np.random.default_rng(20260907)
    n_boot = 10000
    boot: dict[str, dict[str, list[float]]] = {
        f"{a}_minus_{b}": {metric: [] for metric in METRICS} for a, b in comparisons
    }
    # P8 hierarchy: first draw three seed slots with replacement, then draw
    # 128 paired physical cases within each selected seed slot.
    for _ in range(n_boot):
        sampled_seed_slots = rng.integers(0, len(seeds), size=len(seeds))
        for a, b in comparisons:
            key = f"{a}_minus_{b}"
            per_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
            for slot in sampled_seed_slots:
                seed = seeds[int(slot)]
                case_draw = rng.integers(0, len(case_ids), size=len(case_ids))
                chosen_ids = [case_ids[int(i)] for i in case_draw]
                ea = estimate([by[(a, seed, sid)] for sid in chosen_ids])
                eb = estimate([by[(b, seed, sid)] for sid in chosen_ids])
                for metric in METRICS:
                    per_metric[metric].append(ea[metric] - eb[metric])
            for metric in METRICS:
                boot[key][metric].append(float(np.mean(per_metric[metric])))

    result = {
        "schema_version": "heat3d_v7_g2_p8_hierarchical_seed_case_bootstrap_v1",
        "status": "COMPLETE_VALID_ONLY",
        "protocol": "docs/v7_g2_p8_common_task_comparison_hierarchy.json",
        "protocol_sha256": sha256_file(args.prereg),
        "estimand": "resample training-seed slots, then paired valid_iid cases within each selected seed; recompute full estimator per seed and average seed-level differences",
        "comparisons": [f"{a} - {b}" for a, b in comparisons],
        "metrics": list(METRICS),
        "n_cases": len(case_ids),
        "n_training_seeds": len(seeds),
        "resamples": n_boot,
        "bootstrap_seed": 20260907,
        "interval": "percentile 95% CI",
        "point_estimate_matches_main_table": point_checks,
        "point_estimate": {f"{a}_minus_{b}": {metric: point[a][metric] - point[b][metric] for metric in METRICS} for a, b in comparisons},
        "per_comparison": {},
        "source_per_case_sha256": sha256_file(args.per_case),
        "source_main_table_sha256": sha256_file(args.main_table),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    for a, b in comparisons:
        key = f"{a}_minus_{b}"
        result["per_comparison"][key] = {
            metric: {"estimate": result["point_estimate"][key][metric], "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}
            for metric, values in boot[key].items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
