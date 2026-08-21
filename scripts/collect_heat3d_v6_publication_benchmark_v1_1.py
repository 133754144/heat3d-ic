#!/usr/bin/env python3
"""Pre-registered final collector for V6 publication benchmark v1.1.

The collector never pools the 96 route observations.  It first computes each
route/seed statistic, then forms same-seed FVM/neural ratios, and only then
summarizes the three paired ratios.  Bootstrap randomness is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROUTES = (
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control", "FVM240825_reference",
)
SEEDS = (20260814, 20260815, 20260816)
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_RESAMPLES = 20000


def require(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.all(np.isfinite(array)), "empty/nonfinite series")
    return {
        "count": int(array.size),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _stats_pair(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "missing frozen statistic")
    return {
        "median_seconds": float(value["median_seconds"]),
        "p95_seconds": float(value["p95_seconds"]),
    }


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    route = row["route"]
    mode = row["service_mode"]
    normalized: dict[str, Any] = {
        "route": route, "order_seed": int(row["order_seed"]), "service_mode": mode,
    }
    if route.startswith("E"):
        if mode == "serial":
            order = row["serial_orders"][0]
            normalized.update(
                cold_service_first_case_seconds=float(order["rows"][0]["elapsed_seconds"]),
                fresh_distinct_case=_stats_pair(order["fresh"]),
                Q1_closed_loop=_stats_pair(order["Q1_closed_loop"]),
                repeat_case_cache_hot=_stats_pair(row["repeat_case_cache_hot"]),
                resident_core=_stats_pair(row["resident_core"]),
            )
        else:
            q2 = row["Q2_orders"][0]
            normalized.update(
                Q2_submit_to_result=_stats_pair(q2["submit_to_result"]),
                Q2_inter_completion=_stats_pair(q2["inter_completion"]),
                Q2_samples_per_second=float(q2["samples_per_second"]),
                true_B16_to_B32_marginal_seconds=float(q2["true_B16_to_B32_marginal_seconds"]),
            )
    elif route.startswith("U"):
        if mode == "serial":
            fresh = row["runtime"]["fresh_sample"]["matched_continuous_e2e"]
            normalized.update(
                cold_service_first_case_seconds=float(row["samples"][0]["stages"]["matched_continuous_e2e"]),
                fresh_distinct_case=_stats_pair(fresh),
                Q1_closed_loop=_stats_pair(row["streaming"]["submit_to_result"]),
                repeat_case_cache_hot=_stats_pair(row["runtime"]["repeat_case_cache_hot"]),
                resident_core=_stats_pair(row["runtime"]["resident_core"]),
            )
        else:
            q2 = row["true_concurrent_streaming"]
            normalized.update(
                Q2_submit_to_result=_stats_pair(q2["submit_to_result"]),
                Q2_inter_completion=_stats_pair(q2["inter_completion"]),
                Q2_samples_per_second=float(q2["samples_per_second"]),
                true_B16_to_B32_marginal_seconds=float(q2["actual_B16_to_B32_marginal_seconds"]),
            )
    else:
        if mode == "serial":
            normalized.update(
                cold_service_first_case_seconds=float(row["rows"][0]["submit_to_result_seconds"]),
                fresh_distinct_case=summary([x["submit_to_result_seconds"] for x in row["rows"]]),
                Q1_closed_loop=summary([x["submit_to_result_seconds"] for x in row["rows"]]),
                repeat_case_cache_hot=_stats_pair(row["repeat_case_cache_hot"]),
                resident_core=_stats_pair(row["resident_core"]),
            )
        else:
            q2 = row["Q2"]
            normalized.update(
                Q2_submit_to_result=_stats_pair(q2["submit_to_result"]),
                Q2_inter_completion=_stats_pair(q2["inter_completion"]),
                Q2_samples_per_second=float(q2["samples_per_second"]),
                true_B16_to_B32_marginal_seconds=float(q2["true_B16_to_B32_marginal_seconds"]),
            )
    return normalized


def paired_bootstrap(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (3,), "paired bootstrap requires exactly three seed ratios")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.median(array[rng.integers(0, 3, size=(BOOTSTRAP_RESAMPLES, 3))], axis=1)
    return {
        "paired_seed_ratios": array.tolist(),
        "median": float(np.median(array)),
        "bootstrap_percentile_95pct": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    rows = raw["rows"]
    require(len(rows) == 30, "formal matrix must contain 30 independent lifecycle rows")
    normalized = [normalize(row) for row in rows]
    keys = {(row["route"], row["order_seed"], row["service_mode"]) for row in normalized}
    require(len(keys) == 30, "route/seed/mode lifecycle duplication")
    paired: dict[str, Any] = {}
    fvm = {(row["order_seed"], row["service_mode"]): row for row in normalized
           if row["route"] == "FVM240825_reference"}
    for route in ROUTES[:-1]:
        for metric, mode, direction in (
            ("fresh_distinct_case", "serial", "latency"),
            ("Q1_closed_loop", "serial", "latency"),
            ("Q2_submit_to_result", "Q2", "latency"),
            ("Q2_samples_per_second", "Q2", "throughput"),
        ):
            ratios = []
            for seed in SEEDS:
                neural = next(row for row in normalized if row["route"] == route
                              and row["order_seed"] == seed and row["service_mode"] == mode)
                reference = fvm[(seed, mode)]
                n = (float(neural[metric]["median_seconds"])
                     if isinstance(neural[metric], dict) else float(neural[metric]))
                f = (float(reference[metric]["median_seconds"])
                     if isinstance(reference[metric], dict) else float(reference[metric]))
                ratios.append(f / n if direction == "latency" else n / f)
            paired[f"{route}:{metric}"] = paired_bootstrap(ratios)
    result = {
        "schema_version": "heat3d_v6_publication_benchmark_final_collector_v1_1",
        "status": "collected_without_pooled_96",
        "route_seed_statistics": normalized,
        "paired_speedups": paired,
        "aggregation_contract": {
            "within_route_seed": "median_and_p95",
            "pairing": "same_seed_before_ratio",
            "across_seed": "median_of_three_ratios",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "pooled_96_ratio_used": False
        }
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
