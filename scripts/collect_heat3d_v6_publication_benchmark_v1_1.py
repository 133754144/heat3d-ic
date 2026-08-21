#!/usr/bin/env python3
"""Pre-registered final collector for the V6 publication benchmark v1.1.

Fresh/Q1 uncertainty is a paired workload bootstrap over the same 32 sample
IDs within each route/seed lifecycle. The three randomized lifecycles are not
treated as a bootstrap population: their point estimates are reported as
median and min--max. Q2 and B16->B32 likewise use three-repeat median/range.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from heat3d_v6_publication_lifecycle_schema import validate_cell

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


def serial_samples(row: dict[str, Any]) -> list[dict[str, Any]]:
    route = row["route"]
    if route.startswith("E"):
        raw = row["serial_orders"][0]["rows"]
        return [{"sample_id": str(item["sample_id"]),
                 "fresh_seconds": float(item["elapsed_seconds"]),
                 "Q1_seconds": float(item["elapsed_seconds"])} for item in raw]
    if route.startswith("U"):
        return [{"sample_id": str(item["sample_id"]),
                 "fresh_seconds": float(item["stages"]["matched_continuous_e2e"]),
                 "Q1_seconds": float(item["streaming"]["submit_to_result_seconds"])}
                for item in row["samples"]]
    return [{"sample_id": str(item["sample_id"]),
             "fresh_seconds": float(item["submit_to_result_seconds"]),
             "Q1_seconds": float(item["submit_to_result_seconds"])}
            for item in row["rows"]]


def ordered_ids(row: dict[str, Any]) -> list[str]:
    value = row["ordered_sample_ids"]
    if isinstance(value, dict):
        return list(value[str(int(row["order_seed"]))])
    return list(value)


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    validate_cell(row, formal=True)
    route = row["route"]
    mode = row["service_mode"]
    lifecycle = row["lifecycle_metrics"]
    if route.startswith("E"):
        peak_vram = int(row["peak_vram_bytes"])
    elif route.startswith("U"):
        peak_vram = int(row["memory"]["peak_bytes_in_use"])
    else:
        peak_vram = None
    normalized: dict[str, Any] = {
        "route": route, "order_seed": int(row["order_seed"]), "service_mode": mode,
        "process_id": int(row["process_id"]),
        "peak_RAM_bytes": int(row["aggregate_service_worker_peak_RAM_bytes"]),
        "peak_VRAM_bytes": peak_vram,
    }
    if mode == "serial":
        samples = serial_samples(row)
        require(len(samples) == 32, f"{route}: serial workload must contain 32 samples")
        require([item["sample_id"] for item in samples] == ordered_ids(row),
                f"{route}: serial sample order drift")
        normalized.update(
            ordered_sample_ids=[item["sample_id"] for item in samples],
            paired_workload_samples=samples,
            cold_service_first_case_seconds=float(lifecycle["cold"]["seconds"]),
            fresh_distinct_case=_stats_pair(lifecycle["fresh_Q1"]),
            Q1_closed_loop=_stats_pair(lifecycle["fresh_Q1"]),
            repeat_case_cache_hot=_stats_pair(lifecycle["cache_hot"]),
            resident_core=_stats_pair(lifecycle["resident"]),
        )
    else:
        normalized.update(
            ordered_sample_ids=ordered_ids(row),
            Q2_submit_to_result=_stats_pair(lifecycle["submit_to_result"]),
            Q2_inter_completion=_stats_pair(lifecycle["inter_completion"]),
            Q2_samples_per_second=float(lifecycle["throughput_samples_per_second"]),
            true_B16_to_B32_marginal_seconds=float(lifecycle["B16_to_B32_marginal_seconds"]),
        )
    return normalized


def paired_workload_bootstrap(
    neural: dict[str, Any], reference: dict[str, Any], field: str,
) -> dict[str, Any]:
    n_rows = neural["paired_workload_samples"]
    f_rows = reference["paired_workload_samples"]
    n_ids = [row["sample_id"] for row in n_rows]
    f_ids = [row["sample_id"] for row in f_rows]
    require(n_ids == f_ids and len(n_ids) == 32, "paired workload IDs/order differ")
    ratios = np.asarray(
        [float(f[field]) / float(n[field]) for n, f in zip(n_rows, f_rows, strict=True)],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(ratios)) and np.all(ratios > 0), "invalid paired ratios")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, ratios.size, size=(BOOTSTRAP_RESAMPLES, ratios.size))
    draws = np.median(ratios[indices], axis=1)
    return {
        "sample_count": 32, "ordered_sample_ids": n_ids,
        "paired_sample_ratios": ratios.tolist(),
        "median_speedup": float(np.median(ratios)),
        "paired_workload_bootstrap_percentile_95pct": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "three_seed_bootstrap_used": False,
    }


def repeat_summary(values: list[float], unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (3,) and np.all(np.isfinite(array)), "three lifecycle repeats required")
    return {
        "repeat_count": 3, "unit": unit, "values": array.tolist(),
        "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()),
        "bootstrap_CI_claimed": False,
    }


def write_csv(path: Path, normalized: list[dict[str, Any]]) -> None:
    def statistic(row: dict[str, Any], key: str, field: str) -> Any:
        value = row.get(key)
        return None if value is None else value.get(field)

    fields = [
        "route", "order_seed", "service_mode", "cold_first_s", "fresh_median_s",
        "fresh_p95_s", "cache_hot_median_s", "resident_median_s", "Q1_median_s",
        "Q1_p95_s", "Q2_submit_median_s", "Q2_submit_p95_s", "Q2_inter_median_s",
        "Q2_samples_per_second", "B16_to_B32_marginal_s", "peak_VRAM_bytes", "peak_RAM_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in normalized:
            writer.writerow({
                "route": row["route"], "order_seed": row["order_seed"],
                "service_mode": row["service_mode"],
                "cold_first_s": row.get("cold_service_first_case_seconds"),
                "fresh_median_s": statistic(row, "fresh_distinct_case", "median_seconds"),
                "fresh_p95_s": statistic(row, "fresh_distinct_case", "p95_seconds"),
                "cache_hot_median_s": statistic(row, "repeat_case_cache_hot", "median_seconds"),
                "resident_median_s": statistic(row, "resident_core", "median_seconds"),
                "Q1_median_s": statistic(row, "Q1_closed_loop", "median_seconds"),
                "Q1_p95_s": statistic(row, "Q1_closed_loop", "p95_seconds"),
                "Q2_submit_median_s": statistic(row, "Q2_submit_to_result", "median_seconds"),
                "Q2_submit_p95_s": statistic(row, "Q2_submit_to_result", "p95_seconds"),
                "Q2_inter_median_s": statistic(row, "Q2_inter_completion", "median_seconds"),
                "Q2_samples_per_second": row.get("Q2_samples_per_second"),
                "B16_to_B32_marginal_s": row.get("true_B16_to_B32_marginal_seconds"),
                "peak_VRAM_bytes": row["peak_VRAM_bytes"], "peak_RAM_bytes": row["peak_RAM_bytes"],
            })


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# V6 authoritative valid32 publication timing", "",
        f"- Status: `{result['status']}`", "- Population: frozen valid32; test/sealed unopened.",
        "- Fresh/Q1 CI: paired 32-sample workload bootstrap within each seed; seed 20260821; 20,000 resamples.",
        "- Three lifecycle repeats and Q2/B16→B32: median and min–max only; no n=3 bootstrap CI.", "",
        "## Three-lifecycle summary", "", "| Route | Metric | Median | Min | Max |",
        "|---|---|---:|---:|---:|",
    ]
    for key, value in sorted(result["three_lifecycle_repeat_summary"].items()):
        route, metric = key.split(":", 1)
        lines.append(f"| {route} | {metric} | {value['median']:.6g} | {value['min']:.6g} | {value['max']:.6g} |")
    lines.extend(["", "## Paired speedups", "",
                  "Speedup is paired by sample ID and seed before any across-seed summary.", ""])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    require(raw["status"] == "passed", "authoritative raw matrix did not pass hard gates")
    require(raw.get("formal_measurement_attempted") is True, "formal measurement was not attempted")
    require(raw.get("formal_matrix_completed") is True, "formal matrix did not complete")
    require(raw.get("publication_results_generated") is False,
            "publication result provenance must remain false before collector")
    rows = raw["rows"]
    require(len(rows) == 30, "formal matrix must contain 30 independent lifecycle rows")
    normalized = [normalize(row) for row in rows]
    keys = {(row["route"], row["order_seed"], row["service_mode"]) for row in normalized}
    require(len(keys) == 30, "route/seed/mode lifecycle duplication")
    lookup = {(row["route"], row["order_seed"], row["service_mode"]): row for row in normalized}

    paired: dict[str, Any] = {}
    lifecycle: dict[str, Any] = {}
    for route in ROUTES:
        for metric, mode, accessor, unit in (
            ("fresh", "serial", lambda row: row["fresh_distinct_case"]["median_seconds"], "seconds"),
            ("Q1", "serial", lambda row: row["Q1_closed_loop"]["median_seconds"], "seconds"),
            ("Q2_submit", "Q2", lambda row: row["Q2_submit_to_result"]["median_seconds"], "seconds"),
            ("Q2_throughput", "Q2", lambda row: row["Q2_samples_per_second"], "samples_per_second"),
            ("B16_to_B32_marginal", "Q2", lambda row: row["true_B16_to_B32_marginal_seconds"], "seconds"),
        ):
            lifecycle[f"{route}:{metric}"] = repeat_summary(
                [float(accessor(lookup[(route, seed, mode)])) for seed in SEEDS], unit)

    for route in ROUTES[:-1]:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            neural = lookup[(route, seed, "serial")]
            reference = lookup[("FVM240825_reference", seed, "serial")]
            per_seed[str(seed)] = {
                "fresh": paired_workload_bootstrap(neural, reference, "fresh_seconds"),
                "Q1": paired_workload_bootstrap(neural, reference, "Q1_seconds"),
            }
        paired[route] = {
            "per_seed_paired_workload_bootstrap": per_seed,
            "fresh_three_lifecycle_median_range": repeat_summary(
                [per_seed[str(seed)]["fresh"]["median_speedup"] for seed in SEEDS], "speedup"),
            "Q1_three_lifecycle_median_range": repeat_summary(
                [per_seed[str(seed)]["Q1"]["median_speedup"] for seed in SEEDS], "speedup"),
            "Q2_throughput_three_lifecycle_median_range": repeat_summary([
                lookup[(route, seed, "Q2")]["Q2_samples_per_second"] /
                lookup[("FVM240825_reference", seed, "Q2")]["Q2_samples_per_second"]
                for seed in SEEDS], "speedup"),
            "B16_to_B32_three_lifecycle_median_range": repeat_summary([
                lookup[("FVM240825_reference", seed, "Q2")]["true_B16_to_B32_marginal_seconds"] /
                lookup[(route, seed, "Q2")]["true_B16_to_B32_marginal_seconds"]
                for seed in SEEDS], "speedup"),
        }

    result = {
        "schema_version": "heat3d_v6_publication_benchmark_final_collector_v1_2",
        "status": "collected_authoritative_valid32_without_pooled_96",
        "publication_timing_freeze": "GO",
        "formal_measurement_attempted": True,
        "formal_matrix_completed": True,
        "publication_results_generated": True,
        "route_seed_statistics": normalized,
        "three_lifecycle_repeat_summary": lifecycle,
        "paired_speedups": paired,
        "aggregation_contract": {
            "within_route_seed": "median_and_p95",
            "fresh_Q1_CI": "paired_32_sample_workload_bootstrap_within_each_seed",
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "three_lifecycle_repeats": "median_and_min_max_only",
            "Q2_and_B16_to_B32": "three_repeat_median_and_range_no_bootstrap_CI",
            "speedup_pairing": "same_sample_id_then_same_seed_before_three_repeat_summary",
            "pooled_96_ratio_used": False, "post_hoc_aggregation_change": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, normalized)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_md(args.output_md, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
