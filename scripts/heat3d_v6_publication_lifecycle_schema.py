#!/usr/bin/env python3
"""Shared lifecycle schema for the frozen V6 publication benchmark.

The schema deliberately makes mode-inapplicable measurements JSON ``null``.
It is imported by route workers, the orchestrator, the collector, and the
schema-only regression fixtures so those four layers cannot silently diverge.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
    "FVM240825_reference",
)
MODES = ("serial", "Q2")
SERIAL_FIELDS = ("cold", "fresh_Q1", "cache_hot", "resident")
Q2_FIELDS = (
    "submit_to_result", "inter_completion", "throughput_samples_per_second",
    "B16_to_B32_marginal_seconds",
)


def require(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def timing_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0, "empty timing series is forbidden")
    require(np.all(np.isfinite(array)), "nonfinite timing series")
    return {
        "count": int(array.size),
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "std_seconds": float(np.std(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _stat(value: Any, field: str) -> None:
    require(isinstance(value, dict), f"{field}: statistic must be an object")
    require(int(value.get("count", 0)) > 0, f"{field}: empty statistic")
    for key in ("median_seconds", "p95_seconds"):
        require(key in value and np.isfinite(float(value[key])), f"{field}: invalid {key}")


def serial_metrics(
    *, cold_seconds: float, fresh_q1: dict[str, Any],
    cache_hot: dict[str, Any], resident: dict[str, Any],
) -> dict[str, Any]:
    require(np.isfinite(float(cold_seconds)), "cold timing is nonfinite")
    result = {
        "cold": {"seconds": float(cold_seconds)},
        "fresh_Q1": fresh_q1,
        "cache_hot": cache_hot,
        "resident": resident,
        "submit_to_result": None,
        "inter_completion": None,
        "throughput_samples_per_second": None,
        "B16_to_B32_marginal_seconds": None,
    }
    validate_metrics("serial", result)
    return result


def q2_metrics(
    *, submit_to_result: dict[str, Any], inter_completion: dict[str, Any],
    throughput_samples_per_second: float, b16_to_b32_marginal_seconds: float,
) -> dict[str, Any]:
    result = {
        "cold": None,
        "fresh_Q1": None,
        "cache_hot": None,
        "resident": None,
        "submit_to_result": submit_to_result,
        "inter_completion": inter_completion,
        "throughput_samples_per_second": float(throughput_samples_per_second),
        "B16_to_B32_marginal_seconds": float(b16_to_b32_marginal_seconds),
    }
    validate_metrics("Q2", result)
    return result


def validate_metrics(mode: str, metrics: dict[str, Any]) -> None:
    require(mode in MODES, f"unknown lifecycle mode: {mode}")
    require(set(metrics) == set(SERIAL_FIELDS + Q2_FIELDS), "lifecycle metric key drift")
    if mode == "serial":
        require(isinstance(metrics["cold"], dict), "serial cold missing")
        require(np.isfinite(float(metrics["cold"]["seconds"])), "serial cold nonfinite")
        for field in ("fresh_Q1", "cache_hot", "resident"):
            _stat(metrics[field], field)
        for field in Q2_FIELDS:
            require(metrics[field] is None, f"serial field {field} must be null")
    else:
        for field in SERIAL_FIELDS:
            require(metrics[field] is None, f"Q2 field {field} must be null")
        _stat(metrics["submit_to_result"], "submit_to_result")
        _stat(metrics["inter_completion"], "inter_completion")
        require(
            np.isfinite(float(metrics["throughput_samples_per_second"]))
            and float(metrics["throughput_samples_per_second"]) > 0,
            "Q2 throughput invalid",
        )
        require(
            np.isfinite(float(metrics["B16_to_B32_marginal_seconds"]))
            and float(metrics["B16_to_B32_marginal_seconds"]) >= 0,
            "Q2 B16-to-B32 marginal invalid",
        )


def provenance(*, attempted: bool, matrix_completed: bool, generated: bool) -> dict[str, bool]:
    require(not generated or matrix_completed, "publication results require complete matrix")
    require(not matrix_completed or attempted, "complete matrix requires attempted measurement")
    return {
        "formal_measurement_attempted": bool(attempted),
        "formal_matrix_completed": bool(matrix_completed),
        "publication_results_generated": bool(generated),
    }


def validate_cell(row: dict[str, Any], *, formal: bool) -> None:
    require(row.get("route") in ROUTES, "cell route drift")
    mode = row.get("service_mode")
    require(mode in MODES, "cell mode drift")
    if formal:
        require(row.get("sample_count") == 32, "formal cell must contain 32 samples")
        require(row.get("status") == "passed", "formal full32 cell status must be passed")
    validate_metrics(mode, row["lifecycle_metrics"])
    expected = provenance(attempted=formal, matrix_completed=False, generated=False)
    require(row.get("measurement_provenance") == expected, "cell measurement provenance drift")


def has_empty_statistic(value: Any) -> bool:
    if isinstance(value, dict):
        if "count" in value and int(value["count"]) == 0:
            return True
        return any(has_empty_statistic(item) for item in value.values())
    if isinstance(value, list):
        return any(has_empty_statistic(item) for item in value)
    return False
