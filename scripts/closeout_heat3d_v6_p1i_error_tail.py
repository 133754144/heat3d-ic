#!/usr/bin/env python3
"""Build the V6/P1i peak-error tail closeout from frozen artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
FORMAL_SAMPLES = CFG / "v6_p1i_formal1024_v1_samples.csv"
FORMAL_AUDIT = CFG / "v6_p1i_formal1024_v1_distribution_audit.json"
TEST_RESULT = CFG / "v6_p1i_e16384_test_iid_confirmatory.json"
VALID_RESULT = (
    CFG
    / "v6_p1i_post_freeze_raw/performance/"
    "seed0_E16384_reconstruction_order20260814.json"
)
VALID_REPLAYS = [
    CFG
    / "v6_p1i_post_freeze_raw/performance/"
    f"seed0_E16384_reconstruction_order{seed}.json"
    for seed in (20260813, 20260814, 20260815)
]
OUTPUT_JSON = CFG / "v6_p1i_error_tail_closeout.json"
OUTPUT_CSV = CFG / "v6_p1i_error_tail_samples.csv"
FROZEN_TEMPERATURE_SCALE_K = 180.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    if spearman:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def quartile_summary(
    values: np.ndarray,
    absolute_error: np.ndarray,
    relative_error_pct: np.ndarray,
    cutpoints: np.ndarray,
) -> list[dict[str, float | int | str]]:
    bins = np.searchsorted(cutpoints, values, side="right")
    rows: list[dict[str, float | int | str]] = []
    for index in range(4):
        selected = bins == index
        err = absolute_error[selected]
        rel = relative_error_pct[selected]
        rows.append(
            {
                "quartile": f"Q{index + 1}",
                "count": int(np.sum(selected)),
                "peak_error_rmse_K": float(np.sqrt(np.mean(np.square(err)))),
                "peak_error_abs_median_K": float(np.median(err)),
                "sample_peak_relative_error_median_pct": float(np.median(rel)),
                "sample_peak_relative_error_p95_pct": float(np.quantile(rel, 0.95)),
            }
        )
    return rows


def population_payload(
    name: str,
    rows: list[dict[str, float | str]],
    formal_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    errors = np.asarray([float(row["absolute_peak_error_K"]) for row in rows])
    true_peak = np.asarray([float(row["true_peak_deltaT_K"]) for row in rows])
    power = np.asarray([float(row["total_power_W"]) for row in rows])
    severity = np.asarray([float(row["continuous_severity"]) for row in rows])
    relative = 100.0 * errors / true_peak
    sse = np.square(errors)
    descending = np.argsort(sse, kind="mergesort")[::-1]

    for rank, index in enumerate(descending, start=1):
        rows[int(index)]["sample_peak_relative_error_pct"] = float(relative[index])
        rows[int(index)]["peak_error_sse_K2"] = float(sse[index])
        rows[int(index)]["peak_error_sse_rank"] = rank
        rows[int(index)]["is_top5_sse"] = rank <= 5
        rows[int(index)]["is_top10_sse"] = rank <= 10

    relationships: dict[str, object] = {}
    for key, values in (
        ("true_peak_deltaT_K", true_peak),
        ("total_power_W", power),
        ("continuous_severity", severity),
    ):
        relationships[key] = {
            "absolute_error_pearson": correlation(values, errors),
            "absolute_error_spearman": correlation(values, errors, spearman=True),
            "relative_error_pearson": correlation(values, relative),
            "relative_error_spearman": correlation(values, relative, spearman=True),
            "formal_dataset_quartile_cutpoints": [
                float(value) for value in formal_arrays[key]
            ],
            "quartiles": quartile_summary(
                values, errors, relative, formal_arrays[key]
            ),
        }

    top_rows = []
    for index in descending[:10]:
        row = rows[int(index)]
        top_rows.append(
            {
                "rank": int(row["peak_error_sse_rank"]),
                "sample_id": row["sample_id"],
                "absolute_peak_error_K": row["absolute_peak_error_K"],
                "sample_peak_relative_error_pct": row[
                    "sample_peak_relative_error_pct"
                ],
                "true_peak_deltaT_K": row["true_peak_deltaT_K"],
                "total_power_W": row["total_power_W"],
                "continuous_severity": row["continuous_severity"],
            }
        )

    return {
        "population": name,
        "sample_count": len(rows),
        "peak_rmse_K": float(np.sqrt(np.mean(sse))),
        "peak_rmse_over_frozen_180K_pct": float(
            100.0 * np.sqrt(np.mean(sse)) / FROZEN_TEMPERATURE_SCALE_K
        ),
        "sample_peak_relative_error_pct": quantiles(relative),
        "absolute_peak_error_K": quantiles(errors),
        "peak_error_sse_contribution_pct": {
            "top5": float(100.0 * np.sum(sse[descending[:5]]) / np.sum(sse)),
            "top10": float(100.0 * np.sum(sse[descending[:10]]) / np.sum(sse)),
        },
        "trimmed_peak_rmse_K": {
            "excluding_top5": float(np.sqrt(np.mean(sse[descending[5:]]))),
            "excluding_top10": float(np.sqrt(np.mean(sse[descending[10:]]))),
        },
        "relationships": relationships,
        "top10_samples": top_rows,
    }


def main() -> int:
    with FORMAL_SAMPLES.open(newline="") as handle:
        formal_rows = list(csv.DictReader(handle))
    if len(formal_rows) != 1024:
        raise RuntimeError(f"expected 1024 formal rows, found {len(formal_rows)}")
    metadata = {row["sample_id"]: row for row in formal_rows}
    if len(metadata) != 1024:
        raise RuntimeError("duplicate formal sample IDs")

    formal_peak = np.asarray([float(row["peak_deltaT_K"]) for row in formal_rows])
    formal_power = np.asarray(
        [float(row["package_total_power_W"]) for row in formal_rows]
    )
    formal_severity = np.asarray(
        [float(row["continuous_severity"]) for row in formal_rows]
    )
    cutpoints = {
        "true_peak_deltaT_K": np.quantile(formal_peak, [0.25, 0.50, 0.75]),
        "total_power_W": np.quantile(formal_power, [0.25, 0.50, 0.75]),
        "continuous_severity": np.quantile(formal_severity, [0.25, 0.50, 0.75]),
    }

    test_payload = json.loads(TEST_RESULT.read_text())
    if (
        test_payload["status"] != "passed_frozen_test_iid_confirmatory"
        or test_payload["sample_count"] != 128
    ):
        raise RuntimeError("frozen test result is incomplete")
    if test_payload["route"] != "E16384_reconstruction":
        raise RuntimeError("unexpected test route")
    if test_payload["role_contract"]["accessed_roles"] != [
        "train_inputs_for_frozen_standardizer",
        "test_iid",
    ]:
        raise RuntimeError("test role provenance mismatch")

    valid_payload = json.loads(VALID_RESULT.read_text())
    if valid_payload["status"] != "passed" or valid_payload["sample_count"] != 32:
        raise RuntimeError("frozen valid32 result is incomplete")
    if valid_payload["route"]["route"] != "E16384_reconstruction":
        raise RuntimeError("unexpected valid route")

    valid_rows: list[dict[str, float | str]] = []
    for sample in valid_payload["samples"]:
        sample_id = sample["sample_id"]
        meta = metadata[sample_id]
        if meta["split_role"] != "valid_iid":
            raise RuntimeError(f"non-valid row in valid32: {sample_id}")
        valid_rows.append(
            {
                "population": "valid32",
                "sample_id": sample_id,
                "split_role": meta["split_role"],
                "true_peak_deltaT_K": float(meta["peak_deltaT_K"]),
                "total_power_W": float(meta["package_total_power_W"]),
                "continuous_severity": float(meta["continuous_severity"]),
                "absolute_peak_error_K": float(
                    sample["full_field_metrics"]["peak_rmse_K"]
                ),
            }
        )

    test_rows: list[dict[str, float | str]] = []
    for sample in test_payload["per_sample_metrics"]:
        sample_id = sample["sample_id"]
        meta = metadata[sample_id]
        if meta["split_role"] != "test_iid":
            raise RuntimeError(f"non-test row in test128: {sample_id}")
        test_rows.append(
            {
                "population": "test128",
                "sample_id": sample_id,
                "split_role": meta["split_role"],
                "true_peak_deltaT_K": float(meta["peak_deltaT_K"]),
                "total_power_W": float(meta["package_total_power_W"]),
                "continuous_severity": float(meta["continuous_severity"]),
                "absolute_peak_error_K": float(sample["full_field"]["peak_rmse_K"]),
            }
        )

    valid = population_payload("valid32", valid_rows, cutpoints)
    test = population_payload("test128", test_rows, cutpoints)
    valid_mse = float(valid["peak_rmse_K"]) ** 2
    test_errors = np.asarray(
        [float(row["absolute_peak_error_K"]) for row in test_rows]
    )
    test_sse = np.square(test_errors)
    descending = np.argsort(test_sse, kind="mergesort")[::-1]
    excess_sse = float(np.sum(test_sse) - len(test_sse) * valid_mse)
    excess_attribution: dict[str, float] = {}
    for count in (5, 10):
        contribution = float(np.sum(test_sse[descending[:count]]) - count * valid_mse)
        excess_attribution[f"top{count}"] = 100.0 * contribution / excess_sse

    replay_errors = []
    reference_by_id = {
        row["sample_id"]: float(row["full_field_metrics"]["peak_rmse_K"])
        for row in valid_payload["samples"]
    }
    for path in VALID_REPLAYS:
        replay = json.loads(path.read_text())
        by_id = {
            row["sample_id"]: float(row["full_field_metrics"]["peak_rmse_K"])
            for row in replay["samples"]
        }
        if set(by_id) != set(reference_by_id):
            raise RuntimeError(f"valid32 population drift: {path}")
        replay_errors.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "peak_rmse_K": float(
                    math.sqrt(sum(value * value for value in by_id.values()) / len(by_id))
                ),
                "max_per_sample_abs_drift_from_order20260814_K": max(
                    abs(by_id[key] - reference_by_id[key]) for key in by_id
                ),
            }
        )

    audit = json.loads(FORMAL_AUDIT.read_text())
    observed_max = float(np.max(formal_peak))
    audit_max = float(audit["temperature_coverage"]["summary"]["maximum"])
    if observed_max != audit_max:
        raise RuntimeError("formal truth/QC maximum mismatch")

    output = {
        "schema_version": "heat3d_v6_p1i_error_tail_closeout_v1",
        "status": "passed_offline_frozen_artifact_analysis",
        "scientific_development_status": "CLOSED",
        "normalization_contract": {
            "frozen_temperature_scale_K": FROZEN_TEMPERATURE_SCALE_K,
            "formal_dataset_observed_max_peak_deltaT_K": observed_max,
            "formal_dataset_observed_max_sample_id": formal_rows[
                int(np.argmax(formal_peak))
            ]["sample_id"],
            "split_specific_max_used_for_primary_normalization": False,
        },
        "populations": {"valid32": valid, "test128": test},
        "test_rise_attribution": {
            "valid32_mean_peak_error_sse_K2": valid_mse,
            "test128_mean_peak_error_sse_K2": float(np.mean(test_sse)),
            "test_over_valid_peak_rmse_ratio": float(test["peak_rmse_K"])
            / float(valid["peak_rmse_K"]),
            "test_over_valid_absolute_error_median_ratio": float(
                test["absolute_peak_error_K"]["median"]
            )
            / float(valid["absolute_peak_error_K"]["median"]),
            "test_excluding_top10_peak_rmse_K": float(
                test["trimmed_peak_rmse_K"]["excluding_top10"]
            ),
            "top5_share_of_test_excess_sse_vs_valid32_mean_pct": excess_attribution[
                "top5"
            ],
            "top10_share_of_test_excess_sse_vs_valid32_mean_pct": excess_attribution[
                "top10"
            ],
            "classification": "primarily_tail_driven_with_modest_broad_shift",
            "interpretation_boundary": (
                "descriptive comparison against the frozen valid32 mean SSE; "
                "not selection, tuning, or a population-significance test"
            ),
        },
        "provenance": {
            "formal_samples": {
                "path": str(FORMAL_SAMPLES.relative_to(ROOT)),
                "sha256": sha256(FORMAL_SAMPLES),
                "sample_count": 1024,
            },
            "formal_distribution_audit": {
                "path": str(FORMAL_AUDIT.relative_to(ROOT)),
                "sha256": sha256(FORMAL_AUDIT),
            },
            "test_result": {
                "path": str(TEST_RESULT.relative_to(ROOT)),
                "sha256": sha256(TEST_RESULT),
                "checkpoint_file_sha256": test_payload["checkpoint_file_sha256"],
                "prediction_stream_sha256": test_payload["prediction_stream_sha256"],
            },
            "valid32_result": {
                "path": str(VALID_RESULT.relative_to(ROOT)),
                "sha256": sha256(VALID_RESULT),
                "checkpoint_sha256": valid_payload["checkpoint_sha256"],
                "replay_numeric_diagnostics": replay_errors,
            },
        },
        "role_contract": {
            "training": False,
            "inference": False,
            "valid_iid_existing_artifact_only": True,
            "test_iid_existing_artifact_only": True,
            "sealed_iid_opened": False,
            "used_for_selection_or_tuning": False,
            "interpretation": "applicability_boundary_only",
        },
        "integrity": {
            "evaluator_error_found": False,
            "data_error_found": False,
            "provenance_error_found": False,
        },
    }

    all_rows = valid_rows + test_rows
    fields = [
        "population",
        "sample_id",
        "split_role",
        "true_peak_deltaT_K",
        "total_power_W",
        "continuous_severity",
        "absolute_peak_error_K",
        "sample_peak_relative_error_pct",
        "peak_error_sse_K2",
        "peak_error_sse_rank",
        "is_top5_sse",
        "is_top10_sse",
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    output["sample_table"] = {
        "path": str(OUTPUT_CSV.relative_to(ROOT)),
        "sha256": sha256(OUTPUT_CSV),
        "row_count": len(all_rows),
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "valid_peak_rmse_K": valid["peak_rmse_K"],
                "test_peak_rmse_K": test["peak_rmse_K"],
                "test_top10_excess_sse_pct": excess_attribution["top10"],
                "sealed": False,
                "inference": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
