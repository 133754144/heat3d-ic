#!/usr/bin/env python3
"""Zero-solve postmortem and target-independent split comparison for V6-P1i."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import heat3d_v6_p1i_continuous_core as core
import heat3d_v6_p1i_split as splitlib


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
DOCS_DIR = ROOT / "docs"
CONFIG = CONFIG_DIR / "v6_p1i_formal1024_v0.yaml"
SAMPLES = CONFIG_DIR / "v6_p1i_formal1024_v0_samples.csv"
REGIONS = CONFIG_DIR / "v6_p1i_formal1024_v0_regions.csv"
PILOT_CONFIG = CONFIG_DIR / "v6_p1i_pilot128_v2.yaml"
PILOT_SAMPLES = CONFIG_DIR / "v6_p1i_pilot128_v2_samples.csv"
OUTPUT = CONFIG_DIR / "v6_p1i_formal1024_v0_postmortem.json"
SPLIT_CSV = CONFIG_DIR / "v6_p1i_split_candidate_comparison.csv"
SPLIT_JSON = CONFIG_DIR / "v6_p1i_split_candidate_assignments.json"
GAP_CSV = CONFIG_DIR / "v6_p1i_formal1024_v0_gap_samples.csv"
REPORT = DOCS_DIR / "v6_p1i_formal1024_v0_postmortem.md"
FIGURE = DOCS_DIR / "v6_p1i_formal1024_v0_postmortem.png"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std <= 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x, y = map(_standardize, (left, right))
    return float(np.mean(x * y))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort")
    y = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort")
    return _pearson(x, y)


def _partial_correlation(
    left: np.ndarray, right: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    x, y, z = map(_standardize, (left, right, control))
    x_residual = x - float(np.mean(x * z)) * z
    y_residual = y - float(np.mean(y * z)) * z
    rank_x = _standardize(np.argsort(np.argsort(x, kind="mergesort")))
    rank_y = _standardize(np.argsort(np.argsort(y, kind="mergesort")))
    rank_z = _standardize(np.argsort(np.argsort(z, kind="mergesort")))
    rank_x_residual = rank_x - float(np.mean(rank_x * rank_z)) * rank_z
    rank_y_residual = rank_y - float(np.mean(rank_y * rank_z)) * rank_z
    return {
        "pearson_partial_r": _pearson(x_residual, y_residual),
        "spearman_partial_rho": _pearson(
            rank_x_residual, rank_y_residual
        ),
    }


def _single_feature_r2(feature: np.ndarray, target: np.ndarray) -> float:
    correlation = _pearson(feature, target)
    return correlation * correlation


def _gap_postmortem(
    sample_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(sample_rows, key=lambda row: float(row["peak_deltaT_K"]))
    candidates = [
        (
            float(right["peak_deltaT_K"]) - float(left["peak_deltaT_K"]),
            index,
            left,
            right,
        )
        for index, (left, right) in enumerate(zip(ordered, ordered[1:]))
    ]
    gap, index, left, right = max(candidates, key=lambda row: row[0])
    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in ordered])
    midpoint = 0.5 * (
        float(left["peak_deltaT_K"]) + float(right["peak_deltaT_K"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kde = gaussian_kde(peaks)
        density_midpoint = float(kde([midpoint])[0])
        grid = np.linspace(float(peaks.min()), float(peaks.max()), 2048)
        density = kde(grid)
    quantile_left = (index + 1) / len(peaks)
    region = (
        "extreme_upper_tail"
        if quantile_left >= 0.95
        else "lower_tail"
        if quantile_left <= 0.05
        else "distribution_core_or_modal_valley"
    )
    gap_rows = []
    for side, row in (("left", left), ("right", right)):
        gap_rows.append(
            {
                "side": side,
                **{
                    key: row[key]
                    for key in (
                        "sample_id",
                        "split_role",
                        "peak_deltaT_K",
                        "mean_deltaT_K",
                        "cv_rms_deltaT_K",
                        "package_total_power_W",
                        "continuous_severity",
                        "top_h_W_m2K",
                        "bottom_h_W_m2K",
                        "total_source_volume_m3",
                        "mean_q_W_m3",
                        "max_q_W_m3",
                        "mean_local_k_W_mK",
                        "source_count",
                        "k_region_count",
                    )
                },
            }
        )
    return (
        {
            "gap_K": gap,
            "midpoint_K": midpoint,
            "left_rank": index,
            "left_empirical_quantile": quantile_left,
            "right_empirical_quantile": (index + 2) / len(peaks),
            "classification": region,
            "kde_density_at_midpoint": density_midpoint,
            "kde_density_relative_to_global_max": density_midpoint
            / float(np.max(density)),
            "samples": gap_rows,
        },
        gap_rows,
    )


def _response_postmortem(
    sample_rows: Sequence[Mapping[str, str]],
    design_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in design_records}
    peak = np.asarray([float(row["peak_deltaT_K"]) for row in sample_rows])
    power = np.asarray(
        [float(row["package_total_power_W"]) for row in sample_rows]
    )
    reff = peak / power
    features = {
        name: np.asarray(
            [float(by_id[row["sample_id"]][name]) for row in sample_rows]
        )
        for name in splitlib.CONTINUOUS_FEATURES
    }
    correlations = []
    for name, values in features.items():
        transformed = np.log(values) if np.all(values > 0.0) else values
        correlations.append(
            {
                "feature": name,
                "peak_pearson_r": _pearson(transformed, peak),
                "peak_spearman_rho": _spearman(values, peak),
                "reff_spearman_rho": _spearman(values, reff),
                **_partial_correlation(
                    transformed, peak, np.log(power)
                ),
                "single_feature_peak_r2": _single_feature_r2(
                    transformed, peak
                ),
            }
        )
    proxy_names = (
        "continuous_severity",
        "package_total_power_W",
        "total_source_area_fraction",
        "mean_source_area_fraction",
        "q_proxy_mean_W_m3",
    )
    proxy_rows = [
        next(row for row in correlations if row["feature"] == name)
        for name in proxy_names
    ]
    return {
        "reff_peak_K_per_W": {
            "minimum": float(np.min(reff)),
            "median": float(np.median(reff)),
            "maximum": float(np.max(reff)),
        },
        "feature_correlations": correlations,
        "proxy_audit": {
            "features": proxy_rows,
            "dominance_threshold_abs_spearman": 0.9,
            "dominant_features": [
                row["feature"]
                for row in proxy_rows
                if abs(row["peak_spearman_rho"]) >= 0.9
            ],
            "severity_source_size_coupling_spearman": _spearman(
                features["continuous_severity"],
                features["mean_source_area_fraction"],
            ),
        },
    }


def _split_comparison(
    records: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, str]],
    sobol_values: np.ndarray,
    *,
    dataset_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_ids = [str(row["sample_id"]) for row in records]
    counts = dict(
        Counter(row["split_role"] for row in current_rows)
    )
    octet_salt = (
        "v6-p1i-formal1024-split-v0"
        if len(records) == 1024
        else "v6-p1i-octet-candidate-v1"
    )
    candidates = {
        "octet_hash": splitlib.octet_hash_assignment(
            sample_ids, counts, salt=octet_salt
        ),
        "global_hash": splitlib.global_hash_assignment(
            sample_ids, counts, salt="v6-p1i-split-global-candidate-v1"
        ),
        "independent_sobol_dimension": splitlib.sobol_dimension_assignment(
            sample_ids,
            counts,
            sobol_values=sobol_values,
            dimension=int(sobol_values.shape[1] - 1),
        ),
        "balanced_pre_solve_assignment": splitlib.balanced_input_assignment(
            records, counts, salt="v6-p1i-balanced-input-v1"
        ),
    }
    current = {row["sample_id"]: row["split_role"] for row in current_rows}
    if len(records) == 1024 and candidates["octet_hash"] != current:
        raise RuntimeError("reconstructed octet split does not match formal1024_v0")
    rows = []
    payload: dict[str, Any] = {
        "selection_inputs": "pre_solve_only",
        "temperature_or_model_error_used": False,
        "methods": {},
    }
    for method, assignment in candidates.items():
        metrics = splitlib.split_metrics(records, assignment)
        score = (
            metrics["maximum_continuous_ks"]
            + metrics["maximum_discrete_tv"]
            + metrics["maximum_joint_discrepancy"]
        )
        rows.append(
            {
                "method": method,
                "dataset_id": dataset_id,
                "sample_count": len(records),
                "maximum_continuous_ks": metrics["maximum_continuous_ks"],
                "mean_continuous_ks": metrics["mean_continuous_ks"],
                "maximum_discrete_tv": metrics["maximum_discrete_tv"],
                "mean_discrete_tv": metrics["mean_discrete_tv"],
                "maximum_joint_discrepancy": metrics[
                    "maximum_joint_discrepancy"
                ],
                "preregistered_composite_score": score,
                "assignment_sha256": splitlib.assignment_sha256(assignment),
            }
        )
        payload["methods"][method] = {
            "assignment": dict(sorted(assignment.items())),
            "assignment_sha256": splitlib.assignment_sha256(assignment),
            "metrics": metrics,
            "composite_score": score,
        }
    winner = min(rows, key=lambda row: row["preregistered_composite_score"])
    payload["selection_rule"] = (
        "minimize maximum_continuous_ks + maximum_discrete_tv + "
        "maximum_joint_discrepancy using pre-solve inputs only"
    )
    payload["selected_method"] = winner["method"]
    payload["selected_assignment_sha256"] = winner["assignment_sha256"]
    return payload, rows


def _combined_split_comparison() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = {}
    rows = []
    for config_path, samples_path, dataset_id in (
        (
            CONFIG,
            SAMPLES,
            "heat3d_v6_p1i_continuous_physics1024_v0",
        ),
        (
            PILOT_CONFIG,
            PILOT_SAMPLES,
            "heat3d_v6_p1i_continuous_physics128_v2",
        ),
    ):
        records, sobol_values = splitlib.design_records(config_path)
        payload, dataset_rows = _split_comparison(
            records,
            _read_csv(samples_path),
            sobol_values,
            dataset_id=dataset_id,
        )
        datasets[dataset_id] = payload
        rows.extend(dataset_rows)
    method_scores = {}
    for method in (
        "octet_hash",
        "global_hash",
        "independent_sobol_dimension",
        "balanced_pre_solve_assignment",
    ):
        method_scores[method] = float(
            sum(
                row["preregistered_composite_score"]
                for row in rows
                if row["method"] == method
            )
        )
    selected = min(method_scores, key=method_scores.get)
    return (
        {
            "selection_inputs": "pre_solve_only",
            "temperature_or_model_error_used": False,
            "selection_rule": (
                "minimize the sum across pilot128 and formal1024 of "
                "maximum_continuous_ks + maximum_discrete_tv + "
                "maximum_joint_discrepancy"
            ),
            "aggregate_scores": method_scores,
            "selected_method": selected,
            "datasets": datasets,
        },
        rows,
    )


def _markdown(report: Mapping[str, Any], split_rows: Sequence[Mapping[str, Any]]) -> str:
    gap = report["gap_postmortem"]
    proxy = report["physical_response"]["proxy_audit"]
    lines = [
        "# V6-P1i formal1024_v0 zero-solve postmortem",
        "",
        "This analysis reuses the frozen formal1024_v0 outputs and performs no "
        "new PDE solve, training, or model inference. Split candidates are scored "
        "only with reconstructed pre-solve inputs.",
        "",
        "## Temperature gap",
        "",
        f"- Largest gap: {gap['gap_K']:.6f} K, from "
        f"`{gap['samples'][0]['sample_id']}` to "
        f"`{gap['samples'][1]['sample_id']}`.",
        f"- Empirical location: q={gap['left_empirical_quantile']:.6f}; "
        f"classification: `{gap['classification']}`.",
        f"- KDE density at the midpoint is "
        f"{100 * gap['kde_density_relative_to_global_max']:.3f}% of the global "
        "maximum. The failed gap is therefore a sparse extreme-tail interval, "
        "not a core/modal-valley discontinuity.",
        "",
        "## Proxy and response audit",
        "",
        f"- Dominant single latents at |Spearman| >= 0.9: "
        f"`{proxy['dominant_features']}`.",
        f"- Severity/source-size Spearman coupling: "
        f"{proxy['severity_source_size_coupling_spearman']:.6f}.",
        "- `Reff = peak DeltaT / package power`. Partial correlations regress "
        "both response and each transformed feature on log(power); they are "
        "descriptive diagnostics, not a power backsolve.",
        "",
        "## Target-independent split comparison",
        "",
        "| dataset | method | max continuous KS | max discrete TV | max joint | score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(split_rows, key=lambda value: value["preregistered_composite_score"]):
        lines.append(
            f"| {row['sample_count']} | {row['method']} | "
            f"{row['maximum_continuous_ks']:.6f} | "
            f"{row['maximum_discrete_tv']:.6f} | "
            f"{row['maximum_joint_discrepancy']:.6f} | "
            f"{row['preregistered_composite_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected method for preregistration: "
            f"`{report['split_comparison']['selected_method']}`.",
            "",
            "formal1024_v0 remains permanently qualification-failed. No sample, "
            "split, threshold, or frozen artifact was repaired.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    sample_rows = _read_csv(SAMPLES)
    records, _ = splitlib.design_records(CONFIG)
    if not (len(records) == len(sample_rows) == 1024):
        raise RuntimeError("formal1024 row mismatch")
    gap, gap_rows = _gap_postmortem(sample_rows)
    response = _response_postmortem(sample_rows, records)
    split_payload, split_rows = _combined_split_comparison()
    report = {
        "schema_version": "heat3d_v6_p1i_postmortem_v1",
        "dataset_id": "heat3d_v6_p1i_continuous_physics1024_v0",
        "formal1024_v0_lifecycle": "permanent_failed_qualification",
        "analysis_mode": "zero_new_solver_runs",
        "gap_postmortem": gap,
        "physical_response": response,
        "split_comparison": {
            key: value
            for key, value in split_payload.items()
            if key != "datasets"
        },
        "split_method_details": {
            dataset_id: {
                method: {
                    key: value
                    for key, value in payload.items()
                    if key != "assignment"
                }
                for method, payload in dataset["methods"].items()
            }
            for dataset_id, dataset in split_payload["datasets"].items()
        },
        "guardrails": {
            "new_solver_runs": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
            "temperature_used_for_split_selection": False,
            "formal1024_v0_modified": False,
            "v6_or_p1h_modified": False,
        },
    }
    _write_json(OUTPUT, report)
    _write_json(SPLIT_JSON, split_payload)
    _write_csv(SPLIT_CSV, split_rows)
    _write_csv(GAP_CSV, gap_rows)
    REPORT.write_text(_markdown(report, split_rows), encoding="utf-8")
    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in sample_rows])
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.hist(peaks, bins=36, color="#4774B3", alpha=0.55, density=True)
    grid = np.linspace(float(peaks.min()), float(peaks.max()), 600)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        density = gaussian_kde(peaks)(grid)
    axis.plot(grid, density, color="#B22222", lw=2)
    axis.axvspan(
        float(gap["samples"][0]["peak_deltaT_K"]),
        float(gap["samples"][1]["peak_deltaT_K"]),
        color="#FFB000",
        alpha=0.35,
        label="largest gap",
    )
    axis.set(xlabel="peak DeltaT (K)", ylabel="density")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "passed",
                "gap_K": gap["gap_K"],
                "gap_classification": gap["classification"],
                "selected_split": split_payload["selected_method"],
                "solver_runs": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
