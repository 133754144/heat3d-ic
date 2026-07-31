#!/usr/bin/env python3
"""Audit frozen V6-P1i formal1024_v1 without training or model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/heat3d_v6_p1i"
DOCS = ROOT / "docs"
PREFIX = "v6_p1i_pilot128_v3"
DATASET_ID = "heat3d_v6_p1i_continuous_physics128_v3"
SAMPLES = CONFIG / f"{PREFIX}_samples.csv"
INPUTS = CONFIG / f"{PREFIX}_input_definitions.csv"
REGIONS = CONFIG / f"{PREFIX}_regions.csv"
MANIFEST = CONFIG / f"{PREFIX}_manifest.json"
SPLIT = CONFIG / f"{PREFIX}_split_manifest.json"
ACCEPTANCE = CONFIG / f"{PREFIX}_acceptance.json"
OUTPUT = CONFIG / f"{PREFIX}_distribution_audit.json"
CORRELATIONS = CONFIG / f"{PREFIX}_correlations.csv"
REPORT = DOCS / f"{PREFIX}_distribution_audit.md"
FIGURE = DOCS / f"{PREFIX}_distribution.png"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(
        np.argsort(values, kind="mergesort"),
        kind="mergesort",
    ).astype(np.float64)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not math.isfinite(value):
        raise RuntimeError("non-finite Spearman statistic")
    return value


def _partial_spearman(
    left: np.ndarray,
    right: np.ndarray,
    control: np.ndarray,
) -> float:
    x = _ranks(left)
    y = _ranks(right)
    c = _ranks(control)
    c = (c - np.mean(c)) / np.std(c)
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    control_energy = float(np.sum(c * c))
    x_residual = x - float(np.sum(x * c)) / control_energy * c
    y_residual = y - float(np.sum(y * c)) / control_energy * c
    value = float(np.corrcoef(x_residual, y_residual)[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("non-finite partial Spearman statistic")
    return value


def _linear_r2(feature: np.ndarray, target: np.ndarray) -> float:
    feature_std = float(np.std(feature))
    target_std = float(np.std(target))
    if feature_std == 0.0 or target_std == 0.0:
        return 0.0
    feature_z = (feature - np.mean(feature)) / feature_std
    target_z = (target - np.mean(target)) / target_std
    correlation = float(np.mean(feature_z * target_z))
    if not math.isfinite(correlation):
        raise RuntimeError("non-finite linear correlation")
    return correlation * correlation


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _kde_modes(values: np.ndarray) -> dict[str, Any]:
    low = float(np.min(values))
    high = float(np.max(values))
    grid = np.linspace(low, high, 512)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        density = gaussian_kde(values)(grid)
    if not np.all(np.isfinite(density)):
        raise RuntimeError("non-finite KDE density")
    peaks = np.flatnonzero(
        (density[1:-1] > density[:-2])
        & (density[1:-1] >= density[2:])
    ) + 1
    peaks = peaks[density[peaks] >= 0.08 * float(np.max(density))]
    return {
        "mode_count": int(peaks.size),
        "mode_locations_K": grid[peaks].tolist(),
        "grid_K": grid.tolist(),
        "density": density.tolist(),
    }


def _maximum_gap(values: np.ndarray) -> dict[str, Any]:
    ordered = np.sort(values)
    differences = np.diff(ordered)
    index = int(np.argmax(differences))
    return {
        "maximum_gap_K": float(differences[index]),
        "lower_K": float(ordered[index]),
        "upper_K": float(ordered[index + 1]),
    }


def _generator_latent_names(input_rows: Sequence[Mapping[str, str]]) -> list[str]:
    forbidden = {
        "sample_id",
        "split_role",
        "sobol_index",
        "sobol_dimensions_consumed",
    }
    names = []
    for name in input_rows[0]:
        if name in forbidden:
            continue
        try:
            np.asarray([float(row[name]) for row in input_rows])
        except (TypeError, ValueError):
            continue
        names.append(name)
    return names


def audit() -> dict[str, Any]:
    samples = _read_csv(SAMPLES)
    inputs = _read_csv(INPUTS)
    regions = _read_csv(REGIONS)
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    expected_count = int(acceptance["sample_count"])
    if len(samples) != expected_count or len(inputs) != expected_count:
        raise RuntimeError("formal1024_v1 row count drifted")
    input_by_id = {row["sample_id"]: row for row in inputs}
    if set(input_by_id) != {row["sample_id"] for row in samples}:
        raise RuntimeError("sample/input identities drifted")

    peak = np.asarray([float(row["peak_deltaT_K"]) for row in samples])
    power = np.asarray(
        [float(row["package_total_power_W"]) for row in samples]
    )
    top_h = np.asarray([float(row["top_h_W_m2K"]) for row in samples])
    severity = np.asarray(
        [float(input_by_id[row["sample_id"]]["continuous_severity"])
         for row in samples]
    )
    source_area = np.asarray(
        [
            float(
                input_by_id[row["sample_id"]]["mean_source_area_fraction"]
            )
            for row in samples
        ]
    )
    reff = peak / power

    temperature_gate = acceptance["temperature_coverage"]
    primary_low, primary_high = map(
        float, temperature_gate["primary_interval_K"]
    )
    outer_low, outer_high = map(
        float, temperature_gate["outer_safety_interval_K"]
    )
    bin_edges = np.linspace(
        primary_low,
        primary_high,
        int(temperature_gate["histogram_equal_width_bins"]) + 1,
    )
    bin_counts, _ = np.histogram(peak, bins=bin_edges)
    nonzero = bin_counts[bin_counts > 0]
    bin_ratio = (
        float(np.max(nonzero) / np.min(nonzero))
        if nonzero.size
        else math.inf
    )
    full_gap = _maximum_gap(peak)
    q_low, q_high = map(
        float, temperature_gate["core_quantile_interval"]
    )
    core_low, core_high = np.quantile(peak, [q_low, q_high])
    core_peak = peak[(peak >= core_low) & (peak <= core_high)]
    core_gap = _maximum_gap(core_peak)
    kde = _kde_modes(peak)
    temperature = {
        "summary": _summary(peak),
        "inside_primary_count": int(
            np.sum((peak >= primary_low) & (peak <= primary_high))
        ),
        "inside_primary_fraction": float(
            np.mean((peak >= primary_low) & (peak <= primary_high))
        ),
        "outside_outer_count": int(
            np.sum((peak < outer_low) | (peak > outer_high))
        ),
        "outside_outer_sample_ids": [
            row["sample_id"]
            for row, value in zip(samples, peak, strict=True)
            if value < outer_low or value > outer_high
        ],
        "histogram_edges_K": bin_edges.tolist(),
        "histogram_counts": bin_counts.tolist(),
        "internal_empty_bins": int(np.sum(bin_counts == 0)),
        "nonzero_bin_count_ratio": bin_ratio,
        "full_sorted_peak_gap": full_gap,
        "core_quantile_interval": [q_low, q_high],
        "core_peak_interval_K": [float(core_low), float(core_high)],
        "core_sorted_peak_gap": core_gap,
        "kde": kde,
    }

    source_regions = [row for row in regions if row["family"] == "q"]
    source_nodes = np.asarray(
        [int(row["control_volume_count"]) for row in source_regions]
    )
    source_power_by_sample: dict[str, float] = {}
    for row in source_regions:
        source_power_by_sample[row["sample_id"]] = (
            source_power_by_sample.get(row["sample_id"], 0.0)
            + float(row["source_power_W"])
        )
    power_relative_error = np.asarray(
        [
            abs(source_power_by_sample[row["sample_id"]] - expected)
            / expected
            for row, expected in zip(samples, power, strict=True)
        ]
    )
    energy = np.abs(
        np.asarray(
            [float(row["energy_balance_relative_error"]) for row in samples]
        )
    )
    residual = np.abs(
        np.asarray([float(row["linear_residual"]) for row in samples])
    )
    support = np.asarray(
        [int(row["minimum_support_nodes_per_region"]) for row in samples]
    )
    physics = {
        "maximum_abs_energy_balance_relative_error": float(np.max(energy)),
        "maximum_linear_residual": float(np.max(residual)),
        "minimum_source_control_volume_count": int(np.min(source_nodes)),
        "minimum_support_nodes_per_local_region": int(np.min(support)),
        "maximum_source_power_relative_error": float(
            np.max(power_relative_error)
        ),
        "all_values_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in (
                    peak,
                    power,
                    top_h,
                    severity,
                    source_area,
                    energy,
                    residual,
                    support,
                    source_nodes,
                    power_relative_error,
                )
            )
        ),
    }

    latent_rows: list[dict[str, Any]] = []
    for name in _generator_latent_names(inputs):
        values = np.asarray(
            [float(input_by_id[row["sample_id"]][name]) for row in samples]
        )
        if float(np.std(values)) == 0.0:
            continue
        latent_rows.append(
            {
                "feature": name,
                "peak_spearman": _spearman(values, peak),
                "peak_linear_r2": _linear_r2(values, peak),
            }
        )
    maximum_abs_spearman = max(
        latent_rows, key=lambda row: abs(float(row["peak_spearman"]))
    )
    maximum_r2 = max(
        latent_rows, key=lambda row: float(row["peak_linear_r2"])
    )
    response = {
        "power_peak_spearman": _spearman(power, peak),
        "top_h_reff_spearman": _spearman(top_h, reff),
        "top_h_peak_partial_spearman_controlling_log_power": (
            _partial_spearman(top_h, peak, np.log(power))
        ),
        "severity_peak_spearman": _spearman(severity, peak),
        "mean_source_area_peak_spearman": _spearman(source_area, peak),
        "maximum_abs_single_generator_latent_peak_spearman": (
            maximum_abs_spearman
        ),
        "maximum_single_generator_latent_peak_r2": maximum_r2,
        "latent_rows": latent_rows,
    }

    split_metrics = split["metrics"]
    split_gate = acceptance["split_qc"]
    qc = acceptance["physics_qc"]
    response_gate = acceptance["physical_response_qc"]
    checks = {
        "temperature_primary_fraction": (
            temperature["inside_primary_fraction"]
            >= float(
                temperature_gate["minimum_inside_primary_fraction"]
            )
        ),
        "temperature_outer_safety": (
            temperature["outside_outer_count"]
            <= int(temperature_gate["maximum_outside_outer_count"])
        ),
        "temperature_no_empty_bins": (
            temperature["internal_empty_bins"]
            <= int(temperature_gate["maximum_internal_empty_bins"])
        ),
        "temperature_bin_ratio": (
            temperature["nonzero_bin_count_ratio"]
            <= float(temperature_gate["maximum_nonzero_bin_count_ratio"])
        ),
        "temperature_core_gap": (
            temperature["core_sorted_peak_gap"]["maximum_gap_K"]
            <= float(temperature_gate["maximum_core_sorted_peak_gap_K"])
        ),
        "temperature_full_gap": (
            temperature["full_sorted_peak_gap"]["maximum_gap_K"]
            <= float(temperature_gate["maximum_full_sorted_peak_gap_K"])
        ),
        "temperature_kde_mode_limit": (
            kde["mode_count"]
            <= int(temperature_gate["maximum_kde_mode_count"])
        ),
        "temperature_kde_not_forbidden": (
            kde["mode_count"]
            != int(temperature_gate["forbidden_kde_mode_count"])
        ),
        "split_exact_counts": (
            split_metrics["counts"] == split_gate["exact_counts"]
        ),
        "split_continuous_ks": (
            split_metrics["maximum_continuous_ks"]
            <= float(split_gate["maximum_continuous_ks"])
        ),
        "split_discrete_tv": (
            split_metrics["maximum_discrete_tv"]
            <= float(split_gate["maximum_discrete_tv"])
        ),
        "split_joint_discrepancy": (
            split_metrics["maximum_joint_discrepancy"]
            <= float(split_gate["maximum_joint_discrepancy"])
        ),
        "split_target_independent": (
            split["target_values_used"] is False
            and split["solver_results_used"] is False
            and split["model_error_used"] is False
        ),
        "physics_energy_balance": (
            physics["maximum_abs_energy_balance_relative_error"]
            <= float(qc["maximum_abs_energy_balance_relative_error"])
        ),
        "physics_linear_residual": (
            physics["maximum_linear_residual"]
            <= float(qc["maximum_linear_residual"])
        ),
        "physics_source_nodes": (
            physics["minimum_source_control_volume_count"]
            >= int(qc["minimum_source_nodes_per_source"])
        ),
        "physics_support": (
            physics["minimum_support_nodes_per_local_region"]
            >= int(qc["minimum_support_nodes_per_local_region"])
        ),
        "physics_power_conservation": (
            physics["maximum_source_power_relative_error"] <= 1.0e-12
        ),
        "physics_finite": physics["all_values_finite"],
        "response_power": (
            response["power_peak_spearman"]
            >= float(response_gate["minimum_power_peak_spearman"])
        ),
        "response_top_h_reff": (
            response["top_h_reff_spearman"]
            <= float(response_gate["maximum_top_h_reff_spearman"])
        ),
        "response_top_h_partial": (
            response[
                "top_h_peak_partial_spearman_controlling_log_power"
            ]
            <= float(
                response_gate[
                    "maximum_top_h_peak_partial_spearman_controlling_power"
                ]
            )
        ),
        "response_severity_not_dominant": (
            abs(response["severity_peak_spearman"])
            <= float(
                response_gate["maximum_abs_severity_peak_spearman"]
            )
        ),
        "response_source_area_not_dominant": (
            abs(response["mean_source_area_peak_spearman"])
            <= float(
                response_gate[
                    "maximum_abs_mean_source_area_peak_spearman"
                ]
            )
        ),
        "response_single_latent_spearman": (
            abs(
                response[
                    "maximum_abs_single_generator_latent_peak_spearman"
                ]["peak_spearman"]
            )
            <= float(
                response_gate[
                    "maximum_abs_single_generator_latent_peak_spearman"
                ]
            )
        ),
        "response_single_latent_r2": (
            response["maximum_single_generator_latent_peak_r2"][
                "peak_linear_r2"
            ]
            <= float(
                response_gate[
                    "maximum_single_generator_latent_peak_r2"
                ]
            )
        ),
    }
    return {
        "schema_version": "heat3d_v6_p1i_formal1024_v1_audit_v1",
        "dataset_id": DATASET_ID,
        "sample_count": len(samples),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "temperature_coverage": temperature,
        "split_qc": split_metrics,
        "physics_qc": physics,
        "physical_response_qc": response,
        "manifest": {
            "dataset_id": manifest["dataset_id"],
            "sample_count": manifest["sample_count"],
        },
        "guardrails": {
            "training_runs": 0,
            "model_inference_runs": 0,
            "post_solve_filtering_or_replacement": False,
            "per_sample_power_backsolve": False,
            "formal1024_v1_generated": True,
            "formal1024_v0_modified": False,
            "v6_or_p1h_modified": False,
        },
    }


def _plot(report: Mapping[str, Any]) -> None:
    rows = _read_csv(SAMPLES)
    peak = np.asarray([float(row["peak_deltaT_K"]) for row in rows])
    power = np.asarray(
        [float(row["package_total_power_W"]) for row in rows]
    )
    top_h = np.asarray([float(row["top_h_W_m2K"]) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].hist(peak, bins=12, color="#4472C4", alpha=0.72)
    axes[0].set_xlabel("peak DeltaT (K)")
    axes[0].set_ylabel("count")
    axes[1].scatter(power, peak, s=18, alpha=0.72)
    axes[1].set_xlabel("package power (W)")
    axes[1].set_ylabel("peak DeltaT (K)")
    axes[2].scatter(top_h, peak / power, s=18, alpha=0.72)
    axes[2].set_xlabel("top h (W/(m2 K))")
    axes[2].set_ylabel("Reff (K/W)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{DATASET_ID}: {report['status']}")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def _markdown(report: Mapping[str, Any]) -> str:
    temperature = report["temperature_coverage"]
    response = report["physical_response_qc"]
    physics = report["physics_qc"]
    failed = [name for name, passed in report["checks"].items() if not passed]
    return "\n".join(
        [
            "# V6-P1i formal1024_v1 qualification",
            "",
            f"Status: **{report['status']}**.",
            "",
            f"The frozen {report['sample_count']} cases were audited without "
            "filtering, replacement, training, or model inference. The formal "
            "dataset qualifies only when every preregistered gate passes.",
            "",
            "## Temperature coverage",
            "",
            f"- 30--150 K: {temperature['inside_primary_count']}/"
            f"{report['sample_count']} "
            f"({100 * temperature['inside_primary_fraction']:.3f}%).",
            f"- 20--180 K violations: {temperature['outside_outer_count']} "
            f"{temperature['outside_outer_sample_ids']}.",
            f"- 12-bin counts: `{temperature['histogram_counts']}`.",
            f"- Core q05--q95 maximum gap: "
            f"{temperature['core_sorted_peak_gap']['maximum_gap_K']:.6f} K.",
            f"- Full maximum gap: "
            f"{temperature['full_sorted_peak_gap']['maximum_gap_K']:.6f} K.",
            f"- KDE modes: {temperature['kde']['mode_count']}.",
            "",
            "## Physical response",
            "",
            f"- power versus peak Spearman: "
            f"{response['power_peak_spearman']:.6f}.",
            f"- top h versus Reff Spearman: "
            f"{response['top_h_reff_spearman']:.6f}.",
            f"- top h versus peak partial Spearman controlling power: "
            f"{response['top_h_peak_partial_spearman_controlling_log_power']:.6f}.",
            f"- severity versus peak Spearman: "
            f"{response['severity_peak_spearman']:.6f}.",
            f"- mean source area versus peak Spearman: "
            f"{response['mean_source_area_peak_spearman']:.6f}.",
            "",
            "## Physics and decision",
            "",
            f"- Minimum source control volumes: "
            f"{physics['minimum_source_control_volume_count']}.",
            f"- Minimum projected local-region support: "
            f"{physics['minimum_support_nodes_per_local_region']}.",
            f"- Maximum energy error / residual: "
            f"{physics['maximum_abs_energy_balance_relative_error']:.3e} / "
            f"{physics['maximum_linear_residual']:.3e}.",
            f"- Failed gates: `{failed}`.",
            "",
            "formal1024_v1 is qualified only if the status above is `passed`.",
            "",
        ]
    )


def main() -> int:
    global PREFIX, DATASET_ID, SAMPLES, INPUTS, REGIONS, MANIFEST, SPLIT
    global ACCEPTANCE, OUTPUT, CORRELATIONS, REPORT, FIGURE
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-prefix", default=PREFIX)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args()
    PREFIX = str(args.artifact_prefix)
    DATASET_ID = str(args.dataset_id)
    SAMPLES = CONFIG / f"{PREFIX}_samples.csv"
    INPUTS = CONFIG / f"{PREFIX}_input_definitions.csv"
    REGIONS = CONFIG / f"{PREFIX}_regions.csv"
    MANIFEST = CONFIG / f"{PREFIX}_manifest.json"
    SPLIT = CONFIG / f"{PREFIX}_split_manifest.json"
    ACCEPTANCE = CONFIG / f"{PREFIX}_acceptance.json"
    OUTPUT = CONFIG / f"{PREFIX}_distribution_audit.json"
    CORRELATIONS = CONFIG / f"{PREFIX}_correlations.csv"
    REPORT = DOCS / f"{PREFIX}_distribution_audit.md"
    FIGURE = DOCS / f"{PREFIX}_distribution.png"
    report = audit()
    _write_json(OUTPUT, report)
    _write_csv(CORRELATIONS, report["physical_response_qc"]["latent_rows"])
    REPORT.write_text(_markdown(report), encoding="utf-8")
    _plot(report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_checks": [
                    name
                    for name, passed in report["checks"].items()
                    if not passed
                ],
                "training_runs": 0,
                "model_inference_runs": 0,
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
