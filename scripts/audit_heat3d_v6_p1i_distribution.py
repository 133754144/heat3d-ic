#!/usr/bin/env python3
"""Audit the frozen V6-P1i pilot distribution without model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
DOCS_DIR = ROOT / "docs"
DEFAULT_SAMPLES = CONFIG_DIR / "v6_p1i_pilot128_v0_samples.csv"
DEFAULT_ACCEPTANCE = CONFIG_DIR / "v6_p1i_pilot_acceptance.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _modes(values: np.ndarray, low: float, high: float) -> dict[str, Any]:
    grid = np.linspace(low, high, 512)
    density = gaussian_kde(values)(grid)
    peaks = np.flatnonzero(
        (density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])
    ) + 1
    threshold = float(np.max(density)) * 0.08
    peaks = peaks[density[peaks] >= threshold]
    return {
        "mode_count": int(peaks.size),
        "mode_locations_K": grid[peaks].tolist(),
        "grid_K": grid.tolist(),
        "density": density.tolist(),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    quantiles = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": {
            f"q{int(q * 100):02d}": float(np.quantile(values, q))
            for q in quantiles
        },
    }


def audit(samples_path: Path, acceptance_path: Path) -> dict[str, Any]:
    rows = _read_csv(samples_path)
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 rows, found {len(rows)}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    metrics = {
        "deltaT_max_K": np.asarray(
            [float(row["peak_deltaT_K"]) for row in rows]
        ),
        "deltaT_mean_K": np.asarray(
            [float(row["mean_deltaT_K"]) for row in rows]
        ),
        "deltaT_cv_rms_K": np.asarray(
            [float(row["cv_rms_deltaT_K"]) for row in rows]
        ),
    }
    gate = acceptance["temperature_coverage"]
    low, high = map(float, gate["primary_interval_K"])
    outer_low, outer_high = map(float, gate["outer_safety_interval_K"])
    bins = int(gate["histogram_equal_width_bins"])
    edges = np.linspace(low, high, bins + 1)
    peaks = metrics["deltaT_max_K"]
    counts, _ = np.histogram(peaks, bins=edges)
    sorted_peaks = np.sort(peaks)
    inside = int(np.sum((peaks >= low) & (peaks <= high)))
    outside_outer = int(
        np.sum((peaks < outer_low) | (peaks > outer_high))
    )
    nonzero = counts[counts > 0]
    ratio = (
        float(np.max(nonzero) / np.min(nonzero))
        if nonzero.size
        else math.inf
    )
    peak_modes = _modes(peaks, min(outer_low, float(peaks.min())), max(outer_high, float(peaks.max())))
    temperature_checks = {
        "inside_primary_fraction": inside / len(peaks),
        "inside_primary_count": inside,
        "outside_outer_count": outside_outer,
        "histogram_edges_K": edges.tolist(),
        "histogram_counts": counts.tolist(),
        "internal_empty_bins": int(np.sum(counts == 0)),
        "nonzero_bin_count_ratio": ratio,
        "maximum_sorted_peak_gap_K": float(np.max(np.diff(sorted_peaks))),
        "kde": peak_modes,
    }
    checks = {
        "primary_fraction": temperature_checks["inside_primary_fraction"]
        >= float(gate["minimum_inside_primary_fraction"]),
        "outer_safety": outside_outer
        <= int(gate["maximum_outside_outer_count"]),
        "empty_bins": temperature_checks["internal_empty_bins"]
        <= int(gate["maximum_internal_empty_bins"]),
        "bin_ratio": ratio <= float(gate["maximum_nonzero_bin_count_ratio"]),
        "sorted_gap": temperature_checks["maximum_sorted_peak_gap_K"]
        <= float(gate["maximum_sorted_peak_gap_K"]),
        "kde_not_four": peak_modes["mode_count"]
        != int(gate["forbidden_kde_mode_count"]),
        "kde_mode_limit": peak_modes["mode_count"]
        <= int(gate["maximum_kde_mode_count"]),
    }
    qc = acceptance["physics_qc"]
    energy = np.abs(
        np.asarray(
            [float(row["energy_balance_relative_error"]) for row in rows]
        )
    )
    residual = np.asarray(
        [float(row["linear_residual"]) for row in rows]
    )
    support = np.asarray(
        [int(row["minimum_support_nodes_per_region"]) for row in rows]
    )
    checks.update(
        {
            "energy_balance": float(np.max(energy))
            <= float(qc["maximum_abs_energy_balance_relative_error"]),
            "linear_residual": float(np.max(residual))
            <= float(qc["maximum_linear_residual"]),
            "support_coverage": int(np.min(support))
            >= int(qc["minimum_support_nodes_per_local_region"]),
            "finite": all(
                bool(np.all(np.isfinite(values)))
                for values in (*metrics.values(), energy, residual)
            ),
        }
    )
    parameter_names = (
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "total_source_volume_m3",
        "mean_q_W_m3",
        "max_q_W_m3",
        "mean_local_k_W_mK",
        "source_count",
        "k_region_count",
    )
    correlation_rows = []
    for parameter in parameter_names:
        x = np.asarray([float(row[parameter]) for row in rows])
        for metric, y in metrics.items():
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            correlation_rows.append(
                {
                    "parameter": parameter,
                    "metric": metric,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    output = {
        "schema_version": "heat3d_v6_p1i_distribution_audit_v1",
        "dataset_id": "heat3d_v6_p1i_continuous_physics128_v0",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "temperature_coverage": temperature_checks,
        "metric_summaries": {
            name: _summary(values) for name, values in metrics.items()
        },
        "physics_qc": {
            "maximum_abs_energy_balance_relative_error": float(np.max(energy)),
            "maximum_linear_residual": float(np.max(residual)),
            "minimum_support_nodes_per_region": int(np.min(support)),
        },
        "correlations": correlation_rows,
        "guardrails": {
            "training_runs": 0,
            "model_inference_runs": 0,
            "test_labels_used_for_design": False,
            "post_solve_filtering_or_replacement": False,
            "formal1024_generated": False,
        },
    }
    return output


def _plot(report: Mapping[str, Any], samples_path: Path, output: Path) -> None:
    rows = _read_csv(samples_path)
    series = (
        ("DeltaT max", np.asarray([float(r["peak_deltaT_K"]) for r in rows])),
        ("DeltaT mean", np.asarray([float(r["mean_deltaT_K"]) for r in rows])),
        ("DeltaT CV-RMS", np.asarray([float(r["cv_rms_deltaT_K"]) for r in rows])),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, (label, values) in zip(axes, series):
        axis.hist(values, bins=12, density=True, alpha=0.55, color="#4472C4")
        grid = np.linspace(float(values.min()), float(values.max()), 400)
        axis.plot(grid, gaussian_kde(values)(grid), color="#C00000", lw=2)
        for q in np.quantile(values, [0.25, 0.5, 0.75]):
            axis.axvline(q, color="#555555", lw=0.8, ls="--")
        axis.set_title(label)
        axis.set_xlabel("K")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"V6-P1i pilot128 continuous coverage: {report['status']}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _markdown(report: Mapping[str, Any]) -> str:
    coverage = report["temperature_coverage"]
    summary = report["metric_summaries"]
    lines = [
        "# V6-P1i pilot128 distribution audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        "No training or model inference was run. The 1024-sample expansion remains "
        "blocked pending an explicit decision after this report.",
        "",
        "## Temperature coverage",
        "",
        f"- Primary 30--150 K: {coverage['inside_primary_count']}/128 "
        f"({100 * coverage['inside_primary_fraction']:.2f}%).",
        f"- Outer-safety violations: {coverage['outside_outer_count']}.",
        f"- Twelve-bin counts: `{coverage['histogram_counts']}`.",
        f"- Empty bins: {coverage['internal_empty_bins']}; nonzero max/min ratio: "
        f"{coverage['nonzero_bin_count_ratio']:.3f}.",
        f"- Largest adjacent sorted peak gap: "
        f"{coverage['maximum_sorted_peak_gap_K']:.3f} K.",
        f"- Significant KDE modes: {coverage['kde']['mode_count']} at "
        f"{[round(x, 3) for x in coverage['kde']['mode_locations_K']]}.",
        "",
        "## DeltaT summaries",
        "",
        "| metric | mean | std | q05 | q50 | q95 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary.items():
        q = values["quantiles"]
        lines.append(
            f"| {key} | {values['mean']:.4f} | {values['std']:.4f} | "
            f"{q['q05']:.4f} | {q['q50']:.4f} | {q['q95']:.4f} | "
            f"{q['q00']:.4f} | {q['q100']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate details",
            "",
            *[
                f"- `{key}`: {'PASS' if value else 'FAIL'}"
                for key, value in report["checks"].items()
            ],
            "",
            "The distribution was not binned or replaced after solving. Correlations "
            "are descriptive only and cannot be used to remove samples.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    args = parser.parse_args()
    report = audit(args.samples.resolve(), args.acceptance.resolve())
    _json(CONFIG_DIR / "v6_p1i_pilot128_v0_distribution_audit.json", report)
    _csv(
        CONFIG_DIR / "v6_p1i_pilot128_v0_correlations.csv",
        report["correlations"],
    )
    _plot(
        report,
        args.samples.resolve(),
        DOCS_DIR / "v6_p1i_pilot128_v0_distribution.png",
    )
    (DOCS_DIR / "v6_p1i_pilot128_v0_distribution_audit.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
