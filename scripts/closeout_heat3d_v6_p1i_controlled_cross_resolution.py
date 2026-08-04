#!/usr/bin/env python3
"""Summarize the controlled P1i cross-resolution result bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "point_global_true_rms_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "peak_rmse_K",
    "source_rmse_K",
    "interface_drop_rmse_K",
)


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main_summary(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    resolutions = sorted({int(item["resolution"]) for item in payload["main"]})
    for n in resolutions:
        cells = [item for item in payload["main"] if int(item["resolution"]) == n]
        row: dict[str, Any] = {"resolution": n, "seed_count": len(cells)}
        for domain in ("support_metrics", "full_metrics", "oracle_reconstruction_metrics"):
            for key in METRIC_KEYS:
                stats = summarize([float(cell[domain][key]) for cell in cells])
                for stat, value in stats.items():
                    row[f"{domain}_{key}_{stat}"] = value
        row["regional_nodes_mean"] = float(np.mean([
            sample["graph"]["regional_nodes"] for cell in cells for sample in cell["samples"]
        ]))
        row["p2r_inactive_regional_mean"] = float(np.mean([
            sample["graph"]["p2r"]["in_degree"]["zero_count"]
            for cell in cells for sample in cell["samples"]
        ]))
        rows.append(row)
    return rows


def factor_summary(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in "ABCD":
        for item in sorted(
            (row for row in payload["factors"] if row["factor_cell"] == cell),
            key=lambda row: int(row["resolution"]),
        ):
            row = {
                "factor_cell": cell,
                "resolution": int(item["resolution"]),
                "support_mode": item["support_mode"],
                "regional_mode": item["regional_mode"],
                "regional_nodes_mean": float(np.mean([
                    sample["graph"]["regional_nodes"] for sample in item["samples"]
                ])),
            }
            row.update({key: float(item["support_metrics"][key]) for key in METRIC_KEYS})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def factor_attribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["factor_cell"], int(row["resolution"])): row for row in rows}
    result = []
    key = "point_global_true_rms_relative_rmse_pct"
    for n in sorted({int(row["resolution"]) for row in rows}):
        a, b, c, d = (lookup[(cell, n)] for cell in "ABCD")
        result.append({
            "resolution": n,
            "A_source_fixed_pct": a[key],
            "B_source_growing_pct": b[key],
            "C_structured_fixed_pct": c[key],
            "D_structured_growing_pct": d[key],
            "regional_effect_source_B_minus_A_pct_point": b[key] - a[key],
            "support_effect_fixed_C_minus_A_pct_point": c[key] - a[key],
            "regional_effect_structured_D_minus_C_pct_point": d[key] - c[key],
            "compound_D_minus_A_pct_point": d[key] - a[key],
        })
    return result


def plot_results(main_rows: Sequence[Mapping[str, Any]], factors: Sequence[Mapping[str, Any]], drift: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    x = np.asarray([row["resolution"] for row in main_rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for domain, label in (("support_metrics", "support"), ("full_metrics", "full 240825"), ("oracle_reconstruction_metrics", "oracle floor")):
        mean = np.asarray([row[f"{domain}_point_global_true_rms_relative_rmse_pct_mean"] for row in main_rows])
        std = np.asarray([row[f"{domain}_point_global_true_rms_relative_rmse_pct_std"] for row in main_rows])
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=label)
    ax.set_xscale("log", base=2); ax.set_xlabel("support nodes N"); ax.set_ylabel("point-global true-RMS (%)")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    path = output_dir / "v6_p1i_controlled_resolution_error.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for cell in "ABCD":
        rows = sorted((row for row in factors if row["factor_cell"] == cell), key=lambda row: row["resolution"])
        axes[0].plot([row["resolution"] for row in rows], [row["regional_nodes_mean"] for row in rows], marker="o", label=cell)
        axes[1].plot([row["resolution"] for row in rows], [row["point_global_true_rms_relative_rmse_pct"] for row in rows], marker="o", label=cell)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3); ax.legend(title="factor cell")
    axes[0].set_yscale("log", base=2); axes[0].set_xlabel("N"); axes[0].set_ylabel("regional nodes")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("support PG true-RMS (%)")
    fig.tight_layout(); path = output_dir / "v6_p1i_controlled_graph_scale.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for key, label in (("global_context_z_l2_drift", "Global Context z"), ("qk_summary_l2_drift", "QK summary"), ("predicted_log_scale_drift", "predicted log-scale")):
        means = []
        for n in x:
            values = [abs(float(row[key])) for row in drift if int(row["resolution"]) == int(n)]
            means.append(float(np.mean(values)))
        ax.plot(x, means, marker="o", label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("support nodes N"); ax.set_ylabel("mean absolute drift vs same-seed N=1024")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    path = output_dir / "v6_p1i_controlled_feature_drift.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))
    return created


def report_markdown(main_rows: Sequence[Mapping[str, Any]], factor_rows: Sequence[Mapping[str, Any]], attribution: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6 P1i controlled cross-resolution closeout",
        "",
        "This is a frozen valid-only diagnostic. Test/sealed remained closed; no training, tuning, or checkpoint mutation occurred. Direct-N results are compound-OOD diagnostics and are not model-selection evidence.",
        "",
        "## Source-aware nested ladder with training-scale regional mesh",
        "",
        "| N | support PG % | sample-first % | full PG % | oracle PG % | Nr | inactive p2r regional |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['resolution']} | {row['support_metrics_point_global_true_rms_relative_rmse_pct_mean']:.4f} ± {row['support_metrics_point_global_true_rms_relative_rmse_pct_std']:.4f} | "
            f"{row['support_metrics_sample_first_cv_relative_rmse_pct_mean']:.4f} ± {row['support_metrics_sample_first_cv_relative_rmse_pct_std']:.4f} | "
            f"{row['full_metrics_point_global_true_rms_relative_rmse_pct_mean']:.4f} ± {row['full_metrics_point_global_true_rms_relative_rmse_pct_std']:.4f} | "
            f"{row['oracle_reconstruction_metrics_point_global_true_rms_relative_rmse_pct_mean']:.4f} ± {row['oracle_reconstruction_metrics_point_global_true_rms_relative_rmse_pct_std']:.4f} | "
            f"{row['regional_nodes_mean']:.1f} | {row['p2r_inactive_regional_mean']:.1f} |"
        )
    lines += [
        "",
        "## A-D factor diagnostic",
        "",
        "| N | A source/fixed | B source/growing | C structured/fixed | D structured/growing | B-A | C-A | D-C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in attribution:
        lines.append(
            f"| {row['resolution']} | {row['A_source_fixed_pct']:.4f} | {row['B_source_growing_pct']:.4f} | "
            f"{row['C_structured_fixed_pct']:.4f} | {row['D_structured_growing_pct']:.4f} | "
            f"{row['regional_effect_source_B_minus_A_pct_point']:+.4f} | {row['support_effect_fixed_C_minus_A_pct_point']:+.4f} | "
            f"{row['regional_effect_structured_D_minus_C_pct_point']:+.4f} |"
        )
    worst_support = max(attribution, key=lambda row: abs(row["support_effect_fixed_C_minus_A_pct_point"]))
    worst_regional = max(attribution, key=lambda row: abs(row["regional_effect_source_B_minus_A_pct_point"]))
    lines += [
        "",
        "## Attribution",
        "",
        f"The largest fixed-Nr support-distribution contrast is C-A={worst_support['support_effect_fixed_C_minus_A_pct_point']:+.4f} percentage points at N={worst_support['resolution']}. "
        f"The largest source-aware regional-scale contrast is B-A={worst_regional['regional_effect_source_B_minus_A_pct_point']:+.4f} points at N={worst_regional['resolution']}.",
        "",
        "Interpretation uses the full A-D pattern together with Global Context, QK, physical-scale, predicted-scale, graph-size, and oracle-floor drift. It does not attribute the existing structured direct-N curve to resolution alone.",
        "",
        "## Reproducibility and governance",
        "",
        f"- Fixed subset: {payload['contract']['sample_count']} valid_iid samples.",
        f"- Discretization seeds: {payload['contract']['discretization_seeds']}.",
        f"- Checkpoint SHA256: `{payload['inputs']['checkpoint_sha256']}`.",
        f"- Dataset manifest SHA256: `{payload['inputs']['manifest_sha256']}`.",
        f"- Full-field SHA256: `{payload['inputs']['full_fields_sha256']}`.",
        "- test accessed: false; sealed accessed: false; training/tuning: false.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--factor-csv", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    main_rows = main_summary(payload)
    factors = factor_summary(payload)
    attribution = factor_attribution(factors)
    figures = plot_results(main_rows, factors, payload["feature_drift"], args.figure_dir)
    summary = {
        "schema_version": "heat3d_v6_p1i_controlled_cross_resolution_summary_v1",
        "status": "passed",
        "main": main_rows,
        "factors": factors,
        "attribution": attribution,
        "figures": figures,
        "contract": payload["contract"],
        "inputs": payload["inputs"],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.main_csv, main_rows)
    write_csv(args.factor_csv, factors)
    args.report_md.write_text(report_markdown(main_rows, factors, attribution, payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
