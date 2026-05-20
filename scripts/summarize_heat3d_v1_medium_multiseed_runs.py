#!/usr/bin/env python3
"""Summarize Heat3D v1 medium multi-seed diagnostic runs.

This script aggregates existing medium-style run artifacts. It is diagnostic
tooling only, not a formal benchmark or model-performance conclusion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any


PREDICTOR_TRAINED = "trained_prediction"
PREDICTOR_ZERO = "zero_delta"

OVERALL_METRICS = (
    "mean_T_rmse",
    "mean_T_mae",
    "mean_DeltaT_rmse",
    "mean_p95_abs",
    "mean_peak_T_err",
    "mean_hotspot_dist",
)
SPLIT_METRICS = (
    "mean_T_rmse",
    "mean_T_mae",
    "mean_p95_abs",
    "mean_peak_T_err",
    "mean_hotspot_dist",
)
SPLITS = (
    "train",
    "valid",
    "test_id",
    "test_ood_bc_candidate",
    "test_ood_stack_candidate",
)
METRIC_FIELD_ALIASES = {
    "mean_T_rmse": "mean_recovered_T_rmse",
    "mean_T_mae": "mean_recovered_T_mae",
    "mean_DeltaT_rmse": "mean_DeltaT_rmse",
    "mean_p95_abs": "mean_p95_abs_error",
    "mean_peak_T_err": "mean_peak_T_error",
    "mean_hotspot_dist": "mean_hotspot_coord_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Heat3D v1 medium multi-seed run diagnostics. "
            "Diagnostics only; not a formal benchmark."
        )
    )
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label=path entry, for example seed0=output/run_seed0.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt_float(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.8e}"


def _fmt_pct(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number * 100.0:+.2f}%"


def _seed_from_label_or_path(label: str, run_dir: Path) -> int | None:
    for text in (label, run_dir.name, str(run_dir)):
        match = re.search(r"seed[_-]?(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def _label_for_run(run_dir: Path, used: set[str]) -> str:
    seed = _seed_from_label_or_path(run_dir.name, run_dir)
    base = f"seed{seed}" if seed is not None else run_dir.name
    label = base
    index = 2
    while label in used:
        label = f"{base}_{index}"
        index += 1
    used.add(label)
    return label


def _run_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    used: set[str] = set()
    for run_dir in args.run_dir:
        label = _label_for_run(run_dir, used)
        specs.append((label, run_dir))
    for item in args.label:
        if "=" not in item:
            raise ValueError(f"--label must be label=path, found: {item}")
        label, path = item.split("=", 1)
        if not label:
            raise ValueError(f"--label has empty label: {item}")
        if label in used:
            raise ValueError(f"duplicate run label: {label}")
        used.add(label)
        specs.append((label, Path(path)))
    if not specs:
        raise ValueError("Provide at least one --run-dir or --label label=path")
    return specs


def _row_by_predictor(rows: list[dict[str, Any]], predictor: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("predictor")) == predictor:
            return row
    return None


def _metric_from_row(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    return _as_float(row.get(METRIC_FIELD_ALIASES.get(metric, metric)))


def _relative_change(trained: float | None, zero: float | None) -> float | None:
    if trained is None or zero is None:
        return None
    denominator = abs(zero)
    if denominator == 0.0:
        return 0.0 if trained == 0.0 else None
    return (trained - zero) / denominator


def _overall_payload(baseline: dict[str, Any] | None) -> tuple[dict[str, float | None], dict[str, float | None]]:
    rows = baseline.get("overall", []) if baseline else []
    trained = _row_by_predictor(rows, PREDICTOR_TRAINED) if isinstance(rows, list) else None
    zero = _row_by_predictor(rows, PREDICTOR_ZERO) if isinstance(rows, list) else None
    metrics = {metric: _metric_from_row(trained, metric) for metric in OVERALL_METRICS}
    changes = {
        f"{metric}_relative_change": _relative_change(metrics[metric], _metric_from_row(zero, metric))
        for metric in OVERALL_METRICS
    }
    return metrics, changes


def _split_payload(baseline: dict[str, Any] | None) -> dict[str, dict[str, float | None]]:
    rows = baseline.get("split_summary", []) if baseline else []
    result: dict[str, dict[str, float | None]] = {}
    if not isinstance(rows, list):
        return result
    for split in SPLITS:
        trained = _row_by_predictor([row for row in rows if str(row.get("split")) == split], PREDICTOR_TRAINED)
        result[split] = {metric: _metric_from_row(trained, metric) for metric in SPLIT_METRICS}
    return result


def _bin_by_name(error_bins: dict[str, Any] | None, bin_name: str) -> dict[str, Any] | None:
    bins = (((error_bins or {}).get("overall") or {}).get("bins") or [])
    if not isinstance(bins, list):
        return None
    for item in bins:
        if isinstance(item, dict) and item.get("bin_name") == bin_name:
            return item
    return None


def _error_bin_payload(error_bins: dict[str, Any] | None) -> dict[str, float | None]:
    bin0 = _bin_by_name(error_bins, "bin_0")
    bin1 = _bin_by_name(error_bins, "bin_1")
    bin3 = _bin_by_name(error_bins, "bin_3")
    bin4 = _bin_by_name(error_bins, "bin_4")
    return {
        "bin_0_trained_bias": _as_float((bin0 or {}).get("trained_signed_bias")),
        "bin_0_over_ratio": _as_float((bin0 or {}).get("trained_overprediction_ratio")),
        "bin_0_relative_rmse_change": _as_float((bin0 or {}).get("relative_rmse_change")),
        "bin_0_relative_mae_change": _as_float((bin0 or {}).get("relative_mae_change")),
        "bin_1_trained_bias": _as_float((bin1 or {}).get("trained_signed_bias")),
        "bin_1_relative_mae_change": _as_float((bin1 or {}).get("relative_mae_change")),
        "bin_3_trained_bias": _as_float((bin3 or {}).get("trained_signed_bias")),
        "bin_3_relative_mae_change": _as_float((bin3 or {}).get("relative_mae_change")),
        "bin_4_trained_bias": _as_float((bin4 or {}).get("trained_signed_bias")),
        "bin_4_relative_mae_change": _as_float((bin4 or {}).get("relative_mae_change")),
        "bin_4_under_ratio": _as_float((bin4 or {}).get("trained_underprediction_ratio")),
    }


def _loss_payload(loss_summary: dict[str, Any] | None) -> dict[str, float | None]:
    valid_metrics = (loss_summary or {}).get("valid_metrics") or {}
    components = (loss_summary or {}).get("final_valid_loss_components") or {}
    return {
        "final_valid_raw_DeltaT_mse": _as_float(valid_metrics.get("raw_delta_mse")),
        "final_valid_background_signed_bias": _as_float(components.get("bg_signed_bias")),
        "final_valid_background_relative_abs": _as_float(components.get("background_relative_abs")),
        "final_valid_hotspot_raw_mae": _as_float(components.get("hotspot_raw_mae")),
    }


def _run_config_payload(run_config: dict[str, Any] | None, loss_summary: dict[str, Any] | None) -> dict[str, Any]:
    source = run_config or {}
    fallback = loss_summary or {}
    return {
        "loss_mode": source.get("loss_mode", fallback.get("loss_mode")),
        "lr": source.get("lr", fallback.get("lr")),
        "epochs": source.get("epochs", fallback.get("epochs")),
        "loss_weight_schedule": source.get("loss_weight_schedule", fallback.get("loss_weight_schedule")),
    }


def _load_run(label: str, run_dir: Path) -> dict[str, Any]:
    baseline = _read_json(run_dir / "baseline_comparison.json")
    error_bins = _read_json(run_dir / "error_bins.json")
    run_analysis = _read_json(run_dir / "run_analysis.json")
    loss_summary = _read_json(run_dir / "loss_summary.json")
    run_config = _read_json(run_dir / "run_config.json")
    if baseline is None:
        raise FileNotFoundError(f"Missing baseline_comparison.json in {run_dir}")
    if error_bins is None:
        raise FileNotFoundError(f"Missing error_bins.json in {run_dir}")

    overall_metrics, overall_relative_changes = _overall_payload(baseline)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "seed": _seed_from_label_or_path(label, run_dir),
        "config": _run_config_payload(run_config, loss_summary),
        "artifact_present": {
            "baseline_comparison": baseline is not None,
            "error_bins": error_bins is not None,
            "run_analysis": run_analysis is not None,
            "loss_summary": loss_summary is not None,
            "run_config": run_config is not None,
        },
        "overall_metrics": overall_metrics,
        "overall_relative_changes": overall_relative_changes,
        "split_metrics": _split_payload(baseline),
        "error_bin_metrics": _error_bin_payload(error_bins),
        "loss_metrics": _loss_payload(loss_summary),
    }


def _metric_direction(metric: str) -> str:
    if metric.endswith("_trained_bias") or metric.endswith("_bias"):
        return "abs_lower"
    if "over_ratio" in metric or "under_ratio" in metric:
        return "lower"
    if "relative_" in metric or metric.endswith("_relative_change"):
        return "lower"
    return "lower"


def _sort_key(value: float, direction: str) -> float:
    return abs(value) if direction == "abs_lower" else value


def _stats_for_metric(runs: list[dict[str, Any]], path: tuple[str, ...], metric: str) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for run in runs:
        item: Any = run
        for key in path:
            item = item.get(key, {}) if isinstance(item, dict) else {}
        value = _as_float(item.get(metric)) if isinstance(item, dict) else None
        if value is not None:
            values.append((run["label"], value))
    if not values:
        return {
            "metric": metric,
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "best_run": None,
            "worst_run": None,
            "median_value": None,
            "median_run": None,
        }

    numeric = [value for _, value in values]
    mean_value = sum(numeric) / len(numeric)
    std_value = math.sqrt(sum((value - mean_value) ** 2 for value in numeric) / len(numeric)) if len(numeric) > 1 else 0.0
    direction = _metric_direction(metric)
    ordered = sorted(values, key=lambda item: _sort_key(item[1], direction))
    median_value = float(median(numeric))
    median_run = min(values, key=lambda item: abs(item[1] - median_value))[0]
    return {
        "metric": metric,
        "count": len(values),
        "mean": mean_value,
        "std": std_value,
        "min": min(numeric),
        "max": max(numeric),
        "best_run": ordered[0][0],
        "worst_run": ordered[-1][0],
        "median_value": median_value,
        "median_run": median_run,
        "direction": direction,
    }


def _stats_table(runs: list[dict[str, Any]], path: tuple[str, ...], metrics: tuple[str, ...]) -> dict[str, Any]:
    return {metric: _stats_for_metric(runs, path, metric) for metric in metrics}


def _split_stats(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split: _stats_table(runs, ("split_metrics", split), SPLIT_METRICS)
        for split in SPLITS
    }


def _all_seeds_beat_zero_delta(runs: list[dict[str, Any]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for metric in OVERALL_METRICS:
        key = f"{metric}_relative_change"
        changes = [run["overall_relative_changes"].get(key) for run in runs]
        finite = [value for value in (_as_float(item) for item in changes) if value is not None]
        result[metric] = bool(finite) and len(finite) == len(runs) and all(value < 0.0 for value in finite)
    return result


def _interpretation(runs: list[dict[str, Any]], overall_stats: dict[str, Any], split_stats: dict[str, Any]) -> dict[str, Any]:
    beat_zero = _all_seeds_beat_zero_delta(runs)
    best_counts: dict[str, int] = {}
    for item in overall_stats.values():
        best = item.get("best_run")
        if best:
            best_counts[best] = best_counts.get(best, 0) + 1
    best_by_count = max(best_counts.items(), key=lambda item: item[1])[0] if best_counts else None
    rmse_stats = overall_stats.get("mean_T_rmse", {})
    rmse_mean = _as_float(rmse_stats.get("mean"))
    rmse_std = _as_float(rmse_stats.get("std"))
    cv = (rmse_std / abs(rmse_mean)) if rmse_mean not in (None, 0.0) and rmse_std is not None else None
    seed_sensitivity = bool(cv is not None and cv > 0.05)
    weak_splits = []
    for split, metrics in split_stats.items():
        rmse = _as_float((metrics.get("mean_T_rmse") or {}).get("mean"))
        overall_rmse = _as_float(rmse_stats.get("mean"))
        if rmse is not None and overall_rmse is not None and rmse > overall_rmse:
            weak_splits.append(split)
    bin0_bias = _stats_for_metric(runs, ("error_bin_metrics",), "bin_0_trained_bias")
    bin0_over = _stats_for_metric(runs, ("error_bin_metrics",), "bin_0_over_ratio")
    return {
        "all_seeds_beat_zero_delta_by_overall_metric": beat_zero,
        "all_seeds_beat_zero_delta_on_all_requested_metrics": all(beat_zero.values()) if beat_zero else False,
        "seed_sensitivity_present": seed_sensitivity,
        "mean_T_rmse_cv": cv,
        "best_run_by_overall_metric_count": best_by_count,
        "seed0_best_count": best_counts.get("seed0", 0),
        "seed0_outlier_like": best_counts.get("seed0", 0) >= max(3, len(OVERALL_METRICS) // 2),
        "weak_split_candidates_by_mean_rmse": weak_splits,
        "low_delta_background_overprediction_remains": (
            (_as_float(bin0_bias.get("mean")) or 0.0) > 0.0
            or (_as_float(bin0_over.get("mean")) or 0.0) > 0.5
        ),
        "notes": [
            "Interpretation is diagnostic only; not a formal benchmark.",
            "High-bin under ratio is reported as a diagnostic, not an automatically bad direction.",
            "Seed sensitivity should be judged with metric tables and error-bin details, not a single best seed.",
        ],
    }


def build_summary(run_specs: list[tuple[str, Path]], output_json: Path, output_md: Path) -> dict[str, Any]:
    runs = [_load_run(label, run_dir) for label, run_dir in run_specs]
    overall_stats = _stats_table(runs, ("overall_metrics",), OVERALL_METRICS)
    split_stats = _split_stats(runs)
    error_bin_metrics = (
        "bin_0_trained_bias",
        "bin_0_over_ratio",
        "bin_0_relative_rmse_change",
        "bin_0_relative_mae_change",
        "bin_1_trained_bias",
        "bin_1_relative_mae_change",
        "bin_3_trained_bias",
        "bin_3_relative_mae_change",
        "bin_4_trained_bias",
        "bin_4_relative_mae_change",
        "bin_4_under_ratio",
    )
    error_bin_stats = _stats_table(runs, ("error_bin_metrics",), error_bin_metrics)
    loss_stats = _stats_table(
        runs,
        ("loss_metrics",),
        (
            "final_valid_raw_DeltaT_mse",
            "final_valid_background_signed_bias",
            "final_valid_background_relative_abs",
            "final_valid_hotspot_raw_mae",
        ),
    )
    payload = {
        "diagnostic_scope": "Heat3D v1 medium multi-seed summary; not a formal benchmark",
        "inputs": {
            "run_dirs": [{"label": label, "run_dir": str(path)} for label, path in run_specs],
            "output_json": str(output_json),
            "output_md": str(output_md),
        },
        "runs": runs,
        "overall_summary": overall_stats,
        "split_summary": split_stats,
        "error_bin_summary": error_bin_stats,
        "loss_summary": loss_stats,
        "interpretation": _interpretation(runs, overall_stats, split_stats),
        "recommended_next_actions": [
            "Run seed=3 or more seeds only if uncertainty remains decision-critical.",
            "Consider medium512/medium1024 only after medium256 multi-seed stability is summarized.",
            "Audit optimizer stability and checkpoint selection before more loss tuning.",
            "Keep zero_delta, MSE, and background_l1_relative comparisons in a fixed baseline protocol.",
            "Do not overclaim publication readiness from one strong seed.",
        ],
    }
    return payload


def _markdown_stats_table(stats: dict[str, Any], title_metric_label: str = "metric") -> list[str]:
    lines = [
        f"| {title_metric_label} | count | mean | std | min | max | best_run | worst_run |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for metric, item in stats.items():
        lines.append(
            f"| {metric} | {item.get('count', 0)} | {_fmt_float(item.get('mean'))} | "
            f"{_fmt_float(item.get('std'))} | {_fmt_float(item.get('min'))} | "
            f"{_fmt_float(item.get('max'))} | {item.get('best_run') or 'n/a'} | "
            f"{item.get('worst_run') or 'n/a'} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v1 Medium Multi-Seed Summary",
        "",
        "This is diagnostic tooling only; not a formal benchmark.",
        "",
        "## Run List",
        "",
        "| run label | run_dir | seed | loss_mode | lr | epochs | loss_weight_schedule |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for run in payload["runs"]:
        config = run.get("config", {})
        lines.append(
            f"| {run['label']} | `{run['run_dir']}` | {run.get('seed') if run.get('seed') is not None else 'n/a'} | "
            f"{config.get('loss_mode') or 'n/a'} | {_fmt_float(config.get('lr'))} | "
            f"{config.get('epochs') if config.get('epochs') is not None else 'n/a'} | "
            f"{config.get('loss_weight_schedule') or 'n/a'} |"
        )

    lines.extend(["", "## Overall Multi-Seed Summary", ""])
    lines.extend(_markdown_stats_table(payload["overall_summary"]))

    lines.extend(
        [
            "",
            "## Per-Run Overall Table",
            "",
            "| run | mean_T_rmse | mean_T_mae | p95 | peak | hotspot |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["runs"]:
        metrics = run["overall_metrics"]
        lines.append(
            f"| {run['label']} | {_fmt_float(metrics.get('mean_T_rmse'))} | "
            f"{_fmt_float(metrics.get('mean_T_mae'))} | {_fmt_float(metrics.get('mean_p95_abs'))} | "
            f"{_fmt_float(metrics.get('mean_peak_T_err'))} | {_fmt_float(metrics.get('mean_hotspot_dist'))} |"
        )

    lines.extend(["", "## Split-Wise Summary", ""])
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        lines.extend(_markdown_stats_table(payload["split_summary"].get(split, {})))
        lines.append("")

    lines.extend(["", "## Error-Bin Summary", ""])
    lines.extend(_markdown_stats_table(payload["error_bin_summary"]))

    lines.extend(["", "## Training/Loss Final Metrics", ""])
    lines.extend(_markdown_stats_table(payload["loss_summary"]))

    interp = payload["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- all seeds beat zero_delta on all requested overall metrics: `{interp['all_seeds_beat_zero_delta_on_all_requested_metrics']}`",
            f"- per-metric zero_delta improvement flags: `{interp['all_seeds_beat_zero_delta_by_overall_metric']}`",
            f"- seed sensitivity present: `{interp['seed_sensitivity_present']}`",
            f"- mean_T_rmse CV: `{_fmt_float(interp.get('mean_T_rmse_cv'))}`",
            f"- best run by overall metric count: `{interp.get('best_run_by_overall_metric_count')}`",
            f"- seed0 outlier-like by best-count heuristic: `{interp.get('seed0_outlier_like')}`",
            f"- weak split candidates by mean RMSE: `{interp.get('weak_split_candidates_by_mean_rmse')}`",
            f"- low-DeltaT background overprediction remains: `{interp.get('low_delta_background_overprediction_remains')}`",
            "",
            "These interpretations are diagnostic aids only. They do not establish formal benchmark status, OOD generalization, or publication-ready model performance.",
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["recommended_next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_specs = _run_specs(args)
    payload = build_summary(run_specs, args.output_json, args.output_md)
    _write_json(args.output_json, payload)
    _write_text(args.output_md, render_markdown(payload))
    print("Heat3D v1 medium multi-seed summary")
    print("  scope: diagnostics only; not formal benchmark or model-performance conclusion")
    print(f"  run_count: {len(payload['runs'])}")
    print(f"  output_json: {args.output_json}")
    print(f"  output_md: {args.output_md}")
    print("  summary_written: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
