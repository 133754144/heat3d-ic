#!/usr/bin/env python3
"""Analyze Heat3D v1 medium run diagnostics.

This script summarizes loss_summary.json and baseline_comparison.json outputs
from medium-style diagnostic runs. It is run analysis tooling only, not formal
benchmark or model-performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRIC_SET = (
    "mean_T_rmse",
    "mean_T_mae",
    "mean_DeltaT_rmse",
    "mean_max_abs",
    "mean_p95_abs",
    "mean_peak_T_err",
    "mean_hotspot_dist",
)
METRIC_FIELD_ALIASES = {
    "mean_T_rmse": "mean_recovered_T_rmse",
    "mean_T_mae": "mean_recovered_T_mae",
    "mean_DeltaT_rmse": "mean_DeltaT_rmse",
    "mean_max_abs": "mean_max_abs_error",
    "mean_p95_abs": "mean_p95_abs_error",
    "mean_peak_T_err": "mean_peak_T_error",
    "mean_hotspot_dist": "mean_hotspot_coord_error",
}
CONDITION_KEYS = (
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
)
PREDICTOR_ZERO = "zero_delta"
PREDICTOR_TRAINED = "trained_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Heat3D v1 medium run loss and comparison summaries. "
            "Diagnostics only; not a formal benchmark."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--loss-summary", type=Path, default=None)
    parser.add_argument("--baseline-comparison", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--stdout-mode", choices=("compact", "full", "quiet"), default="compact")
    parser.add_argument(
        "--metric-set",
        nargs="*",
        default=None,
        help=(
            "Metrics to compare. Accepts space-separated names or comma-separated "
            "tokens. Defaults to medium comparison table metrics."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metric_set(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return list(DEFAULT_METRIC_SET)
    metrics: list[str] = []
    for token in tokens:
        for item in token.split(","):
            item = item.strip()
            if item:
                metrics.append(item)
    if not metrics:
        raise ValueError("--metric-set did not contain any metrics")
    return metrics


def _metric_field(metric: str) -> str:
    if metric in METRIC_FIELD_ALIASES:
        return METRIC_FIELD_ALIASES[metric]
    if metric.startswith("mean_"):
        return metric
    return f"mean_{metric}"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_change(trained: float | None, zero: float | None) -> float | None:
    if trained is None or zero is None:
        return None
    denominator = abs(zero)
    if denominator == 0.0:
        return 0.0 if trained == 0.0 else None
    return (trained - zero) / denominator


def _loss_change(first: float | None, last: float | None) -> dict[str, float | None]:
    rel = _relative_change(last, first)
    return {
        "initial": first,
        "final": last,
        "absolute_change": None if first is None or last is None else last - first,
        "relative_change": rel,
    }


def _sequence_summary(values: list[Any]) -> dict[str, float | int | None]:
    floats = [value for value in (_as_float(item) for item in values) if value is not None]
    if not floats:
        return {"count": 0, "first": None, "last": None, "min": None, "max": None}
    return {
        "count": len(floats),
        "first": floats[0],
        "last": floats[-1],
        "min": min(floats),
        "max": max(floats),
    }


def _history_field_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    return _sequence_summary([row.get(field) for row in rows if isinstance(row, dict)])


def analyze_loss_summary(loss_summary: dict[str, Any]) -> dict[str, Any]:
    train_losses = loss_summary.get("train_losses") or []
    valid_losses = loss_summary.get("valid_losses") or []
    train_metrics = loss_summary.get("train_metrics") or {}
    valid_metrics = loss_summary.get("valid_metrics") or {}
    epoch_history = loss_summary.get("epoch_history") or []
    final_train_components = loss_summary.get("final_train_loss_components") or {}
    final_valid_components = loss_summary.get("final_valid_loss_components") or {}

    train_initial = _as_float(train_losses[0]) if train_losses else None
    train_final = _as_float(train_losses[-1]) if train_losses else None
    valid_initial = _as_float(valid_losses[0]) if valid_losses else None
    valid_final = _as_float(valid_losses[-1]) if valid_losses else None

    history_trend: dict[str, Any] = {
        "present": bool(epoch_history),
        "record_count": len(epoch_history) if isinstance(epoch_history, list) else 0,
        "report_count": len(epoch_history) if isinstance(epoch_history, list) else 0,
    }
    if isinstance(epoch_history, list) and epoch_history:
        first = epoch_history[0]
        last = epoch_history[-1]
        history_trend.update(
            {
                "first_epoch": first.get("epoch"),
                "last_epoch": last.get("epoch"),
                "train_loss": _loss_change(_as_float(first.get("train_loss")), _as_float(last.get("train_loss"))),
                "valid_loss": _loss_change(_as_float(first.get("valid_loss")), _as_float(last.get("valid_loss"))),
            }
        )

    lr_history = loss_summary.get("lr_history") if isinstance(loss_summary.get("lr_history"), list) else []
    loss_weight_history = (
        loss_summary.get("loss_weight_history") if isinstance(loss_summary.get("loss_weight_history"), list) else []
    )
    if not loss_weight_history and isinstance(epoch_history, list):
        loss_weight_history = epoch_history
    stored_weight_summary = loss_summary.get("loss_weight_history_summary")
    if not isinstance(stored_weight_summary, dict):
        stored_weight_summary = {}

    return {
        "train_loss": _loss_change(train_initial, train_final),
        "valid_loss": _loss_change(valid_initial, valid_final),
        "final_train_raw_deltaT_mse": _as_float(train_metrics.get("raw_delta_mse")),
        "final_valid_raw_deltaT_mse": _as_float(valid_metrics.get("raw_delta_mse")),
        "final_train_recovered_T_mse": _as_float(train_metrics.get("recovered_temperature_mse")),
        "final_valid_recovered_T_mse": _as_float(valid_metrics.get("recovered_temperature_mse")),
        "grad_finite": loss_summary.get("grad_finite"),
        "status_ok": loss_summary.get("status_ok"),
        "loss_mode": loss_summary.get("loss_mode"),
        "lr_schedule": loss_summary.get("lr_schedule"),
        "loss_weight_schedule": loss_summary.get("loss_weight_schedule"),
        "selection_metric": loss_summary.get("selection_metric"),
        "best_epoch": loss_summary.get("best_epoch"),
        "best_valid_loss": _as_float(loss_summary.get("best_valid_loss")),
        "best_valid_raw_deltaT_mse": _as_float(loss_summary.get("best_valid_raw_deltaT_mse")),
        "best_valid_base_mse": _as_float(loss_summary.get("best_valid_base_mse")),
        "final_epoch": loss_summary.get("final_epoch"),
        "final_valid_loss": _as_float(loss_summary.get("final_valid_loss")),
        "best_predictions_saved": loss_summary.get("best_predictions_saved"),
        "best_predictions_path": loss_summary.get("best_predictions_path"),
        "lr_history": lr_history,
        "lr_history_summary": _sequence_summary(lr_history),
        "loss_weight_history": loss_weight_history,
        "relative_weight_summary": stored_weight_summary.get(
            "current_background_relative_weight",
            _history_field_summary(loss_weight_history, "current_background_relative_weight"),
        ),
        "hotspot_weight_summary": stored_weight_summary.get(
            "current_hotspot_weight",
            _history_field_summary(loss_weight_history, "current_hotspot_weight"),
        ),
        "final_train_background_relative_abs": _as_float(final_train_components.get("background_relative_abs")),
        "final_valid_background_relative_abs": _as_float(final_valid_components.get("background_relative_abs")),
        "epoch_history": epoch_history if isinstance(epoch_history, list) else [],
        "epoch_history_trend": history_trend,
    }


def _group_key(row: dict[str, Any], group_keys: tuple[str, ...]) -> tuple[str, ...]:
    if not group_keys:
        return ("overall",)
    return tuple(str(row.get(key, "unknown")) for key in group_keys)


def _group_dict(group_keys: tuple[str, ...], key_values: tuple[str, ...]) -> dict[str, str]:
    if not group_keys:
        return {"overall": "overall"}
    return {key: value for key, value in zip(group_keys, key_values)}


def compare_summary_rows(
    rows: list[dict[str, Any]],
    group_type: str,
    group_keys: tuple[str, ...],
    metrics: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        predictor = str(row.get("predictor", ""))
        grouped.setdefault(_group_key(row, group_keys), {})[predictor] = row

    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key_values, by_predictor in sorted(grouped.items()):
        zero_row = by_predictor.get(PREDICTOR_ZERO)
        trained_row = by_predictor.get(PREDICTOR_TRAINED)
        if zero_row is None or trained_row is None:
            warnings.append(
                f"{group_type} {_group_dict(group_keys, key_values)} missing "
                f"{PREDICTOR_ZERO if zero_row is None else PREDICTOR_TRAINED}"
            )
            continue

        metric_payload: dict[str, Any] = {}
        for metric in metrics:
            field = _metric_field(metric)
            zero = _as_float(zero_row.get(field))
            trained = _as_float(trained_row.get(field))
            metric_payload[metric] = {
                "json_field": field,
                "zero_delta": zero,
                "trained_prediction": trained,
                "absolute_delta": None if zero is None or trained is None else trained - zero,
                "relative_change": _relative_change(trained, zero),
            }

        comparisons.append(
            {
                "group_type": group_type,
                "group": _group_dict(group_keys, key_values),
                "sample_count": trained_row.get("sample_count", zero_row.get("sample_count")),
                "row_count": trained_row.get("row_count", zero_row.get("row_count")),
                "metrics": metric_payload,
            }
        )
    return comparisons, warnings


def _metric_change(comparison: dict[str, Any], metric: str) -> float | None:
    item = comparison.get("metrics", {}).get(metric, {})
    if not isinstance(item, dict):
        return None
    return _as_float(item.get("relative_change"))


def _flatten_condition_comparisons(condition_summary: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for condition_key, rows in condition_summary.items():
        for row in rows:
            item = dict(row)
            item["condition_key"] = condition_key
            item["condition_value"] = row.get("group", {}).get(condition_key, "unknown")
            flattened.append(item)
    return flattened


def _top_condition_groups(
    condition_summary: dict[str, list[dict[str, Any]]],
    metric: str,
    improved: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    candidates = []
    for row in _flatten_condition_comparisons(condition_summary):
        change = _metric_change(row, metric)
        if change is None:
            continue
        if improved and change >= 0.0:
            continue
        if not improved and change <= 0.0:
            continue
        metric_payload = row["metrics"][metric]
        candidates.append(
            {
                "condition_key": row["condition_key"],
                "condition_value": row["condition_value"],
                "metric": metric,
                "zero_delta": metric_payload.get("zero_delta"),
                "trained_prediction": metric_payload.get("trained_prediction"),
                "absolute_delta": metric_payload.get("absolute_delta"),
                "relative_change": change,
                "sample_count": row.get("sample_count"),
            }
        )
    return sorted(candidates, key=lambda item: item["relative_change"], reverse=not improved)[:limit]


def _overall_status(overall_comparisons: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    overall = overall_comparisons[0] if overall_comparisons else {"metrics": {}}
    improved = []
    degraded = []
    unchanged = []
    for metric in metrics:
        change = _metric_change(overall, metric)
        if change is None:
            continue
        if change < 0.0:
            improved.append(metric)
        elif change > 0.0:
            degraded.append(metric)
        else:
            unchanged.append(metric)

    def degraded_metric(metric: str) -> bool:
        change = _metric_change(overall, metric)
        return change is not None and change > 0.0

    def improved_metric(metric: str) -> bool:
        change = _metric_change(overall, metric)
        return change is not None and change < 0.0

    likely_hotspot = (
        degraded_metric("mean_T_rmse")
        and degraded_metric("mean_T_mae")
        and improved_metric("mean_peak_T_err")
        and improved_metric("mean_hotspot_dist")
    )
    return {
        "metrics_improved": improved,
        "metrics_degraded": degraded,
        "metrics_unchanged": unchanged,
        "mean_field_worse_but_peak_hotspot_better": likely_hotspot,
        "likely_hotspot_learning_with_background_bias": likely_hotspot,
    }


def analyze_baseline_comparison(baseline: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    warnings: list[str] = []
    overall, overall_warnings = compare_summary_rows(
        baseline.get("overall", []), "overall", (), metrics
    )
    warnings.extend(overall_warnings)
    split_summary, split_warnings = compare_summary_rows(
        baseline.get("split_summary", []), "split", ("split",), metrics
    )
    warnings.extend(split_warnings)

    condition_summary: dict[str, list[dict[str, Any]]] = {}
    raw_condition_summary = baseline.get("condition_summary", {})
    if isinstance(raw_condition_summary, dict):
        for key in CONDITION_KEYS:
            rows = raw_condition_summary.get(key, [])
            compared, condition_warnings = compare_summary_rows(rows, key, (key,), metrics)
            condition_summary[key] = compared
            warnings.extend(condition_warnings)

    return {
        "schema": {
            "overall": "list of summary rows grouped by predictor",
            "split_summary": "list of summary rows grouped by split and predictor",
            "condition_summary": "dict of condition key to summary rows grouped by condition value and predictor",
            "metric_aliases": {metric: _metric_field(metric) for metric in metrics},
        },
        "trained_comparison_status": baseline.get("trained_comparison_status"),
        "overall": overall,
        "overall_status": _overall_status(overall, metrics),
        "split_summary": split_summary,
        "condition_summary": condition_summary,
        "top_improved_groups": {
            "mean_T_rmse": _top_condition_groups(condition_summary, "mean_T_rmse", improved=True),
            "mean_peak_T_err": _top_condition_groups(condition_summary, "mean_peak_T_err", improved=True),
            "mean_hotspot_dist": _top_condition_groups(condition_summary, "mean_hotspot_dist", improved=True),
        },
        "top_degraded_groups": {
            "mean_T_rmse": _top_condition_groups(condition_summary, "mean_T_rmse", improved=False),
            "mean_T_mae": _top_condition_groups(condition_summary, "mean_T_mae", improved=False),
        },
        "warnings": warnings,
    }


def analyze_run(
    run_dir: Path,
    loss_summary_path: Path | None = None,
    baseline_comparison_path: Path | None = None,
    output_json_path: Path | None = None,
    output_md_path: Path | None = None,
    metric_set: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    loss_summary_path = loss_summary_path or run_dir / "loss_summary.json"
    baseline_comparison_path = baseline_comparison_path or run_dir / "baseline_comparison.json"
    output_json_path = output_json_path or run_dir / "run_analysis.json"
    output_md_path = output_md_path or run_dir / "run_analysis.md"
    metrics = metric_set or list(DEFAULT_METRIC_SET)

    loss_summary = analyze_loss_summary(_read_json(loss_summary_path))
    baseline_analysis = analyze_baseline_comparison(_read_json(baseline_comparison_path), metrics)
    payload = {
        "diagnostic_scope": "run analysis tooling; not formal benchmark or model-performance conclusion",
        "run_dir": str(run_dir),
        "inputs": {
            "loss_summary": str(loss_summary_path),
            "baseline_comparison": str(baseline_comparison_path),
        },
        "outputs": {
            "json": str(output_json_path),
            "markdown": str(output_md_path),
        },
        "metric_set": metrics,
        "loss_summary": loss_summary,
        "baseline_comparison": baseline_analysis,
        "recommended_next_experiments": [
            "error-binning / background-bias analysis",
            "optional e100/e200 comparison",
            "weighted loss / background penalty / residual learning",
        ],
    }
    _write_json(output_json_path, payload)
    _write_text(output_md_path, render_markdown(payload))
    return payload


def _fmt_float(value: Any) -> str:
    value = _as_float(value)
    if value is None:
        return "n/a"
    return f"{value:.8e}"


def _fmt_pct(value: Any) -> str:
    value = _as_float(value)
    if value is None:
        return "n/a"
    return f"{value * 100.0:+.2f}%"


def _comparison_table(comparisons: list[dict[str, Any]], metrics: list[str], group_label: str) -> list[str]:
    lines = [f"| {group_label} | metric | zero_delta | trained_prediction | abs delta | relative change |"]
    lines.append("|---|---|---:|---:|---:|---:|")
    for comparison in comparisons:
        group = comparison.get("group", {})
        group_value = " / ".join(f"{key}={value}" for key, value in group.items())
        for metric in metrics:
            item = comparison.get("metrics", {}).get(metric, {})
            lines.append(
                "| "
                f"{group_value} | {metric} | {_fmt_float(item.get('zero_delta'))} | "
                f"{_fmt_float(item.get('trained_prediction'))} | {_fmt_float(item.get('absolute_delta'))} | "
                f"{_fmt_pct(item.get('relative_change'))} |"
            )
    return lines


def _top_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| condition | value | metric | zero_delta | trained_prediction | relative change |")
    lines.append("|---|---|---|---:|---:|---:|")
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row['condition_key']} | {row['condition_value']} | {row['metric']} | "
            f"{_fmt_float(row.get('zero_delta'))} | {_fmt_float(row.get('trained_prediction'))} | "
            f"{_fmt_pct(row.get('relative_change'))} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metric_set"]
    loss = payload["loss_summary"]
    baseline = payload["baseline_comparison"]
    overall_status = baseline["overall_status"]
    lines: list[str] = [
        "# Heat3D v1 Medium Run Analysis",
        "",
        "This report is diagnostic run analysis tooling only. It is not a formal benchmark, model-performance conclusion, OOD generalization claim, or high-fidelity solver validation.",
        "",
        "## Run Summary",
        "",
        f"- run_dir: `{payload['run_dir']}`",
        f"- loss_summary: `{payload['inputs']['loss_summary']}`",
        f"- baseline_comparison: `{payload['inputs']['baseline_comparison']}`",
        f"- trained_comparison_status: `{baseline.get('trained_comparison_status')}`",
        f"- loss_mode: `{loss.get('loss_mode')}`",
        f"- lr_schedule: `{loss.get('lr_schedule')}`",
        f"- loss_weight_schedule: `{loss.get('loss_weight_schedule')}`",
        f"- selection_metric: `{loss.get('selection_metric')}`",
        f"- best_epoch: `{loss.get('best_epoch')}`",
        f"- best_valid_loss: `{_fmt_float(loss.get('best_valid_loss'))}`",
        f"- best_predictions_saved: `{loss.get('best_predictions_saved')}`",
        f"- likely_hotspot_learning_with_background_bias: `{overall_status['likely_hotspot_learning_with_background_bias']}`",
        "",
        "## Loss Trend",
        "",
        "| item | initial | final | abs change | relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("train_loss", "train loss"), ("valid_loss", "valid loss")):
        item = loss[key]
        lines.append(
            f"| {label} | {_fmt_float(item.get('initial'))} | {_fmt_float(item.get('final'))} | "
            f"{_fmt_float(item.get('absolute_change'))} | {_fmt_pct(item.get('relative_change'))} |"
        )
    lines.extend(
        [
            "",
            f"- final train raw DeltaT MSE: `{_fmt_float(loss.get('final_train_raw_deltaT_mse'))}`",
            f"- final valid raw DeltaT MSE: `{_fmt_float(loss.get('final_valid_raw_deltaT_mse'))}`",
            f"- final train recovered T MSE: `{_fmt_float(loss.get('final_train_recovered_T_mse'))}`",
            f"- final valid recovered T MSE: `{_fmt_float(loss.get('final_valid_recovered_T_mse'))}`",
            f"- final train background relative abs: `{_fmt_float(loss.get('final_train_background_relative_abs'))}`",
            f"- final valid background relative abs: `{_fmt_float(loss.get('final_valid_background_relative_abs'))}`",
            f"- lr history summary: `{loss.get('lr_history_summary')}`",
            f"- relative weight summary: `{loss.get('relative_weight_summary')}`",
            f"- hotspot weight summary: `{loss.get('hotspot_weight_summary')}`",
            f"- epoch_history record count: `{loss.get('epoch_history_trend', {}).get('record_count')}`",
            "",
            "## Overall Trained vs Zero-Delta Table",
            "",
        ]
    )
    lines.extend(_comparison_table(baseline["overall"], metrics, "group"))
    lines.extend(
        [
            "",
            "## Overall Relative Changes",
            "",
            f"- improved metrics: `{overall_status['metrics_improved']}`",
            f"- degraded metrics: `{overall_status['metrics_degraded']}`",
            f"- unchanged metrics: `{overall_status['metrics_unchanged']}`",
            "",
            "## Split-Wise Summary",
            "",
        ]
    )
    lines.extend(_comparison_table(baseline["split_summary"], metrics, "split"))
    lines.extend(["", "## Top Improved Condition Groups", ""])
    for metric, rows in baseline["top_improved_groups"].items():
        lines.extend(_top_table(f"Top improved by {metric}", rows))
        lines.append("")
    lines.extend(["## Top Degraded Condition Groups", ""])
    for metric, rows in baseline["top_degraded_groups"].items():
        lines.extend(_top_table(f"Top degraded by {metric}", rows))
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
        ]
    )
    if overall_status["mean_field_worse_but_peak_hotspot_better"]:
        lines.extend(
            [
                "The overall diagnostics show mean-field metrics worsening while peak/hotspot metrics improve.",
                "",
                "`likely_hotspot_learning_with_background_bias = true`",
                "",
                "Treat this as a diagnostic pattern only: it suggests hotspot-related signal may be learned while the recovered-temperature background has bias.",
            ]
        )
    else:
        lines.append("The configured mean-field vs hotspot conflict pattern was not detected in the overall summary.")
    lines.extend(
        [
            "",
            "## Recommended Next Experiments",
            "",
            "- error-binning / background-bias analysis",
            "- optional e100/e200 comparison",
            "- weighted loss / background penalty / residual learning",
            "",
        ]
    )
    return "\n".join(lines)


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _print_stdout_summary(payload: dict[str, Any], stdout_mode: str) -> None:
    loss = payload["loss_summary"]
    outputs = payload["outputs"]
    if stdout_mode == "quiet":
        _emit(f"run_analysis_written: json={outputs['json']} markdown={outputs['markdown']}")
        return

    _emit("Heat3D v1 medium run analysis tooling")
    _emit("  scope: diagnostics only; not formal benchmark or model-performance conclusion")
    _emit(f"  run_dir: {payload['run_dir']}")
    _emit(
        "  loss: "
        f"mode={loss.get('loss_mode')} lr_schedule={loss.get('lr_schedule')} "
        f"loss_weight_schedule={loss.get('loss_weight_schedule')}"
    )
    _emit(
        "  final: "
        f"train_loss={_fmt_float(loss.get('train_loss', {}).get('final'))} "
        f"valid_loss={_fmt_float(loss.get('valid_loss', {}).get('final'))}"
    )
    _emit(
        "  best-valid: "
        f"metric={loss.get('selection_metric')} epoch={loss.get('best_epoch')} "
        f"best_valid_loss={_fmt_float(loss.get('best_valid_loss'))} "
        f"best_predictions_saved={loss.get('best_predictions_saved')}"
    )
    if stdout_mode == "full":
        _emit(f"  final_valid_raw_deltaT_mse: {_fmt_float(loss.get('final_valid_raw_deltaT_mse'))}")
        _emit(f"  final_valid_background_relative_abs: {_fmt_float(loss.get('final_valid_background_relative_abs'))}")
        _emit(f"  lr_history_summary: {loss.get('lr_history_summary')}")
        _emit(f"  relative_weight_summary: {loss.get('relative_weight_summary')}")
        _emit(f"  hotspot_weight_summary: {loss.get('hotspot_weight_summary')}")
    _emit(f"  output_json: {outputs['json']}")
    _emit(f"  output_md: {outputs['markdown']}")
    _emit("  analysis_written: True")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output_json = args.output_json or run_dir / "run_analysis.json"
    output_md = args.output_md or run_dir / "run_analysis.md"
    payload = analyze_run(
        run_dir=run_dir,
        loss_summary_path=args.loss_summary,
        baseline_comparison_path=args.baseline_comparison,
        output_json_path=output_json,
        output_md_path=output_md,
        metric_set=_metric_set(args.metric_set),
    )
    _print_stdout_summary(payload, args.stdout_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
