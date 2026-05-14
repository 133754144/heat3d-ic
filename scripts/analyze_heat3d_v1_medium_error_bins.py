#!/usr/bin/env python3
"""Heat3D v1 medium error-binning / background-bias diagnostics tooling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import find_sample_dirs, load_json, resolve_t_ref  # noqa: E402


DEFAULT_BINS = "p50,p75,p90,p95"
DEFAULT_GROUP_BY = (
    "split",
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
            "Analyze Heat3D v1 medium error bins and background-bias diagnostics. "
            "Diagnostics only; not a formal benchmark."
        )
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--trained-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--bins", type=str, default=DEFAULT_BINS)
    parser.add_argument("--group-by", nargs="*", default=None)
    parser.add_argument("--stdout-mode", choices=("compact", "full", "quiet"), default="compact")
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _plan(meta: dict[str, Any]) -> dict[str, Any]:
    generation_config = meta.get("generation_config", {})
    if isinstance(generation_config, dict):
        sample_plan = generation_config.get("sample_plan", {})
        if isinstance(sample_plan, dict):
            return sample_plan
    return {}


def _condition_value(meta: dict[str, Any], key: str) -> str:
    if key == "split":
        return str(meta.get("split", "unknown"))
    if key == "stack_template":
        stack = meta.get("stack", {})
        if isinstance(stack, dict) and stack.get("stack_template") is not None:
            return str(stack["stack_template"])
    value = _plan(meta).get(key)
    if value is None:
        value = "unknown"
    return str(value)


def _parse_group_by(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return list(DEFAULT_GROUP_BY)
    result: list[str] = []
    for token in tokens:
        for item in token.split(","):
            item = item.strip()
            if item:
                result.append(item)
    if not result:
        raise ValueError("--group-by did not contain any group keys")
    return result


def _parse_percentile_bins(spec: str) -> list[float]:
    percentiles = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.startswith("p"):
            token = token[1:]
        value = float(token)
        if value <= 0.0 or value >= 100.0:
            raise ValueError("--bins percentiles must be between 0 and 100")
        percentiles.append(value)
    if not percentiles:
        raise ValueError("--bins must include at least one percentile")
    return sorted(set(percentiles))


def _prediction_loader(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"trained predictions path does not exist: {path}")
    if path.is_file() and path.suffix == ".npz":
        archive = np.load(path)

        def load_from_npz(sample_id: str) -> np.ndarray:
            if sample_id not in archive:
                raise KeyError(f"trained predictions .npz missing key {sample_id}")
            return np.asarray(archive[sample_id])

        return load_from_npz
    if path.is_dir():

        def load_from_dir(sample_id: str) -> np.ndarray:
            candidates = (
                path / f"{sample_id}.npy",
                path / sample_id / "temperature.npy",
                path / sample_id / "pred_temperature.npy",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return np.load(candidate)
            raise FileNotFoundError(f"trained prediction for {sample_id} not found under {path}")

        return load_from_dir
    raise ValueError(f"unsupported trained predictions format: {path}; expected .npz or directory")


def _as_column(array: np.ndarray, n_points: int, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape == (n_points,):
        values = values.reshape(n_points, 1)
    if values.shape != (n_points, 1):
        raise ValueError(f"{name} must have shape ({n_points}, 1) or ({n_points},), found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _load_sample_records(subset: Path, trained_predictions: Path, group_by: list[str]) -> list[dict[str, Any]]:
    sample_dirs = find_sample_dirs(_sample_root(subset))
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {subset}")
    load_prediction = _prediction_loader(trained_predictions)
    records = []
    for sample_dir in sample_dirs:
        meta = load_json(sample_dir / "sample_meta.json")
        sample_id = str(meta.get("sample_id", sample_dir.name))
        coords = np.load(sample_dir / "coords.npy")
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{sample_id}: coords.npy must have shape (N, 3), found {coords.shape}")
        n_points = coords.shape[0]
        true_temperature = _as_column(np.load(sample_dir / "temperature.npy"), n_points, f"{sample_id} temperature.npy")
        trained_temperature = _as_column(load_prediction(sample_id), n_points, f"{sample_id} trained prediction")
        t_ref_info = resolve_t_ref(meta)
        t_ref = float(t_ref_info["value"])
        true_delta = true_temperature - t_ref
        zero_temperature = np.full_like(true_temperature, t_ref)
        groups = {key: _condition_value(meta, key) for key in group_by}
        records.append(
            {
                "sample_id": sample_id,
                "point_count": int(n_points),
                "T_ref": t_ref,
                "T_ref_source": t_ref_info.get("source"),
                "groups": groups,
                "T_true": true_temperature.reshape(-1),
                "DeltaT_true": true_delta.reshape(-1),
                "T_pred_trained": trained_temperature.reshape(-1),
                "T_pred_zero": zero_temperature.reshape(-1),
            }
        )
    return records


def _bin_edges(delta_values: np.ndarray, percentiles: list[float]) -> dict[str, Any]:
    edges = [float(np.min(delta_values))]
    edges.extend(float(value) for value in np.percentile(delta_values, percentiles))
    edges.append(float(np.max(delta_values)))
    labels = ["min", *(f"p{percentile:g}" for percentile in percentiles), "max"]
    bins = []
    for idx in range(len(edges) - 1):
        left = labels[idx]
        right = labels[idx + 1]
        bins.append(
            {
                "bin_index": idx,
                "bin_name": f"bin_{idx}",
                "left_label": left,
                "right_label": right,
                "lower": edges[idx],
                "upper": edges[idx + 1],
                "interval": f"{'[' if idx == 0 else '('}{left}, {right}]",
            }
        )
    return {"percentiles": percentiles, "labels": labels, "edges": edges, "bins": bins}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def _relative_change(trained: float, zero: float) -> float | None:
    if not np.isfinite(trained) or not np.isfinite(zero):
        return None
    denominator = abs(zero)
    if denominator == 0.0:
        return 0.0 if trained == 0.0 else None
    return float((trained - zero) / denominator)


def _point_mask(delta: np.ndarray, lower: float, upper: float, bin_index: int) -> np.ndarray:
    if bin_index == 0:
        return (delta >= lower) & (delta <= upper)
    return (delta > lower) & (delta <= upper)


def _empty_bin(bin_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **bin_spec,
        "point_count": 0,
        "sample_count": 0,
        "DeltaT_min": None,
        "DeltaT_max": None,
        "DeltaT_mean": None,
        "zero_delta_rmse": None,
        "zero_delta_mae": None,
        "trained_rmse": None,
        "trained_mae": None,
        "trained_signed_bias": None,
        "zero_signed_bias": None,
        "trained_overprediction_ratio": None,
        "trained_underprediction_ratio": None,
        "relative_rmse_change": None,
        "relative_mae_change": None,
    }


def _bin_stats(records: list[dict[str, Any]], bin_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for bin_spec in bin_specs:
        deltas = []
        trained_errors = []
        zero_errors = []
        sample_ids = set()
        for record in records:
            delta = record["DeltaT_true"]
            mask = _point_mask(delta, bin_spec["lower"], bin_spec["upper"], bin_spec["bin_index"])
            if not np.any(mask):
                continue
            sample_ids.add(record["sample_id"])
            deltas.append(delta[mask])
            trained_errors.append(record["T_pred_trained"][mask] - record["T_true"][mask])
            zero_errors.append(record["T_pred_zero"][mask] - record["T_true"][mask])
        if not deltas:
            result.append(_empty_bin(bin_spec))
            continue
        delta_all = np.concatenate(deltas)
        trained_error_all = np.concatenate(trained_errors)
        zero_error_all = np.concatenate(zero_errors)
        zero_rmse = _rmse(zero_error_all)
        trained_rmse = _rmse(trained_error_all)
        zero_mae = _mae(zero_error_all)
        trained_mae = _mae(trained_error_all)
        result.append(
            {
                **bin_spec,
                "point_count": int(delta_all.size),
                "sample_count": int(len(sample_ids)),
                "DeltaT_min": float(np.min(delta_all)),
                "DeltaT_max": float(np.max(delta_all)),
                "DeltaT_mean": float(np.mean(delta_all)),
                "zero_delta_rmse": zero_rmse,
                "zero_delta_mae": zero_mae,
                "trained_rmse": trained_rmse,
                "trained_mae": trained_mae,
                "trained_signed_bias": float(np.mean(trained_error_all)),
                "zero_signed_bias": float(np.mean(zero_error_all)),
                "trained_overprediction_ratio": float(np.mean(trained_error_all > 0.0)),
                "trained_underprediction_ratio": float(np.mean(trained_error_all < 0.0)),
                "relative_rmse_change": _relative_change(trained_rmse, zero_rmse),
                "relative_mae_change": _relative_change(trained_mae, zero_mae),
            }
        )
    return result


def _group_records(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["groups"].get(key, "unknown"))].append(record)
    return dict(sorted(grouped.items()))


def _groupwise_bins(records: list[dict[str, Any]], group_by: list[str], bin_specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped_payload: dict[str, list[dict[str, Any]]] = {}
    for key in group_by:
        grouped_payload[key] = [
            {
                "group_key": key,
                "group_value": value,
                "sample_count": len(group_records),
                "bins": _bin_stats(group_records, bin_specs),
            }
            for value, group_records in _group_records(records, key).items()
        ]
    return grouped_payload


def _finite_values(items: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def _interpret_bins(overall_bins: list[dict[str, Any]]) -> dict[str, Any]:
    if not overall_bins:
        return {
            "likely_background_overprediction": False,
            "likely_hotspot_region_improvement": False,
            "likely_hotspot_learning_with_background_bias": False,
        }
    split_index = max(1, len(overall_bins) // 2)
    low_bins = overall_bins[:split_index]
    high_bins = overall_bins[-2:] if len(overall_bins) >= 2 else overall_bins
    low_rmse_changes = _finite_values(low_bins, "relative_rmse_change")
    low_mae_changes = _finite_values(low_bins, "relative_mae_change")
    low_biases = _finite_values(low_bins, "trained_signed_bias")
    high_rmse_changes = _finite_values(high_bins, "relative_rmse_change")
    high_mae_changes = _finite_values(high_bins, "relative_mae_change")

    low_background_bins = [
        item
        for item in low_bins
        if (
            ((item.get("relative_rmse_change") or 0.0) > 0.0 or (item.get("relative_mae_change") or 0.0) > 0.0)
            and (item.get("trained_signed_bias") or 0.0) > 0.0
        )
    ]
    high_improved_bins = [
        item
        for item in high_bins
        if (item.get("relative_rmse_change") is not None and item["relative_rmse_change"] < 0.0)
        or (item.get("relative_mae_change") is not None and item["relative_mae_change"] < 0.0)
    ]
    likely_background = bool(low_background_bins)
    likely_hotspot = bool(high_improved_bins)
    return {
        "low_bin_names": [item["bin_name"] for item in low_bins],
        "high_bin_names": [item["bin_name"] for item in high_bins],
        "background_overprediction_bin_names": [item["bin_name"] for item in low_background_bins],
        "hotspot_improvement_bin_names": [item["bin_name"] for item in high_improved_bins],
        "low_bins_mean_relative_rmse_change": float(np.mean(low_rmse_changes)) if low_rmse_changes else None,
        "low_bins_mean_relative_mae_change": float(np.mean(low_mae_changes)) if low_mae_changes else None,
        "low_bins_mean_trained_signed_bias": float(np.mean(low_biases)) if low_biases else None,
        "high_bins_mean_relative_rmse_change": float(np.mean(high_rmse_changes)) if high_rmse_changes else None,
        "high_bins_mean_relative_mae_change": float(np.mean(high_mae_changes)) if high_mae_changes else None,
        "likely_background_overprediction": bool(likely_background),
        "likely_hotspot_region_improvement": bool(likely_hotspot),
        "likely_hotspot_learning_with_background_bias": bool(likely_background and likely_hotspot),
    }


def analyze_error_bins(
    *,
    subset: Path,
    trained_predictions: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
    bins: str = DEFAULT_BINS,
    group_by: list[str] | None = None,
) -> dict[str, Any]:
    group_keys = _parse_group_by(group_by)
    percentiles = _parse_percentile_bins(bins)
    output_json = output_json or trained_predictions.parent / "error_bins.json"
    output_md = output_md or trained_predictions.parent / "error_bins.md"
    records = _load_sample_records(subset, trained_predictions, group_keys)
    all_delta = np.concatenate([record["DeltaT_true"] for record in records])
    edge_payload = _bin_edges(all_delta, percentiles)
    overall_bins = _bin_stats(records, edge_payload["bins"])
    payload = {
        "diagnostic_scope": "error-binning / background-bias diagnostics tooling; not formal benchmark or model-performance conclusion",
        "inputs": {
            "subset": str(subset),
            "trained_predictions": str(trained_predictions),
            "prediction_schema": (
                "recovered-temperature predictions loaded from .npz arrays keyed by sample_id; "
                "directory fallback mirrors compare_heat3d_v1_medium_baselines.py"
            ),
        },
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_md),
        },
        "sample_count": len(records),
        "point_count": int(sum(record["point_count"] for record in records)),
        "group_by": group_keys,
        "deltaT_bin_edges": edge_payload,
        "overall": {
            "bins": overall_bins,
        },
        "groupwise": _groupwise_bins(records, group_keys, edge_payload["bins"]),
        "interpretation": _interpret_bins(overall_bins),
        "recommended_next_actions": [
            "background penalty",
            "hotspot + background combined loss",
            "conservative / residual learning",
            "optional e100/e200 comparison",
        ],
    }
    _write_json(output_json, payload)
    _write_text(output_md, render_markdown(payload))
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.8e}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100.0:+.2f}%"


def _bin_table_rows(bins: list[dict[str, Any]], group_label: str = "") -> list[str]:
    lines = [
        "| group | bin | points | samples | DeltaT min | DeltaT max | DeltaT mean | zero RMSE | trained RMSE | RMSE rel change | zero MAE | trained MAE | MAE rel change | trained bias | over ratio | under ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    label = group_label or "overall"
    for item in bins:
        lines.append(
            "| "
            f"{label} | {item['bin_name']} | {item['point_count']} | {item['sample_count']} | "
            f"{_fmt_float(item.get('DeltaT_min'))} | {_fmt_float(item.get('DeltaT_max'))} | "
            f"{_fmt_float(item.get('DeltaT_mean'))} | {_fmt_float(item.get('zero_delta_rmse'))} | "
            f"{_fmt_float(item.get('trained_rmse'))} | {_fmt_pct(item.get('relative_rmse_change'))} | "
            f"{_fmt_float(item.get('zero_delta_mae'))} | {_fmt_float(item.get('trained_mae'))} | "
            f"{_fmt_pct(item.get('relative_mae_change'))} | {_fmt_float(item.get('trained_signed_bias'))} | "
            f"{_fmt_float(item.get('trained_overprediction_ratio'))} | {_fmt_float(item.get('trained_underprediction_ratio'))} |"
        )
    return lines


def _key_findings(payload: dict[str, Any]) -> list[str]:
    findings = []
    for key, groups in payload["groupwise"].items():
        for group in groups:
            changes = _finite_values(group["bins"], "relative_mae_change")
            biases = _finite_values(group["bins"], "trained_signed_bias")
            if not changes:
                continue
            max_degradation = max(changes)
            min_improvement = min(changes)
            mean_bias = float(np.mean(biases)) if biases else 0.0
            findings.append(
                {
                    "key": key,
                    "value": group["group_value"],
                    "max_mae_degradation": max_degradation,
                    "best_mae_improvement": min_improvement,
                    "mean_trained_bias": mean_bias,
                }
            )
    findings = sorted(findings, key=lambda item: (item["max_mae_degradation"], -item["best_mae_improvement"]), reverse=True)
    lines = ["| condition | value | max MAE degradation | best MAE improvement | mean trained bias |"]
    lines.append("|---|---|---:|---:|---:|")
    for item in findings[:12]:
        lines.append(
            f"| {item['key']} | {item['value']} | {_fmt_pct(item['max_mae_degradation'])} | "
            f"{_fmt_pct(item['best_mae_improvement'])} | {_fmt_float(item['mean_trained_bias'])} |"
        )
    if not findings:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    interpretation = payload["interpretation"]
    edges = payload["deltaT_bin_edges"]
    lines: list[str] = [
        "# Heat3D v1 Medium Error Bins",
        "",
        "This report is error-binning / background-bias diagnostics tooling only. It is not a formal benchmark, OOD generalization claim, model-performance conclusion, or high-fidelity solver validation.",
        "",
        "## Run Inputs",
        "",
        f"- subset: `{payload['inputs']['subset']}`",
        f"- trained_predictions: `{payload['inputs']['trained_predictions']}`",
        f"- prediction_schema: `{payload['inputs']['prediction_schema']}`",
        f"- sample_count: `{payload['sample_count']}`",
        f"- point_count: `{payload['point_count']}`",
        "",
        "## Global DeltaT Bin Edges",
        "",
        "| label | value |",
        "|---|---:|",
    ]
    for label, value in zip(edges["labels"], edges["edges"]):
        lines.append(f"| {label} | {_fmt_float(value)} |")
    lines.extend(["", "## Overall Bin Table", ""])
    lines.extend(_bin_table_rows(payload["overall"]["bins"]))
    lines.extend(["", "## Split-Wise Bin Summary", ""])
    for group in payload["groupwise"].get("split", []):
        lines.append(f"### split={group['group_value']}")
        lines.append("")
        lines.extend(_bin_table_rows(group["bins"], group_label=f"split={group['group_value']}"))
        lines.append("")
    lines.extend(["## Condition-Wise Key Findings", ""])
    lines.extend(_key_findings(payload))
    lines.extend(
        [
            "",
            "## Background-Bias Interpretation",
            "",
            f"- likely_background_overprediction: `{interpretation['likely_background_overprediction']}`",
            f"- likely_hotspot_region_improvement: `{interpretation['likely_hotspot_region_improvement']}`",
            f"- likely_hotspot_learning_with_background_bias: `{interpretation['likely_hotspot_learning_with_background_bias']}`",
            f"- low bins: `{interpretation.get('low_bin_names')}`",
            f"- high bins: `{interpretation.get('high_bin_names')}`",
            f"- low-bin mean trained signed bias: `{_fmt_float(interpretation.get('low_bins_mean_trained_signed_bias'))}`",
            "",
        ]
    )
    if interpretation["likely_hotspot_learning_with_background_bias"]:
        lines.extend(
            [
                "The configured diagnostic pattern is present: low-DeltaT bins show worse trained errors with positive signed bias, while high-DeltaT bins improve against zero_delta. Treat this as background-bias diagnostics, not as a performance conclusion.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The configured background-bias plus hotspot-improvement pattern was not fully detected.",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommended Next Actions",
            "",
            "- background penalty",
            "- hotspot + background combined loss",
            "- conservative / residual learning",
            "- optional e100/e200 comparison",
            "",
        ]
    )
    return "\n".join(lines)


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _bin_by_name(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in payload.get("overall", {}).get("bins", []) if item.get("bin_name") == name), None)


def _compact_bin_line(item: dict[str, Any] | None, fields: tuple[str, ...]) -> str:
    if item is None:
        return "missing"
    parts = [str(item.get("bin_name"))]
    for field in fields:
        value = item.get(field)
        parts.append(f"{field}={_fmt_float(value) if 'change' not in field else _fmt_pct(value)}")
    return " ".join(parts)


def _print_stdout_summary(payload: dict[str, Any], stdout_mode: str) -> None:
    outputs = payload["outputs"]
    interpretation = payload["interpretation"]
    if stdout_mode == "quiet":
        _emit(f"error_bins_written: json={outputs['json']} markdown={outputs['markdown']}")
        return

    _emit("Heat3D v1 medium error-binning / background-bias diagnostics tooling")
    _emit("  scope: diagnostics only; not formal benchmark or model-performance conclusion")
    _emit(f"  subset: {payload['inputs']['subset']}")
    _emit(f"  trained_predictions: {payload['inputs']['trained_predictions']}")
    for name in ("bin_0", "bin_1", "bin_2", "bin_3", "bin_4"):
        item = _bin_by_name(payload, name)
        if name == "bin_0":
            fields = ("relative_rmse_change", "relative_mae_change", "trained_signed_bias", "trained_overprediction_ratio")
        else:
            fields = ("relative_rmse_change", "trained_signed_bias")
        _emit(f"  {name}: {_compact_bin_line(item, fields)}")
    _emit(f"  likely_background_overprediction: {interpretation['likely_background_overprediction']}")
    _emit(f"  likely_hotspot_region_improvement: {interpretation['likely_hotspot_region_improvement']}")
    _emit(f"  likely_hotspot_learning_with_background_bias: {interpretation['likely_hotspot_learning_with_background_bias']}")
    if stdout_mode == "full":
        _emit(f"  sample_count: {payload['sample_count']} point_count: {payload['point_count']}")
        _emit(f"  low bins: {interpretation.get('low_bin_names')}")
        _emit(f"  high bins: {interpretation.get('high_bin_names')}")
    _emit(f"  output_json: {outputs['json']}")
    _emit(f"  output_md: {outputs['markdown']}")
    _emit("  analysis_written: True")


def main() -> int:
    args = parse_args()
    output_json = args.output_json or args.trained_predictions.parent / "error_bins.json"
    output_md = args.output_md or args.trained_predictions.parent / "error_bins.md"
    payload = analyze_error_bins(
        subset=args.subset,
        trained_predictions=args.trained_predictions,
        output_json=output_json,
        output_md=output_md,
        bins=args.bins,
        group_by=args.group_by,
    )
    _print_stdout_summary(payload, args.stdout_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
