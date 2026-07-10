#!/usr/bin/env python3
"""Condition-wise Heat3D v1 medium prediction diagnostics.

This script diagnoses low-DeltaT background bias by grouping existing recovered
temperature predictions by split, source, k-region, BC, k-mode, and q-power
range. It does not train, generate data, or make formal benchmark claims.
"""

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
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rigno.heat3d_v1_label_diagnostics import find_sample_dirs, load_json, resolve_t_ref  # noqa: E402
from rigno.heat3d_v4_split_map import (  # noqa: E402
    load_sample_split_map,
    resolve_sample_split,
    split_source_label,
)
from analyze_heat3d_v1_medium_error_bins import (  # noqa: E402
    DEFAULT_BINS,
    _bin_edges,
    _bin_stats,
    _fmt_float,
    _fmt_pct,
    _parse_percentile_bins,
    _prediction_loader,
)


GROUP_KEYS = (
    "split",
    "source_category",
    "k_region_mode",
    "bc_category",
    "k_mode",
    "q_power_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Condition-wise Heat3D v1 medium prediction diagnostics for low-DeltaT "
            "background bias. Diagnostics only; not a formal benchmark."
        )
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--trained-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--prediction-label", type=str, default="trained_prediction")
    parser.add_argument(
        "--split-map",
        type=Path,
        default=None,
        help="Optional sample_id-to-split map. When provided it overrides sample_meta split labels.",
    )
    parser.add_argument("--bins", type=str, default=DEFAULT_BINS)
    parser.add_argument("--q-power-bins", type=str, default="p33,p66")
    parser.add_argument("--stdout-mode", choices=("compact", "full", "quiet"), default="compact")
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _as_column(array: np.ndarray, n_points: int, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape == (n_points,):
        values = values.reshape(n_points, 1)
    if values.shape != (n_points, 1):
        raise ValueError(f"{name} must have shape ({n_points}, 1) or ({n_points},), found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_json(path)


def _plan(meta: dict[str, Any]) -> dict[str, Any]:
    generation_config = meta.get("generation_config", {})
    if isinstance(generation_config, dict):
        sample_plan = generation_config.get("sample_plan", {})
        if isinstance(sample_plan, dict):
            return sample_plan
    return {}


def _meta_value(metadata: dict[str, Any], sample_meta: dict[str, Any], key: str) -> str:
    if metadata.get(key) is not None:
        return str(metadata[key])
    if sample_meta.get(key) is not None:
        return str(sample_meta[key])
    plan = _plan(sample_meta)
    if plan.get(key) is not None:
        return str(plan[key])
    if key == "stack_template":
        stack = sample_meta.get("stack", {})
        if isinstance(stack, dict) and stack.get("stack_template") is not None:
            return str(stack["stack_template"])
    return "unknown"


def _integrated_power(metadata: dict[str, Any], sample_meta: dict[str, Any], q_field: np.ndarray) -> float:
    for key in ("integrated_power_W", "integrated_q_power", "integrated_power"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    source_diag = sample_meta.get("source_diagnostics", {})
    if isinstance(source_diag, dict):
        for key in ("integrated_q_power", "integrated_power_W", "integrated_power"):
            value = source_diag.get(key)
            if value is not None:
                return float(value)
    return float(np.sum(np.abs(q_field)))


def _q_power_edges(values: list[float], spec: str) -> dict[str, Any]:
    if not values:
        raise ValueError("No q power values found")
    percentiles = _parse_percentile_bins(spec)
    edges = [float(np.min(values))]
    edges.extend(float(value) for value in np.percentile(values, percentiles))
    edges.append(float(np.max(values)))
    labels = ["min", *(f"p{percentile:g}" for percentile in percentiles), "max"]
    ranges = []
    for idx in range(len(edges) - 1):
        ranges.append(
            {
                "range_index": idx,
                "range_name": f"q_power_bin_{idx}",
                "left_label": labels[idx],
                "right_label": labels[idx + 1],
                "lower": edges[idx],
                "upper": edges[idx + 1],
                "interval": f"{'[' if idx == 0 else '('}{labels[idx]}, {labels[idx + 1]}]",
            }
        )
    return {"percentiles": percentiles, "labels": labels, "edges": edges, "ranges": ranges}


def _q_power_range(value: float, ranges: list[dict[str, Any]]) -> str:
    for item in ranges:
        lower = float(item["lower"])
        upper = float(item["upper"])
        if item["range_index"] == 0:
            matched = lower <= value <= upper
        else:
            matched = lower < value <= upper
        if matched:
            return str(item["range_name"])
    return str(ranges[-1]["range_name"])


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


def _load_records(
    subset: Path,
    trained_predictions: Path,
    split_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sample_dirs = find_sample_dirs(_sample_root(subset))
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {subset}")
    load_prediction = _prediction_loader(trained_predictions)
    records = []
    failures = []
    for sample_dir in sample_dirs:
        sample_meta = load_json(sample_dir / "sample_meta.json")
        metadata = _read_optional_json(sample_dir / "metadata.json")
        sample_id = str(metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name)
        split = resolve_sample_split(sample_id, sample_meta, metadata=metadata, split_map=split_map)
        coords = np.load(sample_dir / "coords.npy")
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{sample_id}: coords.npy must have shape (N, 3), found {coords.shape}")
        n_points = coords.shape[0]
        true_temperature = _as_column(np.load(sample_dir / "temperature.npy"), n_points, f"{sample_id} temperature.npy")
        try:
            trained_raw = load_prediction(sample_id)
        except (FileNotFoundError, KeyError) as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "error": str(exc),
                }
            )
            continue
        trained_temperature = _as_column(trained_raw, n_points, f"{sample_id} trained prediction")
        q_field = np.load(sample_dir / "q_field.npy")
        t_ref_info = resolve_t_ref(sample_meta)
        t_ref = float(t_ref_info["value"])
        q_power = _integrated_power(metadata, sample_meta, q_field)
        true_delta = true_temperature - t_ref
        records.append(
            {
                "sample_id": sample_id,
                "point_count": int(n_points),
                "q_power": q_power,
                "groups": {
                    "split": split,
                    "source_category": _meta_value(metadata, sample_meta, "source_pattern_tag"),
                    "k_region_mode": _meta_value(metadata, sample_meta, "k_region_mode"),
                    "bc_category": _meta_value(metadata, sample_meta, "bc_category"),
                    "k_mode": _meta_value(metadata, sample_meta, "k_field_mode"),
                    "power_scale_category": _meta_value(metadata, sample_meta, "power_scale_category"),
                },
                "T_true": true_temperature.reshape(-1),
                "DeltaT_true": true_delta.reshape(-1),
                "T_pred_trained": trained_temperature.reshape(-1),
                "T_pred_zero": np.full(n_points, t_ref, dtype=np.float64),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"no prediction-matched samples found for {trained_predictions}"
        )
    q_edges = _q_power_edges([record["q_power"] for record in records], "p33,p66")
    for record in records:
        record["groups"]["q_power_range"] = _q_power_range(record["q_power"], q_edges["ranges"])
    return records, q_edges, failures


def _summary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    trained_errors = np.concatenate([record["T_pred_trained"] - record["T_true"] for record in records])
    zero_errors = np.concatenate([record["T_pred_zero"] - record["T_true"] for record in records])
    trained_rmse = _rmse(trained_errors)
    zero_rmse = _rmse(zero_errors)
    trained_mae = _mae(trained_errors)
    zero_mae = _mae(zero_errors)
    return {
        "sample_count": len({record["sample_id"] for record in records}),
        "point_count": int(sum(record["point_count"] for record in records)),
        "trained_rmse": trained_rmse,
        "trained_mae": trained_mae,
        "trained_bias": float(np.mean(trained_errors)),
        "trained_overprediction_ratio": float(np.mean(trained_errors > 0.0)),
        "trained_underprediction_ratio": float(np.mean(trained_errors < 0.0)),
        "zero_delta_rmse": zero_rmse,
        "zero_delta_mae": zero_mae,
        "relative_rmse_change": _relative_change(trained_rmse, zero_rmse),
        "relative_mae_change": _relative_change(trained_mae, zero_mae),
    }


def _group_records(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["groups"].get(key, "unknown"))].append(record)
    return dict(sorted(grouped.items()))


def _bin_summary(bins: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        item["bin_name"]: {
            "point_count": item.get("point_count"),
            "sample_count": item.get("sample_count"),
            "trained_rmse": item.get("trained_rmse"),
            "trained_mae": item.get("trained_mae"),
            "trained_signed_bias": item.get("trained_signed_bias"),
            "trained_overprediction_ratio": item.get("trained_overprediction_ratio"),
            "trained_underprediction_ratio": item.get("trained_underprediction_ratio"),
            "relative_rmse_change": item.get("relative_rmse_change"),
            "relative_mae_change": item.get("relative_mae_change"),
        }
        for item in bins
    }


def _condition_item(key: str, value: str, records: list[dict[str, Any]], bin_specs: list[dict[str, Any]]) -> dict[str, Any]:
    bins = _bin_stats(records, bin_specs)
    return {
        "group_key": key,
        "group_value": value,
        "metrics": _summary_metrics(records),
        "bins": bins,
        "bin_summary": _bin_summary(bins),
    }


def _top_background_groups(groups: dict[str, list[dict[str, Any]]], limit: int = 20) -> list[dict[str, Any]]:
    candidates = []
    for key, rows in groups.items():
        for row in rows:
            bin0 = row["bin_summary"].get("bin_0", {})
            candidates.append(
                {
                    "group_key": key,
                    "group_value": row["group_value"],
                    "sample_count": row["metrics"]["sample_count"],
                    "bin_0_bias": bin0.get("trained_signed_bias"),
                    "bin_0_overprediction_ratio": bin0.get("trained_overprediction_ratio"),
                    "bin_0_relative_mae_change": bin0.get("relative_mae_change"),
                    "overall_relative_mae_change": row["metrics"].get("relative_mae_change"),
                }
            )
    return sorted(
        candidates,
        key=lambda item: (
            item["bin_0_overprediction_ratio"] if item["bin_0_overprediction_ratio"] is not None else -1.0,
            item["bin_0_bias"] if item["bin_0_bias"] is not None else -1.0,
        ),
        reverse=True,
    )[:limit]


def analyze_condition_diagnostics(
    *,
    subset: Path,
    trained_predictions: Path,
    output_json: Path,
    output_md: Path,
    prediction_label: str = "trained_prediction",
    split_map_path: Path | None = None,
    bins: str = DEFAULT_BINS,
    q_power_bins: str = "p33,p66",
) -> dict[str, Any]:
    split_map = load_sample_split_map(split_map_path)
    records, _, failures = _load_records(subset, trained_predictions, split_map)
    q_edges = _q_power_edges([record["q_power"] for record in records], q_power_bins)
    for record in records:
        record["groups"]["q_power_range"] = _q_power_range(record["q_power"], q_edges["ranges"])
    delta_edges = _bin_edges(np.concatenate([record["DeltaT_true"] for record in records]), _parse_percentile_bins(bins))
    group_payload = {
        key: [
            _condition_item(key, value, group_records, delta_edges["bins"])
            for value, group_records in _group_records(records, key).items()
        ]
        for key in GROUP_KEYS
    }
    payload = {
        "diagnostic_scope": "condition-wise background-bias diagnostics; not formal benchmark or model-performance conclusion",
        "prediction_label": prediction_label,
        "inputs": {
            "subset": str(subset),
            "trained_predictions": str(trained_predictions),
            "split_source": split_source_label(split_map),
            "split_map": str(split_map_path) if split_map_path is not None else None,
            "bins": bins,
            "q_power_bins": q_power_bins,
        },
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_md),
        },
        "sample_count": len(records),
        "failed_sample_count": len(failures),
        "point_count": int(sum(record["point_count"] for record in records)),
        "deltaT_bin_edges": delta_edges,
        "q_power_edges": q_edges,
        "overall": _condition_item("overall", "overall", records, delta_edges["bins"]),
        "condition_groups": group_payload,
        "top_background_bias_groups": _top_background_groups(group_payload),
        "failures": failures,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v1 Medium Condition Diagnostics",
        "",
        "This report is diagnostic tooling for low-DeltaT background bias. It is not a formal benchmark, model-performance conclusion, OOD generalization claim, or high-fidelity solver validation.",
        "",
        "## Inputs",
        "",
        f"- prediction_label: `{payload['prediction_label']}`",
        f"- subset: `{payload['inputs']['subset']}`",
        f"- trained_predictions: `{payload['inputs']['trained_predictions']}`",
        f"- sample_count: `{payload['sample_count']}`",
        f"- failed_sample_count: `{payload['failed_sample_count']}`",
        f"- point_count: `{payload['point_count']}`",
        "",
        "## Overall",
        "",
    ]
    overall = payload["overall"]
    metrics = overall["metrics"]
    bin0 = overall["bin_summary"].get("bin_0", {})
    lines.extend(
        [
            f"- trained RMSE: `{_fmt_float(metrics.get('trained_rmse'))}`",
            f"- trained MAE: `{_fmt_float(metrics.get('trained_mae'))}`",
            f"- trained bias: `{_fmt_float(metrics.get('trained_bias'))}`",
            f"- overprediction ratio: `{_fmt_float(metrics.get('trained_overprediction_ratio'))}`",
            f"- bin_0 bias: `{_fmt_float(bin0.get('trained_signed_bias'))}`",
            f"- bin_0 overprediction ratio: `{_fmt_float(bin0.get('trained_overprediction_ratio'))}`",
            f"- bin_0 MAE relative change: `{_fmt_pct(bin0.get('relative_mae_change'))}`",
            "",
            "## Top Background-Bias Groups",
            "",
            "| group | value | samples | bin_0 bias | bin_0 over ratio | bin_0 MAE rel | overall MAE rel |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["top_background_bias_groups"]:
        lines.append(
            f"| {item['group_key']} | {item['group_value']} | {item['sample_count']} | "
            f"{_fmt_float(item.get('bin_0_bias'))} | {_fmt_float(item.get('bin_0_overprediction_ratio'))} | "
            f"{_fmt_pct(item.get('bin_0_relative_mae_change'))} | "
            f"{_fmt_pct(item.get('overall_relative_mae_change'))} |"
        )
    lines.extend(["", "## Condition Groups", ""])
    for key, rows in payload["condition_groups"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                "| value | samples | RMSE | MAE | bias | over ratio | bin_0 bias | bin_0 over | bin_1 MAE rel | bin_4 MAE rel |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            row_metrics = row["metrics"]
            row_bin0 = row["bin_summary"].get("bin_0", {})
            row_bin1 = row["bin_summary"].get("bin_1", {})
            row_bin4 = row["bin_summary"].get("bin_4", {})
            lines.append(
                f"| {row['group_value']} | {row_metrics['sample_count']} | "
                f"{_fmt_float(row_metrics.get('trained_rmse'))} | {_fmt_float(row_metrics.get('trained_mae'))} | "
                f"{_fmt_float(row_metrics.get('trained_bias'))} | "
                f"{_fmt_float(row_metrics.get('trained_overprediction_ratio'))} | "
                f"{_fmt_float(row_bin0.get('trained_signed_bias'))} | "
                f"{_fmt_float(row_bin0.get('trained_overprediction_ratio'))} | "
                f"{_fmt_pct(row_bin1.get('relative_mae_change'))} | "
                f"{_fmt_pct(row_bin4.get('relative_mae_change'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _print_stdout_summary(payload: dict[str, Any], stdout_mode: str) -> None:
    outputs = payload["outputs"]
    if stdout_mode == "quiet":
        _emit(f"condition_diagnostics_written: json={outputs['json']} markdown={outputs['markdown']}")
        return
    overall = payload["overall"]
    metrics = overall["metrics"]
    bin0 = overall["bin_summary"].get("bin_0", {})
    _emit("Heat3D v1 medium condition diagnostics")
    _emit("  scope: diagnostics only; not formal benchmark or model-performance conclusion")
    _emit(f"  prediction_label: {payload['prediction_label']}")
    _emit(f"  sample_count: {payload['sample_count']} point_count: {payload['point_count']}")
    _emit(
        "  overall: "
        f"rmse={_fmt_float(metrics.get('trained_rmse'))} mae={_fmt_float(metrics.get('trained_mae'))} "
        f"bias={_fmt_float(metrics.get('trained_bias'))} "
        f"over_ratio={_fmt_float(metrics.get('trained_overprediction_ratio'))}"
    )
    _emit(
        "  bin_0: "
        f"bias={_fmt_float(bin0.get('trained_signed_bias'))} "
        f"over_ratio={_fmt_float(bin0.get('trained_overprediction_ratio'))} "
        f"mae_rel={_fmt_pct(bin0.get('relative_mae_change'))}"
    )
    if stdout_mode == "full":
        for item in payload["top_background_bias_groups"][:10]:
            _emit(
                "  top_bg_group: "
                f"{item['group_key']}={item['group_value']} samples={item['sample_count']} "
                f"bin0_bias={_fmt_float(item.get('bin_0_bias'))} "
                f"bin0_over={_fmt_float(item.get('bin_0_overprediction_ratio'))}"
            )
    _emit(f"  output_json: {outputs['json']}")
    _emit(f"  output_md: {outputs['markdown']}")


def main() -> int:
    args = parse_args()
    payload = analyze_condition_diagnostics(
        subset=args.subset,
        trained_predictions=args.trained_predictions,
        output_json=args.output_json,
        output_md=args.output_md,
        prediction_label=args.prediction_label,
        split_map_path=args.split_map,
        bins=args.bins,
        q_power_bins=args.q_power_bins,
    )
    _print_stdout_summary(payload, args.stdout_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
