#!/usr/bin/env python3
"""Read-only Heat3D v3 prediction mechanism diagnostics.

This script reads an existing run directory plus one prediction archive
(``predictions.npz`` or ``best_predictions.npz``) and decomposes errors into
amplitude, shape, hotspot, and condition-wise groups. It does not import JAX,
build graphs, load model code, or train.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import find_sample_dirs, load_json, resolve_t_ref  # noqa: E402


EPS = 1.0e-12
GROUP_KEYS = (
    "split",
    "source_category",
    "q_power_range",
    "k_mode",
    "k_region_mode",
    "bc_category",
)
SUMMARY_METRICS = (
    "pred_mean",
    "target_mean",
    "pred_std",
    "target_std",
    "amplitude_ratio",
    "mean_bias",
    "centered_corr",
    "zscore_rmse",
    "top_k_overlap",
    "hotspot_centroid_distance",
    "peak_abs_error",
    "peak_rel_error",
    "rmse",
    "mae",
)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_ratio(numerator: float, denominator: float, *, eps: float = EPS) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= eps:
        return None
    return _json_float(numerator / denominator)


def _sample_root(path: Path) -> Path:
    if (path / "samples").is_dir():
        return path / "samples"
    return path


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_json(path)


def _as_column(array: np.ndarray, n_points: int, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape == (n_points,):
        values = values.reshape(n_points, 1)
    if values.shape != (n_points, 1):
        raise ValueError(
            f"{name} must have shape ({n_points}, 1) or ({n_points},), found {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


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
        domain = sample_meta.get("domain", {})
        if isinstance(domain, dict) and domain.get("stack_template") is not None:
            return str(domain["stack_template"])
    return "unknown"


def _integrated_power(
    metadata: dict[str, Any],
    sample_meta: dict[str, Any],
    q_field: np.ndarray,
) -> float:
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


def _q_power_edges(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("no q power values found")
    percentiles = (33.0, 66.0)
    edges = [float(np.min(values))]
    edges.extend(float(value) for value in np.percentile(values, percentiles))
    edges.append(float(np.max(values)))
    labels = ["min", "p33", "p66", "max"]
    ranges = []
    for index in range(len(edges) - 1):
        ranges.append(
            {
                "range_index": index,
                "range_name": f"q_power_bin_{index}",
                "left_label": labels[index],
                "right_label": labels[index + 1],
                "lower": edges[index],
                "upper": edges[index + 1],
                "interval": f"{'[' if index == 0 else '('}{labels[index]}, {labels[index + 1]}]",
            }
        )
    return {"percentiles": list(percentiles), "edges": edges, "labels": labels, "ranges": ranges}


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


def _prediction_loader(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"prediction archive not found: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"prediction archive must be .npz, found {path}")
    archive = np.load(path)

    def load(sample_id: str) -> np.ndarray:
        if sample_id not in archive:
            raise KeyError(f"{path} missing prediction key {sample_id}")
        return np.asarray(archive[sample_id])

    return load, sorted(archive.files)


def _centered_corr(target: np.ndarray, pred: np.ndarray) -> float | None:
    target_centered = target - float(np.mean(target))
    pred_centered = pred - float(np.mean(pred))
    denominator = float(np.linalg.norm(target_centered) * np.linalg.norm(pred_centered))
    if denominator <= EPS:
        return None
    return _json_float(float(np.dot(target_centered, pred_centered)) / denominator)


def _zscore_rmse(target: np.ndarray, pred: np.ndarray) -> float | None:
    target_std = float(np.std(target))
    pred_std = float(np.std(pred))
    if target_std <= EPS or pred_std <= EPS:
        return None
    target_z = (target - float(np.mean(target))) / target_std
    pred_z = (pred - float(np.mean(pred))) / pred_std
    return _json_float(float(np.sqrt(np.mean(np.square(pred_z - target_z)))))


def _top_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    k = min(int(top_k), int(values.size))
    if k < 1:
        raise ValueError("top_k must be >= 1")
    return np.argpartition(values, -k)[-k:]


def _top_k_overlap(target: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    target_top = set(int(item) for item in _top_indices(target, top_k).tolist())
    pred_top = set(int(item) for item in _top_indices(pred, top_k).tolist())
    return float(len(target_top & pred_top) / min(top_k, target.size))


def _hotspot_centroid_distance(
    target: np.ndarray,
    pred: np.ndarray,
    coords: np.ndarray,
    top_k: int,
) -> float:
    target_centroid = np.mean(coords[_top_indices(target, top_k)], axis=0)
    pred_centroid = np.mean(coords[_top_indices(pred, top_k)], axis=0)
    return float(np.linalg.norm(pred_centroid - target_centroid))


def _sample_metrics(
    *,
    sample_id: str,
    groups: dict[str, str],
    target_delta: np.ndarray,
    pred_delta: np.ndarray,
    coords: np.ndarray,
    q_power: float,
    top_k: int,
) -> dict[str, Any]:
    target = np.asarray(target_delta, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_delta, dtype=np.float64).reshape(-1)
    if target.shape != pred.shape:
        raise ValueError(f"{sample_id}: target/pred shape mismatch {target.shape} vs {pred.shape}")
    error = pred - target
    target_range = float(np.max(target) - np.min(target))
    pred_range = float(np.max(pred) - np.min(pred))
    target_peak = float(np.max(target))
    pred_peak = float(np.max(pred))
    peak_abs_error = abs(pred_peak - target_peak)
    return {
        "sample_id": sample_id,
        "groups": groups,
        "point_count": int(target.size),
        "q_power": _json_float(q_power),
        "pred_mean": _json_float(float(np.mean(pred))),
        "target_mean": _json_float(float(np.mean(target))),
        "pred_std": _json_float(float(np.std(pred))),
        "target_std": _json_float(float(np.std(target))),
        "amplitude_ratio": _safe_ratio(pred_range, target_range),
        "mean_bias": _json_float(float(np.mean(error))),
        "centered_corr": _centered_corr(target, pred),
        "zscore_rmse": _zscore_rmse(target, pred),
        "top_k_overlap": _json_float(_top_k_overlap(target, pred, top_k)),
        "hotspot_centroid_distance": _json_float(
            _hotspot_centroid_distance(target, pred, coords, top_k)
        ),
        "peak_abs_error": _json_float(peak_abs_error),
        "peak_rel_error": _safe_ratio(peak_abs_error, abs(target_peak)),
        "rmse": _json_float(float(np.sqrt(np.mean(np.square(error))))),
        "mae": _json_float(float(np.mean(np.abs(error)))),
        "peak_target": _json_float(target_peak),
        "peak_pred": _json_float(pred_peak),
    }


def _mean_metric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value = _json_float(row.get(field))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return _json_float(float(np.mean(values)))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "point_count": int(sum(int(row.get("point_count") or 0) for row in rows)),
        "aggregation": "sample_mean_ignoring_null_metrics",
    }
    for field in SUMMARY_METRICS:
        result[field] = _mean_metric(rows, field)
    amplitude_ratio = result.get("amplitude_ratio")
    result["amplitude_error"] = (
        _json_float(abs(float(amplitude_ratio) - 1.0))
        if amplitude_ratio is not None
        else None
    )
    return result


def _grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped_payload: dict[str, list[dict[str, Any]]] = {}
    for key in GROUP_KEYS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row["groups"].get(key, "unknown"))].append(row)
        items = []
        for value, group_rows in sorted(buckets.items()):
            item = {
                "group_key": key,
                "group_value": value,
            }
            item.update(_aggregate(group_rows))
            items.append(item)
        grouped_payload[key] = items
    return grouped_payload


def _weak_groups(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    metric: str,
    reverse: bool = True,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for group_rows in grouped.values():
        rows.extend(group_rows)

    def sort_key(row: dict[str, Any]) -> float:
        value = _json_float(row.get(metric))
        if value is None:
            return float("-inf") if reverse else float("inf")
        return value

    return sorted(rows, key=sort_key, reverse=reverse)[:limit]


def _resolve_subset(run_dir: Path, explicit_subset: Path | None) -> Path:
    if explicit_subset is not None:
        return explicit_subset
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"run_config.json not found under {run_dir}")
    run_config = load_json(run_config_path)
    subset = run_config.get("subset") or run_config.get("subset_path")
    if subset is None:
        raise KeyError(f"{run_config_path} does not contain subset or subset_path")
    return Path(str(subset))


def _resolve_prediction_path(run_dir: Path, prediction_name: str) -> Path:
    path = Path(prediction_name)
    if path.is_absolute():
        return path
    return run_dir / path


def _load_sample_rows(
    *,
    subset: Path,
    prediction_path: Path,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sample_dirs = find_sample_dirs(_sample_root(subset))
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {subset}")
    load_prediction, prediction_keys = _prediction_loader(prediction_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        try:
            sample_meta = load_json(sample_dir / "sample_meta.json")
            metadata = _read_optional_json(sample_dir / "metadata.json")
            sample_id = str(
                metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name
            )
            coords = np.load(sample_dir / "coords.npy")
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"{sample_id}: coords.npy must have shape (N, 3), found {coords.shape}")
            n_points = int(coords.shape[0])
            true_temperature = _as_column(
                np.load(sample_dir / "temperature.npy"),
                n_points,
                f"{sample_id} temperature.npy",
            )
            pred_temperature = _as_column(
                load_prediction(sample_id),
                n_points,
                f"{sample_id} prediction",
            )
            q_field = np.load(sample_dir / "q_field.npy")
            q_power = _integrated_power(metadata, sample_meta, q_field)
            t_ref = float(resolve_t_ref(sample_meta)["value"])
            groups = {
                "split": _meta_value(metadata, sample_meta, "split"),
                "source_category": _meta_value(metadata, sample_meta, "source_pattern_tag"),
                "k_mode": _meta_value(metadata, sample_meta, "k_field_mode"),
                "k_region_mode": _meta_value(metadata, sample_meta, "k_region_mode"),
                "bc_category": _meta_value(metadata, sample_meta, "bc_category"),
            }
            pending.append(
                {
                    "sample_id": sample_id,
                    "groups": groups,
                    "coords": coords.astype(np.float64),
                    "target_delta": true_temperature.reshape(-1) - t_ref,
                    "pred_delta": pred_temperature.reshape(-1) - t_ref,
                    "q_power": q_power,
                }
            )
        except Exception as exc:  # pragma: no cover - per-sample defensive diagnostics
            failures.append({"sample_id": sample_id, "sample_dir": str(sample_dir), "error": str(exc)})

    q_edges = _q_power_edges([float(item["q_power"]) for item in pending]) if pending else {
        "percentiles": [33.0, 66.0],
        "edges": [],
        "labels": [],
        "ranges": [],
    }
    for item in pending:
        item["groups"]["q_power_range"] = _q_power_range(float(item["q_power"]), q_edges["ranges"])
        rows.append(
            _sample_metrics(
                sample_id=str(item["sample_id"]),
                groups=dict(item["groups"]),
                target_delta=np.asarray(item["target_delta"]),
                pred_delta=np.asarray(item["pred_delta"]),
                coords=np.asarray(item["coords"]),
                q_power=float(item["q_power"]),
                top_k=top_k,
            )
        )

    unused_prediction_keys = sorted(set(prediction_keys) - {row["sample_id"] for row in rows})
    q_edges["unused_prediction_key_count"] = len(unused_prediction_keys)
    if unused_prediction_keys:
        q_edges["unused_prediction_key_examples"] = unused_prediction_keys[:10]
    return rows, failures, q_edges


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if "data" in path.parts:
        raise ValueError("--output-json must not be under data/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    numeric = _json_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.6g}"


def _weak_table(title: str, rows: list[dict[str, Any]], metric: str, limit: int = 5) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| group | samples | metric | amplitude_ratio | centered_corr | zscore_rmse | hotspot_distance | peak_rel_error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        group = f"{row.get('group_key')}={row.get('group_value')}"
        lines.append(
            f"| {group} | {row.get('sample_count')} | {_fmt(row.get(metric))} | "
            f"{_fmt(row.get('amplitude_ratio'))} | {_fmt(row.get('centered_corr'))} | "
            f"{_fmt(row.get('zscore_rmse'))} | {_fmt(row.get('hotspot_centroid_distance'))} | "
            f"{_fmt(row.get('peak_rel_error'))} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    lines = [
        "# Heat3D v3 Prediction Mechanism Diagnostics",
        "",
        "Read-only diagnostics for existing prediction archives. No training, graph build, or model execution is performed.",
        "",
        "## Inputs",
        "",
        f"- run_dir: `{payload['inputs']['run_dir']}`",
        f"- prediction_name: `{payload['inputs']['prediction_name']}`",
        f"- prediction_label: `{payload['prediction_label']}`",
        f"- subset: `{payload['inputs']['subset']}`",
        f"- sample_count: `{payload['sample_count']}`",
        f"- failed_sample_count: `{payload['failed_sample_count']}`",
        "",
        "## Overall",
        "",
        "| pred_mean | target_mean | pred_std | target_std | amplitude_ratio | mean_bias | centered_corr | zscore_rmse | top_k_overlap | hotspot_distance | peak_abs_error | peak_rel_error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {_fmt(overall.get('pred_mean'))} | {_fmt(overall.get('target_mean'))} | "
        f"{_fmt(overall.get('pred_std'))} | {_fmt(overall.get('target_std'))} | "
        f"{_fmt(overall.get('amplitude_ratio'))} | {_fmt(overall.get('mean_bias'))} | "
        f"{_fmt(overall.get('centered_corr'))} | {_fmt(overall.get('zscore_rmse'))} | "
        f"{_fmt(overall.get('top_k_overlap'))} | {_fmt(overall.get('hotspot_centroid_distance'))} | "
        f"{_fmt(overall.get('peak_abs_error'))} | {_fmt(overall.get('peak_rel_error'))} |",
        "",
    ]
    lines.extend(_weak_table("Weak Amplitude Groups", payload["weak_amplitude_groups"], "amplitude_error"))
    lines.append("")
    lines.extend(_weak_table("Weak Shape Groups", payload["weak_shape_groups"], "zscore_rmse"))
    lines.append("")
    lines.extend(_weak_table("Weak Hotspot Groups", payload["weak_hotspot_groups"], "hotspot_centroid_distance"))
    lines.append("")
    return "\n".join(lines)


def analyze_prediction_mechanisms(
    *,
    run_dir: Path,
    prediction_name: str,
    prediction_label: str,
    output_json: Path,
    output_md: Path,
    subset: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    subset_path = _resolve_subset(run_dir, subset)
    prediction_path = _resolve_prediction_path(run_dir, prediction_name)
    rows, failures, q_edges = _load_sample_rows(
        subset=subset_path,
        prediction_path=prediction_path,
        top_k=top_k,
    )
    grouped = _grouped(rows)
    payload = {
        "diagnostic_scope": "read-only prediction-level mechanism diagnostics; not formal benchmark or model-performance claim",
        "prediction_label": prediction_label,
        "inputs": {
            "run_dir": str(run_dir),
            "prediction_name": prediction_name,
            "prediction_path": str(prediction_path),
            "subset": str(subset_path),
            "top_k": top_k,
        },
        "sample_count": len(rows),
        "failed_sample_count": len(failures),
        "q_power_edges": q_edges,
        "overall": _aggregate(rows),
        "per_sample": rows,
        "grouped": grouped,
        "weak_amplitude_groups": _weak_groups(grouped, metric="amplitude_error", reverse=True),
        "weak_shape_groups": _weak_groups(grouped, metric="zscore_rmse", reverse=True),
        "weak_hotspot_groups": _weak_groups(
            grouped,
            metric="hotspot_centroid_distance",
            reverse=True,
        ),
        "failures": failures,
    }
    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction-name", required=True)
    parser.add_argument("--prediction-label", choices=("final", "best"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze_prediction_mechanisms(
        run_dir=args.run_dir,
        prediction_name=args.prediction_name,
        prediction_label=args.prediction_label,
        output_json=args.output_json,
        output_md=args.output_md,
        subset=args.subset,
        top_k=args.top_k,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(
        "overall: samples={samples} amp={amp} corr={corr} zrmse={zrmse} topk={topk}".format(
            samples=payload["sample_count"],
            amp=_fmt(payload["overall"].get("amplitude_ratio")),
            corr=_fmt(payload["overall"].get("centered_corr")),
            zrmse=_fmt(payload["overall"].get("zscore_rmse")),
            topk=_fmt(payload["overall"].get("top_k_overlap")),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
