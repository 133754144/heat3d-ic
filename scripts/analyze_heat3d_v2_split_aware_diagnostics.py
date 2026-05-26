#!/usr/bin/env python3
"""Split-aware Heat3D v2 diagnostics for existing prediction archives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import find_sample_dirs, load_json, resolve_t_ref  # noqa: E402
from rigno.heat3d_v2_field_shape_diagnostics import compute_field_shape_metrics  # noqa: E402


DEFAULT_GROUP_KEYS = (
    "power_scale_category",
    "bc_category",
    "k_mode",
    "k_region_mode",
    "source_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute split-aware diagnostics for existing Heat3D v2 predictions. "
            "Read-only; does not train or generate labels."
        )
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--trained-predictions", type=Path, required=True)
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--prediction-label", choices=("final", "best"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--slice-output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-slice-samples", type=int, default=3)
    parser.add_argument("--stdout-mode", choices=("compact", "full", "quiet"), default="compact")
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    return samples if samples.is_dir() else path


def _load_split_map(path: Path) -> dict[str, str]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    mapping = loaded.get("sample_splits", loaded)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: split map must be a mapping or contain sample_splits")
    return {str(sample_id): str(split) for sample_id, split in mapping.items()}


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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _plan(meta: dict[str, Any]) -> dict[str, Any]:
    generation_config = meta.get("generation_config", {})
    if isinstance(generation_config, dict):
        sample_plan = generation_config.get("sample_plan", {})
        if isinstance(sample_plan, dict):
            return sample_plan
    return {}


def _meta_value(metadata: dict[str, Any], sample_meta: dict[str, Any], key: str) -> str:
    if key == "k_mode":
        key = "k_field_mode"
    if key == "source_category":
        key = "source_pattern_tag"
    if metadata.get(key) not in (None, ""):
        return str(metadata[key])
    if sample_meta.get(key) not in (None, ""):
        return str(sample_meta[key])
    plan = _plan(sample_meta)
    if plan.get(key) not in (None, ""):
        return str(plan[key])
    if key == "stack_template":
        stack = sample_meta.get("stack", {})
        if isinstance(stack, dict) and stack.get("stack_template") is not None:
            return str(stack["stack_template"])
    return "unknown"


def _read_optional_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.is_file() else {}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def _sample_row(
    *,
    sample_dir: Path,
    split_name: str,
    load_prediction,
    top_k: int,
) -> dict[str, Any]:
    sample_meta = load_json(sample_dir / "sample_meta.json")
    metadata = _read_optional_json(sample_dir / "metadata.json")
    sample_id = str(metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name)
    coords = np.load(sample_dir / "coords.npy")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{sample_id}: coords.npy must have shape (N, 3), found {coords.shape}")
    n_points = int(coords.shape[0])
    true_temperature = _as_column(np.load(sample_dir / "temperature.npy"), n_points, f"{sample_id} temperature.npy")
    pred_temperature = _as_column(load_prediction(sample_id), n_points, f"{sample_id} prediction")
    t_ref = float(resolve_t_ref(sample_meta)["value"])
    true_delta = true_temperature.reshape(-1) - t_ref
    pred_delta = pred_temperature.reshape(-1) - t_ref
    error = pred_temperature.reshape(-1) - true_temperature.reshape(-1)
    abs_error = np.abs(error)
    hotspot_threshold = float(np.percentile(true_delta, 95))
    hotspot_mask = true_delta >= hotspot_threshold
    low_masks = {
        "le_0p01": true_delta <= 0.01,
        "le_0p02": true_delta <= 0.02,
        "le_0p05": true_delta <= 0.05,
    }
    field_metrics = compute_field_shape_metrics(
        true_delta,
        pred_delta,
        top_k=top_k,
        sample_id=sample_id,
        split=split_name,
    )
    groups = {key: _meta_value(metadata, sample_meta, key) for key in DEFAULT_GROUP_KEYS}
    return {
        "sample_id": sample_id,
        "split": split_name,
        "point_count": n_points,
        "coords_shape": list(coords.shape),
        "groups": groups,
        "raw_deltaT_mse": _rmse(pred_delta - true_delta),
        "raw_deltaT_mae": _mae(pred_delta - true_delta),
        "recovered_T_mse": _rmse(error) ** 2,
        "recovered_T_mae": _mae(error),
        "signed_bias": float(np.mean(error)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "p99_abs_error": float(np.percentile(abs_error, 99)),
        "peak_abs_error": float(abs(float(np.max(pred_delta)) - float(np.max(true_delta)))),
        "hotspot_mae": _mae(abs_error[hotspot_mask]),
        "hotspot_point_fraction": float(np.mean(hotspot_mask)),
        "low_deltaT_bin_errors": {
            name: {
                "point_fraction": float(np.mean(mask)),
                "mae": _mae(abs_error[mask]),
                "rmse": _rmse(error[mask]),
                "signed_bias": float(np.mean(error[mask])) if np.any(mask) else None,
                "overprediction_ratio": float(np.mean(error[mask] > 0.0)) if np.any(mask) else None,
            }
            for name, mask in low_masks.items()
        },
        **{key: field_metrics.get(key) for key in (
            "field_variance_ratio",
            "centered_spatial_correlation",
            "uncentered_cosine_similarity",
            "amplitude_ratio",
            "top_k_overlap",
        )},
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "sample_count": len(rows),
        "point_count": int(sum(int(row.get("point_count") or 0) for row in rows)),
    }
    for key in (
        "raw_deltaT_mse",
        "raw_deltaT_mae",
        "recovered_T_mse",
        "recovered_T_mae",
        "signed_bias",
        "field_variance_ratio",
        "centered_spatial_correlation",
        "uncentered_cosine_similarity",
        "amplitude_ratio",
        "p95_abs_error",
        "p99_abs_error",
        "peak_abs_error",
        "top_k_overlap",
        "hotspot_mae",
    ):
        result[key] = _mean(rows, key)
    low_bins: dict[str, dict[str, Any]] = {}
    for name in ("le_0p01", "le_0p02", "le_0p05"):
        bin_rows = [row["low_deltaT_bin_errors"][name] for row in rows if row.get("low_deltaT_bin_errors")]
        low_bins[name] = {
            "point_fraction": _mean(bin_rows, "point_fraction"),
            "mae": _mean(bin_rows, "mae"),
            "rmse": _mean(bin_rows, "rmse"),
            "signed_bias": _mean(bin_rows, "signed_bias"),
            "overprediction_ratio": _mean(bin_rows, "overprediction_ratio"),
        }
    result["low_deltaT_bin_errors"] = low_bins
    return result


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for key in DEFAULT_GROUP_KEYS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["groups"].get(key, "unknown"))].append(row)
        result[key] = [
            {"group_key": key, "group_value": value, **_summary(group_rows)}
            for value, group_rows in sorted(grouped.items())
        ]
    return result


def _slice_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
    }


def _write_slice_metadata(
    *,
    sample_dirs: list[Path],
    split_name: str,
    load_prediction,
    output_dir: Path,
    max_samples: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for sample_dir in sample_dirs[:max(0, max_samples)]:
        sample_meta = load_json(sample_dir / "sample_meta.json")
        sample_id = str(sample_meta.get("sample_id", sample_dir.name))
        coords = np.load(sample_dir / "coords.npy")
        true_temperature = np.asarray(np.load(sample_dir / "temperature.npy"), dtype=np.float64).reshape(-1)
        pred_temperature = np.asarray(load_prediction(sample_id), dtype=np.float64).reshape(-1)
        z_values = np.unique(coords[:, 2])
        z_value = float(z_values[len(z_values) // 2])
        mask = np.isclose(coords[:, 2], z_value)
        error = pred_temperature - true_temperature
        entry = {
            "sample_id": sample_id,
            "split": split_name,
            "z_value": z_value,
            "point_count": int(np.sum(mask)),
            "true_temperature": _slice_stats(true_temperature[mask]),
            "pred_temperature": _slice_stats(pred_temperature[mask]),
            "error": _slice_stats(error[mask]),
            "protocol": "metadata-only representative z-slice; arrays are not embedded",
        }
        path = output_dir / f"{sample_id}_slice_metadata.json"
        path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries.append({"sample_id": sample_id, "path": str(path), "z_value": z_value})
    return entries


def analyze_split_aware_diagnostics(
    *,
    subset: Path,
    trained_predictions: Path,
    split_map: Path,
    split: str,
    prediction_label: str,
    output_json: Path,
    output_md: Path,
    slice_output_dir: Path | None = None,
    top_k: int = 5,
    max_slice_samples: int = 3,
) -> dict[str, Any]:
    mapping = _load_split_map(split_map)
    load_prediction = _prediction_loader(trained_predictions)
    root = _sample_root(subset)
    sample_dirs_by_id = {
        str(load_json(sample_dir / "sample_meta.json").get("sample_id", sample_dir.name)): sample_dir
        for sample_dir in find_sample_dirs(root)
    }
    selected_ids = sorted(sample_id for sample_id, split_name in mapping.items() if split_name == split)
    selected_dirs = [sample_dirs_by_id[sample_id] for sample_id in selected_ids if sample_id in sample_dirs_by_id]
    missing = sorted(set(selected_ids) - set(sample_dirs_by_id))
    if missing:
        raise FileNotFoundError(f"split map sample ids missing from subset: {missing[:10]}")
    rows = [
        _sample_row(sample_dir=sample_dir, split_name=split, load_prediction=load_prediction, top_k=top_k)
        for sample_dir in selected_dirs
    ]
    slice_entries = (
        _write_slice_metadata(
            sample_dirs=selected_dirs,
            split_name=split,
            load_prediction=load_prediction,
            output_dir=slice_output_dir,
            max_samples=max_slice_samples,
        )
        if slice_output_dir is not None
        else []
    )
    payload = {
        "diagnostic_scope": "Heat3D v2 split-aware diagnostics; read-only; not formal benchmark",
        "prediction_label": prediction_label,
        "inputs": {
            "subset": str(subset),
            "trained_predictions": str(trained_predictions),
            "split_map": str(split_map),
            "split": split,
            "top_k": int(top_k),
        },
        "sample_count": len(rows),
        "split": split,
        "overall": _summary(rows),
        "condition_summary": _condition_summary(rows),
        "low_deltaT_bin_errors": _summary(rows)["low_deltaT_bin_errors"],
        "slice_exports": slice_entries,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.6g}"


def render_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    lines = [
        "# Heat3D v2 Split-Aware Diagnostics",
        "",
        "Read-only diagnostics for existing predictions; not a formal benchmark.",
        "",
        f"- prediction_label: `{payload['prediction_label']}`",
        f"- split: `{payload['split']}`",
        f"- sample_count: `{payload['sample_count']}`",
        "",
        "## Overall",
        "",
        "| raw_deltaT_mse | raw_deltaT_mae | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | p95_abs_error | p99_abs_error | peak_abs_error | top_k_overlap | hotspot_mae |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            _fmt(overall.get(key))
            for key in (
                "raw_deltaT_mse",
                "raw_deltaT_mae",
                "field_variance_ratio",
                "centered_spatial_correlation",
                "amplitude_ratio",
                "p95_abs_error",
                "p99_abs_error",
                "peak_abs_error",
                "top_k_overlap",
                "hotspot_mae",
            )
        )
        + " |",
        "",
        "## Low-DeltaT Bins",
        "",
        "| bin | point_fraction | mae | rmse | signed_bias | overprediction_ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["low_deltaT_bin_errors"].items():
        lines.append(
            f"| {name} | {_fmt(item.get('point_fraction'))} | {_fmt(item.get('mae'))} | "
            f"{_fmt(item.get('rmse'))} | {_fmt(item.get('signed_bias'))} | "
            f"{_fmt(item.get('overprediction_ratio'))} |"
        )
    lines.extend(["", "## Condition Diagnostics", ""])
    for key, rows in payload["condition_summary"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                "| value | samples | raw_deltaT_mse | raw_deltaT_mae | p95_abs_error | hotspot_mae | signed_bias |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['group_value']} | {row['sample_count']} | {_fmt(row.get('raw_deltaT_mse'))} | "
                f"{_fmt(row.get('raw_deltaT_mae'))} | {_fmt(row.get('p95_abs_error'))} | "
                f"{_fmt(row.get('hotspot_mae'))} | {_fmt(row.get('signed_bias'))} |"
            )
        lines.append("")
    if payload["slice_exports"]:
        lines.extend(["## Slice Export Metadata", ""])
        for entry in payload["slice_exports"]:
            lines.append(f"- `{entry['sample_id']}` z={_fmt(entry['z_value'])}: `{entry['path']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    payload = analyze_split_aware_diagnostics(
        subset=args.subset,
        trained_predictions=args.trained_predictions,
        split_map=args.split_map,
        split=args.split,
        prediction_label=args.prediction_label,
        output_json=args.output_json,
        output_md=args.output_md,
        slice_output_dir=args.slice_output_dir,
        top_k=args.top_k,
        max_slice_samples=args.max_slice_samples,
    )
    if args.stdout_mode != "quiet":
        overall = payload["overall"]
        print(
            "Heat3D v2 split-aware diagnostics: "
            f"split={payload['split']} label={payload['prediction_label']} "
            f"samples={payload['sample_count']} raw_mse={_fmt(overall.get('raw_deltaT_mse'))} "
            f"corr={_fmt(overall.get('centered_spatial_correlation'))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
