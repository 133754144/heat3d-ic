"""Compare Heat3D v1 medium zero-delta and optional trained predictions.

This script is diagnostic tooling only. It does not train a model and does not
write generated samples. By default it computes a zero-delta baseline summary
for the physics-label medium v2 subset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_label_diagnostics import (  # noqa: E402
    find_sample_dirs,
    load_json,
    resolve_t_ref,
)
from rigno.heat3d_v1_metrics import (  # noqa: E402
    hotspot_coord_distance,
    hotspot_index,
    mae,
    max_abs_error,
    mse,
    peak_T_abs_error,
    rmse,
    top_k_hotspot_overlap,
)
from rigno.heat3d_v4_split_map import (  # noqa: E402
    load_sample_split_map,
    resolve_sample_split,
    split_source_label,
)


DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_v2"
)
CONDITION_KEYS = (
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
)
SUMMARY_METRICS = (
    "recovered_T_rmse",
    "recovered_T_mae",
    "recovered_T_mse",
    "DeltaT_rmse",
    "DeltaT_mae",
    "DeltaT_mse",
    "max_abs_error",
    "p95_abs_error",
    "peak_T_error",
    "peak_DeltaT_error",
    "hotspot_coord_error",
    "top_k_hotspot_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline comparison diagnostics for Heat3D v1 medium labels. "
            "Default mode computes zero_delta only and is not a benchmark."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--trained-predictions",
        type=Path,
        default=None,
        help=(
            "Optional predictions path. Supported formats: a .npz with arrays "
            "named by sample_id, or a directory containing <sample_id>.npy or "
            "<sample_id>/temperature.npy recovered-temperature predictions."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Prefer ignored output/ paths.",
    )
    parser.add_argument(
        "--split-map",
        type=Path,
        default=None,
        help="Optional sample_id-to-split map. When provided it overrides sample_meta split labels.",
    )
    parser.add_argument("--top-k", type=int, default=5)
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
    if key == "stack_template":
        stack = meta.get("stack", {})
        if isinstance(stack, dict) and stack.get("stack_template") is not None:
            return str(stack["stack_template"])
    value = _plan(meta).get(key)
    if value is None:
        value = "unknown"
    return str(value)


def _load_array(sample_dir: Path, name: str) -> np.ndarray:
    path = sample_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{sample_dir.name}: missing {name}")
    return np.load(path)


def _load_optional_label_meta(sample_dir: Path) -> dict[str, Any]:
    path = sample_dir / "label_meta.json"
    if not path.is_file():
        return {"present": False}
    data = load_json(path)
    return {
        "present": True,
        "solver_name": data.get("solver_name"),
        "solver_version": data.get("solver_version"),
        "convergence_flag": data.get("convergence_flag"),
        "residual_norm": data.get("residual_norm"),
        "warning_count": len(data.get("warnings", []) or []),
        "warnings": data.get("warnings", []) or [],
    }


def _prediction_loader(path: Path | None):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"trained predictions path does not exist: {path}. Expected a .npz "
            "with arrays named by sample_id, or a directory containing "
            "<sample_id>.npy or <sample_id>/temperature.npy files."
        )

    if path.is_file() and path.suffix == ".npz":
        archive = np.load(path)

        def load_from_npz(sample_id: str) -> np.ndarray:
            if sample_id not in archive:
                raise KeyError(
                    f"trained predictions .npz missing key {sample_id}; "
                    "expected one recovered-temperature array per sample_id"
                )
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
            raise FileNotFoundError(
                f"trained prediction for {sample_id} not found under {path}; "
                "expected <sample_id>.npy, <sample_id>/temperature.npy, or "
                "<sample_id>/pred_temperature.npy"
            )

        return load_from_dir

    raise ValueError(
        f"unsupported trained predictions format: {path}. Expected .npz or directory."
    )


def _as_column(array: np.ndarray, n_points: int, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape == (n_points,):
        values = values.reshape(n_points, 1)
    if values.shape != (n_points, 1):
        raise ValueError(f"{name} must have shape ({n_points}, 1) or ({n_points},), found {values.shape}")
    return values


def _metric_row(
    *,
    sample_id: str,
    split: str,
    predictor: str,
    pred_temperature: np.ndarray,
    true_temperature: np.ndarray,
    t_ref: float,
    coords: np.ndarray,
    meta: dict[str, Any],
    label_meta: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    true_delta = true_temperature - t_ref
    pred_delta = pred_temperature - t_ref
    abs_error = np.abs(pred_temperature.reshape(-1) - true_temperature.reshape(-1))
    true_hotspot = hotspot_index(true_temperature)
    pred_hotspot = hotspot_index(pred_temperature)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "predictor": predictor,
        "recovered_T_mse": mse(pred_temperature, true_temperature),
        "recovered_T_rmse": rmse(pred_temperature, true_temperature),
        "recovered_T_mae": mae(pred_temperature, true_temperature),
        "DeltaT_mse": mse(pred_delta, true_delta),
        "DeltaT_rmse": rmse(pred_delta, true_delta),
        "DeltaT_mae": mae(pred_delta, true_delta),
        "max_abs_error": max_abs_error(pred_temperature, true_temperature),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "peak_T_true": float(np.max(true_temperature)),
        "peak_T_pred": float(np.max(pred_temperature)),
        "peak_T_error": peak_T_abs_error(pred_temperature, true_temperature),
        "peak_DeltaT_true": float(np.max(true_delta)),
        "peak_DeltaT_pred": float(np.max(pred_delta)),
        "peak_DeltaT_error": float(abs(np.max(pred_delta) - np.max(true_delta))),
        "true_hotspot_index": true_hotspot,
        "pred_hotspot_index": pred_hotspot,
        "hotspot_coord_error": hotspot_coord_distance(pred_temperature, true_temperature, coords),
        "top_k_hotspot_overlap": top_k_hotspot_overlap(pred_temperature, true_temperature, k=top_k),
        "T_ref": t_ref,
        "label_meta_present": bool(label_meta.get("present")),
        "label_solver_name": label_meta.get("solver_name"),
        "label_solver_version": label_meta.get("solver_version"),
        "label_convergence_flag": label_meta.get("convergence_flag"),
        "label_residual_norm": label_meta.get("residual_norm"),
        "label_warning_count": label_meta.get("warning_count", 0),
    }
    for key in CONDITION_KEYS:
        row[key] = _condition_value(meta, key)
    return row


def _sample_rows(
    sample_dir: Path,
    trained_loader,
    top_k: int,
    split_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    meta = load_json(sample_dir / "sample_meta.json")
    sample_id = str(meta.get("sample_id", sample_dir.name))
    split = resolve_sample_split(sample_id, meta, split_map=split_map)
    coords = _load_array(sample_dir, "coords.npy")
    _load_array(sample_dir, "k_field.npy")
    _load_array(sample_dir, "q_field.npy")
    true_temperature = _as_column(_load_array(sample_dir, "temperature.npy"), coords.shape[0], "temperature.npy")
    t_ref_info = resolve_t_ref(meta)
    t_ref = float(t_ref_info["value"])
    label_meta = _load_optional_label_meta(sample_dir)

    rows = [
        _metric_row(
            sample_id=sample_id,
            split=split,
            predictor="zero_delta",
            pred_temperature=np.full_like(true_temperature, t_ref),
            true_temperature=true_temperature,
            t_ref=t_ref,
            coords=coords,
            meta=meta,
            label_meta=label_meta,
            top_k=top_k,
        )
    ]

    if trained_loader is not None:
        trained_temperature = _as_column(
            trained_loader(sample_id),
            coords.shape[0],
            f"trained prediction for {sample_id}",
        )
        rows.append(
            _metric_row(
                sample_id=sample_id,
                split=split,
                predictor="trained_prediction",
                pred_temperature=trained_temperature,
                true_temperature=true_temperature,
                t_ref=t_ref,
                coords=coords,
                meta=meta,
                label_meta=label_meta,
                top_k=top_k,
            )
        )
    return rows


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len({row["sample_id"] for row in rows}), "row_count": len(rows)}
    for metric in SUMMARY_METRICS:
        summary[f"mean_{metric}"] = _mean([float(row[metric]) for row in rows])
    return summary


def _group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)

    result = []
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(_summary_for_rows(group_rows))
        result.append(item)
    return result


def _build_report(
    rows: list[dict[str, Any]],
    trained_enabled: bool,
    *,
    split_source: str,
    split_map_path: Path | None,
) -> dict[str, Any]:
    report = {
        "diagnostic_scope": "baseline comparison tooling; not benchmark or model performance",
        "trained_comparison_status": "computed" if trained_enabled else "pending_no_trained_predictions",
        "split_source": split_source,
        "split_map_path": str(split_map_path) if split_map_path is not None else None,
        "per_sample": rows,
        "overall": _group_summary(rows, ("predictor",)),
        "split_summary": _group_summary(rows, ("split", "predictor")),
        "condition_summary": {
            key: _group_summary(rows, (key, "predictor"))
            for key in CONDITION_KEYS
        },
    }
    return report


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _print_table(title: str, rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> None:
    _emit(title)
    header = [*group_keys, "predictor", "n", "mean_T_rmse", "mean_T_mae", "mean_DeltaT_rmse",
              "mean_max_abs", "mean_p95_abs", "mean_peak_T_err", "mean_hotspot_dist"]
    _emit(" ".join(header))
    for row in rows:
        values = [str(row.get(key, "")) for key in group_keys]
        values.extend(
            [
                str(row.get("predictor", "")),
                str(row["sample_count"]),
                f"{row['mean_recovered_T_rmse']:.8e}",
                f"{row['mean_recovered_T_mae']:.8e}",
                f"{row['mean_DeltaT_rmse']:.8e}",
                f"{row['mean_max_abs_error']:.8e}",
                f"{row['mean_p95_abs_error']:.8e}",
                f"{row['mean_peak_T_error']:.8e}",
                f"{row['mean_hotspot_coord_error']:.8e}",
            ]
        )
        _emit(" ".join(values))


def _row_by_predictor(rows: list[dict[str, Any]], predictor: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("predictor") == predictor), None)


def _fmt_metric(row: dict[str, Any] | None, field: str) -> str:
    if not row or row.get(field) is None:
        return "n/a"
    return f"{float(row[field]):.6e}"


def _print_compact_report(report: dict[str, Any], subset: Path, output_json: Path | None) -> None:
    overall_zero = _row_by_predictor(report["overall"], "zero_delta")
    overall_trained = _row_by_predictor(report["overall"], "trained_prediction")
    sample_count = overall_zero.get("sample_count") if overall_zero else "unknown"
    _emit("Heat3D v1 medium baseline comparison diagnostics")
    _emit("  scope: diagnostic draft only; not a formal benchmark")
    _emit(f"  subset: {subset}")
    _emit(f"  trained_status: {report['trained_comparison_status']} sample_count={sample_count}")
    _emit(
        "  overall: "
        f"zero_rmse={_fmt_metric(overall_zero, 'mean_recovered_T_rmse')} "
        f"zero_mae={_fmt_metric(overall_zero, 'mean_recovered_T_mae')} "
        f"trained_rmse={_fmt_metric(overall_trained, 'mean_recovered_T_rmse')} "
        f"trained_mae={_fmt_metric(overall_trained, 'mean_recovered_T_mae')}"
    )
    split_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in report["split_summary"]:
        split_rows.setdefault(str(row.get("split", "unknown")), {})[str(row.get("predictor"))] = row
    _emit("  split summary: split zero_rmse zero_mae trained_rmse trained_mae")
    for split, rows in sorted(split_rows.items()):
        zero = rows.get("zero_delta")
        trained = rows.get("trained_prediction")
        _emit(
            "    "
            f"{split} {_fmt_metric(zero, 'mean_recovered_T_rmse')} {_fmt_metric(zero, 'mean_recovered_T_mae')} "
            f"{_fmt_metric(trained, 'mean_recovered_T_rmse')} {_fmt_metric(trained, 'mean_recovered_T_mae')}"
        )
    _emit(f"  output_json: {output_json if output_json is not None else 'not_written'}")


def _print_report(report: dict[str, Any], subset: Path, output_json: Path | None, stdout_mode: str) -> None:
    if stdout_mode == "quiet":
        _emit(f"baseline_comparison_written: {output_json if output_json is not None else 'not_written'}")
        return
    if stdout_mode == "compact":
        _print_compact_report(report, subset, output_json)
        return

    _emit("Heat3D v1 medium baseline comparison diagnostics")
    _emit("  scope: diagnostic draft only; not a formal benchmark")
    _emit(f"  subset: {subset}")
    _emit(f"  trained comparison status: {report['trained_comparison_status']}")
    _emit(f"  per-sample rows: {len(report['per_sample'])}")
    _emit()

    _print_table("overall summary", report["overall"], ())
    _emit()
    _print_table("split-wise summary", report["split_summary"], ("split",))
    _emit()
    for key, rows in report["condition_summary"].items():
        _print_table(f"{key}-wise summary", rows, (key,))
        _emit()

    if output_json is not None:
        _emit(f"  JSON written: {output_json}")
    else:
        _emit("  JSON written: False")
    _emit("  trained predictions were not fabricated from recorded scalar metrics")


def _write_json(path: Path, report: dict[str, Any]) -> None:
    if "data" in path.parts:
        raise ValueError("--output-json must not be under data/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    sample_root = _sample_root(args.subset)
    sample_dirs = find_sample_dirs(sample_root)
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {sample_root}")

    trained_loader = _prediction_loader(args.trained_predictions)
    split_map = load_sample_split_map(args.split_map)
    rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        rows.extend(_sample_rows(sample_dir, trained_loader, args.top_k, split_map))

    report = _build_report(
        rows,
        trained_enabled=trained_loader is not None,
        split_source=split_source_label(split_map),
        split_map_path=args.split_map,
    )
    if args.output_json is not None:
        _write_json(args.output_json, report)
    _print_report(report, args.subset, args.output_json, args.stdout_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
