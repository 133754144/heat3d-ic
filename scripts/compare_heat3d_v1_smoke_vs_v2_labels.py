"""Compare legacy smoke labels against Heat3D v1 physics-label v2 labels.

This script is a label-audit smoke diagnostic. It does not evaluate model
performance and does not write data or generated reports.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_label_diagnostics import resolve_t_ref  # noqa: E402
from rigno.heat3d_v1_metrics import (  # noqa: E402
    hotspot_coord_distance,
    hotspot_index,
    mae,
    max_abs_error,
    rmse,
)


DEFAULT_OLD_SUBSET = Path("data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small")
DEFAULT_V2_SUBSET = Path("data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_small_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label audit comparing legacy smoke temperature labels and v2 "
            "physics-label smoke diagnostics. This is not a model metric."
        )
    )
    parser.add_argument("--old-subset", type=Path, default=DEFAULT_OLD_SUBSET)
    parser.add_argument("--v2-subset", type=Path, default=DEFAULT_V2_SUBSET)
    parser.add_argument("--require-count", type=int, default=16)
    return parser.parse_args()


def _samples_root(path: Path) -> Path:
    if (path / "samples").is_dir():
        return path / "samples"
    return path


def _sample_dirs(path: Path) -> dict[str, Path]:
    root = _samples_root(path)
    if not root.is_dir():
        raise FileNotFoundError(f"sample root does not exist: {root}")
    samples = {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and child.name.startswith("sample_")
    }
    if not samples:
        raise FileNotFoundError(f"no sample_* directories found under {root}")
    return samples


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def _load_sample(sample_dir: Path) -> dict[str, Any]:
    coords_path = sample_dir / "coords.npy"
    temperature_path = sample_dir / "temperature.npy"
    meta_path = sample_dir / "sample_meta.json"
    for path in (coords_path, temperature_path, meta_path):
        _require_file(path)

    label_meta_path = sample_dir / "label_meta.json"
    return {
        "coords": np.load(coords_path),
        "temperature": np.load(temperature_path),
        "sample_meta": _load_json(meta_path),
        "label_meta": _load_json(label_meta_path) if label_meta_path.is_file() else None,
    }


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
    }


def _check_same_shape(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} shape mismatch: old {left.shape}, v2 {right.shape}")


def _check_coords(old_coords: np.ndarray, v2_coords: np.ndarray) -> None:
    _check_same_shape("coords", old_coords, v2_coords)
    if old_coords.ndim != 2 or old_coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N,3), found {old_coords.shape}")
    if not np.allclose(old_coords, v2_coords, rtol=0.0, atol=1e-12):
        raise ValueError("old and v2 coords differ; label audit requires matched point ordering")


def _sample_row(sample_id: str, old: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    old_temperature = np.asarray(old["temperature"], dtype=np.float64)
    v2_temperature = np.asarray(v2["temperature"], dtype=np.float64)
    _check_same_shape("temperature", old_temperature, v2_temperature)
    _check_coords(np.asarray(old["coords"], dtype=np.float64), np.asarray(v2["coords"], dtype=np.float64))

    if old_temperature.ndim != 2 or old_temperature.shape[1] != 1:
        raise ValueError(f"temperature must have shape (N,1), found {old_temperature.shape}")

    split = str(v2["sample_meta"].get("split", old["sample_meta"].get("split", "unknown")))
    old_t_ref = float(resolve_t_ref(old["sample_meta"])["value"])
    v2_t_ref = float(resolve_t_ref(v2["sample_meta"])["value"])
    old_delta = old_temperature - old_t_ref
    v2_delta = v2_temperature - v2_t_ref
    diff = v2_temperature - old_temperature

    old_hotspot = hotspot_index(old_temperature)
    v2_hotspot = hotspot_index(v2_temperature)
    label_meta = v2.get("label_meta") or {}
    solver_warnings = label_meta.get("warnings", [])
    if not isinstance(solver_warnings, list):
        solver_warnings = ["invalid_warnings_field"]

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "old_t_ref": old_t_ref,
        "v2_t_ref": v2_t_ref,
        "diff_mean": float(np.mean(diff)),
        "diff_rmse": rmse(v2_temperature, old_temperature),
        "diff_mae": mae(v2_temperature, old_temperature),
        "diff_max_abs": max_abs_error(v2_temperature, old_temperature),
        "peak_T_old": float(np.max(old_temperature)),
        "peak_T_v2": float(np.max(v2_temperature)),
        "peak_T_diff": float(np.max(v2_temperature) - np.max(old_temperature)),
        "hotspot_index_old": old_hotspot,
        "hotspot_index_v2": v2_hotspot,
        "hotspot_index_match": old_hotspot == v2_hotspot,
        "hotspot_coord_distance": hotspot_coord_distance(v2_temperature, old_temperature, old["coords"]),
        "solver_name": label_meta.get("solver_name", "not_present"),
        "solver_version": label_meta.get("solver_version", "not_present"),
        "convergence_flag": label_meta.get("convergence_flag", "not_present"),
        "residual_norm": label_meta.get("residual_norm"),
        "solver_warning_count": len(solver_warnings),
    }
    row.update(_stats(old_temperature, "T_old"))
    row.update(_stats(v2_temperature, "T_v2"))
    row.update(_stats(old_delta, "DeltaT_old"))
    row.update(_stats(v2_delta, "DeltaT_v2"))
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)

    split_summary = {}
    for split, items in sorted(split_rows.items()):
        split_summary[split] = {
            "sample_count": len(items),
            "mean_diff_rmse": float(np.mean([row["diff_rmse"] for row in items])),
            "mean_diff_mae": float(np.mean([row["diff_mae"] for row in items])),
            "max_diff_max_abs": float(np.max([row["diff_max_abs"] for row in items])),
            "mean_abs_peak_T_diff": float(np.mean([abs(row["peak_T_diff"]) for row in items])),
            "hotspot_match_count": int(sum(1 for row in items if row["hotspot_index_match"])),
            "mean_hotspot_coord_distance": float(np.mean([row["hotspot_coord_distance"] for row in items])),
        }

    residuals = [
        float(row["residual_norm"])
        for row in rows
        if row["residual_norm"] is not None and row["residual_norm"] != "not_present"
    ]
    solver_versions = Counter(f"{row['solver_name']}:{row['solver_version']}" for row in rows)
    convergence = Counter(str(row["convergence_flag"]) for row in rows)
    return {
        "sample_count": len(rows),
        "mean_diff_rmse": float(np.mean([row["diff_rmse"] for row in rows])),
        "mean_diff_mae": float(np.mean([row["diff_mae"] for row in rows])),
        "max_diff_max_abs": float(np.max([row["diff_max_abs"] for row in rows])),
        "mean_abs_peak_T_diff": float(np.mean([abs(row["peak_T_diff"]) for row in rows])),
        "hotspot_match_count": int(sum(1 for row in rows if row["hotspot_index_match"])),
        "split_summary": split_summary,
        "solver_metadata_summary": {
            "label_meta_present_count": int(sum(1 for row in rows if row["solver_name"] != "not_present")),
            "solver_versions": dict(sorted(solver_versions.items())),
            "convergence_flags": dict(sorted(convergence.items())),
            "residual_norm_min": float(np.min(residuals)) if residuals else None,
            "residual_norm_max": float(np.max(residuals)) if residuals else None,
            "residual_norm_mean": float(np.mean(residuals)) if residuals else None,
            "total_solver_warning_count": int(sum(row["solver_warning_count"] for row in rows)),
        },
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print("\nper-sample label audit")
    header = (
        "sample_id split T_old_min T_old_max T_old_mean T_v2_min T_v2_max T_v2_mean "
        "DeltaT_old_min DeltaT_old_max DeltaT_old_mean DeltaT_v2_min DeltaT_v2_max "
        "DeltaT_v2_mean diff_mean diff_rmse diff_mae diff_max_abs peak_T_old peak_T_v2 "
        "peak_T_diff hotspot_index_old hotspot_index_v2 hotspot_index_match "
        "hotspot_coord_distance solver_name solver_version convergence_flag residual_norm "
        "solver_warning_count"
    )
    print(header)
    for row in rows:
        print(
            row["sample_id"],
            row["split"],
            f"{row['T_old_min']:.8e}",
            f"{row['T_old_max']:.8e}",
            f"{row['T_old_mean']:.8e}",
            f"{row['T_v2_min']:.8e}",
            f"{row['T_v2_max']:.8e}",
            f"{row['T_v2_mean']:.8e}",
            f"{row['DeltaT_old_min']:.8e}",
            f"{row['DeltaT_old_max']:.8e}",
            f"{row['DeltaT_old_mean']:.8e}",
            f"{row['DeltaT_v2_min']:.8e}",
            f"{row['DeltaT_v2_max']:.8e}",
            f"{row['DeltaT_v2_mean']:.8e}",
            f"{row['diff_mean']:.8e}",
            f"{row['diff_rmse']:.8e}",
            f"{row['diff_mae']:.8e}",
            f"{row['diff_max_abs']:.8e}",
            f"{row['peak_T_old']:.8e}",
            f"{row['peak_T_v2']:.8e}",
            f"{row['peak_T_diff']:.8e}",
            row["hotspot_index_old"],
            row["hotspot_index_v2"],
            row["hotspot_index_match"],
            f"{row['hotspot_coord_distance']:.8e}",
            row["solver_name"],
            row["solver_version"],
            row["convergence_flag"],
            row["residual_norm"],
            row["solver_warning_count"],
        )


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nsplit summary")
    for split, values in summary["split_summary"].items():
        print(
            f"  {split}: n={values['sample_count']}, "
            f"mean_diff_rmse={values['mean_diff_rmse']:.8e}, "
            f"mean_diff_mae={values['mean_diff_mae']:.8e}, "
            f"max_diff_max_abs={values['max_diff_max_abs']:.8e}, "
            f"mean_abs_peak_T_diff={values['mean_abs_peak_T_diff']:.8e}, "
            f"hotspot_match_count={values['hotspot_match_count']}, "
            f"mean_hotspot_coord_distance={values['mean_hotspot_coord_distance']:.8e}"
        )

    solver = summary["solver_metadata_summary"]
    print("\nsolver metadata summary for v2")
    for key, value in solver.items():
        print(f"  {key}: {value}")

    print("\noverall summary")
    print(f"  sample_count: {summary['sample_count']}")
    print(f"  mean_diff_rmse: {summary['mean_diff_rmse']:.8e}")
    print(f"  mean_diff_mae: {summary['mean_diff_mae']:.8e}")
    print(f"  max_diff_max_abs: {summary['max_diff_max_abs']:.8e}")
    print(f"  mean_abs_peak_T_diff: {summary['mean_abs_peak_T_diff']:.8e}")
    print(f"  hotspot_match_count: {summary['hotspot_match_count']}")
    print("  formal_benchmark: False")
    print("  model_metric: False")
    print("  label_audit_smoke_ok: True")


def main() -> int:
    args = parse_args()
    print("Heat3D v1 smoke-vs-v2 label audit")
    print("  diagnostic only: not a model metric, not a formal benchmark")
    print(f"  old subset: {args.old_subset}")
    print(f"  v2 subset: {args.v2_subset}")

    try:
        old_samples = _sample_dirs(args.old_subset)
        v2_samples = _sample_dirs(args.v2_subset)
        common_ids = sorted(set(old_samples) & set(v2_samples))
        missing_old = sorted(set(v2_samples) - set(old_samples))
        missing_v2 = sorted(set(old_samples) - set(v2_samples))
        if missing_old or missing_v2:
            raise FileNotFoundError(f"sample mismatch, missing_old={missing_old}, missing_v2={missing_v2}")
        if args.require_count is not None and len(common_ids) != args.require_count:
            raise ValueError(f"expected {args.require_count} common samples, found {len(common_ids)}")

        rows = [
            _sample_row(sample_id, _load_sample(old_samples[sample_id]), _load_sample(v2_samples[sample_id]))
            for sample_id in common_ids
        ]
    except Exception as exc:
        print(f"label_audit_smoke_ok: False")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_rows(rows)
    _print_summary(_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
