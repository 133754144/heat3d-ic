#!/usr/bin/env python3
"""Read-only Heat3D v2 field-shape diagnostics for existing predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import find_sample_dirs, load_json, resolve_t_ref  # noqa: E402
from rigno.heat3d_v2_field_shape_diagnostics import (  # noqa: E402
    build_field_shape_report,
    compute_field_shape_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze field-shape diagnostics for existing Heat3D v2 predictions. "
            "This is read-only diagnostics tooling and does not train."
        )
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--trained-predictions", type=Path, required=True)
    parser.add_argument("--prediction-label", choices=("final", "best"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stdout-mode", choices=("compact", "full", "quiet"), default="compact")
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _prediction_loader(path: Path):
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


def _load_sample_rows(subset: Path, trained_predictions: Path, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    sample_dirs = find_sample_dirs(_sample_root(subset))
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {subset}")
    load_prediction = _prediction_loader(trained_predictions)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        try:
            meta = load_json(sample_dir / "sample_meta.json")
            sample_id = str(meta.get("sample_id", sample_dir.name))
            split = str(meta.get("split", "unknown"))
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
                f"{sample_id} trained prediction",
            )
            t_ref_info = resolve_t_ref(meta)
            t_ref = float(t_ref_info["value"])
            true_delta = true_temperature.reshape(-1) - t_ref
            pred_delta = pred_temperature.reshape(-1) - t_ref
            row = compute_field_shape_metrics(
                true_delta,
                pred_delta,
                top_k=top_k,
                sample_id=sample_id,
                split=split,
            )
            row["T_ref"] = t_ref
            row["T_ref_source"] = t_ref_info.get("source")
            if row.get("warnings"):
                warnings.extend(f"{sample_id}: {item}" for item in row["warnings"])
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive per-sample diagnostics
            failures.append({"sample_id": sample_id, "sample_dir": str(sample_dir), "error": str(exc)})
            warnings.append(f"{sample_id}: failed field-shape diagnostics: {exc}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": "unknown",
                    "failed": True,
                    "error": str(exc),
                    "warning_count": 1,
                    "warnings": [str(exc)],
                }
            )

    return rows, failures, warnings


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if "data" in path.parts:
        raise ValueError("--output-json must not be under data/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.8e}"


def _render_metric_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| group | samples | field variance ratio | centered correlation | amplitude ratio | peak abs error | p95 error | p99 error | top-k overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        group = row.get("split") or row.get("group") or "overall"
        lines.append(
            f"| {group} | {row.get('valid_sample_count', row.get('sample_count'))} | "
            f"{_fmt(row.get('field_variance_ratio'))} | "
            f"{_fmt(row.get('centered_spatial_correlation'))} | "
            f"{_fmt(row.get('amplitude_ratio'))} | "
            f"{_fmt(row.get('peak_abs_error'))} | "
            f"{_fmt(row.get('p95_error'))} | "
            f"{_fmt(row.get('p99_error'))} | "
            f"{_fmt(row.get('top_k_overlap'))} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v2 Field-Shape Diagnostics",
        "",
        "This report is read-only diagnostics for existing recovered-temperature predictions. It is not a formal benchmark, production solver validation, or OOD generalization claim.",
        "",
        "## Inputs",
        "",
        f"- prediction_label: `{payload['prediction_label']}`",
        f"- subset: `{payload['inputs']['subset']}`",
        f"- trained_predictions: `{payload['inputs']['trained_predictions']}`",
        f"- field_semantics: `{payload['diagnostic_scope']['field_semantics']}`",
        f"- sample_count: `{payload['sample_count']}`",
        f"- failed_sample_count: `{payload['failed_sample_count']}`",
        "",
    ]
    overall = {"group": "overall", **payload["overall"]}
    lines.extend(_render_metric_table("Overall", [overall]))
    lines.append("")
    lines.extend(_render_metric_table("Split Summary", payload["split_summary"]))
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for item in payload["warnings"][:50]:
            lines.append(f"- {item}")
        if len(payload["warnings"]) > 50:
            lines.append(f"- truncated additional warnings: {len(payload['warnings']) - 50}")
    lines.append("")
    return "\n".join(lines)


def analyze_field_shape_diagnostics(
    *,
    subset: Path,
    trained_predictions: Path,
    prediction_label: str,
    output_json: Path,
    output_md: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")
    rows, failures, warnings = _load_sample_rows(subset, trained_predictions, top_k)
    report = build_field_shape_report(rows, warnings=warnings, failures=failures)
    payload = {
        "diagnostic_scope": {
            "tool": "Heat3D v2 field-shape diagnostics",
            "read_only": True,
            "not_formal_benchmark": True,
            "field_semantics": (
                "Recovered temperature predictions and true temperature are converted to DeltaT "
                "by subtracting the non-leaking T_ref resolved from sample metadata."
            ),
        },
        "prediction_label": prediction_label,
        "inputs": {
            "subset": str(subset),
            "trained_predictions": str(trained_predictions),
            "prediction_schema": (
                "recovered-temperature predictions loaded from .npz arrays keyed by sample_id; "
                "directory fallback supports <sample_id>.npy, <sample_id>/temperature.npy, or "
                "<sample_id>/pred_temperature.npy"
            ),
            "top_k": int(top_k),
        },
        "outputs": {
            "json": str(output_json),
            "markdown": str(output_md),
        },
        "sample_count": len(rows),
        "failed_sample_count": len(failures),
        **report,
    }
    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _print_stdout(payload: dict[str, Any], stdout_mode: str) -> None:
    outputs = payload["outputs"]
    if stdout_mode == "quiet":
        _emit(f"field_shape_diagnostics_written: json={outputs['json']} markdown={outputs['markdown']}")
        return

    overall = payload["overall"]
    _emit("Heat3D v2 field-shape diagnostics")
    _emit("  scope: read-only diagnostics; not formal benchmark")
    _emit(f"  prediction_label: {payload['prediction_label']}")
    _emit(f"  sample_count: {payload['sample_count']} failed_sample_count: {payload['failed_sample_count']}")
    _emit(
        "  overall: "
        f"field_variance_ratio={_fmt(overall.get('field_variance_ratio'))} "
        f"centered_spatial_correlation={_fmt(overall.get('centered_spatial_correlation'))} "
        f"amplitude_ratio={_fmt(overall.get('amplitude_ratio'))} "
        f"peak_abs_error={_fmt(overall.get('peak_abs_error'))} "
        f"p95_error={_fmt(overall.get('p95_error'))} "
        f"p99_error={_fmt(overall.get('p99_error'))} "
        f"top_k_overlap={_fmt(overall.get('top_k_overlap'))}"
    )
    if stdout_mode == "full":
        for row in payload["split_summary"]:
            _emit(
                "  split: "
                f"{row['split']} samples={row['valid_sample_count']} "
                f"field_variance_ratio={_fmt(row.get('field_variance_ratio'))} "
                f"centered_spatial_correlation={_fmt(row.get('centered_spatial_correlation'))} "
                f"top_k_overlap={_fmt(row.get('top_k_overlap'))}"
            )
        for warning in payload["warnings"][:10]:
            _emit(f"  warning: {warning}")
    _emit(f"  output_json: {outputs['json']}")
    _emit(f"  output_md: {outputs['markdown']}")


def main() -> int:
    args = parse_args()
    payload = analyze_field_shape_diagnostics(
        subset=args.subset,
        trained_predictions=args.trained_predictions,
        prediction_label=args.prediction_label,
        output_json=args.output_json,
        output_md=args.output_md,
        top_k=args.top_k,
    )
    _print_stdout(payload, args.stdout_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
