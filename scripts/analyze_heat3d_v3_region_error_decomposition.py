#!/usr/bin/env python3
"""Read-only S5-family region error decomposition.

This script compares existing Heat3D prediction archives and decomposes raw
DeltaT errors over geometric/physics regions plus metadata condition groups.
It does not import JAX, build graphs, execute a model, or train.
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_heat3d_v3_prediction_mechanisms as mech  # noqa: E402


DEFAULT_SUBSET = (
    Path("data")
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_ENTRIES = (
    "S5_base_best=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5_seed0_e1600_"
    "warmupcosine_lr5e-4_minlr5e-5_wd1e-4:best_predictions.npz",
    "S5_base_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5_seed0_e1600_"
    "warmupcosine_lr5e-4_minlr5e-5_wd1e-4:predictions.npz",
    "S5final_FT_nomask_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5final_FT_e100_"
    "lr1e-5_nomask_wd1e-4:predictions.npz",
    "S5final_EM_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5final_EM_e100_"
    "lr1e-5_edgemask0p02_wd1e-4:predictions.npz",
)
REGION_NAMES = (
    "global",
    "top10_deltaT",
    "top5_deltaT",
    "q_positive",
    "strong_q",
    "background_low_deltaT",
)
GROUP_KEYS = (
    "split",
    "source_category",
    "k_region_mode",
    "bc_category",
    "q_power_range",
    "top_h",
)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only S5-family raw DeltaT region error decomposition. "
            "Entries use LABEL=RUN_DIR:PREDICTION_NPZ."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--entry", action="append", default=None)
    parser.add_argument(
        "--strong-q-quantile",
        type=float,
        default=0.90,
        help="Sample-wise q>0 quantile for the strong-q region.",
    )
    parser.add_argument(
        "--background-quantile",
        type=float,
        default=0.50,
        help="Sample-wise target DeltaT quantile for low-background region.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/heat3d_v3_s5_family_error_decomposition/region_error_decomposition.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/heat3d_v3_s5_family_error_decomposition/region_error_decomposition.md"),
    )
    return parser.parse_args()


def _parse_entry(token: str) -> tuple[str, Path, str]:
    if "=" not in token or ":" not in token:
        raise ValueError(f"entry must be LABEL=RUN_DIR:PREDICTION_NPZ, found {token!r}")
    label, rest = token.split("=", 1)
    run_dir_text, prediction_name = rest.rsplit(":", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty entry label in {token!r}")
    return label, Path(run_dir_text), prediction_name


def _json_float(value: Any) -> float | None:
    return mech._json_float(value)


def _region_metrics(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if target.shape != pred.shape or target.shape != mask.shape:
        raise ValueError(f"region shape mismatch target={target.shape} pred={pred.shape} mask={mask.shape}")
    count = int(np.sum(mask))
    if count <= 0:
        return {
            "point_count": 0,
            "mask_fraction": 0.0,
            "rmse": None,
            "mae": None,
            "bias": None,
        }
    error = pred[mask] - target[mask]
    return {
        "point_count": count,
        "mask_fraction": _json_float(count / max(int(mask.size), 1)),
        "rmse": _json_float(float(np.sqrt(np.mean(np.square(error))))),
        "mae": _json_float(float(np.mean(np.abs(error)))),
        "bias": _json_float(float(np.mean(error))),
    }


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64).reshape(-1), quantile))


def _strong_q_mask(q_field: np.ndarray, quantile: float) -> np.ndarray:
    q_values = np.asarray(q_field, dtype=np.float64).reshape(-1)
    positive = q_values > 0.0
    if not np.any(positive):
        return np.zeros_like(q_values, dtype=bool)
    threshold = float(np.quantile(q_values[positive], quantile))
    return np.logical_and(positive, q_values >= threshold)


def _top_h_group(metadata: dict[str, Any], sample_meta: dict[str, Any]) -> str:
    for key in ("top_h", "top_h_W_m2K", "h_top", "top_contact_h"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
        value = sample_meta.get(key)
        if value is not None:
            return str(value)
    boundary = sample_meta.get("boundary_conditions", {})
    if isinstance(boundary, dict):
        for key in ("top_h", "h_top", "top_contact_h"):
            value = boundary.get(key)
            if value is not None:
                return str(value)
    plan = mech._plan(sample_meta)
    for key in ("top_h", "h_top", "top_contact_h"):
        value = plan.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _load_reference_samples(subset: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sample_dirs = mech.find_sample_dirs(mech._sample_root(subset))
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {subset}")
    pending = []
    q_powers = []
    failures = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        try:
            sample_meta = mech.load_json(sample_dir / "sample_meta.json")
            metadata = mech._read_optional_json(sample_dir / "metadata.json")
            sample_id = str(
                metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name
            )
            coords = np.load(sample_dir / "coords.npy")
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"{sample_id}: coords.npy must have shape (N, 3), found {coords.shape}")
            n_points = int(coords.shape[0])
            true_temperature = mech._as_column(
                np.load(sample_dir / "temperature.npy"),
                n_points,
                f"{sample_id} temperature.npy",
            )
            q_field = mech._as_column(
                np.load(sample_dir / "q_field.npy"),
                n_points,
                f"{sample_id} q_field.npy",
            )
            t_ref = float(mech.resolve_t_ref(sample_meta)["value"])
            q_power = mech._integrated_power(metadata, sample_meta, q_field)
            q_powers.append(float(q_power))
            groups = {
                "split": mech._meta_value(metadata, sample_meta, "split"),
                "source_category": mech._meta_value(metadata, sample_meta, "source_pattern_tag"),
                "k_region_mode": mech._meta_value(metadata, sample_meta, "k_region_mode"),
                "bc_category": mech._meta_value(metadata, sample_meta, "bc_category"),
                "top_h": _top_h_group(metadata, sample_meta),
            }
            pending.append(
                {
                    "sample_id": sample_id,
                    "groups": groups,
                    "target_delta": true_temperature.reshape(-1) - t_ref,
                    "t_ref": float(t_ref),
                    "q_field": q_field.reshape(-1),
                    "q_power": float(q_power),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            failures.append({"sample_id": sample_id, "sample_dir": str(sample_dir), "error": str(exc)})

    q_edges = mech._q_power_edges(q_powers) if q_powers else {"ranges": []}
    references = {}
    for item in pending:
        item["groups"]["q_power_range"] = mech._q_power_range(
            float(item["q_power"]),
            q_edges["ranges"],
        )
        references[str(item["sample_id"])] = item
    return references, {"sample_failures": failures, "q_power_edges": q_edges}


def _sample_region_rows(
    *,
    label: str,
    references: dict[str, dict[str, Any]],
    prediction_path: Path,
    strong_q_quantile: float,
    background_quantile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_prediction, prediction_keys = mech._prediction_loader(prediction_path)
    rows = []
    failures = []
    for sample_id, ref in sorted(references.items()):
        try:
            target = np.asarray(ref["target_delta"], dtype=np.float64).reshape(-1)
            t_ref = float(ref["t_ref"])
            pred_temperature = mech._as_column(
                load_prediction(sample_id),
                int(target.size),
                f"{sample_id} prediction",
            )
            pred = pred_temperature.reshape(-1) - t_ref
            q_field = np.asarray(ref["q_field"], dtype=np.float64).reshape(-1)
            top10_threshold = _safe_quantile(target, 0.90)
            top5_threshold = _safe_quantile(target, 0.95)
            background_threshold = _safe_quantile(target, background_quantile)
            masks = {
                "global": np.ones_like(target, dtype=bool),
                "top10_deltaT": target >= top10_threshold,
                "top5_deltaT": target >= top5_threshold,
                "q_positive": q_field > 0.0,
                "strong_q": _strong_q_mask(q_field, strong_q_quantile),
                "background_low_deltaT": np.logical_and(target <= background_threshold, q_field <= 0.0),
            }
            for region_name, mask in masks.items():
                metrics = _region_metrics(target, pred, mask)
                row = {
                    "label": label,
                    "sample_id": sample_id,
                    "region": region_name,
                    "groups": dict(ref["groups"]),
                }
                row.update(metrics)
                rows.append(row)
        except Exception as exc:  # pragma: no cover - per-sample defensive diagnostics
            failures.append({"label": label, "sample_id": sample_id, "error": str(exc)})
    unused_keys = sorted(set(prediction_keys) - set(references))
    if unused_keys:
        failures.append(
            {
                "label": label,
                "sample_id": None,
                "error": "prediction archive has unused sample keys",
                "unused_key_count": len(unused_keys),
                "unused_key_examples": unused_keys[:10],
            }
        )
    return rows, failures


def _aggregate_region_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "sample_count": len({row["sample_id"] for row in rows}),
        "row_count": len(rows),
        "point_count": int(sum(int(row.get("point_count") or 0) for row in rows)),
    }
    for field in ("rmse", "mae", "bias", "mask_fraction"):
        values = [mech._json_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        result[field] = mech._json_float(float(np.mean(values))) if values else None
    return result


def _grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for key in GROUP_KEYS:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[(row["region"], str(row["groups"].get(key, "unknown")))].append(row)
        items = []
        for (region, value), group_rows in sorted(buckets.items()):
            item = {
                "region": region,
                "group_key": key,
                "group_value": value,
            }
            item.update(_aggregate_region_rows(group_rows))
            items.append(item)
        payload[key] = items
    return payload


def _weak_groups(grouped: dict[str, list[dict[str, Any]]], limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for group_rows in grouped.values():
        rows.extend(group_rows)
    return sorted(
        rows,
        key=lambda row: (
            -1.0 if mech._json_float(row.get("rmse")) is None else mech._json_float(row.get("rmse")),
            row.get("group_key", ""),
            row.get("group_value", ""),
        ),
        reverse=True,
    )[:limit]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if "data" in path.parts:
        raise ValueError("--output-json must not be under data/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = mech._json_float(value)
    if number is None:
        return "-"
    return f"{number:.6g}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v3 S5-Family Region Error Decomposition",
        "",
        "Read-only decomposition of existing prediction archives. No training or graph build is performed.",
        "",
        "## Overall Regions",
        "",
        "| label | region | samples | mask_fraction | RMSE | MAE | bias |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in payload["entries"]:
        label = entry["label"]
        for region in REGION_NAMES:
            row = entry["regions"].get(region, {})
            lines.append(
                f"| {label} | {region} | {row.get('sample_count', 0)} | "
                f"{_fmt(row.get('mask_fraction'))} | {_fmt(row.get('rmse'))} | "
                f"{_fmt(row.get('mae'))} | {_fmt(row.get('bias'))} |"
            )
    lines.extend(["", "## Weak Groups", ""])
    lines.append("| label | region | group | samples | RMSE | MAE | mask_fraction |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for entry in payload["entries"]:
        for row in entry["weak_groups"][:10]:
            group = f"{row.get('group_key')}={row.get('group_value')}"
            lines.append(
                f"| {entry['label']} | {row.get('region')} | {group} | "
                f"{row.get('sample_count')} | {_fmt(row.get('rmse'))} | "
                f"{_fmt(row.get('mae'))} | {_fmt(row.get('mask_fraction'))} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.strong_q_quantile <= 1.0:
        raise ValueError("--strong-q-quantile must be in [0, 1]")
    if not 0.0 <= args.background_quantile <= 1.0:
        raise ValueError("--background-quantile must be in [0, 1]")
    entries = [_parse_entry(token) for token in (args.entry or DEFAULT_ENTRIES)]
    references, reference_meta = _load_reference_samples(args.subset)
    payload = {
        "diagnostic_scope": "read-only region error decomposition; no training",
        "subset": str(args.subset),
        "strong_q_quantile": float(args.strong_q_quantile),
        "background_quantile": float(args.background_quantile),
        "region_definitions": {
            "global": "all points",
            "top10_deltaT": "sample-wise target DeltaT >= p90",
            "top5_deltaT": "sample-wise target DeltaT >= p95",
            "q_positive": "q > 0",
            "strong_q": "sample-wise q > 0 and q >= positive-q quantile",
            "background_low_deltaT": "target DeltaT <= background quantile and q <= 0",
        },
        "reference": reference_meta,
        "entries": [],
    }
    for label, run_dir, prediction_name in entries:
        prediction_path = run_dir / prediction_name
        rows, failures = _sample_region_rows(
            label=label,
            references=references,
            prediction_path=prediction_path,
            strong_q_quantile=float(args.strong_q_quantile),
            background_quantile=float(args.background_quantile),
        )
        regions = {}
        for region in REGION_NAMES:
            regions[region] = _aggregate_region_rows([row for row in rows if row["region"] == region])
        grouped = _grouped(rows)
        payload["entries"].append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "prediction_name": prediction_name,
                "prediction_path": str(prediction_path),
                "regions": regions,
                "grouped": grouped,
                "weak_groups": _weak_groups(grouped),
                "per_sample_regions": rows,
                "failure_count": len(failures),
                "failures": failures,
            }
        )

    _write_json(args.output_json, payload)
    if "data" in args.output_md.parts:
        raise ValueError("--output-md must not be under data/")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
