#!/usr/bin/env python3
"""Read-only Heat3D v3 condition and hard-sample error mining.

The script consumes existing prediction archives plus sample metadata. It does
not import JAX, build graphs, execute a model, or train.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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
    "S5final_FT_nomask_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5final_FT_e100_"
    "lr1e-5_nomask_wd1e-4:predictions.npz",
    "P4_best=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL_e100_"
    "lr1e-5_hotspot0p05_strongq0p05_wd1e-4:best_predictions.npz",
    "A_hotspot_only_best=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p05_strongq0p00_lr1e-5_wd1e-4:best_predictions.npz",
    "A_hotspot_only_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p05_strongq0p00_lr1e-5_wd1e-4:predictions.npz",
    "B_hotspot_strongq025_best=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p05_strongq0p025_lr1e-5_wd1e-4:best_predictions.npz",
    "B_hotspot_strongq025_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p05_strongq0p025_lr1e-5_wd1e-4:predictions.npz",
    "C_light_both_best=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p025_strongq0p025_lr1e-5_wd1e-4:best_predictions.npz",
    "C_light_both_final=output/heat3d_v2_runs/"
    "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5finalFT_HL2_"
    "hotspot0p025_strongq0p025_lr1e-5_wd1e-4:predictions.npz",
)
GROUP_KEYS = (
    "split",
    "source_category",
    "k_region_mode",
    "bc_category",
    "q_power_range",
    "top_h_category",
    "k_mode",
)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only condition error mining over existing Heat3D predictions."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--entry", action="append", default=None, help="LABEL=RUN_DIR:PREDICTION_NPZ")
    parser.add_argument("--strong-q-quantile", type=float, default=0.90)
    parser.add_argument("--hard-sample-count", type=int, default=50)
    parser.add_argument("--hard-sample-weight", type=float, default=1.25)
    parser.add_argument("--hard-sample-split", default="train")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/heat3d_v3_condition_error_mining/condition_error_mining.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/heat3d_v3_condition_error_mining/condition_error_mining.md"),
    )
    parser.add_argument(
        "--hard-sample-json",
        type=Path,
        default=Path("output/heat3d_v3_condition_error_mining/hard_sample_weights.json"),
    )
    return parser.parse_args()


def _parse_entry(token: str) -> tuple[str, Path, str]:
    if "=" not in token or ":" not in token:
        raise ValueError(f"entry must be LABEL=RUN_DIR:PREDICTION_NPZ, found {token!r}")
    label, rest = token.split("=", 1)
    run_dir, prediction_name = rest.rsplit(":", 1)
    return label.strip(), Path(run_dir), prediction_name


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _normalization(run_dir: Path) -> tuple[float, float]:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        return 0.0, 1.0
    data = mech.load_json(run_config_path)
    stats = data.get("train_only_normalization", {})
    mean = _scalar(stats.get("target_delta_mean", 0.0))
    std = max(_scalar(stats.get("target_delta_std", 1.0)), EPS)
    return mean, std


def _top_h_category(metadata: dict[str, Any], sample_meta: dict[str, Any]) -> str:
    candidates = [
        metadata.get("top_h"),
        metadata.get("h_top"),
        sample_meta.get("top_h"),
        sample_meta.get("h_top"),
    ]
    boundary = sample_meta.get("boundary_conditions", {})
    if isinstance(boundary, dict):
        candidates.extend([boundary.get("top_h"), boundary.get("h_top")])
    for value in candidates:
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if v < 50:
            return "very_low_top_h"
        if v < 200:
            return "low_top_h"
        if v > 2000:
            return "very_high_top_h"
        if v > 800:
            return "high_top_h"
        return "nominal_top_h"
    return "unknown"


def _reference_samples(subset: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sample_dirs = mech.find_sample_dirs(mech._sample_root(subset))
    pending = []
    q_powers = []
    failures = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        try:
            sample_meta = mech.load_json(sample_dir / "sample_meta.json")
            metadata = mech._read_optional_json(sample_dir / "metadata.json")
            sample_id = str(metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name)
            coords = np.load(sample_dir / "coords.npy")
            n_points = int(coords.shape[0])
            temperature = mech._as_column(np.load(sample_dir / "temperature.npy"), n_points, f"{sample_id} temperature")
            q_field = mech._as_column(np.load(sample_dir / "q_field.npy"), n_points, f"{sample_id} q_field")
            t_ref = float(mech.resolve_t_ref(sample_meta)["value"])
            q_power = float(mech._integrated_power(metadata, sample_meta, q_field))
            q_powers.append(q_power)
            groups = {
                "split": mech._meta_value(metadata, sample_meta, "split"),
                "source_category": mech._meta_value(metadata, sample_meta, "source_pattern_tag"),
                "k_region_mode": mech._meta_value(metadata, sample_meta, "k_region_mode"),
                "bc_category": mech._meta_value(metadata, sample_meta, "bc_category"),
                "k_mode": mech._meta_value(metadata, sample_meta, "k_mode"),
                "top_h_category": _top_h_category(metadata, sample_meta),
            }
            pending.append(
                {
                    "sample_id": sample_id,
                    "target_delta": temperature.reshape(-1) - t_ref,
                    "q_field": q_field.reshape(-1),
                    "t_ref": t_ref,
                    "q_power": q_power,
                    "groups": groups,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic robustness
            failures.append({"sample_id": sample_id, "sample_dir": str(sample_dir), "error": str(exc)})
    q_edges = mech._q_power_edges(q_powers) if q_powers else {"ranges": []}
    refs = {}
    for item in pending:
        item["groups"]["q_power_range"] = mech._q_power_range(float(item["q_power"]), q_edges["ranges"])
        refs[str(item["sample_id"])] = item
    return refs, {"sample_failures": failures, "q_power_edges": q_edges}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _region_rmse(error: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    return _rmse(error[mask])


def _sample_metrics(label: str, run_dir: Path, prediction_name: str, refs: dict[str, dict[str, Any]], strong_q_quantile: float) -> list[dict[str, Any]]:
    prediction_path = run_dir / prediction_name
    load_prediction, _ = mech._prediction_loader(prediction_path)
    norm_mean, norm_std = _normalization(run_dir)
    rows = []
    for sample_id, ref in sorted(refs.items()):
        target = np.asarray(ref["target_delta"], dtype=np.float64).reshape(-1)
        pred_temperature = mech._as_column(load_prediction(sample_id), int(target.size), f"{sample_id} prediction")
        pred = pred_temperature.reshape(-1) - float(ref["t_ref"])
        error = pred - target
        norm_error = error / norm_std
        abs_target = np.abs(target)
        top10 = abs_target >= float(np.quantile(abs_target, 0.90))
        top5 = abs_target >= float(np.quantile(abs_target, 0.95))
        q = np.asarray(ref["q_field"], dtype=np.float64).reshape(-1)
        positive_q = q > 0.0
        if np.any(positive_q):
            q_threshold = float(np.quantile(q[positive_q], strong_q_quantile))
            strong_q = np.logical_and(positive_q, q >= q_threshold)
        else:
            strong_q = np.zeros_like(q, dtype=bool)
        rows.append(
            {
                "label": label,
                "sample_id": sample_id,
                "prediction_name": prediction_name,
                "normalized_mse": float(np.mean(np.square(norm_error))),
                "rmse": _rmse(error),
                "top10_rmse": _region_rmse(error, top10),
                "top5_rmse": _region_rmse(error, top5),
                "strong_q_rmse": _region_rmse(error, strong_q),
                "peak_abs_error": float(np.max(np.abs(error))),
                "tmax_error": float(np.max(pred) - np.max(target)),
                **ref["groups"],
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        for key in GROUP_KEYS:
            groups[(metric, key, str(row.get(key, "unknown")))].append(float(value))
    out = []
    for (metric_name, group_key, group_value), values in groups.items():
        out.append(
            {
                "metric": metric_name,
                "group_key": group_key,
                "group_value": group_value,
                "sample_count": len(values),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
        )
    out.sort(key=lambda item: (item["mean"], item["max"]), reverse=True)
    return out


def _hard_sample_weights(
    rows: list[dict[str, Any]],
    count: int,
    hard_sample_weight: float,
    hard_sample_split: str,
) -> dict[str, Any]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    eligible_rows = [
        row for row in rows if not hard_sample_split or str(row.get("split", "")) == hard_sample_split
    ]
    for row in eligible_rows:
        by_sample[str(row["sample_id"])].append(float(row["normalized_mse"]))
    ranked = [
        {
            "sample_id": sample_id,
            "score": float(max(scores)),
            "mean_score": float(np.mean(scores)),
            "weight": float(hard_sample_weight),
        }
        for sample_id, scores in by_sample.items()
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = ranked[: max(int(count), 0)]
    return {
        "schema_version": "heat3d_v3_hard_sample_weights_v1",
        "description": "Hard sample weights mined from read-only V3 condition error diagnostics.",
        "weight_policy": "hard_sample_list",
        "default_weight": 1.0,
        "hard_sample_weight": float(hard_sample_weight),
        "hard_sample_split": hard_sample_split,
        "source_row_count": int(len(eligible_rows)),
        "recommended_normalize": True,
        "hard_samples": selected,
        "sample_weights": {item["sample_id"]: item["weight"] for item in selected},
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Heat3D v3 Condition Error Mining",
        "",
        "Read-only mining over existing predictions. No training, graph build, or model execution.",
        "",
        "## Top Hard Groups",
        "",
        "| metric | group | value | n | mean | max |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["hard_groups"][:30]:
        lines.append(
            f"| {row['metric']} | {row['group_key']} | {row['group_value']} | "
            f"{row['sample_count']} | {row['mean']:.6g} | {row['max']:.6g} |"
        )
    lines.extend(["", "## Top Hard Samples", "", "| sample_id | score | mean_score | weight |", "| --- | ---: | ---: | ---: |"])
    for row in payload["hard_sample_weights"]["hard_samples"][:30]:
        lines.append(
            f"| {row['sample_id']} | {row['score']:.6g} | {row['mean_score']:.6g} | {row['weight']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    entries = [_parse_entry(item) for item in (args.entry or DEFAULT_ENTRIES)]
    refs, reference_meta = _reference_samples(args.subset)
    all_rows: list[dict[str, Any]] = []
    for label, run_dir, prediction_name in entries:
        all_rows.extend(_sample_metrics(label, run_dir, prediction_name, refs, args.strong_q_quantile))
    metrics = ("normalized_mse", "top5_rmse", "top10_rmse", "strong_q_rmse", "peak_abs_error")
    hard_groups = []
    for metric in metrics:
        hard_groups.extend(_aggregate(all_rows, metric)[:15])
    hard_groups.sort(key=lambda item: (item["mean"], item["max"]), reverse=True)
    weights = _hard_sample_weights(
        all_rows,
        args.hard_sample_count,
        args.hard_sample_weight,
        args.hard_sample_split,
    )
    payload = {
        "diagnostic_scope": "read-only condition error mining; not formal benchmark evidence",
        "entries": [
            {"label": label, "run_dir": str(run_dir), "prediction_name": prediction_name}
            for label, run_dir, prediction_name in entries
        ],
        "reference_meta": reference_meta,
        "group_keys": list(GROUP_KEYS),
        "per_sample": all_rows,
        "hard_groups": hard_groups,
        "hard_sample_weights": weights,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.hard_sample_json.parent.mkdir(parents=True, exist_ok=True)
    args.hard_sample_json.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_md(args.output_md, payload)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.hard_sample_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
