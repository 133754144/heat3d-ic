#!/usr/bin/env python3
"""P23 valid-only common full-field evaluator.

The evaluator deliberately requires an independently staged *valid-only* case
manifest.  It never discovers cases from a mixed-role CSV and never iterates
over unspecified rows in the truth archive.  The input prediction archives are
also indexed only by the authorized valid sample IDs.

This is an implementation artifact for the P23 protocol.  It is not run in
the current closeout because the V6 seed-1/seed-2 per-sample archives are not
available and a prior mixed-role CSV inspection created a governance blocker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GRID = (65, 65, 57)
NODE_COUNT = int(np.prod(GRID))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def load_valid_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("valid case manifest must contain a list under 'cases'")
    if len(cases) != 128:
        raise ValueError(f"P23 requires exactly 128 valid cases, got {len(cases)}")
    ids: set[str] = set()
    rows: set[int] = set()
    for case in cases:
        if case.get("split_role") != "valid_iid":
            raise ValueError("case manifest contains a non-valid_iid row")
        sid = str(case["sample_id"])
        row = int(case["truth_row"])
        if sid in ids or row in rows:
            raise ValueError("duplicate sample_id or truth_row in valid case manifest")
        ids.add(sid)
        rows.add(row)
    return cases


def as_xyz_flat(value: np.ndarray, layout: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if layout == "xyz_flat":
        flat = array.reshape(-1)
    elif layout == "x_y_z":
        if array.shape != GRID:
            raise ValueError(f"x_y_z prediction shape {array.shape} != {GRID}")
        flat = array.reshape(-1)
    elif layout == "z_x_y":
        if array.shape != (GRID[2], GRID[0], GRID[1]):
            raise ValueError(
                f"z_x_y prediction shape {array.shape} != {(GRID[2], GRID[0], GRID[1])}"
            )
        flat = np.transpose(array, (1, 2, 0)).reshape(-1)
    else:
        raise ValueError(f"unknown prediction layout {layout}")
    if flat.size != NODE_COUNT:
        raise ValueError(f"prediction has {flat.size} nodes, expected {NODE_COUNT}")
    return flat


def prediction_delta_t(value: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    flat = as_xyz_flat(value, str(spec["layout"]))
    kind = str(spec.get("value_kind", "deltaT_K"))
    if kind == "deltaT_K":
        return flat
    if kind == "temperature_K":
        return flat - float(spec.get("reference_temperature_K", 300.0))
    raise ValueError(f"unsupported value_kind {kind}")


def one_case_metrics(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    error = pred - truth
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise FloatingPointError("non-finite prediction or truth")
    truth_energy = float(np.sum(truth * truth))
    weighted_truth_energy = float(np.sum(weights * truth * truth))
    if truth_energy <= 0.0 or weighted_truth_energy <= 0.0:
        raise ValueError("truth energy must be positive")
    hotspot_count = int(np.ceil(0.01 * truth.size))
    hotspot_idx = np.argpartition(truth, -hotspot_count)[-hotspot_count:]
    sse = float(np.sum(error * error))
    truth_sse = float(np.sum(truth * truth))
    cv_sse = float(np.sum(weights * error * error))
    cv_truth_energy = float(np.sum(weights * truth * truth))
    hotspot_sse = float(np.sum(error[hotspot_idx] ** 2))
    return {
        "sse": sse,
        "truth_sse": truth_sse,
        "cv_sse": cv_sse,
        "cv_truth_energy": cv_truth_energy,
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "node_count": int(error.size),
        "hotspot_sse": hotspot_sse,
        "hotspot_count": hotspot_count,
        "sample_first_relative_rmse_pct": float(
            100.0 * np.sqrt(cv_sse / weighted_truth_energy)
        ),
        "point_global_relative_rmse_pct": float(
            100.0 * np.sqrt(sse / truth_sse)
        ),
        "rmse_K": float(np.sqrt(sse / error.size)),
        "mae_K": float(np.sum(np.abs(error)) / error.size),
        "peak_temperature_absolute_error_K": float(abs(np.max(pred) - np.max(truth))),
        "true_hotspot_region_rmse_K": float(np.sqrt(hotspot_sse / hotspot_count)),
    }


def summarize(rows: list[dict[str, Any]], require_three_seeds: bool = True) -> dict[str, Any]:
    metrics = (
        "sample_first_relative_rmse_pct",
        "point_global_relative_rmse_pct",
        "rmse_K",
        "mae_K",
        "peak_temperature_absolute_error_K",
        "true_hotspot_region_rmse_K",
    )
    by_model_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_model_seed.setdefault((row["model"], int(row["seed"])), []).append(row)
    seed_summaries: list[dict[str, Any]] = []
    for (model, seed), model_rows in sorted(by_model_seed.items()):
        sse = sum(float(r["sse"]) for r in model_rows)
        truth_sse = sum(float(r["truth_sse"]) for r in model_rows)
        cv_sse = sum(float(r["cv_sse"]) for r in model_rows)
        cv_truth_energy = sum(float(r["cv_truth_energy"]) for r in model_rows)
        abs_error = sum(float(r["absolute_error_sum"]) for r in model_rows)
        node_count = sum(int(r["node_count"]) for r in model_rows)
        hotspot_sse = sum(float(r["hotspot_sse"]) for r in model_rows)
        hotspot_count = sum(int(r["hotspot_count"]) for r in model_rows)
        seed_summaries.append(
            {
                "model": model,
                "seed": seed,
                "valid_cases": len(model_rows),
                "metrics": {
                    "sample_first_relative_rmse_pct": float(
                        np.mean([r["sample_first_relative_rmse_pct"] for r in model_rows])
                    ),
                    "point_global_relative_rmse_pct": float(
                        100.0 * np.sqrt(sse / truth_sse)
                    ),
                    "cv_weighted_point_global_relative_rmse_pct": float(
                        100.0 * np.sqrt(cv_sse / cv_truth_energy)
                    ),
                    "rmse_K": float(np.sqrt(sse / node_count)),
                    "mae_K": float(abs_error / node_count),
                    "peak_temperature_mae_K": float(
                        np.mean([r["peak_temperature_absolute_error_K"] for r in model_rows])
                    ),
                    "peak_temperature_rmse_K": float(
                        np.sqrt(
                            np.mean(
                                [r["peak_temperature_absolute_error_K"] ** 2 for r in model_rows]
                            )
                        )
                    ),
                    "true_hotspot_region_rmse_K": float(
                        np.sqrt(hotspot_sse / hotspot_count)
                    ),
                },
            }
        )
    aggregate: dict[str, Any] = {}
    for model in sorted({r["model"] for r in rows}):
        seeds = [s for s in seed_summaries if s["model"] == model]
        if require_three_seeds and len(seeds) != 3:
            raise ValueError(f"P23 requires three frozen seeds for {model}, got {len(seeds)}")
        aggregate[model] = {
            "n_training_seeds": len(seeds),
            "dispersion_label": "SD across training seeds",
            "metrics_mean": {
                metric: float(np.mean([s["metrics"][metric] for s in seeds]))
                for metric in seeds[0]["metrics"]
            },
            "metrics_sd_across_training_seeds": (
                {
                    metric: float(np.std([s["metrics"][metric] for s in seeds], ddof=1))
                    for metric in seeds[0]["metrics"]
                }
                if len(seeds) >= 2
                else {metric: None for metric in seeds[0]["metrics"]}
            ),
        }
    return {"per_seed": seed_summaries, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--truth-archive", type=Path, required=True)
    parser.add_argument(
        "--prediction-spec",
        type=Path,
        required=True,
        help="JSON mapping labels to {model, seed, path, layout, value_kind}",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Emit a diagnostic partial valid-only artifact; never a complete P23 comparison.",
    )
    args = parser.parse_args()

    cases = load_valid_cases(args.valid_cases)
    specs = json.loads(args.prediction_spec.read_text(encoding="utf-8"))
    if not isinstance(specs, dict) or not specs:
        raise ValueError("prediction spec must be a non-empty object")
    open_predictions: list[tuple[str, dict[str, Any], Any]] = []
    try:
        for label, spec in specs.items():
            if spec.get("model") not in ("Heat3D", "Therm-FM"):
                raise ValueError(f"P23 common evaluator does not accept model {spec.get('model')}")
            path = Path(spec["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            open_predictions.append((str(label), spec, np.load(path, allow_pickle=False)))
        rows: list[dict[str, Any]] = []
        with h5py.File(args.truth_archive, "r") as archive:
            truth_dataset = archive["samples/deltaT_K"]
            role_dataset = archive["samples/split_role"]
            weights = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
            if weights.shape != (NODE_COUNT,):
                raise ValueError(f"control-volume shape {weights.shape} != {(NODE_COUNT,)}")
            for case in cases:
                sid = str(case["sample_id"])
                truth_row = int(case["truth_row"])
                if _decode(role_dataset[truth_row]) != "valid_iid":
                    raise ValueError(f"truth row for {sid} is not valid_iid")
                truth = np.asarray(truth_dataset[truth_row], dtype=np.float64).reshape(-1)
                if truth.size != NODE_COUNT:
                    raise ValueError(f"truth for {sid} has {truth.size} nodes")
                for label, spec, prediction_archive in open_predictions:
                    if sid not in prediction_archive.files:
                        raise ValueError(f"{label} missing authorized valid case {sid}")
                    pred = prediction_delta_t(prediction_archive[sid], spec)
                    row = {
                        "sample_id": sid,
                        "seed": int(spec["seed"]),
                        "model": str(spec["model"]),
                        "label": label,
                    }
                    row.update(one_case_metrics(pred, truth, weights))
                    rows.append(row)
        summary = summarize(rows, require_three_seeds=not args.allow_partial)
        payload = {
            "schema_version": "heat3d_v7_g2_p23_common_fullfield_evaluation_v1",
            "status": (
                "PARTIAL_VALID_ONLY_DIAGNOSTIC"
                if args.allow_partial and len(rows)
                else ("COMPLETE_VALID_ONLY" if len(rows) else "EMPTY")
            ),
            "authorized_role": "valid_iid",
            "valid_case_count": len(cases),
            "nodes_per_sample": NODE_COUNT,
            "truth_archive_sha256": sha256_file(args.truth_archive),
            "valid_cases_sha256": sha256_file(args.valid_cases),
            "prediction_spec_sha256": sha256_file(args.prediction_spec),
            "hotspot_definition": "top 1 percent nodes by T_true within each sample",
            "test_iid_read": False,
            "sealed_read": False,
            "deepoheat_official100_read": False,
            "partial_diagnostic_only": bool(args.allow_partial),
            "rows": rows,
            **summary,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        fieldnames = list(rows[0]) if rows else ["sample_id", "seed", "model", "label"]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        for _, _, archive in open_predictions:
            archive.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
