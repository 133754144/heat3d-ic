#!/usr/bin/env python3
"""Valid-only field-shape and mechanism diagnostics for frozen P1i predictions.

This script is deliberately read-only.  Corr, Amp, peak and top-k metrics are
computed by the historical ``rigno.heat3d_v2_field_shape_diagnostics`` module;
the only presentation transform is multiplying Corr/Amp by 100 for the V4/V5
percentage convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_field_shape_diagnostics import (  # noqa: E402
    compute_field_shape_metrics,
)


GRID = (65, 65, 57)
NODE_COUNT = int(np.prod(GRID))
SHAPE_FIELDS = (
    "centered_spatial_correlation",
    "amplitude_ratio",
    "top_k_overlap",
    "peak_abs_error",
)
COMMON_FIELDS = (
    "sample_first_relative_rmse_pct",
    "rmse_K",
    "mae_K",
    "peak_temperature_absolute_error_K",
    "true_hotspot_region_rmse_K",
)
HIGH_STRATA = (
    "total_power_W",
    "top_h_W_m2K",
    "bottom_h_W_m2K",
    "material_conductivity_heterogeneity",
)


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


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 128:
        raise ValueError(f"expected 128 valid cases, got {len(cases)}")
    for case in cases:
        if case["split_role"] != "valid_iid":
            raise ValueError("non-valid case in field-shape input")
    return cases


def load_common_rows(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    rows: dict[tuple[str, int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], int(row["seed"]), row["model"])
            rows[key] = {field: float(row[field]) for field in COMMON_FIELDS}
    return rows


def as_xyz_flat(value: np.ndarray, layout: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if layout == "xyz_flat":
        flat = array.reshape(-1)
    elif layout == "x_y_z":
        if array.shape != GRID:
            raise ValueError(f"x_y_z shape {array.shape} != {GRID}")
        flat = array.reshape(-1)
    elif layout == "z_x_y":
        expected = (GRID[2], GRID[0], GRID[1])
        if array.shape != expected:
            raise ValueError(f"z_x_y shape {array.shape} != {expected}")
        flat = np.transpose(array, (1, 2, 0)).reshape(-1)
    else:
        raise ValueError(f"unknown prediction layout {layout}")
    if flat.size != NODE_COUNT:
        raise ValueError(f"prediction node count {flat.size} != {NODE_COUNT}")
    if not np.isfinite(flat).all():
        raise FloatingPointError("prediction contains non-finite values")
    return flat


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _seed_aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"case_count": len(rows)}
    for field in SHAPE_FIELDS + COMMON_FIELDS:
        result[field] = _mean(rows, field)
    result["corr_pct"] = None if result["centered_spatial_correlation"] is None else 100.0 * float(result["centered_spatial_correlation"])
    result["amp_pct"] = None if result["amplitude_ratio"] is None else 100.0 * float(result["amplitude_ratio"])
    result["amp_abs_error_pct"] = None if result["amp_pct"] is None else abs(float(result["amp_pct"]) - 100.0)
    result["top_k_overlap_pct"] = None if result["top_k_overlap"] is None else 100.0 * float(result["top_k_overlap"])
    return result


def _across_seed(seed_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"n_training_seeds": len(seed_rows)}
    fields = (
        "corr_pct",
        "amp_pct",
        "amp_abs_error_pct",
        "top_k_overlap_pct",
        "peak_abs_error",
        *COMMON_FIELDS,
    )
    for field in fields:
        values = [float(row[field]) for row in seed_rows if row.get(field) is not None]
        result[f"{field}_mean"] = float(np.mean(values)) if values else None
        result[f"{field}_sd_across_training_seeds"] = (
            float(np.std(values, ddof=1)) if len(values) >= 2 else None
        )
    return result


def _assign_high(case: dict[str, Any], bins: dict[str, Any]) -> list[str]:
    labels = []
    for field in HIGH_STRATA:
        q67 = float(bins["continuous_bins"][field]["q67"])
        if float(case[field]) > q67:
            labels.append(f"high_{field}")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--truth-archive", type=Path, required=True)
    parser.add_argument("--prediction-spec", type=Path, required=True)
    parser.add_argument("--common-csv", type=Path, required=True)
    parser.add_argument("--condition-bins", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.valid_cases)
    specs = json.loads(args.prediction_spec.read_text(encoding="utf-8"))
    common_rows = load_common_rows(args.common_csv)
    bins = json.loads(args.condition_bins.read_text(encoding="utf-8"))
    if bins["source_manifest_sha256"] != sha256_file(args.valid_cases):
        raise ValueError("condition bins are not bound to the supplied valid manifest")
    by_model_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    stratum_rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    with h5py.File(args.truth_archive, "r") as archive:
        truth_data = archive["samples/deltaT_K"]
        truth_ids = archive["samples/sample_id"]
        truth_roles = archive["samples/split_role"]
        for label, spec in specs.items():
            model = str(spec["model"])
            seed = int(spec["seed"])
            with np.load(spec["path"], allow_pickle=False) as prediction_archive:
                if set(prediction_archive.files) != {str(case["sample_id"]) for case in cases}:
                    raise ValueError(f"{label}: prediction IDs do not exactly match valid manifest")
                for case in cases:
                    sid = str(case["sample_id"])
                    row = int(case["truth_row"])
                    if _decode(truth_ids[row]) != sid:
                        raise ValueError(f"truth-row mapping mismatch for {sid}")
                    if _decode(truth_roles[row]) != "valid_iid":
                        raise ValueError(f"truth row is not valid_iid for {sid}")
                    truth = np.asarray(truth_data[row], dtype=np.float64).reshape(-1)
                    pred = as_xyz_flat(prediction_archive[sid], str(spec["layout"]))
                    shape = compute_field_shape_metrics(
                        truth,
                        pred,
                        top_k=5,
                        sample_id=sid,
                        split="valid_iid",
                    )
                    common = common_rows[(sid, seed, model)]
                    joined = {
                        "sample_id": sid,
                        "model": model,
                        "seed": seed,
                        **{field: shape[field] for field in SHAPE_FIELDS},
                        **common,
                        "strata": _assign_high(case, bins),
                    }
                    by_model_seed.setdefault((model, seed), []).append(joined)
                    for stratum in joined["strata"]:
                        stratum_rows.setdefault((model, seed, stratum), []).append(joined)
    per_seed = [
        {"model": model, "seed": seed, "metrics": _seed_aggregate(rows)}
        for (model, seed), rows in sorted(by_model_seed.items())
    ]
    aggregate = {
        model: _across_seed([item["metrics"] for item in per_seed if item["model"] == model])
        for model in sorted({item["model"] for item in per_seed})
    }
    strata = []
    for (model, seed, stratum), rows in sorted(stratum_rows.items()):
        strata.append({"model": model, "seed": seed, "stratum": stratum, "metrics": _seed_aggregate(rows)})
    strata_aggregate = {}
    for stratum in sorted({row["stratum"] for row in strata}):
        strata_aggregate[stratum] = {}
        for model in ("Heat3D", "Therm-FM"):
            seed_metrics = [row["metrics"] for row in strata if row["stratum"] == stratum and row["model"] == model]
            strata_aggregate[stratum][model] = _across_seed(seed_metrics)
    status = {
        "schema_version": "heat3d_v7_g2_p1i_field_shape_mechanism_v1",
        "status": "COMPLETE_VALID_ONLY",
        "definition_source": "rigno/heat3d_v2_field_shape_diagnostics.py",
        "definition_source_sha256": sha256_file(ROOT / "rigno/heat3d_v2_field_shape_diagnostics.py"),
        "valid_manifest_sha256": sha256_file(args.valid_cases),
        "condition_bins_sha256": sha256_file(args.condition_bins),
        "truth_archive_sha256": sha256_file(args.truth_archive),
        "case_count": len(cases),
        "nodes_per_case": NODE_COUNT,
        "top_k": 5,
        "corr_definition": "100 * centered_spatial_correlation",
        "amp_definition": "100 * amplitude_ratio; ideal=100",
        "amp_calibration_diagnostic": "abs(amp_pct - 100)",
        "dispersion_label": "SD across training seeds",
        "strata_definition": "frozen valid-only q67 high bins; model error not used for binning",
        "common_metric_aggregation": "case-mean diagnostics in this artifact; primary pooled/global metrics remain in the P23 common evaluator and corrected bootstrap",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "strata": strata,
        "strata_aggregate": strata_aggregate,
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V7 G2 P1i Field-Shape / Mechanism Diagnostics",
        "",
        "Valid-only, read-only diagnostics using the historical V4/V5 field-shape definitions.",
        "",
        "| model | Corr (%) mean ± seed SD | Amp (%) mean ± seed SD | |Amp-100| (%) | top-5 overlap (%) | hotspot RMSE K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in aggregate.items():
        lines.append(
            f"| {model} | {values['corr_pct_mean']:.4f} ± {values['corr_pct_sd_across_training_seeds']:.4f} | "
            f"{values['amp_pct_mean']:.4f} ± {values['amp_pct_sd_across_training_seeds']:.4f} | "
            f"{values['amp_abs_error_pct_mean']:.4f} | {values['top_k_overlap_pct_mean']:.4f} | "
            f"{values['true_hotspot_region_rmse_K_mean']:.4f} |"
        )
    lines += ["", "## Frozen high-condition mechanism strata", "", "| stratum | model | cases/seed | RMSE K | MAE K | Corr (%) | Amp (%) | hotspot RMSE K | peak abs K | top-5 overlap (%) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in strata:
        m = row["metrics"]
        lines.append(
            f"| {row['stratum']} | {row['model']} seed{row['seed']} | {m['case_count']} | {m['rmse_K']:.4f} | {m['mae_K']:.4f} | "
            f"{100*m['centered_spatial_correlation']:.4f} | {100*m['amplitude_ratio']:.4f} | {m['true_hotspot_region_rmse_K']:.4f} | "
            f"{m['peak_temperature_absolute_error_K']:.4f} | {100*m['top_k_overlap']:.4f} |"
        )
    lines += ["", "Corr/Amp/top-k are secondary diagnostics; primary metric and checkpoint selection are unchanged.", "", "No test_iid, sealed IID, or DeepOHeat official100 was read.", ""]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return status


if __name__ == "__main__":
    run(parse_args())
