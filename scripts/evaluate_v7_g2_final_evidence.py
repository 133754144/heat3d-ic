#!/usr/bin/env python3
"""Final valid-only evidence evaluator for the V7 G2 freeze.

This evaluator is deliberately resolution-agnostic (native 1024 or the
240825-node full field) but resolution-explicit in its input contract.  It
only consumes a frozen valid_iid manifest and prediction archives named in an
explicit prediction spec.  It never discovers rows from a mixed-role file and
never opens test/sealed roles.

The shape metrics reproduce the historical V4 definitions and the V5
control-volume definitions:

* Corr_point (%) = 100 * unweighted centred spatial correlation;
* Amp_range (%) = 100 * range(pred) / range(truth);
* Corr_CV (%) = 100 * control-volume-weighted centred correlation;
* Amp_CVRMS (%) = 100 * CV_RMS(pred) / CV_RMS(truth).

The script writes only compact JSON/CSV receipts.  Prediction archives and
labels remain in their existing ignored locations.
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


EPS = 1.0e-15


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
    if not isinstance(cases, list) or len(cases) != 128:
        raise ValueError("the final evidence route requires exactly 128 valid_iid cases")
    ids: set[str] = set()
    rows: set[int] = set()
    for case in cases:
        if str(case.get("split_role")) != "valid_iid":
            raise ValueError("valid manifest contains a non-valid_iid row")
        sid = str(case["sample_id"])
        row = int(case["truth_row"])
        if sid in ids or row in rows:
            raise ValueError("duplicate valid sample_id or truth_row")
        ids.add(sid)
        rows.add(row)
    return cases


def _corr(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        x = pred - float(np.mean(pred))
        y = truth - float(np.mean(truth))
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        wsum = float(np.sum(w))
        x = pred - float(np.sum(w * pred) / wsum)
        y = truth - float(np.sum(w * truth) / wsum)
        denom = float(np.sqrt(np.sum(w * x * x) * np.sum(w * y * y)))
        x = np.sqrt(w) * x
        y = np.sqrt(w) * y
    if denom <= EPS:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _cv_rms(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * value * value) / np.sum(weights)))


def _prediction_layout(value: np.ndarray, layout: str, node_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if layout in {"flat", "xyz_flat"}:
        flat = array.reshape(-1)
    elif layout == "x_y_z":
        expected = (65, 65, 57)
        if array.shape != expected:
            raise ValueError(f"x_y_z shape {array.shape} != {expected}")
        flat = array.reshape(-1)
    elif layout == "z_x_y":
        expected = (57, 65, 65)
        if array.shape != expected:
            raise ValueError(f"z_x_y shape {array.shape} != {expected}")
        flat = np.transpose(array, (1, 2, 0)).reshape(-1)
    else:
        raise ValueError(f"unsupported prediction layout {layout}")
    if flat.size != node_count:
        raise ValueError(f"prediction node count {flat.size} != {node_count}")
    if not np.isfinite(flat).all():
        raise FloatingPointError("prediction contains non-finite values")
    return flat


def _load_prediction_archive(path: Path, spec: dict[str, Any], valid_ids: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        fmt = str(spec.get("format", "key_npz"))
        if fmt == "batch_npz":
            if "sample_ids" not in archive or "prediction_deltaT_K" not in archive:
                raise ValueError(f"{path} is not a batch prediction archive")
            ids = [_decode(x) for x in archive["sample_ids"]]
            values = archive["prediction_deltaT_K"]
            if len(ids) != values.shape[0]:
                raise ValueError(f"{path}: sample_ids/prediction batch mismatch")
            result = {sid: values[i] for i, sid in enumerate(ids)}
        elif fmt == "key_npz":
            result = {sid: archive[sid] for sid in archive.files}
        else:
            raise ValueError(f"unsupported prediction archive format {fmt}")
    if set(result) != valid_ids:
        missing = sorted(valid_ids - set(result))[:3]
        extra = sorted(set(result) - valid_ids)[:3]
        raise ValueError(f"prediction IDs do not exactly match valid manifest (missing={missing}, extra={extra})")
    return result


def one_case_metrics(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if pred.size != truth.size or truth.size != weights.size:
        raise ValueError("prediction/truth/control-volume node counts differ")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise FloatingPointError("non-finite truth or prediction")
    error = pred - truth
    sse = float(np.sum(error * error))
    truth_sse = float(np.sum(truth * truth))
    cv_sse = float(np.sum(weights * error * error))
    cv_truth_energy = float(np.sum(weights * truth * truth))
    if truth_sse <= EPS or cv_truth_energy <= EPS:
        raise ValueError("truth energy must be positive")
    hotspot_count = int(np.ceil(0.01 * truth.size))
    truth_hotspot = np.argpartition(truth, -hotspot_count)[-hotspot_count:]
    pred_hotspot = np.argpartition(pred, -hotspot_count)[-hotspot_count:]
    truth_top5 = np.argpartition(truth, -5)[-5:]
    pred_top5 = np.argpartition(pred, -5)[-5:]
    amp_range = float((np.max(pred) - np.min(pred)) / max(np.max(truth) - np.min(truth), EPS))
    true_scale = _cv_rms(truth, weights)
    pred_scale = _cv_rms(pred, weights)
    amp_cvrms = float(pred_scale / max(true_scale, EPS))
    hotspot_sse = float(np.sum(error[truth_hotspot] ** 2))
    return {
        "sse": sse,
        "truth_sse": truth_sse,
        "cv_sse": cv_sse,
        "cv_truth_energy": cv_truth_energy,
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "node_count": int(pred.size),
        "hotspot_sse": hotspot_sse,
        "hotspot_count": hotspot_count,
        "sample_first_relative_rmse_pct": float(100.0 * np.sqrt(cv_sse / cv_truth_energy)),
        "point_global_relative_rmse_pct": float(100.0 * np.sqrt(sse / truth_sse)),
        "rmse_K": float(np.sqrt(sse / pred.size)),
        "mae_K": float(np.mean(np.abs(error))),
        "corr_point_pct": float(100.0 * _corr(pred, truth)),
        "amp_range_pct": float(100.0 * amp_range),
        "corr_cv_pct": float(100.0 * _corr(pred, truth, weights)),
        "amp_cvrms_pct": float(100.0 * amp_cvrms),
        "amp_range_abs_error_pct_points": float(abs(100.0 * amp_range - 100.0)),
        "amp_cvrms_abs_error_pct_points": float(abs(100.0 * amp_cvrms - 100.0)),
        "scale_log_error": float(np.log(max(amp_cvrms, EPS))),
        "peak_temperature_absolute_error_K": float(abs(np.max(pred) - np.max(truth))),
        "true_hotspot_region_rmse_K": float(np.sqrt(hotspot_sse / hotspot_count)),
        "legacy_top5_overlap_pct": float(100.0 * len(set(truth_top5.tolist()) & set(pred_top5.tolist())) / 5.0),
        "true_hotspot_top1_overlap_pct": float(100.0 * len(set(truth_hotspot.tolist()) & set(pred_hotspot.tolist())) / hotspot_count),
        "signed_temperature_bias_K": float(np.mean(error)),
        "field_std_ratio": float(np.std(pred) / max(np.std(truth), EPS)),
    }


def aggregate_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "sample_first_relative_rmse_pct", "point_global_relative_rmse_pct",
        "cv_weighted_point_global_relative_rmse_pct", "rmse_K", "mae_K",
        "corr_point_pct", "amp_range_pct", "corr_cv_pct", "amp_cvrms_pct",
        "amp_range_abs_error_pct_points", "amp_cvrms_abs_error_pct_points",
        "scale_log_error_mean", "scale_log_error_rmse", "peak_temperature_mae_K",
        "peak_temperature_rmse_K", "true_hotspot_region_rmse_K",
        "legacy_top5_overlap_pct", "true_hotspot_top1_overlap_pct",
        "signed_temperature_bias_K", "field_std_ratio",
    ]
    sse = sum(float(r["sse"]) for r in rows)
    truth_sse = sum(float(r["truth_sse"]) for r in rows)
    cv_sse = sum(float(r["cv_sse"]) for r in rows)
    cv_truth = sum(float(r["cv_truth_energy"]) for r in rows)
    abs_sum = sum(float(r["absolute_error_sum"]) for r in rows)
    node_count = sum(int(r["node_count"]) for r in rows)
    hotspot_sse = sum(float(r["hotspot_sse"]) for r in rows)
    hotspot_count = sum(int(r["hotspot_count"]) for r in rows)
    values: dict[str, float] = {
        "sample_first_relative_rmse_pct": float(np.mean([r["sample_first_relative_rmse_pct"] for r in rows])),
        "point_global_relative_rmse_pct": float(100.0 * np.sqrt(sse / truth_sse)),
        "cv_weighted_point_global_relative_rmse_pct": float(100.0 * np.sqrt(cv_sse / cv_truth)),
        "rmse_K": float(np.sqrt(sse / node_count)),
        "mae_K": float(abs_sum / node_count),
        "corr_point_pct": float(np.mean([r["corr_point_pct"] for r in rows])),
        "amp_range_pct": float(np.mean([r["amp_range_pct"] for r in rows])),
        "corr_cv_pct": float(np.mean([r["corr_cv_pct"] for r in rows])),
        "amp_cvrms_pct": float(np.mean([r["amp_cvrms_pct"] for r in rows])),
        "amp_range_abs_error_pct_points": float(np.mean([r["amp_range_abs_error_pct_points"] for r in rows])),
        "amp_cvrms_abs_error_pct_points": float(np.mean([r["amp_cvrms_abs_error_pct_points"] for r in rows])),
        "scale_log_error_mean": float(np.mean([r["scale_log_error"] for r in rows])),
        "scale_log_error_rmse": float(np.sqrt(np.mean([r["scale_log_error"] ** 2 for r in rows]))),
        "peak_temperature_mae_K": float(np.mean([r["peak_temperature_absolute_error_K"] for r in rows])),
        "peak_temperature_rmse_K": float(np.sqrt(np.mean([r["peak_temperature_absolute_error_K"] ** 2 for r in rows]))),
        "true_hotspot_region_rmse_K": float(np.sqrt(hotspot_sse / hotspot_count)),
        "legacy_top5_overlap_pct": float(np.mean([r["legacy_top5_overlap_pct"] for r in rows])),
        "true_hotspot_top1_overlap_pct": float(np.mean([r["true_hotspot_top1_overlap_pct"] for r in rows])),
        "signed_temperature_bias_K": float(np.mean([r["signed_temperature_bias_K"] for r in rows])),
        "field_std_ratio": float(np.mean([r["field_std_ratio"] for r in rows])),
    }
    return {"valid_cases": len(rows), "metrics": {field: values[field] for field in fields}}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_model_seed.setdefault((str(row["model"]), int(row["seed"])), []).append(row)
    per_seed = []
    for (model, seed), model_rows in sorted(by_model_seed.items()):
        per_seed.append({"model": model, "seed": seed, **aggregate_seed(model_rows)})
    aggregate: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        entries = [row for row in per_seed if row["model"] == model]
        if len(entries) != 3:
            raise ValueError(f"final evidence requires three seeds for {model}")
        metrics = entries[0]["metrics"]
        aggregate[model] = {
            "n_training_seeds": 3,
            "dispersion_label": "SD across training seeds",
            "metrics_mean": {k: float(np.mean([e["metrics"][k] for e in entries])) for k in metrics},
            "metrics_sd_across_training_seeds": {k: float(np.std([e["metrics"][k] for e in entries], ddof=1)) for k in metrics},
        }
    return {"per_seed": per_seed, "aggregate": aggregate}


def load_truth_native(sample_root: Path, sid: str) -> tuple[np.ndarray, np.ndarray]:
    base = sample_root / sid
    return np.load(base / "deltaT.npy"), np.load(base / "control_volume.npy")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_valid_cases(args.valid_cases)
    valid_ids = {str(case["sample_id"]) for case in cases}
    specs = json.loads(args.prediction_spec.read_text(encoding="utf-8"))
    if not isinstance(specs, dict) or not specs:
        raise ValueError("prediction spec must be a non-empty object")
    archives = []
    for label, spec in specs.items():
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        archives.append((label, spec, _load_prediction_archive(path, spec, valid_ids)))
    rows: list[dict[str, Any]] = []
    archive_path = args.truth_archive
    h5_archive = None
    if args.domain == "full":
        if archive_path is None:
            raise ValueError("full domain requires --truth-archive")
        h5_archive = h5py.File(archive_path, "r")
        truth_ds = h5_archive["samples/deltaT_K"]
        role_ds = h5_archive["samples/split_role"]
        weights_shared = np.asarray(h5_archive["shared/control_volume_m3"][:], dtype=np.float64)
        if weights_shared.size != 240825:
            raise ValueError("full-field control-volume node count is not 240825")
    try:
        for case in cases:
            sid = str(case["sample_id"])
            if args.domain == "native":
                truth, weights = load_truth_native(args.sample_root, sid)
                truth_row = int(case["truth_row"])
            else:
                truth_row = int(case["truth_row"])
                if _decode(role_ds[truth_row]) != "valid_iid":
                    raise ValueError(f"truth row for {sid} is not valid_iid")
                truth = np.asarray(truth_ds[truth_row], dtype=np.float64).reshape(-1)
                weights = weights_shared
            for label, spec, prediction_map in archives:
                node_count = int(truth.size)
                pred = _prediction_layout(prediction_map[sid], str(spec.get("layout", "flat")), node_count)
                metrics = one_case_metrics(pred, truth, np.asarray(weights, dtype=np.float64))
                rows.append({"sample_id": sid, "truth_row": truth_row, "model": str(spec["model"]), "seed": int(spec["seed"]), "label": label, **metrics})
    finally:
        if h5_archive is not None:
            h5_archive.close()
    summary = summarize(rows)
    payload = {
        "schema_version": "heat3d_v7_g2_final_evidence_evaluator_v1",
        "status": "COMPLETE_VALID_ONLY",
        "domain": args.domain,
        "valid_case_count": len(cases),
        "nodes_per_case": int(rows[0]["node_count"]),
        "valid_manifest_sha256": sha256_file(args.valid_cases),
        "truth_archive_sha256": sha256_file(archive_path) if archive_path else None,
        "prediction_spec_sha256": sha256_file(args.prediction_spec),
        "metric_definitions": {
            "sample_first_relative_rmse_pct": "100 * mean_cases(sqrt(sum(CV*(pred-truth)^2)/sum(CV*truth^2)))",
            "point_global_relative_rmse_pct": "100 * sqrt(sum_all_nodes(error^2)/sum_all_nodes(truth^2))",
            "cv_weighted_point_global_relative_rmse_pct": "100 * sqrt(sum_all_nodes(CV*error^2)/sum_all_nodes(CV*truth^2))",
            "corr_point_pct": "100 * unweighted centered spatial correlation",
            "amp_range_pct": "100 * (max(pred)-min(pred))/(max(truth)-min(truth))",
            "corr_cv_pct": "100 * control-volume-weighted centered spatial correlation",
            "amp_cvrms_pct": "100 * CV_RMS(pred)/CV_RMS(truth)",
            "hotspot": "top 1% nodes by T_true within each valid sample",
        },
        "shape_definition_sources": {
            "v4_corr_amp": "rigno/heat3d_v2_field_shape_diagnostics.py",
            "v5_corr_amp_scale": "rigno/heat3d_v5_metrics.py and rigno/heat3d_v5_shape_scale.py",
        },
        "forbidden_access": {"test_iid": False, "sealed": False, "deepoheat_official100": False},
        "rows": rows,
        **summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("native", "full"), required=True)
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--truth-archive", type=Path)
    parser.add_argument("--prediction-spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.domain == "native" and args.sample_root is None:
        parser.error("native domain requires --sample-root")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
