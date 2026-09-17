#!/usr/bin/env python3
"""Check valid-only Therm-FM replay against the frozen P22 metric semantics.

This is a read-only sanity check.  It indexes only the 128 rows in the
independently staged valid_iid manifest and never discovers or iterates over
other split roles.  The peak metrics intentionally match P22's historical
definition (maximum absolute field error), while the P23 common evaluator has
its separately frozen max-of-fields definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GRID = (65, 65, 57)
TOLERANCE = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--truth-archive", type=Path, required=True)
    parser.add_argument("--prediction-spec", type=Path, required=True)
    parser.add_argument("--historical-aggregation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_payload = json.loads(args.valid_cases.read_text(encoding="utf-8"))
    cases = case_payload["cases"]
    if case_payload.get("role") != "valid_iid" or len(cases) != 128:
        raise ValueError("valid-only manifest contract failed")
    specs = json.loads(args.prediction_spec.read_text(encoding="utf-8"))
    expected_payload = json.loads(args.historical_aggregation.read_text(encoding="utf-8"))
    expected_by_seed = {
        int(row["seed"]): row["best_metrics"] for row in expected_payload["per_seed"]
    }
    observed_by_seed: dict[int, dict[str, float]] = {}
    with h5py.File(args.truth_archive, "r") as archive:
        truth_ds = archive["samples/deltaT_K"]
        role_ds = archive["samples/split_role"]
        for label, spec in sorted(specs.items()):
            if spec.get("model") != "Therm-FM" or spec.get("value_kind") != "deltaT_K":
                raise ValueError(f"unsupported replay spec {label}")
            seed = int(spec["seed"])
            prediction = np.load(Path(spec["path"]), allow_pickle=False)
            sample_relative: list[float] = []
            peak_abs: list[float] = []
            sum_sq_error = 0.0
            sum_sq_truth = 0.0
            sum_abs_error = 0.0
            total_points = 0
            try:
                for case in cases:
                    row = int(case["truth_row"])
                    if decode(role_ds[row]) != "valid_iid":
                        raise ValueError("truth row is not valid_iid")
                    sid = str(case["sample_id"])
                    truth = np.asarray(truth_ds[row], dtype=np.float64).reshape(GRID)
                    if spec.get("layout") == "z_x_y":
                        pred = np.asarray(prediction[sid], dtype=np.float64).transpose(1, 2, 0)
                    else:
                        pred = np.asarray(prediction[sid], dtype=np.float64).reshape(GRID)
                    error = pred - truth
                    if not (np.isfinite(error).all() and np.isfinite(truth).all()):
                        raise FloatingPointError("non-finite replay value")
                    ef = error.reshape(-1)
                    tf = truth.reshape(-1)
                    sample_relative.append(float(np.linalg.norm(ef) / np.linalg.norm(tf) * 100.0))
                    peak_abs.append(float(np.max(np.abs(ef))))
                    sum_sq_error += float(np.square(ef).sum())
                    sum_sq_truth += float(np.square(tf).sum())
                    sum_abs_error += float(np.abs(ef).sum())
                    total_points += int(ef.size)
            finally:
                prediction.close()
            observed_by_seed[seed] = {
                "sample_first_relative_rmse_pct": float(np.mean(sample_relative)),
                "point_global_relative_rmse_pct": float(np.sqrt(sum_sq_error / sum_sq_truth) * 100.0),
                "rmse_K": float(np.sqrt(sum_sq_error / total_points)),
                "mae_K": float(sum_abs_error / total_points),
                "peak_abs_error_K_mean": float(np.mean(peak_abs)),
                "peak_error_rmse_K": float(np.sqrt(np.mean(np.square(peak_abs)))),
            }
    comparisons: list[dict[str, Any]] = []
    pass_status = True
    for seed in sorted(observed_by_seed):
        observed = observed_by_seed[seed]
        expected = expected_by_seed[seed]
        diffs = {metric: observed[metric] - float(expected[metric]) for metric in observed}
        max_abs_diff = max(abs(value) for value in diffs.values())
        pass_status = pass_status and max_abs_diff <= TOLERANCE
        comparisons.append({"seed": seed, "observed": observed, "expected_p22": expected, "difference": diffs, "max_abs_difference": max_abs_diff})
    result = {
        "schema_version": "heat3d_v7_g2_p23_thermfm_replay_sanity_v1",
        "status": "REPLAY_REPRODUCTION_PASS" if pass_status else "REPLAY_REPRODUCTION_DISCREPANCY",
        "authorized_role": "valid_iid",
        "valid_count": 128,
        "metric_semantics": "P22 unweighted sample-relative norm; point-global SSE; RMSE/MAE; peak=max absolute field error",
        "numerical_tolerance": TOLERANCE,
        "comparisons": comparisons,
        "valid_cases_sha256": sha256_file(args.valid_cases),
        "prediction_spec_sha256": sha256_file(args.prediction_spec),
        "historical_aggregation_sha256": sha256_file(args.historical_aggregation),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if pass_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
