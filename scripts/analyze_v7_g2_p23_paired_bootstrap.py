#!/usr/bin/env python3
"""Paired case bootstrap for an already completed P23 per-sample artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "sample_first_relative_rmse_pct",
    "rmse_K",
    "mae_K",
    "peak_temperature_absolute_error_K",
    "true_hotspot_region_rmse_K",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.per_sample.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_VALID_ONLY":
        raise SystemExit("refusing bootstrap on incomplete/non-valid-only artifact")
    rows = payload.get("rows", [])
    if len(rows) != 128 * 6:
        raise SystemExit("P23 paired bootstrap requires 128 cases x 3 seeds x 2 models")
    by_model_case_seed: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_model_case_seed.setdefault((row["model"], row["sample_id"]), []).append(row)
    models = {key[0] for key in by_model_case_seed}
    if models != {"Heat3D", "Therm-FM"}:
        raise SystemExit(f"unexpected models: {models}")
    case_ids = sorted({key[1] for key in by_model_case_seed})
    if len(case_ids) != 128:
        raise SystemExit("paired bootstrap requires exactly 128 common cases")
    arrays: dict[str, np.ndarray] = {}
    for metric in METRICS:
        heat = np.array(
            [
                np.mean([r[metric] for r in by_model_case_seed[("Heat3D", sid)]])
                for sid in case_ids
            ],
            dtype=np.float64,
        )
        therm = np.array(
            [
                np.mean([r[metric] for r in by_model_case_seed[("Therm-FM", sid)]])
                for sid in case_ids
            ],
            dtype=np.float64,
        )
        arrays[metric] = heat - therm
    rng = np.random.default_rng(230023)
    draws = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
    result = {
        "schema_version": "heat3d_v7_g2_p23_paired_bootstrap_v1",
        "status": "COMPLETE_VALID_ONLY",
        "comparison": "Heat3D - Therm-FM",
        "unit": "physical case",
        "n_cases": len(case_ids),
        "n_training_seeds_per_model": 3,
        "resamples": 10000,
        "bootstrap_seed": 230023,
        "interval": "percentile 95 percent CI",
        "conditional_on": "frozen three-seed cohort",
        "per_metric": {},
        "per_sample_artifact_sha256": sha256_file(args.per_sample),
    }
    for metric, values in arrays.items():
        boot = values[draws].mean(axis=1)
        result["per_metric"][metric] = {
            "observed_mean_difference": float(values.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
