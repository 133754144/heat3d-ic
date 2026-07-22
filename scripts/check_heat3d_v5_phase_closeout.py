#!/usr/bin/env python3
"""Reproducibility checker for the frozen V5 phase closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATUS = "completed_research_phase_threshold_unmet"
TAG = "v5-final-threshold-unmet"
REQUIRED_METRICS = {
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tag", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def main() -> int:
    args = _args()
    base = ROOT / "configs/heat3d_v5/final_closeout"
    manifest = _read(base / "v5_phase_closeout_manifest.json")
    q4 = _read(base / "v5_final_q4_root_audit.json")
    test = _read(base / "v42_e257_final_test_timing.json")
    if manifest["status"] != STATUS or manifest["final_tag"] != TAG:
        raise AssertionError("closeout status/tag drift")
    if manifest["training_started_by_closeout"] or manifest["checkpoint_selection_modified"]:
        raise AssertionError("closeout mutated training/selection state")
    if q4["scope"] != {
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "test_accessed": False,
        "hard_accessed": False,
        "sealed_iid_accessed": False,
        "training_started": False,
        "target_used_for_coverage_features": False,
    }:
        raise AssertionError("Q4 role boundary drift")
    scope = test["scope"]
    if scope["roles_accessed"] != ["train", "test_iid"]:
        raise AssertionError("final test role boundary drift")
    if scope["hard_accessed"] or scope["sealed_iid_accessed"] or scope["training_started"]:
        raise AssertionError("forbidden final-test action recorded")
    if scope["checkpoint_reselected"] or scope["hyperparameters_tuned"]:
        raise AssertionError("test result affected selection/tuning")
    if test["binding"]["checkpoint_epoch"] != 257 or test["binding"]["test_iid_count"] != 128:
        raise AssertionError("V42 checkpoint/test binding drift")
    if not REQUIRED_METRICS <= set(test["test_iid"]["summary"]):
        raise AssertionError("final test metric suite incomplete")
    if manifest["threshold"]["passed"] or manifest["threshold"]["valid_iid_pct"] < 20.0 or manifest["threshold"]["test_iid_pct"] < 20.0:
        raise AssertionError("threshold closeout drift")
    integration_commit = subprocess.check_output(
        ["git", "rev-parse", "integration/v5-core"], cwd=ROOT, text=True
    ).strip()
    if manifest["integration"]["commit"] != integration_commit or manifest["integration"]["merge_executed"]:
        raise AssertionError("integration branch binding/merge state drift")
    if _csv_count(base / "v5_final_q4_samples.csv") != 128:
        raise AssertionError("Q4 sample CSV count drift")
    if _csv_count(base / "v42_e257_test_iid_per_sample.csv") != 128:
        raise AssertionError("test sample CSV count drift")
    for entry in manifest["artifacts"].values():
        path = ROOT / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise AssertionError(f"artifact hash drift: {path}")

    csv.field_size_limit(sys.maxsize)
    registry = ROOT / "configs/heat3d_v5/v5_gate6q_training_registry.csv"
    with registry.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    v42 = [row for row in rows if row["config_id"] == "V4P5_42_gate6q_objective_only_e600"]
    if len(v42) != 1:
        raise AssertionError("V42 registry row missing/duplicated")
    row = v42[0]
    if row["execution_status"] != "completed_e600" or row["evaluation_status"] != "completed_valid_iid_and_final_test_iid":
        raise AssertionError("V42 lifecycle drift")
    if row["result_v5_threshold_pass"] != "fail" or not row["result_v5_primary_test_point_global_relative_rmse_pct"]:
        raise AssertionError("V42 final test registry fields incomplete")

    if args.require_tag:
        tag_commit = subprocess.check_output(
            ["git", "rev-list", "-n", "1", TAG], cwd=ROOT, text=True
        ).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if tag_commit != head:
            raise AssertionError(f"{TAG} does not bind HEAD")
    print(
        json.dumps(
            {
                "status": "passed",
                "phase_status": STATUS,
                "q4_samples": 128,
                "test_iid_samples": 128,
                "tag_required": args.require_tag,
                "training_started": False,
                "hard_accessed": False,
                "sealed_iid_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
