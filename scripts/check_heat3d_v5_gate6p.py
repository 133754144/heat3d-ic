#!/usr/bin/env python3
"""Validate Gate 6P read-only diagnostics and role boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/heat3d_v5/gate6p"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6p_registry.csv"
EXPECTED_ARTIFACTS = (
    "gate6p_diagnostics.json",
    "gate6p_diagnostics.md",
    "gate6p_sample_level.json",
    "gate6p_sample_level.csv",
    "gate6p_e543_scale_features.csv",
    "gate6p_train_scale_feature_cv.csv",
    "gate6p_q4_high_error_audit.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["roles_accessed"] == "train|valid_iid"
    assert row["training_started"] == "false"
    assert row["test_accessed"] == "false"
    assert row["hard_accessed"] == "false"
    assert row["sealed_iid_accessed"] == "false"
    if args.preflight:
        assert row["execution_status"] in {
            "prepared_not_started",
            "completed_read_only",
        }
        print("Gate 6P preflight passed (training_started=false)")
        return 0
    assert row["execution_status"] == "completed_read_only"
    assert row["evaluation_status"] == "completed_train_valid_only"
    result_path = ROOT / row["result_json"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["training_started"] is False
    assert scope["checkpoint_written_or_modified"] is False
    assert scope["test_accessed"] is False
    assert scope["hard_accessed"] is False
    assert scope["sealed_iid_accessed"] is False
    assert payload["split"]["train_count"] == 672
    assert payload["split"]["valid_iid_count"] == 128
    assert payload["normalization_and_context"]["context_fit_roles"] == ["train"]
    assert payload["normalization_and_context"]["target_or_label_features"] == []
    assert payload["checkpoint_replay"]["passed"] is True
    assert (
        payload["checkpoint_replay"]["e543_feature_export_max_abs_error_K"]
        <= payload["checkpoint_replay"]["tolerance_K"]
    )
    assert payload["decomposition_max_abs_closure_K2"] <= 1.0e-7
    assert set(payload["checkpoint_binding"]) == {"e231", "e543", "v39_e24"}
    assert set(payload["transplant_provenance"]) == {
        "e543_plus_e231_global_scale_mlp",
        "e543_plus_e231_mlp_scale_attention",
        "e543_plus_e231_complete_scale_head",
    }
    for suite in payload["field_metrics"].values():
        for value in suite["summary"].values():
            if isinstance(value, (int, float)):
                assert math.isfinite(value)
    cv = payload["train_only_scale_feature_cv"]
    assert cv["fit_roles"] == ["train"]
    assert cv["query_roles"] == ["train"]
    assert cv["fold_count"] == 5
    assert set(cv["feature_sets"]) == {
        "physics_24",
        "pooled_latent_96",
        "combined_120",
        "physics_operator_no_readout",
    }
    manifest = json.loads((ROOT / row["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["sample_level_row_count"] == 128
    assert manifest["scale_feature_row_count"] == 800
    assert manifest["train_cv_row_count"] == 672
    for name in EXPECTED_ARTIFACTS:
        path = BASE / name
        assert path.is_file()
        assert manifest["artifacts"][name]["sha256"] == _sha256(path)
    samples_json = json.loads((BASE / "gate6p_sample_level.json").read_text())
    assert len(samples_json) == 128
    assert {item["role"] for item in samples_json} == {"valid_iid"}
    with (BASE / "gate6p_sample_level.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 128
    with (BASE / "gate6p_e543_scale_features.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    assert len(feature_rows) == 800
    assert {item["role"] for item in feature_rows} == {"train", "valid_iid"}
    assert row["bottleneck"] == payload["bottleneck_assessment"]["classification"]
    assert (
        row["next_training_candidate"]
        == payload["bottleneck_assessment"]["next_training_candidate"]
    )
    print("Gate 6P checks passed (training_runs=0, forbidden_roles=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
