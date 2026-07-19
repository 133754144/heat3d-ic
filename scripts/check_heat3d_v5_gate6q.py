#!/usr/bin/env python3
"""Validate Gate 6Q oracle/ridge/coverage artifacts and role boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/heat3d_v5/gate6q"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6q_registry.csv"
EXPECTED_ARTIFACTS = (
    "gate6q_diagnostics.json",
    "gate6q_diagnostics.md",
    "gate6q_sample_level.json",
    "gate6q_sample_level.csv",
    "gate6q_knn_coverage.csv",
    "gate6q_conductivity_strata.csv",
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
    assert row["ridge_alpha"] == "0.001"
    assert row["knn_k"] == "10"
    assert row["training_started"] == "false"
    assert row["training_yaml_generated"] == "false"
    assert row["test_accessed"] == "false"
    assert row["hard_accessed"] == "false"
    assert row["sealed_iid_accessed"] == "false"
    if args.preflight:
        assert row["execution_status"] in {
            "prepared_not_started",
            "completed_read_only",
        }
        print("Gate 6Q preflight passed (training_started=false)")
        return 0

    assert row["execution_status"] == "completed_read_only"
    assert row["evaluation_status"] == "completed_train_valid_only"
    payload = json.loads((ROOT / row["result_json"]).read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["ridge_fit_roles"] == ["train"]
    assert scope["ridge_query_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["training_started"] is False
    assert scope["training_yaml_generated"] is False
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
    ridge = payload["ridge_contract"]
    assert ridge["alpha"] == 0.001
    assert ridge["fit_roles"] == ["train"]
    assert ridge["query_roles"] == ["valid_iid"]
    assert ridge["valid_evaluation_count"] == 1
    assert set(ridge["models"]) == {
        "ridge_physics_24",
        "ridge_pooled_latent_96",
        "ridge_combined_120",
    }
    expected_widths = {
        "ridge_physics_24": 24,
        "ridge_pooled_latent_96": 96,
        "ridge_combined_120": 120,
    }
    for name, model in ridge["models"].items():
        assert model["feature_width"] == expected_widths[name]
        assert model["fit_roles"] == ["train"]
        assert model["query_roles"] == ["valid_iid"]
        assert model["fit_sample_count"] == 672
        assert model["query_sample_count"] == 128
        assert len(model["coefficients"]) == expected_widths[name]
    required_fields = {
        "e231",
        "e543",
        "v39_e24",
        "e231_oracle_scale",
        "e543_oracle_scale",
        "v39_e24_oracle_scale",
        "ridge_physics_24",
        "ridge_pooled_latent_96",
        "ridge_combined_120",
    }
    assert set(payload["field_metrics"]) == required_fields
    for field in payload["field_metrics"].values():
        for section in ("summary", "q4_summary", "top_sse"):
            for value in field[section].values():
                if isinstance(value, (int, float)):
                    assert math.isfinite(value)
    assert set(payload["coverage_diagnostics"]) == {
        "physics_24",
        "pooled_latent_96",
        "combined_120",
    }
    for coverage in payload["coverage_diagnostics"].values():
        assert coverage["contract"]["reference_roles"] == ["train"]
        assert coverage["contract"]["query_roles"] == ["valid_iid"]
        assert coverage["contract"]["k"] == 10
        assert coverage["contract"]["target_scale_source"] == "train neighbors only"
    strata = payload["conductivity_stratification"]
    assert strata["boundary_fit_roles"] == ["train"]
    assert strata["query_roles"] == ["valid_iid"]
    assert strata["row_count"] == 112
    manifest = json.loads((ROOT / row["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["sample_level_row_count"] == 128
    assert manifest["coverage_row_count"] == 128
    assert manifest["conductivity_strata_row_count"] == 112
    for name in EXPECTED_ARTIFACTS:
        path = BASE / name
        assert path.is_file()
        assert manifest["artifacts"][name]["sha256"] == _sha256(path)
    samples = json.loads((BASE / "gate6q_sample_level.json").read_text())
    assert len(samples) == 128
    assert {item["role"] for item in samples} == {"valid_iid"}
    with (BASE / "gate6q_sample_level.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 128
    with (BASE / "gate6q_knn_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 128
    with (BASE / "gate6q_conductivity_strata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 112
    decision = payload["bottleneck_assessment"]
    assert row["scale_only_below_20pct"] == str(
        decision["scale_only_theoretical_below_20pct"]
    ).lower()
    assert row["bottleneck"] == decision["classification"]
    assert row["unique_recommended_route"] == decision[
        "unique_recommended_route"
    ]
    print("Gate 6Q checks passed (training_runs=0, forbidden_roles=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
