#!/usr/bin/env python3
"""Deterministic checker for the V6 model/anchored-resolution closeout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "configs/heat3d_v6/v6_model_closeout_anchored_resolution.json"
DEFAULT_LADDER = ROOT / "configs/heat3d_v6/v6_anchored_probe_ladder.json"


def check(result_path: Path, ladder_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    canonical = yaml.safe_load(
        (ROOT / "configs/heat3d_v6/v6_layer_canonical_default.yaml").read_text()
    )
    if canonical["canonical_model_configuration"] != "V6_03_V5best_P1h":
        raise AssertionError("canonical V6 model configuration drifted")
    if canonical["canonical_model_reference_run"] != "seed0":
        raise AssertionError("reference checkpoint role drifted")
    if len(canonical["canonical_model_replication_runs"]) != 2:
        raise AssertionError("replication-run count drifted")
    if (
        result["evaluation_role"] != "valid_iid"
        or result["test_hard_accessed"]
        or result["training_executed"]
        or result["formal_inference_platform"] != "local_CPU"
    ):
        raise AssertionError("evaluation role/platform guardrail failed")
    freeze = result["three_seed_artifact_freeze"]
    if (
        freeze["config_count"] != 3
        or freeze["checkpoint_prediction_pair_count"] != 12
        or len(freeze["artifacts"]) != 12
    ):
        raise AssertionError("three-seed artifact freeze incomplete")
    for row in freeze["artifacts"]:
        if (
            len(row["checkpoint_sha256"]) != 64
            or len(row["prediction_sha256"]) != 64
            or not row["checkpoint_file"]
            or not row["prediction_file"]
        ):
            raise AssertionError("artifact hash/path incomplete")
    if len(result["rows"]) != 24 or len(result["mean_std"]) != 8:
        raise AssertionError("three-seed/resolution/pooling matrix incomplete")
    for row in result["rows"]:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise AssertionError(f"non-finite result: {key}")
    anchors = set(ladder["probes"]["1024"]["indices"])
    previous = set()
    for resolution in (1024, 2048, 4096, 8192):
        probe = ladder["probes"][str(resolution)]
        current = set(probe["indices"])
        if len(current) != resolution or not anchors <= current:
            raise AssertionError(f"{resolution}: anchor retention failed")
        if previous and not previous < current:
            raise AssertionError(f"{resolution}: ladder not strictly nested")
        previous = current
        if (
            not probe["all_layers_covered"]
            or not probe["all_interfaces_covered"]
            or probe["top_count"] < 1
            or probe["bottom_count"] < 1
        ):
            raise AssertionError(f"{resolution}: physical coverage failed")
    decision = result["workflow_decision"]
    if (
        not decision["recognized"]
        or decision["lowest_error_high_resolution"]["pooling_mode"]
        != "anchor_derived_scale_pooling"
        or not decision["canonical_1024_remains_lower_error"]
    ):
        raise AssertionError("workflow decision drifted")
    for row in result["mean_std"]:
        if (
            row["resolution"] > 1024
            and row["pooling_mode"] == "anchor_derived_scale_pooling"
            and (
                row["point_global_cv_relative_rmse_pct_mean"] >= 20.0
                or row["point_global_cv_relative_rmse_pct_std"] >= 1.0
            )
        ):
            raise AssertionError("high-resolution three-seed stability gate failed")
    attribution = result["volume_support_attribution"]
    if set(attribution["conditions"]) != {
        "source_aware_support_canonical_context",
        "volume_only_support_volume_context",
        "volume_only_support_frozen_source_aware_context",
    }:
        raise AssertionError("support/context attribution cells incomplete")
    return {
        "status": "passed",
        "canonical_model": canonical["canonical_model_configuration"],
        "artifact_pairs": len(freeze["artifacts"]),
        "matrix_rows": len(result["rows"]),
        "test_hard_accessed": False,
        "training_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    args = parser.parse_args()
    print(json.dumps(check(args.result, args.ladder), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
