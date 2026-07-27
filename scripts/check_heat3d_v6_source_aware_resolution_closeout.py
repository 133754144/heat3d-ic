#!/usr/bin/env python3
"""Deterministically check the V6 source-aware resolution closeout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RATIOS = {
    "source": 0.5,
    "volume": 0.25,
    "interface": 0.125,
    "top": 0.0625,
    "bottom": 0.0625,
}


def _finite_mapping(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_mapping(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_mapping(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}")


def check(ladder_path: Path, result_path: Path, metrics_path: Path) -> dict:
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    canonical = yaml.safe_load(
        (ROOT / "configs/heat3d_v6/v6_layer_canonical_default.yaml").read_text(
            encoding="utf-8"
        )
    )
    prior = json.loads(
        (
            ROOT / "configs/heat3d_v6/v6_model_closeout_anchored_resolution.json"
        ).read_text(encoding="utf-8")
    )
    if (
        canonical["canonical_model_configuration"] != "V6_03_V5best_P1h"
        or canonical["canonical_model_reference_run"] != "seed0"
        or set(canonical["canonical_model_replication_runs"])
        != {"V6_03_V5best_P1h_seed1", "V6_03_V5best_P1h_seed2"}
        or canonical["registered_ablation"]
        != "V6_04_V5best_P1h_DualAttention"
    ):
        raise AssertionError("canonical model roles drifted")
    freeze = prior["three_seed_artifact_freeze"]
    if (
        freeze["config_count"] != 3
        or freeze["checkpoint_prediction_pair_count"] != 12
        or len(freeze["artifacts"]) != 12
    ):
        raise AssertionError("frozen three-seed artifact inventory is incomplete")
    for row in freeze["artifacts"]:
        if len(row["checkpoint_sha256"]) != 64 or len(row["prediction_sha256"]) != 64:
            raise AssertionError("frozen checkpoint/prediction SHA256 is incomplete")
    if (
        ladder["stratum_ratios"] != EXPECTED_RATIOS
        or not ladder["strictly_nested"]
        or not ladder["canonical_1024_retained_exactly"]
        or ladder["test_hard_accessed"]
        or ladder["training_executed"]
    ):
        raise AssertionError("ladder contract drifted")
    anchors = ladder["probes"]["1024"]["indices"]
    previous: set[int] = set()
    for resolution in (1024, 2048, 4096, 8192, 16384, 32768):
        probe = ladder["probes"][str(resolution)]
        indices = probe["indices"]
        current = set(indices)
        if (
            len(indices) != resolution
            or len(current) != resolution
            or indices[:1024] != anchors
            or (previous and not previous < current)
        ):
            raise AssertionError(f"{resolution}: strict nesting/anchor order failed")
        previous = current
        if (
            probe["stratum_ratios"] != EXPECTED_RATIOS
            or not probe["all_layers_covered"]
            or not probe["all_interfaces_covered"]
            or probe["top_count"] < 1
            or probe["bottom_count"] < 1
            or not probe["source_box_coverage"]["all_source_boxes_covered"]
            or probe["source_box_coverage"]["zero_covered_source_box_count"] != 0
        ):
            raise AssertionError(f"{resolution}: physical coverage failed")
        if probe["source_box_coverage"]["node_count_per_source_box"]["min"] < 1:
            raise AssertionError(f"{resolution}: a registered source box is uncovered")
    if (
        result["status"] != "completed"
        or result["scope"]["evaluation_role"] != "valid_iid"
        or result["scope"]["test_hard_accessed"]
        or result["scope"]["training_executed"]
        or result["scope"]["checkpoint_modified"]
        or result["scope"]["formal_platform"] != "local_CPU"
    ):
        raise AssertionError("result scope/platform guardrail failed")
    search = result["search"]
    if (
        search["seed0_completed_resolutions"] != [1024, 2048, 4096, 8192, 16384]
        or search["first_failed_resolution"] != 32768
        or search["maximum_stable_resolution"] != 16384
        or search["failure_metric_status"] != "not_available_inference_not_reached"
        or search["failure_elapsed_seconds_lower_bound"] < 1800
    ):
        raise AssertionError("resolution stopping rule drifted")
    if (
        ladder["next_resolution"] != 65536
        or ladder["next_resolution_status"]
        != "infeasible_exact_stratum_ratio_on_unique_solver_nodes"
        or search["next_ratio_exact_resolution"] != 65536
    ):
        raise AssertionError("next-resolution capacity audit drifted")
    rows = result["seed0_rows"]
    if len(rows) != 5 or any(not row["gate_passed"] for row in rows):
        raise AssertionError("seed0 completed ladder is incomplete")
    if any(int(row["warm_repeat_count"]) < 10 for row in rows):
        raise AssertionError("stable inference repeat count is below 10")
    if any(row["gpu_memory"] != "N/A_CPU_only" for row in rows):
        raise AssertionError("CPU/GPU accounting was mixed")
    aggregate = result["multi_seed_mean_std"]
    for mode in (
        "upstream_like_joint_context_scale",
        "anchor_derived_context_scale_diagnostic",
    ):
        subset = [row for row in aggregate if row["mode"] == mode]
        if {row["resolution"] for row in subset} != {1024, 4096, 16384}:
            raise AssertionError(f"{mode}: three-seed resolution matrix incomplete")
        if any(row["seed_count"] != 3 for row in subset):
            raise AssertionError(f"{mode}: seed count drifted")
    anchor_rows = [
        row
        for row in aggregate
        if row["mode"] == "anchor_derived_context_scale_diagnostic"
    ]
    if any(
        row["point_global_pct_mean"] >= 20.0
        or row["point_global_pct_std"] >= 1.0
        for row in anchor_rows
    ):
        raise AssertionError("anchor-derived three-seed stability gate failed")
    decision = result["workflow_decision"]
    if (
        not decision["accepted"]
        or "anchor-derived context" not in decision["minimum_error_scheme"]
        or decision["applicability"] != "P1h source-aware support family only"
    ):
        raise AssertionError("workflow decision drifted")
    solver = result["solver_benchmark"]
    if (
        solver["status"] != "passed"
        or solver["evaluation_role"] != "valid_iid"
        or solver["test_hard_accessed"]
        or solver["solver"]["node_count"] != 240825
        or solver["replay"]["temperature_max_abs_error_K"] > 1.0e-8
        or solver["dof_comparability"][
            "accuracy_equivalent_similar_dof_mesh_available"
        ]
    ):
        raise AssertionError("physical-solver benchmark/replay audit failed")
    dof = solver["dof_comparability"]
    candidates = dof["source_resolution_audit_valid_iid_metadata_only"]
    if (
        [row["mesh_intervals_xyz"][0] for row in candidates]
        != [48, 52, 56, 60, 64]
        or dof["smallest_source_resolution_legal_candidate"]["node_count"] != 212097
        or dof["smallest_source_resolution_legal_candidate"][
            "minimum_source_in_plane_intervals"
        ]
        < 7
        or any(row["target_fields_accessed"] for row in candidates)
    ):
        raise AssertionError("similar-DOF solver/source-resolution audit drifted")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != 29:
        raise AssertionError("complete metrics CSV row count drifted")
    if any(row["test_hard_accessed"] == "True" for row in metric_rows):
        raise AssertionError("metrics CSV records forbidden role access")
    _finite_mapping(result)
    return {
        "status": "passed",
        "canonical_model": canonical["canonical_model_configuration"],
        "frozen_checkpoint_prediction_pairs": len(freeze["artifacts"]),
        "maximum_stable_resolution": search["maximum_stable_resolution"],
        "first_failed_resolution": search["first_failed_resolution"],
        "three_seed_rows": len(aggregate),
        "physical_solver_nodes": solver["solver"]["node_count"],
        "test_hard_accessed": False,
        "training_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ladder",
        type=Path,
        default=ROOT
        / "configs/heat3d_v6/v6_source_aware_resolution_ladder.json",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT
        / "configs/heat3d_v6/v6_source_aware_resolution_limit_results.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "configs/heat3d_v6/v6_source_aware_resolution_metrics.csv",
    )
    args = parser.parse_args()
    print(json.dumps(check(args.ladder, args.result, args.metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
