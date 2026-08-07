#!/usr/bin/env python3
"""Deterministic checker for the P1i R0 gate and frozen high-N binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import V6DualRobinExample  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    P1iSampleVaryingAnchorQueryAdapter,
    conservative_selected_control_volume,
    deterministic_nested_query_order,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_smoke() -> None:
    coords = np.zeros((1024, 3), dtype=np.float64)
    features = np.zeros((1024, 11), dtype=np.float64)
    example = V6DualRobinExample(
        sample_id="synthetic",
        condition=V1SteadyConditionInput(coords, features, tuple(map(str, range(11))), "diag3"),
        target=V1SteadyTarget(np.zeros((1024, 1))),
        meta={},
        operator_point_weights=np.ones(1024),
    )
    adapter = P1iSampleVaryingAnchorQueryAdapter(example)
    assert adapter.r0_input_equivalence(adapter.r0_example())["passed"]

    node_count = 2048
    index = np.arange(node_count)
    full_coords = np.column_stack((((index // 4) % 32) / 31, (index // 128) / 15, (index % 4) / 3))
    layer = (index % 4).astype(np.int32)
    cv = np.linspace(1.0, 2.0, node_count)
    q = np.where((index % 11) == 0, 2.0, 0.0)
    boundaries = np.linspace(0.0, 1.0, 5)
    anchors = np.arange(1024, dtype=np.int64)
    first, audit1 = deterministic_nested_query_order(
        sample_id="synthetic", anchor_indices=anchors, full_coords=full_coords,
        full_control_volume=cv, full_layer_id=layer, full_q=q,
        layer_boundaries_m=boundaries,
    )
    second, audit2 = deterministic_nested_query_order(
        sample_id="synthetic", anchor_indices=anchors, full_coords=full_coords,
        full_control_volume=cv, full_layer_id=layer, full_q=q,
        layer_boundaries_m=boundaries,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first[:1024], anchors)
    assert len(np.unique(first)) == node_count
    assert audit1["order_sha256"] == audit2["order_sha256"]
    weights, cv_audit = conservative_selected_control_volume(
        full_coords=full_coords, full_control_volume=cv, full_layer_id=layer,
        selected_indices=first[:1536],
    )
    assert len(weights) == 1536 and cv_audit["relative_volume_error"] <= 1e-12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-result", action="append", type=Path, default=[])
    parser.add_argument("--r0-json", type=Path)
    parser.add_argument("--binding-json", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    assert contract["status"] == "frozen_before_dual_backend_three_seed_execution"
    assert contract["execution_binding"]["hard_adapter_comparison"].startswith("deterministic CPU")
    assert contract["hard_gate"]["adapter_vs_reference_prediction"]["max_abs_error_K"] == 0.0
    assert contract["role_contract"]["test_accessed"] is False
    assert contract["role_contract"]["sealed_accessed"] is False
    implementation_smoke()

    raw_results = []
    for path in args.seed_result:
        row = json.loads(path.read_text())
        assert row["status"] == "passed" and all(row["checks"].values())
        assert row["input_equivalence"]["sample_count"] == 128
        assert row["graph_equivalence"]["sample_count"] == 128
        assert row["full_field_reconstruction_equivalence"]["mapping_sample_count"] == 128
        if row["backend_role"] == "deterministic_cpu_equivalence":
            assert row["jax_backend"] == "cpu"
            assert row["prediction_equivalence"]["adapter_vs_reference"]["max_abs_error_K"] == 0.0
            assert row["feature_and_scale_equivalence"]["predicted_scale"]["max_abs_error"] == 0.0
            assert row["full_field_reconstruction_equivalence"]["adapter_vs_reference"]["max_abs_error_K"] == 0.0
        else:
            assert row["backend_role"] == "historical_gpu_replay" and row["jax_backend"] == "gpu"
            assert row["checks"]["archived_prediction_replay"]
            assert row["checks"]["archived_full_field_replay"]
        roles = row["role_contract"]
        assert not any(roles[key] for key in (
            "test_accessed", "sealed_accessed", "training_executed",
            "checkpoint_modified", "high_n_inference_executed",
        ))
        raw_results.append(row)
    if raw_results:
        assert len(raw_results) == 6
        assert {(int(row["seed"]), row["backend_role"]) for row in raw_results} == {
            (seed, role) for seed in (0, 1, 2)
            for role in ("deterministic_cpu_equivalence", "historical_gpu_replay")
        }

    if args.r0_json:
        r0 = json.loads(args.r0_json.read_text())
        assert r0["status"] == "passed_three_seed_dual_backend_prediction_level_equivalence"
        assert r0["stage_b_released"] is True and r0["seed_count"] == 3
        assert {int(row["seed"]) for row in r0["seeds"]} == {0, 1, 2}
        assert all(row["adapter_reference_max_abs_K"] == 0.0 for row in r0["seeds"])
        assert r0["role_contract"]["high_n_inference_executed"] is False
        for seed in r0["seeds"]:
            for evidence in seed["raw_results"].values():
                path = ROOT / evidence["path"]
                assert path.is_file() and sha256(path) == evidence["sha256"]

    if args.binding_json:
        binding = json.loads(args.binding_json.read_text())
        assert binding["status"] == "frozen_after_three_seed_r0_pass"
        manifest_path = ROOT / binding["dataset"]["manifest_path"]
        assert manifest_path.is_file() and sha256(manifest_path) == binding["dataset"]["manifest_sha256"]
        assert binding["resolutions"]["mandatory"] == [1024, 4096, 8192, 16384]
        assert binding["resolutions"]["optional_valid_only"] == 32768
        assert binding["resolutions"]["optional_enters_mandatory_ranking"] is False
        assert binding["nested_support"]["selection_seed"] == 20260808
        assert binding["development_subset"]["count"] == 32
        assert len(set(binding["development_subset"]["sample_ids"])) == 32
        assert binding["development_subset"]["model_error_or_temperature_used"] is False
        assert binding["timing_contract"]["R0_may_be_reported_as_production_timing"] is False
        assert binding["execution_contract"] == {
            "high_n_inference_executed_this_closeout": False,
            "training_executed": False,
            "test_accessed": False,
            "sealed_accessed": False,
        }
        for evidence in binding["code_fingerprints"].values():
            path = ROOT / evidence["path"]
            assert path.is_file() and sha256(path) == evidence["sha256"]

    print("P1i anchor/query R0 and high-N binding checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
