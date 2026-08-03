#!/usr/bin/env python3
"""Deterministic checks for the V6 inference qualification closeout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, trail: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {trail}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{trail}[{index}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1i", type=Path, required=True)
    parser.add_argument("--randomblock", type=Path, required=True)
    args = parser.parse_args()
    raw = {name: load(path) for name, path in (("p1i", args.p1i), ("randomblock", args.randomblock))}
    for family, payload in raw.items():
        assert payload["status"] == "passed" and payload["family"] == family
        assert payload["sample_count"] == 32 and len(set(payload["sample_ids"])) == 32
        assert payload["contract"] == {
            "independent_process_per_route_state": True,
            "cold_fresh_process_per_sample": True,
            "continuous_wall_clock_not_stage_sum": True,
            "minimum_measurements": 20,
            "batch_size": 1,
            "fixed_threads": 1,
            "production_excludes_oracle": True,
        }
        assert payload["accessed_roles"] == ["valid_iid", "train_frozen_normalization_metadata"]
        assert not payload["test_accessed"] and not payload["sealed_accessed"]
        assert not payload["training_executed"] and not payload["checkpoint_modified"]
        assert set(payload["routes"]) == {"model_support", "production_reconstruction", "fvm"}
        target_sha = payload["fixed_edge_jit_targets"]["sha256"]
        for route_payload in payload["routes"].values():
            assert route_payload["cold"]["fresh_process_count"] == 32
            assert route_payload["cold"]["external_process_wall_seconds"]["count"] == 32
            for state in ("jit_cached_new_case", "fully_cached_repeat"):
                item = route_payload[state]
                assert item["sample_count"] == 32
                assert item["stage_timing"]["continuous_wall_seconds"]["count"] == 32
        for route in ("model_support", "production_reconstruction"):
            for state in ("jit_cached_new_case", "fully_cached_repeat"):
                item = payload["routes"][route][state]
                assert item["fixed_edge_jit_targets_sha256"] == target_sha
                assert item["stage_timing"]["jit_or_forward_seconds"]["median"] < 0.1
            assert payload["routes"][route]["fully_cached_repeat"]["stage_timing"]["graph_seconds"]["median"] < 0.01
        finite(payload)
    assert raw["p1i"]["checkpoint"]["sha256"] == "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e"
    assert raw["randomblock"]["checkpoint"]["sha256"] == "3ad58c2b34a46481acb74722c80bdcadbf55a0d613bc25c4fe2d7646b91aa1f2"
    p1i_env = raw["p1i"]["routes"]["model_support"]["fully_cached_repeat"]["environment"]
    rb_env = raw["randomblock"]["routes"]["model_support"]["fully_cached_repeat"]["environment"]
    for key in ("host", "platform", "python", "jax", "numpy", "device", "cpu_count", "batch_size", "OMP_NUM_THREADS"):
        assert p1i_env[key] == rb_env[key], f"same-host environment drift: {key}"
    # Random-block selection must cover 16 support groups twice.
    rb_groups = [sample_id.rsplit("_v", 1)[0] for sample_id in raw["randomblock"]["sample_ids"]]
    assert len(set(rb_groups)) == 16 and all(rb_groups.count(group) == 2 for group in set(rb_groups))

    closeout = load(ROOT / "configs/heat3d_v6_p1i/v6_inference_qualification_closeout.json")
    assert closeout["status"] == "qualified_valid_only"
    assert closeout["accepted_model_contract"]["primary"] == "point_global_true_rms_relative_rmse"
    assert closeout["accepted_model_contract"]["secondary"] == "sample_first_cv_relative_rmse"
    assert closeout["accepted_model_contract"]["deployment"] == "1024_source_aware_support_plus_layer_aware_reconstruction"
    assert closeout["scope"]["test_accessed"] is False and closeout["scope"]["sealed_accessed"] is False
    assert len(closeout["timing"]) == 18 and len(closeout["accuracy"]) == 10
    assert set(closeout["sample_scope"]) == {"p1i", "randomblock"}
    assert closeout["sample_scope"]["p1i"]["sample_count"] == 32
    assert closeout["sample_scope"]["randomblock"]["sample_count"] == 32
    assert closeout["sample_scope"]["p1i"]["unique_support_hashes"] == 32
    assert closeout["sample_scope"]["randomblock"]["unique_support_hashes"] == 16
    assert closeout["jit_shape_contract"]["p1i"]["equivalence"]["status"] == "passed"
    assert closeout["jit_shape_contract"]["randomblock"]["contract"]["mode"] == "raw_shape_family_v1"
    assert closeout["jit_shape_contract"]["randomblock"]["raw_graph_unmodified"] is True
    assert closeout["jit_shape_contract"]["randomblock"]["rejected_fixed_padding_attempt"]["status"] == "failed"
    assert len(closeout["protocol_deviations"]) == 3
    assert all("passed" not in row["status"] for row in closeout["protocol_deviations"])
    assert closeout["qualification_decision"] == {
        "p1i_native_deployment": "qualified",
        "randomblock_structured_support_ood_compatibility": "failed",
        "randomblock_is_production_claim": False,
    }
    assert closeout["environment"]["payload"]["host"] == p1i_env["host"] == rb_env["host"]
    assert closeout["environment"]["payload"]["model_device_kind"] == raw["p1i"]["routes"]["model_support"]["fully_cached_repeat"]["device_memory"]["device_kind"]
    assert closeout["environment"]["payload"]["model_device_kind"] == raw["randomblock"]["routes"]["model_support"]["fully_cached_repeat"]["device_memory"]["device_kind"]
    assert closeout["environment"]["payload"]["test_accessed"] is False
    assert all(not row["directly_comparable_to_qualification"] for row in closeout["historical_layer_audit"])
    finite(closeout)
    for name, minimum in (("v6_inference_qualification_timing.csv", 18), ("v6_inference_qualification_accuracy.csv", 10), ("v6_inference_historical_layer_audit.csv", 3)):
        with (ROOT / "configs/heat3d_v6_p1i" / name).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == minimum
    report = (ROOT / "docs/v6_inference_qualification_closeout.md").read_text(encoding="utf-8")
    old_report = (ROOT / "docs/v6_p1i_three_seed_inference_closeout.md").read_text(encoding="utf-8")
    for phrase in ("cached steady-state speedup", "structured-support OOD compatibility diagnostic"):
        assert phrase in report and phrase in old_report
    for path in (ROOT / "docs/v6_inference_qualification_accuracy_latency.svg", ROOT / "docs/v6_inference_qualification_scale_time.svg"):
        assert path.read_text(encoding="utf-8").startswith("<svg")
    benchmark_source = (ROOT / "scripts/benchmark_heat3d_v6_inference_qualification.py").read_text(encoding="utf-8")
    assert "Accuracy and oracle work are deliberately after the timed production span." in benchmark_source
    assert "graph_cache[sample_id]" in benchmark_source
    print(json.dumps({"status": "passed", "families": 2, "samples_per_family": 32, "timing_rows": 18}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
