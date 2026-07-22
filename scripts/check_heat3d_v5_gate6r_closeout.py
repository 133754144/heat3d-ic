#!/usr/bin/env python3
"""Validate the Gate 6R result and V5 phase-closeout assessment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/heat3d_v5/gate6r_closeout"
ASSESSMENT = OUT / "v5_final_phase_assessment.json"
METRICS = OUT / "v5_final_checkpoint_metrics.csv"
PAIRED = OUT / "gate6r_paired_samples.csv"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6r_training_registry.csv"
REPORT = ROOT / "docs/v5_final_phase_assessment.md"
EVALUATORS = tuple(sorted((OUT / "evaluator").glob("V*_gate6r_cpu_replay.json")))
EXPECTED_MODELS = {"V38", "V42", "V43", "V44", "V45", "V46"}
EXPECTED_CHECKPOINTS = {"point_global_best", "sample_first_best", "legacy_best", "final"}
EXPECTED_GATE6R = {
    "V4P5_45_gate6r_deepsets_only_e600": "wsl2",
    "V4P5_46_gate6r_objective_deepsets_e600": "devbox",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    payload = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heat3d_v5_final_phase_assessment_v1"
    assert payload["status"] == "ready_to_close_threshold_unmet"
    assert payload["scope"]["roles_accessed"] == ["train", "valid_iid"]
    assert payload["scope"]["evaluation_roles"] == ["valid_iid"]
    assert payload["scope"]["forbidden_roles_accessed"] == []
    for key in ("test_accessed", "hard_accessed", "sealed_iid_accessed", "training_started"):
        assert payload["scope"][key] is False
    assert payload["v38_cross_evaluator_max_abs_summary_diff"] <= 1.0e-12
    ranking = payload["formal_ranking"]
    assert {row["model"] for row in ranking} == EXPECTED_MODELS
    assert ranking[0]["model"] == "V42"
    assert abs(ranking[0]["point_global_relative_rmse_pct"] - 21.936815354126658) < 1.0e-12
    assert all(not row["valid_threshold_pass"] for row in ranking)
    verdict = payload["phase_closeout_assessment"]
    assert verdict["scientific_success"] is False
    assert verdict["can_close_v5"] is True
    assert verdict["closure_class"] == "completed_research_phase_threshold_unmet"
    assert verdict["additional_v5_training_recommended"] is False
    merge = payload["merge_assessment"]
    assert merge["main_is_ancestor"] is True
    assert merge["technical_fast_forward_possible"] is True
    assert merge["direct_fast_forward_recommended"] is False
    assert merge["behind_commits"] == 0
    assert merge["changed_files"] >= 375

    for path in EVALUATORS:
        evaluator = json.loads(path.read_text(encoding="utf-8"))
        assert evaluator["status"] == "completed_valid_iid_only"
        assert evaluator["evaluator_commit"] == payload["gate6r_evaluator_commit"]
        assert evaluator["metric_source"]["sha256"] == payload["metric_source_sha256"]
        assert evaluator["split"]["valid_iid_ids_sha256"] == payload["valid_iid_ids_sha256"]
        assert evaluator["scope"]["evaluation_roles"] == ["valid_iid"]
        assert evaluator["scope"]["forbidden_roles_accessed"] == []
        assert set(evaluator["metrics"]) == EXPECTED_CHECKPOINTS
        assert evaluator["training_completion"]["epoch_history_count"] == 600
        assert evaluator["training_completion"]["grad_finite"] is True
        for checkpoint in EXPECTED_CHECKPOINTS:
            assert evaluator["reload_audit"][checkpoint]["passed"] is True
            assert evaluator["reload_audit"][checkpoint]["metric_prediction_source"] == "checkpoint_bound_saved_npz"
        rel = str(path.relative_to(ROOT))
        assert payload["input_artifacts"][rel]["sha256"] == _sha256(path)

    with METRICS.open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 24
    assert {row["model"] for row in metrics} == EXPECTED_MODELS
    for model in EXPECTED_MODELS:
        assert {row["checkpoint"] for row in metrics if row["model"] == model} == EXPECTED_CHECKPOINTS
    for row in metrics:
        for field in (
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
        ):
            assert math.isfinite(float(row[field]))

    with PAIRED.open(newline="", encoding="utf-8") as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 3 * 128
    assert {row["comparison"] for row in paired} == {
        "V45_minus_V38",
        "V46_minus_V38",
        "V46_minus_V45",
    }
    for comparison in {row["comparison"] for row in paired}:
        rows = [row for row in paired if row["comparison"] == comparison]
        assert len({row["sample_id"] for row in rows}) == 128

    csv.field_size_limit(sys.maxsize)
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        registry = {row["config_id"]: row for row in csv.DictReader(stream)}
    for config_id, host in EXPECTED_GATE6R.items():
        row = registry[config_id]
        assert row["plan_status"] == "completed"
        assert row["execution_status"] == "completed_e600"
        assert row["evaluation_status"] == "completed_valid_iid_four_checkpoint"
        assert row["training_started"] == "true"
        assert row["result_v5_status"] == "completed_valid_only"
        assert row["result_v5_source"] == host
        assert row["result_v5_required_metrics_complete"] == "true"
        assert row["result_v5_threshold_pass"] == "valid_only_fail"
        assert row["result_v5_primary_test_point_global_relative_rmse_pct"] == ""
        assert row["result_v5_primary_test_sample_first_cv_relative_rmse_pct"] == ""
        assert row["result_v5_primary_test_raw_cv_weighted_rmse_K"] == ""

    report = REPORT.read_text(encoding="utf-8")
    for phrase in (
        "Scientific success: **no**",
        "Phase closure: **yes",
        "do not fast-forward the full research branch",
        "No merge, tag, test/hard/sealed evaluation, or training",
    ):
        assert phrase in report
    print("Gate 6R / V5 final phase closeout checker: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
