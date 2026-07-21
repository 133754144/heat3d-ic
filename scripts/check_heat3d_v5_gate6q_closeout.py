#!/usr/bin/env python3
"""Validate the frozen Gate 6Q valid-only closeout artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


MODELS = ("V38", "V42", "V43", "V44")
CHECKPOINTS = ("point_global_best", "sample_first_best", "legacy_best", "final")
METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "legacy_normalized_valid_base_mse",
    "shape_cv_rmse",
    "scale_log_rmse",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
)
EXPECTED_COMPARISONS = {
    "V42_minus_V38": ("V38", "V42"),
    "V43_minus_V38": ("V38", "V43"),
    "V44_minus_V43": ("V43", "V44"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--attribution-csv", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    args = _args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == "heat3d_v5_gate6q_final_closeout_v1", "schema mismatch")
    _require(payload.get("status") == "completed_valid_iid_only", "status mismatch")
    _require(payload.get("training_started_by_closeout") is False, "closeout started training")
    _require(payload.get("roles_accessed") == ["train", "valid_iid"], "role scope mismatch")
    _require(payload.get("forbidden_roles_accessed") == [], "forbidden role accessed")
    _require(payload.get("checkpoint_selection_modified") is False, "checkpoint selection changed")
    _require(payload.get("model_parameters_modified") is False, "model parameters changed")
    models = payload.get("models") or {}
    _require(set(models) == set(MODELS), "four model payloads missing")

    evaluator = {value.get("evaluator_commit") for value in models.values()}
    evaluator_hash = {value.get("evaluator_source_sha256") for value in models.values()}
    metric_source = {
        (value.get("metric_source") or {}).get("sha256") for value in models.values()
    }
    _require(len(evaluator) == len(evaluator_hash) == len(metric_source) == 1, "evaluator provenance differs")
    for model, value in models.items():
        scope = value.get("scope") or {}
        _require(scope.get("backend") == "cpu", f"{model}: backend is not CPU")
        _require(scope.get("roles_accessed") == ["train", "valid_iid"], f"{model}: role scope mismatch")
        _require(scope.get("forbidden_roles_accessed") == [], f"{model}: forbidden role accessed")
        _require(scope.get("training_started") is False, f"{model}: evaluator started training")
        _require(set(value.get("metrics") or {}) == set(CHECKPOINTS), f"{model}: checkpoints missing")
        for checkpoint in CHECKPOINTS:
            metadata = value["checkpoint_metadata"][checkpoint]
            _require(len(str(metadata.get("sha256") or "")) == 64, f"{model}/{checkpoint}: checkpoint SHA missing")
            _require(metadata.get("parameter_reload_max_abs_error") == 0.0, f"{model}/{checkpoint}: parameter reload mismatch")
            _require(float(metadata.get("training_reload_max_abs_error_K", 1.0)) <= 0.02, f"{model}/{checkpoint}: training reload failed")
            replay = value["reload_audit"][checkpoint]
            _require(replay.get("passed") is True, f"{model}/{checkpoint}: evaluator replay incomplete")
            _require(replay.get("metric_prediction_source") == "checkpoint_bound_saved_npz", f"{model}/{checkpoint}: metric source mismatch")
            suite = value["metrics"][checkpoint]
            _require(len(suite.get("per_sample") or ()) == 128, f"{model}/{checkpoint}: sample count mismatch")
            for metric in METRICS:
                _require(_finite(suite["summary"].get(metric)), f"{model}/{checkpoint}: {metric} missing")
            for row in suite["per_sample"]:
                _require(row.get("split") == "valid_iid", f"{model}/{checkpoint}: non-valid sample")
                _require(float(row.get("decomposition_closure_abs_K2", 1.0)) < 1.0e-8, f"{model}/{checkpoint}: decomposition closure failed")

    ranking = payload.get("formal_ranking") or []
    _require([row["model"] for row in ranking] == ["V42", "V38", "V44", "V43"], "formal ranking mismatch")
    _require(all(not row["valid_point_global_lt_20pct"] for row in ranking), "unexpected threshold pass")
    comparisons = payload.get("comparisons") or {}
    _require(set(comparisons) == set(EXPECTED_COMPARISONS), "comparison set mismatch")
    for name, (baseline, candidate) in EXPECTED_COMPARISONS.items():
        row = comparisons[name]
        _require(row["baseline"] == baseline and row["candidate"] == candidate, f"{name}: direction mismatch")
        direct = sum(
            float(item["delta_point_error_squared_sum"])
            for item in _read_csv(args.paired_csv)
            if item["comparison"] == name
        )
        _require(abs(direct - float(row["point_sse_delta_K2"])) < 1.0e-8, f"{name}: paired SSE mismatch")

    metric_rows = _read_csv(args.metrics_csv)
    paired_rows = _read_csv(args.paired_csv)
    attribution_rows = _read_csv(args.attribution_csv)
    _require(len(metric_rows) == 16, "metrics CSV row count mismatch")
    _require(len(paired_rows) == 3 * 128, "paired CSV row count mismatch")
    _require(len(attribution_rows) == 3 * 5, "attribution CSV row count mismatch")

    registry = {row["config_id"]: row for row in _read_csv(args.registry)}
    for model in ("V42", "V43", "V44"):
        config_id = models[model]["config_id"]
        row = registry.get(config_id) or {}
        _require(row.get("execution_status") == "completed_e600", f"{model}: registry execution status")
        _require(row.get("evaluation_status") == "completed_valid_iid_only", f"{model}: registry evaluation status")
        _require(row.get("training_started") == "true", f"{model}: registry training state")
        _require(row.get("result_v5_status") == "completed_valid_only", f"{model}: registry result state")
        _require(row.get("result_v5_required_metrics_complete") == "true", f"{model}: registry metrics incomplete")
        expected_epoch = str(models[model]["checkpoint_metadata"]["point_global_best"]["epoch"])
        _require(row.get("result_v5_primary_epoch") == expected_epoch, f"{model}: registry epoch mismatch")
        _require(row.get("result_v5_threshold_pass") == "valid_only_fail", f"{model}: threshold status mismatch")

    print("Gate 6Q closeout checker: PASS")
    print("roles=train,valid_iid; test/hard/sealed=not accessed; training=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
