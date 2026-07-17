#!/usr/bin/env python3
"""Validate the frozen Gate 6H engineering closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.heat3d_v5_result_contract import (  # noqa: E402
    V5_FROZEN_METRICS,
    V5_REPORT_ROLES,
)


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6h_v13_scale_ablation_registry.csv"
PREFLIGHT = ROOT / "configs/heat3d_v5/gate6h/preflight_audit.json"
CONTRACT = ROOT / "configs/heat3d_v5/gate6h/v13_scratch_scale_ablation_contract.json"
VALID_ONLY = ROOT / "configs/heat3d_v5/gate6h/valid_only_true_rms_evaluation.json"
MANIFEST = ROOT / "configs/heat3d_v5/gate6h/v31_evaluation_manifest.json"
V31_ADAPTER = ROOT / "configs/heat3d_v5/gate6h/v31_gate5_compat_adapter.txt"
V31_COMPAT_CONTRACT = ROOT / "configs/heat3d_v5/gate6h/v31_gate5_compat_contract.json"
EVALUATOR = ROOT / "scripts/evaluate_heat3d_v5_gate5_checkpoints.py"

V28 = "V4P5_28_gate6h_v13_stopgrad_scratch_e600"
V29 = "V4P5_29_gate6h_v13_scale_attention_scratch_e600"
V30 = "V4P5_30_gate6h_v13_deep_scale_head_scratch_e600"
V31 = "V4P5_31_gate6h_v29_validation_b32_retry_e600"
BASE_EVALUATOR_COMMIT = "639872abcb0f7afd3b6c2d319a7d395bde75c9a4"
EXECUTION_EVALUATOR_COMMIT = "bcd5c581cb4a20e6257179f7a5c6f7528fe05a54"
TRAINING_COMMIT = "a61bb00"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    args = _args()
    rows = {
        row["config_id"]: row
        for row in csv.DictReader(REGISTRY.open(encoding="utf-8", newline=""))
    }
    assert tuple(rows) == (V28, V29, V30, V31)
    v31 = rows[V31]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    valid_only = json.loads(VALID_ONLY.read_text(encoding="utf-8"))
    metrics_text = v31["result_v5_metrics_json"]
    metrics = json.loads(metrics_text)

    # Lifecycle and immutable run artifacts.
    assert v31["plan_status"] == "frozen_closed"
    assert v31["execution_status"] == "completed_e600"
    assert v31["evaluation_status"] == "completed_full_v5_metrics"
    assert v31["threshold_status"] == "failed_clean_valid_test_lt20"
    assert v31["training_host"] == manifest["host"] == "wsl2"
    assert v31["training_commit"] == manifest["run"]["training_commit"] == TRAINING_COMMIT
    assert int(v31["best_epoch"]) == manifest["run"]["best_epoch"] == 330
    assert int(v31["final_epoch"]) == manifest["run"]["final_epoch"] == 600
    assert v31["long_training_started"] == "true"
    artifact_columns = {
        "best_checkpoint_sha256": "params_best.pkl",
        "final_checkpoint_sha256": "params_final.pkl",
        "loss_summary_sha256": "loss_summary.json",
        "run_config_sha256": "run_config.json",
        "best_predictions_sha256": "best_predictions.npz",
        "final_predictions_sha256": "predictions.npz",
    }
    for column, name in artifact_columns.items():
        assert v31[column] == manifest["artifacts"][name]

    # Complete frozen V5 metric payload for V31.
    assert v31["result_v5_status"] == "completed"
    assert v31["result_v5_required_metrics_complete"] == "true"
    assert v31["result_v5_missing_metrics"] == ""
    assert metrics["config_id"] == V31
    assert metrics["training_git_commit"] == TRAINING_COMMIT
    assert metrics["evaluator_git_commit"] == EXECUTION_EVALUATOR_COMMIT
    assert metrics["metric_schema_version"] == manifest["evaluation"]["metric_schema"]
    for checkpoint in ("primary_relative", "legacy_metric"):
        reports = metrics["reports"][checkpoint]
        assert set(reports) == set(V5_REPORT_ROLES)
        for role in V5_REPORT_ROLES:
            row = reports[role]
            for metric in V5_FROZEN_METRICS:
                assert _finite(row.get(metric)), (checkpoint, role, metric)
    assert "sealed_iid" not in metrics["reports"]
    assert all(
        "sealed_iid" not in reports
        for reports in metrics["reports"].values()
        if isinstance(reports, dict)
    )

    # Adapter, contract and evaluator provenance binding.
    evaluation = manifest["evaluation"]
    binding = metrics["compatibility_binding"]
    assert evaluation["base_frozen_evaluator_commit"] == BASE_EVALUATOR_COMMIT
    assert binding["base_frozen_evaluator_commit"] == BASE_EVALUATOR_COMMIT
    assert evaluation["execution_evaluator_commit"] == EXECUTION_EVALUATOR_COMMIT
    assert binding["execution_evaluator_commit"] == EXECUTION_EVALUATOR_COMMIT
    assert v31["evaluator_commit"] == EXECUTION_EVALUATOR_COMMIT
    assert evaluation["evaluator_source_sha256"] == binding["evaluator_source_sha256"]
    assert evaluation["evaluator_source_sha256"] == _sha256(EVALUATOR)
    assert evaluation["adapter_sha256"] == binding["adapter_sha256"]
    assert evaluation["adapter_sha256"] == _sha256(V31_ADAPTER)
    assert evaluation["contract_remote_sha256"] == metrics["contract_sha256"]
    assert evaluation["contract_repo_sha256"] == _sha256(V31_COMPAT_CONTRACT)
    assert hashlib.sha256(metrics_text.encode("utf-8")).hexdigest() == evaluation[
        "registry_embedded_metrics_sha256"
    ]
    assert len(metrics_text.encode("utf-8")) == evaluation["registry_embedded_metrics_bytes"]
    assert all(
        binding[key] is True
        for key in (
            "formulas_unchanged",
            "normalization_context_unchanged",
            "split_handling_unchanged",
        )
    )
    assert v31["authoritative_evaluation_json"] == str(MANIFEST.relative_to(ROOT))

    # Role isolation: train fit, valid selection, report-only test/hard, no sealed.
    role_access = manifest["role_access"]
    assert role_access["fit_roles"] == ["train"]
    assert role_access["selection_roles"] == ["valid_iid"]
    assert tuple(role_access["report_only_roles_accessed"]) == V5_REPORT_ROLES[1:]
    assert role_access["sealed_iid_accessed"] is False
    assert v31["test_role_status"] == "observed_report_only"
    assert v31["hard_role_status"] == "observed_report_only"

    # Preserve the historical preflight snapshot while closing current lifecycle.
    assert preflight["status"] == "passed"
    assert preflight["lifecycle_statuses"][V31] == "not_started"
    assert preflight["scientific_diffs"][V31] == {
        "model.pooled_latent_stop_gradient": True,
        "model.scale_attention_mode": "physics_gate",
        "run.prediction_batch_size": 32,
        "run.validation_batch_size": 32,
    }
    assert contract["status"] == "frozen_closed"
    assert contract["current_lifecycle"][V31] == "completed_e600"

    # V13 remains the best valid-only point-global model; no Gate 6H candidate advances.
    v13_point = float(
        contract["historical_audit"]["models"]["V13"]["point_global_relative_rmse_pct"]
    )
    candidate_points = {
        V28: float(valid_only["models"][V28]["metrics"]["best"]["point_global_relative_rmse_pct"]),
        V30: float(valid_only["models"][V30]["metrics"]["best"]["point_global_relative_rmse_pct"]),
        V31: float(metrics["reports"]["primary_relative"]["valid_iid"]["point_global_relative_rmse_pct"]),
    }
    assert all(v13_point < value for value in candidate_points.values())
    assert all(value >= 20.0 for value in candidate_points.values())
    assert float(
        metrics["reports"]["primary_relative"]["test_iid"]["point_global_relative_rmse_pct"]
    ) >= 20.0
    assert manifest["threshold"]["passed"] is False

    payload = {
        "status": "passed",
        "lifecycle": {config_id: row["execution_status"] for config_id, row in rows.items()},
        "v31_metrics_complete": True,
        "v31_adapter_evaluator_binding": True,
        "roles_accessed": ["train", *V5_REPORT_ROLES],
        "sealed_iid_accessed": False,
        "v13_valid_point_global_relative_rmse_pct": v13_point,
        "gate6h_candidate_valid_point_global_relative_rmse_pct": candidate_points,
        "promoted_candidates": [],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
