#!/usr/bin/env python3
"""Fixture checks for the V5 result collector's frozen-metric gate."""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heat3d_v5_result_contract import (  # noqa: E402
    V5_FROZEN_METRICS,
    V5_REGISTRY_RESULT_FIELDS,
    V5_REPORT_ROLES,
)
from summarize_heat3d_v5_run_result import _result_fields, _update_csv  # noqa: E402


def _reports() -> dict:
    return {
        checkpoint: {
            role: {
                metric: 1.0
                for metric in V5_FROZEN_METRICS
            }
            for role in V5_REPORT_ROLES
        }
        for checkpoint in ("primary_relative", "legacy_metric")
    }


def main() -> int:
    row = {
        "config_id": "fixture",
        "output_dir": "output/fixture",
        "log_path": "output/fixture.log",
    }
    payload = {
        "loss_summary": {"status_ok": True, "grad_finite": True, "best_epoch": 1},
        "run_config": {"final_probe_eval_after_training": False},
        "metrics": {"reports": _reports()},
        "source": "fixture",
    }
    complete = _result_fields(row, Path("/tmp/fixture"), payload, "fixture")
    assert complete["result_v5_status"] == "completed"
    assert complete["result_v5_required_metrics_complete"] == "true"
    assert complete["result_v5_missing_metrics"] == ""
    assert complete["result_v5_threshold_pass"] == "pass"

    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "registry.csv"
        fieldnames = ["config_id", *V5_REGISTRY_RESULT_FIELDS]
        stale = {field: "" for field in fieldnames}
        stale.update({
            "config_id": "fixture",
            "result_v5_required_metrics_complete": "false",
            "result_v5_missing_metrics": "stale.path",
            "result_v5_notes": "stale note",
        })
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(stale)
        _update_csv(csv_path, "fixture", complete)
        updated = next(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        assert updated["result_v5_required_metrics_complete"] == "true"
        assert updated["result_v5_missing_metrics"] == ""
        assert updated["result_v5_notes"] == ""

    incomplete_payload = copy.deepcopy(payload)
    del incomplete_payload["metrics"]["reports"]["primary_relative"]["test_iid"][V5_FROZEN_METRICS[0]]
    incomplete = _result_fields(row, Path("/tmp/fixture"), incomplete_payload, "fixture")
    assert incomplete["result_v5_status"] == "completed_with_missing_v5_metrics"
    assert incomplete["result_v5_required_metrics_complete"] == "false"
    assert "primary_relative.test_iid." in incomplete["result_v5_missing_metrics"]

    valid_only_payload = {
        "schema_version": "heat3d_v5_v32_valid_only_closeout_v1",
        "training_commit": "fixture-commit",
        "checkpoint_metadata": {
            "point_global_best": {"epoch": 10},
            "sample_first_best": {"epoch": 9},
            "legacy_best": {"epoch": 8},
            "final": {"epoch": 600},
        },
        "metrics": {
            checkpoint: {
                "summary": {metric: 1.0 for metric in V5_FROZEN_METRICS}
            }
            for checkpoint in (
                "point_global_best",
                "sample_first_best",
                "legacy_best",
                "final",
            )
        },
    }
    valid_only = _result_fields(
        row, Path("/tmp/fixture"), valid_only_payload, "fixture"
    )
    assert valid_only["result_v5_status"] == "completed_valid_only"
    assert valid_only["result_v5_required_metrics_complete"] == "true"
    assert valid_only["result_v5_primary_checkpoint"] == "point_global_best"
    assert valid_only["result_v5_primary_epoch"] == "10"
    assert valid_only["result_v5_legacy_checkpoint"] == "legacy_best"
    assert valid_only["result_v5_legacy_epoch"] == "8"
    assert valid_only["result_v5_threshold_pass"] == "valid_only_pass"
    assert (
        valid_only["result_v5_final_probe_status"]
        == "not_applicable_valid_only"
    )
    assert (
        valid_only["result_v5_post_training_diagnostics_status"]
        == "not_applicable_valid_only"
    )
    embedded = json.loads(valid_only["result_v5_metrics_json"])
    assert set(embedded["metrics"]) == {
        "point_global_best",
        "sample_first_best",
        "legacy_best",
        "final",
    }
    gate6q_payload = copy.deepcopy(valid_only_payload)
    gate6q_payload["schema_version"] = "heat3d_v5_gate6q_cpu_replay_v1"
    gate6q = _result_fields(row, Path("/tmp/fixture"), gate6q_payload, "fixture")
    assert gate6q["result_v5_status"] == "completed_valid_only"
    assert gate6q["result_v5_required_metrics_complete"] == "true"
    print("V5 result collector fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
