#!/usr/bin/env python3
"""Validate the immutable V6 final-performance test-opening contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/heat3d_v6/v6_final_performance_preregistration.json"
)


def main() -> int:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_test_open"
    assert payload["model"] == {
        "config_id": "V6_03_V5best_P1h",
        "seed": 0,
        "checkpoint_epoch": 111,
        "checkpoint_selection": "valid_point_global_true_rms",
        "checkpoint_sha256": (
            "3ad58c2b34a46481acb74722c80bdcadb"
            "f55a0d613bc25c4fe2d7646b91aa1f2"
        ),
    }
    assert payload["pre_registered_resolutions"] == {
        "default": 4096,
        "full_field": 8192,
        "high_resolution": 16384,
        "experimental_excluded_from_primary_test_table": 32768,
    }
    policy = payload["test_policy"]
    assert policy["test_iid_status"] == "sealed_pending_one_time_evaluation"
    assert policy["test_must_not_be_used_for_selection_or_tuning"] is True
    assert policy["hard_status"] == "sealed"
    assert policy["hard_access_allowed"] is False
    assert payload["training_allowed"] is False
    assert payload["checkpoint_modification_allowed"] is False
    forbidden_outputs = list(
        (ROOT / "configs/heat3d_v6").glob("v6_final_performance_test_*.json")
    )
    assert not forbidden_outputs, forbidden_outputs
    print(
        json.dumps(
            {
                "status": "passed",
                "test_iid": "sealed_pending_committed_preregistration",
                "hard": "sealed",
                "training_allowed": False,
                "checkpoint_modification_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
