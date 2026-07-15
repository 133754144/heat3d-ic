#!/usr/bin/env python3
"""Reproducibility and no-training checks for the Gate 6D preflight audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GATE6D = ROOT / "configs/heat3d_v5/gate6d"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6c_scratch_loss_registry.csv"
EVALUATOR_COMMIT = "639872abcb0f7afd3b6c2d319a7d395bde75c9a4"
EVALUATOR_SHA256 = "aed63bbfa0e23aa69b944960f222feac05dc3682783ab601a3e90ae54581911d"
ROLES = {"valid_iid", "test_iid", "hard_train_holdout", "hard_challenge_valid", "hard_challenge_test"}
CHECKPOINTS = {"best", "final"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _existing_ids() -> set[str]:
    split = _read(ROOT / "configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json")
    return set(split["sample_splits"])


def main() -> int:
    contract = _read(ROOT / "configs/heat3d_v5/v5_gate6d_frozen_evaluator_contract.json")
    assert contract["evaluator_engine_commit"] == EVALUATOR_COMMIT
    assert contract["evaluator_engine_sha256"] == EVALUATOR_SHA256
    evaluations = {}
    for number in (11, 12):
        path = GATE6D / f"V4P5_{number}_frozen_gate5_evaluation.json"
        payload = _read(path)
        _finite(payload)
        assert payload["evaluator_git_commit"] == EVALUATOR_COMMIT
        assert CHECKPOINTS.issubset(payload["reports"])
        assert all(set(payload["reports"][checkpoint]) == ROLES for checkpoint in CHECKPOINTS)
        assert payload["data"]["nodes_per_sample"] == 1024
        audit = payload["validation_audit"]
        for key in (
            "config_id_bound", "run_directory_bound", "checkpoint_kind_epoch_and_run_bound",
            "split_hashes_match_contract", "normalization_recomputed_from_train_only",
            "global_context_recomputed_from_train_only",
        ):
            assert audit[key] is True
        evaluations[payload["config_id"]] = payload
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    for row in rows:
        assert row["plan_status"] == "frozen"
        assert row["execution_status"] == "completed_e600"
        assert row["evaluation_status"] == "completed_evaluated"
        assert row["threshold_status"] == "failed"
        assert row["test_role_status"] == "legacy_observed_test"
        assert row["hard_role_status"] == "observed_report_only"
        assert row["evaluator_commit"] == EVALUATOR_COMMIT
        payload = evaluations[row["config_id"]]
        assert row["best_checkpoint_sha256"] == payload["checkpoint_metadata"]["best"]["sha256"]
        assert row["final_checkpoint_sha256"] == payload["checkpoint_metadata"]["final"]["sha256"]
        assert int(row["best_epoch"]) == payload["checkpoint_metadata"]["best"]["epoch"]
        assert int(row["final_epoch"]) == 600
        assert json.loads(row["result_v5_metrics_json"])["evaluator_git_commit"] == EVALUATOR_COMMIT
    equivalence = _read(GATE6D / "evaluator_equivalence.json")
    assert equivalence["authoritative_source"] == "frozen_gate5_evaluator"
    assert equivalence["exact_equivalent"] is False
    paired = _read(GATE6D / "n3_l2_valid_paired.json")
    coverage = _read(GATE6D / "global_context_coverage.json")
    assert paired["data_roles"] == ["valid_iid"] and paired["forbidden_roles_accessed"] == []
    assert coverage["fit_roles"] == ["train"] and coverage["query_roles"] == ["valid_iid"]
    assert coverage["forbidden_roles_accessed"] == [] and coverage["feature_count"] == 24
    assert coverage["target_or_label_features"] == []
    _finite(paired)
    _finite(coverage)
    sealed = _read(GATE6D / "sealed_iid_contract.json")
    manifest = _read(GATE6D / "sealed_iid_manifest.json")
    split = _read(GATE6D / "sealed_iid_split_map.json")
    assert sealed["status"] == "frozen_not_generated"
    assert sealed["model_inference_run"] is False and sealed["training_started"] is False
    assert manifest["status"] == "planned_not_generated" and manifest["labels_generated"] is False
    assert split["role"] == "sealed_iid_test" and split["sample_count"] == 128
    ids = split["sample_ids"]
    assert len(ids) == len(set(ids)) == 128
    assert set(ids).isdisjoint(_existing_ids())
    sha_payload = _read(GATE6D / "sealed_iid_contract_sha256.json")
    for name, digest in sha_payload["sha256"].items():
        assert hashlib.sha256((GATE6D / name).read_bytes()).hexdigest() == digest
    closeout = _read(GATE6D / "gate6d_preflight_closeout.json")
    assert closeout["training_started"] is False
    assert closeout["multi_seed_started"] is False
    assert closeout["new_loss_config_created"] is False
    assert closeout["authoritative_evaluator_commit"] == EVALUATOR_COMMIT
    print(json.dumps({
        "status": "passed", "evaluations": sorted(evaluations),
        "paired_roles": paired["data_roles"], "coverage_roles": ["train", "valid_iid"],
        "sealed_status": sealed["status"], "training_started": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
