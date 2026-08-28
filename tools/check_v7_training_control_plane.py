"""Fail-closed validation for the V7.1-T training readiness control plane."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "heat3d_v7"


def _load(name: str) -> dict:
    path = CONTROL / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _load_path(relative: str) -> dict:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def main() -> None:
    metric = _load("v7_metric_contract.json")
    registry = _load("v7_experiment_registry.json")
    claims = _load("v7_claim_evidence_mapping.json")
    denylist = _load("v7_frozen_artifact_denylist.json")
    p1i = _load_path("configs/heat3d_v7/v7_g1_full_p1i.json")

    required_metrics = {
        "point_global_relative_rmse_pct",
        "sample_first_relative_rmse_pct",
        "raw_K_CV_RMSE_K",
        "source_region_RMSE_K",
        "peak_RMSE_K",
        "interface_RMSE_K",
    }
    actual = {row["name"] for row in metric["level_a_training_resolution"]["metrics"]}
    if actual != required_metrics:
        raise ValueError(f"metric contract drift: {sorted(actual)}")
    if metric["level_b_high_resolution"]["metrics"] != "level_a_training_resolution.metrics":
        raise ValueError("Level-B must reference the frozen Level-A metric definitions")
    if registry["split_policy"]["forbidden_in_this_task"] != ["test_iid", "sealed"]:
        raise ValueError("test/sealed split policy drifted")
    if denylist["policy"] != "read_only; do_not_delete; do_not_clean; do_not_overwrite; do_not_regenerate_in_place":
        raise ValueError("frozen artifact policy drifted")
    if not claims["prohibited_claims"]:
        raise ValueError("publication claim denylist cannot be empty")
    with (CONTROL / "v7_experiment_registry.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_ids = {
        "V7.1-T-readiness-fixture",
        "V7-G1-template",
        "V7-G1-Full-P1i",
        "V7-G1-Full-P1i:vanilla-RIGNO",
        "V7-G1-Full-P1i:generic-uniform-support",
        "V7-G1-Full-P1i:volume-only-support",
        "V7-G1-Full-P1i:no-context",
        "V7-G1-Full-P1i:no-scale",
    }
    if {row["experiment_id"] for row in rows} != expected_ids:
        raise ValueError("CSV registry is not synchronized with JSON registry")
    if registry.get("formal_training_entrypoint") != "scripts/run_heat3d_v7_formal_p1i_training.py":
        raise ValueError("formal P1i entrypoint binding drifted")
    if p1i.get("experiment_id") != "V7-G1-Full-P1i" or p1i.get("status") != "registered_not_executed":
        raise ValueError("Full P1i registration must remain planned and unexecuted")
    dataset = p1i.get("dataset", {})
    if dataset.get("dataset_id") != "heat3d_v6_p1i_continuous_physics1024_v1":
        raise ValueError("Full P1i dataset binding drifted")
    if dataset.get("manifest_sha256") != "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514":
        raise ValueError("Full P1i manifest binding drifted")
    if dataset.get("roles", {}).get("train") != 768 or dataset.get("roles", {}).get("valid_iid") != 128:
        raise ValueError("Full P1i train/valid population drifted")
    if dataset.get("label_access", {}).get("test_iid") != "forbidden" or dataset.get("label_access", {}).get("sealed") != "forbidden":
        raise ValueError("Full P1i test/sealed access policy drifted")
    batching = p1i.get("batching", {})
    expected_batching = {
        "batch_size": 24,
        "micro_batch_size": 24,
        "validation_batch_size": 32,
        "prediction_batch_size": 32,
        "batch_plan": "sample_shuffle",
        "shuffle_train_batches": True,
        "batch_build_seed": 0,
        "sample_weight_policy": "none",
        "drop_last": False,
        "train_batches_per_epoch": 32,
    }
    for key, value in expected_batching.items():
        if batching.get(key) != value:
            raise ValueError(f"Full P1i batching drifted: {key}={batching.get(key)!r}")
    json_runs = {row["experiment_id"] for row in registry.get("registered_runs", [])}
    if not expected_ids.issubset(json_runs):
        raise ValueError("JSON registry is missing registered Full/variant entries")
    full_entry = next(row for row in registry["registered_runs"] if row["experiment_id"] == "V7-G1-Full-P1i")
    if full_entry.get("execution_started") is not False or full_entry.get("g1_eligible") is not True:
        raise ValueError("Full P1i execution state drifted")
    for entry in registry["registered_runs"]:
        if entry["experiment_id"].startswith("V7-G1-Full-P1i:"):
            if entry.get("status") != "planned_not_executed" or not entry.get("delta"):
                raise ValueError(f"variant registration is not delta-only/planned: {entry['experiment_id']}")
    for path in (CONTROL / "v7_metric_contract.json", CONTROL / "v7_experiment_registry.json", CONTROL / "v7_claim_evidence_mapping.json", CONTROL / "v7_frozen_artifact_denylist.json"):
        if any(token in path.read_text(encoding="utf-8").lower() for token in ("test_iid_labels", "sealed_labels")):
            raise ValueError(f"forbidden label marker in {path}")
    print("V7 training control plane: PASS")


if __name__ == "__main__":
    main()
