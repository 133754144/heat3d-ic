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


def main() -> None:
    metric = _load("v7_metric_contract.json")
    registry = _load("v7_experiment_registry.json")
    claims = _load("v7_claim_evidence_mapping.json")
    denylist = _load("v7_frozen_artifact_denylist.json")

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
    if {row["experiment_id"] for row in rows} != {"V7.1-T-readiness-fixture", "V7-G1-template"}:
        raise ValueError("CSV registry is not synchronized with JSON registry")
    for path in (CONTROL / "v7_metric_contract.json", CONTROL / "v7_experiment_registry.json", CONTROL / "v7_claim_evidence_mapping.json", CONTROL / "v7_frozen_artifact_denylist.json"):
        if any(token in path.read_text(encoding="utf-8").lower() for token in ("test_iid_labels", "sealed_labels")):
            raise ValueError(f"forbidden label marker in {path}")
    print("V7 training control plane: PASS")


if __name__ == "__main__":
    main()
