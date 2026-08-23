#!/usr/bin/env python3
"""Fail-closed checks for P6-A archival and confirmatory evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
PROTOCOL = CFG / "v6_p1i_p6a_publication_archival_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def protocol_checks() -> dict[str, object]:
    payload = json.loads(PROTOCOL.read_text())
    require(payload["status"] == "preregistered_before_test_iid_model_inference", "protocol status")
    scope = payload["scope"]
    require(not scope["training"] and not scope["timing_rerun"], "forbidden runtime scope")
    require(scope["test_iid_open_once_for_frozen_confirmatory_evaluation"], "test opening")
    require(scope["sealed_iid_opened"] is False, "sealed opening")
    evaluation = payload["frozen_confirmatory_evaluation"]
    require(evaluation["route"] == "E16384_reconstruction", "route")
    require(evaluation["model_seed_label"] == "model_seed0", "model seed label")
    require(evaluation["checkpoint_epoch"] == 559, "checkpoint epoch")
    require(evaluation["sample_count"] == 128, "test count")
    require(evaluation["selection_allowed"] is False, "selection guard")
    manifest_path = CFG / "v6_p1i_formal1024_v1_manifest.json"
    require(sha256(manifest_path) == evaluation["dataset_manifest_sha256"], "manifest SHA")
    manifest = json.loads(manifest_path.read_text())
    sample_ids = sorted(
        (row["sample_id"] for row in manifest["samples"] if row["split_role"] == "test_iid"),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    order_sha = hashlib.sha256(("\n".join(sample_ids) + "\n").encode()).hexdigest()
    require(len(sample_ids) == 128 and order_sha == evaluation["sample_order_sha256"], "test order")
    sealed = json.loads((CFG / "v6_p1i_sealed_iid_confirmatory_preregistration.json").read_text())
    require(sealed["labels_generated"] is False and sealed["labels_opened"] is False, "sealed labels")
    return {"protocol": True, "test_count": 128, "test_order_sha256": order_sha, "sealed": False}


def closeout_checks() -> dict[str, object]:
    full_source = CFG / "v6_p1i_u_v2_16384_valid32_accuracy_only_full.json"
    result_path = CFG / "v6_p1i_e16384_test_iid_confirmatory.json"
    closeout_path = CFG / "v6_p1i_p6a_publication_archival_closeout.json"
    require(full_source.is_file() and result_path.is_file() and closeout_path.is_file(), "closeout files")
    source = json.loads(full_source.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    require(sha256(full_source) == protocol["archival"]["source_sha256"], "full source SHA")
    require(source["sample_count"] == 32 and len(source["samples"]) == 32, "full source rows")
    result = json.loads(result_path.read_text())
    require(result["status"] == "passed_frozen_test_iid_confirmatory", "test result status")
    require(result["sample_count"] == 128 and len(result["per_sample_metrics"]) == 128, "test rows")
    require(result["model_seed_label"] == "model_seed0", "test model seed")
    require(result["graph_replay"]["all_real_graph_hashes_exact"], "graph replay")
    require(result["role_contract"] == {
        "accessed_roles": ["train_inputs_for_frozen_standardizer", "test_iid"],
        "checkpoint_modified": False,
        "model_or_route_selection": False,
        "sealed_iid": False,
        "test_iid": True,
        "training": False,
    }, "test role contract")
    closeout = json.loads(closeout_path.read_text())
    require(closeout["status"] == "passed" and closeout["publication_evidence_completeness"] == "GO", "closeout")
    tables = {
        "v6_p1i_p6a_publication_main_table.csv": 5,
        "v6_p1i_p6a_supplementary_lifecycle_table.csv": 10,
        "v6_p1i_p6a_replication_table.csv": 5,
        "v6_p1i_p6a_stage_decomposition_table.csv": 56,
        "v6_p1i_p6a_claim_evidence_mapping.csv": 7,
    }
    for name, count in tables.items():
        with (CFG / name).open() as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == count, f"{name} row count")
    manifest = CFG / "v6_p1i_p6a_publication_evidence_sha256.txt"
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file() and sha256(path) == digest, f"SHA drift: {relative}")
    return {"closeout": True, "test_count": 128, "sealed": False, "tables": len(tables)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    checks = protocol_checks()
    if args.closeout:
        checks.update(closeout_checks())
    print(json.dumps({"status": "passed", **checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
