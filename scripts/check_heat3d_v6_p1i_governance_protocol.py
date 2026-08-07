#!/usr/bin/env python3
"""Validate V6 dataset-role governance, P1i inventory and frozen protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "configs/heat3d_v6/v6_phase_index.json"
LIFECYCLE = ROOT / "configs/heat3d_v6/v6_training_dataset_lifecycle.csv"
GOVERNANCE = ROOT / "configs/heat3d_v6_p1i/v6_dataset_role_governance.json"
INVENTORY = ROOT / "configs/heat3d_v6_p1i/v6_p1i_inference_artifact_inventory.json"
INVENTORY_CSV = ROOT / "configs/heat3d_v6_p1i/v6_p1i_inference_artifact_inventory.csv"
PROTOCOL = ROOT / "configs/heat3d_v6_p1i/v6_p1i_anchor_query_resolution_protocol.json"
PHASE_DOC = ROOT / "docs/v6_phase_index.md"
INVENTORY_DOC = ROOT / "docs/v6_p1i_inference_artifact_inventory.md"
PROTOCOL_DOC = ROOT / "docs/v6_p1i_anchor_query_resolution_protocol.md"

P1H_ID = "heat3d_v6_p1h_shared_support1024_v0"
P1H_SHA = "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
P1I_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
P1I_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
P1I_FULL_SHA = "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb"
LEGACY_ID = "heat3d_v6_randomblock_formal1024_v2"
LEGACY_SHA = "cdd1e9ad57b442c6b3840a74740335cbfd90f96c420da118b6b515c0e09ddc8c"
RUNS = {
    "V6_06_V5best_P1i_seed0_reliable_B24": (0, 559, "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e"),
    "V6_07_V5best_P1i_seed1_reliable_B24": (1, 455, "7197157969278d99648ef9b40d74005d759f32e52e9282e72a5586003d1e71f7"),
    "V6_08_V5best_P1i_seed2_reliable_B24": (2, 587, "d67e0dac2e8ed8009ce7dcdf0b2de4543b10bc005c0bfaa51027ea721bb2ab49"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("non-finite numeric value")


def main() -> int:
    governance = read_json(GOVERNANCE)
    phase = read_json(PHASE)
    inventory = read_json(INVENTORY)
    protocol = read_json(PROTOCOL)

    assert governance["status"] == "frozen"
    roles = governance["canonical_roles"]
    assert roles["formal_v6_layer"] == {
        "dataset_id": P1H_ID,
        "manifest_sha256": P1H_SHA,
        "status": "unchanged",
    }
    p1i_role = roles["formal_v6_randomblock"]
    assert p1i_role["dataset_id"] == P1I_ID
    assert p1i_role["manifest_sha256"] == P1I_SHA
    assert p1i_role["full_field_archive_sha256"] == P1I_FULL_SHA
    assert set(p1i_role["not_roles"]) == {
        "out_of_distribution_benchmark",
        "independent_benchmark",
    }
    deprecated = governance["deprecated_dataset"]
    assert deprecated["dataset_id"] == LEGACY_ID
    assert deprecated["manifest_sha256"] == LEGACY_SHA
    assert deprecated["status"] == "deprecated_engineering_history"
    assert deprecated["superseded_by"] == P1I_ID
    assert deprecated["historical_artifacts_mutable"] is False
    assert deprecated["current_formal_ood_role"] is False
    assert deprecated["current_independent_benchmark_role"] is False
    for binding in governance["immutable_history_bindings"]:
        path = ROOT / binding["path"]
        assert path.is_file(), path
        assert sha256(path) == binding["sha256"], path
    assert not any(governance["access_and_action_contract"].values())

    assert phase["canonical_dataset"]["dataset_id"] == P1H_ID
    assert phase["canonical_dataset"]["manifest_sha256"] == P1H_SHA
    assert phase["canonical_roles"]["formal_v6_randomblock"]["dataset_id"] == P1I_ID
    assert phase["canonical_roles"]["formal_v6_randomblock"]["manifest_sha256"] == P1I_SHA
    data_versions = {row["dataset_id"]: row for row in phase["data_versions"]}
    assert data_versions[P1I_ID]["status"] == "canonical_role_formal_v6_randomblock"
    assert data_versions[LEGACY_ID]["status"] == "deprecated_engineering_history"
    assert data_versions[LEGACY_ID]["superseded_by"] == P1I_ID
    assert phase["test_and_hard"]["true_ood_available"] is False
    assert phase["test_and_hard"]["current_independent_ood_benchmark"] is None
    assert phase["test_and_hard"]["legacy_randomblock_v2_current_benchmark_role"] is False

    with LIFECYCLE.open(newline="", encoding="utf-8") as handle:
        lifecycle = {row["dataset_id"]: row for row in csv.DictReader(handle)}
    assert lifecycle[P1I_ID]["lifecycle_status"] == "canonical_role_formal_v6_randomblock"
    assert lifecycle[LEGACY_ID]["lifecycle_status"] == "deprecated_engineering_history"
    assert lifecycle[LEGACY_ID]["training_reference_allowed"] == "false"

    assert inventory["status"] == "completed_read_only_inventory"
    assert inventory["dataset"]["dataset_id"] == P1I_ID
    assert inventory["dataset"]["canonical_role"] == "formal_v6_randomblock"
    assert inventory["dataset"]["manifest_sha256"] == P1I_SHA
    assert inventory["dataset"]["full_field_archive_sha256"] == P1I_FULL_SHA
    inventory_runs = {row["config_id"]: row for row in inventory["runs"]}
    assert set(inventory_runs) == set(RUNS)
    for config_id, (seed, epoch, checkpoint_sha) in RUNS.items():
        row = inventory_runs[config_id]
        assert row["seed"] == seed
        assert row["execution_status"] == "completed_e600"
        assert row["primary_checkpoint"]["epoch"] == epoch
        assert row["primary_checkpoint"]["sha256"] == checkpoint_sha
        assert row["training_and_checkpoint_bundle"] == "complete"
        assert row["valid_1024_support"] == "complete"
        assert row["valid_1024_to_full_field"] == "complete"
        assert row["independent_reload"] == "complete"
    assert len(inventory["gaps"]) == 6
    assert inventory["available_scope"]["anchor_derived_full_N_query_results"] is False
    assert inventory["remote_read_only_verification"]["core_artifact_hashes_match_frozen_manifest"] is True
    assert inventory["remote_read_only_verification"]["remote_training_or_inference_executed"] is False
    assert inventory["remote_read_only_verification"]["test_or_sealed_accessed"] is False
    inventory_contract = inventory["role_contract"]
    for key in ("test_accessed", "sealed_accessed", "training_executed", "inference_executed", "checkpoint_modified"):
        assert inventory_contract[key] is False
    with INVENTORY_CSV.open(newline="", encoding="utf-8") as handle:
        inventory_rows = list(csv.DictReader(handle))
    assert len(inventory_rows) == 25
    assert {row["config_id"] for row in inventory_rows if row["config_id"] != "ALL"} == set(RUNS)

    assert protocol["status"] == "frozen_before_implementation_and_execution"
    assert protocol["dataset"]["dataset_id"] == P1I_ID
    assert protocol["dataset"]["canonical_role"] == "formal_v6_randomblock"
    assert protocol["dataset"]["manifest_sha256"] == P1I_SHA
    assert protocol["dataset"]["full_field_archive_sha256"] == P1I_FULL_SHA
    protocol_runs = {row["config_id"]: row for row in protocol["checkpoints"]}
    assert set(protocol_runs) == set(RUNS)
    for config_id, (seed, epoch, checkpoint_sha) in RUNS.items():
        row = protocol_runs[config_id]
        assert (row["seed"], row["epoch"], row["sha256"]) == (seed, epoch, checkpoint_sha)
        assert sha256(ROOT / f"configs/heat3d_v6_p1i/{config_id}.yaml") == row["config_sha256"]
    assert protocol["query_contract"]["mandatory_resolutions"] == [1024, 4096, 8192, 16384]
    assert protocol["query_contract"]["optional_resolution"]["nodes"] == 32768
    assert protocol["query_contract"]["optional_resolution"]["may_enter_mandatory_ranking"] is False
    assert protocol["query_contract"]["all_1024_anchors_retained"] is True
    assert protocol["query_contract"]["temperature_or_test_label_use"] is False
    assert protocol["anchor_contract"]["global_context_source"] == "anchors_only"
    assert protocol["anchor_contract"]["predicted_scale_source"] == "anchor_forward_only"
    assert protocol["anchor_contract"]["target_or_label_use"] is False
    graph = protocol["graph_contract"]
    assert graph["backend"] == "sparse_kdtree_v1"
    assert set(graph["cache_key_fields"]) == {
        "support_hash", "resolved_graph_config", "graph_seed", "graph_builder_code_fingerprint"
    }
    assert sha256(ROOT / graph["graph_builder_path"]) == graph["graph_builder_sha256_at_freeze"]
    reconstruction = protocol["reconstruction_contract"]
    assert sha256(ROOT / reconstruction["implementation_path"]) == reconstruction["implementation_sha256_at_freeze"]
    assert protocol["historical_result_policy"]["may_be_relabelled_as_anchor_query"] is False
    assert protocol["historical_result_policy"]["legacy_randomblock_v2_may_be_used_for_selection_or_benchmarking"] is False
    assert protocol["implementation_gate"]["adapter_status"] == "not_implemented_this_round"
    assert protocol["implementation_gate"]["execution_authorized_this_round"] is False
    role = protocol["role_contract"]
    for key in ("test_accessed", "sealed_accessed", "training_executed", "checkpoint_modified", "model_selection_allowed", "large_scale_inference_executed_this_round"):
        assert role[key] is False

    for path, terms in (
        (PHASE_DOC, ("formal_v6_randomblock", "deprecated_engineering_history", "not make P1i an OOD or independent benchmark")),
        (INVENTORY_DOC, ("full-graph re-discretization diagnostic", "anchor-derived valid results", "test/sealed roles remained closed")),
        (PROTOCOL_DOC, ("1024-anchor", "sparse_kdtree_v1", "Status: frozen before implementation and execution")),
    ):
        text = path.read_text(encoding="utf-8")
        for term in terms:
            assert term in text, (path, term)

    for payload in (governance, phase, inventory, protocol):
        finite_tree(payload)
    print(json.dumps({
        "status": "passed",
        "canonical_randomblock_dataset": P1I_ID,
        "deprecated_dataset": LEGACY_ID,
        "inventory_rows": len(inventory_rows),
        "mandatory_resolutions": protocol["query_contract"]["mandatory_resolutions"],
        "optional_resolution": protocol["query_contract"]["optional_resolution"]["nodes"],
        "training_executed": False,
        "inference_executed": False,
        "test_accessed": False,
        "sealed_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
