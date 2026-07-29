#!/usr/bin/env python3
"""Validate V6 governance terminology, bindings, and absolute-path hygiene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    performance = _load(CONFIG / "v6_final_performance_closeout.json")
    amendment = _load(CONFIG / "v6_governance_amendment.json")
    total = _load(CONFIG / "v6_total_governance_manifest.json")
    prereg = _load(CONFIG / "v6_hard_ood_preregistration.json")

    assert performance["confirmatory_holdout_classification"] == (
        "corrected_confirmatory_holdout"
    )
    assert performance["confirmatory_holdout_used_for_selection"] is False
    deviation = performance["protocol_deviation"]
    assert deviation["deviation_id"] == (
        "V6-PROTOCOL-DEVIATION-TEST-LADDER-001"
    )
    assert deviation["selection_or_workflow_changed"] is False
    assert {row["resolution"] for row in deviation["excluded_temporary_results"]} == {
        4096,
        8192,
    }
    assert all(
        len(row["sha256"]) == 64
        and row["status"] == "excluded_wrong_ladder_input"
        and row["used_for_selection_or_reporting"] is False
        for row in deviation["excluded_temporary_results"]
    )
    assert "legal_structured_fvm_mesh_sensitivity" in performance
    assert "matched_accuracy_fvm" not in performance
    assert performance["frozen_decision_unchanged"] == {
        "default_hotspot_oriented": 4096,
        "balanced_full_field": 8192,
        "maximum_full_field_accuracy": 16384,
        "experimental_excluded_from_primary_test_table": 32768,
    }
    assert amendment["cpu_hardware"] == {
        "model": "Apple M4",
        "physical_cores": 10,
        "core_configuration": "4 performance + 6 efficiency",
        "memory_GB": 16,
    }
    assert amendment["speedup_semantics"]["nonmatched_DOF"] is True
    assert total["canonical_dataset"]["dataset_id"] == (
        "heat3d_v6_p1h_shared_support1024_v0"
    )
    assert total["canonical_model"]["config_id"] == "V6_03_V5best_P1h"
    assert total["canonical_model"]["ablation"]["config_id"] == (
        "V6_04_V5best_P1h_DualAttention"
    )
    assert total["canonical_model"]["reference_checkpoint_sha256"] == (
        "3ad58c2b34a46481acb74722c80bdcadb"
        "f55a0d613bc25c4fe2d7646b91aa1f2"
    )
    assert total["source_aware_ladder"]["role_names"] == {
        "4096": "default_hotspot_oriented",
        "8192": "balanced_full_field",
        "16384": "maximum_full_field_accuracy",
    }
    assert total["source_aware_ladder"]["32768"] == "experimental_excluded"
    assert total["governance"]["canonical_ood_status"] == "not_available"
    assert total["local_absolute_paths_allowed"] is False
    assert total["training_executed_by_governance_closeout"] is False
    assert total["checkpoint_sampling_graph_reconstruction_modified"] is False
    assert prereg["roles"]["canonical_ood"]["status"] == "not_available"
    for row in total["official_artifacts"]:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert _sha256(path) == row["sha256"], row["path"]

    hygiene_paths = [
        CONFIG / "v6_final_performance_closeout.json",
        CONFIG / "v6_governance_amendment.json",
        CONFIG / "v6_total_governance_manifest.json",
        CONFIG / "v6_hard_ood_preregistration.json",
        ROOT / "docs/v6_total_closeout.md",
        ROOT / "docs/v6_governance_amendment.md",
    ]
    absolute_pattern = re.compile(r"/(?:Users|private/tmp|home)/")
    assert all(
        not absolute_pattern.search(path.read_text(encoding="utf-8"))
        for path in hygiene_paths
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "confirmatory_holdout": "corrected",
                "legal_structured_fvm_mesh_sensitivity": True,
                "canonical_ood": "not_available",
                "absolute_path_hygiene": True,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
