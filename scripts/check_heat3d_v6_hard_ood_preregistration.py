#!/usr/bin/env python3
"""Deterministically validate the frozen V6 hard/OOD preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"
PREREG = CONFIG / "v6_hard_ood_preregistration.json"
ROLE = CONFIG / "v6_hard_input_stress_role.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    prereg = _load(PREREG)
    role = _load(ROLE)
    assert prereg["status"] == "frozen_before_hard_specific_metric_open"
    assert prereg["training_allowed"] is False
    assert (
        prereg["checkpoint_sampling_graph_reconstruction_changes_allowed"]
        is False
    )
    assert prereg["hard_labels_read_by_preregistration_generator"] is False
    assert prereg["ood_labels_read_by_preregistration_generator"] is False
    assert prereg["checkpoint"] == {
        "config_id": "V6_03_V5best_P1h",
        "seed": 0,
        "kind": "point_global_best",
        "epoch": 111,
        "sha256": (
            "3ad58c2b34a46481acb74722c80bdcadb"
            "f55a0d613bc25c4fe2d7646b91aa1f2"
        ),
        "modified": False,
    }
    assert prereg["workflow"]["resolutions"] == {
        "4096": "default_hotspot_oriented",
        "8192": "balanced_full_field",
        "16384": "maximum_full_field_accuracy",
    }
    assert prereg["workflow"]["excluded_resolution"] == 32768
    assert prereg["roles"]["canonical_ood"]["status"] == "not_available"
    assert prereg["roles"]["canonical_ood"]["labels_must_not_be_accessed"]
    assert (
        prereg["roles"]["hard_input_stress"]["role_manifest_sha256"]
        == _sha256(ROLE)
    )
    assert role["role_id"] == "hard_input_stress_corner_v1"
    assert role["parent_split_role"] == "test"
    assert role["sample_count"] == len(role["sample_ids"]) == 16
    assert len(set(role["sample_ids"])) == 16
    assert len(set(role["group_ids"])) == 16
    assert role["selection_uses_target_labels"] is False
    assert role["selection_uses_model_errors"] is False
    assert role["is_distribution_shift_ood"] is False
    assert all(sample_id.endswith("_t0_b0_p1") for sample_id in role["sample_ids"])
    for relative, expected in role["selection_source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert _sha256(
        ROOT / prereg["dataset"]["manifest_path"]
    ) == prereg["dataset"]["manifest_sha256"]
    assert _sha256(
        ROOT / prereg["workflow"]["ladder_path"]
    ) == prereg["workflow"]["ladder_sha256"]
    assert _sha256(
        ROOT / prereg["workflow"]["evaluator_path"]
    ) == prereg["workflow"]["evaluator_sha256"]
    commands = prereg["command_plan"]
    assert len(commands) == 3
    for resolution, command in zip((4096, 8192, 16384), commands):
        assert f"--resolution {resolution}" in command
        assert "--role hard_input_stress" in command
        assert "--role-manifest configs/heat3d_v6/v6_hard_input_stress_role.json" in command
        assert "--ladder configs/heat3d_v6/v6_source_aware_resolution_ladder.json" in command
        assert "32768" not in command
        assert "hard_challenge" not in command
        assert "--mode cached" in command
    policy = prereg["selection_policy"]
    assert (
        policy["hard_or_ood_used_for_model_checkpoint_resolution_selection"]
        is False
    )
    assert policy["posthoc_reselection_or_tuning_allowed"] is False
    print(
        json.dumps(
            {
                "status": "passed",
                "hard_input_stress_samples": 16,
                "canonical_ood": "not_available",
                "hard_labels_read": False,
                "ood_labels_read": False,
                "training_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
