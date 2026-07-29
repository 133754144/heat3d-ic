#!/usr/bin/env python3
"""Validate the compact V6 phase index against frozen lifecycle evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"


def main() -> int:
    index = json.loads(
        (CONFIG / "v6_phase_index.json").read_text(encoding="utf-8")
    )
    total = json.loads(
        (CONFIG / "v6_total_governance_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with (CONFIG / "v6_training_dataset_lifecycle.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        lifecycle = list(csv.DictReader(handle))
    with (CONFIG / "v6_model_lifecycle.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        models = list(csv.DictReader(handle))

    assert index["status"] == "closed"
    assert index["canonical_dataset"]["dataset_id"] == (
        total["canonical_dataset"]["dataset_id"]
    )
    assert index["canonical_dataset"]["manifest_sha256"] == (
        total["canonical_dataset"]["manifest_sha256"]
    )
    assert index["canonical_model"]["checkpoint_sha256"] == (
        total["canonical_model"]["reference_checkpoint_sha256"]
    )
    assert {row["dataset_id"] for row in index["data_versions"]} == {
        row["dataset_id"] for row in lifecycle
    }
    assert {row["config_id"] for row in index["model_versions"]} == {
        "V6_01_V4best",
        "V6_02_V5best",
        "V6_03_V5best_P1h",
        "V6_04_V5best_P1h_DualAttention",
    }
    assert {row["config_id"] for row in models} == {
        "V6_03_V5best_P1h",
        "V6_03_V5best_P1h_seed1",
        "V6_03_V5best_P1h_seed2",
        "V6_04_V5best_P1h_DualAttention",
    }
    assert index["governance_terms"] == {
        "fvm": "legal structured-FVM mesh sensitivity",
        "hard": (
            "preregistered IID stress subgroup within the already-opened "
            "corrected confirmatory holdout"
        ),
        "resolution_16384": (
            "highest IID-average full-field accuracy mode"
        ),
    }
    assert index["test_and_hard"]["hard_used_for_selection"] is False
    assert index["test_and_hard"]["test_used_for_selection"] is False
    assert index["test_and_hard"]["true_ood_available"] is False
    assert index["protocol_deviation"]["used_for_selection"] is False
    assert (ROOT / "docs/v6_phase_index.md").is_file()
    print(
        json.dumps(
            {
                "status": "passed",
                "dataset_versions": len(index["data_versions"]),
                "model_versions": len(index["model_versions"]),
                "true_ood_available": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
