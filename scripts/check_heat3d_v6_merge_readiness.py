#!/usr/bin/env python3
"""Check V6 merge-readiness declarations without performing a merge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"


def _load(name: str):
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    readiness = _load("v6_merge_readiness.json")
    total = _load("v6_total_governance_manifest.json")
    hard = _load("v6_hard_ood_closeout.json")
    preflight = _load("v6_hard_ood_preflight.json")

    assert readiness["status"] == "ready_for_final_audit"
    assert readiness["merge_executed"] is False
    assert readiness["training_executed"] is False
    assert all(
        row["status"] == "passed"
        for row in readiness["check_results"].values()
    )
    assert total["status"] == "closed"
    assert readiness["canonical_dataset"]["dataset_id"] == (
        total["canonical_dataset"]["dataset_id"]
    )
    assert readiness["canonical_dataset"]["manifest_sha256"] == (
        total["canonical_dataset"]["manifest_sha256"]
    )
    assert readiness["canonical_dataset"]["full_field_archive_sha256"] == (
        total["canonical_dataset"]["full_field_archive_sha256"]
    )
    assert readiness["canonical_model"]["checkpoint_sha256"] == (
        total["canonical_model"]["reference_checkpoint_sha256"]
    )
    assert readiness["hard_input_stress"]["sample_count"] == 16
    assert readiness["hard_input_stress"]["used_for_selection_or_tuning"] is False
    assert hard["hard_used_for_selection_or_tuning"] is False
    assert hard["training_executed"] is False
    assert hard["canonical_ood"] == {
        "labels_accessed": False,
        "reason": (
            "P1h inherits P1g train/valid/test only. Archived P1e OOD uses a "
            "different dataset/support and is outside the canonical checkpoint "
            "applicability boundary."
        ),
        "status": "not_available_not_run",
    }
    assert preflight["status"] == "passed"
    assert preflight["temperature_deltaT_or_full_field_labels_opened"] is False
    assert preflight["ood_labels_opened"] is False

    manifest_path = CONFIG / "v6_p1h_shared_support1024_manifest.json"
    assert _sha256(manifest_path) == readiness["canonical_dataset"][
        "manifest_sha256"
    ]
    docs = [
        ROOT / "docs/v6_merge_readiness.md",
        ROOT / "docs/v6_changelog.md",
        ROOT / "docs/v6_main_merge_instructions.md",
    ]
    assert all(path.is_file() for path in docs)
    absolute_pattern = re.compile(r"/(?:Users|private/tmp|home)/")
    assert all(
        not absolute_pattern.search(path.read_text(encoding="utf-8"))
        for path in docs + [CONFIG / "v6_merge_readiness.json"]
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "merge_executed": False,
                "canonical_ood": "not_available_not_run",
                "hard_used_for_selection": False,
                "absolute_path_hygiene": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
