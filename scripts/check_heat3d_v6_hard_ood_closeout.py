#!/usr/bin/env python3
"""Validate the frozen V6 hard-stress closeout and OOD non-execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-dir", type=Path, required=True)
    args = parser.parse_args()
    closeout_path = CONFIG / "v6_hard_ood_closeout.json"
    csv_path = CONFIG / "v6_hard_ood_metrics.csv"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert closeout["status"] == "passed"
    assert closeout["preregistration_commit"] == (
        "63ef72007c6973805f08e68fad9c2f0dfe5122b6"
    )
    assert closeout["preflight_execution_head"] == (
        "b598211325ae53304de6f44a47a0f7b68845117b"
    )
    assert closeout["checkpoint"]["sha256"] == (
        "3ad58c2b34a46481acb74722c80bdcadb"
        "f55a0d613bc25c4fe2d7646b91aa1f2"
    )
    assert closeout["workflow"]["resolutions"] == {
        "4096": "default_hotspot_oriented",
        "8192": "balanced_full_field",
        "16384": "iid_average_best_full_field_accuracy",
    }
    assert closeout["workflow"]["excluded_resolution"] == 32768
    assert closeout["hard_role"]["selection_uses_target_labels"] is False
    assert closeout["hard_role"]["sample_count"] == 16
    assert closeout["hard_role"]["role_classification"] == (
        "preregistered_iid_stress_subgroup_within_already_"
        "opened_corrected_confirmatory_holdout"
    )
    assert closeout["canonical_ood"]["status"] == "not_available_not_run"
    assert closeout["canonical_ood"]["labels_accessed"] is False
    assert closeout["hard_used_for_selection_or_tuning"] is False
    assert closeout["confirmatory_holdout_used_for_selection_or_tuning"] is False
    assert closeout["posthoc_reselection_allowed"] is False
    assert closeout["training_executed"] is False
    assert closeout["checkpoint_sampling_graph_reconstruction_modified"] is False
    assert len(rows) == 9
    assert {int(row["resolution"]) for row in rows} == {4096, 8192, 16384}
    assert {row["role"] for row in rows} == {
        "valid_iid",
        "corrected_confirmatory_holdout",
        "hard_input_stress",
    }
    assert {
        row["role_classification"]
        for row in rows
        if row["role"] == "hard_input_stress"
    } == {
        "preregistered_iid_stress_subgroup_within_already_"
        "opened_corrected_confirmatory_holdout"
    }
    assert all(row["used_for_selection_or_tuning"] == "False" for row in rows)
    numeric = [
        key
        for key in rows[0]
        if key
        not in {
            "role",
            "role_classification",
            "checkpoint_sha256",
            "used_for_selection_or_tuning",
        }
    ]
    assert all(
        math.isfinite(float(row[key])) for row in rows for key in numeric
    )
    for resolution in (4096, 8192, 16384):
        path = args.hard_dir / f"hard_input_stress_{resolution}.json"
        assert _sha256(path) == closeout["hard_result_sha256"][str(resolution)]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["hard_accessed"] is True
        assert payload["ood_accessed"] is False
        assert payload["sample_count"] == 16
        assert payload["training_executed"] is False
        assert payload["checkpoint_modified"] is False
    assert not re.search(
        r"/(?:Users|private/tmp|home)/",
        closeout_path.read_text(encoding="utf-8"),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "metric_rows": len(rows),
                "hard_samples": 16,
                "canonical_ood": "not_available_not_run",
                "hard_used_for_selection": False,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
