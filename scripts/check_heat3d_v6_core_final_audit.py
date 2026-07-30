#!/usr/bin/env python3
"""Validate the no-new-merge V6 clean-integration final audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/heat3d_v6/v6_core_final_audit_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["branch"] == "integration/v6-core-final-audit"
    assert manifest["status"] == "validated_no_new_main_merge"
    assert manifest["external_state_correction"] == {
        "main_already_contains_v6_before_this_goal": True,
        "main_merge_commit": "332ef3f463d91442632c3ebddd4f7549c7895b8d",
        "main_rewritten_or_reverted": False,
        "new_merge_to_main_authorized": False,
        "new_merge_to_main_performed": False,
    }
    assert manifest["base_main_commit"] == (
        "332ef3f463d91442632c3ebddd4f7549c7895b8d"
    )
    assert manifest["base_main_tree_sha1"] == (
        "b08defef2c39910c7e52152039597ef482d77c19"
    )
    assert manifest["prior_integration_branch"]["tree_identical_to_current_main"]
    assert manifest["prior_integration_branch"]["tree_sha1"] == (
        manifest["base_main_tree_sha1"]
    )
    assert _git("rev-parse", "332ef3f^{tree}") == manifest[
        "base_main_tree_sha1"
    ]
    assert _git("rev-parse", "82beed94^{tree}") == manifest[
        "base_main_tree_sha1"
    ]
    for entry in manifest["frozen_evidence"].values():
        path = ROOT / entry["path"]
        assert path.is_file(), path
        assert _sha256(path) == entry["sha256"], path
    phase = json.loads(
        (ROOT / manifest["frozen_evidence"]["phase_index"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert phase["governance_terms"] == {
        "fvm": "legal structured-FVM mesh sensitivity",
        "hard": "preregistered IID stress subgroup within the already-opened corrected confirmatory holdout",
        "resolution_16384": "highest IID-average full-field accuracy mode",
    }
    assert phase["test_and_hard"]["true_ood_available"] is False
    assert phase["test_and_hard"]["hard_used_for_selection"] is False
    assert phase["test_and_hard"]["test_used_for_selection"] is False
    assert manifest["stable_scope"] == {
        "source_of_truth": "configs/heat3d_v6/v6_core_integration_manifest.json",
        "allowlist_only": True,
        "full_research_history_merged": False,
        "frozen_v6_results_modified": False,
    }
    assert manifest["validation"]["v5_checker_count"] == 6
    assert all(
        manifest["validation"][key] is True
        for key in (
            "v5_checkers_passed",
            "v6_core_checker_passed",
            "canonical_config_dry_run_passed",
            "python_compile_passed",
            "json_yaml_csv_validation_passed",
            "git_diff_check_passed",
        )
    )
    assert manifest["validation"]["training_executed"] is False
    assert manifest["validation"]["model_inference_executed"] is False
    assert manifest["validation"]["test_hard_accessed"] is False
    assert not any(
        (ROOT / path).exists()
        for path in (
            "data",
            "output/heat3d_v6_runs",
            "checkpoints",
            "logs",
            "predictions",
        )
    )
    assert manifest["inherited_main_exception"] == {
        "path": "output/heat3d_ic/heat3d_operator_best.pkl",
        "reason": "pre-existing tracked V5 integration fixture inherited from main; not a V6 output and not modified by this branch",
    }
    assert (ROOT / manifest["inherited_main_exception"]["path"]).is_file()
    assert _git("diff", "--quiet", "332ef3f", "--", "output") == ""
    print(
        json.dumps(
            {
                "status": "passed",
                "base_main_commit": manifest["base_main_commit"],
                "tree_identity": True,
                "governance_terms": manifest["governance"],
                "v5_checker_count": 6,
                "v6_core_checker_passed": True,
                "new_merge_to_main_performed": False,
                "frozen_v6_results_modified": False,
                "training_executed": False,
                "model_inference_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
