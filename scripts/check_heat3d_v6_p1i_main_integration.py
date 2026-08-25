#!/usr/bin/env python3
"""Fail-closed audit for strict-allowlist V6/P1i main integration."""

from __future__ import annotations

import fnmatch
import hashlib
import csv
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
MANIFEST_PATH = CFG / "v6_p1i_main_integration_manifest.json"
RECEIPT_PATH = CFG / "v6_p1i_main_integration_receipt.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def canonical_lines_sha(rows: list[str]) -> str:
    payload = "".join(f"{row}\n" for row in sorted(rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    receipt = json.loads(RECEIPT_PATH.read_text())
    require(manifest["status"] == "validated_ready_for_pr", "manifest status")
    require(receipt["status"] == "merged_main_verified", "receipt status")
    require(receipt["base"]["commit"] == manifest["audit"]["base_commit"], "base")
    require(
        receipt["source"]["commit"]
        == manifest["audit"]["source_head_before_integration"],
        "source",
    )
    require(receipt["integration"]["whole_history_merge"] is False, "whole merge")
    require(receipt["integration"]["strict_allowlist_only"] is True, "allowlist")
    require(receipt["integration"]["pull_request"] == {
        "number": 3,
        "url": "https://github.com/133754144/heat3d-ic/pull/3",
        "state": "merged",
        "required_merge_method": "merge_commit",
        "head_commit": "e1e89f6608ff353bee26534d8cde14b17804255a",
        "merge_commit": "2205883c23f387a08aa7c5ef8b2f5e06688f4793",
        "merged_at_utc": "2026-08-25T04:48:10Z",
    }, "pull request binding")
    merge_commit = receipt["integration"]["merged_main_commit"]
    require(
        merge_commit == "2205883c23f387a08aa7c5ef8b2f5e06688f4793",
        "merged main commit",
    )
    require(
        git("rev-parse", f"{merge_commit}^1") == receipt["base"]["commit"],
        "merge first parent",
    )
    require(
        git("rev-parse", f"{merge_commit}^2")
        == receipt["integration"]["pull_request"]["head_commit"],
        "merge second parent",
    )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", merge_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode == 0,
        "merge commit ancestry",
    )

    allowlist = [
        path for paths in manifest["allowlist"].values() for path in paths
    ]
    require(len(allowlist) == len(set(allowlist)), "duplicate allowlist path")
    require(len(allowlist) == receipt["integration"]["allowlist_path_count"], "count")
    for relative in allowlist:
        require((ROOT / relative).is_file(), f"missing allowlist file: {relative}")
        require(
            not any(
                fnmatch.fnmatch(relative, pattern)
                for pattern in manifest["denylist_patterns"]
            ),
            f"denylist match: {relative}",
        )

    staged = [
        row for row in git("diff", "--cached", "--name-only").splitlines() if row
    ]
    if staged:
        require(set(staged) <= set(allowlist), "staged path outside allowlist")
        committed = [
            row
            for row in git(
                "diff", "--name-only", receipt["base"]["commit"], "HEAD"
            ).splitlines()
            if row
        ]
        integrated_paths = sorted(set(committed) | set(staged))
        require(
            canonical_lines_sha(integrated_paths)
            == receipt["integration"]["staged_path_list_sha256"],
            "integrated path list SHA",
        )
        require(
            len(integrated_paths) == receipt["integration"]["staged_path_count"],
            "integrated path count",
        )
    else:
        committed = [
            row
            for row in git(
                "diff", "--name-only", receipt["base"]["commit"], "HEAD"
            ).splitlines()
            if row
        ]
        require(
            len(committed) == receipt["integration"]["staged_path_count"],
            "committed path count",
        )
        require(
            canonical_lines_sha(committed)
            == receipt["integration"]["staged_path_list_sha256"],
            "committed list SHA",
        )
        require(
            subprocess.run(
                [
                    "git", "merge-base", "--is-ancestor",
                    receipt["integration"]["content_commit"], "HEAD",
                ],
                cwd=ROOT,
                check=False,
            ).returncode == 0,
            "content commit ancestry",
        )

    phase = json.loads(
        (ROOT / "configs/heat3d_v6/v6_phase_index.json").read_text()
    )
    require(
        phase["canonical_roles"]["formal_v6_layer"]["dataset_id"]
        == "heat3d_v6_p1h_shared_support1024_v0",
        "P1h role",
    )
    require(
        phase["canonical_roles"]["formal_v6_randomblock"]["dataset_id"]
        == "heat3d_v6_p1i_continuous_physics1024_v1",
        "P1i role",
    )
    require(
        phase["governance_terms"]["sealed_iid"]
        == "post-development final confirmation; ungenerated and unopened",
        "sealed role",
    )
    require(phase["p1i_final_closeout"]["sealed_iid_opened"] is False, "sealed")
    require(
        phase["inference_evolution"][-2]["decision"]
        == "frozen_noninferiority_latency_pareto_reference",
        "E16384 decision",
    )
    with (ROOT / "configs/heat3d_v6/v6_model_lifecycle.csv").open() as handle:
        lifecycle = {row["config_id"]: row for row in csv.DictReader(handle)}
    require(
        lifecycle["V6_06_V5best_P1i_seed0_reliable_B24"]["run_role"]
        == "reference_seed0",
        "V6_06 lifecycle",
    )
    for config_id, role in (
        ("V6_07_V5best_P1i_seed1_reliable_B24", "replication_seed1"),
        ("V6_08_V5best_P1i_seed2_reliable_B24", "replication_seed2"),
    ):
        require(lifecycle[config_id]["run_role"] == role, f"{config_id} lifecycle")
    for relative in (
        "docs/v6_total_closeout.md",
        "docs/v6_p1i_closeout.md",
    ):
        text = (ROOT / relative).read_text()
        require("E240825_direct_control" in text, f"canonical route: {relative}")
        require("`E240825_direct`" not in text, f"legacy route: {relative}")

    validation = receipt["validation"]
    for key in (
        "exact_staged_path_audit",
        "v5_regressions",
        "v6_p1i_checkers",
        "config_dry_run",
        "python_compile",
        "json_yaml_csv",
        "git_diff_check",
        "clean_checkout_replay",
    ):
        require(validation[key] == "passed", f"validation: {key}")
    require(validation["sealed_iid_accessed"] is False, "sealed access")
    require(validation["training_executed"] is False, "training")
    require(validation["merge_method"] == "merge_commit_verified", "merge method")
    require(validation["post_merge_main_checks"] == "passed", "post-merge checks")
    require(validation["pr_head_second_parent_verified"] is True, "PR second parent")

    print(json.dumps({
        "status": "passed",
        "allowlist_paths": len(allowlist),
        "staged_paths": len(staged),
        "base": receipt["base"]["commit"],
        "source": receipt["source"]["commit"],
        "sealed_iid_accessed": False,
        "training_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
