#!/usr/bin/env python3
"""Fail-closed checks for the final V6/P1i offline closeout."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result_path = CFG / "v6_p1i_error_tail_closeout.json"
    result = json.loads(result_path.read_text())
    require(result["status"] == "passed_offline_frozen_artifact_analysis", "status")
    require(result["scientific_development_status"] == "CLOSED", "scientific status")
    require(result["normalization_contract"]["frozen_temperature_scale_K"] == 180.0, "scale")
    require(
        math.isclose(
            result["normalization_contract"]["formal_dataset_observed_max_peak_deltaT_K"],
            173.0984308776595,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "formal observed maximum",
    )
    require(
        result["normalization_contract"]["split_specific_max_used_for_primary_normalization"]
        is False,
        "split-specific normalization",
    )
    valid = result["populations"]["valid32"]
    test = result["populations"]["test128"]
    require(valid["sample_count"] == 32 and test["sample_count"] == 128, "population")
    require(math.isclose(valid["peak_rmse_K"], 4.0180122985009215, abs_tol=1e-12), "valid RMSE")
    require(math.isclose(test["peak_rmse_K"], 5.726284777743643, abs_tol=1e-12), "test RMSE")
    for population in (valid, test):
        require(
            math.isclose(
                population["peak_rmse_over_frozen_180K_pct"],
                100.0 * population["peak_rmse_K"] / 180.0,
                abs_tol=1e-12,
            ),
            "normalized peak formula",
        )
    attribution = result["test_rise_attribution"]
    require(
        attribution["classification"] == "primarily_tail_driven_with_modest_broad_shift",
        "tail classification",
    )
    require(
        95.0 < attribution["top10_share_of_test_excess_sse_vs_valid32_mean_pct"] < 96.0,
        "top10 excess attribution",
    )
    require(result["integrity"] == {
        "data_error_found": False,
        "evaluator_error_found": False,
        "provenance_error_found": False,
    }, "integrity")
    require(result["role_contract"] == {
        "inference": False,
        "interpretation": "applicability_boundary_only",
        "sealed_iid_opened": False,
        "test_iid_existing_artifact_only": True,
        "training": False,
        "used_for_selection_or_tuning": False,
        "valid_iid_existing_artifact_only": True,
    }, "role contract")

    sample_path = ROOT / result["sample_table"]["path"]
    require(sha256(sample_path) == result["sample_table"]["sha256"], "sample SHA")
    with sample_path.open() as handle:
        samples = list(csv.DictReader(handle))
    require(len(samples) == 160, "sample row count")
    require(
        sum(row["population"] == "valid32" for row in samples) == 32
        and sum(row["population"] == "test128" for row in samples) == 128,
        "sample populations",
    )
    for source in (
        result["provenance"]["formal_samples"],
        result["provenance"]["formal_distribution_audit"],
        result["provenance"]["test_result"],
        result["provenance"]["valid32_result"],
    ):
        require(sha256(ROOT / source["path"]) == source["sha256"], f"source SHA: {source['path']}")

    sealed = json.loads((CFG / "v6_p1i_sealed_iid_confirmatory_preregistration.json").read_text())
    require(
        sealed["labels_generated"] is False and sealed["labels_opened"] is False,
        "sealed IID state",
    )
    for relative, tokens in {
        "docs/v6_p1i_error_tail_closeout.md": (
            "173.098431 K", "95.45%", "primarily driven",
            "Sealed IID remains ungenerated and unopened",
        ),
        "docs/v6_p1i_closeout.md": (
            "V6/P1i scientific development = CLOSED", "Frozen claims",
            "No further valid/test analysis may change these claims",
        ),
        "docs/v6_p1i_handoff.md": (
            "Errors encountered and durable lessons", "E16384 role", "U-v2 role",
            "32,768", "sealed IID set remains ungenerated and unopened",
        ),
        "docs/v6_p1i_main_integration_plan.md": (
            "**NO-GO**",
            "GO only as a separate allowlist-based integration task",
        ),
    }.items():
        text = (ROOT / relative).read_text()
        for token in tokens:
            require(token in text, f"missing token in {relative}: {token}")

    manifest = json.loads((CFG / "v6_p1i_main_integration_manifest.json").read_text())
    require(
        manifest["audit"]["base_commit"]
        == "159d3490be661bda9dcabbd2fce7a20de7ebb734",
        "main audit base",
    )
    require(manifest["audit"]["direct_branch_merge"] == "NO_GO", "direct merge gate")
    require(
        manifest["audit"]["allowlist_integration"]
        == "GO_pending_separate_clean_checkout_validation",
        "allowlist gate",
    )
    allowlist = [
        path
        for paths in manifest["allowlist"].values()
        for path in paths
    ]
    require(len(allowlist) == len(set(allowlist)), "duplicate allowlist path")
    for relative in allowlist:
        require((ROOT / relative).is_file(), f"missing allowlist path: {relative}")
        require(
            not any(fnmatch.fnmatch(relative, pattern) for pattern in manifest["denylist_patterns"]),
            f"allowlist path matches denylist: {relative}",
        )
    require(manifest["execution_contract"]["merge_main_in_this_task"] is False, "merge contract")

    final_manifest = CFG / "v6_p1i_final_closeout_sha256.txt"
    for line in final_manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file() and sha256(path) == digest, f"final SHA drift: {relative}")

    with (CFG / "v6_p1i_p6a_claim_evidence_mapping.csv").open() as handle:
        claims = {row["claim_id"]: row for row in csv.DictReader(handle)}
    require("error-tail" in claims["C6"]["evidence"], "C6 tail evidence")
    require("tail" in claims["C6"]["boundary"], "C6 tail boundary")
    publication = json.loads((CFG / "v6_p1i_publication_evidence_summary.json").read_text())
    require(
        publication["p6a_peak_error_tail_closeout"]["classification"]
        == "primarily_tail_driven_with_modest_broad_shift",
        "publication tail binding",
    )

    print(json.dumps({
        "status": "passed",
        "offline_only": True,
        "valid_rows": 32,
        "test_rows": 128,
        "sealed": False,
        "scientific_closeout": "GO",
        "direct_main_merge": "NO_GO",
        "allowlist_paths": len(allowlist),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
