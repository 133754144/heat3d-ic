#!/usr/bin/env python3
"""Check the V6/P1i publication evidence consolidation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
ROUTES = {
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control", "FVM240825_reference",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    payload = json.loads((CFG / "v6_p1i_publication_evidence_summary.json").read_text())
    require(payload["status"] == "passed", "summary status")
    require(payload["publication_evidence_completeness"] == "GO", "completeness")
    require(payload["performance_roles"]["primary"] == "WSL2_Attempt4", "primary role")
    require(payload["performance_roles"]["cross_host_seed_pooling"] is False, "host pooling")
    require(payload["role_contract"] == {
        "accuracy_only_route_count": 1, "machines_pooled_as_six_seeds": False,
        "sealed": False, "test": True, "test_confirmatory_route_count": 1,
        "timing_rerun": False, "training": False,
    }, "role contract")
    confirmation = payload["p6a_confirmatory_test_iid"]
    require(confirmation["status"] == "passed_frozen_test_iid_confirmatory", "test confirmation")
    require(confirmation["route"] == "E16384_reconstruction", "test route")
    require(confirmation["model_seed_label"] == "model_seed0", "test model seed")
    require(confirmation["sample_count"] == 128 and confirmation["selection_or_tuning_use"] is False,
            "test population/selection")
    require(confirmation["sealed_iid_opened"] is False, "sealed opening")
    require(payload["accuracy_only_inference"]["timing_claimed"] is False, "accuracy timing claim")
    require(payload["frozen_inputs"]["sample_order_cross_host_exact"] is True, "sample order")
    require(payload["frozen_inputs"]["cpu_policy_cross_host_exact"] is True, "CPU policy")
    require(payload["frozen_inputs"]["current_golden_exactness_seal"]["status"] == "passed", "seal")
    require(len(payload["provenance_reconciliation"]["historical_high_n_fingerprint_differences"]) == 3,
            "fingerprint differences")
    require(all(x["different"] for x in payload["provenance_reconciliation"]["historical_high_n_fingerprint_differences"]),
            "fingerprints unexpectedly equal")
    with (CFG / "v6_p1i_master_strategy_table.csv").open() as handle:
        master = list(csv.DictReader(handle))
    require(len(master) == 10 and {x["route"] for x in master} == ROUTES, "master table")
    require({x["machine"] for x in master} == {"wsl2", "devbox"}, "master machines")
    for field in ("cold_s_median", "fresh_s_median", "fresh_p95_s_median",
                  "cache_hot_s_median", "resident_s_median", "q2_submit_s_median",
                  "q2_submit_p95_s_median", "q2_throughput_samples_s_median",
                  "b16_to_b32_marginal_s_median", "ram_bytes_median", "vram_bytes_median",
                  "fresh_speedup_vs_fvm_median", "q2_speedup_vs_fvm_median"):
        require(field in master[0], f"master field missing: {field}")
    with (CFG / "v6_p1i_cross_machine_replication.csv").open() as handle:
        cross = list(csv.DictReader(handle))
    require(len(cross) == 5 and {x["route"] for x in cross} == ROUTES, "cross table")
    for field in ("wsl2_fresh_latency_rank", "devbox_fresh_latency_rank",
                  "wsl2_q2_throughput_rank", "devbox_q2_throughput_rank",
                  "devbox_over_wsl2_fresh_ratio", "devbox_over_wsl2_q2_throughput_ratio"):
        require(field in cross[0], f"cross field missing: {field}")
    with (CFG / "v6_p1i_stage_decomposition.csv").open() as handle:
        stages = list(csv.DictReader(handle))
    require(len(stages) == 56, "stage row count")
    require({x["machine"] for x in stages} == {"wsl2", "devbox"}, "stage machines")
    with (CFG / "v6_p1i_pareto_data.csv").open() as handle:
        pareto = list(csv.DictReader(handle))
    require(len(pareto) == 5 and {x["route"] for x in pareto} == ROUTES, "pareto table")
    accuracy = json.loads((CFG / "v6_p1i_u_v2_16384_valid32_accuracy_only.json").read_text())
    require(accuracy["status"] == "passed_accuracy_only" and accuracy["sample_count"] == 32,
            "accuracy-only evidence")
    require(accuracy["role_contract"] == {
        "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
        "sealed": False, "test": False, "timing_claimed": False, "training": False,
    }, "accuracy-only role contract")
    manifest = CFG / "v6_p1i_publication_evidence_sha256.txt"
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"manifest path missing: {relative}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"manifest SHA drift: {relative}")
    text = (ROOT / "docs/v6_p1i_publication_evidence_summary.md").read_text()
    for token in ("WSL2 Attempt 4", "independent, overclock-enabled", "preprocessing-bound",
                  "publication evidence completeness = GO", "P6-A confirmatory amendment",
                  "sealed IID` remains ungenerated and unopened"):
        require(token in text, f"missing report token: {token}")
    print(json.dumps({"status": "passed", "routes": 5, "lifecycle_rows": 10,
                      "stage_rows": 56, "training": False, "test": True, "sealed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
