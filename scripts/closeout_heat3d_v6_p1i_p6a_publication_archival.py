#!/usr/bin/env python3
"""Build paper-ready P6-A tables from immutable evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOCS = ROOT / "docs"
PAIRED = (
    "within each lifecycle seed and identical ordered sample IDs compute "
    "FVM/neural paired workload ratio first; report median and min-max over "
    "the three lifecycle seeds; never pool 96 samples across machines"
)
NN_TAIL = (
    "WSL2 E16384 NN/reconstruction median=0.007738753 s and p95=0.085637189 s; "
    "the observed upper tail is retained without exclusion or replacement"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None or value == "" else f"{float(value):.{digits}f}"


def main() -> int:
    protocol_path = CFG / "v6_p1i_p6a_publication_archival_protocol.json"
    full_source_path = CFG / "v6_p1i_u_v2_16384_valid32_accuracy_only_full.json"
    test_path = CFG / "v6_p1i_e16384_test_iid_confirmatory.json"
    evidence_path = CFG / "v6_p1i_publication_evidence_summary.json"
    protocol = load_json(protocol_path)
    source = load_json(full_source_path)
    test = load_json(test_path)
    evidence = load_json(evidence_path)
    if sha256(full_source_path) != protocol["archival"]["source_sha256"]:
        raise RuntimeError("U16384 complete source SHA drifted")
    if source.get("sample_count") != 32 or len(source.get("samples", [])) != 32:
        raise RuntimeError("U16384 complete per-sample rows missing")
    if test.get("status") != "passed_frozen_test_iid_confirmatory" or test.get("sample_count") != 128:
        raise RuntimeError("test_iid confirmation is incomplete")
    if test.get("model_seed_label") != "model_seed0":
        raise RuntimeError("ambiguous model seed label")
    test_accuracy = test["accuracy"]["full_field"]
    historical_accuracy = load_json(CFG / "v6_p1i_u4_direct240825_closeout.json")["historical_aggregate_metrics"]
    valid_sample_first = {
        "E16384_reconstruction": historical_accuracy["E16384_reconstruction"]["sample_first_cv_relative_rmse_pct"],
        "U_v2_16384_reconstruction": source["accuracy"]["full_field"]["sample_first_cv_relative_rmse_pct"],
        "U_v2_direct240825": historical_accuracy["U_direct240825"]["sample_first_cv_relative_rmse_pct"],
        "E240825_direct_control": historical_accuracy["E240825_direct"]["sample_first_cv_relative_rmse_pct"],
        "FVM240825_reference": None,
    }
    strategies = evidence["strategy_table"]
    if len(strategies) != 5:
        raise RuntimeError("strategy table route count drifted")

    evidence["p6a_confirmatory_test_iid"] = {
        "status": "passed_frozen_test_iid_confirmatory",
        "route": "E16384_reconstruction",
        "model_seed_label": "model_seed0",
        "sample_count": 128,
        "artifact_path": str(test_path.relative_to(ROOT)),
        "artifact_sha256": sha256(test_path),
        "accuracy": test_accuracy,
        "selection_or_tuning_use": False,
        "sealed_iid_opened": False,
    }
    evidence["claims"]["claim_boundary"] = (
        "valid32 timing/accuracy evidence plus one-time corrected confirmatory "
        "test_iid evaluation of the already-frozen E16384 route; sealed IID remains unopened"
    )
    evidence["role_contract"] = {
        "accuracy_only_route_count": 1,
        "test_confirmatory_route_count": 1,
        "machines_pooled_as_six_seeds": False,
        "sealed": False,
        "test": True,
        "timing_rerun": False,
        "training": False,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")

    evidence_md_path = DOCS / "v6_p1i_publication_evidence_summary.md"
    amendment_marker = "## P6-A confirmatory amendment"
    historical_text = evidence_md_path.read_text().split(amendment_marker, 1)[0].rstrip()
    evidence_md_path.write_text(
        historical_text + "\n\n" + amendment_marker + "\n\n"
        "The earlier valid32 closeout statement that test/sealed confirmation remained pending "
        "is superseded only for the corrected `test_iid` holdout. After route and checkpoint "
        "freeze, `E16384_reconstruction` with `model_seed0` was evaluated once on all 128 "
        "ordered test samples: full-field PG `{:.6f}%`, sample-first `{:.6f}%`, raw CV RMSE "
        "`{:.6f} K`, source `{:.6f} K`, peak `{:.6f} K`, and interface `{:.6f} K`. "
        "The result is descriptive confirmation and was not used for selection, tuning, or "
        "threshold revision. `sealed IID` remains ungenerated and unopened.\n".format(
            test_accuracy["point_global_true_rms_relative_rmse_pct"],
            test_accuracy["sample_first_cv_relative_rmse_pct"],
            test_accuracy["raw_cv_weighted_rmse_K"], test_accuracy["source_rmse_K"],
            test_accuracy["peak_rmse_K"], test_accuracy["interface_drop_rmse_K"],
        )
    )

    old_manifest = CFG / "v6_p1i_publication_evidence_sha256.txt"
    old_manifest_paths = [ROOT / line.split("  ", 1)[1]
                          for line in old_manifest.read_text().splitlines()]
    old_manifest.write_text("".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in old_manifest_paths
    ))

    main_rows = []
    for row in strategies:
        neural = row["route"] != "FVM240825_reference"
        main_rows.append({
            "route": row["route"],
            "model_seed_label": "model_seed0" if neural else "N/A",
            "valid_population": "frozen_valid32" if neural else "reference_solution",
            "valid_point_global_rmse_pct": row["point_global_rmse_pct"],
            "valid_sample_first_cv_relative_rmse_pct": valid_sample_first[row["route"]],
            "valid_raw_cv_rmse_K": row["raw_cv_rmse_K"],
            "valid_source_rmse_K": row["source_rmse_K"],
            "valid_peak_rmse_K": row["peak_rmse_K"],
            "valid_interface_rmse_K": row["interface_rmse_K"],
            "test_confirmatory_population": "test_iid_128" if row["route"] == "E16384_reconstruction" else "N/A",
            "test_point_global_rmse_pct": test_accuracy["point_global_true_rms_relative_rmse_pct"] if row["route"] == "E16384_reconstruction" else None,
            "test_sample_first_cv_relative_rmse_pct": test_accuracy["sample_first_cv_relative_rmse_pct"] if row["route"] == "E16384_reconstruction" else None,
            "test_raw_cv_rmse_K": test_accuracy["raw_cv_weighted_rmse_K"] if row["route"] == "E16384_reconstruction" else None,
            "test_source_rmse_K": test_accuracy["source_rmse_K"] if row["route"] == "E16384_reconstruction" else None,
            "test_peak_rmse_K": test_accuracy["peak_rmse_K"] if row["route"] == "E16384_reconstruction" else None,
            "test_interface_rmse_K": test_accuracy["interface_drop_rmse_K"] if row["route"] == "E16384_reconstruction" else None,
            "wsl2_fresh_median_s": row["fresh_median_s"],
            "wsl2_fresh_p95_s": row["fresh_p95_median_s"],
            "wsl2_resident_median_s": row["resident_median_s"],
            "wsl2_q2_samples_s": row["q2_throughput_samples_s"],
            "wsl2_fresh_paired_speedup_vs_fvm": row["fresh_speedup_vs_fvm"],
            "wsl2_q2_paired_speedup_vs_fvm": row["q2_speedup_vs_fvm"],
            "paired_speedup_definition": PAIRED,
            "evidence_role": "WSL2_Attempt4_primary_authoritative",
        })

    lifecycle_rows = []
    for row in load_csv(CFG / "v6_p1i_master_strategy_table.csv"):
        lifecycle_rows.append({
            **row,
            "model_seed_label": "N/A" if row["route"] == "FVM240825_reference" else "model_seed0",
            "paired_speedup_definition": PAIRED,
        })
    replication_rows = []
    for row in load_csv(CFG / "v6_p1i_cross_machine_replication.csv"):
        replication_rows.append({
            **row,
            "model_seed_label": "N/A" if row["route"] == "FVM240825_reference" else "model_seed0",
            "paired_speedup_definition": PAIRED,
            "replication_interpretation": "devbox is independent overclock-enabled hardware-state replication; it does not replace WSL2 or add model seeds",
        })
    stage_rows = []
    for row in load_csv(CFG / "v6_p1i_stage_decomposition.csv"):
        is_tail = (row["machine"] == "wsl2" and row["route"] == "E16384_reconstruction"
                   and row["stage"] == "nn_reconstruction")
        stage_rows.append({**row, "model_seed_label": "model_seed0", "anomaly_note": NN_TAIL if is_tail else ""})

    claim_rows = [
        {"claim_id": "C1", "claim": "16k plus reconstruction is the primary valid32 accuracy-latency Pareto family", "status": "supported_with_valid32_scope", "evidence": "Main Table; WSL2 Attempt4; frozen valid32 accuracy", "boundary": "not a universal topology or dataset claim"},
        {"claim_id": "C2", "claim": "E16384 and U-v2 16384 are in the same end-to-end performance class", "status": "supported_on_two_reported_hosts", "evidence": "Main Table; Replication Table", "boundary": "hosts are reported separately and not pooled"},
        {"claim_id": "C3", "claim": "U-v2 direct improves valid32 direct-output accuracy at approximately equal WSL2 fresh latency versus E-direct", "status": "supported_on_valid32", "evidence": "Main Table", "boundary": "diagnostic direct strategies; no test comparison"},
        {"claim_id": "C4", "claim": "paired neural/FVM speedup is reproducible across WSL2 primary and devbox replication", "status": "supported_with_hardware_state_caveat", "evidence": "Supplementary Lifecycle; Replication Table", "boundary": PAIRED},
        {"claim_id": "C5", "claim": "the measured neural service is preprocessing-bound", "status": "supported_for_frozen_valid32_workload", "evidence": "Stage Decomposition Table", "boundary": NN_TAIL},
        {"claim_id": "C6", "claim": "frozen E16384 has one-time quantified accuracy on the corrected confirmatory test_iid holdout", "status": "confirmatory_descriptive_only", "evidence": "Main Table; test_iid per-sample artifact", "boundary": "test opened once after route freeze and never used for selection or tuning; test peak RMSE is higher than valid32"},
        {"claim_id": "C7", "claim": "sealed IID remains an unopened future confirmation boundary", "status": "not_evaluated", "evidence": "sealed preregistration and P6-A role contract", "boundary": "labels not generated or opened"},
    ]

    outputs = {
        "main": CFG / "v6_p1i_p6a_publication_main_table.csv",
        "lifecycle": CFG / "v6_p1i_p6a_supplementary_lifecycle_table.csv",
        "replication": CFG / "v6_p1i_p6a_replication_table.csv",
        "stage": CFG / "v6_p1i_p6a_stage_decomposition_table.csv",
        "claims": CFG / "v6_p1i_p6a_claim_evidence_mapping.csv",
    }
    for path, rows in zip(outputs.values(), (main_rows, lifecycle_rows, replication_rows, stage_rows, claim_rows)):
        write_csv(path, rows)

    closeout = {
        "schema_version": "heat3d_v6_p1i_p6a_publication_archival_closeout_v1",
        "status": "passed",
        "publication_evidence_completeness": "GO",
        "model_seed_label": "model_seed0",
        "u16384_complete_source": {"path": str(full_source_path.relative_to(ROOT)), "sha256": sha256(full_source_path), "per_sample_rows": 32},
        "test_iid_confirmatory": {
            "path": str(test_path.relative_to(ROOT)), "sha256": sha256(test_path),
            "execution_host": "devbox",
            "execution_commit": "49188c6cbc1f6fa4bbd5414fd978c769272b79fc",
            "evaluator_sha256": sha256(ROOT / "scripts/evaluate_heat3d_v6_p1i_e16384_test_confirmatory.py"),
            "protocol_sha256": sha256(protocol_path),
            "route": "E16384_reconstruction", "sample_count": 128,
            "accuracy": test_accuracy, "selection_or_tuning_use": False,
            "execution_attempts": [
                {"jax_platforms": "gpu", "status": "failed_before_label_access", "reason": "ROCm backend unavailable"},
                {"jax_platforms": "cuda", "status": "failed_before_label_access", "reason": "CPU graph backend hidden"},
                {"jax_platforms": "cuda,cpu", "status": "passed", "reason": None},
            ],
        },
        "paired_speedup_definition": PAIRED,
        "wsl2_nn_tail": NN_TAIL,
        "paper_tables": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
        "sealed_iid": {"opened": False, "labels_generated": False, "labels_opened": False},
        "role_contract": {"training": False, "timing_rerun": False, "test_iid": True, "sealed_iid": False, "model_or_route_selection": False},
    }
    closeout_path = CFG / "v6_p1i_p6a_publication_archival_closeout.json"
    closeout_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")

    md_path = DOCS / "v6_p1i_p6a_publication_tables.md"
    lines = [
        "# V6/P1i P6-A publication evidence tables", "",
        "All neural accuracy rows use `model_seed0`. WSL2 Attempt 4 is the primary performance result; devbox is a separate overclock-enabled hardware-state replication and is never pooled as additional model seeds.", "",
        "Paired speedup definition: " + PAIRED + ".", "",
        "## Main Table", "",
        "| Route | valid PG/sample-first (%) | valid raw/source/peak/interface (K) | test PG/sample-first (%) | test raw/source/peak/interface (K) | Fresh med/p95 (s) | Q2 (sample/s) | Fresh/Q2 paired speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['route']} | {fmt(row['valid_point_global_rmse_pct'])}/{fmt(row['valid_sample_first_cv_relative_rmse_pct'])} | "
            f"{fmt(row['valid_raw_cv_rmse_K'])}/{fmt(row['valid_source_rmse_K'])}/{fmt(row['valid_peak_rmse_K'])}/{fmt(row['valid_interface_rmse_K'])} | "
            f"{fmt(row['test_point_global_rmse_pct'])}/{fmt(row['test_sample_first_cv_relative_rmse_pct'])} | "
            f"{fmt(row['test_raw_cv_rmse_K'])}/{fmt(row['test_source_rmse_K'])}/{fmt(row['test_peak_rmse_K'])}/{fmt(row['test_interface_rmse_K'])} | "
            f"{fmt(row['wsl2_fresh_median_s'])}/{fmt(row['wsl2_fresh_p95_s'])} | "
            f"{fmt(row['wsl2_q2_samples_s'])} | {fmt(row['wsl2_fresh_paired_speedup_vs_fvm'], 3)}×/{fmt(row['wsl2_q2_paired_speedup_vs_fvm'], 3)}× |"
        )
    lines += [
        "", "FVM is the reference solution; surrogate-error cells are N/A. The E16384 test row is a one-time corrected confirmatory holdout result obtained after route freeze and was not used for selection. Relative to frozen valid32, test PG is +{:.4f} percentage points, raw CV RMSE is +{:.4f} K, source RMSE is +{:.4f} K, peak RMSE is +{:.4f} K, and interface RMSE is {:+.4f} K; the peak tail increase is retained rather than hidden.".format(
            test_accuracy["point_global_true_rms_relative_rmse_pct"] - historical_accuracy["E16384_reconstruction"]["point_global_true_rms_relative_rmse_pct"],
            test_accuracy["raw_cv_weighted_rmse_K"] - historical_accuracy["E16384_reconstruction"]["raw_cv_weighted_rmse_K"],
            test_accuracy["source_rmse_K"] - historical_accuracy["E16384_reconstruction"]["source_rmse_K"],
            test_accuracy["peak_rmse_K"] - historical_accuracy["E16384_reconstruction"]["peak_rmse_K"],
            test_accuracy["interface_drop_rmse_K"] - historical_accuracy["E16384_reconstruction"]["interface_drop_rmse_K"],
        ), "",
        "## Supplementary lifecycle table", "",
        "The complete 10-row machine/route lifecycle table is frozen in `v6_p1i_p6a_supplementary_lifecycle_table.csv`; it retains cold, fresh, cache-hot, resident, Q2, B16-to-B32, RAM/VRAM, and three-lifecycle median/min/max fields.", "",
        "## Replication table", "",
        "The 5-route WSL2-versus-devbox table is frozen in `v6_p1i_p6a_replication_table.csv`. Devbox does not replace the WSL2 primary benchmark.", "",
        "## Stage decomposition table", "",
        "The 56-row decomposition is frozen in `v6_p1i_p6a_stage_decomposition_table.csv`. " + NN_TAIL + ". No tail row was deleted or winsorized.", "",
        "## Claim/evidence mapping", "",
        "| ID | Claim | Status | Boundary |", "|---|---|---|---|",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim_id']} | {row['claim']} | {row['status']} | {row['boundary']} |")
    lines += ["", "`sealed IID` remains unopened because its labels have not been generated. Publication evidence completeness: **GO**."]
    md_path.write_text("\n".join(lines) + "\n")

    manifest_paths = [
        full_source_path, test_path, closeout_path, md_path, protocol_path,
        *outputs.values(),
        CFG / "v6_p1i_publication_evidence_summary.json",
        CFG / "v6_p1i_publication_evidence_sha256.txt",
        ROOT / "scripts/evaluate_heat3d_v6_p1i_e16384_test_confirmatory.py",
        ROOT / "scripts/closeout_heat3d_v6_p1i_p6a_publication_archival.py",
        ROOT / "scripts/check_heat3d_v6_p1i_p6a_publication_archival.py",
    ]
    manifest_path = CFG / "v6_p1i_p6a_publication_evidence_sha256.txt"
    manifest_path.write_text("".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in manifest_paths
    ))
    print(json.dumps({"status": "passed", "publication_evidence_completeness": "GO", "outputs": [str(path.relative_to(ROOT)) for path in [*outputs.values(), closeout_path, md_path, manifest_path]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
