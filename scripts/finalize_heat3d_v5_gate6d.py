#!/usr/bin/env python3
"""Freeze Gate 6D registry, closeout, and an ungenerated sealed-IID contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6c_scratch_loss_registry.csv"
GATE6D = ROOT / "configs/heat3d_v5/gate6d"
DOCS = ROOT / "docs"
EVALUATOR_COMMIT = "639872abcb0f7afd3b6c2d319a7d395bde75c9a4"
TRAINING_COMMIT = "62bee1f3591568bdd97f17ded0128e4f4bb8569c"
REGISTRY_SOURCE_COMMIT = "2cb20af5be8f9e8f2d6d2e409baf4305ffd458bf"
GENERATOR_COMMIT = "ac018088c337689fdad828d3bb2c8296c77edb16"
GENERATOR_SHA256 = "a77afe7d2fde5ce43f4ac642575cfac4b99bd62ccaaf2e7756ba740871513e19"
PARAMETER_REGISTRY_SHA256 = "121cd5e54d54f37ff51cb1347d9813b2ac3363fa6706eee37b82331a5f562b"
SEALED_SEED = 2026071501
SEALED_COUNT = 128

RUNS = {
    "V4P5_11_gate6c_scratch_l1_tail_balanced": {
        "host": "devbox", "best_epoch": 346,
        "best_sha256": "6f398b2a971c2a40e7c71dabaa1b7012a36044862ea8ca481f02f9a54083e817",
        "final_sha256": "dbb5b3f8b93b606bcf180afab1b0ab4fa45694ef823fd9b61590af4738989755",
        "evaluation": "configs/heat3d_v5/gate6d/V4P5_11_frozen_gate5_evaluation.json",
    },
    "V4P5_12_gate6c_scratch_l2_shape_balanced": {
        "host": "wsl2", "best_epoch": 353,
        "best_sha256": "7d144eeee9f111d933d40f1975dfd3ea3c6c4af3945a4e6ab7660fa886f99091",
        "final_sha256": "625e67906ccde07b5c7b2959842cde956d3830813cd706804c2bd758184afc46",
        "evaluation": "configs/heat3d_v5/gate6d/V4P5_12_frozen_gate5_evaluation.json",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_registry() -> None:
    csv.field_size_limit(sys.maxsize)
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)
    lifecycle = [
        "plan_status", "execution_status", "evaluation_status", "threshold_status",
        "training_host", "training_commit", "best_epoch", "final_epoch",
        "best_checkpoint_sha256", "final_checkpoint_sha256", "evaluator_commit",
        "authoritative_evaluation_json", "test_role_status", "hard_role_status",
    ]
    fields = []
    for name in old_fields:
        if name == "status":
            fields.extend(lifecycle)
        else:
            fields.append(name)
    for row in rows:
        meta = RUNS[row["config_id"]]
        payload = json.loads((ROOT / meta["evaluation"]).read_text(encoding="utf-8"))
        best, final = payload["reports"]["best"], payload["reports"]["final"]
        valid, test = best["valid_iid"], best["test_iid"]
        row.pop("status", None)
        row.update({
            "plan_status": "frozen",
            "execution_status": "completed_e600",
            "evaluation_status": "completed_evaluated",
            "threshold_status": "failed",
            "training_host": meta["host"],
            "training_commit": TRAINING_COMMIT,
            "best_epoch": str(meta["best_epoch"]),
            "final_epoch": "600",
            "best_checkpoint_sha256": meta["best_sha256"],
            "final_checkpoint_sha256": meta["final_sha256"],
            "evaluator_commit": EVALUATOR_COMMIT,
            "authoritative_evaluation_json": meta["evaluation"],
            "test_role_status": "legacy_observed_test",
            "hard_role_status": "observed_report_only",
            "result_v5_status": "completed",
            "result_v5_source": meta["host"],
            "result_v5_commit": TRAINING_COMMIT,
            "result_v5_metrics_json": json.dumps(payload, separators=(",", ":"), sort_keys=True),
            "result_v5_required_metrics_complete": "true",
            "result_v5_missing_metrics": "",
            "result_v5_primary_checkpoint": "mse_best",
            "result_v5_primary_epoch": str(meta["best_epoch"]),
            "result_v5_legacy_checkpoint": "mse_best",
            "result_v5_legacy_epoch": str(meta["best_epoch"]),
            "result_v5_primary_valid_point_global_relative_rmse_pct": f"{valid['point_global_relative_rmse_pct']:.12g}",
            "result_v5_primary_valid_sample_first_cv_relative_rmse_pct": f"{valid['sample_first_cv_relative_rmse_pct']:.12g}",
            "result_v5_primary_valid_raw_cv_weighted_rmse_K": f"{valid['raw_cv_weighted_rmse_K']:.12g}",
            "result_v5_primary_test_point_global_relative_rmse_pct": f"{test['point_global_relative_rmse_pct']:.12g}",
            "result_v5_primary_test_sample_first_cv_relative_rmse_pct": f"{test['sample_first_cv_relative_rmse_pct']:.12g}",
            "result_v5_primary_test_raw_cv_weighted_rmse_K": f"{test['raw_cv_weighted_rmse_K']:.12g}",
            "result_v5_legacy_valid_base_mse": f"{valid['legacy_normalized_valid_base_mse']:.12g}",
            "result_v5_legacy_test_point_global_relative_rmse_pct": f"{test['point_global_relative_rmse_pct']:.12g}",
            "result_v5_threshold_pass": "fail",
            "result_v5_notes": (
                "Gate 6D frozen evaluator is authoritative; test was observed only after "
                "training; hard roles are observed report-only and are barred from later selection/tuning."
            ),
        })
        if final["valid_iid"]["point_global_relative_rmse_pct"] <= 0:
            raise ValueError("non-positive final metric")
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _freeze_sealed_contract() -> dict[str, Any]:
    ids = [f"sealed_iid_{SEALED_SEED}_{index:04d}" for index in range(SEALED_COUNT)]
    split_map = {
        "schema_version": "heat3d_v5_gate6d_sealed_split_v1",
        "role": "sealed_iid_test",
        "sample_count": SEALED_COUNT,
        "sample_ids": ids,
    }
    manifest = {
        "schema_version": "heat3d_v5_gate6d_sealed_manifest_plan_v1",
        "status": "planned_not_generated",
        "dataset_id": f"heat3d_v5_gate6d_sealed_iid_seed{SEALED_SEED}",
        "sample_count": SEALED_COUNT,
        "sample_ids": ids,
        "labels_generated": False,
        "model_inference_run": False,
    }
    provenance = {
        "schema_version": "heat3d_v5_gate6d_sealed_provenance_v1",
        "generator_commit": GENERATOR_COMMIT,
        "generator_path": "scripts/generate_heat3d_v4_p3c_smoke16.py",
        "generator_sha256": GENERATOR_SHA256,
        "parameter_registry": "configs/heat3d_v4/p3c_parameter_registry.json",
        "parameter_registry_sha256": PARAMETER_REGISTRY_SHA256,
        "seed": SEALED_SEED,
        "sample_count": SEALED_COUNT,
        "accepted_qc_classes": ["clean_keep"],
        "max_candidates": 1024,
        "distribution_contract": "same physical generator and clean_keep acceptance policy as current clean P5",
        "seed_selection_policy": "chosen once before generation; never changed after observing labels or sample statistics",
    }
    contract = {
        "schema_version": "heat3d_v5_gate6d_sealed_iid_contract_v1",
        "status": "frozen_not_generated",
        "seed": SEALED_SEED,
        "sample_count": SEALED_COUNT,
        "role": "sealed_iid_test",
        "data_dir": f"data/heat3d_v5_gate6d_sealed_iid_seed{SEALED_SEED}",
        "audit_output_dir": f"output/heat3d_v5_gate6d_sealed_iid_seed{SEALED_SEED}",
        "generation_command": (
            "python scripts/generate_heat3d_v5_gate6d_sealed_iid.py "
            "--contract configs/heat3d_v5/gate6d/sealed_iid_contract.json"
        ),
        "first_open_condition": "candidate and complete training plan are fully frozen",
        "selection_or_tuning_use_allowed": False,
        "label_statistics_may_change_seed": False,
        "model_inference_run": False,
        "training_started": False,
        "artifacts": {
            "planned_manifest": "configs/heat3d_v5/gate6d/sealed_iid_manifest.json",
            "planned_provenance": "configs/heat3d_v5/gate6d/sealed_iid_provenance.json",
            "planned_split_map": "configs/heat3d_v5/gate6d/sealed_iid_split_map.json",
        },
    }
    paths = {
        "sealed_iid_contract.json": contract,
        "sealed_iid_manifest.json": manifest,
        "sealed_iid_provenance.json": provenance,
        "sealed_iid_split_map.json": split_map,
    }
    for name, payload in paths.items():
        _write_json(GATE6D / name, payload)
    hashes = {name: _sha(GATE6D / name) for name in paths}
    _write_json(GATE6D / "sealed_iid_contract_sha256.json", {
        "schema_version": "heat3d_v5_gate6d_sealed_contract_sha256_v1",
        "status": "contract_artifacts_only_dataset_not_generated",
        "sha256": hashes,
    })
    return contract


def _write_closeouts(sealed: dict[str, Any]) -> None:
    run_rows = []
    for config_id, meta in RUNS.items():
        payload = json.loads((ROOT / meta["evaluation"]).read_text(encoding="utf-8"))
        valid = payload["reports"]["best"]["valid_iid"]
        run_rows.append({
            "config_id": config_id, **meta,
            "training_commit": TRAINING_COMMIT,
            "evaluator_commit": EVALUATOR_COMMIT,
            "registry_source_commit": REGISTRY_SOURCE_COMMIT,
            "final_epoch": 600,
            "threshold_status": "failed",
            "best_valid_point_global_relative_rmse_pct": valid["point_global_relative_rmse_pct"],
        })
    gate6c = {
        "schema_version": "heat3d_v5_gate6c_closeout_v1",
        "status": "closed_evaluated_threshold_failed",
        "runs": run_rows,
        "role_policy": {
            "selection_roles": ["valid_iid"],
            "test_iid": "legacy_observed_test",
            "hard_roles": "observed_report_only",
            "test_hard_participated_in_training_or_checkpoint_selection": False,
            "test_hard_allowed_for_future_selection_or_tuning": False,
        },
    }
    _write_json(ROOT / "configs/heat3d_v5/v5_gate6c_closeout.json", gate6c)
    paired_payload = json.loads(
        (GATE6D / "n3_l2_valid_paired.json").read_text(encoding="utf-8")
    )
    coverage = json.loads((GATE6D / "global_context_coverage.json").read_text(encoding="utf-8"))
    closeout = {
        "schema_version": "heat3d_v5_gate6d_preflight_closeout_v1",
        "status": "completed_no_training",
        "authoritative_evaluator_commit": EVALUATOR_COMMIT,
        "collector_equivalence": "not_equivalent_frozen_evaluator_authoritative",
        "gate6c_runs": run_rows,
        "paired_analysis": paired_payload["aggregate"],
        "true_delta_point_sse_attribution": paired_payload["true_delta_point_sse_attribution"],
        "paired_inference": paired_payload["paired_inference"],
        "coverage_summary": {
            "fit_roles": coverage["fit_roles"], "query_roles": coverage["query_roles"],
            "feature_count": coverage["feature_count"], "distance_summary": coverage["distance_summary"],
            "correlations": coverage["correlations"],
        },
        "sealed_iid": sealed,
        "training_started": False,
        "multi_seed_started": False,
        "new_loss_config_created": False,
    }
    _write_json(GATE6D / "gate6d_preflight_closeout.json", closeout)
    lines = [
        "# Gate 6C closeout", "",
        "Scratch-L1/L2 均已完成 e600，并在训练与 checkpoint 选择完成后打开 test/hard。",
        "`test_iid` 记为 `legacy_observed_test`；hard roles 记为 `observed_report_only`，后续不得用于候选选择或调参。",
        "", "| candidate | host | best epoch | valid point-global | threshold |",
        "|---|---|---:|---:|---|",
    ]
    for row in run_rows:
        lines.append(f"| {row['config_id']} | {row['host']} | {row['best_epoch']} | {row['best_valid_point_global_relative_rmse_pct']:.6f}% | failed |")
    (DOCS / "v5_gate6c_closeout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DOCS / "v5_gate6d_preflight_closeout.md").write_text(
        "# Gate 6D preflight closeout\n\n"
        f"冻结 evaluator `{EVALUATOR_COMMIT}` 与 collector 不等价，故冻结 evaluator 结果为权威。\n\n"
        "N3-L2 成对归因只使用 valid_iid；24D coverage 只用 train 拟合、valid 查询。"
        "sample-relative 改善不集中于少数 top-10；但 point-global SSE 的 true-DeltaT Q1-Q3 总体退化，Q4 提供全部净改善。\n\n"
        f"sealed IID seed 固定为 `{SEALED_SEED}`，当前仅冻结可执行合同，未生成标签、未推理。"
        "首次开启条件是候选与完整训练方案完全冻结。\n\n"
        "本轮 training_started=false，未启动 multi-seed，未新增 loss 配置。\n",
        encoding="utf-8",
    )


def main() -> int:
    _update_registry()
    sealed = _freeze_sealed_contract()
    _write_closeouts(sealed)
    print(json.dumps({"status": "passed", "training_started": False, "sealed_seed": SEALED_SEED}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
