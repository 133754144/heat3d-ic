#!/usr/bin/env python3
"""Freeze Gate 6R results and assess whether the V5 phase can close."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/heat3d_v5/gate6r_closeout"
EVALUATORS = {
    "V38": OUT / "evaluator/V38_gate6r_cpu_replay.json",
    "V45": OUT / "evaluator/V45_gate6r_cpu_replay.json",
    "V46": OUT / "evaluator/V46_gate6r_cpu_replay.json",
}
GATE6Q = ROOT / "configs/heat3d_v5/gate6q/gate6q_final_closeout.json"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6r_training_registry.csv"
CONTRACT = ROOT / "configs/heat3d_v5/v5_clean_first_performance_contract.json"
JSON_OUT = OUT / "v5_final_phase_assessment.json"
METRICS_CSV = OUT / "v5_final_checkpoint_metrics.csv"
PAIRED_CSV = OUT / "gate6r_paired_samples.csv"
MD_OUT = ROOT / "docs/v5_final_phase_assessment.md"
CHECKPOINTS = ("point_global_best", "sample_first_best", "legacy_best", "final")
SUMMARY_FIELDS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "legacy_normalized_valid_base_mse",
    "shape_cv_rmse",
    "scale_log_rmse",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
)
COMPARISONS = (
    ("V45_minus_V38", "V38", "V45"),
    ("V46_minus_V38", "V38", "V46"),
    ("V46_minus_V45", "V45", "V46"),
)
SEED = 20260722
BOOTSTRAP_REPLICATES = 20_000


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_model(label: str, payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "completed_valid_iid_only":
        raise ValueError(f"{label}: evaluator is incomplete")
    scope = payload.get("scope", {})
    if scope.get("evaluation_roles") != ["valid_iid"]:
        raise ValueError(f"{label}: non-valid evaluation role")
    if scope.get("forbidden_roles_accessed") or any(
        scope.get(key) for key in ("test_accessed", "hard_accessed", "sealed_iid_accessed")
    ):
        raise ValueError(f"{label}: forbidden role accessed")
    if set(payload.get("metrics", {})) != set(CHECKPOINTS):
        raise ValueError(f"{label}: incomplete checkpoint metrics")
    completion = payload.get("training_completion", {})
    if completion.get("final_epoch") != 600 or completion.get("epoch_history_count") != 600:
        raise ValueError(f"{label}: incomplete e600")
    if not completion.get("grad_finite"):
        raise ValueError(f"{label}: non-finite gradient record")
    for checkpoint in CHECKPOINTS:
        summary = payload["metrics"][checkpoint]["summary"]
        if any(not math.isfinite(float(summary[field])) for field in SUMMARY_FIELDS):
            raise ValueError(f"{label}/{checkpoint}: non-finite metric")
        if not payload["reload_audit"][checkpoint].get("passed"):
            raise ValueError(f"{label}/{checkpoint}: reload audit failed")


def _samples(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["metrics"]["point_global_best"]["per_sample"]
    result = {str(row["sample_id"]): dict(row) for row in rows}
    if len(result) != 128:
        raise ValueError("expected 128 paired valid_iid samples")
    return result


def _paired(
    name: str,
    baseline_label: str,
    candidate_label: str,
    models: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = _samples(models[baseline_label])
    candidate = _samples(models[candidate_label])
    if set(baseline) != set(candidate):
        raise ValueError(f"{name}: sample IDs differ")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(baseline):
        left, right = baseline[sample_id], candidate[sample_id]
        rows.append(
            {
                "comparison": name,
                "baseline": baseline_label,
                "candidate": candidate_label,
                "sample_id": sample_id,
                "deltaT_quartile": left["deltaT_quartile"],
                "true_scale_cv_rms_K": float(left["true_scale_cv_rms_K"]),
                "point_true_squared_sum": float(left["point_true_squared_sum"]),
                "baseline_point_sse_K2": float(left["point_error_squared_sum"]),
                "candidate_point_sse_K2": float(right["point_error_squared_sum"]),
                "delta_point_sse_K2": float(right["point_error_squared_sum"])
                - float(left["point_error_squared_sum"]),
                "baseline_sample_relative_pct": 100.0 * float(left["sample_cv_relative_rmse"]),
                "candidate_sample_relative_pct": 100.0 * float(right["sample_cv_relative_rmse"]),
                "delta_sample_relative_pct": 100.0
                * (float(right["sample_cv_relative_rmse"]) - float(left["sample_cv_relative_rmse"])),
            }
        )
    sse_left = np.asarray([row["baseline_point_sse_K2"] for row in rows])
    sse_right = np.asarray([row["candidate_point_sse_K2"] for row in rows])
    true = np.asarray([row["point_true_squared_sum"] for row in rows])
    sample_delta = np.asarray([row["delta_sample_relative_pct"] for row in rows])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    pg_delta = 100.0 * (
        np.sqrt(np.sum(sse_right[indices], axis=1) / np.sum(true[indices], axis=1))
        - np.sqrt(np.sum(sse_left[indices], axis=1) / np.sum(true[indices], axis=1))
    )
    sf_delta = np.mean(sample_delta[indices], axis=1)
    left_summary = models[baseline_label]["metrics"]["point_global_best"]["summary"]
    right_summary = models[candidate_label]["metrics"]["point_global_best"]["summary"]
    quartiles = {
        quartile: float(sum(row["delta_point_sse_K2"] for row in rows if row["deltaT_quartile"] == quartile))
        for quartile in ("Q1", "Q2", "Q3", "Q4")
    }
    summary = {
        "comparison": name,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "point_global_delta_pp": float(right_summary["point_global_relative_rmse_pct"])
        - float(left_summary["point_global_relative_rmse_pct"]),
        "sample_first_delta_pp": float(right_summary["sample_first_cv_relative_rmse_pct"])
        - float(left_summary["sample_first_cv_relative_rmse_pct"]),
        "raw_cv_delta_K": float(right_summary["raw_cv_weighted_rmse_K"])
        - float(left_summary["raw_cv_weighted_rmse_K"]),
        "point_sse_delta_K2": float(np.sum(sse_right - sse_left)),
        "point_sse_win_rate": float(np.mean(sse_right < sse_left)),
        "sample_relative_win_rate": float(np.mean(sample_delta < 0.0)),
        "median_sample_relative_delta_pp": float(np.median(sample_delta)),
        "point_global_delta_bootstrap_95_ci_pp": [float(v) for v in np.quantile(pg_delta, [0.025, 0.975])],
        "sample_first_delta_bootstrap_95_ci_pp": [float(v) for v in np.quantile(sf_delta, [0.025, 0.975])],
        "quartile_point_sse_delta_K2": quartiles,
    }
    return summary, rows


def _merge_audit() -> dict[str, Any]:
    main = _git("rev-parse", "origin/main")
    head = _git("rev-parse", "HEAD")
    base = _git("merge-base", "origin/main", "HEAD")
    counts = _git("rev-list", "--left-right", "--count", "origin/main...HEAD").split()
    names = _git("diff", "--name-only", "origin/main...HEAD").splitlines()
    insertions = deletions = 0
    for line in _git("diff", "--numstat", "origin/main...HEAD").splitlines():
        added, removed, _path = line.split("\t", 2)
        if added.isdigit():
            insertions += int(added)
        if removed.isdigit():
            deletions += int(removed)
    return {
        "main_commit": main,
        "research_v5_commit_at_assessment": head,
        "merge_base": base,
        "main_is_ancestor": base == main,
        "ahead_commits": int(counts[1]),
        "behind_commits": int(counts[0]),
        "changed_files": len(names),
        "insertions": insertions,
        "deletions": deletions,
        "technical_fast_forward_possible": base == main and int(counts[0]) == 0,
        "direct_fast_forward_recommended": False,
        "recommendation": "Do not fast-forward main with the full research branch; integrate reviewed runner/model/metric changes separately from generated experiment evidence.",
    }


def _update_registry() -> None:
    csv.field_size_limit(sys.maxsize)
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    hosts = {
        "V4P5_45_gate6r_deepsets_only_e600": "wsl2",
        "V4P5_46_gate6r_objective_deepsets_e600": "devbox",
    }
    for row in rows:
        if row.get("config_id") not in hosts:
            continue
        row["plan_status"] = "completed"
        row["execution_status"] = "completed_e600"
        row["evaluation_status"] = "completed_valid_iid_four_checkpoint"
        row["training_started"] = "true"
        row["notes"] = (
            f"Gate 6R e600 completed on {hosts[row['config_id']]}; frozen four-checkpoint valid_iid closeout complete; "
            "test/hard/sealed not accessed; point-global threshold failed."
        )
    with REGISTRY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V5 final phase assessment",
        "",
        "Scope: frozen true-RMS evaluation on `valid_iid`; train was read only to verify frozen normalization/context. No test/hard/sealed access and no training.",
        "Metrics use checkpoint-bound saved prediction NPZ artifacts. Training-time reload audits passed; documented direct CPU cross-backend replay drift is retained only as a diagnostic.",
        "",
        "## Point-global-best ranking",
        "",
        "| rank | model | epoch | point-global % | sample-first % | raw CV K | shape | scale log | <20% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for rank, row in enumerate(payload["formal_ranking"], 1):
        lines.append(
            f"| {rank} | {row['model']} | {row['epoch']} | {row['point_global_relative_rmse_pct']:.6f} | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f} | {row['raw_cv_weighted_rmse_K']:.6f} | "
            f"{row['shape_cv_rmse']:.6f} | {row['scale_log_rmse']:.6f} | {'yes' if row['valid_threshold_pass'] else 'no'} |"
        )
    lines.extend(["", "## Gate 6R paired attribution", "", "Negative deltas mean improvement.", "", "| comparison | point-global pp | sample-first pp | raw CV K | point SSE K2 | point win | Q4 SSE |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["gate6r_comparisons"]:
        lines.append(
            f"| {row['comparison']} | {row['point_global_delta_pp']:.6f} | {row['sample_first_delta_pp']:.6f} | "
            f"{row['raw_cv_delta_K']:.6f} | {row['point_sse_delta_K2']:.6f} | {row['point_sse_win_rate']:.4f} | "
            f"{row['quartile_point_sse_delta_K2']['Q4']:.6f} |"
        )
    verdict = payload["phase_closeout_assessment"]
    merge = payload["merge_assessment"]
    lines.extend(
        [
            "",
            "## Closeout verdict",
            "",
            f"- Scientific success: **no**. Best valid point-global result is {verdict['best_valid_point_global_relative_rmse_pct']:.6f}%, above the frozen <20% threshold; required valid+test success therefore cannot be established.",
            "- Phase closure: **yes, as a completed negative/inconclusive research phase**. V45/V46 add no point-global improvement over the V38 lineage; no new V5 training is recommended.",
            "- The sealed/test/hard roles remain unopened in this closeout. This is not a generalization claim.",
            "",
            "## Main merge assessment",
            "",
            f"- Git topology is technically fast-forwardable: main `{merge['main_commit'][:12]}` is the merge base and V5 is ahead by {merge['ahead_commits']} commits with no main-only commits.",
            f"- The branch changes {merge['changed_files']} files (+{merge['insertions']}/-{merge['deletions']}).",
            "- Recommendation: **do not fast-forward the full research branch into main as-is**. Create a reviewed integration branch/PR containing reusable runner, metric, architecture, and tests; keep generated YAML/registries/large research evidence as an archival V5 ref or separate research-history merge.",
            "",
            "No merge, tag, test/hard/sealed evaluation, or training was executed by this assessment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate6q = _read(GATE6Q)
    models: dict[str, dict[str, Any]] = {
        label: gate6q["models"][label] for label in ("V38", "V42", "V43", "V44")
    }
    current = {label: _read(path) for label, path in EVALUATORS.items()}
    models["V38"] = current["V38"]
    models.update({label: current[label] for label in ("V45", "V46")})
    for label, model in models.items():
        _validate_model(label, model)
    split_hashes = {model["split"]["valid_iid_ids_sha256"] for model in models.values()}
    metric_hashes = {model["metric_source"]["sha256"] for model in models.values()}
    if len(split_hashes) != 1 or len(metric_hashes) != 1:
        raise ValueError("cross-model split or metric formula mismatch")
    prior_v38 = gate6q["models"]["V38"]
    max_diff = max(
        abs(float(prior_v38["metrics"][checkpoint]["summary"][field]) - float(models["V38"]["metrics"][checkpoint]["summary"][field]))
        for checkpoint in CHECKPOINTS
        for field in SUMMARY_FIELDS
    )
    comparisons: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for name, baseline, candidate in COMPARISONS:
        summary, rows = _paired(name, baseline, candidate, models)
        comparisons.append(summary)
        paired_rows.extend(rows)
    metric_rows: list[dict[str, Any]] = []
    for label, model in models.items():
        for checkpoint in CHECKPOINTS:
            row = {
                "model": label,
                "config_id": model["config_id"],
                "checkpoint": checkpoint,
                "epoch": model["checkpoint_metadata"][checkpoint]["epoch"],
                "checkpoint_sha256": model["checkpoint_metadata"][checkpoint]["sha256"],
                "training_commit": model["training_commit"],
                "source_host": model["source_host"],
            }
            row.update({field: model["metrics"][checkpoint]["summary"][field] for field in SUMMARY_FIELDS})
            metric_rows.append(row)
    ranking = []
    for label, model in models.items():
        summary = model["metrics"]["point_global_best"]["summary"]
        ranking.append(
            {
                "model": label,
                "config_id": model["config_id"],
                "epoch": model["checkpoint_metadata"]["point_global_best"]["epoch"],
                **{field: float(summary[field]) for field in SUMMARY_FIELDS},
                "valid_threshold_pass": float(summary["point_global_relative_rmse_pct"]) < 20.0,
            }
        )
    ranking.sort(key=lambda row: row["point_global_relative_rmse_pct"])
    contract = _read(CONTRACT)
    merge = _merge_audit()
    payload = {
        "schema_version": "heat3d_v5_final_phase_assessment_v1",
        "status": "ready_to_close_threshold_unmet",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "research/v5",
        "head": _git("rev-parse", "HEAD"),
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "evaluation_roles": ["valid_iid"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "model_or_checkpoint_modified": False,
        },
        "contract_thresholds": contract["clean_success_thresholds"],
        "metric_source_sha256": next(iter(metric_hashes)),
        "valid_iid_ids_sha256": next(iter(split_hashes)),
        "gate6r_evaluator_commit": current["V38"]["evaluator_commit"],
        "gate6q_evaluator_commit": gate6q["evaluator_commit"],
        "evaluator_change_scope": "Gate 6R commit adds only V45/V46 config IDs to the evaluator allowlist.",
        "v38_cross_evaluator_max_abs_summary_diff": max_diff,
        "evaluation_caveat": (
            "Frozen metrics use checkpoint-bound saved prediction NPZ artifacts. "
            "Training-time checkpoint reload audits passed, while direct CPU parameter replay "
            "retains the documented cross-backend drift and is not used to replace metric fields."
        ),
        "input_artifacts": {
            str(path.relative_to(ROOT)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (*EVALUATORS.values(), GATE6Q, CONTRACT)
        },
        "formal_ranking": ranking,
        "gate6r_comparisons": comparisons,
        "phase_closeout_assessment": {
            "scientific_success": False,
            "valid_threshold_pass": False,
            "test_threshold_evaluated": False,
            "valid_and_test_success_established": False,
            "best_model": ranking[0]["model"],
            "best_valid_point_global_relative_rmse_pct": ranking[0]["point_global_relative_rmse_pct"],
            "can_close_v5": True,
            "closure_class": "completed_research_phase_threshold_unmet",
            "additional_v5_training_recommended": False,
            "recommended_close_action": "freeze research/v5 as threshold-unmet research history after review; do not open sealed/test/hard roles merely to seek a passing result",
        },
        "merge_assessment": merge,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    with PAIRED_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_rows)
    MD_OUT.write_text(_markdown(payload), encoding="utf-8")
    _update_registry()
    print(json.dumps({"status": payload["status"], "best": ranking[0], "merge": merge}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
