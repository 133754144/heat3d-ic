#!/usr/bin/env python3
"""Freeze V5 closeout artifacts and update the V42 result registry row."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


STATUS = "completed_research_phase_threshold_unmet"
TAG = "v5-final-threshold-unmet"
V42 = "V4P5_42_gate6q_objective_only_e600"
THRESHOLD = 20.0
INTEGRATION_COMMIT = "3a85d53ac6169f5ab17603c6cf2d146ec07132f3"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _update_registry(path: Path, test: Mapping[str, Any], now: str) -> None:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    matches = [row for row in rows if row.get("config_id") == V42]
    if len(matches) != 1:
        raise RuntimeError(f"expected one V42 registry row, found {len(matches)}")
    summary = test["test_iid"]["summary"]
    row = matches[0]
    row.update(
        {
            "evaluation_status": "completed_valid_iid_and_final_test_iid",
            "notes": (
                "Unweighted point SSE uses one fixed train-global target-energy-per-point denominator; "
                "scale squared weights are train-fitted and clipped; scalar loss weights unchanged.; "
                "Gate 6Q final valid-only CPU closeout complete; final frozen test_iid opened only after "
                "checkpoint/model/phase decisions were fixed; hard/sealed not accessed"
            ),
            "result_v5_status": "completed_valid_and_final_test_iid",
            "result_v5_updated_at": now,
            "result_v5_primary_test_point_global_relative_rmse_pct": (
                f"{summary['point_global_relative_rmse_pct']:.12g}"
            ),
            "result_v5_primary_test_sample_first_cv_relative_rmse_pct": (
                f"{summary['sample_first_cv_relative_rmse_pct']:.12g}"
            ),
            "result_v5_primary_test_raw_cv_weighted_rmse_K": (
                f"{summary['raw_cv_weighted_rmse_K']:.12g}"
            ),
            "result_v5_threshold_pass": "fail",
            "result_v5_notes": (
                "valid_iid four-checkpoint metrics complete; final test_iid evaluated once at frozen "
                "point-global checkpoint e257; no checkpoint reselection or tuning; hard/sealed not accessed; "
                "phase closed threshold unmet"
            ),
        }
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _q4_summary(q4: Mapping[str, Any]) -> dict[str, Any]:
    decomposition = q4["shape_scale_cross_decomposition"]
    q4_shares = {
        model: values["Q4"]["point_error_squared_sum"] / values["all"]["point_error_squared_sum"]
        for model, values in decomposition.items()
    }
    overlaps = q4["difficult_sample_overlap"].values()
    top5 = [row["top5"]["intersection_count"] for row in overlaps]
    top10 = [row["top10"]["intersection_count"] for row in overlaps]
    nn_spearman = {
        model: values["train_nn_distance_24d"]["spearman"]
        for model, values in q4["physical_feature_error_correlations"].items()
    }
    inv_k_spearman = {
        model: values["q_weighted_inverse_kz_mK_W"]["spearman"]
        for model, values in q4["physical_feature_error_correlations"].items()
    }
    stack_spearman = {
        model: max(
            abs(values["stack_unique_conductivity_triplets"]["spearman"]),
            abs(values["stack_conductivity_transition_fraction"]["spearman"]),
        )
        for model, values in q4["physical_feature_error_correlations"].items()
    }
    top_h_spearman = {
        model: values["top_h_W_m2K"]["spearman"]
        for model, values in q4["physical_feature_error_correlations"].items()
    }
    return {
        "top5_pairwise_intersection_range": [min(top5), max(top5)],
        "top10_pairwise_intersection_range": [min(top10), max(top10)],
        "q4_point_sse_share": q4_shares,
        "train_nn_distance_error_spearman": nn_spearman,
        "q_weighted_inverse_kz_error_spearman": inv_k_spearman,
        "stack_sparsity_error_max_abs_spearman": stack_spearman,
        "top_h_error_spearman": top_h_spearman,
        "interpretation": (
            "The identical top-5 and high top-10 overlap, Q4-dominated SSE, and repeatable positive "
            "q-weighted inverse-kz association indicate an energy/path-resistance failure shared across "
            "architectures. Weak 24D train-nearest-neighbor, top-h, and stack-count correlations do not "
            "support generic coverage distance or simple categorical stack sparsity as the primary cause."
        ),
    }


def main() -> int:
    args = _args()
    assessment = _read(args.assessment)
    q4 = _read(args.q4)
    test = _read(args.test)
    if q4["status"] != "completed_train_valid_read_only":
        raise RuntimeError("Q4 audit incomplete")
    if test["status"] != "completed_frozen_checkpoint_test_and_timing":
        raise RuntimeError("V42 final test/timing incomplete")
    valid = test["frozen_valid_iid"]["summary"]
    test_summary = test["test_iid"]["summary"]
    if int(test["binding"]["checkpoint_epoch"]) != 257:
        raise RuntimeError("V42 checkpoint drifted from frozen e257")
    if valid["point_global_relative_rmse_pct"] < THRESHOLD or test_summary["point_global_relative_rmse_pct"] < THRESHOLD:
        raise RuntimeError("threshold status no longer matches requested closeout")
    if test["scope"]["hard_accessed"] or test["scope"]["sealed_iid_accessed"]:
        raise RuntimeError("forbidden role accessed")
    now = datetime.now(timezone.utc).isoformat()
    _update_registry(args.registry, test, now)
    q4_summary = _q4_summary(q4)
    artifact_paths = [
        args.q4,
        args.q4.with_name("v5_final_q4_samples.csv"),
        args.q4.with_name("v5_final_q4_root_audit.md"),
        args.test,
        args.test.with_name("v42_e257_test_iid_per_sample.csv"),
        args.test.with_name("v42_e257_final_test_timing.md"),
        args.assessment,
        args.registry,
    ]
    hypothesis = (
        "V6 should test a stack-aware source-to-sink thermal-resistance representation that preserves "
        "layer/path structure for the interaction of source power, through-plane inverse conductivity, "
        "vertical distance, and terminal BCs; it should not begin with more generic capacity, flat XY "
        "features, or undirected coverage expansion."
    )
    manifest = {
        "schema_version": "heat3d_v5_phase_closeout_manifest_v1",
        "status": STATUS,
        "generated_at_utc": now,
        "branch": "research/v5",
        "final_tag": TAG,
        "tag_target_policy": "annotated tag binds the commit containing this manifest",
        "precloseout_evaluator_commit": test["binding"]["evaluator_commit"],
        "training_started_by_closeout": False,
        "checkpoint_selection_modified": False,
        "model_parameters_modified": False,
        "roles": {
            "q4_audit": ["train", "valid_iid"],
            "final_test": ["train", "test_iid"],
            "hard_accessed": False,
            "sealed_iid_accessed": False,
        },
        "threshold": {
            "metric": "point_global_relative_rmse_pct_true_rms",
            "operator": "<",
            "value_pct": THRESHOLD,
            "valid_iid_pct": valid["point_global_relative_rmse_pct"],
            "test_iid_pct": test_summary["point_global_relative_rmse_pct"],
            "passed": False,
        },
        "frozen_candidate": {
            "config_id": V42,
            "checkpoint": "params_best_valid_point_global.pkl",
            "epoch": 257,
            "sha256": test["binding"]["checkpoint_sha256"],
            "training_commit": test["binding"]["training_commit"],
            "valid_iid": valid,
            "test_iid": test_summary,
            "test_minus_valid": test["test_minus_valid"],
        },
        "q4_root_cause": q4_summary,
        "v6_unique_scientific_hypothesis": hypothesis,
        "integration": {
            "branch": "integration/v5-core",
            "commit": INTEGRATION_COMMIT,
            "base": "main@11e9d2feb1b920b2fbd06b8e626a706b1eb4cd40",
            "scope": "stable implementation and tests only",
            "merge_executed": False,
        },
        "artifacts": {_artifact(path)["path"]: _artifact(path) for path in artifact_paths},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    ranking = [row for row in assessment["formal_ranking"] if row["model"] in {"V38", "V42", "V44", "V45", "V46"}]
    lines = [
        "# Heat3D V5 phase closeout",
        "",
        f"Status: `{STATUS}`. The frozen V5 candidate remains V42 point-global best e257. No training, checkpoint reselection, model mutation, hard-role access, or sealed-IID access occurred during closeout.",
        "",
        "## Frozen valid ranking",
        "",
        "| model | epoch | point-global | sample-first | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {row['point_global_relative_rmse_pct']:.6f}% | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f}% | {row['raw_cv_weighted_rmse_K']:.6f} | "
            f"{row['shape_cv_rmse']:.6f} | {row['scale_log_rmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Final V42 e257 test_iid",
            "",
            "The checkpoint was frozen before test access. Test was not used for selection or tuning.",
            "",
            "| metric | valid_iid | test_iid | test-valid |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (
        "point_global_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
        "amplitude_ratio",
        "spatial_correlation",
        "hotspot_cv_weighted_rmse_K",
        "top5_cv_weighted_rmse_K",
        "strong_q_cv_weighted_rmse_K",
        "low_deltaT_background_bias_K",
        "low_deltaT_background_rmse_K",
        "low_deltaT_background_over_ratio",
        "shape_cv_rmse",
        "scale_log_rmse",
        "legacy_normalized_valid_base_mse",
    ):
        lines.append(f"| {key} | {valid[key]:.9g} | {test_summary[key]:.9g} | {test['test_minus_valid'][key]:+.9g} |")
    timing = test["timing"]
    lines.extend(
        [
            "",
            "## Batch-1 inference timing",
            "",
            f"Device `{timing['device']}`, backend `{timing['backend']}`, float32 parameters, {timing['warmup_iterations']} warmups, synchronized per sample; checkpoint load and file I/O excluded.",
            "",
            "| path | mean ms | median ms | P90 ms | N |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in (("model_forward", "model forward"), ("graph_preprocess_and_model_forward", "graph/preprocess + forward")):
        row = timing[key]
        lines.append(f"| {label} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | {row['sample_count']} |")
    lines.extend(
        [
            "",
            "## Q4 root cause and V6 hypothesis",
            "",
            f"All pairwise top-5 difficult sets are identical; top-10 intersections range {q4_summary['top10_pairwise_intersection_range'][0]}–{q4_summary['top10_pairwise_intersection_range'][1]} samples. Q4 contributes "
            + ", ".join(f"{model} {share:.1%}" for model, share in q4_summary["q4_point_sse_share"].items())
            + " of point SSE.",
            "",
            q4_summary["interpretation"],
            "",
            f"Unique V6 hypothesis: {hypothesis}",
            "",
            "## Integration",
            "",
            f"`integration/v5-core@{INTEGRATION_COMMIT[:8]}` contains only stable runner/model/metric/context/pooling implementation and synthetic tests. Research tables, generated YAML, closeout utilities, datasets and run artifacts are excluded. No merge was executed.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": STATUS, "manifest": str(args.manifest), "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
