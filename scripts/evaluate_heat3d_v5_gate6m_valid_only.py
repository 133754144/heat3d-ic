#!/usr/bin/env python3
"""Evaluate the completed Gate 6M scale-head run on valid_iid only.

This evaluator consumes persisted predictions and checkpoint metadata.  It does
not train, alter checkpoints, or open test, hard, or sealed roles.  The metric
calculation is the frozen V5 true-RMS contract shared by the Gate 6H/6L
evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_heat3d_v5_v32_valid_only as frozen  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402
from rigno.heat3d_v5_global_context import fit_train_only_standardizer  # noqa: E402
from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402


CONFIG_ID = "V4P5_35_gate6m_v32_scale_head_only_e100"
CHECKPOINTS = {
    "point_global_best": (
        "params_best_valid_point_global.pkl",
        "point_global_best_predictions.npz",
    ),
    "sample_first_best": (
        "params_best_valid_sample_first.pkl",
        "sample_first_best_predictions.npz",
    ),
    "legacy_best": (
        "params_best_valid_base_mse.pkl",
        "base_mse_best_predictions.npz",
    ),
    "final": ("params_final.pkl", "predictions.npz"),
}


class EvaluationError(RuntimeError):
    """Raised when a persisted run violates the valid-only contract."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _path_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=ROOT,
        text=True,
    ).strip()


def _reload_rows(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    audit = summary.get("checkpoint_prediction_reload_audit") or {}
    if audit.get("status") != "passed":
        raise EvaluationError("persisted checkpoint reload audit did not pass")
    rows = {str(row["label"]): row for row in audit.get("entries", ())}
    required = {"point_global_best", "sample_first_best", "base_mse_best", "final"}
    if not required <= set(rows):
        raise EvaluationError("checkpoint reload audit lacks a required entry")
    if any(not bool(rows[label].get("passed")) for label in required):
        raise EvaluationError("checkpoint reload audit contains a failed entry")
    return rows


def _checkpoint_metadata(
    path: Path, checkpoint_name: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    payload = frozen._load_params_checkpoint(path)
    second = frozen._load_params_checkpoint(path)
    parameter_reload = runner_module._tree_max_abs_difference(
        payload["params"], second["params"]
    )
    expected_epoch = int(
        summary[
            {
                "point_global_best": "point_global_best_epoch",
                "sample_first_best": "sample_first_best_epoch",
                "legacy_best": "base_mse_best_epoch",
                "final": "final_epoch",
            }[checkpoint_name]
        ]
    )
    if int(payload["epoch"]) != expected_epoch:
        raise EvaluationError(f"{path}: checkpoint epoch mismatch")
    reload_label = {
        "point_global_best": "point_global_best",
        "sample_first_best": "sample_first_best",
        "legacy_best": "base_mse_best",
        "final": "final",
    }[checkpoint_name]
    row = _reload_rows(summary)[reload_label]
    leaves = [np.asarray(value) for value in __import__("jax").tree_util.tree_leaves(payload["params"])]
    if parameter_reload != 0.0:
        raise EvaluationError(f"{path}: non-exact parameter reload")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "checkpoint_kind": str(payload.get("checkpoint_kind") or ""),
        "epoch": expected_epoch,
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "parameter_count": int(sum(value.size for value in leaves)),
        "parameter_leaf_count": len(leaves),
        "parameter_reload_max_abs_error": float(parameter_reload),
        "training_reload_audit": {
            "passed": bool(row["passed"]),
            "checkpoint_reload_max_abs_error_K": float(
                row["checkpoint_reload_max_abs_error_K"]
            ),
            "npz_reload_max_abs_error_K": float(row["npz_reload_max_abs_error_K"]),
            "tolerance_K": float(row["tolerance_K"]),
        },
    }


def _check_train_only_context(
    run_config: Mapping[str, Any],
    checkpoint_stats: Mapping[str, Any],
    train_ids: list[str],
    train_examples: list[Any],
) -> dict[str, Any]:
    recomputed = training_normalization_stats(
        train_examples,
        normalization_profile=str(
            checkpoint_stats.get("normalization_profile", "legacy_zscore")
        ),
        condition_feature_transform=checkpoint_stats.get("condition_feature_transform"),
        input_feature_schema=str(
            checkpoint_stats.get("input_feature_schema", "legacy_bc_flags")
        ),
        coord_policy=str(checkpoint_stats.get("coord_policy", "train_minmax_to_unit_box")),
        extent_feature_policy=str(checkpoint_stats.get("extent_feature_policy", "none")),
        bridge_fn=runner_module._bridge_for,
    )
    if not frozen._normalization_equal(checkpoint_stats, recomputed):
        raise EvaluationError("normalization is not reproducible from train only")
    cache = frozen._physics_cache(train_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    stored = run_config.get("global_context", {}).get("standardizer", {})
    if stored.get("fit_population") != "train_only":
        raise EvaluationError("global-context standardizer is not train-only")
    if int(stored.get("fit_sample_count", -1)) != len(train_ids):
        raise EvaluationError("global-context fit sample count drifted")
    if stored.get("fit_sample_ids_sha256") != standardizer["fit_sample_ids_sha256"]:
        raise EvaluationError("global-context fit sample IDs drifted")
    return {
        "normalization_recomputed_from_train_only": True,
        "context_recomputed_from_train_only": True,
        "fit_roles": ["train"],
        "fit_sample_count": len(train_ids),
        "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
        "target_or_label_features": [],
    }


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    output_json = args.output_json.resolve()
    if output_json.exists() and not args.overwrite:
        raise EvaluationError(f"output exists: {output_json}")
    if run_dir.name != CONFIG_ID:
        raise EvaluationError("run/config binding failed")
    if run_dir == output_json.parent or run_dir in output_json.parents:
        raise EvaluationError("evaluation output overlaps run directory")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "loss_summary.json").read_text(encoding="utf-8"))
    if Path(run_config["output_dir"]).name != CONFIG_ID:
        raise EvaluationError("run_config output binding failed")
    history = summary.get("epoch_history", ())
    if (
        int(summary.get("final_epoch", -1)) != 100
        or len(history) != 100
        or [int(row["epoch"]) for row in history] != list(range(1, 101))
        or not bool(summary.get("grad_finite"))
    ):
        raise EvaluationError("e100 completion/history audit failed")
    _reload_rows(summary)

    checkpoint_paths = {
        name: run_dir / values[0] for name, values in CHECKPOINTS.items()
    }
    prediction_paths = {
        name: run_dir / values[1] for name, values in CHECKPOINTS.items()
    }
    for path in (*checkpoint_paths.values(), *prediction_paths.values()):
        if not path.is_file():
            raise EvaluationError(f"missing artifact: {path}")
    checkpoint_metadata = {
        name: _checkpoint_metadata(path, name, summary)
        for name, path in checkpoint_paths.items()
    }
    if {row["training_commit"] for row in checkpoint_metadata.values()} != {
        str(summary.get("code_version_or_git_commit"))
    }:
        raise EvaluationError("checkpoint training commit drifted")
    if len({row["parameter_count"] for row in checkpoint_metadata.values()}) != 1:
        raise EvaluationError("checkpoint parameter count drifted")

    canonical_checkpoint = frozen._load_params_checkpoint(checkpoint_paths["legacy_best"])
    checkpoint_stats = dict(canonical_checkpoint["train_only_normalization"])
    for name, path in checkpoint_paths.items():
        payload = frozen._load_params_checkpoint(path)
        if not frozen._normalization_equal(
            checkpoint_stats, payload["train_only_normalization"]
        ):
            raise EvaluationError(f"{name}: normalization drifted")

    frozen.install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = frozen.load_training_examples(run_config, checkpoint_stats)
    sample_root = frozen._sample_root(Path(run_config["subset"]))
    split_ids, split_source, _, _ = frozen._resolve_training_splits(
        sample_root, Path(run_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise EvaluationError("train/valid_iid split counts drifted")
    context_audit = _check_train_only_context(
        run_config, checkpoint_stats, train_ids, train_examples
    )

    metrics = {
        name: frozen._metric_report(
            prediction_paths[name],
            ids=valid_ids,
            data_root=sample_root,
            checkpoint_stats=checkpoint_stats,
        )
        for name in CHECKPOINTS
    }
    artifacts = {
        path.name: {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            run_dir / "run_config.json",
            run_dir / "loss_summary.json",
            *checkpoint_paths.values(),
            *prediction_paths.values(),
        )
    }
    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    payload = {
        "schema_version": "heat3d_v5_gate6m_valid_only_four_checkpoint_v1",
        "metric_schema_version": frozen.METRIC_SCHEMA_VERSION,
        "status": "completed_valid_iid_only",
        "config_id": CONFIG_ID,
        "training_commit": str(summary["code_version_or_git_commit"]),
        "evaluator_commit": _git_commit(),
        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        "frozen_formula_source": {
            "path": str(metric_path.relative_to(ROOT)),
            "commit": _path_commit(metric_path),
            "sha256": _sha256(metric_path),
        },
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "evaluation_roles": ["valid_iid"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "model_parameters_modified": False,
            "checkpoint_selection_modified": False,
            "sample_count": 128,
            "nodes_per_sample": 1024,
            "valid_sample_ids_sha256": frozen._ids_hash(valid_ids),
        },
        "split": {
            "source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "train_ids_sha256": frozen._ids_hash(train_ids),
            "valid_iid_ids_sha256": frozen._ids_hash(valid_ids),
        },
        "normalization_and_context": context_audit,
        "training_completion": {
            "final_epoch": 100,
            "epoch_history_count": 100,
            "epoch_history_contiguous_1_to_100": True,
            "grad_finite": bool(summary["grad_finite"]),
            "loss_summary_sha256": _sha256(run_dir / "loss_summary.json"),
            "run_config_sha256": _sha256(run_dir / "run_config.json"),
            "completion_evidence": "loss_summary_and_four_checkpoint_prediction_archives",
        },
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_selection_caveat": {
            "saved_selection_metric": str(run_config.get("selection_metric")),
            "point_global_best_epoch": int(summary["point_global_best_epoch"]),
            "sample_first_best_epoch": int(summary["sample_first_best_epoch"]),
            "legacy_best_epoch": int(summary["base_mse_best_epoch"]),
            "reselection_performed": False,
            "metric_role": "post_training_diagnostic_only_for_this_closeout",
        },
        "formulas": {
            "point_global_relative_rmse_pct": "100*sqrt(sum_point(error^2)/sum_point(true_deltaT^2))",
            "sample_first_cv_relative_rmse_pct": "100*mean_samples(CV_RMS(error)/CV_RMS(true_deltaT))",
            "raw_cv_weighted_rmse_K": "sqrt(sum(error^2*control_volume)/sum(control_volume))",
        },
        "metrics": metrics,
        "training_diagnostics": {
            "attention_diagnostics_by_checkpoint": run_config.get(
                "attention_diagnostics_by_checkpoint", {}
            ),
            "native_runtime_architecture_audit": run_config.get(
                "native_runtime_architecture_audit", {}
            ),
            "checkpoint_prediction_reload_audit": summary.get(
                "checkpoint_prediction_reload_audit", {}
            ),
            "epoch_history": history,
            "loss_weight_history_summary": summary.get("loss_weight_history_summary"),
            "timing_diagnostics": run_config.get("timing_diagnostics", {}),
        },
        "artifacts": artifacts,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_json": str(output_json),
                "epochs": {
                    name: metadata["epoch"]
                    for name, metadata in checkpoint_metadata.items()
                },
                "metrics": {
                    name: metrics[name]["summary"] for name in CHECKPOINTS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
