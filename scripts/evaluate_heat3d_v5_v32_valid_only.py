#!/usr/bin/env python3
"""Frozen-formula valid-only V32 metric and attention evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v5_global_context import fit_train_only_standardizer  # noqa: E402
from rigno.heat3d_v5_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    control_volume_weights,
    evaluate_metric_suite,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    _attach_qk_region_features_to_groups,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)
from run_heat3d_v5_clean_first import (  # noqa: E402
    _attach_v5_physics,
    _load_examples,
    _physics_cache,
)
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402


CONFIG_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
CHECKPOINTS = {
    "point_global_best": (
        "params_best_valid_point_global.pkl",
        "point_global_best_predictions.npz",
        "point_global_best",
    ),
    "sample_first_best": (
        "params_best_valid_sample_first.pkl",
        "sample_first_best_predictions.npz",
        "sample_first_best",
    ),
    "legacy_best": ("params_best.pkl", "best_predictions.npz", "best"),
    "final": ("params_final.pkl", "predictions.npz", "final"),
}
ATTENTION_FEATURES = (
    "source_present_fraction",
    "log1p_q_relative",
    "log_inverse_kz_relative",
    "log1p_q_inverse_kz_relative",
)


class EvaluationError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
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


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _leaves(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _leaves(child)
    else:
        yield np.asarray(value)


def _unwrap_intermediate(value: Any) -> np.ndarray:
    while isinstance(value, (tuple, list)):
        value = value[0]
    return np.asarray(value, dtype=np.float64)


def _normalization_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    scalar_fields = (
        "normalization_profile",
        "condition_feature_transform",
        "input_feature_schema",
        "coord_policy",
        "extent_feature_policy",
    )
    if any(left.get(field) != right.get(field) for field in scalar_fields):
        return False
    if tuple(left.get("feature_names") or ()) != tuple(right.get("feature_names") or ()):
        return False
    for field in ("target_delta_mean", "target_delta_std", "condition_mean", "condition_std"):
        if not np.allclose(
            np.asarray(left.get(field), dtype=np.float64),
            np.asarray(right.get(field), dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            return False
    return True


def _finite_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return None
    value = spearmanr(x, y).statistic if rank else np.corrcoef(x, y)[0, 1]
    return float(value) if math.isfinite(float(value)) else None


def _distribution(values: Sequence[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "finite_sample_count": 0,
            "mean": None,
            "median": None,
            "mean_abs": None,
            "min": None,
            "max": None,
        }
    return {
        "finite_sample_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _checkpoint_metadata(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_params_checkpoint(path)
    leaves = list(_leaves(payload["params"]))
    label = str(payload["checkpoint_kind"])
    reload_rows = {
        str(row["label"]): row
        for row in summary["checkpoint_prediction_reload_audit"]["entries"]
    }
    reload_label = {
        "point_global_best": "point_global_best",
        "sample_first_best": "sample_first_best",
        "best": "best",
        "final": "final",
    }[label]
    reload_row = reload_rows[reload_label]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "checkpoint_kind": label,
        "epoch": int(payload["epoch"]),
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "parameter_count": int(sum(array.size for array in leaves)),
        "parameter_leaf_count": len(leaves),
        "parameter_reload_max_abs_error": float(
            reload_row["parameter_reload_max_abs_error"]
        ),
        "training_replay_max_abs_error_K": float(
            reload_row["checkpoint_reload_max_abs_error_K"]
        ),
        "training_replay_tolerance_K": float(reload_row["tolerance_K"]),
        "training_replay_passed": bool(reload_row["passed"]),
    }


def _prediction_fields(path: Path, ids: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ids):
            raise EvaluationError(f"{path}: keys differ from valid_iid")
        fields = {
            sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            for sample_id in ids
        }
    if any(field.size != 1024 or not np.all(np.isfinite(field)) for field in fields.values()):
        raise EvaluationError(f"{path}: invalid prediction fields")
    return fields


def _metric_report(
    prediction_path: Path,
    *,
    ids: Sequence[str],
    data_root: Path,
    checkpoint_stats: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = _prediction_fields(prediction_path, ids)
    mean = float(np.asarray(checkpoint_stats["target_delta_mean"]).reshape(-1)[0])
    std = float(np.asarray(checkpoint_stats["target_delta_std"]).reshape(-1)[0])
    samples = []
    for sample_id in ids:
        sample_dir = data_root / sample_id
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        if meta.get("split") != "valid_iid":
            raise EvaluationError(f"{sample_id}: non-valid role encountered")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        target = (
            np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1)
            - bottom
        )
        prediction = predictions[sample_id] - bottom
        samples.append(
            {
                "sample_id": sample_id,
                "split": "valid_iid",
                "prediction_deltaT_K": prediction,
                "target_deltaT_K": target,
                "control_volumes_m3": control_volume_weights(
                    np.load(sample_dir / "coords.npy").astype(np.float64)
                ),
                "q_W_m3": np.load(sample_dir / "q_field.npy").astype(np.float64),
                "prediction_normalized": (prediction - mean) / std,
                "target_normalized": (target - mean) / std,
            }
        )
    suite = evaluate_metric_suite(samples)
    return {
        "metric_schema_version": suite["metric_schema_version"],
        "summary": suite["summary"],
    }


def _apply_with_intermediates(model: Any, params: Any, group: Mapping[str, Any]):
    physics = group["native_physics"]
    return model.apply(
        {"params": params},
        inputs=group["inputs"],
        graphs=group["graphs"],
        global_context=group["global_context"],
        control_volumes=physics["control_volumes"],
        log_s_phys=physics["log_s_phys"],
        reference_temperature=physics["reference_temperature"],
        dirichlet_mask=physics["dirichlet_mask"],
        prescribed_temperature=physics["prescribed_temperature"],
        qk_region_features=group["qk_region_features"],
        method=model.predict_native_shape_scale,
        mutable=["intermediates"],
    )


def _attention_report(
    checkpoint_path: Path,
    prediction_path: Path,
    *,
    groups: Sequence[Mapping[str, Any]],
    ids: Sequence[str],
) -> dict[str, Any]:
    checkpoint = _load_params_checkpoint(checkpoint_path)
    stats = checkpoint["train_only_normalization"]
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), dict(stats)
    )
    if model_config.get("scale_attention_mode") != "physics_gate":
        raise EvaluationError("V32 checkpoint lacks physics_gate scale attention")
    if model_config.get("qk_region_feature_version") != "sparse_safe_v2":
        raise EvaluationError("V32 checkpoint lacks sparse_safe_v2 features")
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    saved = _prediction_fields(prediction_path, ids)
    per_sample = []
    replay_max = 0.0
    for group in groups:
        prediction, state = _apply_with_intermediates(model, params, group)
        weights = _unwrap_intermediate(
            state["intermediates"]["scale_attention_weights"]
        )
        residual = _unwrap_intermediate(
            state["intermediates"]["scale_attention_residual"]
        )
        rnodes = np.asarray(prediction["rnodes_processed"], dtype=np.float64)
        mean_pool = np.mean(rnodes, axis=1)
        features = np.asarray(group["qk_region_features"], dtype=np.float64)
        names = tuple(group["qk_region_feature_names"])
        raw_temperature = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        for index, sample_id in enumerate(group["sample_ids"]):
            sample_id = str(sample_id)
            replay_max = max(
                replay_max,
                float(
                    np.max(
                        np.abs(
                            raw_temperature[index].reshape(-1) - saved[sample_id]
                        )
                    )
                ),
            )
            row_weights = weights[index]
            entropy = -float(
                np.sum(row_weights * np.log(np.maximum(row_weights, 1.0e-12)))
            )
            normalized_entropy = (
                entropy / math.log(float(row_weights.size))
                if row_weights.size > 1
                else 0.0
            )
            mean_norm = float(np.linalg.norm(mean_pool[index]))
            residual_norm = float(np.linalg.norm(residual[index]))
            correlations = {}
            for feature_name in ATTENTION_FEATURES:
                feature = features[index, :, names.index(feature_name)]
                correlations[feature_name] = {
                    "pearson": _finite_correlation(
                        row_weights, feature, rank=False
                    ),
                    "spearman": _finite_correlation(
                        row_weights, feature, rank=True
                    ),
                }
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "normalized_entropy": normalized_entropy,
                    "maximum_weight": float(np.max(row_weights)),
                    "residual_l2_norm": residual_norm,
                    "mean_pool_l2_norm": mean_norm,
                    "residual_to_mean_pool_l2_ratio": residual_norm
                    / max(mean_norm, 1.0e-12),
                    "correlations": correlations,
                }
            )
    if {row["sample_id"] for row in per_sample} != set(ids):
        raise EvaluationError("attention diagnostics did not cover valid_iid")
    correlations = {
        feature_name: {
            method: _distribution(
                [
                    row["correlations"][feature_name][method]
                    for row in per_sample
                ]
            )
            for method in ("pearson", "spearman")
        }
        for feature_name in ATTENTION_FEATURES
    }
    entropy = _distribution([row["normalized_entropy"] for row in per_sample])
    maximum = _distribution([row["maximum_weight"] for row in per_sample])
    residual = _distribution([row["residual_l2_norm"] for row in per_sample])
    ratio = _distribution(
        [row["residual_to_mean_pool_l2_ratio"] for row in per_sample]
    )
    regional_count = int(np.asarray(groups[0]["qk_region_features"]).shape[1])
    uniform_max = 1.0 / regional_count
    if float(entropy["mean"]) < 0.20 or float(maximum["mean"]) > 0.80:
        classification = "attention_collapse"
    elif (
        float(entropy["mean"]) > 0.98
        and float(maximum["mean"]) < 2.0 * uniform_max
    ):
        classification = "approximately_uniform"
    else:
        classification = "effective_regional_selection"
    return {
        "sample_count": len(per_sample),
        "regional_node_count": regional_count,
        "uniform_weight": uniform_max,
        "normalized_entropy": entropy,
        "maximum_weight": maximum,
        "residual_l2_norm": residual,
        "residual_to_mean_pool_l2_ratio": ratio,
        "weight_feature_correlations_computed_per_sample_then_aggregated": correlations,
        "classification": classification,
        "evaluator_replay_max_abs_error_K": replay_max,
        "finite": True,
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
        raise EvaluationError("evaluation output must be outside training directory")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "loss_summary.json").read_text(encoding="utf-8"))
    if Path(run_config["output_dir"]).name != CONFIG_ID:
        raise EvaluationError("run_config output binding failed")
    if (
        int(summary["final_epoch"]) != 600
        or not bool(summary["grad_finite"])
        or len(summary["epoch_history"]) != 600
        or [int(row["epoch"]) for row in summary["epoch_history"]] != list(range(1, 601))
    ):
        raise EvaluationError("training completion/history audit failed")

    checkpoint_paths = {
        name: run_dir / values[0] for name, values in CHECKPOINTS.items()
    }
    prediction_paths = {
        name: run_dir / values[1] for name, values in CHECKPOINTS.items()
    }
    checkpoint_metadata = {
        name: _checkpoint_metadata(path, summary)
        for name, path in checkpoint_paths.items()
    }
    if {row["training_commit"] for row in checkpoint_metadata.values()} != {"fcdb01d"}:
        raise EvaluationError("checkpoint training commit drifted")
    if {row["parameter_count"] for row in checkpoint_metadata.values()} != {893736}:
        raise EvaluationError("checkpoint parameter count drifted")
    expected_epochs = {
        "point_global_best": int(summary["point_global_best_epoch"]),
        "sample_first_best": int(summary["sample_first_best_epoch"]),
        "legacy_best": int(summary["best_epoch"]),
        "final": 600,
    }
    if {
        name: row["epoch"] for name, row in checkpoint_metadata.items()
    } != expected_epochs:
        raise EvaluationError("checkpoint epochs differ from loss summary")

    first_checkpoint = _load_params_checkpoint(checkpoint_paths["legacy_best"])
    checkpoint_stats = dict(first_checkpoint["train_only_normalization"])
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    recomputed_stats = training_normalization_stats(
        train_examples,
        normalization_profile=str(
            checkpoint_stats.get("normalization_profile", "legacy_zscore")
        ),
        condition_feature_transform=checkpoint_stats.get(
            "condition_feature_transform"
        ),
        input_feature_schema=str(
            checkpoint_stats.get("input_feature_schema", "legacy_bc_flags")
        ),
        coord_policy=str(
            checkpoint_stats.get("coord_policy", "train_minmax_to_unit_box")
        ),
        extent_feature_policy=str(
            checkpoint_stats.get("extent_feature_policy", "none")
        ),
        bridge_fn=runner_module._bridge_for,
    )
    if not _normalization_equal(checkpoint_stats, recomputed_stats):
        raise EvaluationError("normalization does not reproduce from train only")
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(Path(run_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(run_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise EvaluationError("train/valid split count drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    all_examples = list(train_examples) + list(valid_examples)
    if any(np.asarray(example.condition.coords).shape != (1024, 3) for example in all_examples):
        raise EvaluationError("expected 1024 nodes per sample")
    cache = _physics_cache(all_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    stored_standardizer = run_config["global_context"]["standardizer"]
    if (
        standardizer["fit_population"] != "train_only"
        or int(standardizer["fit_sample_count"]) != 672
        or standardizer["fit_sample_ids_sha256"]
        != stored_standardizer["fit_sample_ids_sha256"]
    ):
        raise EvaluationError("global context standardizer is not train-only")
    for field in ("mean", "std"):
        if not np.allclose(
            np.asarray(standardizer[field]),
            np.asarray(stored_standardizer[field]),
            rtol=1.0e-9,
            atol=1.0e-10,
        ):
            raise EvaluationError(f"global context {field} differs from training")

    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "v32_valid_iid_only",
        False,
        "basic",
        int(run_config.get("graph_seed", 0)),
        batch_size=args.prediction_batch_size,
        drop_last=False,
    )
    _attach_v5_physics(groups, cache, standardizer)
    valid_by_id = {example.sample_id: example for example in valid_examples}
    _attach_qk_region_features_to_groups(
        groups, valid_by_id, feature_version="sparse_safe_v2"
    )
    for group in groups:
        group["native_physics"] = group["v5_physics"]
        group["global_context"] = group["v5_physics"]["global_context"]

    metrics = {
        name: _metric_report(
            prediction_paths[name],
            ids=valid_ids,
            data_root=sample_root,
            checkpoint_stats=checkpoint_stats,
        )
        for name in CHECKPOINTS
    }
    attention = {
        name: _attention_report(
            checkpoint_paths[name],
            prediction_paths[name],
            groups=groups,
            ids=valid_ids,
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
    evaluator_path = Path(__file__).resolve()
    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    payload = {
        "schema_version": "heat3d_v5_v32_valid_only_closeout_v1",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "status": "completed_valid_only",
        "config_id": CONFIG_ID,
        "host": "wsl2",
        "training_commit": str(summary["code_version_or_git_commit"]),
        "evaluator_commit": _git_commit(),
        "evaluator_source_sha256": _sha256(evaluator_path),
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
            "valid_sample_ids_sha256": _ids_hash(valid_ids),
        },
        "training_completion": {
            "final_epoch": 600,
            "epoch_history_count": 600,
            "epoch_history_contiguous_1_to_600": True,
            "grad_finite": True,
            "loss_summary_sha256": _sha256(run_dir / "loss_summary.json"),
            "run_config_sha256": _sha256(run_dir / "run_config.json"),
            "declared_log_path": (
                "output/heat3d_v5_gate6h_attention_fix_logs/"
                f"{CONFIG_ID}.log"
            ),
            "declared_log_exists": False,
            "log_integrity_status": "declared_log_missing_cannot_verify_completeness",
            "completion_evidence": "loss_summary_and_final_checkpoint",
        },
        "normalization_and_context": {
            "normalization_recomputed_from_train_only": True,
            "context_recomputed_from_train_only": True,
            "fit_roles": ["train"],
            "fit_sample_count": 672,
            "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
            "target_or_label_features": [],
        },
        "split": {
            "source": split_source,
            "train_count": 672,
            "valid_iid_count": 128,
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_selection_caveat": {
            "saved_sample_first_checkpoint_primary": (
                "valid_native_joint_relative_rmse"
            ),
            "saved_tie_break_actual": "ordinary_raw_rmse_K",
            "saved_tie_break_condition": "exact_primary_metric_equality_only",
            "correct_cv_metric_role": "post_hoc_diagnostic_only",
            "checkpoint_reselection_performed": False,
        },
        "formulas": {
            "point_global_relative_rmse_pct": (
                "100*sqrt(sum(error_deltaT_K^2)/sum(true_deltaT_K^2))"
            ),
            "sample_first_cv_relative_rmse_pct": (
                "100*mean_samples(CV_RMS(error)/CV_RMS(true_deltaT))"
            ),
            "raw_cv_weighted_rmse_K": "sqrt(sum(error^2*CV)/sum(CV))",
        },
        "metrics": metrics,
        "attention_diagnostics": attention,
        "artifacts": artifacts,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_json": str(output_json),
                "epochs": expected_epochs,
                "metrics": {
                    name: {
                        key: metrics[name]["summary"][key]
                        for key in (
                            "point_global_relative_rmse_pct",
                            "sample_first_cv_relative_rmse_pct",
                            "raw_cv_weighted_rmse_K",
                            "legacy_normalized_valid_base_mse",
                        )
                    }
                    for name in CHECKPOINTS
                },
                "attention_classification": {
                    name: attention[name]["classification"] for name in CHECKPOINTS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
