#!/usr/bin/env python3
"""Gate 6P read-only scale-path attribution on train/valid_iid only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    fit_train_only_standardizer,
)
from diagnose_heat3d_v5_gate6m import (  # noqa: E402
    _decompose_fields,
    _suite_from_fields,
)
from diagnose_heat3d_v5_gate6o import (  # noqa: E402
    CHECKPOINTS as V38_CHECKPOINTS,
    _delta_fields,
    _ids_hash,
    _infer_raw_temperature,
    _normalization_equal,
    _sha256,
    _targets_for_role,
)
from evaluate_heat3d_v5_gate6l_valid_only import _build_groups  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    _device_params,
    _load_params_checkpoint,
    _model_apply,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)
from run_heat3d_v5_clean_first import _load_examples, _physics_cache  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402


V38_CONFIG_ID = "V4P5_38_gate6n_v36_r2r_mask_p005_e600"
V39_CONFIG_ID = "V4P5_39_gate6o_e543_scale_mlp_calibration_e40"
V39_CHECKPOINT = {
    "filename": "params_best_valid_point_global.pkl",
    "epoch": 24,
    "sha256": "edc14650b36a6b5b4068efc39ed3db608bac1fb00fc0cd4a544b06e28d9df823",
}
CV_SEED = 2026071902
CV_FOLDS = 5
RIDGE_ALPHA = 1.0e-3
EPS = 1.0e-12


class Gate6PError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v38-run-dir", type=Path, required=True)
    parser.add_argument("--v39-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fields} for row in rows]
        )


def _mutable_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _mutable_tree(child) for key, child in value.items()}
    return copy.deepcopy(value)


def _module_names(params: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(name) for name in params))


def _transplant(
    skeleton: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    prefixes: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    target = _mutable_tree(skeleton)
    source = _mutable_tree(donor)
    names = [
        name
        for name in _module_names(target)
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not names:
        raise Gate6PError(f"no parameter module matched prefixes {prefixes}")
    for name in names:
        if name not in source:
            raise Gate6PError(f"donor misses scale module {name}")
        target[name] = copy.deepcopy(source[name])
    return target, names


def _infer_with_scale_features(
    checkpoint: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    stats = dict(checkpoint["train_only_normalization"])
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    raw: dict[str, np.ndarray] = {}
    features: dict[str, dict[str, Any]] = {}
    for group in groups:
        prediction = _model_apply(model, params, group)
        temperature = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        pooled = np.asarray(prediction["pooled_rnodes"], dtype=np.float64)
        context = np.asarray(group["global_context"], dtype=np.float64)
        s_hat = np.asarray(prediction["s_hat"], dtype=np.float64).reshape(-1)
        log_s_phys = np.asarray(
            group["native_physics"]["log_s_phys"], dtype=np.float64
        ).reshape(-1)
        for index, sample_id_value in enumerate(group["sample_ids"]):
            sample_id = str(sample_id_value)
            raw[sample_id] = temperature[index].reshape(-1)
            features[sample_id] = {
                "physics": context[index].reshape(-1),
                "pooled_latent": pooled[index].reshape(-1),
                "predicted_scale_K": float(s_hat[index]),
                "log_s_phys_K": float(log_s_phys[index]),
            }
    for sample_id, row in features.items():
        if row["physics"].shape != (24,):
            raise Gate6PError(f"{sample_id}: physics feature width is not 24")
        if row["pooled_latent"].shape != (96,):
            raise Gate6PError(f"{sample_id}: pooled latent width is not 96")
        if not all(
            np.all(np.isfinite(row[name]))
            for name in ("physics", "pooled_latent")
        ):
            raise Gate6PError(f"{sample_id}: non-finite frozen scale feature")
    return raw, features


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def _error_terms(
    fields: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    raw_temperature = {
        sample_id: np.asarray(fields[sample_id], dtype=np.float64)
        + float(targets[sample_id]["bottom_temperature_K"])
        for sample_id in ids
    }
    shapes, scales = _decompose_fields(raw_temperature, ids, targets)
    result = {}
    for sample_id in ids:
        target = np.asarray(
            targets[sample_id]["target_deltaT_K"], dtype=np.float64
        )
        true_scale = float(targets[sample_id]["true_scale_cv_rms_K"])
        true_shape = target / max(true_scale, EPS)
        predicted_shape = shapes[sample_id]
        predicted_scale = scales[sample_id]
        shape_term = true_scale * (predicted_shape - true_shape)
        scale_term = (predicted_scale - true_scale) * predicted_shape
        residual = shape_term + scale_term
        total = float(np.sum(np.square(residual)))
        shape = float(np.sum(np.square(shape_term)))
        scale = float(np.sum(np.square(scale_term)))
        cross = float(2.0 * np.sum(shape_term * scale_term))
        direct = float(np.sum(np.square(fields[sample_id] - target)))
        result[sample_id] = {
            "point_sse_K2": direct,
            "shape_point_sse_K2": shape,
            "scale_point_sse_K2": scale,
            "cross_point_sse_K2": cross,
            "decomposition_closure_abs_K2": abs(direct - (shape + scale + cross)),
            "reconstruction_abs_K2": abs(direct - total),
            "predicted_scale_K": predicted_scale,
            "signed_log_scale_error": math.log(predicted_scale / true_scale),
        }
    return result


def _quartile_summary(
    rows: Sequence[Mapping[str, Any]], comparison: str
) -> list[dict[str, Any]]:
    output = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        selected = [row for row in rows if row["deltaT_quartile"] == quartile]
        output.append(
            {
                "comparison": comparison,
                "deltaT_quartile": quartile,
                "sample_count": len(selected),
                "point_sse_net_delta_K2": float(
                    sum(row[f"{comparison}_point_sse_delta_K2"] for row in selected)
                ),
                "shape_point_sse_net_delta_K2": float(
                    sum(row[f"{comparison}_shape_sse_delta_K2"] for row in selected)
                ),
                "scale_point_sse_net_delta_K2": float(
                    sum(row[f"{comparison}_scale_sse_delta_K2"] for row in selected)
                ),
                "cross_point_sse_net_delta_K2": float(
                    sum(row[f"{comparison}_cross_sse_delta_K2"] for row in selected)
                ),
                "v39_win_rate": float(
                    np.mean(
                        [
                            row[f"{comparison}_point_sse_delta_K2"] < 0.0
                            for row in selected
                        ]
                    )
                ),
            }
        )
    return output


def _top_sse(
    rows: Sequence[Mapping[str, Any]], comparison: str
) -> dict[str, Any]:
    ranked_v39 = sorted(
        rows, key=lambda row: row["v39_e24_point_sse_K2"], reverse=True
    )
    total_v39 = sum(row["v39_e24_point_sse_K2"] for row in rows)
    delta_name = f"{comparison}_point_sse_delta_K2"
    regression = sorted(rows, key=lambda row: row[delta_name], reverse=True)
    improvement = sorted(rows, key=lambda row: row[delta_name])
    return {
        "v39_top5_cumulative_sse_fraction": float(
            sum(row["v39_e24_point_sse_K2"] for row in ranked_v39[:5])
            / max(total_v39, EPS)
        ),
        "v39_top10_cumulative_sse_fraction": float(
            sum(row["v39_e24_point_sse_K2"] for row in ranked_v39[:10])
            / max(total_v39, EPS)
        ),
        "v39_top10_sample_ids": [row["sample_id"] for row in ranked_v39[:10]],
        "top_regression": [
            {
                "sample_id": row["sample_id"],
                "deltaT_quartile": row["deltaT_quartile"],
                "point_sse_delta_K2": row[delta_name],
            }
            for row in regression[:10]
        ],
        "top_improvement": [
            {
                "sample_id": row["sample_id"],
                "deltaT_quartile": row["deltaT_quartile"],
                "point_sse_delta_K2": row[delta_name],
            }
            for row in improvement[:10]
        ],
    }


def _ridge_oof(
    matrix: np.ndarray,
    y_residual: np.ndarray,
    log_s_phys: np.ndarray,
    true_log_scale: np.ndarray,
    fold_ids: np.ndarray,
    q4_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions = np.empty_like(y_residual)
    for fold in range(CV_FOLDS):
        test = fold_ids == fold
        fit = ~test
        mean = matrix[fit].mean(axis=0)
        std = matrix[fit].std(axis=0)
        std = np.where(std > EPS, std, 1.0)
        x_fit = (matrix[fit] - mean) / std
        x_test = (matrix[test] - mean) / std
        design = np.concatenate([np.ones((x_fit.shape[0], 1)), x_fit], axis=1)
        penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ y_residual[fit]
        )
        predictions[test] = (
            np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
            @ coefficients
        )
    predicted_log = log_s_phys + predictions
    error = predicted_log - true_log_scale
    variance = float(np.sum(np.square(true_log_scale - true_log_scale.mean())))
    return predicted_log, {
        "log_scale_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "log_scale_mae": float(np.mean(np.abs(error))),
        "log_scale_signed_bias": float(np.mean(error)),
        "log_scale_r2": float(
            1.0 - np.sum(np.square(error)) / max(variance, EPS)
        ),
        "q4_log_scale_rmse": float(
            np.sqrt(np.mean(np.square(error[q4_mask])))
        ),
        "q4_log_scale_signed_bias": float(np.mean(error[q4_mask])),
    }


def _cross_validate_scale_features(
    train_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    physics = np.stack([features[sample_id]["physics"] for sample_id in train_ids])
    pooled = np.stack(
        [features[sample_id]["pooled_latent"] for sample_id in train_ids]
    )
    log_s_phys = np.asarray(
        [features[sample_id]["log_s_phys_K"] for sample_id in train_ids]
    )
    true_log = np.log(
        np.asarray(
            [targets[sample_id]["true_scale_cv_rms_K"] for sample_id in train_ids]
        )
    )
    y_residual = true_log - log_s_phys
    threshold = float(np.quantile(true_log, 0.75))
    q4_mask = true_log > threshold
    rng = np.random.default_rng(CV_SEED)
    permutation = rng.permutation(len(train_ids))
    fold_ids = np.empty(len(train_ids), dtype=np.int64)
    fold_ids[permutation] = np.arange(len(train_ids)) % CV_FOLDS
    matrices = {
        "physics_24": physics,
        "pooled_latent_96": pooled,
        "combined_120": np.concatenate([physics, pooled], axis=1),
    }
    predictions = {}
    summaries = {}
    for name, matrix in matrices.items():
        predictions[name], summaries[name] = _ridge_oof(
            matrix,
            y_residual,
            log_s_phys,
            true_log,
            fold_ids,
            q4_mask,
        )
        summaries[name]["feature_width"] = int(matrix.shape[1])
    baseline_error = log_s_phys - true_log
    summaries["physics_operator_no_readout"] = {
        "feature_width": 0,
        "log_scale_rmse": float(np.sqrt(np.mean(np.square(baseline_error)))),
        "log_scale_mae": float(np.mean(np.abs(baseline_error))),
        "log_scale_signed_bias": float(np.mean(baseline_error)),
        "q4_log_scale_rmse": float(
            np.sqrt(np.mean(np.square(baseline_error[q4_mask])))
        ),
        "q4_log_scale_signed_bias": float(np.mean(baseline_error[q4_mask])),
    }
    rows = []
    for index, sample_id in enumerate(train_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "role": "train",
                "fold": int(fold_ids[index]),
                "true_log_scale": float(true_log[index]),
                "log_s_phys": float(log_s_phys[index]),
                "is_train_Q4": bool(q4_mask[index]),
                **{
                    f"{name}_oof_log_scale": float(value[index])
                    for name, value in predictions.items()
                },
            }
        )
    return rows, {
        "fit_roles": ["train"],
        "query_roles": ["train"],
        "fold_count": CV_FOLDS,
        "seed": CV_SEED,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "log(s_true)-log(s_phys)",
        "representation_note": (
            "e543 frozen representation was trained on the full train split; "
            "OOF isolation applies to the linear readout only"
        ),
        "feature_sets": summaries,
    }


def _raw_physics(
    sample_id: str,
    cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    row = cache[sample_id]
    context = row["context"]
    q = np.asarray(row["q"], dtype=np.float64).reshape(-1)
    volumes = np.asarray(row["control_volumes"], dtype=np.float64).reshape(-1)
    positive = q > max(EPS, float(np.max(q)) * 1.0e-12)
    total_volume = float(np.sum(volumes))
    return {
        "total_power_W": float(context["P_operator_W"]),
        "q_cv_mean_W_m3": float(np.sum(q * volumes) / total_volume),
        "q_cv_rms_W_m3": float(
            np.sqrt(np.sum(np.square(q) * volumes) / total_volume)
        ),
        "q_max_W_m3": float(np.max(q)),
        "source_present_fraction": float(
            np.sum(volumes[positive]) / total_volume
        ),
        "q_weighted_inverse_kz_mK_W": float(
            context["q_weighted_inverse_kz_mK_W"]
        ),
        "q_weighted_local_kz_W_mK": float(
            context["q_weighted_local_kz_W_mK"]
        ),
        "harmonic_kx_W_mK": float(context["harmonic_kx_W_mK"]),
        "harmonic_ky_W_mK": float(context["harmonic_ky_W_mK"]),
        "harmonic_kz_W_mK": float(context["harmonic_kz_W_mK"]),
        "anisotropy_xy_over_z": float(context["anisotropy_xy_over_z"]),
        "top_h_W_m2K": float(math.exp(context["log_top_h_W_m2K"])),
        "T_inf_minus_T_bottom_K": float(
            context["T_inf_minus_T_bottom_K"]
        ),
        "bc_top_cv_fraction": float(context["bc_top_cv_fraction"]),
        "bc_bottom_cv_fraction": float(context["bc_bottom_cv_fraction"]),
        "bc_side_cv_fraction": float(context["bc_side_cv_fraction"]),
        "source_concentration": float(context["source_concentration"]),
    }


def _coverage(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    train_matrix = np.stack([features[sample_id]["physics"] for sample_id in train_ids])
    output = {}
    for sample_id in valid_ids:
        query = features[sample_id]["physics"]
        distance = np.linalg.norm(train_matrix - query[None, :], axis=1)
        index = int(np.argmin(distance))
        output[sample_id] = {
            "train_nn_distance_24d": float(distance[index]),
            "train_nn_sample_id": train_ids[index],
        }
    return output


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 6P read-only scale-path diagnostics",
        "",
        "仅访问 `train` 与 `valid_iid`；未启动训练，未修改 checkpoint，"
        "`test/hard/sealed` 均未访问。",
        "",
        "## Checkpoint transplant",
        "",
        "| field | point-global % | sample-first % | raw CV K | shape CV | scale log |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, suite in payload["field_metrics"].items():
        summary = suite["summary"]
        lines.append(
            f"| {name} | {summary['point_global_relative_rmse_pct']:.6f} | "
            f"{summary['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{summary['raw_cv_weighted_rmse_K']:.6f} | "
            f"{summary['shape_cv_rmse']:.6f} | "
            f"{summary['scale_log_rmse']:.6f} |"
        )
    lines += [
        "",
        "## Frozen scale-feature readout CV",
        "",
        "| feature set | width | log-scale RMSE | Q4 RMSE | bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["train_only_scale_feature_cv"]["feature_sets"].items():
        lines.append(
            f"| {name} | {row['feature_width']} | "
            f"{row['log_scale_rmse']:.6f} | {row['q4_log_scale_rmse']:.6f} | "
            f"{row['log_scale_signed_bias']:.6f} |"
        )
    conclusion = payload["bottleneck_assessment"]
    lines += [
        "",
        "## Conclusion",
        "",
        f"- bottleneck: `{conclusion['classification']}`",
        f"- evidence: {conclusion['basis']}",
        f"- next candidate: `{conclusion['next_training_candidate']}`",
        "",
        "该候选仅为诊断结论，本轮没有生成或启动训练。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _args()
    v38_run = args.v38_run_dir.resolve()
    v39_run = args.v39_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "gate6p_diagnostics.json",
        "md": output_dir / "gate6p_diagnostics.md",
        "samples_json": output_dir / "gate6p_sample_level.json",
        "samples_csv": output_dir / "gate6p_sample_level.csv",
        "features_csv": output_dir / "gate6p_e543_scale_features.csv",
        "cv_csv": output_dir / "gate6p_train_scale_feature_cv.csv",
        "q4_csv": output_dir / "gate6p_q4_high_error_audit.csv",
        "manifest": output_dir / "gate6p_artifact_manifest.json",
    }
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise Gate6PError("output already exists")
    if v38_run.name != V38_CONFIG_ID or v39_run.name != V39_CONFIG_ID:
        raise Gate6PError("run directory/config binding failed")
    v38_config = json.loads((v38_run / "run_config.json").read_text())
    v38_summary = json.loads((v38_run / "loss_summary.json").read_text())
    v39_summary = json.loads((v39_run / "loss_summary.json").read_text())
    if int(v38_summary.get("final_epoch", -1)) != 600:
        raise Gate6PError("V38 did not complete e600")
    if int(v39_summary.get("final_epoch", -1)) != 40:
        raise Gate6PError("V39 did not complete e40")

    checkpoints = {}
    binding = {}
    for name, (filename, _, epoch, digest) in V38_CHECKPOINTS.items():
        path = v38_run / filename
        if _sha256(path) != digest:
            raise Gate6PError(f"{name}: checkpoint SHA256 drifted")
        checkpoint = _load_params_checkpoint(path)
        if int(checkpoint["epoch"]) != epoch:
            raise Gate6PError(f"{name}: checkpoint epoch drifted")
        checkpoints[name] = checkpoint
        binding[name] = {
            "config_id": V38_CONFIG_ID,
            "epoch": epoch,
            "sha256": digest,
            "path": str(path),
        }
    v39_path = v39_run / V39_CHECKPOINT["filename"]
    if _sha256(v39_path) != V39_CHECKPOINT["sha256"]:
        raise Gate6PError("V39 e24 checkpoint SHA256 drifted")
    checkpoints["v39_e24"] = _load_params_checkpoint(v39_path)
    if int(checkpoints["v39_e24"]["epoch"]) != V39_CHECKPOINT["epoch"]:
        raise Gate6PError("V39 checkpoint epoch drifted")
    binding["v39_e24"] = {
        "config_id": V39_CONFIG_ID,
        "epoch": V39_CHECKPOINT["epoch"],
        "sha256": V39_CHECKPOINT["sha256"],
        "path": str(v39_path),
    }

    canonical_stats = dict(checkpoints["e543"]["train_only_normalization"])
    for name, checkpoint in checkpoints.items():
        if not _normalization_equal(
            canonical_stats, checkpoint["train_only_normalization"]
        ):
            raise Gate6PError(f"{name}: train-only normalization differs")
    install_checkpoint_feature_hooks(canonical_stats)
    train_examples = load_training_examples(v38_config, canonical_stats)
    recomputed = training_normalization_stats(
        train_examples,
        normalization_profile=str(
            canonical_stats.get("normalization_profile", "legacy_zscore")
        ),
        condition_feature_transform=canonical_stats.get(
            "condition_feature_transform"
        ),
        input_feature_schema=str(
            canonical_stats.get("input_feature_schema", "legacy_bc_flags")
        ),
        coord_policy=str(
            canonical_stats.get("coord_policy", "train_minmax_to_unit_box")
        ),
        extent_feature_policy=str(
            canonical_stats.get("extent_feature_policy", "none")
        ),
        bridge_fn=runner_module._bridge_for,
    )
    if not _normalization_equal(canonical_stats, recomputed):
        raise Gate6PError("train-only normalization does not reproduce")
    stats = stats_from_checkpoint_payload(canonical_stats, train_examples)
    sample_root = _sample_root(Path(v38_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(v38_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6PError("split counts drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=canonical_stats,
        boundary_mask_fallback=bool(
            v38_config.get("boundary_mask_fallback", True)
        ),
    )
    cache = _physics_cache(list(train_examples) + list(valid_examples))
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    train_groups = _build_groups(
        run_config=v38_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=train_examples,
        valid_ids=train_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    valid_groups = _build_groups(
        run_config=v38_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=valid_examples,
        valid_ids=valid_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    train_targets = _targets_for_role(
        sample_root=sample_root,
        sample_ids=train_ids,
        expected_role="train",
    )
    valid_targets = _targets_for_role(
        sample_root=sample_root,
        sample_ids=valid_ids,
        expected_role="valid_iid",
    )

    raw = {}
    for name, checkpoint in checkpoints.items():
        raw[name] = _infer_raw_temperature(checkpoint, valid_groups)
    e543_train_raw, train_features = _infer_with_scale_features(
        checkpoints["e543"], train_groups
    )
    e543_valid_raw, valid_features = _infer_with_scale_features(
        checkpoints["e543"], valid_groups
    )
    feature_replay_max_abs_error_K = max(
        float(np.max(np.abs(e543_valid_raw[sample_id] - raw["e543"][sample_id])))
        for sample_id in valid_ids
    )
    if feature_replay_max_abs_error_K > 0.02:
        raise Gate6PError(
            "e543 feature-export replay failed: "
            f"{feature_replay_max_abs_error_K} K"
        )
    all_features = {**train_features, **valid_features}

    transplant_specs = {
        "e543_plus_e231_global_scale_mlp": ("global_scale_",),
        "e543_plus_e231_mlp_scale_attention": (
            "global_scale_",
            "scale_attention_",
        ),
        "e543_plus_e231_complete_scale_head": (
            "global_scale_",
            "scale_attention_",
            "latent_attention_",
            "qk_attention_",
        ),
    }
    transplants = {}
    for name, prefixes in transplant_specs.items():
        params, modules = _transplant(
            checkpoints["e543"]["params"],
            checkpoints["e231"]["params"],
            prefixes=prefixes,
        )
        synthetic = {**checkpoints["e543"], "params": params}
        raw[name] = _infer_raw_temperature(synthetic, valid_groups)
        transplants[name] = {
            "skeleton": "e543",
            "donor": "e231",
            "prefixes": list(prefixes),
            "transplanted_parameter_modules": modules,
            "saved_checkpoint": False,
        }

    delta = {
        name: _delta_fields(fields, valid_ids, valid_targets)
        for name, fields in raw.items()
    }
    suites = {
        name: _suite_from_fields(
            fields=fields,
            ids=valid_ids,
            targets=valid_targets,
            stats=canonical_stats,
        )
        for name, fields in delta.items()
    }
    terms = {
        name: _error_terms(fields, valid_ids, valid_targets)
        for name, fields in delta.items()
        if name in {"e231", "e543", "v39_e24"}
    }
    coverage = _coverage(train_ids, valid_ids, all_features)
    sample_rows = []
    for sample_id in valid_ids:
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "role": "valid_iid",
            "deltaT_quartile": valid_targets[sample_id]["deltaT_quartile"],
            "true_scale_cv_rms_K": valid_targets[sample_id][
                "true_scale_cv_rms_K"
            ],
            "generator_condition_category": valid_targets[sample_id][
                "generator_condition_category"
            ],
            **_raw_physics(sample_id, cache),
            **coverage[sample_id],
        }
        for name in ("e231", "e543", "v39_e24"):
            for field, value in terms[name][sample_id].items():
                row[f"{name}_{field}"] = value
        for baseline in ("e543", "e231"):
            comparison = f"v39_minus_{baseline}"
            row[f"{comparison}_point_sse_delta_K2"] = (
                terms["v39_e24"][sample_id]["point_sse_K2"]
                - terms[baseline][sample_id]["point_sse_K2"]
            )
            row[f"{comparison}_shape_sse_delta_K2"] = (
                terms["v39_e24"][sample_id]["shape_point_sse_K2"]
                - terms[baseline][sample_id]["shape_point_sse_K2"]
            )
            row[f"{comparison}_scale_sse_delta_K2"] = (
                terms["v39_e24"][sample_id]["scale_point_sse_K2"]
                - terms[baseline][sample_id]["scale_point_sse_K2"]
            )
            row[f"{comparison}_cross_sse_delta_K2"] = (
                terms["v39_e24"][sample_id]["cross_point_sse_K2"]
                - terms[baseline][sample_id]["cross_point_sse_K2"]
            )
        sample_rows.append(row)
    closure = max(
        row[f"{name}_decomposition_closure_abs_K2"]
        for row in sample_rows
        for name in ("e231", "e543", "v39_e24")
    )
    if closure > 1.0e-7:
        raise Gate6PError(f"shape/scale/cross decomposition failed: {closure}")

    cv_rows, cv_summary = _cross_validate_scale_features(
        train_ids, all_features, train_targets
    )
    feature_rows = []
    for role, ids in (("train", train_ids), ("valid_iid", valid_ids)):
        for sample_id in ids:
            feature = all_features[sample_id]
            feature_rows.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "log_s_phys_K": feature["log_s_phys_K"],
                    "e543_predicted_scale_K": feature["predicted_scale_K"],
                    **{
                        f"physics_{index:02d}_{name}": value
                        for index, (name, value) in enumerate(
                            zip(GLOBAL_CONTEXT_FEATURES, feature["physics"])
                        )
                    },
                    **{
                        f"pooled_latent_{index:02d}": value
                        for index, value in enumerate(feature["pooled_latent"])
                    },
                }
            )

    v39_q4 = [
        row for row in sample_rows if row["deltaT_quartile"] == "Q4"
    ]
    q4_ranked = sorted(
        v39_q4, key=lambda row: row["v39_e24_point_sse_K2"], reverse=True
    )
    q4_fields = (
        "total_power_W",
        "q_cv_mean_W_m3",
        "q_cv_rms_W_m3",
        "q_max_W_m3",
        "source_present_fraction",
        "q_weighted_inverse_kz_mK_W",
        "q_weighted_local_kz_W_mK",
        "harmonic_kx_W_mK",
        "harmonic_ky_W_mK",
        "harmonic_kz_W_mK",
        "anisotropy_xy_over_z",
        "top_h_W_m2K",
        "T_inf_minus_T_bottom_K",
        "bc_top_cv_fraction",
        "bc_bottom_cv_fraction",
        "bc_side_cv_fraction",
        "source_concentration",
        "train_nn_distance_24d",
    )
    q4_audit = {
        "sample_count": len(v39_q4),
        "top_error_sample_ids": [row["sample_id"] for row in q4_ranked[:10]],
        "feature_correlations_with_v39_point_sse": {
            field: _correlation(
                [row[field] for row in v39_q4],
                [row["v39_e24_point_sse_K2"] for row in v39_q4],
            )
            for field in q4_fields
        },
        "coverage_distance_correlations": {
            "with_v39_point_sse": _correlation(
                [row["train_nn_distance_24d"] for row in v39_q4],
                [row["v39_e24_point_sse_K2"] for row in v39_q4],
            ),
            "with_v39_minus_e543_point_sse": _correlation(
                [row["train_nn_distance_24d"] for row in v39_q4],
                [row["v39_minus_e543_point_sse_delta_K2"] for row in v39_q4],
            ),
        },
        "condition_category_counts": {
            category: sum(
                row["generator_condition_category"] == category
                for row in v39_q4
            )
            for category in sorted(
                {row["generator_condition_category"] for row in v39_q4}
            )
        },
    }

    feature_cv = cv_summary["feature_sets"]
    physics_q4 = feature_cv["physics_24"]["q4_log_scale_rmse"]
    combined_q4 = feature_cv["combined_120"]["q4_log_scale_rmse"]
    coverage_corr = q4_audit["coverage_distance_correlations"][
        "with_v39_point_sse"
    ]["spearman"]
    transplant_best = min(
        (
            suites[name]["summary"]["point_global_relative_rmse_pct"],
            name,
        )
        for name in transplant_specs
    )
    e543_point = suites["e543"]["summary"]["point_global_relative_rmse_pct"]
    mlp_point = suites["e543_plus_e231_global_scale_mlp"]["summary"][
        "point_global_relative_rmse_pct"
    ]
    mlp_sample = suites["e543_plus_e231_global_scale_mlp"]["summary"][
        "sample_first_cv_relative_rmse_pct"
    ]
    e543_sample = suites["e543"]["summary"][
        "sample_first_cv_relative_rmse_pct"
    ]
    attention_point = suites["e543_plus_e231_mlp_scale_attention"]["summary"][
        "point_global_relative_rmse_pct"
    ]
    frozen_representation_is_predictive = bool(
        combined_q4 < 0.5 * physics_q4
    )
    if (
        coverage_corr is not None
        and coverage_corr > 0.5
        and combined_q4 >= 0.95 * physics_q4
    ):
        classification = "data_coverage"
        next_candidate = "coverage-targeted train augmentation before scale-head changes"
    elif (
        frozen_representation_is_predictive
        and mlp_point < e543_point
        and mlp_sample < e543_sample
        and attention_point > mlp_point
    ):
        classification = "objective"
        next_candidate = (
            "e543 frozen backbone/shape/scale-attention plus e231 global-scale-MLP "
            "initialization; train only global scale MLP with a preregistered "
            "sample-first and Q4-balanced scale objective"
        )
    else:
        classification = "scale_representation"
        next_candidate = (
            "full-graph scratch scale-head representation ablation with frozen "
            "selection metrics"
        )
    payload = {
        "schema_version": "heat3d_v5_gate6p_read_only_diagnostics_v1",
        "status": "completed_train_valid_only",
        "evaluator_commit": _git_commit(),
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "evaluation_roles": ["valid_iid"],
            "feature_readout_fit_roles": ["train"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "checkpoint_written_or_modified": False,
        },
        "split": {
            "source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "checkpoint_binding": binding,
        "normalization_and_context": {
            "normalization_recomputed_from_train_only": True,
            "context_fit_roles": ["train"],
            "context_fit_sample_count": 672,
            "context_fit_sample_ids_sha256": standardizer[
                "fit_sample_ids_sha256"
            ],
            "target_or_label_features": [],
        },
        "checkpoint_replay": {
            "e543_feature_export_max_abs_error_K": (
                feature_replay_max_abs_error_K
            ),
            "tolerance_K": 0.02,
            "passed": True,
        },
        "formulas": {
            "error_decomposition": (
                "e=s_true*(phi_hat-phi_true)+(s_hat-s_true)*phi_hat; "
                "SSE=shape_SSE+scale_SSE+2*dot(shape_term,scale_term)"
            ),
            "coverage": (
                "Euclidean nearest-neighbor distance in frozen standardized "
                "24D Global Context; train is reference, valid_iid is query"
            ),
        },
        "decomposition_max_abs_closure_K2": closure,
        "field_metrics": {
            name: {"summary": suite["summary"]} for name, suite in suites.items()
        },
        "transplant_provenance": transplants,
        "quartile_attribution": {
            comparison: _quartile_summary(sample_rows, comparison)
            for comparison in ("v39_minus_e543", "v39_minus_e231")
        },
        "top_sse_attribution": {
            comparison: _top_sse(sample_rows, comparison)
            for comparison in ("v39_minus_e543", "v39_minus_e231")
        },
        "train_only_scale_feature_cv": cv_summary,
        "q4_high_error_audit": q4_audit,
        "bottleneck_assessment": {
            "classification": classification,
            "basis": (
                f"best transplant point-global={transplant_best[0]:.6f}% "
                f"({transplant_best[1]}); physics/combined Q4 readout "
                f"RMSE={physics_q4:.6f}/{combined_q4:.6f}; "
                f"coverage Spearman={coverage_corr}; e231 MLP transplant "
                f"point/sample={mlp_point:.6f}/{mlp_sample:.6f}%; "
                f"MLP+attention point={attention_point:.6f}%"
            ),
            "next_training_candidate": next_candidate,
            "training_started": False,
        },
    }

    sample_fields = list(sample_rows[0])
    feature_fields = list(feature_rows[0])
    cv_fields = list(cv_rows[0])
    q4_output_rows = q4_ranked[:10]
    _write_json(paths["samples_json"], sample_rows)
    _write_csv(paths["samples_csv"], sample_rows, sample_fields)
    _write_csv(paths["features_csv"], feature_rows, feature_fields)
    _write_csv(paths["cv_csv"], cv_rows, cv_fields)
    _write_csv(paths["q4_csv"], q4_output_rows, list(q4_output_rows[0]))
    _write_json(paths["json"], payload)
    paths["md"].write_text(_markdown(payload), encoding="utf-8")
    manifest = {
        "schema_version": "heat3d_v5_gate6p_artifact_manifest_v1",
        "evaluator_commit": _git_commit(),
        "sample_level_row_count": len(sample_rows),
        "scale_feature_row_count": len(feature_rows),
        "train_cv_row_count": len(cv_rows),
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    _write_json(paths["manifest"], manifest)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "bottleneck": classification,
                "next_candidate": next_candidate,
                "output_dir": str(output_dir),
                "training_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
