#!/usr/bin/env python3
"""Gate 6J valid-only V13/V32 pairing and V32 inference-only alpha sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v5_global_context import fit_train_only_standardizer  # noqa: E402
from rigno.heat3d_v5_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    compute_sample_metrics,
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
from evaluate_heat3d_v5_v32_valid_only import (  # noqa: E402
    _apply_with_intermediates,
    _normalization_equal,
    _prediction_fields,
    _unwrap_intermediate,
)
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402


V13_ID = "V4P5_13_gate6e_scratch_branch_rebalance"
V32_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
V13_CHECKPOINT_SHA256 = (
    "dac34633392015d7a1752367cca5ed9cb58fdb62331c46cdf31b0105fc49923d"
)
V32_CHECKPOINT_SHA256 = (
    "f3063b53ca26a2b91fffc090ad4de98fe260ac5d7b669bcfbfd77c1fcf045d24"
)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_SEED = 2026071801
BOOTSTRAP_RESAMPLES = 20_000
EPS = 1.0e-12


class Gate6JError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v13-run-dir", type=Path, required=True)
    parser.add_argument("--v32-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--alpha-csv", type=Path, required=True)
    parser.add_argument("--strata-csv", type=Path, required=True)
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


def _flat_field(value: Any, target_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == target_shape:
        return array
    batch, _, nodes, _ = target_shape
    if array.shape == (batch, nodes):
        return array[:, None, :, None]
    if array.shape == (batch, nodes, 1):
        return array[:, None, :, :]
    if array.shape == (batch,):
        return array[:, None, None, None]
    if array.shape == (batch, 1):
        return array[:, :, None, None]
    raise Gate6JError(f"{name}: cannot coerce {array.shape} to {target_shape}")


def _dense(x: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        jnp.asarray(x, dtype=jnp.float32)
        @ jnp.asarray(params["kernel"], dtype=jnp.float32)
        + jnp.asarray(params["bias"], dtype=jnp.float32),
        dtype=np.float64,
    )


def _alpha_predictions(
    *,
    model: Any,
    params_device: Any,
    params_host: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    valid_ids: Sequence[str],
    saved_v32: Mapping[str, np.ndarray],
) -> tuple[
    dict[float, dict[str, np.ndarray]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    fields = {alpha: {} for alpha in ALPHAS}
    residual_rows: dict[str, dict[str, float]] = {}
    model_alpha1_replay_max = 0.0
    manual_alpha1_model_max = 0.0
    for group in groups:
        prediction, state = _apply_with_intermediates(
            model, params_device, group
        )
        phi = np.asarray(prediction["phi_hat"], dtype=np.float64)
        model_raw = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        target_shape = tuple(model_raw.shape)
        processed = np.asarray(
            prediction["rnodes_processed"], dtype=np.float64
        )
        mean_pool = np.mean(processed, axis=1)
        residual = _unwrap_intermediate(
            state["intermediates"]["scale_attention_residual"]
        )
        if residual.shape != mean_pool.shape:
            raise Gate6JError("attention residual/mean-pool shapes differ")
        context = np.asarray(group["global_context"], dtype=np.float64)
        physics = group["native_physics"]
        log_s_phys = np.asarray(
            physics["log_s_phys"], dtype=np.float64
        ).reshape(-1, 1)
        reference = _flat_field(
            physics["reference_temperature"], target_shape, "reference"
        )
        prescribed = _flat_field(
            physics["prescribed_temperature"], target_shape, "prescribed"
        )
        dirichlet = (
            _flat_field(
                physics["dirichlet_mask"], target_shape, "dirichlet_mask"
            )
            > 0.5
        )
        for alpha in ALPHAS:
            pooled = mean_pool + alpha * residual
            scale_features = np.concatenate([context, pooled], axis=-1)
            hidden = np.asarray(
                jax.nn.gelu(
                    jnp.asarray(
                        _dense(
                            scale_features,
                            params_host["global_scale_hidden"],
                        ),
                        dtype=jnp.float32,
                    )
                ),
                dtype=np.float64,
            )
            scale_residual = _dense(
                hidden, params_host["global_scale_output"]
            )
            scale = np.exp(log_s_phys + scale_residual)
            raw = reference + scale[:, :, None, None] * phi
            raw = np.where(dirichlet, prescribed, raw)
            for index, sample_id in enumerate(group["sample_ids"]):
                fields[alpha][str(sample_id)] = raw[index].reshape(-1)
            if alpha == 1.0:
                manual_alpha1_model_max = max(
                    manual_alpha1_model_max,
                    float(np.max(np.abs(raw - model_raw))),
                )
        for index, sample_id in enumerate(group["sample_ids"]):
            sample_id = str(sample_id)
            mean = mean_pool[index]
            res = residual[index]
            mean_norm = float(np.linalg.norm(mean))
            residual_norm = float(np.linalg.norm(res))
            cosine = float(
                np.dot(mean, res)
                / max(mean_norm * residual_norm, EPS)
            )
            residual_rows[sample_id] = {
                "attention_residual_mean_pool_cosine": cosine,
                "attention_residual_l2_norm": residual_norm,
                "mean_pool_l2_norm": mean_norm,
                "attention_residual_to_mean_pool_norm_ratio": (
                    residual_norm / max(mean_norm, EPS)
                ),
            }
            model_alpha1_replay_max = max(
                model_alpha1_replay_max,
                float(
                    np.max(
                        np.abs(
                            model_raw[index].reshape(-1)
                            - saved_v32[sample_id]
                        )
                    )
                ),
            )
    if any(set(values) != set(valid_ids) for values in fields.values()):
        raise Gate6JError("alpha sweep did not cover valid_iid exactly")
    if set(residual_rows) != set(valid_ids):
        raise Gate6JError("residual diagnostics did not cover valid_iid")
    alpha1_saved_max = max(
        float(np.max(np.abs(fields[1.0][sample_id] - saved_v32[sample_id])))
        for sample_id in valid_ids
    )
    replay = {
        "model_alpha1_vs_saved_max_abs_error_K": model_alpha1_replay_max,
        "manual_alpha1_vs_model_max_abs_error_K": manual_alpha1_model_max,
        "manual_alpha1_vs_saved_max_abs_error_K": alpha1_saved_max,
        "tolerance_K": 0.02,
        "passed": max(
            model_alpha1_replay_max,
            manual_alpha1_model_max,
            alpha1_saved_max,
        )
        <= 0.02,
    }
    if not replay["passed"]:
        raise Gate6JError(f"alpha=1 replay failed: {replay}")
    return fields, residual_rows, replay


def _targets(
    *,
    data_root: Path,
    valid_ids: Sequence[str],
    context_cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for sample_id in valid_ids:
        sample_dir = data_root / sample_id
        meta = json.loads(
            (sample_dir / "sample_meta.json").read_text(encoding="utf-8")
        )
        if meta.get("split") != "valid_iid":
            raise Gate6JError(f"{sample_id}: non-valid role encountered")
        bottom = float(
            meta["boundary_params"]["bottom"]["fixed_temperature_K"]
        )
        q = np.load(sample_dir / "q_field.npy").astype(np.float64).reshape(-1)
        coords = np.load(sample_dir / "coords.npy").astype(np.float64)
        target = (
            np.load(sample_dir / "temperature.npy")
            .astype(np.float64)
            .reshape(-1)
            - bottom
        )
        condition_categories = sorted(
            {
                str(row["DeltaT_target_bin"])
                for row in meta.get("q_block_metadata", ())
                if row.get("DeltaT_target_bin") is not None
            }
        )
        if len(condition_categories) != 1:
            raise Gate6JError(
                f"{sample_id}: expected one generator condition category"
            )
        context = context_cache[sample_id]["context"]
        result[sample_id] = {
            "target_deltaT_K": target,
            "bottom_temperature_K": bottom,
            "control_volumes_m3": control_volume_weights(coords),
            "q_W_m3": q,
            "total_power_W": float(context["P_operator_W"]),
            "source_occupancy_fraction": float(np.mean(q > EPS)),
            "q_weighted_inverse_kz_mK_W": float(
                context["q_weighted_inverse_kz_mK_W"]
            ),
            "generator_condition_category": condition_categories[0],
        }
    return result


def _metric_suite(
    fields: Mapping[str, np.ndarray],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    valid_ids: Sequence[str],
    checkpoint_stats: Mapping[str, Any],
) -> dict[str, Any]:
    mean = float(
        np.asarray(checkpoint_stats["target_delta_mean"]).reshape(-1)[0]
    )
    std = float(
        np.asarray(checkpoint_stats["target_delta_std"]).reshape(-1)[0]
    )
    samples = []
    for sample_id in valid_ids:
        target = targets[sample_id]
        prediction_delta = (
            np.asarray(fields[sample_id], dtype=np.float64)
            - target["bottom_temperature_K"]
        )
        samples.append(
            {
                "sample_id": sample_id,
                "split": "valid_iid",
                "prediction_deltaT_K": prediction_delta,
                "target_deltaT_K": target["target_deltaT_K"],
                "control_volumes_m3": target["control_volumes_m3"],
                "q_W_m3": target["q_W_m3"],
                "prediction_normalized": (prediction_delta - mean) / std,
                "target_normalized": (
                    target["target_deltaT_K"] - mean
                )
                / std,
            }
        )
    return evaluate_metric_suite(samples)


def _point_sample_relative(row: Mapping[str, Any]) -> float:
    return 100.0 * math.sqrt(
        float(row["point_error_squared_sum"])
        / float(row["point_true_squared_sum"])
    )


def _paired_rows(
    *,
    v13_suite: Mapping[str, Any],
    v32_suite: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    residual_rows: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    v13 = {row["sample_id"]: row for row in v13_suite["per_sample"]}
    v32 = {row["sample_id"]: row for row in v32_suite["per_sample"]}
    if set(v13) != set(v32) or set(v13) != set(targets):
        raise Gate6JError("paired sample IDs differ")
    result = []
    for sample_id in sorted(v13):
        left, right = v13[sample_id], v32[sample_id]
        features = targets[sample_id]
        row = {
            "sample_id": sample_id,
            "true_cv_rms_deltaT_K": float(
                left["true_scale_cv_rms_K"]
            ),
            "total_power_W": features["total_power_W"],
            "source_occupancy_fraction": features[
                "source_occupancy_fraction"
            ],
            "q_weighted_inverse_kz_mK_W": features[
                "q_weighted_inverse_kz_mK_W"
            ],
            "generator_condition_category": features[
                "generator_condition_category"
            ],
            "true_point_squared_sum_K2": float(
                left["point_true_squared_sum"]
            ),
            "cv_volume_m3": float(left["raw_cv_volume_m3"]),
        }
        row.update(residual_rows[sample_id])
        values = {
            "sample_relative_rmse_pct": (
                100.0 * float(left["sample_cv_relative_rmse"]),
                100.0 * float(right["sample_cv_relative_rmse"]),
            ),
            "sample_point_relative_rmse_pct": (
                _point_sample_relative(left),
                _point_sample_relative(right),
            ),
            "raw_cv_rmse_K": (
                float(left["raw_cv_weighted_rmse_K"]),
                float(right["raw_cv_weighted_rmse_K"]),
            ),
            "shape_cv_rmse": (
                float(left["shape_cv_rmse"]),
                float(right["shape_cv_rmse"]),
            ),
            "scale_log_abs_error": (
                abs(float(left["scale_log_error"])),
                abs(float(right["scale_log_error"])),
            ),
            "scale_log_signed_error": (
                float(left["scale_log_error"]),
                float(right["scale_log_error"]),
            ),
            "point_sse_K2": (
                float(left["point_error_squared_sum"]),
                float(right["point_error_squared_sum"]),
            ),
            "raw_cv_sse_K2_m3": (
                float(left["raw_cv_error_squared_integral_K2_m3"]),
                float(right["raw_cv_error_squared_integral_K2_m3"]),
            ),
        }
        for metric, (left_value, right_value) in values.items():
            row[f"v13_{metric}"] = left_value
            row[f"v32_{metric}"] = right_value
            row[f"v32_minus_v13_{metric}"] = right_value - left_value
        result.append(row)
    return result


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    count = len(rows)
    if count < 2:
        raise Gate6JError("paired bootstrap requires at least two samples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, count, size=(BOOTSTRAP_RESAMPLES, count)
    )

    def array(name: str, model: str) -> np.ndarray:
        return np.asarray(
            [float(row[f"{model}_{name}"]) for row in rows],
            dtype=np.float64,
        )

    truth = np.asarray(
        [float(row["true_point_squared_sum_K2"]) for row in rows],
        dtype=np.float64,
    )
    volumes = np.asarray(
        [float(row["cv_volume_m3"]) for row in rows], dtype=np.float64
    )
    v13_point, v32_point = array("point_sse_K2", "v13"), array(
        "point_sse_K2", "v32"
    )
    v13_raw, v32_raw = array("raw_cv_sse_K2_m3", "v13"), array(
        "raw_cv_sse_K2_m3", "v32"
    )
    v13_sample, v32_sample = array(
        "sample_relative_rmse_pct", "v13"
    ), array("sample_relative_rmse_pct", "v32")
    v13_shape, v32_shape = array("shape_cv_rmse", "v13"), array(
        "shape_cv_rmse", "v32"
    )
    v13_scale, v32_scale = array(
        "scale_log_signed_error", "v13"
    ), array("scale_log_signed_error", "v32")

    bootstrap_values = {
        "point_global_relative_rmse_pct": (
            100.0
            * np.sqrt(
                np.sum(v32_point[indices], axis=1)
                / np.sum(truth[indices], axis=1)
            )
            - 100.0
            * np.sqrt(
                np.sum(v13_point[indices], axis=1)
                / np.sum(truth[indices], axis=1)
            )
        ),
        "sample_first_cv_relative_rmse_pct": np.mean(
            v32_sample[indices] - v13_sample[indices], axis=1
        ),
        "raw_cv_weighted_rmse_K": (
            np.sqrt(
                np.sum(v32_raw[indices], axis=1)
                / np.sum(volumes[indices], axis=1)
            )
            - np.sqrt(
                np.sum(v13_raw[indices], axis=1)
                / np.sum(volumes[indices], axis=1)
            )
        ),
        "shape_cv_rmse": np.mean(
            v32_shape[indices] - v13_shape[indices], axis=1
        ),
        "scale_log_rmse": (
            np.sqrt(np.mean(np.square(v32_scale[indices]), axis=1))
            - np.sqrt(np.mean(np.square(v13_scale[indices]), axis=1))
        ),
    }
    observed = {
        "point_global_relative_rmse_pct": (
            100.0 * math.sqrt(float(np.sum(v32_point) / np.sum(truth)))
            - 100.0
            * math.sqrt(float(np.sum(v13_point) / np.sum(truth)))
        ),
        "sample_first_cv_relative_rmse_pct": float(
            np.mean(v32_sample - v13_sample)
        ),
        "raw_cv_weighted_rmse_K": float(
            math.sqrt(float(np.sum(v32_raw) / np.sum(volumes)))
            - math.sqrt(float(np.sum(v13_raw) / np.sum(volumes)))
        ),
        "shape_cv_rmse": float(np.mean(v32_shape - v13_shape)),
        "scale_log_rmse": float(
            math.sqrt(float(np.mean(np.square(v32_scale))))
            - math.sqrt(float(np.mean(np.square(v13_scale))))
        ),
    }
    per_sample_delta = {
        "point_global_relative_rmse_pct": array(
            "sample_point_relative_rmse_pct", "v32"
        )
        - array("sample_point_relative_rmse_pct", "v13"),
        "sample_first_cv_relative_rmse_pct": v32_sample - v13_sample,
        "raw_cv_weighted_rmse_K": array("raw_cv_rmse_K", "v32")
        - array("raw_cv_rmse_K", "v13"),
        "shape_cv_rmse": v32_shape - v13_shape,
        "scale_log_rmse": array("scale_log_abs_error", "v32")
        - array("scale_log_abs_error", "v13"),
    }
    return {
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "method": (
            "paired sample bootstrap with replacement; percentile 95% CI; "
            "delta is V32 minus V13"
        ),
        "metrics": {
            metric: {
                "observed_aggregate_difference": observed[metric],
                "bootstrap_95pct_ci": np.quantile(
                    bootstrap_values[metric], [0.025, 0.975]
                ).tolist(),
                "bootstrap_probability_v32_improves": float(
                    np.mean(bootstrap_values[metric] < 0.0)
                ),
                "per_sample_win_rate": float(
                    np.mean(per_sample_delta[metric] < 0.0)
                ),
                "per_sample_tie_rate": float(
                    np.mean(per_sample_delta[metric] == 0.0)
                ),
                "per_sample_median_difference": float(
                    np.median(per_sample_delta[metric])
                ),
                "per_sample_difference_definition": (
                    "sample point-relative RMSE"
                    if metric == "point_global_relative_rmse_pct"
                    else (
                        "absolute scale-log error"
                        if metric == "scale_log_rmse"
                        else metric
                    )
                ),
            }
            for metric in observed
        },
    }


def _subset_summary(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    inference = _bootstrap(rows, seed=seed)
    return {
        "sample_count": len(rows),
        "v13_sample_first_cv_relative_rmse_pct": float(
            np.mean([row["v13_sample_relative_rmse_pct"] for row in rows])
        ),
        "v32_sample_first_cv_relative_rmse_pct": float(
            np.mean([row["v32_sample_relative_rmse_pct"] for row in rows])
        ),
        "v32_minus_v13_sample_first_pct_points": float(
            np.mean(
                [
                    row["v32_minus_v13_sample_relative_rmse_pct"]
                    for row in rows
                ]
            )
        ),
        "sample_first_win_rate": float(
            np.mean(
                [
                    row["v32_minus_v13_sample_relative_rmse_pct"] < 0.0
                    for row in rows
                ]
            )
        ),
        "sample_first_median_difference_pct_points": float(
            np.median(
                [
                    row["v32_minus_v13_sample_relative_rmse_pct"]
                    for row in rows
                ]
            )
        ),
        "v13_point_sse_K2": float(
            sum(row["v13_point_sse_K2"] for row in rows)
        ),
        "v32_point_sse_K2": float(
            sum(row["v32_point_sse_K2"] for row in rows)
        ),
        "v32_minus_v13_point_sse_K2": float(
            sum(row["v32_minus_v13_point_sse_K2"] for row in rows)
        ),
        "paired_bootstrap": inference,
    }


def _strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes = {
        "true_cv_rms_deltaT_K": "quartile",
        "total_power_W": "quartile",
        "source_occupancy_fraction": "quartile",
        "q_weighted_inverse_kz_mK_W": "quartile",
        "generator_condition_category": "category",
    }
    result = {}
    for axis_index, (feature, kind) in enumerate(axes.items()):
        bins = []
        if kind == "quartile":
            values = np.asarray(
                [float(row[feature]) for row in rows], dtype=np.float64
            )
            edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
            assignments = np.searchsorted(
                edges[1:-1], values, side="right"
            )
            for index in range(4):
                selected = [
                    row
                    for row, assignment in zip(
                        rows, assignments, strict=True
                    )
                    if assignment == index
                ]
                bins.append(
                    {
                        "label": f"Q{index + 1}",
                        "lower": float(edges[index]),
                        "upper": float(edges[index + 1]),
                        "feature_mean": float(
                            np.mean([row[feature] for row in selected])
                        ),
                        **_subset_summary(
                            selected,
                            seed=BOOTSTRAP_SEED
                            + 100 * axis_index
                            + index
                            + 1,
                        ),
                    }
                )
            result[feature] = {
                "kind": kind,
                "edges": edges.tolist(),
                "bins": bins,
            }
        else:
            categories = sorted({str(row[feature]) for row in rows})
            for index, category in enumerate(categories):
                selected = [
                    row for row in rows if str(row[feature]) == category
                ]
                bins.append(
                    {
                        "label": category,
                        **_subset_summary(
                            selected,
                            seed=BOOTSTRAP_SEED
                            + 100 * axis_index
                            + index
                            + 1,
                        ),
                    }
                )
            result[feature] = {"kind": kind, "bins": bins}
    return result


def _correlation(
    left: Sequence[float], right: Sequence[float]
) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if (
        x.size < 3
        or np.std(x) <= EPS
        or np.std(y) <= EPS
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        return {
            "sample_count": int(x.size),
            "pearson": None,
            "pearson_p": None,
            "spearman": None,
            "spearman_p": None,
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "sample_count": int(x.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _residual_correlations(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    residual_fields = (
        "attention_residual_mean_pool_cosine",
        "attention_residual_to_mean_pool_norm_ratio",
    )
    outcomes = (
        "v32_minus_v13_shape_cv_rmse",
        "v32_minus_v13_scale_log_abs_error",
        "v32_minus_v13_sample_relative_rmse_pct",
        "v32_minus_v13_sample_point_relative_rmse_pct",
        "v32_minus_v13_raw_cv_rmse_K",
    )
    physics = (
        "true_cv_rms_deltaT_K",
        "total_power_W",
        "source_occupancy_fraction",
        "q_weighted_inverse_kz_mK_W",
    )
    return {
        residual: {
            "error_change": {
                outcome: _correlation(
                    [row[residual] for row in rows],
                    [row[outcome] for row in rows],
                )
                for outcome in outcomes
            },
            "physics_context": {
                feature: _correlation(
                    [row[residual] for row in rows],
                    [row[feature] for row in rows],
                )
                for feature in physics
            },
        }
        for residual in residual_fields
    }


def _alpha_q1_delta(
    *,
    alpha_suite: Mapping[str, Any],
    v13_suite: Mapping[str, Any],
    q1_ids: set[str],
) -> float:
    left = {
        row["sample_id"]: row for row in v13_suite["per_sample"]
    }
    right = {
        row["sample_id"]: row for row in alpha_suite["per_sample"]
    }
    return float(
        np.mean(
            [
                100.0
                * (
                    float(right[sample_id]["sample_cv_relative_rmse"])
                    - float(left[sample_id]["sample_cv_relative_rmse"])
                )
                for sample_id in sorted(q1_ids)
            ]
        )
    )


def _route(
    *,
    alpha_suites: Mapping[float, Mapping[str, Any]],
    v13_suite: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    residual_correlations: Mapping[str, Any],
) -> dict[str, Any]:
    v13 = v13_suite["summary"]
    delta_values = np.asarray(
        [row["true_cv_rms_deltaT_K"] for row in paired_rows],
        dtype=np.float64,
    )
    q1_threshold = float(np.quantile(delta_values, 0.25))
    q1_ids = {
        str(row["sample_id"])
        for row in paired_rows
        if float(row["true_cv_rms_deltaT_K"]) <= q1_threshold
    }
    rows = []
    for alpha in ALPHAS:
        summary = alpha_suites[alpha]["summary"]
        rows.append(
            {
                "alpha": alpha,
                "point_global_relative_rmse_pct": summary[
                    "point_global_relative_rmse_pct"
                ],
                "sample_first_cv_relative_rmse_pct": summary[
                    "sample_first_cv_relative_rmse_pct"
                ],
                "point_global_gain_preserved_vs_v13": (
                    summary["point_global_relative_rmse_pct"]
                    < v13["point_global_relative_rmse_pct"]
                ),
                "sample_first_recovered_vs_v13": (
                    summary["sample_first_cv_relative_rmse_pct"]
                    <= v13["sample_first_cv_relative_rmse_pct"]
                ),
                "q1_sample_first_delta_vs_v13_pct_points": (
                    _alpha_q1_delta(
                        alpha_suite=alpha_suites[alpha],
                        v13_suite=v13_suite,
                        q1_ids=q1_ids,
                    )
                ),
            }
        )
    successful = [
        row
        for row in rows
        if row["alpha"] < 1.0
        and row["point_global_gain_preserved_vs_v13"]
        and row["sample_first_recovered_vs_v13"]
    ]
    point_preserving = [
        row for row in rows if row["point_global_gain_preserved_vs_v13"]
    ]
    if successful:
        recommendation = "residual_strength_control"
        reason = (
            "at least one alpha<1 preserves point-global gain and restores "
            "sample-first relative RMSE"
        )
        sensitivity = [
            min(row["alpha"] for row in successful),
            max(row["alpha"] for row in successful),
        ]
    else:
        best_sample = min(
            point_preserving or rows,
            key=lambda row: row["sample_first_cv_relative_rmse_pct"],
        )
        sensitivity_rows = [
            row
            for row in (point_preserving or rows)
            if row["sample_first_cv_relative_rmse_pct"]
            <= best_sample["sample_first_cv_relative_rmse_pct"] + 0.25
        ]
        sensitivity = [
            min(row["alpha"] for row in sensitivity_rows),
            max(row["alpha"] for row in sensitivity_rows),
        ]
        low_delta_regressed_for_all = all(
            row["q1_sample_first_delta_vs_v13_pct_points"] > 0.0
            for row in rows
        )
        if low_delta_regressed_for_all:
            recommendation = "objective_alignment"
            reason = (
                "alpha scaling does not restore the paired low-DeltaT Q1 "
                "sample-first error"
            )
        else:
            physics_correlations = []
            for residual_payload in residual_correlations.values():
                for value in residual_payload["physics_context"].values():
                    if value["spearman"] is not None:
                        physics_correlations.append(abs(value["spearman"]))
            weak_physics = (
                not physics_correlations
                or max(physics_correlations) < 0.20
            )
            recommendation = (
                "parameter_matched_control"
                if weak_physics
                else "objective_alignment"
            )
            reason = (
                "residual diagnostics have insufficient physics correlation"
                if weak_physics
                else "alpha scaling is ineffective despite measurable physics correlation"
            )
    return {
        "decision_order": [
            "residual_strength_control",
            "objective_alignment",
            "parameter_matched_control",
        ],
        "rows": rows,
        "successful_small_alphas": [
            row["alpha"] for row in successful
        ],
        "optimal_sensitivity_interval_alpha": sensitivity,
        "recommendation": recommendation,
        "reason": reason,
        "q1_threshold_true_cv_rms_deltaT_K": q1_threshold,
        "q1_sample_count": len(q1_ids),
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fields} for row in rows]
        )


def main() -> int:
    args = _args()
    v13_run = args.v13_run_dir.resolve()
    v32_run = args.v32_run_dir.resolve()
    outputs = (
        args.output_json.resolve(),
        args.paired_csv.resolve(),
        args.alpha_csv.resolve(),
        args.strata_csv.resolve(),
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise Gate6JError("one or more output paths already exist")
    if v13_run.name != V13_ID or v32_run.name != V32_ID:
        raise Gate6JError("run/config binding failed")
    if any(run in path.parents for run in (v13_run, v32_run) for path in outputs):
        raise Gate6JError("diagnostic outputs must be outside run directories")

    v13_config = json.loads(
        (v13_run / "run_config.json").read_text(encoding="utf-8")
    )
    v32_config = json.loads(
        (v32_run / "run_config.json").read_text(encoding="utf-8")
    )
    if (
        Path(v13_config["output_dir"]).name != V13_ID
        or Path(v32_config["output_dir"]).name != V32_ID
    ):
        raise Gate6JError("run_config output binding failed")
    v13_checkpoint_path = v13_run / "params_best.pkl"
    v32_checkpoint_path = (
        v32_run / "params_best_valid_point_global.pkl"
    )
    if _sha256(v13_checkpoint_path) != V13_CHECKPOINT_SHA256:
        raise Gate6JError("V13 checkpoint hash mismatch")
    if _sha256(v32_checkpoint_path) != V32_CHECKPOINT_SHA256:
        raise Gate6JError("V32 checkpoint hash mismatch")
    v13_checkpoint = _load_params_checkpoint(v13_checkpoint_path)
    v32_checkpoint = _load_params_checkpoint(v32_checkpoint_path)
    if int(v13_checkpoint["epoch"]) != 318:
        raise Gate6JError("V13 checkpoint is not e318")
    if int(v32_checkpoint["epoch"]) != 474:
        raise Gate6JError("V32 checkpoint is not e474")
    v13_stats = dict(v13_checkpoint["train_only_normalization"])
    v32_stats = dict(v32_checkpoint["train_only_normalization"])
    if not _normalization_equal(v13_stats, v32_stats):
        raise Gate6JError("V13/V32 normalization differs")

    install_checkpoint_feature_hooks(v32_stats)
    train_examples = load_training_examples(v32_config, v32_stats)
    recomputed_stats = training_normalization_stats(
        train_examples,
        normalization_profile=str(
            v32_stats.get("normalization_profile", "legacy_zscore")
        ),
        condition_feature_transform=v32_stats.get(
            "condition_feature_transform"
        ),
        input_feature_schema=str(
            v32_stats.get("input_feature_schema", "legacy_bc_flags")
        ),
        coord_policy=str(
            v32_stats.get("coord_policy", "train_minmax_to_unit_box")
        ),
        extent_feature_policy=str(
            v32_stats.get("extent_feature_policy", "none")
        ),
        bridge_fn=runner_module._bridge_for,
    )
    if not _normalization_equal(v32_stats, recomputed_stats):
        raise Gate6JError("normalization does not reproduce from train only")
    stats = stats_from_checkpoint_payload(v32_stats, train_examples)
    sample_root = _sample_root(Path(v32_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(v32_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6JError("train/valid split count drifted")
    v13_split_ids, _, _, _ = _resolve_training_splits(
        _sample_root(Path(v13_config["subset"])),
        Path(v13_config["split_map_path"]),
    )
    if list(v13_split_ids["valid_iid"]) != valid_ids:
        raise Gate6JError("V13/V32 valid_iid sample IDs differ")

    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=v32_stats,
        boundary_mask_fallback=bool(
            v32_config.get("boundary_mask_fallback", True)
        ),
    )
    all_examples = list(train_examples) + list(valid_examples)
    if any(
        np.asarray(example.condition.coords).shape != (1024, 3)
        for example in all_examples
    ):
        raise Gate6JError("expected 1024 nodes per sample")
    cache = _physics_cache(all_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    stored_standardizer = v32_config["global_context"]["standardizer"]
    if (
        standardizer["fit_population"] != "train_only"
        or int(standardizer["fit_sample_count"]) != 672
        or standardizer["fit_sample_ids_sha256"]
        != stored_standardizer["fit_sample_ids_sha256"]
    ):
        raise Gate6JError("global context standardizer is not train-only")
    for field in ("mean", "std"):
        if not np.allclose(
            np.asarray(standardizer[field]),
            np.asarray(stored_standardizer[field]),
            rtol=1.0e-9,
            atol=1.0e-10,
        ):
            raise Gate6JError(f"global context {field} differs")

    builder = Heat3DGraphBuilder(**dict(v32_config["graph_config"]))
    groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "gate6j_valid_iid_only",
        False,
        "basic",
        int(v32_config.get("graph_seed", 0)),
        batch_size=args.prediction_batch_size,
        drop_last=False,
    )
    _attach_v5_physics(groups, cache, standardizer)
    valid_by_id = {
        example.sample_id: example for example in valid_examples
    }
    _attach_qk_region_features_to_groups(
        groups, valid_by_id, feature_version="sparse_safe_v2"
    )
    for group in groups:
        group["native_physics"] = group["v5_physics"]
        group["global_context"] = group["v5_physics"]["global_context"]

    v13_fields = _prediction_fields(
        v13_run / "best_predictions.npz", valid_ids
    )
    v32_saved_fields = _prediction_fields(
        v32_run / "point_global_best_predictions.npz", valid_ids
    )
    model_config = _resolve_decoder_bypass_model_config(
        dict(v32_checkpoint["model_config"]), v32_stats
    )
    if (
        model_config.get("scale_attention_mode") != "physics_gate"
        or model_config.get("qk_region_feature_version")
        != "sparse_safe_v2"
    ):
        raise Gate6JError("V32 attention configuration drifted")
    model = GraphNeuralOperator(**model_config)
    alpha_fields, residual_rows, alpha_replay = _alpha_predictions(
        model=model,
        params_device=_device_params(v32_checkpoint["params"]),
        params_host=v32_checkpoint["params"],
        groups=groups,
        valid_ids=valid_ids,
        saved_v32=v32_saved_fields,
    )
    targets = _targets(
        data_root=sample_root,
        valid_ids=valid_ids,
        context_cache=cache,
    )
    v13_suite = _metric_suite(
        v13_fields,
        targets=targets,
        valid_ids=valid_ids,
        checkpoint_stats=v13_stats,
    )
    v32_suite = _metric_suite(
        v32_saved_fields,
        targets=targets,
        valid_ids=valid_ids,
        checkpoint_stats=v32_stats,
    )
    alpha_suites = {
        alpha: _metric_suite(
            alpha_fields[alpha],
            targets=targets,
            valid_ids=valid_ids,
            checkpoint_stats=v32_stats,
        )
        for alpha in ALPHAS
    }
    paired = _paired_rows(
        v13_suite=v13_suite,
        v32_suite=v32_suite,
        targets=targets,
        residual_rows=residual_rows,
    )
    paired_inference = _bootstrap(paired, seed=BOOTSTRAP_SEED)
    stratified = _strata(paired)
    residual_correlation = _residual_correlations(paired)
    route = _route(
        alpha_suites=alpha_suites,
        v13_suite=v13_suite,
        paired_rows=paired,
        residual_correlations=residual_correlation,
    )

    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    payload = {
        "schema_version": "heat3d_v5_gate6j_valid_causal_v1",
        "status": "completed_valid_iid_only",
        "evaluator_commit": _git_commit(),
        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
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
            "checkpoints_modified": False,
            "checkpoint_selection_performed": False,
            "sample_count": 128,
            "nodes_per_sample": 1024,
        },
        "split": {
            "source": split_source,
            "train_count": 672,
            "valid_iid_count": 128,
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "normalization_and_context": {
            "normalization_recomputed_from_train_only": True,
            "context_recomputed_from_train_only": True,
            "fit_roles": ["train"],
            "fit_sample_count": 672,
            "fit_sample_ids_sha256": standardizer[
                "fit_sample_ids_sha256"
            ],
            "target_or_label_features": [],
        },
        "models": {
            "v13": {
                "config_id": V13_ID,
                "checkpoint": "params_best.pkl",
                "epoch": 318,
                "sha256": V13_CHECKPOINT_SHA256,
                "training_commit": str(
                    v13_checkpoint.get("git_commit") or ""
                ),
                "prediction_artifact": "best_predictions.npz",
                "metrics": {
                    "summary": v13_suite["summary"],
                    "per_sample": v13_suite["per_sample"],
                },
            },
            "v32": {
                "config_id": V32_ID,
                "checkpoint": "params_best_valid_point_global.pkl",
                "epoch": 474,
                "sha256": V32_CHECKPOINT_SHA256,
                "training_commit": str(
                    v32_checkpoint.get("git_commit") or ""
                ),
                "prediction_artifact": (
                    "point_global_best_predictions.npz"
                ),
                "metrics": {
                    "summary": v32_suite["summary"],
                    "per_sample": v32_suite["per_sample"],
                },
            },
        },
        "paired_bootstrap": paired_inference,
        "paired_samples": paired,
        "stratified_paired_analysis": stratified,
        "attention_residual_analysis": {
            "definition": (
                "pool(alpha)=mean_pool+alpha*attention_residual"
            ),
            "correlations": residual_correlation,
        },
        "alpha_sweep": {
            "alphas": list(ALPHAS),
            "inference_only": True,
            "checkpoint_fixed": "V32 e474",
            "checkpoint_selection_performed": False,
            "replay": alpha_replay,
            "metrics": {
                str(alpha): {
                    "summary": alpha_suites[alpha]["summary"],
                    "per_sample": alpha_suites[alpha]["per_sample"],
                }
                for alpha in ALPHAS
            },
        },
        "route_decision": route,
        "artifacts": {
            "v13_checkpoint": {
                "path": str(v13_checkpoint_path),
                "sha256": _sha256(v13_checkpoint_path),
            },
            "v13_predictions": {
                "path": str(v13_run / "best_predictions.npz"),
                "sha256": _sha256(v13_run / "best_predictions.npz"),
            },
            "v32_checkpoint": {
                "path": str(v32_checkpoint_path),
                "sha256": _sha256(v32_checkpoint_path),
            },
            "v32_predictions": {
                "path": str(
                    v32_run / "point_global_best_predictions.npz"
                ),
                "sha256": _sha256(
                    v32_run / "point_global_best_predictions.npz"
                ),
            },
        },
    }
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paired_fields = list(paired[0])
    _write_csv(args.paired_csv.resolve(), paired, paired_fields)
    alpha_rows = []
    for alpha in ALPHAS:
        route_row = next(
            row for row in route["rows"] if row["alpha"] == alpha
        )
        alpha_rows.append(
            {
                "alpha": alpha,
                **{
                    key: value
                    for key, value in alpha_suites[alpha][
                        "summary"
                    ].items()
                    if isinstance(value, (int, float))
                },
                **{
                    key: value
                    for key, value in route_row.items()
                    if key != "alpha"
                },
            }
        )
    _write_csv(
        args.alpha_csv.resolve(),
        alpha_rows,
        list(alpha_rows[0]),
    )
    strata_rows = []
    for feature, analysis in stratified.items():
        for row in analysis["bins"]:
            bootstrap = row["paired_bootstrap"]["metrics"]
            strata_rows.append(
                {
                    "feature": feature,
                    "kind": analysis["kind"],
                    "bin": row["label"],
                    "lower": row.get("lower", ""),
                    "upper": row.get("upper", ""),
                    "sample_count": row["sample_count"],
                    "v13_sample_first_cv_relative_rmse_pct": row[
                        "v13_sample_first_cv_relative_rmse_pct"
                    ],
                    "v32_sample_first_cv_relative_rmse_pct": row[
                        "v32_sample_first_cv_relative_rmse_pct"
                    ],
                    "v32_minus_v13_sample_first_pct_points": row[
                        "v32_minus_v13_sample_first_pct_points"
                    ],
                    "sample_first_win_rate": row[
                        "sample_first_win_rate"
                    ],
                    "sample_first_median_difference_pct_points": row[
                        "sample_first_median_difference_pct_points"
                    ],
                    "sample_first_bootstrap_ci_low": bootstrap[
                        "sample_first_cv_relative_rmse_pct"
                    ]["bootstrap_95pct_ci"][0],
                    "sample_first_bootstrap_ci_high": bootstrap[
                        "sample_first_cv_relative_rmse_pct"
                    ]["bootstrap_95pct_ci"][1],
                    "v32_minus_v13_point_sse_K2": row[
                        "v32_minus_v13_point_sse_K2"
                    ],
                    "point_global_bootstrap_ci_low": bootstrap[
                        "point_global_relative_rmse_pct"
                    ]["bootstrap_95pct_ci"][0],
                    "point_global_bootstrap_ci_high": bootstrap[
                        "point_global_relative_rmse_pct"
                    ]["bootstrap_95pct_ci"][1],
                }
            )
    _write_csv(
        args.strata_csv.resolve(),
        strata_rows,
        list(strata_rows[0]),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_json": str(args.output_json.resolve()),
                "v13": v13_suite["summary"],
                "v32": v32_suite["summary"],
                "alpha_rows": alpha_rows,
                "route_decision": route,
                "alpha_replay": alpha_replay,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
