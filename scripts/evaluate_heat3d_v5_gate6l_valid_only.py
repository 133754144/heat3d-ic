#!/usr/bin/env python3
"""Gate 6L frozen-formula valid_iid evaluation for O075, Dual, and V32."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v5_global_context import fit_train_only_standardizer  # noqa: E402
from rigno.heat3d_v5_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    REQUIRED_SUMMARY_FIELDS,
    control_volume_weights,
    evaluate_metric_suite,
    summarize_metric_rows,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    _attach_qk_region_features_to_groups,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _model_apply,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
    _tree_max_abs_difference,
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
from evaluate_heat3d_v5_v32_valid_only import _normalization_equal  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402


CONFIGS = {
    "O075": "V4P5_33_gate6k_o075_log_scale",
    "Dual": "V4P5_34_gate6k_dual_physics_attention",
}
V32_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
CHECKPOINTS = {
    "point_global_best": (
        "params_best_valid_point_global.pkl",
        "point_global_best_predictions.npz",
        "point_global_best",
        "point_global_best_epoch",
    ),
    "sample_first_best": (
        "params_best_valid_sample_first.pkl",
        "sample_first_best_predictions.npz",
        "sample_first_best",
        "sample_first_best_epoch",
    ),
    "legacy_best": (
        "params_best_valid_base_mse.pkl",
        "base_mse_best_predictions.npz",
        "base_mse_best",
        "base_mse_best_epoch",
    ),
    "final": ("params_final.pkl", "predictions.npz", "final", "final_epoch"),
}
V32_CHECKPOINT = "params_best_valid_point_global.pkl"
V32_PREDICTIONS = "point_global_best_predictions.npz"
BOOTSTRAP_SEED = 2026071802
BOOTSTRAP_RESAMPLES = 20_000
EPS = 1.0e-12
PAIR_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "shape_cv_rmse",
    "scale_log_rmse",
)


class Gate6LError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o075-run-dir", type=Path, required=True)
    parser.add_argument("--dual-run-dir", type=Path, required=True)
    parser.add_argument("--v32-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fields} for row in rows]
        )


def _prediction_fields(path: Path, ids: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ids):
            raise Gate6LError(f"{path}: prediction keys differ from valid_iid")
        fields = {
            sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            for sample_id in ids
        }
    if any(
        field.size != 1024 or not np.all(np.isfinite(field))
        for field in fields.values()
    ):
        raise Gate6LError(f"{path}: invalid prediction fields")
    return fields


def _run_binding(run_dir: Path, config_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if run_dir.name != config_id:
        raise Gate6LError(f"{run_dir}: run/config directory binding failed")
    run_config = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "loss_summary.json").read_text(encoding="utf-8")
    )
    if Path(run_config["output_dir"]).name != config_id:
        raise Gate6LError(f"{config_id}: run_config output binding failed")
    epochs = [int(row["epoch"]) for row in summary.get("epoch_history", ())]
    if (
        int(summary.get("final_epoch", -1)) != 600
        or epochs != list(range(1, 601))
        or not bool(summary.get("grad_finite"))
    ):
        raise Gate6LError(f"{config_id}: e600 completion audit failed")
    if str(summary.get("code_version_or_git_commit")) != "461d810":
        raise Gate6LError(f"{config_id}: unexpected training commit")
    return run_config, summary


def _training_reload_rows(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    audit = summary.get("checkpoint_prediction_reload_audit") or {}
    if audit.get("status") != "passed":
        raise Gate6LError("training checkpoint reload audit did not pass")
    rows = {str(row["label"]): row for row in audit.get("entries", ())}
    required = {"point_global_best", "sample_first_best", "base_mse_best", "final"}
    if not required <= set(rows) or any(not bool(rows[name]["passed"]) for name in required):
        raise Gate6LError("training reload audit is incomplete")
    return rows


def _checkpoint_metadata(
    *,
    checkpoint_path: Path,
    checkpoint_name: str,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_params_checkpoint(checkpoint_path)
    second = _load_params_checkpoint(checkpoint_path)
    parameter_reload = _tree_max_abs_difference(payload["params"], second["params"])
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
        raise Gate6LError(f"{checkpoint_path}: checkpoint epoch mismatch")
    training_label = {
        "point_global_best": "point_global_best",
        "sample_first_best": "sample_first_best",
        "legacy_best": "base_mse_best",
        "final": "final",
    }[checkpoint_name]
    training_reload = _training_reload_rows(summary)[training_label]
    leaves = [
        np.asarray(value)
        for value in __import__("jax").tree_util.tree_leaves(payload["params"])
    ]
    metadata = {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "epoch": expected_epoch,
        "checkpoint_kind": str(payload.get("checkpoint_kind") or ""),
        "training_commit": str(payload.get("git_commit") or ""),
        "parameter_count": int(sum(value.size for value in leaves)),
        "parameter_leaf_count": len(leaves),
        "parameter_reload_max_abs_error": float(parameter_reload),
        "training_reload_audit": {
            "label": training_label,
            "passed": bool(training_reload["passed"]),
            "checkpoint_reload_max_abs_error_K": float(
                training_reload["checkpoint_reload_max_abs_error_K"]
            ),
            "tolerance_K": float(training_reload["tolerance_K"]),
        },
    }
    if parameter_reload != 0.0:
        raise Gate6LError(f"{checkpoint_path}: parameter reload is not exact")
    return payload, metadata


def _targets(
    *,
    sample_root: Path,
    valid_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    for sample_id in valid_ids:
        sample_dir = sample_root / sample_id
        meta = json.loads(
            (sample_dir / "sample_meta.json").read_text(encoding="utf-8")
        )
        if meta.get("split") != "valid_iid":
            raise Gate6LError(f"{sample_id}: forbidden role encountered")
        categories = sorted(
            {
                str(item["DeltaT_target_bin"])
                for item in meta.get("q_block_metadata", ())
                if item.get("DeltaT_target_bin") is not None
            }
        )
        if len(categories) != 1:
            raise Gate6LError(f"{sample_id}: condition category is ambiguous")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        coords = np.load(sample_dir / "coords.npy").astype(np.float64)
        target = (
            np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1)
            - bottom
        )
        volumes = control_volume_weights(coords)
        true_scale = float(
            math.sqrt(np.sum(np.square(target) * volumes) / np.sum(volumes))
        )
        result[sample_id] = {
            "bottom_temperature_K": bottom,
            "target_deltaT_K": target,
            "control_volumes_m3": volumes,
            "q_W_m3": np.load(sample_dir / "q_field.npy")
            .astype(np.float64)
            .reshape(-1),
            "true_scale_cv_rms_K": true_scale,
            "generator_condition_category": categories[0],
        }
    scales = np.asarray(
        [result[sample_id]["true_scale_cv_rms_K"] for sample_id in valid_ids],
        dtype=np.float64,
    )
    q25, q50, q75 = np.quantile(scales, [0.25, 0.50, 0.75])
    for sample_id in valid_ids:
        value = float(result[sample_id]["true_scale_cv_rms_K"])
        result[sample_id]["deltaT_quartile"] = (
            "Q1"
            if value <= q25
            else "Q2"
            if value <= q50
            else "Q3"
            if value <= q75
            else "Q4"
        )
        result[sample_id]["is_nominal_to_hard"] = (
            result[sample_id]["generator_condition_category"] == "nominal_to_hard"
        )
    return result


def _metric_suite(
    *,
    prediction_path: Path,
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = _prediction_fields(prediction_path, ids)
    mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    samples = []
    for sample_id in ids:
        target = targets[sample_id]
        prediction = (
            predictions[sample_id] - float(target["bottom_temperature_K"])
        )
        true = np.asarray(target["target_deltaT_K"], dtype=np.float64)
        samples.append(
            {
                "sample_id": sample_id,
                "split": "valid_iid",
                "prediction_deltaT_K": prediction,
                "target_deltaT_K": true,
                "control_volumes_m3": target["control_volumes_m3"],
                "q_W_m3": target["q_W_m3"],
                "prediction_normalized": (prediction - mean) / std,
                "target_normalized": (true - mean) / std,
            }
        )
    return evaluate_metric_suite(samples)


def _build_groups(
    *,
    run_config: Mapping[str, Any],
    stats: Mapping[str, Any],
    train_examples: Sequence[Any],
    valid_examples: Sequence[Any],
    valid_ids: Sequence[str],
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    groups = _make_groups_with_progress(
        list(valid_examples),
        stats,
        builder,
        "gate6l_valid_iid_only",
        False,
        "basic",
        int(run_config.get("graph_seed", 0)),
        batch_size=batch_size,
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
    if [str(value) for group in groups for value in group["sample_ids"]] != list(
        valid_ids
    ):
        raise Gate6LError("group order differs from valid_iid split")
    return groups


def _replay_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    saved_prediction_path: Path,
    valid_ids: Sequence[str],
) -> dict[str, Any]:
    checkpoint_stats = dict(checkpoint["train_only_normalization"])
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), checkpoint_stats
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    saved = _prediction_fields(saved_prediction_path, valid_ids)
    maximum = 0.0
    sample_count = 0
    for group in groups:
        prediction = _model_apply(model, params, group)
        raw = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        for index, sample_id in enumerate(group["sample_ids"]):
            sample_id = str(sample_id)
            maximum = max(
                maximum,
                float(
                    np.max(np.abs(raw[index].reshape(-1) - saved[sample_id]))
                ),
            )
            sample_count += 1
    payload = {
        "sample_count": sample_count,
        "max_abs_error_K": maximum,
        "tolerance_K": 0.02,
        "passed": bool(sample_count == 128 and maximum <= 0.02),
    }
    if not payload["passed"]:
        raise Gate6LError(f"checkpoint replay failed: {payload}")
    return payload


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise Gate6LError("distribution requires finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _strata(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(row["sample_id"]): dict(row) for row in rows}
    signed = np.asarray(
        [float(by_id[sample_id]["scale_log_error"]) for sample_id in sorted(by_id)],
        dtype=np.float64,
    )
    signed_p10, signed_p90 = np.quantile(signed, [0.10, 0.90])
    abs_p90 = float(np.quantile(np.abs(signed), 0.90))
    memberships: dict[str, set[str]] = {
        quartile: {
            sample_id
            for sample_id in by_id
            if targets[sample_id]["deltaT_quartile"] == quartile
        }
        for quartile in ("Q1", "Q2", "Q3", "Q4")
    }
    memberships["nominal_to_hard"] = {
        sample_id
        for sample_id in by_id
        if targets[sample_id]["is_nominal_to_hard"]
    }
    memberships["Q2_intersection_nominal_to_hard"] = (
        memberships["Q2"] & memberships["nominal_to_hard"]
    )
    memberships["scale_signed_low_p10"] = {
        sample_id
        for sample_id, row in by_id.items()
        if float(row["scale_log_error"]) <= signed_p10
    }
    memberships["scale_signed_central_p10_p90"] = {
        sample_id
        for sample_id, row in by_id.items()
        if signed_p10 < float(row["scale_log_error"]) < signed_p90
    }
    memberships["scale_signed_high_p90"] = {
        sample_id
        for sample_id, row in by_id.items()
        if float(row["scale_log_error"]) >= signed_p90
    }
    memberships["scale_abs_error_top10pct"] = {
        sample_id
        for sample_id, row in by_id.items()
        if abs(float(row["scale_log_error"])) >= abs_p90
    }
    payload = {}
    flat_rows = []
    for name, ids in memberships.items():
        selected = [by_id[sample_id] for sample_id in sorted(ids)]
        if not selected:
            raise Gate6LError(f"empty stratum: {name}")
        summary = summarize_metric_rows(selected)
        scale_values = [float(row["scale_log_error"]) for row in selected]
        payload[name] = {
            "sample_count": len(selected),
            "metrics": summary,
            "signed_scale_log_error": _distribution(scale_values),
            "sample_ids": sorted(ids),
        }
        flat = {
            "stratum": name,
            "sample_count": len(selected),
            **{
                metric: summary[metric]
                for metric in REQUIRED_SUMMARY_FIELDS
            },
            "signed_scale_error_mean": float(np.mean(scale_values)),
            "signed_scale_error_median": float(np.median(scale_values)),
        }
        flat_rows.append(flat)
    return {
        "thresholds": {
            "signed_scale_error_p10": float(signed_p10),
            "signed_scale_error_p90": float(signed_p90),
            "absolute_scale_error_p90": abs_p90,
        },
        "reports": payload,
    }, flat_rows


def _per_sample_point_relative(row: Mapping[str, Any]) -> float:
    return 100.0 * math.sqrt(
        float(row["point_error_squared_sum"])
        / max(float(row["point_true_squared_sum"]), EPS)
    )


def _paired_rows(
    *,
    left_name: str,
    right_name: str,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left = {str(row["sample_id"]): row for row in left_rows}
    right = {str(row["sample_id"]): row for row in right_rows}
    if set(left) != set(right) or set(left) != set(targets):
        raise Gate6LError("paired sample IDs differ")
    result = []
    for sample_id in sorted(left):
        lrow, rrow = left[sample_id], right[sample_id]
        base = {
            "pair": f"{right_name}_minus_{left_name}",
            "sample_id": sample_id,
            "deltaT_quartile": targets[sample_id]["deltaT_quartile"],
            "generator_condition_category": targets[sample_id][
                "generator_condition_category"
            ],
            "is_nominal_to_hard": targets[sample_id]["is_nominal_to_hard"],
            "true_scale_cv_rms_K": targets[sample_id]["true_scale_cv_rms_K"],
            "point_true_squared_sum_K2": float(lrow["point_true_squared_sum"]),
            "cv_volume_m3": float(lrow["raw_cv_volume_m3"]),
        }
        values = {
            "sample_point_relative_rmse_pct": (
                _per_sample_point_relative(lrow),
                _per_sample_point_relative(rrow),
            ),
            "sample_first_cv_relative_rmse_pct": (
                100.0 * float(lrow["sample_cv_relative_rmse"]),
                100.0 * float(rrow["sample_cv_relative_rmse"]),
            ),
            "raw_cv_rmse_K": (
                float(lrow["raw_cv_weighted_rmse_K"]),
                float(rrow["raw_cv_weighted_rmse_K"]),
            ),
            "shape_cv_rmse": (
                float(lrow["shape_cv_rmse"]),
                float(rrow["shape_cv_rmse"]),
            ),
            "absolute_scale_log_error": (
                abs(float(lrow["scale_log_error"])),
                abs(float(rrow["scale_log_error"])),
            ),
            "signed_scale_log_error": (
                float(lrow["scale_log_error"]),
                float(rrow["scale_log_error"]),
            ),
            "point_sse_K2": (
                float(lrow["point_error_squared_sum"]),
                float(rrow["point_error_squared_sum"]),
            ),
            "raw_cv_sse_K2_m3": (
                float(lrow["raw_cv_error_squared_integral_K2_m3"]),
                float(rrow["raw_cv_error_squared_integral_K2_m3"]),
            ),
        }
        for metric, (left_value, right_value) in values.items():
            base[f"left_{metric}"] = left_value
            base[f"right_{metric}"] = right_value
            base[f"delta_{metric}"] = right_value - left_value
        result.append(base)
    return result


def _bootstrap_pair(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    count = len(rows)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(BOOTSTRAP_RESAMPLES, count))

    def values(prefix: str, field: str) -> np.ndarray:
        return np.asarray(
            [float(row[f"{prefix}_{field}"]) for row in rows],
            dtype=np.float64,
        )

    truth = np.asarray(
        [float(row["point_true_squared_sum_K2"]) for row in rows],
        dtype=np.float64,
    )
    volumes = np.asarray(
        [float(row["cv_volume_m3"]) for row in rows], dtype=np.float64
    )
    left_point = values("left", "point_sse_K2")
    right_point = values("right", "point_sse_K2")
    left_sample = values("left", "sample_first_cv_relative_rmse_pct")
    right_sample = values("right", "sample_first_cv_relative_rmse_pct")
    left_raw = values("left", "raw_cv_sse_K2_m3")
    right_raw = values("right", "raw_cv_sse_K2_m3")
    left_shape = values("left", "shape_cv_rmse")
    right_shape = values("right", "shape_cv_rmse")
    left_scale = values("left", "signed_scale_log_error")
    right_scale = values("right", "signed_scale_log_error")
    boot = {
        "point_global_relative_rmse_pct": (
            100.0
            * np.sqrt(np.sum(right_point[indices], axis=1) / np.sum(truth[indices], axis=1))
            - 100.0
            * np.sqrt(np.sum(left_point[indices], axis=1) / np.sum(truth[indices], axis=1))
        ),
        "sample_first_cv_relative_rmse_pct": np.mean(
            right_sample[indices] - left_sample[indices], axis=1
        ),
        "raw_cv_weighted_rmse_K": (
            np.sqrt(np.sum(right_raw[indices], axis=1) / np.sum(volumes[indices], axis=1))
            - np.sqrt(np.sum(left_raw[indices], axis=1) / np.sum(volumes[indices], axis=1))
        ),
        "shape_cv_rmse": np.mean(
            right_shape[indices] - left_shape[indices], axis=1
        ),
        "scale_log_rmse": (
            np.sqrt(np.mean(np.square(right_scale[indices]), axis=1))
            - np.sqrt(np.mean(np.square(left_scale[indices]), axis=1))
        ),
    }
    observed = {
        "point_global_relative_rmse_pct": (
            100.0 * math.sqrt(float(np.sum(right_point) / np.sum(truth)))
            - 100.0 * math.sqrt(float(np.sum(left_point) / np.sum(truth)))
        ),
        "sample_first_cv_relative_rmse_pct": float(
            np.mean(right_sample - left_sample)
        ),
        "raw_cv_weighted_rmse_K": float(
            math.sqrt(float(np.sum(right_raw) / np.sum(volumes)))
            - math.sqrt(float(np.sum(left_raw) / np.sum(volumes)))
        ),
        "shape_cv_rmse": float(np.mean(right_shape - left_shape)),
        "scale_log_rmse": float(
            math.sqrt(float(np.mean(np.square(right_scale))))
            - math.sqrt(float(np.mean(np.square(left_scale))))
        ),
    }
    sample_deltas = {
        "point_global_relative_rmse_pct": values(
            "right", "sample_point_relative_rmse_pct"
        )
        - values("left", "sample_point_relative_rmse_pct"),
        "sample_first_cv_relative_rmse_pct": right_sample - left_sample,
        "raw_cv_weighted_rmse_K": values("right", "raw_cv_rmse_K")
        - values("left", "raw_cv_rmse_K"),
        "shape_cv_rmse": right_shape - left_shape,
        "scale_log_rmse": values("right", "absolute_scale_log_error")
        - values("left", "absolute_scale_log_error"),
    }
    metrics = {}
    for metric in PAIR_METRICS:
        metrics[metric] = {
            "observed_difference": observed[metric],
            "bootstrap_95pct_ci": np.quantile(
                boot[metric], [0.025, 0.975]
            ).tolist(),
            "bootstrap_probability_right_improves": float(
                np.mean(boot[metric] < 0.0)
            ),
            "per_sample_win_rate": float(np.mean(sample_deltas[metric] < 0.0)),
            "per_sample_median_difference": float(
                np.median(sample_deltas[metric])
            ),
        }
    point_delta = right_point - left_point
    order_improve = np.argsort(point_delta)
    order_regress = np.argsort(point_delta)[::-1]
    abs_total = float(np.sum(np.abs(point_delta)))
    q4 = np.asarray(
        [row["deltaT_quartile"] == "Q4" for row in rows], dtype=bool
    )
    scale_tail_threshold = float(
        np.quantile(values("left", "absolute_scale_log_error"), 0.90)
    )
    scale_tail = (
        values("left", "absolute_scale_log_error") >= scale_tail_threshold
    )
    return {
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "difference_orientation": "right_minus_left; negative is improvement",
        "metrics": metrics,
        "tail_contribution": {
            "point_sse_total_delta_K2": float(np.sum(point_delta)),
            "point_sse_absolute_delta_sum_K2": abs_total,
            "top5_improvement_sample_ids": [
                str(rows[index]["sample_id"]) for index in order_improve[:5]
            ],
            "top5_regression_sample_ids": [
                str(rows[index]["sample_id"]) for index in order_regress[:5]
            ],
            "top5_improvement_net_delta_K2": float(
                np.sum(point_delta[order_improve[:5]])
            ),
            "top5_regression_net_delta_K2": float(
                np.sum(point_delta[order_regress[:5]])
            ),
            "top10_improvement_absolute_fraction": float(
                np.sum(np.abs(point_delta[order_improve[:10]]))
                / max(abs_total, EPS)
            ),
            "top10_regression_absolute_fraction": float(
                np.sum(np.abs(point_delta[order_regress[:10]]))
                / max(abs_total, EPS)
            ),
            "deltaT_Q4_net_delta_K2": float(np.sum(point_delta[q4])),
            "deltaT_Q4_absolute_fraction": float(
                np.sum(np.abs(point_delta[q4])) / max(abs_total, EPS)
            ),
            "left_model_scale_abs_error_p90": scale_tail_threshold,
            "left_model_scale_tail_sample_count": int(np.sum(scale_tail)),
            "left_model_scale_tail_net_delta_K2": float(
                np.sum(point_delta[scale_tail])
            ),
            "left_model_scale_tail_absolute_fraction": float(
                np.sum(np.abs(point_delta[scale_tail])) / max(abs_total, EPS)
            ),
        },
    }


def _pruned_loss(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "code_version_or_git_commit",
        "best_valid_base_mse",
        "final_valid_base_mse",
        "best_valid_iid_loss",
        "final_valid_iid_loss",
        "best_valid_iid_raw_deltaT_rmse_K",
        "final_valid_iid_raw_deltaT_rmse_K",
        "best_valid_iid_recovered_T_rmse_K",
        "final_valid_iid_recovered_T_rmse_K",
        "best_valid_iid_relative_rmse_pct_v4",
        "final_valid_iid_relative_rmse_pct_v4",
        "final_valid_loss_components",
    )
    return {key: summary.get(key) for key in keys}


def main() -> int:
    args = _args()
    output_dir = args.output_dir.resolve()
    output_paths = {
        "combined": output_dir / "gate6l_valid_only_evaluation.json",
        "checkpoints": output_dir / "gate6l_checkpoint_comparison.csv",
        "paired": output_dir / "gate6l_paired_samples.csv",
        "bootstrap": output_dir / "gate6l_paired_bootstrap.csv",
        "strata": output_dir / "gate6l_strata.csv",
        "o075": output_dir / "gate6l_o075_valid_only_metrics.json",
        "dual": output_dir / "gate6l_dual_valid_only_metrics.json",
    }
    if not args.overwrite and any(path.exists() for path in output_paths.values()):
        raise Gate6LError("one or more Gate 6L outputs already exist")

    run_dirs = {
        "O075": args.o075_run_dir.resolve(),
        "Dual": args.dual_run_dir.resolve(),
    }
    v32_run = args.v32_run_dir.resolve()
    for run_dir in (*run_dirs.values(), v32_run):
        if run_dir in output_dir.parents or run_dir == output_dir:
            raise Gate6LError("evaluation output overlaps a frozen run directory")
    run_data = {
        model: _run_binding(run_dirs[model], CONFIGS[model])
        for model in CONFIGS
    }
    v32_config = json.loads(
        (v32_run / "run_config.json").read_text(encoding="utf-8")
    )
    if v32_run.name != V32_ID or Path(v32_config["output_dir"]).name != V32_ID:
        raise Gate6LError("V32 run binding failed")

    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for model in CONFIGS:
        checkpoint_payloads[model] = {}
        checkpoint_metadata[model] = {}
        summary = run_data[model][1]
        for checkpoint, (filename, _, _, _) in CHECKPOINTS.items():
            payload, metadata = _checkpoint_metadata(
                checkpoint_path=run_dirs[model] / filename,
                checkpoint_name=checkpoint,
                summary=summary,
            )
            checkpoint_payloads[model][checkpoint] = payload
            checkpoint_metadata[model][checkpoint] = metadata

    canonical_stats = dict(
        checkpoint_payloads["O075"]["legacy_best"]["train_only_normalization"]
    )
    for model in CONFIGS:
        for checkpoint in CHECKPOINTS:
            stats = checkpoint_payloads[model][checkpoint][
                "train_only_normalization"
            ]
            if not _normalization_equal(canonical_stats, stats):
                raise Gate6LError("checkpoint normalization drifted")
    v32_checkpoint = _load_params_checkpoint(v32_run / V32_CHECKPOINT)
    if not _normalization_equal(
        canonical_stats, v32_checkpoint["train_only_normalization"]
    ):
        raise Gate6LError("V32 normalization differs")

    install_checkpoint_feature_hooks(canonical_stats)
    canonical_run_config = run_data["O075"][0]
    train_examples = load_training_examples(canonical_run_config, canonical_stats)
    recomputed_stats = training_normalization_stats(
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
    if not _normalization_equal(canonical_stats, recomputed_stats):
        raise Gate6LError("normalization does not reproduce from train only")
    stats = stats_from_checkpoint_payload(canonical_stats, train_examples)
    sample_root = _sample_root(Path(canonical_run_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(canonical_run_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6LError("split count drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=canonical_stats,
        boundary_mask_fallback=bool(
            canonical_run_config.get("boundary_mask_fallback", True)
        ),
    )
    all_examples = list(train_examples) + list(valid_examples)
    if any(
        np.asarray(example.condition.coords).shape != (1024, 3)
        for example in all_examples
    ):
        raise Gate6LError("expected 1024 nodes per sample")
    cache = _physics_cache(all_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    for model, (run_config, _) in run_data.items():
        stored = run_config["global_context"]["standardizer"]
        if (
            stored["fit_population"] != "train_only"
            or int(stored["fit_sample_count"]) != 672
            or stored["fit_sample_ids_sha256"]
            != standardizer["fit_sample_ids_sha256"]
        ):
            raise Gate6LError(f"{model}: context standardizer drifted")
    groups = _build_groups(
        run_config=canonical_run_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=valid_examples,
        valid_ids=valid_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    targets = _targets(sample_root=sample_root, valid_ids=valid_ids)

    metric_suites: dict[str, dict[str, Any]] = {}
    strata_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_csv = []
    strata_csv = []
    replay_audits = {}
    for model in CONFIGS:
        metric_suites[model] = {}
        strata_payloads[model] = {}
        replay_audits[model] = {}
        for checkpoint, (_, prediction_name, _, _) in CHECKPOINTS.items():
            suite = _metric_suite(
                prediction_path=run_dirs[model] / prediction_name,
                ids=valid_ids,
                targets=targets,
                stats=canonical_stats,
            )
            metric_suites[model][checkpoint] = suite
            strata, flat_strata = _strata(suite["per_sample"], targets)
            strata_payloads[model][checkpoint] = strata
            for row in flat_strata:
                strata_csv.append(
                    {"model": model, "checkpoint": checkpoint, **row}
                )
            replay = _replay_checkpoint(
                checkpoint=checkpoint_payloads[model][checkpoint],
                groups=groups,
                saved_prediction_path=run_dirs[model] / prediction_name,
                valid_ids=valid_ids,
            )
            replay_audits[model][checkpoint] = replay
            metadata = checkpoint_metadata[model][checkpoint]
            checkpoint_csv.append(
                {
                    "model": model,
                    "config_id": CONFIGS[model],
                    "checkpoint": checkpoint,
                    "epoch": metadata["epoch"],
                    "sha256": metadata["sha256"],
                    "parameter_count": metadata["parameter_count"],
                    "training_commit": metadata["training_commit"],
                    "training_reload_max_abs_error_K": metadata[
                        "training_reload_audit"
                    ]["checkpoint_reload_max_abs_error_K"],
                    "evaluator_replay_max_abs_error_K": replay["max_abs_error_K"],
                    **{
                        field: suite["summary"][field]
                        for field in REQUIRED_SUMMARY_FIELDS
                    },
                }
            )

    v32_suite = _metric_suite(
        prediction_path=v32_run / V32_PREDICTIONS,
        ids=valid_ids,
        targets=targets,
        stats=canonical_stats,
    )
    primary_rows = {
        "V32": v32_suite["per_sample"],
        "O075": metric_suites["O075"]["point_global_best"]["per_sample"],
        "Dual": metric_suites["Dual"]["point_global_best"]["per_sample"],
    }
    pair_specs = (
        ("V32", "O075"),
        ("V32", "Dual"),
        ("O075", "Dual"),
    )
    paired_rows = []
    paired_payload = {}
    bootstrap_csv = []
    for index, (left, right) in enumerate(pair_specs):
        rows = _paired_rows(
            left_name=left,
            right_name=right,
            left_rows=primary_rows[left],
            right_rows=primary_rows[right],
            targets=targets,
        )
        pair_name = f"{right}_minus_{left}"
        inference = _bootstrap_pair(rows, seed=BOOTSTRAP_SEED + index)
        paired_rows.extend(rows)
        paired_payload[pair_name] = inference
        for metric, values in inference["metrics"].items():
            bootstrap_csv.append(
                {
                    "pair": pair_name,
                    "metric": metric,
                    "observed_difference": values["observed_difference"],
                    "ci95_low": values["bootstrap_95pct_ci"][0],
                    "ci95_high": values["bootstrap_95pct_ci"][1],
                    "probability_right_improves": values[
                        "bootstrap_probability_right_improves"
                    ],
                    "win_rate": values["per_sample_win_rate"],
                    "median_difference": values[
                        "per_sample_median_difference"
                    ],
                }
            )

    evaluator_path = Path(__file__).resolve()
    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    scope = {
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
    }
    combined = {
        "schema_version": "heat3d_v5_gate6l_valid_only_evaluation_v1",
        "status": "completed_valid_iid_only",
        "evaluator_commit": _git_commit(),
        "evaluator_source_sha256": _sha256(evaluator_path),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "frozen_formula_source": {
            "path": str(metric_path.relative_to(ROOT)),
            "commit": _path_commit(metric_path),
            "sha256": _sha256(metric_path),
        },
        "scope": scope,
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
            "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
            "target_or_label_features": [],
        },
        "models": {
            model: {
                "config_id": CONFIGS[model],
                "source_host": "devbox" if model == "O075" else "wsl2",
                "training_commit": "461d810",
                "checkpoint_metadata": checkpoint_metadata[model],
                "metrics": {
                    checkpoint: {
                        "metric_schema_version": suite["metric_schema_version"],
                        "summary": suite["summary"],
                    }
                    for checkpoint, suite in metric_suites[model].items()
                },
                "strata": strata_payloads[model],
                "reload_audit": replay_audits[model],
                "training_attention_diagnostics": run_data[model][1].get(
                    "attention_diagnostics_by_checkpoint"
                ),
                "artifacts": {
                    path.name: {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in (
                        run_dirs[model] / "run_config.json",
                        run_dirs[model] / "loss_summary.json",
                        *[
                            run_dirs[model] / values[0]
                            for values in CHECKPOINTS.values()
                        ],
                        *[
                            run_dirs[model] / values[1]
                            for values in CHECKPOINTS.values()
                        ],
                    )
                },
            }
            for model in CONFIGS
        },
        "v32_reference": {
            "config_id": V32_ID,
            "checkpoint": "point_global_best",
            "epoch": int(v32_checkpoint["epoch"]),
            "sha256": _sha256(v32_run / V32_CHECKPOINT),
            "metrics": v32_suite["summary"],
        },
        "paired_primary_point_global_best": paired_payload,
        "checkpoint_comparison_policy": (
            "paired comparisons use each model's frozen point-global-best "
            "checkpoint; no checkpoint was reselected"
        ),
    }
    _write_json(output_paths["combined"], combined)

    for model, output_key in (("O075", "o075"), ("Dual", "dual")):
        run_config, loss_summary = run_data[model]
        compatibility = {
            "schema_version": "heat3d_v5_valid_only_four_checkpoint_v1",
            "status": "completed_valid_only",
            "config_id": CONFIGS[model],
            "source": "devbox" if model == "O075" else "wsl2",
            "run_dir": str(run_config["output_dir"]),
            "log_path": (
                f"output/heat3d_v5_gate6k_{'o075' if model == 'O075' else 'dual'}_logs/"
                f"{CONFIGS[model]}.log"
            ),
            "training_commit": "461d810",
            "evaluator_commit": combined["evaluator_commit"],
            "evaluator_source_sha256": combined["evaluator_source_sha256"],
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "scope": scope,
            "checkpoint_metadata": checkpoint_metadata[model],
            "metrics": combined["models"][model]["metrics"],
            "loss_summary": _pruned_loss(loss_summary),
            "run_config": {
                "code_version_or_git_commit": "461d810",
                "final_probe_eval_after_training": run_config.get(
                    "final_probe_eval_after_training"
                ),
                "post_training_diagnostics": run_config.get(
                    "post_training_diagnostics"
                ),
            },
        }
        _write_json(output_paths[output_key], compatibility)

    checkpoint_fields = (
        "model",
        "config_id",
        "checkpoint",
        "epoch",
        "sha256",
        "parameter_count",
        "training_commit",
        "training_reload_max_abs_error_K",
        "evaluator_replay_max_abs_error_K",
        *REQUIRED_SUMMARY_FIELDS,
    )
    _write_csv(output_paths["checkpoints"], checkpoint_csv, checkpoint_fields)
    paired_fields = tuple(paired_rows[0].keys())
    _write_csv(output_paths["paired"], paired_rows, paired_fields)
    _write_csv(
        output_paths["bootstrap"],
        bootstrap_csv,
        (
            "pair",
            "metric",
            "observed_difference",
            "ci95_low",
            "ci95_high",
            "probability_right_improves",
            "win_rate",
            "median_difference",
        ),
    )
    strata_fields = tuple(strata_csv[0].keys())
    _write_csv(output_paths["strata"], strata_csv, strata_fields)
    print(
        json.dumps(
            {
                "status": "completed_valid_iid_only",
                "outputs": {
                    key: str(path) for key, path in output_paths.items()
                },
                "epochs": {
                    model: {
                        checkpoint: metadata["epoch"]
                        for checkpoint, metadata in checkpoint_metadata[model].items()
                    }
                    for model in CONFIGS
                },
                "primary_metrics": {
                    model: metric_suites[model]["point_global_best"]["summary"]
                    for model in CONFIGS
                },
                "paired": {
                    pair: {
                        metric: values["observed_difference"]
                        for metric, values in report["metrics"].items()
                    }
                    for pair, report in paired_payload.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
