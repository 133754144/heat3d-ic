#!/usr/bin/env python3
"""Gate 6K read-only V13/V32 train and valid_iid loss audit."""

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
    evaluate_metric_suite,
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


V13_ID = "V4P5_13_gate6e_scratch_branch_rebalance"
V32_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
V13_CHECKPOINT = "params_best.pkl"
V32_CHECKPOINT = "params_best_valid_point_global.pkl"
V13_SHA256 = "dac34633392015d7a1752367cca5ed9cb58fdb62331c46cdf31b0105fc49923d"
V32_SHA256 = "f3063b53ca26a2b91fffc090ad4de98fe260ac5d7b669bcfbfd77c1fcf045d24"
EPS = 1.0e-12
LOSS_NAMES = (
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
)
CORE_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
)


class Gate6KError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v13-run-dir", type=Path, required=True)
    parser.add_argument("--v32-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
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


def _as_field(value: Any, batch: int, nodes: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (batch, 1, nodes, 1):
        return array
    if array.shape == (batch, nodes):
        return array[:, None, :, None]
    if array.shape == (batch, nodes, 1):
        return array[:, None, :, :]
    if array.shape == (nodes,):
        return np.broadcast_to(array[None, None, :, None], (batch, 1, nodes, 1))
    raise Gate6KError(f"{name}: unsupported field shape {array.shape}")


def _as_scalar(value: Any, batch: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (batch, 1, 1, 1):
        return array.reshape(batch)
    if array.shape in {(batch,), (batch, 1)}:
        return array.reshape(batch)
    raise Gate6KError(f"{name}: unsupported scalar shape {array.shape}")


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise Gate6KError("distribution requires finite non-empty values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _signed_scale_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    payload = _distribution(array)
    payload.update(
        {
            "rmse": float(np.sqrt(np.mean(np.square(array)))),
            "mean_abs": float(np.mean(np.abs(array))),
            "p10": float(np.quantile(array, 0.10)),
            "positive_fraction": float(np.mean(array > 0.0)),
            "negative_fraction": float(np.mean(array < 0.0)),
        }
    )
    return payload


def _loss_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in LOSS_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        order = np.argsort(values)[::-1]
        total = float(np.sum(values))
        payload[name] = {
            "distribution": _distribution(values),
            "worst_sample": {
                "sample_id": str(rows[int(order[0])]["sample_id"]),
                "value": float(values[order[0]]),
                "fraction_of_total": float(values[order[0]] / max(total, EPS)),
            },
            "top5_cumulative_fraction": float(
                np.sum(values[order[:5]]) / max(total, EPS)
            ),
        }
    payload["signed_scale_log_error"] = _signed_scale_summary(
        [float(row["signed_scale_log_error"]) for row in rows]
    )
    return payload


def _suite(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> dict[str, Any]:
    target_mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    target_std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    samples = []
    for row in rows:
        target = np.asarray(row["target_deltaT_K"], dtype=np.float64)
        prediction = np.asarray(row["prediction_deltaT_K"], dtype=np.float64)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "prediction_deltaT_K": prediction,
                "target_deltaT_K": target,
                "control_volumes_m3": row["control_volumes_m3"],
                "q_W_m3": row["q_W_m3"],
                "prediction_normalized": (prediction - target_mean) / target_std,
                "target_normalized": (target - target_mean) / target_std,
            }
        )
    return evaluate_metric_suite(samples)


def _subset_payload(
    rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "sample_count": len(rows),
        "metrics": _suite(rows, stats)["summary"],
        "losses": _loss_summary(rows),
    }


def _infer_split(
    *,
    split: str,
    examples: Sequence[Any],
    sample_root: Path,
    stats: Mapping[str, Any],
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
    graph_config: Mapping[str, Any],
    graph_seed: int,
    batch_size: int,
    models: Mapping[str, tuple[Any, Any]],
) -> dict[str, list[dict[str, Any]]]:
    builder = Heat3DGraphBuilder(**dict(graph_config))
    groups = _make_groups_with_progress(
        list(examples),
        stats,
        builder,
        f"gate6k_{split}",
        False,
        "basic",
        graph_seed,
        batch_size=batch_size,
        drop_last=False,
    )
    _attach_v5_physics(groups, cache, standardizer)
    examples_by_id = {example.sample_id: example for example in examples}
    _attach_qk_region_features_to_groups(
        groups, examples_by_id, feature_version="sparse_safe_v2"
    )
    for group in groups:
        group["native_physics"] = group["v5_physics"]
        group["global_context"] = group["v5_physics"]["global_context"]

    rows_by_model = {name: [] for name in models}
    for group in groups:
        sample_ids = [str(value) for value in group["sample_ids"]]
        batch = len(sample_ids)
        nodes = int(group["target_delta_raw"].shape[2])
        physics = group["native_physics"]
        target = _as_field(group["target_delta_raw"], batch, nodes, "target")
        volumes = _as_field(
            physics["control_volumes"], batch, nodes, "control_volumes"
        )
        mask = _as_field(
            physics["dirichlet_mask"], batch, nodes, "dirichlet_mask"
        )
        target_free = (1.0 - np.clip(mask, 0.0, 1.0)) * target
        volume_sum = np.sum(volumes, axis=2, keepdims=True)
        true_scale = np.sqrt(
            np.sum(np.square(target_free) * volumes, axis=2, keepdims=True)
            / np.maximum(volume_sum, EPS)
        )
        true_shape = target_free / np.maximum(true_scale, EPS)
        for model_name, (model, params) in models.items():
            prediction = _model_apply(model, params, group)
            phi_hat = _as_field(prediction["phi_hat"], batch, nodes, "phi_hat")
            s_hat = _as_scalar(prediction["s_hat"], batch, "s_hat")
            delta_hat = _as_field(
                prediction["deltaT_hat"], batch, nodes, "deltaT_hat"
            )
            raw_temperature = _as_field(
                prediction["raw_temperature"], batch, nodes, "raw_temperature"
            )
            shape_mse = (
                np.sum(np.square(phi_hat - true_shape) * volumes, axis=2)
                / np.maximum(volume_sum.reshape(batch, 1, 1), EPS)
            ).reshape(batch)
            raw_mse = (
                np.sum(np.square(delta_hat - target_free) * volumes, axis=2)
                / np.maximum(volume_sum.reshape(batch, 1, 1), EPS)
            ).reshape(batch)
            signed_scale = np.log(np.maximum(s_hat, EPS)) - np.log(
                np.maximum(true_scale.reshape(batch), EPS)
            )
            for index, sample_id in enumerate(sample_ids):
                meta = json.loads(
                    (sample_root / sample_id / "sample_meta.json").read_text(
                        encoding="utf-8"
                    )
                )
                if meta.get("split") != split:
                    raise Gate6KError(
                        f"{sample_id}: expected split {split}, found {meta.get('split')}"
                    )
                categories = sorted(
                    {
                        str(item["DeltaT_target_bin"])
                        for item in meta.get("q_block_metadata", ())
                        if item.get("DeltaT_target_bin") is not None
                    }
                )
                if len(categories) != 1:
                    raise Gate6KError(
                        f"{sample_id}: expected one generator condition category"
                    )
                bottom = float(
                    meta["boundary_params"]["bottom"]["fixed_temperature_K"]
                )
                q = np.load(sample_root / sample_id / "q_field.npy").astype(
                    np.float64
                ).reshape(-1)
                row = {
                    "model": model_name,
                    "split": split,
                    "sample_id": sample_id,
                    "true_cv_rms_deltaT_K": float(true_scale[index].reshape(-1)[0]),
                    "generator_condition_category": categories[0],
                    "shape_cv_loss": float(shape_mse[index]),
                    "log_scale_loss": float(signed_scale[index] ** 2),
                    "relative_field_loss": float(
                        raw_mse[index]
                        / max(float(true_scale[index].reshape(-1)[0] ** 2), EPS)
                    ),
                    "raw_absolute_field_loss": float(raw_mse[index]),
                    "signed_scale_log_error": float(signed_scale[index]),
                    "prediction_deltaT_K": (
                        raw_temperature[index].reshape(-1) - bottom
                    ),
                    "target_deltaT_K": target[index].reshape(-1),
                    "control_volumes_m3": volumes[index].reshape(-1),
                    "q_W_m3": q,
                }
                rows_by_model[model_name].append(row)

    for model_name, rows in rows_by_model.items():
        scales = np.asarray(
            [row["true_cv_rms_deltaT_K"] for row in rows], dtype=np.float64
        )
        q25, q50, q75 = np.quantile(scales, [0.25, 0.50, 0.75])
        for row in rows:
            value = float(row["true_cv_rms_deltaT_K"])
            row["deltaT_quartile"] = (
                "Q1"
                if value <= q25
                else "Q2"
                if value <= q50
                else "Q3"
                if value <= q75
                else "Q4"
            )
            row["is_nominal_to_hard"] = (
                row["generator_condition_category"] == "nominal_to_hard"
            )
            row["is_q2_nominal_to_hard_intersection"] = bool(
                row["deltaT_quartile"] == "Q2" and row["is_nominal_to_hard"]
            )
        if len(rows) != len(examples):
            raise Gate6KError(f"{model_name}/{split}: sample count drifted")
    return rows_by_model


def _comparison(
    summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "valid_iid"):
        left = summaries["V13"][split]
        right = summaries["V32"][split]
        result[split] = {
            "core_metric_v32_minus_v13": {
                name: float(right["metrics"][name] - left["metrics"][name])
                for name in CORE_METRICS
            },
            "mean_loss_v32_minus_v13": {
                name: float(
                    right["losses"][name]["distribution"]["mean"]
                    - left["losses"][name]["distribution"]["mean"]
                )
                for name in LOSS_NAMES
            },
            "signed_scale_log_error_mean_delta": float(
                right["losses"]["signed_scale_log_error"]["mean"]
                - left["losses"]["signed_scale_log_error"]["mean"]
            ),
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "model",
        "split",
        "sample_id",
        "true_cv_rms_deltaT_K",
        "deltaT_quartile",
        "generator_condition_category",
        "is_nominal_to_hard",
        "is_q2_nominal_to_hard_intersection",
        *LOSS_NAMES,
        "signed_scale_log_error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: row[field] for field in fields} for row in rows]
        )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate 6K train/valid_iid loss audit",
        "",
        "- Scope: train + valid_iid only; no test/hard/sealed access.",
        "- Checkpoints: V13 legacy base-MSE best e318; V32 point-global best e474.",
        "- Four loss terms are reported per sample before configured weighting.",
        "",
        "## Core frozen V5 metrics",
        "",
        "| split | model | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "valid_iid"):
        for model in ("V13", "V32"):
            summary = payload["models"][model][split]["metrics"]
            lines.append(
                f"| {split} | {model} | "
                f"{summary['point_global_relative_rmse_pct']:.6f} | "
                f"{summary['sample_first_cv_relative_rmse_pct']:.6f} | "
                f"{summary['raw_cv_weighted_rmse_K']:.6f} | "
                f"{summary['shape_cv_rmse']:.6f} | "
                f"{summary['scale_log_rmse']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Loss distribution and signed scale error",
            "",
            "The JSON contains mean/median/P90/P95/P99, worst-sample contribution, "
            "top-5 cumulative contribution, and signed scale-error summaries for "
            "each model/split. It also contains separate Q2, nominal_to_hard, and "
            "Q2 ∩ nominal_to_hard summaries.",
            "",
            "## Interpretation",
            "",
            payload["initial_conclusion"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _args()
    outputs = tuple(
        path.resolve()
        for path in (args.output_json, args.output_csv, args.output_md)
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise Gate6KError("one or more outputs already exist")
    v13_run = args.v13_run_dir.resolve()
    v32_run = args.v32_run_dir.resolve()
    if v13_run.name != V13_ID or v32_run.name != V32_ID:
        raise Gate6KError("run/config binding failed")
    if any(run in path.parents for run in (v13_run, v32_run) for path in outputs):
        raise Gate6KError("audit outputs must remain outside frozen run directories")

    run_configs = {
        "V13": json.loads((v13_run / "run_config.json").read_text(encoding="utf-8")),
        "V32": json.loads((v32_run / "run_config.json").read_text(encoding="utf-8")),
    }
    checkpoint_paths = {
        "V13": v13_run / V13_CHECKPOINT,
        "V32": v32_run / V32_CHECKPOINT,
    }
    expected_hashes = {"V13": V13_SHA256, "V32": V32_SHA256}
    checkpoints = {}
    for model_name, path in checkpoint_paths.items():
        if _sha256(path) != expected_hashes[model_name]:
            raise Gate6KError(f"{model_name}: checkpoint hash mismatch")
        checkpoints[model_name] = _load_params_checkpoint(path)
    if int(checkpoints["V13"]["epoch"]) != 318 or int(checkpoints["V32"]["epoch"]) != 474:
        raise Gate6KError("checkpoint epoch binding failed")

    checkpoint_stats = {
        model: dict(payload["train_only_normalization"])
        for model, payload in checkpoints.items()
    }
    if not _normalization_equal(checkpoint_stats["V13"], checkpoint_stats["V32"]):
        raise Gate6KError("V13/V32 normalization differs")
    v32_stats = checkpoint_stats["V32"]
    install_checkpoint_feature_hooks(v32_stats)
    train_examples = load_training_examples(run_configs["V32"], v32_stats)
    recomputed_stats = training_normalization_stats(
        train_examples,
        normalization_profile=str(
            v32_stats.get("normalization_profile", "legacy_zscore")
        ),
        condition_feature_transform=v32_stats.get("condition_feature_transform"),
        input_feature_schema=str(
            v32_stats.get("input_feature_schema", "legacy_bc_flags")
        ),
        coord_policy=str(
            v32_stats.get("coord_policy", "train_minmax_to_unit_box")
        ),
        extent_feature_policy=str(v32_stats.get("extent_feature_policy", "none")),
        bridge_fn=runner_module._bridge_for,
    )
    if not _normalization_equal(v32_stats, recomputed_stats):
        raise Gate6KError("normalization does not reproduce from train only")
    stats = stats_from_checkpoint_payload(v32_stats, train_examples)
    sample_root = _sample_root(Path(run_configs["V32"]["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(run_configs["V32"]["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6KError("train/valid split count drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=v32_stats,
        boundary_mask_fallback=bool(
            run_configs["V32"].get("boundary_mask_fallback", True)
        ),
    )
    all_examples = list(train_examples) + list(valid_examples)
    if any(np.asarray(example.condition.coords).shape != (1024, 3) for example in all_examples):
        raise Gate6KError("expected 1024 nodes per sample")
    cache = _physics_cache(all_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    stored_standardizer = run_configs["V32"]["global_context"]["standardizer"]
    if (
        standardizer["fit_population"] != "train_only"
        or int(standardizer["fit_sample_count"]) != 672
        or standardizer["fit_sample_ids_sha256"]
        != stored_standardizer["fit_sample_ids_sha256"]
    ):
        raise Gate6KError("global-context standardizer is not train-only")

    models = {}
    for model_name in ("V13", "V32"):
        model_config = _resolve_decoder_bypass_model_config(
            dict(checkpoints[model_name]["model_config"]),
            checkpoint_stats[model_name],
        )
        models[model_name] = (
            GraphNeuralOperator(**model_config),
            _device_params(checkpoints[model_name]["params"]),
        )

    rows_by_model_split = {name: {} for name in models}
    for split, examples in (
        ("train", train_examples),
        ("valid_iid", valid_examples),
    ):
        inferred = _infer_split(
            split=split,
            examples=examples,
            sample_root=sample_root,
            stats=stats,
            cache=cache,
            standardizer=standardizer,
            graph_config=run_configs["V32"]["graph_config"],
            graph_seed=int(run_configs["V32"].get("graph_seed", 0)),
            batch_size=args.prediction_batch_size,
            models=models,
        )
        for model_name, rows in inferred.items():
            rows_by_model_split[model_name][split] = rows

    summaries: dict[str, dict[str, Any]] = {name: {} for name in models}
    csv_rows = []
    for model_name in ("V13", "V32"):
        for split in ("train", "valid_iid"):
            rows = rows_by_model_split[model_name][split]
            suite = _suite(rows, checkpoint_stats[model_name])
            summaries[model_name][split] = {
                "sample_count": len(rows),
                "metrics": suite["summary"],
                "losses": _loss_summary(rows),
                "subsets": {
                    "deltaT_Q2": _subset_payload(
                        [row for row in rows if row["deltaT_quartile"] == "Q2"],
                        checkpoint_stats[model_name],
                    ),
                    "nominal_to_hard": _subset_payload(
                        [row for row in rows if row["is_nominal_to_hard"]],
                        checkpoint_stats[model_name],
                    ),
                    "deltaT_Q2_intersection_nominal_to_hard": _subset_payload(
                        [
                            row
                            for row in rows
                            if row["is_q2_nominal_to_hard_intersection"]
                        ],
                        checkpoint_stats[model_name],
                    ),
                },
            }
            csv_rows.extend(rows)

    comparison = _comparison(summaries)
    valid_delta = comparison["valid_iid"]["core_metric_v32_minus_v13"]
    initial_conclusion = (
        "V32 相对 V13 的 valid point-global 改善，但 sample-first 退化；"
        "Gate 6K 仅作 train/valid 误差归因，不据此自动晋级，也不触发后续 seed。"
        if valid_delta["point_global_relative_rmse_pct"] < 0.0
        and valid_delta["sample_first_cv_relative_rmse_pct"] > 0.0
        else "Gate 6K 结果仅用于误差归因；不自动晋级，也不触发后续 seed。"
    )
    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    payload = {
        "schema_version": "heat3d_v5_gate6k_train_valid_loss_audit_v1",
        "status": "completed_read_only",
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
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "model_or_checkpoint_modified": False,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "nodes_per_sample": 1024,
        },
        "split": {
            "source": split_source,
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "normalization_and_context": {
            "fit_roles": ["train"],
            "fit_population": standardizer["fit_population"],
            "fit_sample_count": int(standardizer["fit_sample_count"]),
            "target_or_label_features": [],
        },
        "checkpoints": {
            "V13": {
                "config_id": V13_ID,
                "file": V13_CHECKPOINT,
                "epoch": 318,
                "sha256": V13_SHA256,
            },
            "V32": {
                "config_id": V32_ID,
                "file": V32_CHECKPOINT,
                "epoch": 474,
                "sha256": V32_SHA256,
            },
        },
        "loss_formulas": {
            "shape_cv_loss": "mean_samples(sum((phi_hat-phi_true)^2*CV)/sum(CV))",
            "log_scale_loss": "mean_samples((log(s_hat)-log(s_true))^2)",
            "relative_field_loss": "mean_samples(CV_MSE(deltaT_hat-deltaT_true)/s_true^2)",
            "raw_absolute_field_loss": "mean_samples(CV_MSE(deltaT_hat-deltaT_true))",
            "signed_scale_log_error": "log(s_hat)-log(s_true); positive means scale overprediction",
        },
        "models": summaries,
        "comparison": comparison,
        "initial_conclusion": initial_conclusion,
    }
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(outputs[1], csv_rows)
    _write_markdown(outputs[2], payload)
    print(json.dumps({"status": payload["status"], "outputs": [str(p) for p in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
