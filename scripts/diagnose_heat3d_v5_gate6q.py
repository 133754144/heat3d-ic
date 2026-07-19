#!/usr/bin/env python3
"""Gate 6Q oracle-scale, fixed-ridge, and coverage diagnostics."""

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
from diagnose_heat3d_v5_gate6p import (  # noqa: E402
    V38_CONFIG_ID,
    V39_CHECKPOINT,
    V39_CONFIG_ID,
    _correlation,
    _infer_with_scale_features,
    _raw_physics,
)
from evaluate_heat3d_v5_gate6l_valid_only import _build_groups  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _load_params_checkpoint,
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


RIDGE_ALPHA = 0.001
K_NEIGHBORS = 10
EPS = 1.0e-12
GATE6P_MANIFEST = (
    ROOT / "configs/heat3d_v5/gate6p/gate6p_artifact_manifest.json"
)
STRATIFY_FIELDS = (
    "harmonic_kx_W_mK",
    "harmonic_ky_W_mK",
    "harmonic_kz_W_mK",
    "anisotropy_xy_over_z",
)


class Gate6QError(RuntimeError):
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


def _ridge_fit_predict(
    train_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    train_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_matrix.mean(axis=0)
    raw_std = train_matrix.std(axis=0)
    std = np.where(raw_std > EPS, raw_std, 1.0)
    x_train = (train_matrix - mean) / std
    x_valid = (valid_matrix - mean) / std
    design = np.concatenate(
        [np.ones((x_train.shape[0], 1)), x_train], axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ train_target
    )
    train_prediction = design @ coefficients
    valid_prediction = (
        np.concatenate([np.ones((x_valid.shape[0], 1)), x_valid], axis=1)
        @ coefficients
    )
    return valid_prediction, {
        "alpha": RIDGE_ALPHA,
        "fit_roles": ["train"],
        "query_roles": ["valid_iid"],
        "fit_sample_count": int(train_matrix.shape[0]),
        "query_sample_count": int(valid_matrix.shape[0]),
        "feature_width": int(train_matrix.shape[1]),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "zero_variance_feature_count": int(np.sum(raw_std <= EPS)),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients[1:])),
        "train_residual_rmse": float(
            np.sqrt(np.mean(np.square(train_prediction - train_target)))
        ),
    }


def _feature_matrices(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_physics = np.stack(
        [features[sample_id]["physics"] for sample_id in train_ids]
    )
    valid_physics = np.stack(
        [features[sample_id]["physics"] for sample_id in valid_ids]
    )
    train_pooled = np.stack(
        [features[sample_id]["pooled_latent"] for sample_id in train_ids]
    )
    valid_pooled = np.stack(
        [features[sample_id]["pooled_latent"] for sample_id in valid_ids]
    )
    return {
        "physics_24": (train_physics, valid_physics),
        "pooled_latent_96": (train_pooled, valid_pooled),
        "combined_120": (
            np.concatenate([train_physics, train_pooled], axis=1),
            np.concatenate([valid_physics, valid_pooled], axis=1),
        ),
    }


def _oracle_fields(
    raw_temperature: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    shapes, _ = _decompose_fields(raw_temperature, ids, targets)
    return {
        sample_id: shapes[sample_id]
        * float(targets[sample_id]["true_scale_cv_rms_K"])
        for sample_id in ids
    }


def _top_sse(suite: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(
        suite["per_sample"],
        key=lambda row: float(row["point_error_squared_sum"]),
        reverse=True,
    )
    total = float(sum(row["point_error_squared_sum"] for row in rows))
    return {
        "total_point_sse_K2": total,
        "top5_point_sse_K2": float(
            sum(row["point_error_squared_sum"] for row in rows[:5])
        ),
        "top10_point_sse_K2": float(
            sum(row["point_error_squared_sum"] for row in rows[:10])
        ),
        "top5_cumulative_fraction": float(
            sum(row["point_error_squared_sum"] for row in rows[:5])
            / max(total, EPS)
        ),
        "top10_cumulative_fraction": float(
            sum(row["point_error_squared_sum"] for row in rows[:10])
            / max(total, EPS)
        ),
        "top10_sample_ids": [str(row["sample_id"]) for row in rows[:10]],
    }


def _knn_coverage(
    *,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    train_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    train_log_scale: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    mean = train_matrix.mean(axis=0)
    raw_std = train_matrix.std(axis=0)
    std = np.where(raw_std > EPS, raw_std, 1.0)
    train = (train_matrix - mean) / std
    valid = (valid_matrix - mean) / std
    rows = {}
    for index, sample_id in enumerate(valid_ids):
        distances = np.linalg.norm(train - valid[index][None, :], axis=1)
        neighbors = np.argsort(distances, kind="mergesort")[:K_NEIGHBORS]
        neighbor_scales = train_log_scale[neighbors]
        rows[sample_id] = {
            "nearest_train_sample_id": train_ids[int(neighbors[0])],
            "nearest_distance": float(distances[neighbors[0]]),
            "mean_k_distance": float(np.mean(distances[neighbors])),
            "neighbor_log_target_scale_mean": float(np.mean(neighbor_scales)),
            "neighbor_log_target_scale_variance": float(
                np.var(neighbor_scales)
            ),
            "neighbor_log_target_scale_std": float(np.std(neighbor_scales)),
        }
    return rows, {
        "reference_roles": ["train"],
        "query_roles": ["valid_iid"],
        "k": K_NEIGHBORS,
        "distance": "Euclidean after train-fit per-feature z-score",
        "target_scale_source": "train neighbors only",
        "feature_width": int(train_matrix.shape[1]),
        "zero_variance_feature_count": int(np.sum(raw_std <= EPS)),
    }


def _quartile_label(value: float, boundaries: Sequence[float]) -> str:
    return (
        "Q1"
        if value <= boundaries[0]
        else "Q2"
        if value <= boundaries[1]
        else "Q3"
        if value <= boundaries[2]
        else "Q4"
    )


def _stratified_rows(
    *,
    fields: Mapping[str, Mapping[str, np.ndarray]],
    valid_ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
    train_physics: Mapping[str, Mapping[str, float]],
    valid_physics: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows = []
    boundaries_by_field = {}
    for physical_field in STRATIFY_FIELDS:
        boundaries = np.quantile(
            [train_physics[sample_id][physical_field] for sample_id in train_physics],
            [0.25, 0.50, 0.75],
        )
        boundaries_by_field[physical_field] = boundaries.tolist()
        labels = {
            sample_id: _quartile_label(
                valid_physics[sample_id][physical_field], boundaries
            )
            for sample_id in valid_ids
        }
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            selected = [
                sample_id for sample_id in valid_ids if labels[sample_id] == quartile
            ]
            if not selected:
                raise Gate6QError(
                    f"{physical_field}/{quartile}: empty valid stratum"
                )
            for name, field in fields.items():
                summary = _suite_from_fields(
                    fields=field,
                    ids=selected,
                    targets=targets,
                    stats=stats,
                )["summary"]
                rows.append(
                    {
                        "physical_field": physical_field,
                        "train_quartile": quartile,
                        "field": name,
                        "sample_count": len(selected),
                        "point_global_relative_rmse_pct": summary[
                            "point_global_relative_rmse_pct"
                        ],
                        "sample_first_cv_relative_rmse_pct": summary[
                            "sample_first_cv_relative_rmse_pct"
                        ],
                        "raw_cv_weighted_rmse_K": summary[
                            "raw_cv_weighted_rmse_K"
                        ],
                    }
                )
    return rows, boundaries_by_field


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 6Q oracle-scale and fixed-ridge diagnostics",
        "",
        "仅访问 `train` 与 `valid_iid`；未训练、未生成 YAML、"
        "未访问 `test/hard/sealed`。",
        "",
        "## Oracle-scale upper bounds",
        "",
        "| field | point-global % | sample-first % | raw CV K | Q4 point-global % | top10 SSE fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("e231_oracle_scale", "e543_oracle_scale", "v39_e24_oracle_scale"):
        row = payload["field_metrics"][name]["summary"]
        q4 = payload["field_metrics"][name]["q4_summary"]
        top = payload["field_metrics"][name]["top_sse"]
        lines.append(
            f"| {name} | {row['point_global_relative_rmse_pct']:.6f} | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{row['raw_cv_weighted_rmse_K']:.6f} | "
            f"{q4['point_global_relative_rmse_pct']:.6f} | "
            f"{top['top10_cumulative_fraction']:.6f} |"
        )
    lines += [
        "",
        "## Train-fit fixed ridge, one-shot valid_iid",
        "",
        "| readout | point-global % | sample-first % | raw CV K | Q4 point-global % |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("ridge_physics_24", "ridge_pooled_latent_96", "ridge_combined_120"):
        row = payload["field_metrics"][name]["summary"]
        q4 = payload["field_metrics"][name]["q4_summary"]
        lines.append(
            f"| {name} | {row['point_global_relative_rmse_pct']:.6f} | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{row['raw_cv_weighted_rmse_K']:.6f} | "
            f"{q4['point_global_relative_rmse_pct']:.6f} |"
        )
    conclusion = payload["bottleneck_assessment"]
    lines += [
        "",
        "## Conclusion",
        "",
        f"- scale-only theoretical <20%: `{conclusion['scale_only_theoretical_below_20pct']}`",
        f"- bottleneck: `{conclusion['classification']}`",
        f"- basis: {conclusion['basis']}",
        f"- unique route: `{conclusion['unique_recommended_route']}`",
        "",
        "该路线仅为只读诊断建议，本轮没有生成配置或启动训练。",
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
        "json": output_dir / "gate6q_diagnostics.json",
        "md": output_dir / "gate6q_diagnostics.md",
        "samples_json": output_dir / "gate6q_sample_level.json",
        "samples_csv": output_dir / "gate6q_sample_level.csv",
        "coverage_csv": output_dir / "gate6q_knn_coverage.csv",
        "strata_csv": output_dir / "gate6q_conductivity_strata.csv",
        "manifest": output_dir / "gate6q_artifact_manifest.json",
    }
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise Gate6QError("output already exists")
    if v38_run.name != V38_CONFIG_ID or v39_run.name != V39_CONFIG_ID:
        raise Gate6QError("run directory/config binding failed")
    gate6p_manifest = json.loads(GATE6P_MANIFEST.read_text(encoding="utf-8"))
    if (
        gate6p_manifest["artifacts"]["gate6p_e543_scale_features.csv"]["sha256"]
        != "46c4380c33af0f88ec30d81b2516fdb032b57d7c023b68e9106694968bf24ec9"
    ):
        raise Gate6QError("Gate 6P frozen feature SHA drifted")

    v38_config = json.loads((v38_run / "run_config.json").read_text())
    v38_summary = json.loads((v38_run / "loss_summary.json").read_text())
    v39_summary = json.loads((v39_run / "loss_summary.json").read_text())
    if int(v38_summary.get("final_epoch", -1)) != 600:
        raise Gate6QError("V38 did not complete e600")
    if int(v39_summary.get("final_epoch", -1)) != 40:
        raise Gate6QError("V39 did not complete e40")

    checkpoints = {}
    binding = {}
    for name, (filename, _, epoch, digest) in V38_CHECKPOINTS.items():
        path = v38_run / filename
        if _sha256(path) != digest:
            raise Gate6QError(f"{name}: checkpoint SHA256 drifted")
        checkpoint = _load_params_checkpoint(path)
        if int(checkpoint["epoch"]) != epoch:
            raise Gate6QError(f"{name}: checkpoint epoch drifted")
        checkpoints[name] = checkpoint
        binding[name] = {
            "config_id": V38_CONFIG_ID,
            "epoch": epoch,
            "sha256": digest,
            "path": str(path),
        }
    v39_path = v39_run / V39_CHECKPOINT["filename"]
    if _sha256(v39_path) != V39_CHECKPOINT["sha256"]:
        raise Gate6QError("V39 e24 checkpoint SHA256 drifted")
    checkpoints["v39_e24"] = _load_params_checkpoint(v39_path)
    if int(checkpoints["v39_e24"]["epoch"]) != V39_CHECKPOINT["epoch"]:
        raise Gate6QError("V39 checkpoint epoch drifted")
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
            raise Gate6QError(f"{name}: train-only normalization differs")
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
        raise Gate6QError("train-only normalization does not reproduce")
    stats = stats_from_checkpoint_payload(canonical_stats, train_examples)
    sample_root = _sample_root(Path(v38_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(v38_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6QError("split counts drifted")
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

    raw = {
        name: _infer_raw_temperature(checkpoint, valid_groups)
        for name, checkpoint in checkpoints.items()
    }
    _, train_features = _infer_with_scale_features(
        checkpoints["e543"], train_groups
    )
    e543_feature_raw, valid_features = _infer_with_scale_features(
        checkpoints["e543"], valid_groups
    )
    replay_error = max(
        float(np.max(np.abs(e543_feature_raw[sample_id] - raw["e543"][sample_id])))
        for sample_id in valid_ids
    )
    if replay_error > 0.02:
        raise Gate6QError(f"e543 feature replay failed: {replay_error} K")
    features = {**train_features, **valid_features}
    matrices = _feature_matrices(train_ids, valid_ids, features)

    actual_delta = {
        name: _delta_fields(value, valid_ids, valid_targets)
        for name, value in raw.items()
    }
    oracle = {
        f"{name}_oracle_scale": _oracle_fields(
            raw[name], valid_ids, valid_targets
        )
        for name in ("e231", "e543", "v39_e24")
    }
    e543_shapes, _ = _decompose_fields(raw["e543"], valid_ids, valid_targets)
    train_log_true = np.log(
        np.asarray(
            [train_targets[sample_id]["true_scale_cv_rms_K"] for sample_id in train_ids]
        )
    )
    train_log_s_phys = np.asarray(
        [features[sample_id]["log_s_phys_K"] for sample_id in train_ids]
    )
    train_residual = train_log_true - train_log_s_phys
    valid_log_s_phys = np.asarray(
        [features[sample_id]["log_s_phys_K"] for sample_id in valid_ids]
    )
    ridge_fields = {}
    ridge_models = {}
    for feature_name, (train_matrix, valid_matrix) in matrices.items():
        predicted_residual, model = _ridge_fit_predict(
            train_matrix, valid_matrix, train_residual
        )
        predicted_scale = np.exp(valid_log_s_phys + predicted_residual)
        field_name = f"ridge_{feature_name}"
        ridge_fields[field_name] = {
            sample_id: e543_shapes[sample_id] * float(predicted_scale[index])
            for index, sample_id in enumerate(valid_ids)
        }
        ridge_models[field_name] = model

    evaluated_fields = {
        "e231": actual_delta["e231"],
        "e543": actual_delta["e543"],
        "v39_e24": actual_delta["v39_e24"],
        **oracle,
        **ridge_fields,
    }
    q4_ids = [
        sample_id
        for sample_id in valid_ids
        if valid_targets[sample_id]["deltaT_quartile"] == "Q4"
    ]
    suites = {}
    for name, field in evaluated_fields.items():
        suite = _suite_from_fields(
            fields=field,
            ids=valid_ids,
            targets=valid_targets,
            stats=canonical_stats,
        )
        q4 = _suite_from_fields(
            fields=field,
            ids=q4_ids,
            targets=valid_targets,
            stats=canonical_stats,
        )
        suites[name] = {
            "summary": suite["summary"],
            "q4_summary": q4["summary"],
            "top_sse": _top_sse(suite),
            "per_sample": suite["per_sample"],
        }

    train_physics = {
        sample_id: _raw_physics(sample_id, cache) for sample_id in train_ids
    }
    valid_physics = {
        sample_id: _raw_physics(sample_id, cache) for sample_id in valid_ids
    }
    coverage_by_space = {}
    coverage_contract = {}
    for name, (train_matrix, valid_matrix) in matrices.items():
        coverage_by_space[name], coverage_contract[name] = _knn_coverage(
            train_ids=train_ids,
            valid_ids=valid_ids,
            train_matrix=train_matrix,
            valid_matrix=valid_matrix,
            train_log_scale=train_log_true,
        )

    per_sample_by_field = {
        name: {str(row["sample_id"]): row for row in suite["per_sample"]}
        for name, suite in suites.items()
    }
    sample_rows = []
    for sample_id in valid_ids:
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "role": "valid_iid",
            "deltaT_quartile": valid_targets[sample_id]["deltaT_quartile"],
            "true_scale_cv_rms_K": valid_targets[sample_id][
                "true_scale_cv_rms_K"
            ],
            **valid_physics[sample_id],
        }
        for name, values in per_sample_by_field.items():
            item = values[sample_id]
            row[f"{name}_point_sse_K2"] = item["point_error_squared_sum"]
            row[f"{name}_sample_cv_relative_rmse"] = item[
                "sample_cv_relative_rmse"
            ]
            row[f"{name}_raw_cv_rmse_K"] = math.sqrt(
                item["raw_cv_error_squared_integral_K2_m3"]
                / max(item["raw_cv_volume_m3"], EPS)
            )
            row[f"{name}_shape_cv_rmse"] = item["shape_cv_rmse"]
            row[f"{name}_signed_log_scale_error"] = item["scale_log_error"]
        for space, coverage in coverage_by_space.items():
            for field, value in coverage[sample_id].items():
                row[f"{space}_{field}"] = value
        sample_rows.append(row)

    coverage_rows = []
    for sample_id in valid_ids:
        row = {
            "sample_id": sample_id,
            "role": "valid_iid",
            "deltaT_quartile": valid_targets[sample_id]["deltaT_quartile"],
            "true_scale_cv_rms_K": valid_targets[sample_id][
                "true_scale_cv_rms_K"
            ],
        }
        for space, coverage in coverage_by_space.items():
            row.update(
                {
                    f"{space}_{field}": value
                    for field, value in coverage[sample_id].items()
                }
            )
        coverage_rows.append(row)

    stratified_field_names = (
        "e543",
        "e231_oracle_scale",
        "e543_oracle_scale",
        "v39_e24_oracle_scale",
        "ridge_physics_24",
        "ridge_pooled_latent_96",
        "ridge_combined_120",
    )
    strata_rows, strata_boundaries = _stratified_rows(
        fields={name: evaluated_fields[name] for name in stratified_field_names},
        valid_ids=valid_ids,
        targets=valid_targets,
        stats=canonical_stats,
        train_physics=train_physics,
        valid_physics=valid_physics,
    )

    coverage_diagnostics = {}
    for space in matrices:
        corresponding_field = {
            "physics_24": "ridge_physics_24",
            "pooled_latent_96": "ridge_pooled_latent_96",
            "combined_120": "ridge_combined_120",
        }[space]
        distances = [
            coverage_by_space[space][sample_id]["nearest_distance"]
            for sample_id in valid_ids
        ]
        variances = [
            coverage_by_space[space][sample_id][
                "neighbor_log_target_scale_variance"
            ]
            for sample_id in valid_ids
        ]
        errors = [
            per_sample_by_field[corresponding_field][sample_id][
                "point_error_squared_sum"
            ]
            for sample_id in valid_ids
        ]
        coverage_diagnostics[space] = {
            "contract": coverage_contract[space],
            "nearest_distance_distribution": {
                "mean": float(np.mean(distances)),
                "median": float(np.median(distances)),
                "p90": float(np.quantile(distances, 0.90)),
                "max": float(np.max(distances)),
            },
            "neighbor_log_target_scale_variance_distribution": {
                "mean": float(np.mean(variances)),
                "median": float(np.median(variances)),
                "p90": float(np.quantile(variances, 0.90)),
                "max": float(np.max(variances)),
            },
            "nearest_distance_vs_point_sse": _correlation(distances, errors),
            "neighbor_target_variance_vs_point_sse": _correlation(
                variances, errors
            ),
        }

    oracle_candidates = {
        name: suites[name]["summary"]["point_global_relative_rmse_pct"]
        for name in (
            "e231_oracle_scale",
            "e543_oracle_scale",
            "v39_e24_oracle_scale",
        )
    }
    ridge_candidates = {
        name: suites[name]["summary"]["point_global_relative_rmse_pct"]
        for name in (
            "ridge_physics_24",
            "ridge_pooled_latent_96",
            "ridge_combined_120",
        )
    }
    best_oracle_name = min(oracle_candidates, key=oracle_candidates.get)
    best_ridge_name = min(ridge_candidates, key=ridge_candidates.get)
    scale_only_below_20 = bool(oracle_candidates[best_oracle_name] < 20.0)
    ridge_below_20 = bool(ridge_candidates[best_ridge_name] < 20.0)
    combined_coverage_spearman = coverage_diagnostics["combined_120"][
        "nearest_distance_vs_point_sse"
    ]["spearman"]
    if not scale_only_below_20:
        classification = "representation"
        route = "improve frozen shape representation before further scale-only work"
    elif ridge_below_20:
        classification = "objective"
        route = (
            "replace nonlinear scale calibration with a preregistered "
            "combined-120D linear residual-scale head objective, preserving "
            "the e543 shape path and evaluating valid_iid once"
        )
    elif (
        combined_coverage_spearman is not None
        and combined_coverage_spearman > 0.5
    ):
        classification = "coverage"
        route = "coverage-targeted train expansion before scale-head retraining"
    else:
        classification = "representation"
        route = (
            "scale-head representation ablation using the frozen e543 shape path"
        )

    payload = {
        "schema_version": "heat3d_v5_gate6q_read_only_diagnostics_v1",
        "status": "completed_train_valid_only",
        "evaluator_commit": _git_commit(),
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "ridge_fit_roles": ["train"],
            "ridge_query_roles": ["valid_iid"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "training_yaml_generated": False,
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
        "gate6p_frozen_feature_binding": {
            "manifest_path": str(GATE6P_MANIFEST.relative_to(ROOT)),
            "feature_csv_sha256": gate6p_manifest["artifacts"][
                "gate6p_e543_scale_features.csv"
            ]["sha256"],
        },
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
            "e543_feature_export_max_abs_error_K": replay_error,
            "tolerance_K": 0.02,
            "passed": True,
        },
        "ridge_contract": {
            "alpha": RIDGE_ALPHA,
            "fit_roles": ["train"],
            "query_roles": ["valid_iid"],
            "valid_evaluation_count": 1,
            "target": "log(s_true)-log(s_phys)",
            "field_reconstruction": "e543 frozen predicted shape * exp(log_s_phys + ridge residual)",
            "models": ridge_models,
        },
        "field_metrics": {
            name: {
                "summary": suite["summary"],
                "q4_summary": suite["q4_summary"],
                "top_sse": suite["top_sse"],
            }
            for name, suite in suites.items()
        },
        "coverage_diagnostics": coverage_diagnostics,
        "conductivity_stratification": {
            "boundary_fit_roles": ["train"],
            "query_roles": ["valid_iid"],
            "fields": list(STRATIFY_FIELDS),
            "train_quartile_boundaries": strata_boundaries,
            "row_count": len(strata_rows),
        },
        "bottleneck_assessment": {
            "scale_only_theoretical_below_20pct": scale_only_below_20,
            "best_oracle_scale_field": best_oracle_name,
            "best_oracle_scale_point_global_relative_rmse_pct": (
                oracle_candidates[best_oracle_name]
            ),
            "fixed_ridge_below_20pct": ridge_below_20,
            "best_fixed_ridge_field": best_ridge_name,
            "best_fixed_ridge_point_global_relative_rmse_pct": (
                ridge_candidates[best_ridge_name]
            ),
            "classification": classification,
            "basis": (
                f"best oracle={best_oracle_name}/"
                f"{oracle_candidates[best_oracle_name]:.6f}%; "
                f"best fixed ridge={best_ridge_name}/"
                f"{ridge_candidates[best_ridge_name]:.6f}%; "
                f"combined coverage-distance Spearman="
                f"{combined_coverage_spearman}"
            ),
            "unique_recommended_route": route,
            "training_started": False,
        },
    }

    _write_json(paths["samples_json"], sample_rows)
    _write_csv(paths["samples_csv"], sample_rows, list(sample_rows[0]))
    _write_csv(
        paths["coverage_csv"], coverage_rows, list(coverage_rows[0])
    )
    _write_csv(paths["strata_csv"], strata_rows, list(strata_rows[0]))
    _write_json(paths["json"], payload)
    paths["md"].write_text(_markdown(payload), encoding="utf-8")
    manifest = {
        "schema_version": "heat3d_v5_gate6q_artifact_manifest_v1",
        "evaluator_commit": _git_commit(),
        "sample_level_row_count": len(sample_rows),
        "coverage_row_count": len(coverage_rows),
        "conductivity_strata_row_count": len(strata_rows),
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
                "scale_only_below_20pct": scale_only_below_20,
                "best_oracle": best_oracle_name,
                "best_ridge": best_ridge_name,
                "bottleneck": classification,
                "training_started": False,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
