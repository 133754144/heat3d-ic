#!/usr/bin/env python3
"""Gate 6O train/valid-only V38 checkpoint diagnostics and selection."""

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
from evaluate_heat3d_v5_gate6l_valid_only import (  # noqa: E402
    _build_groups,
    _ids_hash,
    _normalization_equal,
    _prediction_fields,
    _targets,
)
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


CONFIG_ID = "V4P5_38_gate6n_v36_r2r_mask_p005_e600"
CHECKPOINTS = {
    "e231": (
        "params_best_valid_point_global.pkl",
        "point_global_best_predictions.npz",
        231,
        "18a3382252858cfd4daf3c5cf4b8a585b903fbaf254e8159922c1df0fcc874bf",
    ),
    "e543": (
        "params_best_valid_sample_first.pkl",
        "sample_first_best_predictions.npz",
        543,
        "e26e50b432ebab2858fcf9c5597d2bb85715d181c48292c3cc85c7749b790869",
    ),
}
BOOTSTRAP_SEED = 2026071901
BOOTSTRAP_RESAMPLES = 20_000
EPS = 1.0e-12


class Gate6OError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
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


def _infer_raw_temperature(
    checkpoint: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    stats = dict(checkpoint["train_only_normalization"])
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    result: dict[str, np.ndarray] = {}
    for group in groups:
        prediction = _model_apply(model, params, group)
        raw = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        for index, sample_id in enumerate(group["sample_ids"]):
            result[str(sample_id)] = raw[index].reshape(-1)
    if any(
        field.size != 1024 or not np.all(np.isfinite(field))
        for field in result.values()
    ):
        raise Gate6OError("inference produced invalid fields")
    return result


def _delta_fields(
    raw: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        sample_id: np.asarray(raw[sample_id], dtype=np.float64)
        - float(targets[sample_id]["bottom_temperature_K"])
        for sample_id in ids
    }


def _paired_rows(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left_rows = {str(row["sample_id"]): row for row in left["per_sample"]}
    right_rows = {str(row["sample_id"]): row for row in right["per_sample"]}
    if set(left_rows) != set(right_rows) or set(left_rows) != set(targets):
        raise Gate6OError("paired sample IDs differ")
    rows = []
    for sample_id in sorted(left_rows):
        a = left_rows[sample_id]
        b = right_rows[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "deltaT_quartile": targets[sample_id]["deltaT_quartile"],
                "true_scale_cv_rms_K": a["true_scale_cv_rms_K"],
                "e231_point_sse_K2": a["point_error_squared_sum"],
                "e543_point_sse_K2": b["point_error_squared_sum"],
                "point_sse_delta_e543_minus_e231_K2": (
                    b["point_error_squared_sum"] - a["point_error_squared_sum"]
                ),
                "e231_sample_cv_relative_rmse": a["sample_cv_relative_rmse"],
                "e543_sample_cv_relative_rmse": b["sample_cv_relative_rmse"],
                "sample_cv_relative_delta_e543_minus_e231": (
                    b["sample_cv_relative_rmse"] - a["sample_cv_relative_rmse"]
                ),
                "e231_shape_cv_rmse": a["shape_cv_rmse"],
                "e543_shape_cv_rmse": b["shape_cv_rmse"],
                "shape_delta_e543_minus_e231": (
                    b["shape_cv_rmse"] - a["shape_cv_rmse"]
                ),
                "e231_scale_log_error": a["scale_log_error"],
                "e543_scale_log_error": b["scale_log_error"],
                "absolute_scale_log_error_delta_e543_minus_e231": (
                    abs(b["scale_log_error"]) - abs(a["scale_log_error"])
                ),
                "e543_point_sse_win": bool(
                    b["point_error_squared_sum"] < a["point_error_squared_sum"]
                ),
                "e543_sample_relative_win": bool(
                    b["sample_cv_relative_rmse"] < a["sample_cv_relative_rmse"]
                ),
            }
        )
    return rows


def _quartiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        selected = [row for row in rows if row["deltaT_quartile"] == quartile]
        sse = np.asarray(
            [row["point_sse_delta_e543_minus_e231_K2"] for row in selected],
            dtype=np.float64,
        )
        relative = np.asarray(
            [row["sample_cv_relative_delta_e543_minus_e231"] for row in selected],
            dtype=np.float64,
        )
        result.append(
            {
                "deltaT_quartile": quartile,
                "sample_count": len(selected),
                "e543_point_sse_win_rate": float(np.mean(sse < 0.0)),
                "point_sse_net_delta_e543_minus_e231_K2": float(np.sum(sse)),
                "point_sse_median_delta_e543_minus_e231_K2": float(
                    np.median(sse)
                ),
                "e543_sample_relative_win_rate": float(
                    np.mean(relative < 0.0)
                ),
                "sample_relative_mean_delta_e543_minus_e231": float(
                    np.mean(relative)
                ),
                "shape_mean_delta_e543_minus_e231": float(
                    np.mean(
                        [
                            row["shape_delta_e543_minus_e231"]
                            for row in selected
                        ]
                    )
                ),
                "absolute_scale_log_error_mean_delta_e543_minus_e231": float(
                    np.mean(
                        [
                            row[
                                "absolute_scale_log_error_delta_e543_minus_e231"
                            ]
                            for row in selected
                        ]
                    )
                ),
            }
        )
    return result


def _bootstrap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_rows = left["per_sample"]
    right_rows = right["per_sample"]
    if len(left_rows) != 128 or len(right_rows) != 128:
        raise Gate6OError("bootstrap requires 128 paired valid samples")
    arrays = {}
    for side, rows in (("left", left_rows), ("right", right_rows)):
        arrays[side] = {
            field: np.asarray([row[field] for row in rows], dtype=np.float64)
            for field in (
                "point_error_squared_sum",
                "point_true_squared_sum",
                "sample_cv_relative_rmse",
                "raw_cv_error_squared_integral_K2_m3",
                "raw_cv_volume_m3",
                "shape_cv_rmse",
                "scale_log_squared_error",
            )
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    observed: dict[str, float] = {}
    draws: dict[str, list[float]] = {
        name: []
        for name in (
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
        )
    }

    def metric(block: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
        return {
            "point_global_relative_rmse_pct": float(
                100.0
                * math.sqrt(
                    np.sum(block["point_error_squared_sum"][indices])
                    / np.sum(block["point_true_squared_sum"][indices])
                )
            ),
            "sample_first_cv_relative_rmse_pct": float(
                100.0 * np.mean(block["sample_cv_relative_rmse"][indices])
            ),
            "raw_cv_weighted_rmse_K": float(
                math.sqrt(
                    np.sum(
                        block["raw_cv_error_squared_integral_K2_m3"][indices]
                    )
                    / np.sum(block["raw_cv_volume_m3"][indices])
                )
            ),
            "shape_cv_rmse": float(np.mean(block["shape_cv_rmse"][indices])),
            "scale_log_rmse": float(
                math.sqrt(np.mean(block["scale_log_squared_error"][indices]))
            ),
        }

    all_indices = np.arange(128)
    left_observed = metric(arrays["left"], all_indices)
    right_observed = metric(arrays["right"], all_indices)
    for name in draws:
        observed[name] = right_observed[name] - left_observed[name]
    chunk = 500
    for _ in range(BOOTSTRAP_RESAMPLES // chunk):
        indices = rng.integers(0, 128, size=(chunk, 128))
        for row in indices:
            left_metric = metric(arrays["left"], row)
            right_metric = metric(arrays["right"], row)
            for name in draws:
                draws[name].append(right_metric[name] - left_metric[name])
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "difference_direction": "e543_minus_e231",
        "metrics": {
            name: {
                "observed_difference": observed[name],
                "ci95": [
                    float(np.quantile(draws[name], 0.025)),
                    float(np.quantile(draws[name], 0.975)),
                ],
                "probability_e543_better": float(
                    np.mean(np.asarray(draws[name]) < 0.0)
                ),
            }
            for name in draws
        },
    }


def _fit_log_affine(
    fields: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    _, predicted_scales = _decompose_fields(fields, ids, targets)
    x = np.log(
        np.asarray([predicted_scales[sample_id] for sample_id in ids])
    )
    y = np.log(
        np.asarray(
            [targets[sample_id]["true_scale_cv_rms_K"] for sample_id in ids]
        )
    )
    design = np.stack([np.ones_like(x), x], axis=1)
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "fit_sample_count": len(ids),
        "fit_log_rmse": float(np.sqrt(np.mean(np.square(fitted - y)))),
    }


def _apply_log_affine(
    raw_fields: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, float],
) -> dict[str, np.ndarray]:
    shapes, scales = _decompose_fields(raw_fields, ids, targets)
    return {
        sample_id: shapes[sample_id]
        * math.exp(
            float(calibration["intercept"])
            + float(calibration["slope"]) * math.log(scales[sample_id])
        )
        for sample_id in ids
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 6O train/valid-only diagnostics",
        "",
        "仅访问 `train` 与 `valid_iid`；`test/hard/sealed` 均未访问。",
        "",
        "## Valid metrics",
        "",
        "| field | point-global % | sample-first % | raw CV K | shape CV | scale log |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["field_metrics"].items():
        row = metrics["summary"]
        lines.append(
            f"| {name} | {row['point_global_relative_rmse_pct']:.6f} | "
            f"{row['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{row['raw_cv_weighted_rmse_K']:.6f} | "
            f"{row['shape_cv_rmse']:.6f} | "
            f"{row['scale_log_rmse']:.6f} |"
        )
    selection = payload["stage2_selection"]
    lines += [
        "",
        "## Stage 2 selection",
        "",
        f"- selected: `{selection['selected_checkpoint']}`",
        f"- checkpoint epoch: `{selection['selected_epoch']}`",
        f"- basis: {selection['basis']}",
        "",
        "Stage 2 冻结 shape 路径，因此预注册 primary 是 valid shape "
        "CV-RMSE；sample-first 与 raw CV RMSE 依次作为 tie-break。branch "
        "swap、ensemble 与 affine calibration 仅作机制诊断，不改变选择规则。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    outputs = {
        "json": output_dir / "gate6o_diagnostics.json",
        "md": output_dir / "gate6o_diagnostics.md",
        "paired": output_dir / "gate6o_paired_samples.csv",
        "quartiles": output_dir / "gate6o_quartiles.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise Gate6OError("output already exists")
    if run_dir.name != CONFIG_ID:
        raise Gate6OError("run directory/config binding failed")
    run_config = json.loads((run_dir / "run_config.json").read_text())
    summary = json.loads((run_dir / "loss_summary.json").read_text())
    if int(summary.get("final_epoch", -1)) != 600:
        raise Gate6OError("V38 did not complete e600")

    checkpoints = {}
    checkpoint_binding = {}
    for name, (filename, _, epoch, digest) in CHECKPOINTS.items():
        path = run_dir / filename
        if _sha256(path) != digest:
            raise Gate6OError(f"{name}: checkpoint SHA256 drifted")
        checkpoint = _load_params_checkpoint(path)
        if int(checkpoint["epoch"]) != epoch:
            raise Gate6OError(f"{name}: checkpoint epoch drifted")
        checkpoints[name] = checkpoint
        checkpoint_binding[name] = {
            "path": str(path),
            "epoch": epoch,
            "sha256": digest,
            "training_commit": str(checkpoint.get("git_commit") or ""),
        }

    canonical_stats = dict(checkpoints["e231"]["train_only_normalization"])
    if not _normalization_equal(
        canonical_stats, checkpoints["e543"]["train_only_normalization"]
    ):
        raise Gate6OError("checkpoint normalization differs")
    install_checkpoint_feature_hooks(canonical_stats)
    train_examples = load_training_examples(run_config, canonical_stats)
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
        raise Gate6OError("train-only normalization does not reproduce")
    stats = stats_from_checkpoint_payload(canonical_stats, train_examples)
    sample_root = _sample_root(Path(run_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(run_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6OError("split counts drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=canonical_stats,
        boundary_mask_fallback=bool(
            run_config.get("boundary_mask_fallback", True)
        ),
    )
    cache = _physics_cache(list(train_examples) + list(valid_examples))
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    train_groups = _build_groups(
        run_config=run_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=train_examples,
        valid_ids=train_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    valid_groups = _build_groups(
        run_config=run_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=valid_examples,
        valid_ids=valid_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    train_targets = _targets(sample_root=sample_root, valid_ids=train_ids)
    valid_targets = _targets(sample_root=sample_root, valid_ids=valid_ids)

    raw = {"train": {}, "valid_iid": {}}
    replay = {}
    for name, checkpoint in checkpoints.items():
        raw["train"][name] = _infer_raw_temperature(checkpoint, train_groups)
        raw["valid_iid"][name] = _infer_raw_temperature(
            checkpoint, valid_groups
        )
        saved = _prediction_fields(run_dir / CHECKPOINTS[name][1], valid_ids)
        maximum = max(
            float(
                np.max(
                    np.abs(raw["valid_iid"][name][sample_id] - saved[sample_id])
                )
            )
            for sample_id in valid_ids
        )
        replay[name] = {
            "max_abs_error_K": maximum,
            "tolerance_K": 0.02,
            "passed": bool(maximum <= 0.02),
        }
        if not replay[name]["passed"]:
            raise Gate6OError(f"{name}: saved prediction replay failed")

    train_delta = {
        name: _delta_fields(fields, train_ids, train_targets)
        for name, fields in raw["train"].items()
    }
    valid_delta = {
        name: _delta_fields(fields, valid_ids, valid_targets)
        for name, fields in raw["valid_iid"].items()
    }
    valid_decomposed = {
        name: _decompose_fields(raw["valid_iid"][name], valid_ids, valid_targets)
        for name in CHECKPOINTS
    }
    derived = {
        "shape_e231+scale_e543": {
            sample_id: (
                valid_decomposed["e231"][0][sample_id]
                * valid_decomposed["e543"][1][sample_id]
            )
            for sample_id in valid_ids
        },
        "shape_e543+scale_e231": {
            sample_id: (
                valid_decomposed["e543"][0][sample_id]
                * valid_decomposed["e231"][1][sample_id]
            )
            for sample_id in valid_ids
        },
        "ensemble_alpha_0.5": {
            sample_id: 0.5
            * (valid_delta["e231"][sample_id] + valid_delta["e543"][sample_id])
            for sample_id in valid_ids
        },
    }
    calibrations = {}
    for name in CHECKPOINTS:
        calibration = _fit_log_affine(
            raw["train"][name], train_ids, train_targets
        )
        calibrations[name] = {
            **calibration,
            "fit_roles": ["train"],
            "target_access_roles": ["train"],
            "formula": "log(s_true)=intercept+slope*log(s_pred)",
        }
        derived[f"{name}_train_affine_scale"] = _apply_log_affine(
            raw["valid_iid"][name],
            valid_ids,
            valid_targets,
            calibration,
        )
    fields = {**valid_delta, **derived}
    suites = {
        name: _suite_from_fields(
            fields=field,
            ids=valid_ids,
            targets=valid_targets,
            stats=canonical_stats,
        )
        for name, field in fields.items()
    }
    paired = _paired_rows(suites["e231"], suites["e543"], valid_targets)
    quartiles = _quartiles(paired)
    bootstrap = _bootstrap(suites["e231"], suites["e543"])

    selection_order = (
        "shape_cv_rmse",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
    )
    candidates = {
        name: suites[name]["summary"] for name in ("e231", "e543")
    }
    selected = min(
        candidates,
        key=lambda name: tuple(candidates[name][metric] for metric in selection_order),
    )
    selected_epoch = CHECKPOINTS[selected][2]
    payload = {
        "schema_version": "heat3d_v5_gate6o_diagnostics_v1",
        "status": "completed_train_valid_only",
        "evaluator_commit": _git_commit(),
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "evaluation_roles": ["valid_iid"],
            "calibration_fit_roles": ["train"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "model_parameters_modified": False,
        },
        "split": {
            "source": split_source,
            "train_count": 672,
            "valid_iid_count": 128,
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "checkpoint_binding": checkpoint_binding,
        "normalization_and_context": {
            "normalization_recomputed_from_train_only": True,
            "context_fit_roles": ["train"],
            "context_fit_sample_count": 672,
            "context_fit_sample_ids_sha256": standardizer[
                "fit_sample_ids_sha256"
            ],
            "target_or_label_features": [],
        },
        "checkpoint_replay": replay,
        "formulas": {
            "branch_swap": "CV-decompose predicted DeltaT and combine shape donor phi with scale donor s",
            "ensemble": "0.5*DeltaT_e231+0.5*DeltaT_e543",
            "affine_scale_calibration": "fit log(s_true)=a+b*log(s_pred) on train only; preserve checkpoint shape on valid_iid",
        },
        "field_metrics": {
            name: {"summary": suite["summary"]} for name, suite in suites.items()
        },
        "paired_bootstrap": bootstrap,
        "quartile_attribution": quartiles,
        "affine_scale_calibration": calibrations,
        "stage2_selection": {
            "selection_frozen_before_gate6o_diagnostic_execution": True,
            "eligible_checkpoints": ["e231", "e543"],
            "primary_metric": selection_order[0],
            "tie_break_metrics": list(selection_order[1:]),
            "reason_for_primary": (
                "Stage 2 freezes backbone/processor/decoder/scale attention, "
                "so checkpoint shape quality is immutable"
            ),
            "diagnostic_transforms_ineligible_for_initialization_selection": [
                "branch_swap",
                "ensemble",
                "train_affine_scale",
            ],
            "selected_checkpoint": selected,
            "selected_epoch": selected_epoch,
            "selected_path": str(run_dir / CHECKPOINTS[selected][0]),
            "selected_sha256": CHECKPOINTS[selected][3],
            "basis": (
                f"{selected} has the lower valid shape CV-RMSE under the "
                "pre-registered frozen-shape rule"
            ),
        },
    }
    _write_json(outputs["json"], payload)
    outputs["md"].write_text(_markdown(payload), encoding="utf-8")
    _write_csv(
        outputs["paired"],
        paired,
        tuple(paired[0]),
    )
    _write_csv(
        outputs["quartiles"],
        quartiles,
        tuple(quartiles[0]),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_checkpoint": selected,
                "selected_epoch": selected_epoch,
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
