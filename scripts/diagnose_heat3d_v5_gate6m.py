#!/usr/bin/env python3
"""Gate 6M valid-only branch swapping and shared-backbone gradient audit."""

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
from rigno.heat3d_v5_metrics import evaluate_metric_suite  # noqa: E402
from diagnose_heat3d_v5_gate6a import (  # noqa: E402
    LOSS_NAMES,
    _aggregate_gradient,
    _cosine,
    _gradient_summary,
)
from evaluate_heat3d_v5_gate6l_valid_only import (  # noqa: E402
    V32_CHECKPOINT,
    _build_groups,
    _ids_hash,
    _normalization_equal,
    _prediction_fields,
    _run_binding,
    _sha256,
    _targets,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    _device_params,
    _load_params_checkpoint,
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
    _load_examples,
    _physics_cache,
)
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402


V32_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
O075_ID = "V4P5_33_gate6k_o075_log_scale"
O075_CHECKPOINT = "params_best_valid_point_global.pkl"
V32_PREDICTIONS = "point_global_best_predictions.npz"
O075_PREDICTIONS = "point_global_best_predictions.npz"
GATE6L = ROOT / "configs/heat3d_v5/gate6l/gate6l_valid_only_evaluation.json"
EPS = 1.0e-12
PHYSICS_FIELDS = (
    "P_operator_W",
    "source_concentration",
    "q_weighted_inverse_kz_mK_W",
    "anisotropy_xy_over_z",
    "log_top_h_W_m2K",
)


class Gate6MError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v32-run-dir", type=Path, required=True)
    parser.add_argument("--o075-run-dir", type=Path, required=True)
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
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fields} for row in rows]
        )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _suite_from_fields(
    *,
    fields: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    samples = []
    for sample_id in ids:
        target = targets[sample_id]
        prediction = np.asarray(fields[sample_id], dtype=np.float64)
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


def _decompose_fields(
    raw_temperature: Mapping[str, np.ndarray],
    ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    shapes: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for sample_id in ids:
        target = targets[sample_id]
        delta = (
            np.asarray(raw_temperature[sample_id], dtype=np.float64)
            - float(target["bottom_temperature_K"])
        )
        volumes = np.asarray(target["control_volumes_m3"], dtype=np.float64)
        scale = math.sqrt(
            float(np.sum(np.square(delta) * volumes) / np.sum(volumes))
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise Gate6MError(f"{sample_id}: invalid predicted scale")
        shapes[sample_id] = delta / scale
        scales[sample_id] = scale
    return shapes, scales


def _paired_rows(
    *,
    reference_name: str,
    candidate_name: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left = {str(row["sample_id"]): row for row in reference["per_sample"]}
    right = {str(row["sample_id"]): row for row in candidate["per_sample"]}
    if set(left) != set(right) or set(left) != set(targets):
        raise Gate6MError("paired sample IDs differ")
    result = []
    for sample_id in sorted(left):
        delta = float(
            right[sample_id]["point_error_squared_sum"]
            - left[sample_id]["point_error_squared_sum"]
        )
        context = contexts[sample_id]["context"]
        result.append(
            {
                "comparison": f"{candidate_name}_minus_{reference_name}",
                "sample_id": sample_id,
                "deltaT_quartile": targets[sample_id]["deltaT_quartile"],
                "point_sse_delta_K2": delta,
                "outcome": "win" if delta < 0.0 else "loss" if delta > 0.0 else "tie",
                **{field: float(context[field]) for field in PHYSICS_FIELDS},
            }
        )
    return result


def _quartile_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        selected = [row for row in rows if row["deltaT_quartile"] == quartile]
        deltas = np.asarray(
            [float(row["point_sse_delta_K2"]) for row in selected],
            dtype=np.float64,
        )
        result.append(
            {
                "comparison": selected[0]["comparison"],
                "deltaT_quartile": quartile,
                "sample_count": len(selected),
                "win_count": sum(row["outcome"] == "win" for row in selected),
                "loss_count": sum(row["outcome"] == "loss" for row in selected),
                "tie_count": sum(row["outcome"] == "tie" for row in selected),
                "win_rate": float(np.mean(deltas < 0.0)),
                "point_sse_net_delta_K2": float(np.sum(deltas)),
                "point_sse_median_delta_K2": float(np.median(deltas)),
                "point_sse_absolute_delta_fraction": float(
                    np.sum(np.abs(deltas))
                    / max(
                        sum(
                            abs(float(row["point_sse_delta_K2"]))
                            for row in rows
                        ),
                        EPS,
                    )
                ),
            }
        )
    return result


def _physical_attribution(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = np.asarray(
        [float(row["point_sse_delta_K2"]) for row in rows], dtype=np.float64
    )
    fields: dict[str, Any] = {}
    for field in PHYSICS_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        edges[0] = np.nextafter(edges[0], -np.inf)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        quartiles = []
        for index in range(4):
            mask = (values > edges[index]) & (values <= edges[index + 1])
            quartiles.append(
                {
                    "quartile": index + 1,
                    "sample_count": int(np.sum(mask)),
                    "point_sse_mean_delta_K2": float(np.mean(deltas[mask])),
                    "point_sse_net_delta_K2": float(np.sum(deltas[mask])),
                    "win_rate": float(np.mean(deltas[mask] < 0.0)),
                }
            )
        fields[field] = {
            "pearson_with_point_sse_delta": _correlation(values, deltas),
            "spearman_with_point_sse_delta": _correlation(
                _rankdata(values), _rankdata(deltas)
            ),
            "quartiles": quartiles,
        }
    return {
        "feature_provenance": (
            "direct inference-time coords/k/q/BC/control-volume context; "
            "no target, train fit, or learned representation"
        ),
        "fields": fields,
    }


def _gradient_audit(
    *,
    checkpoint: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]),
        dict(checkpoint["train_only_normalization"]),
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    gradients = {}
    means = {}
    for loss_name in LOSS_NAMES:
        means[loss_name], gradients[loss_name], count = _aggregate_gradient(
            model, params, groups, loss_name
        )
        if count != 128:
            raise Gate6MError("gradient audit did not cover 128 valid samples")
    cosine = {
        left: {
            right: _cosine(gradients[left], gradients[right])["backbone"]
            for right in LOSS_NAMES
        }
        for left in LOSS_NAMES
    }
    return {
        "sample_count": 128,
        "loss_means": means,
        "shared_backbone_gradient_norms": {
            name: _gradient_summary(gradient)["parameter_group_norms"][
                "backbone"
            ]
            for name, gradient in gradients.items()
        },
        "shared_backbone_gradient_cosine": cosine,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 6M valid-only branch swap and gradient audit",
        "",
        "本轮没有训练或 checkpoint 修改。仅访问 `train`（重建 normalization/context）"
        "与 `valid_iid`（评估）；`test/hard/sealed` 均未访问。",
        "",
        "## Branch swapping",
        "",
        "| field | point-global % | sample-first % | raw CV K | shape CV | scale log |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, suite in payload["field_metrics"].items():
        summary = suite["summary"]
        lines.append(
            f"| {name} | {summary['point_global_relative_rmse_pct']:.4f} | "
            f"{summary['sample_first_cv_relative_rmse_pct']:.4f} | "
            f"{summary['raw_cv_weighted_rmse_K']:.6f} | "
            f"{summary['shape_cv_rmse']:.6f} | "
            f"{summary['scale_log_rmse']:.6f} |"
        )
    lines += [
        "",
        "## Shared-backbone gradient cosine",
        "",
    ]
    for model_name, audit in payload["gradient_audit"].items():
        lines += [
            f"### {model_name}",
            "",
            "| loss | shape | scale | relative | raw |",
            "|---|---:|---:|---:|---:|",
        ]
        labels = dict(zip(LOSS_NAMES, ("shape", "scale", "relative", "raw")))
        for left in LOSS_NAMES:
            row = audit["shared_backbone_gradient_cosine"][left]
            lines.append(
                f"| {labels[left]} | "
                + " | ".join(
                    "n/a" if row[right] is None else f"{row[right]:.4f}"
                    for right in LOSS_NAMES
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Frozen interpretation",
        "",
        payload["conclusion"],
        "",
        "Q1–Q4 win/loss、逐样本 point-SSE 差值和 inference-only 物理条件归因"
        "见 JSON/CSV。本结果不触发自动晋级或训练。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "json": output_dir / "gate6m_branch_swap_gradient_audit.json",
        "paired": output_dir / "gate6m_branch_swap_paired_samples.csv",
        "quartiles": output_dir / "gate6m_branch_swap_quartiles.csv",
        "cosine": output_dir / "gate6m_backbone_gradient_cosine.csv",
        "md": output_dir / "gate6m_branch_swap_gradient_audit.md",
    }
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise Gate6MError("Gate 6M output already exists")

    frozen = json.loads(GATE6L.read_text(encoding="utf-8"))
    v32_run = args.v32_run_dir.resolve()
    o075_run = args.o075_run_dir.resolve()
    o075_config, _ = _run_binding(o075_run, O075_ID)
    v32_config = json.loads((v32_run / "run_config.json").read_text())
    if v32_run.name != V32_ID or Path(v32_config["output_dir"]).name != V32_ID:
        raise Gate6MError("V32 run binding failed")
    expected_v32_sha = frozen["v32_reference"]["sha256"]
    expected_o075_sha = frozen["models"]["O075"]["checkpoint_metadata"][
        "point_global_best"
    ]["sha256"]
    if _sha256(v32_run / V32_CHECKPOINT) != expected_v32_sha:
        raise Gate6MError("V32 checkpoint SHA256 drifted")
    if _sha256(o075_run / O075_CHECKPOINT) != expected_o075_sha:
        raise Gate6MError("O075 checkpoint SHA256 drifted")

    checkpoints = {
        "V32": _load_params_checkpoint(v32_run / V32_CHECKPOINT),
        "O075": _load_params_checkpoint(o075_run / O075_CHECKPOINT),
    }
    canonical_stats = dict(checkpoints["V32"]["train_only_normalization"])
    if not _normalization_equal(
        canonical_stats, checkpoints["O075"]["train_only_normalization"]
    ):
        raise Gate6MError("checkpoint normalization differs")
    install_checkpoint_feature_hooks(canonical_stats)
    train_examples = load_training_examples(o075_config, canonical_stats)
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
        raise Gate6MError("train-only normalization does not reproduce")
    stats = stats_from_checkpoint_payload(canonical_stats, train_examples)
    sample_root = _sample_root(Path(o075_config["subset"]))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(o075_config["split_map_path"])
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6MError("split count drifted")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=canonical_stats,
        boundary_mask_fallback=bool(
            o075_config.get("boundary_mask_fallback", True)
        ),
    )
    cache = _physics_cache(list(train_examples) + list(valid_examples))
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    groups = _build_groups(
        run_config=o075_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=valid_examples,
        valid_ids=valid_ids,
        cache=cache,
        standardizer=standardizer,
        batch_size=args.prediction_batch_size,
    )
    targets = _targets(sample_root=sample_root, valid_ids=valid_ids)

    raw = {
        "V32": _prediction_fields(v32_run / V32_PREDICTIONS, valid_ids),
        "O075": _prediction_fields(o075_run / O075_PREDICTIONS, valid_ids),
    }
    decomposed = {
        name: _decompose_fields(fields, valid_ids, targets)
        for name, fields in raw.items()
    }
    delta_fields = {
        name: {
            sample_id: fields[sample_id]
            - float(targets[sample_id]["bottom_temperature_K"])
            for sample_id in valid_ids
        }
        for name, fields in raw.items()
    }
    delta_fields["shape_V32+scale_O075"] = {
        sample_id: decomposed["V32"][0][sample_id]
        * decomposed["O075"][1][sample_id]
        for sample_id in valid_ids
    }
    delta_fields["shape_O075+scale_V32"] = {
        sample_id: decomposed["O075"][0][sample_id]
        * decomposed["V32"][1][sample_id]
        for sample_id in valid_ids
    }
    suites = {
        name: _suite_from_fields(
            fields=fields,
            ids=valid_ids,
            targets=targets,
            stats=canonical_stats,
        )
        for name, fields in delta_fields.items()
    }

    pair_specs = (
        ("V32", "shape_V32+scale_O075"),
        ("V32", "shape_O075+scale_V32"),
        ("O075", "shape_V32+scale_O075"),
        ("O075", "shape_O075+scale_V32"),
    )
    paired_rows = []
    quartile_rows = []
    physical = {}
    for reference, candidate in pair_specs:
        rows = _paired_rows(
            reference_name=reference,
            candidate_name=candidate,
            reference=suites[reference],
            candidate=suites[candidate],
            targets=targets,
            contexts=cache,
        )
        paired_rows.extend(rows)
        quartile_rows.extend(_quartile_outcomes(rows))
        physical[rows[0]["comparison"]] = _physical_attribution(rows)

    gradient_audit = {
        name: _gradient_audit(checkpoint=checkpoint, groups=groups)
        for name, checkpoint in checkpoints.items()
    }
    cosine_rows = []
    for model_name, audit in gradient_audit.items():
        for left in LOSS_NAMES:
            for right in LOSS_NAMES:
                cosine_rows.append(
                    {
                        "model": model_name,
                        "left_loss": left,
                        "right_loss": right,
                        "shared_backbone_cosine": audit[
                            "shared_backbone_gradient_cosine"
                        ][left][right],
                    }
                )

    v32_point = suites["V32"]["summary"][
        "point_global_relative_rmse_pct"
    ]
    o075_point = suites["O075"]["summary"][
        "point_global_relative_rmse_pct"
    ]
    swap_points = {
        name: suites[name]["summary"]["point_global_relative_rmse_pct"]
        for name in ("shape_V32+scale_O075", "shape_O075+scale_V32")
    }
    conclusion = (
        f"V32={v32_point:.4f}%，O075={o075_point:.4f}%；"
        f"shape_V32+scale_O075={swap_points['shape_V32+scale_O075']:.4f}%，"
        f"shape_O075+scale_V32={swap_points['shape_O075+scale_V32']:.4f}%。"
        "该交换只用于因果诊断，不重新选择 checkpoint。"
    )
    payload = {
        "schema_version": "heat3d_v5_gate6m_valid_only_v1",
        "status": "completed_valid_iid_only",
        "evaluator_commit": _git_commit(),
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "evaluation_roles": ["valid_iid"],
            "forbidden_roles_accessed": [],
            "training_started": False,
            "model_parameters_modified": False,
            "checkpoint_selection_modified": False,
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
        },
        "split": {
            "source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "checkpoint_binding": {
            "V32": {
                "config_id": V32_ID,
                "epoch": int(checkpoints["V32"]["epoch"]),
                "sha256": expected_v32_sha,
            },
            "O075": {
                "config_id": O075_ID,
                "epoch": int(checkpoints["O075"]["epoch"]),
                "sha256": expected_o075_sha,
            },
        },
        "normalization_and_context": {
            "fit_roles": ["train"],
            "fit_sample_count": 672,
            "fit_sample_ids_sha256": standardizer[
                "fit_sample_ids_sha256"
            ],
            "target_or_label_features": [],
        },
        "branch_swap_formula": (
            "CV-normalize predicted DeltaT into phi and s; "
            "reconstruct s_scale_donor * phi_shape_donor"
        ),
        "field_metrics": {
            name: {"summary": suite["summary"]}
            for name, suite in suites.items()
        },
        "gradient_audit": gradient_audit,
        "quartile_win_loss": quartile_rows,
        "physical_condition_attribution": physical,
        "conclusion": conclusion,
    }
    _write_json(outputs["json"], payload)
    _write_csv(
        outputs["paired"],
        paired_rows,
        (
            "comparison",
            "sample_id",
            "deltaT_quartile",
            "point_sse_delta_K2",
            "outcome",
            *PHYSICS_FIELDS,
        ),
    )
    _write_csv(
        outputs["quartiles"],
        quartile_rows,
        (
            "comparison",
            "deltaT_quartile",
            "sample_count",
            "win_count",
            "loss_count",
            "tie_count",
            "win_rate",
            "point_sse_net_delta_K2",
            "point_sse_median_delta_K2",
            "point_sse_absolute_delta_fraction",
        ),
    )
    _write_csv(
        outputs["cosine"],
        cosine_rows,
        ("model", "left_loss", "right_loss", "shared_backbone_cosine"),
    )
    outputs["md"].write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed_valid_iid_only",
                "outputs": {
                    key: str(path) for key, path in outputs.items()
                },
                "training_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
