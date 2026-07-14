#!/usr/bin/env python3
"""Gate 6A train/valid-only loss and gradient diagnosis for N3 best e402."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v5_global_context import fit_train_only_standardizer  # noqa: E402
from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    field_layout,
    free_field,
    native_shape_scale_losses,
    target_shape_scale,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
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


LOSS_NAMES = (
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
)
PARAMETER_GROUPS = ("backbone", "shape_decoder", "scale_head", "film", "bypass")
SPLITS = ("train", "valid_iid")
DEFAULT_INPUT = ROOT / "output/heat3d_v5_gate6_inputs/N3_best_e402"
DEFAULT_CONTRACT = ROOT / "configs/heat3d_v5/v5_gate5_final_evaluator_contract.json"


class Gate6ADiagnosticError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6ADiagnosticError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_input_dir(path: Path) -> dict[str, str]:
    required = (
        "params_best.pkl",
        "run_config.json",
        "loss_summary.json",
        "normalization_context_metadata.json",
    )
    expected: dict[str, str] = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[name.strip()] = checksum
    if set(expected) != set(required):
        raise Gate6ADiagnosticError("SHA256SUMS does not contain the frozen Gate 6 input set")
    actual = {name: _sha256(path / name) for name in required}
    if actual != expected:
        raise Gate6ADiagnosticError("Gate 6 input SHA256 verification failed")
    return actual


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _parameter_group(path: Any) -> str:
    names = tuple(str(getattr(item, "key", getattr(item, "name", item))) for item in path)
    joined = "/".join(names)
    if "global_film_" in joined:
        return "film"
    if "decoder_bypass_" in joined:
        return "bypass"
    if "global_scale_" in joined:
        return "scale_head"
    if "decoder" in joined:
        return "shape_decoder"
    return "backbone"


def _tree_linear_combination(trees: Sequence[Any], coefficients: Sequence[float]) -> Any:
    if len(trees) != len(coefficients) or not trees:
        raise Gate6ADiagnosticError("gradient tree combination is empty or misaligned")
    return tree_util.tree_map(
        lambda *values: sum(float(weight) * value for weight, value in zip(coefficients, values, strict=True)),
        *trees,
    )


def _tree_scale(tree: Any, scale: float) -> Any:
    return tree_util.tree_map(lambda value: float(scale) * value, tree)


def _group_dot(left: Any, right: Any) -> dict[str, float]:
    left_leaves = tree_util.tree_flatten_with_path(left)[0]
    right_leaves = tree_util.tree_flatten_with_path(right)[0]
    if len(left_leaves) != len(right_leaves):
        raise Gate6ADiagnosticError("gradient trees differ")
    result = {name: 0.0 for name in PARAMETER_GROUPS}
    for (left_path, left_value), (right_path, right_value) in zip(
        left_leaves, right_leaves, strict=True
    ):
        if str(left_path) != str(right_path):
            raise Gate6ADiagnosticError("gradient tree paths differ")
        group = _parameter_group(left_path)
        result[group] += float(jnp.sum(jnp.asarray(left_value) * jnp.asarray(right_value)))
    return result


def _gradient_summary(gradient: Any) -> dict[str, Any]:
    squared = _group_dot(gradient, gradient)
    norms = {name: math.sqrt(max(value, 0.0)) for name, value in squared.items()}
    global_norm = math.sqrt(sum(squared.values()))
    return {
        "global_norm": global_norm,
        "parameter_group_norms": norms,
        "parameter_group_fraction_of_global_norm": {
            name: value / max(global_norm, 1.0e-30) for name, value in norms.items()
        },
    }


def _cosine(left: Any, right: Any) -> dict[str, float | None]:
    dot = _group_dot(left, right)
    left_sq = _group_dot(left, left)
    right_sq = _group_dot(right, right)
    result: dict[str, float | None] = {}
    for group in PARAMETER_GROUPS:
        denominator = math.sqrt(max(left_sq[group], 0.0) * max(right_sq[group], 0.0))
        result[group] = dot[group] / denominator if denominator > 0.0 else None
    global_dot = sum(dot.values())
    global_denominator = math.sqrt(sum(left_sq.values()) * sum(right_sq.values()))
    result["global"] = global_dot / global_denominator if global_denominator > 0.0 else None
    return result


def _loss_for_group(model: Any, params: Any, group: Mapping[str, Any], loss_name: str) -> Any:
    prediction = _model_apply(model, params, group)
    physics = group["native_physics"]
    components = native_shape_scale_losses(
        prediction,
        target_deltaT=group["target_delta_raw"],
        control_volumes=physics["control_volumes"],
        dirichlet_mask=physics["dirichlet_mask"],
        loss_weights={
            "shape_cv": 1.0,
            "log_scale": 1.0,
            "relative_field": 1.0,
            "raw_absolute": 1.0,
        },
    )
    return components[loss_name]


def _aggregate_gradient(
    model: Any, params: Any, groups: Sequence[Mapping[str, Any]], loss_name: str
) -> tuple[float, Any, int]:
    weighted_gradients = []
    weighted_losses = []
    total_count = 0
    for group in groups:
        count = int(group["target_delta_raw"].shape[0])
        value, gradient = jax.value_and_grad(
            lambda current_params, current_group=group: _loss_for_group(
                model, current_params, current_group, loss_name
            )
        )(params)
        weighted_losses.append(float(value) * count)
        weighted_gradients.append(_tree_scale(gradient, count))
        total_count += count
    gradient = _tree_scale(
        _tree_linear_combination(weighted_gradients, [1.0] * len(weighted_gradients)),
        1.0 / total_count,
    )
    return sum(weighted_losses) / total_count, gradient, total_count


def _per_sample_rows(
    model: Any,
    params: Any,
    groups: Sequence[Mapping[str, Any]],
    split: str,
    context_cache: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    field_error = {name: 0.0 for name in ("joint", "oracle_scale", "oracle_shape", "physics_scale")}
    field_truth = 0.0
    for group in groups:
        prediction = _model_apply(model, params, group)
        physics = group["native_physics"]
        phi_hat = np.asarray(prediction["phi_hat"], dtype=np.float64)
        s_hat = np.asarray(prediction["s_hat"], dtype=np.float64).reshape(-1)
        delta_hat = np.asarray(prediction["deltaT_hat"], dtype=np.float64)
        target = np.asarray(group["target_delta_raw"], dtype=np.float64)
        weights = np.asarray(physics["control_volumes"], dtype=np.float64)
        mask = np.asarray(physics["dirichlet_mask"], dtype=np.float64)
        log_s_phys = np.asarray(physics["log_s_phys"], dtype=np.float64).reshape(-1)
        free_target = target * (1.0 - np.clip(mask, 0.0, 1.0))
        volume_sum = np.sum(weights, axis=2, keepdims=True)
        true_scale = np.sqrt(np.sum(np.square(free_target) * weights, axis=2, keepdims=True) / volume_sum)
        phi_true = free_target / np.maximum(true_scale, 1.0e-12)
        shape_loss = np.sum(np.square(phi_hat - phi_true) * weights, axis=2, keepdims=True) / volume_sum
        raw_loss = np.sum(np.square(delta_hat - free_target) * weights, axis=2, keepdims=True) / volume_sum
        relative_loss = raw_loss / np.maximum(np.square(true_scale), 1.0e-12)
        scale_loss = np.square(np.log(np.maximum(s_hat, 1.0e-12)) - np.log(np.maximum(true_scale.reshape(-1), 1.0e-12)))
        fields = {
            "joint": delta_hat,
            "oracle_scale": true_scale * phi_hat,
            "oracle_shape": s_hat[:, None, None, None] * phi_true,
            "physics_scale": np.exp(log_s_phys)[:, None, None, None] * phi_hat,
        }
        for name, field in fields.items():
            field_error[name] += float(np.sum(np.square(field - free_target)))
        field_truth += float(np.sum(np.square(free_target)))
        for index, sample_id in enumerate(group["sample_ids"]):
            losses = {
                "shape_cv_loss": float(shape_loss[index].reshape(-1)[0]),
                "log_scale_loss": float(scale_loss[index]),
                "relative_field_loss": float(relative_loss[index].reshape(-1)[0]),
                "raw_absolute_field_loss": float(raw_loss[index].reshape(-1)[0]),
            }
            sample_id = str(sample_id)
            rows.append({
                "sample_id": sample_id,
                "split": split,
                "true_cv_rms_deltaT_K": float(true_scale[index].reshape(-1)[0]),
                "total_power_W": float(context_cache[sample_id]["context"]["P_operator_W"]),
                **losses,
                "unit_weight_total_loss": sum(losses.values()),
            })
    metrics = {
        f"point_global_{name}_relative_rmse_pct": 100.0 * math.sqrt(error / field_truth)
        for name, error in field_error.items()
    }
    return rows, metrics


def _quartile_table(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    result = []
    for index in range(4):
        selected = [
            row for row in rows if edges[index] < float(row[field]) <= edges[index + 1]
        ]
        if not selected:
            raise Gate6ADiagnosticError(f"empty {field} quartile")
        result.append({
            "quartile": index + 1,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "sample_count": len(selected),
            "mean_losses": {
                name: float(np.mean([float(row[name]) for row in selected]))
                for name in LOSS_NAMES
            },
            "mean_unit_weight_total_loss": float(np.mean([
                float(row["unit_weight_total_loss"]) for row in selected
            ])),
        })
    return result


def _subset_groups(
    sample_ids: Sequence[str],
    examples_by_id: Mapping[str, Any],
    stats: Mapping[str, Any],
    builder: Any,
    graph_seed: int,
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    groups = _make_groups_with_progress(
        [examples_by_id[sample_id] for sample_id in sample_ids],
        dict(stats),
        builder,
        label,
        False,
        "basic",
        graph_seed,
        batch_size=min(28, len(sample_ids)),
        drop_last=False,
    )
    _attach_v5_physics(groups, cache, standardizer)
    for group in groups:
        group["native_physics"] = group["v5_physics"]
        group["global_context"] = group["v5_physics"]["global_context"]
    return groups


def _subset_contribution(
    *,
    subset_name: str,
    sample_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    split_count: int,
    split_gradients: Mapping[str, Any],
    split_total_gradient: Any,
    model: Any,
    params: Any,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    subset_count = len(sample_ids)
    result: dict[str, Any] = {
        "subset": subset_name,
        "sample_count": subset_count,
        "sample_ids": list(sample_ids),
        "loss_contribution_fraction": {},
        "gradient_contribution": {},
    }
    subset_gradients: dict[str, Any] = {}
    for loss_name in LOSS_NAMES:
        full_loss_sum = sum(float(row[loss_name]) for row in rows_by_id.values())
        subset_loss_sum = sum(float(rows_by_id[sample_id][loss_name]) for sample_id in sample_ids)
        _, subset_gradient, _ = _aggregate_gradient(model, params, groups, loss_name)
        subset_gradients[loss_name] = subset_gradient
        scaled_gradient = _tree_scale(subset_gradient, subset_count / split_count)
        result["loss_contribution_fraction"][loss_name] = subset_loss_sum / max(full_loss_sum, 1.0e-30)
        result["gradient_contribution"][loss_name] = {
            "scaled_norm_ratio_to_full": (
                _gradient_summary(scaled_gradient)["global_norm"]
                / max(_gradient_summary(split_gradients[loss_name])["global_norm"], 1.0e-30)
            ),
            "cosine_with_full": _cosine(scaled_gradient, split_gradients[loss_name])["global"],
        }
    subset_total = _tree_linear_combination(
        [
            _tree_scale(
                subset_gradients[loss_name],
                subset_count / split_count,
            )
            for loss_name in LOSS_NAMES
        ],
        [1.0] * len(LOSS_NAMES),
    )
    result["gradient_contribution"]["unit_weight_total_loss"] = {
        "scaled_norm_ratio_to_full": (
            _gradient_summary(subset_total)["global_norm"]
            / max(_gradient_summary(split_total_gradient)["global_norm"], 1.0e-30)
        ),
        "cosine_with_full": _cosine(subset_total, split_total_gradient)["global"],
    }
    return result


def _diagnose_split(
    *,
    split: str,
    model: Any,
    params: Any,
    groups: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    examples_by_id: Mapping[str, Any],
    stats: Mapping[str, Any],
    builder: Any,
    graph_seed: int,
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
) -> dict[str, Any]:
    gradients: dict[str, Any] = {}
    loss_values: dict[str, float] = {}
    count = len(rows)
    for loss_name in LOSS_NAMES:
        loss_values[loss_name], gradients[loss_name], gradient_count = _aggregate_gradient(
            model, params, groups, loss_name
        )
        if gradient_count != count:
            raise Gate6ADiagnosticError(f"{split}: gradient count mismatch")
    total_gradient = _tree_linear_combination(
        [gradients[name] for name in LOSS_NAMES], [1.0] * len(LOSS_NAMES)
    )
    cosine = {
        left: {
            right: _cosine(gradients[left], gradients[right])
            for right in LOSS_NAMES
        }
        for left in LOSS_NAMES
    }
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    ordered = sorted(rows, key=lambda row: float(row["unit_weight_total_loss"]), reverse=True)
    true_scale_q4_floor = float(np.quantile(
        [float(row["true_cv_rms_deltaT_K"]) for row in rows], 0.75
    ))
    power_q4_floor = float(np.quantile([float(row["total_power_W"]) for row in rows], 0.75))
    subsets = {
        "true_scale_Q4": [
            str(row["sample_id"]) for row in rows
            if float(row["true_cv_rms_deltaT_K"]) >= true_scale_q4_floor
        ],
        "total_power_Q4": [
            str(row["sample_id"]) for row in rows
            if float(row["total_power_W"]) >= power_q4_floor
        ],
        "top5_unit_weight_total_loss": [str(row["sample_id"]) for row in ordered[:5]],
        "top10_unit_weight_total_loss": [str(row["sample_id"]) for row in ordered[:10]],
    }
    subset_reports = []
    for subset_name, sample_ids in subsets.items():
        subset_groups = _subset_groups(
            sample_ids, examples_by_id, stats, builder, graph_seed, cache,
            standardizer, f"gate6a_{split}_{subset_name}",
        )
        subset_reports.append(_subset_contribution(
            subset_name=subset_name,
            sample_ids=sample_ids,
            rows_by_id=rows_by_id,
            split_count=count,
            split_gradients=gradients,
            split_total_gradient=total_gradient,
            model=model,
            params=params,
            groups=subset_groups,
        ))
    top10_sample_gradient_rows = []
    for row in ordered[:10]:
        sample_id = str(row["sample_id"])
        sample_groups = _subset_groups(
            [sample_id], examples_by_id, stats, builder, graph_seed, cache,
            standardizer, f"gate6a_{split}_{sample_id}",
        )
        sample_gradients = [
            _aggregate_gradient(model, params, sample_groups, loss_name)[1]
            for loss_name in LOSS_NAMES
        ]
        scaled = _tree_scale(
            _tree_linear_combination(sample_gradients, [1.0] * len(LOSS_NAMES)),
            1.0 / count,
        )
        top10_sample_gradient_rows.append({
            **dict(row),
            "unit_weight_total_loss_contribution_fraction": (
                float(row["unit_weight_total_loss"])
                / sum(float(item["unit_weight_total_loss"]) for item in rows)
            ),
            "scaled_gradient_norm_ratio_to_full": (
                _gradient_summary(scaled)["global_norm"]
                / max(_gradient_summary(total_gradient)["global_norm"], 1.0e-30)
            ),
            "gradient_cosine_with_full": _cosine(scaled, total_gradient)["global"],
        })
    global_norms = {
        name: _gradient_summary(gradient)["global_norm"] for name, gradient in gradients.items()
    }
    nonzero_norms = [value for value in global_norms.values() if value > 0.0]
    loss_nonzero = [value for value in loss_values.values() if value > 0.0]
    return {
        "loss_means": loss_values,
        "gradient_summaries": {
            name: _gradient_summary(gradient) for name, gradient in gradients.items()
        },
        "unit_weight_total_gradient": _gradient_summary(total_gradient),
        "loss_gradient_cosine_similarity": cosine,
        "true_cv_rms_deltaT_quartiles": _quartile_table(rows, "true_cv_rms_deltaT_K"),
        "total_power_quartiles": _quartile_table(rows, "total_power_W"),
        "subset_contributions": subset_reports,
        "top10_sample_contributions": top10_sample_gradient_rows,
        "imbalance_signals": {
            "max_to_min_loss_mean": max(loss_nonzero) / min(loss_nonzero),
            "max_to_min_loss_gradient_global_norm": max(nonzero_norms) / min(nonzero_norms),
            "unit_weights_imbalanced": (
                max(loss_nonzero) / min(loss_nonzero) > 3.0
                or max(nonzero_norms) / min(nonzero_norms) > 3.0
            ),
        },
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V5 Gate 6A no-training diagnosis",
        "",
        "仅访问 `train` 与 `valid_iid`；未加载或评估 test/hard。checkpoint 为 N3 best e402。",
        "",
        "## Point-global oracle views",
        "",
        "| split | joint % | oracle-scale % | oracle-shape % | physics-scale % |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = payload["splits"][split]["point_global_metrics"]
        lines.append(
            f"| {split} | {row['point_global_joint_relative_rmse_pct']:.4f} | "
            f"{row['point_global_oracle_scale_relative_rmse_pct']:.4f} | "
            f"{row['point_global_oracle_shape_relative_rmse_pct']:.4f} | "
            f"{row['point_global_physics_scale_relative_rmse_pct']:.4f} |"
        )
    lines += [
        "",
        "## Loss and gradient scale",
        "",
        "| split | loss | mean | global grad norm | backbone | shape decoder | scale head | FiLM | bypass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        diagnostic = payload["splits"][split]["diagnostic"]
        for loss_name in LOSS_NAMES:
            gradient = diagnostic["gradient_summaries"][loss_name]
            groups = gradient["parameter_group_norms"]
            lines.append(
                f"| {split} | {loss_name} | {diagnostic['loss_means'][loss_name]:.6g} | "
                f"{gradient['global_norm']:.6g} | "
                + " | ".join(f"{groups[name]:.6g}" for name in PARAMETER_GROUPS)
                + " |"
            )
    lines += [
        "",
        "完整四分位、gradient cosine、Q4、top-5/top-10 与逐 top-10 样本贡献见 JSON。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    input_dir = args.input_dir.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    if args.batch_size != 28:
        raise Gate6ADiagnosticError("Gate 6A diagnostic batch size is frozen at 28")
    if any(path.exists() for path in (output_json, output_md)) and not args.overwrite:
        raise Gate6ADiagnosticError("output exists; pass --overwrite")
    if input_dir in output_json.parents or input_dir in output_md.parents:
        raise Gate6ADiagnosticError("diagnostic output must not modify the frozen input package")
    checksums = _verify_input_dir(input_dir)
    run_config = _read_json(input_dir / "run_config.json")
    loss_summary = _read_json(input_dir / "loss_summary.json")
    metadata = _read_json(input_dir / "normalization_context_metadata.json")
    contract = _read_json(args.contract.resolve())
    checkpoint = _load_params_checkpoint(input_dir / "params_best.pkl")
    if checkpoint.get("checkpoint_kind") != "best" or int(checkpoint.get("epoch", -1)) != 402:
        raise Gate6ADiagnosticError("Gate 6A requires N3 best e402")
    if loss_summary.get("code_version_or_git_commit") != "f1053d1":
        raise Gate6ADiagnosticError("unexpected N3 training commit")
    checkpoint_stats = dict(checkpoint.get("train_only_normalization") or {})
    if checkpoint_stats != metadata.get("train_only_normalization"):
        raise Gate6ADiagnosticError("normalization metadata differs from checkpoint")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_ids, split_source, _, _ = _resolve_training_splits(
        sample_root, Path(str(run_config["split_map_path"]))
    )
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6ADiagnosticError("Gate 6A requires train=672 and valid_iid=128")
    if _ids_hash(train_ids) != contract["split_hashes"]["train"] or _ids_hash(valid_ids) != contract["split_hashes"]["valid_iid"]:
        raise Gate6ADiagnosticError("Gate 6A train/valid split hash mismatch")
    if [str(example.sample_id) for example in train_examples] != train_ids:
        raise Gate6ADiagnosticError("train normalization population differs from frozen train split")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    examples = {str(example.sample_id): example for example in [*train_examples, *valid_examples]}
    if set(examples) != set(train_ids + valid_ids):
        raise Gate6ADiagnosticError("diagnostic loaded samples outside train/valid or missed samples")
    if any(np.asarray(example.condition.coords).shape != (1024, 3) for example in examples.values()):
        raise Gate6ADiagnosticError("Gate 6A requires exactly 1024 nodes")
    cache = _physics_cache(list(examples.values()))
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids], fit_sample_ids=train_ids
    )
    stored_standardizer = (loss_summary.get("global_context") or {}).get("standardizer") or {}
    for field in ("mean", "std"):
        if not np.allclose(standardizer[field], stored_standardizer[field], rtol=1.0e-9, atol=1.0e-10):
            raise Gate6ADiagnosticError(f"train-only context {field} differs from N3 training")
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    graph_seed = int(run_config.get("graph_seed", 0))
    groups: dict[str, list[dict[str, Any]]] = {}
    for split, ids in (("train", train_ids), ("valid_iid", valid_ids)):
        groups[split] = _subset_groups(
            ids, examples, stats, builder, graph_seed, cache, standardizer, f"gate6a_{split}"
        )
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint.get("model_config") or {}), dict(stats)
    )
    if model_config.get("native_output_mode") != "native_shape_scale" or model_config.get("global_context_mode") != "film":
        raise Gate6ADiagnosticError("input checkpoint is not N3 native pooled-latent + Global FiLM")
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    payload: dict[str, Any] = {
        "schema_version": "heat3d_v5_gate6a_no_training_diagnostic_v1",
        "status": "complete",
        "training_started": False,
        "config_id": "V4P5_07_native_pooled_latent_global_film",
        "checkpoint": {
            "epoch": 402,
            "kind": "best",
            "sha256": checksums["params_best.pkl"],
            "training_git_commit": loss_summary.get("code_version_or_git_commit"),
        },
        "diagnostic_git_commit": _git_commit(),
        "input_dir": str(input_dir),
        "input_checksums": checksums,
        "data_access_contract": {
            "loaded_splits": list(SPLITS),
            "forbidden_splits_loaded": [],
            "selection_split": "valid_iid",
            "split_source": split_source,
            "split_counts": {"train": len(train_ids), "valid_iid": len(valid_ids)},
            "split_hashes": {"train": _ids_hash(train_ids), "valid_iid": _ids_hash(valid_ids)},
            "nodes_per_sample": 1024,
            "normalization_fit": "train_only",
            "context_fit": "train_only",
        },
        "unit_loss_weights": {
            "shape_cv": 1.0,
            "log_scale": 1.0,
            "relative_field": 1.0,
            "raw_absolute": 1.0,
        },
        "splits": {},
    }
    for split in SPLITS:
        rows, point_metrics = _per_sample_rows(
            model, params, groups[split], split, cache
        )
        payload["splits"][split] = {
            "point_global_metrics": point_metrics,
            "per_sample": rows,
            "diagnostic": _diagnose_split(
                split=split,
                model=model,
                params=params,
                groups=groups[split],
                rows=rows,
                examples_by_id=examples,
                stats=stats,
                builder=builder,
                graph_seed=graph_seed,
                cache=cache,
                standardizer=standardizer,
            ),
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "training_started": False,
        "output_json": str(output_json),
        "valid_joint_pct": payload["splits"]["valid_iid"]["point_global_metrics"]["point_global_joint_relative_rmse_pct"],
        "valid_imbalanced": payload["splits"]["valid_iid"]["diagnostic"]["imbalance_signals"]["unit_weights_imbalanced"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
