#!/usr/bin/env python3
"""Frozen read-only Gate-5 evaluation for completed B0/N0/N1/N3 checkpoints."""

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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rigno.heat3d_v1_normalization import normalized_delta_to_raw  # noqa: E402
from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    fit_train_only_standardizer,
)
from rigno.heat3d_v5_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    evaluate_metric_suite,
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
import run_heat3d_v1_medium_controlled_training_export as runner_module  # noqa: E402
from run_heat3d_v5_clean_first import (  # noqa: E402
    _attach_v5_physics,
    _load_examples,
    _physics_cache,
)


ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
ALLOWED_CONFIGS = {
    "V4P5_04_local_bypass_global_film",
    "V4P5_05_native_physics_only",
    "V4P5_06_native_pooled_latent",
    "V4P5_07_native_pooled_latent_global_film",
}
SCHEMA_VERSION = "heat3d_v5_gate5_unified_evaluation_v2_frozen"
DEFAULT_CONTRACT = ROOT / "configs/heat3d_v5/v5_gate5_final_evaluator_contract.json"
DEFAULT_REGISTRY = ROOT / "configs/heat3d_v5/v5_scratch_bypass_film_registry.csv"
ATTRIBUTION_CONTEXT_FIELDS = (
    "P_operator_W",
    "source_concentration",
    "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W",
    "log_top_h_W_m2K",
    "anisotropy_xy_over_z",
)


class EvaluationError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True, choices=sorted(ALLOWED_CONFIGS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--prediction-batch-size", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationError(f"{path}: expected JSON object")
    return payload


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _path_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.resolve())],
        cwd=ROOT,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _registry_row(path: Path, config_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("config_id") == config_id]
    if len(rows) != 1:
        raise EvaluationError(f"registry must contain exactly one {config_id} row")
    return rows[0]


def _validate_contract_binding(
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    registry_row: Mapping[str, str],
    run_dir: Path,
    run_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    configs = contract.get("configs") or {}
    expected = configs.get(args.config_id)
    if not isinstance(expected, Mapping):
        raise EvaluationError(f"contract lacks config {args.config_id}")
    if set(configs) != ALLOWED_CONFIGS:
        raise EvaluationError("contract config set differs from frozen Gate-5 set")
    if registry_row.get("output_dir") != expected.get("run_dir"):
        raise EvaluationError("registry run directory differs from evaluator contract")
    try:
        relative_run_dir = str(run_dir.relative_to(ROOT))
    except ValueError as exc:
        raise EvaluationError("run directory must be inside the repository") from exc
    if relative_run_dir != expected.get("run_dir"):
        raise EvaluationError(
            f"run directory mismatch: {relative_run_dir} != {expected.get('run_dir')}"
        )
    if str(run_config.get("output_dir")) not in {relative_run_dir, str(run_dir)}:
        raise EvaluationError("run_config output_dir does not identify the requested run")
    generated_yaml = ROOT / str(registry_row.get("generated_yaml"))
    if not generated_yaml.is_file() or f"config_id: {args.config_id}" not in generated_yaml.read_text(encoding="utf-8"):
        raise EvaluationError("generated YAML config_id does not match evaluator request")
    return expected


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


def _validate_checkpoint_binding(
    *,
    checkpoint: Mapping[str, Any],
    path: Path,
    kind: str,
    expected_epoch: int,
    expected_run_dir: str,
    expected_model: Mapping[str, Any],
) -> None:
    if checkpoint.get("checkpoint_kind") != kind:
        raise EvaluationError(f"{path}: checkpoint_kind must be {kind}")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise EvaluationError(f"{path}: epoch differs from frozen loss summary")
    metadata = checkpoint.get("run_config_metadata") or {}
    if metadata.get("output_dir") != expected_run_dir:
        raise EvaluationError(f"{path}: checkpoint output_dir binding mismatch")
    model_config = checkpoint.get("model_config") or {}
    for field, value in expected_model.items():
        defaults = {"native_output_mode": "legacy_normalized_deltaT"}
        actual = model_config.get(field, defaults.get(field))
        if actual != value:
            raise EvaluationError(
                f"{path}: model_config.{field}={actual!r} != {value!r}"
            )


def _film_rows(params: Mapping[str, Any], context: Any) -> list[dict[str, float]]:
    if "global_film_hidden" not in params or "global_film_output" not in params:
        return []
    hidden_params = params["global_film_hidden"]
    output_params = params["global_film_output"]
    values = jnp.asarray(context)
    hidden = jax.nn.gelu(
        values @ jnp.asarray(hidden_params["kernel"]) + jnp.asarray(hidden_params["bias"])
    )
    gamma_beta = (
        hidden @ jnp.asarray(output_params["kernel"]) + jnp.asarray(output_params["bias"])
    )
    gamma, beta = np.split(np.asarray(gamma_beta, dtype=np.float64), 2, axis=-1)
    rows = []
    for gamma_row, beta_row in zip(gamma, beta, strict=True):
        rows.append({
            "film_gamma_mean_abs": float(np.mean(np.abs(gamma_row))),
            "film_gamma_rms": float(np.sqrt(np.mean(np.square(gamma_row)))),
            "film_gamma_max_abs": float(np.max(np.abs(gamma_row))),
            "film_beta_mean_abs": float(np.mean(np.abs(beta_row))),
            "film_beta_rms": float(np.sqrt(np.mean(np.square(beta_row)))),
            "film_beta_max_abs": float(np.max(np.abs(beta_row))),
        })
    return rows


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise EvaluationError("native diagnostics are empty or non-finite")
    return float(np.mean(array))


def _cv_rms(field: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(field) * weights) / np.sum(weights)))


def _native_sample_diagnostics(
    native: Mapping[str, Any],
    index: int,
    target: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    log_s_phys: float,
) -> dict[str, float]:
    free = 1.0 - np.clip(mask.reshape(-1), 0.0, 1.0)
    target_free = target.reshape(-1) * free
    true_scale = _cv_rms(target_free, weights)
    phi_true = target_free / max(true_scale, 1.0e-12)
    phi_hat = np.asarray(native["phi_hat"], dtype=np.float64)[index].reshape(-1)
    s_hat = float(np.asarray(native["s_hat"], dtype=np.float64)[index].reshape(-1)[0])
    fields = {
        "joint": s_hat * phi_hat,
        "oracle_scale": true_scale * phi_hat,
        "oracle_shape": s_hat * phi_true,
        "physics_scale": math.exp(float(log_s_phys)) * phi_hat,
    }
    return {
        f"{name}_relative_rmse_pct": 100.0 * _cv_rms(field - target_free, weights) / true_scale
        for name, field in fields.items()
    }


def _evaluate_role(
    model: Any,
    params: Any,
    groups: Sequence[Mapping[str, Any]],
    *,
    role: str,
    stats: Mapping[str, Any],
    native_mode: bool,
    context_cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    native_by_id: dict[str, dict[str, float]] = {}
    film_by_id: dict[str, dict[str, float]] = {}
    for group in groups:
        prediction = (
            _model_apply(model, params, group)
            if native_mode
            else model.apply(
                {"params": params},
                inputs=group["inputs"],
                graphs=group["graphs"],
                global_context=group.get("global_context"),
            )
        )
        native = prediction if native_mode else None
        if native_mode:
            raw = np.asarray(prediction["deltaT_hat"], dtype=np.float64)
            normalized = (
                jnp.asarray(prediction["deltaT_hat"])
                - jnp.asarray(stats["target_delta_mean"])
            ) / jnp.asarray(stats["target_delta_std"])
        else:
            normalized = prediction
            raw = np.asarray(normalized_delta_to_raw(normalized, stats), dtype=np.float64)
        normalized_np = np.asarray(normalized, dtype=np.float64)
        target_raw = np.asarray(group["target_delta_raw"], dtype=np.float64)
        target_normalized = np.asarray(group["target_normalized"], dtype=np.float64)
        physics = group["native_physics"]
        volumes = np.asarray(physics["control_volumes"], dtype=np.float64)
        q_values = np.asarray(physics["q"], dtype=np.float64)
        masks = np.asarray(physics["dirichlet_mask"], dtype=np.float64)
        log_s_phys = np.asarray(physics["log_s_phys"], dtype=np.float64)
        film_rows = _film_rows(params, group["global_context"])
        for index, sample_id in enumerate(group["sample_ids"]):
            sample_id = str(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": role,
                    "prediction_deltaT_K": raw[index].reshape(-1),
                    "target_deltaT_K": target_raw[index].reshape(-1),
                    "control_volumes_m3": volumes[index].reshape(-1),
                    "q_W_m3": q_values[index].reshape(-1),
                    "prediction_normalized": normalized_np[index],
                    "target_normalized": target_normalized[index],
                }
            )
            if native is not None:
                native_by_id[sample_id] = _native_sample_diagnostics(
                    native,
                    index,
                    target_raw[index],
                    volumes[index].reshape(-1),
                    masks[index],
                    float(log_s_phys[index]),
                )
            if film_rows:
                film_by_id[sample_id] = film_rows[index]
    suite = evaluate_metric_suite(samples)
    result = dict(suite["summary"])
    enriched_rows = []
    for row in suite["per_sample"]:
        sample_id = str(row["sample_id"])
        context = context_cache[sample_id]["context"]
        enriched = dict(row)
        enriched["attribution_context"] = {
            field: float(context[field]) for field in ATTRIBUTION_CONTEXT_FIELDS
        }
        if sample_id in native_by_id:
            enriched["native_shape_scale"] = native_by_id[sample_id]
        if sample_id in film_by_id:
            enriched["film_modulation"] = film_by_id[sample_id]
        enriched_rows.append(enriched)
    result["per_sample"] = enriched_rows
    if native_by_id:
        result["native_shape_scale"] = {
            key: _mean([row[key] for row in native_by_id.values()])
            for key in next(iter(native_by_id.values()))
        }
    if film_by_id:
        result["film_modulation"] = {
            "sample_count": len(film_by_id),
            **{
                key: _mean([row[key] for row in film_by_id.values()])
                for key in next(iter(film_by_id.values()))
            },
            "film_gamma_global_max_abs": max(
                row["film_gamma_max_abs"] for row in film_by_id.values()
            ),
            "film_beta_global_max_abs": max(
                row["film_beta_max_abs"] for row in film_by_id.values()
            ),
        }
    return result


def _checkpoint_report(
    path: Path,
    *,
    role_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
    context_cache: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], int, bool]:
    checkpoint = _load_params_checkpoint(path)
    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint.get("model_config") or {}), dict(stats)
    )
    native_mode = str(model_config.get("native_output_mode")) == "native_shape_scale"
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint["params"])
    reports = {
        role: _evaluate_role(
            model,
            params,
            groups,
            role=role,
            stats=stats,
            native_mode=native_mode,
            context_cache=context_cache,
        )
        for role, groups in role_groups.items()
    }
    finite = all(
        math.isfinite(float(report["point_global_relative_rmse_pct"]))
        for report in reports.values()
    )
    if not finite:
        raise EvaluationError(f"{path}: non-finite checkpoint report")
    return reports, int(checkpoint.get("epoch", -1)), native_mode


def main() -> int:
    args = _args()
    if args.prediction_batch_size < 1:
        raise EvaluationError("--prediction-batch-size must be positive")
    run_dir = args.run_dir.resolve()
    output_json = args.output_json.resolve()
    if run_dir == output_json.parent or run_dir in output_json.parents:
        raise EvaluationError("final evaluation output must not modify the training run directory")
    if output_json.exists() and not args.overwrite:
        raise EvaluationError(f"output exists; pass --overwrite: {output_json}")
    run_config = _read_json(run_dir / "run_config.json")
    loss_summary = _read_json(run_dir / "loss_summary.json")
    contract = _read_json(args.contract.resolve())
    registry_row = _registry_row(args.registry.resolve(), args.config_id)
    expected = _validate_contract_binding(
        args=args,
        contract=contract,
        registry_row=registry_row,
        run_dir=run_dir,
        run_config=run_config,
    )
    best_path = run_dir / "params_best.pkl"
    final_path = run_dir / "params_final.pkl"
    for path in (best_path, final_path):
        if not path.is_file():
            raise EvaluationError(f"missing checkpoint: {path}")

    best_payload = _load_params_checkpoint(best_path)
    final_payload = _load_params_checkpoint(final_path)
    best_epoch_expected = int(loss_summary.get("best_epoch", -1))
    final_epoch_expected = int(loss_summary.get("final_epoch", -1))
    if final_epoch_expected != 600:
        raise EvaluationError("Gate-5 final checkpoint must be epoch 600")
    for payload, path, kind, epoch in (
        (best_payload, best_path, "best", best_epoch_expected),
        (final_payload, final_path, "final", final_epoch_expected),
    ):
        _validate_checkpoint_binding(
            checkpoint=payload,
            path=path,
            kind=kind,
            expected_epoch=epoch,
            expected_run_dir=str(expected["run_dir"]),
            expected_model=expected["model"],
        )
    checkpoint_stats = dict(best_payload.get("train_only_normalization") or {})
    final_checkpoint_stats = dict(final_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise EvaluationError("best checkpoint lacks train-only normalization")
    if not _normalization_equal(checkpoint_stats, final_checkpoint_stats):
        raise EvaluationError("best/final train-only normalization differs")
    if best_payload.get("train_stats_hash") != final_payload.get("train_stats_hash"):
        raise EvaluationError("best/final train_stats_hash differs")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    train_ids_from_examples = [str(example.sample_id) for example in train_examples]
    recomputed_train_stats = training_normalization_stats(
        train_examples,
        normalization_profile=str(checkpoint_stats.get("normalization_profile", "legacy_zscore")),
        condition_feature_transform=checkpoint_stats.get("condition_feature_transform"),
        input_feature_schema=str(checkpoint_stats.get("input_feature_schema", "legacy_bc_flags")),
        coord_policy=str(checkpoint_stats.get("coord_policy", "train_minmax_to_unit_box")),
        extent_feature_policy=str(checkpoint_stats.get("extent_feature_policy", "none")),
        bridge_fn=runner_module._bridge_for,
    )
    if not _normalization_equal(checkpoint_stats, recomputed_train_stats):
        raise EvaluationError("checkpoint normalization does not reproduce from train-only examples")
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_map = Path(str(run_config["split_map_path"]))
    split_ids, split_source, _, _ = _resolve_training_splits(sample_root, split_map)
    missing_roles = [role for role in ROLES if not split_ids.get(role)]
    if missing_roles:
        raise EvaluationError(f"missing evaluation roles: {missing_roles}")
    if len(split_ids.get("train", ())) != 672 or len(split_ids["valid_iid"]) != 128:
        raise EvaluationError("Gate-5 requires train=672 and valid_iid=128")
    split_hashes = {
        role: _ids_hash(split_ids[role]) for role in ("train",) + ROLES
    }
    if split_hashes != contract.get("split_hashes"):
        raise EvaluationError("resolved split hashes differ from frozen evaluator contract")
    if train_ids_from_examples != list(split_ids["train"]):
        raise EvaluationError("normalization examples do not exactly match frozen train split")

    evaluation_ids = [sample_id for role in ROLES for sample_id in split_ids[role]]
    evaluation_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=evaluation_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    all_examples = list(train_examples) + list(evaluation_examples)
    if any(np.asarray(example.condition.coords).shape != (1024, 3) for example in all_examples):
        raise EvaluationError("Gate-5 evaluator requires exactly 1024 nodes per sample")
    cache = _physics_cache(all_examples)
    train_ids = list(split_ids["train"])
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    if int(standardizer["fit_sample_count"]) != 672:
        raise EvaluationError("global-context standardizer was not train-only")
    stored_context = loss_summary.get("global_context") or {}
    stored_standardizer = stored_context.get("standardizer") or {}
    if (
        stored_standardizer.get("fit_population") != "train_only"
        or int(stored_standardizer.get("fit_sample_count", -1)) != 672
        or stored_standardizer.get("fit_sample_ids_sha256")
        != contract.get("train_context_fit_sample_ids_sha256")
        or standardizer.get("fit_sample_ids_sha256")
        != contract.get("train_context_fit_sample_ids_sha256")
    ):
        raise EvaluationError("training summary global-context fit is not the frozen train split")
    for field in ("mean", "std"):
        if not np.allclose(
            np.asarray(standardizer[field], dtype=np.float64),
            np.asarray(stored_standardizer[field], dtype=np.float64),
            rtol=1.0e-9,
            atol=1.0e-10,
        ):
            raise EvaluationError(f"recomputed global-context {field} differs from training summary")

    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    examples_by_id = {example.sample_id: example for example in evaluation_examples}
    role_groups: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        examples = [examples_by_id[sample_id] for sample_id in split_ids[role]]
        groups = _make_groups_with_progress(
            examples,
            stats,
            builder,
            f"gate5_{role}",
            False,
            "basic",
            int(run_config.get("graph_seed", 0)),
            batch_size=args.prediction_batch_size,
            drop_last=False,
        )
        _attach_v5_physics(groups, cache, standardizer)
        for group in groups:
            group["native_physics"] = group["v5_physics"]
            group["global_context"] = group["v5_physics"]["global_context"]
        role_groups[role] = groups

    best_reports, best_epoch, best_native = _checkpoint_report(
        best_path, role_groups=role_groups, stats=stats, context_cache=cache
    )
    final_reports, final_epoch, final_native = _checkpoint_report(
        final_path, role_groups=role_groups, stats=stats, context_cache=cache
    )
    if best_native != final_native:
        raise EvaluationError("best/final native-output mode differs")
    commit = _git_commit()
    registry_commit = _path_commit(args.registry.resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "config_id": args.config_id,
        "evaluator_git_commit": commit,
        "training_git_commit": loss_summary.get("code_version_or_git_commit"),
        "registry_git_commit": registry_commit,
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract.resolve()),
        "validation_audit": {
            "config_id_bound": True,
            "run_directory_bound": True,
            "checkpoint_kind_epoch_and_run_bound": True,
            "checkpoint_best_final_normalization_equal": True,
            "normalization_recomputed_from_train_only": True,
            "global_context_recomputed_from_train_only": True,
            "nodes_per_sample": 1024,
            "split_hashes_match_contract": True,
        },
        "formulas": {
            "point_global_relative_rmse_pct": "100 * sqrt(sum(error_deltaT_K^2) / sum(true_deltaT_K^2))",
            "sample_first_cv_relative_rmse_pct": "100 * mean_samples(CV_RMS(error) / CV_RMS(true))",
            "raw_cv_weighted_rmse_K": "sqrt(sum(error_deltaT_K^2 * CV) / sum(CV))",
            "native_oracle_views": "joint=s_hat*phi_hat; oracle_scale=s_true*phi_hat; oracle_shape=s_hat*phi_true; physics_scale=s_phys*phi_hat",
        },
        "data": {
            "subset": str(run_config["subset"]),
            "split_map": str(run_config["split_map_path"]),
            "split_source": split_source,
            "split_counts": {role: len(split_ids[role]) for role in ("train",) + ROLES},
            "split_hashes": split_hashes,
            "nodes_per_sample": 1024,
            "prediction_batch_size": args.prediction_batch_size,
        },
        "global_context_standardizer": {
            "schema": list(GLOBAL_CONTEXT_FEATURES),
            "fit_roles": ["train"],
            "fit_sample_count": int(standardizer["fit_sample_count"]),
            "fit_sample_ids_sha256": standardizer["fit_sample_ids_sha256"],
            "train_split_ordered_ids_sha256": split_hashes["train"],
            "target_or_label_features": [],
        },
        "checkpoint_metadata": {
            "best": {
                "epoch": best_epoch,
                "path": str(best_path),
                "sha256": _sha256(best_path),
                "training_git_commit": best_payload.get("git_commit"),
                "train_stats_hash": best_payload.get("train_stats_hash"),
            },
            "final": {
                "epoch": final_epoch,
                "path": str(final_path),
                "sha256": _sha256(final_path),
                "training_git_commit": final_payload.get("git_commit"),
                "train_stats_hash": final_payload.get("train_stats_hash"),
            },
        },
        "native_shape_scale": bool(best_native),
        "primary_checkpoint": "mse_best",
        "primary_epoch": best_epoch,
        "legacy_checkpoint": "mse_best",
        "legacy_epoch": best_epoch,
        "reports": {
            "primary_relative": best_reports,
            "legacy_metric": best_reports,
            "best": best_reports,
            "final": final_reports,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "config_id": args.config_id,
        "output_json": str(output_json),
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_valid_true_rms_relative_rmse_pct": best_reports["valid_iid"]["point_global_relative_rmse_pct"],
        "final_valid_true_rms_relative_rmse_pct": final_reports["valid_iid"]["point_global_relative_rmse_pct"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
