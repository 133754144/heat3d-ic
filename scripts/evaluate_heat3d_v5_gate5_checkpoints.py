#!/usr/bin/env python3
"""Unified read-only Gate-5 evaluation for completed B0/N0/N1 checkpoints."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rigno.heat3d_v1_normalization import normalized_delta_to_raw  # noqa: E402
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
}
SCHEMA_VERSION = "heat3d_v5_gate5_unified_evaluation_v1"


class EvaluationError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True, choices=sorted(ALLOWED_CONFIGS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


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
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    native_rows: list[dict[str, float]] = []
    for group in groups:
        prediction = _model_apply(model, params, group)
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
        for index, sample_id in enumerate(group["sample_ids"]):
            samples.append(
                {
                    "sample_id": str(sample_id),
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
                native_rows.append(
                    _native_sample_diagnostics(
                        native,
                        index,
                        target_raw[index],
                        volumes[index].reshape(-1),
                        masks[index],
                        float(log_s_phys[index]),
                    )
                )
    suite = evaluate_metric_suite(samples)
    result = dict(suite["summary"])
    if native_rows:
        result["native_shape_scale"] = {
            key: _mean([row[key] for row in native_rows])
            for key in native_rows[0]
        }
    return result


def _checkpoint_report(
    path: Path,
    *,
    role_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
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
            model, params, groups, role=role, stats=stats, native_mode=native_mode
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
    output_json = (args.output_json or run_dir / "v5_metrics.json").resolve()
    if output_json.exists() and not args.overwrite:
        raise EvaluationError(f"output exists; pass --overwrite: {output_json}")
    run_config = _read_json(run_dir / "run_config.json")
    loss_summary = _read_json(run_dir / "loss_summary.json")
    best_path = run_dir / "params_best.pkl"
    final_path = run_dir / "params_final.pkl"
    for path in (best_path, final_path):
        if not path.is_file():
            raise EvaluationError(f"missing checkpoint: {path}")

    best_payload = _load_params_checkpoint(best_path)
    checkpoint_stats = dict(best_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise EvaluationError("best checkpoint lacks train-only normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_map = Path(str(run_config["split_map_path"]))
    split_ids, split_source, _, _ = _resolve_training_splits(sample_root, split_map)
    missing_roles = [role for role in ROLES if not split_ids.get(role)]
    if missing_roles:
        raise EvaluationError(f"missing evaluation roles: {missing_roles}")
    if len(split_ids.get("train", ())) != 672 or len(split_ids["valid_iid"]) != 128:
        raise EvaluationError("Gate-5 requires train=672 and valid_iid=128")

    evaluation_ids = [sample_id for role in ROLES for sample_id in split_ids[role]]
    evaluation_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=evaluation_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    all_examples = list(train_examples) + list(evaluation_examples)
    cache = _physics_cache(all_examples)
    train_ids = list(split_ids["train"])
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    if int(standardizer["fit_sample_count"]) != 672:
        raise EvaluationError("global-context standardizer was not train-only")

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
        best_path, role_groups=role_groups, stats=stats
    )
    final_reports, final_epoch, final_native = _checkpoint_report(
        final_path, role_groups=role_groups, stats=stats
    )
    if best_native != final_native:
        raise EvaluationError("best/final native-output mode differs")
    commit = _git_commit()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "config_id": args.config_id,
        "evaluator_git_commit": commit,
        "training_git_commit": loss_summary.get("code_version_or_git_commit"),
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
            "split_hashes": {role: _ids_hash(split_ids[role]) for role in ("train",) + ROLES},
            "nodes_per_sample": 1024,
            "prediction_batch_size": args.prediction_batch_size,
        },
        "global_context_standardizer": {
            "schema": list(GLOBAL_CONTEXT_FEATURES),
            "fit_roles": ["train"],
            "fit_sample_count": int(standardizer["fit_sample_count"]),
            "fit_sample_ids_sha256": _ids_hash(train_ids),
            "target_or_label_features": [],
        },
        "checkpoint_metadata": {
            "best": {"epoch": best_epoch, "path": str(best_path), "sha256": _sha256(best_path)},
            "final": {"epoch": final_epoch, "path": str(final_path), "sha256": _sha256(final_path)},
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
