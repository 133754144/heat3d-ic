#!/usr/bin/env python3
"""Replay one Gate 6Q/V38 run with the frozen valid-only CPU metric suite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    Heat3DV1NativeSupervisedDataset,
)
from rigno.heat3d_v1_supervised import (  # noqa: E402
    PHYSICS_LABEL_SUPERVISED_STAGES,
    Heat3DV1SupervisedDataset,
)
from rigno.heat3d_v5_metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    control_volume_weights,
    evaluate_metric_suite,
)
from rigno.heat3d_v5_global_context import standardize_contexts  # noqa: E402
from rigno.heat3d_v5_scale_context import (  # noqa: E402
    standardize_scale_contexts,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    _attach_qk_region_features_to_groups,
    _attach_scale_context_to_groups,
    _attach_scale_deepsets_weights_to_groups,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _model_apply,
    _global_context_row_for_example,
    _prepare_global_context_lookup,
    _prepare_scale_context_lookup,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
    _scale_context_row_for_example,
    _tree_max_abs_difference,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    stats_from_checkpoint_payload,
)
from run_heat3d_v5_clean_first import (  # noqa: E402
    _attach_v5_physics,
    _physics_cache,
)


EXPECTED_CONFIGS = {
    "V4P5_38_gate6n_v36_r2r_mask_p005_e600",
    "V4P5_42_gate6q_objective_only_e600",
    "V4P5_43_gate6q_xy_scale_features_e600",
    "V4P5_44_gate6q_xy_deepsets_e600",
}
CHECKPOINTS = {
    "point_global_best": (
        "params_best_valid_point_global.pkl",
        "point_global_best_predictions.npz",
        "point_global_best_epoch",
        "point_global_best",
    ),
    "sample_first_best": (
        "params_best_valid_sample_first.pkl",
        "sample_first_best_predictions.npz",
        "sample_first_best_epoch",
        "sample_first_best",
    ),
    "legacy_best": (
        "params_best_valid_base_mse.pkl",
        "base_mse_best_predictions.npz",
        "base_mse_best_epoch",
        "base_mse_best",
    ),
    "final": ("params_final.pkl", "predictions.npz", "final_epoch", "final"),
}
EPS = 1.0e-12


class Gate6QCloseoutError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True, choices=sorted(EXPECTED_CONFIGS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


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


def _load_examples_for_ids(
    sample_root: Path,
    sample_ids: Sequence[str],
    *,
    role: str,
    boundary_mask_fallback: bool,
) -> list[Any]:
    """Load labels only for explicitly allowed train/valid IDs."""

    legacy = Heat3DV1SupervisedDataset.__new__(Heat3DV1SupervisedDataset)
    legacy.datadir = sample_root
    legacy.sample_dirs = [sample_root / sample_id for sample_id in sample_ids]
    legacy.input_mode = "pure_physics"
    legacy.k_encoding_mode = "diag3"
    legacy.allowed_stages = tuple(PHYSICS_LABEL_SUPERVISED_STAGES)
    legacy.boundary_mask_fallback = bool(boundary_mask_fallback)
    legacy.samples = []
    for sample_dir in legacy.sample_dirs:
        sample = legacy._load_sample(sample_dir)
        if str(sample["meta"].get("split")) != role:
            raise Gate6QCloseoutError(
                f"{sample_dir.name}: expected role={role}, found "
                f"{sample['meta'].get('split')!r}"
            )
        legacy.samples.append(sample)

    native = Heat3DV1NativeSupervisedDataset.__new__(
        Heat3DV1NativeSupervisedDataset
    )
    native._legacy_dataset = legacy
    native.k_encoding_mode = "diag3"
    native.boundary_mask_fallback = bool(boundary_mask_fallback)
    native.samples = [native._to_native(sample) for sample in legacy.samples]
    return native.samples


def _target_cache(
    sample_root: Path, valid_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample_id in valid_ids:
        sample_dir = sample_root / sample_id
        meta = json.loads(
            (sample_dir / "sample_meta.json").read_text(encoding="utf-8")
        )
        if str(meta.get("split")) != "valid_iid":
            raise Gate6QCloseoutError(f"{sample_id}: forbidden role encountered")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        coords = np.load(sample_dir / "coords.npy").astype(np.float64)
        target = (
            np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1)
            - bottom
        )
        weights = control_volume_weights(coords)
        true_scale = math.sqrt(
            float(np.sum(np.square(target) * weights) / np.sum(weights))
        )
        result[sample_id] = {
            "bottom_temperature_K": bottom,
            "target_deltaT_K": target,
            "control_volumes_m3": weights,
            "q_W_m3": np.load(sample_dir / "q_field.npy")
            .astype(np.float64)
            .reshape(-1),
            "true_scale_cv_rms_K": float(true_scale),
        }
    scales = np.asarray(
        [result[sample_id]["true_scale_cv_rms_K"] for sample_id in valid_ids]
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
    return result


def _prediction_fields(path: Path, ids: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ids):
            raise Gate6QCloseoutError(
                f"{path}: prediction keys differ from valid_iid"
            )
        result = {
            sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            for sample_id in ids
        }
    if any(
        values.size != 1024 or not np.all(np.isfinite(values))
        for values in result.values()
    ):
        raise Gate6QCloseoutError(f"{path}: invalid prediction fields")
    return result


def _training_reload_rows(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    audit = summary.get("checkpoint_prediction_reload_audit") or {}
    if audit.get("status") != "passed":
        raise Gate6QCloseoutError("training checkpoint reload audit did not pass")
    rows = {str(row["label"]): row for row in audit.get("entries", ())}
    required = {values[3] for values in CHECKPOINTS.values()}
    if not required <= set(rows) or any(not bool(rows[name]["passed"]) for name in required):
        raise Gate6QCloseoutError("training reload audit is incomplete")
    return rows


def _checkpoint(
    run_dir: Path,
    checkpoint_name: str,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    filename, prediction_name, epoch_key, training_label = CHECKPOINTS[checkpoint_name]
    path = run_dir / filename
    payload = _load_params_checkpoint(path)
    second = _load_params_checkpoint(path)
    parameter_reload = _tree_max_abs_difference(payload["params"], second["params"])
    expected_epoch = int(summary[epoch_key])
    if int(payload["epoch"]) != expected_epoch or parameter_reload != 0.0:
        raise Gate6QCloseoutError(f"{path}: checkpoint binding/reload failed")
    training_reload = _training_reload_rows(summary)[training_label]
    leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(payload["params"])]
    metadata = {
        "checkpoint": checkpoint_name,
        "checkpoint_kind": str(payload.get("checkpoint_kind") or ""),
        "epoch": expected_epoch,
        "path": str(path),
        "prediction_path": str(run_dir / prediction_name),
        "sha256": _sha256(path),
        "prediction_sha256": _sha256(run_dir / prediction_name),
        "bytes": path.stat().st_size,
        "training_commit": str(payload.get("git_commit") or ""),
        "train_stats_hash": str(payload.get("train_stats_hash") or ""),
        "parameter_count": int(sum(value.size for value in leaves)),
        "parameter_leaf_count": len(leaves),
        "parameter_reload_max_abs_error": float(parameter_reload),
        "training_reload_max_abs_error_K": float(
            training_reload["checkpoint_reload_max_abs_error_K"]
        ),
        "training_reload_tolerance_K": float(training_reload["tolerance_K"]),
    }
    return payload, metadata


def _build_groups(
    *,
    run_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    stats: Mapping[str, Any],
    train_examples: list[Any],
    valid_examples: list[Any],
    physics_cache: Mapping[str, Mapping[str, Any]],
    prediction_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    groups = _make_groups_with_progress(
        valid_examples,
        dict(stats),
        builder,
        "gate6q_closeout_valid_iid",
        False,
        "basic",
        int(run_config.get("graph_seed", 0)),
        batch_size=prediction_batch_size,
        drop_last=False,
    )
    global_lookup, global_payload = _prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    stored_global = dict(run_config["global_context"]["standardizer"])
    stored_global_lookup = {
        str(example.sample_id): standardize_contexts(
            [_global_context_row_for_example(example)], stored_global
        )[0]
        for example in valid_examples
    }
    global_recompute_max_abs = max(
        float(
            np.max(
                np.abs(
                    np.asarray(global_lookup[sample_id], dtype=np.float64)
                    - np.asarray(stored_global_lookup[sample_id], dtype=np.float64)
                )
            )
        )
        for sample_id in stored_global_lookup
    )
    # Replay the exact V5 physics attachment used by the established valid-only
    # evaluators.  Besides the broadcast global context this preserves the
    # native control-volume/q/Dirichlet payload consumed by shape-scale models.
    _attach_v5_physics(groups, physics_cache, stored_global)
    for group in groups:
        group["native_physics"] = group["v5_physics"]
        group["global_context"] = group["v5_physics"]["global_context"]
    scale_lookup, scale_payload = _prepare_scale_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    stored_scale = dict(
        (run_config.get("scale_context") or {}).get("standardizer") or {}
    )
    stored_scale_lookup: dict[str, np.ndarray] = {}
    scale_recompute_max_abs = None
    if model_config.get("scale_context_mode", "none") != "none":
        stored_scale_lookup = {
            str(example.sample_id): standardize_scale_contexts(
                [_scale_context_row_for_example(example)], stored_scale
            )[0]
            for example in valid_examples
        }
        scale_recompute_max_abs = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(scale_lookup[sample_id], dtype=np.float64)
                        - np.asarray(stored_scale_lookup[sample_id], dtype=np.float64)
                    )
                )
            )
            for sample_id in stored_scale_lookup
        )
    _attach_scale_context_to_groups(
        groups,
        stored_scale_lookup,
        expected_feature_dim=int(model_config.get("scale_context_feature_dim", 0)),
    )
    by_id = {str(example.sample_id): example for example in valid_examples}
    if (
        model_config.get("scale_pooling") == "qk_gated"
        or model_config.get("shape_attention_mode", "none") != "none"
        or model_config.get("scale_attention_mode", "none") != "none"
    ):
        _attach_qk_region_features_to_groups(
            groups,
            by_id,
            feature_version=str(
                model_config.get("qk_region_feature_version", "bugged_v1")
            ),
        )
    if model_config.get("scale_deepsets_mode", "none") != "none":
        _attach_scale_deepsets_weights_to_groups(groups, by_id)
    return groups, {
        "global_context": global_payload,
        "scale_context": scale_payload,
        "stored_global_standardizer": stored_global,
        "stored_scale_standardizer": stored_scale,
        "global_context_recompute_max_abs": global_recompute_max_abs,
        "scale_context_recompute_max_abs": scale_recompute_max_abs,
    }


def _replay_fields(
    *,
    payload: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    saved_prediction_path: Path,
    valid_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stats = dict(payload["train_only_normalization"])
    model_config = _resolve_decoder_bypass_model_config(
        dict(payload["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(payload["params"])
    saved = _prediction_fields(saved_prediction_path, valid_ids)
    replay: dict[str, np.ndarray] = {}
    maximum = 0.0
    for group in groups:
        prediction = _model_apply(model, params, group)
        raw = np.asarray(prediction["raw_temperature"], dtype=np.float64)
        for index, sample_id_value in enumerate(group["sample_ids"]):
            sample_id = str(sample_id_value)
            field = raw[index].reshape(-1)
            replay[sample_id] = field
            maximum = max(maximum, float(np.max(np.abs(field - saved[sample_id]))))
    del params, model
    gc.collect()
    passed = len(replay) == 128 and maximum <= 0.02
    audit = {
        "sample_count": len(replay),
        "max_abs_error_K": maximum,
        "tolerance_K": 0.02,
        "passed": passed,
    }
    if not passed:
        raise Gate6QCloseoutError(f"checkpoint replay failed: {audit}")
    return replay, audit


def _suite(
    *,
    raw_temperature: Mapping[str, np.ndarray],
    valid_ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    samples = []
    for sample_id in valid_ids:
        target = targets[sample_id]
        prediction_delta = raw_temperature[sample_id] - float(
            target["bottom_temperature_K"]
        )
        true = np.asarray(target["target_deltaT_K"], dtype=np.float64)
        samples.append(
            {
                "sample_id": sample_id,
                "split": "valid_iid",
                "prediction_deltaT_K": prediction_delta,
                "target_deltaT_K": true,
                "control_volumes_m3": target["control_volumes_m3"],
                "q_W_m3": target["q_W_m3"],
                "prediction_normalized": (prediction_delta - mean) / std,
                "target_normalized": (true - mean) / std,
            }
        )
    suite = evaluate_metric_suite(samples)
    for row in suite["per_sample"]:
        sample_id = str(row["sample_id"])
        target = targets[sample_id]
        true = np.asarray(target["target_deltaT_K"], dtype=np.float64)
        prediction = raw_temperature[sample_id] - float(
            target["bottom_temperature_K"]
        )
        true_scale = float(target["true_scale_cv_rms_K"])
        pred_scale = float(row["pred_scale_cv_rms_K"])
        true_shape = true / max(true_scale, EPS)
        pred_shape = prediction / max(pred_scale, EPS)
        shape_term = true_scale * (pred_shape - true_shape)
        scale_term = (pred_scale - true_scale) * pred_shape
        shape_sse = float(np.sum(np.square(shape_term)))
        scale_sse = float(np.sum(np.square(scale_term)))
        cross_sse = float(2.0 * np.sum(shape_term * scale_term))
        direct = float(row["point_error_squared_sum"])
        row.update(
            {
                "deltaT_quartile": target["deltaT_quartile"],
                "shape_point_sse_K2": shape_sse,
                "scale_point_sse_K2": scale_sse,
                "cross_point_sse_K2": cross_sse,
                "decomposition_closure_abs_K2": abs(
                    direct - shape_sse - scale_sse - cross_sse
                ),
            }
        )
    return suite


def main() -> int:
    args = _args()
    if jax.default_backend() != "cpu":
        raise Gate6QCloseoutError(
            f"CPU evaluator required, found backend={jax.default_backend()}"
        )
    run_dir = args.run_dir.resolve()
    if run_dir.name != args.config_id:
        raise Gate6QCloseoutError("config/run directory binding failed")
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "loss_summary.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if Path(str(run_config["output_dir"])).name != args.config_id:
        raise Gate6QCloseoutError("run_config output binding failed")
    history = [int(row["epoch"]) for row in summary.get("epoch_history", ())]
    if (
        int(summary.get("final_epoch", -1)) != 600
        or history != list(range(1, 601))
        or not bool(summary.get("grad_finite"))
    ):
        raise Gate6QCloseoutError("e600 completion audit failed")

    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for checkpoint_name in CHECKPOINTS:
        payload, metadata = _checkpoint(run_dir, checkpoint_name, summary)
        checkpoint_payloads[checkpoint_name] = payload
        checkpoint_metadata[checkpoint_name] = metadata
    canonical = checkpoint_payloads["point_global_best"]
    stats_payload = dict(canonical["train_only_normalization"])
    stats_hashes = {
        metadata["train_stats_hash"] for metadata in checkpoint_metadata.values()
    }
    if len(stats_hashes) != 1:
        raise Gate6QCloseoutError("checkpoint train normalization hashes differ")
    install_checkpoint_feature_hooks(stats_payload)

    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_path = Path(str(run_config["split_map_path"]))
    split_ids, split_source, primary, stress = _resolve_training_splits(
        sample_root, split_path
    )
    train_ids = list(split_ids.get("train") or ())
    valid_ids = list(split_ids.get("valid_iid") or ())
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise Gate6QCloseoutError("unexpected train/valid_iid split counts")
    if set(train_ids).intersection(valid_ids):
        raise Gate6QCloseoutError("train/valid_iid split overlap")
    boundary_fallback = bool(run_config.get("boundary_mask_fallback", True))
    train_examples = _load_examples_for_ids(
        sample_root,
        train_ids,
        role="train",
        boundary_mask_fallback=boundary_fallback,
    )
    valid_examples = _load_examples_for_ids(
        sample_root,
        valid_ids,
        role="valid_iid",
        boundary_mask_fallback=boundary_fallback,
    )
    physics_cache = _physics_cache(train_examples + valid_examples)
    stats = stats_from_checkpoint_payload(stats_payload, train_examples)
    model_config = _resolve_decoder_bypass_model_config(
        dict(canonical["model_config"]), stats
    )
    groups, context_payload = _build_groups(
        run_config=run_config,
        model_config=model_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=valid_examples,
        physics_cache=physics_cache,
        prediction_batch_size=args.prediction_batch_size,
    )
    ordered = [str(value) for group in groups for value in group["sample_ids"]]
    if ordered != valid_ids or any(group["inputs"].x_inp.shape[2] != 1024 for group in groups):
        raise Gate6QCloseoutError("valid group order/node count mismatch")
    targets = _target_cache(sample_root, valid_ids)

    metrics: dict[str, Any] = {}
    reload_audit: dict[str, Any] = {}
    for checkpoint_name, values in CHECKPOINTS.items():
        replay, audit = _replay_fields(
            payload=checkpoint_payloads[checkpoint_name],
            groups=groups,
            saved_prediction_path=run_dir / values[1],
            valid_ids=valid_ids,
        )
        suite = _suite(
            raw_temperature=replay,
            valid_ids=valid_ids,
            targets=targets,
            stats=checkpoint_payloads[checkpoint_name]["train_only_normalization"],
        )
        metrics[checkpoint_name] = suite
        reload_audit[checkpoint_name] = audit

    global_standardizer = context_payload["global_context"].get("standardizer", {})
    if (
        global_standardizer.get("fit_population") != "train_only"
        or int(global_standardizer.get("fit_sample_count", -1)) != 672
    ):
        raise Gate6QCloseoutError("global context was not fit on train only")
    scale_standardizer = context_payload["scale_context"].get("standardizer", {})
    if model_config.get("scale_context_mode", "none") != "none":
        if scale_standardizer.get("fit_roles") != ["train"] or len(
            scale_standardizer.get("fit_sample_ids", ())
        ) != 672:
            raise Gate6QCloseoutError("scale context was not fit on train only")

    metric_path = ROOT / "rigno/heat3d_v5_metrics.py"
    artifacts = {
        path.name: {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            run_config_path,
            summary_path,
            *[run_dir / values[0] for values in CHECKPOINTS.values()],
            *[run_dir / values[1] for values in CHECKPOINTS.values()],
        )
    }
    payload = {
        "schema_version": "heat3d_v5_gate6q_cpu_replay_v1",
        "status": "completed_valid_iid_only",
        "config_id": args.config_id,
        "source_host": args.source_host,
        "run_dir": str(run_dir),
        "training_commit": str(summary.get("code_version_or_git_commit") or ""),
        "evaluator_commit": _git_commit(),
        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "metric_source": {
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
            "checkpoint_selection_modified": False,
            "backend": jax.default_backend(),
            "sample_count": 128,
            "nodes_per_sample": 1024,
        },
        "split": {
            "source": split_source,
            "primary_validation": primary,
            "stress_validation": stress,
            "train_count": 672,
            "valid_iid_count": 128,
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "normalization_and_context": {
            "fit_roles": ["train"],
            "fit_sample_count": 672,
            "global_context_fit_sample_ids_sha256": global_standardizer.get(
                "fit_sample_ids_sha256"
            ),
            "scale_context_fit_sample_ids_sha256": (
                _ids_hash(scale_standardizer["fit_sample_ids"])
                if scale_standardizer
                else None
            ),
            "replay_context_source": "persisted_run_config_standardizers",
            "global_context_recompute_max_abs": context_payload[
                "global_context_recompute_max_abs"
            ],
            "scale_context_recompute_max_abs": context_payload[
                "scale_context_recompute_max_abs"
            ],
            "target_or_label_features": [],
        },
        "training_completion": {
            "final_epoch": 600,
            "epoch_history_count": 600,
            "epoch_history_contiguous": True,
            "grad_finite": True,
            "selection_metric": str(summary.get("selection_metric")),
        },
        "checkpoint_metadata": checkpoint_metadata,
        "metrics": metrics,
        "reload_audit": reload_audit,
        "artifacts": artifacts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "config_id": args.config_id,
                "output_json": str(args.output_json),
                "epochs": {
                    name: metadata["epoch"]
                    for name, metadata in checkpoint_metadata.items()
                },
                "point_global_best": metrics["point_global_best"]["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
