#!/usr/bin/env python3
"""Controlled short V5 clean-first warm-start runner.

This runner intentionally owns only the V5 route.  It uses the frozen V4P5_02
checkpoint, fits all learned context statistics on clean train=672 only, and
selects checkpoints from clean valid_iid=128 only.  Test and hard splits are
evaluated only after selection and never enter optimization or model choice.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "rigno").is_dir() and (Path.cwd() / "rigno").is_dir():
    REPO_ROOT = Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import normalized_delta_to_raw  # noqa: E402
from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    fit_train_only_standardizer,
    global_context_from_raw_condition,
    standardize_contexts,
    validate_global_context_schema,
)
from rigno.heat3d_v5_metrics import (  # noqa: E402
    compute_sample_metrics,
    control_volume_weights,
    evaluate_metric_suite,
)
from rigno.heat3d_v5_shape_scale import native_shape_scale_losses  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _apply_checkpoint_params,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _param_leaf_items,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


RUNNER_ID = "V5-clean-first-short-warmstart-runner"
SCHEMA_VERSION = "heat3d_v5_clean_warmstart_result_v1"
EXPECTED_BASELINE_ID = "V4P5_02_clean_baseline_raw_B28_e600"
EXPECTED_EPOCH = 405
LOCAL_BYPASS_FEATURES = (
    "k_x", "k_y", "k_z", "q", "is_top", "is_bottom", "is_side", "is_interior",
)
REPORT_ROLES = (
    "test_iid", "hard_train_holdout", "hard_challenge_valid", "hard_challenge_test",
)


class WarmStartError(RuntimeError):
    """Raised for a clean-isolation, model, or warm-start invariant violation."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run the explicitly configured short warm-start.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-variant", action="append", default=None)
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WarmStartError(f"cannot read warm-start YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WarmStartError(f"{path}: YAML root must be a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WarmStartError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WarmStartError(f"{path}: JSON root must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(event: str, /, **fields: Any) -> None:
    """Emit one flushed, machine-readable status line for tmux/tail users."""

    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WarmStartError(f"{label} must be a mapping")
    return value


def _as_sequence(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise WarmStartError(f"{label} must be a sequence")
    return tuple(value)


def _resolve_repo_path(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise WarmStartError(f"{label} does not exist: {path}")
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "heat3d_v5_clean_warmstart_short_v1":
        raise WarmStartError("unexpected warm-start config schema")
    if config.get("config_role") != "controlled_short_warm_start":
        raise WarmStartError("warm-start config role must be controlled_short_warm_start")
    dataset = _as_mapping(config.get("dataset"), "dataset")
    if tuple(_as_sequence(dataset.get("fit_roles"), "dataset.fit_roles")) != ("train",):
        raise WarmStartError("warm-start fit roles must be [train]")
    if tuple(_as_sequence(dataset.get("selection_roles"), "dataset.selection_roles")) != ("valid_iid",):
        raise WarmStartError("warm-start selection roles must be [valid_iid]")
    if tuple(_as_sequence(dataset.get("normalization_fit_roles"), "dataset.normalization_fit_roles")) != ("train",):
        raise WarmStartError("normalization fit roles must be [train]")
    run = _as_mapping(config.get("run"), "run")
    epochs = int(run.get("epochs", 0))
    if epochs < 1 or epochs >= 600:
        raise WarmStartError("warm-start epochs must be >=1 and strictly below V4 long budget=600")
    if not bool(run.get("training_allowed", False)):
        raise WarmStartError("warm-start config must explicitly allow its controlled training")
    variants = _as_sequence(config.get("variants"), "variants")
    if len(variants) != 3 or len({str(_as_mapping(item, "variant").get("id")) for item in variants}) != 3:
        raise WarmStartError("warm-start config must define exactly three unique train variants")
    for variant in variants:
        mapping = _as_mapping(variant, "variant")
        family = mapping.get("family")
        if family not in {"legacy_target", "native_shape_scale"}:
            raise WarmStartError(f"unsupported warm-start family {family!r}")
        if mapping.get("global_context_mode") not in {"none", "film"}:
            raise WarmStartError("global_context_mode must be none|film")


def _load_examples(
    *,
    sample_root: Path,
    sample_ids: Sequence[str],
    checkpoint_stats: Mapping[str, Any],
    boundary_mask_fallback: bool,
) -> list[Any]:
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=k_encoding_mode,
        boundary_mask_fallback=boundary_mask_fallback,
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise WarmStartError(f"dataset misses split sample IDs: {missing[:8]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def _raw_context_and_physics(example: Any) -> dict[str, Any]:
    """Derive only inference-time raw coords/k/q/BC/CV values for one sample."""

    relative = example.get_relative_bc_feature_view()
    names = tuple(relative.condition_feature_names)
    values = np.asarray(relative.condition_features, dtype=np.float64)
    coords = np.asarray(example.condition.coords, dtype=np.float64)
    required = ("q", "is_bottom", "bottom_T_fixed_minus_T_ref")
    missing = [name for name in required if name not in names]
    if missing:
        raise WarmStartError(f"{example.sample_id}: raw condition misses {missing}")
    context = global_context_from_raw_condition(
        coords=coords,
        raw_condition=values,
        condition_feature_names=names,
        reference_temperature_K=float(relative.t_ref_value),
    )
    bottom_mask = values[:, names.index("is_bottom")] > 0.5
    if not np.any(bottom_mask):
        raise WarmStartError(f"{example.sample_id}: no Dirichlet bottom nodes")
    reference = np.full(coords.shape[0], float(relative.t_ref_value), dtype=np.float32)
    prescribed = reference + values[:, names.index("bottom_T_fixed_minus_T_ref")].astype(np.float32)
    return {
        "context": context,
        "control_volumes": control_volume_weights(coords).astype(np.float32),
        "q": values[:, names.index("q")].astype(np.float32),
        "reference_temperature": reference,
        "dirichlet_mask": bottom_mask.astype(np.float32),
        "prescribed_temperature": prescribed,
    }


def _physics_cache(examples: Sequence[Any]) -> dict[str, dict[str, Any]]:
    return {str(example.sample_id): _raw_context_and_physics(example) for example in examples}


def _attach_v5_physics(
    groups: Sequence[dict[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
) -> None:
    for group in groups:
        rows = []
        for sample_id in group["sample_ids"]:
            if sample_id not in cache:
                raise WarmStartError(f"{group['name']}: no cached physics for {sample_id}")
            rows.append(cache[sample_id])
        group["v5_physics"] = {
            "control_volumes": jnp.asarray(np.stack([row["control_volumes"] for row in rows])),
            "q": jnp.asarray(np.stack([row["q"] for row in rows])),
            "log_s_phys": jnp.asarray([float(row["context"]["log_s_phys_K"]) for row in rows]),
            "reference_temperature": jnp.asarray(np.stack([row["reference_temperature"] for row in rows])),
            "dirichlet_mask": jnp.asarray(np.stack([row["dirichlet_mask"] for row in rows])),
            "prescribed_temperature": jnp.asarray(np.stack([row["prescribed_temperature"] for row in rows])),
            "global_context": jnp.asarray(standardize_contexts([row["context"] for row in rows], standardizer)),
        }


def _variant_model_config(
    *,
    checkpoint_model_config: Mapping[str, Any],
    stats: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    family = str(variant["family"])
    global_mode = str(variant["global_context_mode"])
    bypass = str(variant["decoder_bypass"])
    config = copy.deepcopy(dict(checkpoint_model_config))
    if bypass == "full_condition":
        config["decoder_bypass_mode"] = "post_decoder_residual"
        config["decoder_bypass_features"] = "full_condition"
        config.pop("decoder_bypass_local_feature_names", None)
    elif bypass == "explicit_local_condition":
        config.update(
            {
                "decoder_bypass_mode": "post_decoder_residual",
                "decoder_bypass_features": "explicit_local_condition",
                "decoder_bypass_local_feature_names": LOCAL_BYPASS_FEATURES,
            }
        )
    elif bypass == "none":
        config.update(
            {
                "decoder_bypass_mode": "none",
                "decoder_bypass_features": "none",
                "decoder_bypass_local_feature_names": (),
            }
        )
    else:
        raise WarmStartError(f"unsupported decoder bypass {bypass!r}")
    config = _resolve_decoder_bypass_model_config(config, dict(stats))
    config.update(
        {
            "global_context_mode": global_mode,
            "global_context_feature_dim": len(GLOBAL_CONTEXT_FEATURES),
            "global_context_feature_names": tuple(GLOBAL_CONTEXT_FEATURES),
            "film_target": "rnodes_processed",
            "film_init": "identity",
            "film_hidden_size": 64,
            "native_output_mode": "native_shape_scale" if family == "native_shape_scale" else "legacy_normalized_deltaT",
            "shape_scale_epsilon": 1.0e-12,
            "scale_head_hidden_size": 64,
            "scale_head_init": "identity",
        }
    )
    if family == "native_shape_scale":
        config["decoder_bypass_output_space"] = "native_psi"
    else:
        config["decoder_bypass_output_space"] = "normalized_deltaT"
    return config


def _native_apply(model, params, group: Mapping[str, Any]):
    physics = group["v5_physics"]
    return model.apply(
        {"params": params},
        inputs=group["inputs"],
        graphs=group["graphs"],
        control_volumes=physics["control_volumes"],
        log_s_phys=physics["log_s_phys"],
        reference_temperature=physics["reference_temperature"],
        dirichlet_mask=physics["dirichlet_mask"],
        prescribed_temperature=physics["prescribed_temperature"],
        global_context=physics["global_context"],
        method=model.predict_native_shape_scale,
    )


def _legacy_apply(model, params, group: Mapping[str, Any]):
    return model.apply(
        {"params": params},
        inputs=group["inputs"],
        graphs=group["graphs"],
        global_context=group["v5_physics"]["global_context"],
    )


def _normalized_from_raw(raw_delta: Any, stats: Mapping[str, Any]) -> jnp.ndarray:
    mean = jnp.asarray(stats["target_delta_mean"], dtype=jnp.float32)
    std = jnp.asarray(stats["target_delta_std"], dtype=jnp.float32)
    return (jnp.asarray(raw_delta) - mean) / std


def _prediction_views(
    model,
    params,
    group: Mapping[str, Any],
    *,
    family: str,
    stats: Mapping[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, Mapping[str, Any] | None]:
    if family == "legacy_target":
        normalized = _legacy_apply(model, params, group)
        return normalized, normalized_delta_to_raw(normalized, stats), None
    native = _native_apply(model, params, group)
    raw = native["deltaT_hat"]
    return _normalized_from_raw(raw, stats), raw, native


def _evaluate_groups(
    model,
    params,
    groups: Sequence[Mapping[str, Any]],
    *,
    family: str,
    stats: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    temperatures: dict[str, np.ndarray] = {}
    for group in groups:
        normalized, raw_delta, native = _prediction_views(
            model, params, group, family=family, stats=stats
        )
        raw_np = np.asarray(raw_delta, dtype=np.float64)
        normalized_np = np.asarray(normalized, dtype=np.float64)
        target_raw = np.asarray(group["target_delta_raw"], dtype=np.float64)
        target_norm = np.asarray(group["target_normalized"], dtype=np.float64)
        physics = group["v5_physics"]
        volumes = np.asarray(physics["control_volumes"], dtype=np.float64)
        q_values = np.asarray(physics["q"], dtype=np.float64)
        t_ref = np.asarray(group["t_ref"], dtype=np.float64)
        if native is not None:
            recovered_temperature = np.asarray(native["raw_temperature"], dtype=np.float64)
        else:
            recovered_temperature = t_ref + raw_np
        for index, sample_id in enumerate(group["sample_ids"]):
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": role,
                    "prediction_deltaT_K": raw_np[index].reshape(-1),
                    "target_deltaT_K": target_raw[index].reshape(-1),
                    "control_volumes_m3": volumes[index],
                    "q_W_m3": q_values[index],
                    "prediction_normalized": normalized_np[index],
                    "target_normalized": target_norm[index],
                }
            )
            temperatures[sample_id] = recovered_temperature[index, 0, :, :].reshape(-1)
    suite = evaluate_metric_suite(samples)
    return {"suite": suite, "temperatures": temperatures}


def _loss_for_group(
    model,
    params,
    group: Mapping[str, Any],
    *,
    family: str,
    stats: Mapping[str, Any],
    native_loss_weights: Mapping[str, float],
) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    if family == "legacy_target":
        prediction = _legacy_apply(model, params, group)
        base = jnp.mean(jnp.square(prediction - group["target_normalized"]))
        return base, {"legacy_normalized_mse": base, "total_loss": base}
    prediction = _native_apply(model, params, group)
    components = native_shape_scale_losses(
        prediction,
        target_deltaT=group["target_delta_raw"],
        control_volumes=group["v5_physics"]["control_volumes"],
        loss_weights=native_loss_weights,
    )
    return components["total_loss"], components


def _global_norm(tree_value: Any) -> float:
    leaves = [jnp.sum(jnp.square(leaf)) for leaf in tree_util.tree_leaves(tree_value)]
    return float(jnp.sqrt(jnp.sum(jnp.asarray(leaves)))) if leaves else 0.0


def _tree_finite(value: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in tree_util.tree_leaves(value))


def _is_primary_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    return (
        float(candidate["sample_first_cv_relative_rmse_pct"]),
        float(candidate["raw_cv_weighted_rmse_K"]),
    ) < (
        float(best["sample_first_cv_relative_rmse_pct"]),
        float(best["raw_cv_weighted_rmse_K"]),
    )


def _host_params(params: Any) -> Any:
    return tree_util.tree_map(lambda value: np.asarray(value), jax.device_get(params))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def _safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, jnp.floating)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checkpoint(path: Path, *, params: Any, epoch: int, model_config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "params": _host_params(params),
                "epoch": int(epoch),
                "model_config": dict(model_config),
                "provenance": dict(provenance),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _write_predictions(path: Path, temperatures: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **{sample_id: np.asarray(value, dtype=np.float64) for sample_id, value in temperatures.items()})


def _initialize_variant(
    *,
    variant: Mapping[str, Any],
    model_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    group: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    family = str(variant["family"])
    model = GraphNeuralOperator(**dict(model_config))
    if family == "native_shape_scale":
        physics = group["v5_physics"]
        initial = model.init(
            jax.random.PRNGKey(seed),
            inputs=group["inputs"],
            graphs=group["graphs"],
            control_volumes=physics["control_volumes"],
            log_s_phys=physics["log_s_phys"],
            reference_temperature=physics["reference_temperature"],
            dirichlet_mask=physics["dirichlet_mask"],
            prescribed_temperature=physics["prescribed_temperature"],
            global_context=physics["global_context"],
            method=model.predict_native_shape_scale,
        )["params"]
    else:
        initial = model.init(
            jax.random.PRNGKey(seed),
            inputs=group["inputs"],
            graphs=group["graphs"],
            global_context=group["v5_physics"]["global_context"],
        )["params"]
    params, info = _apply_checkpoint_params(
        initial,
        checkpoint_payload["params"],
        strict=False,
        partial_load_policy=str(variant["partial_load_policy"]),
    )
    if family == "native_shape_scale":
        initial_map = {path: np.asarray(value) for path, value in _param_leaf_items(initial)}
        loaded_map = {path: np.asarray(value) for path, value in _param_leaf_items(params)}
        frozen_map = {path: np.asarray(value) for path, value in _param_leaf_items(checkpoint_payload["params"])}
        encoder_processor = [
            path for path in initial_map if path.startswith("encoder/") or path.startswith("processor/")
        ]
        if any(path not in frozen_map or not np.array_equal(loaded_map[path], frozen_map[path]) for path in encoder_processor):
            raise WarmStartError(f"{variant['id']}: failed encoder/processor partial load")
    return model, params, info


def _train_variant(
    *,
    variant: Mapping[str, Any],
    model_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    train_groups: Sequence[Mapping[str, Any]],
    valid_groups: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    run: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    native_loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    try:
        import optax
    except ImportError as exc:
        raise WarmStartError("optax is required for controlled V5 warm-start") from exc
    family = str(variant["family"])
    seed = int(run["seed"])
    _progress(
        "variant_start",
        variant=str(variant["id"]),
        family=family,
        epochs=int(run["epochs"]),
    )
    model, params, load_info = _initialize_variant(
        variant=variant,
        model_config=model_config,
        checkpoint_payload=checkpoint_payload,
        group=train_groups[0],
        seed=seed,
    )
    _progress(
        "variant_initialized",
        variant=str(variant["id"]),
        partial_load_policy=str(variant["partial_load_policy"]),
        loaded_leaf_count=int(load_info.get("loaded_key_count", 0)),
        skipped_leaf_count=int(load_info.get("skipped_key_count", 0)),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(float(optimizer_config["gradient_clip_norm"])),
        optax.adamw(
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        ),
    )
    opt_state = tx.init(params)
    initial_eval = _evaluate_groups(model, params, valid_groups, family=family, stats=stats, role="valid_iid")
    primary_params = params
    legacy_params = params
    primary_epoch = 0
    legacy_epoch = 0
    primary_summary = initial_eval["suite"]["summary"]
    legacy_summary = initial_eval["suite"]["summary"]
    history: list[dict[str, Any]] = []
    finite = True
    rng = np.random.default_rng(seed)
    for epoch in range(1, int(run["epochs"]) + 1):
        order = rng.permutation(len(train_groups))
        train_losses: list[float] = []
        train_components: dict[str, list[float]] = {}
        grad_norms: list[float] = []
        for group_index in order:
            group = train_groups[int(group_index)]

            def _objective(current_params):
                return _loss_for_group(
                    model,
                    current_params,
                    group,
                    family=family,
                    stats=stats,
                    native_loss_weights=native_loss_weights,
                )

            (loss_value, components), gradients = jax.value_and_grad(_objective, has_aux=True)(params)
            finite = finite and bool(math.isfinite(float(loss_value))) and _tree_finite(gradients)
            grad_norms.append(_global_norm(gradients))
            updates, opt_state = tx.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            train_losses.append(float(loss_value))
            for key, value in components.items():
                if key in {"target_shape", "target_scale", "s_hat_positive"}:
                    continue
                train_components.setdefault(key, []).append(float(value))
        valid_eval = _evaluate_groups(model, params, valid_groups, family=family, stats=stats, role="valid_iid")
        valid_summary = valid_eval["suite"]["summary"]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_components": {key: float(np.mean(values)) for key, values in train_components.items()},
            "gradient_norm_mean": float(np.mean(grad_norms)),
            "gradient_norm_max": float(np.max(grad_norms)),
            "valid_summary": valid_summary,
        }
        history.append(record)
        _progress(
            "epoch_complete",
            variant=str(variant["id"]),
            epoch=epoch,
            train_loss=record["train_loss"],
            gradient_norm_mean=record["gradient_norm_mean"],
            valid_sample_first_cv_relative_rmse_pct=float(
                valid_summary["sample_first_cv_relative_rmse_pct"]
            ),
            valid_point_global_relative_rmse_pct=float(
                valid_summary["point_global_relative_rmse_pct"]
            ),
            valid_raw_cv_weighted_rmse_K=float(valid_summary["raw_cv_weighted_rmse_K"]),
        )
        if _is_primary_better(valid_summary, primary_summary):
            primary_params = params
            primary_epoch = epoch
            primary_summary = valid_summary
        if float(valid_summary["legacy_normalized_valid_base_mse"]) < float(legacy_summary["legacy_normalized_valid_base_mse"]):
            legacy_params = params
            legacy_epoch = epoch
            legacy_summary = valid_summary
    final_eval = _evaluate_groups(model, params, valid_groups, family=family, stats=stats, role="valid_iid")
    shape_scale_stable = True
    if family == "native_shape_scale":
        for key in ("shape_cv_loss", "log_scale_loss"):
            values = [record["train_components"].get(key) for record in history]
            if not values or any(value is None or not math.isfinite(float(value)) for value in values):
                shape_scale_stable = False
            elif float(values[-1]) >= float(values[0]):
                shape_scale_stable = False
    primary_eval = _evaluate_groups(
        model, primary_params, valid_groups, family=family, stats=stats, role="valid_iid"
    )
    legacy_eval = _evaluate_groups(
        model, legacy_params, valid_groups, family=family, stats=stats, role="valid_iid"
    )
    return {
        "model": model,
        "model_config": model_config,
        "family": family,
        "load_info": load_info,
        "initial_eval": initial_eval,
        "final_params": params,
        "final_eval": final_eval,
        "primary_params": primary_params,
        "primary_epoch": primary_epoch,
        "primary_eval": primary_eval,
        "legacy_params": legacy_params,
        "legacy_epoch": legacy_epoch,
        "legacy_eval": legacy_eval,
        "history": history,
        "finite": finite,
        "shape_scale_stable": shape_scale_stable,
    }


def _baseline_model_and_params(
    checkpoint_payload: Mapping[str, Any], stats: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    config = _resolve_decoder_bypass_model_config(dict(checkpoint_payload["model_config"]), dict(stats))
    config.update(
        {
            "global_context_mode": "none",
            "global_context_feature_dim": 0,
            "global_context_feature_names": (),
            "native_output_mode": "legacy_normalized_deltaT",
            "decoder_bypass_output_space": "normalized_deltaT",
        }
    )
    return GraphNeuralOperator(**config), _device_params(checkpoint_payload["params"]), config


def _build_groups(
    examples: Sequence[Any],
    *,
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    label: str,
    graph_seed: int,
    batch_size: int,
    cache: Mapping[str, Mapping[str, Any]],
    standardizer: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups = _make_groups_with_progress(
        examples, dict(stats), builder, label, False, "basic", graph_seed,
        batch_size=batch_size, drop_last=False,
    )
    _attach_v5_physics(groups, cache, standardizer)
    return groups


def _report_selected(
    result: Mapping[str, Any],
    groups_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for checkpoint_name, params_key in (("primary_relative", "primary_params"), ("legacy_metric", "legacy_params")):
        split_reports = {}
        for role, groups in groups_by_role.items():
            evaluated = _evaluate_groups(
                result["model"], result[params_key], groups,
                family=result["family"], stats=stats, role=role,
            )
            split_reports[role] = evaluated
        reports[checkpoint_name] = split_reports
    return reports


def _gate(
    *,
    baseline: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline["suite"]["summary"]
    primary = result["primary_eval"]["suite"]["summary"]
    initial = result["initial_eval"]["suite"]["summary"]
    final = result["final_eval"]["suite"]["summary"]
    min_improvement_pp = 0.25
    sample_improvement = float(baseline_summary["sample_first_cv_relative_rmse_pct"]) - float(primary["sample_first_cv_relative_rmse_pct"])
    point_improvement = float(baseline_summary["point_global_relative_rmse_pct"]) - float(primary["point_global_relative_rmse_pct"])
    quality_non_degrade = {
        key: float(primary[key]) <= float(baseline_summary[key]) * 1.05
        for key in ("raw_cv_weighted_rmse_K", "hotspot_cv_weighted_rmse_K", "strong_q_cv_weighted_rmse_K")
    }
    no_late_rollback = float(final["sample_first_cv_relative_rmse_pct"]) <= float(primary["sample_first_cv_relative_rmse_pct"]) * 1.10
    return {
        "minimum_direction_improvement_pp": min_improvement_pp,
        "sample_first_improvement_pp": sample_improvement,
        "point_global_improvement_pp": point_improvement,
        "sample_first_improves": sample_improvement >= min_improvement_pp,
        "point_global_improves": point_improvement >= min_improvement_pp,
        "raw_hotspot_strong_q_non_degrade": quality_non_degrade,
        "raw_hotspot_strong_q_non_degrade_pass": all(quality_non_degrade.values()),
        "shape_scale_losses_stable": bool(result["shape_scale_stable"]),
        "finite_gradients": bool(result["finite"]),
        "no_late_immediate_rollback": no_late_rollback,
        "initial_valid_summary": initial,
        "final_valid_summary": final,
        "pass": bool(
            sample_improvement >= min_improvement_pp
            and point_improvement >= min_improvement_pp
            and all(quality_non_degrade.values())
            and result["finite"]
            and result["shape_scale_stable"]
            and no_late_rollback
        ),
    }


def _save_variant_artifacts(
    *,
    root: Path,
    variant_id: str,
    result: Mapping[str, Any],
    reports: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    output = root / variant_id
    output.mkdir(parents=True, exist_ok=False)
    _write_checkpoint(
        output / "params_final.pkl", params=result["final_params"], epoch=len(result["history"]),
        model_config=result["model_config"], provenance=provenance,
    )
    _write_checkpoint(
        output / "params_primary_relative.pkl", params=result["primary_params"], epoch=result["primary_epoch"],
        model_config=result["model_config"], provenance=provenance,
    )
    _write_checkpoint(
        output / "params_legacy_metric.pkl", params=result["legacy_params"], epoch=result["legacy_epoch"],
        model_config=result["model_config"], provenance=provenance,
    )
    _write_predictions(output / "predictions_primary_relative_valid_iid.npz", result["primary_eval"]["temperatures"])
    _write_predictions(output / "predictions_legacy_metric_valid_iid.npz", result["legacy_eval"]["temperatures"])
    _write_predictions(output / "predictions_final_valid_iid.npz", result["final_eval"]["temperatures"])
    compact_reports = {
        checkpoint: {
            role: evaluated["suite"]["summary"]
            for role, evaluated in split_reports.items()
        }
        for checkpoint, split_reports in reports.items()
    }
    _write_json(
        output / "loss_summary.json",
        {
            "runner_id": RUNNER_ID,
            "history": result["history"],
            "initial_valid": result["initial_eval"]["suite"]["summary"],
            "final_valid": result["final_eval"]["suite"]["summary"],
            "primary_relative_epoch": result["primary_epoch"],
            "legacy_metric_epoch": result["legacy_epoch"],
            "checkpoint_load": result["load_info"],
        },
    )
    _write_json(output / "clean_metrics.json", {"reports": compact_reports})
    _write_json(output / "provenance.json", provenance)
    return {"output_dir": str(output)}


def _run(config: Mapping[str, Any], config_path: Path, *, only_variants: set[str] | None) -> dict[str, Any]:
    _validate_config(config)
    frozen = _as_mapping(config["frozen_reference"], "frozen_reference")
    dataset_config = _as_mapping(config["dataset"], "dataset")
    run = _as_mapping(config["run"], "run")
    optimizer_config = _as_mapping(config["optimizer"], "optimizer")
    native_loss_weights = _as_mapping(config["native_loss_weights"], "native_loss_weights")
    checkpoint_path = _resolve_repo_path(frozen["checkpoint"], "frozen checkpoint")
    run_config_path = _resolve_repo_path(frozen["run_config"], "frozen run config")
    subset = _resolve_repo_path(dataset_config["subset_path"], "dataset subset")
    split_map = _resolve_repo_path(dataset_config["split_map_path"], "split map")
    output_root = REPO_ROOT / str(run["output_dir"])
    if output_root.exists():
        raise WarmStartError(f"warm-start output directory already exists: {output_root}")
    if "output" not in output_root.parts:
        raise WarmStartError("warm-start output_dir must remain under output/")
    _progress(
        "warmstart_preflight",
        run_id=str(run["run_id"]),
        epochs=int(run["epochs"]),
        output_dir=str(output_root),
    )
    checkpoint_payload = _load_params_checkpoint(checkpoint_path)
    if int(checkpoint_payload.get("epoch", -1)) != int(frozen.get("checkpoint_epoch", EXPECTED_EPOCH)):
        raise WarmStartError("frozen checkpoint epoch does not match config")
    if EXPECTED_BASELINE_ID not in checkpoint_path.as_posix():
        raise WarmStartError("warm-start must use frozen V4P5_02 baseline")
    raw_run_config = _load_json(run_config_path)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise WarmStartError("frozen checkpoint lacks train-only normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(raw_run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(subset)
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    if len(split_ids.get("train", [])) != 672 or len(split_ids.get("valid_iid", [])) != 128:
        raise WarmStartError("V5 clean warm-start requires train=672 and valid_iid=128")
    train_ids = list(split_ids["train"])
    all_report_ids = [sample_id for role in REPORT_ROLES for sample_id in split_ids.get(role, [])]
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=split_ids["valid_iid"],
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(raw_run_config.get("boundary_mask_fallback", True)),
    )
    report_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=all_report_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(raw_run_config.get("boundary_mask_fallback", True)),
    )
    all_examples = [*train_examples, *valid_examples, *report_examples]
    cache = _physics_cache(all_examples)
    standardizer = fit_train_only_standardizer(
        [cache[sample_id]["context"] for sample_id in train_ids], fit_sample_ids=train_ids
    )
    if standardizer["fit_sample_count"] != 672 or standardizer["fit_population"] != "train_only":
        raise WarmStartError("global context standardizer violates train-only clean contract")
    builder = Heat3DGraphBuilder(**dict(raw_run_config.get("graph_config") or {}))
    graph_seed = int(raw_run_config.get("graph_seed", 0))
    train_groups = _build_groups(
        train_examples, stats=stats, builder=builder, label="v5_train", graph_seed=graph_seed,
        batch_size=int(run["train_batch_size"]), cache=cache, standardizer=standardizer,
    )
    valid_groups = _build_groups(
        valid_examples, stats=stats, builder=builder, label="valid_iid", graph_seed=graph_seed,
        batch_size=int(run["evaluation_batch_size"]), cache=cache, standardizer=standardizer,
    )
    report_groups: dict[str, list[dict[str, Any]]] = {}
    report_index = {example.sample_id: example for example in report_examples}
    for role in REPORT_ROLES:
        ids = list(split_ids.get(role, []))
        examples = [report_index[sample_id] for sample_id in ids]
        report_groups[role] = _build_groups(
            examples, stats=stats, builder=builder, label=role, graph_seed=graph_seed,
            batch_size=int(run["evaluation_batch_size"]), cache=cache, standardizer=standardizer,
        )
    baseline_model, baseline_params, baseline_config = _baseline_model_and_params(checkpoint_payload, stats)
    baseline_valid = _evaluate_groups(
        baseline_model, baseline_params, valid_groups, family="legacy_target", stats=stats, role="valid_iid"
    )
    baseline_reports = {
        role: _evaluate_groups(
            baseline_model, baseline_params, groups, family="legacy_target", stats=stats, role=role
        )
        for role, groups in report_groups.items()
    }
    _progress(
        "baseline_evaluated",
        valid_sample_first_cv_relative_rmse_pct=float(
            baseline_valid["suite"]["summary"]["sample_first_cv_relative_rmse_pct"]
        ),
        valid_point_global_relative_rmse_pct=float(
            baseline_valid["suite"]["summary"]["point_global_relative_rmse_pct"]
        ),
        valid_raw_cv_weighted_rmse_K=float(
            baseline_valid["suite"]["summary"]["raw_cv_weighted_rmse_K"]
        ),
    )
    variants = [dict(_as_mapping(item, "variant")) for item in _as_sequence(config["variants"], "variants")]
    if only_variants is not None:
        variants = [variant for variant in variants if str(variant["id"]) in only_variants]
        missing = only_variants - {str(variant["id"]) for variant in variants}
        if missing:
            raise WarmStartError(f"requested variants not found: {sorted(missing)}")
    output_root.mkdir(parents=True, exist_ok=False)
    provenance = {
        "runner_id": RUNNER_ID,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "frozen_checkpoint": str(checkpoint_path),
        "frozen_checkpoint_sha256": _sha256(checkpoint_path),
        "frozen_epoch": int(checkpoint_payload["epoch"]),
        "split_map": str(split_map),
        "split_map_sha256": _sha256(split_map),
        "fit_roles": ["train"],
        "selection_roles": ["valid_iid"],
        "report_only_roles": list(REPORT_ROLES),
        "global_context_standardizer": standardizer,
        "target_or_label_derived_model_inputs": False,
        "training_epochs": int(run["epochs"]),
        "full_long_training_started": False,
        "scratch_training_started": False,
    }
    _write_json(output_root / "provenance.json", provenance)
    baseline_payload = {
        "model_config": baseline_config,
        "valid_iid": baseline_valid["suite"]["summary"],
        "report_only": {role: item["suite"]["summary"] for role, item in baseline_reports.items()},
    }
    results: dict[str, Any] = {}
    for variant in variants:
        model_config = _variant_model_config(
            checkpoint_model_config=checkpoint_payload["model_config"], stats=stats, variant=variant
        )
        trained = _train_variant(
            variant=variant,
            model_config=model_config,
            checkpoint_payload=checkpoint_payload,
            train_groups=train_groups,
            valid_groups=valid_groups,
            stats=stats,
            run=run,
            optimizer_config=optimizer_config,
            native_loss_weights=native_loss_weights,
        )
        reports = _report_selected(trained, report_groups, stats=stats)
        gate = _gate(baseline=baseline_valid, result=trained)
        _progress(
            "variant_gate",
            variant=str(variant["id"]),
            passed=bool(gate["pass"]),
            sample_first_improvement_pp=float(gate["sample_first_improvement_pp"]),
            point_global_improvement_pp=float(gate["point_global_improvement_pp"]),
        )
        artifact_info = _save_variant_artifacts(
            root=output_root,
            variant_id=str(variant["id"]),
            result=trained,
            reports=reports,
            provenance={**provenance, "variant": variant, "gate": gate},
        )
        results[str(variant["id"])] = {
            "variant": variant,
            "model_config": model_config,
            "load_info": trained["load_info"],
            "initial_valid": trained["initial_eval"]["suite"]["summary"],
            "final_valid": trained["final_eval"]["suite"]["summary"],
            "primary_relative": {
                "epoch": trained["primary_epoch"],
                "valid_iid": trained["primary_eval"]["suite"]["summary"],
                "report_only": {
                    role: reports["primary_relative"][role]["suite"]["summary"] for role in REPORT_ROLES
                },
            },
            "legacy_metric": {
                "epoch": trained["legacy_epoch"],
                "valid_iid": trained["legacy_eval"]["suite"]["summary"],
                "report_only": {
                    role: reports["legacy_metric"][role]["suite"]["summary"] for role in REPORT_ROLES
                },
            },
            "history": trained["history"],
            "gate": gate,
            "artifacts": artifact_info,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "status": "completed",
        "provenance": provenance,
        "baseline": baseline_payload,
        "variants": results,
        "scratch_yaml_allowed": any(item["gate"]["pass"] for item in results.values()),
        "selection_policy": config["selection"],
    }
    _write_json(output_root / "warmstart_result.json", payload)
    _write_json(output_root / "loss_summary.json", {
        "baseline_valid": baseline_valid["suite"]["summary"],
        "variant_histories": {key: value["history"] for key, value in results.items()},
    })
    return payload


def _dry_run(config: Mapping[str, Any], config_path: Path, only_variants: set[str] | None) -> dict[str, Any]:
    _validate_config(config)
    frozen = _as_mapping(config["frozen_reference"], "frozen_reference")
    dataset = _as_mapping(config["dataset"], "dataset")
    run = _as_mapping(config["run"], "run")
    variants = [str(_as_mapping(item, "variant")["id"]) for item in _as_sequence(config["variants"], "variants")]
    if only_variants is not None:
        variants = [variant for variant in variants if variant in only_variants]
    return {
        "runner_id": RUNNER_ID,
        "mode": "dry_run",
        "config": str(config_path),
        "checkpoint": str(frozen["checkpoint"]),
        "subset": str(dataset["subset_path"]),
        "split_map": str(dataset["split_map_path"]),
        "fit_roles": list(dataset["fit_roles"]),
        "selection_roles": list(dataset["selection_roles"]),
        "report_only_roles": list(dataset["report_only_roles"]),
        "epochs": int(run["epochs"]),
        "variants": variants,
        "training_runs": 0,
        "planned_output_dir": str(run["output_dir"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = _load_yaml(config_path)
        selected = set(args.only_variant) if args.only_variant else None
        if args.dry_run or not args.execute:
            print(json.dumps(_dry_run(config, config_path, selected), indent=2, sort_keys=True))
            return 0
        payload = _run(config, config_path, only_variants=selected)
    except (WarmStartError, ValueError, OSError) as exc:
        print(f"V5 warm-start error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "runner_id": RUNNER_ID,
                "status": payload["status"],
                "scratch_yaml_allowed": payload["scratch_yaml_allowed"],
                "variants": {key: value["gate"]["pass"] for key, value in payload["variants"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
