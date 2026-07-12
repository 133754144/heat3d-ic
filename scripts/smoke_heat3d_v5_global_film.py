#!/usr/bin/env python3
"""Read-only V5 Global FiLM smoke against frozen V4P5_02 epoch 405.

The smoke deliberately reuses the frozen model graph and checkpoint.  Its only
new input is a train-standardized, inference-only sample-global context built
from raw coords/k/q/BC/control-volume data.  It never feeds targets or writes
data, checkpoints, logs, or output/ directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "rigno").is_dir() and (Path.cwd() / "rigno").is_dir():
    # Remote verification copies this immutable script to /tmp.
    REPO_ROOT = Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import tree_util  # noqa: E402

from rigno.heat3d_v1_normalization import recover_temperature_from_normalized_delta  # noqa: E402
from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    FEATURE_PROVENANCE,
    batch_global_context_from_raw_condition,
    fit_train_only_standardizer,
    standardize_contexts,
    validate_global_context_schema,
)
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
    _validate_model_config,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


SMOKE_ID = "V5-global-film-identity-smoke"
SCHEMA_VERSION = "heat3d_v5_global_film_smoke_v1"
EXPECTED_BASELINE_ID = "V4P5_02_clean_baseline_raw_B28_e600"
EXPECTED_EPOCH = 405
REPLAY_TOLERANCE_K = 2.0e-2
JIT_EAGER_TOLERANCE_K = 3.0e-2
EPS = 1.0e-12


class FilmSmokeError(RuntimeError):
    """Raised if an identity/replay/provenance invariant is violated."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument("--frozen-valid-predictions", type=Path, required=True)
    parser.add_argument("--gate1-table", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=EXPECTED_EPOCH)
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilmSmokeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FilmSmokeError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_output(path: Path, label: str, overwrite: bool) -> Path:
    resolved = path.resolve()
    forbidden = {"data", "output", "checkpoints", "logs"}
    if any(part in forbidden for part in resolved.parts):
        raise FilmSmokeError(f"--{label} must not write under data/output/checkpoints/logs")
    if resolved.exists() and not overwrite:
        raise FilmSmokeError(f"--{label} already exists (use --overwrite): {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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
        raise FilmSmokeError(f"dataset is missing frozen split sample IDs: {missing[:8]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def _context_rows(
    examples: Sequence[Any],
    checkpoint_stats: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Build raw-condition contexts without accessing any example target."""

    rows: dict[str, dict[str, float]] = {}
    physical_log_scales: dict[str, float] = {}
    for example in examples:
        # Do not reuse the JAX bridge here: it is float32 and would turn this
        # inference-only physical feature path into a lossy target-side view.
        relative_view = example.get_relative_bc_feature_view()
        raw_condition = np.asarray(relative_view.condition_features, dtype=np.float64)
        raw_coords = np.asarray(example.condition.coords, dtype=np.float64).reshape(-1, 3)
        t_ref = float(relative_view.t_ref_value)
        if not math.isfinite(t_ref):
            raise FilmSmokeError(f"{example.sample_id}: invalid prescribed reference temperature")
        rows[example.sample_id] = batch_global_context_from_raw_condition(
            coords_per_sample=(raw_coords,),
            raw_conditions_per_sample=(raw_condition,),
            condition_feature_names=tuple(relative_view.condition_feature_names),
            reference_temperatures_K=(t_ref,),
        )[0]
        physical_log_scales[example.sample_id] = float(rows[example.sample_id]["log_s_phys_K"])
    if len(rows) != len(examples):
        raise FilmSmokeError("global-context construction lost or duplicated a sample")
    return rows, physical_log_scales


def _load_gate1_operator(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FilmSmokeError(f"Gate-1 table does not exist: {path}")
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise FilmSmokeError(f"{path}: Gate-1 row misses sample_id")
            try:
                value = float(row["raw_z_collapsed_1d_operator_K"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FilmSmokeError(f"{path}: bad raw_z_collapsed_1d_operator_K for {sample_id}") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise FilmSmokeError(f"{path}: non-positive Gate-1 operator for {sample_id}")
            values[sample_id] = value
    return values


def _load_frozen_archive(path: Path, expected_ids: Sequence[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FilmSmokeError(f"frozen valid prediction archive does not exist: {path}")
    archive = np.load(path, allow_pickle=False)
    expected = set(expected_ids)
    found = set(archive.files)
    if found != expected:
        raise FilmSmokeError(
            "frozen valid archive coverage differs from valid_iid split: "
            f"missing={sorted(expected - found)[:5]} extra={sorted(found - expected)[:5]}"
        )
    return {
        sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
        for sample_id in expected_ids
    }


def _tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    for leaf in tree_util.tree_leaves(value):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _tree_all_finite(value: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in tree_util.tree_leaves(value))


def _checkpoint_load_assertions(
    initial_params: Any,
    loaded_params: Any,
    checkpoint_params: Any,
) -> dict[str, Any]:
    initial = {path: np.asarray(value) for path, value in _param_leaf_items(initial_params)}
    loaded = {path: np.asarray(value) for path, value in _param_leaf_items(loaded_params)}
    checkpoint = {path: np.asarray(value) for path, value in _param_leaf_items(checkpoint_params)}
    for prefix in ("encoder/", "processor/"):
        paths = [path for path in initial if path.startswith(prefix)]
        if not paths:
            raise FilmSmokeError(f"no {prefix} parameters found in FiLM model")
        bad = [
            path for path in paths
            if path not in checkpoint
            or tuple(initial[path].shape) != tuple(checkpoint[path].shape)
            or not np.array_equal(loaded[path], checkpoint[path])
        ]
        if bad:
            raise FilmSmokeError(f"partial checkpoint load failed for {prefix}: {bad[:3]}")
    output_paths = [path for path in initial if path.startswith("global_film_output/")]
    if not output_paths or any(not np.all(initial[path] == 0.0) for path in output_paths):
        raise FilmSmokeError("FiLM output layer is not zero-initialized")
    if any(not np.array_equal(loaded[path], initial[path]) for path in output_paths):
        raise FilmSmokeError("checkpoint unexpectedly overwrote new FiLM output parameters")
    matching_paths = [
        path
        for path in initial
        if path in checkpoint and tuple(initial[path].shape) == tuple(checkpoint[path].shape)
    ]
    mismatched_loaded = [
        path for path in matching_paths if not np.array_equal(loaded[path], checkpoint[path])
    ]
    if mismatched_loaded:
        raise FilmSmokeError(
            f"matching checkpoint leaves were not restored: {mismatched_loaded[:3]}"
        )
    return {
        "encoder_loaded_leaf_count": sum(path.startswith("encoder/") for path in initial),
        "processor_loaded_leaf_count": sum(path.startswith("processor/") for path in initial),
        "identity_zero_initialized_leaf_paths": output_paths,
        "all_matching_checkpoint_leaf_count": len(matching_paths),
    }


def _group_context(group: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], standardizer: Mapping[str, Any]):
    missing = [sample_id for sample_id in group["sample_ids"] if sample_id not in rows]
    if missing:
        raise FilmSmokeError(f"group lacks global context rows: {missing[:3]}")
    ordered = [rows[sample_id] for sample_id in group["sample_ids"]]
    return jnp.asarray(standardize_contexts(ordered, standardizer))


def _validate_checkpoint(path: Path, payload: Mapping[str, Any], expected_epoch: int) -> None:
    if not path.is_file():
        raise FilmSmokeError(f"checkpoint does not exist: {path}")
    if EXPECTED_BASELINE_ID not in path.as_posix():
        raise FilmSmokeError(f"checkpoint must be frozen baseline {EXPECTED_BASELINE_ID}")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise FilmSmokeError(
            f"checkpoint epoch mismatch: expected {expected_epoch}, found {payload.get('epoch')!r}"
        )


def _intermediate_array(mutable: Mapping[str, Any], name: str) -> np.ndarray:
    value = _mapping_value(mutable.get("intermediates"), name)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise FilmSmokeError(f"intermediate {name!r} is ambiguous")
        value = value[0]
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise FilmSmokeError(f"intermediate {name!r} is invalid")
    return array


def _mapping_value(mapping: Any, name: str) -> Any:
    if not isinstance(mapping, Mapping) or name not in mapping:
        raise FilmSmokeError(f"FiLM smoke did not expose intermediate {name!r}")
    return mapping[name]


def _run(args: argparse.Namespace, output_json: Path, output_md: Path) -> dict[str, Any]:
    if args.prediction_batch_size < 1:
        raise FilmSmokeError("--prediction-batch-size must be >= 1")
    validate_global_context_schema()
    run_config = _load_json(args.run_config)
    checkpoint_payload = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, checkpoint_payload, args.expected_epoch)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise FilmSmokeError("frozen checkpoint lacks train_only_normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    train_ids = list(split_ids.get("train") or ())
    valid_ids = list(split_ids.get("valid_iid") or ())
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise FilmSmokeError(f"expected clean train=672 valid_iid=128, got {len(train_ids)}/{len(valid_ids)}")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    train_rows, train_log_scales = _context_rows(train_examples, checkpoint_stats)
    valid_rows, valid_log_scales = _context_rows(valid_examples, checkpoint_stats)
    standardizer = fit_train_only_standardizer(
        [train_rows[sample_id] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    if standardizer.get("fit_sample_count") != 672 or standardizer.get("fit_population") != "train_only":
        raise FilmSmokeError("global context standardizer was not fit exactly on train=672")
    gate1 = _load_gate1_operator(args.gate1_table)
    scale_errors = []
    for sample_id, log_scale in {**train_log_scales, **valid_log_scales}.items():
        if sample_id not in gate1:
            raise FilmSmokeError(f"Gate-1 table lacks context sample {sample_id}")
        scale_errors.append(abs(math.exp(log_scale) - gate1[sample_id]))
    gate1_error = max(scale_errors) if scale_errors else math.inf
    if gate1_error > 1.0e-8:
        raise FilmSmokeError(f"global-context physics-scale mismatch versus Gate-1: {gate1_error:g} K")

    baseline_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint_payload.get("model_config") or {}), stats
    )
    _validate_model_config(baseline_config)
    baseline_config["global_context_mode"] = "none"
    baseline_config["global_context_feature_dim"] = 0
    baseline_config["global_context_feature_names"] = ()
    film_config = dict(baseline_config)
    film_config.update(
        {
            "global_context_mode": "film",
            "global_context_feature_dim": len(GLOBAL_CONTEXT_FEATURES),
            "global_context_feature_names": tuple(GLOBAL_CONTEXT_FEATURES),
            "film_target": "rnodes_processed",
            "film_init": "identity",
            "film_hidden_size": 64,
        }
    )
    builder = Heat3DGraphBuilder(**dict(run_config.get("graph_config") or {}))
    graph_seed = int(run_config.get("graph_seed", 0))
    valid_groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "v5_global_film_valid",
        False,
        "basic",
        graph_seed,
        batch_size=args.prediction_batch_size,
        drop_last=False,
    )
    one_groups = _make_groups_with_progress(
        valid_examples[:1],
        stats,
        builder,
        "v5_global_film_batch1",
        False,
        "basic",
        graph_seed,
        batch_size=1,
        drop_last=False,
    )
    if not valid_groups or not one_groups:
        raise FilmSmokeError("failed to build smoke graph batches")

    baseline_model = GraphNeuralOperator(**baseline_config)
    film_model = GraphNeuralOperator(**film_config)
    initial_context = _group_context(valid_groups[0], valid_rows, standardizer)
    initial_params = film_model.init(
        jax.random.PRNGKey(20260712),
        inputs=valid_groups[0]["inputs"],
        graphs=valid_groups[0]["graphs"],
        global_context=initial_context,
    )["params"]
    loaded_params, load_info = _apply_checkpoint_params(
        initial_params,
        checkpoint_payload["params"],
        strict=False,
        partial_load_policy="matching",
    )
    load_assertions = _checkpoint_load_assertions(initial_params, loaded_params, checkpoint_payload["params"])
    baseline_params = _device_params(checkpoint_payload["params"])
    frozen_archive = _load_frozen_archive(args.frozen_valid_predictions, valid_ids)
    target_delta_std_K = float(np.asarray(stats["target_delta_std"], dtype=np.float64).reshape(-1)[0])
    if not math.isfinite(target_delta_std_K) or target_delta_std_K <= 0.0:
        raise FilmSmokeError("checkpoint target_delta_std must be finite and positive")

    _film_probe, film_probe_mutable = film_model.apply(
        {"params": loaded_params},
        inputs=valid_groups[0]["inputs"],
        graphs=valid_groups[0]["graphs"],
        global_context=initial_context,
        mutable=["intermediates"],
    )
    gamma_probe = _intermediate_array(film_probe_mutable, "global_film_gamma")
    beta_probe = _intermediate_array(film_probe_mutable, "global_film_beta")
    pre_probe = _intermediate_array(film_probe_mutable, "rnodes_processed_pre_film")
    post_probe = _intermediate_array(film_probe_mutable, "rnodes_processed")
    gamma_abs_max = float(np.max(np.abs(gamma_probe)))
    beta_abs_max = float(np.max(np.abs(beta_probe)))
    latent_identity_error = float(np.max(np.abs(post_probe - pre_probe)))
    if gamma_abs_max != 0.0 or beta_abs_max != 0.0 or latent_identity_error != 0.0:
        raise FilmSmokeError(
            "identity FiLM must have exact zero gamma/beta and unchanged processed rnodes; "
            f"gamma={gamma_abs_max:g} beta={beta_abs_max:g} latent={latent_identity_error:g}"
        )

    identity_error = 0.0
    archive_error = 0.0
    topology_hashes: list[str] = []
    for group in valid_groups:
        context = _group_context(group, valid_rows, standardizer)
        baseline = baseline_model.apply(
            {"params": baseline_params}, inputs=group["inputs"], graphs=group["graphs"]
        )
        film = film_model.apply(
            {"params": loaded_params},
            inputs=group["inputs"],
            graphs=group["graphs"],
            global_context=context,
        )
        baseline_np = np.asarray(baseline, dtype=np.float64)
        film_np = np.asarray(film, dtype=np.float64)
        if baseline_np.shape != film_np.shape or not np.all(np.isfinite(film_np)):
            raise FilmSmokeError(f"{group['name']}: invalid FiLM prediction shape or values")
        identity_error = max(identity_error, float(np.max(np.abs(film_np - baseline_np))))
        recovered = np.asarray(
            recover_temperature_from_normalized_delta(baseline, group["t_ref"], stats), dtype=np.float64
        )
        for index, sample_id in enumerate(group["sample_ids"]):
            archive_error = max(
                archive_error,
                float(np.max(np.abs(recovered[index, 0, :, :].reshape(-1) - frozen_archive[sample_id]))),
            )
        topology_hashes.append(_tree_hash(group["graphs"]))
    identity_error_K = identity_error * target_delta_std_K
    if identity_error_K > REPLAY_TOLERANCE_K:
        raise FilmSmokeError(
            "identity-initialized FiLM changed V4 prediction beyond frozen replay tolerance: "
            f"{identity_error_K:g} K"
        )
    if archive_error > REPLAY_TOLERANCE_K:
        raise FilmSmokeError(f"FiLM-disabled V4 replay drifted from frozen valid archive by {archive_error:g} K")

    batch1 = one_groups[0]
    context1 = _group_context(batch1, valid_rows, standardizer)
    baseline1 = baseline_model.apply(
        {"params": baseline_params}, inputs=batch1["inputs"], graphs=batch1["graphs"]
    )
    film1 = film_model.apply(
        {"params": loaded_params},
        inputs=batch1["inputs"],
        graphs=batch1["graphs"],
        global_context=context1,
    )
    batch1_error = float(np.max(np.abs(np.asarray(film1) - np.asarray(baseline1))))
    batch1_error_K = batch1_error * target_delta_std_K
    if batch1_error_K > REPLAY_TOLERANCE_K:
        raise FilmSmokeError(
            "batch-size-one identity FiLM changed V4 prediction beyond frozen replay tolerance: "
            f"{batch1_error_K:g} K"
        )

    jit_group = valid_groups[0]
    jit_context = _group_context(jit_group, valid_rows, standardizer)

    @jax.jit
    def _jit_apply(context):
        return film_model.apply(
            {"params": loaded_params},
            inputs=jit_group["inputs"],
            graphs=jit_group["graphs"],
            global_context=context,
        )

    jit_prediction = _jit_apply(jit_context)
    eager_prediction = film_model.apply(
        {"params": loaded_params},
        inputs=jit_group["inputs"],
        graphs=jit_group["graphs"],
        global_context=jit_context,
    )
    jit_error = float(np.max(np.abs(np.asarray(jit_prediction) - np.asarray(eager_prediction))))
    jit_error_K = jit_error * target_delta_std_K
    if jit_error_K > JIT_EAGER_TOLERANCE_K:
        raise FilmSmokeError(f"JIT FiLM prediction differs from eager by {jit_error_K:g} K")

    def _loss_for_grad(params):
        prediction = film_model.apply(
            {"params": params},
            inputs=jit_group["inputs"],
            graphs=jit_group["graphs"],
            global_context=jit_context,
        )
        return jnp.mean(jnp.square(prediction))

    gradient_loss, gradients = jax.value_and_grad(_loss_for_grad)(loaded_params)
    if not math.isfinite(float(gradient_loss)) or not _tree_all_finite(gradients):
        raise FilmSmokeError("FiLM gradient smoke produced a non-finite value")

    payload = {
        "smoke_id": SMOKE_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_checkpoint_identity_smoke",
        "baseline": {
            "config_id": EXPECTED_BASELINE_ID,
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "run_config": args.run_config.as_posix(),
            "run_config_sha256": _sha256(args.run_config),
            "valid_archive": args.frozen_valid_predictions.as_posix(),
            "valid_archive_sha256": _sha256(args.frozen_valid_predictions),
        },
        "dataset": {
            "subset": sample_root.as_posix(),
            "split_map": split_map.as_posix(),
            "split_source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "hard_roles_used": [],
        },
        "global_context": {
            "feature_schema": list(GLOBAL_CONTEXT_FEATURES),
            "feature_provenance": FEATURE_PROVENANCE,
            "target_or_label_derived_inputs": False,
            "standardizer": standardizer,
            "gate1_operator_table": args.gate1_table.as_posix(),
            "gate1_operator_table_sha256": _sha256(args.gate1_table),
            "gate1_operator_max_abs_error_K": gate1_error,
        },
        "film": {
            "global_context_mode": "film",
            "film_target": "rnodes_processed",
            "film_init": "identity",
            "film_hidden_size": 64,
            "latent_dimension_preserved": int(film_config["node_latent_size"]),
            "checkpoint_partial_load": load_info,
            "checkpoint_load_assertions": load_assertions,
        },
        "checks": {
            "film_disabled_v4_archive_replay_max_abs_temperature_error_K": archive_error,
            "film_disabled_v4_archive_replay_pass": bool(archive_error <= REPLAY_TOLERANCE_K),
            "identity_initialization_max_abs_normalized_error_batch_ge_1": identity_error,
            "identity_initialization_max_abs_normalized_error_batch_1": batch1_error,
            "identity_initialization_max_abs_raw_deltaT_error_K_batch_ge_1": identity_error_K,
            "identity_initialization_max_abs_raw_deltaT_error_K_batch_1": batch1_error_K,
            "identity_replay_tolerance_K": REPLAY_TOLERANCE_K,
            "identity_initialization_pass": bool(
                identity_error_K <= REPLAY_TOLERANCE_K and batch1_error_K <= REPLAY_TOLERANCE_K
            ),
            "jit_max_abs_normalized_error": jit_error,
            "jit_max_abs_raw_deltaT_error_K": jit_error_K,
            "jit_eager_tolerance_K": JIT_EAGER_TOLERANCE_K,
            "jit_pass": bool(jit_error_K <= JIT_EAGER_TOLERANCE_K),
            "identity_film_gamma_abs_max": gamma_abs_max,
            "identity_film_beta_abs_max": beta_abs_max,
            "identity_processed_rnodes_max_abs_error": latent_identity_error,
            "gradient_loss": float(gradient_loss),
            "gradient_finite": True,
            "graph_topology_change": False,
            "graph_topology_hashes": topology_hashes,
            "training_runs": 0,
            "checkpoint_writes": 0,
            "data_writes": 0,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _render_markdown(payload: Mapping[str, Any]) -> str:
    checks = payload["checks"]
    context = payload["global_context"]
    film = payload["film"]
    lines = [
        "# V5 Global FiLM Architecture Smoke",
        "",
        "## Result",
        "",
        f"- Frozen baseline: `{payload['baseline']['config_id']}` epoch `{payload['baseline']['checkpoint_epoch']}`.",
        "- Mode: read-only checkpoint smoke; no training, checkpoint, data, or graph-topology write occurred.",
        f"- V4 disabled-FiLM valid archive replay max error: `{checks['film_disabled_v4_archive_replay_max_abs_temperature_error_K']:.6g} K`.",
        f"- Identity FiLM max raw-DeltaT replay error (batched): `{checks['identity_initialization_max_abs_raw_deltaT_error_K_batch_ge_1']:.6g} K`; batch-1: `{checks['identity_initialization_max_abs_raw_deltaT_error_K_batch_1']:.6g} K` (tolerance `{checks['identity_replay_tolerance_K']:.3g} K`).",
        f"- JIT/eager max raw-DeltaT difference: `{checks['jit_max_abs_raw_deltaT_error_K']:.6g} K` (tolerance `{checks['jit_eager_tolerance_K']:.3g} K`); finite gradient: `{checks['gradient_finite']}`.",
        f"- Gate-1 physics-scale crosscheck max error: `{context['gate1_operator_max_abs_error_K']:.6g} K`.",
        "",
        "## Global Context Contract",
        "",
        "- Context uses only inference-time `coords`, `k`, `q`, BC fields, and control-volume weights; it has no target, residual, prediction, or oracle input.",
        f"- The standardizer is fit only on `train={context['standardizer']['fit_sample_count']}` and records the frozen feature order/hash.",
        f"- FiLM target: `{film['film_target']}`; latent dimension remains `{film['latent_dimension_preserved']}`.",
        "- The gamma/beta output layer is zero initialized, so `h' = (1 + gamma) * h + beta` leaves the processed-rnode latent exactly unchanged at initialization. GPU sparse replay is compared in raw K using the frozen V4 archive tolerance.",
        "",
        "## Feature Schema",
        "",
        "| index | feature | provenance |",
        "| ---: | --- | --- |",
    ]
    for index, name in enumerate(context["feature_schema"]):
        lines.append(f"| {index} | `{name}` | {context['feature_provenance'][name]} |")
    lines.extend(
        [
            "",
            "The JSON contains the exact train-only standardizer and checkpoint partial-load evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    run_config = _load_json(args.run_config)
    payload = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, payload, args.expected_epoch)
    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    return {
        "smoke_id": SMOKE_ID,
        "mode": "dry_run",
        "checkpoint_epoch": int(payload["epoch"]),
        "split_source": split_source,
        "roles": {"train": len(split_ids.get("train", [])), "valid_iid": len(split_ids.get("valid_iid", []))},
        "global_context_feature_count": len(GLOBAL_CONTEXT_FEATURES),
        "training_runs": 0,
        "planned_writes": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.run_config.is_file():
            raise FilmSmokeError(f"run config does not exist: {args.run_config}")
        if args.dry_run:
            print(json.dumps(_dry_run(args), indent=2, sort_keys=True))
            return 0
        if args.output_json is None or args.output_md is None:
            raise FilmSmokeError("smoke requires --output-json and --output-md")
        output_json = _ensure_output(args.output_json, "output-json", args.overwrite)
        output_md = _ensure_output(args.output_md, "output-md", args.overwrite)
        if output_json == output_md:
            raise FilmSmokeError("output JSON and Markdown paths must differ")
        payload = _run(args, output_json, output_md)
    except (FilmSmokeError, ValueError, OSError) as exc:
        print(f"global FiLM smoke error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "smoke_id": payload["smoke_id"],
                "identity_error": payload["checks"]["identity_initialization_max_abs_normalized_error_batch_ge_1"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
