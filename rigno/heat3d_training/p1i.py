"""Library-level V6/P1i Full training preparation and semantics.

This module is the publication-training boundary for V7.  It deliberately
contains no imports from ``scripts`` and does not materialize test or sealed
samples.  The numerical contracts are copied from the frozen V6/P1i Full
configuration and expressed through reusable library APIs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_normalization import (
    legacy_train_only_stats,
    normalize_condition,
    normalize_coords,
    normalized_delta_to_raw,
    normalize_target_delta,
)
from rigno.heat3d_v1_training_semantics import (
    COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    build_legacy_zero_delta_bridge,
)
from rigno.heat3d_v5_scale_context import (
    regional_source_volume_weights_from_raw,
)
from rigno.heat3d_v5_scale_pooling import (
    qk_region_feature_names,
    qk_region_features_from_raw,
)
from rigno.heat3d_v5_shape_scale import (
    apply_scale_head_lr_multiplier,
    mask_native_trainable_scope,
    native_shape_scale_losses,
)
from rigno.heat3d_v6_dataset import (
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
    V6DualRobinExample,
)
from rigno.heat3d_v6_global_context import (
    GLOBAL_CONTEXT_FEATURES_V6,
    fit_train_only_v6_standardizer,
    global_context_from_v6_inputs,
    standardize_v6_contexts,
    validate_v6_global_context_schema,
)
from rigno.models.operator import Inputs


P1I_DATASET_ID = CONTINUOUS_PHYSICS_V6_DATASET_ID
P1I_SPLIT_COUNTS = {"train": 768, "valid_iid": 128}
P1I_BATCH_CONTRACT = {
    "batch_size": 24,
    "micro_batch_size": 24,
    "validation_batch_size": 32,
    "prediction_batch_size": 32,
    "batch_plan": "sample_shuffle",
    "shuffle_train_batches": True,
    "batch_build_seed": 0,
    "sample_weight_policy": "none",
    "drop_last": False,
}


@dataclass(frozen=True)
class P1IPreparedData:
    """All train/valid arrays prepared from the registered P1i split only."""

    train_examples: tuple[V6DualRobinExample, ...]
    valid_examples: tuple[V6DualRobinExample, ...]
    stats: dict[str, Any]
    builder: Any
    train_batches: tuple[Any, ...]
    valid_batches: tuple[Any, ...]
    context_standardizer: dict[str, Any]
    train_only_loss_references: dict[str, Any]
    preparation_profile: dict[str, Any]


def _sample_root(subset: str | Path) -> Path:
    root = Path(subset).resolve()
    return root / "samples" if (root / "samples").is_dir() else root


def _canonical_role(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    role = str(row.get("split_role") or "")
    if manifest.get("dataset_id") != P1I_DATASET_ID:
        raise ValueError("P1i loader received a non-P1i dataset")
    if role not in {"train", "valid_iid", "test_iid"}:
        raise ValueError(f"unsupported P1i split_role={role!r}")
    return role


def load_selected_p1i_examples(
    subset: str | Path,
    manifest_path: str | Path,
) -> dict[str, list[V6DualRobinExample]]:
    """Load exactly train and valid_iid rows without materializing test rows."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("samples")
    if manifest.get("dataset_id") != P1I_DATASET_ID:
        raise ValueError("registered P1i dataset id is required")
    if not isinstance(rows, list) or len(rows) != 1024:
        raise ValueError("P1i manifest must contain 1024 rows")
    # Validate role counts without retaining or resolving test/sealed IDs.
    counts = {"train": 0, "valid_iid": 0, "test_iid": 0}
    selected_rows: dict[str, list[Mapping[str, Any]]] = {"train": [], "valid_iid": []}
    for row in rows:
        role = _canonical_role(manifest, row)
        counts[role] += 1
        if role in selected_rows:
            selected_rows[role].append(row)
    if counts != {"train": 768, "valid_iid": 128, "test_iid": 128}:
        raise ValueError(f"P1i split counts drifted: {counts}")

    loader = object.__new__(Heat3DV6DualRobinDataset)
    loader.datadir = _sample_root(subset)
    loader.manifest_path = Path(manifest_path).resolve()
    loader.manifest = manifest
    result: dict[str, list[V6DualRobinExample]] = {"train": [], "valid_iid": []}
    for role, selected in selected_rows.items():
        for row in selected:
            # _load_sample reads only the explicitly selected sample directory.
            result[role].append(loader._load_sample(dict(row)))
    if {name: len(value) for name, value in result.items()} != P1I_SPLIT_COUNTS:
        raise AssertionError("selected P1i split counts are not frozen")
    return result


def _metadata_shape_signature(metadata: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value.shape)
        for value in tree.tree_leaves(metadata)
        if hasattr(value, "shape")
    )


def _metadata_key(graph_seed: int):
    return jax.random.PRNGKey(int(graph_seed))


def _pad_metadata(metadata_list: Sequence[Any]) -> tuple[Any, bool]:
    """Apply the frozen per-batch dummy-edge padding envelope."""

    if not metadata_list:
        raise ValueError("cannot pad an empty metadata list")
    same_coords = all(
        np.array_equal(np.asarray(metadata_list[0].x_pnodes_inp), np.asarray(value.x_pnodes_inp))
        for value in metadata_list[1:]
    )
    if same_coords:
        return (
            tree.tree_map(
                lambda value: jnp.repeat(value, repeats=len(metadata_list), axis=0),
                metadata_list[0],
            ),
            True,
        )

    edge_fields = (
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
        "r2p_edge_indices",
    )
    targets: dict[str, int | None] = {}
    for field in edge_fields:
        values = [getattr(metadata, field) for metadata in metadata_list]
        if all(value is None for value in values):
            targets[field] = None
        elif any(value is None for value in values):
            raise ValueError(f"mixed None/non-None graph metadata for {field}")
        else:
            targets[field] = max(int(value.shape[1]) for value in values)

    padded = []
    for metadata in metadata_list:
        replacements: dict[str, Any] = {}
        for field, target in targets.items():
            value = getattr(metadata, field)
            if target is None:
                replacements[field] = None
                continue
            pad_count = int(target) - int(value.shape[1])
            if pad_count < 0:
                raise AssertionError(f"negative padding for {field}")
            replacements[field] = (
                value
                if pad_count == 0
                else jnp.concatenate(
                    [value, jnp.repeat(value[:, -1:, :], pad_count, axis=1)],
                    axis=1,
                )
            )
        padded.append(
            type(metadata)(
                **{
                    field: replacements.get(field, getattr(metadata, field))
                    for field in metadata._fields
                }
            )
        )
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *padded), False


def _graph_coords(example: V6DualRobinExample, stats: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(example.condition.coords, dtype=np.float64).reshape(1, 1, -1, 3)
    return np.asarray(normalize_coords(raw, dict(stats))).reshape(-1, 3)


def _make_group(
    name: str,
    examples: Sequence[V6DualRobinExample],
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    *,
    graph_seed: int,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError(f"{name}: empty batch")
    feature_started = time.perf_counter()
    bridges = [build_legacy_zero_delta_bridge(example) for example in examples]
    feature_names = bridges[0].condition_feature_names
    if any(bridge.condition_feature_names != feature_names for bridge in bridges[1:]):
        raise ValueError(f"{name}: feature schema mismatch")
    raw_u = jnp.concatenate([bridge.legacy_inputs.u for bridge in bridges], axis=0)
    raw_c = jnp.concatenate([bridge.legacy_inputs.c for bridge in bridges], axis=0)
    raw_coords = jnp.concatenate([bridge.legacy_inputs.x_inp for bridge in bridges], axis=0)
    target_delta = jnp.concatenate([bridge.target_delta_u for bridge in bridges], axis=0)
    t_ref = jnp.concatenate([bridge.t_ref for bridge in bridges], axis=0)
    normalized_coords = normalize_coords(raw_coords, dict(stats))
    inputs = Inputs(
        u=raw_u,
        c=normalize_condition(raw_c, dict(stats)),
        x_inp=normalized_coords,
        x_out=normalized_coords,
        t=None,
        tau=None,
    )
    feature_seconds = time.perf_counter() - feature_started
    if profile is not None:
        profile["feature_preprocessing_seconds"] = float(
            profile.get("feature_preprocessing_seconds", 0.0)
        ) + feature_seconds
        profile["feature_transform_calls"] = int(
            profile.get("feature_transform_calls", 0)
        ) + len(examples)
    graph_started = time.perf_counter()
    metadata_list = [
        builder.build_metadata(_graph_coords(example, stats), key=_metadata_key(graph_seed))
        for example in examples
    ]
    metadata, shared = _pad_metadata(metadata_list)
    graphs = builder.build_graphs(metadata)
    graph_seconds = time.perf_counter() - graph_started
    if profile is not None:
        profile["graph_preparation_seconds"] = float(
            profile.get("graph_preparation_seconds", 0.0)
        ) + graph_seconds
        profile["graph_metadata_build_calls"] = int(
            profile.get("graph_metadata_build_calls", 0)
        ) + len(examples)
        profile["graph_build_calls"] = int(profile.get("graph_build_calls", 0)) + 1
    return {
        "name": name,
        "sample_ids": tuple(str(example.sample_id) for example in examples),
        "split": str(examples[0].meta.get("v6_adapter", {}).get("manifest_split_role")),
        "inputs": inputs,
        "target_normalized": normalize_target_delta(target_delta, dict(stats)),
        "target_delta_raw": target_delta,
        "target_temperature": t_ref + target_delta,
        "t_ref": t_ref,
        "graphs": graphs,
        "metadata": metadata,
        "shared_metadata": shared,
        "feature_names": tuple(feature_names),
    }


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def build_p1i_batches(
    examples: Sequence[V6DualRobinExample],
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    *,
    label: str,
    batch_size: int,
    graph_seed: int = 0,
    batch_build_seed: int | None = None,
    profile: dict[str, Any] | None = None,
) -> list[Any]:
    """Build fixed B24/B32 batches once, preserving sample_shuffle semantics."""

    from .core import TrainingBatch

    ordered = list(examples)
    if batch_build_seed is not None:
        rng = np.random.default_rng(int(batch_build_seed))
        ordered = [ordered[int(index)] for index in rng.permutation(len(ordered))]
    start = time.perf_counter()
    result = []
    for index, batch_examples in enumerate(_chunks(ordered, int(batch_size)), start=1):
        group_start = time.perf_counter()
        group = _make_group(
            f"{label}_sample_shuffle_batch_{index:04d}_B{len(batch_examples)}",
            batch_examples,
            stats,
            builder,
            graph_seed=graph_seed,
            profile=profile,
        )
        result.append(
            TrainingBatch(
                batch_id=f"{label}_batch_{index:04d}",
                sample_ids=tuple(group["sample_ids"]),
                groups=(group,),
            )
        )
        if profile is not None:
            profile["batch_count"] = int(profile.get("batch_count", 0)) + 1
            profile["batch_build_seconds"] = float(profile.get("batch_build_seconds", 0.0)) + (
                time.perf_counter() - group_start
            )
    if not result:
        raise ValueError(f"{label}: no batches were built")
    if profile is not None:
        profile["total_batch_build_seconds"] = float(time.perf_counter() - start)
        profile["sample_count"] = len(ordered)
        profile["unique_batch_sizes"] = sorted({len(batch.sample_ids) for batch in result})
    return result


def _context_row(example: V6DualRobinExample) -> dict[str, float]:
    return global_context_from_v6_inputs(**example.v6_global_context_inputs())


def attach_input_contexts(
    batches: Sequence[Any],
    train_examples: Sequence[V6DualRobinExample],
    required_examples: Sequence[V6DualRobinExample],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach train-only standardized 24-D context and return its provenance."""

    names = tuple(model_config.get("global_context_feature_names") or ())
    validate_v6_global_context_schema(names)
    rows = {str(example.sample_id): _context_row(example) for example in [*train_examples, *required_examples]}
    train_ids = [str(example.sample_id) for example in train_examples]
    standardizer = fit_train_only_v6_standardizer(
        [rows[sample_id] for sample_id in train_ids], fit_sample_ids=train_ids
    )
    encoded = {
        sample_id: standardize_v6_contexts([row], standardizer)[0]
        for sample_id, row in rows.items()
    }
    for batch in batches:
        context = np.stack([encoded[sample_id] for sample_id in batch.sample_ids])
        if context.shape != (len(batch.sample_ids), 24):
            raise ValueError(f"{batch.batch_id}: invalid global context shape {context.shape}")
        batch.groups[0]["global_context"] = jnp.asarray(context, dtype=jnp.float32)
    return {
        "schema": "GLOBAL_CONTEXT_FEATURES_V6",
        "feature_names": list(GLOBAL_CONTEXT_FEATURES_V6),
        "standardizer": standardizer,
        "raw_context_by_id": rows,
        "fit_role": "train_only",
        "target_or_label_derived_inputs": False,
    }


def attach_native_physics(
    batches: Sequence[Any],
    examples_by_id: Mapping[str, V6DualRobinExample],
    *,
    context_by_id: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """Attach native shape--scale inputs derived only from raw V6 inputs."""

    for batch in batches:
        volumes = []
        log_s_phys = []
        references = []
        masks = []
        prescribed = []
        for sample_id in batch.sample_ids:
            example = examples_by_id[sample_id]
            relative = example.get_relative_bc_feature_view()
            context = (
                context_by_id[example.sample_id]
                if context_by_id is not None
                else _context_row(example)
            )
            n_points = int(example.condition.coords.shape[0])
            volumes.append(example.v6_operator_point_weights())
            log_s_phys.append(float(context["log_s_phys_K"]))
            references.append(np.full(n_points, float(relative.t_ref_value), dtype=np.float32))
            masks.append(np.zeros(n_points, dtype=np.float32))
            prescribed.append(np.full(n_points, float(relative.t_ref_value), dtype=np.float32))
        batch.groups[0]["native_physics"] = {
            "control_volumes": jnp.asarray(np.stack(volumes), dtype=jnp.float32),
            "log_s_phys": jnp.asarray(log_s_phys, dtype=jnp.float32),
            "reference_temperature": jnp.asarray(np.stack(references), dtype=jnp.float32),
            "dirichlet_mask": jnp.asarray(np.stack(masks), dtype=jnp.float32),
            "prescribed_temperature": jnp.asarray(np.stack(prescribed), dtype=jnp.float32),
        }


def attach_qk_features(
    batches: Sequence[Any],
    examples_by_id: Mapping[str, V6DualRobinExample],
    *,
    feature_version: str = "sparse_safe_v2",
) -> None:
    feature_names = qk_region_feature_names(feature_version)
    for batch in batches:
        group = batch.groups[0]
        metadata = group["metadata"]
        p2r = np.asarray(metadata.p2r_edge_indices)
        rnode_count = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
        rows = []
        for row, sample_id in enumerate(batch.sample_ids):
            example = examples_by_id[sample_id]
            relative = example.get_relative_bc_feature_view()
            rows.append(
                qk_region_features_from_raw(
                    coords=np.asarray(example.condition.coords, dtype=np.float64),
                    raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
                    condition_feature_names=relative.condition_feature_names,
                    p2r_edge_indices=p2r[row],
                    rnode_count=rnode_count,
                    feature_version=feature_version,
                )
            )
        packed = np.stack(rows)
        expected = (len(batch.sample_ids), rnode_count, len(feature_names))
        if packed.shape != expected or not np.all(np.isfinite(packed)):
            raise ValueError(f"{batch.batch_id}: qk feature shape drifted: {packed.shape} != {expected}")
        group["qk_region_features"] = jnp.asarray(packed, dtype=jnp.float32)
        group["qk_region_feature_names"] = feature_names


def fit_native_loss_references(
    train_examples: Sequence[V6DualRobinExample],
    loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit exactly the frozen V6 train-only raw/scale references."""

    energy = 0.0
    count = 0
    squared_scales = []
    for example in train_examples:
        bridge = build_legacy_zero_delta_bridge(example)
        target = np.asarray(bridge.target_delta_u, dtype=np.float64).reshape(-1)
        energy += float(np.sum(np.square(target)))
        count += int(target.size)
        squared_scales.append(
            float(np.sum(np.square(target) * example.v6_operator_point_weights()))
        )
    raw_reference = energy / max(count, 1)
    scale_reference = float(np.mean(squared_scales))
    if raw_reference <= 0.0 or scale_reference <= 0.0:
        raise ValueError("invalid train-only native loss reference")
    clip_min = float(loss_config.get("native_log_scale_weight_clip_min", 0.25))
    clip_max = float(loss_config.get("native_log_scale_weight_clip_max", 4.0))
    raw_weights = np.asarray(squared_scales) / scale_reference
    clipped = np.clip(raw_weights, clip_min, clip_max)
    return {
        "native_raw_train_target_energy_per_point": raw_reference,
        "native_log_scale_train_true_scale_sq_mean": scale_reference,
        "native_log_scale_weight_diagnostics": {
            "fit_roles": ["train"],
            "sample_count": len(squared_scales),
            "clip_bounds": [clip_min, clip_max],
            "raw_weight_mean": float(np.mean(raw_weights)),
            "clipped_weight_mean": float(np.mean(clipped)),
            "effective_sample_size": float(
                np.square(np.sum(clipped)) / max(float(np.sum(np.square(clipped))), 1.0e-12)
            ),
        },
    }


def model_apply_full(model: Any, params: Any, batch: Any, rng: Any = None) -> tuple[Any, ...]:
    predictions = []
    for index, group in enumerate(batch.groups):
        physics = group["native_physics"]
        key = None if rng is None else jax.random.fold_in(rng, index)
        predictions.append(
            model.apply(
                {"params": params},
                inputs=group["inputs"],
                graphs=group["graphs"],
                global_context=group["global_context"],
                control_volumes=physics["control_volumes"],
                log_s_phys=physics["log_s_phys"],
                reference_temperature=physics["reference_temperature"],
                dirichlet_mask=physics["dirichlet_mask"],
                prescribed_temperature=physics["prescribed_temperature"],
                qk_region_features=group["qk_region_features"],
                scale_context=group.get("scale_context"),
                scale_region_source_weights=group.get("scale_region_source_weights"),
                scale_region_volume_weights=group.get("scale_region_volume_weights"),
                key=key,
                method=model.predict_native_shape_scale,
            )
        )
    return tuple(predictions)


def model_init_full(model: Any, key: Any, batch: Any) -> Any:
    group = batch.groups[0]
    physics = group["native_physics"]
    return model.init(
        key,
        inputs=group["inputs"],
        graphs=group["graphs"],
        global_context=group["global_context"],
        control_volumes=physics["control_volumes"],
        log_s_phys=physics["log_s_phys"],
        reference_temperature=physics["reference_temperature"],
        dirichlet_mask=physics["dirichlet_mask"],
        prescribed_temperature=physics["prescribed_temperature"],
        qk_region_features=group["qk_region_features"],
        scale_context=group.get("scale_context"),
        scale_region_source_weights=group.get("scale_region_source_weights"),
        scale_region_volume_weights=group.get("scale_region_volume_weights"),
        method=model.predict_native_shape_scale,
    )


def model_apply_vanilla(model: Any, params: Any, batch: Any, rng: Any = None) -> tuple[Any, ...]:
    """Apply the capacity-matched RIGNO control without Full-only heads.

    The control uses the same prepared graph, normalized operator inputs and
    batching contract as Full, but calls the ordinary RIGNO operator.  It
    therefore has no global FiLM context, native scale head, q/k regional
    feature input, or decoder bypass.  The returned value is the legacy
    normalized-DeltaT field and is converted to raw DeltaT only by the
    explicit evaluation adapter.
    """

    predictions = []
    for index, group in enumerate(batch.groups):
        key = None if rng is None else jax.random.fold_in(rng, index)
        predictions.append(
            model.apply(
                {"params": params},
                inputs=group["inputs"],
                graphs=group["graphs"],
                key=key,
                method=model.call,
            )
        )
    return tuple(predictions)


def model_init_vanilla(model: Any, key: Any, batch: Any) -> Any:
    """Initialize the ordinary normalized-DeltaT RIGNO control."""

    group = batch.groups[0]
    return model.init(
        key,
        inputs=group["inputs"],
        graphs=group["graphs"],
        key=None,
        method=model.call,
    )


def loss_fn_vanilla(
    predictions: Sequence[Any],
    batch: Any,
    _loss_config: Mapping[str, Any] | None = None,
) -> Any:
    """Frozen base normalized-DeltaT MSE for the vanilla RIGNO control."""

    del _loss_config
    total = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for prediction, group in zip(predictions, batch.groups, strict=True):
        target = group["target_normalized"]
        total = total + jnp.sum(jnp.square(prediction - target))
        count += int(np.prod(target.shape))
    return total / max(count, 1)


def prediction_to_raw_delta(
    prediction: Any,
    *,
    variant: str,
    stats: Mapping[str, Any],
) -> np.ndarray:
    """Convert one model output to raw DeltaT with an explicit variant rule."""

    if variant == "Full":
        if not isinstance(prediction, Mapping) or "deltaT_hat" not in prediction:
            raise ValueError("Full prediction must contain native deltaT_hat")
        return np.asarray(prediction["deltaT_hat"], dtype=np.float64)
    if variant == "vanilla_RIGNO":
        return np.asarray(
            normalized_delta_to_raw(prediction, dict(stats)), dtype=np.float64
        )
    raise ValueError(f"unsupported V7 training prediction variant {variant!r}")


def loss_fn_full(
    predictions: Sequence[Mapping[str, Any]],
    batch: Any,
    loss_config: Mapping[str, Any],
) -> Any:
    weights = {
        "shape_cv": float(loss_config["native_shape_cv_weight"]),
        "log_scale": float(loss_config["native_log_scale_weight"]),
        "relative_field": float(loss_config["native_relative_field_weight"]),
        "raw_absolute": float(loss_config["native_raw_field_weight"]),
    }
    total = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for prediction, group in zip(predictions, batch.groups, strict=True):
        physics = group["native_physics"]
        components = native_shape_scale_losses(
            prediction,
            target_deltaT=group["target_delta_raw"],
            control_volumes=physics["control_volumes"],
            dirichlet_mask=physics["dirichlet_mask"],
            loss_weights=weights,
            raw_loss_mode=str(loss_config["native_raw_loss_mode"]),
            raw_train_target_energy_per_point=float(
                loss_config["native_raw_train_target_energy_per_point"]
            ),
            log_scale_weight_mode=str(loss_config["native_log_scale_weight_mode"]),
            log_scale_train_true_scale_sq_mean=float(
                loss_config["native_log_scale_train_true_scale_sq_mean"]
            ),
            log_scale_weight_clip=(
                float(loss_config["native_log_scale_weight_clip_min"]),
                float(loss_config["native_log_scale_weight_clip_max"]),
            ),
        )
        total = total + components["total_loss"] * int(group["target_delta_raw"].shape[0])
        count += int(group["target_delta_raw"].shape[0])
    return total / max(count, 1)


def make_gradient_transform(
    model_config: Mapping[str, Any], optimizer_config: Mapping[str, Any]
):
    branch_mode = str(model_config.get("native_branch_mode", "joint"))
    trainable_scope = str(optimizer_config.get("native_trainable_scope", "branch"))
    multiplier = float(optimizer_config.get("scale_head_lr_multiplier", 1.0))

    def transform(gradients: Any) -> Any:
        masked = mask_native_trainable_scope(
            gradients, branch_mode=branch_mode, trainable_scope=trainable_scope
        )
        return apply_scale_head_lr_multiplier(masked, multiplier)

    return transform


def _learning_rate_schedule(epochs: int, updates_per_epoch: int, config: Mapping[str, Any]):
    schedule = str(config.get("lr_schedule", "warmup_cosine"))
    base = float(config["lr"])
    if schedule == "constant":
        return base

    def schedule_fn(count: Any):
        update_count = jnp.asarray(count, dtype=jnp.float32)
        epoch = jnp.floor(update_count / float(max(updates_per_epoch, 1))) + 1.0
        base_value = jnp.asarray(base, dtype=jnp.float32)
        if schedule == "warmup_cosine":
            minimum = jnp.asarray(float(config["min_lr"]), dtype=jnp.float32)
            warmup = int(config["warmup_epochs"])
            if warmup > 0:
                warm_progress = jnp.clip(epoch / float(warmup), 0.0, 1.0)
                warm_lr = minimum + warm_progress * (base_value - minimum)
                decay_epochs = max(int(epochs) - warmup, 1)
                decay_progress = jnp.clip(
                    (epoch - float(warmup)) / float(decay_epochs), 0.0, 1.0
                )
                cosine_lr = minimum + 0.5 * (
                    1.0 + jnp.cos(jnp.pi * decay_progress)
                ) * (base_value - minimum)
                return jnp.where(epoch <= float(warmup), warm_lr, cosine_lr)
        raise ValueError(f"unsupported P1i learning-rate schedule: {schedule}")

    return schedule_fn


def make_p1i_optimizer(
    optimizer_config: Mapping[str, Any], *, epochs: int, updates_per_epoch: int
):
    name = str(optimizer_config.get("optimizer", "adamw"))
    if name not in {"adam", "adamw"}:
        raise ValueError("P1i Full frozen optimizer must be adam or adamw")
    transforms = []
    clip = optimizer_config.get("gradient_clip_norm")
    if clip is not None:
        transforms.append(optax.clip_by_global_norm(float(clip)))
    lr = _learning_rate_schedule(epochs, updates_per_epoch, optimizer_config)
    if name == "adam":
        if float(optimizer_config.get("weight_decay", 0.0)) > 0.0:
            transforms.append(optax.add_decayed_weights(float(optimizer_config["weight_decay"])))
        transforms.append(optax.adam(lr))
    else:
        transforms.append(
            optax.adamw(lr, weight_decay=float(optimizer_config.get("weight_decay", 0.0)))
        )
    return optax.chain(*transforms)


def prepare_p1i_data(
    subset: str | Path,
    manifest_path: str | Path,
    *,
    graph_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    batch_size: int = 24,
    validation_batch_size: int = 32,
    batch_build_seed: int = 0,
    graph_seed: int = 0,
    profile: dict[str, Any] | None = None,
) -> P1IPreparedData:
    """Prepare the registered Full train/valid route once for a run."""

    started = time.perf_counter()
    loaded = load_selected_p1i_examples(subset, manifest_path)
    train_examples = tuple(loaded["train"])
    valid_examples = tuple(loaded["valid_iid"])
    stats = legacy_train_only_stats(
        list(train_examples),
        coord_policy=COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    )
    builder = Heat3DGraphBuilder(**dict(graph_config))
    build_profile: dict[str, Any] = {}
    train_batches = build_p1i_batches(
        train_examples,
        stats,
        builder,
        label="train",
        batch_size=batch_size,
        graph_seed=graph_seed,
        batch_build_seed=batch_build_seed,
        profile=build_profile,
    )
    valid_batches = build_p1i_batches(
        valid_examples,
        stats,
        builder,
        label="valid_iid",
        batch_size=validation_batch_size,
        graph_seed=graph_seed,
        profile=build_profile,
    )
    all_examples = [*train_examples, *valid_examples]
    context = attach_input_contexts(
        [*train_batches, *valid_batches], train_examples, all_examples, model_config
    )
    by_id = {example.sample_id: example for example in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(
            batches,
            by_id,
            context_by_id=context["raw_context_by_id"],
        )
        attach_qk_features(batches, by_id, feature_version=str(model_config["qk_region_feature_version"]))
    references = fit_native_loss_references(train_examples, loss_config)
    merged_loss = dict(loss_config)
    merged_loss.update(references)
    prep = P1IPreparedData(
        train_examples=train_examples,
        valid_examples=valid_examples,
        stats=stats,
        builder=builder,
        train_batches=tuple(train_batches),
        valid_batches=tuple(valid_batches),
        context_standardizer=context["standardizer"],
        train_only_loss_references=merged_loss,
        preparation_profile={
            **build_profile,
            "data_loading_and_preprocessing_seconds": float(time.perf_counter() - started),
            "train_batch_count": len(train_batches),
            "valid_batch_count": len(valid_batches),
            "train_sample_count": len(train_examples),
            "valid_sample_count": len(valid_examples),
            "test_and_sealed_access": "closed",
        },
    )
    if profile is not None:
        profile.update(prep.preparation_profile)
    return prep


def tree_max_abs_difference(left: Any, right: Any) -> float:
    left_leaves, left_def = tree.tree_flatten(left)
    right_leaves, right_def = tree.tree_flatten(right)
    if left_def != right_def or len(left_leaves) != len(right_leaves):
        return math.inf
    errors = []
    for a, b in zip(left_leaves, right_leaves, strict=True):
        errors.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return max(errors, default=0.0)


def tree_parameter_count(params: Any) -> int:
    """Count scalar parameters without depending on a model-private API."""

    return int(sum(int(np.asarray(value).size) for value in tree.tree_leaves(params)))


def tree_l2_norm(value: Any) -> float:
    """Return a finite diagnostic norm for gradients, updates, or parameters."""

    leaves = [np.asarray(leaf, dtype=np.float64) for leaf in tree.tree_leaves(value)]
    return float(np.sqrt(sum(float(np.sum(np.square(leaf))) for leaf in leaves)))


def learning_rate_for_epoch(
    epoch: int, *, epochs: int, updates_per_epoch: int, config: Mapping[str, Any]
) -> float:
    """Materialize the registered schedule for a receipt without changing it."""

    schedule = _learning_rate_schedule(
        int(epochs), int(updates_per_epoch), config
    )
    count = max(int(epoch) - 1, 0) * int(updates_per_epoch)
    return float(np.asarray(schedule(count)))


def atomic_training_checkpoint(path: str | Path, *, state: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Write and reload an optimizer-aware checkpoint for rehearsal QA."""

    from .core import TrainingState

    if not isinstance(state, TrainingState):
        raise TypeError("state must be TrainingState")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v7_p1i_training_state_checkpoint_v1",
        **dict(metadata),
        "params": tree.tree_map(lambda value: np.asarray(value), state.params),
        "optimizer_state": tree.tree_map(lambda value: np.asarray(value), state.optimizer_state),
        "step": int(state.step),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    with destination.open("rb") as stream:
        loaded = pickle.load(stream)
    param_error = tree_max_abs_difference(payload["params"], loaded["params"])
    optimizer_error = tree_max_abs_difference(payload["optimizer_state"], loaded["optimizer_state"])
    return {
        "path": str(destination),
        "parameter_reload_max_abs": param_error,
        "optimizer_reload_max_abs": optimizer_error,
        "passed": bool(param_error == 0.0 and optimizer_error == 0.0 and loaded["step"] == state.step),
    }


__all__ = [
    "P1I_DATASET_ID",
    "P1I_BATCH_CONTRACT",
    "P1IPreparedData",
    "load_selected_p1i_examples",
    "build_p1i_batches",
    "prepare_p1i_data",
    "model_apply_full",
    "model_init_full",
    "model_apply_vanilla",
    "model_init_vanilla",
    "loss_fn_vanilla",
    "prediction_to_raw_delta",
    "loss_fn_full",
    "make_gradient_transform",
    "make_p1i_optimizer",
    "atomic_training_checkpoint",
    "tree_max_abs_difference",
    "tree_parameter_count",
    "tree_l2_norm",
    "learning_rate_for_epoch",
]
