"""Label-scoped V1 fixture preparation used by the V7 formal trainer.

Only explicitly selected ``train`` and ``valid`` sample directories are read.
The loader never enumerates or opens test/sealed samples.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_native_supervised import (
    V1SteadyConditionInput,
    V1SteadySupervisedExampleNative,
    V1SteadyTarget,
)
from rigno.heat3d_v1_normalization import (
    legacy_train_only_stats,
    normalize_condition,
    normalize_coords,
    normalize_target_delta,
)
from rigno.heat3d_v1_supervised import (
    PHYSICS_LABEL_SUPERVISED_STAGES,
    Heat3DV1SupervisedDataset,
)
from rigno.heat3d_v1_training_semantics import build_legacy_zero_delta_bridge
from rigno.models.operator import Inputs


def _sample_root(subset: str | Path) -> Path:
    root = Path(subset).resolve()
    return root / "samples" if (root / "samples").is_dir() else root


def _manifest_ids(manifest_path: str | Path) -> tuple[list[str], list[str]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train: list[str] = []
    valid: list[str] = []
    for row in payload.get("samples", []):
        role = str(row.get("split", ""))
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("manifest contains a sample without sample_id")
        if role == "train":
            train.append(sample_id)
        elif role == "valid":
            valid.append(sample_id)
    if not train or not valid:
        raise ValueError("V7 fixture manifest must define train and valid samples")
    return train, valid


def _load_one_supervised_sample(sample_dir: Path) -> V1SteadySupervisedExampleNative:
    """Load one selected sample using the library schema without scanning peers."""

    loader = object.__new__(Heat3DV1SupervisedDataset)
    loader.datadir = sample_dir.parent
    loader.input_mode = "pure_physics"
    loader.k_encoding_mode = "diag3"
    loader.allowed_stages = tuple(PHYSICS_LABEL_SUPERVISED_STAGES)
    loader.boundary_mask_fallback = True
    sample = Heat3DV1SupervisedDataset._load_sample(loader, sample_dir)
    temperature = np.asarray(np.load(sample_dir / "temperature.npy"), dtype=np.float64)
    if temperature.ndim != 2 or temperature.shape[1] != 1:
        raise ValueError(f"{sample_dir}: temperature.npy must be [N,1]")
    if temperature.shape[0] != sample["coords"].shape[0]:
        raise ValueError(f"{sample_dir}: temperature/coords count mismatch")
    return V1SteadySupervisedExampleNative(
        sample_id=str(sample["sample_id"]),
        condition=V1SteadyConditionInput(
            coords=np.asarray(sample["coords"], dtype=np.float64),
            condition_features=np.asarray(sample["physics_input"].features, dtype=np.float64),
            condition_feature_names=tuple(sample["physics_input"].feature_names),
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=temperature),
        meta=dict(sample["meta"]),
    )


def load_selected_v1_examples(
    subset: str | Path,
    manifest_path: str | Path,
    *,
    train_count: int,
    valid_count: int,
) -> dict[str, list[V1SteadySupervisedExampleNative]]:
    """Load only the requested train/valid fixture population."""

    if train_count < 1 or valid_count < 1:
        raise ValueError("train_count and valid_count must be positive")
    train_ids, valid_ids = _manifest_ids(manifest_path)
    selected = {"train": train_ids[:train_count], "valid": valid_ids[:valid_count]}
    root = _sample_root(subset)
    result: dict[str, list[V1SteadySupervisedExampleNative]] = {}
    for role, sample_ids in selected.items():
        rows: list[V1SteadySupervisedExampleNative] = []
        for sample_id in sample_ids:
            sample_dir = root / sample_id
            if not sample_dir.is_dir():
                raise FileNotFoundError(f"missing selected {role} sample: {sample_dir}")
            example = _load_one_supervised_sample(sample_dir)
            if example.meta.get("split") != ("train" if role == "train" else "valid"):
                raise ValueError(f"{sample_id}: manifest/sample split mismatch")
            rows.append(example)
        result[role] = rows
    return result


def build_v1_training_stats(examples: Sequence[Any]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot fit normalization stats on an empty train set")
    return legacy_train_only_stats(list(examples))


def _metadata_shape_signature(metadata: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value.shape)
        for value in tree.tree_leaves(metadata)
        if hasattr(value, "shape")
    )


def _build_batch_metadata(builder: Heat3DGraphBuilder, coords_list: list[np.ndarray]):
    metadata_list = [builder.build_metadata(coords) for coords in coords_list]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return (
            tree.tree_map(
                lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
                metadata_list[0],
            ),
            True,
        )
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def _make_group(
    group_name: str,
    examples: Sequence[Any],
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    profile: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    feature_started = time.perf_counter()
    bridges = [build_legacy_zero_delta_bridge(example) for example in examples]
    feature_names = bridges[0].condition_feature_names
    if any(bridge.condition_feature_names != feature_names for bridge in bridges[1:]):
        raise ValueError(f"feature-name mismatch in {group_name}")
    raw_u = jnp.concatenate([bridge.legacy_inputs.u for bridge in bridges], axis=0)
    raw_c = jnp.concatenate([bridge.legacy_inputs.c for bridge in bridges], axis=0)
    raw_coords = jnp.concatenate([bridge.legacy_inputs.x_inp for bridge in bridges], axis=0)
    target_delta = jnp.concatenate([bridge.target_delta_u for bridge in bridges], axis=0)
    t_ref = jnp.concatenate([bridge.t_ref for bridge in bridges], axis=0)
    inputs = Inputs(
        u=raw_u,
        c=normalize_condition(raw_c, dict(stats)),
        x_inp=normalize_coords(raw_coords, dict(stats)),
        x_out=normalize_coords(raw_coords, dict(stats)),
        t=None,
        tau=None,
    )
    if profile is not None:
        profile["feature_preprocessing_seconds"] = profile.get(
            "feature_preprocessing_seconds", 0.0
        ) + (time.perf_counter() - feature_started)
        profile["feature_transform_calls"] = profile.get("feature_transform_calls", 0) + len(examples)
    graph_started = time.perf_counter()
    metadata, shared = _build_batch_metadata(
        builder,
        [np.asarray(example.condition.coords) for example in examples],
    )
    graph_metadata_seconds = time.perf_counter() - graph_started
    graphs_started = time.perf_counter()
    graphs = builder.build_graphs(metadata)
    graph_seconds = graph_metadata_seconds + (time.perf_counter() - graphs_started)
    if profile is not None:
        profile["graph_build_seconds"] = profile.get("graph_build_seconds", 0.0) + graph_seconds
        profile["graph_metadata_build_calls"] = profile.get("graph_metadata_build_calls", 0) + len(examples)
        profile["graph_build_graphs_calls"] = profile.get("graph_build_graphs_calls", 0) + 1
    return {
        "name": group_name,
        "sample_ids": tuple(str(example.sample_id) for example in examples),
        "split": examples[0].meta.get("split"),
        "inputs": inputs,
        "target_normalized": normalize_target_delta(target_delta, dict(stats)),
        "target_delta_raw": target_delta,
        "target_temperature": t_ref + target_delta,
        "t_ref": t_ref,
        "graphs": graphs,
        "metadata": metadata,
        "shared_metadata": shared,
        "feature_names": feature_names,
    }


def _group_examples(examples: Sequence[Any], builder: Heat3DGraphBuilder) -> list[list[Any]]:
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for example in examples:
        bridge = build_legacy_zero_delta_bridge(example)
        signature = _metadata_shape_signature(builder.build_metadata(example.condition.coords))
        key = (
            int(example.condition.coords.shape[0]),
            tuple(bridge.condition_feature_names),
            signature,
        )
        groups.setdefault(key, []).append(example)
    return list(groups.values())


def build_v1_training_batches(
    examples: Sequence[Any],
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    *,
    batch_prefix: str,
    profile: dict[str, float | int] | None = None,
) -> list["TrainingBatch"]:
    """Build the legacy-compatible shape groups once, before train steps."""

    # Local import avoids a module cycle while keeping the public type opaque.
    from .core import TrainingBatch

    batches = []
    grouped = _group_examples(examples, builder)
    built_groups = []
    for index, group_examples in enumerate(grouped, start=1):
        group = _make_group(
            f"{batch_prefix}_group_{index}",
            group_examples,
            stats,
            builder,
            profile=profile,
        )
        built_groups.append(group)
    if not built_groups:
        return batches
    return [
        TrainingBatch(
            batch_id=f"{batch_prefix}_full_batch",
            sample_ids=tuple(
                sample_id
                for group in built_groups
                for sample_id in group["sample_ids"]
            ),
            groups=tuple(built_groups),
        )
    ]
