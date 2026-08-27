"""Stable V6 graph and model-input assembly for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_runtime.features import FeatureTransform
from rigno.heat3d_v6_dataset import V6DualRobinExample


@dataclass(frozen=True)
class GroupBuilder:
    """Build dense JAX graph groups without importing any script module."""

    feature_transform: FeatureTransform
    graph_config: dict[str, Any]
    graph_seed: int = 0

    def build(self, examples: list[V6DualRobinExample], *, name: str) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot build an empty Heat3D model group")
        transformed = [self.feature_transform.transform(example) for example in examples]
        feature_names = transformed[0].feature_names
        if any(row.feature_names != feature_names for row in transformed[1:]):
            raise ValueError(f"{name}: feature-name mismatch")
        inputs = _concatenate_inputs([row.inputs for row in transformed])
        builder = Heat3DGraphBuilder(**dict(self.graph_config))
        metadata, shared = _build_batch_metadata(
            builder,
            [row.graph_coords for row in transformed],
            graph_seed=int(self.graph_seed),
        )
        graphs = builder.build_graphs(metadata)
        return {
            "name": str(name),
            "sample_ids": tuple(example.sample_id for example in examples),
            "split": examples[0].meta.get("split"),
            "inputs": inputs,
            "graphs": graphs,
            "metadata": metadata,
            "shared_metadata": shared,
            "feature_names": feature_names,
        }


def _concatenate_inputs(inputs: list[Any]) -> Any:
    return inputs[0]._replace(
        u=jnp.concatenate([item.u for item in inputs], axis=0),
        c=jnp.concatenate([item.c for item in inputs], axis=0),
        x_inp=jnp.concatenate([item.x_inp for item in inputs], axis=0),
        x_out=jnp.concatenate([item.x_out for item in inputs], axis=0),
    )


def _build_batch_metadata(
    builder: Heat3DGraphBuilder,
    coords_list: list[np.ndarray],
    *,
    graph_seed: int,
) -> tuple[Any, bool]:
    if not coords_list:
        raise ValueError("coords_list cannot be empty")
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    key = jax.random.PRNGKey(int(graph_seed))
    if same_coords:
        metadata = builder.build_metadata(coords_list[0], key=key)
        return (
            tree.tree_map(
                lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
                metadata,
            ),
            True,
        )

    metadata_list = [builder.build_metadata(coords, key=key) for coords in coords_list]
    edge_fields = (
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
        "r2p_edge_indices",
    )
    edge_targets: dict[str, int | None] = {}
    for field in edge_fields:
        values = [getattr(metadata, field) for metadata in metadata_list]
        if all(value is None for value in values):
            edge_targets[field] = None
        elif any(value is None for value in values):
            raise ValueError(f"mixed None/non-None graph metadata for {field}")
        else:
            edge_targets[field] = max(int(value.shape[1]) for value in values)

    padded = []
    for metadata in metadata_list:
        replacements: dict[str, Any] = {}
        for field, target in edge_targets.items():
            value = getattr(metadata, field)
            if target is None:
                replacements[field] = None
                continue
            pad_count = int(target) - int(value.shape[1])
            if pad_count < 0:
                raise AssertionError(f"negative graph metadata padding for {field}")
            replacements[field] = (
                value
                if pad_count == 0
                else jnp.concatenate(
                    [value, jnp.repeat(value[:, -1:, :], pad_count, axis=1)], axis=1
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
