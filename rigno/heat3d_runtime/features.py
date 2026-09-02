"""Explicit V6/P1i feature and physics-input assembly for V7 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v1_normalization import (
    normalize_condition,
    normalize_coords,
)
from rigno.heat3d_v1_training_semantics import transform_condition_feature_view
from rigno.heat3d_v5_scale_context import (
    regional_source_volume_weights_from_raw,
    standardize_scale_contexts,
    xy_scale_context_from_raw_condition,
)
from rigno.heat3d_v5_scale_pooling import qk_region_features_from_raw
from rigno.heat3d_v6_dataset import V6DualRobinExample
from rigno.heat3d_v6_global_context import (
    GLOBAL_CONTEXT_FEATURES_V6,
    global_context_from_v6_inputs,
    standardize_v6_contexts,
)
from rigno.models.operator import Inputs


@dataclass(frozen=True)
class TransformedExample:
    """The model-visible tensors for one example, before graph assembly."""

    inputs: Inputs
    graph_coords: np.ndarray
    raw_condition: np.ndarray
    raw_condition_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    reference_temperature: float


@dataclass(frozen=True)
class FeatureTransform:
    """Pure, checkpoint-configured feature transform.

    The transform reproduces the V6 bridge and normalization semantics while
    constructing ``Inputs`` directly.  It never reads ``example.target`` and
    never assigns a function into another module.
    """

    stats: Mapping[str, Any]
    # Alternative P1i support providers intentionally need the frozen
    # full-field context row used during formal training.  The selected
    # 1024-node support may contain no source node even though the global
    # physical input has positive q.  Keeping this override explicit avoids
    # silently recomputing a different context from the support subset.
    context_rows_by_id: Mapping[str, Mapping[str, float]] | None = None

    def transform(self, example: V6DualRobinExample) -> TransformedExample:
        relative = example.get_relative_bc_feature_view()
        raw_condition = np.asarray(relative.condition_features, dtype=np.float64)
        raw_coords = np.asarray(example.condition.coords, dtype=np.float64).reshape(-1, 3)
        transformed, names = transform_condition_feature_view(
            raw_condition,
            tuple(relative.condition_feature_names),
            raw_coords,
            input_feature_schema=str(self.stats["input_feature_schema"]),
            coord_policy=str(self.stats["coord_policy"]),
            extent_feature_policy=str(self.stats["extent_feature_policy"]),
        )
        n_points = raw_coords.shape[0]
        raw_c = jnp.asarray(transformed.reshape(1, 1, n_points, -1), dtype=jnp.float32)
        # The frozen V6 runner normalizes model coordinates from its JAX
        # bridge tensor, but computes graph coordinates from the raw NumPy
        # coordinates before Heat3DGraphBuilder performs its float32 cast.
        # Keeping these two arithmetic paths explicit is required for exact
        # old/new graph metadata equivalence; converting raw coordinates to
        # float32 before normalization changes regional nodes and KD-tree
        # edge selection at tie-sensitive boundaries.
        normalized_coords = normalize_coords(
            jnp.asarray(raw_coords.reshape(1, 1, n_points, 3), dtype=jnp.float32),
            dict(self.stats),
        )
        graph_normalized_coords = normalize_coords(
            raw_coords.reshape(1, 1, n_points, 3), dict(self.stats)
        )
        normalized_condition = normalize_condition(raw_c, dict(self.stats))
        zeros = jnp.zeros((1, 1, n_points, 1), dtype=jnp.float32)
        inputs = Inputs(
            u=zeros,
            c=normalized_condition,
            x_inp=normalized_coords,
            x_out=normalized_coords,
            t=None,
            tau=None,
        )
        if str(self.stats["coord_policy"]) == "sample_local_isotropic":
            graph_coords = np.asarray(graph_normalized_coords).reshape(n_points, 3)
        else:
            graph_coords = raw_coords
        return TransformedExample(
            inputs=inputs,
            graph_coords=graph_coords,
            raw_condition=raw_condition,
            raw_condition_names=tuple(relative.condition_feature_names),
            feature_names=tuple(names),
            reference_temperature=float(relative.t_ref_value),
        )

    def global_context_row(self, example: V6DualRobinExample) -> dict[str, float]:
        if not isinstance(example, V6DualRobinExample):
            raise TypeError("V7 runtime currently accepts V6DualRobinExample only")
        if self.context_rows_by_id is not None:
            row = self.context_rows_by_id.get(str(example.sample_id))
            if row is not None:
                return dict(row)
        return global_context_from_v6_inputs(**example.v6_global_context_inputs())

    def standardize_global_contexts(
        self,
        examples: list[V6DualRobinExample],
        standardizer: Mapping[str, Any],
    ) -> np.ndarray:
        rows = [self.global_context_row(example) for example in examples]
        if tuple(standardizer["feature_names"]) != GLOBAL_CONTEXT_FEATURES_V6:
            raise ValueError("V6 global-context standardizer schema drifted")
        return standardize_v6_contexts(rows, standardizer)

    def standardize_scale_contexts(
        self,
        examples: list[V6DualRobinExample],
        standardizer: Mapping[str, Any],
    ) -> np.ndarray:
        rows = []
        for example in examples:
            relative = example.get_relative_bc_feature_view()
            rows.append(
                xy_scale_context_from_raw_condition(
                    coords=np.asarray(example.condition.coords, dtype=np.float64),
                    raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
                    condition_feature_names=tuple(relative.condition_feature_names),
                )
            )
        return standardize_scale_contexts(rows, standardizer)

    def native_physics(
        self,
        example: V6DualRobinExample,
        *,
        context_row: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        transformed = self.transform(example)
        n_points = transformed.inputs.x_inp.shape[2]
        context = dict(context_row) if context_row is not None else self.global_context_row(example)
        reference = transformed.reference_temperature
        return {
            "control_volumes": jnp.asarray(
                example.v6_operator_point_weights(), dtype=jnp.float32
            ),
            "log_s_phys": jnp.asarray(context["log_s_phys_K"], dtype=jnp.float32),
            "reference_temperature": jnp.full(
                (n_points,), reference, dtype=jnp.float32
            ),
            "dirichlet_mask": jnp.zeros((n_points,), dtype=jnp.float32),
            "prescribed_temperature": jnp.full(
                (n_points,), reference, dtype=jnp.float32
            ),
        }

    def qk_region_features(
        self,
        example: V6DualRobinExample,
        p2r_edge_indices: np.ndarray,
        rnode_count: int,
        *,
        feature_version: str,
    ) -> np.ndarray:
        relative = example.get_relative_bc_feature_view()
        return qk_region_features_from_raw(
            coords=np.asarray(example.condition.coords, dtype=np.float64),
            raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
            condition_feature_names=tuple(relative.condition_feature_names),
            p2r_edge_indices=np.asarray(p2r_edge_indices),
            rnode_count=int(rnode_count),
            feature_version=feature_version,
        )

    def scale_region_weights(
        self,
        example: V6DualRobinExample,
        p2r_edge_indices: np.ndarray,
        rnode_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        relative = example.get_relative_bc_feature_view()
        return regional_source_volume_weights_from_raw(
            coords=np.asarray(example.condition.coords, dtype=np.float64),
            raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
            condition_feature_names=tuple(relative.condition_feature_names),
            p2r_edge_indices=np.asarray(p2r_edge_indices),
            rnode_count=int(rnode_count),
        )
