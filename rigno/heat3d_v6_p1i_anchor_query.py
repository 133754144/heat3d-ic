"""Checkpoint-preserving anchor/query adapter for sample-varying P1i."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput
from rigno.heat3d_v6_dataset import V6DualRobinExample


TRAINING_ANCHOR_COUNT = 1024


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorPayload:
    sample_id: str
    coords: np.ndarray
    condition_features: np.ndarray
    operator_point_weights: np.ndarray
    condition_feature_names: tuple[str, ...]
    k_encoding_mode: str
    support_hashes: Mapping[str, str]


class P1iSampleVaryingAnchorQueryAdapter:
    """Copy one frozen P1i sample without changing any R0 model input."""

    def __init__(self, example: V6DualRobinExample) -> None:
        coords = np.asarray(example.condition.coords)
        features = np.asarray(example.condition.condition_features)
        weights = np.asarray(example.operator_point_weights)
        if coords.shape != (TRAINING_ANCHOR_COUNT, 3):
            raise ValueError(f"{example.sample_id}: invalid anchor shape {coords.shape}")
        if features.shape[0] != TRAINING_ANCHOR_COUNT:
            raise ValueError(f"{example.sample_id}: feature/support count drift")
        if weights.reshape(-1).shape != (TRAINING_ANCHOR_COUNT,):
            raise ValueError(f"{example.sample_id}: control-volume/support count drift")
        self._example = example
        self.anchor = AnchorPayload(
            sample_id=example.sample_id,
            coords=coords.copy(),
            condition_features=features.copy(),
            operator_point_weights=weights.copy(),
            condition_feature_names=tuple(example.condition.condition_feature_names),
            k_encoding_mode=str(example.condition.k_encoding_mode),
            support_hashes={
                "coords": array_sha256(coords),
                "condition_features": array_sha256(features),
                "operator_point_weights": array_sha256(weights),
            },
        )

    def r0_example(self) -> V6DualRobinExample:
        """Return an independent, byte-equivalent 1024 anchor/query example."""
        meta: dict[str, Any] = deepcopy(self._example.meta)
        meta["p1i_anchor_query_adapter"] = {
            "mode": "R0_exact_anchor_query_identity",
            "anchor_count": TRAINING_ANCHOR_COUNT,
            "target_or_label_used": False,
            "support_hashes": dict(self.anchor.support_hashes),
        }
        return V6DualRobinExample(
            sample_id=self.anchor.sample_id,
            condition=V1SteadyConditionInput(
                coords=self.anchor.coords.copy(),
                condition_features=self.anchor.condition_features.copy(),
                condition_feature_names=self.anchor.condition_feature_names,
                k_encoding_mode=self.anchor.k_encoding_mode,
            ),
            # Required by the legacy group container, but not read by adapter.
            target=self._example.target,
            meta=meta,
            operator_point_weights=self.anchor.operator_point_weights.copy(),
        )

    def r0_input_equivalence(self, adapted: V6DualRobinExample) -> dict[str, Any]:
        observed = {
            "coords": np.asarray(adapted.condition.coords),
            "condition_features": np.asarray(adapted.condition.condition_features),
            "operator_point_weights": np.asarray(adapted.operator_point_weights),
        }
        expected = {
            "coords": self.anchor.coords,
            "condition_features": self.anchor.condition_features,
            "operator_point_weights": self.anchor.operator_point_weights,
        }
        rows = {}
        for name in expected:
            exact = bool(np.array_equal(expected[name], observed[name]))
            rows[name] = {
                "exact": exact,
                "reference_sha256": array_sha256(expected[name]),
                "adapter_sha256": array_sha256(observed[name]),
                "max_abs_error": float(np.max(np.abs(
                    np.asarray(expected[name], dtype=np.float64)
                    - np.asarray(observed[name], dtype=np.float64)
                ))),
            }
        order_exact = adapted.sample_id == self.anchor.sample_id
        schema_exact = (
            tuple(adapted.condition.condition_feature_names)
            == self.anchor.condition_feature_names
            and adapted.condition.k_encoding_mode == self.anchor.k_encoding_mode
        )
        return {
            "passed": bool(all(row["exact"] for row in rows.values()) and order_exact and schema_exact),
            "sample_id_order_exact": order_exact,
            "feature_schema_exact": schema_exact,
            "arrays": rows,
        }
