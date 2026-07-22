#!/usr/bin/env python3
"""Synthetic regression checks for input-only V5 scale context and pooling."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v5_scale_context import (  # noqa: E402
    XY_SCALE_CONTEXT_FEATURES,
    fit_train_only_scale_context_standardizer,
    p2r_partition_of_unity_audit,
    standardize_scale_contexts,
    xy_scale_context_from_raw_condition,
)
from rigno.heat3d_v5_scale_pooling import (  # noqa: E402
    QK_REGION_FEATURE_SCHEMAS,
    qk_region_features_from_raw,
)


FEATURE_NAMES = (
    "k_x", "k_y", "k_z", "q", "is_top", "is_bottom", "is_side",
    "is_interior", "top_h", "bottom_T_fixed_minus_T_ref",
)


def _fixture(q: np.ndarray, kz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray((0.0, 1.0), dtype=np.float64)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    top = coords[:, 2] == 1.0
    bottom = coords[:, 2] == 0.0
    side = np.logical_or(coords[:, 0] == 0.0, coords[:, 1] == 1.0)
    condition = np.column_stack(
        (
            np.linspace(20.0, 80.0, 8),
            np.linspace(30.0, 90.0, 8),
            kz,
            q,
            top.astype(float),
            bottom.astype(float),
            side.astype(float),
            (~(top | bottom | side)).astype(float),
            np.full(8, 2.0e4),
            np.zeros(8),
        )
    )
    edges = np.asarray(
        [[0, 0], [0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0]],
        dtype=np.int64,
    )
    return coords, condition, edges


def main() -> int:
    sparse_q = np.asarray((0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 11.0, 0.0))
    tied_kz = np.full(8, 25.0)
    fixtures = {
        "sparse_q": sparse_q,
        "zero_q": np.zeros(8),
        "same_positive_q": np.full(8, 3.0),
    }
    feature_rows = {}
    for name, q in fixtures.items():
        coords, condition, edges = _fixture(q, tied_kz)
        features = qk_region_features_from_raw(
            coords=coords,
            raw_condition=condition,
            condition_feature_names=FEATURE_NAMES,
            p2r_edge_indices=edges,
            rnode_count=2,
            feature_version="sparse_safe_v2",
        )
        if features.shape != (2, len(QK_REGION_FEATURE_SCHEMAS["sparse_safe_v2"])):
            raise AssertionError("sparse_safe_v2 regional feature width drifted")
        if not np.all(np.isfinite(features)):
            raise AssertionError(f"{name}: non-finite regional feature")
        feature_rows[name] = features
    present_index = QK_REGION_FEATURE_SCHEMAS["sparse_safe_v2"].index(
        "source_present_fraction"
    )
    if np.any(feature_rows["zero_q"][:, present_index] != 0.0):
        raise AssertionError("all-zero q must have zero source-present fraction")
    if not np.allclose(feature_rows["same_positive_q"][:, present_index], 1.0):
        raise AssertionError("identical positive q must have unit source-present fraction")

    coords, condition, edges = _fixture(sparse_q, tied_kz)
    audit = p2r_partition_of_unity_audit(
        coords=coords,
        raw_condition=condition,
        condition_feature_names=FEATURE_NAMES,
        p2r_edge_indices=edges,
        rnode_count=2,
    )
    if audit["zero_degree_node_count"] != 0 or audit["maximum_partition_of_unity_error"] > 1.0e-12:
        raise AssertionError("P2R partition-of-unity failed")
    if not audit["source_conserved"] or not audit["volume_conserved"]:
        raise AssertionError("P2R source/volume conservation failed")

    xy_a = xy_scale_context_from_raw_condition(
        coords=coords,
        raw_condition=condition,
        condition_feature_names=FEATURE_NAMES,
    )
    _, condition_b, _ = _fixture(sparse_q * 1.5, tied_kz * 1.1)
    xy_b = xy_scale_context_from_raw_condition(
        coords=coords,
        raw_condition=condition_b,
        condition_feature_names=FEATURE_NAMES,
    )
    standardizer = fit_train_only_scale_context_standardizer(
        [xy_a, xy_b], fit_sample_ids=("train_a", "train_b")
    )
    encoded = standardize_scale_contexts([xy_a], standardizer)
    if encoded.shape != (1, len(XY_SCALE_CONTEXT_FEATURES)) or not np.all(np.isfinite(encoded)):
        raise AssertionError("train-only XY scale context standardization failed")

    forbidden = {"target", "label", "temperature", "prediction", "oracle"}
    public_functions = (
        qk_region_features_from_raw,
        xy_scale_context_from_raw_condition,
        p2r_partition_of_unity_audit,
    )
    for function in public_functions:
        if forbidden.intersection(inspect.signature(function).parameters):
            raise AssertionError(f"{function.__name__} accepts a label-derived argument")

    print(
        json.dumps(
            {
                "status": "passed",
                "qk_feature_version": "sparse_safe_v2",
                "qk_feature_count": feature_rows["sparse_q"].shape[1],
                "xy_feature_count": len(XY_SCALE_CONTEXT_FEATURES),
                "partition": audit,
                "target_or_label_inputs": [],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
