#!/usr/bin/env python3
"""Deterministic fixture for the V5 inference-only Global FiLM context."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    GlobalContextError,
    batch_global_context_from_raw_condition,
    context_vector,
    fit_train_only_standardizer,
    global_context_from_raw_condition,
    standardize_contexts,
    validate_global_context_schema,
)


FEATURE_NAMES = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "top_h",
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
)


def _synthetic_condition(*, power_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.meshgrid(
        np.asarray((0.0, 2.0e-3)),
        np.asarray((0.0, 3.0e-3)),
        np.asarray((0.0, 1.0e-3, 4.0e-3)),
        indexing="ij",
    )
    coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    n = coords.shape[0]
    kx = 120.0 + 5.0 * (coords[:, 2] > 0.0)
    ky = 110.0 + 3.0 * (coords[:, 0] > 0.0)
    kz = 25.0 + 10.0 * (coords[:, 2] >= 1.0e-3)
    q = np.where(coords[:, 2] >= 1.0e-3, 4.0e7 * power_scale, 0.0)
    is_top = (coords[:, 2] == coords[:, 2].max()).astype(np.float64)
    is_bottom = (coords[:, 2] == coords[:, 2].min()).astype(np.float64)
    is_side = np.logical_or(coords[:, 0] == coords[:, 0].min(), coords[:, 1] == coords[:, 1].max())
    is_interior = np.logical_not(np.logical_or(np.logical_or(is_top > 0.5, is_bottom > 0.5), is_side))
    condition = np.column_stack(
        (
            kx,
            ky,
            kz,
            q,
            np.full(n, 2.0e4),
            np.full(n, 5.0),
            np.zeros(n),
            is_top,
            is_bottom,
            is_side.astype(np.float64),
            is_interior.astype(np.float64),
        )
    )
    return coords, condition


def main() -> int:
    validate_global_context_schema()
    coords_a, condition_a = _synthetic_condition(power_scale=1.0)
    coords_b, condition_b = _synthetic_condition(power_scale=1.7)
    context_a = global_context_from_raw_condition(
        coords=coords_a,
        raw_condition=condition_a,
        condition_feature_names=FEATURE_NAMES,
        reference_temperature_K=300.0,
    )
    context_b = global_context_from_raw_condition(
        coords=coords_b,
        raw_condition=condition_b,
        condition_feature_names=FEATURE_NAMES,
        reference_temperature_K=300.0,
    )
    vector_a = context_vector(context_a)
    if vector_a.shape != (len(GLOBAL_CONTEXT_FEATURES),) or not np.all(np.isfinite(vector_a)):
        raise AssertionError("global context vector is not finite fixed-width")
    if not context_a["log_s_phys_K"] > 0.0:
        raise AssertionError("physics scale must be positive before log")
    if not context_b["P_operator_W"] > context_a["P_operator_W"]:
        raise AssertionError("operator power must follow source-power change")

    standardizer = fit_train_only_standardizer(
        [context_a, context_b],
        fit_sample_ids=("train_a", "train_b"),
    )
    expected_mean = np.vstack((context_vector(context_a), context_vector(context_b))).mean(axis=0)
    if not np.allclose(np.asarray(standardizer["mean"]), expected_mean, rtol=0.0, atol=1.0e-12):
        raise AssertionError("global-context standardizer did not use exactly its train rows")
    held_out = standardize_contexts([context_a], standardizer)
    if held_out.shape != (1, len(GLOBAL_CONTEXT_FEATURES)) or not np.all(np.isfinite(held_out)):
        raise AssertionError("held-out context standardization failed")
    batch = batch_global_context_from_raw_condition(
        coords_per_sample=(coords_a, coords_b),
        raw_conditions_per_sample=(condition_a, condition_b),
        condition_feature_names=FEATURE_NAMES,
        reference_temperatures_K=(300.0, 300.0),
    )
    if len(batch) != 2:
        raise AssertionError("batch global-context assembly lost a sample")

    node_varying = condition_a.copy()
    node_varying[:, FEATURE_NAMES.index("top_h")] *= np.linspace(1.0, 1.01, node_varying.shape[0])
    try:
        global_context_from_raw_condition(
            coords=coords_a,
            raw_condition=node_varying,
            condition_feature_names=FEATURE_NAMES,
            reference_temperature_K=300.0,
        )
    except GlobalContextError:
        pass
    else:
        raise AssertionError("node-varying BC broadcast input was accepted")

    signature_names = set(inspect.signature(global_context_from_raw_condition).parameters)
    forbidden = {"target", "label", "prediction", "residual", "oracle"}
    if signature_names.intersection(forbidden):
        raise AssertionError("global context function accepts a label-derived input")

    print(
        json.dumps(
            {
                "status": "passed",
                "feature_count": len(GLOBAL_CONTEXT_FEATURES),
                "batch_size": len(batch),
                "operator_power_ratio": context_b["P_operator_W"] / context_a["P_operator_W"],
                "train_only_fit_sample_count": standardizer["fit_sample_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
