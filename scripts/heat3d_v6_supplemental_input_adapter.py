#!/usr/bin/env python3
"""Label-free fixed-geometry k/q adapter for the P1i supplemental benchmark."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v6_dataset import V6DualRobinExample, V6_DUAL_ROBIN_CONDITION_FEATURES


REFERENCE_K = 300.0


class AdapterError(RuntimeError):
    pass


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sample_directory(dataset_root: Path, row: Mapping[str, Any]) -> Path:
    relative = Path(str(row.get("sample_dir") or row.get("relative_path") or row["sample_id"]))
    if dataset_root.name == "samples" and relative.parts[:1] == ("samples",):
        relative = Path(*relative.parts[1:])
    return dataset_root / relative


def load_input_only_example(dataset_root: Path, row: Mapping[str, Any]) -> V6DualRobinExample:
    if str(row["split_role"]) != "train":
        raise AdapterError(f"{row['sample_id']}: only train inputs are allowed")
    sample_dir = _sample_directory(dataset_root, row)
    meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
    if str(meta["split_role"]) != "train":
        raise AdapterError(f"{row['sample_id']}: metadata role drift")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float64).reshape(-1, 1)
    bc = np.asarray(np.load(sample_dir / "bc_features.npy"), dtype=np.float64)
    cv = np.asarray(np.load(sample_dir / "control_volume.npy"), dtype=np.float64).reshape(-1)
    if coords.shape != (1024, 3) or k.shape != (1024, 3) or q.shape != (1024, 1):
        raise AdapterError(f"{row['sample_id']}: frozen input shape drift")
    ambient = float(meta["physics"]["ambient_K"])
    features = np.concatenate((
        k, q, bc[:, :4],
        np.column_stack((
            np.full(1024, float(meta["top_h_W_m2K"])),
            np.full(1024, float(meta["bottom_h_W_m2K"])),
            np.zeros(1024),
        )),
    ), axis=1)
    enriched = deepcopy(meta)
    enriched["v6_adapter"] = {
        "dataset_id": str(meta["dataset_id"]),
        "manifest_split_role": "train",
        "group_id": str(row.get("group_id") or row["sample_id"]),
        "reference_temperature_K": ambient,
        "top_T_inf_K": ambient,
        "bottom_T_inf_K": ambient,
        "bottom_boundary_semantics": "robin_not_dirichlet",
        "operator_point_measure": "control_volume_frozen_irregular_1024",
    }
    return V6DualRobinExample(
        sample_id=str(row["sample_id"]),
        condition=V1SteadyConditionInput(
            coords=coords, condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((1024, 1), ambient, dtype=np.float64)),
        meta=enriched, operator_point_weights=cv,
    )


def _quantile(lo: float, hi: float, value: float, *, log: bool) -> float:
    if log:
        return float(math.exp(math.log(lo) + value * (math.log(hi) - math.log(lo))))
    return float(lo + value * (hi - lo))


def _blend(base: np.ndarray, target: np.ndarray, fraction: float, *, log: bool) -> np.ndarray:
    if log:
        return np.exp((1.0 - fraction) * np.log(base) + fraction * np.log(target))
    return (1.0 - fraction) * base + fraction * target


def transform_k(
    base: np.ndarray, layer_id: np.ndarray, meta: Mapping[str, Any],
    target_quantile: float, blend_fraction: float,
) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    background_masks: list[np.ndarray] = []
    bounds_by_layer: list[list[Any]] = []
    for index, layer in enumerate(meta["physics"]["layers_bottom_to_top"]):
        sampling = layer["sampling"]
        logarithmic = "log_uniform" in str(sampling["distribution"])
        old = np.asarray(layer["background_k_xyz_W_mK"], dtype=np.float64)
        if "kxy_range_W_mK" in sampling:
            xy = _quantile(*map(float, sampling["kxy_range_W_mK"]), target_quantile, log=logarithmic)
            z = _quantile(*map(float, sampling["kz_range_W_mK"]), target_quantile, log=logarithmic)
            target = np.asarray([xy, xy, z], dtype=np.float64)
            bounds = [sampling["kxy_range_W_mK"], sampling["kxy_range_W_mK"], sampling["kz_range_W_mK"]]
        else:
            scalar = _quantile(*map(float, sampling["range_W_mK"]), target_quantile, log=logarithmic)
            target = np.full(3, scalar, dtype=np.float64)
            bounds = [sampling["range_W_mK"]] * 3
        mask = (layer_id == index) & np.all(np.isclose(result, old[None, :], rtol=0.0, atol=1e-10), axis=1)
        if not np.any(mask):
            raise AdapterError(f"{layer['id']}: empty background mask")
        background_masks.append(mask)
        bounds_by_layer.append(bounds)
        result[mask] = _blend(old, target, blend_fraction, log=logarithmic)
    background = np.logical_or.reduce(background_masks)
    local = ~background
    local_target = _quantile(20.0, 400.0, target_quantile, log=True)
    result[local] = _blend(result[local], np.full_like(result[local], local_target), blend_fraction, log=True)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise AdapterError("nonphysical k transform")
    if np.any(result[local] < 20.0 - 1e-10) or np.any(result[local] > 400.0 + 1e-10):
        raise AdapterError("local k left formal bounds")
    for mask, bounds in zip(background_masks, bounds_by_layer, strict=True):
        values = result[mask]
        for axis, (lo, hi) in enumerate(bounds):
            if np.min(values[:, axis]) < float(lo) - 1e-10 or np.max(values[:, axis]) > float(hi) + 1e-10:
                raise AdapterError("background k left formal bounds")
    return result


def materialize_dynamic_example(
    base: V6DualRobinExample, *, anchor_k: np.ndarray, anchor_q: np.ndarray,
    total_power_W: float,
) -> V6DualRobinExample:
    features = np.asarray(base.condition.condition_features, dtype=np.float64).copy()
    features[:, :3] = np.asarray(anchor_k, dtype=np.float64)
    features[:, 3] = np.asarray(anchor_q, dtype=np.float64).reshape(-1)
    meta = deepcopy(base.meta)
    meta["package_total_power_W"] = float(total_power_W)
    return V6DualRobinExample(
        sample_id=base.sample_id,
        condition=V1SteadyConditionInput(
            coords=np.asarray(base.condition.coords, dtype=np.float64),
            condition_features=features,
            condition_feature_names=base.condition.condition_feature_names,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((1024, 1), REFERENCE_K, dtype=np.float64)),
        meta=meta,
        operator_point_weights=np.asarray(base.operator_point_weights, dtype=np.float64),
    )


def make_case(
    *, base: V6DualRobinExample, full: Mapping[str, np.ndarray], full_k: np.ndarray,
    full_q: np.ndarray, sweep: str, quantile: float | None, alpha: float,
    blend_fraction: float, protocol: Mapping[str, Any], anchor_indices: np.ndarray,
) -> dict[str, Any]:
    anchor_k_base = np.asarray(base.condition.condition_features[:, :3], dtype=np.float64)
    anchor_q_base = np.asarray(base.condition.condition_features[:, 3], dtype=np.float64)
    if quantile is None:
        anchor_k = anchor_k_base.copy(); query_k = np.asarray(full_k).copy()
    else:
        anchor_k = transform_k(anchor_k_base, full["layer"][anchor_indices], base.meta, quantile, blend_fraction)
        query_k = transform_k(full_k, full["layer"], base.meta, quantile, blend_fraction)
    anchor_q = anchor_q_base.copy() * alpha
    query_q = np.asarray(full_q, dtype=np.float64).copy() * alpha
    base_mask = np.asarray(full_q) > 0.0
    if not np.array_equal(query_q > 0.0, base_mask):
        raise AdapterError("q mask drift")
    if sweep == "K_only" and not np.array_equal(query_q, np.asarray(full_q)):
        raise AdapterError("K-only q is not bitwise fixed")
    positive = query_q > 0.0
    base_dist = np.asarray(full_q)[base_mask] / np.sum(np.asarray(full_q)[base_mask])
    new_dist = query_q[positive] / np.sum(query_q[positive])
    normalized_drift = float(np.max(np.abs(base_dist - new_dist)))
    if normalized_drift > 1e-12:
        raise AdapterError("normalized q distribution drift")
    total_power = float(base.meta["package_total_power_W"]) * alpha
    power_lo, power_hi = protocol["parameter_domain"]["total_power_W"]
    q_lo, q_hi = protocol["parameter_domain"]["q_nonzero_W_m3"]
    source_q = np.asarray([
        float(row["q_W_m3"]) * alpha for row in base.meta["region_rows"]
        if row["family"] == "q"
    ], dtype=np.float64)
    if not float(power_lo) <= total_power <= float(power_hi):
        raise AdapterError("total power left formal bounds")
    if np.min(source_q) < float(q_lo) - 1e-6 or np.max(source_q) > float(q_hi) + 1e-6:
        raise AdapterError("source q left formal bounds")
    return {
        "case_id": f"{base.sample_id}__{sweep}__{quantile if quantile is not None else 'identity'}",
        "base_sample_id": base.sample_id,
        "sweep": sweep,
        "quantile": quantile,
        "alpha": float(alpha),
        "base": base,
        "anchor_indices": np.asarray(anchor_indices, dtype=np.int64),
        "anchor_k": anchor_k,
        "anchor_q": anchor_q,
        "full_k": query_k,
        "full_q": query_q,
        "total_power_W": total_power,
        "q_mask_sha256": array_sha256(positive),
        "normalized_q_max_abs_drift": normalized_drift,
        "dynamic_physics_sha256": array_sha256(np.column_stack((query_k, query_q))),
    }
