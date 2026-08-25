#!/usr/bin/env python3
"""Train-only fixed-geometry runtime benchmark for frozen V6/P1i routes.

The script never opens temperature targets.  It changes only inference inputs
within the formal P1i domain and compares standard preparation against two
increasingly strong geometry-only cache contracts.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as u1  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadyTarget,
)
from rigno.heat3d_v6_dataset import (  # noqa: E402
    V6DualRobinExample,
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.heat3d_v6_full_field import (  # noqa: E402
    build_reconstruction_map,
    prepare_reconstruction_domain_partition,
)
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


REFERENCE_K = 300.0
ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
)
MODES = ("fresh_new_case", "graph_only_reuse", "full_static_reuse")
EDGE_FIELDS = tuple(qualification.EDGE_FIELDS)


class SupplementalRuntimeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest.update(str(treedef).encode())
    for leaf in leaves:
        if leaf is None or not hasattr(leaf, "shape"):
            digest.update(repr(leaf).encode())
            continue
        digest.update(array_sha256(np.asarray(leaf)).encode())
    return digest.hexdigest()


def tree_leaf_audit(value: Any) -> list[dict[str, Any]]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        label = "/".join(
            str(getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry))))
            for entry in path
        )
        if leaf is None or not hasattr(leaf, "shape"):
            rows.append({"path": label, "scalar_repr": repr(leaf)})
            continue
        array = np.asarray(leaf)
        rows.append({
            "path": label, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": array_sha256(array),
        })
    return rows


def block(value: Any) -> None:
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def host_tree(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if hasattr(leaf, "shape") else leaf,
        value,
    )


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median_s": None, "p95_s": None, "mean_s": None,
                "std_s": None, "min_s": None, "max_s": None}
    return {
        "count": int(array.size),
        "median_s": float(np.median(array)),
        "p95_s": float(np.quantile(array, 0.95)),
        "mean_s": float(np.mean(array)),
        "std_s": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min_s": float(np.min(array)),
        "max_s": float(np.max(array)),
    }


def device_memory() -> dict[str, Any]:
    device = jax.devices("gpu")[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:
        stats = {}
    return {
        "device": str(device),
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def prediction_difference(left: Any, right: Any) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs_K": float(np.max(np.abs(delta))),
        "rmse_K": float(np.sqrt(np.mean(np.square(delta)))),
    }


def mapping_hash(mapping: Any | None) -> str:
    if mapping is None:
        return "direct_no_reconstruction_map"
    return tree_sha256((mapping.support_indices, mapping.neighbor_local_indices,
                        mapping.neighbor_weights))


def graph_hash(metadata: Any) -> str:
    payload = [
        getattr(metadata, name) for name in (
            "x_pnodes_inp", "x_pnodes_out", "x_rnodes", "r_rnodes", *EDGE_FIELDS
        )
    ]
    return tree_sha256(tuple(payload))


def edge_targets(metadata: Any) -> dict[str, int | None]:
    return {
        name: None if getattr(metadata, name) is None
        else int(np.asarray(getattr(metadata, name)).shape[1])
        for name in EDGE_FIELDS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _manifest_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in payload["samples"]}
    return payload, rows


def _sample_directory(dataset_root: Path, row: Mapping[str, Any]) -> Path:
    relative = Path(str(row.get("sample_dir") or row.get("relative_path") or row["sample_id"]))
    if dataset_root.name == "samples" and relative.parts[:1] == ("samples",):
        relative = Path(*relative.parts[1:])
    return dataset_root / relative


def load_input_only_example(
    dataset_root: Path, row: Mapping[str, Any]
) -> V6DualRobinExample:
    """Load a formal P1i input without opening temperature.npy."""
    if str(row["split_role"]) != "train":
        raise SupplementalRuntimeError(f"{row['sample_id']}: only train input is allowed")
    sample_dir = _sample_directory(dataset_root, row)
    meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
    if str(meta["split_role"]) != "train":
        raise SupplementalRuntimeError(f"{row['sample_id']}: sample metadata is not train")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float64).reshape(-1, 1)
    bc = np.asarray(np.load(sample_dir / "bc_features.npy"), dtype=np.float64)
    cv = np.asarray(np.load(sample_dir / "control_volume.npy"), dtype=np.float64).reshape(-1)
    if coords.shape != (1024, 3) or k.shape != (1024, 3) or q.shape != (1024, 1):
        raise SupplementalRuntimeError(f"{row['sample_id']}: input shape drift")
    flags = bc[:, :4]
    top_h = float(meta["top_h_W_m2K"])
    bottom_h = float(meta["bottom_h_W_m2K"])
    ambient = float(meta["physics"]["ambient_K"])
    broadcast = np.column_stack((
        np.full(1024, top_h), np.full(1024, bottom_h), np.zeros(1024)
    ))
    features = np.concatenate((k, q, flags, broadcast), axis=1)
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
            coords=coords,
            condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((1024, 1), ambient, dtype=np.float64)),
        meta=enriched,
        operator_point_weights=cv,
    )


def load_full_mesh(path: Path) -> dict[str, np.ndarray]:
    """Read only shared geometry/measure arrays, never sample temperature."""
    with h5py.File(path, "r") as handle:
        result = {
            "coords": np.asarray(handle["shared/coords_m"], dtype=np.float64),
            "cv": np.asarray(handle["shared/control_volume_m3"], dtype=np.float64),
            "layer": np.asarray(handle["shared/layer_id"], dtype=np.int32),
        }
    if result["coords"].shape != (240825, 3):
        raise SupplementalRuntimeError("full-field shared mesh shape drift")
    return result


def _quantile(lo: float, hi: float, t: float, *, logarithmic: bool) -> float:
    if logarithmic:
        return float(math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo))))
    return float(lo + t * (hi - lo))


def _blend(base: np.ndarray, target: np.ndarray, fraction: float, *, logarithmic: bool) -> np.ndarray:
    if logarithmic:
        return np.exp((1.0 - fraction) * np.log(base) + fraction * np.log(target))
    return (1.0 - fraction) * base + fraction * target


def transform_k(
    base: np.ndarray,
    layer_id: np.ndarray,
    meta: Mapping[str, Any],
    target_quantile: float,
    blend_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.asarray(base, dtype=np.float64).copy()
    layers = meta["physics"]["layers_bottom_to_top"]
    background_masks: list[np.ndarray] = []
    audit_layers = []
    for index, layer in enumerate(layers):
        sampling = layer["sampling"]
        distribution_name = str(sampling["distribution"])
        logarithmic = "log_uniform" in distribution_name
        old = np.asarray(layer["background_k_xyz_W_mK"], dtype=np.float64)
        if "kxy_range_W_mK" in sampling:
            xy = _quantile(*map(float, sampling["kxy_range_W_mK"]), target_quantile,
                           logarithmic=logarithmic)
            z = _quantile(*map(float, sampling["kz_range_W_mK"]), target_quantile,
                          logarithmic=logarithmic)
            target = np.asarray([xy, xy, z], dtype=np.float64)
            bounds = [sampling["kxy_range_W_mK"], sampling["kxy_range_W_mK"],
                      sampling["kz_range_W_mK"]]
        else:
            scalar = _quantile(*map(float, sampling["range_W_mK"]), target_quantile,
                               logarithmic=logarithmic)
            target = np.full(3, scalar, dtype=np.float64)
            bounds = [sampling["range_W_mK"]] * 3
        mask = (layer_id == index) & np.all(
            np.isclose(result, old[None, :], rtol=0.0, atol=1e-10), axis=1
        )
        background_masks.append(mask)
        new = _blend(old, target, blend_fraction, logarithmic=logarithmic)
        result[mask] = new
        audit_layers.append({
            "layer_id": str(layer["id"]), "background_node_count": int(np.sum(mask)),
            "old": old.tolist(), "new": new.tolist(), "bounds": bounds,
        })
    background = np.logical_or.reduce(background_masks)
    local = ~background
    local_target = _quantile(20.0, 400.0, target_quantile, logarithmic=True)
    result[local] = _blend(
        result[local], np.full_like(result[local], local_target),
        blend_fraction, logarithmic=True,
    )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise SupplementalRuntimeError("conductivity transform produced nonphysical values")
    if np.any(result[local] < 20.0 - 1e-10) or np.any(result[local] > 400.0 + 1e-10):
        raise SupplementalRuntimeError("local conductivity left formal range")
    for row, mask in zip(audit_layers, background_masks, strict=True):
        if not np.any(mask):
            raise SupplementalRuntimeError(f"{row['layer_id']}: background mask is empty")
        values = result[mask]
        for axis, (lo, hi) in enumerate(row["bounds"]):
            if np.min(values[:, axis]) < float(lo) - 1e-10 or np.max(values[:, axis]) > float(hi) + 1e-10:
                raise SupplementalRuntimeError(f"{row['layer_id']}: background k left formal range")
    return result, {
        "target_quantile": target_quantile,
        "blend_fraction": blend_fraction,
        "background_node_count": int(np.sum(background)),
        "local_node_count": int(np.sum(local)),
        "minimum_W_mK": float(np.min(result)),
        "maximum_W_mK": float(np.max(result)),
        "layers": audit_layers,
    }


def dynamic_example(
    base: V6DualRobinExample,
    *,
    sample_id: str,
    k_xyz: np.ndarray,
    q_W_m3: np.ndarray,
    total_power_W: float,
) -> V6DualRobinExample:
    features = np.asarray(base.condition.condition_features, dtype=np.float64).copy()
    features[:, :3] = np.asarray(k_xyz, dtype=np.float64)
    features[:, 3] = np.asarray(q_W_m3, dtype=np.float64).reshape(-1)
    meta = deepcopy(base.meta)
    meta["sample_id"] = sample_id
    meta["package_total_power_W"] = float(total_power_W)
    meta["v6_adapter"] = dict(meta["v6_adapter"])
    meta["v6_adapter"]["group_id"] = str(base.meta["v6_adapter"]["group_id"])
    return V6DualRobinExample(
        sample_id=sample_id,
        condition=V1SteadyConditionInput(
            coords=np.asarray(base.condition.coords, dtype=np.float64),
            condition_features=features,
            condition_feature_names=base.condition.condition_feature_names,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((len(features), 1), REFERENCE_K, dtype=np.float64)),
        meta=meta,
        operator_point_weights=np.asarray(base.operator_point_weights, dtype=np.float64),
    )


def prepare_cases(
    *,
    protocol: Mapping[str, Any],
    bases: Sequence[V6DualRobinExample],
    full: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    geometry_audits = []
    q_bounds = protocol["parameter_domain"]["q_nonzero_W_m3"]
    power_bounds = protocol["parameter_domain"]["total_power_W"]
    for base in bases:
        anchor_indices, maximum_distance = highn._anchor_indices(
            base, full["coords"], 1.0e-12
        )
        _, full_k, full_q, power_audit = highn._physics_fields(base, full)
        base_anchor_k = np.asarray(base.condition.condition_features[:, :3], dtype=np.float64)
        base_anchor_q = np.asarray(base.condition.condition_features[:, 3], dtype=np.float64)
        base_power = float(base.meta["package_total_power_W"])
        if not math.isclose(base_power, float(power_audit["requested_power_W"]), rel_tol=1e-10):
            raise SupplementalRuntimeError(f"{base.sample_id}: base power provenance drift")
        base_positive = full_q > 0.0
        if not np.any(base_positive):
            raise SupplementalRuntimeError(f"{base.sample_id}: no source support")
        geometry_audits.append({
            "sample_id": base.sample_id,
            "split_role": base.meta["v6_adapter"]["manifest_split_role"],
            "source_count": int(base.meta["source_region_count"]),
            "k_region_count": int(base.meta["k_region_count"]),
            "anchor_coordinate_max_distance_m": maximum_distance,
            "anchor_indices_sha256": array_sha256(anchor_indices),
            "source_mask_sha256": array_sha256(base_positive),
            "base_total_power_W": base_power,
            "base_q_positive_min_W_m3": float(np.min(full_q[base_positive])),
            "base_q_max_W_m3": float(np.max(full_q)),
        })
        quantiles = protocol["sweeps"]["K_only"]["k_target_quantiles"]
        blend_fraction = float(protocol["sweeps"]["K_only"]["base_to_target_blend"])
        alphas = protocol["sweeps"]["K_plus_Q_scale"]["q_multipliers"]
        for sweep_name in ("K_only", "K_plus_Q_scale"):
            for case_index, target_quantile in enumerate(quantiles):
                alpha = 1.0 if sweep_name == "K_only" else float(alphas[case_index])
                anchor_k, anchor_k_audit = transform_k(
                    base_anchor_k, full["layer"][anchor_indices], base.meta,
                    float(target_quantile), blend_fraction,
                )
                query_k, query_k_audit = transform_k(
                    full_k, full["layer"], base.meta,
                    float(target_quantile), blend_fraction,
                )
                anchor_q = base_anchor_q.copy() if sweep_name == "K_only" else base_anchor_q * alpha
                query_q = full_q.copy() if sweep_name == "K_only" else full_q * alpha
                total_power = base_power * alpha
                positive = query_q > 0.0
                source_region_q = np.asarray([
                    float(row["q_W_m3"]) * alpha
                    for row in base.meta["region_rows"] if row["family"] == "q"
                ], dtype=np.float64)
                if not (float(power_bounds[0]) <= total_power <= float(power_bounds[1])):
                    raise SupplementalRuntimeError(f"{base.sample_id}/{sweep_name}: power left formal range")
                if (
                    source_region_q.size != int(base.meta["source_region_count"])
                    or np.min(source_region_q) < float(q_bounds[0]) - 1e-6
                    or np.max(source_region_q) > float(q_bounds[1]) + 1e-6
                    or np.max(query_q) > float(q_bounds[1]) + 1e-6
                ):
                    raise SupplementalRuntimeError(f"{base.sample_id}/{sweep_name}: source-region q left formal range")
                if not np.array_equal(positive, base_positive):
                    raise SupplementalRuntimeError(f"{base.sample_id}/{sweep_name}: q source mask drift")
                if sweep_name == "K_only" and (
                    not np.array_equal(query_q, full_q) or not np.array_equal(anchor_q, base_anchor_q)
                ):
                    raise SupplementalRuntimeError(f"{base.sample_id}: K-only q is not bitwise fixed")
                base_distribution = full_q[base_positive] / np.sum(full_q[base_positive])
                new_distribution = query_q[positive] / np.sum(query_q[positive])
                normalized_drift = float(np.max(np.abs(base_distribution - new_distribution)))
                if normalized_drift > 1.0e-12:
                    raise SupplementalRuntimeError(f"{base.sample_id}/{sweep_name}: normalized q drift")
                case_id = f"{base.sample_id}__{sweep_name}__q{case_index}"
                cases.append({
                    "case_id": case_id,
                    "base_sample_id": base.sample_id,
                    "sweep": sweep_name,
                    "target_quantile": float(target_quantile),
                    "alpha": alpha,
                    "base": base,
                    "anchor_k": anchor_k,
                    "anchor_q": anchor_q,
                    "anchor_indices": anchor_indices,
                    "full_k": query_k,
                    "full_q": query_q,
                    "total_power_W": total_power,
                    "q_mask_sha256": array_sha256(positive),
                    "source_region_q_min_W_m3": float(np.min(source_region_q)),
                    "source_region_q_max_W_m3": float(np.max(source_region_q)),
                    "normalized_q_max_abs_drift": normalized_drift,
                    "anchor_k_audit": anchor_k_audit,
                    "query_k_audit": query_k_audit,
                    "dynamic_physics_sha256": tree_sha256((anchor_k, anchor_q, query_k, query_q)),
                })
    return cases, {"geometries": geometry_audits, "case_count": len(cases)}


def materialize_anchor(case: Mapping[str, Any]) -> V6DualRobinExample:
    return dynamic_example(
        case["base"], sample_id=str(case["case_id"]),
        k_xyz=np.asarray(case["anchor_k"]), q_W_m3=np.asarray(case["anchor_q"]),
        total_power_W=float(case["total_power_W"]),
    )


def load_runtime(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = args.run_dir / protocol["checkpoint"]["checkpoint_name"]
    if sha256_file(checkpoint_path) != protocol["checkpoint"]["checkpoint_sha256"]:
        raise SupplementalRuntimeError("frozen checkpoint SHA256 drift")
    checkpoint = highn.runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != int(protocol["checkpoint"]["checkpoint_epoch"]):
        raise SupplementalRuntimeError("frozen checkpoint epoch drift")
    run_config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    highn.install_checkpoint_feature_hooks(stats)
    model_config = highn.runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    highn.runner._validate_model_config(model_config)
    standardizer = run_config["global_context"]["standardizer"]
    if standardizer["fit_population"] != "train_only" or int(standardizer["fit_sample_count"]) != 768:
        raise SupplementalRuntimeError("Global Context standardizer is not train-only")
    graph_config = dict(run_config["graph_config"])
    graph_config.update(
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    graph_config = dict(Heat3DGraphBuilder(**graph_config).config)
    return {
        "checkpoint": checkpoint,
        "run_config": run_config,
        "stats": stats,
        "model_config": model_config,
        "graph_config": graph_config,
    }


def route_spec(route: str) -> dict[str, Any]:
    return {
        "route": route,
        "resolution": 16384 if "16384" in route else 240825,
        "u_v2": route.startswith("U_v2"),
        "direct": "direct240825" in route or route == "E240825_direct_control",
    }


def make_builder(runtime: Mapping[str, Any], *, resolution: int, native: bool) -> Heat3DGraphBuilder:
    config = dict(runtime["graph_config"])
    config["subsample_factor"] = 4.0 if native else resolution / 256.0
    config["discrete_graph_backend"] = "sparse_kdtree_v1"
    config["reuse_exact_p2r_for_r2p"] = True
    return Heat3DGraphBuilder(**config)


def support_for_case(
    case: Mapping[str, Any], *, resolution: int, direct: bool,
    full: Mapping[str, np.ndarray], geometry_cache: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    if direct:
        selected = np.arange(len(full["coords"]), dtype=np.int64)
        selected_cv = np.asarray(full["cv"], dtype=np.float64)
    else:
        selected, _ = deterministic_nested_query_prefix(
            sample_id=str(case["base_sample_id"]),
            anchor_indices=np.asarray(case["anchor_indices"], dtype=np.int64),
            full_q=np.asarray(case["full_q"], dtype=np.float64),
            target_count=resolution,
            geometry_cache=geometry_cache,
        )
        selected_cv, _ = conservative_selected_control_volume(
            full_coords=full["coords"], full_control_volume=full["cv"],
            full_layer_id=full["layer"], selected_indices=selected, query_workers=1,
        )
    return selected, selected_cv, {"support_plus_cv": time.perf_counter() - started}


def query_example(
    case: Mapping[str, Any], anchor: V6DualRobinExample,
    selected: np.ndarray, selected_cv: np.ndarray,
    full: Mapping[str, np.ndarray],
) -> V6DualRobinExample:
    support = {
        "selected_indices": selected,
        "operator_control_volume": selected_cv,
        "k_xyz": np.asarray(case["full_k"])[selected],
        "q_W_m3": np.asarray(case["full_q"])[selected],
        "layer_id": np.asarray(full["layer"])[selected],
    }
    return highn._query_example(anchor, support, full["coords"])


def build_graphs(
    *, spec: Mapping[str, Any], anchor: V6DualRobinExample,
    query: V6DualRobinExample, runtime: Mapping[str, Any], graph_key: Any,
) -> tuple[dict[str, Any], dict[str, float]]:
    cpu = jax.devices("cpu")[0]
    stages: dict[str, float] = {}
    native_builder = make_builder(runtime, resolution=1024, native=True)
    query_builder = make_builder(runtime, resolution=int(spec["resolution"]), native=False)
    with jax.default_device(cpu):
        phase = time.perf_counter()
        anchor_coords = highn.runner._graph_coords_for_example(anchor, runtime["stats"])
        native = native_builder.build_metadata(anchor_coords, key=graph_key)
        block(native)
        stages["anchor_graph"] = time.perf_counter() - phase
        phase = time.perf_counter()
        query_coords = highn.runner._graph_coords_for_example(query, runtime["stats"])
        if spec["u_v2"]:
            query_metadata, audit = u1.prior_u1._u_v2_asymmetric_metadata(
                native_builder, native, anchor_coords, query_coords,
                numerical_tolerance=1.0e-6,
                maximum_normalized_overshoot=0.25,
            )
            if not all(audit["native_exact"].values()) or audit["repaired_uncovered_count"] != 0:
                raise SupplementalRuntimeError("U-v2 frozen graph hard gate failed")
        else:
            query_metadata = query_builder.build_metadata(query_coords, key=graph_key)
            audit = None
        block(query_metadata)
        stages["query_graph"] = time.perf_counter() - phase
    return {
        "native_builder": native_builder,
        "query_builder": native_builder if spec["u_v2"] else query_builder,
        "native_metadata": native,
        "query_metadata": query_metadata,
        "u_v2_audit": audit,
    }, stages


def _context(
    anchor: V6DualRobinExample, runtime: Mapping[str, Any]
) -> tuple[dict[str, float], jnp.ndarray]:
    row = highn.runner._global_context_row_for_example(anchor)
    encoded = highn.common.standardize_v6_contexts(
        [row], runtime["run_config"]["global_context"]["standardizer"]
    )[0]
    with jax.default_device(jax.devices("cpu")[0]):
        packed = jnp.asarray(encoded[None, :], dtype=jnp.float32)
    return row, packed


def _dynamic_inputs(
    example: V6DualRobinExample, runtime: Mapping[str, Any], template_inputs: Any
) -> Inputs:
    bridge = highn.runner._bridge_for(example)
    with jax.default_device(jax.devices("cpu")[0]):
        # Preserve the frozen single-example concat operation as well as the
        # CPU device used by _make_batch_group_with_seed.
        raw_u = jnp.concatenate([bridge.legacy_inputs.u], axis=0)
        raw_c = jnp.concatenate([bridge.legacy_inputs.c], axis=0)
        condition = highn.runner.normalize_condition(raw_c, runtime["stats"])
        return Inputs(
            u=raw_u, c=condition,
            x_inp=template_inputs.x_inp, x_out=template_inputs.x_out,
            t=None, tau=None,
        )


def _dynamic_native_physics(
    example: V6DualRobinExample, context_row: Mapping[str, float]
) -> dict[str, Any]:
    count = int(example.condition.coords.shape[0])
    reference = float(example.meta["v6_adapter"]["reference_temperature_K"])
    with jax.default_device(jax.devices("cpu")[0]):
        return {
            "control_volumes": jnp.asarray(
                example.v6_operator_point_weights()[None, :], dtype=jnp.float32
            ),
            "log_s_phys": jnp.asarray([float(context_row["log_s_phys_K"])], dtype=jnp.float32),
            "reference_temperature": jnp.full((1, count), reference, dtype=jnp.float32),
            "dirichlet_mask": jnp.zeros((1, count), dtype=jnp.float32),
            "prescribed_temperature": jnp.full((1, count), reference, dtype=jnp.float32),
        }


def _dynamic_qk(
    example: V6DualRobinExample, padded_metadata: Any, runtime: Mapping[str, Any]
) -> jnp.ndarray:
    relative = example.get_relative_bc_feature_view()
    p2r = np.asarray(padded_metadata.p2r_edge_indices)[0]
    rnodes = int(np.asarray(padded_metadata.x_rnodes).shape[1] - 1)
    value = highn.runner.qk_region_features_from_raw(
        coords=np.asarray(example.condition.coords, dtype=np.float64),
        raw_condition=np.asarray(relative.condition_features, dtype=np.float64),
        condition_feature_names=tuple(relative.condition_feature_names),
        p2r_edge_indices=p2r,
        rnode_count=rnodes,
        feature_version=runtime["model_config"]["qk_region_feature_version"],
    )
    with jax.default_device(jax.devices("cpu")[0]):
        return jnp.asarray(value[None, :, :], dtype=jnp.float32)


def pack_standard(
    *, spec: Mapping[str, Any], anchor: V6DualRobinExample,
    query: V6DualRobinExample, runtime: Mapping[str, Any], graphs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    cpu = jax.devices("cpu")[0]
    stages: dict[str, float] = {}
    native_targets = edge_targets(graphs["native_metadata"])
    query_targets = edge_targets(graphs["query_metadata"])
    with jax.default_device(cpu):
        phase = time.perf_counter()
        anchor_full = highn._prepare_group(
            example=anchor, anchor=anchor, runtime=runtime,
            builder=graphs["native_builder"], metadata=graphs["native_metadata"],
            edge_targets=p5r._compatible_targets(native_targets, graphs["native_metadata"]),
        )
        anchor_group = host_tree(highn._model_group(anchor_full))
        stages["anchor_group_pack"] = time.perf_counter() - phase
        phase = time.perf_counter()
        if spec["u_v2"]:
            combined = u1._combined_targets(native_targets, query_targets)
            output_full = u1._prepare_output_query_group_lean(
                example=query, anchor=anchor, runtime=runtime,
                builder=graphs["query_builder"], metadata=graphs["query_metadata"],
                edge_targets=p5r._compatible_targets(combined, graphs["query_metadata"]),
            )
            output_group = host_tree(output_full)
        else:
            output_full = highn._prepare_group(
                example=query, anchor=anchor, runtime=runtime,
                builder=graphs["query_builder"], metadata=graphs["query_metadata"],
                edge_targets=p5r._compatible_targets(query_targets, graphs["query_metadata"]),
            )
            output_group = host_tree(highn._model_group(output_full))
        stages["query_group_pack"] = time.perf_counter() - phase
    return {
        "anchor_group": anchor_group,
        "query_group": output_group,
        "anchor_full": host_tree(anchor_full),
        "query_full": host_tree(output_full),
        "native_targets": native_targets,
        "query_targets": query_targets,
    }, stages


def pack_full_static(
    *, spec: Mapping[str, Any], anchor: V6DualRobinExample,
    query: V6DualRobinExample, runtime: Mapping[str, Any], template: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    stages: dict[str, float] = {}
    phase = time.perf_counter()
    context_row, context = _context(anchor, runtime)
    anchor_template = template["anchor_group"]
    anchor_metadata = template["anchor_full"]["metadata"]
    anchor_group = {
        "inputs": _dynamic_inputs(anchor, runtime, anchor_template["inputs"]),
        "graphs": anchor_template["graphs"],
        "global_context": context,
        "native_physics": _dynamic_native_physics(anchor, context_row),
        "qk_region_features": _dynamic_qk(anchor, anchor_metadata, runtime),
    }
    stages["anchor_dynamic_pack"] = time.perf_counter() - phase
    phase = time.perf_counter()
    query_template = template["query_group"]
    if spec["u_v2"]:
        query_group = {
            "inputs": _dynamic_inputs(query, runtime, query_template["inputs"]),
            "graphs": query_template["graphs"],
            "native_physics": _dynamic_native_physics(query, context_row),
        }
    else:
        query_metadata = template["query_full"]["metadata"]
        query_group = {
            "inputs": _dynamic_inputs(query, runtime, query_template["inputs"]),
            "graphs": query_template["graphs"],
            "global_context": context,
            "native_physics": _dynamic_native_physics(query, highn.runner._global_context_row_for_example(query)),
            "qk_region_features": _dynamic_qk(query, query_metadata, runtime),
        }
        # Frozen E query groups consume anchor-derived FiLM context while their
        # native shape normalization uses query CV.  Scale is replaced below
        # by the independently predicted anchor scale.
        query_group["global_context"] = context
    stages["query_dynamic_pack"] = time.perf_counter() - phase
    return {"anchor_group": host_tree(anchor_group), "query_group": host_tree(query_group)}, stages


def build_mapping(
    *, spec: Mapping[str, Any], selected: np.ndarray,
    full: Mapping[str, np.ndarray], partition: Any, boundaries: np.ndarray,
) -> tuple[Any | None, dict[str, float]]:
    started = time.perf_counter()
    if spec["direct"]:
        mapping = None
    else:
        mapping, _ = build_reconstruction_map(
            coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
            support_indices=selected, empty_domain_fallback="same_layer",
            prepared_partition=partition, query_workers=1,
        )
    return mapping, {"reconstruction_map": time.perf_counter() - started}


def make_forward(spec: Mapping[str, Any], model: Any) -> tuple[Any, Any, Any]:
    @jax.jit
    def e_forward(params: Any, anchor_group: Any, query_group: Any,
                  query_weights: Any, map_indices: Any, map_weights: Any) -> Any:
        anchor_result = highn.runner._model_apply(model, params, anchor_group)
        anchor_scale = anchor_result["s_hat"].reshape(-1)[0]
        raw = highn.runner._model_apply(model, params, query_group)["raw_temperature"][0, 0, :, 0]
        delta = raw - REFERENCE_K
        normalized = query_weights / jnp.sum(query_weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        del map_indices, map_weights
        return delta / query_scale * anchor_scale

    @jax.jit
    def u_forward(params: Any, inputs_in: Any, inputs_out: Any, graphs: Any,
                  local_p2r: Any, kwargs: Any, map_indices: Any, map_weights: Any) -> Any:
        del map_indices, map_weights
        return model.apply(
            {"params": params}, inputs_in=inputs_in, inputs_out=inputs_out,
            graphs=graphs, output_local_p2r=local_p2r, split=True,
            method=u1._trace_method, **kwargs,
        )["raw_temperature"][0, 0, :, 0] - REFERENCE_K

    @jax.jit
    def reconstruct(support: Any, map_indices: Any, map_weights: Any) -> Any:
        if bool(spec["direct"]):
            return support
        return jnp.sum(support[map_indices] * map_weights.astype(support.dtype), axis=1)

    return e_forward, u_forward, reconstruct


def device_payload(
    *, spec: Mapping[str, Any], packed: Mapping[str, Any],
    mapping: Any | None, query_cv: np.ndarray, graphs: Mapping[str, Any], gpu: Any,
    hash_payload: bool = False,
) -> tuple[Any, dict[str, float], str | None]:
    if mapping is None:
        map_indices = np.zeros((1,), dtype=np.int32)
        map_weights = np.ones((1,), dtype=np.float64)
    else:
        map_indices = np.asarray(mapping.neighbor_local_indices, dtype=np.int32)
        map_weights = np.asarray(mapping.neighbor_weights, dtype=np.float64)
    started = time.perf_counter()
    if spec["u_v2"]:
        anchor_group = packed["anchor_group"]
        query_group = packed["query_group"]
        local = u1._dummy_local_p2r(graphs["native_builder"], graphs["query_metadata"])
        payload = (
            anchor_group["inputs"], query_group["inputs"], query_group["graphs"],
            host_tree(local), host_tree(u1._model_kwargs(anchor_group, query_group)),
            map_indices, map_weights,
        )
    else:
        payload = (
            packed["anchor_group"], packed["query_group"],
            np.asarray(query_cv, dtype=np.float32), map_indices, map_weights,
        )
    prepared_audit = None
    if hash_payload:
        prepared_audit = {"sha256": tree_sha256(payload), "leaves": tree_leaf_audit(payload)}
    device = jax.device_put(payload, gpu)
    enqueue = time.perf_counter() - started
    started = time.perf_counter()
    block(device)
    sync = time.perf_counter() - started
    return device, {"h2d_enqueue": enqueue, "h2d_sync": sync}, prepared_audit


def predict_device(
    *, spec: Mapping[str, Any], device: Any, params: Any,
    e_forward: Any, u_forward: Any, reconstruct: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    if spec["u_v2"]:
        support = u_forward(params, *device)
    else:
        support = e_forward(params, *device)
    block(support)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    full = reconstruct(support, device[-2], device[-1])
    block(full)
    reconstruction_seconds = time.perf_counter() - started
    value = np.asarray(full, dtype=np.float64)
    if value.shape != (240825,) or not np.all(np.isfinite(value)):
        raise SupplementalRuntimeError(f"{spec['route']}: invalid 240825 prediction")
    return value, {
        "neural_forward": forward_seconds,
        "reconstruction_apply": reconstruction_seconds,
    }


def setup_geometry(
    *, case: Mapping[str, Any], spec: Mapping[str, Any], runtime: Mapping[str, Any],
    full: Mapping[str, np.ndarray], geometry_cache: Any, partition: Any,
    boundaries: np.ndarray, graph_key: Any, params: Any, e_forward: Any,
    u_forward: Any, reconstruct: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    anchor = materialize_anchor(case)
    selected, selected_cv, support_stages = support_for_case(
        case, resolution=int(spec["resolution"]), direct=bool(spec["direct"]),
        full=full, geometry_cache=geometry_cache,
    )
    query = query_example(case, anchor, selected, selected_cv, full)
    graphs, graph_stages = build_graphs(
        spec=spec, anchor=anchor, query=query, runtime=runtime, graph_key=graph_key,
    )
    template, pack_stages = pack_standard(
        spec=spec, anchor=anchor, query=query, runtime=runtime, graphs=graphs,
    )
    mapping, map_stages = build_mapping(
        spec=spec, selected=selected, full=full, partition=partition,
        boundaries=boundaries,
    )
    static_packed, static_pack_stages = pack_full_static(
        spec=spec, anchor=anchor, query=query, runtime=runtime, template=template,
    )
    gpu = jax.devices("gpu")[0]
    device, h2d_stages, _ = device_payload(
        spec=spec, packed=static_packed, mapping=mapping, query_cv=selected_cv,
        graphs=graphs, gpu=gpu,
    )
    prediction, prediction_stages = predict_device(
        spec=spec, device=device, params=params, e_forward=e_forward,
        u_forward=u_forward, reconstruct=reconstruct,
    )
    # Second call completes the route/shape JIT warmup outside formal timing.
    prediction_repeat, _ = predict_device(
        spec=spec, device=device, params=params, e_forward=e_forward,
        u_forward=u_forward, reconstruct=reconstruct,
    )
    floor = prediction_difference(prediction, prediction_repeat)
    # The hard gate below compares standard and cached paths.  Same-shape GPU
    # replay is retained as a numerical-floor diagnostic and must not replace
    # or relax that frozen cached/uncached tolerance.
    return {
        "base_sample_id": case["base_sample_id"],
        "selected": selected,
        "selected_cv": selected_cv,
        "selected_sha256": array_sha256(selected),
        "selected_cv_sha256": array_sha256(selected_cv),
        "query_coords_sha256": array_sha256(full["coords"][selected]),
        "graphs": graphs,
        "native_graph_sha256": graph_hash(graphs["native_metadata"]),
        "query_graph_sha256": graph_hash(graphs["query_metadata"]),
        "template": template,
        "mapping": mapping,
        "mapping_sha256": mapping_hash(mapping),
        "setup_seconds": time.perf_counter() - started,
        "setup_stage_seconds": {
            **support_stages, **graph_stages, **pack_stages, **map_stages,
            **{f"static_{key}": value for key, value in static_pack_stages.items()},
            **h2d_stages, **{f"compile_{key}": value for key, value in prediction_stages.items()},
        },
        "same_shape_prediction_floor": floor,
    }


def run_case(
    *, mode: str, case: Mapping[str, Any], spec: Mapping[str, Any],
    runtime: Mapping[str, Any], setup: Mapping[str, Any], full: Mapping[str, np.ndarray],
    geometry_cache: Any, partition: Any, boundaries: np.ndarray, graph_key: Any,
    params: Any, e_forward: Any, u_forward: Any, reconstruct: Any,
    gpu: Any,
    hash_payload: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    stages: dict[str, float] = {}
    phase = time.perf_counter()
    anchor = materialize_anchor(case)
    stages["dynamic_anchor_assembly"] = time.perf_counter() - phase
    if mode == "fresh_new_case":
        selected, selected_cv, values = support_for_case(
            case, resolution=int(spec["resolution"]), direct=bool(spec["direct"]),
            full=full, geometry_cache=geometry_cache,
        )
        stages.update(values)
    else:
        selected = np.asarray(setup["selected"])
        selected_cv = np.asarray(setup["selected_cv"])
    phase = time.perf_counter()
    query = query_example(case, anchor, selected, selected_cv, full)
    stages["dynamic_query_assembly"] = time.perf_counter() - phase
    if mode == "fresh_new_case":
        graphs, values = build_graphs(
            spec=spec, anchor=anchor, query=query, runtime=runtime, graph_key=graph_key,
        )
        stages.update(values)
    else:
        graphs = setup["graphs"]
    if mode == "full_static_reuse":
        packed, values = pack_full_static(
            spec=spec, anchor=anchor, query=query, runtime=runtime,
            template=setup["template"],
        )
    else:
        packed, values = pack_standard(
            spec=spec, anchor=anchor, query=query, runtime=runtime, graphs=graphs,
        )
    stages.update(values)
    if mode == "full_static_reuse":
        mapping = setup["mapping"]
        stages["reconstruction_map_cache_lookup"] = 0.0
    else:
        mapping, values = build_mapping(
            spec=spec, selected=selected, full=full, partition=partition,
            boundaries=boundaries,
        )
        stages.update(values)
    device, values, prepared_audit = device_payload(
        spec=spec, packed=packed, mapping=mapping, query_cv=selected_cv,
        graphs=graphs, gpu=gpu, hash_payload=hash_payload,
    )
    stages.update(values)
    prediction, values = predict_device(
        spec=spec, device=device, params=params, e_forward=e_forward,
        u_forward=u_forward, reconstruct=reconstruct,
    )
    stages.update(values)
    elapsed = time.perf_counter() - started
    stage_sum = float(sum(stages.values()))
    residual = elapsed - stage_sum
    if residual < -1.0e-6 or residual > max(0.025, 0.05 * elapsed):
        raise SupplementalRuntimeError(
            f"{case['case_id']}/{mode}: timing residual {residual:.6f}s"
        )
    return {
        "case_id": case["case_id"], "base_sample_id": case["base_sample_id"],
        "sweep": case["sweep"], "mode": mode, "elapsed_seconds": elapsed,
        "stages": stages, "stage_sum_seconds": stage_sum,
        "residual_seconds": residual, "prediction": prediction,
        "selected_sha256": array_sha256(selected),
        "selected_cv_sha256": array_sha256(selected_cv),
        "native_graph_sha256": graph_hash(graphs["native_metadata"]),
        "query_graph_sha256": graph_hash(graphs["query_metadata"]),
        "mapping_sha256": mapping_hash(mapping),
        "prepared_payload_sha256": None if prepared_audit is None else prepared_audit["sha256"],
        "prepared_payload_leaf_audit": prepared_audit,
    }


def historical_devbox(route: str) -> dict[str, Any] | None:
    path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_p6a_supplementary_lifecycle_table.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("machine") == "devbox" and row.get("route") == route:
                return {
                    "source": str(path.relative_to(ROOT)),
                    "fresh_median_s": float(row["fresh_s_median"]),
                    "fresh_p95_s": float(row["fresh_p95_s_median"]),
                    "role": row["evidence_role"],
                }
    return None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for sweep in ("K_only", "K_plus_Q_scale"):
        for mode in MODES:
            selected = [row for row in rows if row["sweep"] == sweep and row["mode"] == mode]
            elapsed = [float(row["elapsed_seconds"]) for row in selected]
            timing = distribution(elapsed)
            timing["throughput_samples_per_second"] = float(len(elapsed) / sum(elapsed))
            stage_names = sorted({name for row in selected for name in row["stages"]})
            stage_summary = {
                name: distribution([float(row["stages"].get(name, 0.0)) for row in selected])
                for name in stage_names
            }
            output.append({
                "sweep": sweep, "mode": mode, "sample_measurement_count": len(selected),
                "timing": timing, "stage_summary": stage_summary,
            })
    lookup = {(row["sweep"], row["mode"]): row for row in output}
    for row in output:
        fresh = lookup[(row["sweep"], "fresh_new_case")]["timing"]["median_s"]
        row["speedup_vs_fresh_median"] = float(fresh / row["timing"]["median_s"])
    return output


def compact_setup(setup: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: setup[key] for key in (
            "base_sample_id", "selected_sha256", "selected_cv_sha256",
            "query_coords_sha256", "native_graph_sha256", "query_graph_sha256",
            "mapping_sha256", "setup_seconds", "setup_stage_seconds",
            "same_shape_prediction_floor",
        )
    }


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_supplemental_execution":
        raise SupplementalRuntimeError("supplemental protocol is not preregistered")
    if subprocess.check_output(["git", "merge-base", "--is-ancestor",
                                protocol["base_main_commit"], "HEAD"]).strip() != b"":
        pass
    if jax.devices()[0].platform != "gpu":
        raise SupplementalRuntimeError("supplemental benchmark requires devbox GPU")
    if sha256_file(args.manifest) != protocol["dataset"]["manifest_sha256"]:
        raise SupplementalRuntimeError("formal manifest SHA256 drift")
    if sha256_file(args.full_fields) != protocol["dataset"]["full_field_archive_sha256"]:
        raise SupplementalRuntimeError("full-field sidecar SHA256 drift")
    manifest, rows_by_id = _manifest_rows(args.manifest)
    if manifest["dataset_id"] != protocol["dataset"]["dataset_id"]:
        raise SupplementalRuntimeError("dataset ID drift")
    geometry_rows = protocol["representative_geometry_selection"]["samples"]
    bases = []
    for registered in geometry_rows:
        row = rows_by_id[registered["sample_id"]]
        example = load_input_only_example(args.dataset_root, row)
        if int(example.meta["source_region_count"]) != int(registered["source_count"]):
            raise SupplementalRuntimeError("registered source-count stratum drift")
        bases.append(example)
    full = load_full_mesh(args.full_fields)
    cases, input_audit = prepare_cases(protocol=protocol, bases=bases, full=full)
    if args.smoke:
        keep_ids = {bases[0].sample_id}
        cases = [row for row in cases if row["base_sample_id"] in keep_ids and row["target_quantile"] == 0.2]
    runtime = load_runtime(args, protocol)
    spec = route_spec(args.route)
    params_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])
    # Resolve the production device once, outside every per-case timing span.
    # Repeated jax.devices() calls can occasionally synchronize runtime state;
    # that service-level lookup is neither dynamic-physics preparation nor
    # model execution and previously appeared as an unexplained timing residual.
    gpu = jax.devices("gpu")[0]
    model = GraphNeuralOperator(**runtime["model_config"])
    e_forward, u_forward, reconstruct = make_forward(spec, model)
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    boundaries = highn._boundaries(bases[0], float(np.min(full["coords"][:, 2])))
    geometry_cache = prepare_nested_query_geometry_cache(
        full_coords=full["coords"], full_control_volume=full["cv"],
        full_layer_id=full["layer"], layer_boundaries_m=boundaries,
    )
    partition = prepare_reconstruction_domain_partition(
        coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
    )
    cases_by_geometry = {
        base.sample_id: [row for row in cases if row["base_sample_id"] == base.sample_id]
        for base in bases if any(row["base_sample_id"] == base.sample_id for row in cases)
    }
    setups = {}
    for sample_id, geometry_cases in cases_by_geometry.items():
        setups[sample_id] = setup_geometry(
            case=geometry_cases[0], spec=spec, runtime=runtime, full=full,
            geometry_cache=geometry_cache, partition=partition,
            boundaries=boundaries, graph_key=graph_key, params=params,
            e_forward=e_forward, u_forward=u_forward, reconstruct=reconstruct,
        )

    correctness_rows = []
    for case in cases:
        setup = setups[case["base_sample_id"]]
        results = {
            mode: run_case(
                mode=mode, case=case, spec=spec, runtime=runtime, setup=setup,
                full=full, geometry_cache=geometry_cache, partition=partition,
                boundaries=boundaries, graph_key=graph_key, params=params,
                e_forward=e_forward, u_forward=u_forward, reconstruct=reconstruct,
                gpu=gpu,
                hash_payload=True,
            )
            for mode in MODES
        }
        standard = results["fresh_new_case"]
        comparisons = {
            mode: prediction_difference(standard["prediction"], results[mode]["prediction"])
            for mode in ("graph_only_reuse", "full_static_reuse")
        }
        identities = {
            mode: {
                key: results[mode][key] == standard[key]
                for key in (
                    "selected_sha256", "selected_cv_sha256", "native_graph_sha256",
                    "query_graph_sha256", "mapping_sha256",
                    "prepared_payload_sha256",
                )
            }
            for mode in ("graph_only_reuse", "full_static_reuse")
        }
        same_shape_floor = float(setup["same_shape_prediction_floor"]["max_abs_K"])
        numerical_limit = max(1.0e-3, 20.0 * same_shape_floor)
        passed = all(all(row.values()) for row in identities.values()) and all(
            row["max_abs_K"] <= numerical_limit for row in comparisons.values()
        )
        correctness_rows.append({
            "case_id": case["case_id"], "base_sample_id": case["base_sample_id"],
            "sweep": case["sweep"], "comparisons": comparisons,
            "static_identities": identities,
            "same_shape_repeat_max_abs_K": same_shape_floor,
            "frozen_gpu_numerical_limit_K": numerical_limit,
            "passed": passed,
        })
        if not passed:
            prepared_differences = {}
            reference_leaves = {
                row["path"]: row for row in standard["prepared_payload_leaf_audit"]["leaves"]
            }
            for mode in ("graph_only_reuse", "full_static_reuse"):
                candidate_leaves = {
                    row["path"]: row for row in results[mode]["prepared_payload_leaf_audit"]["leaves"]
                }
                prepared_differences[mode] = [
                    {"path": path, "reference": reference_leaves.get(path),
                     "candidate": candidate_leaves.get(path)}
                    for path in sorted(set(reference_leaves) | set(candidate_leaves))
                    if reference_leaves.get(path) != candidate_leaves.get(path)
                ]
            failure = {
                "schema_version": "heat3d_v6_p1i_fixed_geometry_runtime_failure_v1",
                "status": "failed_correctness_gate",
                "route": args.route,
                "case": correctness_rows[-1],
                "prepared_payload_leaf_differences": prepared_differences,
                "setup": compact_setup(setup),
                "protocol_sha256": sha256_file(args.protocol),
            }
            failure_path = args.output.with_suffix(args.output.suffix + ".failed.json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
            raise SupplementalRuntimeError(f"{case['case_id']}: correctness gate failed")

    timing_rows = []
    if not args.correctness_only:
        order_seeds = protocol["timing"]["randomized_order_seeds"][:1 if args.smoke else None]
        for sweep in ("K_only", "K_plus_Q_scale"):
            sweep_cases = [row for row in cases if row["sweep"] == sweep]
            for seed in order_seeds:
                order = np.random.default_rng(int(seed)).permutation(len(sweep_cases))
                for mode in MODES:
                    for position in order:
                        case = sweep_cases[int(position)]
                        result = run_case(
                            mode=mode, case=case, spec=spec, runtime=runtime,
                            setup=setups[case["base_sample_id"]], full=full,
                            geometry_cache=geometry_cache, partition=partition,
                            boundaries=boundaries, graph_key=graph_key, params=params,
                            e_forward=e_forward, u_forward=u_forward, reconstruct=reconstruct,
                            gpu=gpu,
                        )
                        result.pop("prediction")
                        result["order_seed"] = int(seed)
                        timing_rows.append(result)
    summaries = summarize_rows(timing_rows) if timing_rows else []
    setup_rows = [compact_setup(value) for value in setups.values()]
    setup_median = float(np.median([row["setup_seconds"] for row in setup_rows]))
    amortization = []
    for sweep in ("K_only", "K_plus_Q_scale"):
        lookup = {(row["sweep"], row["mode"]): row for row in summaries}
        if not lookup:
            continue
        fresh = float(lookup[(sweep, "fresh_new_case")]["timing"]["median_s"])
        cached = float(lookup[(sweep, "full_static_reuse")]["timing"]["median_s"])
        savings = fresh - cached
        amortization.append({
            "sweep": sweep,
            "setup_median_s_per_geometry_route": setup_median,
            "break_even_repeated_cases": None if savings <= 0.0 else int(math.ceil(setup_median / savings)),
            "amortized_latency_s": {
                str(count): float(cached + setup_median / count) for count in (4, 16, 100, 1000)
            },
        })
    result = {
        "schema_version": "heat3d_v6_p1i_fixed_geometry_runtime_result_v1",
        "status": "passed_smoke" if args.smoke else "passed",
        "route": args.route, "resolution": spec["resolution"],
        "protocol_sha256": sha256_file(args.protocol),
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": "devbox", "device": str(gpu),
        "input_audit": input_audit,
        "temperature_files_opened": 0,
        "accessed_roles": ["train_inputs", "shared_full_mesh_without_temperature"],
        "correctness": {
            "status": "passed", "case_count": len(correctness_rows), "rows": correctness_rows,
            "maximum_cached_vs_standard_max_abs_K": max(
                row["comparisons"][mode]["max_abs_K"]
                for row in correctness_rows for mode in row["comparisons"]
            ),
        },
        "static_setup": setup_rows,
        "timing_rows": timing_rows,
        "summary": summaries,
        "amortization": amortization,
        "historical_devbox_comparison": historical_devbox(args.route),
        "memory": {
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "device_memory": device_memory(),
        },
        "checkpoint_parameter_sha256_before": params_before,
        "checkpoint_parameter_sha256_after": highn._tree_sha256(runtime["checkpoint"]["params"]),
        "checkpoint_unchanged": params_before == highn._tree_sha256(runtime["checkpoint"]["params"]),
        "guardrails": {
            "training": False, "temperature_labels_read": False,
            "test_iid_accessed": False, "sealed_iid_accessed": False,
            "FVM_run": False,
        },
    }
    if not result["checkpoint_unchanged"]:
        raise SupplementalRuntimeError("checkpoint parameter tree changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"], "route": args.route,
        "correctness_max_abs_K": result["correctness"]["maximum_cached_vs_standard_max_abs_K"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
