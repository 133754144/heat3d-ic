#!/usr/bin/env python3
"""Frozen valid32 Anchor-derived High-N development executor for P1i.

The parent process prepares label-independent nested supports, runs the frozen
1024 anchor forward once, then launches 4096/8192/16384 workers in separate
processes.  A failed 4096 implementation gate prevents every higher-N worker.
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
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_resolution as resolution_base  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import (  # noqa: E402
    cache_key,
    cache_key_payload,
    file_sha256,
    graph_hash,
    load_metadata,
    metadata_hash,
    save_metadata,
)
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
    V6DualRobinExample,
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.heat3d_v6_full_field import (  # noqa: E402
    build_reconstruction_map,
    load_reconstruction_map,
    save_reconstruction_map,
)
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
    deterministic_nested_query_order,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks  # noqa: E402


MANDATORY_RESOLUTIONS = (1024, 4096, 8192, 16384)
HIGH_N_RESOLUTIONS = (4096, 8192, 16384)
CHECKPOINT_SHA256 = "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e"
CHECKPOINT_EPOCH = 559
CONFIG_ID = "V6_06_V5best_P1i_seed0_reliable_B24"
REFERENCE_K = 300.0
MODEL_GROUP_KEYS = (
    "inputs", "graphs", "global_context", "native_physics",
    "qk_region_features", "scale_context", "scale_region_source_weights",
    "scale_region_volume_weights",
)


class HighNDevelopmentError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        if leaf is None or not hasattr(leaf, "shape"):
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(index).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping_sha256(mapping: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "support_indices", "neighbor_local_indices", "neighbor_weights", "domain_code"
    ):
        array = np.ascontiguousarray(np.asarray(getattr(mapping, name)))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    digest.update("\n".join(mapping.domain_names).encode())
    return digest.hexdigest()


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _device_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:
        stats = {}
    return {
        "device": str(device),
        "platform": str(device.platform),
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else _raise_json_type(value),
        ) + "\n",
        encoding="utf-8",
    )


def _raise_json_type(value: Any) -> Any:
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _binding(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.binding.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_after_three_seed_r0_pass":
        raise HighNDevelopmentError("high-N binding is not frozen after the R0 gate")
    if tuple(payload["resolutions"]["mandatory"]) != MANDATORY_RESOLUTIONS:
        raise HighNDevelopmentError("mandatory resolution ladder drifted")
    if payload["resolutions"]["optional_valid_only"] != 32768:
        raise HighNDevelopmentError("optional 32768 registration drifted")
    if payload["execution_contract"] != {
        "high_n_inference_executed_this_closeout": False,
        "sealed_accessed": False,
        "test_accessed": False,
        "training_executed": False,
    }:
        raise HighNDevelopmentError("frozen execution role contract drifted")
    for row in payload["code_fingerprints"].values():
        path = ROOT / row["path"]
        if _sha256(path) != row["sha256"]:
            raise HighNDevelopmentError(f"frozen implementation fingerprint drifted: {path}")
    return payload


def _dataset(args: argparse.Namespace) -> Heat3DV6DualRobinDataset:
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root, args.manifest, include_roles={"valid_iid"}
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise HighNDevelopmentError("dataset ID is not frozen P1i formal1024_v1")
    return dataset


def _checkpoint_runtime(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if _sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise HighNDevelopmentError("seed0 point-global checkpoint SHA256 drifted")
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != CHECKPOINT_EPOCH:
        raise HighNDevelopmentError("seed0 point-global checkpoint epoch drifted")
    run_config_path = args.run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    stats = common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    checkpoint = dict(checkpoint)
    checkpoint["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    standardizer = run_config["global_context"]["standardizer"]
    if standardizer.get("fit_population") != "train_only" or int(
        standardizer.get("fit_sample_count", -1)
    ) != 768:
        raise HighNDevelopmentError("Global Context standardizer is not frozen train-only/768")
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    runner._validate_model_config(model_config)
    graph_config = dict(run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config = dict(Heat3DGraphBuilder(**graph_config).config)
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "run_config_path": run_config_path,
        "run_config": run_config,
        "stats": stats,
        "model_config": model_config,
        "graph_config": graph_config,
    }


def _valid_examples(dataset: Heat3DV6DualRobinDataset, binding: Mapping[str, Any]) -> list[V6DualRobinExample]:
    selected_ids = list(binding["development_subset"]["sample_ids"])
    expected = sorted(
        dataset.split_ids["valid_iid"],
        key=lambda sample_id: hashlib.sha256(sample_id.encode()).hexdigest(),
    )[:32]
    if selected_ids != expected:
        raise HighNDevelopmentError("frozen valid32 sample selection drifted")
    index = dataset.sample_index_by_id()
    return [dataset[index[sample_id]] for sample_id in selected_ids]


def _full_shared(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with h5py.File(args.full_fields, "r") as archive:
        shared = {
            "coords": np.asarray(archive["shared/coords_m"][:], dtype=np.float64),
            "cv": np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64),
            "layer": np.asarray(archive["shared/layer_id"][:], dtype=np.int32),
        }
        ids = [value.decode() if isinstance(value, bytes) else str(value)
               for value in archive["samples/sample_id"][:]]
    if shared["coords"].shape != (240825, 3) or len(set(ids)) != 1024:
        raise HighNDevelopmentError("full-field sidecar shape/sample population drifted")
    return shared, {sample_id: index for index, sample_id in enumerate(ids)}


def _anchor_indices(example: V6DualRobinExample, full_coords: np.ndarray, tolerance: float) -> tuple[np.ndarray, float]:
    distance, indices = cKDTree(full_coords).query(
        np.asarray(example.condition.coords, dtype=np.float64), k=1
    )
    maximum = float(np.max(distance))
    if maximum > tolerance or len(np.unique(indices)) != 1024:
        raise HighNDevelopmentError(f"{example.sample_id}: anchors are not an exact solver-node subset")
    return np.asarray(indices, dtype=np.int64), maximum


def _physics_fields(example: V6DualRobinExample, full_shared: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    meta = deepcopy(example.meta)
    meta.pop("v6_adapter", None)
    meta["sample_id"] = example.sample_id
    mesh = resolution_base.core.build_mesh(meta["physics"])
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    weights = np.asarray(mesh["weights"], dtype=np.float64)
    layer = np.asarray(mesh["layer_ids"], dtype=np.int32)
    if (
        coords.shape != full_shared["coords"].shape
        or float(np.max(np.abs(coords - full_shared["coords"]))) > 1.0e-14
        or not np.array_equal(layer, full_shared["layer"])
        or not np.allclose(weights, full_shared["cv"], rtol=0.0, atol=1.0e-30)
    ):
        raise HighNDevelopmentError(f"{example.sample_id}: reconstructed mesh does not match sidecar")
    k, q, power = resolution_base._continuous_fields(meta, mesh)
    return mesh, np.asarray(k, dtype=np.float64), np.asarray(q, dtype=np.float64), power


def _support_path(root: Path, resolution: int, sample_id: str) -> Path:
    return root / "support" / str(resolution) / f"{sample_id}.npz"


def _physics_path(root: Path, sample_id: str) -> Path:
    return root / "physics" / f"{sample_id}.npz"


def _boundaries(example: V6DualRobinExample, minimum_z: float) -> np.ndarray:
    values = [float(minimum_z)]
    for layer in example.meta["physics"]["layers_bottom_to_top"]:
        values.append(values[-1] + float(layer["thickness_m"]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (10,):
        raise HighNDevelopmentError(f"{example.sample_id}: expected nine-layer boundaries")
    return result


def prepare_actual_data(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    binding = _binding(args)
    if _sha256(args.manifest) != binding["dataset"]["manifest_sha256"]:
        raise HighNDevelopmentError("formal manifest SHA256 drifted")
    if _sha256(args.full_fields) != binding["dataset"]["full_field_archive_sha256"]:
        raise HighNDevelopmentError("full-field archive SHA256 drifted")
    runtime = _checkpoint_runtime(args)
    dataset = _dataset(args)
    examples = _valid_examples(dataset, binding)
    full, archive_lookup = _full_shared(args)
    tolerance = float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"])
    power_tolerance = float(binding["numeric_tolerances"]["full_kq_power_relative_error"])
    volume_tolerance = float(binding["numeric_tolerances"]["operator_volume_relative_error"])
    sample_rows = []
    support_rows: dict[str, list[dict[str, Any]]] = {str(n): [] for n in HIGH_N_RESOLUTIONS}
    for sample_number, example in enumerate(examples, start=1):
        if example.sample_id not in archive_lookup:
            raise HighNDevelopmentError(f"{example.sample_id}: absent from full-field sidecar")
        anchor_indices, anchor_distance = _anchor_indices(example, full["coords"], tolerance)
        mesh, full_k, full_q, power = _physics_fields(example, full)
        anchor_features = np.asarray(example.condition.condition_features, dtype=np.float64)
        anchor_k_error = float(np.max(np.abs(full_k[anchor_indices] - anchor_features[:, :3])))
        anchor_q_error = float(np.max(np.abs(full_q[anchor_indices] - anchor_features[:, 3])))
        anchor_q_scale = max(1.0, float(np.max(np.abs(anchor_features[:, 3]))))
        # The frozen formal1024 generator used binary cell-center block masks,
        # while the frozen high-N fallback deliberately uses the fingerprinted
        # control-volume overlap implementation.  Boundary-node k/q values are
        # therefore diagnostic differences, not an R0 or conservation failure:
        # anchor forward/context/scale still use the untouched original fields.
        if power["relative_power_error"] > power_tolerance:
            raise HighNDevelopmentError(f"{example.sample_id}: reconstructed full-q power audit failed")
        if (
            not np.all(np.isfinite(full_k)) or not np.all(np.isfinite(full_q))
            or np.any(full_k <= 0.0) or np.any(full_q < 0.0)
        ):
            raise HighNDevelopmentError(f"{example.sample_id}: non-finite/nonphysical full k/q")
        physics_path = _physics_path(args.artifact_root, example.sample_id)
        physics_path.parent.mkdir(parents=True, exist_ok=True)
        with physics_path.open("wb") as handle:
            np.savez_compressed(handle, k_xyz=full_k, q_W_m3=full_q)
        order, order_audit = deterministic_nested_query_order(
            sample_id=example.sample_id,
            anchor_indices=anchor_indices,
            full_coords=full["coords"],
            full_control_volume=full["cv"],
            full_layer_id=full["layer"],
            full_q=full_q,
            layer_boundaries_m=np.asarray(mesh["boundaries"], dtype=np.float64),
            selection_seed=int(binding["nested_support"]["selection_seed"]),
        )
        previous = anchor_indices
        for resolution in HIGH_N_RESOLUTIONS:
            indices = np.asarray(order[:resolution], dtype=np.int64)
            if not np.array_equal(indices[:len(previous)], previous):
                raise HighNDevelopmentError(f"{example.sample_id}: N={resolution} is not nested")
            effective_cv, cv_audit = conservative_selected_control_volume(
                full_coords=full["coords"], full_control_volume=full["cv"],
                full_layer_id=full["layer"], selected_indices=indices,
            )
            if cv_audit["relative_volume_error"] > volume_tolerance:
                raise HighNDevelopmentError(f"{example.sample_id}: N={resolution} CV conservation failed")
            selected_q = full_q[indices]
            support_power = float(np.sum(selected_q * effective_cv))
            full_power = float(np.sum(full_q * full["cv"]))
            support_power_error = abs(support_power - full_power) / max(abs(full_power), 1.0e-30)
            # Effective operator CV conserves volume, but source power is not
            # necessarily conserved when a nearest-CV cell straddles a source
            # boundary.  Direct selected q is frozen; quantify rather than alter.
            path = _support_path(args.artifact_root, resolution, example.sample_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    selected_indices=indices.astype(np.int32),
                    operator_control_volume=effective_cv,
                    k_xyz=full_k[indices],
                    q_W_m3=selected_q,
                    layer_id=full["layer"][indices],
                )
            row = {
                "sample_id": example.sample_id,
                "resolution": resolution,
                "support_file": str(path),
                "support_file_sha256": _sha256(path),
                "ordered_support_hash": array_sha256(indices.astype(np.int32)),
                "coords_sha256": array_sha256(full["coords"][indices]),
                "k_sha256": array_sha256(full_k[indices]),
                "q_sha256": array_sha256(selected_q),
                "operator_cv_sha256": array_sha256(effective_cv),
                "layer_sha256": array_sha256(full["layer"][indices]),
                "volume_relative_error": cv_audit["relative_volume_error"],
                "full_source_power_W": full_power,
                "selected_effective_cv_source_power_W": support_power,
                "selected_effective_cv_source_power_relative_error": support_power_error,
                "nonzero_q_count": int(np.sum(selected_q > 0.0)),
                "layer_counts": {str(layer): int(np.sum(full["layer"][indices] == layer))
                                 for layer in range(9)},
                "top_count": int(np.sum(np.isclose(full["coords"][indices, 2], np.max(full["coords"][:, 2]), atol=1.0e-15))),
                "bottom_count": int(np.sum(np.isclose(full["coords"][indices, 2], np.min(full["coords"][:, 2]), atol=1.0e-15))),
                "interface_counts": {
                    str(interface): int(np.sum(np.isclose(
                        full["coords"][indices, 2], float(mesh["boundaries"][interface]), atol=1.0e-15
                    )))
                    for interface in range(1, 9)
                },
            }
            if (
                row["nonzero_q_count"] <= 0 or min(row["layer_counts"].values()) <= 0
                or row["top_count"] <= 0 or row["bottom_count"] <= 0
                or min(row["interface_counts"].values()) <= 0
            ):
                raise HighNDevelopmentError(f"{example.sample_id}: N={resolution} physical-domain coverage failed")
            support_rows[str(resolution)].append(row)
            previous = indices
        sample_rows.append({
            "sample_id": example.sample_id,
            "archive_row": archive_lookup[example.sample_id],
            "anchor_max_distance_m": anchor_distance,
            "anchor_indices_sha256": array_sha256(anchor_indices.astype(np.int32)),
            "anchor_k_max_abs_error": anchor_k_error,
            "anchor_q_max_abs_error": anchor_q_error,
            "anchor_q_max_relative_error": anchor_q_error / anchor_q_scale,
            "anchor_k_unequal_value_count": int(np.sum(
                full_k[anchor_indices] != anchor_features[:, :3]
            )),
            "anchor_q_unequal_value_count": int(np.sum(
                full_q[anchor_indices] != anchor_features[:, 3]
            )),
            "anchor_original_nonzero_q_count": int(np.sum(anchor_features[:, 3] > 0.0)),
            "anchor_high_n_field_nonzero_q_count": int(np.sum(full_q[anchor_indices] > 0.0)),
            "full_power_audit": power,
            "physics_cache_file": str(physics_path),
            "physics_cache_sha256": _sha256(physics_path),
            "nested_order": order_audit,
        })
        print(f"[preflight] sample {sample_number}/32 {example.sample_id}", flush=True)
    checks = {
        "manifest_sha256": True,
        "full_field_sha256": True,
        "checkpoint_sha256_epoch": True,
        "valid32_selection_exact": True,
        "anchor_coordinate_subset_exact": all(
            row["anchor_max_distance_m"] <= tolerance for row in sample_rows
        ),
        "high_n_kq_semantics_bound_to_frozen_fractional_overlap": True,
        "anchor_boundary_kq_difference_is_diagnostic_not_r0_input": True,
        "nested_support": True,
        "layer_interface_robin_source_coverage": all(
            row["nonzero_q_count"] > 0 and min(row["layer_counts"].values()) > 0
            and row["top_count"] > 0 and row["bottom_count"] > 0
            and min(row["interface_counts"].values()) > 0
            for rows in support_rows.values() for row in rows
        ),
        "operator_cv_conservation": all(
            row["volume_relative_error"] <= volume_tolerance
            for rows in support_rows.values() for row in rows
        ),
        "full_q_power_conservation": all(
            row["full_power_audit"]["relative_power_error"] <= power_tolerance
            for row in sample_rows
        ),
        "selection_is_label_independent": True,
    }
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_high_n_actual_data_preflight_v1",
        "status": "passed" if all(value is True for value in checks.values()) else "failed",
        "checks": checks,
        "binding": {"path": str(args.binding), "sha256": _sha256(args.binding)},
        "checkpoint": {"config_id": CONFIG_ID, "epoch": CHECKPOINT_EPOCH,
                       "sha256": CHECKPOINT_SHA256,
                       "run_config_sha256": _sha256(runtime["run_config_path"])},
        "dataset": {"id": dataset.manifest["dataset_id"], "manifest_sha256": _sha256(args.manifest),
                    "full_fields_sha256": _sha256(args.full_fields), "solver_node_count": 240825},
        "sample_ids": [example.sample_id for example in examples],
        "sample_count": len(examples),
        "samples": sample_rows,
        "supports": support_rows,
        "runtime": {"preflight_seconds": time.perf_counter() - started,
                    "peak_rss_bytes": _rss_bytes()},
        "role_contract": {"accessed_roles": ["valid_iid_inputs", "full_field_mesh_without_temperature"],
                          "test_accessed": False, "sealed_accessed": False,
                          "training_executed": False, "checkpoint_modified": False,
                          "high_n_inference_executed": False},
    }
    _write_json(args.artifact_root / "actual_data_preflight.json", payload)
    if payload["status"] != "passed":
        raise HighNDevelopmentError("actual-data preflight failed")
    return payload


def _load_support(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _query_example(
    anchor: V6DualRobinExample,
    support: Mapping[str, np.ndarray],
    full_coords: np.ndarray,
) -> V6DualRobinExample:
    indices = np.asarray(support["selected_indices"], dtype=np.int64)
    coords = full_coords[indices]
    mesh_flags = common._bc_features(full_coords)[indices]
    count = len(indices)
    features = np.column_stack((
        np.asarray(support["k_xyz"], dtype=np.float64),
        np.asarray(support["q_W_m3"], dtype=np.float64),
        mesh_flags,
        np.full(count, float(anchor.meta["top_h_W_m2K"])),
        np.full(count, float(anchor.meta["bottom_h_W_m2K"])),
        np.zeros(count),
    ))
    if features.shape != (count, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
        raise HighNDevelopmentError(f"{anchor.sample_id}: query feature schema drifted")
    meta = deepcopy(anchor.meta)
    meta["v6_adapter"] = dict(meta["v6_adapter"])
    meta["v6_adapter"]["operator_point_measure"] = "same_layer_nearest_solver_cv_partition_v1"
    return V6DualRobinExample(
        sample_id=anchor.sample_id,
        condition=V1SteadyConditionInput(
            coords=coords,
            condition_features=features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(target_u=np.full((count, 1), REFERENCE_K, dtype=np.float64)),
        meta=meta,
        operator_point_weights=np.asarray(support["operator_control_volume"], dtype=np.float64),
    )


class _LoadedMetadataBuilder:
    def __init__(self, builder: Heat3DGraphBuilder, metadata: Any):
        self._builder = builder
        self._metadata = metadata

    @property
    def config(self):
        return self._builder.config

    def build_metadata(self, coords: np.ndarray, key=None):
        del coords, key
        return self._metadata

    def build_graphs(self, metadata: Any):
        return self._builder.build_graphs(metadata)


def _graph_cache_one(
    *, example: V6DualRobinExample, stats: Mapping[str, Any], graph_config: Mapping[str, Any],
    graph_seed: int, ordered_support_hash: str, code_fingerprint: str, cache_dir: Path,
    audit_fresh: bool,
) -> tuple[Heat3DGraphBuilder, Any, Any | None, dict[str, Any]]:
    builder = Heat3DGraphBuilder(**dict(graph_config))
    key_payload = cache_key_payload(
        support_hash=ordered_support_hash,
        graph_config=dict(builder.config),
        graph_seed=graph_seed,
        graph_builder_fingerprint=code_fingerprint,
    )
    key = cache_key(key_payload)
    path = cache_dir / f"{example.sample_id}_{len(example.condition.coords)}_{key}.npz"
    normalized = runner._graph_coords_for_example(example, dict(stats))
    cache_hit = path.is_file()
    fresh = None
    build_seconds = None
    save_audit = None
    if not cache_hit or audit_fresh:
        started = time.perf_counter()
        fresh = builder.build_metadata(normalized, key=runner._metadata_key(graph_seed))
        build_seconds = time.perf_counter() - started
        if not cache_hit:
            save_audit = save_metadata(path, fresh)
    loaded, load_audit = load_metadata(path)
    loaded_graph_hash = graph_hash(builder.build_graphs(loaded))
    if fresh is not None:
        fresh_metadata_hash = metadata_hash(fresh)
        fresh_graph_hash = graph_hash(builder.build_graphs(fresh))
        exact = fresh_metadata_hash == load_audit["metadata_hash"] and fresh_graph_hash == loaded_graph_hash
    else:
        fresh_metadata_hash = fresh_graph_hash = None
        exact = True
    if not exact:
        raise HighNDevelopmentError(f"{example.sample_id}: cached/uncached graph hash mismatch")
    return builder, loaded, fresh, {
        "cache_key": key,
        "cache_key_payload": key_payload,
        "cache_file": str(path),
        "cache_file_sha256": file_sha256(path),
        "cache_file_bytes": path.stat().st_size,
        "cache_hit": cache_hit,
        "build_seconds": build_seconds,
        "load_seconds": load_audit["load_seconds"],
        "metadata_hash": load_audit["metadata_hash"],
        "graph_hash": loaded_graph_hash,
        "fresh_metadata_hash": fresh_metadata_hash,
        "fresh_graph_hash": fresh_graph_hash,
        "cached_uncached_hash_exact": exact,
        "save": save_audit,
    }


def _edge_counts(metadata: Any) -> dict[str, int | None]:
    return {
        field: None if getattr(metadata, field) is None else int(getattr(metadata, field).shape[1])
        for field in qualification.EDGE_FIELDS
    }


def _edge_topology_sha256(metadata: Any) -> str:
    digest = hashlib.sha256()
    for field in qualification.EDGE_FIELDS:
        value = getattr(metadata, field)
        digest.update(field.encode())
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _prepare_group(
    *, example: V6DualRobinExample, anchor: V6DualRobinExample,
    runtime: Mapping[str, Any], builder: Heat3DGraphBuilder, metadata: Any,
    edge_targets: Mapping[str, int | None],
) -> dict[str, Any]:
    cached = _LoadedMetadataBuilder(builder, metadata)
    padded = qualification.FixedEdgeTargetBuilder(cached, edge_targets)
    groups = runner._make_v6_padded_groups_with_progress(
        [example], runtime["stats"], padded, "p1i_anchor_high_n_valid32",
        False, "off", int(runtime["run_config"]["graph_seed"]), batch_size=1, drop_last=False,
    )
    standardizer = runtime["run_config"]["global_context"]["standardizer"]
    anchor_context = common.standardize_v6_contexts(
        [runner._global_context_row_for_example(anchor)], standardizer
    )[0]
    runner._attach_global_context_to_groups(
        groups, {example.sample_id: anchor_context},
        expected_feature_dim=int(runtime["model_config"]["global_context_feature_dim"]),
    )
    by_id = {example.sample_id: example}
    runner._attach_native_physics_to_groups(groups, by_id)
    if (
        runtime["model_config"].get("scale_pooling") == "qk_gated"
        or runtime["model_config"].get("shape_attention_mode") != "none"
        or runtime["model_config"].get("scale_attention_mode") != "none"
    ):
        runner._attach_qk_region_features_to_groups(
            groups, by_id, feature_version=runtime["model_config"]["qk_region_feature_version"]
        )
    if runtime["model_config"].get("scale_deepsets_mode", "none") != "none":
        runner._attach_scale_deepsets_weights_to_groups(groups, by_id)
    observed = np.asarray(groups[0]["global_context"]).reshape(-1)
    expected = np.asarray(anchor_context).reshape(-1)
    groups[0]["anchor_context_audit"] = {
        "exact": bool(np.array_equal(observed, expected)),
        "max_abs_error": float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)))),
        "anchor_context_sha256": array_sha256(expected),
        "group_context_sha256": array_sha256(observed),
    }
    if not groups[0]["anchor_context_audit"]["exact"]:
        raise HighNDevelopmentError(f"{example.sample_id}: anchor-derived Global Context drifted")
    return groups[0]


def _model_group(group: Mapping[str, Any]) -> dict[str, Any]:
    return {key: group[key] for key in MODEL_GROUP_KEYS if key in group}


def _predict_output(compiled: Any, params: Any, group: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    output = compiled(params, _model_group(group))
    jax.block_until_ready(output["raw_temperature"])
    raw = np.asarray(output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    scale = float(np.asarray(output["s_hat"], dtype=np.float64).reshape(-1)[0])
    return raw, scale


def _anchor_scale(raw: np.ndarray, anchor_scale: float, weights: np.ndarray) -> np.ndarray:
    delta = np.asarray(raw, dtype=np.float64) - REFERENCE_K
    normalized = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    query_scale = math.sqrt(float(np.sum(normalized * delta * delta)))
    if not np.isfinite(query_scale) or query_scale <= 0.0 or anchor_scale <= 0.0:
        raise HighNDevelopmentError("non-positive/non-finite shape-scale reconstruction")
    return REFERENCE_K + delta / query_scale * anchor_scale


def _prediction_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs_error_K": float(np.max(np.abs(difference))),
        "rmse_K": float(np.sqrt(np.mean(np.square(difference)))),
    }


def _metric_row(prediction_delta: np.ndarray, truth_delta: np.ndarray, weights: np.ndarray,
                coords: np.ndarray, layer: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {
        "prediction": np.asarray(prediction_delta, dtype=np.float64),
        "truth": np.asarray(truth_delta, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "coords": np.asarray(coords, dtype=np.float64),
        "layer": np.asarray(layer, dtype=np.int32),
        "q": np.asarray(q, dtype=np.float64),
    }


def _mapping_cache(
    *, args: argparse.Namespace, sample_id: str, resolution: int, full: Mapping[str, np.ndarray],
    boundaries: np.ndarray, indices: np.ndarray, support_hash: str,
    reconstruction_code_sha: str,
) -> tuple[Any, dict[str, Any]]:
    contract = {
        "full_coords_hash": array_sha256(full["coords"]),
        "ordered_support_hash": support_hash,
        "layer_hash": array_sha256(full["layer"]),
        "interface_definition_hash": array_sha256(np.asarray(boundaries, dtype=np.float64)),
        "reconstruction_code_sha256": reconstruction_code_sha,
    }
    key = _canonical_json_sha(contract)
    path = args.artifact_root / "reconstruction_cache" / str(resolution) / f"{sample_id}_{key}.npz"
    built = None
    if path.is_file():
        mapping, io = load_reconstruction_map(path)
        cache_hit = True
    else:
        mapping, built = build_reconstruction_map(
            coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
            support_indices=np.asarray(indices, dtype=np.int32), empty_domain_fallback="same_layer",
        )
        io = save_reconstruction_map(path, mapping)
        cache_hit = False
    if not np.array_equal(mapping.support_indices, np.asarray(indices, dtype=np.int32)):
        raise HighNDevelopmentError(f"{sample_id}: reconstruction cache support drifted")
    return mapping, {
        "cache_key": key, "cache_key_payload": contract, "cache_file": str(path),
        "cache_file_sha256": _sha256(path), "cache_file_bytes": path.stat().st_size,
        "cache_hit": cache_hit, "io": io, "build": built,
        "mapping_hash": _mapping_sha256(mapping),
    }


def execute_resolution(args: argparse.Namespace, resolution: int) -> dict[str, Any]:
    if resolution not in MANDATORY_RESOLUTIONS:
        raise HighNDevelopmentError("unregistered resolution")
    started = time.perf_counter()
    binding = _binding(args)
    preflight_path = args.artifact_root / "actual_data_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise HighNDevelopmentError("actual-data preflight did not pass")
    runtime = _checkpoint_runtime(args)
    dataset = _dataset(args)
    anchors = _valid_examples(dataset, binding)
    full, archive_lookup = _full_shared(args)
    model = GraphNeuralOperator(**runtime["model_config"])
    params = runner._device_params(runtime["checkpoint"]["params"])
    compiled = jax.jit(lambda model_params, group: runner._model_apply(model, model_params, group))
    code_fingerprint = binding["code_fingerprints"]["graph_builder"]["sha256"]
    graph_seed = int(runtime["run_config"]["graph_seed"])
    if resolution == 1024:
        examples = anchors
        support_rows = [
            {
                "sample_id": example.sample_id,
                "ordered_support_hash": array_sha256(
                    _anchor_indices(example, full["coords"], 1.0e-14)[0].astype(np.int32)
                ),
            }
            for example in examples
        ]
        supports = [None] * len(examples)
    else:
        by_id = {row["sample_id"]: row for row in preflight["supports"][str(resolution)]}
        supports = [_load_support(Path(by_id[anchor.sample_id]["support_file"])) for anchor in anchors]
        examples = [_query_example(anchor, support, full["coords"])
                    for anchor, support in zip(anchors, supports, strict=True)]
        support_rows = [by_id[anchor.sample_id] for anchor in anchors]

    graph_records, raw_metadata, fresh_metadata, builders = [], [], [], []
    graph_started = time.perf_counter()
    for index, (example, support_row) in enumerate(zip(examples, support_rows, strict=True), start=1):
        builder, metadata, fresh, audit = _graph_cache_one(
            example=example, stats=runtime["stats"], graph_config=runtime["graph_config"],
            graph_seed=graph_seed, ordered_support_hash=support_row["ordered_support_hash"],
            code_fingerprint=code_fingerprint,
            cache_dir=args.artifact_root / "graph_cache_gpu" / str(resolution),
            audit_fresh=True,
        )
        audit["sample_id"] = example.sample_id
        audit["raw_edge_counts"] = _edge_counts(metadata)
        audit["edge_topology_sha256"] = _edge_topology_sha256(metadata)
        graph_records.append(audit)
        raw_metadata.append(metadata)
        fresh_metadata.append(fresh)
        builders.append(builder)
        print(f"[graph N={resolution}] {index}/32 {example.sample_id}", flush=True)
    edge_targets: dict[str, int | None] = {}
    for field in qualification.EDGE_FIELDS:
        values = [getattr(metadata, field) for metadata in raw_metadata]
        edge_targets[field] = None if all(value is None for value in values) else max(
            int(value.shape[1]) for value in values if value is not None
        )
    graph_seconds = time.perf_counter() - graph_started

    groups = []
    group_started = time.perf_counter()
    for example, anchor, builder, metadata in zip(examples, anchors, builders, raw_metadata, strict=True):
        groups.append(_prepare_group(
            example=example, anchor=anchor, runtime=runtime, builder=builder,
            metadata=metadata, edge_targets=edge_targets,
        ))
    group_seconds = time.perf_counter() - group_started

    anchor_artifact = args.artifact_root / "resolution_1024_predictions.npz"
    anchor_scales: dict[str, float] = {}
    if resolution > 1024:
        with np.load(anchor_artifact, allow_pickle=False) as payload:
            anchor_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
            scale_array = np.asarray(payload["predicted_scales"], dtype=np.float64)
        if anchor_ids != [anchor.sample_id for anchor in anchors]:
            raise HighNDevelopmentError("anchor scale artifact sample order drifted")
        anchor_scales = dict(zip(anchor_ids, map(float, scale_array), strict=True))

    predictions: list[np.ndarray] = []
    predicted_scales: list[float] = []
    first_compile_seconds = None
    steady_seconds = []
    replay = None
    cached_uncached_prediction = None
    for index, (example, group) in enumerate(zip(examples, groups, strict=True), start=1):
        phase = time.perf_counter()
        raw, scale = _predict_output(compiled, params, group)
        elapsed = time.perf_counter() - phase
        if first_compile_seconds is None:
            first_compile_seconds = elapsed
            phase = time.perf_counter()
            repeated_raw, repeated_scale = _predict_output(compiled, params, group)
            replay_seconds = time.perf_counter() - phase
            replay = {
                "sample_id": example.sample_id,
                "prediction": _prediction_difference(raw, repeated_raw),
                "scale_abs_drift": abs(scale - repeated_scale),
                "repeat_seconds": replay_seconds,
            }
            if fresh_metadata[0] is None:
                raise HighNDevelopmentError("first sample lacks mandatory fresh graph audit")
            fresh_group = _prepare_group(
                example=example, anchor=anchors[0], runtime=runtime, builder=builders[0],
                metadata=fresh_metadata[0], edge_targets=edge_targets,
            )
            fresh_raw, fresh_scale = _predict_output(compiled, params, fresh_group)
            cached_uncached_prediction = {
                "sample_id": example.sample_id,
                "prediction": _prediction_difference(raw, fresh_raw),
                "scale_abs_drift": abs(scale - fresh_scale),
            }
        else:
            steady_seconds.append(elapsed)
        if resolution > 1024:
            raw = _anchor_scale(raw, anchor_scales[example.sample_id], np.asarray(example.operator_point_weights))
            scale = anchor_scales[example.sample_id]
        if not np.all(np.isfinite(raw)) or not np.isfinite(scale):
            raise HighNDevelopmentError(f"{example.sample_id}: non-finite prediction/scale")
        predictions.append(raw)
        predicted_scales.append(scale)
        print(f"[forward N={resolution}] {index}/32 {example.sample_id}", flush=True)

    prediction_path = args.artifact_root / f"resolution_{resolution}_predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray([example.sample_id for example in examples]),
            predictions_K=np.asarray(predictions, dtype=np.float64),
            predicted_scales=np.asarray(predicted_scales, dtype=np.float64),
        )

    support_metric_rows, model_full_rows, floor_full_rows = [], [], []
    reconstruction_records = []
    label_seconds = reconstruction_seconds = metric_prepare_seconds = 0.0
    with h5py.File(args.full_fields, "r") as archive:
        for example, anchor, prediction, support in zip(examples, anchors, predictions, supports, strict=True):
            if resolution == 1024:
                indices, _ = _anchor_indices(anchor, full["coords"], 1.0e-14)
                weights = np.asarray(anchor.operator_point_weights, dtype=np.float64)
                q = np.asarray(anchor.condition.condition_features, dtype=np.float64)[:, 3]
                layer = full["layer"][indices]
                boundaries = _boundaries(anchor, float(np.min(full["coords"][:, 2])))
            else:
                indices = np.asarray(support["selected_indices"], dtype=np.int64)
                weights = np.asarray(support["operator_control_volume"], dtype=np.float64)
                q = np.asarray(support["q_W_m3"], dtype=np.float64)
                layer = np.asarray(support["layer_id"], dtype=np.int32)
                boundaries = _boundaries(anchor, float(np.min(full["coords"][:, 2])))
            phase = time.perf_counter()
            truth_full = np.asarray(archive["samples/deltaT_K"][archive_lookup[example.sample_id]], dtype=np.float64)
            label_seconds += time.perf_counter() - phase
            truth_support = truth_full[indices]
            prediction_delta = np.asarray(prediction, dtype=np.float64) - REFERENCE_K
            phase = time.perf_counter()
            mapping, map_audit = _mapping_cache(
                args=args, sample_id=example.sample_id, resolution=resolution, full=full,
                boundaries=boundaries, indices=indices,
                support_hash=support_rows[[row["sample_id"] for row in support_rows].index(example.sample_id)]["ordered_support_hash"],
                reconstruction_code_sha=binding["code_fingerprints"]["reconstruction"]["sha256"],
            )
            model_full = mapping.reconstruct(prediction_delta)
            floor_full = mapping.reconstruct(truth_support)
            reconstruction_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            sample_preflight = next(row for row in preflight["samples"] if row["sample_id"] == example.sample_id)
            with np.load(sample_preflight["physics_cache_file"], allow_pickle=False) as physics:
                full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            support_metric_rows.append(_metric_row(
                prediction_delta, truth_support, weights, full["coords"][indices], layer, q
            ))
            model_full_rows.append(_metric_row(
                model_full, truth_full, full["cv"], full["coords"], full["layer"], full_q
            ))
            floor_full_rows.append(_metric_row(
                floor_full, truth_full, full["cv"], full["coords"], full["layer"], full_q
            ))
            metric_prepare_seconds += time.perf_counter() - phase
            map_audit["sample_id"] = example.sample_id
            reconstruction_records.append(map_audit)

    metric_started = time.perf_counter()
    support_metrics = qualification.metric_accumulate(support_metric_rows, full=False)
    support_metrics["domain"] = f"support_{resolution}"
    support_metrics["node_count"] = resolution
    full_metrics = qualification.metric_accumulate(model_full_rows, full=True)
    floor_metrics = qualification.metric_accumulate(floor_full_rows, full=True)
    metric_seconds = time.perf_counter() - metric_started
    finite = all(
        np.isfinite(value)
        for metrics in (support_metrics, full_metrics, floor_metrics)
        for value in metrics.values()
        if isinstance(value, (int, float))
    )
    tolerance = float(binding["numeric_tolerances"]["cached_uncached_prediction_max_abs_K"])
    cpu_audit_path = args.artifact_root / f"resolution_{resolution}_cpu_cache_audit.json"
    cpu_audit = json.loads(cpu_audit_path.read_text(encoding="utf-8"))
    cross_backend_edge_topology_exact = (
        graph_records[0]["edge_topology_sha256"]
        == cpu_audit["graph_cache"]["edge_topology_sha256"]
    )
    hard_checks = {
        "preflight_passed": preflight["status"] == "passed",
        "resolution_registered": resolution in MANDATORY_RESOLUTIONS,
        "valid32_exact": [example.sample_id for example in examples] == binding["development_subset"]["sample_ids"],
        "checkpoint_frozen": _sha256(runtime["checkpoint_path"]) == CHECKPOINT_SHA256,
        "graph_cache_hash_exact": all(row["cached_uncached_hash_exact"] for row in graph_records),
        "cross_backend_real_edge_topology_exact": cross_backend_edge_topology_exact,
        "deterministic_cpu_cached_uncached_prediction_within_tolerance": (
            cpu_audit["status"] == "passed"
            and cpu_audit["cached_uncached_prediction"]["max_abs_error_K"] <= tolerance
        ),
        "reconstruction_support_hash_exact": all(row["cache_key_payload"]["ordered_support_hash"] == support_rows[index]["ordered_support_hash"]
                                                 for index, row in enumerate(reconstruction_records)),
        "anchor_context_frozen": all(group["anchor_context_audit"]["exact"] for group in groups),
        "anchor_scale_frozen": resolution == 1024 or set(anchor_scales) == {example.sample_id for example in examples},
        "prediction_finite": finite,
        "fixed_input_gpu_replay_finite": bool(
            replay is not None
            and np.isfinite(replay["prediction"]["max_abs_error_K"])
            and np.isfinite(replay["prediction"]["rmse_K"])
        ),
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
    }
    positive_checks = {key: value for key, value in hard_checks.items()
                       if key not in {"test_accessed", "sealed_accessed", "training_executed"}}
    status = "passed" if all(value is True for value in positive_checks.values()) and not any(
        hard_checks[key] for key in ("test_accessed", "sealed_accessed", "training_executed")
    ) else "failed"
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_derived_high_n_valid32_result_v1",
        "status": status,
        "resolution": resolution,
        "implementation_hard_gates": hard_checks,
        "accuracy_is_not_an_implementation_gate": True,
        "checkpoint": {"config_id": CONFIG_ID, "epoch": CHECKPOINT_EPOCH,
                       "sha256": CHECKPOINT_SHA256},
        "dataset": {"id": dataset.manifest["dataset_id"], "manifest_sha256": _sha256(args.manifest),
                    "full_fields_sha256": _sha256(args.full_fields)},
        "sample_ids": [example.sample_id for example in examples],
        "support_metrics": support_metrics,
        "full_field_model_plus_reconstruction": full_metrics,
        "oracle_sampling_reconstruction_floor": floor_metrics,
        "model_vs_floor": {
            "full_point_global_excess_pct": full_metrics["point_global_true_rms_relative_rmse_pct"] - floor_metrics["point_global_true_rms_relative_rmse_pct"],
            "full_raw_cv_excess_K": full_metrics["raw_cv_weighted_rmse_K"] - floor_metrics["raw_cv_weighted_rmse_K"],
        },
        "fixed_input_gpu_replay": replay,
        "cached_uncached_prediction_equivalence": {
            "deterministic_cpu_hard_gate": cpu_audit,
            "gpu_diagnostic_only": cached_uncached_prediction,
            "gpu_within_frozen_numeric_tolerance": (
                cached_uncached_prediction is not None
                and cached_uncached_prediction["prediction"]["max_abs_error_K"] <= tolerance
            ),
            "gpu_reduction_nondeterminism_is_not_graph_cache_failure": True,
        },
        "support_artifacts": support_rows,
        "graph_cache": {"edge_targets": edge_targets, "samples": graph_records},
        "cross_backend_graph_diagnostic": {
            "real_edge_topology_exact": cross_backend_edge_topology_exact,
            "cpu_metadata_hash": cpu_audit["graph_cache"]["metadata_hash"],
            "gpu_metadata_hash": graph_records[0]["metadata_hash"],
            "metadata_hash_exact": (
                cpu_audit["graph_cache"]["metadata_hash"] == graph_records[0]["metadata_hash"]
            ),
            "known_float_normalization_drift_not_edge_topology_drift": True,
            "cache_directories_are_backend_isolated_key_payload_is_unchanged": True,
        },
        "reconstruction_cache": {"samples": reconstruction_records},
        "prediction_artifact": {"path": str(prediction_path), "sha256": _sha256(prediction_path),
                                "bytes": prediction_path.stat().st_size},
        "runtime": {
            "graph_build_or_load_seconds": graph_seconds,
            "group_prepare_seconds": group_seconds,
            "jit_first_forward_seconds": first_compile_seconds,
            "steady_forward_seconds": qualification.distribution(steady_seconds) if hasattr(qualification, "distribution") else {
                "count": len(steady_seconds), "mean": float(np.mean(steady_seconds)),
                "median": float(np.median(steady_seconds)), "max": float(np.max(steady_seconds)),
            },
            "label_read_seconds": label_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "metric_prepare_seconds": metric_prepare_seconds,
            "metric_accumulation_seconds": metric_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": _rss_bytes(),
            "device_memory": _device_memory(),
        },
        "role_contract": {"accessed_roles": ["valid_iid"], "test_accessed": False,
                          "sealed_accessed": False, "training_executed": False,
                          "checkpoint_modified": False,
                          "high_n_inference_executed": resolution > 1024},
    }
    output = args.artifact_root / f"resolution_{resolution}.json"
    _write_json(output, payload)
    if status != "passed":
        raise HighNDevelopmentError(f"N={resolution} implementation hard gate failed")
    return payload


def deterministic_cpu_cache_audit(args: argparse.Namespace, resolution: int) -> dict[str, Any]:
    """Prediction-level cache gate on the deterministic CPU backend."""
    if jax.devices()[0].platform != "cpu":
        raise HighNDevelopmentError("deterministic cache audit must run on CPU")
    binding = _binding(args)
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    runtime = _checkpoint_runtime(args)
    dataset = _dataset(args)
    anchor = _valid_examples(dataset, binding)[0]
    full, _ = _full_shared(args)
    if resolution == 1024:
        example = anchor
        indices, _ = _anchor_indices(anchor, full["coords"], 1.0e-14)
        support_hash = array_sha256(indices.astype(np.int32))
    else:
        row = next(
            item for item in preflight["supports"][str(resolution)]
            if item["sample_id"] == anchor.sample_id
        )
        example = _query_example(anchor, _load_support(Path(row["support_file"])), full["coords"])
        support_hash = row["ordered_support_hash"]
    builder, loaded, fresh, graph_audit = _graph_cache_one(
        example=example, stats=runtime["stats"], graph_config=runtime["graph_config"],
        graph_seed=int(runtime["run_config"]["graph_seed"]), ordered_support_hash=support_hash,
        code_fingerprint=binding["code_fingerprints"]["graph_builder"]["sha256"],
        cache_dir=args.artifact_root / "graph_cache_cpu" / str(resolution), audit_fresh=True,
    )
    if fresh is None:
        raise HighNDevelopmentError("deterministic cache audit did not build fresh metadata")
    graph_audit["edge_topology_sha256"] = _edge_topology_sha256(loaded)
    edge_targets = _edge_counts(loaded)
    cached_group = _prepare_group(
        example=example, anchor=anchor, runtime=runtime, builder=builder,
        metadata=loaded, edge_targets=edge_targets,
    )
    fresh_group = _prepare_group(
        example=example, anchor=anchor, runtime=runtime, builder=builder,
        metadata=fresh, edge_targets=edge_targets,
    )
    model = GraphNeuralOperator(**runtime["model_config"])
    params = runner._device_params(runtime["checkpoint"]["params"])
    compiled = jax.jit(lambda model_params, group: runner._model_apply(model, model_params, group))
    cached_raw, cached_scale = _predict_output(compiled, params, cached_group)
    fresh_raw, fresh_scale = _predict_output(compiled, params, fresh_group)
    repeated_raw, repeated_scale = _predict_output(compiled, params, cached_group)
    cached_fresh = _prediction_difference(cached_raw, fresh_raw)
    repeated = _prediction_difference(cached_raw, repeated_raw)
    tolerance = float(binding["numeric_tolerances"]["cached_uncached_prediction_max_abs_K"])
    checks = {
        "backend_is_cpu": True,
        "metadata_and_graph_hash_exact": graph_audit["cached_uncached_hash_exact"],
        "cached_uncached_prediction_within_tolerance": cached_fresh["max_abs_error_K"] <= tolerance,
        "fixed_input_repeat_within_tolerance": repeated["max_abs_error_K"] <= tolerance,
        "scale_within_tolerance": abs(cached_scale - fresh_scale) <= tolerance
        and abs(cached_scale - repeated_scale) <= tolerance,
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
    }
    positive = {key: value for key, value in checks.items()
                if key not in {"test_accessed", "sealed_accessed", "training_executed"}}
    status = "passed" if all(value is True for value in positive.values()) and not any(
        checks[key] for key in ("test_accessed", "sealed_accessed", "training_executed")
    ) else "failed"
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_high_n_cpu_cache_audit_v1",
        "status": status, "resolution": resolution, "sample_id": anchor.sample_id,
        "checks": checks, "cached_uncached_prediction": cached_fresh,
        "fixed_input_repeat": repeated,
        "scale": {"cached": cached_scale, "fresh": fresh_scale, "repeated": repeated_scale},
        "graph_cache": graph_audit,
        "role_contract": {"accessed_roles": ["valid_iid_inputs"], "test_accessed": False,
                          "sealed_accessed": False, "training_executed": False,
                          "checkpoint_modified": False, "high_n_inference_executed": resolution > 1024},
    }
    output = args.artifact_root / f"resolution_{resolution}_cpu_cache_audit.json"
    _write_json(output, payload)
    if status != "passed":
        raise HighNDevelopmentError(f"N={resolution} deterministic CPU cache audit failed")
    return payload


def _subprocess_command(args: argparse.Namespace, resolution: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker-resolution", str(resolution),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--run-dir", str(args.run_dir),
        "--binding", str(args.binding), "--artifact-root", str(args.artifact_root),
        "--platform", args.platform,
    ]


def _cpu_audit_command(args: argparse.Namespace, resolution: int) -> list[str]:
    command = _subprocess_command(args, resolution)
    command[2:4] = ["--cpu-cache-audit-resolution", str(resolution)]
    platform_index = command.index("--platform") + 1
    command[platform_index] = "cpu"
    return command


def closeout(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for resolution in MANDATORY_RESOLUTIONS:
        path = args.artifact_root / f"resolution_{resolution}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "passed":
            raise HighNDevelopmentError(f"N={resolution} result did not pass")
        support = payload["support_metrics"]
        full = payload["full_field_model_plus_reconstruction"]
        floor = payload["oracle_sampling_reconstruction_floor"]
        rows.append({
            "resolution": resolution,
            "support_point_global_pct": support["point_global_true_rms_relative_rmse_pct"],
            "support_sample_first_pct": support["sample_first_cv_relative_rmse_pct"],
            "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
            "support_peak_rmse_K": support["peak_rmse_K"],
            "support_source_rmse_K": support["source_rmse_K"],
            "support_background_rmse_K": support["background_rmse_K"],
            "support_interface_drop_rmse_K": support["interface_drop_rmse_K"],
            "full_point_global_pct": full["point_global_true_rms_relative_rmse_pct"],
            "full_sample_first_pct": full["sample_first_cv_relative_rmse_pct"],
            "full_raw_cv_rmse_K": full["raw_cv_weighted_rmse_K"],
            "full_peak_rmse_K": full["peak_rmse_K"],
            "full_source_rmse_K": full["source_rmse_K"],
            "full_background_rmse_K": full["background_rmse_K"],
            "full_interface_drop_rmse_K": full["interface_drop_rmse_K"],
            "oracle_full_point_global_pct": floor["point_global_true_rms_relative_rmse_pct"],
            "oracle_full_raw_cv_rmse_K": floor["raw_cv_weighted_rmse_K"],
            "gpu_replay_rmse_K": payload["fixed_input_gpu_replay"]["prediction"]["rmse_K"],
            "gpu_replay_max_abs_K": payload["fixed_input_gpu_replay"]["prediction"]["max_abs_error_K"],
            "end_to_end_seconds": payload["runtime"]["end_to_end_seconds"],
            "peak_rss_bytes": payload["runtime"]["process_peak_rss_bytes"],
            "peak_vram_bytes": payload["runtime"]["device_memory"].get("peak_bytes_in_use"),
            "result_sha256": _sha256(args.artifact_root / f"resolution_{resolution}.json"),
        })
    csv_path = args.artifact_root / "anchor_high_n_valid32_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_derived_high_n_valid32_closeout_v1",
        "status": "passed",
        "rows": rows,
        "preflight": {"path": str(args.artifact_root / "actual_data_preflight.json"),
                      "sha256": _sha256(args.artifact_root / "actual_data_preflight.json")},
        "summary_csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        "binding": {"path": str(args.binding), "sha256": _sha256(args.binding), "modified": False},
        "execution_order": list(MANDATORY_RESOLUTIONS),
        "accuracy_used_as_gate": False,
        "model_error_and_sampling_floor_reported_separately": True,
        "role_contract": {"accessed_roles": ["valid_iid"], "test_accessed": False,
                          "sealed_accessed": False, "training_executed": False,
                          "checkpoint_modified": False, "three_seed_valid128_executed": False,
                          "resolution_32768_executed": False},
    }
    json_path = args.artifact_root / "anchor_high_n_valid32_closeout.json"
    _write_json(json_path, payload)
    md_path = args.artifact_root / "anchor_high_n_valid32_closeout.md"
    lines = [
        "# P1i Anchor-derived High-N valid32 closeout", "",
        "Implementation hard gates passed at 4096 before 8192/16384 were launched. "
        "Accuracy was reported but never used as a gate or tuning signal.", "",
        "| N | support PG % | support SF % | support raw K | full PG % | full raw K | oracle floor PG % | replay RMSE K | E2E s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['support_point_global_pct']:.6f} | "
            f"{row['support_sample_first_pct']:.6f} | {row['support_raw_cv_rmse_K']:.6f} | "
            f"{row['full_point_global_pct']:.6f} | {row['full_raw_cv_rmse_K']:.6f} | "
            f"{row['oracle_full_point_global_pct']:.6f} | {row['gpu_replay_rmse_K']:.9f} | "
            f"{row['end_to_end_seconds']:.3f} |"
        )
    lines += ["", "Roles: valid_iid only; test/sealed closed; no training; checkpoint and frozen binding unchanged.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload["closeout_markdown"] = {"path": str(md_path), "sha256": _sha256(md_path)}
    _write_json(json_path, payload)
    return payload


def orchestrate(args: argparse.Namespace) -> int:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    prepare_actual_data(args)
    execution = []
    for resolution in MANDATORY_RESOLUTIONS:
        cpu_command = _cpu_audit_command(args, resolution)
        cpu_log_path = args.artifact_root / f"resolution_{resolution}_cpu_cache_audit.log"
        cpu_environment = dict(os.environ)
        cpu_environment["JAX_PLATFORMS"] = "cpu"
        cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
        with cpu_log_path.open("w", encoding="utf-8") as log:
            cpu_process = subprocess.run(
                cpu_command, stdout=log, stderr=subprocess.STDOUT, text=True,
                env=cpu_environment,
            )
        if cpu_process.returncode != 0:
            _write_json(args.artifact_root / "execution_state.json", {
                "status": "failed", "execution": execution,
                "failed_cpu_audit_resolution": resolution,
                "cpu_audit_log": str(cpu_log_path),
                "higher_resolutions_started_after_4096_failure": False,
            })
            raise HighNDevelopmentError(
                f"N={resolution} deterministic CPU cache audit failed; see {cpu_log_path}"
            )
        command = _subprocess_command(args, resolution)
        log_path = args.artifact_root / f"resolution_{resolution}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        execution.append({"resolution": resolution,
                          "cpu_cache_audit_command": cpu_command,
                          "cpu_cache_audit_returncode": cpu_process.returncode,
                          "cpu_cache_audit_log": str(cpu_log_path),
                          "cpu_cache_audit_log_sha256": _sha256(cpu_log_path),
                          "command": command, "returncode": process.returncode,
                          "log": str(log_path), "log_sha256": _sha256(log_path),
                          "started_unix": started, "finished_unix": time.time()})
        _write_json(args.artifact_root / "execution_state.json", {
            "status": "running" if process.returncode == 0 else "failed",
            "execution": execution,
            "higher_resolutions_started_after_4096_failure": False,
        })
        if process.returncode != 0:
            raise HighNDevelopmentError(
                f"N={resolution} worker failed; higher resolutions were not started; see {log_path}"
            )
        result = json.loads((args.artifact_root / f"resolution_{resolution}.json").read_text())
        if result["status"] != "passed":
            raise HighNDevelopmentError(f"N={resolution} hard gate failed")
    result = closeout(args)
    _write_json(args.artifact_root / "execution_state.json", {
        "status": "passed", "execution": execution,
        "higher_resolutions_started_after_4096_failure": False,
        "closeout_sha256": _sha256(args.artifact_root / "anchor_high_n_valid32_closeout.json"),
    })
    print(json.dumps({"status": result["status"], "artifact_root": str(args.artifact_root)}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json")
    parser.add_argument("--full-fields", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--binding", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--worker-resolution", type=int, choices=MANDATORY_RESOLUTIONS)
    parser.add_argument("--cpu-cache-audit-resolution", type=int, choices=MANDATORY_RESOLUTIONS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binding = _binding(args)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_passed", "mandatory_order": list(MANDATORY_RESOLUTIONS),
            "valid32_count": binding["development_subset"]["count"],
            "selection_rule": binding["development_subset"]["selection_rule"],
            "selection_seed": binding["nested_support"]["selection_seed"],
            "checkpoint": {"config_id": CONFIG_ID, "epoch": CHECKPOINT_EPOCH, "sha256": CHECKPOINT_SHA256},
            "forbidden": ["training", "test", "sealed", "three_seed_valid128", "32768"],
        }, indent=2))
        return 0
    for name in ("dataset_root", "full_fields", "run_dir", "artifact_root"):
        if getattr(args, name) is None:
            raise HighNDevelopmentError(f"--{name.replace('_', '-')} is required")
    actual_platform = jax.devices()[0].platform
    if args.platform == "gpu" and actual_platform not in {"gpu", "cuda"}:
        raise HighNDevelopmentError("GPU execution requested but JAX is not on CUDA")
    if args.platform == "cpu" and actual_platform != "cpu":
        raise HighNDevelopmentError("CPU execution requested but JAX did not select CPU")
    if args.cpu_cache_audit_resolution is not None:
        deterministic_cpu_cache_audit(args, args.cpu_cache_audit_resolution)
        return 0
    if args.worker_resolution is not None:
        execute_resolution(args, args.worker_resolution)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
