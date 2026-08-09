#!/usr/bin/env python3
"""Run one preregistered P1i graph-scale candidate on frozen valid32."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_publication_gpu_pipeline as publication  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import save_metadata  # noqa: E402
from rigno.heat3d_v6_graph_scale import (  # noqa: E402
    NATIVE_POLICY,
    Native1024PhysicalCoverageGraphBuilder,
)
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


CANDIDATES = {
    "B": {"subsample_factor": 8, "coverage_mode": "discrete_physical_coverage"},
    "C": {"subsample_factor": 4, "coverage_mode": NATIVE_POLICY},
    "D": {"subsample_factor": 8, "coverage_mode": NATIVE_POLICY},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dist(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p5": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _diff(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _builder(
    candidate: str,
    *,
    anchor: Any,
    runtime: Mapping[str, Any],
    graph_key: Any,
) -> Any:
    policy = CANDIDATES[candidate]
    graph_config = dict(runtime["graph_config"])
    graph_config["subsample_factor"] = policy["subsample_factor"]
    if policy["coverage_mode"] == "discrete_physical_coverage":
        return Heat3DGraphBuilder(**graph_config)
    anchor_coords = highn.runner._graph_coords_for_example(anchor, runtime["stats"])
    return Native1024PhysicalCoverageGraphBuilder(
        anchor_coords=anchor_coords,
        graph_config=graph_config,
        graph_key=graph_key,
        regional_subsample_factor=policy["subsample_factor"],
    )


def _real_edges(value: Any, n_sender: int, n_receiver: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(value)[0].astype(np.int64)
    return edges[(edges[:, 0] < n_sender) & (edges[:, 1] < n_receiver)]


def _graph_diagnostics(
    metadata: Any,
    *,
    coords: np.ndarray,
    q: np.ndarray,
    boundaries: np.ndarray,
    native_reference: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    x_p = np.asarray(metadata.x_pnodes_inp)[0, :-1]
    x_r = np.asarray(metadata.x_rnodes)[0, :-1]
    target_radius = np.asarray(metadata.r_rnodes)[0, :-1]
    n_p, n_r = len(x_p), len(x_r)
    p2r = _real_edges(metadata.p2r_edge_indices, n_p, n_r)
    r2r = _real_edges(metadata.r2r_edge_indices, n_r, n_r)
    r2p = (
        _real_edges(metadata.r2p_edge_indices, n_r, n_p)
        if metadata.r2p_edge_indices is not None
        else np.flip(p2r, axis=1)
    )
    lower, upper = np.min(coords, axis=0), np.max(coords, axis=0)
    p_phys = (x_p + 1.0) * 0.5 * (upper - lower) + lower
    r_phys = (x_r + 1.0) * 0.5 * (upper - lower) + lower
    p2r_norm = np.linalg.norm(x_p[p2r[:, 0]] - x_r[p2r[:, 1]], axis=1)
    r2r_norm = np.linalg.norm(x_r[r2r[:, 0]] - x_r[r2r[:, 1]], axis=1)
    r2p_norm = np.linalg.norm(x_r[r2p[:, 0]] - x_p[r2p[:, 1]], axis=1)
    p2r_phys = np.linalg.norm(p_phys[p2r[:, 0]] - r_phys[p2r[:, 1]], axis=1)
    r2r_phys = np.linalg.norm(r_phys[r2r[:, 0]] - r_phys[r2r[:, 1]], axis=1)
    r2p_phys = np.linalg.norm(r_phys[r2p[:, 0]] - p_phys[r2p[:, 1]], axis=1)
    p2r_p = np.bincount(p2r[:, 0], minlength=n_p)
    p2r_r = np.bincount(p2r[:, 1], minlength=n_r)
    r2r_r = np.bincount(r2r[:, 0], minlength=n_r)
    r2p_r = np.bincount(r2p[:, 0], minlength=n_r)
    r2p_p = np.bincount(r2p[:, 1], minlength=n_p)
    realized_norm = np.zeros(n_r, dtype=np.float64)
    realized_phys = np.zeros(n_r, dtype=np.float64)
    np.maximum.at(realized_norm, p2r[:, 1], p2r_norm)
    np.maximum.at(realized_phys, p2r[:, 1], p2r_phys)
    source = np.asarray(q).reshape(-1) > 0.0
    interface = np.any(
        np.isclose(coords[:, 2, None], boundaries[None, 1:-1], atol=1.0e-15), axis=1
    )
    categories = np.full(n_p, "background", dtype="U10")
    categories[interface] = "interface"
    categories[source] = "source"
    partition = {}
    for name in ("source", "interface", "background"):
        mask = categories == name
        partition[name] = {
            "node_count": int(np.count_nonzero(mask)),
            "p2r_physical_degree": _dist(p2r_p[mask]),
            "r2p_physical_degree": _dist(r2p_p[mask]),
        }
    adjacency = coo_matrix(
        (np.ones(len(r2r), dtype=np.uint8), (r2r[:, 0], r2r[:, 1])),
        shape=(n_r, n_r),
    )
    components = int(connected_components(adjacency, directed=False, return_labels=False))
    reference = None
    if native_reference is not None:
        reference = {
            "regional_node_count": int(len(native_reference["rnodes"])),
            "normalized_radius": _dist(native_reference["radii"]),
        }
    return {
        "regional_node_count": n_r,
        "edge_count": {"p2r": int(len(p2r)), "r2r": int(len(r2r)), "r2p": int(len(r2p))},
        "degree": {
            "p2r_physical": _dist(p2r_p), "p2r_regional": _dist(p2r_r),
            "r2r_regional": _dist(r2r_r), "r2p_regional": _dist(r2p_r),
            "r2p_physical": _dist(r2p_p),
        },
        "partition": partition,
        "target_normalized_radius": _dist(target_radius),
        "realized_normalized_radius": _dist(realized_norm),
        "physical_coverage_radius_m": _dist(realized_phys),
        "normalized_edge_length": {"p2r": _dist(p2r_norm), "r2r": _dist(r2r_norm), "r2p": _dist(r2p_norm)},
        "physical_edge_length_m": {"p2r": _dist(p2r_phys), "r2r": _dist(r2r_phys), "r2p": _dist(r2p_phys)},
        "isolated_fraction": {
            "p2r_physical": float(np.mean(p2r_p == 0)),
            "r2r_regional": float(np.mean(r2r_r == 0)),
            "r2p_physical": float(np.mean(r2p_p == 0)),
        },
        "undercovered_fraction_degree_lt_1": float(np.mean(p2r_p < 1)),
        "r2r_connected_components": components,
        "effective_physical_receptive_field_proxy_m": float(
            np.percentile(p2r_phys, 95) + np.percentile(r2r_phys, 95) + np.percentile(r2p_phys, 95)
        ),
        "native1024_reference": reference,
    }


def _mean_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    summary = {
        "sample_count": len(rows),
        "regional_node_count": mean(("regional_node_count",)),
        "edge_count": {name: mean(("edge_count", name)) for name in ("p2r", "r2r", "r2p")},
        "r2r_connected_components": _dist([row["r2r_connected_components"] for row in rows]),
        "undercovered_fraction": mean(("undercovered_fraction_degree_lt_1",)),
        "effective_physical_receptive_field_proxy_m": mean(("effective_physical_receptive_field_proxy_m",)),
    }
    for name in ("p2r_physical", "p2r_regional", "r2r_regional", "r2p_regional", "r2p_physical"):
        summary.setdefault("degree", {})[name] = {
            stat: mean(("degree", name, stat)) for stat in ("mean", "p5", "median", "p95", "max")
        }
    for name in ("target_normalized_radius", "realized_normalized_radius", "physical_coverage_radius_m"):
        summary[name] = {stat: mean((name, stat)) for stat in ("mean", "p5", "median", "p95", "max")}
    for domain in ("normalized_edge_length", "physical_edge_length_m"):
        summary[domain] = {
            edge: {stat: mean((domain, edge, stat)) for stat in ("mean", "p5", "median", "p95", "max")}
            for edge in ("p2r", "r2r", "r2p")
        }
    for category in ("source", "interface", "background"):
        summary.setdefault("partition", {})[category] = {
            "node_count": mean(("partition", category, "node_count")),
            "p2r_degree": {
                stat: mean(("partition", category, "p2r_physical_degree", stat))
                for stat in ("mean", "p5", "median", "p95", "max")
            },
        }
    return summary


def _runtime_distribution(values: list[float]) -> dict[str, float | int]:
    base = publication._distribution(values)
    base["p5_seconds"] = float(np.percentile(np.asarray(values), 5))
    return base


def execute(args: argparse.Namespace) -> dict[str, Any]:
    policy_contract = json.loads(args.policy_contract.read_text())
    if policy_contract["status"] != "preregistered_before_candidate_execution":
        raise RuntimeError("graph-scale policy contract is not preregistered")
    if args.candidate not in CANDIDATES or args.resolution not in (4096, 8192, 16384, 32768):
        raise RuntimeError("unregistered candidate or resolution")
    binding = highn._binding(args)
    highn._protocol_amendment(args)
    runtime = highn._checkpoint_runtime(args)
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    full, archive_lookup = highn._full_shared(args)
    preflight = json.loads((args.baseline_artifact_root / "actual_data_preflight.json").read_text())
    supports_by_id = {
        row["sample_id"]: row for row in preflight["supports"][str(args.resolution)]
    }
    baseline_result = json.loads(
        (args.baseline_artifact_root / f"resolution_{args.resolution}.json").read_text()
    )
    baseline_maps = {
        row["sample_id"]: Path(row["cache_file"])
        for row in baseline_result["reconstruction_cache"]["samples"]
    }
    with np.load(args.native_predictions, allow_pickle=False) as payload:
        native_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
        native_prediction = {
            sample_id: np.asarray(prediction, dtype=np.float64)
            for sample_id, prediction in zip(native_ids, np.asarray(payload["predictions_K"]), strict=True)
        }
        anchor_scales = dict(zip(native_ids, map(float, np.asarray(payload["predicted_scales"])), strict=True))
    if native_ids != [anchor.sample_id for anchor in anchors]:
        raise RuntimeError("native prediction order drifted")
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    examples, support_payloads, boundaries_by_id = [], [], {}
    for anchor in anchors:
        row = supports_by_id[anchor.sample_id]
        support = highn._load_support(Path(row["support_file"]))
        examples.append(highn._query_example(anchor, support, full["coords"]))
        support_payloads.append(support)
        boundaries_by_id[anchor.sample_id] = highn._boundaries(
            anchor, float(np.min(full["coords"][:, 2]))
        )

    # Qualification pass determines only padding envelopes. No prediction is run.
    qualification_metadata, qualification_builders = [], []
    qualification_build_seconds = []
    for example, anchor in zip(examples, anchors, strict=True):
        builder = _builder(args.candidate, anchor=anchor, runtime=runtime, graph_key=graph_key)
        started = time.perf_counter()
        metadata = builder.build_metadata(
            highn.runner._graph_coords_for_example(example, runtime["stats"]), key=graph_key
        )
        jax.block_until_ready(metadata.r_rnodes)
        qualification_build_seconds.append(time.perf_counter() - started)
        qualification_metadata.append(metadata)
        qualification_builders.append(builder)
    edge_targets = {}
    for field in qualification.EDGE_FIELDS:
        values = [getattr(metadata, field) for metadata in qualification_metadata]
        edge_targets[field] = None if all(value is None for value in values) else max(
            int(value.shape[1]) for value in values if value is not None
        )

    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])

    @jax.jit
    def model_core(model_params, group, weights, anchor_scale):
        output = highn.runner._model_apply(model, model_params, group)
        raw = output["raw_temperature"][0, 0, :, 0]
        delta = raw - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * anchor_scale

    @jax.jit
    def production_apply(model_params, group, weights, anchor_scale, indices, map_weights):
        support_delta = model_core(model_params, group, weights, anchor_scale)
        full_delta = jnp.sum(
            support_delta[indices] * map_weights.astype(support_delta.dtype), axis=1
        )
        return support_delta, full_delta

    # Compile outside production timing using qualification metadata and cached map.
    first_group = highn._prepare_group(
        example=examples[0], anchor=anchors[0], runtime=runtime,
        builder=qualification_builders[0], metadata=qualification_metadata[0],
        edge_targets=edge_targets,
    )
    first_group = highn._model_group(first_group)
    first_map = publication._load_mapping_no_audit(baseline_maps[anchors[0].sample_id])
    first_device_map = to_device_reconstruction_map(first_map)
    first_weights = jnp.asarray(examples[0].operator_point_weights, dtype=jnp.float32)
    started = time.perf_counter()
    compiled = production_apply(
        params, first_group, first_weights, jnp.asarray(anchor_scales[anchors[0].sample_id]),
        first_device_map.neighbor_local_indices, first_device_map.neighbor_weights,
    )
    jax.block_until_ready(compiled[1])
    compile_seconds = time.perf_counter() - started

    graph_seconds = []
    group_seconds = []
    map_load_transfer_seconds = []
    neural_seconds = []
    reconstruction_apply_seconds = []
    new_case_seconds = []
    predictions = []
    full_predictions = []
    diagnostics = []
    response_drift = []
    anchor_feature_drift = []
    metadata_rows = []
    retained = []

    for example, anchor, support in zip(examples, anchors, support_payloads, strict=True):
        new_started = time.perf_counter()
        builder = _builder(args.candidate, anchor=anchor, runtime=runtime, graph_key=graph_key)
        phase = time.perf_counter()
        metadata = builder.build_metadata(
            highn.runner._graph_coords_for_example(example, runtime["stats"]), key=graph_key
        )
        jax.block_until_ready(metadata.r_rnodes)
        graph_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        group = highn._prepare_group(
            example=example, anchor=anchor, runtime=runtime, builder=builder,
            metadata=metadata, edge_targets=edge_targets,
        )
        group = highn._model_group(group)
        group_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        cpu_map = publication._load_mapping_no_audit(baseline_maps[anchor.sample_id])
        device_map = to_device_reconstruction_map(cpu_map)
        map_load_transfer_seconds.append(time.perf_counter() - phase)
        weights = jnp.asarray(example.operator_point_weights, dtype=jnp.float32)
        phase = time.perf_counter()
        support_delta, full_delta = production_apply(
            params, group, weights, jnp.asarray(anchor_scales[anchor.sample_id]),
            device_map.neighbor_local_indices, device_map.neighbor_weights,
        )
        jax.block_until_ready(support_delta)
        neural_plus_reconstruction = time.perf_counter() - phase
        new_case_seconds.append(time.perf_counter() - new_started)
        support_np = np.asarray(support_delta, dtype=np.float64)
        full_np = np.asarray(full_delta, dtype=np.float64)
        predictions.append(support_np)
        full_predictions.append(full_np)
        response_drift.append(_diff(support_np[:1024], native_prediction[anchor.sample_id] - highn.REFERENCE_K))
        anchor_k = np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64)
        anchor_q = np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64)
        anchor_feature_drift.append({
            "sample_id": anchor.sample_id,
            "delta_k": _diff(np.asarray(support["k_xyz"][:1024]), anchor_k),
            "delta_q": _diff(np.asarray(support["q_W_m3"][:1024]), anchor_q),
            "delta_cv": _diff(np.asarray(support["operator_control_volume"][:1024]), np.asarray(anchor.operator_point_weights)),
        })
        native_reference = getattr(builder, "native_reference", None)
        diag = _graph_diagnostics(
            metadata, coords=np.asarray(example.condition.coords),
            q=np.asarray(support["q_W_m3"]), boundaries=boundaries_by_id[anchor.sample_id],
            native_reference=native_reference,
        )
        diag["sample_id"] = anchor.sample_id
        diagnostics.append(diag)
        cache_path = args.output_dir / "graph_metadata" / f"{anchor.sample_id}.npz"
        save_metadata(cache_path, metadata)
        metadata_rows.append({"sample_id": anchor.sample_id, "path": str(cache_path), "sha256": _sha256(cache_path)})
        # Only the first case is retained for steady-state component timing.
        # Keeping all 32 device groups and 240825-node maps would inflate VRAM
        # without adding any timing samples or scientific evidence.
        if not retained:
            retained.append((group, weights, device_map, support_delta, neural_plus_reconstruction))

    # Component timing uses fixed in-memory first case and excludes all audits.
    warm_seconds = []
    forward_seconds = []
    apply_seconds = []
    group, weights, device_map, support_delta, _ = retained[0]
    scale = jnp.asarray(anchor_scales[anchors[0].sample_id])
    for _ in range(args.timing_repeats):
        phase = time.perf_counter()
        value = model_core(params, group, weights, scale)
        jax.block_until_ready(value)
        forward_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        value = device_map.reconstruct(support_delta)
        jax.block_until_ready(value)
        apply_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        _, value = production_apply(
            params, group, weights, scale,
            device_map.neighbor_local_indices, device_map.neighbor_weights,
        )
        jax.block_until_ready(value)
        warm_seconds.append(time.perf_counter() - phase)

    support_metric_rows, full_metric_rows = [], []
    with h5py.File(args.full_fields, "r") as archive:
        for example, anchor, support, support_pred, full_pred in zip(
            examples, anchors, support_payloads, predictions, full_predictions, strict=True
        ):
            truth_full = np.asarray(
                archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]], dtype=np.float64
            )
            indices = np.asarray(support["selected_indices"], dtype=np.int64)
            with np.load(
                next(row for row in preflight["samples"] if row["sample_id"] == anchor.sample_id)["physics_cache_file"],
                allow_pickle=False,
            ) as physics:
                full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            support_metric_rows.append(highn._metric_row(
                support_pred, truth_full[indices], np.asarray(support["operator_control_volume"]),
                full["coords"][indices], np.asarray(support["layer_id"]), np.asarray(support["q_W_m3"]),
            ))
            full_metric_rows.append(highn._metric_row(
                full_pred, truth_full, full["cv"], full["coords"], full["layer"], full_q,
            ))
    support_metrics = qualification.metric_accumulate(support_metric_rows, full=False)
    full_metrics = qualification.metric_accumulate(full_metric_rows, full=True)
    maximum_feature_drift = {
        name: {
            "max_abs": max(row[name]["max_abs"] for row in anchor_feature_drift),
            "max_rmse": max(row[name]["rmse"] for row in anchor_feature_drift),
        }
        for name in ("delta_k", "delta_q", "delta_cv")
    }
    graph_summary = _mean_summary(diagnostics)
    result = {
        "schema_version": "heat3d_v6_p1i_graph_scale_candidate_result_v1",
        "status": "passed",
        "candidate": args.candidate,
        "policy": CANDIDATES[args.candidate],
        "resolution": args.resolution,
        "checkpoint": {"epoch": highn.CHECKPOINT_EPOCH, "sha256": highn.CHECKPOINT_SHA256},
        "sample_ids": [anchor.sample_id for anchor in anchors],
        "accuracy": {
            "support": support_metrics,
            "full_field": full_metrics,
            "oracle_reconstruction_floor_reused": baseline_result["oracle_sampling_reconstruction_floor"],
            "oracle_source": str(args.baseline_artifact_root / f"resolution_{args.resolution}.json"),
        },
        "common_anchor_response_drift": {
            "sample_count": len(response_drift),
            "rmse_K": float(math.sqrt(np.mean([row["rmse"] ** 2 for row in response_drift]))),
            "max_abs_K": max(row["max_abs"] for row in response_drift),
        },
        "common_anchor_input_drift": maximum_feature_drift,
        "common_anchor_input_drift_interpretation": (
            "report_only_frozen_high_n_overlap_fields_vs_native1024_binary_mask_fields"
        ),
        "graph_diagnostics": graph_summary,
        "graph_diagnostics_per_sample": diagnostics,
        "timing": {
            "contract": {
                "single_sample_B1": True, "gpu_synchronized": True,
                "qualification_hash_metrics_serialization_excluded": True,
                "jit_compile_excluded": True,
            },
            "qualification_graph_build": _runtime_distribution(qualification_build_seconds),
            "jit_compile_seconds_excluded": compile_seconds,
            "graph_construction": _runtime_distribution(graph_seconds),
            "group_prepare": _runtime_distribution(group_seconds),
            "cached_reconstruction_map_load_and_h2d": _runtime_distribution(map_load_transfer_seconds),
            "neural_core": _runtime_distribution(forward_seconds),
            "reconstruction_apply_gpu": _runtime_distribution(apply_seconds),
            "warm_cache_e2e": _runtime_distribution(warm_seconds),
            "new_case_e2e": _runtime_distribution(new_case_seconds),
        },
        "device_memory": publication._device_memory(),
        "graph_metadata_artifacts": metadata_rows,
        "role_contract": {
            "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
            "training": False, "test": False, "sealed": False,
            "checkpoint_modified": False, "support_or_physics_modified": False,
        },
    }
    prediction_path = args.output_dir / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(result["sample_ids"]),
            support_deltaT_K=np.asarray(predictions),
            full_deltaT_K=np.asarray(full_predictions),
        )
    result["prediction_artifact"] = {
        "path": str(prediction_path), "sha256": _sha256(prediction_path),
        "bytes": prediction_path.stat().st_size,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--policy-contract", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--gpu-only-amendment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--baseline-artifact-root", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("graph-scale candidate execution requires formal GPU backend")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = execute(args)
    _write_json(args.output_dir / "result.json", result)
    print(json.dumps({
        "status": result["status"], "candidate": args.candidate,
        "resolution": args.resolution,
        "full_pg_pct": result["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
