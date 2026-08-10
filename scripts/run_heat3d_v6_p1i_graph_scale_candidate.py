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
import run_heat3d_v1_medium_controlled_training_export as legacy_runner  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import save_metadata  # noqa: E402
from rigno.heat3d_v6_graph_scale import (  # noqa: E402
    NATIVE_POLICY,
    Native1024PhysicalCoverageGraphBuilder,
)
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


CANDIDATES = {
    "A": {"subsample_factor": 4, "coverage_mode": "discrete_physical_coverage"},
    "B": {"subsample_factor": 8, "coverage_mode": "discrete_physical_coverage"},
    "C": {"subsample_factor": 4, "coverage_mode": NATIVE_POLICY},
    "D": {"subsample_factor": 8, "coverage_mode": NATIVE_POLICY},
    "E": {
        "regional_node_target": 256,
        "coverage_mode": "discrete_physical_coverage",
    },
}


def _resolved_policy(candidate: str, physical_node_count: int) -> dict[str, Any]:
    policy = dict(CANDIDATES[candidate])
    if candidate == "E":
        target = int(policy["regional_node_target"])
        if physical_node_count not in (1024, 4096, 8192, 16384, 32768, 240825):
            raise RuntimeError("E requires a registered N with frozen Nr=256")
        # _subsample_pointset keeps int(N / factor) nodes.  The exact rational
        # N/256 therefore preserves the frozen E capacity even when the full
        # solver grid is not divisible by 256.
        policy["subsample_factor"] = physical_node_count / target
    return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _confirmation_runtime(args: argparse.Namespace, contract: Mapping[str, Any]) -> dict[str, Any]:
    frozen = contract["checkpoints"][str(args.seed)]
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if _sha256(checkpoint_path) != frozen["sha256"]:
        raise RuntimeError("confirmation checkpoint SHA drifted")
    checkpoint = legacy_runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != int(frozen["epoch"]):
        raise RuntimeError("confirmation checkpoint epoch drifted")
    run_config_path = args.run_dir / "run_config.json"
    if _sha256(run_config_path) != frozen["run_config_sha256"]:
        raise RuntimeError("confirmation run_config SHA drifted")
    run_config = json.loads(run_config_path.read_text())
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    checkpoint = dict(checkpoint)
    checkpoint["train_only_normalization"] = stats
    highn.install_checkpoint_feature_hooks(stats)
    standardizer = run_config["global_context"]["standardizer"]
    if standardizer.get("fit_population") != "train_only" or int(standardizer.get("fit_sample_count", -1)) != 768:
        raise RuntimeError("confirmation Global Context standardizer drifted")
    model_config = legacy_runner._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    legacy_runner._validate_model_config(model_config)
    graph_config = dict(run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config = dict(Heat3DGraphBuilder(**graph_config).config)
    return {
        "checkpoint_path": checkpoint_path, "checkpoint": checkpoint,
        "run_config_path": run_config_path, "run_config": run_config,
        "stats": stats, "model_config": model_config, "graph_config": graph_config,
    }


def _native_cache(
    path: Path, *, anchors: list[Any], runtime: Mapping[str, Any], checkpoint_sha: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
            if ids != [anchor.sample_id for anchor in anchors]:
                raise RuntimeError("native cache sample order drifted")
            if str(np.asarray(payload["checkpoint_sha256"]).item()) != checkpoint_sha:
                raise RuntimeError("native cache checkpoint drifted")
            predictions = np.asarray(payload["predictions_K"], dtype=np.float64)
            scales = np.asarray(payload["predicted_scales"], dtype=np.float64)
    else:
        builder_rows = []
        graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
        for anchor in anchors:
            builder = Heat3DGraphBuilder(**dict(runtime["graph_config"]))
            metadata = builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor, runtime["stats"]), key=graph_key
            )
            builder_rows.append((builder, metadata))
        edge_targets = {}
        for field in qualification.EDGE_FIELDS:
            values = [getattr(metadata, field) for _, metadata in builder_rows]
            edge_targets[field] = None if all(value is None for value in values) else max(
                int(value.shape[1]) for value in values if value is not None
            )
        model = GraphNeuralOperator(**runtime["model_config"])
        params = highn.runner._device_params(runtime["checkpoint"]["params"])
        compiled = jax.jit(lambda model_params, group: highn.runner._model_apply(model, model_params, group))
        predictions, scales = [], []
        for anchor, (builder, metadata) in zip(anchors, builder_rows, strict=True):
            group = highn._prepare_group(
                example=anchor, anchor=anchor, runtime=runtime, builder=builder,
                metadata=metadata, edge_targets=edge_targets,
            )
            output = compiled(params, highn._model_group(group))
            jax.block_until_ready(output["raw_temperature"])
            predictions.append(np.asarray(output["raw_temperature"], dtype=np.float64)[0, 0, :, 0])
            scales.append(float(np.asarray(output["s_hat"], dtype=np.float64).reshape(-1)[0]))
        predictions = np.asarray(predictions)
        scales = np.asarray(scales)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle, sample_ids=np.asarray([anchor.sample_id for anchor in anchors]),
                predictions_K=predictions, predicted_scales=scales,
                checkpoint_sha256=np.asarray(checkpoint_sha),
            )
    return (
        dict(zip([anchor.sample_id for anchor in anchors], predictions, strict=True)),
        dict(zip([anchor.sample_id for anchor in anchors], map(float, scales), strict=True)),
    )


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
    physical_node_count: int,
) -> Any:
    policy = _resolved_policy(candidate, physical_node_count)
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
    full_grid = args.resolution == 240825
    full_grid_contract = (
        json.loads(args.full_grid_protocol.read_text())
        if args.full_grid_protocol is not None else None
    )
    confirmation = policy_contract["status"] == "frozen_after_E_no_go_before_confirmation"
    resolution_closeout = policy_contract["status"] == "preregistered_before_graph_resolution_closeout"
    allowed_status = (
        "frozen_after_E_no_go_before_confirmation" if confirmation
        else "preregistered_before_graph_resolution_closeout" if resolution_closeout
        else "preregistered_before_E_execution" if args.candidate == "E"
        else "preregistered_before_candidate_execution"
    )
    if not full_grid and policy_contract["status"] != allowed_status:
        raise RuntimeError("graph-scale policy contract is not preregistered for candidate")
    if full_grid:
        if full_grid_contract is None or full_grid_contract.get("status") != "preregistered_before_full_grid_execution":
            raise RuntimeError("full-grid execution protocol is not preregistered")
        if args.candidate not in full_grid_contract["policies"]:
            raise RuntimeError("candidate is not registered for full-grid execution")
        registered = full_grid_contract["policies"][args.candidate]
        resolved = _resolved_policy(args.candidate, args.resolution)
        if (
            int(registered["regional_node_count"]) != int(args.resolution / resolved["subsample_factor"])
            or registered["coverage_mode"] != resolved["coverage_mode"]
        ):
            raise RuntimeError("full-grid policy definition drifted")
    if args.candidate == "E" and not resolution_closeout and not full_grid:
        registered = policy_contract["policies"]["E"]
        if (
            registered["regional_node_count"] != 256
            or registered["coverage_mode"] != "discrete_physical_coverage"
            or args.resolution not in policy_contract["resolutions"]
        ):
            raise RuntimeError("E scientific contract drifted")
    if args.candidate not in CANDIDATES or args.resolution not in (1024, 4096, 8192, 16384, 32768, 240825):
        raise RuntimeError("unregistered candidate or resolution")
    if full_grid:
        if args.seed != 0 or args.timing_only:
            raise RuntimeError("full-grid execution requires seed0 accuracy plus timing")
    elif resolution_closeout:
        registered = policy_contract["policies"].get(args.candidate)
        cell = {"policy": args.candidate, "resolution": args.resolution}
        if registered is None or cell not in policy_contract["new_execution_cells"]:
            raise RuntimeError("cell is not preregistered for graph-resolution closeout")
        resolved = _resolved_policy(args.candidate, args.resolution)
        if (
            resolved["subsample_factor"] != registered["resolved_subsample_factor"][str(args.resolution)]
            or resolved["coverage_mode"] != registered["coverage_mode"]
        ):
            raise RuntimeError("graph-resolution policy definition drifted")
        if args.timing_only or args.seed != 0:
            raise RuntimeError("graph-resolution closeout requires seed0 accuracy plus timing")
    elif confirmation:
        if args.candidate not in policy_contract["policies"] or args.resolution not in policy_contract["resolutions"]:
            raise RuntimeError("unregistered confirmation cell")
        if args.timing_only:
            raise RuntimeError("confirmation requires accuracy and timing")
    elif args.candidate == "A":
        if not args.timing_only or args.timing_amendment is None:
            raise RuntimeError("A may only run under the timing-only amendment")
        amendment = json.loads(args.timing_amendment.read_text())
        if amendment["status"] != "preregistered_before_a_timing_only_execution":
            raise RuntimeError("A timing amendment is not preregistered")
        scope = amendment["authorized_scope"]
        if args.resolution not in scope["resolutions"] or not scope["timing_only"]:
            raise RuntimeError("A timing-only resolution is not authorized")
    elif args.timing_only:
        raise RuntimeError("timing-only mode is registered only for baseline A")
    binding = highn._binding(args)
    if confirmation:
        runtime = _confirmation_runtime(args, policy_contract)
        frozen_checkpoint = policy_contract["checkpoints"][str(args.seed)]
    else:
        highn._protocol_amendment(args)
        runtime = highn._checkpoint_runtime(args)
        frozen_checkpoint = {
            "config_id": highn.CONFIG_ID, "epoch": highn.CHECKPOINT_EPOCH,
            "sha256": highn.CHECKPOINT_SHA256,
        }
    dataset = highn._dataset(args)
    full, archive_lookup = highn._full_shared(args)
    preflight = json.loads((args.baseline_artifact_root / "actual_data_preflight.json").read_text())
    if confirmation:
        expected = sorted(
            dataset.split_ids["valid_iid"], key=lambda sample_id: hashlib.sha256(sample_id.encode()).hexdigest()
        )[32:]
        if preflight["sample_ids"] != expected or len(expected) != 96:
            raise RuntimeError("confirmation remaining-valid96 selection drifted")
        index = dataset.sample_index_by_id()
        anchors = [dataset[index[sample_id]] for sample_id in expected]
    else:
        anchors = highn._valid_examples(dataset, binding)
        if args.sample_count is not None:
            anchors = anchors[: args.sample_count]
    supports_by_id = {
        row["sample_id"]: row for row in preflight.get("supports", {}).get(str(args.resolution), [])
    }
    if confirmation:
        baseline_result = None
        baseline_maps = {
            row["sample_id"]: Path(row["cache_file"])
            for row in preflight["reconstruction_maps"][str(args.resolution)]
        }
        native_prediction, anchor_scales = _native_cache(
            args.native_cache, anchors=anchors, runtime=runtime,
            checkpoint_sha=frozen_checkpoint["sha256"],
        )
    else:
        baseline_result = None if full_grid else json.loads(
            (args.baseline_artifact_root / f"resolution_{args.resolution}.json").read_text()
        )
        baseline_maps = {} if full_grid else {
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
        if native_ids[: len(anchors)] != [anchor.sample_id for anchor in anchors]:
            raise RuntimeError("native prediction order drifted")
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    examples, support_payloads, boundaries_by_id, anchor_indices_by_id = [], [], {}, {}
    for anchor in anchors:
        anchor_indices, anchor_maximum = highn._anchor_indices(
            anchor, full["coords"],
            float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        if anchor_maximum > float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]):
            raise RuntimeError("native1024 anchor coordinate replay drifted")
        anchor_indices_by_id[anchor.sample_id] = anchor_indices
        if args.resolution == 1024:
            features = np.asarray(anchor.condition.condition_features, dtype=np.float64)
            support = {
                "selected_indices": anchor_indices,
                "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
                "k_xyz": features[:, :3],
                "q_W_m3": features[:, 3],
                "layer_id": np.asarray(full["layer"][anchor_indices], dtype=np.int32),
            }
        elif full_grid:
            physics_path = Path(next(
                row for row in preflight["samples"] if row["sample_id"] == anchor.sample_id
            )["physics_cache_file"])
            with np.load(physics_path, allow_pickle=False) as physics:
                support = {
                    "selected_indices": np.arange(len(full["coords"]), dtype=np.int64),
                    "operator_control_volume": np.asarray(full["cv"], dtype=np.float64),
                    "k_xyz": np.asarray(physics["k_xyz"], dtype=np.float64),
                    "q_W_m3": np.asarray(physics["q_W_m3"], dtype=np.float64),
                    "layer_id": np.asarray(full["layer"], dtype=np.int32),
                }
        else:
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
    qualification_graph_stage_rows = []
    for example, anchor in zip(examples, anchors, strict=True):
        builder = _builder(
            args.candidate, anchor=anchor, runtime=runtime, graph_key=graph_key,
            physical_node_count=len(example.condition.coords),
        )
        started = time.perf_counter()
        metadata = builder.build_metadata(
            highn.runner._graph_coords_for_example(example, runtime["stats"]), key=graph_key
        )
        jax.block_until_ready(metadata.r_rnodes)
        qualification_build_seconds.append(time.perf_counter() - started)
        qualification_graph_stage_rows.append(dict(getattr(builder.builder, "last_build_timings", {})))
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

    @jax.jit
    def direct_full_grid_apply(model_params, group, weights, anchor_scale):
        support_delta = model_core(model_params, group, weights, anchor_scale)
        return support_delta, support_delta

    # Compile outside production timing using qualification metadata and cached map.
    first_group = highn._prepare_group(
        example=examples[0], anchor=anchors[0], runtime=runtime,
        builder=qualification_builders[0], metadata=qualification_metadata[0],
        edge_targets=edge_targets,
    )
    first_group = highn._model_group(first_group)
    first_map = None if full_grid else publication._load_mapping_no_audit(baseline_maps[anchors[0].sample_id])
    first_device_map = None if full_grid else to_device_reconstruction_map(first_map)
    first_weights = jnp.asarray(examples[0].operator_point_weights, dtype=jnp.float32)
    started = time.perf_counter()
    if full_grid:
        compiled = direct_full_grid_apply(
            params, first_group, first_weights, jnp.asarray(anchor_scales[anchors[0].sample_id])
        )
    else:
        compiled = production_apply(
            params, first_group, first_weights, jnp.asarray(anchor_scales[anchors[0].sample_id]),
            first_device_map.neighbor_local_indices, first_device_map.neighbor_weights,
        )
    jax.block_until_ready(compiled[1])
    compile_seconds = time.perf_counter() - started

    graph_seconds = []
    graph_stage_rows = []
    group_seconds = []
    map_load_transfer_seconds = []
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
        builder = _builder(
            args.candidate, anchor=anchor, runtime=runtime, graph_key=graph_key,
            physical_node_count=len(example.condition.coords),
        )
        phase = time.perf_counter()
        metadata = builder.build_metadata(
            highn.runner._graph_coords_for_example(example, runtime["stats"]), key=graph_key
        )
        jax.block_until_ready(metadata.r_rnodes)
        graph_seconds.append(time.perf_counter() - phase)
        graph_stage_rows.append(dict(getattr(builder.builder, "last_build_timings", {})))
        phase = time.perf_counter()
        group = highn._prepare_group(
            example=example, anchor=anchor, runtime=runtime, builder=builder,
            metadata=metadata, edge_targets=edge_targets,
        )
        group = highn._model_group(group)
        group_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        cpu_map = None if full_grid else publication._load_mapping_no_audit(baseline_maps[anchor.sample_id])
        device_map = None if full_grid else to_device_reconstruction_map(cpu_map)
        map_load_transfer_seconds.append(time.perf_counter() - phase)
        weights = jnp.asarray(example.operator_point_weights, dtype=jnp.float32)
        phase = time.perf_counter()
        if full_grid:
            support_delta, full_delta = direct_full_grid_apply(
                params, group, weights, jnp.asarray(anchor_scales[anchor.sample_id])
            )
        else:
            support_delta, full_delta = production_apply(
                params, group, weights, jnp.asarray(anchor_scales[anchor.sample_id]),
                device_map.neighbor_local_indices, device_map.neighbor_weights,
            )
        jax.block_until_ready(support_delta)
        neural_plus_reconstruction = time.perf_counter() - phase
        new_case_seconds.append(time.perf_counter() - new_started)
        support_np = np.asarray(support_delta, dtype=np.float64)
        full_np = np.asarray(full_delta, dtype=np.float64)
        if not args.timing_only:
            predictions.append(support_np)
            full_predictions.append(full_np)
        anchor_local = (
            anchor_indices_by_id[anchor.sample_id]
            if full_grid else np.arange(1024, dtype=np.int64)
        )
        response_drift.append(_diff(
            support_np[anchor_local], native_prediction[anchor.sample_id] - highn.REFERENCE_K
        ))
        anchor_k = np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64)
        anchor_q = np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64)
        anchor_feature_drift.append({
            "sample_id": anchor.sample_id,
            "delta_k": _diff(np.asarray(support["k_xyz"])[anchor_local], anchor_k),
            "delta_q": _diff(np.asarray(support["q_W_m3"])[anchor_local], anchor_q),
            "delta_cv": _diff(np.asarray(support["operator_control_volume"])[anchor_local], np.asarray(anchor.operator_point_weights)),
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
    # Compile the standalone component functions before collecting their
    # production distributions.  The combined production_apply was compiled
    # above, but JAX caches these two entry points independently.
    jax.block_until_ready(model_core(params, group, weights, scale))
    if not full_grid:
        jax.block_until_ready(device_map.reconstruct(support_delta))
    for _ in range(args.timing_repeats):
        phase = time.perf_counter()
        value = model_core(params, group, weights, scale)
        jax.block_until_ready(value)
        forward_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        if full_grid:
            value = support_delta
            jax.block_until_ready(value)
            apply_seconds.append(0.0)
        else:
            value = device_map.reconstruct(support_delta)
            jax.block_until_ready(value)
            apply_seconds.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        if full_grid:
            _, value = direct_full_grid_apply(params, group, weights, scale)
        else:
            _, value = production_apply(
                params, group, weights, scale,
                device_map.neighbor_local_indices, device_map.neighbor_weights,
            )
        jax.block_until_ready(value)
        warm_seconds.append(time.perf_counter() - phase)

    per_sample_metrics = []
    if args.timing_only:
        support_metrics = full_metrics = None
    else:
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
                support_row = highn._metric_row(
                    support_pred, truth_full[indices], np.asarray(support["operator_control_volume"]),
                    full["coords"][indices], np.asarray(support["layer_id"]), np.asarray(support["q_W_m3"]),
                )
                full_row = highn._metric_row(
                    full_pred, truth_full, full["cv"], full["coords"], full["layer"], full_q,
                )
                support_metric_rows.append(support_row)
                full_metric_rows.append(full_row)
                single = qualification.metric_accumulate([full_row], full=True)
                per_sample_metrics.append({
                    "sample_id": anchor.sample_id,
                    "full_rmse_K": single["raw_cv_weighted_rmse_K"],
                    "source_rmse_K": single["source_rmse_K"],
                    "peak_rmse_K": single["peak_rmse_K"],
                    "interface_rmse_K": single["interface_drop_rmse_K"],
                })
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
        "status": "passed_timing_only" if args.timing_only else "passed",
        "candidate": args.candidate,
        "policy": _resolved_policy(args.candidate, args.resolution),
        "resolution": args.resolution,
        "checkpoint": frozen_checkpoint,
        "sample_ids": [anchor.sample_id for anchor in anchors],
        "accuracy": ({
            "status": "not_evaluated_timing_only",
            "metrics_evaluated": False,
            "accuracy_recomputed": False,
        } if args.timing_only else {
            "support": support_metrics,
            "full_field": full_metrics,
            "oracle_reconstruction_floor_reused": ({
                "point_global_true_rms_relative_rmse_pct": 0.0,
                "raw_cv_weighted_rmse_K": 0.0,
                "reason": "query support exactly equals the 240825-node solver field",
            } if full_grid else (
                preflight["oracle_reconstruction_floor"][str(args.resolution)]
                if confirmation else baseline_result["oracle_sampling_reconstruction_floor"]
            )),
            "oracle_source": (
                str(args.full_fields) if full_grid else
                str(args.baseline_artifact_root / "actual_data_preflight.json")
                if confirmation else str(args.baseline_artifact_root / f"resolution_{args.resolution}.json")
            ),
        }),
        "per_sample_metrics": per_sample_metrics,
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
            "qualification_graph_stage": {
                key: _runtime_distribution([row[key] for row in qualification_graph_stage_rows])
                for key in qualification_graph_stage_rows[0]
            },
            "jit_compile_seconds_excluded": compile_seconds,
            "graph_construction": _runtime_distribution(graph_seconds),
            "graph_stage": {
                key: _runtime_distribution([row[key] for row in graph_stage_rows])
                for key in graph_stage_rows[0]
            },
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
            "timing_only": bool(args.timing_only),
            "metrics_evaluated": not args.timing_only,
            "prediction_artifact_saved": not args.timing_only and not args.no_save_predictions,
            "confirmation_remaining_valid96": confirmation,
            "direct_full_grid_output": full_grid,
        },
    }
    if not args.timing_only and not args.no_save_predictions:
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
    parser.add_argument("--native-predictions", type=Path)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--timing-amendment", type=Path)
    parser.add_argument("--full-grid-protocol", type=Path)
    parser.add_argument("--sample-count", type=int, choices=[1, 32])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("graph-scale candidate execution requires formal GPU backend")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.native_cache is None:
        if args.native_predictions is None:
            raise RuntimeError("native predictions or native cache path is required")
        args.native_cache = args.native_predictions
    result = execute(args)
    _write_json(args.output_dir / "result.json", result)
    summary = {"status": result["status"], "candidate": args.candidate, "resolution": args.resolution}
    if not args.timing_only:
        summary["full_pg_pct"] = result["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"]
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
