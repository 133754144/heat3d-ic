#!/usr/bin/env python3
"""P6 matched B1 runtime decomposition and GPU batch-throughput benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_p5_a4_p2r_r2p as a4  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import (  # noqa: E402
    build_reconstruction_map,
    prepare_reconstruction_domain_partition,
)
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)), "std_seconds": float(np.std(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def stack_batch(trees: list[Any]) -> Any:
    return jax.tree_util.tree_map(
        lambda *xs: np.concatenate([np.asarray(x) for x in xs], axis=0), *trees
    )


def tree_nbytes(tree: Any) -> int:
    return int(sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(tree)))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "native_padding_result", "query_padding_result",
        "p5r_closeout", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--route", choices=["native1024", "E16384", "E32768"], required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sample-count", type=int, choices=[1, 32], default=32)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("P6 requires the frozen GPU host")
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_before_execution":
        raise RuntimeError("P6 protocol is not frozen")
    route = {row["route"]: row for row in protocol["routes"]}[args.route]
    resolution = int(route["resolution"])
    runtime = p5r._runtime(args)
    binding = json.loads(args.binding.read_text())
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    if len(anchors) != 32:
        raise RuntimeError("P6 requires frozen valid32")
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    if preflight["sample_ids"] != [row.sample_id for row in anchors]:
        raise RuntimeError("P6 valid32 order drift")
    anchors = anchors[: args.sample_count]
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}
    full, _ = highn._full_shared(args)
    coords = np.asarray(full["coords"], dtype=np.float64)
    cv = np.asarray(full["cv"], dtype=np.float64)
    layer = np.asarray(full["layer"], dtype=np.int32)
    boundaries = highn._boundaries(anchors[0], float(np.min(coords[:, 2])))
    geometry = prepare_nested_query_geometry_cache(
        full_coords=coords, full_control_volume=cv, full_layer_id=layer,
        layer_boundaries_m=boundaries,
    )
    partition = prepare_reconstruction_domain_partition(
        coords=coords, layer_id=layer, boundaries=boundaries,
    )
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    anchor_targets = p5r._edge_targets(args.native_padding_result)
    query_targets = p5r._edge_targets(args.query_padding_result)
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])
    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]

    @jax.jit
    def anchor_forward(model_params: Any, anchor_group: Any) -> tuple[Any, Any]:
        anchor_output = highn.runner._model_apply(model, model_params, anchor_group)
        anchor_raw = anchor_output["raw_temperature"][:, 0, :, 0]
        anchor_scale = anchor_output["s_hat"].reshape((anchor_raw.shape[0], -1))[:, 0]
        return anchor_raw - highn.REFERENCE_K, anchor_scale

    @jax.jit
    def query_forward(model_params: Any, query_group: Any, weights: Any, anchor_scale: Any) -> Any:
        query_output = highn.runner._model_apply(model, model_params, query_group)
        delta = query_output["raw_temperature"][:, 0, :, 0] - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights, axis=1, keepdims=True)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta, axis=1))
        return delta / query_scale[:, None] * anchor_scale[:, None]

    @jax.jit
    def reconstruct_batch(values: Any, indices: Any, weights: Any) -> Any:
        gathered = values[jnp.arange(values.shape[0])[:, None, None], indices]
        return jnp.sum(gathered * weights.astype(values.dtype), axis=2)

    def builder(count: int) -> Heat3DGraphBuilder:
        config = dict(runtime["graph_config"])
        factor = 4.0 if count == 1024 else count / 256.0
        config.update(
            subsample_factor=factor, discrete_graph_backend="sparse_kdtree_v1",
            reuse_exact_p2r_for_r2p=True,
        )
        return Heat3DGraphBuilder(**config)

    # Qualification-only shape envelope.  It is outside every production
    # timing span and changes no real edge, ordering, or graph semantics.
    tracked_anchor_targets = dict(anchor_targets)
    tracked_query_targets = dict(query_targets)
    frozen_support_rows = {
        row["sample_id"]: row
        for row in preflight.get("supports", {}).get(str(resolution), [])
    }
    for anchor in anchors:
        with jax.default_device(cpu):
            anchor_metadata = builder(1024).build_metadata(
                highn.runner._graph_coords_for_example(anchor, runtime["stats"]), key=graph_key,
            )
            block(anchor_metadata)
            for field in qualification.EDGE_FIELDS:
                value = getattr(anchor_metadata, field)
                if value is not None:
                    anchor_targets[field] = max(
                        int(anchor_targets.get(field) or 0), int(np.asarray(value).shape[1])
                    )
            if resolution != 1024:
                frozen_support = highn._load_support(
                    Path(frozen_support_rows[anchor.sample_id]["support_file"])
                )
                query_example = highn._query_example(anchor, frozen_support, coords)
                query_metadata = builder(resolution).build_metadata(
                    highn.runner._graph_coords_for_example(query_example, runtime["stats"]),
                    key=graph_key,
                )
                block(query_metadata)
                for field in qualification.EDGE_FIELDS:
                    value = getattr(query_metadata, field)
                    if value is not None:
                        query_targets[field] = max(
                            int(query_targets.get(field) or 0), int(np.asarray(value).shape[1])
                        )

    def prepare_one(anchor: Any) -> dict[str, Any]:
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            full_k = np.asarray(physics["k_xyz"], dtype=np.float64)
            full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
        anchor_indices, distance = highn._anchor_indices(
            anchor, coords,
            float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        anchor_support = {
            "selected_indices": anchor_indices,
            "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
            "k_xyz": np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
            "q_W_m3": np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
            "layer_id": layer[anchor_indices],
        }
        total_started = time.perf_counter()
        phase = time.perf_counter()
        if resolution == 1024:
            selected = anchor_indices
            selected_cv = anchor_support["operator_control_volume"]
        else:
            selected, _ = deterministic_nested_query_prefix(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
                target_count=resolution, geometry_cache=geometry,
            )
            selected_cv, _ = conservative_selected_control_volume(
                full_coords=coords, full_control_volume=cv, full_layer_id=layer,
                selected_indices=selected,
            )
        support_s = time.perf_counter() - phase
        query_support = {
            "selected_indices": selected, "operator_control_volume": selected_cv,
            "k_xyz": anchor_support["k_xyz"] if resolution == 1024 else full_k[selected],
            "q_W_m3": anchor_support["q_W_m3"] if resolution == 1024 else full_q[selected],
            "layer_id": layer[selected],
        }
        if distance != 0.0:
            raise RuntimeError("anchor coordinate drift")
        anchor_example = highn._query_example(anchor, anchor_support, coords)
        query_example = anchor_example if resolution == 1024 else highn._query_example(anchor, query_support, coords)
        anchor_builder = builder(1024)
        query_builder = anchor_builder if resolution == 1024 else builder(resolution)
        with jax.default_device(cpu):
            phase = time.perf_counter()
            anchor_metadata = anchor_builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor_example, runtime["stats"]), key=graph_key,
            )
            block(anchor_metadata)
            anchor_graph_s = time.perf_counter() - phase
            if resolution == 1024:
                query_metadata = anchor_metadata
                query_graph_s = 0.0
            else:
                phase = time.perf_counter()
                query_metadata = query_builder.build_metadata(
                    highn.runner._graph_coords_for_example(query_example, runtime["stats"]), key=graph_key,
                )
                block(query_metadata)
                query_graph_s = time.perf_counter() - phase
            phase = time.perf_counter()
            anchor_host = host_tree(highn._model_group(highn._prepare_group(
                example=anchor_example, anchor=anchor, runtime=runtime, builder=anchor_builder,
                metadata=anchor_metadata,
                edge_targets=p5r._compatible_targets(anchor_targets, anchor_metadata),
            )))
            anchor_pack_s = time.perf_counter() - phase
            if resolution == 1024:
                query_host = anchor_host
                query_pack_s = 0.0
            else:
                phase = time.perf_counter()
                query_host = host_tree(highn._model_group(highn._prepare_group(
                    example=query_example, anchor=anchor, runtime=runtime, builder=query_builder,
                    metadata=query_metadata,
                    edge_targets=p5r._compatible_targets(query_targets, query_metadata),
                )))
                query_pack_s = time.perf_counter() - phase
        phase = time.perf_counter()
        mapping, _ = build_reconstruction_map(
            coords=coords, layer_id=layer, boundaries=boundaries,
            support_indices=selected, empty_domain_fallback="same_layer",
            prepared_partition=partition, query_workers=-1,
        )
        map_s = time.perf_counter() - phase
        host_weights = np.asarray(selected_cv, dtype=np.float32)[None, :]
        map_indices = np.asarray(mapping.neighbor_local_indices, dtype=np.int32)[None, :, :]
        map_weights = np.asarray(mapping.neighbor_weights, dtype=np.float64)[None, :, :]
        phase = time.perf_counter()
        device = jax.device_put(
            (anchor_host, query_host, host_weights, map_indices, map_weights), gpu
        )
        h2d_enqueue_s = time.perf_counter() - phase
        phase = time.perf_counter(); block(device); h2d_sync_s = time.perf_counter() - phase
        anchor_device, query_device, device_weights, device_indices, device_map_weights = device
        phase = time.perf_counter()
        anchor_delta, anchor_scale = anchor_forward(params, anchor_device)
        block((anchor_delta, anchor_scale)); anchor_forward_s = time.perf_counter() - phase
        if resolution == 1024:
            support_delta = anchor_delta
            query_forward_s = 0.0
        else:
            phase = time.perf_counter()
            support_delta = query_forward(params, query_device, device_weights, anchor_scale)
            block(support_delta); query_forward_s = time.perf_counter() - phase
        phase = time.perf_counter()
        full_delta = reconstruct_batch(support_delta, device_indices, device_map_weights)
        block(full_delta); reconstruction_s = time.perf_counter() - phase
        total_s = time.perf_counter() - total_started
        return {
            "sample_id": anchor.sample_id,
            "anchor_host": anchor_host, "query_host": query_host,
            "host_weights": host_weights, "map_indices": map_indices,
            "map_weights": map_weights, "anchor_device": anchor_device,
            "query_device": query_device, "device_weights": device_weights,
            "device_indices": device_indices, "device_map_weights": device_map_weights,
            "stages": {
                "support_plus_cv": support_s, "anchor_graph": anchor_graph_s,
                "query_graph": query_graph_s, "reconstruction_map": map_s,
                "anchor_group_pack": anchor_pack_s, "query_group_pack": query_pack_s,
                "h2d_enqueue": h2d_enqueue_s, "h2d_sync": h2d_sync_s,
                "anchor_forward": anchor_forward_s,
                "query_forward": query_forward_s,
                "reconstruction_apply": reconstruction_s,
                "matched_continuous_e2e": total_s,
            },
            "graph": {
                "anchor_hash": a4._canonical_hash(anchor_metadata),
                "query_hash": a4._canonical_hash(query_metadata),
                "anchor_query_reused": resolution == 1024,
                "regional_nodes": int(np.asarray(query_metadata.x_rnodes).shape[1] - 1),
                "p2r_edges": int(np.asarray(query_metadata.p2r_edge_indices).shape[1]),
                "r2r_edges": int(np.asarray(query_metadata.r2r_edge_indices).shape[1]),
            },
            "support_delta": support_delta, "full_delta": full_delta,
        }

    # One outside-timing preparation and compile.  Smoke uses the same path.
    warm = prepare_one(anchors[0])
    warm_support, warm_scale = anchor_forward(params, warm["anchor_device"])
    if resolution != 1024:
        warm_support = query_forward(
            params, warm["query_device"], warm["device_weights"], warm_scale
        )
    block(reconstruct_batch(warm_support, warm["device_indices"], warm["device_map_weights"]))

    prepared = [prepare_one(anchor) for anchor in anchors]
    native_reuse_gate = True
    if resolution == 1024:
        first = prepared[0]
        native_reuse_gate = (
            first["graph"]["anchor_hash"] == first["graph"]["query_hash"]
            and highn._tree_sha256(first["anchor_host"]) == highn._tree_sha256(first["query_host"])
        )
        if not native_reuse_gate:
            raise RuntimeError("native1024 anchor/query reuse exact gate failed")

    stage_keys = list(prepared[0]["stages"])
    timing = {key: dist([row["stages"][key] for row in prepared]) for key in stage_keys}

    # Same-input replay uses sample 0 with everything resident.
    replay_values = []
    for _ in range(args.repeats):
        phase = time.perf_counter()
        support, scale = anchor_forward(params, warm["anchor_device"])
        if resolution != 1024:
            support = query_forward(params, warm["query_device"], warm["device_weights"], scale)
        full_value = reconstruct_batch(support, warm["device_indices"], warm["device_map_weights"])
        block(full_value); replay_values.append(time.perf_counter() - phase)

    # Batch benchmark: prepared resident, and host-prepared streamed H2D + compute.
    batch_rows = []
    memory = candidate.publication._device_memory()
    limit = int(memory.get("bytes_limit", 0))
    base_bytes = sum(
        tree_nbytes(row["anchor_host"]) + tree_nbytes(row["query_host"])
        + row["host_weights"].nbytes + row["map_indices"].nbytes + row["map_weights"].nbytes
        for row in prepared[:1]
    )
    t1 = None
    for batch_size in protocol["batch_contract"]["sizes"]:
        if batch_size > len(prepared):
            continue
        if limit and base_bytes * batch_size > 0.8 * limit:
            batch_rows.append({
                "batch_size": batch_size, "status": "skipped_memory_feasibility",
                "estimated_input_bytes": base_bytes * batch_size,
            })
            continue
        subset = prepared[:batch_size]
        anchor_host = stack_batch([row["anchor_host"] for row in subset])
        query_host = anchor_host if resolution == 1024 else stack_batch([row["query_host"] for row in subset])
        weights_host = np.concatenate([row["host_weights"] for row in subset], axis=0)
        indices_host = np.concatenate([row["map_indices"] for row in subset], axis=0)
        map_weights_host = np.concatenate([row["map_weights"] for row in subset], axis=0)
        device_batch = jax.device_put(
            (anchor_host, query_host, weights_host, indices_host, map_weights_host), gpu
        )
        block(device_batch)
        ad, qd, wd, idd, mwd = device_batch
        support, scale = anchor_forward(params, ad)
        if resolution != 1024:
            support = query_forward(params, qd, wd, scale)
        block(support)
        full_value = reconstruct_batch(support, idd, mwd); block(full_value)
        forward_values = []; prepared_values = []; streamed_values = []
        for _ in range(args.repeats):
            phase = time.perf_counter(); support, scale = anchor_forward(params, ad)
            if resolution != 1024:
                support = query_forward(params, qd, wd, scale)
            block(support); forward_values.append(time.perf_counter() - phase)
            phase = time.perf_counter(); support, scale = anchor_forward(params, ad)
            if resolution != 1024:
                support = query_forward(params, qd, wd, scale)
            full_value = reconstruct_batch(support, idd, mwd); block(full_value)
            prepared_values.append(time.perf_counter() - phase)
            phase = time.perf_counter()
            streamed = jax.device_put((anchor_host, query_host, weights_host, indices_host, map_weights_host), gpu)
            block(streamed); sad, sqd, swd, sidd, smwd = streamed
            support, scale = anchor_forward(params, sad)
            if resolution != 1024:
                support = query_forward(params, sqd, swd, scale)
            full_value = reconstruct_batch(support, sidd, smwd); block(full_value)
            streamed_values.append(time.perf_counter() - phase)
        forward_stats = dist(forward_values); prepared_stats = dist(prepared_values)
        streamed_stats = dist(streamed_values)
        batch_wall = prepared_stats["median_seconds"]
        if t1 is None:
            t1 = batch_wall
        batch_rows.append({
            "batch_size": batch_size, "status": "passed",
            "resident_forward_only": forward_stats,
            "prepared_group_steady_inference": prepared_stats,
            "streamed_prepared_host_batch": streamed_stats,
            "batch_wall_seconds": batch_wall,
            "samples_per_second": batch_size / batch_wall,
            "average_per_case_seconds": batch_wall / batch_size,
            "marginal_per_case_seconds": None if batch_size == 1 else (batch_wall - t1) / (batch_size - 1),
            "peak_vram_bytes": int(candidate.publication._device_memory().get("peak_bytes_in_use", 0)),
        })

    p5r_closeout = json.loads(args.p5r_closeout.read_text())
    accuracy_row = next(row for row in p5r_closeout["rows"] if row["N"] == resolution and (
        args.route == "native1024" or row["route"].startswith("E")
    ))
    result = {
        "schema_version": "heat3d_v6_p1i_p6_runtime_cell_v1",
        "status": "passed" if args.sample_count == 32 else "passed_smoke",
        "route": route, "sample_count": args.sample_count,
        "protocol_sha256": sha256(args.protocol),
        "checkpoint_sha256": args.checkpoint_sha256,
        "accuracy_reused_from_p5r": accuracy_row,
        "runtime": {
            "fresh_sample": timing,
            "same_shape_new_sample": {
                key: dist([row["stages"][key] for row in prepared[1:]])
                for key in stage_keys
            } if len(prepared) > 1 else None,
            "same_input_replay": dist(replay_values),
        },
        "native1024_reuse_exact_gate": native_reuse_gate,
        "jit_padding_envelope": {
            "semantics": "qualification_only_shape_max; real_edges_unchanged; outside_timing",
            "tracked_anchor": tracked_anchor_targets,
            "actual_anchor": anchor_targets,
            "tracked_query": tracked_query_targets,
            "actual_query": query_targets,
        },
        "batch": batch_rows,
        "memory": candidate.publication._device_memory(),
        "samples": [
            {"sample_id": row["sample_id"], "stages": row["stages"], "graph": row["graph"]}
            for row in prepared
        ],
        "role_contract": protocol["role_contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "route": args.route,
        "fresh_median_s": timing["matched_continuous_e2e"]["median_seconds"],
        "max_batch": max((row["batch_size"] for row in batch_rows if row["status"] == "passed"), default=0),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
