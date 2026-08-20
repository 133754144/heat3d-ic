#!/usr/bin/env python3
"""Final matched valid32 service benchmark for frozen E inference routes."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)), "std_seconds": float(np.std(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "native_padding_result", "query_padding_result",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--route", choices=("E16384_reconstruction", "E240825_direct_control"), required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--resident-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "preregistered_before_execution":
        raise RuntimeError("final correction protocol is not preregistered")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("final E service benchmark requires GPU")
    state = p8.runtime_state(args)
    runtime = state["runtime"]
    checkpoint_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    if checkpoint_before != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA drift")
    resolution = 16384 if args.route.startswith("E16384") else 240825
    direct = resolution == 240825
    query_config = dict(runtime["graph_config"])
    query_config.update(
        subsample_factor=resolution / 256.0,
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    anchor_config = dict(runtime["graph_config"])
    anchor_config.update(
        subsample_factor=4.0,
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    anchors = state["anchors"]
    if len(anchors) != 32:
        raise RuntimeError("frozen valid32 drift")
    physics_memory: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for anchor in anchors:
        with np.load(state["physics"][anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            physics_memory[anchor.sample_id] = (
                np.asarray(physics["k_xyz"], dtype=np.float64),
                np.asarray(physics["q_W_m3"], dtype=np.float64),
            )
    anchor_targets = p5r._edge_targets(args.native_padding_result)
    query_targets = p5r._edge_targets(args.query_padding_result)
    graph_key = state["graph_key"]
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])

    @jax.jit
    def forward(model_params: Any, anchor_group: Any, query_group: Any,
                weights: Any, map_indices: Any, map_weights: Any) -> Any:
        anchor_result = highn.runner._model_apply(model, model_params, anchor_group)
        anchor_scale = anchor_result["s_hat"].reshape(-1)[0]
        query = highn.runner._model_apply(model, model_params, query_group)["raw_temperature"][0, 0, :, 0]
        query = query - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * query * query))
        support = query / query_scale * anchor_scale
        if direct:
            return support
        gathered = support[map_indices]
        return jnp.sum(gathered * map_weights.astype(support.dtype), axis=1)

    def prepare_host(anchor: Any) -> tuple[dict[str, Any], dict[str, float]]:
        stages: dict[str, float] = {}
        full_k, full_q = physics_memory[anchor.sample_id]
        anchor_indices, distance = highn._anchor_indices(
            anchor, state["coords"],
            float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        if distance != 0.0:
            raise RuntimeError(f"{anchor.sample_id}: anchor coordinate drift")
        phase = time.perf_counter()
        if direct:
            selected = np.arange(len(state["coords"]), dtype=np.int64)
            selected_cv = state["cv"]
        else:
            selected, _ = deterministic_nested_query_prefix(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices,
                full_q=full_q, target_count=resolution, geometry_cache=state["geometry"],
            )
            selected_cv, _ = conservative_selected_control_volume(
                full_coords=state["coords"], full_control_volume=state["cv"],
                full_layer_id=state["layer"], selected_indices=selected, query_workers=1,
            )
        stages["support_plus_cv"] = time.perf_counter() - phase
        anchor_support = {
            "selected_indices": anchor_indices,
            "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
            "k_xyz": np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
            "q_W_m3": np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
            "layer_id": state["layer"][anchor_indices],
        }
        query_support = {
            "selected_indices": selected, "operator_control_volume": selected_cv,
            "k_xyz": full_k[selected], "q_W_m3": full_q[selected],
            "layer_id": state["layer"][selected],
        }
        anchor_example = highn._query_example(anchor, anchor_support, state["coords"])
        query_example = highn._query_example(anchor, query_support, state["coords"])
        anchor_builder = Heat3DGraphBuilder(**anchor_config)
        query_builder = Heat3DGraphBuilder(**query_config)
        with jax.default_device(cpu):
            phase = time.perf_counter()
            anchor_metadata = anchor_builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor_example, runtime["stats"]), key=graph_key)
            block(anchor_metadata)
            stages["anchor_graph"] = time.perf_counter() - phase
            phase = time.perf_counter()
            query_metadata = query_builder.build_metadata(
                highn.runner._graph_coords_for_example(query_example, runtime["stats"]), key=graph_key)
            block(query_metadata)
            stages["query_graph"] = time.perf_counter() - phase
            phase = time.perf_counter()
            anchor_group = host_tree(highn._model_group(highn._prepare_group(
                example=anchor_example, anchor=anchor, runtime=runtime,
                builder=anchor_builder, metadata=anchor_metadata,
                edge_targets=p5r._compatible_targets(anchor_targets, anchor_metadata),
            )))
            stages["anchor_group_pack"] = time.perf_counter() - phase
            phase = time.perf_counter()
            query_group = host_tree(highn._model_group(highn._prepare_group(
                example=query_example, anchor=anchor, runtime=runtime,
                builder=query_builder, metadata=query_metadata,
                edge_targets=p5r._compatible_targets(query_targets, query_metadata),
            )))
            stages["query_group_pack"] = time.perf_counter() - phase
        phase = time.perf_counter()
        if direct:
            map_indices = np.zeros((1,), dtype=np.int32)
            map_weights = np.ones((1,), dtype=np.float64)
        else:
            mapping, _ = build_reconstruction_map(
                coords=state["coords"], layer_id=state["layer"], boundaries=state["boundaries"],
                support_indices=selected, empty_domain_fallback="same_layer",
                prepared_partition=state["partition"], query_workers=1,
            )
            map_indices = np.asarray(mapping.neighbor_local_indices, dtype=np.int32)
            map_weights = np.asarray(mapping.neighbor_weights, dtype=np.float64)
        stages["reconstruction_map"] = time.perf_counter() - phase
        payload = {
            "anchor": anchor_group, "query": query_group,
            "weights": np.asarray(selected_cv, dtype=np.float32),
            "map_indices": map_indices, "map_weights": map_weights,
        }
        return payload, stages

    def service_one(anchor: Any) -> dict[str, Any]:
        started = time.perf_counter()
        payload, stages = prepare_host(anchor)
        phase = time.perf_counter()
        device = jax.device_put((
            payload["anchor"], payload["query"], payload["weights"],
            payload["map_indices"], payload["map_weights"],
        ), gpu)
        enqueue = time.perf_counter() - phase
        phase = time.perf_counter(); block(device); h2d_sync = time.perf_counter() - phase
        phase = time.perf_counter(); prediction = forward(params, *device); block(prediction)
        neural = time.perf_counter() - phase
        elapsed = time.perf_counter() - started
        stages.update(h2d_enqueue=enqueue, h2d_sync=h2d_sync, neural_forward_and_reconstruction=neural)
        exclusive = float(sum(stages.values()))
        residual = elapsed - exclusive
        limit = max(0.025, 0.05 * elapsed)
        if residual < -1.0e-6 or residual > limit:
            raise RuntimeError(
                f"{anchor.sample_id}: exclusive timing residual {residual} exceeds limit {limit}")
        if not np.all(np.isfinite(np.asarray(prediction))):
            raise RuntimeError(f"{anchor.sample_id}: nonfinite prediction")
        return {
            "sample_id": anchor.sample_id, "elapsed_seconds": elapsed,
            "exclusive_stage_sum_seconds": exclusive,
            "residual_seconds": residual, "residual_limit_seconds": limit,
            "stages": stages,
        }

    # Compile model kernels once with a real payload. This does not prebuild
    # every sample-varying graph shape; the shared host graph gate covers those.
    warm_payload, _ = prepare_host(anchors[0])
    warm_device = jax.device_put((
        warm_payload["anchor"], warm_payload["query"], warm_payload["weights"],
        warm_payload["map_indices"], warm_payload["map_weights"],
    ), gpu)
    block(warm_device); block(forward(params, *warm_device))
    resident_values = []
    for _ in range(args.resident_repeats):
        phase = time.perf_counter(); prediction = forward(params, *warm_device); block(prediction)
        resident_values.append(time.perf_counter() - phase)

    serial_orders = []
    q2_orders = []
    first_hit_values: list[float] = []
    steady_values: list[float] = []
    order_seeds = protocol["timing"]["randomized_order_seeds"]
    for order_seed in order_seeds:
        order = np.random.default_rng(order_seed).permutation(32).tolist()
        rows = []
        for position, index in enumerate(order):
            row = service_one(anchors[index]); rows.append(row)
            (first_hit_values if position == 0 else steady_values).append(row["elapsed_seconds"])
        serial_orders.append({
            "order_seed": order_seed, "order": order, "rows": rows,
            "fresh": stats([row["elapsed_seconds"] for row in rows]),
            "Q1_closed_loop": stats([row["elapsed_seconds"] for row in rows]),
            "wall_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
        })

        started = time.perf_counter()
        next_position = 0
        inflight: dict[Any, tuple[int, float]] = {}
        completions: list[dict[str, Any]] = []
        q2_failed = None
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="final-e-q2") as pool:
                while next_position < 2:
                    submitted = time.perf_counter()
                    future = pool.submit(service_one, anchors[order[next_position]])
                    inflight[future] = (next_position, submitted); next_position += 1
                while inflight:
                    done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                    for future in done:
                        position, submitted = inflight.pop(future)
                        row = future.result(); completed = time.perf_counter()
                        completions.append({
                            "position": position, "sample_id": row["sample_id"],
                            "submit_offset_seconds": submitted - started,
                            "completion_offset_seconds": completed - started,
                            "submit_to_result_seconds": completed - submitted,
                        })
                        if next_position < len(order):
                            submitted = time.perf_counter()
                            future = pool.submit(service_one, anchors[order[next_position]])
                            inflight[future] = (next_position, submitted); next_position += 1
        except Exception as exc:  # retain the exact failed randomized order
            q2_failed = f"{type(exc).__name__}: {exc}"
        completions.sort(key=lambda row: row["completion_offset_seconds"])
        completion_offsets = [row["completion_offset_seconds"] for row in completions]
        if q2_failed is None and len(completions) == 32:
            inter = np.diff(np.asarray([0.0] + completion_offsets)).tolist()
            q2_orders.append({
                "status": "passed", "order_seed": order_seed, "order": order,
                "rows": completions,
                "submit_to_result": stats([row["submit_to_result_seconds"] for row in completions]),
                "inter_completion": stats(inter),
                "wall_seconds": completion_offsets[-1],
                "samples_per_second": 32.0 / completion_offsets[-1],
                "true_B16_to_B32_marginal_seconds":
                    (completion_offsets[31] - completion_offsets[15]) / 16.0,
            })
        else:
            q2_orders.append({
                "status": "failed_hard_gate", "order_seed": order_seed,
                "order": order, "completed_count": len(completions), "failure": q2_failed,
                "rows": completions,
            })
    q2_all_passed = all(row["status"] == "passed" for row in q2_orders)
    stage_names = tuple(serial_orders[0]["rows"][0]["stages"])
    stage_summary = {
        name: stats([
            row["stages"][name] for order in serial_orders for row in order["rows"]
        ]) for name in stage_names
    }
    result = {
        "schema_version": "heat3d_v6_p1i_final_e_service_v1",
        "status": "passed" if q2_all_passed else "passed_serial_Q2_not_qualified",
        "route": args.route, "resolution": resolution, "output_nodes": 240825,
        "timing_boundary": protocol["timing"]["boundary"],
        "unseen_shape_first_hit": stats(first_hit_values),
        "steady_shape_fresh": stats(steady_values),
        "fresh_single_case": stats([
            row["elapsed_seconds"] for order in serial_orders for row in order["rows"]
        ]),
        "resident_core": stats(resident_values),
        "stage_summary": stage_summary,
        "serial_orders": serial_orders, "Q2_orders": q2_orders,
        "Q2_all_randomized_orders_passed": q2_all_passed,
        "publication_speedup_allowed": q2_all_passed,
        "peak_vram_bytes": int(candidate.publication._device_memory().get("peak_bytes_in_use", 0)),
        "checkpoint_parameter_sha256_before": checkpoint_before,
        "checkpoint_parameter_sha256_after": highn._tree_sha256(runtime["checkpoint"]["params"]),
        "checkpoint_unchanged": checkpoint_before == highn._tree_sha256(runtime["checkpoint"]["params"]),
        "role_contract": protocol["role_contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "route": args.route,
        "fresh_median_s": result["fresh_single_case"]["median_seconds"],
        "Q2_all_passed": q2_all_passed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
