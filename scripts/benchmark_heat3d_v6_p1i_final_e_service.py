#!/usr/bin/env python3
"""Final matched valid32 service benchmark for frozen E inference routes."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
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
from heat3d_v6_publication_lifecycle_schema import (  # noqa: E402
    provenance as lifecycle_provenance,
    q2_metrics,
    serial_metrics,
    timing_stats,
    validate_cell,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def stats(values: list[float]) -> dict[str, float | int]:
    return timing_stats(values)


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def peak_ram_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def tree_sha(tree: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest.update(str(treedef).encode())
    for leaf in leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def metadata_hashes(metadata: Any) -> dict[str, Any]:
    fields = (
        "x_pnodes_inp", "x_pnodes_out", "x_rnodes", "r_rnodes",
        "p2r_edge_indices", "p2r_domains", "r2r_edge_indices",
        "r2r_domains", "r2p_edge_indices", "r2p_domains",
    )
    rows = {
        field: (None if getattr(metadata, field, None) is None
                else tree_sha(np.asarray(getattr(metadata, field))))
        for field in fields
    }
    rows["combined_sha256"] = tree_sha(tuple(
        np.asarray(getattr(metadata, field))
        for field in fields if getattr(metadata, field, None) is not None
    ))
    return rows


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
    parser.add_argument("--sample-count", type=int, choices=(4, 8, 32), default=32)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--service-mode", choices=("serial", "Q2", "both"), default="both")
    parser.add_argument("--standard-v1-1-smoke", action="store_true")
    parser.add_argument("--publication-v1-1", action="store_true")
    parser.add_argument("--cache-hot-repeats", type=int, default=20)
    parser.add_argument("--golden-seal", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") not in {
        "preregistered_before_execution", "frozen_before_real_route_conformance_smoke",
        "pre_measurement_sealed",
    }:
        raise RuntimeError("final correction protocol is not preregistered")
    if args.publication_v1_1 and (args.standard_v1_1_smoke or args.sample_count != 32):
        raise RuntimeError("publication v1.1 requires a distinct full-valid32 lifecycle")
    if args.publication_v1_1 and args.golden_seal is None:
        raise RuntimeError("publication v1.1 requires historical golden seal")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("final E service benchmark requires GPU")
    state = p8.runtime_state(args)
    runtime = state["runtime"]
    checkpoint_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    # p5r._runtime already verifies the checkpoint file against the registered
    # file SHA supplied on the command line.  The tree SHA is a separate
    # before/after immutability witness and must not be compared to file bytes.
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
    train_rows = [
        row for row in state["dataset"].manifest["samples"]
        if str(row["split_role"]) == "train"
    ]
    warmup_row = min(
        train_rows,
        key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(),
    )
    warmup_id = str(warmup_row["sample_id"])
    warmup_anchor = state["dataset"]._load_sample(warmup_row)
    _, warmup_k, warmup_q, _ = highn._physics_fields(
        warmup_anchor,
        {"coords": state["coords"], "cv": state["cv"], "layer": state["layer"]},
    )
    physics_memory[warmup_id] = (warmup_k, warmup_q)
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

    def prepare_host(
        anchor: Any, *, allow_static_envelope_widen: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        stages: dict[str, float] = {}
        phase = time.perf_counter()
        full_k, full_q = physics_memory[anchor.sample_id]
        anchor_indices, distance = highn._anchor_indices(
            anchor, state["coords"],
            float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        if distance != 0.0:
            raise RuntimeError(f"{anchor.sample_id}: anchor coordinate drift")
        stages["input_lookup_and_anchor_index"] = time.perf_counter() - phase
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
        phase = time.perf_counter()
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
        stages["support_payload_and_builder_setup"] = time.perf_counter() - phase
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
            if allow_static_envelope_widen:
                for targets, metadata in (
                    (anchor_targets, anchor_metadata), (query_targets, query_metadata)):
                    for field in (
                        "p2r_edge_indices", "p2r_domains", "r2r_edge_indices",
                        "r2r_domains", "r2p_edge_indices", "r2p_domains",
                    ):
                        value = getattr(metadata, field, None)
                        if value is not None:
                            targets[field] = max(
                                int(targets.get(field) or 0),
                                int(np.asarray(value).shape[1]),
                            )
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
            "graph_hashes": {
                "native1024": metadata_hashes(anchor_metadata),
                "query": metadata_hashes(query_metadata),
            },
        }
        return payload, stages

    host_payload_cache: dict[str, dict[str, Any]] = {}

    def service_one(anchor: Any, *, cache_hot: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        if cache_hot:
            phase = time.perf_counter()
            payload = host_payload_cache[anchor.sample_id]
            stages = {"declared_case_cache_lookup": time.perf_counter() - phase}
        else:
            payload, stages = prepare_host(anchor)
            if args.publication_v1_1 and args.service_mode in {"serial", "both"}:
                host_payload_cache[anchor.sample_id] = payload
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

    order_seeds = ([args.order_seed] if args.order_seed is not None
                   else protocol.get("timing", protocol)["randomized_order_seeds"])
    orders = {seed: np.random.default_rng(seed).permutation(32)[:args.sample_count].tolist()
              for seed in order_seeds}
    # Compile only with a train-input payload.  No target is loaded and no
    # timed valid case graph or packing path is touched before measurement.
    envelope_before_warmup = {"anchor": dict(anchor_targets), "query": dict(query_targets)}
    warm_payload, _ = prepare_host(warmup_anchor, allow_static_envelope_widen=True)
    envelope_after_warmup = {"anchor": dict(anchor_targets), "query": dict(query_targets)}
    warm_device = jax.device_put((
        warm_payload["anchor"], warm_payload["query"], warm_payload["weights"],
        warm_payload["map_indices"], warm_payload["map_weights"],
    ), gpu)
    block(warm_device); block(forward(params, *warm_device))
    warmup_resident_values = []
    if not args.publication_v1_1 and args.service_mode in {"serial", "both"}:
        for _ in range(args.resident_repeats):
            phase = time.perf_counter(); prediction = forward(params, *warm_device); block(prediction)
            warmup_resident_values.append(time.perf_counter() - phase)

    serial_orders = []
    q2_orders = []
    first_hit_values: list[float] = []
    steady_values: list[float] = []
    cache_hot_values: list[float] = []
    valid_resident_values: list[float] = []
    for order_seed in order_seeds:
        order = orders[order_seed]
        rows = []
        if args.service_mode in {"serial", "both"}:
            for position, index in enumerate(order):
                row = service_one(anchors[index]); rows.append(row)
                (first_hit_values if position == 0 else steady_values).append(row["elapsed_seconds"])
            serial_orders.append({
                "order_seed": order_seed, "order": order, "sample_ids": [anchors[index].sample_id for index in order], "rows": rows,
                "fresh": stats([row["elapsed_seconds"] for row in rows]),
                "Q1_closed_loop": stats([row["elapsed_seconds"] for row in rows]),
                "wall_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
            })
            if args.publication_v1_1:
                repeated_anchor = anchors[order[0]]
                for _ in range(args.cache_hot_repeats):
                    cache_hot_values.append(
                        service_one(repeated_anchor, cache_hot=True)["elapsed_seconds"])
                cached = host_payload_cache[repeated_anchor.sample_id]
                resident_device = jax.device_put((
                    cached["anchor"], cached["query"], cached["weights"],
                    cached["map_indices"], cached["map_weights"],
                ), gpu)
                block(resident_device)
                for _ in range(args.resident_repeats):
                    phase = time.perf_counter()
                    resident_prediction = forward(params, *resident_device)
                    block(resident_prediction)
                    valid_resident_values.append(time.perf_counter() - phase)

        if args.service_mode not in {"Q2", "both"}:
            continue
        started = time.perf_counter()
        next_position = 0
        inflight: dict[Any, tuple[int, float]] = {}
        completions: list[dict[str, Any]] = []
        q2_failed = None
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="final-e-q2") as pool:
                while next_position < min(2, len(order)):
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
                            "service_residual_seconds": row["residual_seconds"],
                            "service_residual_limit_seconds": row["residual_limit_seconds"],
                        })
                        if next_position < len(order):
                            submitted = time.perf_counter()
                            future = pool.submit(service_one, anchors[order[next_position]])
                            inflight[future] = (next_position, submitted); next_position += 1
        except Exception as exc:  # retain the exact failed randomized order
            q2_failed = f"{type(exc).__name__}: {exc}"
        completions.sort(key=lambda row: row["completion_offset_seconds"])
        completion_offsets = [row["completion_offset_seconds"] for row in completions]
        if q2_failed is None and len(completions) == len(order):
            inter = np.diff(np.asarray([0.0] + completion_offsets)).tolist()
            q2_orders.append({
                "status": "passed", "order_seed": order_seed, "order": order,
                "rows": completions,
                "submit_to_result": stats([row["submit_to_result_seconds"] for row in completions]),
                "inter_completion": stats(inter),
                "wall_seconds": completion_offsets[-1],
                "samples_per_second": len(order) / completion_offsets[-1],
                "true_B16_to_B32_marginal_seconds": (
                    (completion_offsets[31] - completion_offsets[15]) / 16.0
                    if len(order) == 32 else None
                ),
            })
        else:
            q2_orders.append({
                "status": "failed_hard_gate", "order_seed": order_seed,
                "order": order, "completed_count": len(completions), "failure": q2_failed,
                "rows": completions,
            })
    q2_all_passed = all(row["status"] == "passed" for row in q2_orders) if q2_orders else True
    exactness = None
    if args.standard_v1_1_smoke:
        first_anchor = anchors[orders[order_seeds[0]][0]]
        candidate_payload, _ = prepare_host(first_anchor)
        reference_payload, _ = prepare_host(first_anchor)
        candidate_prepared_sha256 = tree_sha(tuple(
            candidate_payload[key] for key in (
                "anchor", "query", "weights", "map_indices", "map_weights")))
        reference_prepared_sha256 = tree_sha(tuple(
            reference_payload[key] for key in (
                "anchor", "query", "weights", "map_indices", "map_weights")))
        exactness = {
            "sample_id": first_anchor.sample_id,
            "graph_candidate_hashes": candidate_payload["graph_hashes"],
            "graph_reference_hashes": reference_payload["graph_hashes"],
            "graph_metadata_edge_hash_exact": (
                candidate_payload["graph_hashes"] == reference_payload["graph_hashes"]),
            "candidate_prepared_payload_sha256": candidate_prepared_sha256,
            "reference_prepared_payload_sha256": reference_prepared_sha256,
            "prepared_payload_exact": candidate_prepared_sha256 == reference_prepared_sha256,
            "audit_outside_service_timing": True,
        }
        if not exactness["graph_metadata_edge_hash_exact"] or not exactness["prepared_payload_exact"]:
            raise RuntimeError(f"{first_anchor.sample_id}: E route exactness audit failed")
    if args.publication_v1_1:
        first_anchor = anchors[orders[order_seeds[0]][0]]
        candidate_payload, _ = prepare_host(first_anchor)
        candidate_prepared_sha256 = tree_sha(tuple(
            candidate_payload[key] for key in (
                "anchor", "query", "weights", "map_indices", "map_weights")))
        frozen = json.loads(args.golden_seal.read_text())["historical_golden"]["records"]
        matches = [row for row in frozen if row["route"] == args.route
                   and row["order_seed"] == order_seeds[0]
                   and row["sample_id"] == first_anchor.sample_id]
        if len(matches) != 1:
            raise RuntimeError("E historical golden record missing")
        golden = matches[0]
        if (candidate_payload["graph_hashes"]["native1024"] != golden["native1024_graph_hashes"]
                or candidate_payload["graph_hashes"]["query"] != golden["query_graph_hashes"]
                or candidate_prepared_sha256 != golden["prepared_payload_sha256"]):
            raise RuntimeError(f"{first_anchor.sample_id}: E historical golden drift")
        exactness = {
            "sample_id": first_anchor.sample_id,
            "graph_candidate_hashes": candidate_payload["graph_hashes"],
            "graph_reference_hashes": {
                "native1024": golden["native1024_graph_hashes"],
                "query": golden["query_graph_hashes"],
            },
            "candidate_prepared_payload_sha256": candidate_prepared_sha256,
            "reference_prepared_payload_sha256": golden["prepared_payload_sha256"],
            "historical_golden_record_sha256": golden["record_sha256"],
            "reference_semantics": "immutable_historical_git_bound_golden",
            "audit_outside_service_timing": True,
        }
    stage_names = tuple(serial_orders[0]["rows"][0]["stages"]) if serial_orders else ()
    stage_summary = {
        name: stats([
            row["stages"][name] for order in serial_orders for row in order["rows"]
        ]) for name in stage_names
    }
    lifecycle_metrics = None
    if args.service_mode == "serial" and args.publication_v1_1:
        flat_serial_rows = [row for order in serial_orders for row in order["rows"]]
        lifecycle_metrics = serial_metrics(
            cold_seconds=float(flat_serial_rows[0]["elapsed_seconds"]),
            fresh_q1=stats([row["elapsed_seconds"] for row in flat_serial_rows]),
            cache_hot=stats(cache_hot_values),
            resident=stats(valid_resident_values),
        )
    elif args.service_mode == "Q2" and args.publication_v1_1:
        if len(q2_orders) != 1 or q2_orders[0].get("status") != "passed":
            raise RuntimeError("Q2 lifecycle did not produce one passed order")
        q2_row = q2_orders[0]
        lifecycle_metrics = q2_metrics(
            submit_to_result=q2_row["submit_to_result"],
            inter_completion=q2_row["inter_completion"],
            throughput_samples_per_second=q2_row["samples_per_second"],
            b16_to_b32_marginal_seconds=q2_row["true_B16_to_B32_marginal_seconds"],
        )
    result = {
        "schema_version": "heat3d_v6_p1i_final_e_service_v1",
        "status": (
            "passed" if q2_all_passed and args.sample_count == 32
            else "passed_smoke" if q2_all_passed
            else "passed_serial_Q2_not_qualified"
        ),
        "route": args.route, "resolution": resolution, "output_nodes": 240825,
        "sample_count": args.sample_count, "service_mode": args.service_mode,
        "process_id": os.getpid(), "order_seeds": order_seeds,
        "ordered_sample_ids": {str(seed): [anchors[index].sample_id for index in orders[seed]] for seed in order_seeds},
        "warmup": {"kind": "train_input_static_padded_envelope", "source_sample_id": warmup_id,
                   "source_split": "train", "target_read": False,
                   "source_is_timed": False, "timed_graph_or_packing_prebuilt": False,
                   "padding_envelope_before": envelope_before_warmup,
                   "padding_envelope_after": envelope_after_warmup,
                   "padding_widening_semantics": "max_frozen_valid_and_train_warmup_actual"},
        "timing_boundary": protocol.get("timing", {}).get("boundary", protocol.get("timing_boundary")),
        "unseen_shape_first_hit": None if not first_hit_values else stats(first_hit_values),
        "steady_shape_fresh": None if not steady_values else stats(steady_values),
        "fresh_single_case": None if not serial_orders else stats([
            row["elapsed_seconds"] for order in serial_orders for row in order["rows"]
        ]),
        "resident_core": (
            stats(valid_resident_values)
            if args.service_mode == "serial" and valid_resident_values else None),
        "repeat_case_cache_hot": (
            stats(cache_hot_values)
            if args.service_mode == "serial" and cache_hot_values else None),
        "lifecycle_metrics": lifecycle_metrics,
        "stage_summary": stage_summary,
        "timing_pool_classification": {
            "cold_service_first_case": "first_serial_row" if serial_orders else None,
            "fresh_distinct_case": "serial_rows" if serial_orders else None,
            "repeat_case_cache_hot": (
                "separate_post_fresh_declared_case_cache_pool"
                if args.service_mode == "serial" and args.publication_v1_1 else None),
            "resident_core": (
                "separate_valid_prepared_device_payload_pool"
                if args.service_mode == "serial" and args.publication_v1_1 else None),
            "pools_mixed": False,
        },
        "serial_orders": serial_orders, "Q2_orders": q2_orders,
        "Q2_all_randomized_orders_passed": q2_all_passed,
        "publication_speedup_allowed": False,
        "measurement_provenance": lifecycle_provenance(
            attempted=bool(args.publication_v1_1), matrix_completed=False, generated=False),
        "peak_vram_bytes": int(candidate.publication._device_memory().get("peak_bytes_in_use", 0)),
        "aggregate_service_worker_peak_RAM_bytes": peak_ram_bytes(),
        "memory_measurement": {
            "field": "service_process_HWM_bytes",
            "value": peak_ram_bytes(),
            "semantics": "single_process_service_HWM",
        },
        "cpu_policy": protocol["resources"]["neural"],
        "thread_env": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "checkpoint_parameter_sha256_before": checkpoint_before,
        "checkpoint_parameter_sha256_after": highn._tree_sha256(runtime["checkpoint"]["params"]),
        "checkpoint_unchanged": checkpoint_before == highn._tree_sha256(runtime["checkpoint"]["params"]),
        "exactness_provenance": exactness,
        "role_contract": protocol["role_contract"],
    }
    if args.publication_v1_1:
        validate_cell(result, formal=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"], "route": args.route,
        "fresh_median_s": (
            None if result["fresh_single_case"] is None
            else result["fresh_single_case"]["median_seconds"]),
        "Q2_all_passed": q2_all_passed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
