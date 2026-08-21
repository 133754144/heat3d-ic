#!/usr/bin/env python3
"""Two-case model equivalence for exact-edge versus dummy-padded envelopes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_final_e_service as e_service  # noqa: E402
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields",
        "run_dir", "native_padding_result", "query_padding_result", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=559)
    parser.add_argument("--sample-count", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("padding prediction equivalence requires production GPU")
    state = p8.runtime_state(args)
    binding = json.loads(args.binding.read_text())
    native_targets = p5r._edge_targets(args.native_padding_result)
    query_targets = p5r._edge_targets(args.query_padding_result)
    gpu = jax.devices("gpu")[0]
    model = GraphNeuralOperator(**state["runtime"]["model_config"])
    params = highn.runner._device_params(state["runtime"]["checkpoint"]["params"])
    before = highn._tree_sha256(state["runtime"]["checkpoint"]["params"])
    records = []

    def predict(payload: tuple[Any, ...]) -> np.ndarray:
        anchor_group, query_group, weights, indices, map_weights = jax.device_put(payload, gpu)
        block((anchor_group, query_group, weights, indices, map_weights))
        anchor_result = highn.runner._model_apply(model, params, anchor_group)
        anchor_scale = anchor_result["s_hat"].reshape(-1)[0]
        query = highn.runner._model_apply(model, params, query_group)["raw_temperature"][0, 0, :, 0]
        query = query - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * query * query))
        support = query / query_scale * anchor_scale
        full = jnp.sum(support[indices] * map_weights.astype(support.dtype), axis=1)
        block(full)
        return np.asarray(full, dtype=np.float64)

    for anchor in state["anchors"][:args.sample_count]:
        with np.load(state["physics"][anchor.sample_id]["physics_cache_file"], allow_pickle=False) as archive:
            full_k = np.asarray(archive["k_xyz"], dtype=np.float64)
            full_q = np.asarray(archive["q_W_m3"], dtype=np.float64)
        anchor_indices, distance = highn._anchor_indices(
            anchor, state["coords"],
            float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]))
        if distance != 0.0:
            raise RuntimeError("anchor drift")
        selected, _ = deterministic_nested_query_prefix(
            sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
            target_count=16384, geometry_cache=state["geometry"])
        selected_cv, _ = conservative_selected_control_volume(
            full_coords=state["coords"], full_control_volume=state["cv"],
            full_layer_id=state["layer"], selected_indices=selected, query_workers=1)
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
        anchor_config = dict(state["runtime"]["graph_config"])
        anchor_config.update(subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1",
                             reuse_exact_p2r_for_r2p=True)
        query_config = dict(state["runtime"]["graph_config"])
        query_config.update(subsample_factor=64.0, discrete_graph_backend="sparse_kdtree_v1",
                            reuse_exact_p2r_for_r2p=True)
        anchor_builder = Heat3DGraphBuilder(**anchor_config)
        query_builder = Heat3DGraphBuilder(**query_config)
        with jax.default_device(jax.devices("cpu")[0]):
            anchor_metadata = anchor_builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor_example, state["runtime"]["stats"]),
                key=state["graph_key"]); block(anchor_metadata)
            query_metadata = query_builder.build_metadata(
                highn.runner._graph_coords_for_example(query_example, state["runtime"]["stats"]),
                key=state["graph_key"]); block(query_metadata)
            exact_native = highn._edge_counts(anchor_metadata)
            exact_query = highn._edge_counts(query_metadata)
            mapping, _ = build_reconstruction_map(
                coords=state["coords"], layer_id=state["layer"], boundaries=state["boundaries"],
                support_indices=selected, empty_domain_fallback="same_layer",
                prepared_partition=state["partition"], query_workers=1)

            def payload(ntargets: dict[str, int | None], qtargets: dict[str, int | None]) -> tuple[Any, ...]:
                ag = highn._model_group(highn._prepare_group(
                    example=anchor_example, anchor=anchor, runtime=state["runtime"],
                    builder=anchor_builder, metadata=anchor_metadata,
                    edge_targets=p5r._compatible_targets(ntargets, anchor_metadata)))
                qg = highn._model_group(highn._prepare_group(
                    example=query_example, anchor=anchor, runtime=state["runtime"],
                    builder=query_builder, metadata=query_metadata,
                    edge_targets=p5r._compatible_targets(qtargets, query_metadata)))
                return (
                    ag, qg, np.asarray(selected_cv, dtype=np.float32),
                    np.asarray(mapping.neighbor_local_indices, dtype=np.int32),
                    np.asarray(mapping.neighbor_weights, dtype=np.float64),
                )
            exact_payload = payload(exact_native, exact_query)
            padded_payload = payload(native_targets, query_targets)
        exact_prediction = predict(exact_payload)
        padded_prediction = predict(padded_payload)
        delta = padded_prediction - exact_prediction
        maximum = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta * delta)))
        tolerance = float(binding["numeric_tolerances"]["cached_uncached_prediction_max_abs_K"])
        passed = maximum <= tolerance
        records.append({
            "sample_id": anchor.sample_id,
            "real_graph_hashes": {
                "native1024": e_service.metadata_hashes(anchor_metadata),
                "query": e_service.metadata_hashes(query_metadata)},
            "exact_edge_targets": {"native": exact_native, "query": exact_query},
            "frozen_dummy_capacity": {"native": native_targets, "query": query_targets},
            "prediction_max_abs_K": maximum, "prediction_RMSE_K": rmse,
            "tolerance_K": tolerance, "passed": passed,
        })
        if not passed:
            raise RuntimeError(f"{anchor.sample_id}: dummy padding changed prediction")
    after = highn._tree_sha256(state["runtime"]["checkpoint"]["params"])
    if before != after:
        raise RuntimeError("checkpoint parameters changed")
    result = {
        "schema_version": "heat3d_v6_publication_padding_prediction_equivalence_v1",
        "status": "passed", "sample_count": args.sample_count, "records": records,
        "checkpoint_parameter_tree_sha256_before": before,
        "checkpoint_parameter_tree_sha256_after": after, "checkpoint_unchanged": True,
        "real_edges_changed": False, "dummy_capacity_only": True,
        "role_contract": {
            "accessed_roles": ["valid_iid_inputs"], "temperature_truth_read": False,
            "training": False, "accuracy_tuning": False, "test": False, "sealed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "samples": args.sample_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
