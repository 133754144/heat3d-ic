#!/usr/bin/env python3
"""Build padding-adjusted prepared-payload golden hashes without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_final_e_service as e_service  # noqa: E402
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import benchmark_heat3d_v6_p1i_u2_asymmetric_runtime as u2  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as u1  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def host(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields",
        "run_dir", "native_padding_result", "query_padding_result", "e240825_padding_result",
        "u16384_padding_result", "u240825_padding_result", "old_seal", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=559)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "cpu":
        raise RuntimeError("padding golden construction must be CPU graph/pack only")
    protocol = json.loads(args.protocol.read_text())
    state = p8.runtime_state(args)
    anchors = {anchor.sample_id: anchor for anchor in state["anchors"]}
    old_records = json.loads(args.old_seal.read_text())["historical_golden"]["records"]
    native_targets = p5r._edge_targets(args.native_padding_result)
    route_targets = {
        "E16384_reconstruction": p5r._edge_targets(args.query_padding_result),
        "E240825_direct_control": p5r._edge_targets(args.e240825_padding_result),
    }
    u_sources = {
        "U_v2_16384_reconstruction": json.loads(args.u16384_padding_result.read_text()),
        "U_v2_direct240825": json.loads(args.u240825_padding_result.read_text()),
    }
    physics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample_id, anchor in anchors.items():
        with np.load(state["physics"][sample_id]["physics_cache_file"], allow_pickle=False) as archive:
            physics[sample_id] = (
                np.asarray(archive["k_xyz"], dtype=np.float64),
                np.asarray(archive["q_W_m3"], dtype=np.float64),
            )

    def supports(anchor: Any, resolution: int) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        full_k, full_q = physics[anchor.sample_id]
        anchor_indices, distance = highn._anchor_indices(
            anchor, state["coords"],
            float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        if distance != 0.0:
            raise RuntimeError("anchor drift")
        if resolution == 240825:
            selected = np.arange(len(state["coords"]), dtype=np.int64); selected_cv = state["cv"]
        else:
            selected, _ = deterministic_nested_query_prefix(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
                target_count=resolution, geometry_cache=state["geometry"])
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
        return (
            highn._query_example(anchor, anchor_support, state["coords"]),
            highn._query_example(anchor, query_support, state["coords"]),
            selected, selected_cv,
        )

    records = []
    for old in old_records:
        route = old["route"]; anchor = anchors[old["sample_id"]]
        resolution = 16384 if "16384" in route else 240825
        anchor_example, query_example, selected, selected_cv = supports(anchor, resolution)
        anchor_config = dict(state["runtime"]["graph_config"])
        anchor_config.update(subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1",
                             reuse_exact_p2r_for_r2p=True)
        builder = Heat3DGraphBuilder(**anchor_config)
        anchor_coords = highn.runner._graph_coords_for_example(anchor_example, state["runtime"]["stats"])
        native = builder.build_metadata(anchor_coords, key=state["graph_key"]); block(native)
        anchor_group_full = host(highn._prepare_group(
            example=anchor_example, anchor=anchor, runtime=state["runtime"], builder=builder,
            metadata=native, edge_targets=p5r._compatible_targets(native_targets, native)))
        anchor_group = host(highn._model_group(anchor_group_full))
        query_coords = highn.runner._graph_coords_for_example(query_example, state["runtime"]["stats"])
        if route.startswith("E"):
            query_config = dict(state["runtime"]["graph_config"])
            query_config.update(subsample_factor=resolution / 256.0,
                                discrete_graph_backend="sparse_kdtree_v1",
                                reuse_exact_p2r_for_r2p=True)
            query_builder = Heat3DGraphBuilder(**query_config)
            metadata = query_builder.build_metadata(query_coords, key=state["graph_key"]); block(metadata)
            query_group = host(highn._model_group(highn._prepare_group(
                example=query_example, anchor=anchor, runtime=state["runtime"], builder=query_builder,
                metadata=metadata,
                edge_targets=p5r._compatible_targets(route_targets[route], metadata))))
            if resolution == 240825:
                map_indices = np.zeros((1,), dtype=np.int32); map_weights = np.ones((1,), dtype=np.float64)
            else:
                mapping, _ = build_reconstruction_map(
                    coords=state["coords"], layer_id=state["layer"], boundaries=state["boundaries"],
                    support_indices=selected, empty_domain_fallback="same_layer",
                    prepared_partition=state["partition"], query_workers=1)
                map_indices = np.asarray(mapping.neighbor_local_indices, dtype=np.int32)
                map_weights = np.asarray(mapping.neighbor_weights, dtype=np.float64)
            payload_sha = e_service.tree_sha((
                anchor_group, query_group, np.asarray(selected_cv, dtype=np.float32),
                map_indices, map_weights))
            query_hash = e_service.metadata_hashes(metadata)
        else:
            source = u_sources[route]["padding"]["actual_padding_envelope"]
            metadata, audit = u1.prior_u1._u_v2_asymmetric_metadata(
                builder, native, anchor_coords, query_coords,
                numerical_tolerance=float(protocol["u_v2"]["normalized_numerical_tolerance"]),
                maximum_normalized_overshoot=float(protocol["u_v2"]["maximum_normalized_overshoot"]),
            ); block(metadata)
            if not all(audit["native_exact"].values()) or audit["repaired_uncovered_count"] != 0:
                raise RuntimeError("U-v2 graph gate")
            combined = u1._combined_targets(source["native"], source["query"])
            output_group = host(u1._prepare_output_query_group_lean(
                example=query_example, anchor=anchor, runtime=state["runtime"], builder=builder,
                metadata=metadata, edge_targets=p5r._compatible_targets(combined, metadata)))
            graphs = host(output_group["graphs"])
            local = host(u1._dummy_local_p2r(builder, metadata))
            inputs_out = host(output_group["inputs"])
            kwargs = host(u1._model_kwargs(anchor_group_full, output_group))
            payload_sha = u2.tree_sha((inputs_out, graphs, local, kwargs))
            query_hash = u2.metadata_hashes(metadata)
        record = {
            "route": route, "order_seed": old["order_seed"], "resolution": resolution,
            "sample_id": anchor.sample_id,
            "native1024_graph_hashes": e_service.metadata_hashes(native),
            "query_graph_hashes": query_hash,
            "prepared_payload_sha256": payload_sha,
            "padding_adjusted": True, "model_inference_executed": False,
        }
        if (record["native1024_graph_hashes"] != old["native1024_graph_hashes"]
                or record["query_graph_hashes"] != old["query_graph_hashes"]):
            raise RuntimeError(f"{route}: real graph drift while padding golden changed")
        record["record_sha256"] = canonical_sha(record)
        records.append(record)
    result = {
        "schema_version": "heat3d_v6_publication_padding_adjusted_golden_v1",
        "status": "passed", "record_count": len(records), "records": records,
        "model_inference_executed": False,
        "role_contract": {"training": False, "test": False, "sealed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
