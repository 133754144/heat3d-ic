#!/usr/bin/env python3
"""Graph-only qualification for the V6 publication padding envelope.

This program never instantiates or applies the model. It builds the five
frozen production graph routes from valid inputs plus one target-free train
warmup input, records real (unpadded) graph hashes/counts, and derives capacity
only as an observed maximum. Padding is not applied here.
"""
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

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import benchmark_heat3d_v6_p1i_u2_asymmetric_runtime as u2  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as u1  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)

ROUTES = (
    "native1024_anchor", "E16384_reconstruction", "U_v2_16384_reconstruction",
    "E240825_direct_control", "U_v2_direct240825",
)


def sha_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def counts(metadata: Any) -> dict[str, int | None]:
    return {
        field: None if getattr(metadata, field, None) is None
        else int(np.asarray(getattr(metadata, field)).shape[1])
        for field in qualification.EDGE_FIELDS
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "native_padding_result", "query_padding_result",
        "golden_seal", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=559)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "cpu":
        raise RuntimeError("graph-only qualification must use CPU production graph backend")
    protocol = json.loads(args.protocol.read_text())
    state = p8.runtime_state(args)
    anchors = list(state["anchors"])
    if len(anchors) != 32:
        raise RuntimeError("frozen valid32 drift")
    train_rows = [
        row for row in state["dataset"].manifest["samples"]
        if str(row["split_role"]) == "train"
    ]
    warmup_row = min(train_rows, key=lambda row: hashlib.sha256(
        str(row["sample_id"]).encode()).hexdigest())
    warmup_anchor = state["dataset"]._load_sample(warmup_row)
    _, warmup_k, warmup_q, _ = highn._physics_fields(
        warmup_anchor,
        {"coords": state["coords"], "cv": state["cv"], "layer": state["layer"]},
    )
    physics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for anchor in anchors:
        row = state["physics"][anchor.sample_id]
        with np.load(row["physics_cache_file"], allow_pickle=False) as archive:
            physics[anchor.sample_id] = (
                np.asarray(archive["k_xyz"], dtype=np.float64),
                np.asarray(archive["q_W_m3"], dtype=np.float64),
            )
    physics[warmup_anchor.sample_id] = (warmup_k, warmup_q)

    seal = json.loads(args.golden_seal.read_text())
    golden = seal["historical_golden"]["records"]
    golden_by_key = {(row["route"], row["sample_id"]): row for row in golden}
    records: list[dict[str, Any]] = []
    envelopes = {route: {field: None for field in qualification.EDGE_FIELDS} for route in ROUTES}
    graph_key = state["graph_key"]
    anchor_config = dict(state["runtime"]["graph_config"])
    anchor_config.update(
        subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )

    def update(route: str, sample_id: str, split: str, metadata: Any,
               selected: np.ndarray) -> None:
        edge_counts = counts(metadata)
        for field, value in edge_counts.items():
            if value is not None:
                current = envelopes[route][field]
                envelopes[route][field] = value if current is None else max(current, value)
        records.append({
            "route": route, "sample_id": sample_id, "population": split,
            "resolution": int(len(selected)), "selected_indices_sha256": sha_array(selected),
            "edge_counts": edge_counts, "metadata_hashes": u2.metadata_hashes(metadata),
        })

    for anchor in anchors + [warmup_anchor]:
        population = "train_only_static_warmup" if anchor is warmup_anchor else "valid_iid"
        full_k, full_q = physics[anchor.sample_id]
        anchor_indices, distance = highn._anchor_indices(
            anchor, state["coords"],
            float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
        )
        if distance != 0.0:
            raise RuntimeError(f"{anchor.sample_id}: anchor coordinate drift")
        anchor_support = {
            "selected_indices": anchor_indices,
            "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
            "k_xyz": np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
            "q_W_m3": np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
            "layer_id": state["layer"][anchor_indices],
        }
        anchor_example = highn._query_example(anchor, anchor_support, state["coords"])
        builder = Heat3DGraphBuilder(**anchor_config)
        anchor_coords = highn.runner._graph_coords_for_example(anchor_example, state["runtime"]["stats"])
        native = builder.build_metadata(anchor_coords, key=graph_key); block(native)
        update("native1024_anchor", anchor.sample_id, population, native, anchor_indices)

        for resolution in (16384, 240825):
            if resolution == 240825:
                selected = np.arange(len(state["coords"]), dtype=np.int64)
                selected_cv = state["cv"]
            else:
                selected, _ = deterministic_nested_query_prefix(
                    sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
                    target_count=resolution, geometry_cache=state["geometry"],
                )
                selected_cv, _ = conservative_selected_control_volume(
                    full_coords=state["coords"], full_control_volume=state["cv"],
                    full_layer_id=state["layer"], selected_indices=selected, query_workers=1,
                )
            query_support = {
                "selected_indices": selected, "operator_control_volume": selected_cv,
                "k_xyz": full_k[selected], "q_W_m3": full_q[selected],
                "layer_id": state["layer"][selected],
            }
            query_example = highn._query_example(anchor, query_support, state["coords"])
            query_coords = highn.runner._graph_coords_for_example(query_example, state["runtime"]["stats"])
            e_config = dict(state["runtime"]["graph_config"])
            e_config.update(
                subsample_factor=resolution / 256.0,
                discrete_graph_backend="sparse_kdtree_v1", reuse_exact_p2r_for_r2p=True,
            )
            e_metadata = Heat3DGraphBuilder(**e_config).build_metadata(query_coords, key=graph_key)
            block(e_metadata)
            e_route = "E16384_reconstruction" if resolution == 16384 else "E240825_direct_control"
            update(e_route, anchor.sample_id, population, e_metadata, selected)
            u_metadata, audit = u1.prior_u1._u_v2_asymmetric_metadata(
                builder, native, anchor_coords, query_coords,
                numerical_tolerance=float(protocol["u_v2"]["normalized_numerical_tolerance"]),
                maximum_normalized_overshoot=float(protocol["u_v2"]["maximum_normalized_overshoot"]),
            )
            block(u_metadata)
            if not all(audit["native_exact"].values()) or audit["repaired_uncovered_count"] != 0:
                raise RuntimeError(f"{anchor.sample_id}: U-v2 graph semantics drift")
            u_route = "U_v2_16384_reconstruction" if resolution == 16384 else "U_v2_direct240825"
            update(u_route, anchor.sample_id, population, u_metadata, selected)

    records.sort(key=lambda row: (row["population"], row["sample_id"], row["route"]))
    indexed = {(row["route"], row["sample_id"]): row for row in records}
    golden_checks = []
    for key, expected in sorted(golden_by_key.items()):
        actual = indexed[key]
        native = indexed[("native1024_anchor", expected["sample_id"])]
        exact = (
            native["metadata_hashes"] == expected["native1024_graph_hashes"]
            and actual["metadata_hashes"] == expected["query_graph_hashes"]
        )
        golden_checks.append({
            "route": key[0], "sample_id": key[1], "real_unpadded_graph_hash_exact": exact,
            "native_actual": native["metadata_hashes"],
            "native_reference": expected["native1024_graph_hashes"],
            "query_actual": actual["metadata_hashes"],
            "query_reference": expected["query_graph_hashes"],
        })
    if not all(row["real_unpadded_graph_hash_exact"] for row in golden_checks):
        raise RuntimeError("historical real-unpadded graph golden mismatch")
    result = {
        "schema_version": "heat3d_v6_publication_graph_envelope_qualification_run_v1",
        "status": "passed", "envelope_qualification": "candidate_run_passed",
        "sample_count": 32, "train_only_warmup_count": 1,
        "sample_ids": [anchor.sample_id for anchor in anchors],
        "train_only_warmup_sample_id": warmup_anchor.sample_id,
        "routes": list(ROUTES), "records": records, "observed_max_envelopes": envelopes,
        "historical_golden_checks": golden_checks,
        "graph_contract": {
            "backend": "sparse_kdtree_v1", "graph_seed": int(state["runtime"]["run_config"]["graph_seed"]),
            "radius_changed": False, "backend_semantics_changed": False,
            "u_v2_repair_changed": False, "model_inference_executed": False,
            "padding_applied": False,
        },
        "role_contract": {
            "accessed_roles": ["valid_iid_inputs", "train_input_warmup_without_target"],
            "temperature_truth_read": False, "model_inference": False, "training": False,
            "test": False, "sealed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
