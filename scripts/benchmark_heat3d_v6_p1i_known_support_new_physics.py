#!/usr/bin/env python3
"""Time true known-support/new-physics inference on frozen P1i valid32.

The first registered sample supplies only coordinates, graph and reconstruction
map. Every timed case supplies its own real k/q/BC, anchor context and scale.
No temperature label or metric is read during this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v1_medium_controlled_training_export as legacy  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(tuple(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def dist(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median_seconds": float(np.median(array)),
            "p95_seconds": float(np.percentile(array, 95)), "mean_seconds": float(np.mean(array)),
            "std_seconds": float(np.std(array))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["B", "E"], required=True)
    parser.add_argument("--resolution", type=int, choices=[8192, 32768, 240825], required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--baseline-artifact-root", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_after_p1_before_p2_execution":
        raise RuntimeError("P2 protocol is not frozen")
    expected = {("B", 8192), ("E", 32768), ("B", 240825), ("E", 240825)}
    if (args.candidate, args.resolution) not in expected:
        raise RuntimeError("unregistered P2 cell")
    binding = json.loads(args.binding.read_text())
    preflight = json.loads(args.preflight.read_text())
    if preflight["status"] != "passed" or preflight["sample_ids"] != binding["development_subset"]["sample_ids"]:
        raise RuntimeError("frozen valid32 preflight mismatch")
    dataset = highn.Heat3DV6DualRobinDataset(args.dataset_root, args.manifest, include_roles={"valid_iid"})
    by_id = dataset.sample_index_by_id()
    anchors = [dataset[by_id[sample_id]] for sample_id in preflight["sample_ids"]]
    group_ids = [str(anchor.meta["group_id"]) for anchor in anchors]
    if len(set(group_ids)) != 32:
        raise RuntimeError("known-support workload requires 32 distinct groups")
    full, _ = highn._full_shared(args)
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}

    first = anchors[0]
    if args.resolution == 240825:
        fixed_indices = np.arange(len(full["coords"]), dtype=np.int64)
        fixed_cv = np.asarray(full["cv"], dtype=np.float64)
        fixed_layer = np.asarray(full["layer"], dtype=np.int32)
        reconstruction = None
    else:
        support_row = next(row for row in preflight["supports"][str(args.resolution)] if row["sample_id"] == first.sample_id)
        support = highn._load_support(Path(support_row["support_file"]))
        fixed_indices = np.asarray(support["selected_indices"], dtype=np.int64)
        fixed_cv = np.asarray(support["operator_control_volume"], dtype=np.float64)
        fixed_layer = np.asarray(support["layer_id"], dtype=np.int32)
        baseline = json.loads((args.baseline_artifact_root / f"resolution_{args.resolution}.json").read_text())
        map_row = next(row for row in baseline["reconstruction_cache"]["samples"] if row["sample_id"] == first.sample_id)
        reconstruction = candidate.publication._load_mapping_no_audit(Path(map_row["cache_file"]))
    support_hash = sha_array(fixed_indices)

    checkpoint = legacy._load_params_checkpoint(args.run_dir / "params_best_valid_point_global.pkl")
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    highn.install_checkpoint_feature_hooks(stats)
    model_config = legacy._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    graph_config = dict(run_config["graph_config"]); graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    runtime = {"checkpoint": checkpoint, "run_config": run_config, "stats": stats,
               "model_config": model_config, "graph_config": graph_config}
    graph_key = highn.runner._metadata_key(int(run_config["graph_seed"]))

    def sample_support(anchor):
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            return {"selected_indices": fixed_indices, "operator_control_volume": fixed_cv,
                    "k_xyz": np.asarray(physics["k_xyz"], dtype=np.float64)[fixed_indices],
                    "q_W_m3": np.asarray(physics["q_W_m3"], dtype=np.float64)[fixed_indices],
                    "layer_id": fixed_layer}

    first_support = sample_support(first)
    first_example = highn._query_example(first, first_support, full["coords"])
    builder = candidate._builder(args.candidate, anchor=first, runtime=runtime, graph_key=graph_key,
                                 physical_node_count=args.resolution, optimization_mode="baseline")
    metadata = builder.build_metadata(highn.runner._graph_coords_for_example(first_example, stats), key=graph_key)
    jax.block_until_ready(metadata.r_rnodes)
    edge_targets = {field: None if getattr(metadata, field) is None else int(getattr(metadata, field).shape[1])
                    for field in candidate.qualification.EDGE_FIELDS}
    graph_hash = highn._tree_sha256(metadata)
    with np.load(args.native_predictions, allow_pickle=False) as native:
        native_ids = [str(value) for value in np.asarray(native["sample_ids"]).tolist()]
        native_scales = {sample_id: float(np.asarray(native["predicted_scales"])[native_ids.index(sample_id)])
                         for sample_id in preflight["sample_ids"]}

    model = GraphNeuralOperator(**model_config)
    params = highn.runner._device_params(checkpoint["params"])
    device_map = None if reconstruction is None else to_device_reconstruction_map(reconstruction)

    @jax.jit
    def apply(model_params, group, weights, frozen_scale):
        output = highn.runner._model_apply(model, model_params, group)
        delta = output["raw_temperature"][0, 0, :, 0] - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        support_delta = delta / query_scale * frozen_scale
        if device_map is None:
            return support_delta
        return device_map.reconstruct(support_delta)

    compile_group = highn._model_group(highn._prepare_group(
        example=first_example, anchor=first, runtime=runtime, builder=builder,
        metadata=metadata, edge_targets=edge_targets))
    compiled = apply(params, compile_group, jnp.asarray(fixed_cv, dtype=jnp.float32), jnp.asarray(native_scales[first.sample_id]))
    jax.block_until_ready(compiled)

    total_times: list[float] = []; prepare_times: list[float] = []; h2d_times: list[float] = []
    forward_times: list[float] = []; signatures = []; output_finite = []
    for anchor in anchors:
        started = time.perf_counter()
        support = sample_support(anchor)
        example = highn._query_example(anchor, support, full["coords"])
        phase = time.perf_counter()
        group = highn._model_group(highn._prepare_group(
            example=example, anchor=anchor, runtime=runtime, builder=builder,
            metadata=metadata, edge_targets=edge_targets))
        prepare_times.append(time.perf_counter() - phase)
        physics_signature = hashlib.sha256()
        for value in (support["k_xyz"], support["q_W_m3"], np.asarray([anchor.meta["top_h_W_m2K"], anchor.meta["bottom_h_W_m2K"]])):
            physics_signature.update(sha_array(np.asarray(value)).encode())
        signatures.append(physics_signature.hexdigest())
        phase = time.perf_counter()
        device_group, device_weights, device_scale = jax.device_put(
            (group, np.asarray(fixed_cv, dtype=np.float32), np.asarray(native_scales[anchor.sample_id], dtype=np.float32)))
        jax.block_until_ready(device_weights)
        h2d_times.append(time.perf_counter() - phase)
        phase = time.perf_counter()
        prediction = apply(params, device_group, device_weights, device_scale)
        jax.block_until_ready(prediction)
        forward_times.append(time.perf_counter() - phase)
        total_times.append(time.perf_counter() - started)
        output_finite.append(bool(np.all(np.isfinite(np.asarray(prediction)))))
    if len(set(signatures)) != 32:
        raise RuntimeError("dynamic k/q/BC signatures are not unique")

    graph_hash_repeat = highn._tree_sha256(builder.build_metadata(
        highn.runner._graph_coords_for_example(first_example, stats), key=graph_key))
    payload = {
        "schema_version": "heat3d_v6_p1i_known_support_new_physics_v1", "status": "passed",
        "candidate": args.candidate, "resolution": args.resolution, "sample_count": 32,
        "sample_ids": preflight["sample_ids"], "group_ids": group_ids,
        "fixed_support_hash": support_hash, "graph_metadata_hash": graph_hash,
        "graph_repeat_hash": graph_hash_repeat, "graph_repeat_exact": graph_hash == graph_hash_repeat,
        "unique_group_count": len(set(group_ids)), "unique_physics_signature_count": len(set(signatures)),
        "dynamic_fields": protocol["known_support_new_physics"]["dynamic_fields"],
        "timing": {"dynamic_prepare": dist(prepare_times), "h2d": dist(h2d_times),
                   "forward_plus_reconstruction": dist(forward_times), "continuous_total": dist(total_times)},
        "output_all_finite": all(output_finite),
        "temperature_read": False, "metrics_computed": False,
        "memory": candidate.publication._device_memory(),
        "role_contract": {"valid_iid": True, "training": False, "test": False, "sealed": False},
    }
    if not payload["graph_repeat_exact"] or not payload["output_all_finite"]:
        payload["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else value) + "\n")
    print(json.dumps({"status": payload["status"], "candidate": args.candidate, "resolution": args.resolution,
                      "known_support_median_s": payload["timing"]["continuous_total"]["median_seconds"]}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
