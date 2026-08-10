#!/usr/bin/env python3
"""Exact static-cache benchmark for P1i known-support/new-physics."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    if str(value) not in sys.path: sys.path.insert(0, str(value))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash, metadata_hash  # noqa: E402
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(); digest.update(str(array.dtype).encode()); digest.update(str(tuple(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def dist(values):
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "median_seconds": float(np.median(array)),
            "p95_seconds": float(np.percentile(array, 95)), "mean_seconds": float(np.mean(array)),
            "std_seconds": float(np.std(array))}


def difference(left, right):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(delta))), float(np.sqrt(np.mean(np.square(delta))))


def cached_cpu_group(example, anchor, runtime, metadata, template):
    bridge = runner._bridge_for(example)
    c = runner.normalize_condition(bridge.legacy_inputs.c, runtime["stats"])
    old = template["inputs"]
    inputs = type(old)(u=old.u, c=c, x_inp=old.x_inp, x_out=old.x_out, t=old.t, tau=old.tau)
    group = {"sample_ids": (example.sample_id,), "inputs": inputs,
             "graphs": template["graphs"], "metadata": metadata}
    standardizer = runtime["run_config"]["global_context"]["standardizer"]
    context = highn.common.standardize_v6_contexts(
        [runner._global_context_row_for_example(anchor)], standardizer)[0]
    runner._attach_global_context_to_groups(
        [group], {example.sample_id: context},
        expected_feature_dim=int(runtime["model_config"]["global_context_feature_dim"]))
    runner._attach_native_physics_to_groups([group], {example.sample_id: example})
    if (runtime["model_config"].get("scale_pooling") == "qk_gated"
            or runtime["model_config"].get("shape_attention_mode") != "none"
            or runtime["model_config"].get("scale_attention_mode") != "none"):
        runner._attach_qk_region_features_to_groups(
            [group], {example.sample_id: example},
            feature_version=runtime["model_config"]["qk_region_feature_version"])
    if runtime["model_config"].get("scale_deepsets_mode", "none") != "none":
        runner._attach_scale_deepsets_weights_to_groups([group], {example.sample_id: example})
    return highn._model_group(group)


def merge_dynamic_device(static_device, dynamic_cpu):
    result = dict(static_device)
    old = static_device["inputs"]
    dynamic_c = jax.device_put(dynamic_cpu["inputs"].c)
    result["inputs"] = type(old)(u=old.u, c=dynamic_c, x_inp=old.x_inp, x_out=old.x_out, t=old.t, tau=old.tau)
    dynamic_leaves = [dynamic_c]
    for key in ("global_context", "qk_region_features", "scale_context", "scale_region_source_weights"):
        if key in dynamic_cpu:
            result[key] = jax.device_put(dynamic_cpu[key]); dynamic_leaves.append(result[key])
    if "native_physics" in dynamic_cpu:
        native = dict(static_device["native_physics"])
        native["log_s_phys"] = jax.device_put(dynamic_cpu["native_physics"]["log_s_phys"])
        dynamic_leaves.append(native["log_s_phys"]); result["native_physics"] = native
    return result, dynamic_leaves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["B", "E"], required=True)
    parser.add_argument("--resolution", type=int, choices=[8192, 32768, 240825], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--baseline-artifact-root", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_after_p2_before_p3_execution": raise RuntimeError("P3 protocol not frozen")
    if (args.candidate, args.resolution) not in {("B",8192),("E",32768),("B",240825),("E",240825)}:
        raise RuntimeError("unregistered P3 cell")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    compilation_dir = args.cache_root / "jax_compilation"
    compilation_dir.mkdir(parents=True, exist_ok=True)
    compilation_enabled = True
    try: jax.config.update("jax_compilation_cache_dir", str(compilation_dir))
    except Exception: compilation_enabled = False

    preflight = json.loads(args.preflight.read_text())
    dataset = highn.Heat3DV6DualRobinDataset(args.dataset_root, args.manifest, include_roles={"valid_iid"})
    by_id = dataset.sample_index_by_id(); anchors = [dataset[by_id[sid]] for sid in preflight["sample_ids"]]
    group_ids = [str(anchor.meta["group_id"]) for anchor in anchors]
    if len(set(group_ids)) != 32: raise RuntimeError("expected 32 distinct groups")
    full, _ = highn._full_shared(args)
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}
    first = anchors[0]
    if args.resolution == 240825:
        fixed_indices = np.arange(len(full["coords"]), dtype=np.int64)
        fixed_cv = np.asarray(full["cv"], dtype=np.float64); fixed_layer = np.asarray(full["layer"], dtype=np.int32)
        reconstruction = None; reconstruction_file = None
    else:
        row = next(row for row in preflight["supports"][str(args.resolution)] if row["sample_id"] == first.sample_id)
        support0 = highn._load_support(Path(row["support_file"]))
        fixed_indices = np.asarray(support0["selected_indices"], dtype=np.int64)
        fixed_cv = np.asarray(support0["operator_control_volume"], dtype=np.float64)
        fixed_layer = np.asarray(support0["layer_id"], dtype=np.int32)
        baseline = json.loads((args.baseline_artifact_root / f"resolution_{args.resolution}.json").read_text())
        map_row = next(item for item in baseline["reconstruction_cache"]["samples"] if item["sample_id"] == first.sample_id)
        reconstruction_file = Path(map_row["cache_file"])
        phase = time.perf_counter(); reconstruction = candidate.publication._load_mapping_no_audit(reconstruction_file)
        reconstruction_load_s = time.perf_counter() - phase
    if reconstruction is None: reconstruction_load_s = 0.0

    checkpoint = runner._load_params_checkpoint(args.run_dir / "params_best_valid_point_global.pkl")
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    highn.install_checkpoint_feature_hooks(stats)
    model_config = runner._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    graph_config = dict(run_config["graph_config"]); graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    runtime = {"checkpoint": checkpoint, "run_config": run_config, "stats": stats,
               "model_config": model_config, "graph_config": graph_config}
    graph_key = highn.runner._metadata_key(int(run_config["graph_seed"]))

    def support_for(anchor):
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            return {"selected_indices": fixed_indices, "operator_control_volume": fixed_cv,
                    "k_xyz": np.asarray(physics["k_xyz"], dtype=np.float64)[fixed_indices],
                    "q_W_m3": np.asarray(physics["q_W_m3"], dtype=np.float64)[fixed_indices],
                    "layer_id": fixed_layer}

    first_example = highn._query_example(first, support_for(first), full["coords"])
    builder = candidate._builder(args.candidate, anchor=first, runtime=runtime, graph_key=graph_key,
                                 physical_node_count=args.resolution, optimization_mode="baseline")
    phase = time.perf_counter(); fresh_metadata = builder.build_metadata(
        highn.runner._graph_coords_for_example(first_example, stats), key=graph_key)
    jax.block_until_ready(fresh_metadata.r_rnodes); graph_build_s = time.perf_counter() - phase
    cache_file = args.cache_root / "static_graph_metadata.npz"
    cache_hit = cache_file.is_file()
    save_info = None if cache_hit else highn.save_metadata(cache_file, fresh_metadata)
    phase = time.perf_counter(); metadata, load_info = highn.load_metadata(cache_file)
    graph_load_s = time.perf_counter() - phase
    if metadata_hash(metadata) != metadata_hash(fresh_metadata) or graph_hash(metadata) != graph_hash(fresh_metadata):
        raise RuntimeError("static graph cache equivalence failed")
    edge_targets = {field: None if getattr(metadata, field) is None else int(getattr(metadata, field).shape[1])
                    for field in candidate.qualification.EDGE_FIELDS}
    template_full = highn._model_group(highn._prepare_group(
        example=first_example, anchor=first, runtime=runtime, builder=builder,
        metadata=metadata, edge_targets=edge_targets))
    template_cached = cached_cpu_group(first_example, first, runtime, metadata, template_full)
    if highn._tree_sha256(template_full) != highn._tree_sha256(template_cached):
        raise RuntimeError("static/dynamic packing template mismatch")
    phase = time.perf_counter(); static_device = jax.device_put(template_cached)
    jax.block_until_ready(jax.tree.leaves(static_device)); static_h2d_s = time.perf_counter() - phase
    phase = time.perf_counter(); device_map = None if reconstruction is None else to_device_reconstruction_map(reconstruction)
    if device_map is not None: jax.block_until_ready((device_map.neighbor_local_indices, device_map.neighbor_weights))
    reconstruction_h2d_s = time.perf_counter() - phase
    fixed_weights = jax.device_put(np.asarray(fixed_cv, dtype=np.float32))

    model = GraphNeuralOperator(**model_config); params = highn.runner._device_params(checkpoint["params"])
    @jax.jit
    def model_core(model_params, group, weights, frozen_scale):
        output = highn.runner._model_apply(model, model_params, group)
        delta = output["raw_temperature"][0, 0, :, 0] - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights); query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * frozen_scale
    @jax.jit
    def reconstruct(support_delta):
        if device_map is None: return support_delta
        return device_map.reconstruct(support_delta)

    with np.load(args.native_predictions, allow_pickle=False) as native:
        ids = [str(v) for v in np.asarray(native["sample_ids"]).tolist()]
        scales = {sid: float(np.asarray(native["predicted_scales"])[ids.index(sid)]) for sid in preflight["sample_ids"]}
    compile_started = time.perf_counter()
    compiled_support = model_core(params, static_device, fixed_weights, jnp.asarray(scales[first.sample_id]))
    compiled_full = reconstruct(compiled_support); jax.block_until_ready(compiled_full)
    compile_s = time.perf_counter() - compile_started

    dynamic_prepare=[]; h2d=[]; forward=[]; recon=[]; total=[]; max_errors=[]; rmse_errors=[]; tree_exact=[]
    reference_repeat_max=[]; reference_repeat_rmse=[]
    for anchor in anchors:
        support = support_for(anchor); example = highn._query_example(anchor, support, full["coords"])
        reference_cpu = highn._model_group(highn._prepare_group(
            example=example, anchor=anchor, runtime=runtime, builder=builder,
            metadata=metadata, edge_targets=edge_targets))
        reference_device = jax.device_put(reference_cpu)
        reference_support = model_core(params, reference_device, fixed_weights, jnp.asarray(scales[anchor.sample_id]))
        reference_full = reconstruct(reference_support); jax.block_until_ready(reference_full)
        reference_repeat_support = model_core(params, reference_device, fixed_weights, jnp.asarray(scales[anchor.sample_id]))
        reference_repeat_full = reconstruct(reference_repeat_support); jax.block_until_ready(reference_repeat_full)
        control_max, control_rmse = difference(reference_full, reference_repeat_full)
        reference_repeat_max.append(control_max); reference_repeat_rmse.append(control_rmse)

        started = time.perf_counter(); phase = time.perf_counter()
        cached_cpu = cached_cpu_group(example, anchor, runtime, metadata, template_cached)
        dynamic_prepare.append(time.perf_counter() - phase)
        phase = time.perf_counter(); cached_device, dynamic_leaves = merge_dynamic_device(static_device, cached_cpu)
        device_scale = jax.device_put(np.asarray(scales[anchor.sample_id], dtype=np.float32))
        jax.block_until_ready(dynamic_leaves + [device_scale]); h2d.append(time.perf_counter() - phase)
        phase = time.perf_counter(); cached_support = model_core(params, cached_device, fixed_weights, device_scale)
        jax.block_until_ready(cached_support); forward.append(time.perf_counter() - phase)
        phase = time.perf_counter(); cached_full = reconstruct(cached_support)
        jax.block_until_ready(cached_full); recon.append(time.perf_counter() - phase)
        total.append(time.perf_counter() - started)
        # Qualification is deliberately outside the production timing span.
        tree_exact.append(highn._tree_sha256(reference_cpu) == highn._tree_sha256(cached_cpu))
        maximum, rms = difference(reference_full, cached_full); max_errors.append(maximum); rmse_errors.append(rms)
    tolerance = protocol["equivalence_tolerance"]
    equivalence = {"group_tree_all_exact": all(tree_exact), "prediction_max_abs_K": max(max_errors),
                   "prediction_max_rmse_K": max(rmse_errors),
                   "same_gpu_reference_repeat_max_abs_K": max(reference_repeat_max),
                   "same_gpu_reference_repeat_max_rmse_K": max(reference_repeat_rmse),
                   "passed": all(tree_exact)
                             and max(max_errors) <= tolerance["same_gpu_prediction_max_abs_K"]
                             and max(rmse_errors) <= tolerance["same_gpu_prediction_rmse_K"]}
    baseline_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_performance_p2_raw" / f"{args.candidate}_{args.resolution}.json"
    baseline = json.loads(baseline_path.read_text())
    old_median = baseline["timing"]["continuous_total"]["median_seconds"]
    new_median = dist(total)["median_seconds"]
    benefit = float(old_median) / new_median
    payload = {"schema_version": "heat3d_v6_p1i_static_cache_v1",
               "status": "passed" if equivalence["passed"] else "failed_equivalence",
               "candidate": args.candidate, "resolution": args.resolution, "sample_count": 32,
               "static_graph_cache": {"path": str(cache_file), "file_sha256": file_sha(cache_file),
                                      "metadata_hash": metadata_hash(metadata), "graph_hash": graph_hash(metadata),
                                      "build_seconds": graph_build_s, "save": save_info, "load_seconds": graph_load_s,
                                      "cache_hit": cache_hit, "fresh_cached_exact": True},
               "reconstruction_cache": {"path": None if reconstruction_file is None else str(reconstruction_file),
                                         "file_sha256": None if reconstruction_file is None else file_sha(reconstruction_file),
                                         "load_seconds": reconstruction_load_s, "h2d_seconds": reconstruction_h2d_s},
               "static_packing": {"static_h2d_seconds": static_h2d_s,
                                  "static_fields": protocol["cache_scope"]["static_packing"],
                                  "dynamic_fields": protocol["cache_scope"]["dynamic_packing"]},
               "jax_compilation_cache": {"enabled": compilation_enabled, "directory": str(compilation_dir),
                                         "compile_and_first_apply_seconds": compile_s,
                                         "file_count_after": sum(1 for p in compilation_dir.rglob("*") if p.is_file())},
               "equivalence": equivalence,
               "timing": {"static_graph_load": graph_load_s, "reconstruction_map_load": reconstruction_load_s,
                          "dynamic_input_preparation": dist(dynamic_prepare), "dynamic_h2d": dist(h2d),
                          "forward": dist(forward), "reconstruction": dist(recon),
                          "known_support_new_physics_total": dist(total)},
               "baseline_known_support_median_s": old_median, "speedup_vs_p2_dynamic_full_pack": benefit,
               "promote": bool(equivalence["passed"] and benefit > 1.0),
               "memory": candidate.publication._device_memory(),
               "role_contract": {"valid_iid": True, "training": False, "test": False, "sealed": False,
                                 "temperature_read": False, "metrics_computed": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: value.item() if isinstance(value,np.generic) else value) + "\n")
    print(json.dumps({"status": payload["status"], "route": f"{args.candidate}{args.resolution}",
                      "median_s": new_median, "speedup": benefit, "promote": payload["promote"]}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
