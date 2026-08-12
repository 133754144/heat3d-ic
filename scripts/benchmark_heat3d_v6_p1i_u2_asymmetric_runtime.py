#!/usr/bin/env python3
"""U2 checkpoint-preserving asymmetric-query production benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as u1  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map, prepare_reconstruction_domain_partition  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    return {"count": len(x), "median_seconds": float(np.median(x)), "mean_seconds": float(np.mean(x)),
            "std_seconds": float(np.std(x)), "p95_seconds": float(np.quantile(x, .95))}


def block(tree: Any) -> None:
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def stack(trees: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: np.concatenate([np.asarray(x) for x in xs], axis=0), *trees)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields",
                 "run_dir", "native_padding_result", "query_padding_result", "output"):
        parser.add_argument(f"--{name.replace('_','-')}", dest=name, type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=[32768, 240825], required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sample-count", type=int, choices=[1, 32], default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-sizes", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("U2 requires GPU")
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_before_execution":
        raise RuntimeError("U2 protocol not frozen")
    direct = args.resolution == 240825
    runtime = p5r._runtime(args)
    checkpoint_parameter_sha256_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    binding = json.loads(args.binding.read_text())
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    if len(anchors) != 32:
        raise RuntimeError("U2 requires frozen valid32")
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    if preflight["sample_ids"] != [x.sample_id for x in anchors]:
        raise RuntimeError("valid32 order drift")
    anchors = anchors[:args.sample_count]
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}
    full, archive_lookup = highn._full_shared(args)
    coords = np.asarray(full["coords"], dtype=np.float64); cv = np.asarray(full["cv"], dtype=np.float64)
    layer = np.asarray(full["layer"], dtype=np.int32)
    boundaries = highn._boundaries(anchors[0], float(np.min(coords[:,2])))
    geometry = prepare_nested_query_geometry_cache(
        full_coords=coords, full_control_volume=cv, full_layer_id=layer, layer_boundaries_m=boundaries)
    partition = prepare_reconstruction_domain_partition(coords=coords, layer_id=layer, boundaries=boundaries)
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    graph_config = dict(runtime["graph_config"]); graph_config.update(
        subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1", reuse_exact_p2r_for_r2p=True)
    native_targets = p5r._edge_targets(args.native_padding_result)
    query_targets = p5r._edge_targets(args.query_padding_result)
    gpu = jax.devices("gpu")[0]; cpu = jax.devices("cpu")[0]
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])

    def support(anchor: Any, full_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anchor_indices, _ = highn._anchor_indices(
            anchor, coords, float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]))
        if direct:
            return np.arange(len(coords), dtype=np.int64), cv
        selected, _ = deterministic_nested_query_prefix(
            sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
            target_count=args.resolution, geometry_cache=geometry)
        selected_cv, _ = conservative_selected_control_volume(
            full_coords=coords, full_control_volume=cv, full_layer_id=layer, selected_indices=selected)
        return selected, selected_cv

    # Shape envelope is qualification-only and outside all timed spans.
    tracked_native = dict(native_targets); tracked_query = dict(query_targets)
    for anchor in anchors:
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            full_k = np.asarray(physics["k_xyz"], dtype=np.float64)
        selected, selected_cv = support(anchor, full_q)
        query_support = {"selected_indices": selected, "operator_control_volume": selected_cv,
                         "k_xyz": full_k[selected], "q_W_m3": full_q[selected], "layer_id": layer[selected]}
        query_example = highn._query_example(anchor, query_support, coords)
        builder = Heat3DGraphBuilder(**graph_config)
        with jax.default_device(cpu):
            anchor_coords = highn.runner._graph_coords_for_example(anchor, runtime["stats"])
            native = builder.build_metadata(anchor_coords, key=graph_key); block(native)
            asymmetric, audit = u1.prior_u1._strict_asymmetric_metadata(
                builder, native, anchor_coords,
                highn.runner._graph_coords_for_example(query_example, runtime["stats"]))
            if not audit["query_inside_native_domain"]:
                raise RuntimeError("query outside native domain")
            for targets, metadata in ((native_targets, native), (query_targets, asymmetric)):
                for field in qualification.EDGE_FIELDS:
                    value = getattr(metadata, field)
                    if value is not None:
                        targets[field] = max(int(targets.get(field) or 0), int(np.asarray(value).shape[1]))
    asymmetric_targets = u1._combined_targets(native_targets, query_targets)

    @jax.jit
    def split_forward(model_params: Any, inputs_in: Any, inputs_out: Any, graphs: Any, local_p2r: Any, kwargs: Any) -> Any:
        output = model.apply(
            {"params": model_params}, inputs_in=inputs_in, inputs_out=inputs_out,
            graphs=graphs, output_local_p2r=local_p2r, split=True,
            method=u1._trace_method, **kwargs)
        return output["raw_temperature"][:,0,:,0] - highn.REFERENCE_K

    @jax.jit
    def reconstruct(values: Any, indices: Any, weights: Any) -> Any:
        if direct:
            return values
        gathered = values[jnp.arange(values.shape[0])[:,None,None], indices]
        return jnp.sum(gathered * weights.astype(values.dtype), axis=2)

    def prepare_one(anchor: Any) -> dict[str, Any]:
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
            full_k = np.asarray(physics["k_xyz"], dtype=np.float64)
            full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
        anchor_indices, _ = highn._anchor_indices(
            anchor, coords, float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]))
        anchor_support = {"selected_indices": anchor_indices,
                          "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
                          "k_xyz": np.asarray(anchor.condition.condition_features[:,:3], dtype=np.float64),
                          "q_W_m3": np.asarray(anchor.condition.condition_features[:,3], dtype=np.float64),
                          "layer_id": layer[anchor_indices]}
        total_start = time.perf_counter(); phase = time.perf_counter()
        selected, selected_cv = support(anchor, full_q); support_s = time.perf_counter()-phase
        query_support = {"selected_indices": selected, "operator_control_volume": selected_cv,
                         "k_xyz": full_k[selected], "q_W_m3": full_q[selected], "layer_id": layer[selected]}
        query_example = highn._query_example(anchor, query_support, coords)
        builder = Heat3DGraphBuilder(**graph_config)
        with jax.default_device(cpu):
            anchor_coords = highn.runner._graph_coords_for_example(anchor, runtime["stats"])
            phase=time.perf_counter(); native=builder.build_metadata(anchor_coords,key=graph_key);block(native)
            anchor_graph_s=time.perf_counter()-phase
            phase=time.perf_counter(); asymmetric,audit=u1.prior_u1._strict_asymmetric_metadata(
                builder,native,anchor_coords,highn.runner._graph_coords_for_example(query_example,runtime["stats"]));block(asymmetric)
            query_graph_s=time.perf_counter()-phase
            if not audit["query_inside_native_domain"]: raise RuntimeError("query domain")
            phase=time.perf_counter(); anchor_group=host_tree(highn._prepare_group(
                example=anchor,anchor=anchor,runtime=runtime,builder=builder,metadata=native,
                edge_targets=p5r._compatible_targets(native_targets,native))); anchor_pack_s=time.perf_counter()-phase
            phase=time.perf_counter(); output_group=host_tree(highn._prepare_group(
                example=query_example,anchor=anchor,runtime=runtime,builder=builder,metadata=asymmetric,
                edge_targets=p5r._compatible_targets(asymmetric_targets,asymmetric))); query_pack_s=time.perf_counter()-phase
            detail_started=time.perf_counter()
            phase=time.perf_counter(); graphs_raw=output_group["graphs"]; graph_extraction_s=time.perf_counter()-phase
            phase=time.perf_counter(); local_raw=u1._dummy_local_p2r(builder,asymmetric); dummy_local_p2r_s=time.perf_counter()-phase
            phase=time.perf_counter(); graphs=host_tree(graphs_raw); local=host_tree(local_raw); host_tree_s=time.perf_counter()-phase
            phase=time.perf_counter(); inputs_in=host_tree(anchor_group["inputs"]); inputs_out=host_tree(output_group["inputs"]); inputs_s=time.perf_counter()-phase
            phase=time.perf_counter(); kwargs=host_tree(u1._model_kwargs(anchor_group,output_group)); kwargs_s=time.perf_counter()-phase
            detail_total_s=time.perf_counter()-detail_started
            other_s=max(0.0,detail_total_s-(graph_extraction_s+dummy_local_p2r_s+host_tree_s+inputs_s+kwargs_s))
        phase=time.perf_counter(); mapping=None if direct else build_reconstruction_map(
            coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,
            empty_domain_fallback="same_layer",prepared_partition=partition,query_workers=-1)[0]
        map_s=time.perf_counter()-phase
        if direct:
            map_indices=np.zeros((1,1,1),dtype=np.int32); map_weights=np.ones((1,1,1),dtype=np.float64)
        else:
            map_indices=np.asarray(mapping.neighbor_local_indices,dtype=np.int32)[None,:,:]
            map_weights=np.asarray(mapping.neighbor_weights,dtype=np.float64)[None,:,:]
        phase=time.perf_counter(); device=jax.device_put((inputs_in,inputs_out,graphs,local,kwargs,map_indices,map_weights),gpu)
        enqueue_s=time.perf_counter()-phase; phase=time.perf_counter();block(device);sync_s=time.perf_counter()-phase
        ii,io,g,l,kw,mi,mw=device
        phase=time.perf_counter(); values=split_forward(params,ii,io,g,l,kw);block(values);forward_s=time.perf_counter()-phase
        phase=time.perf_counter(); full_value=reconstruct(values,mi,mw);block(full_value);recon_s=time.perf_counter()-phase
        return {"sample_id":anchor.sample_id,"inputs_in":inputs_in,"inputs_out":inputs_out,"graphs":graphs,"local":local,
                "kwargs":kwargs,"map_indices":map_indices,"map_weights":map_weights,"device":device,"selected":selected,
                "selected_cv":selected_cv,"full_q":full_q,"support_delta":values,"full_delta":full_value,
                "stages":{"support_plus_cv":support_s,"anchor_graph":anchor_graph_s,"query_graph":query_graph_s,
                          "reconstruction_map":map_s,"anchor_group_pack":anchor_pack_s,"query_group_pack":query_pack_s,
                          "h2d_enqueue":enqueue_s,"h2d_sync":sync_s,"asymmetric_forward":forward_s,
                          "reconstruction_apply":recon_s,"dummy_local_p2r":dummy_local_p2r_s,
                          "graph_extraction":graph_extraction_s,"host_tree":host_tree_s,"inputs":inputs_s,
                          "kwargs":kwargs_s,"profiled_other":other_s,
                          "matched_continuous_e2e":time.perf_counter()-total_start},
                "shape":{"output_nodes":int(np.asarray(values).shape[1]),"regional_nodes":int(np.asarray(asymmetric.x_rnodes).shape[1]-1),
                         "p2r_edges":int(np.asarray(asymmetric.p2r_edge_indices).shape[1]),
                         "r2r_edges":int(np.asarray(asymmetric.r2r_edge_indices).shape[1]),
                         "r2p_edges":int(np.asarray(asymmetric.r2p_edge_indices).shape[1])}}

    warm=prepare_one(anchors[0]); ii,io,g,l,kw,mi,mw=warm["device"]
    value=split_forward(params,ii,io,g,l,kw);block(reconstruct(value,mi,mw))
    prepared=[prepare_one(anchor) for anchor in anchors]
    if any(row["shape"]["output_nodes"] != args.resolution for row in prepared): raise RuntimeError("output shape")
    if any(not np.all(np.isfinite(np.asarray(row["full_delta"]))) for row in prepared): raise RuntimeError("nonfinite")
    stage_keys=list(prepared[0]["stages"]); timing={k:dist([r["stages"][k] for r in prepared]) for k in stage_keys}
    replay=[]
    for _ in range(args.repeats):
        phase=time.perf_counter(); value=split_forward(params,ii,io,g,l,kw);full_value=reconstruct(value,mi,mw);block(full_value)
        replay.append(time.perf_counter()-phase)
    batch_rows=[]
    if args.batch_sizes is not None:
        sizes=[int(value) for value in args.batch_sizes.split(",")]
    elif not direct:
        sizes=protocol.get("batch_sizes_32768",protocol.get("u1_32768",{}).get("batch_sizes",[1]))
    else:
        sizes=[1]
    limit=int(candidate.publication._device_memory().get("bytes_limit",0)); one_peak=int(candidate.publication._device_memory().get("peak_bytes_in_use",0))
    t1=None
    for b in sizes:
        if b>len(prepared): continue
        if args.batch_sizes is None and limit and b*one_peak>0.8*limit:
            batch_rows.append({"batch_size":b,"status":"skipped_memory_feasibility","estimated_peak_bytes":b*one_peak});continue
        subset=prepared[:b]; host=(stack([r["inputs_in"] for r in subset]),stack([r["inputs_out"] for r in subset]),
            stack([r["graphs"] for r in subset]),stack([r["local"] for r in subset]),stack([r["kwargs"] for r in subset]),
            np.concatenate([r["map_indices"] for r in subset]),np.concatenate([r["map_weights"] for r in subset]))
        device=jax.device_put(host,gpu);block(device); bii,bio,bg,bl,bkw,bmi,bmw=device
        value=split_forward(params,bii,bio,bg,bl,bkw);block(reconstruct(value,bmi,bmw))
        forward=[];prepared_times=[];streamed=[]
        for _ in range(args.repeats):
            phase=time.perf_counter();value=split_forward(params,bii,bio,bg,bl,bkw);block(value);forward.append(time.perf_counter()-phase)
            phase=time.perf_counter();value=split_forward(params,bii,bio,bg,bl,bkw);full_value=reconstruct(value,bmi,bmw);block(full_value);prepared_times.append(time.perf_counter()-phase)
            phase=time.perf_counter();dev=jax.device_put(host,gpu);block(dev);sii,sio,sg,sl,skw,smi,smw=dev
            value=split_forward(params,sii,sio,sg,sl,skw);full_value=reconstruct(value,smi,smw);block(full_value);streamed.append(time.perf_counter()-phase)
        stats=dist(prepared_times);wall=stats["median_seconds"];t1=wall if t1 is None else t1
        batch_rows.append({"batch_size":b,"status":"passed","resident_forward_only":dist(forward),
            "prepared_group_steady_inference":stats,"streamed_prepared_host_batch":dist(streamed),"batch_wall_seconds":wall,
            "samples_per_second":b/wall,"average_per_case_seconds":wall/b,
            "marginal_per_case_seconds":None if b==1 else (wall-t1)/(b-1),
            "peak_vram_bytes":int(candidate.publication._device_memory().get("peak_bytes_in_use",0))})

    metric_support=[];metric_full=[]
    with h5py.File(args.full_fields,"r") as archive:
        for anchor,row in zip(anchors,prepared,strict=True):
            truth=np.asarray(archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]],dtype=np.float64)
            support_np=np.asarray(row["support_delta"],dtype=np.float64)[0];full_np=np.asarray(row["full_delta"],dtype=np.float64)[0]
            metric_support.append(highn._metric_row(support_np,truth[row["selected"]],row["selected_cv"],coords[row["selected"]],layer[row["selected"]],row["full_q"][row["selected"]]))
            metric_full.append(highn._metric_row(full_np,truth,cv,coords,layer,row["full_q"]))
    result={"schema_version":"heat3d_v6_p1i_u2_asymmetric_runtime_cell_v1","status":"passed" if args.sample_count==32 else "passed_smoke",
        "resolution":args.resolution,"output_mode":"direct" if direct else "reconstruction","sample_count":args.sample_count,
        "protocol_sha256":sha256(args.protocol),"checkpoint_sha256":args.checkpoint_sha256,
        "checkpoint_parameter_sha256_before": checkpoint_parameter_sha256_before,
        "checkpoint_parameter_sha256_after": highn._tree_sha256(runtime["checkpoint"]["params"]),
        "checkpoint_parameters_unchanged":checkpoint_parameter_sha256_before==highn._tree_sha256(runtime["checkpoint"]["params"]),
        "accuracy":{"support":qualification.metric_accumulate(metric_support,full=False),"full_field":qualification.metric_accumulate(metric_full,full=True)},
        "runtime":{"fresh_sample":timing,"same_input_replay":dist(replay)},"batch":batch_rows,
        "padding":{"tracked_native":tracked_native,"actual_native":native_targets,"tracked_query":tracked_query,"actual_query":query_targets},
        "memory":candidate.publication._device_memory(),"samples":[{"sample_id":r["sample_id"],"stages":r["stages"],"shape":r["shape"]} for r in prepared],
        "role_contract":protocol["role_contract"]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"resolution":args.resolution,"pg":result["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"],"e2e":timing["matched_continuous_e2e"]["median_seconds"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
