#!/usr/bin/env python3
"""Evaluate frozen fixed-support production candidates on valid32 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
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


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--route",choices=["B8192_recon","E32768_recon"],required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--full-fields",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--baseline-artifact-root",type=Path,required=True)
    parser.add_argument("--native-predictions",type=Path,required=True)
    parser.add_argument("--adaptive-b-predictions",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text()); assert protocol["status"]=="frozen_before_p4a_execution"
    route=protocol["routes"][args.route]; policy=route["policy"]; resolution=int(route["resolution"])
    adaptive_path=ROOT/route["adaptive_reference_path"]
    if file_sha(adaptive_path)!=route["adaptive_reference_sha256"]: raise RuntimeError("adaptive reference SHA drift")
    adaptive=json.loads(adaptive_path.read_text())
    preflight=json.loads(args.preflight.read_text()); sample_ids=preflight["sample_ids"]
    dataset=highn.Heat3DV6DualRobinDataset(args.dataset_root,args.manifest,include_roles={"valid_iid"})
    by_id=dataset.sample_index_by_id(); anchors=[dataset[by_id[sid]] for sid in sample_ids]
    full,archive_lookup=highn._full_shared(args); first=anchors[0]
    support_row=next(row for row in preflight["supports"][str(resolution)] if row["sample_id"]==first.sample_id)
    support0=highn._load_support(Path(support_row["support_file"])); fixed_indices=np.asarray(support0["selected_indices"],dtype=np.int64)
    fixed_cv=np.asarray(support0["operator_control_volume"],dtype=np.float64); fixed_layer=np.asarray(support0["layer_id"],dtype=np.int32)
    baseline=json.loads((args.baseline_artifact_root/f"resolution_{resolution}.json").read_text())
    map_row=next(row for row in baseline["reconstruction_cache"]["samples"] if row["sample_id"]==first.sample_id)
    cpu_map=candidate.publication._load_mapping_no_audit(Path(map_row["cache_file"])); device_map=to_device_reconstruction_map(cpu_map)
    physics_rows={row["sample_id"]:row for row in preflight["samples"]}

    checkpoint=runner._load_params_checkpoint(args.run_dir/"params_best_valid_point_global.pkl")
    run_config=json.loads((args.run_dir/"run_config.json").read_text())
    stats=highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"]); highn.install_checkpoint_feature_hooks(stats)
    model_config=runner._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]),stats)
    graph_config=dict(run_config["graph_config"]); graph_config["discrete_graph_backend"]="sparse_kdtree_v1"
    runtime={"checkpoint":checkpoint,"run_config":run_config,"stats":stats,"model_config":model_config,"graph_config":graph_config}
    graph_key=highn.runner._metadata_key(int(run_config["graph_seed"]))

    def support_for(anchor):
        with np.load(physics_rows[anchor.sample_id]["physics_cache_file"],allow_pickle=False) as physics:
            return {"selected_indices":fixed_indices,"operator_control_volume":fixed_cv,
                    "k_xyz":np.asarray(physics["k_xyz"],dtype=np.float64)[fixed_indices],
                    "q_W_m3":np.asarray(physics["q_W_m3"],dtype=np.float64)[fixed_indices],"layer_id":fixed_layer}
    first_example=highn._query_example(first,support_for(first),full["coords"])
    builder=candidate._builder(policy,anchor=first,runtime=runtime,graph_key=graph_key,physical_node_count=resolution,optimization_mode="baseline")
    metadata=builder.build_metadata(highn.runner._graph_coords_for_example(first_example,stats),key=graph_key); jax.block_until_ready(metadata.r_rnodes)
    edge_targets={field:None if getattr(metadata,field) is None else int(getattr(metadata,field).shape[1]) for field in candidate.qualification.EDGE_FIELDS}
    with np.load(args.native_predictions,allow_pickle=False) as native:
        ids=[str(v) for v in np.asarray(native["sample_ids"]).tolist()]
        scales={sid:float(np.asarray(native["predicted_scales"])[ids.index(sid)]) for sid in sample_ids}
    model=GraphNeuralOperator(**model_config); params=highn.runner._device_params(checkpoint["params"])
    @jax.jit
    def apply(model_params,group,weights,scale):
        output=highn.runner._model_apply(model,model_params,group); delta=output["raw_temperature"][0,0,:,0]-highn.REFERENCE_K
        norm=weights/jnp.sum(weights); query_scale=jnp.sqrt(jnp.sum(norm*delta*delta)); support_delta=delta/query_scale*scale
        return support_delta,device_map.reconstruct(support_delta)
    compile_group=highn._model_group(highn._prepare_group(example=first_example,anchor=first,runtime=runtime,builder=builder,metadata=metadata,edge_targets=edge_targets))
    compiled=apply(params,compile_group,jnp.asarray(fixed_cv,dtype=jnp.float32),jnp.asarray(scales[first.sample_id])); jax.block_until_ready(compiled[1])

    adaptive_predictions={}
    if args.route=="B8192_recon":
        if args.adaptive_b_predictions is None or file_sha(args.adaptive_b_predictions)!=adaptive["prediction_artifact"]["sha256"]:
            raise RuntimeError("adaptive B prediction artifact missing/SHA mismatch")
        with np.load(args.adaptive_b_predictions,allow_pickle=False) as payload:
            pids=[str(v) for v in np.asarray(payload["sample_ids"]).tolist()]
            adaptive_predictions={sid:np.asarray(payload["full_deltaT_K"])[pids.index(sid)] for sid in sample_ids}
        adaptive_rows=None
    else:
        adaptive_rows={row["sample_id"]:row for row in adaptive["per_sample_metrics"]}

    fixed_metric_rows=[]; adaptive_metric_rows=[]; paired=[]; predictions=[]
    with h5py.File(args.full_fields,"r") as archive:
        for anchor in anchors:
            support=support_for(anchor); example=highn._query_example(anchor,support,full["coords"])
            group=highn._model_group(highn._prepare_group(example=example,anchor=anchor,runtime=runtime,builder=builder,metadata=metadata,edge_targets=edge_targets))
            support_pred,full_pred=apply(params,group,jnp.asarray(fixed_cv,dtype=jnp.float32),jnp.asarray(scales[anchor.sample_id])); jax.block_until_ready(full_pred)
            full_np=np.asarray(full_pred,dtype=np.float64); predictions.append(full_np)
            truth=np.asarray(archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]],dtype=np.float64)
            with np.load(physics_rows[anchor.sample_id]["physics_cache_file"],allow_pickle=False) as physics: q=np.asarray(physics["q_W_m3"],dtype=np.float64)
            fixed_row=highn._metric_row(full_np,truth,full["cv"],full["coords"],full["layer"],q); fixed_metric_rows.append(fixed_row)
            fixed_single=candidate.qualification.metric_accumulate([fixed_row],full=True)
            if adaptive_predictions:
                adaptive_row=highn._metric_row(adaptive_predictions[anchor.sample_id],truth,full["cv"],full["coords"],full["layer"],q)
                adaptive_metric_rows.append(adaptive_row); adaptive_single=candidate.qualification.metric_accumulate([adaptive_row],full=True)
                adaptive_values={"full_rmse_K":adaptive_single["raw_cv_weighted_rmse_K"],"source_rmse_K":adaptive_single["source_rmse_K"],
                                 "peak_rmse_K":adaptive_single["peak_rmse_K"],"interface_rmse_K":adaptive_single["interface_drop_rmse_K"]}
            else:
                source=adaptive_rows[anchor.sample_id]; adaptive_values={k:source[k] for k in ("full_rmse_K","source_rmse_K","peak_rmse_K","interface_rmse_K")}
            fixed_values={"full_rmse_K":fixed_single["raw_cv_weighted_rmse_K"],"source_rmse_K":fixed_single["source_rmse_K"],
                          "peak_rmse_K":fixed_single["peak_rmse_K"],"interface_rmse_K":fixed_single["interface_drop_rmse_K"]}
            paired.append({"sample_id":anchor.sample_id,**{f"adaptive_{k}":v for k,v in adaptive_values.items()},
                           **{f"fixed_{k}":v for k,v in fixed_values.items()},
                           **{f"delta_{k}":fixed_values[k]-adaptive_values[k] for k in fixed_values}})
    fixed_agg=candidate.qualification.metric_accumulate(fixed_metric_rows,full=True)
    adaptive_agg=adaptive["accuracy"]["full_field"]
    margins=protocol["production_go_gate"]["aggregate_noninferiority_margins"]
    comparisons={"point_global_pct":fixed_agg["point_global_true_rms_relative_rmse_pct"]-adaptive_agg["point_global_true_rms_relative_rmse_pct"],
                 "raw_cv_rmse_K":fixed_agg["raw_cv_weighted_rmse_K"]-adaptive_agg["raw_cv_weighted_rmse_K"],
                 "source_rmse_K":fixed_agg["source_rmse_K"]-adaptive_agg["source_rmse_K"],
                 "peak_rmse_K":fixed_agg["peak_rmse_K"]-adaptive_agg["peak_rmse_K"],
                 "interface_rmse_K":fixed_agg["interface_drop_rmse_K"]-adaptive_agg["interface_drop_rmse_K"]}
    gates={"point_global":comparisons["point_global_pct"]<=margins["point_global_pct_absolute"],
           "raw":comparisons["raw_cv_rmse_K"]<=margins["raw_cv_rmse_K_absolute"],
           "source":comparisons["source_rmse_K"]<=margins["source_rmse_K_absolute"],
           "peak":comparisons["peak_rmse_K"]<=margins["peak_rmse_K_absolute"],
           "interface":comparisons["interface_rmse_K"]<=margins["interface_rmse_K_absolute"],
           "worst_sample_raw":max(row["delta_full_rmse_K"] for row in paired)<=protocol["production_go_gate"]["worst_sample_raw_rmse_delta_K_max"],
           "finite_complete":len(paired)==32 and bool(np.all(np.isfinite(np.asarray(predictions))))}
    output_predictions=args.output.with_suffix(".npz"); np.savez_compressed(output_predictions,sample_ids=np.asarray(sample_ids),full_deltaT_K=np.asarray(predictions))
    result={"schema_version":"heat3d_v6_p1i_p4a_fixed_support_accuracy_v1","status":"passed_execution",
            "route":args.route,"policy":policy,"resolution":resolution,"sample_count":32,
            "fixed_support":{"sample_id":first.sample_id,"selected_indices_sha256":highn.array_sha256(fixed_indices.astype(np.int32)),
                             "metadata_hash":metadata_hash(metadata),"graph_hash":graph_hash(metadata)},
            "fixed_accuracy":fixed_agg,"adaptive_accuracy":adaptive_agg,"fixed_minus_adaptive":comparisons,
            "paired":paired,"worst_case":{"fixed_raw_rmse_K":max(row["fixed_full_rmse_K"] for row in paired),
                                           "adaptive_raw_rmse_K":max(row["adaptive_full_rmse_K"] for row in paired),
                                           "max_raw_delta_K":max(row["delta_full_rmse_K"] for row in paired)},
            "production_gates":gates,"production_go":all(gates.values()),
            "prediction_artifact":{"path":str(output_predictions),"sha256":file_sha(output_predictions),"bytes":output_predictions.stat().st_size},
            "role_contract":{"valid_iid":True,"training":False,"test":False,"sealed":False,"adaptive_accuracy_reexecuted":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True,default=lambda value:value.item() if isinstance(value,np.generic) else value)+"\n")
    print(json.dumps({"status":result["status"],"route":args.route,"production_go":result["production_go"],"fixed_pg":fixed_agg["point_global_true_rms_relative_rmse_pct"],"delta_pg":comparisons["point_global_pct"]}))
    return 0
if __name__=="__main__": raise SystemExit(main())
