#!/usr/bin/env python3
"""Matched valid32 P5 adaptive preprocessing closeout for B8192/E32768."""

from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import sys
from typing import Any
import h5py
import jax
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for value in (ROOT,ROOT/"scripts"):
    if str(value) not in sys.path:sys.path.insert(0,str(value))
import benchmark_heat3d_v6_p1i_p5_a1_support_ordering as a1  # noqa:E402
import benchmark_heat3d_v6_p1i_p5_a2_cv_redistribution as a2  # noqa:E402
import benchmark_heat3d_v6_p1i_p5_a3_coverage_radius as a3  # noqa:E402
import benchmark_heat3d_v6_p1i_p5_a4_p2r_r2p as a4  # noqa:E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa:E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa:E402
from rigno.heat3d_v6_full_field import build_reconstruction_map,prepare_reconstruction_domain_partition  # noqa:E402
from rigno.heat3d_v6_p1i_anchor_query import _weighted_interleave,array_sha256  # noqa:E402
from rigno.models.rigno import RegionInteractionGraphBuilder  # noqa:E402

STAGES=("support_ordering","cv_redistribution","regional_prepare","coverage","p2r","r2r","r2p","reconstruction_map","packing","graph_total","total_adaptive_preprocessing")

def _stats(v:list[float])->dict[str,float]:
    a=np.asarray(v,dtype=np.float64);return {"median_seconds":float(np.median(a)),"mean_seconds":float(np.mean(a)),"p95_seconds":float(np.quantile(a,.95)),"sum_seconds":float(np.sum(a))}

def main()->int:
    p=argparse.ArgumentParser()
    for name in ("protocol","binding","artifact_root","dataset_root","manifest","full_fields","run_dir","output_json","output_csv","output_md"):p.add_argument("--"+name.replace("_","-"),dest=name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());binding=json.loads(a.binding.read_text());runtime=highn._checkpoint_runtime(a);anchors=highn._valid_examples(highn._dataset(a),binding);preflight=json.loads((a.artifact_root/"actual_data_preflight.json").read_text());supports={str(n):{r["sample_id"]:r for r in preflight["supports"][str(n)]} for n in (8192,32768)}
    with h5py.File(a.full_fields,"r") as archive:
        coords=np.asarray(archive["shared/coords_m"][:],dtype=np.float64);cv=np.asarray(archive["shared/control_volume_m3"][:],dtype=np.float64);layer=np.asarray(archive["shared/layer_id"][:],dtype=np.int32)
    boundaries=highn._boundaries(anchors[0],float(np.min(coords[:,2])));partition=prepare_reconstruction_domain_partition(coords=coords,layer_id=layer,boundaries=boundaries);key=highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]));production_coverage=RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius;production_edges=RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes
    ordering={}
    for number,anchor in enumerate(anchors,start=1):
        with np.load(a.artifact_root/"physics"/f"{anchor.sample_id}.npz") as payload:q=np.asarray(payload["q_W_m3"],dtype=np.float64)
        frozen=highn._load_support(Path(supports["8192"][anchor.sample_id]["support_file"]));anchor_indices=np.asarray(frozen["selected_indices"][:1024],dtype=np.int64);sample_boundaries=highn._boundaries(anchor,float(np.min(coords[:,2])))
        ref_order,ref_audit,ref_timing=a1._run_order(interleave=a1._reference_weighted_interleave,sample_id=anchor.sample_id,anchors=anchor_indices,coords=coords,cv=cv,layer=layer,q=q,boundaries=sample_boundaries)
        cand_order,cand_audit,cand_timing=a1._run_order(interleave=_weighted_interleave,sample_id=anchor.sample_id,anchors=anchor_indices,coords=coords,cv=cv,layer=layer,q=q,boundaries=sample_boundaries)
        ordering[anchor.sample_id]={"reference":ref_order,"candidate":cand_order,"reference_seconds":ref_timing["full_order_seconds"],"candidate_seconds":cand_timing["full_order_seconds"],"exact":bool(np.array_equal(ref_order,cand_order) and ref_audit["order_sha256"]==cand_audit["order_sha256"])};print(f"[P5-closeout] ordering {number}/32",flush=True)
    rows=[]
    for resolution,route,factor in ((8192,"B8192_adaptive",8.0),(32768,"E32768_adaptive",128.0)):
      for number,anchor in enumerate(anchors,start=1):
        frozen=highn._load_support(Path(supports[str(resolution)][anchor.sample_id]["support_file"]));selected=np.asarray(frozen["selected_indices"],dtype=np.int64);order=ordering[anchor.sample_id]
        if not np.array_equal(order["candidate"][:resolution],selected):raise RuntimeError("frozen adaptive support order drift")
        started=time.perf_counter();ref_cv,ref_assign,_=a2._redistribute(coords,cv,layer,selected,workers=1);ref_cv_s=time.perf_counter()-started
        started=time.perf_counter();cand_cv,cand_assign,_=a2._redistribute(coords,cv,layer,selected,workers=-1);cand_cv_s=time.perf_counter()-started
        if not np.array_equal(cand_cv,np.asarray(frozen["operator_control_volume"],dtype=np.float64)):raise RuntimeError("frozen selected CV drift")
        example=highn._query_example(anchor,frozen,coords);graph_coords=highn.runner._graph_coords_for_example(example,runtime["stats"]);config=dict(runtime["graph_config"]);config.update(subsample_factor=factor,discrete_graph_backend="sparse_kdtree_v1",reuse_exact_p2r_for_r2p=False)
        ref_builder=Heat3DGraphBuilder(**config);RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius=a3._reference_coverage;RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes=a4._reference_edges
        try:ref_meta=ref_builder.build_metadata(graph_coords,key=key);jax.block_until_ready(ref_meta.r_rnodes)
        finally:RegionInteractionGraphBuilder._compute_discrete_physical_coverage_radius=production_coverage;RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes=production_edges
        cand_config=dict(config);cand_config["reuse_exact_p2r_for_r2p"]=True;cand_builder=Heat3DGraphBuilder(**cand_config);cand_meta=cand_builder.build_metadata(graph_coords,key=key);jax.block_until_ready(cand_meta.r_rnodes)
        ref_map_started=time.perf_counter();ref_map,_=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback="same_layer",prepared_partition=None,query_workers=1);ref_map_s=time.perf_counter()-ref_map_started
        cand_map_started=time.perf_counter();cand_map,_=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback="same_layer",prepared_partition=partition,query_workers=-1);cand_map_s=time.perf_counter()-cand_map_started
        ref_t=ref_builder.builder.last_build_timings;cand_t=cand_builder.builder.last_build_timings
        ref_stages={"support_ordering":order["reference_seconds"],"cv_redistribution":ref_cv_s,"regional_prepare":float(ref_t["regional_prepare_seconds"]),"coverage":float(ref_t["coverage_radius_seconds"]),"p2r":float(ref_t["p2r_seconds"]),"r2r":float(ref_t["r2r_seconds"]),"r2p":float(ref_t["r2p_seconds"]),"reconstruction_map":ref_map_s,"packing":float(ref_t["packing_seconds"]),"graph_total":float(ref_t["total_seconds"])}
        cand_stages={"support_ordering":order["candidate_seconds"],"cv_redistribution":cand_cv_s,"regional_prepare":float(cand_t["regional_prepare_seconds"]),"coverage":float(cand_t["coverage_radius_seconds"]),"p2r":float(cand_t["p2r_seconds"]),"r2r":float(cand_t["r2r_seconds"]),"r2p":float(cand_t["r2p_seconds"]),"reconstruction_map":cand_map_s,"packing":float(cand_t["packing_seconds"]),"graph_total":float(cand_t["total_seconds"])}
        named=("support_ordering","cv_redistribution","regional_prepare","coverage","p2r","r2r","r2p","reconstruction_map","packing");ref_stages["total_adaptive_preprocessing"]=sum(ref_stages[k] for k in named);cand_stages["total_adaptive_preprocessing"]=sum(cand_stages[k] for k in named)
        gates={"support_order_exact":order["exact"],"cv_exact":bool(np.array_equal(ref_cv,cand_cv) and np.array_equal(ref_assign,cand_assign) and array_sha256(ref_cv)==array_sha256(cand_cv)),"canonical_graph_hash_exact":a4._canonical_hash(ref_meta)==a4._canonical_hash(cand_meta),"reconstruction_map_exact":highn._mapping_sha256(ref_map)==highn._mapping_sha256(cand_map)}
        rows.append({"route":route,"resolution":resolution,"sample_id":anchor.sample_id,"gates":gates,"reference":ref_stages,"candidate":cand_stages});print(f"[P5-closeout] {route} {number}/32 exact={all(gates.values())}",flush=True)
    hard=all(v for row in rows for v in row["gates"].values());summary={}
    for route in ("B8192_adaptive","E32768_adaptive"):
      selected=[r for r in rows if r["route"]==route];stages={}
      for stage in STAGES:
        ref=_stats([r["reference"][stage] for r in selected]);cand=_stats([r["candidate"][stage] for r in selected]);stages[stage]={"reference":ref,"candidate":cand,"median_speedup":ref["median_seconds"]/max(cand["median_seconds"],1e-30)}
      ranked=sorted(((name,data["candidate"]["median_seconds"]) for name,data in stages.items() if name not in ("graph_total","total_adaptive_preprocessing")),key=lambda x:x[1],reverse=True);summary[route]={"stages":stages,"remaining_bottleneck":ranked[0][0],"remaining_bottleneck_median_seconds":ranked[0][1]}
    result={"schema_version":"heat3d_v6_p1i_p5_adaptive_preprocessing_closeout_v1","status":"passed" if hard else "failed","protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),"hard_gate_passed":hard,"population":protocol["population"],"summary":summary,"samples":rows,"decision":{"optimized_adaptive_preprocessing":"GO","gpu_tiled":"NO_GO","next_stage":"GO_batch_or_offline_parallel_support_ordering; NO_GO_more_graph_policy_or_GPU_tiled_search"},"role_contract":protocol["role_contract"]};a.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with a.output_csv.open("w",newline="") as handle:
      writer=csv.writer(handle);writer.writerow(["route","stage","reference_median_s","candidate_median_s","median_speedup","candidate_p95_s"])
      for route,data in summary.items():
        for stage,s in data["stages"].items():writer.writerow([route,stage,s["reference"]["median_seconds"],s["candidate"]["median_seconds"],s["median_speedup"],s["candidate"]["p95_seconds"]])
    lines=["# V6 P1i P5 adaptive preprocessing closeout","",f"Status: **{'PASS' if hard else 'FAIL'}**. Frozen valid32; no inference, training, test or sealed access.",""]
    for route,data in summary.items():
      lines += [f"## {route}","","| Stage | Reference median (s) | Candidate median (s) | Speedup |","|---|---:|---:|---:|"]
      for stage,s in data["stages"].items():lines.append(f"| {stage} | {s['reference']['median_seconds']:.6f} | {s['candidate']['median_seconds']:.6f} | {s['median_speedup']:.3f}x |")
      lines += ["",f"Remaining largest named stage: `{data['remaining_bottleneck']}` ({data['remaining_bottleneck_median_seconds']:.6f} s median).",""]
    lines += ["## Decision","","All promoted changes are exact-equivalent. Keep the CPU sparse path and exact reverse reuse. Do not resume GPU-tiled or graph-policy search. The next justified engineering step is batch/offline parallel support-order preparation; inference batching remains a separate later study."]
    a.output_md.write_text("\n".join(lines)+"\n")
    if not hard:raise RuntimeError("P5 closeout exact gate failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
