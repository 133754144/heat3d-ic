#!/usr/bin/env python3
"""P5-S exact lazy-prefix and continuous adaptive preprocessing benchmark."""

from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import sys
from typing import Any,Callable
import h5py
import jax
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for value in (ROOT,ROOT/"scripts"):
    if str(value) not in sys.path:sys.path.insert(0,str(value))
import benchmark_heat3d_v6_p1i_p5_a4_p2r_r2p as a4  # noqa:E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa:E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa:E402
from rigno.heat3d_v6_full_field import build_reconstruction_map,prepare_reconstruction_domain_partition  # noqa:E402
from rigno.heat3d_v6_p1i_anchor_query import array_sha256,conservative_selected_control_volume,deterministic_nested_query_order,deterministic_nested_query_prefix,prepare_nested_query_geometry_cache  # noqa:E402

STAGES=("support_ordering","cv_redistribution","regional_prepare","coverage","p2r","r2r","r2p","packing","graph_total","reconstruction_map","continuous_total")

def _block(metadata:Any)->None:
    jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,"block_until_ready") else x,metadata)

def _stats(values:list[float])->dict[str,float]:
    a=np.asarray(values,dtype=np.float64);return {"median_seconds":float(np.median(a)),"mean_seconds":float(np.mean(a)),"p95_seconds":float(np.quantile(a,.95))}

def main()->int:
    p=argparse.ArgumentParser()
    for name in ("protocol","binding","artifact_root","dataset_root","manifest","full_fields","run_dir","output_json","output_csv","output_md"):p.add_argument("--"+name.replace("_","-"),dest=name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());binding=json.loads(a.binding.read_text());runtime=highn._checkpoint_runtime(a);anchors=highn._valid_examples(highn._dataset(a),binding);preflight=json.loads((a.artifact_root/"actual_data_preflight.json").read_text());supports={str(n):{r["sample_id"]:r for r in preflight["supports"][str(n)]} for n in (8192,32768)}
    with h5py.File(a.full_fields,"r") as archive:
        coords=np.asarray(archive["shared/coords_m"][:],dtype=np.float64);cv=np.asarray(archive["shared/control_volume_m3"][:],dtype=np.float64);layer=np.asarray(archive["shared/layer_id"][:],dtype=np.int32)
    boundaries=highn._boundaries(anchors[0],float(np.min(coords[:,2])));cache_started=time.perf_counter();geometry=prepare_nested_query_geometry_cache(full_coords=coords,full_control_volume=cv,full_layer_id=layer,layer_boundaries_m=boundaries);geometry_cache_seconds=time.perf_counter()-cache_started;reconstruction_partition=prepare_reconstruction_domain_partition(coords=coords,layer_id=layer,boundaries=boundaries)
    z=coords[:,2];internal=boundaries[1:-1];direct_interface=np.flatnonzero(np.any(np.isclose(z[:,None],internal[None,:],atol=1e-15),axis=1));direct_top=np.flatnonzero(np.isclose(z,np.max(z),atol=1e-15));direct_bottom=np.flatnonzero(np.isclose(z,np.min(z),atol=1e-15));geometry_exact=bool(np.array_equal(geometry.coords,coords) and np.array_equal(geometry.control_volume,cv) and np.array_equal(geometry.layer_id,layer) and np.array_equal(geometry.interface_indices,direct_interface) and np.array_equal(geometry.top_indices,direct_top) and np.array_equal(geometry.bottom_indices,direct_bottom) and all(np.array_equal(values,np.flatnonzero(layer==index)) for index,values in enumerate(geometry.layer_indices)))
    key=highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]));rows=[]
    for resolution,route,factor in ((8192,"B8192_adaptive",8.0),(32768,"E32768_adaptive",128.0)):
      for number,anchor in enumerate(anchors,start=1):
        frozen=highn._load_support(Path(supports[str(resolution)][anchor.sample_id]["support_file"]));anchors_idx=np.asarray(frozen["selected_indices"][:1024],dtype=np.int64)
        with np.load(a.artifact_root/"physics"/f"{anchor.sample_id}.npz") as physics:q=np.asarray(physics["q_W_m3"],dtype=np.float64)
        config=dict(runtime["graph_config"]);config.update(subsample_factor=factor,discrete_graph_backend="sparse_kdtree_v1",reuse_exact_p2r_for_r2p=True);warm_builder=Heat3DGraphBuilder(**config);warm_example=highn._query_example(anchor,frozen,coords);_block(warm_builder.build_metadata(highn.runner._graph_coords_for_example(warm_example,runtime["stats"]),key=key))
        def run(kind:str)->dict[str,Any]:
          continuous_started=time.perf_counter();order_started=time.perf_counter()
          if kind=="reference":full,audit=deterministic_nested_query_order(sample_id=anchor.sample_id,anchor_indices=anchors_idx,full_coords=coords,full_control_volume=cv,full_layer_id=layer,full_q=q,layer_boundaries_m=boundaries);selected=np.asarray(full[:resolution],dtype=np.int64);prefix_hash=array_sha256(selected)
          else:selected,audit=deterministic_nested_query_prefix(sample_id=anchor.sample_id,anchor_indices=anchors_idx,full_q=q,target_count=resolution,geometry_cache=geometry);prefix_hash=audit["prefix_sha256"]
          ordering_seconds=time.perf_counter()-order_started;cv_started=time.perf_counter();selected_cv,cv_audit=conservative_selected_control_volume(full_coords=coords,full_control_volume=cv,full_layer_id=layer,selected_indices=selected);cv_seconds=time.perf_counter()-cv_started
          support=dict(frozen);support["selected_indices"]=selected;support["operator_control_volume"]=selected_cv;example=highn._query_example(anchor,support,coords);builder=Heat3DGraphBuilder(**config);graph_started=time.perf_counter();metadata=builder.build_metadata(highn.runner._graph_coords_for_example(example,runtime["stats"]),key=key);_block(metadata);graph_seconds=time.perf_counter()-graph_started;map_started=time.perf_counter();mapping,_=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback="same_layer",prepared_partition=reconstruction_partition,query_workers=-1);map_seconds=time.perf_counter()-map_started;timing=builder.builder.last_build_timings
          return {"selected":selected,"prefix_hash":prefix_hash,"cv":selected_cv,"cv_hash":cv_audit["weights_sha256"],"metadata":metadata,"mapping":mapping,"stages":{"support_ordering":ordering_seconds,"cv_redistribution":cv_seconds,"regional_prepare":float(timing["regional_prepare_seconds"]),"coverage":float(timing["coverage_radius_seconds"]),"p2r":float(timing["p2r_seconds"]),"r2r":float(timing["r2r_seconds"]),"r2p":float(timing["r2p_seconds"]),"packing":float(timing["packing_seconds"]),"graph_total":graph_seconds,"reconstruction_map":map_seconds,"continuous_total":time.perf_counter()-continuous_started}}
        if number%2: candidate=run("candidate");reference=run("reference")
        else: reference=run("reference");candidate=run("candidate")
        gates={"prefix_array_equal_historical":bool(np.array_equal(reference["selected"],candidate["selected"]) and np.array_equal(reference["selected"],np.asarray(frozen["selected_indices"]))),"prefix_sha256_equal":reference["prefix_hash"]==candidate["prefix_hash"]==array_sha256(np.asarray(frozen["selected_indices"])),"anchor_prefix_exact":bool(np.array_equal(candidate["selected"][:1024],anchors_idx)),"geometry_partition_exact":geometry_exact,"cv_exact":bool(np.array_equal(reference["cv"],candidate["cv"]) and reference["cv_hash"]==candidate["cv_hash"]),"canonical_graph_hash_exact":a4._canonical_hash(reference["metadata"])==a4._canonical_hash(candidate["metadata"]),"reconstruction_map_exact":highn._mapping_sha256(reference["mapping"])==highn._mapping_sha256(candidate["mapping"])}
        rows.append({"route":route,"resolution":resolution,"sample_id":anchor.sample_id,"gates":gates,"reference":reference["stages"],"candidate":candidate["stages"]});print(f"[P5-S] {route} {number}/32 gates={gates}",flush=True)
        if not all(gates.values()):
            raise RuntimeError(f"P5-S fail-fast exact gate: {route}/{anchor.sample_id}: {gates}")
    hard=all(value for row in rows for value in row["gates"].values());summary={}
    for route in ("B8192_adaptive","E32768_adaptive"):
      selected=[r for r in rows if r["route"]==route];stages={}
      for stage in STAGES:
        ref=_stats([r["reference"][stage] for r in selected]);cand=_stats([r["candidate"][stage] for r in selected]);stages[stage]={"reference":ref,"candidate":cand,"median_speedup":ref["median_seconds"]/max(cand["median_seconds"],1e-30)}
      ranked=sorted(((stage,data["candidate"]["median_seconds"]) for stage,data in stages.items() if stage not in ("graph_total","continuous_total")),key=lambda item:item[1],reverse=True);summary[route]={"stages":stages,"remaining_bottleneck":ranked[0][0],"remaining_bottleneck_median_seconds":ranked[0][1],"ordering_is_secondary":ranked[0][0]!="support_ordering"}
    result={"schema_version":"heat3d_v6_p1i_p5s_closeout_v1","status":"passed" if hard else "failed","protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),"hard_gate_passed":hard,"geometry_cache_prepare_seconds":geometry_cache_seconds,"geometry_static_hashes":dict(geometry.static_hashes),"summary":summary,"samples":rows,"decision":{"lazy_prefix":"GO" if hard else "NO_GO","c_cpp":"NOT_IMPLEMENTED","approximate_q_cluster_cache":"NOT_IMPLEMENTED","further_micro_optimization":"STOP_this_phase","next":"proceed_to_U1" if hard else "STOP_before_U1"},"role_contract":protocol["role_contract"]};a.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with a.output_csv.open("w",newline="") as handle:
      w=csv.writer(handle);w.writerow(["route","stage","reference_median_s","candidate_median_s","speedup","candidate_p95_s"])
      for route,data in summary.items():
        for stage,s in data["stages"].items():w.writerow([route,stage,s["reference"]["median_seconds"],s["candidate"]["median_seconds"],s["median_speedup"],s["candidate"]["p95_seconds"]])
    lines=["# V6 P1i P5-S lazy-prefix closeout","",f"Status: **{'PASS' if hard else 'FAIL'}**; geometry cache preparation {geometry_cache_seconds:.6f} s.",""]
    for route,data in summary.items():
      lines += [f"## {route}","","| Stage | Full-order baseline median (s) | Lazy-prefix median (s) | Speedup |","|---|---:|---:|---:|"]
      for stage,s in data["stages"].items():lines.append(f"| {stage} | {s['reference']['median_seconds']:.6f} | {s['candidate']['median_seconds']:.6f} | {s['median_speedup']:.3f}x |")
      lines += ["",f"Remaining bottleneck: `{data['remaining_bottleneck']}` ({data['remaining_bottleneck_median_seconds']:.6f} s); ordering secondary={data['ordering_is_secondary']}.",""]
    lines += ["## Decision","","Every valid32 prefix is bitwise equal to the historical full-order prefix and frozen support. No C/C++ or approximate q-cluster cache was implemented. Further support-order micro-optimization stops in this phase; U1 may proceed only because all hard gates passed."]
    a.output_md.write_text("\n".join(lines)+"\n")
    if not hard:raise RuntimeError("P5-S hard gate failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
