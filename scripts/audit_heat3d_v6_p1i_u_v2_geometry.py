#!/usr/bin/env python3
"""Label-free formal-valid geometry and R2P repair audit for U-v2."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import time
import numpy as np
ROOT=Path(os.environ.get("HEAT3D_REPO_ROOT",Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT,ROOT/"scripts"):
 if str(value) not in sys.path:sys.path.insert(0,str(value))
import run_heat3d_v6_p1i_anchor_high_n_development as highn
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r
import probe_heat3d_v6_p1i_u1_asymmetric_query as u1
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()
def parse():
 p=argparse.ArgumentParser()
 for n in ("protocol","binding","artifact_root","dataset_root","manifest","full_fields","run_dir","native_padding_result","query_padding_result","output"):p.add_argument(f"--{n.replace('_','-')}",dest=n,type=Path,required=True)
 p.add_argument("--checkpoint-sha256",required=True);p.add_argument("--population-preflight",type=Path,required=True);return p.parse_args()
def main():
 a=parse();q=json.loads(a.protocol.read_text());a.padding_result=a.query_padding_result;runtime=p5r._runtime(a);binding=json.loads(a.binding.read_text());dataset=highn._dataset(a)
 ordered=sorted(dataset.split_ids["valid_iid"],key=lambda x:hashlib.sha256(x.encode()).hexdigest());valid32=binding["development_subset"]["sample_ids"]
 if ordered[:32]!=valid32:raise RuntimeError("valid32 subset drift")
 index=dataset.sample_index_by_id();anchors=[dataset[index[x]] for x in ordered[32:]];pre=json.loads(a.population_preflight.read_text())
 if pre["sample_ids"]!=[x.sample_id for x in anchors]:raise RuntimeError("valid96 order drift")
 full,_=highn._full_shared(a);coords=np.asarray(full["coords"],dtype=np.float64);graph_key=highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]));cfg=dict(runtime["graph_config"]);cfg.update(subsample_factor=4.0,discrete_graph_backend="sparse_kdtree_v1",reuse_exact_p2r_for_r2p=True)
 rows=[]
 for i,anchor in enumerate(anchors,1):
  builder=Heat3DGraphBuilder(**cfg);anchor_coords=highn.runner._graph_coords_for_example(anchor,runtime["stats"]);native=builder.build_metadata(anchor_coords,key=graph_key)
  support={"selected_indices":np.arange(len(coords),dtype=np.int64),"operator_control_volume":np.asarray(full["cv"],dtype=np.float64),"k_xyz":np.zeros((len(coords),3),dtype=np.float64),"q_W_m3":np.zeros(len(coords),dtype=np.float64),"layer_id":np.asarray(full["layer"],dtype=np.int32)}
  query=highn._query_example(anchor,support,coords);query_coords=highn.runner._graph_coords_for_example(query,runtime["stats"])
  centers=np.asarray(native.x_rnodes)[0,:-1];base_radii=np.asarray(native.r_rnodes)[0,:-1];impl=builder.builder
  lower=np.asarray(anchor_coords,dtype=np.float64).min(axis=0);upper=np.asarray(anchor_coords,dtype=np.float64).max(axis=0)
  normalized=2.0*(np.asarray(query_coords,dtype=np.float64)-lower)/(upper-lower)-1.0
  raw=impl._get_supported_pnodes_by_rnodes(centers=centers,points=normalized,radii=impl._get_effective_support_radii(base_radii,impl.overlap_factor_r2p),apply_legacy_hard_reset=(impl.radius_policy=="legacy_kdtree_mean4"))
  phase=time.perf_counter();reference=impl._repair_physical_node_coverage(edge_indices=raw,centers=centers,points=normalized);reference_seconds=time.perf_counter()-phase
  phase=time.perf_counter();candidate=u1._repair_uncovered_physical_nodes_exact(edge_indices=raw,centers=centers,points=normalized,min_physical_coverage=impl.min_physical_coverage);candidate_seconds=time.perf_counter()-phase
  if not np.array_equal(np.asarray(reference),np.asarray(candidate)) or array_sha(reference)!=array_sha(candidate):raise RuntimeError(f"{anchor.sample_id}: repair edge drift")
  _,audit=u1._u_v2_asymmetric_metadata(builder,native,anchor_coords,query_coords,numerical_tolerance=float(q["u_v2"]["normalized_numerical_tolerance"]),maximum_normalized_overshoot=float(q["u_v2"]["maximum_normalized_overshoot"]))
  audit["repair_exact_gate"]={"edge_array_equal":True,"reference_sha256":array_sha(reference),"candidate_sha256":array_sha(candidate),"reference_seconds":reference_seconds,"candidate_seconds":candidate_seconds}
  audit["sample_id"]=anchor.sample_id;rows.append(audit);print(f"[U-v2 geometry] {i}/96",flush=True)
 result={"schema_version":"heat3d_v6_p1i_u_v2_geometry_audit_v2_exact_repair","status":"passed","sample_count":96,"temperature_labels_read":False,"sample_ids":[x.sample_id for x in anchors],"rows":rows,"summary":{"outside_sample_count":sum(r["outside_node_count"]>0 for r in rows),"outside_node_count_total":sum(r["outside_node_count"] for r in rows),"outside_node_ratio_mean":float(np.mean([r["outside_node_ratio"] for r in rows])),"maximum_normalized_overshoot":max(r["maximum_normalized_overshoot"] for r in rows),"raw_uncovered_count_total":sum(r["raw_uncovered_count"] for r in rows),"repaired_uncovered_count_total":sum(r["repaired_uncovered_count"] for r in rows),"native_exact_all":all(all(r["native_exact"].values()) for r in rows),"repair_edge_exact_all":all(r["repair_exact_gate"]["edge_array_equal"] and r["repair_exact_gate"]["reference_sha256"]==r["repair_exact_gate"]["candidate_sha256"] for r in rows),"reference_repair_median_seconds":float(np.median([r["repair_exact_gate"]["reference_seconds"] for r in rows])),"candidate_repair_median_seconds":float(np.median([r["repair_exact_gate"]["candidate_seconds"] for r in rows]))},"role_contract":q["role_contract"]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result["summary"]));return 0
if __name__=="__main__":raise SystemExit(main())
