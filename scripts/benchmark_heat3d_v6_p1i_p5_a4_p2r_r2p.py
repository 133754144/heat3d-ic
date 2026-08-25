#!/usr/bin/env python3
"""P5-A4 actual-data P2R/R2P batched-radius and reverse-reuse gate."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import sys
from typing import Any

import h5py
import jax
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
for value in (ROOT,ROOT/"scripts"):
    if str(value) not in sys.path: sys.path.insert(0,str(value))
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa:E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa:E402
from rigno.heat3d_graph_cache import graph_hash  # noqa:E402
from rigno.heat3d_v6_p1i_anchor_query import array_sha256  # noqa:E402
from rigno.models.rigno import RegionInteractionGraphBuilder  # noqa:E402


def _reference_edges(self: Any, centers: Any, points: Any, radii: Any, ord_distance: int=2, apply_legacy_hard_reset: bool=True) -> Any:
    if apply_legacy_hard_reset: radii=np.where(radii<.5,radii,.2)
    centers_array=np.asarray(centers); points_array=np.asarray(points); radii_array=np.asarray(radii)
    if self.radius_policy!="discrete_physical_coverage" or self.discrete_graph_backend!="sparse_kdtree_v1":
        return _candidate_edges(self,centers,points,radii,ord_distance,apply_legacy_hard_reset)
    tree=cKDTree(np.asarray(points_array,dtype=np.float64)); point_parts=[]; center_parts=[]
    for center_index,(center,radius) in enumerate(zip(centers_array,radii_array)):
        query_radius=float(radius)+max(1e-7,abs(float(radius))*1e-6)
        candidates=np.asarray(tree.query_ball_point(np.asarray(center,dtype=np.float64),query_radius),dtype=np.int64)
        if not len(candidates): continue
        candidates=candidates[np.linalg.norm(points_array[candidates]-center,axis=1)<=radius]
        point_parts.append(candidates); center_parts.append(np.full(len(candidates),center_index,dtype=np.int64))
    if point_parts:
        point_index=np.concatenate(point_parts); center_index=np.concatenate(center_parts); order=np.lexsort((center_index,point_index)); edges=np.column_stack((point_index[order],center_index[order]))
    else: edges=np.empty((0,2),dtype=np.int64)
    return jax.numpy.asarray(edges,dtype=jax.numpy.int32)


_candidate_edges=RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes


def _resolved_edges(metadata: Any, kind: str) -> np.ndarray:
    if kind=="p2r": return np.asarray(metadata.p2r_edge_indices)[0]
    if metadata.r2p_edge_indices is None: return np.flip(np.asarray(metadata.p2r_edge_indices)[0],axis=1)
    return np.asarray(metadata.r2p_edge_indices)[0]


def _canonical_hash(metadata: Any) -> str:
    digest=hashlib.sha256()
    for name,array in (("p2r",_resolved_edges(metadata,"p2r")),("r2p",_resolved_edges(metadata,"r2p")),("r2r",np.asarray(metadata.r2r_edge_indices)[0]),("domains",np.asarray(metadata.r2r_edge_domains)[0]),("radii",np.asarray(metadata.r_rnodes)[0])):
        digest.update(name.encode()); digest.update(array_sha256(np.asarray(array)).encode())
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str,float]:
    a=np.asarray(values,dtype=np.float64); return {"median_seconds":float(np.median(a)),"mean_seconds":float(np.mean(a)),"p95_seconds":float(np.quantile(a,.95))}


def main()->int:
    p=argparse.ArgumentParser()
    for name in ("protocol","binding","artifact_root","dataset_root","manifest","full_fields","run_dir","output"): p.add_argument("--"+name.replace("_","-"),dest=name,type=Path,required=True)
    a=p.parse_args(); protocol=json.loads(a.protocol.read_text()); binding=json.loads(a.binding.read_text())
    if binding.get("status")!="frozen_after_three_seed_r0_pass": raise RuntimeError("binding drift")
    runtime=highn._checkpoint_runtime(a); anchors=highn._valid_examples(highn._dataset(a),binding)
    preflight=json.loads((a.artifact_root/"actual_data_preflight.json").read_text()); supports={str(n):{r["sample_id"]:r for r in preflight["supports"][str(n)]} for n in (8192,32768)}
    with h5py.File(a.full_fields,"r") as archive: full_coords=np.asarray(archive["shared/coords_m"][:],dtype=np.float64)
    key=highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"])); rows=[]
    for resolution,route,factor in ((8192,"B8192",8.0),(32768,"E32768",128.0)):
      for number,anchor in enumerate(anchors,start=1):
        support=highn._load_support(Path(supports[str(resolution)][anchor.sample_id]["support_file"])); example=highn._query_example(anchor,support,full_coords); coords=highn.runner._graph_coords_for_example(example,runtime["stats"])
        config=dict(runtime["graph_config"]); config.update(subsample_factor=factor,discrete_graph_backend="sparse_kdtree_v1",reuse_exact_p2r_for_r2p=False)
        reference=Heat3DGraphBuilder(**config); RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes=_reference_edges
        try: ref_meta=reference.build_metadata(coords,key=key); jax.block_until_ready(ref_meta.r_rnodes)
        finally: RegionInteractionGraphBuilder._get_supported_pnodes_by_rnodes=_candidate_edges
        candidate=Heat3DGraphBuilder(**config); cand_meta=candidate.build_metadata(coords,key=key); jax.block_until_ready(cand_meta.r_rnodes)
        reuse_config=dict(config); reuse_config["reuse_exact_p2r_for_r2p"]=True; reuse=Heat3DGraphBuilder(**reuse_config); reuse_meta=reuse.build_metadata(coords,key=key); jax.block_until_ready(reuse_meta.r_rnodes)
        ref_p2r,ref_r2p=_resolved_edges(ref_meta,"p2r"),_resolved_edges(ref_meta,"r2p"); cand_p2r,cand_r2p=_resolved_edges(cand_meta,"p2r"),_resolved_edges(cand_meta,"r2p"); reuse_p2r,reuse_r2p=_resolved_edges(reuse_meta,"p2r"),_resolved_edges(reuse_meta,"r2p")
        row={"route":route,"resolution":resolution,"sample_id":anchor.sample_id,"p2r_edge_array_hash_equal":bool(np.array_equal(ref_p2r,cand_p2r) and array_sha256(ref_p2r)==array_sha256(cand_p2r)),"r2p_edge_array_hash_equal":bool(np.array_equal(ref_r2p,cand_r2p) and array_sha256(ref_r2p)==array_sha256(cand_r2p)),"canonical_graph_hash_equal":_canonical_hash(ref_meta)==_canonical_hash(cand_meta),"stored_graph_hash_equal":graph_hash(ref_meta)==graph_hash(cand_meta),"reuse_p2r_exact":bool(np.array_equal(ref_p2r,reuse_p2r) and array_sha256(ref_p2r)==array_sha256(reuse_p2r)),"reuse_r2p_exact":bool(np.array_equal(ref_r2p,reuse_r2p) and array_sha256(ref_r2p)==array_sha256(reuse_r2p)),"reuse_canonical_graph_hash_equal":_canonical_hash(ref_meta)==_canonical_hash(reuse_meta),"reuse_representation_is_implicit_reverse":reuse_meta.r2p_edge_indices is None,"reference_p2r_seconds":float(reference.builder.last_build_timings["p2r_seconds"]),"reference_r2p_seconds":float(reference.builder.last_build_timings["r2p_seconds"]),"candidate_p2r_seconds":float(candidate.builder.last_build_timings["p2r_seconds"]),"candidate_r2p_seconds":float(candidate.builder.last_build_timings["r2p_seconds"]),"reuse_p2r_seconds":float(reuse.builder.last_build_timings["p2r_seconds"]),"reuse_r2p_seconds":float(reuse.builder.last_build_timings["r2p_seconds"])}
        rows.append(row); print(f"[P5-A4] {route} {number}/32 batch={row['canonical_graph_hash_equal']} reuse={row['reuse_canonical_graph_hash_equal']}",flush=True)
    hard=all(r[k] for r in rows for k in ("p2r_edge_array_hash_equal","r2p_edge_array_hash_equal","canonical_graph_hash_equal")); reuse_exact=all(r[k] for r in rows for k in ("reuse_p2r_exact","reuse_r2p_exact","reuse_canonical_graph_hash_equal")); summary={}
    for route in ("B8192","E32768","pooled"):
      selected=rows if route=="pooled" else [r for r in rows if r["route"]==route]
      ref=_stats([r["reference_p2r_seconds"]+r["reference_r2p_seconds"] for r in selected]); cand=_stats([r["candidate_p2r_seconds"]+r["candidate_r2p_seconds"] for r in selected]); reuse_t=_stats([r["reuse_p2r_seconds"]+r["reuse_r2p_seconds"] for r in selected]); summary[route]={"reference":ref,"candidate":cand,"reuse":reuse_t,"candidate_median_speedup":ref["median_seconds"]/cand["median_seconds"],"reuse_median_speedup_vs_candidate":cand["median_seconds"]/reuse_t["median_seconds"]}
    batch_go=hard and summary["pooled"]["candidate_median_speedup"]>1; reuse_go=reuse_exact and summary["pooled"]["reuse_median_speedup_vs_candidate"]>1
    result={"schema_version":"heat3d_v6_p1i_p5_a4_result_v1","status":"passed" if hard else "failed","phase":protocol["phase"],"protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),"hard_gate_passed":hard,"batched_candidate_promoted":batch_go,"reverse_reuse_exact":reuse_exact,"reverse_reuse_promoted":reuse_go,"decision":{"batched":"GO" if batch_go else "NO_GO","reverse_reuse":"GO" if reuse_go else "NO_GO"},"summary":summary,"samples":rows,"role_contract":protocol["role_contract"]}; a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if not hard: raise RuntimeError("P5-A4 exact gate failed")
    return 0
if __name__=="__main__": raise SystemExit(main())
