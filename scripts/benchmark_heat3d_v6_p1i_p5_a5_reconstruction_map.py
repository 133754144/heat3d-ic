#!/usr/bin/env python3
"""P5-A5 fixed-domain reconstruction-map exact timing gate."""

from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import sys
from typing import Any
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for value in (ROOT,ROOT/"scripts"):
    if str(value) not in sys.path: sys.path.insert(0,str(value))
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa:E402
from rigno.heat3d_v6_full_field import build_reconstruction_map,prepare_reconstruction_domain_partition  # noqa:E402

def _stats(values:list[float])->dict[str,float]:
    a=np.asarray(values,dtype=np.float64);return {"median_seconds":float(np.median(a)),"mean_seconds":float(np.mean(a)),"p95_seconds":float(np.quantile(a,.95))}

def main()->int:
    p=argparse.ArgumentParser()
    for name in ("protocol","binding","artifact_root","dataset_root","manifest","full_fields","output"):p.add_argument("--"+name.replace("_","-"),dest=name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());binding=json.loads(a.binding.read_text())
    dataset=highn._dataset(a);anchors=highn._valid_examples(dataset,binding);preflight=json.loads((a.artifact_root/"actual_data_preflight.json").read_text());supports={str(n):{r["sample_id"]:r for r in preflight["supports"][str(n)]} for n in (8192,32768)}
    with h5py.File(a.full_fields,"r") as archive:
        coords=np.asarray(archive["shared/coords_m"][:],dtype=np.float64);layer=np.asarray(archive["shared/layer_id"][:],dtype=np.int32)
    boundaries=highn._boundaries(anchors[0],float(np.min(coords[:,2])));partition_started=time.perf_counter();partition=prepare_reconstruction_domain_partition(coords=coords,layer_id=layer,boundaries=boundaries);partition_seconds=time.perf_counter()-partition_started
    rows=[]
    for resolution,route in ((8192,"B8192"),(32768,"E32768")):
      for number,anchor in enumerate(anchors,start=1):
        support=highn._load_support(Path(supports[str(resolution)][anchor.sample_id]["support_file"]));selected=np.asarray(support["selected_indices"],dtype=np.int32)
        started=time.perf_counter();reference,ref_audit=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback="same_layer",prepared_partition=None,query_workers=1);reference_seconds=time.perf_counter()-started
        started=time.perf_counter();candidate,cand_audit=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback="same_layer",prepared_partition=partition,query_workers=-1);candidate_seconds=time.perf_counter()-started
        row={"route":route,"resolution":resolution,"sample_id":anchor.sample_id,"neighbor_indices_equal":bool(np.array_equal(reference.neighbor_local_indices,candidate.neighbor_local_indices)),"neighbor_weights_array_equal":bool(np.array_equal(reference.neighbor_weights,candidate.neighbor_weights)),"domain_code_equal":bool(np.array_equal(reference.domain_code,candidate.domain_code)),"mapping_hash_equal":highn._mapping_sha256(reference)==highn._mapping_sha256(candidate),"partition_of_unity_equal":ref_audit["partition_of_unity_max_abs_error"]==cand_audit["partition_of_unity_max_abs_error"],"reference_seconds":reference_seconds,"candidate_seconds":candidate_seconds}
        rows.append(row);print(f"[P5-A5] {route} {number}/32 exact={row['mapping_hash_equal']}",flush=True)
    gates=("neighbor_indices_equal","neighbor_weights_array_equal","domain_code_equal","mapping_hash_equal","partition_of_unity_equal");hard=all(r[k] for r in rows for k in gates);summary={}
    for route in ("B8192","E32768","pooled"):
      selected=rows if route=="pooled" else [r for r in rows if r["route"]==route];ref=_stats([r["reference_seconds"] for r in selected]);cand=_stats([r["candidate_seconds"] for r in selected]);summary[route]={"reference":ref,"candidate":cand,"median_speedup":ref["median_seconds"]/cand["median_seconds"]}
    promoted=hard and summary["pooled"]["median_speedup"]>1
    result={"schema_version":"heat3d_v6_p1i_p5_a5_result_v1","status":"passed" if hard else "failed","phase":protocol["phase"],"protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),"hard_gate_passed":hard,"candidate_promoted":promoted,"decision":"GO_partition_cache_parallel_knn" if promoted else "NO_GO_keep_reference","partition_prepare_seconds":partition_seconds,"partition_domain_count":len(partition.domain_names),"summary":summary,"samples":rows,"role_contract":protocol["role_contract"]};a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if not hard:raise RuntimeError("P5-A5 exact gate failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
