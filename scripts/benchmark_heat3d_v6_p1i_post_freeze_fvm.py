#!/usr/bin/env python3
"""Persistent FVM timing under the post-freeze four-layer contract."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8

def stats(values):
 x=np.asarray(values,dtype=np.float64);return {'count':len(values),'median_seconds':float(np.median(x)),'mean_seconds':float(np.mean(x)),'std_seconds':float(np.std(x)),'p95_seconds':float(np.quantile(x,.95))}

def valid96_indices(dataset_root,manifest,full_fields):
 data=p8.qualification.FamilyData(family='p1i',dataset_root=dataset_root,manifest_path=manifest,full_fields_path=full_fields,randomblock_config=None)
 ranked=sorted(data.valid_rows,key=lambda row:hashlib.sha256(str(row['sample_id']).encode()).hexdigest())
 return list(range(32,128)),[str(row['sample_id']) for row in ranked[32:]]
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--full-fields',type=Path,required=True);p.add_argument('--process-counts',default='1,2,4');p.add_argument('--population-mode',choices=['frozen_valid32','remaining_valid96'],default='frozen_valid32');p.add_argument('--queue-depth',type=int,default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=json.loads(a.protocol.read_text());counts=[int(value) for value in a.process_counts.split(',')];count=32 if a.population_mode=='frozen_valid32' else 96;base_indices=list(range(count)) if count==32 else list(range(32,128));sample_ids=None if count==32 else valid96_indices(a.dataset_root,a.manifest,a.full_fields)[1];seeds=q.get('timing',q.get('performance'))['randomized_order_seeds'];orders=[[base_indices[int(i)] for i in np.random.default_rng(seed).permutation(count)] for seed in seeds];rows=[]
 for count in counts:
  os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');resident_index=base_indices[0];serialized={'dataset_root':str(a.dataset_root),'manifest':str(a.manifest),'full_fields':str(a.full_fields),'selected_count':'128' if count==96 else '32','prepare_indices':str(resident_index)};started=time.perf_counter();pool=ProcessPoolExecutor(max_workers=count,mp_context=mp.get_context('spawn'),initializer=p8.init_fvm_worker,initargs=(serialized,));ready={f.result() for f in [pool.submit(p8.worker_ready,0.01) for _ in range(count*4)]};startup=time.perf_counter()-started
  if len(ready)!=count:raise RuntimeError('FVM workers not ready')
  list(pool.map(p8.fvm_solve_cached_worker,[resident_index]*count))
  repeats=[]
  for repeat,order in enumerate(orders):
   # Fresh/new physics includes data, assembly and solve.
   tick=time.perf_counter();fresh=list(pool.map(p8.fvm_worker,order));fresh_wall=time.perf_counter()-tick
   # Resident core is prepared-system solve-only and explicitly not E2E.
   tick=time.perf_counter();resident=list(pool.map(p8.fvm_solve_cached_worker,[resident_index]*len(order)));resident_wall=time.perf_counter()-tick
   # True stream submits distinct cases and records submit-to-result and completions.
   # Fixed-depth arrival protocol: initially submit Q, then submit one new
   # distinct case after each completion.  Qualification and result checking
   # remain outside the service span.
   tick=time.perf_counter();pending={};cursor=0;completion=[];submit_to=[];previous=tick
   while cursor<min(a.queue_depth,len(order)):
    index=order[cursor];submitted_at=time.perf_counter();pending[pool.submit(p8.fvm_worker,index)]=(index,submitted_at);cursor+=1
   while pending:
    future=next(as_completed(pending));index,submitted_at=pending.pop(future);future.result();now=time.perf_counter();submit_to.append(now-submitted_at);completion.append(now-previous);previous=now
    if cursor<len(order):
     next_index=order[cursor];new_submitted=time.perf_counter();pending[pool.submit(p8.fvm_worker,next_index)]=(next_index,new_submitted);cursor+=1
   stream_wall=time.perf_counter()-tick
   repeats.append({'repeat':repeat,'order_seed':seeds[repeat],'order':order,'fresh_wall_seconds':fresh_wall,'fresh_samples_per_second':count/fresh_wall,'fresh_single_case':stats([x['continuous_compute_seconds'] for x in fresh]),'resident_core_wall_seconds':resident_wall,'resident_core_solve_only':stats([x['solve_seconds'] for x in resident]),'streaming_wall_seconds':stream_wall,'streaming_samples_per_second':count/stream_wall,'stream_submit_to_result':stats(submit_to),'stream_inter_completion':stats(completion)})
  pool.shutdown();rows.append({'process_count':count,'startup_and_prepare_all_seconds':startup,'untimed_warmup':True,'repeats':repeats,'fresh_valid32_wall':stats([x['fresh_wall_seconds'] for x in repeats]),'resident_valid32_wall':stats([x['resident_core_wall_seconds'] for x in repeats]),'streaming_valid32_wall':stats([x['streaming_wall_seconds'] for x in repeats]),'fresh_throughput':stats([x['fresh_samples_per_second'] for x in repeats]),'streaming_throughput':stats([x['streaming_samples_per_second'] for x in repeats])})
 saturation=max(rows,key=lambda x:x['streaming_throughput']['median_seconds']);result={'schema_version':'heat3d_v6_p1i_post_freeze_fvm_v2','status':'passed','output_nodes':240825,'population_mode':a.population_mode,'sample_count':count,'sample_ids':sample_ids,'queue_depth':a.queue_depth,'rows':rows,'tested_process_counts':counts,'process_count_cap_reason':'persistent prepared-system workers are bounded by host RAM; cap is resource-feasibility, not performance selection','saturation_process_count':saturation['process_count'],'randomized_order_count':len(orders),'role_contract':q['role_contract']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','saturation':saturation['process_count'],'streaming_samples_s':saturation['streaming_throughput']['median_seconds']}));return 0
if __name__=='__main__':raise SystemExit(main())
