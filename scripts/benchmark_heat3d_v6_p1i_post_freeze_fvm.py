#!/usr/bin/env python3
"""Persistent FVM timing under the post-freeze four-layer contract."""
from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8

def stats(values):
 x=np.asarray(values,dtype=np.float64);return {'count':len(values),'median_seconds':float(np.median(x)),'mean_seconds':float(np.mean(x)),'std_seconds':float(np.std(x)),'p95_seconds':float(np.quantile(x,.95))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--full-fields',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=json.loads(a.protocol.read_text());counts=[1,2,4,8];orders=[np.random.default_rng(seed).permutation(32).tolist() for seed in q['performance']['randomized_order_seeds']];rows=[]
 for count in counts:
  os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');serialized={'dataset_root':str(a.dataset_root),'manifest':str(a.manifest),'full_fields':str(a.full_fields),'prepare_all':'true'};started=time.perf_counter();pool=ProcessPoolExecutor(max_workers=count,mp_context=mp.get_context('spawn'),initializer=p8.init_fvm_worker,initargs=(serialized,));ready={f.result() for f in [pool.submit(p8.worker_ready,0.01) for _ in range(count*4)]};startup=time.perf_counter()-started
  if len(ready)!=count:raise RuntimeError('FVM workers not ready')
  list(pool.map(p8.fvm_solve_cached_worker,[0]*count))
  repeats=[]
  for repeat,order in enumerate(orders):
   # Fresh/new physics includes data, assembly and solve.
   tick=time.perf_counter();fresh=list(pool.map(p8.fvm_worker,order));fresh_wall=time.perf_counter()-tick
   # Resident core is prepared-system solve-only and explicitly not E2E.
   tick=time.perf_counter();resident=list(pool.map(p8.fvm_solve_cached_worker,order));resident_wall=time.perf_counter()-tick
   # True stream submits distinct cases and records submit-to-result and completions.
   tick=time.perf_counter();submitted={pool.submit(p8.fvm_worker,index):(index,time.perf_counter()) for index in order};completion=[];submit_to=[];previous=tick
   for future in as_completed(submitted):
    now=time.perf_counter();index,submitted_at=submitted[future];future.result();submit_to.append(now-submitted_at);completion.append(now-previous);previous=now
   stream_wall=time.perf_counter()-tick
   repeats.append({'repeat':repeat,'order_seed':q['performance']['randomized_order_seeds'][repeat],'order':order,'fresh_wall_seconds':fresh_wall,'fresh_samples_per_second':32/fresh_wall,'fresh_single_case':stats([x['continuous_compute_seconds'] for x in fresh]),'resident_core_wall_seconds':resident_wall,'resident_core_solve_only':stats([x['solve_seconds'] for x in resident]),'streaming_wall_seconds':stream_wall,'streaming_samples_per_second':32/stream_wall,'stream_submit_to_result':stats(submit_to),'stream_inter_completion':stats(completion)})
  pool.shutdown();rows.append({'process_count':count,'startup_and_prepare_all_seconds':startup,'untimed_warmup':True,'repeats':repeats,'fresh_valid32_wall':stats([x['fresh_wall_seconds'] for x in repeats]),'resident_valid32_wall':stats([x['resident_core_wall_seconds'] for x in repeats]),'streaming_valid32_wall':stats([x['streaming_wall_seconds'] for x in repeats]),'fresh_throughput':stats([x['fresh_samples_per_second'] for x in repeats]),'streaming_throughput':stats([x['streaming_samples_per_second'] for x in repeats])})
 saturation=max(rows,key=lambda x:x['streaming_throughput']['median_seconds']);result={'schema_version':'heat3d_v6_p1i_post_freeze_fvm_v1','status':'passed','output_nodes':240825,'rows':rows,'saturation_process_count':saturation['process_count'],'randomized_order_count':len(orders),'role_contract':q['role_contract']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','saturation':saturation['process_count'],'streaming_samples_s':saturation['streaming_throughput']['median_seconds']}));return 0
if __name__=='__main__':raise SystemExit(main())
