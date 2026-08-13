#!/usr/bin/env python3
"""Measure true valid96 FVM B16-to-B32 persistent-pool marginal cost."""
from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--full-fields',type=Path,required=True);p.add_argument('--process-count',type=int,default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=json.loads(a.protocol.read_text());seeds=q['timing']['randomized_order_seeds'];base=np.arange(32,128);os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 serialized={'dataset_root':str(a.dataset_root),'manifest':str(a.manifest),'full_fields':str(a.full_fields),'selected_count':'128','prepare_indices':'32'};pool=ProcessPoolExecutor(max_workers=a.process_count,mp_context=mp.get_context('spawn'),initializer=p8.init_fvm_worker,initargs=(serialized,));ready={x for x in pool.map(p8.worker_ready,[.01]*(a.process_count*4))};
 if len(ready)!=a.process_count:raise RuntimeError('workers not ready')
 list(pool.map(p8.fvm_worker,base[:a.process_count]));rows=[]
 for seed in seeds:
  order=base[np.random.default_rng(seed).permutation(96)]
  walls={}
  for size in (16,32):
   started=time.perf_counter();list(pool.map(p8.fvm_worker,order[:size].tolist()));walls[str(size)]=time.perf_counter()-started
  rows.append({'order_seed':seed,'B16_wall_seconds':walls['16'],'B32_wall_seconds':walls['32'],'marginal_B16_to_B32_seconds':(walls['32']-walls['16'])/16.0})
 pool.shutdown();result={'schema_version':'heat3d_v6_p1i_valid96_fvm_batch_prefix_v1','status':'passed','process_count':a.process_count,'population':'remaining_valid96','batch_sizes':[16,32],'rows':rows,'role_contract':q['role_contract']};a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','median_marginal':float(np.median([x['marginal_B16_to_B32_seconds'] for x in rows]))}));return 0
if __name__=='__main__':raise SystemExit(main())
