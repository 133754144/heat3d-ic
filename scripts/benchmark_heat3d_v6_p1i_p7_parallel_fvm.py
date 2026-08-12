#!/usr/bin/env python3
"""P7 same-host parallel independent-case FVM throughput benchmark."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def parse()->argparse.Namespace:
    p=argparse.ArgumentParser()
    for name in ('protocol','dataset_root','manifest','full_fields','run_dir','edge_targets','valid32_source','work_dir','output'):
        p.add_argument(f'--{name.replace("_","-")}',dest=name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--checkpoint-epoch',type=int,required=True)
    p.add_argument('--sample-count',type=int,choices=[2,32],required=True);p.add_argument('--process-counts',default='1,2,4,8');return p.parse_args()

def main()->int:
    args=parse();protocol=json.loads(args.protocol.read_text());source=json.loads(args.valid32_source.read_text())
    ids=list(source['sample_ids'])[:args.sample_count];counts=[int(x) for x in args.process_counts.split(',')]
    args.work_dir.mkdir(parents=True,exist_ok=True);env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    rows=[]
    for count in counts:
        cell_dir=args.work_dir/f'p{count}';cell_dir.mkdir(parents=True,exist_ok=True)
        def run(item:tuple[int,str])->dict[str,Any]:
            index,sample_id=item;output=cell_dir/f'{index:02d}_{sample_id}.json'
            command=[sys.executable,str(ROOT/'scripts/benchmark_heat3d_v6_inference_qualification.py'),'--worker','--family','p1i','--route','fvm','--state','known_topology_new_physics','--sample-count','32','--sample-id',sample_id,
                     '--dataset-root',str(args.dataset_root),'--manifest',str(args.manifest),'--full-fields',str(args.full_fields),'--run-dir',str(args.run_dir),'--checkpoint-sha256',args.checkpoint_sha256,'--checkpoint-epoch',str(args.checkpoint_epoch),'--edge-targets',str(args.edge_targets),'--output',str(output)]
            completed=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
            if completed.returncode:raise RuntimeError(f'{sample_id}: FVM worker failed: {completed.stderr[-2000:]}')
            payload=json.loads(output.read_text());measurement=payload['measurements'][0]
            return {'sample_id':sample_id,'continuous_wall_seconds':measurement['continuous_wall_seconds'],'assembly_seconds':measurement['assembly_seconds'],'linear_solve_seconds':measurement['linear_solve_seconds'],'process_internal_wall_seconds':payload['process_internal_wall_seconds'],'peak_ram_bytes':payload['process_peak_ram_bytes']}
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=count) as executor:measurements=list(executor.map(run,enumerate(ids)))
        wall=time.perf_counter()-started;rows.append({'process_count':count,'threads_per_process':1,'sample_count':len(ids),'status':'passed','total_wall_seconds':wall,'samples_per_second':len(ids)/wall,'average_per_case_seconds':wall/len(ids),'worker_continuous_seconds_sum':float(sum(x['continuous_wall_seconds'] for x in measurements)),'worker_continuous_seconds_median':float(np.median([x['continuous_wall_seconds'] for x in measurements])),'peak_ram_bytes_max_per_process':max(x['peak_ram_bytes'] for x in measurements),'sample_ids':ids,'measurements':measurements})
    result={'schema_version':'heat3d_v6_p1i_p7_parallel_fvm_v1','status':'passed','sample_count':len(ids),'process_counts':counts,'threads_per_process':1,'worker_startup_included':True,'process_environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'rows':rows,'saturation':max(rows,key=lambda x:x['samples_per_second']),'protocol_sha256':sha256(args.protocol),'role_contract':{'accessed_roles':['valid_iid'],'training':False,'test':False,'sealed':False,'checkpoint_modified':False,'dataset_modified':False,'graph_semantics_modified':False}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','best_processes':result['saturation']['process_count'],'samples_per_s':result['saturation']['samples_per_second']}));return 0
if __name__=='__main__':raise SystemExit(main())
