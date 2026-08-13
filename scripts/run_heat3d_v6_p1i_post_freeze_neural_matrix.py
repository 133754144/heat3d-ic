#!/usr/bin/env python3
"""Execute frozen post-freeze neural routes without changing route semantics."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser()
 for name in ('protocol','p5r_protocol','u5_protocol','binding','artifact_root','dataset_root','manifest','full_fields','native_padding','e16384_padding','e240825_padding','u240825_padding','output_root'):
  p.add_argument(f'--{name.replace("_","-")}',dest=name,type=Path,required=True)
 p.add_argument('--population-mode',choices=['frozen_valid32','remaining_valid96'],required=True);p.add_argument('--population-preflight',type=Path);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--seed0-run-dir',type=Path,required=True);p.add_argument('--seeds');p.add_argument('--routes',default='E16384_reconstruction,E240825_direct,U_direct240825');a=p.parse_args();q=json.loads(a.protocol.read_text());count=32 if a.population_mode=='frozen_valid32' else 96;seeds=([0] if count==32 else q['confirmation']['seeds']) if a.seeds is None else [int(value) for value in a.seeds.split(',')];routes=a.routes.split(',');allowed={'E16384_reconstruction','E240825_direct','U_direct240825'}
 if not routes or not set(routes)<=allowed:raise RuntimeError('unregistered route subset')
 orders=q['performance']['randomized_order_seeds'] if count==32 else [None];run_dirs={0:a.seed0_run_dir,1:a.runs_root/'V6_07_V5best_P1i_seed1_reliable_B24',2:a.runs_root/'V6_08_V5best_P1i_seed2_reliable_B24'};state={'schema_version':'heat3d_v6_p1i_post_freeze_neural_matrix_v1','status':'running','population_mode':a.population_mode,'requested_seeds':seeds,'requested_routes':routes,'cells':[],'role_contract':q['role_contract']};a.output_root.mkdir(parents=True,exist_ok=True);write(a.output_root/'execution_state.json',state)
 for seed in seeds:
  checkpoint=q['checkpoint_sha256'][f'seed{seed}']
  for order_seed in orders:
   suffix=f'_order{order_seed}' if order_seed is not None else ''
   for route in [value for value in ('E16384_reconstruction','E240825_direct') if value in routes]:
    out=a.output_root/f'seed{seed}_{route}{suffix}.json';cmd=[sys.executable,str(ROOT/'scripts/run_heat3d_v6_p1i_p5r_resolution_cell.py'),'--protocol',str(a.p5r_protocol),'--binding',str(a.binding),'--artifact-root',str(a.artifact_root),'--dataset-root',str(a.dataset_root),'--manifest',str(a.manifest),'--full-fields',str(a.full_fields),'--run-dir',str(run_dirs[seed]),'--padding-result',str(a.e16384_padding if route.startswith('E16384') else a.e240825_padding),'--native-padding-result',str(a.native_padding),'--route',route,'--checkpoint-sha256',checkpoint,'--sample-count',str(count),'--population-mode',a.population_mode,'--output',str(out)]
    if a.population_preflight:cmd += ['--population-preflight',str(a.population_preflight)]
    if order_seed is not None:cmd += ['--order-seed',str(order_seed)]
    completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);log=out.with_suffix('.log');log.write_text(completed.stdout);cell={'seed':seed,'route':route,'order_seed':order_seed,'returncode':completed.returncode,'result':str(out),'log':str(log),'log_sha256':sha(log)};cell.update({'result_sha256':sha(out)} if out.is_file() else {});state['cells'].append(cell);write(a.output_root/'execution_state.json',state)
    if completed.returncode:state['status']='failed';write(a.output_root/'execution_state.json',state);return completed.returncode
   if 'U_direct240825' not in routes:continue
   out=a.output_root/f'seed{seed}_U_direct240825{suffix}.json';cmd=[sys.executable,str(ROOT/'scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py'),'--protocol',str(a.u5_protocol),'--binding',str(a.binding),'--artifact-root',str(a.artifact_root),'--dataset-root',str(a.dataset_root),'--manifest',str(a.manifest),'--full-fields',str(a.full_fields),'--run-dir',str(run_dirs[seed]),'--native-padding-result',str(a.native_padding),'--query-padding-result',str(a.u240825_padding),'--resolution','240825','--checkpoint-sha256',checkpoint,'--sample-count',str(count),'--repeats','20','--batch-sizes','1','--population-mode',a.population_mode,'--output',str(out)]
   if a.population_preflight:cmd += ['--population-preflight',str(a.population_preflight)]
   if order_seed is not None:cmd += ['--order-seed',str(order_seed)]
   completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);log=out.with_suffix('.log');log.write_text(completed.stdout);cell={'seed':seed,'route':'U_direct240825','order_seed':order_seed,'returncode':completed.returncode,'result':str(out),'log':str(log),'log_sha256':sha(log)};cell.update({'result_sha256':sha(out)} if out.is_file() else {});state['cells'].append(cell);write(a.output_root/'execution_state.json',state)
   if completed.returncode:state['status']='failed';write(a.output_root/'execution_state.json',state);return completed.returncode
 state['status']='passed';write(a.output_root/'execution_state.json',state);print(json.dumps({'status':'passed','cells':len(state['cells'])}));return 0
if __name__=='__main__':raise SystemExit(main())
