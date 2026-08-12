#!/usr/bin/env python3
"""Close out P7 fresh neural batch versus saturated parallel FVM."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--neural',type=Path,required=True);p.add_argument('--fvm',type=Path,required=True);p.add_argument('--p6',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-csv',type=Path,required=True);p.add_argument('--output-md',type=Path,required=True);a=p.parse_args()
 protocol,neural,fvm,p6=map(load,(a.protocol,a.neural,a.fvm,a.p6));sat=fvm['saturation'];rows=[]
 for r in neural['fresh_batch']:
  rows.append({'system':'E16384','semantic':'fresh_distinct_case_batch','batch_or_processes':r['batch_size'],'total_wall_s':r['total_wall_seconds'],'samples_per_s':r['samples_per_second'],'average_per_case_s':r['average_per_case_seconds'],'marginal_per_case_s':r['marginal_per_case_seconds'],'cpu_preprocess_s':r['cpu_preprocessing_wall_seconds'],'h2d_s':r['h2d_enqueue_seconds']+r['h2d_sync_seconds'],'forward_reconstruction_s':r['gpu_forward_and_reconstruction_seconds'],'peak_vram_bytes':r['peak_vram_bytes'],'provenance':'P7_new_valid32'})
 for r in fvm['rows']:
  rows.append({'system':'FVM','semantic':'parallel_independent_known_topology_new_physics','batch_or_processes':r['process_count'],'total_wall_s':r['total_wall_seconds'],'samples_per_s':r['samples_per_second'],'average_per_case_s':r['average_per_case_seconds'],'marginal_per_case_s':None,'cpu_preprocess_s':None,'h2d_s':None,'forward_reconstruction_s':None,'peak_vram_bytes':None,'provenance':'P7_new_valid32'})
 best=max(neural['fresh_batch'],key=lambda x:x['samples_per_second']);resident=next(x for x in p6['systems'] if x['route']['route']=='E16384')['batch']
 result={'schema_version':'heat3d_v6_p1i_p7_fresh_batch_closeout_v1','status':'completed','artifacts':{'neural':{'sha256':sha(a.neural)},'fvm':{'sha256':sha(a.fvm)}},'neural_best':best,'fvm_saturation':sat,'fresh_neural_speedup_vs_saturated_fvm':best['samples_per_second']/sat['samples_per_second'],'resident_reference':resident,'rows':rows,'decision':{'production_route':'E16384','fresh_batch_GO':True,'saturation_note':'E16384 fresh throughput remains CPU-preprocessing limited; B32 is highest tested and fastest','semantic_guard':'resident, streamed-prepared, fresh, and parallel-FVM remain separate'},'role_contract':protocol['role_contract']}
 a.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 with a.output_csv.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 lines=['# V6 P1i P7 fresh-batch closeout','','All runs use frozen valid32 on WSL2. Accuracy is unchanged and reused from the frozen E16384 route. JIT, metrics, hashes, and serialization are outside fresh spans.','','| system | batch/processes | wall (s) | samples/s | avg/case (s) |','|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['system']} ({r['semantic']}) | {r['batch_or_processes']} | {r['total_wall_s']:.6f} | {r['samples_per_s']:.3f} | {r['average_per_case_s']:.6f} |")
 lines += ['',f"Best fresh neural throughput: B{best['batch_size']} = {best['samples_per_second']:.3f} samples/s. Saturated FVM: {sat['process_count']} processes = {sat['samples_per_second']:.3f} samples/s. Semantically explicit throughput ratio = {result['fresh_neural_speedup_vs_saturated_fvm']:.2f}x.",'','Fresh neural throughput is dominated by CPU preprocessing, while resident throughput excludes that work. These figures are not interchangeable. No training or test/sealed access occurred.']
 a.output_md.write_text('\n'.join(lines)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())
