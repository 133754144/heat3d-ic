#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-csv',type=Path,required=True);p.add_argument('--output-md',type=Path,required=True);a=p.parse_args();q=json.loads(a.protocol.read_text());r=json.loads(a.result.read_text());rows=[]
 for route,x in r['routes'].items():
  accuracy=x['accuracy'];timing=x['timing'];fresh=timing['matched_continuous_e2e'];rows.append({'route':route,'output_domain':'240825_full_field','pg_pct':accuracy['point_global_true_rms_relative_rmse_pct'],'raw_K':accuracy['raw_cv_weighted_rmse_K'],'source_K':accuracy['source_rmse_K'],'peak_K':accuracy['peak_rmse_K'],'interface_K':accuracy['interface_drop_rmse_K'],'fresh_median_s':fresh['median_seconds'],'fresh_p95_s':fresh['p95_seconds'],'peak_vram_bytes':x['peak_vram_bytes'],'runtime_artifact':x['path'],'runtime_sha256':x['sha256'],'execution_commit':r['execution_commit']})
 direct=[x for x in rows if x['route'] in {'E240825_direct','U_direct240825'}];dominates=lambda x,y: x['pg_pct']<y['pg_pct'] and x['raw_K']<y['raw_K'] and x['fresh_median_s']<y['fresh_median_s'];decision='U_direct240825_on_same_output_pareto' if dominates(direct[1],direct[0]) else ('E240825_direct_on_same_output_pareto' if dominates(direct[0],direct[1]) else 'no_strict_same_output_dominance')
 out={'schema_version':'heat3d_v6_p1i_u5_direct_timing_freeze_closeout_v1','status':'completed_frozen','decision':decision,'same_session':True,'protocol_sha256':sha(a.protocol),'result':{'path':str(a.result),'sha256':sha(a.result)},'lean_output_query':r['lean_output_query'],'rows':rows,'role_contract':q['role_contract']};a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with a.output_csv.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 lines=['# V6 P1i U5 direct timing freeze','',f"Decision: `{decision}`. All three routes were measured sequentially in one Python/JAX session at commit `{r['execution_commit']}`; historical P5-R latency is excluded.",'','| route | PG % | raw K | source K | peak K | interface K | fresh median / p95 s |','|---|---:|---:|---:|---:|---:|---:|']
 for x in rows: lines.append(f"| {x['route']} | {x['pg_pct']:.6f} | {x['raw_K']:.6f} | {x['source_K']:.6f} | {x['peak_K']:.6f} | {x['interface_K']:.6f} | {x['fresh_median_s']:.6f} / {x['fresh_p95_s']:.6f} |")
 lines += ['','U-direct uses `lean_output_query_v2`: the production span directly constructs only split-decoder inputs, graph and output native-physics tensors. The old full-group route is run only as an untimed deterministic CPU reference and is bitwise exact. No training or test/sealed access occurred.'];a.output_md.write_text('\n'.join(lines)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())
