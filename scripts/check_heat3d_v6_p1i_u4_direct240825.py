#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def req(v,m):
 if not v:raise RuntimeError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--identity',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--optimized',type=Path);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);p.add_argument('--sample-csv',type=Path);a=p.parse_args();q=json.loads(a.protocol.read_text());req(q['status']=='preregistered_before_execution','protocol');req(q['historical_u3']['status']=='completed_unchanged','U3');req(q['historical_u3']['plus_0p1_pp_gate'].startswith('withdrawn_for_U4'),'gate');req(not q['role_contract']['training'] and not q['role_contract']['test'] and not q['role_contract']['sealed'],'roles');checked=[]
 if a.identity:
  d=json.loads(a.identity.read_text());req(d['identity_hard_gate_passed'] and all(s['passed'] for s in d['samples']),'identity');checked.append('identity')
 if a.reference and a.optimized:
  r=json.loads(a.reference.read_text());o=json.loads(a.optimized.read_text());req(o['status']=='passed' and o['sample_count']==32,'optimized');req(o['packing_optimization']['prediction_bitwise_exact_vs_U3'],'prediction');req(o['checkpoint_parameters_unchanged'],'checkpoint');req(o['accuracy']['full_field']['domain']=='full_240825','domain');req(o['accuracy']['query_full_grid']['domain']=='query_full_grid_240825','label');checked.append('optimized')
 if a.closeout:
  c=json.loads(a.closeout.read_text());req(c['status']=='passed_valid32','closeout');req(c['decision'] in {'GO_architecture_freeze_candidate','NO_GO'},'decision');req(c['role_contract']==q['role_contract'],'role');req(c['same_240825_output_pareto']['u_direct_dominates_E240825'],'direct Pareto');req(set(c['prediction_artifacts'])==set(q['comparison_routes']),'prediction routes');req(c['packing']['prediction_bitwise_exact_vs_U3'],'packing exact');req(c['checkpoint_parameters_unchanged'],'checkpoint')
  for comparison in c['paired_valid32'].values():
   for metric in comparison.values():req(0.0<=metric['win_rate']<=1.0 and metric['ci95_low']<=metric['ci95_high'],'paired stats')
  for comparison in c['paired_valid32_preregistered_sse'].values():
   req(set(comparison)==set(q['paired_metrics']),'paired metric schema')
   for metric in comparison.values():req(0.0<=metric['win_rate']<=1.0 and metric['ci95_low']<=metric['ci95_high'],'paired SSE stats')
  if a.csv:req(len(list(csv.DictReader(a.csv.open())))==4,'CSV')
  if a.sample_csv:req(len(list(csv.DictReader(a.sample_csv.open())))==128,'sample CSV')
  checked.append('closeout')
 print(json.dumps({'u4_protocol_checked':True,'checked':checked}));return 0
if __name__=='__main__':raise SystemExit(main())
