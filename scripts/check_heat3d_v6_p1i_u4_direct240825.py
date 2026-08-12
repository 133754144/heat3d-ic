#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def req(v,m):
 if not v:raise RuntimeError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--identity',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--optimized',type=Path);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args();q=json.loads(a.protocol.read_text());req(q['status']=='preregistered_before_execution','protocol');req(q['historical_u3']['status']=='completed_unchanged','U3');req(q['historical_u3']['plus_0p1_pp_gate'].startswith('withdrawn_for_U4'),'gate');req(not q['role_contract']['training'] and not q['role_contract']['test'] and not q['role_contract']['sealed'],'roles');checked=[]
 if a.identity:
  d=json.loads(a.identity.read_text());req(d['identity_hard_gate_passed'] and all(s['passed'] for s in d['samples']),'identity');checked.append('identity')
 if a.reference and a.optimized:
  r=json.loads(a.reference.read_text());o=json.loads(a.optimized.read_text());req(o['status']=='passed' and o['sample_count']==32,'optimized');req(o['packing_optimization']['prediction_bitwise_exact_vs_U3'],'prediction');req(o['checkpoint_parameters_unchanged'],'checkpoint');req(o['accuracy']['full_field']['domain']=='full_240825','domain');req(o['accuracy']['query_full_grid']['domain']=='query_full_grid_240825','label');checked.append('optimized')
 if a.closeout:
  c=json.loads(a.closeout.read_text());req(c['status']=='completed','closeout');req(c['decision']['historical_u3_unchanged'],'historical');req(c['role_contract']==q['role_contract'],'role')
  if a.csv:req(len(list(csv.DictReader(a.csv.open())))==len(c['rows']),'CSV')
  checked.append('closeout')
 print(json.dumps({'u4_protocol_checked':True,'checked':checked}));return 0
if __name__=='__main__':raise SystemExit(main())
