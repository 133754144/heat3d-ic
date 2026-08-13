#!/usr/bin/env python3
"""Validate post-freeze protocol and optional result bundle."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def req(x,m):
    if not x: raise RuntimeError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--performance',type=Path);p.add_argument('--confirmation',type=Path);p.add_argument('--performance-csv',type=Path);p.add_argument('--confirmation-csv',type=Path);a=p.parse_args();q=json.loads(a.protocol.read_text());req(q['status']=='preregistered_before_execution','protocol');req(q['baseline_commit'].startswith('ded92ad'),'baseline');req(q['freeze_scope']['no_valid32_route_graph_packing_or_model_optimization'],'freeze');req(q['performance']['output_nodes']==240825 and len(q['performance']['randomized_order_seeds'])>=3,'performance');req(q['confirmation']['selection_or_tuning'] is False,'confirmation');req(not q['role_contract']['training'] and not q['role_contract']['test'] and not q['role_contract']['sealed'],'roles');checked=[]
 if a.performance:
  r=json.loads(a.performance.read_text());req(r['status']=='passed','performance');req(r['output_domain_nodes']==240825,'performance contract');req(all(x['randomized_order_count']>=3 and x['exact_payload_across_orders'] for x in r['performance'].values()),'exact gates');req(r['role_contract']==q['role_contract'],'performance roles');
  if a.performance_csv:req(len(list(csv.DictReader(a.performance_csv.open())))>=4,'performance CSV');checked.append('performance')
 if a.confirmation:
  r=json.loads(a.confirmation.read_text());req(r['status']=='passed','confirmation');req(r['valid32_subset_formal_valid128'] and r['valid96_count']==96,'population');req(set(r['confirmation_valid96_three_seed_mean_std'])=={'E16384_reconstruction','U_direct240825','E240825_direct'},'routes');req(set(r['paired_bootstrap'])=={'seed0','seed1','seed2'},'seeds');req(r['role_contract']==q['role_contract'],'confirmation roles');
  if a.confirmation_csv:req(len(list(csv.DictReader(a.confirmation_csv.open())))>=9,'confirmation CSV');checked.append('confirmation')
 print(json.dumps({'post_freeze_checked':True,'checked':checked}));return 0
if __name__=='__main__':raise SystemExit(main())
