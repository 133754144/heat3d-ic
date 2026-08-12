#!/usr/bin/env python3
"""Validate P8 preregistration and optional frozen results."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def req(value,message):
    if not value:raise RuntimeError(message)

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--neural',type=Path);p.add_argument('--fvm',type=Path);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args();q=json.loads(a.protocol.read_text())
    req(q['status']=='preregistered_before_execution','protocol status')
    req(q['baseline_commit'].startswith('611f390'),'baseline')
    req(q['neural_preprocessing']['kdtree_workers_per_case']==1,'KDTree')
    req(q['fvm']['persistent_worker_pool'] and q['fvm']['threads_per_process']==1,'FVM')
    req(q['role_contract']['training'] is False and q['role_contract']['test'] is False and q['role_contract']['sealed'] is False,'roles')
    checked=[]
    if a.neural:
        n=json.loads(a.neural.read_text());req(n['status']=='passed','neural');req(n['sample_count']==32,'valid32');req(n['all_backends_exact'],'exact');req(n['role_contract']==q['role_contract'],'role');checked.append('neural')
    if a.fvm:
        f=json.loads(a.fvm.read_text());req(f['status']=='passed','fvm');req(f['sample_count']==32,'valid32');req(all(r['persistent_worker_pool'] for r in f['rows']),'persistent');req(all(r['threads_per_process']==1 for r in f['rows']),'threads');checked.append('fvm')
    if a.closeout:
        c=json.loads(a.closeout.read_text());req(c['status']=='completed','closeout');req(c['publication_safe']['semantic_match'],'semantics');req(c['role_contract']==q['role_contract'],'closeout role')
        if a.csv:req(len(list(csv.DictReader(a.csv.open())))==len(c['rows']),'CSV')
        checked.append('closeout')
    print(json.dumps({'p8_protocol_checked':True,'checked':checked}));return 0
if __name__=='__main__':raise SystemExit(main())
