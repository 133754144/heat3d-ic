#!/usr/bin/env python3
"""Validate the frozen P7 fresh-batch protocol and optional closeout."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--neural',type=Path);p.add_argument('--fvm',type=Path)
    p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());require(protocol['status']=='preregistered_before_execution','protocol')
    require(protocol['route']['name']=='E16384' and protocol['route']['resolution']==16384,'route')
    require(protocol['fresh_batch_contract']['batch_sizes']==[1,2,4,8,16,32],'batch sizes')
    require(protocol['fvm_contract']['process_counts']==[1,2,4,8],'FVM processes')
    require(protocol['fvm_contract']['threads_per_process']==1,'FVM threads')
    role=protocol['role_contract'];require(not role['training'] and not role['test'] and not role['sealed'],'roles')
    checked=[]
    if a.neural:
        d=json.loads(a.neural.read_text());require(d['status'] in {'passed_smoke','passed'},'neural status')
        require(d['role_contract']==role and d['checkpoint_sha256']==protocol['checkpoint']['sha256'],'neural binding')
        for row in d['fresh_batch']:
            if row['status']=='passed':
                require(math.isfinite(row['samples_per_second']) and row['samples_per_second']>0,'neural finite')
                require(len(row['sample_ids'])==row['batch_size']==len(set(row['sample_ids'])),'distinct batch')
        checked.append('neural')
    if a.fvm:
        d=json.loads(a.fvm.read_text());require(d['status']=='passed','FVM status')
        require(d['process_counts']==[1,2,4,8] and d['threads_per_process']==1,'FVM contract')
        require(not d['role_contract']['test'] and not d['role_contract']['sealed'],'FVM roles');checked.append('fvm')
    if a.closeout:
        d=json.loads(a.closeout.read_text());require(d['status']=='completed','closeout')
        require(d['role_contract']==role and d['decision']['production_route']=='E16384','decision')
        if a.csv:
            rows=list(csv.DictReader(a.csv.open()));require(len(rows)==len(d['rows']),'CSV rows')
        checked.append('closeout')
    print(json.dumps({'p7_protocol_checked':True,'checked':checked}));return 0

if __name__=='__main__':raise SystemExit(main())
