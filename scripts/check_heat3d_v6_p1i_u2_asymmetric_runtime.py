#!/usr/bin/env python3
"""Validate U2 protocol and optional cell."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def req(x,m):
    if not x: raise RuntimeError(m)
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--result',type=Path);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());req(protocol['status']=='preregistered_before_execution','protocol')
    req(protocol['u1_32768_accuracy_gate']['passed_before_runtime'],'accuracy gate');req(protocol['runtime_go_gate']['both_required'],'both gates')
    role=protocol['role_contract'];req(not role['training'] and not role['test'] and not role['sealed'],'roles');checked=False
    if a.result:
        d=json.loads(a.result.read_text());req(d['status'] in {'passed','passed_smoke'},'status');req(d['role_contract']==role,'role')
        req(d['checkpoint_parameters_unchanged'],'checkpoint');req(d['accuracy']['full_field']['point_global_true_rms_relative_rmse_pct']>0,'metric')
        req(all(r['shape']['output_nodes']==d['resolution'] for r in d['samples']),'shape');checked=True
    closeout_checked=False
    if a.closeout:
        d=json.loads(a.closeout.read_text());req(d['status']=='completed_no_go','closeout status')
        req(d['accuracy_gate']['passed'],'accuracy noninferiority');req(not d['runtime_gate']['passed'],'runtime gate')
        req(not d['runtime_gate']['b1_passed'] and d['runtime_gate']['resident_throughput_passed'],'gate attribution')
        req(d['decision']['u1_240825']=='not_executed_fail_fast_runtime_gate','240825 fail fast')
        req(d['role_contract']==role,'closeout role');req(not any('U1-240825'==r['system'] for r in d['rows']),'240825 row')
        if a.csv:
            rows=list(csv.DictReader(a.csv.open()));req(len(rows)==len(d['rows']),'CSV rows')
            req({'FVM','E16384','E32768','U1-32768'}=={r['system'] for r in rows},'CSV systems')
        closeout_checked=True
    print(json.dumps({'u2_protocol_checked':True,'result_checked':checked,'closeout_checked':closeout_checked}));return 0
if __name__=='__main__':raise SystemExit(main())
