#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
def req(x,m):
 if not x:raise RuntimeError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--direct',type=Path);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args();q=json.loads(a.protocol.read_text())
 req(q['status']=='preregistered_before_execution','protocol');req(q['historical_u2']['status']=='completed_no_go_unchanged','U2')
 req(q['u1_32768']['batch_sizes']==[1,4,8,16],'batch');req(q['direct_240825']['independent_preregistration'],'direct prereg')
 role=q['role_contract'];req(not role['training'] and not role['test'] and not role['sealed'],'roles');checked=[]
 if a.profile:
  d=json.loads(a.profile.read_text());req(d['status'] in {'passed_smoke','passed'},'profile');req(d['role_contract']==role,'profile role');req(d['checkpoint_parameters_unchanged'],'checkpoint');checked.append('profile')
 if a.direct:
  d=json.loads(a.direct.read_text());req(d['status'] in {'passed_smoke','passed'},'direct');req(d['resolution']==240825 and d['output_mode']=='direct','direct mode');req(d['role_contract']==role,'direct role');checked.append('direct')
 if a.closeout:
  d=json.loads(a.closeout.read_text());req(d['status']=='completed','closeout');req(d['role_contract']==role,'closeout role')
  if a.csv:req(len(list(csv.DictReader(a.csv.open())))==len(d['rows']),'CSV')
  checked.append('closeout')
 print(json.dumps({'u3_protocol_checked':True,'checked':checked}));return 0
if __name__=='__main__':raise SystemExit(main())
