#!/usr/bin/env python3
"""Check frozen U-v2 valid96 geometry, qualification, timing, and closeout."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def req(value,message):
    if not value: raise RuntimeError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--closeout',type=Path);p.add_argument('--csv',type=Path);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());geometry=json.loads(a.geometry.read_text());qualification=json.loads(a.qualification.read_text())
    req(protocol['baseline_commit'].startswith('42c0fca'),'baseline');req(protocol['population']['sample_count']==96 and protocol['population']['role']=='valid96_diagnostic_characterization','population')
    req(not protocol['role_contract']['training'] and not protocol['role_contract']['test'] and not protocol['role_contract']['sealed'],'roles')
    req(geometry['status']=='passed' and geometry['sample_count']==96,'geometry');req(geometry['temperature_labels_read'] is False,'geometry label access')
    req(geometry['summary']['native_exact_all'],'native graph');req(geometry['summary']['repaired_uncovered_count_total']==0,'repair coverage')
    req(qualification['status']=='passed' and qualification['sample_count']==96,'qualification');req(qualification['checkpoint_parameters_unchanged'],'checkpoint')
    req(all(row['asymmetric_graph_audit']['native_exact'] and not row['asymmetric_graph_audit']['native_graph_policy_or_radius_changed'] for row in qualification['samples']),'native exact')
    if a.closeout:
        result=json.loads(a.closeout.read_text());req(result['status']=='passed_final_freeze','closeout');req(result['population']=='valid96_diagnostic_characterization','role')
        req(result['decision']['production_reference']=='E16384-reconstruction','production');req(result['decision']['parallel_direct_strategy']=='U-v2-direct240825','U-v2')
        req(not result['decision']['test_or_sealed_opened'],'sealed');req(len(result['paired_statistics'])==2,'paired')
        for route in result['routes'].values():req(route['timing']['randomized_order_count']==3,'orders')
        if a.csv:req(len(list(csv.DictReader(a.csv.open())))==4,'CSV rows')
    print(json.dumps({'u_v2_valid96_checked':True,'closeout_checked':bool(a.closeout)}));return 0
if __name__=='__main__':raise SystemExit(main())
