#!/usr/bin/env python3
"""Cross-check the V6 stage summary against frozen machine-readable evidence."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.loads((ROOT/name).read_text())
def main():
 p9=load('configs/heat3d_v6_p1i/v6_p1i_p9_performance_freeze_closeout.json');u5=load('configs/heat3d_v6_p1i/v6_p1i_u5_direct_timing_freeze_closeout.json');p5=load('configs/heat3d_v6_p1i/v6_p1i_p5r_resolution_sweep_closeout.json');three=load('configs/heat3d_v6_p1i/v6_p1i_three_seed_inference_closeout.json');text=(ROOT/'docs/v6_stage_summary_and_performance.md').read_text()
 assert p9['status']=='completed_frozen' and p9['preprocessing_exact']['all_backends_exact'];assert u5['status']=='completed_frozen' and u5['lean_output_query']['cpu_prediction_bitwise_exact'];assert p5['recommended_production_route']=='E16384_reconstruction';assert three['test_accessed'] is False and three['sealed_accessed'] is False
 for token in ('Fresh single-case latency','Warm/resident latency','Marginal added-case latency','Batch throughput','E16384-reconstruction','U-direct240825','停止 valid32 架构调优'): assert token in text
 assert p9['role_contract']['training'] is False and p9['role_contract']['test'] is False and p9['role_contract']['sealed'] is False; assert u5['role_contract']['training'] is False and u5['role_contract']['test'] is False and u5['role_contract']['sealed'] is False
 print(json.dumps({'v6_summary_checked':True,'architecture_frozen':True,'test_sealed_accessed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
