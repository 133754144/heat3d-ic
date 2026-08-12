#!/usr/bin/env python3
"""Validate the U5 preregistration and optional runtime closeout."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def req(value, message):
    if not value: raise RuntimeError(message)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--result",type=Path); p.add_argument("--closeout",type=Path); p.add_argument("--csv",type=Path); a=p.parse_args(); q=json.loads(a.protocol.read_text())
    req(q["status"]=="preregistered_before_execution","protocol"); req(q["same_session"],"same session"); req(q["timing"]["historical_latency_reuse_for_pareto"] is False,"timing reuse"); req(q["lean_output_query"]["forbidden"].startswith("prepare_full"),"lean"); req(not q["role_contract"]["training"] and not q["role_contract"]["test"] and not q["role_contract"]["sealed"],"roles"); checked=[]
    if a.result:
        r=json.loads(a.result.read_text()); req(r["status"]=="passed","result"); req(r["same_session"],"session"); req(set(r["routes"])==set(q["routes"]),"routes"); req(r["lean_output_query"]["cpu_prediction_bitwise_exact"],"CPU exact"); checked.append("result")
    if a.closeout:
        c=json.loads(a.closeout.read_text()); req(c["status"]=="completed_frozen","closeout"); req(c["decision"] in {"U_direct240825_on_same_output_pareto","E240825_direct_on_same_output_pareto","no_strict_same_output_dominance"},"decision"); req(c["role_contract"]==q["role_contract"],"role");
        if a.csv: req(len(list(csv.DictReader(a.csv.open())))==3,"CSV")
        checked.append("closeout")
    print(json.dumps({"u5_checked":True,"checked":checked})); return 0

if __name__=="__main__": raise SystemExit(main())
