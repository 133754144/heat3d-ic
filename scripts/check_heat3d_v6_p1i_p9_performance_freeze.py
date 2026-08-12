#!/usr/bin/env python3
"""Check preregistered or completed P9 artifacts."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def require(value, message):
    if not value: raise RuntimeError(message)

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",type=Path,required=True); parser.add_argument("--closeout",type=Path); parser.add_argument("--csv",type=Path); args=parser.parse_args(); protocol=json.loads(args.protocol.read_text())
    require(protocol["status"]=="preregistered_before_execution","protocol status"); require(protocol["baseline_commit"].startswith("5d8ee431"),"baseline"); require(protocol["neural"]["untimed_worker_warmup"],"neural warmup"); require(protocol["fvm"]["untimed_worker_warmup"],"FVM warmup"); require(len(protocol["randomized_order_seeds"])>=3,"orders"); require(not protocol["role_contract"]["training"] and not protocol["role_contract"]["test"] and not protocol["role_contract"]["sealed"],"roles")
    checked=[]
    if args.closeout:
        out=json.loads(args.closeout.read_text()); require(out["status"]=="completed_frozen","closeout"); require(out["preprocessing_exact"]["all_backends_exact"],"exact gate"); require(out["decision"].endswith("no_more_valid32_architecture_tuning"),"freeze"); require(len(out["neural"]["repeat_rows"])>=3,"neural repeats"); require(all(len(row["order"])==32 for row in out["neural"]["repeat_rows"]),"neural valid32"); require(all(len(row["repeats"])>=3 for row in out["fvm"]["rows"]),"FVM repeats"); require(out["role_contract"]==protocol["role_contract"],"role contract")
        if args.csv: require(len(list(csv.DictReader(args.csv.open())))==len(out["rows"]),"CSV")
        checked.append("closeout")
    print(json.dumps({"p9_checked":True,"checked":checked})); return 0

if __name__=="__main__": raise SystemExit(main())
