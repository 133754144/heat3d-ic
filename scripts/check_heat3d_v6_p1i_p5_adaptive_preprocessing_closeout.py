#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--json",type=Path,required=True);p.add_argument("--csv",type=Path,required=True);p.add_argument("--md",type=Path,required=True);a=p.parse_args();d=json.loads(a.json.read_text());assert d["status"]=="passed" and d["hard_gate_passed"] and len(d["samples"])==64
for row in d["samples"]:assert all(row["gates"].values())
assert not any(d["role_contract"][k] for k in ("training","test","sealed","inference","temperature_or_prediction_used","gpu_tiled_used","graph_policy_modified"))
rows=list(csv.DictReader(a.csv.open()));assert len(rows)==22 and set(r["route"] for r in rows)=={"B8192_adaptive","E32768_adaptive"};assert "P5 adaptive preprocessing closeout" in a.md.read_text()
print(json.dumps({"status":"passed","rows":len(rows),"routes":sorted(d["summary"])},sort_keys=True))
