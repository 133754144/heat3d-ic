#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--result",type=Path,required=True); a=p.parse_args(); d=json.loads(a.result.read_text())
assert d["status"]=="passed" and d["hard_gate_passed"] and d["timer_boundary_corrected"] and len(d["samples"])==64
for r in d["samples"]:
    assert r["nearest_assignment_equal"] and r["radii_array_equal"] and r["radii_sha256_equal"] and r["final_graph_hash_equal"]
assert not any(d["role_contract"][k] for k in ("training","test","sealed","temperature_or_prediction_used","graph_policy_modified"))
print(json.dumps({"status":"passed","candidate_promoted":d["candidate_promoted"],"pooled_median_speedup":d["summary"]["pooled"]["median_speedup"]},sort_keys=True))
