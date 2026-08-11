#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--result",type=Path,required=True);a=p.parse_args();d=json.loads(a.result.read_text());assert d["status"]=="passed" and d["hard_gate_passed"] and len(d["samples"])==64
for r in d["samples"]: assert r["p2r_edge_array_hash_equal"] and r["r2p_edge_array_hash_equal"] and r["canonical_graph_hash_equal"]
assert not any(d["role_contract"][k] for k in ("training","test","sealed","temperature_or_prediction_used","graph_policy_modified","gpu_tiled_used"))
print(json.dumps({"status":"passed","batched":d["decision"]["batched"],"reverse_reuse":d["decision"]["reverse_reuse"],"speedup":d["summary"]["pooled"]["candidate_median_speedup"]},sort_keys=True))
