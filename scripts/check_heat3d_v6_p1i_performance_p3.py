#!/usr/bin/env python3
"""Checker for isolated P3 and final performance table."""

import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CFG=ROOT/"configs/heat3d_v6_p1i"; DOC=ROOT/"docs"

def main():
    protocol=json.loads((CFG/"v6_p1i_performance_p3_protocol.json").read_text())
    result=json.loads((CFG/"v6_p1i_performance_p3_closeout.json").read_text())
    assert protocol["p2_commit"]=="28d31ee" and result["status"]=="completed"
    assert result["p1_sha256"]==hashlib.sha256((CFG/"v6_p1i_performance_p1_closeout.json").read_bytes()).hexdigest()
    assert result["p2_sha256"]==hashlib.sha256((CFG/"v6_p1i_performance_p2_closeout.json").read_bytes()).hexdigest()
    assert all(not result["role_contract"][k] for k in ("training","test","sealed","accuracy_recomputed","checkpoint_modified","dataset_modified","graph_policy_search"))
    assert len(result["cache_gates"])==4
    for gate in result["cache_gates"].values():
        assert gate["status"]=="passed" and gate["promote"] is True
        assert gate["graph_cache_exact"] and gate["group_tree_exact"] and gate["prediction_within_replay_envelope"]
        assert gate["static_cache_speedup"]>1.0
    assert result["persistent_jax_cache"]["cache_hit_compile_s"] < result["persistent_jax_cache"]["first_compile_s"]
    assert result["hybrid_gpu_tiled_continued"] is False and result["unseen_topology_gpu_builder_deferred"] is True
    with (DOC/"v6_p1i_performance_final.csv").open(newline="") as h: rows=list(csv.DictReader(h))
    assert [r["route"] for r in rows]==["B8192_recon","E32768_recon","B240825_direct","E240825_direct","FVM240825"]
    assert all(float(r["reconstruction_median_s"])>0 for r in rows[:2])
    assert all(float(r["reconstruction_median_s"])==0 for r in rows[2:4])
    for path,expected in result["source_hashes"].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    text=(DOC/"v6_p1i_performance_final.md").read_text(); assert "Process-cold" in text and "Known-support/new-physics" in text
    print(json.dumps({"status":"passed","rows":len(rows),"cache_routes":4}))
    return 0
if __name__=="__main__": raise SystemExit(main())
