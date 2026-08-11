#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CFG=ROOT/"configs/heat3d_v6_p1i"; DOC=ROOT/"docs"
def main():
    result=json.loads((CFG/"v6_p1i_p4a_fixed_support_accuracy_closeout.json").read_text())
    assert result["status"]=="completed_fail_fast" and result["p4b_allowed"] is False
    assert result["production_routes"]==[] and all(not row["production_go"] for row in result["summaries"])
    assert result["subsequent_stages"]=={"P4-B":"not_executed","P4-C":"not_executed","P4-D":"not_executed","P4-E":"not_executed"}
    assert not any(result["role_contract"][k] for k in ("training","test","sealed","checkpoint_modified","dataset_modified","graph_policy_search","adaptive_accuracy_reexecuted"))
    for path,expected in result["sources"].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    with (DOC/"v6_p1i_p4a_fixed_support_accuracy.csv").open(newline="") as h: assert len(list(csv.DictReader(h)))==2
    with (DOC/"v6_p1i_p4a_fixed_support_paired.csv").open(newline="") as h: assert len(list(csv.DictReader(h)))==64
    text=(DOC/"v6_p1i_p4_closeout.md").read_text(); assert "不执行 P4-B/C/D/E" in text and "NO-GO" in text
    print(json.dumps({"status":"passed","fail_fast":True,"paired_rows":64})); return 0
if __name__=="__main__": raise SystemExit(main())
