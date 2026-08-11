#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
p = json.loads(args.result.read_text())
assert p["status"] == "passed" and p["hard_gate_passed"]
assert len(p["samples"]) == 64
for row in p["samples"]:
    for key in ("selected_cv_array_equal", "production_array_equal_candidate", "selected_cv_sha256_equal", "nearest_assignment_equal", "volume_sum_bitwise_equal", "relative_volume_error_not_increased"):
        assert row[key]
assert not p["role_contract"]["training"] and not p["role_contract"]["test"] and not p["role_contract"]["sealed"]
print(json.dumps({"status": "passed", "candidate_promoted": p["candidate_promoted"], "pooled_median_speedup": p["summary"]["pooled"]["median_speedup"]}, sort_keys=True))
