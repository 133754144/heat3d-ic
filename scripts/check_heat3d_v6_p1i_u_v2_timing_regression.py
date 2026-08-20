#!/usr/bin/env python3
"""Deterministic checker for the corrected U-v2 timing closeout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closeout", type=Path, required=True)
    parser.add_argument("--performance-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.closeout.read_text())
    errors: list[str] = []
    if data.get("status") != "passed_with_Q2_not_qualified":
        errors.append("closeout status drift")
    timing = data["timing_regression"]
    corrected = timing["corrected_U_v2"]
    if corrected.get("orders") != 3:
        errors.append("corrected serial order count is not three")
    if not (1.40 < corrected["stages"]["matched_continuous_e2e"]["median"] < 1.70):
        errors.append("corrected U-v2 fresh timing outside frozen audit envelope")
    if corrected["stages"]["anchor_graph"]["median"] >= 0.08:
        errors.append("anchor graph regression remains")
    if corrected["stages"]["query_graph"]["median"] >= 1.30:
        errors.append("query graph regression exceeds corrected repair envelope")
    if "old U-v2 3.229151 s fresh latency" not in timing["deprecated"]:
        errors.append("old U-v2 latency not deprecated")
    edge = data["edge_exact_optimization"]
    if not edge["golden_edge_exact_all_96"] or edge["speedup"] <= 1.0:
        errors.append("edge-exact optimization gate failed")
    if data["Q2"]["decision"] != "not_qualified_stop_after_second_order_residual_gate_failure":
        errors.append("Q2 fail-closed decision drift")
    rows = list(csv.DictReader(args.performance_csv.open()))
    if {row["strategy"] for row in rows} != {
        "E16384-reconstruction", "E240825-direct-control", "U-v2-direct240825", "FVM240825"
    }:
        errors.append("performance route set drift")
    by_name = {row["strategy"]: row for row in rows}
    if by_name["FVM240825"]["accuracy_role"] != "reference_solution":
        errors.append("FVM accuracy role is not reference_solution")
    if any(by_name["FVM240825"][key] for key in ("PG_pct", "raw_K", "source_K", "peak_K", "interface_K")):
        errors.append("FVM reference incorrectly has surrogate error values")
    if by_name["U-v2-direct240825"]["Q2_status"] != "not_qualified_one_pass_one_residual_gate_failure":
        errors.append("U-v2 Q2 status drift")
    for name in ("E16384-reconstruction", "E240825-direct-control"):
        if by_name[name]["Q2_status"] != "deprecated_serial_trace_not_concurrent":
            errors.append(f"{name} old serial Q2 not deprecated")
    attribution = data["repair_error_attribution"]
    if set(attribution) != {"seed0", "seed1", "seed2"}:
        errors.append("repair attribution seed set drift")
    for seed in attribution.values():
        if set(seed["categories"]) != {"covered", "repaired_inside", "repaired_outside"}:
            errors.append("repair attribution category drift")
    role = data["role_contract"]
    if any(role[key] for key in (
        "training", "test", "sealed", "checkpoint_modified", "dataset_modified",
        "E_architecture_modified", "accuracy_driven_optimization",
    )):
        errors.append("role contract violated")
    for artifact in data["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            errors.append(f"artifact mismatch: {path}")
    report = args.report.read_text()
    summary = args.summary.read_text()
    for required in ("3.229151", "deprecated", "1.015156", "未取得 publication qualification"):
        if required not in report:
            errors.append(f"report missing: {required}")
    for required in ("1.520111", "serial-trace Q2", "reference/—"):
        if required not in summary:
            errors.append(f"summary missing: {required}")
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "passed", "artifacts": len(data["artifacts"]),
                      "performance_rows": len(rows), "test_or_sealed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
