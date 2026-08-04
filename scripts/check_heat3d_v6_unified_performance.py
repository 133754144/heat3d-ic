#!/usr/bin/env python3
"""Deterministic checker for the valid-only V6 unified performance closeout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RESOLUTIONS = {4096, 8192, 16384, 32768, 65536, 240825}
FAMILIES = {"p1i", "randomblock"}
STATES = {"process_cold", "new_topology", "known_topology_new_physics", "fully_cached"}


def finite_tree(value, path="root") -> None:
    if isinstance(value, dict):
        for key, item in value.items(): finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value): finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite numeric leaf: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--resolution-csv", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.closeout.read_text())
    if payload["status"] != "passed": raise RuntimeError("closeout is not passed")
    if any(len(payload[key]) != 40 for key in ("execution_commit", "direct_execution_commit", "closeout_commit_parent")):
        raise RuntimeError("commit binding missing")
    if payload["test_accessed"] or payload["sealed_accessed"]: raise RuntimeError("closed role accessed")
    if payload["training_executed"] or payload["tuning_executed"]: raise RuntimeError("training/tuning executed")
    contract = payload["formal_quality_contract"]
    if contract["sample_count"] != 128 or contract["seed_count"] != 3 or contract["timing_queue_replaces_formal_quality"]:
        raise RuntimeError("formal quality scope drift")
    if payload["timing_contract"]["sample_count"] != 32 or payload["timing_contract"]["batch_size"] != 1:
        raise RuntimeError("timing queue contract drift")
    if not payload["timing_contract"]["production_excludes_sha_metrics_oracle_json_checker"]:
        raise RuntimeError("production timing boundary drift")
    governance = payload["randomblock_governance"]
    if not governance["runtime_only_structured_support_OOD_diagnostic"] or governance["production_acceleration_claim_allowed"]:
        raise RuntimeError("random-block governance drift")
    direct_governance = payload["direct_model_governance"]
    if not direct_governance["diagnostic_only"] or direct_governance["production_speedup_claim_allowed"]:
        raise RuntimeError("direct-model diagnostic was promoted to production")
    if payload["first_resolution_speedup_over_1x_by_state"]["p1i"]["process_cold"] is not None:
        raise RuntimeError("cold-start speedup was overstated")
    if payload["first_resolution_speedup_over_1x_by_state"]["p1i"]["fully_cached"] != 4096:
        raise RuntimeError("P1i cached crossover drift")
    if payload["historical_values_directly_comparable"] or not payload["historical_layer_audit"]:
        raise RuntimeError("historical comparability audit missing or overstated")
    resolution_rows = payload["resolution_rows"]
    if {(row["family"], int(row["resolution"])) for row in resolution_rows} != {(family, node) for family in FAMILIES for node in RESOLUTIONS}:
        raise RuntimeError("resolution/family coverage incomplete")
    if {row["state"] for row in resolution_rows} != STATES:
        raise RuntimeError("four-state coverage incomplete")
    expected_routes = {"production_reconstruction", "fvm", "direct_model"}
    for family in FAMILIES:
        for node in RESOLUTIONS:
            routes = {row["route"] for row in resolution_rows if row["family"] == family and int(row["resolution"]) == node}
            if routes != expected_routes:
                raise RuntimeError(f"route coverage incomplete: {family} N={node} {routes}")
    for row in resolution_rows:
        if row["family"] == "randomblock" and int(row["resolution"]) in {4096, 8192, 16384, 32768} and row["route"] == "fvm":
            if row["status"] != "not_run_resolution_infeasible":
                raise RuntimeError("underresolved random-block FVM was not failed closed")
        if row["family"] == "randomblock" and int(row["resolution"]) in {4096, 8192, 16384, 32768} and row["route"] == "direct_model":
            expected_status = "not_applicable_fixed_structured_support" if row["state"] == "new_topology" else "not_run_resolution_infeasible"
            if row["status"] != expected_status:
                raise RuntimeError("underresolved random-block direct model was not failed closed")
        if row["family"] == "randomblock" and int(row["resolution"]) in {65536, 240825} and row["route"] == "fvm":
            expected_status = "not_applicable_under_frozen_contract" if row["state"] == "new_topology" else "passed"
            if row["status"] != expected_status:
                raise RuntimeError("qualified random-block FVM resolution did not run")
        if row.get("continuous_wall_median_s") is not None and float(row["continuous_wall_median_s"]) <= 0.0:
            raise RuntimeError("non-positive wall time")
    for path in (args.timing_csv, args.resolution_csv, args.accuracy_csv):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows: raise RuntimeError(f"empty CSV: {path}")
    for artifact in payload["input_artifacts"]:
        artifact_path = args.closeout.parents[2] / artifact["path"]
        if not artifact_path.is_file(): raise RuntimeError(f"missing frozen input: {artifact_path}")
        import hashlib
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise RuntimeError(f"frozen input hash drift: {artifact_path}")
    finite_tree(payload)
    print(json.dumps({"status": "passed", "timing_rows": len(payload["timing_rows"]), "resolution_rows": len(resolution_rows), "accuracy_rows": len(payload["accuracy_rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
