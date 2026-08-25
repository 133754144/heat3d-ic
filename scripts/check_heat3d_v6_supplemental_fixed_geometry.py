#!/usr/bin/env python3
"""Fail-closed checker for the V6 fixed-geometry runtime supplement."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_supplemental"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> int:
    protocol_path = CONFIG / "v6_fixed_geometry_runtime_protocol.json"
    closeout_path = CONFIG / "v6_fixed_geometry_runtime_closeout.json"
    csv_path = CONFIG / "v6_fixed_geometry_runtime_summary.csv"
    stage_csv_path = CONFIG / "v6_fixed_geometry_runtime_stage_decomposition.csv"
    hash_path = CONFIG / "v6_fixed_geometry_runtime_sha256.txt"
    protocol = json.loads(protocol_path.read_text())
    closeout = json.loads(closeout_path.read_text())

    if closeout.get("status") != "completed_passed":
        fail("closeout status")
    if closeout.get("protocol_sha256") != sha256(protocol_path):
        fail("protocol binding")
    if protocol.get("base_main_commit") != "6922e80c392385a8ae3d09b720c5307aaee1fffd":
        fail("base main commit drift")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["base_main_commit"], "HEAD"],
        cwd=ROOT,
    ).returncode:
        fail("base main is not an ancestor")
    if closeout.get("temperature_labels_opened") != 0:
        fail("temperature label access")
    if closeout.get("roles_accessed") != [
        "train_inputs", "shared_full_mesh_without_temperature"
    ]:
        fail("role access")
    if any(closeout["guardrails"].values()):
        fail("guardrail violation")

    routes = protocol["routes"]
    if [row["route"] for row in closeout["routes"]] != routes:
        fail("route order/completeness")
    for record in closeout["routes"]:
        path = ROOT / record["raw_path"]
        if sha256(path) != record["raw_sha256"]:
            fail(f"raw hash: {record['route']}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "passed" or len(payload.get("timing_rows", [])) != 288:
            fail(f"raw lifecycle: {record['route']}")
        if payload["correctness"].get("case_count") != 32:
            fail(f"correctness count: {record['route']}")
        if not all(row.get("passed") for row in payload["correctness"]["rows"]):
            fail(f"correctness result: {record['route']}")
        if not all(
            all(identity.values())
            for row in payload["correctness"]["rows"]
            for identity in row["static_identities"].values()
        ):
            fail(f"static identity: {record['route']}")
        if not payload.get("checkpoint_unchanged"):
            fail(f"checkpoint mutation: {record['route']}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (sweep, route, mode)
        for sweep in ("K_only", "K_plus_Q_scale")
        for route in routes
        for mode in ("fresh_new_case", "graph_only_reuse", "full_static_reuse")
    }
    observed = {(row["sweep"], row["route"], row["mode"]) for row in rows}
    if observed != expected or len(rows) != 24:
        fail("CSV matrix completeness")
    for row in rows:
        for key in ("median_s", "p95_s", "throughput_samples_per_second"):
            if float(row[key]) <= 0.0:
                fail(f"non-positive timing: {key}")
    with stage_csv_path.open(newline="", encoding="utf-8") as handle:
        stage_rows = list(csv.DictReader(handle))
    if not stage_rows:
        fail("empty stage decomposition")
    if {(row["sweep"], row["route"], row["mode"]) for row in stage_rows} != expected:
        fail("stage decomposition matrix completeness")
    if any(float(row["median_s"]) < 0.0 or float(row["p95_s"]) < 0.0 for row in stage_rows):
        fail("negative stage timing")

    manifest = {}
    for line in hash_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    for relative, digest in manifest.items():
        if sha256(ROOT / relative) != digest:
            fail(f"SHA manifest mismatch: {relative}")

    if protocol["scientific_immutability"] != {
        "training": False,
        "checkpoint_modified": False,
        "dataset_modified": False,
        "graph_policy_modified": False,
        "reconstruction_modified": False,
        "temperature_labels_read": False,
        "test_iid_accessed": False,
        "sealed_iid_accessed": False,
    }:
        fail("scientific immutability contract drift")
    print("PASS: V6 fixed-geometry supplemental runtime closeout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
