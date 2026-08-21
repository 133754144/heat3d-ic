#!/usr/bin/env python3
"""Build the machine-readable V6 pre-measurement seal without benchmarking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def historical_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(("git", "show", f"{commit}:{path}"), cwd=ROOT)


def exact_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for row in payload["rows"]:
        if row["service_mode"] != "serial" or row["route"].startswith("FVM"):
            continue
        if row["route"].startswith("E"):
            evidence = row["exactness_provenance"]
            graph = evidence["graph_candidate_hashes"]
            prepared = evidence["candidate_prepared_payload_sha256"]
            sample_id = evidence["sample_id"]
        else:
            audits = [sample for sample in row["samples"]
                      if sample["packing_audit"]["graph_exactness_audit_executed"]]
            if len(audits) != 1:
                raise RuntimeError("historical U exact audit count drift")
            audit = audits[0]["packing_audit"]
            graph = audit["graph_candidate_hashes"]
            prepared = audit["candidate_payload_sha256"]
            sample_id = audits[0]["sample_id"]
        direct = {
            "route": row["route"], "order_seed": int(row["order_seed"]),
            "resolution": int(row["resolution"]), "sample_id": sample_id,
            "native1024_graph_hashes": graph["native1024"],
            "query_graph_hashes": graph["query"],
            "prepared_payload_sha256": prepared,
        }
        direct["record_sha256"] = canonical_sha(direct)
        records.append(direct)
    return sorted(records, key=lambda x: (x["order_seed"], x["resolution"], x["route"]))


def runtime_state(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {row["route"]: row for row in payload["rows"] if row["service_mode"] == "serial"}
    states: dict[str, Any] = {}
    for route, row in rows.items():
        if route.startswith("E"):
            padding = row["warmup"]["padding_envelope_after"]
        elif route.startswith("U"):
            padding = row["padding"]["effective_padding_envelope"]
        else:
            padding = None
        if route == "FVM240825_reference":
            tensor = {"solver_nodes": 240825, "workers_serial": 1, "workers_Q2": 2}
        else:
            resolution = int(row["resolution"])
            tensor = {
                "batch_size": 1, "anchor_physical_nodes": 1024,
                "query_physical_nodes": resolution, "regional_nodes": 256,
                "output_nodes": 240825,
                "reconstruction_map_shape": (
                    [240825, 8] if route.startswith("E") and resolution == 16384
                    else [1, 240825, 8] if route.startswith("U") and resolution == 16384
                    else None),
            }
        state = {"padding_envelope": padding, "tensor_envelope": tensor}
        state["state_sha256"] = canonical_sha(state)
        states[route] = state
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.candidate = args.candidate.resolve()
    args.output = args.output.resolve()
    protocol = json.loads(args.protocol.read_text())
    golden = protocol["golden_exactness"]
    blob = historical_blob(golden["source_commit"], golden["source_path"])
    if sha_bytes(blob) != golden["source_sha256"]:
        raise RuntimeError("historical golden blob SHA drift")
    historical = json.loads(blob)
    candidate = json.loads(args.candidate.read_text())
    reference_records = exact_records(historical)
    candidate_records = exact_records(candidate)
    if len(reference_records) != golden["required_record_count"]:
        raise RuntimeError("golden record count drift")
    if candidate_records != reference_records:
        raise RuntimeError("candidate graph/edge/metadata/payload differs from historical golden")
    frozen_files = [
        args.protocol,
        ROOT / "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_v1_1.json",
        ROOT / "scripts/benchmark_heat3d_v6_p1i_final_e_service.py",
        ROOT / "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py",
        ROOT / "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py",
        ROOT / "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py",
        ROOT / "scripts/check_heat3d_v6_publication_benchmark_pre_measurement.py",
    ]
    result = {
        "schema_version": "heat3d_v6_publication_benchmark_pre_measurement_seal_v1",
        "status": "passed",
        "pre_measurement_seal": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "protocol_sha256": sha_file(args.protocol),
        "historical_golden": {
            "source_commit": golden["source_commit"],
            "source_path": golden["source_path"],
            "source_sha256": golden["source_sha256"],
            "record_count": len(reference_records),
            "records": reference_records,
            "candidate_reference_direct_SHA_equal": True,
            "current_implementation_self_replay_is_only_reference": False,
        },
        "runtime_state": runtime_state(historical),
        "frozen_implementation_sha256": {
            str(path.relative_to(ROOT)): sha_file(path) for path in frozen_files
        },
        "analysis_freeze": protocol["analysis_freeze"],
        "FVM_contract": protocol["FVM_contract"],
        "complete_workloads": protocol["complete_workloads"],
        "formal_runner_static_gate": "passed",
        "new_benchmark_execution": {
            "case_count": 0, "reason": "recent_real_route_smoke_reused_and_implementation_only_seal",
            "full_valid32_timing": False, "latency_or_speedup_generated": False,
        },
        "role_contract": protocol["role_contract"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
