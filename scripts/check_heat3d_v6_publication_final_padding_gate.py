#!/usr/bin/env python3
"""Validate monotonic publication padding and its persisted numerical gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROUTES = (
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "E240825_direct_control", "U_v2_direct240825",
)
EDGE_FIELDS = (
    "p2r_edge_indices", "r2r_edge_indices", "r2r_edge_domains", "r2p_edge_indices",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum(*sources: dict[str, int | None]) -> dict[str, int | None]:
    result = {}
    for field in EDGE_FIELDS:
        values = [int(row[field]) for row in sources if row.get(field) is not None]
        result[field] = max(values) if values else None
    return result


def check_capacity(payload: dict[str, Any], expected: dict[str, int | None], label: str) -> None:
    inputs = payload["capacity_inputs"]
    require(inputs["monotonic_envelope"] == expected, f"{label}: recorded envelope drift")
    recomputed = maximum(
        inputs["previous_same_semantic_frozen_capacity"],
        inputs["qualified_valid32_max"], inputs["train_only_warmup"],
    )
    require(recomputed == expected, f"{label}: monotonic max formula drift")
    for field in EDGE_FIELDS:
        old = inputs["previous_same_semantic_frozen_capacity"].get(field)
        new = expected.get(field)
        require(old is None or (new is not None and int(new) >= int(old)),
                f"{label}: capacity shrank for {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    qualification = json.loads(args.qualification.read_text())
    require(qualification["envelope_qualification"] == "GO", "frozen graph qualification drift")
    require(manifest["padding_semantics"] == "monotonic_dummy_capacity_only",
            "padding semantics drift")
    artifacts = {}
    for name, evidence in manifest["artifacts"].items():
        path = Path(evidence["path"])
        require(path.is_file() and sha(path) == evidence["sha256"], f"{name}: artifact SHA drift")
        artifacts[name] = json.loads(path.read_text())
        require(artifacts[name]["dummy_capacity_only"] is True, f"{name}: not dummy-only")
        require(artifacts[name]["real_graph_semantics_changed"] is False,
                f"{name}: real graph semantics changed")
    check_capacity(artifacts["native1024"],
                   artifacts["native1024"]["graph_cache"]["edge_targets"], "native1024")
    for key in ("E16384", "E240825"):
        check_capacity(artifacts[key], artifacts[key]["graph_cache"]["edge_targets"], key)
    for key in ("U16384", "U240825"):
        actual = artifacts[key]["padding"]["actual_padding_envelope"]
        check_capacity({"capacity_inputs": artifacts[key]["capacity_inputs"]["native"]},
                       actual["native"], f"{key}.native")
        check_capacity({"capacity_inputs": artifacts[key]["capacity_inputs"]["query"]},
                       actual["query"], f"{key}.query")

    gate = json.loads(args.gate_result.read_text())
    require(gate["status"] == "passed", "padding numerical gate did not pass")
    require(gate["padding_numerical_equivalence"] == "GO", "padding equivalence is not GO")
    require(gate["ready_for_authoritative_valid32"] == "GO", "authoritative run not ready")
    require(gate["routes"] == list(ROUTES), "route order drift")
    require(len(gate["route_results"]) == 4, "route result count drift")
    for route in gate["route_results"]:
        require(route["status"] == "passed" and route["checkpoint_unchanged"],
                f"{route['route']}: route gate failed")
        require([row["sample_id"] for row in route["records"]]
                == ["v6p1if1_0308", "v6p1if1_0029"], "sample contract drift")
        for row in route["records"]:
            require(row["real_graph_hashes_exact"] and row["passed"],
                    f"{route['route']}/{row['sample_id']}: exactness failed")
            tolerance = max(1.0e-3, 20.0 * float(row["same_shape_floor_K"]))
            require(abs(float(row["tolerance_K"]) - tolerance) <= 1.0e-15,
                    "numerical tolerance formula drift")
            for name in row["required_comparisons"]:
                require(float(row["comparisons"][name]["max_abs_K"]) <= tolerance,
                        f"{route['route']}/{row['sample_id']}/{name}: numerical gate failed")
    role = gate["role_contract"]
    require(not any(role[key] for key in ("training", "test", "sealed", "temperature_truth_read")),
            "forbidden role accessed")
    seal_checked = False
    if args.seal is not None:
        seal = json.loads(args.seal.read_text())
        require(seal["pre_measurement_seal"] == "GO", "seal GO drift")
        require(seal["ready_for_authoritative_valid32"] == "GO", "seal readiness drift")
        final_gate = seal["final_padding_gate"]
        require(final_gate["result_sha256"] == sha(args.gate_result), "seal gate SHA drift")
        require(final_gate["padding_manifest_sha256"] == sha(args.manifest),
                "seal padding manifest SHA drift")
        seal_checked = True
    print(json.dumps({
        "status": "passed", "envelope_qualification": "GO_frozen_reused",
        "padding_numerical_equivalence": "GO", "ready_for_authoritative_valid32": "GO",
        "seal_checked": seal_checked, "training": False, "test": False, "sealed": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
