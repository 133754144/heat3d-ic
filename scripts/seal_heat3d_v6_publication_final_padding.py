#!/usr/bin/env python3
"""Create the versioned publication seal after the final monotonic padding gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-seal", type=Path, required=True)
    parser.add_argument("--padding-manifest", type=Path, required=True)
    parser.add_argument("--padding-gate", type=Path, required=True)
    parser.add_argument("--padding-golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.previous_seal.read_text())
    manifest = json.loads(args.padding_manifest.read_text())
    gate = json.loads(args.padding_gate.read_text())
    golden = json.loads(args.padding_golden.read_text())
    if gate["status"] != "passed" or gate["padding_numerical_equivalence"] != "GO":
        raise RuntimeError("final padding numerical gate did not pass")
    if manifest["padding_semantics"] != "monotonic_dummy_capacity_only":
        raise RuntimeError("padding manifest is not monotonic")
    if golden["status"] != "passed" or golden["record_count"] != 12:
        raise RuntimeError("padding-adjusted golden is incomplete")
    artifacts = {
        key: json.loads((ROOT / row["path"]).read_text())
        for key, row in manifest["artifacts"].items()
    }
    for key, row in manifest["artifacts"].items():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"padding artifact SHA drift: {key}")
    native = artifacts["native1024"]["graph_cache"]["edge_targets"]
    route_padding = {
        "E16384_reconstruction": {"anchor": native,
                                   "query": artifacts["E16384"]["graph_cache"]["edge_targets"]},
        "E240825_direct_control": {"anchor": native,
                                    "query": artifacts["E240825"]["graph_cache"]["edge_targets"]},
        "U_v2_16384_reconstruction": artifacts["U16384"]["padding"]["actual_padding_envelope"],
        "U_v2_direct240825": artifacts["U240825"]["padding"]["actual_padding_envelope"],
    }
    runtime_state = json.loads(json.dumps(previous["runtime_state"]))
    for route, padding in route_padding.items():
        runtime_state[route]["padding_envelope"] = padding
        state = {
            "padding_envelope": runtime_state[route]["padding_envelope"],
            "tensor_envelope": runtime_state[route]["tensor_envelope"],
        }
        runtime_state[route]["state_sha256"] = canonical_sha(state)

    frozen_files = [
        "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_protocol.json",
        "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_v1_1.json",
        "scripts/benchmark_heat3d_v6_p1i_final_e_service.py",
        "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py",
        "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py",
        "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py",
        "scripts/check_heat3d_v6_publication_benchmark_pre_measurement.py",
        "scripts/manifest_heat3d_v6_publication_benchmark_artifacts.py",
        "scripts/freeze_heat3d_v6_publication_padding_envelopes.py",
        "scripts/run_heat3d_v6_publication_final_padding_gate.py",
        "scripts/check_heat3d_v6_publication_final_padding_gate.py",
        "scripts/build_heat3d_v6_publication_padding_golden.py",
    ]
    result = json.loads(json.dumps(previous))
    result.update({
        "schema_version": "heat3d_v6_publication_benchmark_pre_measurement_seal_v2_final_padding",
        "status": "passed", "pre_measurement_seal": "GO",
        "ready_for_authoritative_valid32": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "runtime_state": runtime_state,
        "historical_golden": {
            "source_kind": "padding_adjusted_prepared_payload_golden_bound_to_historical_real_graphs",
            "source_path": str(args.padding_golden), "source_sha256": sha(args.padding_golden),
            "record_count": len(golden["records"]), "records": golden["records"],
            "candidate_reference_direct_SHA_equal": True,
            "current_implementation_self_replay_is_only_reference": False,
            "real_graph_hashes_inherited_from_previous_historical_golden": True,
            "previous_historical_source_commit": previous["historical_golden"]["source_commit"],
            "previous_historical_source_path": previous["historical_golden"]["source_path"],
            "previous_historical_source_sha256": previous["historical_golden"]["source_sha256"],
        },
        "final_padding_gate": {
            "status": "passed", "envelope_qualification": "GO_frozen_reused_not_rerun",
            "padding_numerical_equivalence": "GO",
            "padding_manifest_path": str(args.padding_manifest),
            "padding_manifest_sha256": sha(args.padding_manifest),
            "result_path": str(args.padding_gate), "result_sha256": sha(args.padding_gate),
            "padding_golden_path": str(args.padding_golden),
            "padding_golden_sha256": sha(args.padding_golden),
            "numeric_gate": "max_abs_K<=max(1e-3,20*same_shape_floor_K)",
            "sample_ids": ["v6p1if1_0308", "v6p1if1_0029"],
            "neural_routes": [
                "E16384_reconstruction", "U_v2_16384_reconstruction",
                "E240825_direct_control", "U_v2_direct240825",
            ],
        },
        "source_previous_seal": {
            "path": str(args.previous_seal), "sha256": sha(args.previous_seal),
            "retained_unchanged": True,
        },
        "frozen_implementation_sha256": {
            relative: sha(ROOT / relative) for relative in frozen_files
        },
        "new_benchmark_execution": {
            "case_count": 8, "route_process_count": 4,
            "reason": "final_padding_prediction_equivalence_only",
            "full_valid32_timing": False, "latency_or_speedup_generated": False,
        },
    })
    result["role_contract"].update({
        "training": False, "test": False, "sealed": False, "accuracy_tuning": False,
        "full_valid32_timing_executed_in_this_seal": False,
        "formal_latency_or_speedup_generated_in_this_seal": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "passed", "pre_measurement_seal": "GO",
        "ready_for_authoritative_valid32": "GO", "records": len(golden["records"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
