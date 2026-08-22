#!/usr/bin/env python3
"""Versioned final runtime-isolation amendment for Attempt 4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_protocol.json",
    "configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_v1_1.json",
    "scripts/benchmark_heat3d_v6_p1i_final_e_service.py",
    "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py",
    "scripts/check_heat3d_v6_publication_benchmark_pre_measurement.py",
    "scripts/check_heat3d_v6_publication_lifecycle_schema.py",
    "scripts/check_heat3d_v6_publication_runtime_isolation.py",
    "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py",
    "scripts/heat3d_v6_publication_lifecycle_schema.py",
    "scripts/heat3d_v6_publication_runtime_isolation.py",
    "scripts/manifest_heat3d_v6_publication_benchmark_artifacts.py",
    "scripts/seal_heat3d_v6_publication_runtime_isolation.py",
    "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-seal", type=Path, required=True)
    parser.add_argument("--runtime-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prior_seal = args.prior_seal.resolve()
    args.runtime_gate = args.runtime_gate.resolve()
    args.output = args.output.resolve()
    prior = json.loads(args.prior_seal.read_text())
    gate = json.loads(args.runtime_gate.read_text())
    if gate.get("benchmark_runtime_isolation") != "GO":
        raise RuntimeError("runtime isolation gate did not pass")
    result = dict(prior)
    result.update({
        "schema_version": "heat3d_v6_publication_pre_measurement_seal_v4_runtime_isolation",
        "status": "passed",
        "benchmark_runtime_isolation": "GO",
        "ready_for_authoritative_valid32": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "prior_versioned_seal": {
            "path": str(args.prior_seal.relative_to(ROOT)),
            "sha256": sha(args.prior_seal),
        },
        "runtime_isolation_regression": {
            "artifact_path": str(args.runtime_gate.relative_to(ROOT)),
            "artifact_sha256": sha(args.runtime_gate),
            "audit_outside_production_timing": True,
            "audit_outside_Q2_completion_refill": True,
            "service_HWM_captured_before_untimed_audit": True,
            "inner_failure_artifact_verified": True,
            "gpu_execution": False,
        },
        "attempt3_semantics": {
            "formal_measurement_attempted": True,
            "formal_matrix_completed": False,
            "publication_results_generated": False,
            "historical_artifacts_unchanged": True,
        },
        "frozen_implementation_sha256": {
            relative: sha(ROOT / relative) for relative in FROZEN
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "benchmark_runtime_isolation": "GO",
                      "frozen_files": len(FROZEN)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
