#!/usr/bin/env python3
"""Create the versioned lifecycle-schema amendment to the final-padding seal."""
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
    "scripts/build_heat3d_v6_publication_padding_golden.py",
    "scripts/check_heat3d_v6_publication_benchmark_pre_measurement.py",
    "scripts/check_heat3d_v6_publication_final_padding_gate.py",
    "scripts/check_heat3d_v6_publication_lifecycle_schema.py",
    "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py",
    "scripts/freeze_heat3d_v6_publication_padding_envelopes.py",
    "scripts/heat3d_v6_publication_lifecycle_schema.py",
    "scripts/manifest_heat3d_v6_publication_benchmark_artifacts.py",
    "scripts/run_heat3d_v6_publication_final_padding_gate.py",
    "scripts/seal_heat3d_v6_publication_lifecycle_schema.py",
    "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-seal", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prior_seal = args.prior_seal.resolve()
    args.fixture = args.fixture.resolve()
    args.output = args.output.resolve()
    prior = json.loads(args.prior_seal.read_text())
    fixture = json.loads(args.fixture.read_text())
    if fixture.get("status") != "passed" or fixture.get("fixture_count") != 10:
        raise RuntimeError("lifecycle fixture gate did not pass")
    result = dict(prior)
    result.update({
        "schema_version": "heat3d_v6_publication_benchmark_pre_measurement_seal_v3_lifecycle_schema",
        "status": "passed",
        "benchmark_lifecycle_schema": "GO",
        "ready_for_authoritative_valid32": "GO",
        "publication_timing_freeze": "NO_GO_ready_for_full_valid32",
        "prior_versioned_seal": {
            "path": str(args.prior_seal.relative_to(ROOT)),
            "sha256": sha(args.prior_seal),
        },
        "lifecycle_schema_regression": {
            "artifact_path": str(args.fixture.relative_to(ROOT)),
            "artifact_sha256": sha(args.fixture),
            "fixture_count": 10,
            "collector_parsed_all_10": True,
            "gpu_smoke_executed": False,
        },
        "measurement_provenance_initial_state": {
            "formal_measurement_attempted": False,
            "formal_matrix_completed": False,
            "publication_results_generated": False,
        },
        "frozen_implementation_sha256": {
            relative: sha(ROOT / relative) for relative in FROZEN
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "benchmark_lifecycle_schema": "GO",
                      "frozen_files": len(FROZEN)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
