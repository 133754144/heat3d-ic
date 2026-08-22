#!/usr/bin/env python3
"""Low-cost static/fixture gate for final V6 publication runtime isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from heat3d_v6_publication_runtime_isolation import failure_record, validate_failure_record


ROOT = Path(__file__).resolve().parents[1]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e_text = (ROOT / "scripts/benchmark_heat3d_v6_p1i_final_e_service.py").read_text()
    u_text = (ROOT / "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py").read_text()
    harness = (ROOT / "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py").read_text()

    checks = {
        "E_hash_opt_in_untimed": (
            "include_untimed_hash_audit: bool = False" in e_text
            and "prepare_host(first_anchor, include_untimed_hash_audit=True)" in e_text),
        "E_failure_written_before_validation": (
            "if args.publication_v1_1 and q2_all_passed:" in e_text
            and "return 0 if q2_all_passed else 1" in e_text),
        "U_Q2_returns_before_post_result_audit": (
            "if production_completion_only:" in u_text
            and "outside_Q2_completion_and_refill" in u_text),
        "service_HWM_precedes_untimed_audit": (
            "captured_before_untimed_audit" in e_text
            and "captured_before_untimed_audit" in u_text),
        "FVM_case_worker_pid_authoritative": (
            'worker_pids = sorted({int(row["worker_pid"]) for row in rows_out})' in harness
            and "FVM P2 case worker participation drift" in harness),
        "formal_role_not_smoke": (
            '"real_route_smoke_only": not bool(args.formal_measurement)' in harness
            and '"formal_full_valid32" if args.formal_measurement' in harness),
        "immediate_cell_validation": "validate_completed_cell(" in harness,
        "outer_failure_embeds_inner_artifact": '"inner_failure_artifact": persisted' in harness,
    }
    for name, passed in checks.items():
        require(passed, name)

    error = RuntimeError("synthetic residual violation")
    error.failure_observability = {
        "failure_stage": "timing_residual_hard_gate",
        "residual_seconds": 0.125,
        "residual_limit_seconds": 0.025,
    }
    record = failure_record(
        error, sample_id="fixture_sample", order_position=7,
        completed_rows=[{"sample_id": "completed_fixture"}],
        failure_stage="service_future_result")
    validate_failure_record(record)
    json.dumps(record, allow_nan=False)
    result = {
        "schema_version": "heat3d_v6_publication_runtime_isolation_gate_v1",
        "status": "passed",
        "benchmark_runtime_isolation": "GO",
        "checks": checks,
        "failure_observability_fixture": record,
        "gpu_execution": False,
        "training": False,
        "test": False,
        "sealed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
