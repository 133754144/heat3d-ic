#!/usr/bin/env python3
"""Fail-closed checks for the preregistered P1i graph-scale ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GRAPH_BUILDER_SHA = "fce189e90aa3e182a418cd1ef50a9b5d24558fc3d24e50f9d6d1e734c3129cc3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    require(payload["status"] == "preregistered_before_candidate_execution", "protocol status")
    scope = payload["scientific_scope"]
    require(scope["mandatory_candidate_resolutions"] == [8192, 16384], "mandatory resolutions")
    require(scope["winner_extension_resolutions"] == [4096, 32768], "winner extensions")
    require(scope["forbidden_resolutions"] == [65536], "forbidden resolution")
    require(not scope["training"] and not scope["test_accessed"] and not scope["sealed_accessed"], "role contract")
    candidates = payload["candidates"]
    require(candidates["A"]["run"] is False, "baseline A must be reused")
    require(candidates["D"]["trigger"] == "B_and_C_both_clear_improvement_over_A", "D trigger")
    native = payload["native1024_physical_coverage_v1"]
    require(native["uses_temperature_prediction_or_error"] is False, "native policy leakage")
    require(native["changes_legacy_discrete_physical_coverage"] is False, "legacy semantics")
    require(sha256(ROOT / "rigno/graphBuilder_Heat3D.py") == EXPECTED_GRAPH_BUILDER_SHA, "legacy graph builder drift")
    return {"protocol_checked": True, "legacy_graph_builder_unchanged": True}


def check_result(path: Path, candidate: str, resolution: int) -> dict[str, object]:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed", "candidate execution failed")
    require(payload["candidate"] == candidate and payload["resolution"] == resolution, "identity")
    require(len(payload["sample_ids"]) == 32 and len(set(payload["sample_ids"])) == 32, "valid32")
    role = payload["role_contract"]
    require(not role["training"] and not role["test"] and not role["sealed"], "role access")
    require(not role["checkpoint_modified"] and not role["support_or_physics_modified"], "frozen inputs")
    for name in ("delta_k", "delta_q", "delta_cv"):
        drift = payload["common_anchor_input_drift"][name]
        require(float(drift["max_abs"]) >= 0.0 and float(drift["max_rmse"]) >= 0.0, f"{name} drift")
    require(
        payload["common_anchor_input_drift_interpretation"]
        == "report_only_frozen_high_n_overlap_fields_vs_native1024_binary_mask_fields",
        "anchor input drift semantics",
    )
    graph = payload["graph_diagnostics"]
    require(0.0 <= float(graph["undercovered_fraction"]) <= 1.0, "under-covered fraction")
    require(float(graph["r2r_connected_components"]["max"]) >= 1.0, "connected-component diagnostic")
    for domain in ("support", "full_field"):
        values = payload["accuracy"][domain]
        for key in (
            "point_global_true_rms_relative_rmse_pct", "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K", "source_rmse_K", "background_rmse_K",
            "interface_drop_rmse_K", "peak_rmse_K",
        ):
            require(float(values[key]) >= 0.0, f"missing/non-finite {domain}.{key}")
    for name in ("neural_core", "reconstruction_apply_gpu", "warm_cache_e2e", "new_case_e2e"):
        timing = payload["timing"][name]
        require(timing["count"] >= 20 and timing["median_seconds"] > 0 and timing["p95_seconds"] > 0, f"timing {name}")
    return {"result_checked": True, "candidate": candidate, "resolution": resolution}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_scale_ablation_protocol.json")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--candidate", choices=["B", "C", "D"])
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()
    report = check_protocol(args.protocol)
    if args.result:
        require(args.candidate is not None and args.resolution is not None, "result identity arguments")
        report.update(check_result(args.result, args.candidate, args.resolution))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
