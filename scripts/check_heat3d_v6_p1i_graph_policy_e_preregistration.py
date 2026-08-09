#!/usr/bin/env python3
"""Fail-closed checks for the P1i E screen and confirmation preregistration."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/heat3d_v6_p1i/v6_p1i_graph_policy_e_confirmation_preregistration.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    payload = json.loads(PROTOCOL.read_text())
    require(payload["status"] == "preregistered_before_E_execution", "status")
    require(payload["base_commit"].startswith("7a5adcc"), "base commit")
    scope = payload["scope"]
    require(not any(scope.values()), "scope must be entirely false")
    e = payload["policies"]["E"]
    require(e["regional_node_count"] == 256, "E Nr")
    require(e["resolved_subsample_factor"] == {"8192": 32, "16384": 64}, "E factors")
    margins = payload["historical_margin_source"]["values"]
    require(margins == {
        "full_point_global_pct": 0.05843506828224949,
        "full_raw_cv_rmse_K": 0.019179597941702667,
        "source_rmse_K": 0.13095819024668282,
        "peak_rmse_K": 0.3785072688630482,
        "interface_rmse_K": 0.04746620520450112,
    }, "historical margins")
    confirmation = payload["confirmation"]
    require(confirmation["seeds"] == [0, 1, 2], "seeds")
    require(confirmation["paired_bootstrap"] == {
        "seed": 20260810, "replicates": 10000, "confidence": 0.95,
        "unit": "sample", "difference": "candidate_minus_A",
    }, "bootstrap")
    require(confirmation["policy_set_if_E_fails"] == ["A", "B"], "fail policy set")
    require(confirmation["policy_set_if_E_passes"] == ["A", "B", "E"], "pass policy set")
    print(json.dumps({"status": "passed", "preregistration_checked": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
