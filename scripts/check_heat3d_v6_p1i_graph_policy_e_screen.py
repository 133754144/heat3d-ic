#!/usr/bin/env python3
"""Validate E fail-closed screen and frozen A/B confirmation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)

def main() -> int:
    screen_path = CONFIG / "v6_p1i_graph_policy_e_screen_closeout.json"
    screen = json.loads(screen_path.read_text())
    require(screen["status"] == "completed_E_no_go_stopped_after_8192", "screen status")
    require(screen["decision"] == "E_NO_GO" and not screen["E_16384_executed"], "E stop")
    require(all(row["passed"] for row in screen["accuracy"].values()), "E accuracy screen")
    require(screen["coverage"]["passed"], "E coverage")
    require(screen["latency"]["speed_passed"] and not screen["latency"]["vram_passed"], "E latency cause")
    raw = ROOT / screen["artifacts"]["E_8192"]["path"]
    require(sha(raw) == screen["artifacts"]["E_8192"]["sha256"], "E SHA")
    protocol = json.loads((CONFIG / "v6_p1i_graph_policy_ab_confirmation_protocol.json").read_text())
    require(protocol["status"] == "frozen_after_E_no_go_before_confirmation", "confirmation status")
    require(protocol["policies"] == ["A", "B"] and protocol["primary_candidate"] == "B", "policies")
    require(protocol["population"]["count"] == 96 and not protocol["population"]["frozen_valid32_recomputed"], "population")
    require(protocol["paired_bootstrap"]["cluster_unit"] == "sample_id_with_all_three_seeds", "bootstrap unit")
    require(sha(screen_path) == protocol["screen_closeout"]["sha256"], "screen binding")
    print(json.dumps({"status": "passed", "E_no_go": True, "confirmation_policies": ["A", "B"]}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
