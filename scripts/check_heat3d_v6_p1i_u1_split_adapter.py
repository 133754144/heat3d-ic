#!/usr/bin/env python3
"""Check U1 split-adapter preregistration and optional closeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    require(protocol["status"] == "preregistered_before_execution", "protocol status")
    require(protocol["adapter"]["new_parameters"] == 0, "new params")
    require(protocol["identity_hard_gate"]["tolerance"] == 0.0, "identity tolerance")
    require(protocol["fail_fast"]["8192_failure_stops_32768"], "fail-fast")
    role = protocol["role_contract"]
    require(not role["training"] and not role["test"] and not role["sealed"], "role boundary")
    checked = False
    if args.result is not None:
        require(args.md is not None, "MD required")
        result = json.loads(args.result.read_text())
        require(result["status"] == "completed", "result status")
        identity = result["identity"]
        require(identity["identity_hard_gate_passed"] and identity["sample_count"] == 32, "identity valid32")
        for sample in identity["samples"]:
            require(sample["passed"], "identity sample")
            require(sample["encoder_pnode_local_transform_exact"]["array_equal"], "encoder locality")
        resolutions = [int(row["resolution"]) for row in result["high_n"]]
        require(resolutions in ([8192], [8192, 32768]), "high-N order")
        if resolutions == [8192, 32768]:
            require(result["decision"]["worth_entering_1024_to_240825"], "240825 discussion gate")
        require(result["role_contract"] == role, "result role contract")
        require("No training" in args.md.read_text(), "MD role statement")
        checked = True
    print(json.dumps({"u1_split_protocol_checked": True, "result_checked": checked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
