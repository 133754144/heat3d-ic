#!/usr/bin/env python3
"""Validate P6 protocol and optional runtime closeout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    require(protocol["status"] == "preregistered_before_execution", "protocol")
    require([row["route"] for row in protocol["routes"]] == ["native1024", "E16384", "E32768"], "routes")
    require(protocol["batch_contract"]["sizes"] == [1, 2, 4, 8, 16], "batch sizes")
    role = protocol["role_contract"]
    require(not role["training"] and not role["test"] and not role["sealed"], "roles")
    checked = False
    if args.result:
        result = json.loads(args.result.read_text())
        require(result["status"] in {"passed", "passed_smoke"}, "status")
        require(result["role_contract"] == role, "role contract")
        require(result["checkpoint_sha256"] == protocol["checkpoint"]["sha256"], "checkpoint")
        if result["route"]["route"] == "native1024":
            require(result["native1024_reuse_exact_gate"], "native reuse")
        for state in ("fresh_sample", "same_input_replay"):
            require(result["runtime"][state], state)
        for row in result["batch"]:
            if row["status"] == "passed":
                require(math.isfinite(row["samples_per_second"]) and row["samples_per_second"] > 0, "batch finite")
        checked = True
    print(json.dumps({"p6_protocol_checked": True, "result_checked": checked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
