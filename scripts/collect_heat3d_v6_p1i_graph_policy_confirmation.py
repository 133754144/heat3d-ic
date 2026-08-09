#!/usr/bin/env python3
"""Compact the remote A/B confirmation matrix while preserving SHA provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.artifact_root / "execution_state.json"
    state = json.loads(state_path.read_text())
    if state["status"] != "passed" or len(state["cells"]) != 12:
        raise RuntimeError("confirmation matrix incomplete")
    cells = []
    for row in state["cells"]:
        path = Path(row["result"])
        if sha(path) != row["result_sha256"]:
            raise RuntimeError("result SHA drifted")
        source = json.loads(path.read_text())
        if source["status"] != "passed" or len(source["sample_ids"]) != 96:
            raise RuntimeError("confirmation result invalid")
        cells.append({
            "seed": row["seed"], "policy": row["policy"], "resolution": row["resolution"],
            "checkpoint": source["checkpoint"], "sample_ids": source["sample_ids"],
            "accuracy": source["accuracy"], "per_sample_metrics": source["per_sample_metrics"],
            "timing": source["timing"], "device_memory": source["device_memory"],
            "graph_diagnostics": source["graph_diagnostics"], "role_contract": source["role_contract"],
            "raw_result": {"remote_path": str(path), "sha256": row["result_sha256"]},
            "log": {"remote_path": row["log"], "sha256": row["log_sha256"]},
        })
    payload = {
        "schema_version": "heat3d_v6_p1i_graph_policy_confirmation_compact_v1",
        "status": "passed", "execution_state_sha256": sha(state_path),
        "cells": cells, "role_contract": state["role_contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "cells": len(cells), "sha256": sha(args.output)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
