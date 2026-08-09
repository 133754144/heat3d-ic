#!/usr/bin/env python3
"""Run the preregistered A/B remaining-valid96 confirmation matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--gpu-only-amendment", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed0-run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_after_E_no_go_before_confirmation" or protocol["policies"] != ["A", "B"]:
        raise RuntimeError("A/B confirmation protocol drifted")
    preflight = json.loads((args.shared_root / "actual_data_preflight.json").read_text())
    if preflight["status"] != "passed" or preflight["sample_count"] != 96:
        raise RuntimeError("remaining-valid96 preflight missing")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        0: args.seed0_run_dir,
        1: args.runs_root / "V6_07_V5best_P1i_seed1_reliable_B24",
        2: args.runs_root / "V6_08_V5best_P1i_seed2_reliable_B24",
    }
    state = {
        "schema_version": "heat3d_v6_p1i_graph_policy_confirmation_execution_v1",
        "status": "running", "protocol_sha256": sha(args.protocol), "cells": [],
        "role_contract": {"training": False, "test": False, "sealed": False, "remaining_valid96_only": True},
    }
    state_path = args.artifact_root / "execution_state.json"
    write(state_path, state)
    for seed in (0, 1, 2):
        for resolution in (8192, 16384):
            for policy in ("A", "B"):
                output = args.artifact_root / f"seed{seed}_{policy}_{resolution}"
                log = args.artifact_root / f"seed{seed}_{policy}_{resolution}.log"
                command = [
                    sys.executable, str(ROOT / "scripts/run_heat3d_v6_p1i_graph_scale_candidate.py"),
                    "--candidate", policy, "--resolution", str(resolution),
                    "--policy-contract", str(args.protocol), "--binding", str(args.binding),
                    "--gpu-only-amendment", str(args.gpu_only_amendment),
                    "--seed", str(seed), "--run-dir", str(run_dirs[seed]),
                    "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
                    "--full-fields", str(args.full_fields), "--baseline-artifact-root", str(args.shared_root),
                    "--native-cache", str(args.artifact_root / f"native_seed{seed}.npz"),
                    "--output-dir", str(output), "--timing-repeats", "20", "--no-save-predictions",
                ]
                with log.open("w") as handle:
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
                row = {
                    "seed": seed, "policy": policy, "resolution": resolution,
                    "returncode": completed.returncode, "log": str(log), "log_sha256": sha(log),
                }
                result = output / "result.json"
                if result.is_file():
                    row.update({"result": str(result), "result_sha256": sha(result)})
                state["cells"].append(row)
                write(state_path, state)
                print(f"[confirmation] seed={seed} N={resolution} policy={policy} rc={completed.returncode}", flush=True)
                if completed.returncode != 0:
                    state["status"] = "failed"
                    write(state_path, state)
                    return completed.returncode
    state["status"] = "passed"
    write(state_path, state)
    print(json.dumps({"status": "passed", "cells": len(state["cells"])}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
