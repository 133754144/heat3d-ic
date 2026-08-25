#!/usr/bin/env python3
"""Validate the fail-closed publication-runtime supplemental gate."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "configs/heat3d_v6_supplemental_publication/fixed_geometry_publication_runtime_gate.json"


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def module_functions(source: bytes) -> set[str]:
    tree = ast.parse(source.decode("utf-8"))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def nested_functions(source: bytes, parent: str) -> set[str]:
    tree = ast.parse(source.decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == parent:
            return {
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def main() -> int:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    base = gate["base_main_commit"]
    frozen = gate["publication_runtime_provenance"]["frozen_runtime_commit"]
    require(
        subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT).returncode == 0,
        "specified main base is not an ancestor",
    )
    require(gate["status"] == "failed_closed_before_gpu_null_gate", "gate status drift")
    require(gate["null_gate"]["status"] == "FAIL_PRE_EXECUTION_CAPABILITY", "null status drift")
    require(gate["execution"] == {
        "formal_sweep_started": False,
        "smoke_started": False,
        "training_started": False,
        "test_accessed": False,
        "sealed_accessed": False,
        "new_FVM_labels_generated": False,
        "new_timing_results_generated": False,
        "unified_execution_commit": None,
    }, "execution guardrail drift")

    invalid = gate["invalidated_timing_attempt"]
    require(invalid["head"] == "171cd5e468fdae3ee599eac11bc1508097f7dd7e", "invalidated head drift")
    require(invalid["timing_speedup_setup_and_break_even_excluded"] is True, "invalidated result reuse")

    route_rows = gate["publication_runtime_provenance"]["routes"]
    for route, row in route_rows.items():
        content = git_bytes(frozen, row["runner"])
        require(hashlib.sha256(content).hexdigest() == row["sha256"], f"{route}: runner SHA")
        require(row["function"] in module_functions(content), f"{route}: function missing")
        replication = git_bytes(
            gate["publication_runtime_provenance"]["devbox_replication_commit"], row["runner"]
        )
        require(content == replication, f"{route}: WSL2/devbox runner content drift")

    for path, expected in gate["publication_runtime_provenance"]["shared_code_sha256"].items():
        require(hashlib.sha256(git_bytes(frozen, path)).hexdigest() == expected, f"shared SHA: {path}")

    e_path = route_rows["E16384_reconstruction"]["runner"]
    u_path = route_rows["U_v2_16384_reconstruction"]["runner"]
    e_source = git_bytes(frozen, e_path)
    u_source = git_bytes(frozen, u_path)
    e_text = e_source.decode("utf-8")
    u_text = u_source.decode("utf-8")
    require('choices=(4, 8, 32)' in e_text, "E sample-count contract drift")
    require('choices=[1, 4, 8, 32, 96]' in u_text, "U sample-count contract drift")
    require("known_topology_new_physics" not in e_text, "unexpected E known-topology entrypoint")
    require("known_topology_new_physics" not in u_text, "unexpected U known-topology entrypoint")
    require({"prepare_host", "service_one"}.issubset(nested_functions(e_source, "main")), "E closures drift")
    require("prepare_one" in nested_functions(u_source, "main"), "U closure drift")
    require('payload = host_payload_cache[anchor.sample_id]' in e_text, "E cache-hot semantics drift")
    require('cached_host=declared_case_cache[cached["sample_id"]]' in u_text, "U cache-hot semantics drift")

    raw_paths = list((ROOT / "configs/heat3d_v6_supplemental_publication").glob("**/*raw*"))
    require(not raw_paths, f"unexpected new raw timing artifacts: {raw_paths}")
    print(json.dumps({
        "governance_checker": "PASS",
        "null_gate": "FAIL_PRE_EXECUTION_CAPABILITY",
        "formal_sweep_started": False,
        "reason": gate["null_gate"]["difference"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
