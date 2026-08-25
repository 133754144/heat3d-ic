#!/usr/bin/env python3
"""Fail-closed S0 audit for the publication-compatible supplemental path."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/heat3d_v6_supplemental_publication/known_topology_new_physics_protocol.json"
FROZEN = "04dc85c6ec1b620f026ea546f28a045cd43bbc9c"
E = "scripts/benchmark_heat3d_v6_p1i_final_e_service.py"
U = "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py"
EXPECTED = {
    E: "ef8087d3ffe19d4d3d044097baa14d5de39f029652cab8a4064105d53695f326",
    U: "ec916aa2e4bf37cbc2fb27d6a862f610b185e9b918781ae82b3c6c4e5fb6a834",
    "rigno/graphBuilder_Heat3D.py": "4d40e0f851e5b04f30d30192e13c2c533735490cc9a120d91489b9d3702eadd5",
    "rigno/heat3d_v6_full_field.py": "45d8e8ea8d06f022d6d405cb22d2e6dd05ebfb1e93fcc1a54901646312c4a2c8",
    "rigno/heat3d_v6_p1i_anchor_query.py": "196a13e823b1af512e0192be72c2a8c4c8ee1bfd67f3e5b7a3f9dc5495e4d7f9",
    "rigno/models/rigno.py": "f564e3870c02f3674ebb13bf3ce7380773e100383b8d7ad2eaaa7f25ae92f41c",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def nested_function(source: bytes, name: str) -> str:
    tree = ast.parse(source.decode("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            copy = ast.FunctionDef(
                name="frozen_numeric_primitive", args=node.args, body=node.body,
                decorator_list=[], returns=node.returns, type_comment=node.type_comment,
            )
            return ast.dump(copy, include_attributes=False)
    raise RuntimeError(f"function {name} absent")


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(protocol["status"] in {"preregistered_before_s0", "s0_passed_ready_for_gpu"}, "protocol status")
    require(protocol["frozen_publication_commit"] == FROZEN, "frozen commit drift")
    for path, expected in EXPECTED.items():
        frozen = git_bytes(FROZEN, path)
        require(sha(frozen) == expected, f"historical SHA drift: {path}")
        if path not in {E, U}:
            require(sha((ROOT / path).read_bytes()) == expected, f"shared code drift: {path}")

    current_e = (ROOT / E).read_bytes(); frozen_e = git_bytes(FROZEN, E)
    current_u = (ROOT / U).read_bytes(); frozen_u = git_bytes(FROZEN, U)
    require(nested_function(current_e, "forward") == nested_function(frozen_e, "forward"), "E forward drift")
    require(nested_function(current_u, "split_forward") == nested_function(frozen_u, "split_forward"), "U forward drift")
    require(nested_function(current_u, "reconstruct") == nested_function(frozen_u, "reconstruct"), "U reconstruction drift")

    orchestration_paths = [
        "scripts/heat3d_v6_supplemental_input_adapter.py",
        "scripts/heat3d_v6_supplemental_runtime.py",
        "scripts/run_heat3d_v6_supplemental_publication_known_topology.py",
    ]
    forbidden = ("jax.jit", "model.apply", "_model_apply(", "_prepare_group(",
                 "build_reconstruction_map(", "Heat3DGraphBuilder(")
    for path in orchestration_paths:
        text = (ROOT / path).read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"{path}: numerical/runtime primitive reimplemented: {token}")
        require("temperature.npy" not in text and "samples/deltaT_K" not in text,
                f"{path}: target access")

    run_text = (ROOT / orchestration_paths[-1]).read_text(encoding="utf-8")
    require("benchmark_heat3d_v6_p1i_final_e_service.py" in run_text, "E entrypoint not reused")
    require("benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py" in run_text, "U entrypoint not reused")
    require("fresh_new_case" in run_text and "known_topology_new_physics" in run_text, "workloads absent")
    require("171cd5e" not in run_text, "invalid timing attempt reused")

    payload = {
        "S0": "PASS",
        "gpu_authorized_after_protocol_status_update": True,
        "frozen_runner_sha256": {E: EXPECTED[E], U: EXPECTED[U]},
        "current_execution_files_sha256": {
            path: sha((ROOT / path).read_bytes())
            for path in [E, U, *orchestration_paths]
        },
        "numeric_primitives_ast_exact": True,
        "shared_code_exact": True,
        "forbidden_roles_accessed": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
