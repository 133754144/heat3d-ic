#!/usr/bin/env python3
"""Fail-closed validation for the G2-A external-baseline control plane."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/heat3d_v7/g2_candidate_registry.json"
RECEIPT = ROOT / "docs/v7_g2_a_baseline_reproduction_receipt.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    registry = _load(REGISTRY)
    receipt = _load(RECEIPT)
    _require(registry["schema_version"] == "heat3d_v7_g2_candidate_registry_v1", "registry schema drifted")
    _require(registry["status"] == "G2-A_adapter_smoke_and_qualification_only", "G2 scope opened unexpectedly")
    _require(registry["base_v7_commit"] == "78e7651bab5ef41a8ca4e42c45f64b1b98f04ea7", "G2 base commit drifted")
    scope = registry["scope"]
    for key in ("formal_multi_seed", "publication_evidence", "test_iid_access", "sealed_access", "solver_execution"):
        _require(scope[key] is False, f"forbidden G2 scope flag opened: {key}")
    p1i = registry["p1i_binding"]
    _require(p1i["dataset_id"] == "heat3d_v6_p1i_continuous_physics1024_v1", "P1i dataset drifted")
    _require(p1i["manifest_sha256"] == "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514", "P1i manifest drifted")
    _require(p1i["allowed_input_roles"] == ["train", "valid_iid"], "G2 input roles opened")
    _require(len(p1i["input_feature_names"]) == 11, "P1i feature schema width drifted")
    candidates = registry["candidates"]
    _require([row["id"] for row in candidates] == ["GINO", "Transolver", "Geo-FNO", "Therm-FM", "DeepOHeat", "DeepOHeat-v2"], "candidate set drifted")
    for row in candidates:
        if row["id"] != "DeepOHeat-v2":
            _require(len(str(row["upstream_commit"])) == 40, f"{row['id']}: upstream commit not frozen")
            _require(row["license"] in {"MIT", "Apache-2.0"}, f"{row['id']}: unsupported license")
    _require(receipt["source_control"]["v7_base_commit"] == registry["base_v7_commit"], "receipt base mismatch")
    policy = receipt["policy"]
    for key in ("formal_g2_multi_seed", "publication_evidence", "scientific_evidence_eligible", "training_executed", "solver_executed", "test_iid_access", "sealed_access", "model_selection"):
        _require(policy[key] is False, f"receipt policy flag opened: {key}")

    for relative in (
        "rigno/heat3d_g2/inputs.py",
        "rigno/heat3d_g2/p1i.py",
        "rigno/heat3d_g2/adapters.py",
        "scripts/run_v7_g2_a_adapter_smoke.py",
        "scripts/run_v7_g2_a_p1i_smoke.py",
        "scripts/run_v7_g2_a_p1i_level_a_smoke.py",
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = path.read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        _require(not any(module.startswith("scripts.") for module in imported), f"{relative}: script dependency")
        # The smoke receipt may describe the closed splits.  Audit executable
        # access patterns instead of rejecting those provenance field names.
        _require("np.load(sample_dir / \"temperature.npy\")" not in source, f"{relative}: label file access")
        _require("np.load(sample_dir / \"test_iid" not in source, f"{relative}: forbidden split file access")
        _require("np.load(sample_dir / \"sealed" not in source, f"{relative}: forbidden sealed file access")
        _require("sys.path.insert" not in source, f"{relative}: sys.path mutation")
        _require("sys.path.append" not in source, f"{relative}: sys.path mutation")
    print("V7 G2-A control plane: PASS (external adapters; no formal multi-seed work)")


if __name__ == "__main__":
    main()
