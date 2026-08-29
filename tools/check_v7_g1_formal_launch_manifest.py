#!/usr/bin/env python3
"""Fail-closed prelaunch validation for the frozen V7 G1 matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "heat3d_v7" / "v7_g1_formal_launch_manifest.json"
PARENT = ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json"
PREREG = ROOT / "configs" / "heat3d_v7" / "v7_g1_statistical_preregistration.json"
PROVIDER = ROOT / "configs" / "heat3d_v7" / "v7_g1_support_provider_contract.json"

EXPECTED = [
    ("V7-G1-Full-P1i", "Full"),
    ("V7-G1-Full-P1i:vanilla-RIGNO", "vanilla_RIGNO"),
    ("V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched", "vanilla_RIGNO_capacity_matched"),
    ("V7-G1-Full-P1i:layout-agnostic-stratified-support", "layout_agnostic_stratified_support"),
    ("V7-G1-Full-P1i:cv-only-support", "cv_only_support"),
    ("V7-G1-Full-P1i:no-film", "no_film"),
    ("V7-G1-Full-P1i:physics-scale-only", "physics_scale_only"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    manifest = _load(MANIFEST)
    _require(
        manifest.get("schema_version") == "heat3d_v7_g1_formal_launch_manifest_v1",
        "launch manifest schema drifted",
    )
    _require(manifest.get("status") == "frozen_launch_manifest", "launch manifest is not frozen")
    _require(manifest.get("branch") == "research/v7", "launch branch drifted")
    _require(manifest.get("test_iid_access") is False, "test_iid access is open")
    _require(manifest.get("sealed_access") is False, "sealed access is open")
    _require(manifest.get("matrix", {}).get("formal_execution_started") is False, "formal matrix is already open")
    code_sha = str(manifest.get("g1_formal_code_sha", ""))
    _require(len(code_sha) == 40 and all(c in "0123456789abcdef" for c in code_sha), "formal code SHA is not pinned")
    current_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "cat-file", "-e", f"{code_sha}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", code_sha, current_sha],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [
        name
        for name in subprocess.run(
            ["git", "diff", "--name-only", code_sha, current_sha],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if name
    ]
    _require(
        changed in ([], [str(MANIFEST.relative_to(ROOT))]),
        "scientific files changed after formal code freeze: " + ", ".join(changed),
    )

    parent = _load(PARENT)
    prereg = _load(PREREG)
    _require(manifest.get("parent_config_sha256") == _sha256(PARENT), "parent config SHA mismatch")
    _require(manifest.get("preregistration_file_sha256") == _sha256(PREREG), "preregistration file SHA mismatch")
    _require(
        manifest.get("preregistration_sha256") == prereg.get("preregistration_sha256"),
        "preregistration canonical SHA mismatch",
    )
    _require(
        manifest.get("support_provider_contract_sha256") == _sha256(PROVIDER),
        "support-provider contract SHA mismatch",
    )
    _require(
        manifest.get("dataset", {}).get("dataset_id") == parent.get("dataset", {}).get("dataset_id"),
        "dataset ID mismatch",
    )
    for key in ("manifest_sha256", "full_field_archive_sha256"):
        _require(
            manifest.get("dataset", {}).get(key) == parent.get("dataset", {}).get(key),
            f"dataset binding mismatch: {key}",
        )

    matrix = manifest.get("matrix", {})
    _require(matrix.get("variants") == [variant for _experiment, variant in EXPECTED], "variant order drifted")
    _require(matrix.get("seeds") == [0, 1, 2], "seed set drifted")
    _require(matrix.get("epochs") == 200, "epoch budget drifted")
    _require(matrix.get("run_count") == 21, "run count drifted")
    runs = manifest.get("runs", [])
    _require(len(runs) == 21, "manifest does not contain 21 runs")
    expected_rows = {
        (experiment, seed): {"experiment_id": experiment, "variant": variant, "seed": seed}
        for experiment, variant in EXPECTED
        for seed in (0, 1, 2)
    }
    actual_rows = {}
    run_ids = set()
    for row in runs:
        key = (row.get("experiment_id"), int(row.get("seed", -1)))
        _require(key in expected_rows, f"unexpected formal run row: {key}")
        _require(key not in actual_rows, f"duplicate formal run row: {key}")
        _require(row.get("variant") == expected_rows[key]["variant"], f"variant mismatch: {key}")
        run_id = str(row.get("run_id", ""))
        _require(run_id and run_id not in run_ids, "formal run IDs are not unique")
        run_ids.add(run_id)
        actual_rows[key] = row
    _require(set(actual_rows) == set(expected_rows), "formal run matrix is incomplete")
    print(f"V7 G1 formal launch manifest: PASS ({len(runs)} runs; code={code_sha}; prelaunch)")


if __name__ == "__main__":
    main()
