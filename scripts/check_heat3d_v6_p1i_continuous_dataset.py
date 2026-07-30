#!/usr/bin/env python3
"""Deterministic contract and artifact checker for V6-P1i."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import py_compile
from typing import Any

import numpy as np
import yaml

import heat3d_v6_p1i_continuous_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
CONFIG = CONFIG_DIR / "v6_p1i_pilot128_v1.yaml"
ACCEPTANCE = CONFIG_DIR / "v6_p1i_pilot_acceptance.json"
BACKGROUND = CONFIG_DIR / "v6_p1i_background_k_contract.csv"
LITERATURE = CONFIG_DIR / "v6_p1i_literature.json"
MANIFEST = CONFIG_DIR / "v6_p1i_pilot128_v1_manifest.json"
AUDIT = CONFIG_DIR / "v6_p1i_pilot128_v1_distribution_audit.json"
CLOSEOUT = CONFIG_DIR / "v6_p1i_pilot128_v1_closeout.json"
ATTEMPTS = CONFIG_DIR / "v6_p1i_generation_attempts.csv"
ARTIFACT_MANIFEST = CONFIG_DIR / "v6_p1i_pilot128_v1_artifact_manifest.json"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _base_contract() -> dict[str, Any]:
    config = core.load_config(CONFIG)
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    literature = json.loads(LITERATURE.read_text(encoding="utf-8"))
    with BACKGROUND.open(encoding="utf-8", newline="") as handle:
        background = list(csv.DictReader(handle))
    layers = config["physics"]["layers_bottom_to_top"]
    _assert(len(layers) == 9, "nine-layer stack required")
    _assert(len(background) == 9, "background-k table must cover nine layers")
    _assert(
        {row["layer_id"] for row in background}
        == {row["id"] for row in layers},
        "background-k table/layer mismatch",
    )
    for layer in layers:
        _assert(
            "background_k_xyz_W_mK" in layer,
            f"{layer['id']}: explicit background missing",
        )
        values = list(map(float, layer["background_k_xyz_W_mK"]))
        _assert(len(values) == 3 and min(values) > 0.0, "invalid background k")
        _assert("sampling" in layer, f"{layer['id']}: sampling missing")
    missing = copy.deepcopy(config["physics"])
    del missing["layers_bottom_to_top"][0]["background_k_xyz_W_mK"]
    try:
        core.build_mesh(missing)
    except core.ContinuousPhysicsError as exc:
        _assert("missing explicit" in str(exc), "wrong missing-background error")
    else:
        raise AssertionError("missing background k did not raise")
    config_text = CONFIG.read_text(encoding="utf-8").lower()
    _assert("temperature_bin" not in config_text, "four-bin design leaked")
    _assert(
        config["sampling"]["method"] == "scrambled_sobol",
        "Sobol method not frozen",
    )
    _assert(
        int(config["sample_count"]) == 128
        and sum(map(int, config["split_counts"].values())) == 128,
        "pilot cardinality mismatch",
    )
    _assert(
        acceptance["decision"]["report_pilot_before_formal_decision"],
        "pilot-first decision gate missing",
    )
    _assert(len(literature["sources"]) >= 5, "literature inventory too small")
    for key, value in config["guardrails"].items():
        _assert(value is False, f"guardrail {key} must be false")
    return config


def _artifact_contract(config: dict[str, Any]) -> None:
    if not MANIFEST.exists():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert(manifest["sample_count"] == 128, "manifest sample count")
    _assert(
        manifest["split_role_counts"]
        == {"test_iid": 16, "train": 96, "valid_iid": 16},
        "split role counts",
    )
    _assert(
        manifest["config_sha256"] == core.file_sha256(CONFIG),
        "config hash drift",
    )
    hash_payload = copy.deepcopy(manifest)
    expected_payload_hash = hash_payload.pop("manifest_payload_sha256")
    _assert(
        core.canonical_json_sha256(hash_payload) == expected_payload_hash,
        "manifest payload hash drift",
    )
    dataset_root = ROOT / manifest["dataset_root"]
    _assert(dataset_root.exists(), "dataset root missing")
    _assert(len(manifest["samples"]) == 128, "sample manifest count")
    ids = set()
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        _assert(sample_id not in ids, "duplicate sample ID")
        ids.add(sample_id)
        sample_dir = dataset_root / row["relative_path"]
        _assert(sample_dir.is_dir(), f"sample directory missing: {sample_id}")
        for filename, expected in row["file_sha256"].items():
            path = sample_dir / filename
            _assert(path.is_file(), f"missing {sample_id}/{filename}")
            _assert(core.file_sha256(path) == expected, "sample SHA drift")
        coords = np.load(sample_dir / "coords.npy", allow_pickle=False)
        temperature = np.load(
            sample_dir / "temperature.npy", allow_pickle=False
        )
        k = np.load(sample_dir / "k_field.npy", allow_pickle=False)
        q = np.load(sample_dir / "q_field.npy", allow_pickle=False)
        _assert(coords.shape == (1024, 3), "coordinate shape")
        _assert(temperature.shape == (1024,), "temperature shape")
        _assert(k.shape == (1024, 3), "k shape")
        _assert(q.shape == (1024, 1), "q shape")
        _assert(
            all(np.all(np.isfinite(value)) for value in (coords, temperature, k, q)),
            "non-finite sample array",
        )
        _assert(float(np.min(k)) > 0.0, "non-positive k")
    _assert(AUDIT.exists(), "distribution audit missing")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    _assert(audit["status"] in {"passed", "failed"}, "audit status invalid")
    _assert(
        audit["guardrails"]["formal1024_generated"] is False,
        "formal1024 must remain blocked",
    )
    _assert(CLOSEOUT.exists(), "pilot closeout missing")
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    _assert(
        closeout["status"] == "failed_distribution_gate",
        "closeout status must preserve failed gate",
    )
    _assert(
        closeout["decision"]["formal1024_allowed"] is False,
        "formal1024 must remain disallowed",
    )
    with ATTEMPTS.open(encoding="utf-8", newline="") as handle:
        attempts = {row["attempt_id"]: row for row in csv.DictReader(handle)}
    _assert(
        attempts["pilot128_v0"]["status"] == "failed_generator_geometry_gate",
        "v0 failure provenance missing",
    )
    _assert(
        attempts["pilot128_v1"]["status"] == "failed_distribution_gate",
        "v1 failure provenance missing",
    )
    artifact_manifest = json.loads(
        ARTIFACT_MANIFEST.read_text(encoding="utf-8")
    )
    _assert(
        artifact_manifest["formal1024_allowed"] is False,
        "artifact manifest opened formal1024",
    )
    for artifact in artifact_manifest["artifacts"]:
        path = ROOT / artifact["path"]
        _assert(path.is_file(), f"artifact missing: {artifact['path']}")
        _assert(
            core.file_sha256(path) == artifact["sha256"],
            f"artifact SHA drift: {artifact['path']}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-generated", action="store_true", help="require pilot artifacts"
    )
    args = parser.parse_args()
    config = _base_contract()
    if args.require_generated:
        _assert(MANIFEST.exists(), "generated manifest required")
    _artifact_contract(config)
    for path in (
        ROOT / "scripts/heat3d_v6_p1i_continuous_core.py",
        ROOT / "scripts/generate_heat3d_v6_p1i_continuous_dataset.py",
        ROOT / "scripts/audit_heat3d_v6_p1i_distribution.py",
        Path(__file__),
    ):
        py_compile.compile(str(path), doraise=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "config": str(CONFIG.relative_to(ROOT)),
                "manifest_present": MANIFEST.exists(),
                "formal1024_generated": False,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
