#!/usr/bin/env python3
"""Protocol, freeze, and generated-artifact checker for V6-P1i formal1024."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import py_compile
from typing import Any

import numpy as np
import yaml

import generate_heat3d_v6_p1i_continuous_dataset as generator
import heat3d_v6_p1i_continuous_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
CONFIG = CONFIG_DIR / "v6_p1i_formal1024_v0.yaml"
ACCEPTANCE = CONFIG_DIR / "v6_p1i_formal1024_acceptance.json"
FREEZE = CONFIG_DIR / "v6_p1i_formal1024_v0_freeze_manifest.json"
MANIFEST = CONFIG_DIR / "v6_p1i_formal1024_v0_manifest.json"
AUDIT = CONFIG_DIR / "v6_p1i_formal1024_v0_distribution_audit.json"
CLOSEOUT = CONFIG_DIR / "v6_p1i_formal1024_v0_closeout.json"
ARTIFACTS = CONFIG_DIR / "v6_p1i_formal1024_v0_artifact_manifest.json"
SAMPLES = CONFIG_DIR / "v6_p1i_formal1024_v0_samples.csv"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _protocol() -> dict[str, Any]:
    config = core.load_config(CONFIG)
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    _assert(config["stage"] == "formal1024", "formal stage")
    _assert(int(config["sample_count"]) == 1024, "sample count")
    _assert(
        config["split_counts"]
        == {"train": 768, "valid_iid": 128, "test_iid": 128},
        "split counts",
    )
    _assert(
        config["split_assignment"]["method"]
        == "hash_within_sobol_octets",
        "split method",
    )
    _assert(
        config["sampling"]["method"] == "scrambled_sobol"
        and int(config["sampling"]["seed"]) == 612804,
        "Sobol contract",
    )
    _assert(config["sample_id_prefix"] == "v6p1if_", "sample prefix")
    _assert(
        config["provenance"]["parent_pilot"]
        == "heat3d_v6_p1i_continuous_physics128_v2",
        "parent pilot",
    )
    _assert(
        acceptance["split_qc"]["target_independent"] is True,
        "target-independent split",
    )
    for layer in config["physics"]["layers_bottom_to_top"]:
        _assert(
            "background_k_xyz_W_mK" in layer,
            f"{layer['id']}: background k missing",
        )
    for key, value in config["guardrails"].items():
        _assert(value is False, f"guardrail {key}")
    sample_ids = [
        f"{config['sample_id_prefix']}{index:04d}" for index in range(1024)
    ]
    split_map = generator._split_map(
        sample_ids, config["split_counts"], config["split_assignment"]
    )
    _assert(
        Counter(split_map.values())
        == Counter({"train": 768, "valid_iid": 128, "test_iid": 128}),
        "split realization",
    )
    for offset in range(0, 1024, 8):
        roles = Counter(split_map[sample_ids[index]] for index in range(offset, offset + 8))
        _assert(
            roles == Counter({"train": 6, "valid_iid": 1, "test_iid": 1}),
            f"octet split {offset // 8}",
        )
    return config


def _freeze(config: dict[str, Any], required: bool) -> None:
    if not FREEZE.exists():
        _assert(not required, "freeze manifest required")
        return
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    _assert(payload["status"] == "frozen_before_generation", "freeze status")
    _assert(payload["sample_count"] == 1024, "freeze sample count")
    for artifact in payload["frozen_artifacts"]:
        path = ROOT / artifact["path"]
        _assert(path.is_file(), f"frozen artifact missing: {artifact['path']}")
        _assert(
            core.file_sha256(path) == artifact["sha256"],
            f"frozen artifact SHA drift: {artifact['path']}",
        )


def _generated(config: dict[str, Any], required: bool) -> None:
    if not MANIFEST.exists():
        _assert(not required, "generated manifest required")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert(manifest["sample_count"] == 1024, "manifest count")
    _assert(
        manifest["split_role_counts"]
        == {"test_iid": 128, "train": 768, "valid_iid": 128},
        "manifest split counts",
    )
    _assert(
        manifest["config_sha256"] == core.file_sha256(CONFIG),
        "config hash drift",
    )
    payload = copy.deepcopy(manifest)
    expected = payload.pop("manifest_payload_sha256")
    _assert(core.canonical_json_sha256(payload) == expected, "manifest payload SHA")
    dataset_root = ROOT / manifest["dataset_root"]
    _assert(dataset_root.is_dir(), "dataset root")
    _assert(len(manifest["samples"]) == 1024, "manifest samples")
    ids = set()
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        _assert(sample_id.startswith("v6p1if_"), "formal sample prefix")
        _assert(sample_id not in ids, "duplicate sample")
        ids.add(sample_id)
        sample_dir = dataset_root / row["relative_path"]
        for filename, expected_sha in row["file_sha256"].items():
            path = sample_dir / filename
            _assert(path.is_file(), f"missing {sample_id}/{filename}")
            _assert(core.file_sha256(path) == expected_sha, "sample SHA drift")
        coords = np.load(sample_dir / "coords.npy", allow_pickle=False)
        temperature = np.load(sample_dir / "temperature.npy", allow_pickle=False)
        k = np.load(sample_dir / "k_field.npy", allow_pickle=False)
        q = np.load(sample_dir / "q_field.npy", allow_pickle=False)
        _assert(coords.shape == (1024, 3), "coords shape")
        _assert(temperature.shape == (1024,), "temperature shape")
        _assert(k.shape == (1024, 3) and q.shape == (1024, 1), "field shape")
        _assert(
            all(np.all(np.isfinite(value)) for value in (coords, temperature, k, q)),
            "non-finite arrays",
        )
    _assert(AUDIT.is_file() and CLOSEOUT.is_file() and ARTIFACTS.is_file(), "closeout files")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    _assert(audit["status"] == "passed", "formal distribution gate")
    _assert(all(audit["checks"].values()), "formal checks")
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    _assert(closeout["status"] == "passed", "formal closeout")
    artifact_manifest = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    for artifact in artifact_manifest["artifacts"]:
        path = ROOT / artifact["path"]
        _assert(path.is_file(), f"artifact missing: {artifact['path']}")
        _assert(core.file_sha256(path) == artifact["sha256"], "artifact SHA drift")
    with SAMPLES.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _assert(len(rows) == 1024, "samples CSV count")
    _assert(
        Counter(row["split_role"] for row in rows)
        == Counter({"train": 768, "valid_iid": 128, "test_iid": 128}),
        "samples CSV split",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("protocol", "frozen", "generated"),
        default="protocol",
    )
    parser.add_argument("--full-preflight", action="store_true")
    args = parser.parse_args()
    config = _protocol()
    _freeze(config, required=args.phase in {"frozen", "generated"})
    _generated(config, required=args.phase == "generated")
    preflight = None
    if args.full_preflight:
        preflight = generator.preflight(CONFIG)
        _assert(preflight["sample_count"] == 1024, "preflight sample count")
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
                "phase": args.phase,
                "full_preflight": preflight,
                "training_runs": 0,
                "model_inference_runs": 0,
                "frozen_v6_modified": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
