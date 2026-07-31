#!/usr/bin/env python3
"""Deterministically verify the V6 P1h canonical dataset closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_heat3d_v6_common_valid_probe import build as rebuild_probe  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CANONICAL_V6_DATASET_ID,
    P1G_GEOMETRY_ADAPTIVE_V6_DATASET_ID,
    SHARED_SUPPORT_V6_DATASET_ID,
    SUPPORTED_V6_DATASET_IDS,
)


P1G = "heat3d_v6_p1g_geometry_deconfounded1024_v0"
P1H = "heat3d_v6_p1h_shared_support1024_v0"
MANIFEST_SHA256 = "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
ARCHIVE_SHA256 = "f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00"
FROZEN_CONFIG_SHA256 = {
    "V6_01_V4best.yaml": "e4024f4bbbc4dab173c5512216bed895c0175ca45a02e616061af31456d50ad7",
    "V6_02_V5best.yaml": "5aa236b6aaffc46d604f0d9c1bea741f0c6c6acabdae262cd056b47136b44fd5",
    "V6_03_V5best_P1h.yaml": "1f2c0564fb87a6ac067df6deb2a661a78ca104623775c9616a5936c990fc1ddf",
    "V6_04_V5best_P1h_DualAttention.yaml": "5c7ec0ae37a8c9495044c4f498f219dd954f60ba65c713f1821962c1fc333931",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()

    _assert(CANONICAL_V6_DATASET_ID == P1H, "loader canonical constant drifted")
    _assert(SHARED_SUPPORT_V6_DATASET_ID == P1H, "shared-support ID drifted")
    _assert(
        P1G_GEOMETRY_ADAPTIVE_V6_DATASET_ID == P1G,
        "P1g historical ID drifted",
    )
    _assert(SUPPORTED_V6_DATASET_IDS == {P1G, P1H}, "loader support set drifted")

    default = yaml.safe_load(
        (ROOT / "configs/heat3d_v6/v6_layer_canonical_default.yaml").read_text(
            encoding="utf-8"
        )
    )
    _assert(default["dataset_id"] == P1H, "default dataset is not P1h")
    _assert(default["lifecycle_status"] == "canonical", "P1h is not canonical")
    _assert(
        default["canonical_model_configuration"] == "V6_03_V5best_P1h",
        "canonical model configuration drifted",
    )
    _assert(
        default["registered_ablation"] == "V6_04_V5best_P1h_DualAttention",
        "ablation drift",
    )
    _assert(
        default["archived_geometry_adaptive_baseline"]["dataset_id"] == P1G,
        "P1g archive binding drifted",
    )
    _assert(
        default["archived_geometry_adaptive_baseline"]["historical_run_bindings"]
        == ["V6_01_V4best", "V6_02_V5best"],
        "historical P1g run bindings drifted",
    )

    with (ROOT / "configs/heat3d_v6/v6_training_dataset_lifecycle.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        lifecycle = list(csv.DictReader(handle))
    canonical = [row for row in lifecycle if row["lifecycle_status"] == "canonical"]
    _assert(
        len(canonical) == 1 and canonical[0]["dataset_id"] == P1H,
        "lifecycle must have exactly one P1h canonical row",
    )
    p1g_row = next(row for row in lifecycle if row["dataset_id"] == P1G)
    _assert(
        p1g_row["lifecycle_status"] == "archived_geometry_adaptive_baseline",
        "P1g lifecycle drifted",
    )

    for name, expected in FROZEN_CONFIG_SHA256.items():
        _assert(
            _sha256(ROOT / "configs/heat3d_v6" / name) == expected,
            f"frozen config was modified: {name}",
        )

    manifest_path = dataset / "manifest.json"
    archive_path = dataset / "full_fields.h5"
    _assert(_sha256(manifest_path) == MANIFEST_SHA256, "manifest SHA256 mismatch")
    _assert(_sha256(archive_path) == ARCHIVE_SHA256, "archive SHA256 mismatch")
    manifest = _read_json(manifest_path)
    _assert(manifest["dataset_id"] == P1H, "dataset ID mismatch")
    _assert(len(manifest["samples"]) == 1024, "sample count drifted")
    _assert(
        len({str(row["sample_dir"]) for row in manifest["samples"]}) == 1024,
        "sample directories are not unique",
    )
    _assert(
        all((dataset / str(row["sample_dir"])).is_dir() for row in manifest["samples"]),
        "one or more sample directories are absent",
    )

    freeze = _read_json(ROOT / "configs/heat3d_v6/v6_run_artifact_freeze.json")
    _assert(freeze["status"] == "frozen", "run artifacts are not frozen")
    _assert(freeze["run_count"] == 4, "freeze run count drifted")
    _assert(freeze["artifact_count"] == 14, "freeze artifact count drifted")
    _assert(
        [row["config_id"] for row in freeze["runs"]]
        == [
            "V6_01_V4best",
            "V6_02_V5best",
            "V6_03_V5best_P1h",
            "V6_04_V5best_P1h_DualAttention",
        ],
        "freeze run IDs drifted",
    )
    for run in freeze["runs"]:
        for artifact in run["artifacts"]:
            _assert(
                len(artifact["checkpoint_sha256"]) == 64
                and len(artifact["prediction_sha256"]) == 64,
                "artifact SHA256 missing",
            )
    _assert(
        not freeze["immutability_policy"]["overwrite_allowed"]
        and not freeze["immutability_policy"]["historical_run_directories_mutated"],
        "historical artifact immutability was relaxed",
    )

    probe_path = ROOT / "configs/heat3d_v6/v6_valid_common_probe4096.json"
    probe = _read_json(probe_path)
    rebuilt = rebuild_probe(
        ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_acceptance.json",
        ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json",
    )
    _assert(probe == rebuilt, "4096-node probe is not deterministic")
    _assert(
        probe["node_count"] == 4096
        and probe["sample_count"] == 128
        and probe["evaluation_role"] == "valid_iid",
        "probe shape/role drifted",
    )
    _assert(
        probe["label_independent"]
        and not probe["test_hard_accessed"]
        and probe["coverage"]["all_layers_covered"]
        and probe["coverage"]["all_interfaces_covered"],
        "probe leakage/coverage contract failed",
    )

    results = _read_json(
        ROOT / "configs/heat3d_v6/v6_common_valid_probe4096_results.json"
    )
    _assert(results["status"] == "passed", "common-probe evaluation failed")
    _assert(
        results["evaluation_role"] == "valid_iid"
        and not results["test_hard_accessed"]
        and not results["training_executed"]
        and not results["checkpoint_selection_modified"],
        "common-probe role or mutation guard failed",
    )
    _assert(
        set(results["models"])
        == {
            "V6_02_V5best",
            "V6_03_V5best_P1h",
            "V6_04_V5best_P1h_DualAttention",
        },
        "common-probe model set drifted",
    )
    _assert(_all_finite(results["models"]), "common-probe metrics are non-finite")
    for row in results["models"].values():
        _assert(row["metrics"]["sample_count"] == 128, "valid sample count drifted")
        _assert(row["metrics"]["node_count"] == 4096, "probe node count drifted")

    receipt = _read_json(ROOT / "configs/heat3d_v6/v6_hf_sync_receipt.json")
    _assert(receipt["status"] == "verified", "HF mirror is not verified")
    _assert(receipt["dataset_id"] == P1H, "HF dataset ID drifted")
    _assert(receipt["manifest_sha256"] == MANIFEST_SHA256, "HF manifest hash drifted")
    _assert(receipt["archive_sha256"] == ARCHIVE_SHA256, "HF archive hash drifted")
    _assert(receipt["remote_sample_directory_count"] == 1024, "HF sample count drifted")
    _assert(receipt["remote_file_count"] == 10243, "HF file count drifted")
    _assert(
        receipt["path_in_repo"]
        == f"subsets/{P1H}/",
        "HF destination path drifted",
    )
    _assert(not receipt["other_subsets_modified"], "another HF subset was modified")
    _assert(len(receipt["hf_commit"]) == 40, "HF commit is not frozen")

    print(
        json.dumps(
            {
                "status": "passed",
                "canonical_dataset": P1H,
                "archived_baseline": P1G,
                "frozen_runs": freeze["run_count"],
                "frozen_artifacts": freeze["artifact_count"],
                "probe_nodes": probe["node_count"],
                "valid_samples": results["models"]["V6_03_V5best_P1h"]["metrics"][
                    "sample_count"
                ],
                "hf_commit": receipt["hf_commit"],
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
