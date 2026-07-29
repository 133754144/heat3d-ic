#!/usr/bin/env python3
"""Validate the complete V6-RandomBlock staged-generation closeout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import generate_heat3d_v6_randomblock_dataset as generator
import heat3d_v6_randomblock_core as core
from check_heat3d_v6_randomblock_dataset import check as check_dataset


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_randomblock"
MANIFEST_PATH = CONFIG_DIR / "v6_randomblock_closeout_manifest.json"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise core.RandomBlockError(f"{path}: expected object")
    return payload


def _manifest_payload_hash(path: Path) -> str:
    payload = _json(path)
    expected = str(payload.pop("manifest_payload_sha256"))
    actual = core.canonical_json_sha256(payload)
    if actual != expected:
        raise core.RandomBlockError(f"{path}: payload SHA mismatch")
    return actual


def check(dataset_root: Path | None) -> dict[str, Any]:
    closeout = _json(MANIFEST_PATH)
    if closeout["branch"] != "research/v6-randomblock":
        raise core.RandomBlockError("closeout branch mismatch")
    if any(bool(value) for value in closeout["guardrails"].values()):
        raise core.RandomBlockError("closeout guardrail violation")
    stage_files = {
        "smoke16_v0": {
            "config": CONFIG_DIR / "v6_randomblock_smoke16.yaml",
            "manifest": CONFIG_DIR / "v6_randomblock_smoke16_manifest.json",
            "audit": CONFIG_DIR / "v6_randomblock_smoke16_audit.json",
        },
        "smoke16_v1": {
            "config": CONFIG_DIR / "v6_randomblock_smoke16_v1.yaml",
            "manifest": CONFIG_DIR / "v6_randomblock_smoke16_v1_manifest.json",
            "audit": CONFIG_DIR / "v6_randomblock_smoke16_v1_audit.json",
        },
        "pilot128_v1": {
            "config": CONFIG_DIR / "v6_randomblock_pilot128_v1.yaml",
            "manifest": CONFIG_DIR / "v6_randomblock_pilot128_v1_manifest.json",
            "audit": CONFIG_DIR / "v6_randomblock_pilot128_v1_audit.json",
        },
        "pilot128_v2": {
            "config": CONFIG_DIR / "v6_randomblock_pilot128_v2.yaml",
            "manifest": CONFIG_DIR / "v6_randomblock_pilot128_v2_manifest.json",
            "audit": CONFIG_DIR / "v6_randomblock_pilot128_v2_audit.json",
        },
        "formal1024_v2": {
            "config": CONFIG_DIR / "v6_randomblock_formal1024_v2.yaml",
            "manifest": CONFIG_DIR / "v6_randomblock_formal1024_v2_manifest.json",
            "audit": CONFIG_DIR / "v6_randomblock_formal1024_v2_audit.json",
        },
    }
    for stage, files in stage_files.items():
        config = core.load_config(files["config"])
        generator._config_contract(config)
        tracked = closeout["stages"][stage]
        if config["provenance"]["protocol_sha256"] != tracked["protocol_sha256"]:
            raise core.RandomBlockError(f"{stage}: protocol SHA mismatch")
        manifest_hash = _manifest_payload_hash(files["manifest"])
        if manifest_hash != tracked["manifest_payload_sha256"]:
            raise core.RandomBlockError(f"{stage}: manifest SHA mismatch")
        manifest = _json(files["manifest"])
        if (
            str(manifest["full_field_archive"]["sha256"])
            != tracked["full_field_archive_sha256"]
        ):
            raise core.RandomBlockError(f"{stage}: archive SHA mismatch")
        audit = _json(files["audit"])
        if (
            int(audit["guardrails"]["training_runs"]) != 0
            or int(audit["guardrails"]["model_inference_runs"]) != 0
            or int(audit["guardrails"]["temperature_filtered_samples"]) != 0
            or int(audit["guardrails"]["sample_replacements"]) != 0
        ):
            raise core.RandomBlockError(f"{stage}: lifecycle violation")
    expected_status = {
        "smoke16_v0": "failed_temperature_gate",
        "smoke16_v1": "passed",
        "pilot128_v1": "failed_temperature_gate",
        "pilot128_v2": "passed",
        "formal1024_v2": "failed_temperature_gate",
    }
    for stage, status in expected_status.items():
        actual = _json(stage_files[stage]["audit"])["status"]
        if actual != status:
            raise core.RandomBlockError(
                f"{stage}: audit status {actual} != {status}"
            )

    formal = closeout["stages"]["formal1024_v2"]
    formal_manifest = _json(stage_files["formal1024_v2"]["manifest"])
    formal_audit = _json(stage_files["formal1024_v2"]["audit"])
    joint = _json(
        CONFIG_DIR / "v6_randomblock_formal1024_v2_joint_audit.json"
    )
    if (
        core.file_sha256(stage_files["formal1024_v2"]["config"])
        != formal["config_file_sha256"]
        or core.file_sha256(stage_files["formal1024_v2"]["manifest"])
        != formal["manifest_file_sha256"]
        or core.file_sha256(stage_files["formal1024_v2"]["audit"])
        != formal["audit_file_sha256"]
        or core.file_sha256(
            CONFIG_DIR / "v6_randomblock_formal1024_v2_joint_audit.json"
        )
        != formal["joint_audit_file_sha256"]
    ):
        raise core.RandomBlockError("formal tracked-file SHA mismatch")
    if (
        int(formal_manifest["sample_count"]) != 1024
        or int(formal_manifest["group_count"]) != 128
        or formal_audit["split_role_counts"]
        != {"train": 768, "valid": 128, "test": 128}
        or formal_audit["intended_temperature_bin_counts"]
        != {"0": 256, "1": 256, "2": 256, "3": 256}
        or formal_audit["realized_temperature_bin_counts"]
        != {"0": 256, "1": 256, "2": 255, "3": 257}
        or int(formal_audit["peak_deltaT_K"]["inside_30_150_count"]) != 1024
    ):
        raise core.RandomBlockError("formal cardinality/bin contract mismatch")
    with (
        CONFIG_DIR / "v6_randomblock_formal1024_v2_samples.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    crossings = [
        row
        for row in rows
        if int(row["intended_temperature_bin"])
        != int(row["realized_temperature_bin"])
    ]
    if len(crossings) != 1:
        raise core.RandomBlockError("formal deviation is not exactly one sample")
    crossing = crossings[0]
    if not (
        crossing["variant_id"] == "v4"
        and int(crossing["intended_temperature_bin"]) == 2
        and int(crossing["realized_temperature_bin"]) == 3
    ):
        raise core.RandomBlockError("unexpected formal crossing")
    if not bool(joint["passed"]) or int(
        joint["support_coverage"]["zero_coverage_count"]
    ):
        raise core.RandomBlockError("formal joint audit failed")
    if not bool(closeout["downstream_status"]["dataset_generation_complete"]):
        raise core.RandomBlockError("dataset not marked complete")
    if bool(closeout["downstream_status"]["canonical_promotion"]):
        raise core.RandomBlockError("unexpected canonical promotion")

    dataset_check = None
    if dataset_root is not None:
        dataset_check = check_dataset(
            stage_files["formal1024_v2"]["config"],
            dataset_root,
            stage_files["formal1024_v2"]["manifest"],
            stage_files["formal1024_v2"]["audit"],
            require_temperature_gate=False,
        )
        if dataset_check["full_field_archive_sha256"] != formal[
            "full_field_archive_sha256"
        ]:
            raise core.RandomBlockError("remote dataset archive mismatch")
    return {
        "status": "passed",
        "stage_count": len(stage_files),
        "formal_dataset_id": formal["dataset_id"],
        "formal_sample_count": 1024,
        "formal_group_count": 128,
        "formal_physical_window_gate": True,
        "formal_strict_exact_realized_bin_gate": False,
        "formal_realized_bin_max_count_deviation": 1,
        "formal_crossing_sample_id": crossing["sample_id"],
        "formal_archive_sha256": formal["full_field_archive_sha256"],
        "dataset_files_checked": dataset_root is not None,
        "training_runs": 0,
        "model_inference_runs": 0,
        "canonical_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    dataset_root = None
    if args.dataset_root is not None:
        dataset_root = (
            args.dataset_root
            if args.dataset_root.is_absolute()
            else ROOT / args.dataset_root
        ).resolve()
    print(json.dumps(check(dataset_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
