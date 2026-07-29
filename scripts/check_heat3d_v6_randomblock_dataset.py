#!/usr/bin/env python3
"""Deterministically validate a generated V6-RandomBlock dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

import generate_heat3d_v6_randomblock_dataset as generator
import heat3d_v6_randomblock_core as core


ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise core.RandomBlockError(f"{path}: expected JSON object")
    return payload


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    expected = str(payload.pop("manifest_payload_sha256"))
    actual = core.canonical_json_sha256(payload)
    if actual != expected:
        raise core.RandomBlockError(
            f"manifest payload hash mismatch: {actual} != {expected}"
        )
    return actual


def check(
    config_path: Path,
    dataset: Path,
    manifest_path: Path,
    audit_path: Path,
    *,
    require_temperature_gate: bool,
) -> dict[str, Any]:
    config = core.load_config(config_path)
    generator._config_contract(config)
    manifest = _load_json(manifest_path)
    audit = _load_json(audit_path)
    dataset_manifest = _load_json(dataset / "manifest.json")
    dataset_audit = _load_json(dataset / "audit.json")
    if manifest != dataset_manifest or audit != dataset_audit:
        raise core.RandomBlockError("tracked and dataset metadata diverge")
    manifest_hash = _manifest_hash(manifest)
    expected_dataset = str(config["dataset_id"])
    for payload, label in (
        (manifest, "manifest"),
        (audit, "audit"),
        (dataset_manifest, "dataset manifest"),
    ):
        if str(payload["dataset_id"]) != expected_dataset:
            raise core.RandomBlockError(f"{label}: dataset ID mismatch")
    if str(manifest["protocol_sha256"]) != str(
        config["provenance"]["protocol_sha256"]
    ):
        raise core.RandomBlockError("manifest/config protocol mismatch")
    if int(manifest["sample_count"]) != int(config["sample_count"]):
        raise core.RandomBlockError("manifest sample count mismatch")
    if len(manifest["samples"]) != int(config["sample_count"]):
        raise core.RandomBlockError("manifest sample list mismatch")
    if int(manifest["solver_mesh"]["node_count"]) != 240825:
        raise core.RandomBlockError("unexpected solver node count")
    if any(bool(value) for value in (
        manifest["guardrails"]["training_runs"],
        manifest["guardrails"]["model_inference_runs"],
        manifest["guardrails"]["temperature_filtered_samples"],
        manifest["guardrails"]["sample_replacements"],
    )):
        raise core.RandomBlockError("forbidden dataset lifecycle activity")

    samples = list(manifest["samples"])
    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise core.RandomBlockError("duplicate manifest sample ID")
    config_cases = {str(row["sample_id"]): row for row in config["cases"]}
    if set(sample_ids) != set(config_cases):
        raise core.RandomBlockError("manifest/config sample set mismatch")
    group_roles: defaultdict[str, set[str]] = defaultdict(set)
    coordinate_hashes: defaultdict[str, set[str]] = defaultdict(set)
    minimum_block_coverage = math.inf
    max_power_error = 0.0
    peak_values: list[float] = []
    realized_bins: Counter[str] = Counter()

    for row in samples:
        sample_id = str(row["sample_id"])
        case = config_cases[sample_id]
        group_id = str(row["group_id"])
        role = str(row["split_role"])
        if role != str(case["split_role"]) or group_id != str(case["group_id"]):
            raise core.RandomBlockError(f"{sample_id}: identity mismatch")
        group_roles[group_id].add(role)
        coordinate_hashes[group_id].add(str(row["point_coordinates_sha256"]))
        sample_dir = dataset / str(row["sample_dir"])
        meta = _load_json(sample_dir / "sample_meta.json")
        if (
            str(meta["sample_id"]) != sample_id
            or str(meta["group_id"]) != group_id
            or str(meta["split_role"]) != role
        ):
            raise core.RandomBlockError(f"{sample_id}: sample metadata mismatch")
        if any(bool(value) for value in meta["guardrails"].values()):
            raise core.RandomBlockError(f"{sample_id}: forbidden sample activity")
        for filename, expected_hash in row["file_sha256"].items():
            actual_hash = core.file_sha256(sample_dir / filename)
            if actual_hash != str(expected_hash):
                raise core.RandomBlockError(
                    f"{sample_id}/{filename}: SHA256 mismatch"
                )
        coords = np.load(sample_dir / "coords.npy", allow_pickle=False)
        temperature = np.load(
            sample_dir / "temperature.npy", allow_pickle=False
        )
        delta = np.load(sample_dir / "deltaT.npy", allow_pickle=False)
        k_field = np.load(sample_dir / "k_field.npy", allow_pickle=False)
        q_field = np.load(sample_dir / "q_field.npy", allow_pickle=False)
        volume = np.load(
            sample_dir / "control_volume.npy", allow_pickle=False
        )
        flags = np.load(sample_dir / "bc_features.npy", allow_pickle=False)
        if not (
            coords.shape == (1024, 3)
            and temperature.shape == (1024, 1)
            and delta.shape == (1024, 1)
            and k_field.shape == (1024, 3)
            and q_field.shape == (1024, 1)
            and volume.shape == (1024, 1)
            and flags.shape == (1024, 4)
        ):
            raise core.RandomBlockError(f"{sample_id}: array shape mismatch")
        if not all(
            np.all(np.isfinite(value))
            for value in (coords, temperature, delta, k_field, q_field, volume)
        ):
            raise core.RandomBlockError(f"{sample_id}: non-finite array")
        if np.min(k_field) <= 0.0 or np.min(q_field) < 0.0 or np.min(volume) <= 0.0:
            raise core.RandomBlockError(f"{sample_id}: invalid physical field")
        if not np.array_equal(temperature - 300.0, delta):
            raise core.RandomBlockError(f"{sample_id}: deltaT mismatch")
        if not np.all(np.sum(flags, axis=1) == 1.0):
            raise core.RandomBlockError(f"{sample_id}: BC flags not one-hot")
        coverage = list(map(int, meta["support"]["block_coverage"]))
        minimum_block_coverage = min(minimum_block_coverage, min(coverage))
        if min(coverage) <= 0:
            raise core.RandomBlockError(f"{sample_id}: zero block coverage")
        block_power = sum(
            float(block["source_power_W"])
            for block in meta["blocks"]
            if str(block["family"]) == "q"
        )
        expected_power = float(meta["package_total_power_W"])
        max_power_error = max(max_power_error, abs(block_power - expected_power))
        if not math.isclose(
            block_power, expected_power, rel_tol=2.0e-12, abs_tol=1.0e-12
        ):
            raise core.RandomBlockError(f"{sample_id}: block power mismatch")
        peak = float(meta["metrics"]["peak_deltaT_K"])
        peak_values.append(peak)
        realized = meta["metrics"]["realized_temperature_bin"]
        realized_bins["outside" if realized is None else str(realized)] += 1

    if any(len(roles) != 1 for roles in group_roles.values()):
        raise core.RandomBlockError("group split leakage")
    if any(len(hashes) != 1 for hashes in coordinate_hashes.values()):
        raise core.RandomBlockError("support varies within a layout group")

    archive_path = dataset / str(manifest["full_field_archive"]["path"])
    archive_sha = core.file_sha256(archive_path)
    if archive_sha != str(manifest["full_field_archive"]["sha256"]):
        raise core.RandomBlockError("full-field archive SHA256 mismatch")
    with h5py.File(archive_path, "r") as archive:
        expected_shape = (int(config["sample_count"]), 240825)
        if archive["temperature_K"].shape != expected_shape:
            raise core.RandomBlockError("temperature archive shape mismatch")
        if archive["q_W_m3"].shape != expected_shape:
            raise core.RandomBlockError("q archive shape mismatch")
        if archive["k_xyz_W_mK"].shape != expected_shape + (3,):
            raise core.RandomBlockError("k archive shape mismatch")
        if archive["coords"].shape != (240825, 3):
            raise core.RandomBlockError("mesh archive shape mismatch")
        sample_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in archive["sample_id"][:]
        ]
        if sample_names != sample_ids:
            raise core.RandomBlockError("archive sample order mismatch")

    temperature_gate = str(audit["status"]) == "passed"
    if str(audit["status"]) not in {
        "passed",
        "failed_temperature_gate",
    }:
        raise core.RandomBlockError(
            f"physical gate failed: {audit['status']}"
        )
    if require_temperature_gate and not temperature_gate:
        raise core.RandomBlockError("temperature gate is not passed")
    return {
        "status": "passed",
        "dataset_id": expected_dataset,
        "stage": config["stage"],
        "sample_count": len(samples),
        "group_count": len(group_roles),
        "split_role_counts": dict(
            sorted(Counter(str(row["split_role"]) for row in samples).items())
        ),
        "group_split_leakage": False,
        "unique_support_per_group": True,
        "minimum_support_nodes_per_block": int(minimum_block_coverage),
        "maximum_block_power_absolute_error_W": max_power_error,
        "peak_deltaT_K": {
            "minimum": min(peak_values),
            "median": float(np.median(peak_values)),
            "maximum": max(peak_values),
        },
        "realized_temperature_bin_counts": dict(sorted(realized_bins.items())),
        "temperature_gate": temperature_gate,
        "manifest_payload_sha256": manifest_hash,
        "full_field_archive_sha256": archive_sha,
        "training_runs": 0,
        "model_inference_runs": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--allow-temperature-gate-failure", action="store_true")
    args = parser.parse_args()
    result = check(
        _resolve(args.config),
        _resolve(args.dataset),
        _resolve(args.manifest),
        _resolve(args.audit),
        require_temperature_gate=not args.allow_temperature_gate_failure,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
