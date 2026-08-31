#!/usr/bin/env python3
"""Materialize immutable train-only P1i normalization statistics.

Only the frozen ``train`` role is accepted. The script verifies every opened
file against the dataset manifest, never opens ``valid_iid`` targets, and uses
population (ddof=0) global/channel-wise moments that do not depend on node
index correspondence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED = ("coords.npy", "k_field.npy", "q_field.npy", "bc_features.npy", "deltaT.npy")
FEATURE_NAMES = (
    "kx_W_mK",
    "ky_W_mK",
    "kz_W_mK",
    "q_W_m3",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
    "top_h_W_m2K",
    "bottom_h_W_m2K",
    "top_T_inf_minus_T_ref_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    rows = {row["sample_id"]: row for row in dataset["samples"]}
    train_ids = sorted(
        sample_id
        for sample_id, role in split["assignment"].items()
        if role == "train"
    )
    if len(train_ids) != 768:
        raise ValueError(f"expected 768 train samples, got {len(train_ids)}")
    if any("test" in sample_id.lower() or "sealed" in sample_id.lower() for sample_id in train_ids):
        raise ValueError("closed-role token reached train statistics")

    coordinate_min = np.full(3, np.inf, dtype=np.float64)
    coordinate_max = np.full(3, -np.inf, dtype=np.float64)
    feature_sum = np.zeros(11, dtype=np.float64)
    feature_sum_sq = np.zeros(11, dtype=np.float64)
    target_sum = 0.0
    target_sum_sq = 0.0
    point_count = 0
    verified_files = 0

    for sample_id in train_ids:
        directory = args.samples_root / sample_id
        row = rows[sample_id]
        if row["split_role"] != "train":
            raise ValueError(f"manifest role mismatch: {sample_id}")
        for name in REQUIRED:
            path = directory / name
            if sha256(path) != row["file_sha256"][name]:
                raise ValueError(f"SHA mismatch: {sample_id}/{name}")
            verified_files += 1
        coords = np.asarray(np.load(directory / "coords.npy", allow_pickle=False), dtype=np.float64)
        k_field = np.asarray(np.load(directory / "k_field.npy", allow_pickle=False), dtype=np.float64)
        q_field = np.asarray(np.load(directory / "q_field.npy", allow_pickle=False), dtype=np.float64).reshape(-1, 1)
        bc = np.asarray(np.load(directory / "bc_features.npy", allow_pickle=False), dtype=np.float64)
        target = np.asarray(np.load(directory / "deltaT.npy", allow_pickle=False), dtype=np.float64).reshape(-1, 1)
        if coords.shape != (1024, 3) or k_field.shape != (1024, 3):
            raise ValueError(f"unexpected point layout: {sample_id}")
        if bc.shape == (1024, 4):
            metadata_path = directory / "sample_meta.json"
            expected = row["file_sha256"]["sample_meta.json"]
            if sha256(metadata_path) != expected:
                raise ValueError(f"SHA mismatch: {sample_id}/sample_meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bc = np.column_stack(
                (
                    bc,
                    np.full(1024, metadata["top_h_W_m2K"]),
                    np.full(1024, metadata["bottom_h_W_m2K"]),
                    np.zeros(1024),
                )
            )
            verified_files += 1
        if bc.shape != (1024, 7) or q_field.shape != (1024, 1) or target.shape != (1024, 1):
            raise ValueError(f"unexpected field layout: {sample_id}")
        features = np.concatenate((k_field, q_field, bc), axis=1)
        coordinate_min = np.minimum(coordinate_min, coords.min(axis=0))
        coordinate_max = np.maximum(coordinate_max, coords.max(axis=0))
        feature_sum += features.sum(axis=0)
        feature_sum_sq += np.square(features).sum(axis=0)
        target_sum += float(target.sum())
        target_sum_sq += float(np.square(target).sum())
        point_count += coords.shape[0]

    feature_mean = feature_sum / point_count
    feature_var = np.maximum(feature_sum_sq / point_count - np.square(feature_mean), 0.0)
    target_mean = target_sum / point_count
    target_var = max(target_sum_sq / point_count - target_mean**2, 0.0)
    arrays = {
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "feature_mean": feature_mean,
        "feature_std": np.sqrt(feature_var),
        "target_mean": np.asarray([target_mean]),
        "target_std": np.asarray([np.sqrt(target_var)]),
    }
    core: dict[str, Any] = {
        "schema_version": "heat3d_v7_g2_p1i_train_statistics_v1",
        "dataset_id": dataset["dataset_id"],
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "fit_role": "train_only",
        "sample_count": len(train_ids),
        "point_count": point_count,
        "verified_file_count": verified_files,
        "feature_names": list(FEATURE_NAMES),
        "moment_contract": "global_channelwise_population_ddof0_float64_accumulation",
        "statistics": {name: value.tolist() for name, value in arrays.items()},
        "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
        "valid_or_test_statistics_opened": False,
        "formal_accuracy_observed": False,
    }
    core["payload_sha256"] = json_sha256(core)
    args.output.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(core, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
