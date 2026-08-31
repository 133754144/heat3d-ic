#!/usr/bin/env python3
"""Freeze the DeepOHeat-v1 volumetric 768/128 supervised subset.

Selection depends only on integer indices and a predeclared RNG contract.  The
script verifies the official 100,000-function input array, hashes the selected
input bytes, and never opens any temperature/reference array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
SEED = 20260831
TRAIN_COUNT = 768
VALID_COUNT = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def selected_input_sha256(source: np.ndarray, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index in indices:
        values = np.ascontiguousarray(source[int(index)], dtype="<f8")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("official fs_train_volume.npy SHA mismatch")
    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    if source.shape != (100000, 101, 101) or source.dtype != np.float64:
        raise ValueError(f"unexpected official train input contract: {source.shape}/{source.dtype}")
    generator = np.random.Generator(np.random.PCG64(SEED))
    permutation = generator.permutation(source.shape[0])
    train = np.sort(permutation[:TRAIN_COUNT].astype(np.int64))
    valid = np.sort(permutation[TRAIN_COUNT : TRAIN_COUNT + VALID_COUNT].astype(np.int64))
    if len(set(train.tolist()) & set(valid.tolist())):
        raise RuntimeError("train/valid overlap")
    payload = {
        "schema_version": "heat3d_v7_g2_deepoheat_v1_volumetric_subset_v1",
        "status": "IMMUTABLE_SELECTION_BEFORE_TEMPERATURE_ACCURACY",
        "upstream": "xlyu0127/DeepOHeat-v1@3ef3d9c41666a56b5940b39a61166ccaa5aaedb2",
        "source": {
            "file": "fs_train_volume.npy",
            "shape": [100000, 101, 101],
            "dtype": "float64",
            "sha256": SOURCE_SHA256,
        },
        "selection": {
            "algorithm": "numpy_Generator_PCG64_permutation_then_first_768_train_next_128_valid_sort_within_role",
            "numpy_version": np.__version__,
            "seed": SEED,
            "accuracy_or_temperature_observed": False,
        },
        "roles": {
            "train": {
                "count": TRAIN_COUNT,
                "indices": train.tolist(),
                "indices_little_endian_int64_sha256": array_sha256(train, "<i8"),
                "selected_inputs_little_endian_float64_sha256": selected_input_sha256(source, train),
            },
            "valid": {
                "count": VALID_COUNT,
                "indices": valid.tolist(),
                "indices_little_endian_int64_sha256": array_sha256(valid, "<i8"),
                "selected_inputs_little_endian_float64_sha256": selected_input_sha256(source, valid),
            },
        },
        "official_test_release": {
            "count": 100,
            "separate_file": "fs_test_volume.npy",
            "untouched_by_selection": True,
            "used_for_training_or_validation": False,
        },
        "temperature_labels_generated": 0,
        "p1i_test_or_sealed_access": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
