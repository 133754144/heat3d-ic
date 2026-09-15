#!/usr/bin/env python3
"""Build the immutable DeepOHeat full-pool-minus-valid128 manifest.

Only source shape/bytes and the already frozen subset manifest are consulted;
no temperature values, predictions, accuracy, or test files are read.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


SUBSET_SHA = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
SOURCE_SHA = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(manifest: dict, role: str) -> np.ndarray:
    row = manifest["roles"][role]
    values = np.frombuffer(base64.b64decode(row["indices_base64"]), dtype="<u4").astype(np.int64)
    if len(values) != int(row["count"]):
        raise ValueError(f"{role} count mismatch")
    return values


def int64_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.subset_manifest) != SUBSET_SHA:
        raise SystemExit("frozen subset manifest SHA mismatch")
    if args.fs_train.name != "fs_train_volume.npy" or sha256(args.fs_train) != SOURCE_SHA:
        raise SystemExit("official fs_train_volume.npy SHA/name mismatch")
    subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if subset.get("selection", {}).get("accuracy_or_temperature_observed") is not False:
        raise SystemExit("subset selection was not accuracy-blind")
    valid = decode(subset, "valid")
    frozen_train = decode(subset, "train")
    source = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    if source.shape != (100000, 101, 101) or source.dtype != np.float64:
        raise SystemExit(f"unexpected source shape/dtype: {source.shape}/{source.dtype}")
    pool = np.setdiff1d(np.arange(source.shape[0], dtype=np.int64), valid)
    duplicate_valid = int(len(valid) - np.unique(valid).size)
    duplicate_pool = int(len(pool) - np.unique(pool).size)
    overlap = np.intersect1d(pool, valid)
    payload = {
        "schema_version": "heat3d_v7_g2_p17_deepoheat_full_minus_valid128_exclusion_manifest_v1",
        "status": "FROZEN_EXCLUSION_MANIFEST",
        "selection_policy": "exclude exactly frozen Heat3D valid128 source IDs; no target/accuracy/test access",
        "source": {
            "file": args.fs_train.name,
            "sha256": SOURCE_SHA,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "original_pool_size": int(source.shape[0]),
        },
        "subset_manifest": {
            "path": str(args.subset_manifest),
            "sha256": SUBSET_SHA,
            "heat3d_train_count": int(len(frozen_train)),
            "heat3d_valid_count": int(len(valid)),
            "valid_indices_int64_sha256": int64_sha(valid),
        },
        "excluded_valid": {
            "count": int(len(valid)),
            "source_indices": valid.tolist(),
            "source_indices_int64_sha256": int64_sha(valid),
            "role": "Heat3D valid128; validation only",
        },
        "training_pool": {
            "count": int(len(pool)),
            "included_source_indices": pool.tolist(),
            "included_source_indices_int64_sha256": int64_sha(pool),
            "construction": "np.setdiff1d(np.arange(original_pool_size), excluded_valid.source_indices)",
        },
        "overlap_audit": {
            "excluded_valid_duplicate_count": duplicate_valid,
            "included_pool_duplicate_count": duplicate_pool,
            "included_vs_excluded_overlap_count": int(len(overlap)),
            "heat3d_train_vs_excluded_valid_overlap_count": int(len(np.intersect1d(frozen_train, valid))),
            "required_zero_overlap": True,
        },
        "roles": {"training": "full pool minus valid128", "validation": "excluded valid128", "test": "sealed"},
        "temperature_or_accuracy_used": False,
        "test_or_sealed_access": False,
    }
    if duplicate_valid or duplicate_pool or len(overlap) or np.intersect1d(frozen_train, valid).size:
        raise SystemExit("exclusion overlap/duplicate audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "source_pool": source.shape[0], "excluded": len(valid), "training_pool": len(pool), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
