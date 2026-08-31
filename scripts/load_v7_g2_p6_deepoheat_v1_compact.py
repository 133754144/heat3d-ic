#!/usr/bin/env python3
"""Verified compact loader for Heat3D-on-DeepOHeat-v1 formal training.

Train rows expose only the frozen 1024 support, eleven physical features, and
1024 supervised targets. Valid rows additionally expose the frozen complete
571256-point reference field (and its 10201-point source slice). The official
test files are not accepted as arguments or discovered by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FS_TRAIN_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
EXPANDED_RECEIPT_SHA256 = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
NORMALIZATION_PAYLOAD_SHA256 = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class CompactDeepOHeatV1Dataset:
    def __init__(self, *, fs_train: Path, labels_root: Path, role: str, verify_source_file: bool = True):
        if role not in {"train", "valid"}:
            raise ValueError("compact formal loader permits only train or valid")
        if fs_train.name != "fs_train_volume.npy":
            raise ValueError("only official fs_train_volume.npy is accepted")
        if verify_source_file and file_sha256(fs_train) != FS_TRAIN_SHA256:
            raise ValueError("official training input pool SHA mismatch")
        receipt_path = labels_root / "label_generation_receipt.json"
        if file_sha256(receipt_path) != EXPANDED_RECEIPT_SHA256:
            raise ValueError("expanded label receipt SHA mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.rows = [row for row in receipt["rows"] if row["role"] == role]
        expected = 768 if role == "train" else 128
        if len(self.rows) != expected:
            raise ValueError(f"{role} row count mismatch")
        self.fs_train = np.load(fs_train, mmap_mode="r", allow_pickle=False)
        if self.fs_train.shape != (100000, 101, 101):
            raise ValueError("official training input pool shape mismatch")
        self.labels_root, self.role = labels_root, role
        self.converter = load_script("convert_v7_g2_semiconductor_case.py")

    def __len__(self) -> int:
        return len(self.rows)

    def _artifact(self, directory: Path, row: dict[str, Any], key: str) -> np.ndarray:
        metadata = row["artifacts"][key]
        value = np.load(directory / metadata["file"], mmap_mode="r", allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            raise ValueError(f"artifact shape/dtype drift: {row['sample_id']}/{key}")
        if array_sha256(value) != metadata["sha256"]:
            raise ValueError(f"artifact SHA drift: {row['sample_id']}/{key}")
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_index = int(row["source_index"])
        power = np.asarray(self.fs_train[source_index])
        if array_sha256(power) != row["source_input_sha256"]:
            raise ValueError(f"source input SHA drift: {row['sample_id']}")
        directory = self.labels_root / self.role / row["sample_id"]
        support = np.asarray(self._artifact(directory, row, "support_indices"), dtype=np.int64)
        target = np.asarray(self._artifact(directory, row, "support_target"), dtype=np.float32)
        arrays = self.converter.volume_v1_arrays(power)
        result = {
            "sample_id": row["sample_id"],
            "source_index": source_index,
            "coords": np.asarray(arrays["coords"])[support].astype(np.float32),
            "features": np.asarray(arrays["features"])[support].astype(np.float32),
            "support_indices": support,
            "target_1024": target,
        }
        if result["coords"].shape != (1024, 3) or result["features"].shape != (1024, 11) or target.shape != (1024,):
            raise ValueError("compact row schema mismatch")
        if self.role == "valid":
            result["target_full_571256"] = self._artifact(directory, row, "full_reference")
            result["target_slice_10201"] = self._artifact(directory, row, "u_slice_z015")
        return result


def load_normalization(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual or actual != NORMALIZATION_PAYLOAD_SHA256:
        raise ValueError("train-only normalization payload mismatch")
    if payload["valid_or_test_used_to_fit"] is not False:
        raise ValueError("normalization is not train-only")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    args = parser.parse_args()
    normalization = load_normalization(args.normalization)
    train = CompactDeepOHeatV1Dataset(fs_train=args.fs_train, labels_root=args.labels_root, role="train")
    valid = CompactDeepOHeatV1Dataset(fs_train=args.fs_train, labels_root=args.labels_root, role="valid", verify_source_file=False)
    train_row, valid_row = train[0], valid[0]
    receipt = {
        "status": "PASS_COMPACT_LOADER_SCHEMA_AND_SHA",
        "roles": {"train": len(train), "valid": len(valid)},
        "train_keys": sorted(train_row), "valid_keys": sorted(valid_row),
        "train_shapes": {key: list(value.shape) for key, value in train_row.items() if isinstance(value, np.ndarray)},
        "valid_shapes": {key: list(value.shape) for key, value in valid_row.items() if isinstance(value, np.ndarray)},
        "normalization_payload_sha256": NORMALIZATION_PAYLOAD_SHA256,
        "test_or_sealed_access": False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
