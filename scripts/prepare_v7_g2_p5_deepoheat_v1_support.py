#!/usr/bin/env python3
"""Freeze label-independent 1024-point support for DeepOHeat-v1 volume cases.

The selector adapts the frozen Heat3D/G1 physics-layout-aware quota semantics
to the released regular 101x101x56 benchmark mesh.  It sees only source
occupancy/layout, the fixed material interface, coordinates, boundaries, and
control-volume weights.  Source amplitudes, temperatures, predictions, errors,
and official test data are prohibited inputs.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage


SELECTION_SEED = 20260831
QUOTAS = {"block": 256, "interface": 128, "top": 64, "bottom": 64, "volume": 512}
MESH_SHAPE = (101, 101, 56)
SOURCE_Z_INDICES = tuple(range(10, 16))


def canonical_array_sha256(value: Any, dtype: str | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def mesh_arrays() -> dict[str, np.ndarray]:
    axes = (np.linspace(0.0, 1.0, 101), np.linspace(0.0, 1.0, 101), np.linspace(0.0, 0.55, 56))
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    one_d = []
    for axis in axes:
        spacing = float(axis[1] - axis[0])
        weight = np.full(len(axis), spacing, dtype=np.float64)
        weight[[0, -1]] *= 0.5
        one_d.append(weight)
    cv = np.einsum("i,j,k->ijk", *one_d).reshape(-1)
    layer_id = np.where(coords[:, 2] < 0.1 - 1.0e-15, 0, 1).astype(np.int32)
    return {"coords": coords, "control_volume": cv, "layer_id": layer_id}


def _weighted_choice(
    rng: np.random.Generator, candidates: np.ndarray, count: int, weights: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if count < 0 or candidates.size < count:
        raise ValueError(f"candidate shortage: {candidates.size} < {count}")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    probability = np.asarray(weights[candidates], dtype=np.float64)
    probability /= probability.sum()
    return np.asarray(rng.choice(candidates, size=count, replace=False, p=probability), dtype=np.int64)


def source_layout_masks(power: np.ndarray) -> list[np.ndarray]:
    occupancy = np.asarray(power) != 0
    if occupancy.shape != (101, 101) or not np.any(occupancy):
        raise ValueError("expected a nonempty 101x101 source layout")
    component, count = ndimage.label(occupancy, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    masks: list[np.ndarray] = []
    for component_id in range(1, count + 1):
        xy = component == component_id
        volume = np.zeros(MESH_SHAPE, dtype=bool)
        volume[:, :, SOURCE_Z_INDICES] = xy[:, :, None]
        masks.append(volume.reshape(-1))
    if not masks:
        raise ValueError("source component labeling returned no blocks")
    return masks


def select_support(power: np.ndarray, *, source_index: int, role: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    mesh = mesh_arrays()
    coords, cv = mesh["coords"], mesh["control_volume"]
    digest = hashlib.sha256(
        f"{SELECTION_SEED}:deepoheat_v1_volume:{role}:{source_index}:support_v1".encode()
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    selected: list[int] = []
    strata: list[str] = []
    used: set[int] = set()

    def add(values: Sequence[int], label: str) -> None:
        for value in values:
            node = int(value)
            if node not in used:
                selected.append(node); strata.append(label); used.add(node)

    masks = source_layout_masks(power)
    for mask in masks:
        candidates = np.flatnonzero(mask)
        add([int(candidates[rng.integers(len(candidates))])], "block")
    block_pool = np.flatnonzero(np.logical_or.reduce(masks))
    available = np.asarray([v for v in block_pool if int(v) not in used], dtype=np.int64)
    add(_weighted_choice(rng, available, QUOTAS["block"] - strata.count("block"), cv), "block")

    interface_pool = np.flatnonzero(np.isclose(coords[:, 2], 0.1, atol=1.0e-15))
    available = np.asarray([v for v in interface_pool if int(v) not in used], dtype=np.int64)
    add([int(available[rng.integers(len(available))])], "interface")
    available = np.asarray([v for v in interface_pool if int(v) not in used], dtype=np.int64)
    add(_weighted_choice(rng, available, QUOTAS["interface"] - strata.count("interface"), cv), "interface")

    for label, z_value in (("top", 0.55), ("bottom", 0.0)):
        candidates = np.flatnonzero(np.isclose(coords[:, 2], z_value, atol=1.0e-15))
        available = np.asarray([v for v in candidates if int(v) not in used], dtype=np.int64)
        add(_weighted_choice(rng, available, QUOTAS[label], cv), label)

    available_mask = np.ones(len(coords), dtype=bool)
    available_mask[np.fromiter(used, dtype=np.int64, count=len(used))] = False
    available = np.flatnonzero(available_mask)
    add(_weighted_choice(rng, available, QUOTAS["volume"], cv), "volume")
    counts = Counter(strata)
    if len(selected) != 1024 or counts != Counter(QUOTAS):
        raise AssertionError(f"support quota invariant failed: {counts}")
    indices = np.asarray(selected, dtype=np.int64)
    coverage = [int(np.sum(mask[indices])) for mask in masks]
    if min(coverage) <= 0:
        raise AssertionError("support missed a released source-layout component")
    audit = {
        "algorithm_seed_digest": digest,
        "source_component_count": len(masks),
        "source_component_support_counts": coverage,
        "source_occupancy_sha256": canonical_array_sha256(np.asarray(power) != 0),
        "support_indices_sha256": canonical_array_sha256(indices, "<i8"),
        "support_coords_sha256": canonical_array_sha256(coords[indices], "<f8"),
        "strata_sequence_sha256": hashlib.sha256("\n".join(strata).encode()).hexdigest(),
        "temperature_prediction_error_or_test_used": False,
    }
    return indices, strata, audit


def decode_indices(payload: dict[str, Any], role: str) -> np.ndarray:
    row = payload["roles"][role]
    values = np.frombuffer(base64.b64decode(row["indices_base64"]), dtype="<u4").astype(np.int64)
    if len(values) != int(row["count"]):
        raise ValueError(f"{role} subset count mismatch")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if subset["selection"]["accuracy_or_temperature_observed"] is not False:
        raise ValueError("subset selection must be temperature/accuracy blind")
    fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    rows = []
    for role in ("train", "valid"):
        for index in decode_indices(subset, role):
            power = np.asarray(fs_train[int(index)])
            support, _strata, audit = select_support(power, source_index=int(index), role=role)
            rows.append({"role": role, "source_index": int(index), **audit})
    row_hashes = [hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for row in rows]
    receipt = {
        "schema_version": "heat3d_v7_g2_p5_deepoheat_v1_support_rows_v1",
        "status": "FROZEN_LABEL_INDEPENDENT",
        "selection_seed": SELECTION_SEED,
        "quotas": QUOTAS,
        "roles": {"train": 768, "valid": 128},
        "rows": rows,
        "ordered_row_hashes_sha256": hashlib.sha256("\n".join(row_hashes).encode()).hexdigest(),
        "fs_train_file_sha256": file_sha256(args.fs_train),
        "official_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
