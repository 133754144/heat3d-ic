#!/usr/bin/env python3
"""Materialize a V6 Heat3D valid-only full-field prediction archive.

The input is the frozen V6 point-global-best 1024-node prediction archive.  A
deterministic V6 utility maps those values to the canonical 65x65x57 solver
grid.  No model is loaded, no truth values are read, and only the 128 IDs in
the independent valid-only manifest are touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Running a script by path places ``scripts/`` (rather than the repository
# root) first on sys.path.  Make the frozen repository utility import explicit
# so the remote invocation cannot depend on the caller's PYTHONPATH.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v6_full_field import build_reconstruction_map


GRID = (65, 65, 57)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boundaries(meta: dict[str, Any], minimum_z: float) -> np.ndarray:
    thicknesses = [float(row["thickness_m"]) for row in meta["physics"]["layers_bottom_to_top"]]
    return minimum_z + np.concatenate(([0.0], np.cumsum(thicknesses)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--support-predictions", type=Path, required=True)
    parser.add_argument("--support-predictions-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    required = (args.valid_cases, args.dataset_root, args.full_field_archive, args.support_predictions)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if args.output.exists() or args.receipt.exists():
        raise SystemExit("refusing to overwrite existing replay artifact")
    support_sha = sha256_file(args.support_predictions)
    if support_sha != args.support_predictions_sha256:
        raise SystemExit(f"support prediction SHA mismatch: {support_sha} != {args.support_predictions_sha256}")
    payload = json.loads(args.valid_cases.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if payload.get("role") != "valid_iid" or len(cases) != 128:
        raise ValueError("valid-only manifest contract failed")
    ids = [str(case["sample_id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate valid sample ID")
    support = np.load(args.support_predictions, allow_pickle=False)
    predictions: dict[str, np.ndarray] = {}
    mapping_audits: list[dict[str, Any]] = []
    with h5py.File(args.full_field_archive, "r") as archive:
        full_coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        full_layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
        full_tree = cKDTree(full_coords)
        minimum_z = float(np.min(full_coords[:, 2]))
        for index, sid in enumerate(ids):
            sample_dir = args.dataset_root / sid
            meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
            support_coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
            distance, support_indices = full_tree.query(support_coords, k=1)
            if float(np.max(distance)) > 1e-14:
                raise RuntimeError(f"{sid}: support is not a subset of solver nodes")
            mapping, audit = build_reconstruction_map(
                coords=full_coords,
                layer_id=full_layer,
                boundaries=boundaries(meta, minimum_z),
                support_indices=np.asarray(support_indices, dtype=np.int32),
                empty_domain_fallback="same_layer",
            )
            if index in (0, len(ids) - 1):
                mapping_audits.append({"sample_id": sid, **audit})
            if sid not in support.files:
                raise RuntimeError(f"support prediction archive missing {sid}")
            support_temperature = np.asarray(support[sid], dtype=np.float64).reshape(-1)
            if support_temperature.size != 1024:
                raise RuntimeError(f"{sid}: expected 1024 support predictions, got {support_temperature.size}")
            ambient = float(meta["physics"]["ambient_K"])
            dense_delta_t = mapping.reconstruct(support_temperature - ambient)
            dense_delta_t = np.asarray(dense_delta_t, dtype=np.float32).reshape(GRID)
            if not np.isfinite(dense_delta_t).all():
                raise FloatingPointError(f"{sid}: non-finite dense prediction")
            predictions[sid] = dense_delta_t
    support.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **predictions)
    receipt = {
        "schema_version": "heat3d_v7_g2_p23_heat3d_v6_valid_only_replay_v1",
        "status": "PASS_VALID_ONLY_REPLAY",
        "authorized_role": "valid_iid",
        "valid_count": len(predictions),
        "grid": list(GRID),
        "layout": "x_y_z float32 arrays; value_kind=deltaT_K",
        "value_kind": "deltaT_K",
        "checkpoint_role": "V6 point_global_best frozen support prediction",
        "support_predictions_sha256": support_sha,
        "valid_case_manifest_sha256": sha256_file(args.valid_cases),
        "full_field_archive_sha256": sha256_file(args.full_field_archive),
        "mapping_utility": "rigno.heat3d_v6_full_field.build_reconstruction_map",
        "mapping_audit_examples": mapping_audits,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
        "training_or_optimizer_step": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
