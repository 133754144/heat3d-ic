#!/usr/bin/env python3
"""Audit frozen V7 plotting inputs and materialize one compact visualization fixture.

The script only reads an existing FVM field and frozen prediction archives.  It
does not load a model or run inference.  Therm-FM's native ``z_x_y`` array is
converted once to the canonical ``x_y_z`` C-order used by the P1i full-field
archive; the conversion is recorded in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_index(values: np.ndarray, sample_id: str) -> int:
    ids = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]
    if sample_id not in ids:
        raise KeyError(f"sample_id not found: {sample_id}")
    return ids.index(sample_id)


def metric(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        p, t = pred.reshape(-1), truth.reshape(-1)
    else:
        p, t = pred[mask], truth[mask]
    err = p - t
    return {
        "rmse_K": float(np.sqrt(np.mean(err * err))),
        "mae_K": float(np.mean(np.abs(err))),
        "peak_abs_error_K": float(abs(float(np.max(p)) - float(np.max(t)))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-fields", type=Path, required=True)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--heat3d", type=Path, required=True)
    ap.add_argument("--thermfm", type=Path, required=True)
    ap.add_argument("--gino", type=Path)
    ap.add_argument("--transolver", type=Path)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with h5py.File(args.full_fields, "r") as f:
        idx = sample_index(np.asarray(f["samples/sample_id"]), args.sample_id)
        truth = np.asarray(f["samples/deltaT_K"][idx], dtype=np.float32)
        coords = np.asarray(f["shared/coords_m"], dtype=np.float64)
        layer_id = np.asarray(f["shared/layer_id"], dtype=np.int32)
    native_coords = np.asarray(np.load(args.sample_dir / "coords.npy"), dtype=np.float64)
    native_layer = np.asarray(np.load(args.sample_dir / "layer_id.npy"), dtype=np.int32)
    if native_coords.shape != (1024, 3) or native_layer.shape != (1024,):
        raise ValueError("native P1i sample does not have the frozen 1024-point shape")

    heat = np.load(args.heat3d, allow_pickle=False)
    hids = np.asarray(heat["sample_ids"]).astype(str)
    hi = int(np.flatnonzero(hids == args.sample_id)[0])
    heat_pred = np.asarray(heat["prediction_deltaT_K"][hi], dtype=np.float32)
    therm_raw = np.load(args.thermfm, allow_pickle=False)[args.sample_id]
    if therm_raw.shape != (57, 65, 65):
        raise ValueError(f"unexpected Therm-FM native shape: {therm_raw.shape}")
    # Therm-FM/Poseidon adapter writes z,x,y; common evaluator consumes x,y,z.
    therm_pred = np.transpose(np.asarray(therm_raw, dtype=np.float32), (1, 2, 0)).reshape(-1)

    native = {}
    for name, path in (("GINO", args.gino), ("Transolver", args.transolver)):
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        ids = np.asarray(arr["sample_ids"]).astype(str)
        ni = int(np.flatnonzero(ids == args.sample_id)[0])
        native[name] = np.asarray(arr["prediction_deltaT_K"][ni], dtype=np.float32)
        if native[name].shape != (1024,):
            raise ValueError(f"unexpected {name} native shape: {native[name].shape}")

    # Pick the same fixed, input-independent visual slice rule used by the
    # existing V7 figures: the dense truth peak layer.
    peak = int(np.argmax(truth))
    z = float(coords[peak, 2])
    dense_slice = np.isclose(coords[:, 2], z, atol=1e-12, rtol=0.0)
    slice_stats = {
        "truth": metric(truth[dense_slice], truth[dense_slice]),
        "Heat3D_U_v2": metric(heat_pred[dense_slice], truth[dense_slice]),
        "ThermFM": metric(therm_pred[dense_slice], truth[dense_slice]),
    }
    payload = {
        "sample_id": args.sample_id,
        "split_role": "valid_iid" if args.sample_id.startswith("v6p1if1_") else "visualization_only_unclassified",
        "truth_row": idx,
        "dense_shape": [65, 65, 57],
        "dense_node_count": int(truth.size),
        "dense_layout": "x_y_z C-order; shared coords_m order",
        "native_layout": "per-sample coords.npy order",
        "thermfm_source_layout": "z_x_y (57,65,65)",
        "thermfm_conversion": "transpose(z,x,y -> x,y,z), then C-order flatten",
        "z_slice_m": z,
        "z_slice_index_count": int(dense_slice.sum()),
        "peak_truth_node": peak,
        "slice_stats": slice_stats,
        "inputs": {
            "full_fields": {"path": str(args.full_fields), "sha256": sha256(args.full_fields)},
            "sample_dir": str(args.sample_dir),
            "sample_coords_sha256": sha256(args.sample_dir / "coords.npy"),
            "sample_layer_id_sha256": sha256(args.sample_dir / "layer_id.npy"),
            "heat3d": {"path": str(args.heat3d), "sha256": sha256(args.heat3d)},
            "thermfm": {"path": str(args.thermfm), "sha256": sha256(args.thermfm)},
            **{name: {"path": str(path), "sha256": sha256(path)} for name, path in (("GINO", args.gino), ("Transolver", args.transolver)) if path is not None},
        },
        "prediction_fresh_inference": False,
        "test_iid_read": not args.sample_id.startswith("v6p1if1_"),
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {"coords": coords, "truth": truth, "layer_id": layer_id, "heat3d_u_v2": heat_pred, "thermfm": therm_pred, "native_coords": native_coords, "native_layer_id": native_layer}
    arrays.update({name.lower() + "_native": value for name, value in native.items()})
    np.savez_compressed(args.out / f"{args.sample_id}.npz", **arrays)
    (args.out / f"{args.sample_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
