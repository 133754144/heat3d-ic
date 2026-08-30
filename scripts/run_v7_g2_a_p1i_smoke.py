#!/usr/bin/env python3
"""Run G2-A adapters on one explicitly selected frozen P1i input sample.

Only ``coords.npy``, ``k_field.npy``, ``q_field.npy``, ``bc_features.npy`` and
input metadata are read.  The target temperature file is deliberately not
opened; accuracy remains the responsibility of the existing Level-A
EvaluationCore path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from rigno.heat3d_g2 import P1IInputBatch, build_gino_model, build_transolver_model
from rigno.heat3d_g2.adapters import prediction_sha256


DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
MANIFEST_SHA256 = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
ALLOWED_ROLES = {"train", "valid_iid"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_sample(root: Path, sample_id: str) -> Path:
    candidates = (root / "samples" / sample_id, root / sample_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"P1i sample directory not found for {sample_id!r}")


def load_input_only(subset: Path, manifest: Path, sample_id: str) -> tuple[P1IInputBatch, dict[str, object]]:
    if _sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("P1i manifest SHA does not match the frozen binding")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected P1i dataset ID")
    row = next((item for item in payload.get("samples", []) if item.get("sample_id") == sample_id), None)
    if row is None:
        raise ValueError(f"sample ID not present in frozen manifest: {sample_id}")
    split = str(row.get("split_role"))
    if split not in ALLOWED_ROLES:
        raise ValueError(f"sample role is not allowed for input-only smoke: {split}")
    sample_dir = _resolve_sample(subset, sample_id)
    meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
    if meta.get("dataset_id") != DATASET_ID:
        raise ValueError("sample metadata dataset ID drifted")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float32)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float32)
    q_field = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float32).reshape(-1, 1)
    bc = np.asarray(np.load(sample_dir / "bc_features.npy"), dtype=np.float32)
    if coords.shape != (1024, 3) or k_field.shape != (1024, 3) or q_field.shape != (1024, 1):
        raise ValueError("frozen P1i input shape drifted")
    if bc.shape not in {(1024, 4), (1024, 7)}:
        raise ValueError("frozen P1i BC feature shape drifted")
    ambient = float(meta["physics"]["ambient_K"])
    flags = bc[:, :4]
    broadcast = (
        bc[:, 4:7]
        if bc.shape[1] == 7
        else np.column_stack(
            (
                np.full(1024, float(meta["top_h_W_m2K"]), dtype=np.float32),
                np.full(1024, float(meta["bottom_h_W_m2K"]), dtype=np.float32),
                np.zeros(1024, dtype=np.float32),
            )
        )
    )
    features = np.concatenate(
        (
            k_field,
            q_field,
            flags,
            broadcast,
        ),
        axis=1,
    )
    return (
        P1IInputBatch(
            sample_ids=(sample_id,),
            coords=coords[None, ...],
            features=features[None, ...],
            split=split,
        ),
        {
            "sample_id": sample_id,
            "split": split,
            "reference_temperature_K": ambient,
            "labels_read": False,
            "temperature_path_opened": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-id", default="v6p1if1_0000")
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    inputs, provenance = load_input_only(args.subset.resolve(), args.manifest.resolve(), args.sample_id)
    result: dict[str, object] = {
        "schema_version": "heat3d_v7_g2_a_p1i_input_smoke_v1",
        "status": "PASS",
        "execution_role": "compatibility_audit",
        "scientific_evidence_eligible": False,
        "formal_g2": False,
        "dataset_id": DATASET_ID,
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "input_contract": {
            "sample_count": inputs.batch_size,
            "point_count": inputs.point_count,
            "feature_names": list(inputs.feature_names),
            "normalization": "raw V6/P1i condition features; no hidden adapter normalization",
        },
        "provenance": provenance,
        "backends": {},
        "training_executed": False,
        "solver_executed": False,
    }
    try:
        gino = build_gino_model(device=args.device, latent_resolution=3)
        prediction = gino.predict(inputs)
        result["backends"]["GINO"] = {
            "status": "PASS",
            "identity": gino.upstream_identity,
            "output_shape": list(prediction.shape),
            "prediction_sha256": prediction_sha256(prediction),
        }
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        result["backends"]["GINO"] = {
            "status": "BLOCKED_MISSING_OPTIONAL_DEPENDENCY",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    transolver = build_transolver_model(
        upstream_root=args.transolver_root.resolve(), device=args.device
    )
    prediction = transolver.predict(inputs)
    result["backends"]["Transolver"] = {
        "status": "PASS",
        "identity": transolver.upstream_identity,
        "output_shape": list(prediction.shape),
        "prediction_sha256": prediction_sha256(prediction),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
