"""Frozen P1i input and explicit valid-only evaluation boundaries for G2-A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample

from .inputs import P1IInputBatch


DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
MANIFEST_SHA256 = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
ALLOWED_INPUT_ROLES = frozenset({"train", "valid_iid"})


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


def _manifest_row(manifest: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("P1i manifest SHA does not match the frozen binding")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected P1i dataset ID")
    row = next(
        (item for item in payload.get("samples", []) if item.get("sample_id") == sample_id),
        None,
    )
    if row is None:
        raise ValueError(f"sample ID not present in frozen manifest: {sample_id}")
    return payload, row


def load_frozen_p1i_input_only(
    subset: Path,
    manifest: Path,
    sample_id: str,
) -> tuple[P1IInputBatch, dict[str, object]]:
    """Load only frozen condition inputs; no target file is opened."""

    _payload, row = _manifest_row(manifest, sample_id)
    split = str(row.get("split_role"))
    if split not in ALLOWED_INPUT_ROLES:
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
    features = np.concatenate((k_field, q_field, flags, broadcast), axis=1)
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
            "reference_temperature_K": float(meta["physics"]["ambient_K"]),
            "labels_read": False,
            "temperature_path_opened": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )


def load_frozen_valid_evaluation_sample(
    subset: Path,
    manifest: Path,
    sample_id: str,
) -> EvaluationSample:
    """Read one valid_iid truth row for the external EvaluationCore only."""

    _payload, row = _manifest_row(manifest, sample_id)
    if str(row.get("split_role")) != "valid_iid":
        raise ValueError("G2 Level-A smoke may load truth only for valid_iid")
    sample_dir = _resolve_sample(subset, sample_id)
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    q_field = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float64).reshape(-1)
    control_volume = np.asarray(np.load(sample_dir / "control_volume.npy"), dtype=np.float64).reshape(-1)
    layer_id = np.asarray(np.load(sample_dir / "layer_id.npy"), dtype=np.int32).reshape(-1)
    # Truth is intentionally loaded in this separate EvaluationCore-owned
    # boundary, never by P1IInputBatch or an external model adapter.
    truth_delta = np.asarray(np.load(sample_dir / "deltaT.npy"), dtype=np.float64).reshape(-1)
    return EvaluationSample(
        sample_id=sample_id,
        prediction_deltaT_K=np.zeros_like(truth_delta),
        truth_deltaT_K=truth_delta,
        control_volumes_m3=control_volume,
        coords=coords,
        layer_id=layer_id,
        q_W_m3=q_field,
    ).validated()


def evaluate_valid_prediction(
    truth_sample: EvaluationSample,
    prediction_deltaT_K: Any,
) -> dict[str, Any]:
    """Evaluate one explicit prediction with the frozen Level-A core."""

    row = EvaluationSample(
        sample_id=truth_sample.sample_id,
        prediction_deltaT_K=np.asarray(prediction_deltaT_K, dtype=np.float64).reshape(-1),
        truth_deltaT_K=truth_sample.truth_deltaT_K,
        control_volumes_m3=truth_sample.control_volumes_m3,
        coords=truth_sample.coords,
        layer_id=truth_sample.layer_id,
        q_W_m3=truth_sample.q_W_m3,
        split="valid_iid",
    )
    return EvaluationCore().evaluate([row])


__all__ = [
    "DATASET_ID",
    "MANIFEST_SHA256",
    "evaluate_valid_prediction",
    "load_frozen_p1i_input_only",
    "load_frozen_valid_evaluation_sample",
]
