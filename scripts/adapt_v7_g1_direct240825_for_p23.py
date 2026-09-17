#!/usr/bin/env python3
"""Adapt a frozen G1 U-v2 direct240825 archive to the P23 NPZ-key contract.

This is a lossless container adapter only.  It never reads truth/labels, does
not reorder nodes or cases, and does not perform inference.  The P23 evaluator
expects each authorized sample id to be an NPZ key, whereas the archived G1
export stores ``sample_ids`` and a stacked ``prediction_deltaT_K`` array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NODE_COUNT = 65 * 65 * 57


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_valid_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or len(cases) != 128:
        raise ValueError("P23 valid manifest must contain exactly 128 cases")
    ids = [str(case["sample_id"]) for case in cases]
    if len(set(ids)) != len(ids) or any(case.get("split_role") != "valid_iid" for case in cases):
        raise ValueError("valid manifest has duplicate ids or non-valid_iid rows")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    valid_ids = load_valid_ids(args.valid_manifest)
    with np.load(args.source, allow_pickle=False) as source:
        required = {"sample_ids", "prediction_deltaT_K", "split", "domain_id", "point_count"}
        if set(source.files) != required:
            raise ValueError(f"unexpected source keys: {source.files}")
        sample_ids = [str(value) for value in source["sample_ids"].tolist()]
        predictions = np.asarray(source["prediction_deltaT_K"])
        split = str(source["split"].item())
        domain_id = str(source["domain_id"].item())
        point_count = int(source["point_count"].item())

    if split != "valid_iid" or domain_id != "heat3d_v6_p1i_full_field_240825":
        raise ValueError("source split/domain is not the frozen P23 contract")
    if point_count != NODE_COUNT or predictions.shape != (128, NODE_COUNT):
        raise ValueError(f"source prediction shape/point_count mismatch: {predictions.shape}, {point_count}")
    if sample_ids != valid_ids:
        raise ValueError("source sample id order does not exactly match frozen valid manifest")
    if predictions.dtype.kind not in "fiu" or not np.isfinite(predictions).all():
        raise ValueError("source predictions must be finite numeric values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    named = {sample_id: np.asarray(predictions[index], dtype=np.float32) for index, sample_id in enumerate(sample_ids)}
    np.savez(args.output, **named)
    output_sha = sha256_file(args.output)
    receipt = {
        "schema_version": "heat3d_v7_g2_p23_g1_direct240825_adapter_v1",
        "status": "PASS_LOSSLESS_CONTAINER_ADAPTER",
        "source_sha256": sha256_file(args.source),
        "output_sha256": output_sha,
        "valid_manifest_sha256": sha256_file(args.valid_manifest),
        "sample_count": len(sample_ids),
        "point_count_per_sample": NODE_COUNT,
        "source_layout": "sample_ids + prediction_deltaT_K[128,240825]",
        "evaluator_layout": "NPZ key=sample_id, flat x_y_z canonical order",
        "value_kind": "deltaT_K",
        "lossless": True,
        "truth_or_label_access": False,
        "inference_or_training": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
