#!/usr/bin/env python3
"""Run forward-only G2-A upstream adapter smoke checks.

This command deliberately uses a synthetic schema fixture for the upstream
forward call.  It is a software qualification, not a P1i scientific result.
The P1i input adapter is exercised separately with an explicitly supplied
frozen sample path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from rigno.heat3d_g2 import (
    P1IInputBatch,
    build_gino_model,
    build_transolver_model,
)
from rigno.heat3d_g2.adapters import prediction_sha256


def _synthetic_inputs(point_count: int) -> P1IInputBatch:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 1.0, size=(1, point_count, 3)).astype(np.float32)
    features = rng.normal(0.0, 1.0, size=(1, point_count, 11)).astype(np.float32)
    return P1IInputBatch.from_arrays(
        sample_ids=("g2_native_smoke_0000",),
        coords=coords,
        features=features,
        split="train",
    )


def _run_gino(inputs: P1IInputBatch, device: str) -> dict[str, object]:
    adapter = build_gino_model(device=device, latent_resolution=3)
    output = adapter.predict(inputs)
    return {
        "status": "PASS",
        "identity": adapter.upstream_identity,
        "output_shape": list(output.shape),
        "prediction_sha256": prediction_sha256(output),
    }


def _run_transolver(
    inputs: P1IInputBatch,
    device: str,
    upstream_root: Path,
) -> dict[str, object]:
    adapter = build_transolver_model(upstream_root=upstream_root, device=device)
    output = adapter.predict(inputs)
    return {
        "status": "PASS",
        "identity": adapter.upstream_identity,
        "output_shape": list(output.shape),
        "prediction_sha256": prediction_sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gino", "transolver", "both"), default="both")
    parser.add_argument("--point-count", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    if args.point_count < 8:
        raise ValueError("--point-count must be >= 8")

    inputs = _synthetic_inputs(args.point_count)
    result: dict[str, object] = {
        "schema_version": "heat3d_v7_g2_a_adapter_smoke_v1",
        "status": "PASS",
        "execution_role": "compatibility_audit",
        "scientific_evidence_eligible": False,
        "formal_g2": False,
        "dataset_id": inputs.dataset_id,
        "split": inputs.split,
        "labels_read": False,
        "solver_executed": False,
        "training_executed": False,
        "device": args.device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "point_count": inputs.point_count,
        "feature_names": list(inputs.feature_names),
        "backends": {},
    }
    if args.backend in {"gino", "both"}:
        result["backends"]["GINO"] = _run_gino(inputs, args.device)
    if args.backend in {"transolver", "both"}:
        result["backends"]["Transolver"] = _run_transolver(
            inputs, args.device, args.transolver_root.resolve()
        )

    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
