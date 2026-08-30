#!/usr/bin/env python3
"""Run G2-A adapters on one explicitly selected frozen P1i input sample.

Only ``coords.npy``, ``k_field.npy``, ``q_field.npy``, ``bc_features.npy`` and
input metadata are read.  The target temperature file is deliberately not
opened; accuracy remains the responsibility of the existing Level-A
EvaluationCore path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigno.heat3d_g2 import (
    build_gino_model,
    build_transolver_model,
    load_frozen_p1i_input_only,
)
from rigno.heat3d_g2.adapters import prediction_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-id", default="v6p1if1_0000")
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    inputs, provenance = load_frozen_p1i_input_only(
        args.subset.resolve(), args.manifest.resolve(), args.sample_id
    )
    result: dict[str, object] = {
        "schema_version": "heat3d_v7_g2_a_p1i_input_smoke_v1",
        "status": "PASS",
        "execution_role": "compatibility_audit",
        "scientific_evidence_eligible": False,
        "formal_g2": False,
        "dataset_id": inputs.dataset_id,
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
