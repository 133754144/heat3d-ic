#!/usr/bin/env python3
"""Run external adapters plus one explicit valid-only Level-A evaluator smoke.

The adapter phase remains label-free.  This command loads one frozen
``valid_iid`` truth row only after the forward call and sends the explicit
prediction/truth pair to the existing EvaluationCore.  The resulting metrics
are diagnostic interface evidence only, never publication evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigno.heat3d_g2 import (
    build_gino_model,
    build_transolver_model,
    evaluate_valid_prediction,
    load_frozen_p1i_input_only,
    load_frozen_valid_evaluation_sample,
)
from rigno.heat3d_g2.adapters import prediction_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-id", default="v6p1if1_0003")
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    inputs, input_provenance = load_frozen_p1i_input_only(
        args.subset.resolve(), args.manifest.resolve(), args.sample_id
    )
    if inputs.split != "valid_iid":
        raise ValueError("Level-A smoke requires one valid_iid input")
    truth_sample = load_frozen_valid_evaluation_sample(
        args.subset.resolve(), args.manifest.resolve(), args.sample_id
    )
    result: dict[str, object] = {
        "schema_version": "heat3d_v7_g2_a_p1i_level_a_smoke_v1",
        "status": "PASS",
        "execution_role": "compatibility_audit",
        "scientific_evidence_eligible": False,
        "formal_g2": False,
        "dataset_id": inputs.dataset_id,
        "sample_id": args.sample_id,
        "split": inputs.split,
        "input_provenance": input_provenance,
        "evaluation_provenance": {
            "truth_source": "frozen valid_iid deltaT.npy",
            "labels_read_by_adapter": False,
            "labels_read_by_evaluation_core": True,
            "evaluation_core": "rigno.heat3d_runtime.evaluation.EvaluationCore",
            "prediction_representation": "deltaT_K (explicit diagnostic adapter contract)",
            "accuracy_claim": False,
        },
        "backends": {},
        "training_executed": False,
        "solver_executed": False,
        "test_iid_access": False,
        "sealed_access": False,
    }

    gino = build_gino_model(device=args.device, latent_resolution=3)
    gino_prediction = gino.predict(inputs)
    result["backends"]["GINO"] = {
        "status": "PASS",
        "identity": gino.upstream_identity,
        "output_shape": list(gino_prediction.shape),
        "prediction_sha256": prediction_sha256(gino_prediction),
        "level_a": evaluate_valid_prediction(truth_sample, gino_prediction),
    }

    transolver = build_transolver_model(
        upstream_root=args.transolver_root.resolve(), device=args.device
    )
    transolver_prediction = transolver.predict(inputs)
    result["backends"]["Transolver"] = {
        "status": "PASS",
        "identity": transolver.upstream_identity,
        "output_shape": list(transolver_prediction.shape),
        "prediction_sha256": prediction_sha256(transolver_prediction),
        "level_a": evaluate_valid_prediction(truth_sample, transolver_prediction),
    }

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
