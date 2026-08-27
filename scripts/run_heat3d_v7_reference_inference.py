#!/usr/bin/env python3
"""V7 reference inference entrypoint for the frozen V6/P1i native-1024 route.

This entrypoint is deliberately inference-only.  It materializes only the
``valid_iid`` role, calls ``rigno.heat3d_runtime.RuntimeSession``, and prints a
provenance/count summary.  It does not calculate metrics, invoke an FVM solver,
write an output artifact, or import another script module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from rigno.heat3d_runtime import RuntimeSession
from rigno.heat3d_v6_dataset import (
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("valid_iid",),
        default="valid_iid",
        help="The only role permitted by the V7 reference inference entrypoint.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split != "valid_iid":
        raise ValueError("V7 reference inference is restricted to valid_iid")
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root,
        args.manifest,
        include_roles={"valid_iid"},
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise ValueError("reference inference requires frozen V6/P1i continuous_physics1024_v1")
    examples = list(dataset.samples)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        examples = examples[: int(args.max_samples)]
    if not examples:
        raise ValueError("valid_iid fixture contains no examples")
    session = RuntimeSession.from_paths(
        args.checkpoint,
        args.run_config,
        execution_role="production_inference",
    )
    predictions = session.predict_native_1024(examples, batch_size=int(args.batch_size))
    node_counts = sorted(
        {int(np.asarray(row["raw_temperature"]).shape[0]) for row in predictions.values()}
    )
    return {
        "status": "inference_complete",
        "execution_role": "production_inference",
        "experiment_role": "reference_inference",
        "dataset_role": "valid_iid",
        "dataset_id": dataset.manifest["dataset_id"],
        "sample_count": len(predictions),
        "native_node_counts": node_counts,
        "checkpoint": session.checkpoint.descriptor(),
        "runtime": session.descriptor(),
        "labels_used_for_model_input": False,
        "metrics_or_solver_invoked": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
