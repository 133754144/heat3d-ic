#!/usr/bin/env python3
"""Check fixed-edge dummy padding preserves a frozen qualification forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import benchmark_heat3d_v6_inference_qualification as bench


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("p1i", "randomblock"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--randomblock-config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--edge-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = bench.FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    row = data.selected_rows(32)[0]
    example, _ = data.load_example(row)
    raw = bench.ModelRuntime(args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, None)
    padded = bench.ModelRuntime(args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, args.edge_targets)
    raw_prediction = raw.forward(raw.graph(example))
    padded_prediction = padded.forward(padded.graph(example))
    error = np.asarray(padded_prediction) - np.asarray(raw_prediction)
    payload = {
        "schema_version": "heat3d_v6_qualification_edge_padding_equivalence_v1",
        "status": "passed",
        "family": args.family, "sample_id": row["sample_id"],
        "edge_targets_sha256": bench.sha256(args.edge_targets),
        "padding_semantics": "repeat_existing_dummy_edge_only",
        "max_abs_error_K": float(np.max(np.abs(error))),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_tolerance_K": 0.01,
        "rmse_tolerance_K": 0.002,
        "test_accessed": False, "sealed_accessed": False, "training_executed": False,
    }
    if (
        payload["max_abs_error_K"] > payload["max_abs_tolerance_K"]
        or payload["rmse_K"] > payload["rmse_tolerance_K"]
    ):
        payload["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "passed":
        raise RuntimeError(payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
