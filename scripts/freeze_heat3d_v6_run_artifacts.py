#!/usr/bin/env python3
"""Freeze registered V6_01--V6_04 checkpoint/prediction identities and hashes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "configs/heat3d_v6/v6_training_result_sources.json"
DEFAULT_METRICS = ROOT / "configs/heat3d_v6/v6_training_checkpoint_metrics.csv"
DEFAULT_OUTPUT = ROOT / "configs/heat3d_v6/v6_run_artifact_freeze.json"
EXPECTED_RUNS = {
    "V6_01_V4best",
    "V6_02_V5best",
    "V6_03_V5best_P1h",
    "V6_04_V5best_P1h_DualAttention",
}


def build(sources_path: Path, metrics_path: Path) -> dict[str, Any]:
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    source_runs = {
        str(run["config_id"]): run for run in sources["runs"]
    }
    if set(source_runs) != EXPECTED_RUNS:
        raise RuntimeError("V6 result source run IDs drifted")
    metric_by_key = {
        (row["config_id"], row["checkpoint_kind"]): row for row in metrics
    }
    if len(metric_by_key) != 14:
        raise RuntimeError("expected 14 unique registered checkpoint metrics")
    runs = []
    for config_id in sorted(source_runs):
        source = source_runs[config_id]
        artifacts = []
        for kind, checkpoint in sorted(source["checkpoints"].items()):
            row = metric_by_key.get((config_id, kind))
            if row is None:
                raise RuntimeError(f"{config_id}/{kind}: metric row missing")
            if (
                row["checkpoint_file"] != checkpoint["checkpoint_file"]
                or row["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
                or int(row["checkpoint_epoch"]) != int(checkpoint["epoch"])
                or row["prediction_file"] != checkpoint["prediction_file"]
            ):
                raise RuntimeError(f"{config_id}/{kind}: artifact registry drift")
            if len(row["prediction_sha256"]) != 64:
                raise RuntimeError(f"{config_id}/{kind}: prediction SHA missing")
            artifacts.append(
                {
                    "checkpoint_kind": kind,
                    "epoch": int(checkpoint["epoch"]),
                    "checkpoint_file": checkpoint["checkpoint_file"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "prediction_file": checkpoint["prediction_file"],
                    "prediction_sha256": row["prediction_sha256"],
                }
            )
        runs.append(
            {
                "run_id": config_id,
                "config_id": config_id,
                "dataset_id": source["dataset_id"],
                "source_host": source["source_host"],
                "training_commit": source["training_commit"],
                "remote_run_dir": source["remote_run_dir"],
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
        )
    return {
        "schema_version": "heat3d_v6_run_artifact_freeze_v1",
        "status": "frozen",
        "scope": "V6_01_through_V6_04_registered_checkpoints_and_predictions",
        "immutability_policy": {
            "overwrite_allowed": False,
            "historical_run_directories_mutated": False,
            "checkpoint_selection_changed": False,
            "hash_source": (
                "configs/heat3d_v6/v6_training_checkpoint_metrics.csv"
            ),
        },
        "run_count": len(runs),
        "artifact_count": sum(run["artifact_count"] for run in runs),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build(args.sources, args.metrics)
    if args.write:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_count": payload["run_count"],
                "artifact_count": payload["artifact_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
