"""Smoke-check the detailed Heat3D v2 timing profile schema."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


REQUIRED_EPOCH_FIELDS = {
    "epoch_index",
    "epoch_total_time",
    "train_total_time",
    "validation_total_time",
    "num_train_batches",
    "num_valid_batches",
    "mean_train_batch_time",
    "median_train_batch_time",
    "max_train_batch_time",
    "first_train_batch_time",
    "later_train_batch_median_time",
    "possible_recompile_batch_count",
}
REQUIRED_BATCH_FIELDS = {
    "epoch_index",
    "batch_index",
    "batch_size",
    "group_count",
    "total_batch_time",
    "loss_grad_time",
    "grad_norm_time",
    "optimizer_update_time",
    "other_time",
    "batch_shape_signature",
}


def main() -> int:
    train_batches = [
        _batch_record(1, 1, 10.0, "(4,1,2048,3)"),
        _batch_record(1, 2, 1.0, "(4,1,2048,3)"),
        _batch_record(1, 3, 1.0, "(4,1,2048,3)"),
        _batch_record(1, 4, 4.1, "(2,1,2048,3)"),
    ]
    summary = runner._summarize_batch_records(train_batches)
    if summary["possible_recompile_batch_count"] != 1:
        raise AssertionError("expected one later-batch possible recompile")
    if train_batches[0]["possible_recompile"]:
        raise AssertionError("first batch should not be marked as recompile")
    if not train_batches[-1]["possible_recompile"]:
        raise AssertionError("slow later batch should be marked as possible recompile")

    epoch_record = {
        "epoch": 1,
        "epoch_total_time_s": 18.0,
        "epoch_train_time_s": 16.1,
        "epoch_train_metrics_time_s": 0.7,
        "epoch_validation_time_s": 1.2,
        "train_batch_count": len(train_batches),
        "valid_batch_count": 2,
        "train_batch_timing_summary": summary,
    }
    payload = runner._profile_timing_payload(
        timings={
            "dataset_load": 0.1,
            "group_build": 17.0,
            "epoch_loop": 18.0,
            "prediction_export": 0.5,
        },
        profile_counts={
            "graph_metadata_build_calls": 12,
            "graph_build_graphs_calls": 5,
        },
        epoch_records=[epoch_record],
        train_batch_records=train_batches,
        validation_batch_records=[
            {
                "epoch_index": 1,
                "batch_index": 1,
                "split": "valid",
                "batch_size": 4,
                "group_count": 1,
                "total_batch_time": 0.3,
                "batch_shape_signature": {"group_count": 1, "input_x_inp_shape": [4, 1, 2048, 3]},
            }
        ],
        train_group_count=4,
        valid_group_count=2,
        all_group_count=6,
        train_batch_counts=[4],
        subset=runner.DEFAULT_SUBSET,
        output_dir=REPO_DIR / "output" / "heat3d_v2_runs" / "timing_schema_smoke",
        total_run_time_so_far=36.0,
    )
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    epoch_fields = set(decoded["per_epoch"][0])
    batch_fields = set(decoded["train_batches"][0])
    missing_epoch = REQUIRED_EPOCH_FIELDS - epoch_fields
    missing_batch = REQUIRED_BATCH_FIELDS - batch_fields
    if missing_epoch:
        raise AssertionError(f"missing per-epoch fields: {sorted(missing_epoch)}")
    if missing_batch:
        raise AssertionError(f"missing per-batch fields: {sorted(missing_batch)}")
    if decoded["run_level"]["metadata_calls"] != 12:
        raise AssertionError("run-level metadata call count missing")
    print("Heat3D v2 timing profile schema smoke passed.")
    return 0


def _batch_record(epoch: int, batch: int, total_time: float, x_shape: str) -> dict:
    return {
        "epoch_index": epoch,
        "batch_index": batch,
        "split": "train",
        "batch_size": 4 if batch < 4 else 2,
        "group_count": 1,
        "total_batch_time": total_time,
        "loss_grad_time": total_time * 0.80,
        "grad_norm_time": total_time * 0.05,
        "optimizer_update_time": total_time * 0.10,
        "output_scalar_extraction_time": 0.0,
        "other_time": total_time * 0.05,
        "batch_shape_signature": {
            "group_count": 1,
            "sample_count": 4 if batch < 4 else 2,
            "input_x_inp_shape": x_shape,
            "target_shape": x_shape.replace(",3)", ",1)"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
