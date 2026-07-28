#!/usr/bin/env python3
"""Check the frozen V6 production high-resolution inference closeout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6/v6_production_highres_inference.yaml"
CLOSEOUT = ROOT / "configs/heat3d_v6/v6_production_highres_closeout.json"
METRICS = ROOT / "configs/heat3d_v6/v6_production_highres_metrics.csv"
TIMING = ROOT / "configs/heat3d_v6/v6_production_highres_timing.csv"
CACHE = ROOT / "configs/heat3d_v6/v6_production_graph_cache_manifest.json"
COMMIT = "d5c06263ee5a5cf0b925b6e1d35ade205ada8bce"
RESOLUTIONS = ("1024", "2048", "4096", "8192", "16384", "32768")


def _finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    if config["evaluation_commit"] != COMMIT or closeout["evaluation_commit"] != COMMIT:
        raise SystemExit("evaluation commit drifted")
    if closeout["status"] != "passed" or closeout["decision"] != (
        "freeze_anchor_derived_sparse_query_subsample8"
    ):
        raise SystemExit("closeout decision drifted")
    if closeout["training_executed"] or closeout["checkpoint_modified"]:
        raise SystemExit("read-only boundary drifted")
    if closeout["evaluation_role"] != "valid_iid" or closeout["test_hard_accessed"]:
        raise SystemExit("role boundary drifted")
    if closeout["upstream_like_preforward_executed"]:
        raise SystemExit("obsolete upstream-like preforward was executed")
    if config["resolution"]["default"] != 4096:
        raise SystemExit("default resolution drifted")
    if config["resolution"]["maximum_production_verified"] != 16384:
        raise SystemExit("production maximum drifted")
    graph = closeout["production_graph"]
    if (
        graph["backend"] != "sparse_kdtree_v1"
        or graph["anchor_subsample_factor"] != 4
        or graph["query_subsample_factor"] != 8
        or not graph["accepted"]
    ):
        raise SystemExit("production graph decision drifted")
    if graph["point_global_relative_rmse_pct"] >= 20.0:
        raise SystemExit("point-global acceptance failed")
    if graph["full_field_rmse_ratio"] > 1.10:
        raise SystemExit("full-field acceptance failed")

    with METRICS.open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    expected = {
        *(("cpu", resolution, "1") for resolution in RESOLUTIONS),
        *(("gpu", resolution, "1") for resolution in RESOLUTIONS),
        ("cpu", "4096", "8"),
        ("gpu", "4096", "8"),
    }
    observed = {
        (row["platform"], row["resolution"], row["batch_size"]) for row in metrics
    }
    if observed != expected:
        raise SystemExit("metrics coverage drifted")
    numeric_metrics = (
        "support_point_global_relative_rmse_pct",
        "support_sample_first_relative_rmse_pct",
        "support_raw_cv_rmse_K",
        "full_point_global_relative_rmse_pct",
        "full_raw_cv_rmse_K",
        "sampling_floor_raw_cv_rmse_K",
    )
    if any(not _finite(row[field]) for row in metrics for field in numeric_metrics):
        raise SystemExit("non-finite metric")

    with TIMING.open(encoding="utf-8", newline="") as handle:
        timing = list(csv.DictReader(handle))
    if len(timing) != len(metrics):
        raise SystemExit("timing coverage drifted")
    numeric_timing = (
        "graph_cached_load_seconds",
        "compile_seconds",
        "warm_batch_seconds_mean",
        "formal_inference_seconds",
        "samples_per_second",
        "end_to_end_seconds_valid128",
        "process_peak_ram_GB",
    )
    if any(not _finite(row[field]) for row in timing for field in numeric_timing):
        raise SystemExit("non-finite timing")
    if any(
        row["batch_size"] == "1"
        and not _finite(row["graph_uncached_build_seconds"])
        for row in timing
    ):
        raise SystemExit("missing uncached graph-build timing")

    if cache["evaluation_commit"] != COMMIT or cache["backend"] != (
        "sparse_kdtree_v1"
    ):
        raise SystemExit("cache provenance drifted")
    if cache["production_resolutions"] != [1024, 2048, 4096, 8192, 16384]:
        raise SystemExit("cache resolution set drifted")
    if len(cache["entries"]) != 10:
        raise SystemExit("cache entry count drifted")
    for row in cache["entries"]:
        if (
            not row["metadata_hash_equal"]
            or not row["graph_hash_equal"]
            or row["prediction_max_abs_error_K"] != 0.0
            or row["prediction_rmse_K"] != 0.0
        ):
            raise SystemExit("cache equivalence drifted")
        payload = row["cache_key_payload"]
        if payload["commit"] != COMMIT:
            raise SystemExit("cache commit drifted")
        if set(payload) != {
            "schema_version",
            "support_hash",
            "graph_config",
            "graph_seed",
            "commit",
        }:
            raise SystemExit("cache-key fields drifted")

    solver = closeout["solver_benchmark"]
    if (
        solver["evaluation_role"] != "valid_iid"
        or solver["test_hard_accessed"]
        or solver["sample_count"] != 128
    ):
        raise SystemExit("solver role or coverage drifted")
    if not closeout["speedup"]["nonmatched_dof"]:
        raise SystemExit("nonmatched-DOF qualification missing")
    print(
        json.dumps(
            {
                "status": "passed",
                "metric_rows": len(metrics),
                "cache_entries": len(cache["entries"]),
                "experimental_32768": "passed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
