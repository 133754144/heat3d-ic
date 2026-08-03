#!/usr/bin/env python3
"""Deterministic gate for the P1i three-seed valid-only closeout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "heat3d_v6_p1i"
DOCS = ROOT / "docs"
TRAINING_COMMIT = "3884de07525b7e8c0f8fa3382b24bf94322bebe9"
MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
FULL_FIELD_SHA = "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb"
RESOLUTIONS = [1024, 4096, 16384, 65536, 240825]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    closeout = json.loads(
        (CONFIG / "v6_p1i_three_seed_inference_closeout.json").read_text()
    )
    artifact_manifest = json.loads(
        (CONFIG / "v6_p1i_three_seed_artifact_manifest.json").read_text()
    )
    _require(closeout["status"] == "completed_three_seed_valid_only", "closeout status")
    _require(closeout["training_commit"] == TRAINING_COMMIT, "training commit drift")
    _require(closeout["test_accessed"] is False, "test must remain closed")
    _require(closeout["sealed_accessed"] is False, "sealed IID must remain closed")
    _require(closeout["retrained"] is False and closeout["tuned"] is False, "no retraining/tuning")
    _require(closeout["primary_checkpoint"] == "point_global_best", "primary checkpoint drift")
    floor = closeout["valid_full_field_reconstruction_only_sampling_floor"]
    for key in (
        "point_global_true_rms_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
    ):
        _require(_finite(floor[key]), f"non-finite reconstruction floor: {key}")
    _require([item["seed"] for item in closeout["seeds"]] == [0, 1, 2], "three seeds")

    for seed in closeout["seeds"]:
        _require(seed["training_commit"] == TRAINING_COMMIT, "per-seed commit drift")
        _require(seed["log"]["epoch_600_recorded"] is True, "incomplete training log")
        replay = seed["independent_replay"]
        _require(replay["status"] == "passed", "independent checkpoint replay")
        _require(replay["test_accessed"] is False and replay["sealed_accessed"] is False, "role leak")
        _require(replay["checkpoint_modified"] is False and replay["training_executed"] is False, "read-only replay")
        standardizer = replay["global_context_standardizer"]
        _require(standardizer["fit_population"] == "train_only", "normalizer must be train-only")
        _require(int(standardizer["fit_sample_count"]) == 768, "normalizer population")
        labels = {entry["label"] for entry in replay["entries"]}
        _require(
            {"best", "point_global_alias", "sample_first_best", "base_mse_best", "final", "latest"} <= labels,
            "missing checkpoint replay",
        )
        for entry in replay["entries"]:
            _require(entry["status"] == "passed", "checkpoint replay failed")
            _require(len(entry["checkpoint"]["sha256"]) == 64, "checkpoint SHA")
            _require(len(entry["archived_predictions"]["sha256"]) == 64, "prediction SHA")
            _require(entry["prediction_difference"]["rmse_K"] <= 0.01, "prediction replay RMSE")
            _require(entry["prediction_difference"]["fraction_abs_gt_0p1_K"] <= 1e-4, "prediction replay tail")

    checkpoint_rows = list(csv.DictReader(
        (CONFIG / "v6_p1i_three_seed_checkpoint_metrics.csv").open(encoding="utf-8")
    ))
    _require(len(checkpoint_rows) == 12, "expected 3 seeds x 4 checkpoint comparisons")
    for row in checkpoint_rows:
        _require(row["checkpoint_label"] in {"point_global_best", "sample_first_best", "base_mse_best", "final"}, "checkpoint label")
        for key in (
            "support_point_global_pct", "support_sample_first_pct", "support_raw_cv_rmse_K",
            "full_point_global_pct", "full_sample_first_pct", "full_raw_cv_rmse_K",
            "peak_rmse_K", "source_rmse_K", "background_rmse_K", "layer_mean_rmse_K",
            "interface_rmse_K", "top_rmse_K", "bottom_rmse_K",
        ):
            _require(_finite(row[key]), f"non-finite checkpoint metric: {key}")

    raw = closeout["benchmark"]["raw_results"]
    _require(closeout["benchmark"]["repeats"] == 20, "benchmark repeats")
    _require([item["resolution"] for item in raw] == RESOLUTIONS, "resolution ladder")
    _require(len(set(tuple(item["sample_ids"]) for item in raw)) == 1, "same valid sample set")
    for item in raw:
        _require(
            item["accessed_roles"] == ["train_inputs_for_frozen_standardizer", "valid_iid"],
            "benchmark role boundary",
        )
        _require(item["test_accessed"] is False and item["sealed_accessed"] is False, "benchmark role leak")
        _require(item["training_executed"] is False and item["checkpoint_modified"] is False, "benchmark mutation")
        _require(int(item["actual_target_node_count"]) == int(item["resolution"]), "target node count")
        _require(abs(float(item["power_audit"]["relative_power_error"])) <= 1e-12, "source power conservation")
        direct = item["route_A_direct"]
        interpolated = item["route_B_1024_plus_interpolation"]
        solver = item["route_C_structured_FVM"]
        _require(direct["status"] == "passed", "direct compatibility execution")
        _require(direct["graph_backend"] == "sparse_kdtree_v1", "direct sparse graph backend")
        _require(direct["scientific_graph_parameters_unchanged"] is True, "graph semantics drift")
        for route in (direct, interpolated, solver):
            _require(route["steady_end_to_end_seconds"]["count"] == 20, "steady repeat count")
            _require(_finite(route["metrics"]["point_global_true_rms_relative_rmse_pct"]), "benchmark accuracy")
        _require(
            interpolated["oracle_reconstruction_metrics"]["point_global_true_rms_relative_rmse_pct"]
            <= interpolated["metrics"]["point_global_true_rms_relative_rmse_pct"],
            "oracle reconstruction floor must not exceed model+reconstruction",
        )

    registry = list(csv.DictReader((CONFIG / "v6_p1i_training_registry.csv").open(encoding="utf-8")))
    selected = [row for row in registry if row["config_id"].startswith(("V6_06_", "V6_07_", "V6_08_"))]
    _require(len(selected) == 3, "registry rows")
    for row in selected:
        _require(row["execution_status"] == "completed_e600", "registry execution")
        _require(row["evaluation_status"] == "completed_valid_support_fullfield_replay", "registry evaluation")
        _require(row["test_access"].startswith("closed"), "registry test role")
        _require(row["sealed_access"].startswith("closed"), "registry sealed role")
        _require(_finite(row["best_valid_point_global_pct"]), "registry metric")

    _require(artifact_manifest["training_commit"] == TRAINING_COMMIT, "artifact commit")
    _require(artifact_manifest["dataset_manifest_sha256"] == MANIFEST_SHA, "dataset manifest SHA")
    _require(artifact_manifest["full_field_archive_sha256"] == FULL_FIELD_SHA, "full-field SHA")
    _require(artifact_manifest["test_accessed"] is False and artifact_manifest["sealed_accessed"] is False, "artifact role leak")
    for path in (
        DOCS / "v6_p1i_three_seed_inference_closeout.md",
        DOCS / "v6_p1i_accuracy_latency.svg",
        DOCS / "v6_p1i_memory_resolution.svg",
    ):
        _require(path.is_file() and path.stat().st_size > 500, f"missing output {path.name}")
    print(json.dumps({"status": "passed", "seeds": 3, "resolutions": RESOLUTIONS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
