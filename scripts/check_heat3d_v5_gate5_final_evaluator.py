#!/usr/bin/env python3
"""Check the frozen Gate-5 evaluator contract and optional result payloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/heat3d_v5/v5_gate5_final_evaluator_contract.json"
REGISTRY = ROOT / "configs/heat3d_v5/v5_scratch_bypass_film_registry.csv"
EVALUATOR = ROOT / "scripts/evaluate_heat3d_v5_gate5_checkpoints.py"
ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
REQUIRED_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen"
    assert contract["nodes_per_sample"] == 1024
    assert len(contract["configs"]) == 4
    assert set(contract["split_hashes"]) == {"train", *ROLES}
    rows = {row["config_id"]: row for row in csv.DictReader(REGISTRY.open(encoding="utf-8"))}
    assert set(contract["configs"]).issubset(rows)
    assert rows["V4P5_07_native_pooled_latent_global_film"]["loss_mode"] == "native_shape_scale_four_term"
    source = EVALUATOR.read_text(encoding="utf-8")
    for fragment in (
        "normalization_recomputed_from_train_only",
        "global_context_recomputed_from_train_only",
        "split_hashes_match_contract",
        "film_gamma_mean_abs",
        "per_sample",
    ):
        assert fragment in source

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.result_json]
    if payloads:
        assert len(payloads) == 4
        assert {payload["config_id"] for payload in payloads} == set(contract["configs"])
        commits = {payload["evaluator_git_commit"] for payload in payloads}
        assert len(commits) == 1, f"evaluator commits differ: {commits}"
        for payload in payloads:
            standardizer = payload["global_context_standardizer"]
            assert standardizer["fit_roles"] == ["train"]
            assert standardizer["fit_sample_count"] == 672
            assert standardizer["fit_sample_ids_sha256"] == contract["train_context_fit_sample_ids_sha256"]
            assert standardizer["train_split_ordered_ids_sha256"] == contract["split_hashes"]["train"]
            assert standardizer["target_or_label_features"] == []
            assert payload["validation_audit"] == {
                "checkpoint_best_final_normalization_equal": True,
                "checkpoint_kind_epoch_and_run_bound": True,
                "config_id_bound": True,
                "global_context_recomputed_from_train_only": True,
                "nodes_per_sample": 1024,
                "normalization_recomputed_from_train_only": True,
                "run_directory_bound": True,
                "split_hashes_match_contract": True,
            }
            for checkpoint in ("best", "final"):
                for role in ROLES:
                    report = payload["reports"][checkpoint][role]
                    assert report["per_sample"]
                    for metric in REQUIRED_METRICS:
                        assert math.isfinite(float(report[metric])), (
                            payload["config_id"], checkpoint, role, metric
                        )
                    for row in report["per_sample"]:
                        assert row["attribution_context"]
                        assert math.isfinite(float(row["point_error_squared_sum"]))
    print(json.dumps({
        "status": "passed",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "result_count": len(payloads),
        "evaluator_commit": next(iter({p["evaluator_git_commit"] for p in payloads}), None),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
