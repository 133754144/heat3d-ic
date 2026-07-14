#!/usr/bin/env python3
"""Check tracked Gate-5 final closeout and error-attribution artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSEOUT = ROOT / "configs/heat3d_v5/v5_gate5_final_closeout.json"
DIAGNOSTIC = ROOT / "configs/heat3d_v5/v5_gate5_final_error_attribution.json"
CLOSEOUT_MD = ROOT / "docs/v5_gate5_final_closeout.md"
DIAGNOSTIC_MD = ROOT / "docs/v5_gate5_final_error_attribution.md"
LABELS = ("B0", "N0", "N1", "N3")
ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
METRICS = (
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
    "legacy_normalized_valid_base_mse",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    assert CLOSEOUT_MD.stat().st_size > 0 and DIAGNOSTIC_MD.stat().st_size > 0
    assert closeout["status"] == "complete_no_followup_training_started"
    assert closeout["next_phase_started"] is False
    assert diagnostic["status"] == "complete"
    assert closeout["evaluator_git_commit"] == diagnostic["evaluator_git_commit"]
    assert set(closeout["runs"]) == set(LABELS)
    for label, run in closeout["runs"].items():
        assert run["training_git_commit"]
        assert run["evaluator_git_commit"] == closeout["evaluator_git_commit"]
        assert run["registry_git_commit"]
        for checkpoint in ("best", "final"):
            assert run["checkpoint_epochs"][checkpoint] > 0
            assert len(run["checkpoint_sha256"][checkpoint]) == 64
            for role in ROLES:
                report = run["reports"][checkpoint][role]
                assert report["sample_count"] > 0
                assert all(_finite(report[metric]) for metric in METRICS), (label, checkpoint, role)
        expected = (
            run["reports"]["best"]["valid_iid"]["point_global_relative_rmse_pct"] < 20.0
            and run["reports"]["best"]["test_iid"]["point_global_relative_rmse_pct"] < 20.0
        )
        assert closeout["threshold_assessment"][label] is expected

    mechanism = diagnostic["n1_to_n3_mechanism"]
    assert mechanism["classification"] in {
        "joint_path", "shape_dominant", "scale_dominant", "no_clean_component_gain"
    }
    for role in ("valid_iid", "test_iid"):
        film = mechanism["n3_film_modulation"][role]
        assert film and all(_finite(value) for value in film.values())
    tail = diagnostic["high_deltaT_tail"]
    assert tail["sample_count"] == 256
    assert len(tail["true_cv_rms_deltaT_bins"]) == 4
    for label in ("B0", "N1", "N3"):
        share = sum(
            row["models"][label]["total_squared_error_contribution"]
            for row in tail["true_cv_rms_deltaT_bins"]
        )
        assert math.isclose(share, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9)
    for label in LABELS:
        concentration = tail["squared_error_concentration"][label]
        assert 0.0 <= concentration["top5_cumulative_contribution"] <= 1.0
        assert concentration["top5_cumulative_contribution"] <= concentration["top10_cumulative_contribution"] <= 1.0
        assert len(concentration["top10_samples"]) == 10
    print(json.dumps({
        "status": "passed",
        "evaluator_git_commit": closeout["evaluator_git_commit"],
        "mechanism": mechanism["classification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
