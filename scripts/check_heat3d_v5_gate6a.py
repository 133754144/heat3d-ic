#!/usr/bin/env python3
"""Static and result checks for the Gate 6A no-training diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose_heat3d_v5_gate6a.py"
LOSSES = (
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
)
GROUPS = ("backbone", "shape_decoder", "scale_head", "film", "bypass")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    args = _args()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SPLITS = (\"train\", \"valid_iid\")" in source
    assert "forbidden_splits_loaded" in source
    assert "training_started\": False" in source
    for forbidden in ("test_iid", "hard_train_holdout", "hard_challenge"):
        assert forbidden not in source
    if args.result_json is None:
        print(json.dumps({"status": "passed", "mode": "static"}, indent=2))
        return 0
    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["training_started"] is False
    assert payload["checkpoint"]["epoch"] == 402
    assert len(payload["checkpoint"]["sha256"]) == 64
    access = payload["data_access_contract"]
    assert access["loaded_splits"] == ["train", "valid_iid"]
    assert access["forbidden_splits_loaded"] == []
    assert access["split_counts"] == {"train": 672, "valid_iid": 128}
    assert access["nodes_per_sample"] == 1024
    for split, expected_count in (("train", 672), ("valid_iid", 128)):
        result = payload["splits"][split]
        assert len(result["per_sample"]) == expected_count
        assert all(_finite(value) for value in result["point_global_metrics"].values())
        diagnostic = result["diagnostic"]
        assert set(diagnostic["loss_means"]) == set(LOSSES)
        assert all(_finite(value) for value in diagnostic["loss_means"].values())
        for loss in LOSSES:
            gradient = diagnostic["gradient_summaries"][loss]
            assert _finite(gradient["global_norm"])
            assert set(gradient["parameter_group_norms"]) == set(GROUPS)
            assert all(_finite(value) for value in gradient["parameter_group_norms"].values())
            for other in LOSSES:
                cosine = diagnostic["loss_gradient_cosine_similarity"][loss][other]
                for value in cosine.values():
                    assert value is None or (_finite(value) and -1.000001 <= float(value) <= 1.000001)
        for table in (
            diagnostic["true_cv_rms_deltaT_quartiles"],
            diagnostic["total_power_quartiles"],
        ):
            assert len(table) == 4
            assert sum(row["sample_count"] for row in table) == expected_count
            assert all(
                _finite(value)
                for row in table
                for value in row["mean_losses"].values()
            )
        subsets = {row["subset"]: row for row in diagnostic["subset_contributions"]}
        assert set(subsets) == {
            "true_scale_Q4", "total_power_Q4",
            "top5_unit_weight_total_loss", "top10_unit_weight_total_loss",
        }
        assert subsets["top5_unit_weight_total_loss"]["sample_count"] == 5
        assert subsets["top10_unit_weight_total_loss"]["sample_count"] == 10
        assert len(diagnostic["top10_sample_contributions"]) == 10
        assert all(
            _finite(row["scaled_gradient_norm_ratio_to_full"])
            and _finite(row["unit_weight_total_loss_contribution_fraction"])
            for row in diagnostic["top10_sample_contributions"]
        )
    print(json.dumps({
        "status": "passed",
        "mode": "result",
        "diagnostic_git_commit": payload["diagnostic_git_commit"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
