#!/usr/bin/env python3
"""Validate Gate 6K audit, checkpoint selection, and two frozen YAML plans."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _sample_first_candidate_is_better,
)


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_32_gate6h_attention_sparse_safe_v2_e600.yaml"
CONFIGS = {
    "O075": ROOT / "configs/heat3d_v5/generated/V4P5_33_gate6k_o075_log_scale.yaml",
    "Dual": ROOT / "configs/heat3d_v5/generated/V4P5_34_gate6k_dual_physics_attention.yaml",
}
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6k_v32_single_variable_registry.csv"
AUDIT = ROOT / "configs/heat3d_v5/gate6k/gate6k_train_valid_loss_audit.json"
REPORT = ROOT / "docs/v5_gate6k_train_valid_loss_audit.md"
RUNNER = ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
FORBIDDEN = (
    "test_iid|hard_train_holdout|hard_challenge_valid|"
    "hard_challenge_test|sealed_iid"
)
LOSS_NAMES = (
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
)


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _science(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    for field in ("schema_version", "config_id", "description", "metadata"):
        payload.pop(field, None)
    payload["run"].pop("final_probe_output_dir", None)
    payload["run"].pop("post_training_diagnostics_output_dir", None)
    payload["export"].pop("output_dir", None)
    payload["export"].pop("run_name", None)
    return payload


def _diff(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: set[str] = set()
        for key in set(left) | set(right):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.add(name)
            else:
                result |= _diff(left[key], right[key], name)
        return result
    return set() if left == right else {prefix}


def _check_audit() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert payload["status"] == "completed_read_only"
    assert payload["scope"]["roles_accessed"] == ["train", "valid_iid"]
    assert payload["scope"]["forbidden_roles_accessed"] == []
    assert payload["scope"]["training_started"] is False
    assert payload["scope"]["test_accessed"] is False
    assert payload["scope"]["hard_accessed"] is False
    assert payload["scope"]["sealed_iid_accessed"] is False
    assert payload["scope"]["train_count"] == 672
    assert payload["scope"]["valid_iid_count"] == 128
    assert payload["normalization_and_context"]["fit_roles"] == ["train"]
    assert payload["normalization_and_context"]["target_or_label_features"] == []
    for model in ("V13", "V32"):
        for split in ("train", "valid_iid"):
            entry = payload["models"][model][split]
            assert entry["sample_count"] == (672 if split == "train" else 128)
            for loss in LOSS_NAMES:
                distribution = entry["losses"][loss]["distribution"]
                assert set(("mean", "median", "p90", "p95", "p99")) <= set(distribution)
                assert entry["losses"][loss]["worst_sample"]["fraction_of_total"] >= 0.0
            assert entry["losses"]["signed_scale_log_error"]["rmse"] >= 0.0
            assert entry["subsets"]["deltaT_Q2"]["sample_count"] > 0
            assert entry["subsets"]["nominal_to_hard"]["sample_count"] > 0
            intersection = entry["subsets"][
                "deltaT_Q2_intersection_nominal_to_hard"
            ]
            assert intersection is None or intersection["sample_count"] > 0
    assert REPORT.is_file()


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    by_candidate = {row["candidate"]: row for row in rows}
    assert set(by_candidate) == set(CONFIGS)
    baseline = _resolved(BASELINE)
    allowed = {
        "O075": {"loss.native_log_scale_weight"},
        "Dual": {"model.shape_attention_mode"},
    }
    for candidate, path in CONFIGS.items():
        row = by_candidate[candidate]
        resolved = _resolved(path)
        assert ROOT / row["generated_yaml"] == path
        assert row["plan_status"] == "frozen_prepared"
        assert row["execution_status"] == "not_started"
        assert row["evaluation_status"] == "not_evaluated"
        assert row["training_started"] == "false"
        assert row["launch_policy"] == "explicit_user_instruction_only"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert resolved["run"]["init_checkpoint"] is None
        assert resolved["run"]["epochs"] == 600
        assert resolved["run"]["batch_size"] == 28
        assert resolved["run"]["validation_batch_size"] == 32
        assert resolved["run"]["prediction_batch_size"] == 32
        assert resolved["run"]["seed"] == 0
        assert resolved["optimizer"]["multi_seed"] == []
        assert resolved["export"]["prediction_split"] == "valid_iid"
        assert resolved["export"]["save_point_global_best_checkpoint"] is True
        assert resolved["export"]["save_base_mse_best_checkpoint"] is True
        assert resolved["export"]["save_sample_first_best_checkpoint"] is True
        assert resolved["model"]["scale_attention_mode"] == "physics_gate"
        assert resolved["model"]["qk_region_feature_version"] == "sparse_safe_v2"
        assert _diff(_science(baseline), _science(resolved)) == allowed[candidate]
        command = build_training_command(resolved, python_executable="python")
        text = shlex.join(command)
        for flag in (
            "--epochs 600",
            "--batch-size 28",
            "--validation-batch-size 32",
            "--prediction-batch-size 32",
            "--prediction-split valid_iid",
            "--save-point-global-best-checkpoint",
            "--save-base-mse-best-checkpoint",
            "--save-sample-first-best-checkpoint",
            "--scale-attention-mode physics_gate",
            "--qk-region-feature-version sparse_safe_v2",
        ):
            assert flag in text, (candidate, flag)
        assert "--init-checkpoint" not in command
        if candidate == "O075":
            assert "--native-log-scale-weight 0.75" in text
            assert "--shape-attention-mode none" in text
        else:
            assert "--native-log-scale-weight 0.5" in text
            assert "--shape-attention-mode physics_gate" in text
        dry = subprocess.run(
            [
                sys.executable,
                "scripts/run_heat3d_v4_config.py",
                "--config",
                str(path.relative_to(ROOT)),
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert dry.stdout.strip() == text

    assert _sample_first_candidate_is_better(0.20, 0.16, 0.21, 0.15)
    assert _sample_first_candidate_is_better(
        0.20 + 1.0e-12, 0.14, 0.20, 0.15
    )
    assert not _sample_first_candidate_is_better(
        0.20 + 1.0e-12, 0.16, 0.20, 0.15
    )
    assert not _sample_first_candidate_is_better(0.21, 0.14, 0.20, 0.15)
    source = RUNNER.read_text(encoding="utf-8")
    assert "sample_score == sample_first_best_score" not in source
    assert "valid_native_sample_first_cv_relative_rmse" in source
    assert "valid_raw_cv_weighted_rmse_K" in source
    assert "tolerance_aware_lexicographic" in source
    assert 'payload["shape_attention"]' in source
    _check_audit()
    print(
        json.dumps(
            {
                "status": "passed",
                "configs": [str(path.relative_to(ROOT)) for path in CONFIGS.values()],
                "scientific_diffs": {
                    key: sorted(value) for key, value in allowed.items()
                },
                "training_started": False,
                "roles_accessed": ["train", "valid_iid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
