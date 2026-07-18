#!/usr/bin/env python3
"""Validate Gate 6M diagnostics, single-variable configs, and remote sync."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_32_gate6h_attention_sparse_safe_v2_e600.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6m_registry.csv"
CONFIGS = {
    "A_scale_head_only": ROOT
    / "configs/heat3d_v5/generated/V4P5_35_gate6m_v32_scale_head_only_e100.yaml",
    "B_epoch_wise_batch_regrouping": ROOT
    / "configs/heat3d_v5/generated/V4P5_36_gate6m_v32_epoch_regroup_e200.yaml",
}
RESULT_DIR = ROOT / "configs/heat3d_v5/gate6m"
RESULT = RESULT_DIR / "gate6m_branch_swap_gradient_audit.json"
SYNC = RESULT_DIR / "gate6m_remote_sync_manifest.json"
REPORT = ROOT / "docs/v5_gate6m_closeout.md"
LAUNCH = ROOT / "docs/v5_gate6m_launch_commands.md"
V32_SHA256 = "f3063b53ca26a2b91fffc090ad4de98fe260ac5d7b669bcfbfd77c1fcf045d24"
FORBIDDEN = (
    "test_iid|hard_train_holdout|hard_challenge_valid|"
    "hard_challenge_test|sealed_iid"
)
LOSS_NAMES = {
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


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


def _check_configs() -> None:
    baseline = _resolved(BASELINE)
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_candidate = {row["candidate"]: row for row in rows}
    assert set(by_candidate) == set(CONFIGS)
    allowed = {
        "A_scale_head_only": {
            "model.native_branch_mode",
            "run.epochs",
            "run.init_checkpoint",
            "run.checkpoint_load_strict",
            "run.partial_load_policy",
        },
        "B_epoch_wise_batch_regrouping": {
            "run.epochs",
            "run.epoch_wise_batch_regrouping",
        },
    }
    for candidate, path in CONFIGS.items():
        row = by_candidate[candidate]
        config = _resolved(path)
        assert ROOT / row["generated_yaml"] == path
        assert config["metadata"]["assigned_host"] == row["assigned_host"]
        assert config["run"]["epochs"] == int(row["epochs"])
        assert config["run"]["batch_size"] == baseline["run"]["batch_size"] == 28
        assert (
            config["run"]["validation_batch_size"]
            == baseline["run"]["validation_batch_size"]
            == 32
        )
        assert (
            config["run"]["prediction_batch_size"]
            == baseline["run"]["prediction_batch_size"]
            == 32
        )
        assert config["export"]["prediction_split"] == "valid_iid"
        assert config["export"]["save_point_global_best_checkpoint"] is True
        assert config["export"]["save_sample_first_best_checkpoint"] is True
        assert config["export"]["save_base_mse_best_checkpoint"] is True
        assert row["primary_checkpoint"] == "params_best_valid_point_global.pkl"
        assert row["sample_first_checkpoint"] == "params_best_valid_sample_first.pkl"
        assert row["base_mse_checkpoint"] == "params_best_valid_base_mse.pkl"
        assert row["final_checkpoint"] == "params_final.pkl"
        assert row["fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["launch_policy"] == "explicit_user_instruction_only"
        assert row["plan_status"] == "prepared_not_started"
        assert row["training_started"] == "false"
        assert row["test_accessed"] == "false"
        assert row["hard_accessed"] == "false"
        assert row["sealed_iid_accessed"] == "false"
        assert _diff(_science(baseline), _science(config)) == allowed[candidate]
        command = build_training_command(config, python_executable="python")
        text = shlex.join(command)
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
        for flag in (
            "--batch-size 28",
            "--validation-batch-size 32",
            "--prediction-batch-size 32",
            "--prediction-split valid_iid",
            "--save-point-global-best-checkpoint",
            "--save-sample-first-best-checkpoint",
            "--save-base-mse-best-checkpoint",
        ):
            assert flag in text
        if candidate == "A_scale_head_only":
            assert "--epochs 100" in text
            assert "--native-branch-mode scale_only" in text
            assert "--checkpoint-load-strict true" in text
            assert row["checkpoint_sha256"] == V32_SHA256
            assert "--epoch-wise-batch-regrouping" not in command
        else:
            assert "--epochs 200" in text
            assert "--native-branch-mode joint" in text
            assert "--epoch-wise-batch-regrouping" in command
            assert "--init-checkpoint" not in command
    assert LAUNCH.is_file()


def _check_results() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "completed_valid_iid_only"
    assert payload["schema_version"] == "heat3d_v5_gate6m_valid_only_v1"
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["evaluation_roles"] == ["valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["training_started"] is False
    assert scope["model_parameters_modified"] is False
    assert scope["checkpoint_selection_modified"] is False
    assert not scope["test_accessed"]
    assert not scope["hard_accessed"]
    assert not scope["sealed_iid_accessed"]
    assert payload["split"]["train_count"] == 672
    assert payload["split"]["valid_iid_count"] == 128
    assert payload["checkpoint_binding"]["V32"]["epoch"] == 474
    assert payload["checkpoint_binding"]["V32"]["sha256"] == V32_SHA256
    assert payload["checkpoint_binding"]["O075"]["epoch"] == 280
    assert set(payload["field_metrics"]) == {
        "V32",
        "O075",
        "shape_V32+scale_O075",
        "shape_O075+scale_V32",
    }
    for field in payload["field_metrics"].values():
        for value in field["summary"].values():
            if isinstance(value, (int, float)):
                assert math.isfinite(float(value))
    assert set(payload["gradient_audit"]) == {"V32", "O075"}
    for audit in payload["gradient_audit"].values():
        assert audit["sample_count"] == 128
        assert set(audit["loss_means"]) == LOSS_NAMES
        assert set(audit["shared_backbone_gradient_cosine"]) == LOSS_NAMES
        for row in audit["shared_backbone_gradient_cosine"].values():
            assert set(row) == LOSS_NAMES
            assert all(value is None or math.isfinite(value) for value in row.values())
    assert len(payload["quartile_win_loss"]) == 16
    for row in payload["quartile_win_loss"]:
        assert row["sample_count"] == 32
        assert row["win_count"] + row["loss_count"] + row["tie_count"] == 32
    assert len(payload["physical_condition_attribution"]) == 4
    for report in payload["physical_condition_attribution"].values():
        assert "no target" in report["feature_provenance"]
        assert len(report["fields"]) == 5
    for name in (
        "gate6m_branch_swap_paired_samples.csv",
        "gate6m_branch_swap_quartiles.csv",
        "gate6m_backbone_gradient_cosine.csv",
    ):
        assert (RESULT_DIR / name).is_file()
    assert REPORT.is_file()


def _check_sync() -> None:
    payload = json.loads(SYNC.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["branch"] == "research/v5"
    assert payload["checkpoint"]["sha256"] == V32_SHA256
    assert payload["training_started"] is False
    assert payload["test_accessed"] is False
    assert payload["hard_accessed"] is False
    assert payload["sealed_iid_accessed"] is False
    assert set(payload["hosts"]) == {"devbox", "wsl2"}
    for host, entry in payload["hosts"].items():
        assert entry["assigned_candidate"] == (
            "A_scale_head_only"
            if host == "devbox"
            else "B_epoch_wise_batch_regrouping"
        )
        assert entry["branch"] == "research/v5"
        assert entry["head"] == payload["head"]
        assert entry["worktree_clean"] is True
        assert entry["checkpoint_sha256"] == V32_SHA256
        assert entry["checkpoint_matches"] is True


def main() -> int:
    args = _args()
    _check_configs()
    if not args.preflight_only:
        _check_results()
        _check_sync()
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "preflight_only" if args.preflight_only else "full",
                "training_started": False,
                "roles": ["train", "valid_iid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
