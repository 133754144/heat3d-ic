#!/usr/bin/env python3
"""Validate Gate 6O diagnostics, calibration, and prepared seed1 pair."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import jax.numpy as jnp
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_shape_scale import mask_native_trainable_scope  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p005_e600.yaml"
STAGE2 = ROOT / "configs/heat3d_v5/generated/V4P5_39_gate6o_e543_scale_mlp_calibration_e40.yaml"
SEED_P0 = ROOT / "configs/heat3d_v5/generated/V4P5_40_gate6o_seed1_full_graph_e600.yaml"
SEED_P005 = ROOT / "configs/heat3d_v5/generated/V4P5_41_gate6o_seed1_r2r_mask_p005_e600.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6o_registry.csv"
CONTRACT = ROOT / "configs/heat3d_v5/gate6o/gate6o_selection_contract.json"
DIAGNOSTICS = ROOT / "configs/heat3d_v5/gate6o/gate6o_diagnostics.json"
RESULT = ROOT / "configs/heat3d_v5/gate6o/gate6o_stage2_valid_only_metrics.json"
FROZEN_AUDIT = (
    ROOT / "configs/heat3d_v5/gate6o/gate6o_frozen_parameter_audit.json"
)
FORBIDDEN = (
    "test_iid|hard_train_holdout|hard_challenge_valid|"
    "hard_challenge_test|sealed_iid"
)


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    validate_v2_config(resolved, config_path=path)
    return resolved


def _science(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in ("schema_version", "config_id", "description", "metadata"):
        result.pop(key, None)
    result["run"].pop("final_probe_output_dir", None)
    result["run"].pop("post_training_diagnostics_output_dir", None)
    result["export"].pop("output_dir", None)
    result["export"].pop("run_name", None)
    return result


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


def _check_scope_mask() -> None:
    values = {
        "global_scale_hidden": {"kernel": jnp.ones((2, 2))},
        "global_scale_output": {"kernel": jnp.ones((2, 1))},
        "scale_attention_logits": {"kernel": jnp.ones((2, 1))},
        "processor": {"kernel": jnp.ones((2, 2))},
        "decoder": {"kernel": jnp.ones((2, 1))},
    }
    masked = mask_native_trainable_scope(
        values,
        branch_mode="scale_only",
        trainable_scope="global_scale_mlp_only",
    )
    assert float(jnp.sum(masked["global_scale_hidden"]["kernel"])) == 4.0
    assert float(jnp.sum(masked["global_scale_output"]["kernel"])) == 2.0
    assert float(jnp.sum(masked["scale_attention_logits"]["kernel"])) == 0.0
    assert float(jnp.sum(masked["processor"]["kernel"])) == 0.0
    assert float(jnp.sum(masked["decoder"]["kernel"])) == 0.0


def _check_diagnostics() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_gate6o_diagnostic_execution"
    assert contract["selection_rule"]["primary"] == "valid_iid shape_cv_rmse"
    payload = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert scope["roles_accessed"] == ["train", "valid_iid"]
    assert scope["forbidden_roles_accessed"] == []
    assert scope["test_accessed"] is False
    assert scope["hard_accessed"] is False
    assert scope["sealed_iid_accessed"] is False
    assert payload["stage2_selection"]["selected_checkpoint"] == "e543"
    assert payload["stage2_selection"]["selected_epoch"] == 543
    assert payload["stage2_selection"]["primary_metric"] == "shape_cv_rmse"
    bootstrap = payload["paired_bootstrap"]["metrics"]
    assert bootstrap["shape_cv_rmse"]["ci95"][1] < 0.0
    assert bootstrap["sample_first_cv_relative_rmse_pct"]["ci95"][1] < 0.0
    assert {row["deltaT_quartile"] for row in payload["quartile_attribution"]} == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }
    assert payload["affine_scale_calibration"]["fit_roles"] == ["train"]


def _check_configs_and_registry() -> None:
    baseline = _resolved(BASELINE)
    stage2 = _resolved(STAGE2)
    seed_p0 = _resolved(SEED_P0)
    seed_p005 = _resolved(SEED_P005)
    assert _diff(_science(baseline), _science(stage2)) == {
        "model.native_branch_mode",
        "model.p_edge_masking",
        "optimizer.lr",
        "optimizer.lr_schedule",
        "optimizer.native_trainable_scope",
        "run.checkpoint_load_strict",
        "run.epochs",
        "run.init_checkpoint",
        "run.partial_load_policy",
    }
    assert _diff(_science(seed_p0), _science(seed_p005)) == {
        "model.p_edge_masking"
    }
    assert _diff(_science(baseline), _science(seed_p0)) == {
        "model.p_edge_masking",
        "optimizer.model_seed",
    }
    assert _diff(_science(baseline), _science(seed_p005)) == {
        "optimizer.model_seed"
    }
    assert stage2["run"]["epochs"] == 40
    assert stage2["optimizer"]["lr"] == 0.0001
    assert stage2["optimizer"]["lr_schedule"] == "constant"
    assert stage2["model"]["p_edge_masking"] == 0.0
    assert stage2["model"]["native_branch_mode"] == "scale_only"
    assert (
        stage2["optimizer"]["native_trainable_scope"]
        == "global_scale_mlp_only"
    )
    assert stage2["run"]["init_checkpoint"].endswith(
        "params_best_valid_sample_first.pkl"
    )
    for config in (stage2, seed_p0, seed_p005):
        assert config["export"]["prediction_split"] == "valid_iid"
        assert config["export"]["save_point_global_best_checkpoint"] is True
        assert config["export"]["save_sample_first_best_checkpoint"] is True
        assert config["export"]["save_base_mse_best_checkpoint"] is True
    for config, path in (
        (stage2, STAGE2),
        (seed_p0, SEED_P0),
        (seed_p005, SEED_P005),
    ):
        command = build_training_command(config, python_executable="python")
        dry = subprocess.run(
            [
                sys.executable,
                "scripts/run_heat3d_v4_config.py",
                "--config",
                str(path.relative_to(ROOT)),
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert dry.stdout.strip() == shlex.join(command)
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    by_id = {row["config_id"]: row for row in rows}
    for row in rows:
        assert row["assigned_host"] == "wsl2"
        assert row["fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["test_accessed"] == "false"
        assert row["hard_accessed"] == "false"
        assert row["sealed_iid_accessed"] == "false"
    for config_id in (
        seed_p0["config_id"],
        seed_p005["config_id"],
    ):
        row = by_id[config_id]
        assert row["execution_status"] == "not_started"
        assert row["training_started"] == "false"
        assert row["launch_policy"] == "explicit_user_instruction_only"
    stage_row = by_id[stage2["config_id"]]
    if stage_row["execution_status"] == "completed_e40":
        assert stage2["metadata"]["status"] == "completed_e40"
        assert stage2["metadata"]["training_started"] is True
        assert stage2["metadata"]["training_commit"] == stage_row["training_commit"]
        assert stage_row["training_started"] == "true"
        assert stage_row["plan_status"] == "completed"
        assert stage_row["evaluation_status"] == "completed_valid_iid_only"
        assert stage_row["result_json"] == str(RESULT.relative_to(ROOT))
        assert RESULT.is_file()
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        assert result["status"] == "completed_valid_iid_only"
        assert result["training_host"] == "wsl2"
        assert result["training_commit"] == stage_row["training_commit"]
        assert result["scope"]["roles_accessed"] == ["train", "valid_iid"]
        assert result["scope"]["forbidden_roles_accessed"] == []
        assert result["scope"]["test_accessed"] is False
        assert result["scope"]["hard_accessed"] is False
        assert result["scope"]["sealed_iid_accessed"] is False
        assert result["training_completion"]["final_epoch"] == 40
        assert result["training_completion"]["epoch_history_count"] == 40
        assert result["training_completion"]["grad_finite"] is True
        expected_epochs = {
            "point_global_best": 24,
            "sample_first_best": 0,
            "legacy_best": 24,
            "final": 40,
        }
        for name, epoch in expected_epochs.items():
            checkpoint = result["checkpoint_metadata"][name]
            assert checkpoint["epoch"] == epoch
            assert len(checkpoint["sha256"]) == 64
            assert checkpoint["parameter_count"] == 893736
            assert checkpoint["parameter_reload_max_abs_error"] == 0.0
            metrics = result["metrics"][name]
            assert metrics["epoch"] == epoch
            for value in metrics.values():
                if isinstance(value, (int, float)):
                    assert bool(jnp.isfinite(value))
        assert FROZEN_AUDIT.is_file()
        audit = json.loads(FROZEN_AUDIT.read_text(encoding="utf-8"))
        assert audit["status"] == "passed"
        assert audit["trainable_scope"] == "global_scale_mlp_only"
        for name, report in audit["reports"].items():
            assert report["passed"] is True, name
            assert report["frozen_max_abs_difference"] == 0.0, name
            expected_changed = 0 if report["epoch"] == 0 else 4
            assert report["trainable_changed_leaf_count"] == expected_changed
    else:
        assert stage_row["execution_status"] == "not_started"
        assert stage_row["training_started"] == "false"


def main() -> int:
    _check_scope_mask()
    _check_diagnostics()
    _check_configs_and_registry()
    print("Gate 6O checks passed (seed1_training_runs=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
