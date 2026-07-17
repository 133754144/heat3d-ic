#!/usr/bin/env python3
"""Check the sole post-freeze Gate 6H sparse-safe attention candidate."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_scale_pooling import (  # noqa: E402
    QK_REGION_FEATURE_SCHEMAS,
    qk_region_features_from_raw,
)
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


CONFIG_ID = "V4P5_32_gate6h_attention_sparse_safe_v2_e600"
CONFIG = ROOT / f"configs/heat3d_v5/generated/{CONFIG_ID}.yaml"
BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6h_attention_fix_registry.csv"
CONTRACT = ROOT / "configs/heat3d_v5/gate6h/attention_sparse_safe_v2_contract.json"
AUDIT = ROOT / "configs/heat3d_v5/gate6h/attention_sparse_safe_v2_feature_audit.json"
RUNNER = ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
FORBIDDEN = (
    "test_iid|hard_train_holdout|hard_challenge_valid|"
    "hard_challenge_test|sealed_iid"
)


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value, name))
        return result
    return {prefix: payload}


def _feature_fixture(q: np.ndarray, kz: np.ndarray, *, version: str) -> np.ndarray:
    count = len(q)
    coords = np.stack(
        [np.linspace(0.0, 1.0, count), np.zeros(count), np.linspace(0.0, 2.0, count)],
        axis=-1,
    )
    names = (
        "k_z",
        "q",
        "is_top",
        "is_bottom",
        "is_side",
        "is_interior",
        "top_h",
        "bottom_T_fixed_minus_T_ref",
    )
    raw = np.stack(
        [
            kz,
            q,
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
            np.ones(count),
            np.zeros(count),
            np.zeros(count),
        ],
        axis=-1,
    )
    edges = np.stack([np.arange(count), np.arange(count) % 2], axis=-1)
    return qk_region_features_from_raw(
        coords=coords,
        raw_condition=raw,
        condition_feature_names=names,
        p2r_edge_indices=edges,
        rnode_count=2,
        feature_version=version,
    )


def _fixture_checks() -> dict[str, Any]:
    sparse_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0])
    kz_ties = np.full(8, 2.0)
    v1 = _feature_fixture(sparse_q, kz_ties, version="bugged_v1")
    v2 = _feature_fixture(sparse_q, kz_ties, version="sparse_safe_v2")
    assert v1.shape == v2.shape == (2, 11)
    assert np.allclose(v1[:, :3], v2[:, :3])
    assert np.allclose(v1[:, 4], v2[:, 4])
    assert np.allclose(v2[:, 3], [0.0, 0.25])
    assert np.all(v1[:, 3] > v2[:, 3])

    zero = _feature_fixture(np.zeros(8), kz_ties, version="sparse_safe_v2")
    assert np.all(zero[:, 3] == 0.0)
    assert np.all(np.isfinite(zero))

    same_positive = _feature_fixture(np.full(8, 3.0), kz_ties, version="sparse_safe_v2")
    assert np.allclose(same_positive[:, 3], 1.0)
    assert np.all(np.isfinite(same_positive))

    tied_k = _feature_fixture(
        np.arange(1.0, 9.0), np.full(8, 5.0), version="sparse_safe_v2"
    )
    assert np.all(np.isfinite(tied_k))
    assert np.allclose(tied_k[:, 1], 0.0)

    return {
        "sparse_q": "passed",
        "all_zero_q": "passed",
        "identical_positive_q": "passed",
        "k_ties": "passed",
        "v1_preserved": list(QK_REGION_FEATURE_SCHEMAS["bugged_v1"]),
        "v2_schema": list(QK_REGION_FEATURE_SCHEMAS["sparse_safe_v2"]),
    }


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["config_id"] == CONFIG_ID
    if row["execution_status"] == "not_started":
        assert row["long_training_started"] == "false"
    else:
        assert row["execution_status"] == "completed_e600"
        assert row["long_training_started"] == "true"
        assert row["evaluation_status"] == (
            "completed_valid_iid_four_checkpoint"
        )
    assert row["forbidden_access_roles"] == FORBIDDEN
    assert row["test_accessed"] == row["hard_accessed"] == "false"
    assert row["sealed_iid_accessed"] == "false"

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["training_started"] is False
    assert contract["access_contract"]["sealed_iid_accessed"] is False
    assert contract["checkpoint_contract"]["params_best.pkl"] == (
        "legacy_valid_base_mse_best"
    )
    assert "only_advancement_checkpoint" in contract["checkpoint_contract"][
        "params_best_valid_point_global.pkl"
    ]

    baseline = _resolved(BASELINE)
    candidate = _resolved(CONFIG)
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["graph"] == baseline["graph"]
    assert candidate["loss"] == baseline["loss"]
    assert candidate["run"]["batch_size"] == baseline["run"]["batch_size"] == 28
    assert candidate["run"]["validation_batch_size"] == 32
    assert candidate["run"]["prediction_batch_size"] == 32
    assert candidate["run"]["epochs"] == baseline["run"]["epochs"] == 600
    assert candidate["run"]["init_checkpoint"] is None
    assert candidate["model"]["scale_attention_mode"] == "physics_gate"
    assert candidate["model"]["pooled_latent_stop_gradient"] is False
    assert candidate["model"]["qk_region_feature_version"] == "sparse_safe_v2"
    assert candidate["model"]["scale_pooling"] == baseline["model"]["scale_pooling"] == "mean"
    assert candidate["model"].get("scale_head_depth", 1) == baseline["model"].get(
        "scale_head_depth", 1
    ) == 1
    for key in (
        "node_latent_size",
        "edge_latent_size",
        "processor_steps",
        "mlp_hidden_layers",
        "scale_head_mode",
        "scale_head_hidden_size",
    ):
        assert candidate["model"][key] == baseline["model"][key]
    for key in ("seed", "model_seed", "batch_order_seed", "graph_seed"):
        assert candidate["optimizer"][key] == baseline["optimizer"][key] == 0
    assert candidate["run"]["batch_build_seed"] == baseline["run"]["batch_build_seed"] == 0
    assert candidate["optimizer"]["multi_seed"] == []
    assert candidate["export"]["selection_metric"] == "valid_base_mse"
    assert candidate["export"]["prediction_split"] == "valid_iid"

    actual_optimizer = {
        "name": "adamw",
        "lr": 0.0005,
        "lr_schedule": "warmup_cosine",
        "warmup_epochs": 10,
        "min_lr": 0.00005,
        "second_stage_epoch": 0,
        "second_stage_lr": 0.0001,
        "lr_init": 0.00001,
        "lr_peak": 0.0002,
        "lr_base": 0.00001,
        "lr_lowr": 0.000001,
        "pct_start": 0.02,
        "pct_final": 0.1,
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.0001,
    }
    for key, value in actual_optimizer.items():
        assert candidate["optimizer"][key] == value

    effective_baseline = copy.deepcopy(baseline)
    effective_baseline["model"].update(
        {
            "scale_attention_mode": "none",
            "pooled_latent_stop_gradient": False,
            "qk_region_feature_version": "bugged_v1",
        }
    )
    effective_candidate = copy.deepcopy(candidate)
    ignored_prefixes = ("metadata.",)
    ignored = {
        "config_id",
        "description",
        "export.output_dir",
        "export.run_name",
        "run.final_probe_output_dir",
        "run.post_training_diagnostics_output_dir",
    }
    left, right = _flatten(effective_baseline), _flatten(effective_candidate)
    diff = {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
        and key not in ignored
        and not key.startswith(ignored_prefixes)
    }
    allowed = {
        "model.qk_region_feature_version",
        "model.scale_attention_mode",
        "run.prediction_batch_size",
        "run.validation_batch_size",
        "optimizer.second_stage_epoch",
        "optimizer.second_stage_lr",
        "optimizer.lr_init",
        "optimizer.lr_peak",
        "optimizer.lr_base",
        "optimizer.lr_lowr",
        "optimizer.pct_start",
        "optimizer.pct_final",
        "export.save_point_global_best_checkpoint",
        "export.point_global_best_checkpoint_name",
        "export.save_base_mse_best_checkpoint",
        "export.base_mse_best_checkpoint_name",
        "export.save_sample_first_best_checkpoint",
        "export.sample_first_best_checkpoint_name",
    }
    assert set(diff) == allowed, diff

    command = build_training_command(candidate, python_executable="python")
    text = shlex.join(command)
    for flag in (
        "--epochs 600",
        "--batch-size 28",
        "--validation-batch-size 32",
        "--prediction-batch-size 32",
        "--scale-pooling mean",
        "--scale-attention-mode physics_gate",
        "--no-pooled-latent-stop-gradient",
        "--qk-region-feature-version sparse_safe_v2",
        "--selection-metric valid_base_mse",
        "--prediction-split valid_iid",
        "--save-point-global-best-checkpoint",
        "--save-base-mse-best-checkpoint",
        "--save-sample-first-best-checkpoint",
        "--no-final-probe-eval-after-training",
    ):
        assert flag in text, flag
    assert "--init-checkpoint" not in command

    dry = subprocess.run(
        [
            sys.executable,
            "scripts/run_heat3d_v4_config.py",
            "--config",
            str(CONFIG.relative_to(ROOT)),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert dry.stdout.strip() == text

    source = RUNNER.read_text(encoding="utf-8")
    assert 'sample_score == sample_first_best_score' not in source
    assert "valid_native_sample_first_cv_relative_rmse" in source
    assert "valid_raw_cv_weighted_rmse_K" in source
    assert "tolerance_aware_lexicographic" in source
    assert "attention_diagnostics_by_checkpoint" in source
    assert "valid_iid_only_after_all_checkpoint_selection_frozen" in source
    assert "target" not in qk_region_features_from_raw.__code__.co_varnames

    fixture = _fixture_checks()
    audit = None
    if AUDIT.is_file():
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        assert audit["status"] == "passed"
        assert audit["roles_accessed"] == ["train", "valid_iid"]
        assert audit["forbidden_roles_accessed"] == []
        assert audit["sealed_iid_accessed"] is False
        assert audit["target_or_label_files_read"] == []

    expected_tag_target = "96fa6fb5af451d8bafd6219f4372e36c792648d6"
    local_tag = subprocess.run(
        ["git", "rev-list", "-n", "1", "v5-gate6h-frozen"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if local_tag.returncode == 0:
        tag_target = local_tag.stdout.strip()
        tag_verification_source = "local"
    else:
        remote_tag = subprocess.check_output(
            [
                "git",
                "ls-remote",
                "--tags",
                "origin",
                "refs/tags/v5-gate6h-frozen^{}",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        tag_target = remote_tag.split()[0]
        tag_verification_source = "origin"
    assert tag_target == expected_tag_target

    print(
        json.dumps(
            {
                "status": "passed",
                "config_id": CONFIG_ID,
                "checker_training_runs": 0,
                "output_writes": 0,
                "resolved_diff": diff,
                "fixture_checks": fixture,
                "feature_audit_present": audit is not None,
                "frozen_tag_target": tag_target,
                "frozen_tag_verification_source": tag_verification_source,
                "launch_command": text,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
