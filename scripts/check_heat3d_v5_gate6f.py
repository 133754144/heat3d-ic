#!/usr/bin/env python3
"""Static Gate 6F registry, runner and low-memory control checks."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shlex
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
    QK_REGION_FEATURES,
    SCALE_POOLING_MODES,
    qk_region_features_from_raw,
)
from rigno.models.rigno import RIGNO  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6f_scale_probe_registry.csv"
N3 = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
EXPECTED_IDS = tuple(f"V4P5_{index:02d}_gate6f_{suffix}_smoke" for index, suffix in (
    (14, "mean_pool"),
    (15, "mean_std"),
    (16, "mean_max"),
    (17, "pre_film_mean_std"),
    (18, "deep_scale_head"),
    (19, "latent_attention"),
    (20, "qk_gated"),
    (21, "mean_decoupled"),
))
FORBIDDEN = "test_iid|hard_train_holdout|hard_challenge_valid|hard_challenge_test"


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _assert_qk_input_only() -> None:
    names = (
        "k_z", "q", "is_top", "is_bottom", "is_side", "is_interior",
        "top_h", "bottom_T_fixed_minus_T_ref",
    )
    coords = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.5], [0.8, 0.0, 0.7], [1.0, 0.0, 1.0]])
    raw = np.asarray([
        [2.0, 0.0, 1, 0, 1, 0, 1000.0, 0.0],
        [1.0, 3.0, 0, 0, 0, 1, 1000.0, 0.0],
        [0.5, 5.0, 0, 0, 0, 1, 1000.0, 0.0],
        [1.5, 0.0, 0, 1, 1, 0, 1000.0, 5.0],
    ])
    region = qk_region_features_from_raw(
        coords=coords,
        raw_condition=raw,
        condition_feature_names=names,
        p2r_edge_indices=np.asarray([[0, 0], [1, 0], [2, 1], [3, 1], [4, 2]]),
        rnode_count=2,
    )
    assert region.shape == (2, len(QK_REGION_FEATURES))
    assert np.all(np.isfinite(region))


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert tuple(row["config_id"] for row in rows) == EXPECTED_IDS
    assert len({row["output_dir"] for row in rows}) == len(rows)
    assert len({row["memory_audit_jsonl"] for row in rows}) == len(rows)
    n3 = _resolved(N3)
    commands: dict[str, str] = {}
    for row in rows:
        path = ROOT / row["generated_yaml"]
        resolved = _resolved(path)
        assert row["baseline_config_id"] == "V4P5_07_native_pooled_latent_global_film"
        assert resolved["dataset"] == n3["dataset"]
        assert resolved["graph"] == n3["graph"]
        assert resolved["run"]["epochs"] == int(row["epochs"]) == 1
        assert resolved["run"]["batch_size"] == 28
        assert resolved["run"]["validation_batch_size"] == 128
        assert resolved["run"]["init_checkpoint"] is None
        assert resolved["optimizer"]["multi_seed"] == []
        assert resolved["export"]["prediction_split"] == "valid_iid"
        assert resolved["export"]["selection_metric"] == "valid_base_mse"
        assert row["fit_roles"] == row["normalization_fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["launch_policy"] == "explicit_user_instruction_only"
        assert row["long_training_started"] == "false"
        assert row["plan_status"] == "frozen_prepared"
        assert row["execution_status"] in {"not_started", "completed_e1_smoke"}
        assert row["evaluation_status"] in {"not_evaluated", "completed_smoke"}
        assert bool(resolved["metadata"]["long_training_started"]) is False
        assert bool(resolved["metadata"]["e600_started"]) is False
        assert bool(resolved["metadata"]["e600_completed"]) is False
        assert resolved["metadata"]["registry_config_id"] == row["config_id"]
        assert str(resolved["export"]["output_dir"]) == row["output_dir"]
        assert str(resolved["run"]["memory_audit_jsonl"]) == row["memory_audit_jsonl"]
        assert not (ROOT / row["output_dir"]).exists()
        assert not (ROOT / row["memory_audit_jsonl"]).exists()
        assert resolved["model"]["scale_pooling"] == row["scale_pooling"]
        assert int(resolved["model"]["scale_head_depth"]) == int(row["scale_head_depth"])
        assert bool(resolved["model"]["pooled_latent_stop_gradient"]) == (
            row["pooled_latent_stop_gradient"] == "true"
        )
        assert float(resolved["optimizer"]["scale_head_lr_multiplier"]) == float(
            row["scale_head_lr_multiplier"]
        )
        if row["scale_pooling"] == "qk_gated":
            assert row["qk_input_provenance"] == "raw_coords_k_q_bc_only"
            assert resolved["metadata"]["qk_region_feature_schema"] == "heat3d_v5_qk_region_features_v1"
        command = build_training_command(resolved, python_executable="python")
        text = shlex.join(command)
        commands[row["config_id"]] = text
        assert "--epochs 1" in text and "--batch-size 28" in text
        assert "--prediction-split valid_iid" in text
        assert "--init-checkpoint" not in command
        assert "--scale-pooling " + row["scale_pooling"] in text
        assert "--scale-head-depth " + row["scale_head_depth"] in text
        if row["pooled_latent_stop_gradient"] == "true":
            assert "--pooled-latent-stop-gradient" in command
        else:
            assert "--no-pooled-latent-stop-gradient" in command
        assert "--scale-head-lr-multiplier " + row["scale_head_lr_multiplier"] in text
    assert set(SCALE_POOLING_MODES) == {
        "mean", "mean_std", "mean_max", "pre_film_mean_std", "latent_attention", "qk_gated"
    }
    assert RIGNO.scale_head_depth == 1
    assert RIGNO.pooled_latent_stop_gradient is False
    # The N3 YAML leaves all Gate 6F controls absent, therefore model defaults
    # retain the old parameter graph and output route.  The cache exporter
    # performs the corresponding exact parameter/output replay check on devbox.
    assert "scale_head_depth" not in n3["model"]
    assert "pooled_latent_stop_gradient" not in n3["model"]
    assert "scale_head_lr_multiplier" not in n3["optimizer"]
    _assert_qk_input_only()
    v4_registry = (ROOT / "configs/heat3d_v4/run_registry.csv").read_text(encoding="utf-8")
    assert not any(config_id in v4_registry for config_id in EXPECTED_IDS)
    print(json.dumps({
        "status": "passed",
        "registry": str(REGISTRY),
        "config_count": len(rows),
        "pooling_modes": list(SCALE_POOLING_MODES),
        "qk_region_feature_count": len(QK_REGION_FEATURES),
        "training_started": False,
        "long_training_started": False,
        "commands": commands,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
