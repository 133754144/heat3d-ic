#!/usr/bin/env python3
"""Validate Gate 6N upstream evidence, runner scope, configs, and e3 smoke."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

csv.field_size_limit(sys.maxsize)

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.models.rigno import edge_masking_probabilities  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _build_optax_state,
    _loss_components,
    _model_apply,
    _training_edge_masking_key,
)


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_36_gate6m_v32_epoch_regroup_e600.yaml"
SMOKE = ROOT / "configs/heat3d_v5/generated/V4P5_37_gate6n_v36_r2r_mask_p005_smoke_e3.yaml"
E600 = ROOT / "configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p005_e600.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6n_edge_masking_registry.csv"
UPSTREAM = ROOT / "configs/heat3d_v5/gate6n/gate6n_upstream_evidence.json"
DEGREES = ROOT / "configs/heat3d_v5/gate6n/gate6n_graph_degree_audit.json"
SMOKE_RESULT = ROOT / "configs/heat3d_v5/gate6n/gate6n_e3_smoke.json"
CLOSEOUT = ROOT / "configs/heat3d_v5/gate6n/gate6n_closeout.json"
P0_REGRESSION = ROOT / "configs/heat3d_v5/gate6n/gate6n_p0_runner_regression.json"
RUNNER = ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
FORBIDDEN = (
    "test_iid|hard_train_holdout|hard_challenge_valid|"
    "hard_challenge_test|sealed_iid"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


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


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    leaves = [
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    ]
    return max(leaves, default=0.0)


def _p0_runner_regression_payload() -> dict[str, Any]:
    """Compare the legacy unkeyed update with the new runner at p=0."""

    class ToyModel:
        def apply(
            self,
            variables: dict[str, Any],
            *,
            inputs: Any,
            graphs: Any,
            global_context: Any,
            key: Any,
        ) -> Any:
            del graphs, global_context, key
            params = variables["params"]
            return inputs * params["weight"] + params["bias"]

    params = {
        "weight": jnp.asarray(0.75, dtype=jnp.float32),
        "bias": jnp.asarray(-0.125, dtype=jnp.float32),
    }
    inputs = jnp.asarray(
        [
            [[[-1.0], [0.5], [2.0]]],
            [[[1.5], [-0.25], [0.75]]],
        ],
        dtype=jnp.float32,
    )
    target = jnp.asarray(
        [
            [[[-0.5], [0.25], [1.25]]],
            [[[0.75], [0.0], [0.5]]],
        ],
        dtype=jnp.float32,
    )
    stats = {
        "target_delta_mean": jnp.asarray(0.25, dtype=jnp.float32),
        "target_delta_std": jnp.asarray(1.75, dtype=jnp.float32),
    }
    group = {
        "inputs": inputs,
        "graphs": None,
        "global_context": None,
        "target_normalized": target,
        "target_delta_raw": (
            target * stats["target_delta_std"] + stats["target_delta_mean"]
        ),
    }
    loss_config = {
        "loss_mode": "mse",
        "background_quantile": 0.25,
        "hotspot_quantile": 0.75,
    }
    p0_key = _training_edge_masking_key(
        {"p_edge_masking": 0.0},
        model_seed=0,
        epoch=1,
        batch_index=1,
    )

    def legacy_loss(current_params: Any) -> Any:
        return _loss_components(
            ToyModel(),
            current_params,
            [group],
            stats,
            loss_config,
        )["total_loss"]

    def revised_p0_loss(current_params: Any) -> Any:
        return _loss_components(
            ToyModel(),
            current_params,
            [group],
            stats,
            loss_config,
            key=p0_key,
        )["total_loss"]

    old_loss, old_grad = jax.value_and_grad(legacy_loss)(params)
    new_loss, new_grad = jax.value_and_grad(revised_p0_loss)(params)
    lr_config = {
        "lr": 0.0005,
        "lr_schedule": "warmup_cosine",
        "warmup_epochs": 10,
        "min_lr": 5.0e-05,
        "lr_init": 1.0e-05,
        "lr_peak": 2.0e-04,
        "lr_base": 1.0e-05,
        "lr_lowr": 1.0e-06,
        "pct_start": 0.02,
        "pct_final": 0.1,
        "updates_per_epoch": 24,
    }
    optimizer_config = {
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
    }
    old_optimizer = _build_optax_state(
        params,
        epochs=600,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
    )
    new_optimizer = _build_optax_state(
        params,
        epochs=600,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
    )
    old_updates, _ = old_optimizer["tx"].update(
        old_grad,
        old_optimizer["state"],
        params,
    )
    new_updates, _ = new_optimizer["tx"].update(
        new_grad,
        new_optimizer["state"],
        params,
    )
    old_params = old_optimizer["apply_updates"](params, old_updates)
    new_params = new_optimizer["apply_updates"](params, new_updates)
    payload = {
        "schema_version": "heat3d_v5_gate6n_p0_runner_regression_v1",
        "comparison": "legacy_unkeyed_vs_revised_runner_p_edge_masking_zero",
        "optimizer": "V36 AdamW first update",
        "p0_training_key_is_none": p0_key is None,
        "legacy_loss": float(old_loss),
        "revised_loss": float(new_loss),
        "loss_abs_difference": abs(float(old_loss) - float(new_loss)),
        "gradient_max_abs_difference": _tree_max_abs_difference(old_grad, new_grad),
        "update_max_abs_difference": _tree_max_abs_difference(
            old_updates,
            new_updates,
        ),
        "updated_parameter_max_abs_difference": _tree_max_abs_difference(
            old_params,
            new_params,
        ),
    }
    payload["exact_match"] = bool(
        payload["p0_training_key_is_none"]
        and payload["loss_abs_difference"] == 0.0
        and payload["gradient_max_abs_difference"] == 0.0
        and payload["update_max_abs_difference"] == 0.0
        and payload["updated_parameter_max_abs_difference"] == 0.0
    )
    return payload


def _check_upstream_and_degrees() -> None:
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    assert upstream["commit"] == "3e4b307c90f34237d0c1e5e497d4301116e9c3db"
    assert upstream["upstream_training_default"] == 0.5
    assert upstream["upstream_training_edge_types"] == ["p2r", "r2r", "r2p"]
    assert upstream["upstream_experimental_strengths"] == [0.0, 0.5, 0.8]
    degrees = json.loads(DEGREES.read_text(encoding="utf-8"))
    assert degrees["schema_version"] == "heat3d_v5_gate6n_graph_degree_audit_v2"
    assert degrees["key_schedule"] == "exact_native_processor_call_chain"
    assert degrees["train_sample_count"] == 672
    assert degrees["node_count"] == 1024
    assert degrees["coordinate_topology_count"] == 1
    topology = degrees["topologies"][0]
    assert topology["degree"]["r2r"]["zero_in_degree_count"] == 0
    chosen = {
        row["rate"]: row for row in topology["r2r_mask_rate_audit"]
    }[0.05]
    assert chosen["seed_count"] == 128
    assert chosen["zero_in_degree_max"] == 0
    assert chosen["same_seed_reproducible"] is True
    assert chosen["distinct_seed_changes_mask"] is True
    schedules = {
        row["rate"]: row for row in topology["planned_e600_key_schedules"]
    }
    assert set(schedules) == {0.02, 0.05}
    chosen_schedule = schedules[0.05]
    assert chosen_schedule["mask_count"] == 14400
    assert chosen_schedule["group_index"] == 0
    assert chosen_schedule["same_schedule_reproducible"] is True
    assert chosen_schedule["zero_in_degree_sum"] == 0
    assert chosen_schedule["zero_out_degree_sum"] == 0
    assert chosen_schedule["isolated_node_sum"] == 0
    assert chosen_schedule["disconnected_mask_count"] == 0
    assert chosen_schedule["weak_component_max"] == 1
    assert chosen_schedule["all_masks_safe"] is True
    assert all(
        value is None for value in chosen_schedule["first_failures"].values()
    )
    assert len(chosen_schedule["processor_key_sha256"]) == 64
    assert len(chosen_schedule["mask_sequence_sha256"]) == 64
    unsafe = {
        row["rate"]: row for row in topology["r2r_mask_rate_audit"]
    }[0.5]
    assert unsafe["zero_in_degree_sum"] > 0


def _check_runner_prng_and_scope() -> None:
    assert edge_masking_probabilities(0.05, "r2r_only") == (0.0, 0.05, 0.0)
    left = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.05}, model_seed=0, epoch=9, batch_index=4
        )
    )
    repeat = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.05}, model_seed=0, epoch=9, batch_index=4
        )
    )
    changed = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.05}, model_seed=0, epoch=9, batch_index=5
        )
    )
    assert np.array_equal(left, repeat)
    assert not np.array_equal(left, changed)
    assert (
        _training_edge_masking_key(
            {"p_edge_masking": 0.0}, model_seed=0, epoch=9, batch_index=4
        )
        is None
    )

    class Spy:
        def __init__(self) -> None:
            self.keys: list[Any] = []

        def apply(self, variables: Any, **kwargs: Any) -> Any:
            del variables
            self.keys.append(kwargs.get("key"))
            return kwargs.get("key")

    group = {
        "inputs": object(),
        "graphs": object(),
        "global_context": None,
    }
    spy = Spy()
    assert _model_apply(spy, object(), group) is None
    explicit = jax.random.PRNGKey(3)
    returned = _model_apply(spy, object(), group, key=explicit)
    assert np.array_equal(np.asarray(returned), np.asarray(explicit))
    assert spy.keys[0] is None
    assert np.array_equal(np.asarray(spy.keys[1]), np.asarray(explicit))

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    fit = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_fit_once"
    )
    key_build_calls = [
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_training_edge_masking_key"
    ]
    assert len(key_build_calls) == 2
    loss_calls = [
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_loss_components"
    ]
    keyed = [
        node for node in loss_calls if any(item.arg == "key" for item in node.keywords)
    ]
    unkeyed = [
        node for node in loss_calls if not any(item.arg == "key" for item in node.keywords)
    ]
    assert len(keyed) == 2
    assert len(unkeyed) >= 6

    actual = _p0_runner_regression_payload()
    frozen = json.loads(P0_REGRESSION.read_text(encoding="utf-8"))
    assert frozen == actual
    assert actual["exact_match"] is True


def _check_configs_and_registry() -> None:
    baseline = _resolved(BASELINE)
    smoke = _resolved(SMOKE)
    e600 = _resolved(E600)
    assert _diff(_science(baseline), _science(e600)) == {
        "model.p_edge_masking",
        "model.edge_masking_scope",
    }
    assert _diff(_science(e600), _science(smoke)) == {
        "run.epochs",
        "run.post_training_diagnostics",
    }
    for config, epochs in ((smoke, 3), (e600, 600)):
        assert config["run"]["epochs"] == epochs
        assert config["run"]["batch_size"] == 28
        assert config["run"]["validation_batch_size"] == 32
        assert config["run"]["prediction_batch_size"] == 32
        assert config["run"]["epoch_wise_batch_regrouping"] is True
        assert config["model"]["p_edge_masking"] == 0.05
        assert config["model"]["edge_masking_scope"] == "r2r_only"
        assert config["metadata"]["edge_masking_key_schedule"] == (
            "epoch_batch_fold_in_then_group_index_fold_in_then_"
            "native_processor_split"
        )
        assert config["export"]["prediction_split"] == "valid_iid"
        command = build_training_command(config, python_executable="python")
        text = shlex.join(command)
        assert "--p-edge-masking 0.05" in text
        assert "--edge-masking-scope r2r_only" in text
        assert "--batch-size 28" in text
        assert "--validation-batch-size 32" in text
        assert "--prediction-batch-size 32" in text
        dry = subprocess.run(
            [
                sys.executable,
                "scripts/run_heat3d_v4_config.py",
                "--config",
                str((SMOKE if epochs == 3 else E600).relative_to(ROOT)),
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert dry.stdout.strip() == text

    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_id = {row["config_id"]: row for row in rows}
    assert set(by_id) == {smoke["config_id"], e600["config_id"]}
    for row in rows:
        assert row["fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["test_accessed"] == "false"
        assert row["hard_accessed"] == "false"
        assert row["sealed_iid_accessed"] == "false"
        assert row["edge_masking_scope"] == "r2r_only"
        assert float(row["p_edge_masking"]) == 0.05
        assert row["mask_audit_path"] == (
            "configs/heat3d_v5/gate6n/gate6n_graph_degree_audit.json"
        )
        assert int(row["exact_mask_count"]) == 14400
        assert int(row["zero_in_degree_events"]) == 0
        assert int(row["zero_out_degree_events"]) == 0
        assert int(row["isolated_node_events"]) == 0
        assert int(row["disconnected_mask_count"]) == 0
        assert row["p0_regression_exact"] == "true"
    formal = by_id[e600["config_id"]]
    assert formal["execution_status"] == "not_started"
    assert formal["training_started"] == "false"
    assert formal["formal_e600_started"] == "false"
    assert formal["launch_policy"] == "explicit_user_instruction_only"
    smoke_row = by_id[smoke["config_id"]]
    assert smoke_row["execution_status"] == "completed_e3_smoke"
    assert smoke_row["training_started"] == "true"
    assert smoke_row["formal_e600_started"] == "false"
    assert smoke_row["smoke_commit"] == "c792a61"
    assert len(smoke_row["smoke_checkpoint_sha256"]) == 64
    assert float(smoke_row["smoke_reload_max_abs_error_K"]) <= 0.02
    assert smoke["metadata"]["status"] == "completed_e3_smoke"
    assert smoke["metadata"]["training_started"] is True
    assert smoke["metadata"]["formal_e600_started"] is False


def _check_smoke() -> None:
    result = json.loads(SMOKE_RESULT.read_text(encoding="utf-8"))
    assert result["schema_version"] == "heat3d_v5_gate6n_e3_smoke_v1"
    assert result["status"] == "completed_e3_smoke"
    assert result["config_id"] == "V4P5_37_gate6n_v36_r2r_mask_p005_smoke_e3"
    assert result["epochs_completed"] == 3
    assert result["node_count"] == 1024
    assert result["train_batch_size"] == 28
    assert result["finite"] is True
    assert result["checkpoint_reload"]["passed"] is True
    assert result["roles_accessed"] == ["train", "valid_iid"]
    assert result["forbidden_roles_accessed"] == []
    assert result["formal_e600_started"] is False
    checkpoint = result["remote_artifact"]
    assert checkpoint["host"] == "wsl2"
    assert checkpoint["path"].endswith("/params_final.pkl")
    assert len(checkpoint["sha256"]) == 64
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    assert closeout["status"] == "completed_revised_preflight_e3_only"
    assert closeout["formal_e600_started"] is False
    assert closeout["devbox_connected"] is False
    assert closeout["v36_run_directory_modified"] is False
    assert closeout["rate_selection"]["exact_mask_count"] == 14400
    assert closeout["rate_selection"]["zero_in_degree_events"] == 0
    assert closeout["rate_selection"]["zero_out_degree_events"] == 0
    assert closeout["rate_selection"]["isolated_node_events"] == 0
    assert closeout["rate_selection"]["disconnected_mask_count"] == 0
    assert closeout["p0_runner_regression"]["exact_match"] is True


def main() -> int:
    args = _args()
    _check_upstream_and_degrees()
    _check_runner_prng_and_scope()
    _check_configs_and_registry()
    if not args.preflight_only:
        _check_smoke()
    print(
        "Gate 6N checks passed "
        f"(preflight_only={str(args.preflight_only).lower()}, training_runs=0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
