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

import jax
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
    _model_apply,
    _training_edge_masking_key,
)


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_36_gate6m_v32_epoch_regroup_e600.yaml"
SMOKE = ROOT / "configs/heat3d_v5/generated/V4P5_37_gate6n_v36_r2r_mask_p010_smoke_e3.yaml"
E600 = ROOT / "configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p010_e600.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6n_edge_masking_registry.csv"
UPSTREAM = ROOT / "configs/heat3d_v5/gate6n/gate6n_upstream_evidence.json"
DEGREES = ROOT / "configs/heat3d_v5/gate6n/gate6n_graph_degree_audit.json"
SMOKE_RESULT = ROOT / "configs/heat3d_v5/gate6n/gate6n_e3_smoke.json"
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


def _check_upstream_and_degrees() -> None:
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    assert upstream["commit"] == "3e4b307c90f34237d0c1e5e497d4301116e9c3db"
    assert upstream["upstream_training_default"] == 0.5
    assert upstream["upstream_training_edge_types"] == ["p2r", "r2r", "r2p"]
    assert upstream["upstream_experimental_strengths"] == [0.0, 0.5, 0.8]
    degrees = json.loads(DEGREES.read_text(encoding="utf-8"))
    assert degrees["train_sample_count"] == 672
    assert degrees["node_count"] == 1024
    assert degrees["coordinate_topology_count"] == 1
    topology = degrees["topologies"][0]
    assert topology["degree"]["r2r"]["zero_in_degree_count"] == 0
    chosen = {
        row["rate"]: row for row in topology["r2r_mask_rate_audit"]
    }[0.1]
    assert chosen["seed_count"] == 128
    assert chosen["zero_in_degree_max"] == 0
    assert chosen["same_seed_reproducible"] is True
    assert chosen["distinct_seed_changes_mask"] is True
    unsafe = {
        row["rate"]: row for row in topology["r2r_mask_rate_audit"]
    }[0.5]
    assert unsafe["zero_in_degree_sum"] > 0


def _check_runner_prng_and_scope() -> None:
    assert edge_masking_probabilities(0.1, "r2r_only") == (0.0, 0.1, 0.0)
    left = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.1}, model_seed=0, epoch=9, batch_index=4
        )
    )
    repeat = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.1}, model_seed=0, epoch=9, batch_index=4
        )
    )
    changed = np.asarray(
        _training_edge_masking_key(
            {"p_edge_masking": 0.1}, model_seed=0, epoch=9, batch_index=5
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
        assert config["model"]["p_edge_masking"] == 0.1
        assert config["model"]["edge_masking_scope"] == "r2r_only"
        assert config["export"]["prediction_split"] == "valid_iid"
        command = build_training_command(config, python_executable="python")
        text = shlex.join(command)
        assert "--p-edge-masking 0.1" in text
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
        assert float(row["p_edge_masking"]) == 0.1
    formal = by_id[e600["config_id"]]
    assert formal["execution_status"] == "not_started"
    assert formal["training_started"] == "false"
    assert formal["formal_e600_started"] == "false"
    assert formal["launch_policy"] == "explicit_user_instruction_only"


def _check_smoke() -> None:
    result = json.loads(SMOKE_RESULT.read_text(encoding="utf-8"))
    assert result["schema_version"] == "heat3d_v5_gate6n_e3_smoke_v1"
    assert result["status"] == "completed_e3_smoke"
    assert result["config_id"] == "V4P5_37_gate6n_v36_r2r_mask_p010_smoke_e3"
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
