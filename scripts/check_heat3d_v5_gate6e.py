#!/usr/bin/env python3
"""Validate the Gate 6E branch-only missing-cell plan without training."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6e_scratch_loss_registry.csv"
N3_PATH = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
GATE6E_PATH = ROOT / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml"
LOSS_FIELDS = (
    "native_shape_cv_weight", "native_log_scale_weight",
    "native_relative_field_weight", "native_raw_field_weight",
)
ALLOWED_SCIENTIFIC_DIFFS = {
    "loss.native_shape_cv_weight",
    "loss.native_log_scale_weight",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        paths: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths |= _diff(left[key], right[key], path)
        return paths
    return set() if left == right else {prefix}


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["config_id"] == "V4P5_13_gate6e_scratch_branch_rebalance"
    assert row["phase"] == "v5_gate6e"
    assert row["plan_status"] == "frozen_prepared"
    assert row["execution_status"] == "not_started"
    assert row["evaluation_status"] == "not_evaluated"
    assert row["training_started"] == "false"
    assert row["launch_policy"] == "explicit_user_instruction_only"
    assert row["forbidden_access_roles"] == "test_iid|hard_train_holdout|hard_challenge_valid|hard_challenge_test"
    n3, gate6e = _resolved(N3_PATH), _resolved(GATE6E_PATH)
    assert ROOT / row["generated_yaml"] == GATE6E_PATH
    assert ROOT / row["baseline_yaml"] == N3_PATH
    assert row["baseline_config_id"] == "V4P5_07_native_pooled_latent_global_film"
    differences = _diff(_science(n3), _science(gate6e))
    assert differences == ALLOWED_SCIENTIFIC_DIFFS, differences
    weights = [float(gate6e["loss"][field]) for field in LOSS_FIELDS]
    assert weights == [1.5, 0.5, 1.0, 1.0]
    assert row["loss_weights"] == "1.5|0.5|1|1"
    assert gate6e["run"]["init_checkpoint"] is None
    assert gate6e["run"]["epochs"] == 600 and gate6e["run"]["batch_size"] == 28
    assert gate6e["optimizer"] == n3["optimizer"]
    assert gate6e["optimizer"]["multi_seed"] == []
    assert gate6e["dataset"] == n3["dataset"]
    assert gate6e["graph"] == n3["graph"]
    assert gate6e["model"] == n3["model"]
    assert gate6e["export"]["selection_metric"] == "valid_base_mse"
    assert gate6e["export"]["prediction_split"] == "valid_iid"
    identities = {
        "output_dir": gate6e["export"]["output_dir"],
        "run_name": gate6e["export"]["run_name"],
        "log_path": row["log_path"],
        "final_probe_output_dir": gate6e["run"]["final_probe_output_dir"],
        "post_training_diagnostics_output_dir": gate6e["run"]["post_training_diagnostics_output_dir"],
    }
    assert all(row[name] == value for name, value in identities.items())
    assert all(not (ROOT / value).exists() for value in identities.values() if value != identities["run_name"])
    metadata = gate6e["metadata"]
    assert metadata["training_started"] is False
    assert metadata["fit_roles"] == ["train"]
    assert metadata["normalization_fit_roles"] == ["train"]
    assert metadata["selection_roles"] == ["valid_iid"]
    assert metadata["forbidden_access_roles"] == [
        "test_iid", "hard_train_holdout", "hard_challenge_valid", "hard_challenge_test",
    ]
    command = build_training_command(gate6e, python_executable="python")
    joined = shlex.join(command)
    for fragment in (
        "--epochs 600", "--batch-size 28", "--global-context-mode film",
        "--native-output-mode native_shape_scale",
        "--scale-head-mode physics_plus_pooled_latent",
        "--native-shape-cv-weight 1.5", "--native-log-scale-weight 0.5",
        "--native-relative-field-weight 1.0", "--native-raw-field-weight 1.0",
        "--prediction-split valid_iid", "--selection-metric valid_base_mse",
    ):
        assert fragment in joined, fragment
    assert "--init-checkpoint" not in command
    v4_registry = (ROOT / "configs/heat3d_v4/run_registry.csv").read_text(encoding="utf-8")
    assert row["config_id"] not in v4_registry
    paired = json.loads(
        (ROOT / "configs/heat3d_v5/gate6d/n3_l2_valid_paired.json").read_text(encoding="utf-8")
    )
    sse = paired["true_delta_point_sse_attribution"]
    assert all(
        quartile["l2_minus_n3_point_sse_K2"] > 0.0 for quartile in sse["quartiles"][:3]
    )
    assert sse["quartiles"][3]["l2_minus_n3_point_sse_K2"] < 0.0
    assert sse["q1_q3_overall_regressed"] is True
    assert sse["q4_provides_all_net_point_global_improvement"] is True
    inference = paired["paired_inference"]
    assert inference["seed"] == 2026071502
    assert inference["bootstrap_resamples"] == inference["permutation_resamples"] == 20000
    ensemble = json.loads(
        (ROOT / "configs/heat3d_v5/gate6e/valid_only_ensemble_audit.json").read_text(encoding="utf-8")
    )
    assert ensemble["roles_accessed"] == ["valid_iid"]
    assert ensemble["forbidden_roles_accessed"] == []
    assert ensemble["training_started"] is False and ensemble["model_inference_run"] is False
    assert ensemble["gate6e_config_not_modified_from_ensemble"] is True
    assert ensemble["gate6e_config_sha256"] == _sha256(GATE6E_PATH)
    assert len(ensemble["ensembles"]) == 9
    assert {float(item["alpha"]) for item in ensemble["ensembles"]} == {0.25, 0.5, 0.75}
    for item in ensemble["ensembles"]:
        for field in (
            "valid_base_mse", "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct", "raw_cv_weighted_rmse_K",
        ):
            assert math.isfinite(float(item[field])) and float(item[field]) >= 0.0
    sealed = json.loads(
        (ROOT / "configs/heat3d_v5/gate6d/sealed_iid_contract.json").read_text(encoding="utf-8")
    )
    assert sealed["status"] == "frozen_not_generated"
    assert sealed["model_inference_run"] is False and sealed["training_started"] is False
    print(json.dumps({
        "status": "passed",
        "config_id": row["config_id"],
        "scientific_differences_from_n3": sorted(differences),
        "run_identity_differences": identities,
        "loss_weights": weights,
        "training_started": False,
        "multi_seed_started": False,
        "test_or_hard_accessed": False,
        "ensemble_roles_accessed": ensemble["roles_accessed"],
        "sealed_iid_status": sealed["status"],
        "dry_run_command": joined,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
