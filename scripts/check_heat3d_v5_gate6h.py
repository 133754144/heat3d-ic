#!/usr/bin/env python3
"""Check the V13-based Gate 6H scratch scale-path e600 plans."""

from __future__ import annotations

import argparse
import copy
import csv
import json
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


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6h_v13_scale_ablation_registry.csv"
CONTRACT = ROOT / "configs/heat3d_v5/gate6h/v13_scratch_scale_ablation_contract.json"
V13 = ROOT / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml"
E1_SUMMARY = ROOT / "configs/heat3d_v5/gate6h/e1_smoke_summary.json"
RUNNER = ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
IDS = (
    "V4P5_28_gate6h_v13_stopgrad_scratch_e600",
    "V4P5_29_gate6h_v13_scale_attention_scratch_e600",
    "V4P5_30_gate6h_v13_deep_scale_head_scratch_e600",
)
FORBIDDEN = "test_iid|hard_train_holdout|hard_challenge_valid|hard_challenge_test|sealed_iid"
EXPECTED_MODEL_DIFFS = {
    IDS[0]: {"model.pooled_latent_stop_gradient": True},
    IDS[1]: {
        "model.pooled_latent_stop_gradient": True,
        "model.scale_attention_mode": "physics_gate",
    },
    IDS[2]: {"model.scale_head_depth": 3},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


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


def _scientific_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left, right = _flatten(baseline), _flatten(candidate)
    ignored = {
        "config_id",
        "description",
        "export.output_dir",
        "export.run_name",
        "run.final_probe_output_dir",
        "run.post_training_diagnostics_output_dir",
    }
    return {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
        and key not in ignored
        and not key.startswith("metadata.")
    }


def _runtime_baseline(v13: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    baseline = copy.deepcopy(v13)
    actual = contract["actual_lr_contract"]
    baseline["optimizer"].update({
        "name": actual["optimizer"],
        **{key: actual[key] for key in (
            "lr", "lr_schedule", "warmup_epochs", "min_lr",
            "second_stage_epoch", "second_stage_lr", "lr_init", "lr_peak",
            "lr_base", "lr_lowr", "pct_start", "pct_final",
            "gradient_clip_norm", "weight_decay",
        )},
    })
    baseline["run"].update({
        "epochs": 600,
        "final_probe_eval_after_training": False,
        "post_training_diagnostics": True,
    })
    baseline["export"].update({
        "prediction_split": "valid_iid",
        "selection_metric": "valid_rel_rmse_v4_pct",
        "save_point_global_best_checkpoint": True,
        "point_global_best_checkpoint_name": "params_best_valid_point_global.pkl",
        "save_base_mse_best_checkpoint": True,
        "base_mse_best_checkpoint_name": "params_best_valid_base_mse.pkl",
        "save_sample_first_best_checkpoint": True,
        "sample_first_best_checkpoint_name": "params_best_valid_sample_first.pkl",
    })
    return baseline


def main() -> int:
    args = _args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert tuple(row["config_id"] for row in rows) == IDS
    assert "V4P5_22" not in REGISTRY.read_text(encoding="utf-8")

    metrics = contract["historical_audit"]["models"]
    assert metrics["V13"]["point_global_relative_rmse_pct"] < metrics["L2"]["point_global_relative_rmse_pct"]
    assert metrics["V13"]["point_global_relative_rmse_pct"] < metrics["N3"]["point_global_relative_rmse_pct"]
    assert metrics["V13"]["sample_first_cv_relative_rmse_pct"] < metrics["L2"]["sample_first_cv_relative_rmse_pct"]
    assert metrics["V13"]["sample_first_cv_relative_rmse_pct"] < metrics["N3"]["sample_first_cv_relative_rmse_pct"]
    assert metrics["V13"]["raw_cv_weighted_rmse_K"] < metrics["L2"]["raw_cv_weighted_rmse_K"]
    assert metrics["V13"]["raw_cv_weighted_rmse_K"] < metrics["N3"]["raw_cv_weighted_rmse_K"]

    v13 = _resolved(V13)
    baseline = _runtime_baseline(v13, contract)
    commands: dict[str, str] = {}
    diffs: dict[str, dict[str, Any]] = {}
    output_paths: set[str] = set()
    diagnostic_paths: set[str] = set()
    log_paths: set[str] = set()
    for row in rows:
        path = ROOT / row["generated_yaml"]
        resolved = _resolved(path)
        config_id = row["config_id"]
        assert resolved["dataset"] == v13["dataset"]
        assert resolved["graph"] == v13["graph"]
        assert resolved["loss"] == v13["loss"]
        assert resolved["run"]["epochs"] == int(row["epochs"]) == 600
        assert resolved["run"]["batch_size"] == int(row["batch_size"]) == 28
        assert resolved["run"]["init_checkpoint"] is None
        assert resolved["optimizer"]["multi_seed"] == []
        for seed in ("seed", "model_seed", "batch_order_seed", "graph_seed"):
            assert int(resolved["optimizer"][seed]) == int(row[seed]) == 0
        assert int(resolved["run"]["batch_build_seed"]) == int(row["batch_build_seed"]) == 0
        actual = contract["actual_lr_contract"]
        assert resolved["optimizer"]["name"] == actual["optimizer"] == row["optimizer"]
        for key in (
            "lr", "warmup_epochs", "min_lr", "second_stage_epoch", "second_stage_lr",
            "lr_init", "lr_peak", "lr_base", "lr_lowr", "pct_start", "pct_final",
            "gradient_clip_norm", "weight_decay",
        ):
            assert float(resolved["optimizer"][key]) == float(actual[key]) == float(row[key])
        assert resolved["optimizer"]["lr_schedule"] == actual["lr_schedule"] == row["lr_schedule"]
        assert resolved["export"]["prediction_split"] == row["selection_roles"] == "valid_iid"
        assert resolved["export"]["selection_metric"] == row["primary_selection"] == "valid_rel_rmse_v4_pct"
        for key in (
            "save_point_global_best_checkpoint",
            "save_base_mse_best_checkpoint",
            "save_sample_first_best_checkpoint",
        ):
            assert resolved["export"][key] is True
        assert resolved["run"]["final_probe_eval_after_training"] is False
        assert resolved["run"]["post_training_diagnostics"] is True
        assert row["final_probe_eval_after_training"] == "false"
        assert row["post_training_diagnostics"] == "true"
        assert row["fit_roles"] == row["normalization_fit_roles"] == "train"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["summary_persisted_before_replay"] == "true"
        assert row["launch_policy"] == "explicit_user_instruction_only"
        execution_status = row["execution_status"]
        if execution_status == "not_started":
            assert row["long_training_started"] == "false"
        else:
            assert execution_status in {"running_e600", "completed_e600"}
            assert row["long_training_started"] == "true"
        diffs[config_id] = _scientific_diff(baseline, resolved)
        assert diffs[config_id] == EXPECTED_MODEL_DIFFS[config_id], (config_id, diffs[config_id])

        command = build_training_command(resolved, python_executable="python")
        text = shlex.join(command)
        for flag in (
            "--epochs 600", "--batch-size 28", "--lr 0.0005",
            "--lr-schedule warmup_cosine", "--warmup-epochs 10", "--min-lr 5e-05",
            "--second-stage-epoch 0", "--second-stage-lr 0.0001",
            "--lr-init 1e-05", "--lr-peak 0.0002", "--lr-base 1e-05",
            "--lr-lowr 1e-06", "--pct-start 0.02", "--pct-final 0.1",
            "--prediction-split valid_iid", "--no-final-probe-eval-after-training",
            "--post-training-diagnostics", "--save-point-global-best-checkpoint",
            "--save-base-mse-best-checkpoint", "--save-sample-first-best-checkpoint",
        ):
            assert flag in text, (config_id, flag)
        assert "--init-checkpoint" not in command
        commands[config_id] = text
        assert row["output_dir"] not in output_paths
        assert row["post_training_diagnostics_output_dir"] not in diagnostic_paths
        assert row["log_path"] not in log_paths
        output_paths.add(row["output_dir"])
        diagnostic_paths.add(row["post_training_diagnostics_output_dir"])
        log_paths.add(row["log_path"])

    source = RUNNER.read_text(encoding="utf-8")
    pending_at = source.index('"status": "pending" if reload_entries')
    first_summary_write = source.index('_write_json(output_dir / "loss_summary.json", loss_summary)', pending_at)
    replay_at = source.index("checkpoint_prediction_reload_audit = _checkpoint_prediction_reload_audit(", first_summary_write)
    assert pending_at < first_summary_write < replay_at

    e1 = None
    if E1_SUMMARY.is_file():
        e1 = json.loads(E1_SUMMARY.read_text(encoding="utf-8"))
        assert e1["status"] == "completed"
        assert [row["config_id"] for row in e1["results"]] == list(IDS)
        assert e1["roles_accessed"] == ["train", "valid_iid"]
        assert e1["forbidden_roles_accessed"] == []
        assert e1["sealed_iid_accessed"] is False
        assert e1["long_training_started"] is False
        for result in e1["results"]:
            assert result["status"] == "passed"
            assert result["checkpoint_reload_passed"] is True
            assert result["summary_persisted_before_replay"] is True
            assert result["post_training_diagnostics_status"] == "completed"

    lifecycle_statuses = {row["config_id"]: row["execution_status"] for row in rows}
    payload = {
        "status": "passed",
        "config_count": len(rows),
        "baseline_best_single_model_confirmed": True,
        "actual_lr_contract": contract["actual_lr_contract"],
        "scientific_diffs": diffs,
        "summary_persisted_before_replay": True,
        "e1_smoke_checked": e1 is not None,
        "roles_accessed": [],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "long_training_started": any(
            row["long_training_started"] == "true" for row in rows
        ),
        "lifecycle_statuses": lifecycle_statuses,
        "commands": commands,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
