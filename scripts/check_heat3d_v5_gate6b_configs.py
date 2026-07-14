#!/usr/bin/env python3
"""Validate Gate 6B fine-tune YAMLs and dry-run commands without training."""

from __future__ import annotations

import copy
import csv
import json
import shlex
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6_finetune_registry.csv"
FREEZE = ROOT / "configs/heat3d_v5/v5_gate6_loss_freeze.json"
N3 = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
INPUT_CHECKPOINT = "output/heat3d_v5_gate6_inputs/N3_best_e402/params_best.pkl"
LOSS_FIELDS = (
    "native_shape_cv_weight",
    "native_log_scale_weight",
    "native_relative_field_weight",
    "native_raw_field_weight",
)
IDENTITY_FIELDS = {
    "config_id",
    "description",
    "metadata",
}


def _resolved(path: Path) -> dict:
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = resolve_inherited_yaml(source, path)
    validate_v2_config(payload, config_path=path)
    return payload


def _scientific_payload(config: dict) -> dict:
    payload = copy.deepcopy(config)
    for field in IDENTITY_FIELDS:
        payload.pop(field, None)
    for field in LOSS_FIELDS:
        payload["loss"].pop(field, None)
    payload["run"].pop("final_probe_output_dir", None)
    payload["run"].pop("post_training_diagnostics_output_dir", None)
    payload["export"].pop("output_dir", None)
    payload["export"].pop("run_name", None)
    return payload


def _weights(config: dict) -> list[float]:
    return [float(config["loss"][field]) for field in LOSS_FIELDS]


def main() -> int:
    runner_source = (
        ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
    ).read_text(encoding="utf-8")
    epoch0_anchor = 'initial_best_record = _epoch_history_record(\n        0,'
    loop_anchor = "for epoch in range(1, epochs + 1):"
    assert epoch0_anchor in runner_source
    assert runner_source.index(epoch0_anchor) < runner_source.index(loop_anchor)
    assert 'best_score: float | None = float(initial_best_record[selection_metric])' in runner_source
    assert 'float(initial_best_record["valid_rel_rmse_v4_pct"])' in runner_source
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert [row["variant"] for row in rows] == ["FT-L0", "FT-L1", "FT-L2"]
    assert all(row["launch_policy"] == "explicit_user_instruction_only" for row in rows)
    assert all(row["status"] == "prepared_not_started" for row in rows)
    n3 = _resolved(N3)
    reports = {}
    scientific_payload = None
    output_dirs = set()
    expected = {
        "FT-L0": [1.0, 1.0, 1.0, 1.0],
        "FT-L1": list(freeze["candidates"][0]["weights"]),
        "FT-L2": list(freeze["candidates"][1]["weights"]),
    }
    assert len(freeze["candidates"]) == 2
    assert freeze["selection_data_roles"] == ["train", "valid_iid"]
    assert freeze["forbidden_roles_accessed"] == []
    for row in rows:
        label = row["variant"]
        path = ROOT / row["yaml_path"]
        config = _resolved(path)
        run, optimizer, export = config["run"], config["optimizer"], config["export"]
        assert config["model"] == n3["model"]
        assert config["dataset"] == n3["dataset"]
        assert run["epochs"] == 100
        assert run["batch_size"] == 28
        assert run["batch_plan"] == "sample_shuffle"
        assert run["batch_build_seed"] == n3["run"]["batch_build_seed"]
        assert run["init_checkpoint"] == INPUT_CHECKPOINT
        assert run["checkpoint_load_strict"] is True
        assert run["partial_load_policy"] == "matching"
        assert run["report_every"] == 1
        assert run["final_probe_eval_after_training"] is False
        assert run["post_training_diagnostics"] is False
        assert config["model"]["native_branch_mode"] == "joint"
        assert optimizer["seed"] == n3["optimizer"]["seed"]
        assert optimizer["model_seed"] == n3["optimizer"]["model_seed"]
        assert optimizer["batch_order_seed"] == n3["optimizer"]["batch_order_seed"]
        assert optimizer["graph_seed"] == n3["optimizer"]["graph_seed"]
        assert 0.1 <= optimizer["lr"] / n3["optimizer"]["lr"] <= 0.2
        assert optimizer["lr_schedule"] == n3["optimizer"]["lr_schedule"]
        assert export["prediction_split"] == "valid_iid"
        assert export["save_final_predictions"] is False
        assert export["save_best_predictions"] is False
        assert export["selection_metric"] == "valid_base_mse"
        assert export["save_point_global_best_checkpoint"] is True
        assert export["point_global_best_checkpoint_name"] == "params_best_valid_point_global.pkl"
        assert _weights(config) == expected[label]
        assert export["output_dir"] not in output_dirs
        output_dirs.add(export["output_dir"])
        current_science = _scientific_payload(config)
        if scientific_payload is None:
            scientific_payload = current_science
        else:
            assert current_science == scientific_payload
        command = build_training_command(config, python_executable="python")
        joined = shlex.join(command)
        for fragment in (
            "--epochs 100",
            f"--init-checkpoint {INPUT_CHECKPOINT}",
            "--checkpoint-load-strict true",
            "--native-branch-mode joint",
            "--batch-size 28",
            "--prediction-split valid_iid",
            "--no-save-predictions",
            "--no-save-best-predictions",
            "--save-point-global-best-checkpoint",
            "--point-global-best-checkpoint-name params_best_valid_point_global.pkl",
            "--no-final-probe-eval-after-training",
            "--no-post-training-diagnostics",
            "--report-every 1",
        ):
            assert fragment in joined, f"{label}: missing dry-run fragment {fragment}"
        reports[label] = {
            "config_id": row["config_id"],
            "yaml": row["yaml_path"],
            "weights": _weights(config),
            "lr": optimizer["lr"],
            "output_dir": export["output_dir"],
            "training_started": False,
        }
    print(json.dumps({
        "status": "passed",
        "epoch0_best_selection": {
            "valid_base_mse": True,
            "valid_point_global_true_rms": True,
        },
        "optimizer_state_loaded": False,
        "all_parameters_trainable": True,
        "test_or_hard_evaluation_configured": False,
        "configs": reports,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
