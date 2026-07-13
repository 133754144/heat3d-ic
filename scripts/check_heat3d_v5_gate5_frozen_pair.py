#!/usr/bin/env python3
"""Assert the N0/N1 frozen e600 pair has one scientific difference."""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


N0 = ROOT / "configs/heat3d_v5/generated/V4P5_05_native_physics_only.yaml"
N1 = ROOT / "configs/heat3d_v5/generated/V4P5_06_native_pooled_latent.yaml"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate5_native_preflight_registry.csv"
PREFLIGHT = ROOT / "configs/heat3d_v5/v5_gate5_e600_preflight.json"
LOSS_FREEZE = ROOT / "configs/heat3d_v5/v5_gate5_loss_freeze.json"
IDENTITY_FIELDS = {
    "run": {
        "final_probe_output_dir",
        "post_training_diagnostics_output_dir",
        "profile_timing_json",
        "memory_audit_jsonl",
    },
    "export": {"output_dir", "run_name"},
}


def _resolved(path: Path) -> dict:
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = resolve_inherited_yaml(source, path)
    validate_v2_config(payload, config_path=path)
    return payload


def _scientific_payload(config: dict) -> dict:
    payload = copy.deepcopy(config)
    for field in ("schema_version", "config_id", "description", "metadata"):
        payload.pop(field, None)
    payload["model"].pop("scale_head_mode", None)
    for section, fields in IDENTITY_FIELDS.items():
        for field in fields:
            payload[section].pop(field, None)
    return payload


def main() -> int:
    n0_source = yaml.safe_load(N0.read_text(encoding="utf-8"))
    n1_source = yaml.safe_load(N1.read_text(encoding="utf-8"))
    n0, n1 = _resolved(N0), _resolved(N1)
    assert n0["model"]["scale_head_mode"] == "physics_only"
    assert n1["model"]["scale_head_mode"] == "physics_plus_pooled_latent"
    assert n0["run"]["epochs"] == n1["run"]["epochs"] == 600
    assert n0["export"]["selection_metric"] == n1["export"]["selection_metric"] == "valid_base_mse"
    assert _scientific_payload(n0) == _scientific_payload(n1)
    weight_fields = (
        "native_shape_cv_weight",
        "native_log_scale_weight",
        "native_relative_field_weight",
        "native_raw_field_weight",
    )
    weights = [float(n0["loss"][field]) for field in weight_fields]
    assert weights == [float(n1["loss"][field]) for field in weight_fields]
    freeze = json.loads(LOSS_FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "frozen"
    assert list(freeze["frozen_weights"].values()) == weights
    assert freeze["e600_started"] is False
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert preflight["status"] == "passed"
    assert preflight["formal_performance_result"] is False
    assert preflight["e600_started"] is False
    registry_rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert [row["config_id"] for row in registry_rows] == [
        n0_source["config_id"],
        n1_source["config_id"],
    ]
    for row in registry_rows:
        assert row["execution_smoke_status"] == "passed"
        assert row["calibration_status"] == "passed"
        assert row["frozen_loss_weights"] == "1|1|1|1"
        assert row["e600_status"] == "not_started"
        assert row["selection_metric"] == "valid_base_mse"
        assert row["formal_performance_result"] == "false"
        assert row["launch_policy"] == "explicit_user_instruction_only"
        for field in (
            "execution_smoke_config", "calibration_config", "loss_freeze_json",
            "preflight_json", "preflight_md", "final_e600_yaml",
        ):
            assert (ROOT / row[field]).is_file(), f"missing registry path: {row[field]}"
    commands = {
        "N0": " ".join(build_training_command(n0, python_executable="python")),
        "N1": " ".join(build_training_command(n1, python_executable="python")),
    }
    for command in commands.values():
        assert "--epochs 600" in command
        assert "--selection-metric valid_base_mse" in command
    print(json.dumps({
        "status": "passed",
        "allowed_scientific_difference": "model.scale_head_mode",
        "N0_scale_head_mode": n0["model"]["scale_head_mode"],
        "N1_scale_head_mode": n1["model"]["scale_head_mode"],
        "shared_loss_weights": weights,
        "e600_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
