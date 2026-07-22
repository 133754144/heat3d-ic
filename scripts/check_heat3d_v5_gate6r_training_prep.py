#!/usr/bin/env python3
"""Validate Gate 6R V45/V46 preparation without starting e600."""

from __future__ import annotations

import argparse
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
from scripts.check_heat3d_v5_gate6q_training_prep import (  # noqa: E402
    _diff,
    _flatten,
    _objective_and_feature_fixtures,
    _semantic_defaults,
)
from scripts.smoke_heat3d_v5_gate6q_single_batch import run_smoke  # noqa: E402


V38 = ROOT / "configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p005_e600.yaml"
V42 = ROOT / "configs/heat3d_v5/generated/V4P5_42_gate6q_objective_only_e600.yaml"
V44 = ROOT / "configs/heat3d_v5/generated/V4P5_44_gate6q_xy_deepsets_e600.yaml"
CONFIGS = {
    "V45": ROOT / "configs/heat3d_v5/generated/V4P5_45_gate6r_deepsets_only_e600.yaml",
    "V46": ROOT / "configs/heat3d_v5/generated/V4P5_46_gate6r_objective_deepsets_e600.yaml",
}
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6r_training_registry.csv"
REPORT = ROOT / "configs/heat3d_v5/gate6r_training/gate6r_training_prep.json"
FORBIDDEN = {
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
    "sealed_iid",
}
INVARIANTS = (
    "dataset",
    "graph",
    "optimizer",
    "run.epochs",
    "run.batch_size",
    "run.validation_batch_size",
    "run.prediction_batch_size",
    "run.batch_plan",
    "run.batch_build_seed",
    "model.node_latent_size",
    "model.edge_latent_size",
    "model.processor_steps",
    "model.mlp_hidden_layers",
    "model.p_edge_masking",
    "model.edge_masking_scope",
)


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    validate_v2_config(resolved, config_path=path)
    return resolved


def _assert_invariants(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    flat_base = _flatten(baseline)
    flat_candidate = _flatten(candidate)
    for path in INVARIANTS:
        if path in {"dataset", "graph", "optimizer"}:
            assert candidate[path] == baseline[path]
        else:
            assert flat_candidate[path] == flat_base[path]
    assert candidate["run"]["init_checkpoint"] is None
    assert candidate["run"]["epochs"] == 600
    assert candidate["run"]["batch_size"] == 28
    assert candidate["model"]["p_edge_masking"] == 0.05
    assert candidate["model"]["edge_masking_scope"] == "r2r_only"
    assert candidate["export"]["prediction_split"] == "valid_iid"
    assert set(candidate["metadata"]["forbidden_access_roles"]) == FORBIDDEN
    assert candidate["metadata"]["training_started"] is False
    assert candidate["metadata"]["formal_e600_started"] is False
    command = shlex.join(build_training_command(candidate, python_executable="python"))
    assert "--epochs 600" in command
    assert "--batch-size 28" in command
    assert "--p-edge-masking 0.05" in command
    assert "--edge-masking-scope r2r_only" in command
    assert "--prediction-split valid_iid" in command
    assert "--init-checkpoint" not in command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    _objective_and_feature_fixtures()
    baseline = _resolved(V38)
    objective = _resolved(V42)
    deepsets_reference = _resolved(V44)
    configs = {name: _resolved(path) for name, path in CONFIGS.items()}
    diffs = {
        "V45_vs_V38": _diff(baseline, configs["V45"]),
        "V46_vs_V45": _diff(configs["V45"], configs["V46"]),
    }
    assert set(diffs["V45_vs_V38"]) == {"model.scale_deepsets_mode"}
    assert set(diffs["V46_vs_V45"]) == {
        "loss.native_log_scale_weight_mode",
        "loss.native_raw_loss_mode",
    }

    for candidate in configs.values():
        _assert_invariants(baseline, candidate)
        semantic = _semantic_defaults(candidate)
        assert semantic["model"]["scale_context_mode"] == "none"
        assert semantic["model"]["scale_context_feature_names"] == []
        assert semantic["model"]["scale_deepsets_mode"] == "source_volume_residual"
    assert configs["V45"]["loss"] == baseline["loss"]
    assert configs["V46"]["model"] == configs["V45"]["model"]
    assert (
        configs["V45"]["model"]["scale_deepsets_mode"]
        == deepsets_reference["model"]["scale_deepsets_mode"]
    )
    assert (
        _semantic_defaults(configs["V45"])["model"]["scale_deepsets_hidden_size"]
        == _semantic_defaults(deepsets_reference)["model"][
            "scale_deepsets_hidden_size"
        ]
    )
    for field in (
        "native_raw_loss_mode",
        "native_log_scale_weight_mode",
        "native_log_scale_weight_clip_min",
        "native_log_scale_weight_clip_max",
    ):
        assert configs["V46"]["loss"][field] == objective["loss"][field]
    for field in (
        "native_shape_cv_weight",
        "native_log_scale_weight",
        "native_relative_field_weight",
        "native_raw_field_weight",
    ):
        assert configs["V46"]["loss"][field] == baseline["loss"][field]

    runtime: dict[str, Any] = {}
    if not args.skip_runtime:
        runtime["V38"] = run_smoke(V38, batch_size=2, grid=(3, 3, 3))
        for name, path in CONFIGS.items():
            runtime[name] = run_smoke(path, batch_size=2, grid=(3, 3, 3))
            assert runtime[name]["finite_forward_backward"]
            assert runtime[name]["all_parameters_trainable"]
            assert runtime[name]["deepsets_output_init_max_abs"] == 0.0
        assert runtime["V45"]["parameter_count"] - runtime["V38"]["parameter_count"] == 28896
        assert runtime["V46"]["parameter_count"] == runtime["V45"]["parameter_count"]
        assert runtime["V45"]["initial_prediction_checksums"] == runtime["V38"]["initial_prediction_checksums"]
        assert runtime["V46"]["initial_prediction_checksums"] == runtime["V45"]["initial_prediction_checksums"]

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert [row["config_id"] for row in rows] == [
        configs[name]["config_id"] for name in ("V45", "V46")
    ]
    for row in rows:
        config = (
            configs["V45"]
            if row["candidate"] == "V45_deepsets_only"
            else configs["V46"]
        )
        assert row["launch_policy"] == "explicit_user_instruction_only"
        if row["execution_status"] == "not_started":
            assert row["evaluation_status"] == "not_evaluated"
            assert row["training_started"] == "false"
        else:
            assert row["plan_status"] == "completed"
            assert row["execution_status"] == "completed_e600"
            assert row["evaluation_status"] == "completed_valid_iid_four_checkpoint"
            assert row["training_started"] == "true"
        assert row["forbidden_access_roles"].split("|") == sorted(FORBIDDEN)
        assert row["scale_context_mode"] == "none"
        assert row["scale_deepsets_mode"] == "source_volume_residual"
        assert row["expected_parameter_increment"] == "28896"
        assert row["output_dir"] == config["export"]["output_dir"]
        assert row["log_path"] == config["metadata"]["log_path"]

    if not args.skip_report:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        assert report["status"] == "prepared_not_started"
        assert report["training_started"] is False
        assert report["formal_e600_started"] is False
        assert report["roles_accessed"] == ["train"]
        assert report["forbidden_roles_accessed"] == []
        assert report["resolved_config_diffs"] == diffs
        assert report["expected_parameter_increment"]["V45_vs_V38"] == 28896
        assert report["expected_parameter_increment"]["V46_vs_V45"] == 0
        real = report["real_train_update_smoke"]
        assert real["status"] == "passed"
        assert real["batch_size"] == 28
        assert real["node_count"] == 1024
        assert real["memory_fraction"] == 0.85
        assert set(real["results"]) == {"V45", "V46"}
        for name, result in real["results"].items():
            assert result["finite_loss"]
            assert result["finite_gradients"]
            assert result["finite_updated_parameters"]
            assert result["optimizer_update_applied"]
            assert result["update_nonzero"]
            assert result["checkpoint_written"] is False
            assert result["training_started"] is False
            assert result["formal_e600_started"] is False
            assert result["accessed_roles"] == ["train"]
            assert result["forbidden_roles_accessed"] == []
            assert result["parameter_count"] == 922632
            partition = result["p2r_partition_of_unity"]
            assert partition["zero_degree_node_count"] == 0
            assert partition["maximum_partition_of_unity_error"] <= 1.0e-12
            assert partition["source_conserved"]
            assert partition["volume_conserved"]
        assert rows[0]["smoke_status"] == rows[1]["smoke_status"] == "passed_real_B28_update"
        assert rows[0]["smoke_commit"] == rows[1]["smoke_commit"] == report["smoke_commit"]
        assert float(rows[0]["smoke_peak_GiB"]) > 0.0
        assert float(rows[1]["smoke_peak_GiB"]) > 0.0

    print(json.dumps({
        "status": "passed",
        "resolved_config_diffs": diffs,
        "runtime": runtime,
        "training_started": False,
        "formal_e600_started": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
