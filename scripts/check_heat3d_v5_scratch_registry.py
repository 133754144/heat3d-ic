#!/usr/bin/env python3
"""Validate the phase-specific V5 scratch registry and print dry-run commands."""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from heat3d_v5_result_contract import (  # noqa: E402
    V5_FROZEN_METRICS,
    V5_REGISTRY_RESULT_FIELDS,
    V5_REPORT_ROLES,
)


REGISTRY = ROOT / "configs/heat3d_v5/v5_scratch_bypass_film_registry.csv"
V4_JSON = ROOT / "configs/heat3d_v4/v4_run_registry.json"
V4_CSV = ROOT / "configs/heat3d_v4/run_registry.csv"
ALLOWED_BASELINES = {
    "V4P5_02_clean_baseline_raw_B28_e600",
    "V4P5_04_local_bypass_global_film",
    "V4P5_06_native_pooled_latent",
}
EXPECTED_LOCAL = (
    "k_x", "k_y", "k_z", "q", "is_top", "is_bottom", "is_side", "is_interior"
)
EXPECTED_REPORT = (
    "test_iid", "hard_train_holdout", "hard_challenge_valid", "hard_challenge_test"
)
EXPECTED_CHECKPOINTS = (
    "valid_sample_first_cv_relative_rmse",
    "valid_raw_cv_weighted_rmse_K",
    "valid_point_global_relative_rmse",
    "valid_base_mse",
)


def _parts(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split("|") if item)


def _resolved(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"invalid YAML root: {path}")
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _assert_same(base: dict, candidate: dict, section: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        assert candidate[section].get(key) == base[section].get(key), (
            f"{section}.{key} drifted: {candidate[section].get(key)!r} != "
            f"{base[section].get(key)!r}"
        )


def main() -> int:
    reader = csv.DictReader(REGISTRY.open(encoding="utf-8", newline=""))
    fieldnames = tuple(reader.fieldnames or ())
    missing_result_columns = [
        field for field in V5_REGISTRY_RESULT_FIELDS if field not in fieldnames
    ]
    assert not missing_result_columns, (
        "V5 registry must expose result columns: "
        + ", ".join(missing_result_columns)
    )
    rows = list(reader)
    assert rows, "V5 registry is empty"
    assert len({row["config_id"] for row in rows}) == len(rows), "duplicate config_id"
    v4_json = json.loads(V4_JSON.read_text(encoding="utf-8"))
    v4_ids = set(v4_json["runs"])
    v4_csv_ids = {row["config_id"] for row in csv.DictReader(V4_CSV.open(encoding="utf-8"))}
    seen_paths: set[str] = set()
    plans = []
    for row in rows:
        config_id = row["config_id"]
        assert row["phase"] == "v5"
        assert row["baseline_config_id"] in ALLOWED_BASELINES
        assert config_id not in v4_ids and config_id not in v4_csv_ids, (
            f"V5 config leaked into V4 registry: {config_id}"
        )
        assert row["launch_policy"] == "explicit_user_instruction_only"
        for key in (
            "generated_yaml", "output_dir", "log_path", "final_probe_output_dir",
            "post_training_diagnostics_output_dir",
        ):
            value = row[key]
            assert value not in seen_paths, f"duplicate registry path: {value}"
            seen_paths.add(value)
        path = ROOT / row["generated_yaml"]
        baseline_path = ROOT / row["baseline_yaml"]
        assert path.is_file() and baseline_path.is_file()
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert source["config_id"] == config_id
        assert (path.parent / source["extends"]).resolve() == baseline_path.resolve()
        candidate, base = _resolved(path), _resolved(baseline_path)
        assert candidate["run"].get("init_checkpoint") is None
        assert row["initialization"] == "random"
        _assert_same(base, candidate, "dataset", (
            "name", "subset_path", "manifest_path", "split_map_path", "target",
            "normalization_profile", "coord_policy", "extent_feature_policy",
            "condition_feature_transform",
        ))
        _assert_same(base, candidate, "model", (
            "architecture", "node_latent_size", "edge_latent_size",
            "processor_steps", "mlp_hidden_layers", "p_edge_masking",
            "decoder_bypass_hidden_size", "decoder_bypass_layers",
            "decoder_bypass_residual_scale",
        ))
        _assert_same(base, candidate, "graph", tuple(base["graph"].keys()))
        _assert_same(base, candidate, "optimizer", tuple(base["optimizer"].keys()))
        _assert_same(base, candidate, "loss", tuple(base["loss"].keys()))
        _assert_same(base, candidate, "run", (
            "epochs", "batch_size", "validation_batch_size", "prediction_batch_size",
            "batch_plan", "batch_build_seed", "shuffle_train_batches", "drop_last",
        ))
        model = candidate["model"]
        assert model["decoder_bypass_features"] == "explicit_local_condition"
        assert tuple(model["decoder_bypass_local_feature_names"]) == EXPECTED_LOCAL
        assert tuple(model["global_context_feature_names"]) == GLOBAL_CONTEXT_FEATURES
        native = row["candidate"].startswith("N")
        expected_global_mode = (
            "film" if config_id == "V4P5_07_native_pooled_latent_global_film"
            else ("none" if native else "film")
        )
        assert model["global_context_mode"] == expected_global_mode
        assert model["film_target"] == "rnodes_processed"
        assert model["film_init"] == "identity"
        metadata = candidate["metadata"]
        assert tuple(metadata["fit_roles"]) == ("train",)
        assert tuple(metadata["normalization_fit_roles"]) == ("train",)
        assert tuple(metadata["selection_roles"]) == ("valid_iid",)
        assert tuple(metadata["report_only_roles"]) == EXPECTED_REPORT
        assert _parts(row["decoder_bypass_local_feature_names"]) == EXPECTED_LOCAL
        assert _parts(row["report_only_roles"]) == EXPECTED_REPORT
        expected_checkpoints = ("valid_base_mse",) if native else EXPECTED_CHECKPOINTS
        assert _parts(row["checkpoint_metrics"]) == expected_checkpoints
        result_status = row.get("result_v5_status", "")
        result_complete = row.get("result_v5_required_metrics_complete", "")
        if result_status == "completed":
            assert result_complete == "true", (
                f"{config_id}: completed result must include all frozen V5 metrics"
            )
        if result_complete == "true":
            payload = json.loads(row.get("result_v5_metrics_json") or "{}")
            reports = payload.get("reports", payload)
            for checkpoint in ("primary_relative", "legacy_metric"):
                for role in V5_REPORT_ROLES:
                    metric_row = reports.get(checkpoint, {}).get(role, {})
                    missing = [
                        metric for metric in V5_FROZEN_METRICS
                        if metric not in metric_row
                    ]
                    assert not missing, (
                        f"{config_id}: {checkpoint}.{role} missing {missing}"
                    )
        contract = metadata["checkpoint_contract"]
        if native:
            assert contract["primary"] == "valid_base_mse"
            assert candidate["export"]["selection_metric"] == "valid_base_mse"
            assert model["native_output_mode"] == "native_shape_scale"
            assert model["native_branch_mode"] == "joint"
            assert model["decoder_bypass_output_space"] == "native_psi"
            assert model["scale_pooling"] == "mean"
            assert model["scale_head_mode"] == (
                "physics_only" if row["candidate"].startswith("N0")
                else "physics_plus_pooled_latent"
            )
            for key in (
                "native_shape_cv_weight", "native_log_scale_weight",
                "native_relative_field_weight", "native_raw_field_weight",
            ):
                assert candidate["loss"][key] == 1.0
        else:
            assert contract["primary"] == EXPECTED_CHECKPOINTS[0]
            assert contract["tie_break"] == EXPECTED_CHECKPOINTS[1]
            assert contract["secondary"] == EXPECTED_CHECKPOINTS[2]
            assert contract["legacy_control"] == EXPECTED_CHECKPOINTS[3]
        command = build_training_command(candidate, python_executable="python")
        joined = shlex.join(command)
        assert "--init-checkpoint" not in command
        fragments = [
            "--decoder-bypass-features explicit_local_condition",
            f"--global-context-mode {expected_global_mode}",
            "--film-target rnodes_processed",
            "--film-init identity",
            "--epochs 600",
            "--loss-mode mse",
        ]
        if native:
            fragments.extend((
                "--native-output-mode native_shape_scale",
                "--native-branch-mode joint",
                f"--scale-head-mode {model['scale_head_mode']}",
                "--scale-pooling mean",
                "--decoder-bypass-output-space native_psi",
                "--native-shape-cv-weight 1.0",
                "--native-log-scale-weight 1.0",
                "--native-relative-field-weight 1.0",
                "--native-raw-field-weight 1.0",
                "--selection-metric valid_base_mse",
            ))
        for fragment in fragments:
            assert fragment in joined, f"dry-run missing {fragment}"
        plans.append({"config_id": config_id, "training_started": False, "command": joined})
    print(json.dumps({"status": "ok", "registry": str(REGISTRY.relative_to(ROOT)), "plans": plans}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
