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


REGISTRY = ROOT / "configs/heat3d_v5/v5_scratch_bypass_film_registry.csv"
V4_JSON = ROOT / "configs/heat3d_v4/v4_run_registry.json"
V4_CSV = ROOT / "configs/heat3d_v4/run_registry.csv"
BASELINE_ID = "V4P5_02_clean_baseline_raw_B28_e600"
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
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
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
        assert row["baseline_config_id"] == BASELINE_ID
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
        assert model["global_context_mode"] == "film"
        assert model["film_target"] == "rnodes_processed"
        assert model["film_init"] == "identity"
        metadata = candidate["metadata"]
        assert tuple(metadata["fit_roles"]) == ("train",)
        assert tuple(metadata["normalization_fit_roles"]) == ("train",)
        assert tuple(metadata["selection_roles"]) == ("valid_iid",)
        assert tuple(metadata["report_only_roles"]) == EXPECTED_REPORT
        assert _parts(row["decoder_bypass_local_feature_names"]) == EXPECTED_LOCAL
        assert _parts(row["report_only_roles"]) == EXPECTED_REPORT
        assert _parts(row["checkpoint_metrics"]) == EXPECTED_CHECKPOINTS
        contract = metadata["checkpoint_contract"]
        assert contract["primary"] == EXPECTED_CHECKPOINTS[0]
        assert contract["tie_break"] == EXPECTED_CHECKPOINTS[1]
        assert contract["secondary"] == EXPECTED_CHECKPOINTS[2]
        assert contract["legacy_control"] == EXPECTED_CHECKPOINTS[3]
        command = build_training_command(candidate, python_executable="python")
        joined = shlex.join(command)
        assert "--init-checkpoint" not in command
        for fragment in (
            "--decoder-bypass-features explicit_local_condition",
            "--global-context-mode film",
            "--film-target rnodes_processed",
            "--film-init identity",
            "--epochs 600",
            "--loss-mode mse",
        ):
            assert fragment in joined, f"dry-run missing {fragment}"
        plans.append({"config_id": config_id, "training_started": False, "command": joined})
    print(json.dumps({"status": "ok", "registry": str(REGISTRY.relative_to(ROOT)), "plans": plans}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
