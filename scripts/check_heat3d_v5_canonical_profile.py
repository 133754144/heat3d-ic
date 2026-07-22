#!/usr/bin/env python3
"""Verify the durable V42 canonical profile and default-entry isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import posixpath
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from run_heat3d_v4_config import (  # noqa: E402
    DEFAULT_TRAINING_PROFILE,
    _load_config,
    _selected_config_path,
)


CANONICAL = ROOT / "configs/heat3d_v5/V4P5_42_canonical.yaml"
SCIENTIFIC_PAYLOAD_SHA256 = "f4bc08542abecc667f0833bca22e7fe510f0bcc6ec7793db8b6d08e0c30bd6d6"
CANONICAL_COMMAND_SHA256 = "ccfab6d3940f2b12d1e885e67046d0d689ef9144bddd496df485038fbaa635c6"
LEGACY_COMMANDS = {
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_dryrun.yaml": (
        "b46cdd3f4c9d33a567a9851a77c9ed22add4a14699ed4a3d4a4e9fb01d25b809"
    ),
    "configs/heat3d_v4/generated/V4P5_02_clean_baseline_raw_B28_e600.yaml": (
        "3a4ad18d2b1a6c0c1ae826981d76e3dcf0e17558055d1642ad08a0ba2676331c"
    ),
}
FROZEN_TAG = "v5-final-threshold-unmet"
FROZEN_V42 = "configs/heat3d_v5/generated/V4P5_42_gate6q_objective_only_e600.yaml"


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _frozen_yaml(path: str) -> dict:
    raw = subprocess.check_output(
        ["git", "show", f"{FROZEN_TAG}:{path}"], cwd=ROOT, text=True
    )
    payload = yaml.safe_load(raw)
    if payload.get("schema_version") != "heat3d_v4_inherited_config_v0":
        return payload
    parent = posixpath.normpath(f"{Path(path).parent.as_posix()}/{payload['extends']}")
    base = _frozen_yaml(parent)
    return _deep_merge(base, dict(payload.get("overrides") or {}))


def _assert_frozen_v42_field_equivalence(canonical: dict) -> None:
    frozen = _frozen_yaml(FROZEN_V42)
    if canonical["dataset"] != frozen["dataset"]:
        raise AssertionError("canonical dataset differs from frozen V42/V38 contract")
    frozen_model = dict(frozen["model"])
    for key, value in (
        ("shape_attention_mode", "none"),
        ("scale_context_mode", "none"),
        ("scale_deepsets_mode", "none"),
    ):
        frozen_model.setdefault(key, value)
    if canonical["model"] != frozen_model:
        raise AssertionError("canonical model differs from frozen V42/V38 contract")
    for section in ("graph", "optimizer", "loss", "diagnostics", "baseline_reference"):
        if canonical[section] != frozen[section]:
            raise AssertionError(f"canonical {section} differs from frozen V42")
    frozen_run = dict(frozen["run"])
    for key in ("device_policy", "final_probe_output_dir", "post_training_diagnostics_output_dir"):
        frozen_run.pop(key, None)
    if canonical["run"] != frozen_run:
        raise AssertionError("canonical run contract differs beyond removed runtime fields")
    frozen_export = dict(frozen["export"])
    for key in ("output_dir", "run_name"):
        frozen_export.pop(key, None)
    frozen_selection = frozen_export.pop("selection_metric")
    canonical_export = dict(canonical["export"])
    canonical_selection = canonical_export.pop("selection_metric")
    if canonical_export != frozen_export:
        raise AssertionError("canonical export differs beyond runtime identity/selection")
    if frozen_selection != "valid_base_mse" or canonical_selection != "valid_rel_rmse_v4_pct":
        raise AssertionError("primary checkpoint policy is not the requested point-global transition")


def main() -> int:
    yaml_files = sorted((ROOT / "configs/heat3d_v5").rglob("*.yaml"))
    if yaml_files != [CANONICAL]:
        raise AssertionError(f"expected exactly one migrated V5 YAML, got={yaml_files}")
    if DEFAULT_TRAINING_PROFILE != CANONICAL or _selected_config_path(None) != CANONICAL:
        raise AssertionError("omitted --config does not select canonical V42")
    explicit_probe = "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_dryrun.yaml"
    if _selected_config_path(explicit_probe) != ROOT / explicit_probe:
        raise AssertionError("explicit config path resolution changed")

    config = _load_config(CANONICAL)
    _assert_frozen_v42_field_equivalence(config)
    science = {key: config[key] for key in ("dataset", "model", "graph", "optimizer", "loss", "run", "export")}
    if _hash(science) != SCIENTIFIC_PAYLOAD_SHA256:
        raise AssertionError("canonical scientific payload drift")
    forbidden_keys = {
        "output_dir",
        "run_name",
        "log_path",
        "final_probe_output_dir",
        "post_training_diagnostics_output_dir",
        "assigned_host",
        "launch_host",
        "launch_tmux_session",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise AssertionError(f"canonical embeds runtime fields: {sorted(overlap)}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(config)
    if config["run"]["init_checkpoint"] is not None:
        raise AssertionError("canonical profile is not random-init")
    if config["model"]["scale_context_mode"] != "none":
        raise AssertionError("canonical profile enables XY scale context")
    if config["model"]["scale_deepsets_mode"] != "none":
        raise AssertionError("canonical profile enables DeepSets")

    command = build_training_command(config, python_executable="python")
    if _command_hash(command) != CANONICAL_COMMAND_SHA256:
        raise AssertionError("canonical command drift")
    expected_options = {
        "--epochs": "600",
        "--batch-size": "28",
        "--model-seed": "0",
        "--batch-order-seed": "0",
        "--graph-seed": "0",
        "--p-edge-masking": "0.05",
        "--edge-masking-scope": "r2r_only",
        "--native-raw-loss-mode": "point_global_fixed_train_energy_sse",
        "--native-log-scale-weight-mode": "train_true_scale_squared_clipped",
        "--selection-metric": "valid_rel_rmse_v4_pct",
        "--scale-context-mode": "none",
        "--scale-deepsets-mode": "none",
    }
    for flag, expected in expected_options.items():
        if _value(command, flag) != expected:
            raise AssertionError(f"{flag} drifted from {expected}")
    if "--init-checkpoint" in command or "--output-dir" in command:
        raise AssertionError("canonical command embeds checkpoint/output path")

    default = subprocess.run(
        [sys.executable, "scripts/run_heat3d_v4_config.py", "--python-executable", "python", "--dry-run"],
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    explicit = subprocess.run(
        [
            sys.executable,
            "scripts/run_heat3d_v4_config.py",
            "--config",
            str(CANONICAL.relative_to(ROOT)),
            "--python-executable",
            "python",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    if default != explicit:
        raise AssertionError("default and explicit canonical commands differ")

    legacy = {}
    for relative, expected_hash in LEGACY_COMMANDS.items():
        legacy_config = _load_config(ROOT / relative)
        legacy_command = build_training_command(legacy_config, python_executable="python")
        actual = _command_hash(legacy_command)
        if actual != expected_hash:
            raise AssertionError(f"legacy command drift: {relative} {actual}")
        legacy[relative] = actual

    print(
        json.dumps(
            {
                "status": "passed",
                "canonical": str(CANONICAL.relative_to(ROOT)),
                "scientific_payload_sha256": SCIENTIFIC_PAYLOAD_SHA256,
                "command_sha256": CANONICAL_COMMAND_SHA256,
                "legacy_command_sha256": legacy,
                "training_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
