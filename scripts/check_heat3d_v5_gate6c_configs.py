#!/usr/bin/env python3
"""Validate Gate 6C scratch loss-only configs without training or evaluation."""

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


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6c_scratch_loss_registry.csv"
N3_PATH = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
LOSS_FIELDS = (
    "native_shape_cv_weight",
    "native_log_scale_weight",
    "native_relative_field_weight",
    "native_raw_field_weight",
)
EXPECTED = {
    "Scratch-L1": [1.0, 1.0, 0.5, 1.5],
    "Scratch-L2": [1.5, 0.5, 0.5, 1.5],
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
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert [row["candidate"] for row in rows] == ["Scratch-L1", "Scratch-L2"]
    assert all(row["status"] == "prepared_not_started" for row in rows)
    assert all(row["launch_policy"] == "explicit_user_instruction_only" for row in rows)
    n3 = _resolved(N3_PATH)
    expected_science = _scientific_payload(n3)
    identities: dict[str, set[str]] = {
        name: set() for name in (
            "output_dir", "run_name", "log_path", "final_probe_output_dir",
            "post_training_diagnostics_output_dir",
        )
    }
    reports = {}
    for row in rows:
        label = row["candidate"]
        path = ROOT / row["generated_yaml"]
        config = _resolved(path)
        run, optimizer, export = config["run"], config["optimizer"], config["export"]
        assert row["baseline_config_id"] == "V4P5_07_native_pooled_latent_global_film"
        assert ROOT / row["baseline_yaml"] == N3_PATH
        assert config["dataset"] == n3["dataset"]
        assert config["model"] == n3["model"]
        assert optimizer == n3["optimizer"]
        assert _scientific_payload(config) == expected_science
        assert run["epochs"] == 600
        assert run["batch_size"] == 28
        assert run["init_checkpoint"] is None
        assert optimizer["multi_seed"] == []
        assert export["prediction_split"] == "valid_iid"
        assert export["selection_metric"] == "valid_base_mse"
        assert _weights(config) == EXPECTED[label]
        assert row["loss_weights"] == "|".join(str(value).rstrip("0").rstrip(".") for value in EXPECTED[label])
        for name, value in (
            ("output_dir", export["output_dir"]),
            ("run_name", export["run_name"]),
            ("log_path", row["log_path"]),
            ("final_probe_output_dir", run["final_probe_output_dir"]),
            ("post_training_diagnostics_output_dir", run["post_training_diagnostics_output_dir"]),
        ):
            assert value == row[name]
            assert value not in identities[name]
            identities[name].add(value)
        command = build_training_command(config, python_executable="python")
        joined = shlex.join(command)
        for fragment in (
            "--epochs 600",
            "--batch-size 28",
            "--global-context-mode film",
            "--native-output-mode native_shape_scale",
            "--scale-head-mode physics_plus_pooled_latent",
            "--prediction-split valid_iid",
            "--selection-metric valid_base_mse",
        ):
            assert fragment in joined, f"{label}: missing dry-run fragment {fragment}"
        assert "--init-checkpoint" not in joined
        for field, value in zip(LOSS_FIELDS, EXPECTED[label], strict=True):
            flag = "--" + field.replace("native_", "native-").replace("_", "-")
            assert f"{flag} {value}" in joined
        reports[label] = {
            "config_id": row["config_id"],
            "yaml": row["generated_yaml"],
            "weights": EXPECTED[label],
            "epochs": 600,
            "initialization": "random",
            "training_started_by_checker": False,
        }
    v4_registry = (ROOT / "configs/heat3d_v4/run_registry.csv").read_text(encoding="utf-8")
    assert all(row["config_id"] not in v4_registry for row in rows)
    print(json.dumps({
        "status": "passed",
        "scientific_differences_from_n3": ["loss.native_*_weight"],
        "test_or_hard_accessed": False,
        "e600_started": False,
        "multi_seed_started": False,
        "configs": reports,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
