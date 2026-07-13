#!/usr/bin/env python3
"""Assert N1/N3 differ scientifically only by Global FiLM enablement."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


N1 = ROOT / "configs/heat3d_v5/generated/V4P5_06_native_pooled_latent.yaml"
N3 = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
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
    payload["model"].pop("global_context_mode", None)
    for section, fields in IDENTITY_FIELDS.items():
        for field in fields:
            payload[section].pop(field, None)
    return payload


def main() -> int:
    n1, n3 = _resolved(N1), _resolved(N3)
    assert n1["model"]["global_context_mode"] == "none"
    assert n3["model"]["global_context_mode"] == "film"
    for field in ("global_context_feature_names", "film_target", "film_init"):
        assert n1["model"][field] == n3["model"][field]
    assert n1["model"]["scale_head_mode"] == n3["model"]["scale_head_mode"] == "physics_plus_pooled_latent"
    assert n1["run"]["init_checkpoint"] is None and n3["run"]["init_checkpoint"] is None
    assert n1["run"]["epochs"] == n3["run"]["epochs"] == 600
    assert n1["run"]["batch_size"] == n3["run"]["batch_size"] == 28
    assert n1["export"]["selection_metric"] == n3["export"]["selection_metric"] == "valid_base_mse"
    assert _scientific_payload(n1) == _scientific_payload(n3)
    command = " ".join(build_training_command(n3, python_executable="python"))
    for fragment in (
        "--global-context-mode film",
        "--film-target rnodes_processed",
        "--film-init identity",
        "--native-output-mode native_shape_scale",
        "--scale-head-mode physics_plus_pooled_latent",
        "--epochs 600",
        "--batch-size 28",
        "--selection-metric valid_base_mse",
    ):
        assert fragment in command, f"N3 command missing {fragment}"
    assert "--init-checkpoint" not in command
    print(json.dumps({
        "status": "passed",
        "allowed_scientific_difference": "model.global_context_mode none->film",
        "shared_global_context_schema": n3["model"]["global_context_feature_names"],
        "random_initialization": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
