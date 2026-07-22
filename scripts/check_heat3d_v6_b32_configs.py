#!/usr/bin/env python3
"""Validate V6 B32 single-variable configs and exact 4xB8 batching."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402


PAIRS = {
    "V6_01_V4best_B32": (
        ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
        ROOT / "configs/heat3d_v6/V6_01_V4best_B32.yaml",
        ROOT / "configs/heat3d_v6/preflight/V6_01_V4best_B32_e5_gpu_preflight.yaml",
    ),
    "V6_02_V5best_B32": (
        ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
        ROOT / "configs/heat3d_v6/V6_02_V5best_B32.yaml",
        ROOT / "configs/heat3d_v6/preflight/V6_02_V5best_B32_e5_gpu_preflight.yaml",
    ),
}
GATE = ROOT / "configs/heat3d_v6/v6_b32_selective_launch_gate.json"


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = resolve_inherited_yaml(payload, path)
    value["config_id"] = payload["config_id"]
    return value


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaf_diffs(left.get(key), right.get(key), path))
        return result
    return [] if left == right else [prefix]


def _scientific_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("config_id", None)
    payload.pop("description", None)
    payload.pop("metadata", None)
    payload.get("export", {}).pop("output_dir", None)
    payload.get("export", {}).pop("run_name", None)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-resolved", action="store_true")
    args = parser.parse_args()
    reports = []
    for config_id, (b24_path, b32_path, e5_path) in PAIRS.items():
        b24 = _resolved(b24_path)
        b32 = _resolved(b32_path)
        e5 = _resolved(e5_path)
        validate_v2_config(b32, config_path=b32_path)
        validate_v2_config(e5, config_path=e5_path)
        left = _scientific_payload(b24)
        right = _scientific_payload(b32)
        assert _leaf_diffs(left, right) == ["run.batch_size"]
        assert b32["run"]["batch_size"] == 32
        assert b32["run"]["micro_batch_size"] == 8
        assert b32["run"]["validation_batch_size"] == 32
        assert b32["run"]["prediction_batch_size"] == 32
        assert b32["run"]["drop_last"] is False
        assert b32["run"]["epochs"] == 600
        assert b32["metadata"]["micro_batches_per_epoch"] == 96
        assert b32["metadata"]["optimizer_updates_per_epoch"] == 24
        assert b32["metadata"]["micro_batches_per_optimizer_update"] == 4
        assert b32["metadata"]["final_partial_effective_batch_size"] is None
        assert b32["metadata"]["training_started"] is False
        assert e5["run"]["epochs"] == 5
        assert e5["run"]["train_metrics_schedule"] == "every_epoch"
        assert e5["run"]["memory_audit_every_batch"] is True
        assert e5["export"]["output_dir"] != b32["export"]["output_dir"]
        assert e5["metadata"]["preflight_only"] is True
        if args.write_resolved:
            path = ROOT / f"configs/heat3d_v6/resolved/{config_id}.resolved.yaml"
            path.write_text(yaml.safe_dump(b32, sort_keys=False), encoding="utf-8")
        reports.append(
            {
                "config_id": config_id,
                "scientific_diff_paths": ["run.batch_size"],
                "micro_batch_size": 8,
                "micro_batches_per_epoch": 96,
                "micro_batches_per_update": 4,
                "updates_per_epoch": 24,
                "tail": None,
                "geometry_split": False,
                "epochs": 600,
                "e5_preflight": str(e5_path.relative_to(ROOT)),
            }
        )

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert gate["status"] == "prepared_not_evaluated"
    assert gate["formal_training_started"] is False
    assert gate["candidate"]["micro_batches_per_epoch"] == 96
    assert gate["candidate"]["updates_per_epoch"] == 24
    assert gate["candidate"]["tail_batch"] is None
    assert gate["forbidden_roles"] == ["test_iid", "all", "hard", "sealed"]
    print(json.dumps({"status": "passed", "configs": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
