#!/usr/bin/env python3
"""Static and artifact checks for the frozen P1i reliable-training contract."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402

FORMAL = (
    "V6_06_V5best_P1i_seed0_reliable_B24.yaml",
    "V6_07_V5best_P1i_seed1_reliable_B24.yaml",
    "V6_08_V5best_P1i_seed2_reliable_B24.yaml",
)


def _resolve(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_inherited_yaml(payload, path)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, name))
        return result
    return {prefix: value}


def _duplicate_literal_keys(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = {}
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, (str, int, float)):
                continue
            if key.value in seen:
                duplicates.append({"key": key.value, "first_line": seen[key.value], "line": key.lineno})
            else:
                seen[key.value] = key.lineno
    return duplicates


def _checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    checks = {
        "schema_v2": payload.get("schema_version") == "heat3d_training_state_checkpoint_v2",
        "optimizer_saved": payload.get("optimizer_state_saved") is True and "optimizer_state" in payload,
        "epoch_present": isinstance(payload.get("epoch"), int),
        "best_state_present": isinstance(payload.get("best_state"), dict),
        "params_present": "params" in payload,
    }
    audit = Path(str(path) + ".reload.json")
    checks["reload_audit_passed"] = audit.is_file() and json.loads(audit.read_text())["passed"] is True
    return {"path": str(path), "checks": checks, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs/heat3d_v6_p1i")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    configs = [_resolve(args.config_dir / name) for name in FORMAL]
    checks: dict[str, bool] = {}
    for seed, config in enumerate(configs):
        run = config["run"]
        export = config["export"]
        optimizer = config["optimizer"]
        checks[f"seed{seed}_frozen_execution"] = (
            run["epochs"] == 600
            and run["batch_size"] == 24
            and run["micro_batch_size"] == 24
            and run["validation_batch_size"] == 32
            and run["prediction_batch_size"] == 32
            and run["drop_last"] is False
            and run["init_checkpoint"] is None
            and export["prediction_split"] == "valid_iid"
            and export["selection_metric"] == "valid_rel_rmse_v4_pct"
            and export["reliable_checkpointing"] is True
            and export["latest_checkpoint_every"] == 10
            and optimizer["seed"] == optimizer["model_seed"] == optimizer["batch_order_seed"] == optimizer["graph_seed"] == seed
            and run["batch_build_seed"] == seed
        )
        checks[f"seed{seed}_roles_closed"] = "test_and_sealed_closed" in config["metadata"]["split_access_policy"]
    base = _flatten(configs[0])
    allowed_identity_prefixes = ("config_id", "description", "export.output_dir", "export.run_name", "metadata.")
    allowed_seed_paths = {
        "optimizer.seed", "optimizer.model_seed", "optimizer.batch_order_seed", "optimizer.graph_seed", "run.batch_build_seed"
    }
    diff_payload = {}
    for seed, config in enumerate(configs[1:], start=1):
        other = _flatten(config)
        diff = {key: [base.get(key), other.get(key)] for key in sorted(set(base) | set(other)) if base.get(key) != other.get(key)}
        unexpected = [key for key in diff if key not in allowed_seed_paths and not key.startswith(allowed_identity_prefixes)]
        checks[f"seed{seed}_only_seed_scientific_diff"] = not unexpected
        diff_payload[f"seed0_to_seed{seed}"] = {"differences": diff, "unexpected": unexpected}
    runner_path = REPO_ROOT / "scripts/run_heat3d_v1_medium_controlled_training_export.py"
    wrapper_path = REPO_ROOT / "scripts/run_heat3d_v4_controlled_training.py"
    duplicate_audit = {str(path.relative_to(REPO_ROOT)): _duplicate_literal_keys(path) for path in (runner_path, wrapper_path)}
    checks["no_duplicate_literal_dict_keys"] = not any(duplicate_audit.values())
    checkpoints = []
    if args.output_dir is not None:
        for filename in ("params_best.pkl", "params_latest.pkl", "params_final.pkl"):
            path = args.output_dir / filename
            checkpoints.append(_checkpoint(path) if path.is_file() else {"path": str(path), "passed": False, "checks": {"exists": False}})
        checks["required_checkpoint_triplet"] = all(item["passed"] for item in checkpoints)
        checks["pretraining_contract"] = all((args.output_dir / name).is_file() for name in ("resolved_config_pretraining.yaml", "resolved_command.txt", "pretraining_provenance.json", "environment.json"))
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "resolved_diff": diff_payload,
        "duplicate_key_audit": duplicate_audit,
        "checkpoint_audit": checkpoints,
        "accessed_roles": ["configuration_only"],
        "test_accessed": False,
        "sealed_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
