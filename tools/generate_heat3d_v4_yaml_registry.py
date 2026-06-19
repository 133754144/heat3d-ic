#!/usr/bin/env python3
"""Generate inherited Heat3D V4 YAML configs and dry-run command plans.

This tool is control-plane only. It does not execute training, start tmux,
read datasets, or create run output directories. It resolves each inherited
YAML in memory and reuses the existing v2 validator and dry-run command builder.
"""

from __future__ import annotations

import argparse
import csv
import copy
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue.
    raise SystemExit("PyYAML is required for V4 YAML generation.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import (  # noqa: E402
    build_v2_command_plan,
    summarize_command_plan,
)


DEFAULT_REGISTRY = Path("configs/heat3d_v4/run_registry.csv")
INHERITED_SCHEMA_VERSION = "heat3d_v4_inherited_config_v0"


def main() -> int:
    args = _parse_args()
    registry_path = _repo_path(args.registry)
    rows = _load_registry(registry_path)
    generated_paths: list[Path] = []
    output_rows: list[dict[str, str]] = []

    for row in rows:
        config_id = row["config_id"]
        base_path = _repo_path(row["base_yaml"])
        generated_path = _repo_path(row["generated_yaml"])
        base_config = load_v2_config(base_path)
        inherited = _build_inherited_yaml(row, base_config, generated_path)

        if args.write_yaml:
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            _write_yaml(generated_path, inherited)
            generated_paths.append(generated_path)

        resolved = _resolve_inherited(inherited, generated_path=generated_path)
        validate_v2_config(resolved, config_path=base_path)
        _assert_registry_matches_resolved(row, resolved)
        _assert_no_output_collision(resolved)

        if args.dry_run:
            plan = build_v2_command_plan(resolved, python_executable=args.python_executable)
            print(f"config_id: {config_id}")
            print(f"generated_yaml: {row['generated_yaml']}")
            print(summarize_command_plan(plan))

        output_rows.append(row)

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_registry_csv(output_csv, output_rows)
        print(f"wrote registry csv: {output_csv}")

    if generated_paths:
        print("generated inherited YAML:")
        for path in generated_paths:
            print(f"  {_relative_to_repo(path)}")

    print(f"validated registry rows: {len(rows)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--write-yaml", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--python-executable", default="python3")
    return parser.parse_args()


def _load_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path}: registry has no rows")
    for index, row in enumerate(rows, start=2):
        if not row.get("config_id"):
            raise ValueError(f"{path}:{index}: missing config_id")
        if not row.get("base_yaml"):
            raise ValueError(f"{path}:{index}: missing base_yaml")
        if not row.get("generated_yaml"):
            raise ValueError(f"{path}:{index}: missing generated_yaml")
    return rows


def _build_inherited_yaml(
    row: Mapping[str, str],
    base_config: Mapping[str, Any],
    generated_path: Path,
) -> dict[str, Any]:
    desired = _desired_config_from_row(row)
    overrides = _diff_mapping(base_config, desired)
    return {
        "schema_version": INHERITED_SCHEMA_VERSION,
        "config_id": row["config_id"],
        "extends": _relative_path(_repo_path(row["base_yaml"]), generated_path.parent),
        "overrides": overrides,
    }


def _desired_config_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    config_id = row["config_id"]
    return {
        "description": (
            f"Heat3D V4 registry config {config_id}; standard task "
            f"{row['task']}; generated from {Path(row['base_yaml']).name}."
        ),
        "model": {
            "node_latent_size": _int(row, "node_latent_size"),
            "edge_latent_size": _int(row, "edge_latent_size"),
            "processor_steps": _int(row, "processor_steps"),
            "mlp_hidden_layers": _int(row, "mlp_hidden_layers"),
        },
        "optimizer": {
            "name": row["optimizer"],
            "lr": _float(row, "lr"),
            "lr_schedule": row["lr_schedule"],
            "warmup_epochs": _int(row, "warmup_epochs"),
            "min_lr": _float(row, "min_lr"),
            "weight_decay": _float(row, "weight_decay"),
        },
        "loss": {
            "mode": row["loss_mode"],
        },
        "run": {
            "epochs": _int(row, "epochs"),
            "batch_size": _int(row, "batch_size"),
            "validation_batch_size": _int(row, "validation_batch_size"),
            "prediction_batch_size": _int(row, "prediction_batch_size"),
            "batch_plan": row["batch_plan"],
            "final_probe_output_dir": row["final_probe_output_dir"],
            "post_training_diagnostics_output_dir": row[
                "post_training_diagnostics_output_dir"
            ],
        },
        "graph": {
            "radius_policy": row["graph_radius_policy"],
            "coverage_repair_policy": row["coverage_repair_policy"],
        },
        "export": {
            "output_dir": row["output_dir"],
            "run_name": row["run_name"],
            "selection_metric": row["selection_metric"],
        },
        "metadata": {
            "registry_config_id": config_id,
            "registry_status": row["status"],
            "standard_task": row["task"],
            "variant_label": config_id,
            "model_label": (
                f"latent{row['node_latent_size']}-edge{row['edge_latent_size']}"
                f"-s{row['processor_steps']}-mlp{row['mlp_hidden_layers']}"
            ),
            "batch_policy_label": f"B{row['batch_size']}_{row['batch_plan']}",
            "optimizer_label": (
                f"{row['optimizer']}_{row['lr_schedule']}_lr{row['lr']}"
                f"_minlr{row['min_lr']}"
            ),
            "graph_policy_label": (
                f"{row['graph_radius_policy']}_repair_{row['coverage_repair_policy']}"
            ),
            "loss_label": f"plain_{row['loss_mode']}",
            "launch_policy": row["launch_policy"],
            "log_path": row["log_path"],
            "notes": row["notes"],
        },
    }


def _resolve_inherited(
    inherited: Mapping[str, Any], *, generated_path: Path
) -> dict[str, Any]:
    if inherited.get("schema_version") != INHERITED_SCHEMA_VERSION:
        raise ValueError("inherited YAML has wrong schema_version")
    extends = inherited.get("extends")
    if not isinstance(extends, str) or not extends:
        raise ValueError("inherited YAML requires a non-empty extends field")
    base_path = (generated_path.parent / extends).resolve()
    base_config = load_v2_config(base_path)
    overrides = inherited.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("inherited YAML overrides field must be a mapping")
    return _deep_merge(base_config, overrides)


def _assert_registry_matches_resolved(
    row: Mapping[str, str], config: Mapping[str, Any]
) -> None:
    checks = {
        "model.node_latent_size": _int(row, "node_latent_size"),
        "model.edge_latent_size": _int(row, "edge_latent_size"),
        "model.processor_steps": _int(row, "processor_steps"),
        "model.mlp_hidden_layers": _int(row, "mlp_hidden_layers"),
        "run.batch_size": _int(row, "batch_size"),
        "run.validation_batch_size": _int(row, "validation_batch_size"),
        "run.prediction_batch_size": _int(row, "prediction_batch_size"),
        "run.batch_plan": row["batch_plan"],
        "run.epochs": _int(row, "epochs"),
        "optimizer.name": row["optimizer"],
        "optimizer.lr": _float(row, "lr"),
        "optimizer.lr_schedule": row["lr_schedule"],
        "optimizer.warmup_epochs": _int(row, "warmup_epochs"),
        "optimizer.min_lr": _float(row, "min_lr"),
        "optimizer.weight_decay": _float(row, "weight_decay"),
        "graph.radius_policy": row["graph_radius_policy"],
        "graph.coverage_repair_policy": row["coverage_repair_policy"],
        "loss.mode": row["loss_mode"],
        "export.selection_metric": row["selection_metric"],
        "export.output_dir": row["output_dir"],
        "export.run_name": row["run_name"],
    }
    for dotted, expected in checks.items():
        actual = _get_dotted(config, dotted)
        if actual != expected:
            raise AssertionError(
                f"{row['config_id']}: {dotted} expected {expected!r}, got {actual!r}"
            )


def _assert_no_output_collision(config: Mapping[str, Any]) -> None:
    for dotted in (
        "export.output_dir",
        "run.final_probe_output_dir",
        "run.post_training_diagnostics_output_dir",
    ):
        output_path = _get_dotted(config, dotted)
        if not isinstance(output_path, str):
            continue
        path = REPO_ROOT / output_path
        if path.exists():
            raise FileExistsError(f"output collision for {dotted}: {output_path}")


def _diff_mapping(base: Mapping[str, Any], desired: Mapping[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, desired_value in desired.items():
        base_value = base.get(key)
        if isinstance(base_value, Mapping) and isinstance(desired_value, Mapping):
            nested = _diff_mapping(base_value, desired_value)
            if nested:
                diff[key] = nested
        elif base_value != desired_value:
            diff[key] = desired_value
    return diff


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)


def _write_registry_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _int(row: Mapping[str, str], key: str) -> int:
    return int(row[key])


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _get_dotted(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
