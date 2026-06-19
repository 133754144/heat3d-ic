#!/usr/bin/env python3
"""Check the authoritative Heat3D V4 run registry.

The JSON registry is the only source of truth. It resolves one baseline plus
per-run overrides into CSV mirror rows and generated YAML.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue.
    raise SystemExit("PyYAML is required for V4 registry checks.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


REGISTRY_SCHEMA_VERSION = "heat3d_v4_run_registry_v0"
INHERITED_SCHEMA_VERSION = "heat3d_v4_inherited_config_v0"
DEFAULT_REGISTRY = Path("configs/heat3d_v4/v4_run_registry.json")
CSV_FIELDNAMES = (
    "config_id",
    "phase",
    "status",
    "base_yaml",
    "generated_yaml",
    "task",
    "model_capacity",
    "node_latent_size",
    "edge_latent_size",
    "processor_steps",
    "mlp_hidden_layers",
    "batch_size",
    "validation_batch_size",
    "prediction_batch_size",
    "batch_plan",
    "optimizer",
    "lr",
    "model_seed",
    "batch_order_seed",
    "graph_seed",
    "seed",
    "multi_seed",
    "batch_build_seed",
    "lr_schedule",
    "warmup_epochs",
    "min_lr",
    "weight_decay",
    "epochs",
    "graph_radius_policy",
    "coverage_repair_policy",
    "loss_mode",
    "selection_metric",
    "output_dir",
    "run_name",
    "log_path",
    "final_probe_output_dir",
    "post_training_diagnostics_output_dir",
    "dry_run_required",
    "launch_policy",
    "notes",
)
UNIQUE_RESOLVED_FIELDS = (
    "config_id",
    "generated_yaml",
    "output_dir",
    "run_name",
    "log_path",
    "final_probe_output_dir",
    "post_training_diagnostics_output_dir",
)
EXPECTED_V4_BASELINE = {
    "config_id": "V4_baseline",
    "phase": "v4",
    "status": "registered",
    "base_yaml": "configs/heat3d_v4/V4_base.yaml",
    "generated_yaml": "configs/heat3d_v4/generated/V4_baseline.yaml",
    "task": "coords+k(x)+q(x)+BC->T(x)",
    "model_capacity": "96/96/s6/m2",
    "node_latent_size": "96",
    "edge_latent_size": "96",
    "processor_steps": "6",
    "mlp_hidden_layers": "2",
    "batch_size": "88",
    "validation_batch_size": "88",
    "prediction_batch_size": "88",
    "batch_plan": "sample_shuffle",
    "optimizer": "adamw",
    "lr": "0.0005",
    "model_seed": "0",
    "batch_order_seed": "0",
    "graph_seed": "0",
    "seed": "0",
    "multi_seed": "[]",
    "batch_build_seed": "0",
    "lr_schedule": "warmup_cosine",
    "warmup_epochs": "10",
    "min_lr": "0.00005",
    "weight_decay": "0.0001",
    "epochs": "600",
    "graph_radius_policy": "discrete_physical_coverage",
    "coverage_repair_policy": "none",
    "loss_mode": "mse",
    "selection_metric": "valid_base_mse",
    "dry_run_required": "true",
    "launch_policy": "explicit_user_instruction_only",
}


def main() -> int:
    args = _parse_args()
    rows = check_registry(_repo_path(args.registry), emit_warnings=True)
    print(f"checked authoritative registry: {args.registry}")
    print(f"checked registry rows: {len(rows)}")
    for row in rows:
        print(f"  {row['config_id']}: {row['generated_yaml']}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    return parser.parse_args()


def check_registry(
    registry_path: Path | str = DEFAULT_REGISTRY, *, emit_warnings: bool = False
) -> list[dict[str, str]]:
    registry_path = _repo_path(registry_path)
    registry = load_registry(registry_path)
    rows = registry_rows(registry)
    _check_csv_mirror(registry, rows)
    for row in rows:
        _check_generated_yaml(row)
        if row["config_id"] == "V4_baseline":
            _check_v4_baseline(row)
    if not any(row["config_id"] == "V4_baseline" for row in rows):
        raise AssertionError("registry must contain V4_baseline")
    if emit_warnings:
        for warning in runner_control_warnings(rows):
            print(f"WARNING: {warning}")
    return rows


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    path = _repo_path(path)
    with path.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, dict):
        raise ValueError(f"{path}: registry must be a JSON object")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    if registry.get("registry_role") != "authoritative":
        raise ValueError(f"{path}: registry_role must be 'authoritative'")
    if not isinstance(registry.get("csv_mirror_path"), str):
        raise ValueError(f"{path}: csv_mirror_path must be a string")
    if not isinstance(registry.get("baseline"), dict):
        raise ValueError(f"{path}: baseline must be an object")
    if not isinstance(registry.get("runs"), dict) or not registry["runs"]:
        raise ValueError(f"{path}: runs must be a non-empty object")
    return registry


def registry_rows(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    baseline = _normalize_resolved_row(
        registry.get("baseline", {}), context="registry baseline"
    )
    if baseline["config_id"] != "V4_baseline":
        raise ValueError("registry baseline config_id must be 'V4_baseline'")
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("registry runs must be a mapping")
    for key, raw_run in runs.items():
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"registry run {key!r} must be a mapping")
        extra = sorted(set(raw_run) - {"config_id", "overrides"})
        if extra:
            raise ValueError(
                f"registry run {key!r} must contain only config_id and overrides; "
                f"unsupported fields: {', '.join(extra)}"
            )
        config_id = _stringify(raw_run.get("config_id", key))
        if key != config_id:
            raise ValueError(f"registry key {key!r} must match config_id {config_id!r}")
        overrides = raw_run.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"registry run {key!r} overrides must be a mapping")
        row = _resolve_registry_row(baseline, config_id, overrides)
        config_id = row["config_id"]
        if config_id in seen:
            raise ValueError(f"duplicate config_id {config_id!r}")
        seen.add(config_id)
        rows.append(row)
    _check_unique_resolved_fields(rows)
    return rows


def write_csv_mirror(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(CSV_FIELDNAMES), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_inherited_yaml(row: Mapping[str, str]) -> dict[str, Any]:
    base_path = _repo_path(row["base_yaml"])
    generated_path = _repo_path(row["generated_yaml"])
    base_config = load_v2_config(base_path)
    desired = _desired_config_from_row(row)
    overrides = _diff_mapping(base_config, desired)
    return {
        "schema_version": INHERITED_SCHEMA_VERSION,
        "config_id": row["config_id"],
        "extends": _relative_path(base_path, generated_path.parent),
        "overrides": overrides,
    }


def resolve_inherited_yaml(
    inherited: Mapping[str, Any], generated_path: Path
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


def runner_control_warnings(rows: list[dict[str, str]]) -> list[str]:
    baseline = next((row for row in rows if row["config_id"] == "V4_baseline"), None)
    if baseline is None:
        return []
    baseline_config = _resolved_config_from_row(baseline)
    baseline_unmapped = _runner_unmapped_field_reasons(baseline_config)
    warnings: list[str] = []
    for row in rows:
        if row["config_id"] == "V4_baseline":
            continue
        config = _resolved_config_from_row(row)
        unmapped = _runner_unmapped_field_reasons(config)
        for field in sorted(set(baseline_unmapped) | set(unmapped)):
            actual = _get_dotted(config, field)
            expected = _get_dotted(baseline_config, field)
            if actual != expected:
                reason = unmapped.get(field) or baseline_unmapped.get(field)
                warnings.append(
                    f"{row['config_id']}: {field} differs from V4_baseline "
                    f"({expected!r} -> {actual!r}); V2 runner reports this "
                    f"field as unmapped: {reason}"
                )
    return warnings


def _resolved_config_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    return resolve_inherited_yaml(
        build_inherited_yaml(row), _repo_path(row["generated_yaml"])
    )


def _runner_unmapped_field_reasons(config: Mapping[str, Any]) -> dict[str, str]:
    plan = build_v2_command_plan(config)
    reasons: dict[str, str] = {}
    for entry in plan.get("unmapped_fields", []):
        field = entry.get("field")
        reason = entry.get("reason")
        if isinstance(field, str) and isinstance(reason, str):
            reasons[field] = reason
    return reasons


def _resolve_registry_row(
    baseline: Mapping[str, str], config_id: str, overrides: Mapping[str, Any]
) -> dict[str, str]:
    if "config_id" in overrides:
        raise ValueError(
            f"registry run {config_id!r} overrides must not contain config_id; "
            "use the run key instead"
        )
    extra = sorted(set(overrides) - set(CSV_FIELDNAMES))
    if extra:
        raise ValueError(
            f"registry run {config_id!r} overrides unsupported fields: "
            f"{', '.join(extra)}"
        )
    raw_row: dict[str, Any] = dict(baseline)
    raw_row.update(overrides)
    raw_row["config_id"] = config_id
    return _normalize_resolved_row(
        raw_row, context=f"resolved registry run {config_id}"
    )


def _normalize_resolved_row(
    raw_row: Mapping[str, Any], *, context: str
) -> dict[str, str]:
    missing = [field for field in CSV_FIELDNAMES if field not in raw_row]
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")
    extra = sorted(set(raw_row) - set(CSV_FIELDNAMES))
    if extra:
        raise ValueError(f"{context} has unsupported fields: {', '.join(extra)}")
    row = {field: _stringify(raw_row[field]) for field in CSV_FIELDNAMES}
    if not row["config_id"]:
        raise ValueError(f"{context} has empty config_id")
    return row


def _check_unique_resolved_fields(rows: list[dict[str, str]]) -> None:
    for field in UNIQUE_RESOLVED_FIELDS:
        seen: dict[str, str] = {}
        for row in rows:
            value = row[field]
            prior = seen.get(value)
            if prior is not None:
                raise ValueError(
                    f"registry conflict: {field} {value!r} is used by "
                    f"{prior!r} and {row['config_id']!r}"
                )
            seen[value] = row["config_id"]


def _check_csv_mirror(
    registry: Mapping[str, Any], rows: list[dict[str, str]]
) -> None:
    mirror_path = _repo_path(str(registry["csv_mirror_path"]))
    with mirror_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDNAMES:
            raise AssertionError(f"{mirror_path}: CSV mirror header mismatch")
        csv_rows = list(reader)
    if csv_rows != rows:
        raise AssertionError(
            f"{mirror_path}: CSV audit mirror differs from authoritative JSON"
        )


def _check_generated_yaml(row: Mapping[str, str]) -> None:
    expected = build_inherited_yaml(row)
    generated_path = _repo_path(row["generated_yaml"])
    with generated_path.open("r", encoding="utf-8") as file:
        actual = yaml.safe_load(file)
    if actual != expected:
        raise AssertionError(
            f"{row['generated_yaml']}: generated YAML is not reproducible "
            "from authoritative JSON"
        )
    resolved = resolve_inherited_yaml(actual, generated_path)
    validate_v2_config(resolved, config_path=_repo_path(row["base_yaml"]))
    _assert_registry_matches_resolved(row, resolved)


def _check_v4_baseline(row: Mapping[str, str]) -> None:
    for field, expected in EXPECTED_V4_BASELINE.items():
        actual = row.get(field)
        if actual != expected:
            raise AssertionError(
                f"V4_baseline {field} expected {expected!r}, got {actual!r}"
            )
    base = load_v2_config(_repo_path(row["base_yaml"]))
    resolved = resolve_inherited_yaml(
        build_inherited_yaml(row), _repo_path(row["generated_yaml"])
    )
    for config, label in ((base, "V4_base"), (resolved, "V4_baseline")):
        checks = {
            "model.node_latent_size": 96,
            "model.edge_latent_size": 96,
            "model.processor_steps": 6,
            "model.mlp_hidden_layers": 2,
            "run.batch_size": 88,
            "run.batch_plan": "sample_shuffle",
            "optimizer.name": "adamw",
            "optimizer.lr": 0.0005,
            "optimizer.model_seed": 0,
            "optimizer.batch_order_seed": 0,
            "optimizer.graph_seed": 0,
            "optimizer.seed": 0,
            "optimizer.multi_seed": [],
            "optimizer.lr_schedule": "warmup_cosine",
            "optimizer.warmup_epochs": 10,
            "optimizer.min_lr": 0.00005,
            "optimizer.weight_decay": 0.0001,
            "run.epochs": 600,
            "run.batch_build_seed": 0,
            "graph.radius_policy": "discrete_physical_coverage",
            "graph.coverage_repair_policy": "none",
            "loss.mode": "mse",
            "export.selection_metric": "valid_base_mse",
        }
        for dotted, expected in checks.items():
            actual = _get_dotted(config, dotted)
            if actual != expected:
                raise AssertionError(
                    f"{label} {dotted} expected {expected!r}, got {actual!r}"
                )


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
            "model_seed": _int(row, "model_seed"),
            "batch_order_seed": _int(row, "batch_order_seed"),
            "graph_seed": _int(row, "graph_seed"),
            "seed": _int(row, "seed"),
            "multi_seed": _json_list(row, "multi_seed"),
            "lr_schedule": row["lr_schedule"],
            "warmup_epochs": _int(row, "warmup_epochs"),
            "min_lr": _float(row, "min_lr"),
            "weight_decay": _float(row, "weight_decay"),
        },
        "loss": {"mode": row["loss_mode"]},
        "run": {
            "epochs": _int(row, "epochs"),
            "batch_size": _int(row, "batch_size"),
            "validation_batch_size": _int(row, "validation_batch_size"),
            "prediction_batch_size": _int(row, "prediction_batch_size"),
            "batch_plan": row["batch_plan"],
            "batch_build_seed": _int(row, "batch_build_seed"),
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
        "run.batch_build_seed": _int(row, "batch_build_seed"),
        "run.epochs": _int(row, "epochs"),
        "optimizer.name": row["optimizer"],
        "optimizer.lr": _float(row, "lr"),
        "optimizer.model_seed": _int(row, "model_seed"),
        "optimizer.batch_order_seed": _int(row, "batch_order_seed"),
        "optimizer.graph_seed": _int(row, "graph_seed"),
        "optimizer.seed": _int(row, "seed"),
        "optimizer.multi_seed": _json_list(row, "multi_seed"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)


def write_generated_yaml(row: Mapping[str, str]) -> Path:
    path = _repo_path(row["generated_yaml"])
    _write_yaml(path, build_inherited_yaml(row))
    return path


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _int(row: Mapping[str, str], key: str) -> int:
    return int(row[key])


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _json_list(row: Mapping[str, str], key: str) -> list[Any]:
    value = json.loads(row[key])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return value


def _get_dotted(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
