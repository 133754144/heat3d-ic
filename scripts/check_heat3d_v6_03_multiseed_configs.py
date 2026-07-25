#!/usr/bin/env python3
"""Validate V6_03 seed1/seed2 single-variable inherited configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


BASE_PATH = ROOT / "configs/heat3d_v6/V6_03_V5best_P1h.yaml"
CANDIDATES = {
    1: ROOT / "configs/heat3d_v6/V6_03_V5best_P1h_seed1.yaml",
    2: ROOT / "configs/heat3d_v6/V6_03_V5best_P1h_seed2.yaml",
}
RESOLVED_DIR = ROOT / "configs/heat3d_v6/resolved"
REPORT_JSON = ROOT / "configs/heat3d_v6/v6_03_multiseed_resolved_diff.json"
REPORT_MD = ROOT / "docs/v6_03_multiseed_preparation.md"
SEED_PATHS = {
    "optimizer.seed",
    "optimizer.model_seed",
    "optimizer.batch_order_seed",
    "optimizer.graph_seed",
}
IDENTITY_PATHS = {
    "config_id",
    "description",
    "export.output_dir",
    "export.run_name",
    "metadata.canonical_dataset_id",
    "metadata.dataset_lifecycle_status",
    "metadata.experiment_role",
    "metadata.log_path",
    "metadata.phase",
    "metadata.seed_index",
    "metadata.seed_parent_config_id",
}


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = resolve_inherited_yaml(payload, path)
    result["config_id"] = payload["config_id"]
    return result


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaf_diffs(left.get(key), right.get(key), path))
        return result
    if left == right:
        return []
    return [{"path": prefix, "base": left, "candidate": right}]


def _optimizer_without_seeds(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for key in ("seed", "model_seed", "batch_order_seed", "graph_seed"):
        result.pop(key)
    return result


def _dry_run(config: Mapping[str, Any], seed: int) -> list[str]:
    command = build_training_command(config, python_executable="python")
    expected = {
        "--seed": seed,
        "--model-seed": seed,
        "--batch-order-seed": seed,
        "--graph-seed": seed,
        "--batch-size": 24,
        "--micro-batch-size": 24,
        "--epochs": 600,
    }
    for flag, value in expected.items():
        if flag not in command or command[command.index(flag) + 1] != str(value):
            raise AssertionError(f"seed{seed}: dry-run flag drifted: {flag}")
    if "--epoch-wise-batch-regrouping" in command:
        raise AssertionError(f"seed{seed}: regrouping unexpectedly enabled")
    return command


def build_report() -> dict[str, Any]:
    base = _resolved(BASE_PATH)
    validate_v2_config(base, config_path=BASE_PATH)
    reports = {}
    output_dirs = set()
    run_names = set()
    log_paths = set()
    for seed, path in CANDIDATES.items():
        candidate = _resolved(path)
        validate_v2_config(candidate, config_path=path)
        diffs = _leaf_diffs(base, candidate)
        paths = {row["path"] for row in diffs}
        expected = SEED_PATHS | IDENTITY_PATHS
        if paths != expected:
            raise AssertionError(
                f"seed{seed}: resolved diff mismatch "
                f"missing={sorted(expected - paths)} "
                f"unexpected={sorted(paths - expected)}"
            )
        if candidate["dataset"] != base["dataset"]:
            raise AssertionError(f"seed{seed}: dataset drifted")
        for section in ("model", "graph", "loss", "run", "diagnostics"):
            if candidate[section] != base[section]:
                raise AssertionError(f"seed{seed}: {section} drifted")
        if _optimizer_without_seeds(candidate["optimizer"]) != (
            _optimizer_without_seeds(base["optimizer"])
        ):
            raise AssertionError(f"seed{seed}: optimizer non-seed field drifted")
        actual_seeds = {
            candidate["optimizer"][key]
            for key in ("seed", "model_seed", "batch_order_seed", "graph_seed")
        }
        if actual_seeds != {seed}:
            raise AssertionError(f"seed{seed}: seed fields are not unified")
        if (
            candidate["run"]["epochs"] != 600
            or candidate["run"]["batch_size"] != 24
            or candidate["run"]["micro_batch_size"] != 24
            or candidate["run"]["init_checkpoint"] is not None
            or candidate["run"]["epoch_wise_batch_regrouping"]
        ):
            raise AssertionError(f"seed{seed}: frozen run contract drifted")
        if (
            candidate["export"]["selection_metric"]
            != base["export"]["selection_metric"]
        ):
            raise AssertionError(f"seed{seed}: checkpoint selection drifted")
        metadata = candidate["metadata"]
        if (
            metadata["training_started"]
            or metadata["execution_host"] is not None
            or metadata["training_commit"] is not None
            or metadata["runner_pid"] is not None
            or metadata["launch_timestamp_utc"] is not None
            or metadata["seed_index"] != seed
            or metadata["dataset_lifecycle_status"] != "canonical"
        ):
            raise AssertionError(f"seed{seed}: execution metadata drifted")
        command = _dry_run(candidate, seed)
        output_dirs.add(candidate["export"]["output_dir"])
        run_names.add(candidate["export"]["run_name"])
        log_paths.add(metadata["log_path"])
        reports[str(seed)] = {
            "config_id": candidate["config_id"],
            "config_path": str(path.relative_to(ROOT)),
            "resolved_path": str(
                (
                    RESOLVED_DIR
                    / f"V6_03_V5best_P1h_seed{seed}.resolved.yaml"
                ).relative_to(ROOT)
            ),
            "resolved_diff_paths": sorted(paths),
            "scientific_diff_paths": sorted(paths - IDENTITY_PATHS),
            "resolved_diffs": diffs,
            "manual_command": (
                "python scripts/run_heat3d_v4_config.py --config "
                f"configs/heat3d_v6/V6_03_V5best_P1h_seed{seed}.yaml"
            ),
            "runner_command": command,
            "training_started": False,
        }
    if len(output_dirs) != 2 or len(run_names) != 2 or len(log_paths) != 2:
        raise AssertionError("seed run/output/log identities are not independent")
    return {
        "schema_version": "heat3d_v6_03_multiseed_preparation_v1",
        "status": "passed",
        "base_config_id": base["config_id"],
        "base_config_path": str(BASE_PATH.relative_to(ROOT)),
        "seed_configs": reports,
        "scientific_difference_policy": (
            "only optimizer.seed/model_seed/batch_order_seed/graph_seed"
        ),
        "dataset_model_loss_lr_batch_epochs_selection_equal": True,
        "independent_run_output_log": True,
        "training_started": False,
        "formal_inference_executed": False,
        "test_hard_accessed": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V6_03 P1h multi-seed preparation",
        "",
        "Status: **passed**. Seed1 and seed2 resolve from the frozen V6_03 seed0",
        "configuration. The only scientific differences are the four registered",
        "seed fields; dataset, model, graph, loss, optimizer hyperparameters, LR,",
        "B24/micro24, e600, and checkpoint selection remain unchanged.",
        "",
        "| config | scientific diff | training started |",
        "|---|---|---|",
    ]
    for row in report["seed_configs"].values():
        lines.append(
            f"| `{row['config_id']}` | "
            f"`{', '.join(row['scientific_diff_paths'])}` | false |"
        )
    lines.extend(["", "## Manual launch commands", ""])
    for row in report["seed_configs"].values():
        lines.extend(["```bash", row["manual_command"], "```", ""])
    lines.extend(
        [
            "No training, optimizer update, test/hard access, or formal inference",
            "was performed by this preparation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_artifacts:
        RESOLVED_DIR.mkdir(parents=True, exist_ok=True)
        for seed, path in CANDIDATES.items():
            (RESOLVED_DIR / f"V6_03_V5best_P1h_seed{seed}.resolved.yaml").write_text(
                yaml.safe_dump(_resolved(path), sort_keys=False),
                encoding="utf-8",
            )
        REPORT_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "configs": {
                    seed: {
                        "config_id": row["config_id"],
                        "scientific_diff_paths": row["scientific_diff_paths"],
                        "manual_command": row["manual_command"],
                    }
                    for seed, row in report["seed_configs"].items()
                },
                "training_started": False,
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
