#!/usr/bin/env python3
"""Prepare and freeze a target-independent V6-P1i follow-up design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import py_compile
from typing import Any, Mapping, Sequence

import numpy as np

import generate_heat3d_v6_p1i_v13 as adapter
import heat3d_v6_p1i_continuous_core as core
import heat3d_v6_p1i_split as splitlib


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.argsort(
        np.argsort(left, kind="mergesort"),
        kind="mergesort",
    )
    y = np.argsort(
        np.argsort(right, kind="mergesort"),
        kind="mergesort",
    )
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return float(np.mean(x * y))


def _paths(prefix: str) -> dict[str, Path]:
    return {
        "inputs": CONFIG_DIR / f"{prefix}_input_definitions.csv",
        "split": CONFIG_DIR / f"{prefix}_split_manifest.json",
        "preflight": CONFIG_DIR / f"{prefix}_preflight.json",
        "freeze": CONFIG_DIR / f"{prefix}_freeze_manifest.json",
    }


def prepare(
    config_path: Path,
    acceptance_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = core.load_config(config_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    prefix = str(config["artifact_prefix"])
    paths = _paths(prefix)
    records, sobol = splitlib.design_records(
        config_path,
        case_builder=adapter.case_from_sobol_v3,
    )
    counts = {
        key: int(value) for key, value in config["split_counts"].items()
    }
    assignment = splitlib.balanced_input_assignment(
        records,
        counts,
        salt=f"{prefix}-balanced-input-v1",
    )
    metrics = splitlib.split_metrics(records, assignment)
    rows = [
        {**row, "split_role": assignment[str(row["sample_id"])]}
        for row in records
    ]
    _csv(paths["inputs"], rows)
    split_payload = {
        "schema_version": "heat3d_v6_p1i_split_manifest_v1",
        "dataset_id": config["dataset_id"],
        "sample_count": len(records),
        "method": "balanced_pre_solve_assignment_v1",
        "algorithm_contract": (
            "configs/heat3d_v6_p1i/"
            "v6_p1i_split_preregistration_v1.json"
        ),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": core.file_sha256(config_path),
        "sobol_seed": int(config["sampling"]["seed"]),
        "sobol_design_sha256": core.canonical_json_sha256(sobol.tolist()),
        "input_definition_path": str(paths["inputs"].relative_to(ROOT)),
        "input_definition_sha256": core.file_sha256(paths["inputs"]),
        "input_payload_sha256": core.canonical_json_sha256(records),
        "assignment": dict(sorted(assignment.items())),
        "assignment_sha256": splitlib.assignment_sha256(assignment),
        "metrics": metrics,
        "target_values_used": False,
        "solver_results_used": False,
        "model_error_used": False,
    }
    _json(paths["split"], split_payload)
    adapter_preflight = adapter.preflight(config_path)
    severity = np.asarray(
        [float(row["continuous_severity"]) for row in records]
    )
    source_area = np.asarray(
        [float(row["mean_source_area_fraction"]) for row in records]
    )
    power = np.asarray(
        [float(row["package_total_power_W"]) for row in records]
    )
    independent = np.asarray(
        [float(row["power_independent_latent"]) for row in records]
    )
    split_gate = acceptance["split_qc"]
    exact_counts = {
        key: int(value) for key, value in split_gate["exact_counts"].items()
    }
    gates = {
        "sample_count": len(records) == int(acceptance["sample_count"]),
        "split_count": metrics["counts"] == exact_counts,
        "split_continuous_ks": (
            metrics["maximum_continuous_ks"]
            <= float(split_gate["maximum_continuous_ks"])
        ),
        "split_discrete_tv": (
            metrics["maximum_discrete_tv"]
            <= float(split_gate["maximum_discrete_tv"])
        ),
        "split_joint": (
            metrics["maximum_joint_discrepancy"]
            <= float(split_gate["maximum_joint_discrepancy"])
        ),
        "source_size_severity_decoupled": abs(
            _rank_correlation(severity, source_area)
        )
        <= 0.2,
        "independent_power_latent_effective": abs(
            _rank_correlation(independent, power)
        )
        >= 0.2,
        "support_at_least_four": (
            adapter_preflight["minimum_support_nodes_per_region"] >= 4
        ),
        "power_range": (
            adapter_preflight["package_power_W"][0] >= 1.7
            and adapter_preflight["package_power_W"][1] <= 20.0
        ),
        "q_range": (
            adapter_preflight["q_W_m3"][0] >= 1.0e8
            and adapter_preflight["q_W_m3"][1] <= 8.0e10
        ),
    }
    report = {
        "schema_version": "heat3d_v6_p1i_followup_preflight_v1",
        "status": "passed" if all(gates.values()) else "failed",
        "dataset_id": config["dataset_id"],
        "sample_count": len(records),
        "checks": gates,
        "split_metrics": metrics,
        "response_proxy_inputs": {
            "severity_power_spearman": _rank_correlation(severity, power),
            "independent_power_latent_power_spearman": _rank_correlation(
                independent, power
            ),
            "severity_mean_source_area_spearman": _rank_correlation(
                severity, source_area
            ),
        },
        "adapter_preflight": adapter_preflight,
        "guardrails": {
            "solver_runs": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
            "temperature_or_model_error_used": False,
            "formal1024_v0_modified": False,
            "v6_or_p1h_modified": False,
        },
    }
    _json(paths["preflight"], report)
    if report["status"] != "passed":
        raise RuntimeError(f"follow-up input preflight failed: {gates}")
    return report, paths


def freeze(
    config_path: Path,
    acceptance_path: Path,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    config = core.load_config(config_path)
    artifacts = [
        config_path,
        acceptance_path,
        CONFIG_DIR / "v6_p1i_split_preregistration_v1.json",
        CONFIG_DIR / "v6_p1i_v3_literature_contract.json",
        CONFIG_DIR / "v6_p1i_formal1024_v0_postmortem.json",
        CONFIG_DIR / "v6_p1i_split_candidate_comparison.csv",
        CONFIG_DIR / "v6_p1i_split_candidate_assignments.json",
        CONFIG_DIR / "v6_p1i_v8_population_power_calibration.json",
        CONFIG_DIR / "v6_p1i_v9_population_power_calibration.json",
        CONFIG_DIR / "v6_p1i_v10_population_power_calibration.json",
        CONFIG_DIR / "v6_p1i_v11_population_power_calibration.json",
        CONFIG_DIR / "v6_p1i_v12_population_power_calibration.json",
        CONFIG_DIR / "v6_p1i_v13_population_power_calibration.json",
        paths["split"],
        paths["inputs"],
        paths["preflight"],
        ROOT / "scripts/heat3d_v6_p1i_split.py",
        ROOT / "scripts/generate_heat3d_v6_p1i_v13.py",
        Path(__file__),
    ]
    payload = {
        "schema_version": "heat3d_v6_p1i_followup_freeze_v1",
        "status": "frozen_before_generation",
        "dataset_id": config["dataset_id"],
        "sample_count": int(config["sample_count"]),
        "sobol_seed": int(config["sampling"]["seed"]),
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": core.file_sha256(path),
            }
            for path in artifacts
        ],
        "guardrails": {
            "solver_runs_before_freeze": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
            "temperature_used_for_split": False,
            "formal1024_v0_modified": False,
            "v6_or_p1h_modified": False,
        },
    }
    _json(paths["freeze"], payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    acceptance_path = args.acceptance.resolve()
    report, paths = prepare(config_path, acceptance_path)
    result = (
        freeze(config_path, acceptance_path, paths)
        if args.freeze
        else report
    )
    for path in (
        ROOT / "scripts/heat3d_v6_p1i_split.py",
        ROOT / "scripts/generate_heat3d_v6_p1i_v13.py",
        Path(__file__),
    ):
        py_compile.compile(str(path), doraise=True)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_id": result["dataset_id"],
                "sample_count": result["sample_count"],
                "freeze": args.freeze,
                "solver_runs": 0,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
