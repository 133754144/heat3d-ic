#!/usr/bin/env python3
"""Collect direct timing, structured-FVM sensitivity, and holdout results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESOLUTIONS = (4096, 8192, 16384, 32768)
PRIMARY = (4096, 8192, 16384)
MODES = ("cold", "cached", "persistent")
ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_paths(value: Any) -> Any:
    """Remove machine-local absolute paths from tracked governance artifacts."""
    if isinstance(value, dict):
        return {key: _sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/private/tmp/"):
        return "runtime-artifact://" + Path(value).name
    if isinstance(value, str) and value.startswith("/Users/"):
        marker = "/data/heat3d_v6_p1h_shared_support1024_v0"
        if marker in value:
            suffix = value.split(marker, 1)[1]
            return "data/heat3d_v6_p1h_shared_support1024_v0" + suffix
        return "local-artifact://" + Path(value).name
    if isinstance(value, str) and value.startswith("/home/"):
        return "remote-artifact://" + Path(value).name
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _mean_cycle_field(payload, section, key):
    return float(
        np.mean([cycle[section][key] for cycle in payload["cycles"]])
    )


def _timing_row(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload["timing_summary_seconds"]
    cycles = payload["cycles"]
    memory = [
        cycle["device_memory"].get("peak_bytes_in_use")
        for cycle in cycles
        if cycle["device_memory"].get("peak_bytes_in_use") is not None
    ]
    row = {
        "platform": payload["platform"],
        "mode": payload["mode"],
        "batch_size": payload["batch_size"],
        "resolution": payload["resolution"],
        "sample_count": payload["sample_count"],
        "model_core_seconds": direct["model_core"]["mean"],
        "full_field_production_seconds": direct["full_field_production"]["mean"],
        "evaluation_seconds": direct["evaluation"]["mean"],
        "model_core_seconds_per_sample": direct["model_core"]["mean"] / 128.0,
        "production_seconds_per_sample": direct["full_field_production"]["mean"]
        / 128.0,
        "evaluation_seconds_per_sample": direct["evaluation"]["mean"] / 128.0,
        "production_samples_per_second": 128.0
        / direct["full_field_production"]["mean"],
        "input_seconds": _mean_cycle_field(
            payload, "production_phase_seconds", "input"
        ),
        "graph_load_or_build_seconds": _mean_cycle_field(
            payload, "production_phase_seconds", "graph_load_or_build"
        ),
        "batch_prepare_seconds": _mean_cycle_field(
            payload, "production_phase_seconds", "batch_prepare"
        ),
        "full_field_reconstruction_seconds": _mean_cycle_field(
            payload, "production_phase_seconds", "full_field_reconstruction"
        ),
        "serialization_seconds": _mean_cycle_field(
            payload, "production_phase_seconds", "serialization"
        ),
        "label_read_seconds": _mean_cycle_field(
            payload, "evaluation_phase_seconds", "label_read_seconds"
        ),
        "metrics_diagnostics_seconds": _mean_cycle_field(
            payload,
            "evaluation_phase_seconds",
            "metrics_diagnostics_seconds",
        ),
        "process_peak_ram_GB": max(
            cycle["process_peak_ram_bytes"] for cycle in cycles
        )
        / 1.0e9,
        "device_peak_memory_GB": (
            max(memory) / 1.0e9 if memory else "N/A"
        ),
        "direct_single_cycle_measurements": True,
        "cross_run_phase_addition": False,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-b8-dir", type=Path, required=True)
    parser.add_argument("--gpu-b16-dir", type=Path, required=True)
    parser.add_argument("--legal-structured-fvm", type=Path, required=True)
    parser.add_argument("--runner-smoke", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--excluded-test-dir", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--persistent-gpu-csv", type=Path, required=True)
    parser.add_argument("--solver-comparison-csv", type=Path, required=True)
    parser.add_argument("--pareto-csv", type=Path, required=True)
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--gpu-environment", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    timing_payloads = []
    for resolution in RESOLUTIONS:
        for mode in MODES:
            timing_payloads.append(
                _load(args.cpu_dir / f"{resolution}_{mode}_b8.json")
            )
            timing_payloads.append(
                _load(args.gpu_b8_dir / f"{resolution}_{mode}_b8.json")
            )
    for resolution in PRIMARY:
        timing_payloads.append(
            _load(args.gpu_b16_dir / f"{resolution}_persistent_b16.json")
        )
    timing_rows = [_timing_row(payload) for payload in timing_payloads]
    _write_csv(args.timing_csv, timing_rows)

    fvm = _load(args.legal_structured_fvm)
    fvm["schema_version"] = (
        "heat3d_v6_legal_structured_fvm_mesh_sensitivity_v1"
    )
    reference = fvm["meshes"]["reference"]
    cold_total = (
        reference["cold_mesh_assembly_solve_seconds"]["mean"] * 128.0
    )
    warm_total = reference["warm_solve_seconds"]["mean"] * 128.0
    solver_comparison_rows = []
    for resolution in RESOLUTIONS:
        cpu_cold = _timing_row(
            _load(args.cpu_dir / f"{resolution}_cold_b8.json")
        )
        cpu_cached = _timing_row(
            _load(args.cpu_dir / f"{resolution}_cached_b8.json")
        )
        cpu_persistent_row = _timing_row(
            _load(args.cpu_dir / f"{resolution}_persistent_b8.json")
        )
        gpu_cold = _timing_row(
            _load(args.gpu_b8_dir / f"{resolution}_cold_b8.json")
        )
        gpu_cached = _timing_row(
            _load(args.gpu_b8_dir / f"{resolution}_cached_b8.json")
        )
        gpu_persistent_row = _timing_row(
            _load(args.gpu_b8_dir / f"{resolution}_persistent_b8.json")
        )
        solver_comparison_rows.append(
            {
                "query_resolution": resolution,
                "sample_count": 128,
                "cpu_cold_model_core_s": cpu_cold["model_core_seconds"],
                "cpu_cold_production_with_graph_build_s": cpu_cold[
                    "full_field_production_seconds"
                ],
                "cpu_cached_production_with_graph_load_s": cpu_cached[
                    "full_field_production_seconds"
                ],
                "cpu_persistent_production_s": cpu_persistent_row[
                    "full_field_production_seconds"
                ],
                "gpu_b8_cold_model_core_s": gpu_cold["model_core_seconds"],
                "gpu_b8_cold_production_with_graph_build_s": gpu_cold[
                    "full_field_production_seconds"
                ],
                "gpu_b8_cached_production_with_graph_load_s": gpu_cached[
                    "full_field_production_seconds"
                ],
                "gpu_b8_persistent_production_s": gpu_persistent_row[
                    "full_field_production_seconds"
                ],
                "fvm_reference_nodes": reference["solver_node_count"],
                "fvm_reference_cold_s_per_sample": reference[
                    "cold_mesh_assembly_solve_seconds"
                ]["mean"],
                "fvm_reference_warm_s_per_sample": reference[
                    "warm_solve_seconds"
                ]["mean"],
                "fvm_reference_cold_s_valid128": cold_total,
                "fvm_reference_warm_s_valid128": warm_total,
                "gpu_b8_persistent_fvm_cold_speedup": (
                    cold_total
                    / gpu_persistent_row["full_field_production_seconds"]
                ),
                "gpu_b8_persistent_fvm_warm_speedup": (
                    warm_total
                    / gpu_persistent_row["full_field_production_seconds"]
                ),
                "nonmatched_dof": True,
            }
        )
    _write_csv(args.solver_comparison_csv, solver_comparison_rows)
    persistent_gpu_rows = []
    for batch_size, directory in ((8, args.gpu_b8_dir), (16, args.gpu_b16_dir)):
        for resolution in PRIMARY:
            suffix = (
                f"{resolution}_persistent_b8.json"
                if batch_size == 8
                else f"{resolution}_persistent_b16.json"
            )
            payload = _load(directory / suffix)
            row = _timing_row(payload)
            production = row["full_field_production_seconds"]
            persistent_gpu_rows.append(
                {
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "production_seconds_valid128": production,
                    "latency_seconds_per_sample": production / 128.0,
                    "samples_per_second": 128.0 / production,
                    "device_peak_memory_GB": row["device_peak_memory_GB"],
                    "fvm_cold_speedup": cold_total / production,
                    "fvm_warm_speedup": warm_total / production,
                }
            )
    _write_csv(args.persistent_gpu_csv, persistent_gpu_rows)

    cpu_persistent = {
        resolution: _load(
            args.cpu_dir / f"{resolution}_persistent_b8.json"
        )
        for resolution in PRIMARY
    }
    pareto_rows = []
    for label in ("coarse", "medium", "reference"):
        row = fvm["meshes"][label]
        pareto_rows.append(
            {
                "family": "FVM",
                "method": label,
                "nodes": row["solver_node_count"],
                "runtime_seconds_per_sample": row["warm_solve_seconds"]["mean"],
                "raw_cv_weighted_rmse_K": row[
                    "accuracy_vs_240825_reference"
                ]["raw_cv_weighted_rmse_K"],
                "point_global_relative_rmse_pct": row[
                    "accuracy_vs_240825_reference"
                ]["cv_weighted_point_global_relative_rmse_pct"],
                "runtime_scope": "warm_solve",
                "evaluation_role": "valid_iid",
            }
        )
    for resolution, payload in cpu_persistent.items():
        cycle = payload["cycles"][0]
        metrics = cycle["metrics"]["full_field"]
        pareto_rows.append(
            {
                "family": "V6_anchor_derived",
                "method": str(resolution),
                "nodes": resolution,
                "runtime_seconds_per_sample": payload[
                    "timing_summary_seconds"
                ]["full_field_production"]["mean"]
                / 128.0,
                "raw_cv_weighted_rmse_K": metrics["cv_weighted_rmse_K"],
                "point_global_relative_rmse_pct": metrics[
                    "cv_weighted_point_global_relative_rmse_pct"
                ],
                "runtime_scope": "persistent_full_field_production_CPU_B8",
                "evaluation_role": "valid_iid",
            }
        )
    _write_csv(args.pareto_csv, pareto_rows)

    test_rows = []
    test_payloads = {}
    for resolution in PRIMARY:
        payload = _load(args.test_dir / f"test_{resolution}.json")
        test_payloads[str(resolution)] = payload
        cycle = payload["cycles"][0]
        support = cycle["metrics"]["support"]
        full = cycle["metrics"]["full_field"]
        test_rows.append(
            {
                "resolution": resolution,
                "checkpoint_epoch": payload["checkpoint"]["epoch"],
                "checkpoint_sha256": payload["checkpoint"]["sha256"],
                "support_point_global_pct": support[
                    "point_global_cv_relative_rmse_pct"
                ],
                "support_sample_first_pct": support[
                    "sample_first_cv_relative_rmse_pct"
                ],
                "support_raw_cv_rmse_K": support[
                    "raw_cv_weighted_rmse_K"
                ],
                "full_point_global_pct": full[
                    "cv_weighted_point_global_relative_rmse_pct"
                ],
                "full_sample_first_pct": full[
                    "sample_first_cv_relative_rmse_pct"
                ],
                "full_raw_cv_rmse_K": full["cv_weighted_rmse_K"],
                "full_peak_rmse_K": full["peak_error_rmse_K"],
                "full_source_rmse_K": full["source_cv_weighted_rmse_K"],
                "hard_accessed": False,
                "used_for_selection": False,
                "role_classification": "corrected_confirmatory_holdout",
            }
        )
    _write_csv(args.holdout_csv, test_rows)
    excluded_test_results = []
    for resolution in (4096, 8192):
        path = args.excluded_test_dir / f"test_{resolution}.json"
        excluded_test_results.append(
            {
                "resolution": resolution,
                "sha256": _sha256(path),
                "status": "excluded_wrong_ladder_input",
                "wrong_ladder": "configs/heat3d_v6/v6_anchored_probe_ladder.json",
                "frozen_ladder": (
                    "configs/heat3d_v6/"
                    "v6_source_aware_resolution_ladder.json"
                ),
                "used_for_selection_or_reporting": False,
            }
        )
    test_opening_audit = {
        "deviation_id": "V6-PROTOCOL-DEVIATION-TEST-LADDER-001",
        "classification": "protocol_deviation_corrected_before_formal_reporting",
        "status": "completed_with_corrected_command_input",
        "opening_session": "single_post_preregistration_closeout_session",
        "incident": (
            "The first command used the legacy anchored ladder for 4096/8192. "
            "Those temporary outputs are excluded. The 16384 attempt stopped "
            "at ladder-key lookup before label loading. The frozen source-aware "
            "workflow was then executed for all three preregistered resolutions."
        ),
        "excluded_temporary_results": excluded_test_results,
        "formal_result_sha256": {
            str(resolution): _sha256(
                args.test_dir / f"test_{resolution}.json"
            )
            for resolution in PRIMARY
        },
        "ladder_sha256": {
            "excluded_anchored": _sha256(
                ROOT
                / "configs/heat3d_v6/v6_anchored_probe_ladder.json"
            ),
            "frozen_source_aware": _sha256(
                ROOT
                / "configs/heat3d_v6/v6_source_aware_resolution_ladder.json"
            ),
        },
        "selection_or_workflow_changed": False,
        "hard_accessed": False,
    }

    colors = {"FVM": "tab:blue", "V6_anchor_derived": "tab:orange"}
    markers = {"FVM": "o", "V6_anchor_derived": "s"}
    plt.figure(figsize=(7.6, 5.0))
    for family in colors:
        rows = [row for row in pareto_rows if row["family"] == family]
        plt.plot(
            [float(row["runtime_seconds_per_sample"]) for row in rows],
            [float(row["raw_cv_weighted_rmse_K"]) for row in rows],
            marker=markers[family],
            color=colors[family],
            label=family,
        )
        for row in rows:
            plt.annotate(
                row["method"],
                (
                    float(row["runtime_seconds_per_sample"]),
                    float(row["raw_cv_weighted_rmse_K"]),
                ),
            )
    plt.xlabel("CPU runtime per sample (s)")
    plt.ylabel("CV-weighted full-field RMSE (K)")
    plt.title("V6 valid_iid accuracy-runtime Pareto")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=180)
    plt.close()

    environment = {
        "collector": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "cpu": next(
            payload["environment"]
            for payload in timing_payloads
            if payload["platform"] == "cpu"
        ),
        "gpu": next(
            payload["environment"]
            for payload in timing_payloads
            if payload["platform"] == "gpu"
        ),
        "gpu_hosts": _load(args.gpu_environment),
    }
    args.environment_json.parent.mkdir(parents=True, exist_ok=True)
    args.environment_json.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner_smoke = _load(args.runner_smoke)
    payload = {
        "schema_version": "heat3d_v6_final_performance_governance_v2",
        "status": "passed",
        "preregistration_commit": args.preregistration_commit,
        "checkpoint_modified": False,
        "training_executed": False,
        "confirmatory_holdout_classification": (
            "corrected_confirmatory_holdout"
        ),
        "confirmatory_holdout_opened_after_preregistration": True,
        "confirmatory_holdout_used_for_selection": False,
        "hard_accessed": False,
        "timing_contract": timing_payloads[0]["timing_contract"],
        "timing_rows": timing_rows,
        "persistent_gpu": persistent_gpu_rows,
        "solver_inference_comparison": solver_comparison_rows,
        "runner_graph_reuse": runner_smoke,
        "legal_structured_fvm_mesh_sensitivity": fvm,
        "pareto_rows": pareto_rows,
        "corrected_confirmatory_holdout": test_payloads,
        "protocol_deviation": test_opening_audit,
        "frozen_decision_unchanged": {
            "default_hotspot_oriented": 4096,
            "balanced_full_field": 8192,
            "maximum_full_field_accuracy": 16384,
            "experimental_excluded_from_primary_test_table": 32768,
        },
        "decision_basis": {
            "selection_source": (
                "valid_iid_timing_and_accuracy_before_holdout_open"
            ),
            "confirmatory_holdout_role": "descriptive_confirmation_only",
            "default_4096": "lowest frozen production resolution",
            "full_field_8192": (
                "lower valid full-field error than 4096 with modest persistent "
                "GPU cost"
            ),
            "maximum_full_field_accuracy_16384": (
                "lowest preregistered valid full-field error; higher memory and "
                "latency than 8192"
            ),
            "experimental_32768": (
                "timing and valid experiment only; excluded from primary test "
                "table and production defaults"
            ),
        },
    }
    payload = _sanitize_paths(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V6 final performance closeout",
        "",
        "The model, checkpoint, sampling, graph parameters, and reconstruction "
        "method remained frozen. The corrected confirmatory holdout was opened "
        "after preregistration; "
        "hard remained sealed.",
        "",
        "The first test command used the legacy ladder for temporary 4096/8192 "
        "outputs. They are explicitly excluded; the 16384 attempt stopped before "
        "label loading. The formal table below is the corrected, frozen "
        "source-aware workflow and did not change any selection.",
        "",
        "## Direct timing",
        "",
        "| Platform | Mode | B | Nodes | Model-core s | Production s | Evaluation s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            f"| {row['platform']} | {row['mode']} | {row['batch_size']} | "
            f"{row['resolution']} | {row['model_core_seconds']:.3f} | "
            f"{row['full_field_production_seconds']:.3f} | "
            f"{row['evaluation_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Inference versus 240825-node FVM",
            "",
            "All model values below are direct 128-sample cycles. Cold production "
            "includes graph build; cached production includes graph-cache load; "
            "model-core excludes graph and full-field reconstruction. The FVM "
            "comparison is explicitly nonmatched-DOF.",
            "",
            "| Query nodes | CPU core/no graph s | CPU cold/cached/persistent production s | GPU B8 core/no graph s | GPU B8 cold/cached/persistent production s | FVM cold/warm s per 128 | GPU persistent speedup cold/warm |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in solver_comparison_rows:
        lines.append(
            f"| {row['query_resolution']} | "
            f"{row['cpu_cold_model_core_s']:.3f} | "
            f"{row['cpu_cold_production_with_graph_build_s']:.3f}/"
            f"{row['cpu_cached_production_with_graph_load_s']:.3f}/"
            f"{row['cpu_persistent_production_s']:.3f} | "
            f"{row['gpu_b8_cold_model_core_s']:.3f} | "
            f"{row['gpu_b8_cold_production_with_graph_build_s']:.3f}/"
            f"{row['gpu_b8_cached_production_with_graph_load_s']:.3f}/"
            f"{row['gpu_b8_persistent_production_s']:.3f} | "
            f"{row['fvm_reference_cold_s_valid128']:.3f}/"
            f"{row['fvm_reference_warm_s_valid128']:.3f} | "
            f"{row['gpu_b8_persistent_fvm_cold_speedup']:.2f}×/"
            f"{row['gpu_b8_persistent_fvm_warm_speedup']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Legal structured-FVM mesh sensitivity",
            "",
            "| Mesh | Nodes | Cold mean/median/P95 s | Warm mean/median/P95 s | Full-field RMSE K |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ("coarse", "medium", "reference"):
        row = fvm["meshes"][label]
        cold = row["cold_mesh_assembly_solve_seconds"]
        warm = row["warm_solve_seconds"]
        lines.append(
            f"| {label} | {row['solver_node_count']} | "
            f"{cold['mean']:.3f}/{cold['median']:.3f}/{cold['p95']:.3f} | "
            f"{warm['mean']:.3f}/{warm['median']:.3f}/{warm['p95']:.3f} | "
            f"{row['accuracy_vs_240825_reference']['raw_cv_weighted_rmse_K']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Persistent GPU",
            "",
            "| B | Nodes | Production s/128 | sample/s | VRAM GB | cold/warm FVM speedup |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in persistent_gpu_rows:
        lines.append(
            f"| {row['batch_size']} | {row['resolution']} | "
            f"{row['production_seconds_valid128']:.3f} | "
            f"{row['samples_per_second']:.3f} | "
            f"{float(row['device_peak_memory_GB']):.3f} | "
            f"{row['fvm_cold_speedup']:.2f}×/{row['fvm_warm_speedup']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Corrected confirmatory holdout",
            "",
            "| Nodes | Support point-global | Full point-global | Full RMSE K |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| {row['resolution']} | {row['support_point_global_pct']:.4f}% | "
            f"{row['full_point_global_pct']:.4f}% | "
            f"{row['full_raw_cv_rmse_K']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The confirmatory table is descriptive only. It did not change the frozen "
            "4096/8192/16384 roles. 32768 is excluded; hard remains sealed.",
            "",
            "## Frozen decision",
            "",
            "- 4096 remains the default/hotspot-oriented mode.",
            "- 8192 remains the balanced full-field mode.",
            "- 16384 remains the maximum full-field accuracy mode.",
            "- 32768 remains experimental and was not included in the "
            "confirmatory table.",
            "- The decision was fixed from valid_iid before holdout opening; the "
            "corrected confirmatory holdout is descriptive only.",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"status": "passed", "confirmatory_holdout_rows": len(test_rows)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
