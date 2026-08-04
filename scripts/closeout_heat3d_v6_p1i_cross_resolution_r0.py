#!/usr/bin/env python3
"""Add an exact checkpoint replay baseline to the frozen P1i resolution audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

import audit_heat3d_v6_p1i_controlled_cross_resolution as audit
import benchmark_heat3d_v6_inference_qualification as base


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "point_global_true_rms_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "peak_rmse_K",
    "source_rmse_K",
    "interface_drop_rmse_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def worker_command(
    args: argparse.Namespace,
    *,
    support_mode: str,
    resolution: int,
    seed: int,
    sample_count: int,
    output: Path,
    prediction_npz: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_heat3d_v6_p1i_controlled_cross_resolution.py"),
        "--worker",
        "--resolution", str(resolution),
        "--discretization-seed", str(seed),
        "--support-mode", support_mode,
        "--regional-mode", "fixed_training_nr",
        "--sample-count", str(sample_count),
        "--dataset-root", str(args.dataset_root),
        "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields),
        "--run-dir", str(args.run_dir),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--checkpoint-epoch", str(args.checkpoint_epoch),
        "--checkpoint-sha-preverified",
        "--output", str(output),
    ]
    if support_mode == "checkpoint_replay":
        command.extend(("--edge-targets", str(args.edge_targets)))
    if prediction_npz is not None:
        command.extend(("--prediction-npz", str(prediction_npz)))
    return command


def run_worker(command: Sequence[str], timeout: int) -> tuple[dict[str, Any], float]:
    env = dict(os.environ)
    env.update({
        "HEAT3D_REPO_ROOT": str(ROOT),
        "MEM_FRACTION": "0.85",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, text=True, timeout=timeout)
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"worker failed ({completed.returncode}): {' '.join(command)}")
    output = Path(command[command.index("--output") + 1])
    return json.loads(output.read_text(encoding="utf-8")), wall


def assert_metric_replay(actual: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float, label: str) -> dict[str, float]:
    differences = {}
    for key in METRIC_KEYS:
        difference = float(actual[key]) - float(expected[key])
        differences[key] = difference
        if abs(difference) > tolerance:
            raise RuntimeError(f"{label}: {key} replay drift {difference} exceeds {tolerance}")
    return differences


def fixed32_expected(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    support = payload["routes"]["model_support"]["fully_cached"]["metrics"]["support"]
    production = payload["routes"]["production_reconstruction"]["fully_cached"]["metrics"]
    return {
        "support": support,
        "full": production["full_field"],
        "oracle": production["oracle_reconstruction"],
    }


def formal128_expected(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seed0 = next(row for row in payload["seeds"] if int(row["seed"]) == 0)
    primary = seed0["primary"]
    return {
        "support": {
            "point_global_true_rms_relative_rmse_pct": primary["support_point_global_pct"],
            "sample_first_cv_relative_rmse_pct": primary["support_sample_first_pct"],
            "raw_cv_weighted_rmse_K": primary["support_raw_cv_rmse_K"],
        },
        "full": {
            "point_global_true_rms_relative_rmse_pct": primary["full_point_global_pct"],
            "sample_first_cv_relative_rmse_pct": primary["full_sample_first_pct"],
            "raw_cv_weighted_rmse_K": primary["full_raw_cv_rmse_K"],
        },
    }


def formal_replay_differences(actual: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float, label: str) -> dict[str, float]:
    differences = {}
    for key, value in expected.items():
        difference = float(actual[key]) - float(value)
        differences[key] = difference
        if abs(difference) > tolerance:
            raise RuntimeError(f"{label}: {key} formal replay drift {difference} exceeds {tolerance}")
    return differences


def graph_feature_drift(current: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {
        "graph_hash_equal": current["graph_sha256"] == reference["graph_sha256"],
        "regional_nodes_delta": float(current["graph"]["regional_nodes"] - reference["graph"]["regional_nodes"]),
    }
    for family in ("p2r", "r2r", "r2p"):
        result[f"{family}_edge_count_delta"] = float(
            current["graph"][family]["edge_count"] - reference["graph"][family]["edge_count"]
        )
        result[f"{family}_in_degree_mean_delta"] = float(
            current["graph"][family]["in_degree"]["mean"]
            - reference["graph"][family]["in_degree"]["mean"]
        )
        result[f"{family}_edge_length_mean_delta"] = float(
            current["graph"][family]["edge_length_mean_normalized"]
            - reference["graph"][family]["edge_length_mean_normalized"]
        )
    g = np.asarray(current["features"]["global_context_z"], dtype=np.float64)
    g0 = np.asarray(reference["features"]["global_context_z"], dtype=np.float64)
    qk = np.asarray([
        *current["features"]["qk_mean"], *current["features"]["qk_std"]
    ], dtype=np.float64)
    qk0 = np.asarray([
        *reference["features"]["qk_mean"], *reference["features"]["qk_std"]
    ], dtype=np.float64)
    result.update({
        "global_context_z_l2_drift": float(np.linalg.norm(g - g0)),
        "global_context_z_max_abs_drift": float(np.max(np.abs(g - g0))),
        "qk_summary_l2_drift": float(np.linalg.norm(qk - qk0)),
        "log_s_phys_drift": float(current["features"]["log_s_phys"] - reference["features"]["log_s_phys"]),
        "predicted_log_scale_drift": float(current["features"]["predicted_log_scale"] - reference["features"]["predicted_log_scale"]),
    })
    return result


def mean_drift(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {f"{key}_mean": float(np.mean([float(row[key]) for row in rows])) for key in keys}


def prediction_drift(current_npz: Path, reference_npz: Path, full_cv: np.ndarray) -> dict[str, float]:
    with np.load(current_npz) as current, np.load(reference_npz) as reference:
        if list(current["sample_ids"]) != list(reference["sample_ids"]):
            raise RuntimeError("prediction drift sample identity mismatch")
        prediction = np.asarray(current["full_predictions"], dtype=np.float64)
        baseline = np.asarray(reference["full_predictions"], dtype=np.float64)
        truth = np.asarray(reference["full_truth"], dtype=np.float64)
    error = prediction - baseline
    weights = np.asarray(full_cv, dtype=np.float64)[None, :]
    sample_relative = np.sqrt(
        np.sum(weights * error * error, axis=1) / np.sum(weights * truth * truth, axis=1)
    ) * 100.0
    return {
        "prediction_drift_point_global_true_rms_pct": float(
            math.sqrt(float(np.sum(error * error)) / float(np.sum(truth * truth))) * 100.0
        ),
        "prediction_drift_sample_first_cv_pct": float(np.mean(sample_relative)),
        "prediction_drift_raw_cv_rmse_K": float(
            math.sqrt(float(np.sum(weights * error * error)) / float(len(error) * np.sum(full_cv)))
        ),
        "prediction_drift_max_abs_K": float(np.max(np.abs(error))),
    }


def plot_rows(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    created = []
    main = [row for row in rows if row["reference_label"] != "R0"]
    for y_key, name, ylabel in (
        ("support_point_global_pct", "v6_p1i_r0_resolution_error.png", "support PG true-RMS (%)"),
        ("worker_wall_seconds", "v6_p1i_r0_resolution_time.png", "diagnostic worker wall time (s)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for seed in audit.DISCRETIZATION_SEEDS:
            selected = sorted((row for row in main if int(row["discretization_seed"]) == seed), key=lambda row: int(row["resolution"]))
            ax.plot([row["resolution"] for row in selected], [row[y_key] for row in selected], marker="o", label=f"seed {seed}")
        if y_key.startswith("support"):
            r0 = next(row for row in rows if row["reference_label"] == "R0")
            ax.scatter([1024], [r0[y_key]], marker="*", s=120, color="black", label="R0 exact replay")
        ax.set_xscale("log", base=2); ax.set_xlabel("support nodes N"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        path = figure_dir / name; fig.savefig(path, dpi=180); plt.close(fig); created.append(str(path))
    return created


def report(payload: Mapping[str, Any]) -> str:
    jump = payload["r0_to_r1"]
    return "\n".join([
        "# V6 P1i cross-resolution R0 closeout",
        "",
        "This valid-only closeout formally names Stage A a measure-conservative full-graph re-discretization diagnostic. It is not checkpoint-IID and not a formal same-distribution invariance result.",
        "",
        "## R0 exact checkpoint replay",
        "",
        f"- fixed-32 support PG: {payload['r0']['fixed32']['support_metrics']['point_global_true_rms_relative_rmse_pct']:.6f}%",
        f"- full-128 support PG: {payload['r0']['formal128']['support_metrics']['point_global_true_rms_relative_rmse_pct']:.6f}%",
        f"- formal replay tolerance: {payload['r0']['formal_tolerance']} (passed)",
        "- frozen original 1024 coordinates, pointwise k/q, control-volume weights, Global Context, QK/scale inputs, graph config and graph seed were replayed without re-discretization.",
        "",
        "## R0 to R1 discontinuity",
        "",
        f"- support PG jump: {jump['support_point_global_delta_percentage_points']:+.6f} percentage points",
        f"- common full-field PG jump: {jump['full_point_global_delta_percentage_points']:+.6f} percentage points",
        f"- R1-vs-R0 prediction drift on the common full field: {jump['prediction_drift_point_global_true_rms_pct']:.6f}%",
        f"- graph hash equality fraction: {jump['graph_hash_equal_fraction']:.6f}",
        f"- pointwise coords/k/q/weights equality fractions: {jump['pointwise_hash_equal_fraction']}",
        "",
        "The R0-to-R1 jump demonstrates that the prior N=1024 re-discretization cell is already outside the checkpoint support/measure contract. Oracle improvement with N remains valid, while model degradation is associated primarily with support/measure, p2r graph-scale and context/scale-response drift; Nr growth remains secondary.",
        "",
        "## Governance",
        "",
        "- test accessed: false; sealed accessed: false; training/tuning: false.",
        "- Direct-N cells remain diagnostics only and cannot be used for model selection or production speedup claims.",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--edge-targets", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--historical-qualification", type=Path, required=True)
    parser.add_argument("--formal-closeout", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    args = parser.parse_args()
    if sha256(args.run_dir / "params_best_valid_point_global.pkl") != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA mismatch")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixed_expected = fixed32_expected(args.historical_qualification)
    formal_expected = formal128_expected(args.formal_closeout)
    tolerance = 0.005

    r0_32_json = args.work_dir / "R0_checkpoint_replay_valid32.json"
    r0_32_npz = args.work_dir / "R0_checkpoint_replay_valid32_predictions.npz"
    r0_32, r0_wall = run_worker(worker_command(
        args, support_mode="checkpoint_replay", resolution=1024, seed=0,
        sample_count=32, output=r0_32_json, prediction_npz=r0_32_npz,
    ), args.worker_timeout)
    fixed_diffs = {
        "support": assert_metric_replay(r0_32["support_metrics"], fixed_expected["support"], tolerance, "R0 fixed32 support"),
        "full": assert_metric_replay(r0_32["full_metrics"], fixed_expected["full"], tolerance, "R0 fixed32 full"),
        "oracle": assert_metric_replay(r0_32["oracle_reconstruction_metrics"], fixed_expected["oracle"], tolerance, "R0 fixed32 oracle"),
    }

    r0_128_json = args.work_dir / "R0_checkpoint_replay_valid128.json"
    r0_128, r0_128_wall = run_worker(worker_command(
        args, support_mode="checkpoint_replay", resolution=1024, seed=0,
        sample_count=128, output=r0_128_json, prediction_npz=None,
    ), args.worker_timeout)
    formal_diffs = {
        "support": formal_replay_differences(r0_128["support_metrics"], formal_expected["support"], tolerance, "R0 formal support"),
        "full": formal_replay_differences(r0_128["full_metrics"], formal_expected["full"], tolerance, "R0 formal full"),
    }

    prior = json.loads(args.prior_result.read_text(encoding="utf-8"))
    prior_lookup = {(int(row["discretization_seed"]), int(row["resolution"])): row for row in prior["main"]}
    reference_by_id = {sample["sample_id"]: sample for sample in r0_32["samples"]}
    data = base.FamilyData("p1i", args.dataset_root, args.manifest, args.full_fields, None)
    full_cv = np.asarray(data.full_shared()["cv"], dtype=np.float64)
    rows: list[dict[str, Any]] = [{
        "reference_label": "R0", "resolution": 1024, "discretization_seed": 0,
        "support_mode": "frozen_checkpoint_support", "regional_mode": "frozen_checkpoint_graph",
        "support_point_global_pct": r0_32["support_metrics"]["point_global_true_rms_relative_rmse_pct"],
        "support_sample_first_pct": r0_32["support_metrics"]["sample_first_cv_relative_rmse_pct"],
        "support_raw_cv_rmse_K": r0_32["support_metrics"]["raw_cv_weighted_rmse_K"],
        "full_point_global_pct": r0_32["full_metrics"]["point_global_true_rms_relative_rmse_pct"],
        "oracle_point_global_pct": r0_32["oracle_reconstruction_metrics"]["point_global_true_rms_relative_rmse_pct"],
        "worker_wall_seconds": r0_wall,
    }]
    detailed: list[dict[str, Any]] = []
    commands = []
    r1_payload = None
    r1_prediction_drift = None
    for seed in audit.DISCRETIZATION_SEEDS:
        for resolution in audit.MAIN_RESOLUTIONS:
            cell_json = args.work_dir / f"R2plus_N{resolution}_seed{seed}.json"
            cell_npz = args.work_dir / f"R2plus_N{resolution}_seed{seed}_predictions.npz"
            command = worker_command(
                args, support_mode="source_aware", resolution=resolution, seed=seed,
                sample_count=32, output=cell_json, prediction_npz=cell_npz,
            )
            commands.append(" ".join(command))
            current, wall = run_worker(command, args.worker_timeout)
            expected = prior_lookup[(seed, resolution)]
            replay_diffs = {
                "support": assert_metric_replay(current["support_metrics"], expected["support_metrics"], tolerance, f"N{resolution}/seed{seed}/support"),
                "full": assert_metric_replay(current["full_metrics"], expected["full_metrics"], tolerance, f"N{resolution}/seed{seed}/full"),
                "oracle": assert_metric_replay(current["oracle_reconstruction_metrics"], expected["oracle_reconstruction_metrics"], tolerance, f"N{resolution}/seed{seed}/oracle"),
            }
            current_by_id = {sample["sample_id"]: sample for sample in current["samples"]}
            sample_drifts = [graph_feature_drift(current_by_id[sample_id], reference_by_id[sample_id]) for sample_id in reference_by_id]
            drift = mean_drift(sample_drifts)
            drift["graph_hash_equal_fraction"] = float(np.mean([row["graph_hash_equal"] for row in sample_drifts]))
            pred_drift = prediction_drift(cell_npz, r0_32_npz, full_cv)
            row = {
                "reference_label": "R1" if resolution == 1024 else f"R2+_N{resolution}",
                "resolution": resolution, "discretization_seed": seed,
                "support_mode": "measure_conservative_full_graph_rediscretization",
                "regional_mode": "fixed_training_nr",
                "support_point_global_pct": current["support_metrics"]["point_global_true_rms_relative_rmse_pct"],
                "support_sample_first_pct": current["support_metrics"]["sample_first_cv_relative_rmse_pct"],
                "support_raw_cv_rmse_K": current["support_metrics"]["raw_cv_weighted_rmse_K"],
                "full_point_global_pct": current["full_metrics"]["point_global_true_rms_relative_rmse_pct"],
                "oracle_point_global_pct": current["oracle_reconstruction_metrics"]["point_global_true_rms_relative_rmse_pct"],
                "worker_wall_seconds": wall,
                **drift, **pred_drift,
            }
            rows.append(row)
            detailed.append({
                "resolution": resolution, "discretization_seed": seed,
                "metric_replay_differences": replay_diffs,
                "sample_drift": sample_drifts,
                "prediction_drift": pred_drift,
            })
            if seed == 0 and resolution == 1024:
                r1_payload = current
                r1_prediction_drift = pred_drift
            cell_npz.unlink()

    if r1_payload is None or r1_prediction_drift is None:
        raise RuntimeError("R1 reference cell missing")
    r1_by_id = {sample["sample_id"]: sample for sample in r1_payload["samples"]}
    pointwise_equal = {
        name: float(np.mean([
            r1_by_id[sample_id]["pointwise_hashes"][name] == reference_by_id[sample_id]["pointwise_hashes"][name]
            for sample_id in reference_by_id
        ]))
        for name in ("coords", "k_xyz", "q", "weights")
    }
    graph_equal = float(np.mean([
        r1_by_id[sample_id]["graph_sha256"] == reference_by_id[sample_id]["graph_sha256"]
        for sample_id in reference_by_id
    ]))
    r0_to_r1 = {
        "support_point_global_delta_percentage_points": float(
            r1_payload["support_metrics"]["point_global_true_rms_relative_rmse_pct"]
            - r0_32["support_metrics"]["point_global_true_rms_relative_rmse_pct"]
        ),
        "full_point_global_delta_percentage_points": float(
            r1_payload["full_metrics"]["point_global_true_rms_relative_rmse_pct"]
            - r0_32["full_metrics"]["point_global_true_rms_relative_rmse_pct"]
        ),
        "graph_hash_equal_fraction": graph_equal,
        "pointwise_hash_equal_fraction": pointwise_equal,
        **r1_prediction_drift,
    }
    figures = plot_rows(rows, args.figure_dir)
    payload = {
        "schema_version": "heat3d_v6_p1i_cross_resolution_r0_closeout_v1",
        "status": "passed",
        "diagnostic_name": "measure-conservative full-graph re-discretization diagnostic",
        "checkpoint_iid": False,
        "formal_same_distribution_invariance": False,
        "r0": {
            "fixed32": r0_32,
            "formal128": {
                "support_metrics": r0_128["support_metrics"],
                "full_metrics": r0_128["full_metrics"],
                "oracle_reconstruction_metrics": r0_128["oracle_reconstruction_metrics"],
                "worker_wall_seconds": r0_128_wall,
            },
            "fixed32_replay_differences": fixed_diffs,
            "formal128_replay_differences": formal_diffs,
            "formal_tolerance": tolerance,
        },
        "r0_to_r1": r0_to_r1,
        "resolution_rows": rows,
        "per_sample_drift": detailed,
        "figures": figures,
        "commands": commands,
        "inputs": {
            "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_epoch": args.checkpoint_epoch,
            "manifest_sha256": sha256(args.manifest),
            "full_fields_sha256": sha256(args.full_fields),
            "edge_targets_sha256": sha256(args.edge_targets),
            "prior_result_sha256": sha256(args.prior_result),
            "historical_qualification_sha256": sha256(args.historical_qualification),
            "formal_closeout_sha256": sha256(args.formal_closeout),
        },
        "contract": {
            "valid_only": True, "test_accessed": False, "sealed_accessed": False,
            "training_executed": False, "tuning_executed": False,
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.report_md.write_text(report(payload), encoding="utf-8")
    print(json.dumps({"status": "passed", "rows": len(rows), "r0_to_r1": r0_to_r1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
