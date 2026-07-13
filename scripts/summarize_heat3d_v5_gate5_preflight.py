#!/usr/bin/env python3
"""Summarize Gate-5 execution smoke and e10 calibration artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


LOSS_FIELDS = (
    "shape_cv_loss",
    "log_scale_loss",
    "relative_field_loss",
    "raw_absolute_field_loss",
)
NATIVE_METRICS = (
    "joint_relative_rmse",
    "oracle_scale_relative_rmse",
    "oracle_shape_relative_rmse",
    "shape_cv_rmse",
    "scale_log_abs_error",
    "joint_amplitude_ratio",
)
GRADIENT_GROUPS = ("backbone", "shape_decoder", "scale_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for label in ("n0_smoke", "n1_smoke", "n0_e10", "n1_e10"):
        parser.add_argument(f"--{label.replace('_', '-')}-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _load_run(path: Path) -> dict[str, Any]:
    summary_path = path / "loss_summary.json"
    profile_path = path / "profile_timing.json"
    if not summary_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError(f"incomplete run artifacts under {path}")
    return {
        "dir": str(path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "profile": json.loads(profile_path.read_text(encoding="utf-8")),
    }


def _history_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in summary["epoch_history"]:
        row = {
            "epoch": int(record["epoch"]),
            "train": {name: record.get(f"train_{name}") for name in LOSS_FIELDS},
            "valid": {name: record.get(f"valid_{name}") for name in LOSS_FIELDS},
            "valid_base_mse": record.get("valid_base_mse"),
            "valid_native": {
                name: record.get(f"valid_native_{name}") for name in NATIVE_METRICS
            },
            "gradient_norm": {
                name: record.get(f"epoch_mean_{name}_grad_norm")
                for name in GRADIENT_GROUPS
            },
            "epoch_time_s": record.get("epoch_total_time_s"),
        }
        values = [
            *row["train"].values(),
            *row["valid"].values(),
            row["valid_base_mse"],
            *row["valid_native"].values(),
            *row["gradient_norm"].values(),
            row["epoch_time_s"],
        ]
        if not all(_finite(value) for value in values):
            raise AssertionError(f"non-finite or missing epoch metrics: {row}")
        rows.append(row)
    return rows


def _timing(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary.get("train_batch_records") or []
    times = [float(record["total_batch_time"]) for record in records]
    epoch_times = [float(record["epoch_total_time_s"]) for record in summary["epoch_history"]]
    return {
        "first_batch_compile_time_s": times[0] if times else None,
        "steady_batch_median_time_s": median(times[1:]) if len(times) > 1 else None,
        "epoch_time_s": epoch_times,
        "steady_epoch_median_time_s": median(epoch_times[1:]) if len(epoch_times) > 1 else median(epoch_times),
        "peak_memory": summary.get("memory_audit_summary"),
    }


def _execution_checks(summary: dict[str, Any], *, expected_mode: str) -> dict[str, Any]:
    split_counts = summary["split_counts"]
    runtime = summary["native_runtime_architecture_audit"]
    reload_audit = summary["checkpoint_prediction_reload_audit"]
    context = summary["global_context"]
    standardizer = context["standardizer"]
    expected_width = 96 if expected_mode == "physics_plus_pooled_latent" else 0
    checks = {
        "status_ok": bool(summary["status_ok"]),
        "grad_finite": bool(summary["grad_finite"]),
        "train_count_672": split_counts.get("train") == 672,
        "valid_iid_count_128": split_counts.get("valid_iid") == 128,
        "batch_size_28": summary.get("batch_size") == 28,
        "node_count_1024": all(
            record.get("batch_shape_signature", {}).get("input_x_inp_shape", [None, None, None])[-2] == 1024
            for record in (summary.get("train_batch_records") or [])
        ),
        "scale_head_mode": runtime.get("scale_head_mode") == expected_mode,
        "pooled_latent_width": runtime.get("pooled_latent_width") == expected_width,
        "checkpoint_prediction_reload": reload_audit.get("status") == "passed"
        and all(entry.get("passed") for entry in reload_audit.get("entries", [])),
        "context_fit_train_only": standardizer.get("fit_population") == "train_only"
        and standardizer.get("fit_sample_count") == 672,
        "context_no_target_leakage": context.get("target_or_label_derived_inputs") is False
        and standardizer.get("target_or_label_derived_inputs") is False,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise AssertionError(f"execution checks failed: {checks}")
    return checks


def _run_payload(run: dict[str, Any], *, expected_mode: str) -> dict[str, Any]:
    summary = run["summary"]
    history = _history_rows(summary)
    return {
        "run_dir": run["dir"],
        "epochs": len(history),
        "execution_checks": _execution_checks(summary, expected_mode=expected_mode),
        "history": history,
        "timing": _timing(summary),
        "best_epoch": summary.get("best_epoch"),
        "best_valid_base_mse": summary.get("best_valid_base_mse"),
        "final_valid_base_mse": history[-1]["valid_base_mse"],
    }


def _loss_audit(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_run = {}
    for label in ("n0_e10", "n1_e10"):
        history = runs[label]["history"]
        medians = {
            name: median(float(row["valid"][name]) for row in history)
            for name in LOSS_FIELDS
        }
        positive = [value for value in medians.values() if value > 0.0]
        per_run[label] = {
            "median_valid_unweighted_components": medians,
            "largest_to_smallest_positive_ratio": max(positive) / min(positive),
            "median_gradient_norms": {
                group: median(float(row["gradient_norm"][group]) for row in history)
                for group in GRADIENT_GROUPS
            },
        }
    combined = {
        name: median(per_run[label]["median_valid_unweighted_components"][name] for label in per_run)
        for name in LOSS_FIELDS
    }
    return {"initial_weights": [1.0, 1.0, 1.0, 1.0], "per_run": per_run, "combined_component_medians": combined}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Gate 5 N0/N1 e600 preflight",
        "",
        "Smoke 与 e10 均为真实 P5 execution/calibration，不是正式性能结果；未启动 e600。",
        "",
        "| run | epochs | valid_base_mse(final) | joint rel | oracle-scale rel | oracle-shape rel | peak GPU MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("n0_smoke", "n1_smoke", "n0_e10", "n1_e10"):
        run = payload["runs"][label]
        last = run["history"][-1]
        native = last["valid_native"]
        peak = (run["timing"].get("peak_memory") or {}).get("peak_device_memory_all_mb")
        lines.append(
            f"| {label} | {run['epochs']} | {last['valid_base_mse']:.6g} | "
            f"{native['joint_relative_rmse']:.6g} | {native['oracle_scale_relative_rmse']:.6g} | "
            f"{native['oracle_shape_relative_rmse']:.6g} | {peak if peak is not None else 'n/a'} |"
        )
    lines.extend(["", "## Execution checks", ""])
    for label in ("n0_smoke", "n1_smoke"):
        checks = payload["runs"][label]["execution_checks"]
        lines.append(f"- {label}: passed={str(checks['passed']).lower()}, checkpoint/prediction reload passed, context fit train-only.")
    lines.extend(["", "## Loss freeze", "", "最终冻结权重与依据见同目录 loss-freeze JSON；N0/N1 必须共用同一组权重。", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    raw_runs = {
        "n0_smoke": _load_run(args.n0_smoke_run_dir),
        "n1_smoke": _load_run(args.n1_smoke_run_dir),
        "n0_e10": _load_run(args.n0_e10_run_dir),
        "n1_e10": _load_run(args.n1_e10_run_dir),
    }
    runs = {
        label: _run_payload(
            run,
            expected_mode="physics_plus_pooled_latent" if label.startswith("n1") else "physics_only",
        )
        for label, run in raw_runs.items()
    }
    if runs["n0_smoke"]["epochs"] != 1 or runs["n1_smoke"]["epochs"] != 1:
        raise AssertionError("smoke runs must be e1")
    if runs["n0_e10"]["epochs"] != 10 or runs["n1_e10"]["epochs"] != 10:
        raise AssertionError("calibration runs must be e10")
    payload = {
        "schema_version": "heat3d_v5_gate5_e600_preflight_v1",
        "status": "passed",
        "formal_performance_result": False,
        "e600_started": False,
        "runs": runs,
        "loss_audit": _loss_audit(runs),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "passed", "output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
