#!/usr/bin/env python3
"""Collect finite/reload/memory evidence from Gate 6F full-model e1 smokes.

The collector is intentionally read-only with respect to model artifacts.  It
only reads the eight registered e1 run summaries and memory audit logs, and
writes one ignored JSON report.  It never loads a checkpoint or materializes a
test, hard, or sealed-IID role.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6f_scale_probe_registry.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _reload_passed(payload: dict[str, Any]) -> bool:
    audit = payload.get("checkpoint_prediction_reload_audit") or {}
    return bool(
        audit.get("enabled")
        and audit.get("status") == "passed"
        and audit.get("entries")
        and all(entry.get("passed") for entry in audit["entries"])
    )


def _native_passed(payload: dict[str, Any]) -> bool:
    audit = payload.get("native_runtime_architecture_audit") or {}
    return bool(audit.get("enabled") and audit.get("passed"))


def _memory(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    audit = payload.get("memory_audit_summary") or {}
    rss = audit.get("peak_rss_mb")
    device = audit.get("peak_device_memory_all_mb")
    return (
        float(rss) if rss is not None else None,
        float(device) if device is not None else None,
    )


def _row_result(row: dict[str, str]) -> dict[str, Any]:
    output_dir = ROOT / row["output_dir"]
    summary_path = output_dir / "loss_summary.json"
    run_config_path = output_dir / "run_config.json"
    if not summary_path.is_file() or not run_config_path.is_file():
        raise FileNotFoundError(f"{row['config_id']}: missing e1 output summary")
    summary = _read_json(summary_path)
    run_config = _read_json(run_config_path)
    if int(run_config.get("epochs", -1)) != 1:
        raise ValueError(f"{row['config_id']}: expected e1 run")
    if summary.get("prediction_split") != "valid_iid":
        raise ValueError(f"{row['config_id']}: prediction split escaped valid_iid")
    if int(summary.get("test_iid_group_count", 0)) != 0:
        raise ValueError(f"{row['config_id']}: test_iid was materialized")
    if summary.get("all_groups_status") != "skipped":
        raise ValueError(f"{row['config_id']}: report-only roles were materialized")
    context = summary.get("global_context") or {}
    standardizer = context.get("standardizer") or {}
    if context.get("enabled") and standardizer.get("fit_population") != "train_only":
        raise ValueError(f"{row['config_id']}: global context was not train-only fit")
    native = summary.get("native_runtime_architecture_audit") or {}
    expected_width = int(native.get("expected_pooled_latent_width", -1))
    actual_width = int(native.get("pooled_latent_width", -2))
    if expected_width != actual_width:
        raise ValueError(f"{row['config_id']}: pooled latent width mismatch")
    peak_rss, peak_device = _memory(summary)
    passed = bool(
        summary.get("status_ok")
        and summary.get("grad_finite")
        and _reload_passed(summary)
        and _native_passed(summary)
    )
    if not passed:
        raise ValueError(f"{row['config_id']}: e1 smoke audit failed")
    return {
        "config_id": row["config_id"],
        "candidate": row["candidate"],
        "output_dir": row["output_dir"],
        "memory_audit_jsonl": row["memory_audit_jsonl"],
        "status": "passed",
        "status_ok": bool(summary["status_ok"]),
        "grad_finite": bool(summary["grad_finite"]),
        "checkpoint_reload_passed": _reload_passed(summary),
        "native_runtime_passed": _native_passed(summary),
        "scale_pooling": native.get("scale_pooling"),
        "scale_head_depth": native.get("scale_head_depth"),
        "pooled_latent_stop_gradient": native.get("pooled_latent_stop_gradient"),
        "pooled_latent_width": actual_width,
        "peak_rss_mb": peak_rss,
        "peak_device_memory_mb": peak_device,
        "valid_base_mse": summary.get("valid_base_mse"),
        "valid_point_global_relative_rmse_pct": summary.get("valid_point_global_relative_rmse_pct"),
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
    }


def main() -> int:
    args = _parse_args()
    registry = args.registry.resolve()
    rows = list(csv.DictReader(registry.open(encoding="utf-8", newline="")))
    if len(rows) != 8:
        raise ValueError(f"expected eight Gate 6F smoke rows, got {len(rows)}")
    results = [_row_result(row) for row in rows]
    payload = {
        "schema_version": "heat3d_v5_gate6f_e1_smoke_summary_v1",
        "registry": str(registry),
        "config_count": len(results),
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "long_training_started": False,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
