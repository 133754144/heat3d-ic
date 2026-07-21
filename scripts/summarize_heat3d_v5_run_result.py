#!/usr/bin/env python3
"""Collect V5 frozen metrics and V4-compatible diagnostics into the V5 CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heat3d_v5_result_contract import (  # noqa: E402
    V5_CHECKPOINTS,
    V5_FROZEN_METRICS,
    V5_REGISTRY_RESULT_FIELDS,
    V5_REPORT_ROLES,
)


DEFAULT_REGISTRY = REPO_ROOT / "configs/heat3d_v5/v5_scratch_bypass_film_registry.csv"
DEFAULT_CSV = DEFAULT_REGISTRY


def main() -> int:
    args = _parse_args()
    row = _find_row(args.config_id, _repo_path(args.registry))
    run_dir = _repo_path(args.run_dir or row["output_dir"])
    if args.result_json:
        payload = _read_json(Path(args.result_json))
    else:
        payload = _collect(row, run_dir, args.source_label)
    result = _result_fields(row, run_dir, payload, args.source_label)
    if args.strict and result["result_v5_required_metrics_complete"] != "true":
        missing = result["result_v5_missing_metrics"] or "unknown"
        raise SystemExit(
            f"{args.config_id}: V5 frozen metrics incomplete; missing {missing}"
        )
    if args.update_csv:
        _update_csv(_repo_path(args.csv), args.config_id, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--run-dir")
    parser.add_argument("--source-label", default="local")
    parser.add_argument("--result-json")
    parser.add_argument("--update-csv", action="store_true")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all primary/legacy checkpoint frozen metrics (default: true).",
    )
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _find_row(config_id: str, registry: Path) -> dict[str, str]:
    with registry.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row.get("config_id") == config_id:
            return row
    raise SystemExit(f"{registry}: config_id not found: {config_id}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect(row: dict[str, str], run_dir: Path, source: str) -> dict[str, Any]:
    loss = _read_json(run_dir / "loss_summary.json")
    run_config = _read_json(run_dir / "run_config.json")
    metrics = _read_json(run_dir / "v5_metrics.json")
    if not metrics:
        metrics = _read_json(run_dir / "clean_metrics.json")
    if not metrics:
        best = _read_json(run_dir / "clean_metrics_best.json")
        final = _read_json(run_dir / "clean_metrics_final.json")
        metrics = {"reports": {}}
        if best:
            metrics["reports"]["best"] = best.get("reports", best)
        if final:
            metrics["reports"]["final"] = final.get("reports", final)
    return {
        "loss_summary": loss,
        "run_config": run_config,
        "metrics": metrics,
        "source": source,
        "run_dir": row.get("output_dir", str(run_dir)),
        "log_path": row.get("log_path", ""),
    }


def _result_fields(
    row: dict[str, str], run_dir: Path, payload: dict[str, Any], source: str
) -> dict[str, str]:
    loss = payload.get("loss_summary") if isinstance(payload.get("loss_summary"), dict) else {}
    run_config = payload.get("run_config") if isinstance(payload.get("run_config"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    valid_only_four = _is_valid_only_four_checkpoint_payload(payload, metrics)
    result = {field: "" for field in V5_REGISTRY_RESULT_FIELDS}
    result["result_v5_source"] = str(payload.get("source") or source)
    result["result_v5_updated_at"] = datetime.now(timezone.utc).isoformat()
    result["result_v5_commit"] = _fmt(
        loss.get("code_version_or_git_commit")
        or run_config.get("code_version_or_git_commit")
        or payload.get("training_commit")
    )
    result["result_v5_run_dir"] = str(payload.get("run_dir") or row.get("output_dir") or run_dir)
    result["result_v5_log_path"] = str(payload.get("log_path") or row.get("log_path") or "")
    result["result_v5_loss_summary"] = _relpath(run_dir / "loss_summary.json")
    metrics_for_csv = payload if valid_only_four else metrics
    result["result_v5_metrics_json"] = (
        json.dumps(metrics_for_csv, sort_keys=True, separators=(",", ":"))
        if metrics_for_csv
        else ""
    )

    _fill_v4_values(result, loss, run_config)
    reports = (
        _normalize_valid_only_four_checkpoint_reports(metrics)
        if valid_only_four
        else _normalize_reports(metrics)
    )
    _fill_v4_metric_values(result, reports)
    missing = (
        _missing_valid_only_four_checkpoint_paths(metrics)
        if valid_only_four
        else _missing_metric_paths(reports)
    )
    result["result_v5_required_metrics_complete"] = "true" if not missing else "false"
    result["result_v5_missing_metrics"] = "|".join(missing)
    checkpoint_metadata = (
        payload.get("checkpoint_metadata")
        if isinstance(payload.get("checkpoint_metadata"), dict)
        else {}
    )
    result["result_v5_primary_checkpoint"] = _fmt(
        "point_global_best"
        if valid_only_four
        else _get_any(metrics, "primary_checkpoint", "primary_relative_checkpoint")
    )
    result["result_v5_primary_epoch"] = _fmt(
        checkpoint_metadata.get("point_global_best", {}).get("epoch")
        if valid_only_four
        else (
            _get_any(metrics, "primary_epoch", "primary_relative_epoch")
            or loss.get("primary_relative_epoch")
        )
    )
    result["result_v5_legacy_checkpoint"] = _fmt(
        "legacy_best"
        if valid_only_four
        else _get_any(metrics, "legacy_checkpoint", "legacy_metric_checkpoint")
    )
    result["result_v5_legacy_epoch"] = _fmt(
        checkpoint_metadata.get("legacy_best", {}).get("epoch")
        if valid_only_four
        else (
            _get_any(metrics, "legacy_epoch", "legacy_metric_epoch")
            or loss.get("legacy_metric_epoch")
            or loss.get("best_epoch")
        )
    )
    primary_valid = reports.get("primary_relative", {}).get("valid_iid", {})
    primary_test = reports.get("primary_relative", {}).get("test_iid", {})
    legacy_valid = reports.get("legacy_metric", {}).get("valid_iid", {})
    legacy_test = reports.get("legacy_metric", {}).get("test_iid", {})
    for field, source_value in (
        ("result_v5_primary_valid_point_global_relative_rmse_pct", primary_valid.get("point_global_relative_rmse_pct")),
        ("result_v5_primary_valid_sample_first_cv_relative_rmse_pct", primary_valid.get("sample_first_cv_relative_rmse_pct")),
        ("result_v5_primary_valid_raw_cv_weighted_rmse_K", primary_valid.get("raw_cv_weighted_rmse_K")),
        ("result_v5_primary_test_point_global_relative_rmse_pct", primary_test.get("point_global_relative_rmse_pct")),
        ("result_v5_primary_test_sample_first_cv_relative_rmse_pct", primary_test.get("sample_first_cv_relative_rmse_pct")),
        ("result_v5_primary_test_raw_cv_weighted_rmse_K", primary_test.get("raw_cv_weighted_rmse_K")),
        ("result_v5_legacy_valid_base_mse", legacy_valid.get("legacy_normalized_valid_base_mse")),
        ("result_v5_legacy_test_point_global_relative_rmse_pct", legacy_test.get("point_global_relative_rmse_pct")),
    ):
        result[field] = _fmt(source_value)
    result["result_v5_threshold_pass"] = (
        _valid_only_threshold_status(primary_valid)
        if valid_only_four
        else _threshold_status(primary_valid, primary_test)
    )
    result["result_v5_final_probe_status"] = _status(loss.get("final_probe_eval_result"), run_config.get("final_probe_eval_result"))
    result["result_v5_post_training_diagnostics_status"] = _status(loss.get("post_training_diagnostics_result"), run_config.get("post_training_diagnostics_result"))
    if valid_only_four:
        result["result_v5_final_probe_status"] = "not_applicable_valid_only"
        result["result_v5_post_training_diagnostics_status"] = "not_applicable_valid_only"
    if not result["result_v5_final_probe_status"]:
        result["result_v5_final_probe_status"] = "disabled" if run_config.get("final_probe_eval_after_training") is False else "missing_or_failed"
    if not result["result_v5_post_training_diagnostics_status"]:
        result["result_v5_post_training_diagnostics_status"] = "disabled" if run_config.get("post_training_diagnostics") is False else "missing_or_skipped"
    result["result_v5_status"] = (
        "completed_valid_only" if valid_only_four and not missing
        else _run_status(loss, run_dir / "loss_summary.json", bool(missing))
    )
    notes = []
    if missing:
        notes.append("required V5 frozen metric payload incomplete")
    if valid_only_four and not missing:
        notes.append(
            "four-checkpoint valid_iid metrics complete; test/hard/sealed not accessed"
        )
    if result["result_v5_final_probe_status"] in {"missing_or_failed", "failed"}:
        notes.append("Global FiLM final-probe payload missing or failed")
    result["result_v5_notes"] = "; ".join(notes)
    return result


def _is_valid_only_four_checkpoint_payload(
    payload: dict[str, Any], metrics: dict[str, Any]
) -> bool:
    return (
        str(payload.get("schema_version") or "").startswith(
            (
                "heat3d_v5_v32_valid_only_closeout",
                "heat3d_v5_valid_only_four_checkpoint",
                "heat3d_v5_gate6q_cpu_replay",
            )
        )
        and set(metrics) == {
            "point_global_best",
            "sample_first_best",
            "legacy_best",
            "final",
        }
    )


def _valid_only_summary(
    metrics: dict[str, Any], checkpoint: str
) -> dict[str, Any]:
    payload = metrics.get(checkpoint)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _normalize_valid_only_four_checkpoint_reports(
    metrics: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "primary_relative": {
            "valid_iid": _valid_only_summary(metrics, "point_global_best")
        },
        "legacy_metric": {
            "valid_iid": _valid_only_summary(metrics, "legacy_best")
        },
        "best": {"valid_iid": _valid_only_summary(metrics, "legacy_best")},
        "final": {"valid_iid": _valid_only_summary(metrics, "final")},
    }


def _missing_valid_only_four_checkpoint_paths(
    metrics: dict[str, Any],
) -> list[str]:
    missing = []
    for checkpoint in (
        "point_global_best",
        "sample_first_best",
        "legacy_best",
        "final",
    ):
        row = _valid_only_summary(metrics, checkpoint)
        for metric in V5_FROZEN_METRICS:
            if not _finite(row.get(metric)):
                missing.append(f"{checkpoint}.valid_iid.{metric}")
    return missing


def _fill_v4_values(result: dict[str, str], loss: dict[str, Any], run_config: dict[str, Any]) -> None:
    pairs = {
        "result_v4_best_valid_base_mse": loss.get("best_valid_base_mse"),
        "result_v4_final_valid_base_mse": loss.get("final_valid_base_mse"),
        "result_v4_best_valid_iid_loss": loss.get("best_valid_iid_loss"),
        "result_v4_final_valid_iid_loss": loss.get("final_valid_iid_loss"),
        "result_v4_best_valid_raw_deltaT_rmse_K": loss.get("best_valid_iid_raw_deltaT_rmse_K"),
        "result_v4_final_valid_raw_deltaT_rmse_K": loss.get("final_valid_iid_raw_deltaT_rmse_K"),
        "result_v4_best_valid_recovered_T_rmse_K": loss.get("best_valid_iid_recovered_T_rmse_K"),
        "result_v4_final_valid_recovered_T_rmse_K": loss.get("final_valid_iid_recovered_T_rmse_K"),
        "result_v4_best_valid_relative_rmse_pct_v4": loss.get("best_valid_iid_relative_rmse_pct_v4"),
        "result_v4_final_valid_relative_rmse_pct_v4": loss.get("final_valid_iid_relative_rmse_pct_v4"),
        "result_v4_final_probe_status": _status(loss.get("final_probe_eval_result"), run_config.get("final_probe_eval_result")),
        "result_v4_post_training_diagnostics_status": _status(loss.get("post_training_diagnostics_result"), run_config.get("post_training_diagnostics_result")),
    }
    components = loss.get("final_valid_loss_components")
    if isinstance(components, dict):
        pairs["result_v4_strong_q_rmse"] = _sqrt(components.get("strong_q_mse"))
        pairs["result_v4_hotspot_mae"] = components.get("hotspot_raw_mae")
    for field, value in pairs.items():
        result[field] = _fmt(value)


def _fill_v4_metric_values(
    result: dict[str, str], reports: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Retain V4-style field-fidelity scalars from the V5 metric payload."""

    valid = reports.get("primary_relative", {}).get("valid_iid", {})
    if not valid:
        valid = reports.get("legacy_metric", {}).get("valid_iid", {})
    values = {
        "result_v4_corr_iid": valid.get("spatial_correlation"),
        "result_v4_amp": valid.get("amplitude_ratio"),
        "result_v4_valid_iid_topk": valid.get("top5_cv_weighted_rmse_K"),
        "result_v4_strong_q_rmse": valid.get("strong_q_cv_weighted_rmse_K"),
    }
    for field, value in values.items():
        if not result.get(field):
            result[field] = _fmt(value)


def _normalize_reports(metrics: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    reports = metrics.get("reports", metrics.get("checkpoints", metrics))
    if not isinstance(reports, dict):
        return {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for checkpoint, payload in reports.items():
        if checkpoint not in V5_CHECKPOINTS or not isinstance(payload, dict):
            continue
        # Accept both reports[checkpoint][role] and reports[role][checkpoint].
        if any(role in payload for role in V5_REPORT_ROLES):
            normalized[checkpoint] = {
                role: value for role, value in payload.items()
                if role in V5_REPORT_ROLES and isinstance(value, dict)
            }
    for role in V5_REPORT_ROLES:
        value = reports.get(role)
        if not isinstance(value, dict):
            continue
        for checkpoint in V5_CHECKPOINTS:
            if checkpoint in value and isinstance(value[checkpoint], dict):
                normalized.setdefault(checkpoint, {})[role] = value[checkpoint]
    return normalized


def _missing_metric_paths(reports: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    missing = []
    for checkpoint in ("primary_relative", "legacy_metric"):
        for role in V5_REPORT_ROLES:
            row = reports.get(checkpoint, {}).get(role, {})
            for metric in V5_FROZEN_METRICS:
                value = row.get(metric) if isinstance(row, dict) else None
                if not _finite(value):
                    missing.append(f"{checkpoint}.{role}.{metric}")
    return missing


def _threshold_status(valid: dict[str, Any], test: dict[str, Any]) -> str:
    values = (valid.get("point_global_relative_rmse_pct"), test.get("point_global_relative_rmse_pct"))
    if not all(_finite(value) for value in values):
        return "unknown"
    return "pass" if all(float(value) < 20.0 for value in values) else "fail"


def _valid_only_threshold_status(valid: dict[str, Any]) -> str:
    value = valid.get("point_global_relative_rmse_pct")
    if not _finite(value):
        return "unknown"
    return "valid_only_pass" if float(value) < 20.0 else "valid_only_fail"


def _run_status(loss: dict[str, Any], loss_path: Path, incomplete: bool) -> str:
    if not loss and not loss_path.exists():
        return "missing"
    if loss.get("status_ok") is False or loss.get("grad_finite") is False:
        return "partial"
    return "completed_with_missing_v5_metrics" if incomplete else "completed"


def _status(*payloads: Any) -> str:
    for payload in payloads:
        if isinstance(payload, dict):
            if isinstance(payload.get("status"), str):
                return str(payload["status"])
            if payload.get("enabled") is False:
                return str(payload.get("reason") or "disabled")
            if payload.get("returncode") == 0:
                return "completed"
            if payload.get("returncode") not in (None, 0):
                return "failed"
    return ""


def _get_any(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sqrt(value: Any) -> float | None:
    return math.sqrt(float(value)) if _finite(value) and float(value) >= 0 else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if _finite(value):
        return f"{float(value):.12g}"
    return str(value)


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _update_csv(path: Path, config_id: str, result: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing_fields = [field for field in V5_REGISTRY_RESULT_FIELDS if field not in fieldnames]
    if missing_fields:
        raise SystemExit(f"{path}: missing V5 result columns: {', '.join(missing_fields)}")
    found = False
    for row in rows:
        if row.get("config_id") == config_id:
            found = True
            for field in V5_REGISTRY_RESULT_FIELDS:
                # Result collection is an authoritative snapshot. Overwrite
                # empty values as well so a later complete replay clears stale
                # missing-metric paths and notes from an earlier partial run.
                row[field] = result.get(field, "")
            break
    if not found:
        raise SystemExit(f"{path}: config_id not found: {config_id}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
