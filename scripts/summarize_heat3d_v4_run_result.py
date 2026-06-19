#!/usr/bin/env python3
"""Summarize one Heat3D V4 run directory and optionally update CSV results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v4_registry import (  # noqa: E402
    CSV_FIELDNAMES,
    DEFAULT_REGISTRY,
    RESULT_FIELDNAMES,
    load_registry,
    registry_rows,
)


def main() -> int:
    args = _parse_args()
    row = _row_for_config(args.config_id, args.registry)
    run_dir = _repo_path(args.run_dir or row["output_dir"])
    summary = _result_fields(row, run_dir, args.source_label)
    if args.update_csv:
        _update_csv(_repo_path(args.csv), args.config_id, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--csv", default="configs/heat3d_v4/run_registry.csv")
    parser.add_argument("--run-dir")
    parser.add_argument("--source-label", default="local")
    parser.add_argument("--update-csv", action="store_true")
    return parser.parse_args()


def _row_for_config(config_id: str, registry_path: str) -> dict[str, str]:
    rows = registry_rows(load_registry(_repo_path(registry_path)))
    for row in rows:
        if row["config_id"] == config_id:
            return row
    raise SystemExit(f"missing config_id: {config_id}")


def _result_fields(
    row: dict[str, str], run_dir: Path, source_label: str
) -> dict[str, str]:
    loss_path = run_dir / "loss_summary.json"
    config_path = run_dir / "run_config.json"
    loss = _read_json(loss_path)
    result: dict[str, str] = {field: "" for field in RESULT_FIELDNAMES}
    result["result_source"] = source_label
    result["result_updated_at"] = datetime.now(timezone.utc).isoformat()
    result["result_run_dir"] = row["output_dir"]
    result["result_log_path"] = row["log_path"]
    result["result_loss_summary"] = str(loss_path) if loss_path.exists() else ""
    result["result_params_best"] = _existing_path(loss.get("best_checkpoint_path"))
    result["result_params_final"] = _existing_path(loss.get("final_checkpoint_path"))
    result["result_best_epoch"] = _fmt(loss.get("best_epoch"))

    best_mse = _number(loss.get("best_valid_base_mse"))
    final_mse = _number(loss.get("final_valid_base_mse"))
    if final_mse is None:
        final_mse = _number(_dig(loss, "final_valid_loss_components", "base_mse"))
    result["result_best_valid_base_mse"] = _fmt(best_mse)
    result["result_best_mse"] = _fmt(best_mse)
    result["result_best_rmse"] = _fmt(_sqrt(best_mse))
    result["result_final_valid_base_mse"] = _fmt(final_mse)
    result["result_final_mse"] = _fmt(final_mse)
    result["result_final_rmse"] = _fmt(_sqrt(final_mse))

    result["result_best_valid_iid"] = _fmt(best_mse)
    result["result_final_valid_iid"] = _fmt(final_mse)
    result["result_final_stress"] = _fmt(
        _number(_dig(loss, "valid_stress_metrics", "base_mse"))
    )

    final_probe = loss.get("final_probe_eval_result")
    diagnostics = loss.get("post_training_diagnostics_result")
    result["result_final_probe_status"] = _status_from_payload(final_probe)
    result["result_post_training_diagnostics_status"] = _status_from_payload(diagnostics)
    if not result["result_final_probe_status"]:
        result["result_final_probe_status"] = (
            "disabled" if loss.get("final_probe_eval_after_training") is False else ""
        )
    if not result["result_post_training_diagnostics_status"]:
        result["result_post_training_diagnostics_status"] = (
            "disabled" if loss.get("post_training_diagnostics") is False else ""
        )

    result["result_status"] = "completed" if loss_path.exists() else "missing"
    if config_path.exists():
        result["result_notes"] = "run_config.json present"
    elif not loss_path.exists():
        result["result_notes"] = "loss_summary.json missing"
    return result


def _update_csv(path: Path, config_id: str, result_fields: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if tuple(fieldnames) != CSV_FIELDNAMES:
        raise SystemExit(f"{path}: CSV header does not match V4 registry contract")
    found = False
    for row in rows:
        if row.get("config_id") != config_id:
            continue
        found = True
        for field in RESULT_FIELDNAMES:
            value = result_fields.get(field, "")
            if value != "":
                row[field] = value
        break
    if not found:
        raise SystemExit(f"{path}: config_id not found: {config_id}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return {}
    return payload


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _status_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str):
            return status
        if payload.get("skipped") is True:
            return "skipped"
    return ""


def _existing_path(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value


def _sqrt(value: float | None) -> float | None:
    if value is None or value < 0:
        return None
    return math.sqrt(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _fmt(value: Any) -> str:
    number = _number(value)
    if number is not None:
        return f"{number:.12g}"
    if value is None:
        return ""
    return str(value)


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
