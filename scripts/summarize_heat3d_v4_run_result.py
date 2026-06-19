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
    result["result_commit"] = _fmt(loss.get("code_version_or_git_commit"))
    result["result_run_dir"] = row["output_dir"]
    result["result_log_path"] = row["log_path"]
    result["result_loss_summary"] = _display_path(loss_path) if loss_path.exists() else ""
    result["result_params_best"] = _display_path(loss.get("best_checkpoint_path"))
    result["result_params_final"] = _display_path(loss.get("final_checkpoint_path"))
    result["result_best_epoch"] = _fmt(loss.get("best_epoch"))

    best_mse = _first_number(loss, "best_valid_base_mse", "best_valid_loss")
    final_mse = _first_number(loss, "final_valid_base_mse", "final_valid_loss")
    if final_mse is None:
        final_mse = _number(_dig(loss, "final_valid_loss_components", "base_mse"))
    result["result_best_valid_base_mse"] = _fmt(best_mse)
    result["result_best_mse"] = _fmt(best_mse)
    result["result_best_rmse"] = _fmt(_sqrt(best_mse))
    result["result_best_mae"] = _fmt(_first_number(loss, "best_valid_mae"))
    result["result_final_valid_base_mse"] = _fmt(final_mse)
    result["result_final_mse"] = _fmt(final_mse)
    result["result_final_rmse"] = _fmt(_sqrt(final_mse))
    result["result_final_mae"] = _fmt(_first_number(loss, "final_valid_mae"))

    best_raw_mse = _first_number(
        loss, "best_valid_raw_deltaT_mse", "best_valid_iid_raw_deltaT_mse"
    )
    final_raw_mse = _first_number(
        loss, "final_valid_raw_deltaT_mse", "final_valid_iid_raw_deltaT_mse"
    )
    result["result_best_raw_deltaT_mse"] = _fmt(best_raw_mse)
    result["result_best_raw_deltaT_rmse"] = _fmt(_sqrt(best_raw_mse))
    result["result_best_raw_deltaT_mae"] = _fmt(_first_number(loss, "best_valid_raw_deltaT_mae"))
    result["result_final_raw_deltaT_mse"] = _fmt(final_raw_mse)
    result["result_final_raw_deltaT_rmse"] = _fmt(_sqrt(final_raw_mse))
    result["result_final_raw_deltaT_mae"] = _fmt(_first_number(loss, "final_valid_raw_deltaT_mae"))

    best_iid = _first_number(loss, "best_valid_iid_base_mse", "best_valid_iid_loss")
    final_iid = _first_number(loss, "final_valid_iid_base_mse", "final_valid_iid_loss")
    result["result_best_valid_iid"] = _fmt(best_iid if best_iid is not None else best_mse)
    result["result_final_valid_iid"] = _fmt(final_iid if final_iid is not None else final_mse)
    result["result_best_stress"] = _fmt(
        _first_number(loss, "best_valid_stress_base_mse", "best_valid_stress_loss")
    )
    result["result_final_stress"] = _fmt(
        _first_number(loss, "final_valid_stress_base_mse", "final_valid_stress_loss")
    )

    final_components = loss.get("final_valid_loss_components")
    if isinstance(final_components, dict):
        result["result_strong_q_rmse"] = _fmt(_sqrt(_number(final_components.get("strong_q_mse"))))
        result["result_hotspot_mae"] = _fmt(final_components.get("hotspot_raw_mae"))

    _fill_post_training_diagnostics(result, loss, run_dir)
    _fill_final_probe_metrics(result, loss, run_dir)

    final_probe = loss.get("final_probe_eval_result")
    diagnostics = loss.get("post_training_diagnostics_result")
    result["result_final_probe_status"] = (
        result["result_final_probe_status"] or _status_from_payload(final_probe)
    )
    result["result_post_training_diagnostics_status"] = (
        result["result_post_training_diagnostics_status"]
        or _status_from_payload(diagnostics)
    )
    if not result["result_final_probe_status"]:
        result["result_final_probe_status"] = (
            "disabled" if loss.get("final_probe_eval_after_training") is False else ""
        )
    if not result["result_post_training_diagnostics_status"]:
        result["result_post_training_diagnostics_status"] = (
            "disabled" if loss.get("post_training_diagnostics") is False else ""
        )

    result["result_status"] = _run_status(loss, loss_path)
    if config_path.exists():
        filled_count = sum(1 for field in RESULT_FIELDNAMES if result.get(field))
        result["result_notes"] = f"run_config.json present; filled_result_fields={filled_count}"
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


def _fill_post_training_diagnostics(
    result: dict[str, str], loss: dict[str, Any], run_dir: Path
) -> None:
    diagnostics = loss.get("post_training_diagnostics_result")
    if not isinstance(diagnostics, dict):
        return

    entry = _select_labeled_entry(diagnostics.get("entries"), ("best", "final"))
    if isinstance(entry, dict):
        field = _read_json_ref(entry.get("field_shape_diagnostics_json"), run_dir)
        field_overall = _as_dict(field.get("overall"))
        mechanism = _read_json_ref(entry.get("mechanism_json"), run_dir)
        mechanism_overall = _as_dict(mechanism.get("overall"))
        error_bins = _read_json_ref(entry.get("error_bins_json"), run_dir)
        bin0 = _bin_by_name(error_bins, "bin_0")

        _set_number(result, "result_corr_iid", field_overall, "centered_spatial_correlation")
        _set_number(result, "result_amp", field_overall, "amplitude_ratio")
        _set_number(result, "result_field_variance_iid", field_overall, "field_variance_ratio")
        _set_number(result, "result_valid_iid_topk", field_overall, "top_k_overlap")
        _set_number(result, "result_zrmse", mechanism_overall, "zscore_rmse", "per_sample_zscore_rmse")
        _set_number(result, "result_p95_abs", mechanism_overall, "p95_abs_error")
        _set_number(result, "result_p99_abs", mechanism_overall, "p99_abs_error")
        _set_number(result, "result_peak_abs", mechanism_overall, "peak_abs_error", "max_abs_error")
        _set_number(result, "result_peak_rel", mechanism_overall, "peak_rel_error")
        _set_number(result, "result_bin0_bias", bin0, "trained_signed_bias")
        _set_number(result, "result_bin0_over", bin0, "trained_overprediction_ratio")

    region = _read_json_ref(diagnostics.get("region_error_decomposition_json"), run_dir)
    region_entry = _select_labeled_entry(region.get("entries"), ("best", "final"))
    regions = _as_dict(region_entry.get("regions")) if isinstance(region_entry, dict) else {}
    _set_number(result, "result_top5_rmse", _as_dict(regions.get("top5_deltaT")), "rmse")
    _set_number(result, "result_top10_rmse", _as_dict(regions.get("top10_deltaT")), "rmse")
    strong_q_rmse = _first_number(_as_dict(regions.get("strong_q")), "rmse")
    if strong_q_rmse is not None:
        result["result_strong_q_rmse"] = _fmt(strong_q_rmse)

    le005_bias = _find_named_number(diagnostics, ("le_0p05_signed_bias", "le005_bias"))
    le005_over = _find_named_number(diagnostics, ("le_0p05_over_ratio", "le005_over"))
    if le005_bias is not None:
        result["result_le005_bias"] = _fmt(le005_bias)
    if le005_over is not None:
        result["result_le005_over"] = _fmt(le005_over)


def _fill_final_probe_metrics(
    result: dict[str, str], loss: dict[str, Any], run_dir: Path
) -> None:
    final_probe = loss.get("final_probe_eval_result")
    if not isinstance(final_probe, dict):
        return
    entry = _select_labeled_entry(final_probe.get("entries"), ("best", "final"))
    if not isinstance(entry, dict):
        return
    payload = _read_json_ref(entry.get("metrics_path"), run_dir)
    rows = payload.get("metrics")
    if not isinstance(rows, list):
        return
    result["result_final_probe_rmse"] = _fmt(_mean_metric(rows, "RMSE"))
    result["result_final_probe_relrmse"] = _fmt(
        _mean_metric(rows, "relative_RMSE_on_DeltaT")
    )
    result["result_final_probe_tmax_error"] = _fmt(_mean_metric(rows, "Tmax_error"))
    for probe_id, field in (
        ("P02", "result_probe_p02_rmse"),
        ("P03", "result_probe_p03_rmse"),
        ("P09", "result_probe_p09_rmse"),
    ):
        value = _probe_metric(rows, probe_id, "RMSE")
        if value is not None:
            result[field] = _fmt(value)


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


def _first_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(payload.get(key))
        if value is not None:
            return value
    return None


def _set_number(
    result: dict[str, str], field: str, payload: dict[str, Any], *keys: str
) -> None:
    if result.get(field):
        return
    value = _first_number(payload, *keys)
    if value is not None:
        result[field] = _fmt(value)


def _run_status(loss: dict[str, Any], loss_path: Path) -> str:
    if not loss_path.exists():
        return "missing"
    if loss.get("status_ok") is False or loss.get("grad_finite") is False:
        return "partial"
    return "completed"


def _status_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str):
            return status
        if payload.get("enabled") is False:
            reason = payload.get("reason")
            return str(reason) if reason else "disabled"
        if payload.get("skipped") is True:
            return "skipped"
        if payload.get("returncode") == 0:
            return "completed"
        if payload.get("enabled") is True:
            return "completed"
    return ""


def _display_path(value: Any) -> str:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(value)
    else:
        return ""
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
    return str(path)


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


def _read_json_ref(value: Any, run_dir: Path) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    path = Path(value)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, run_dir / path]
    for candidate in candidates:
        payload = _read_json(candidate)
        if payload:
            return payload
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _select_labeled_entry(entries: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(entries, list):
        return {}
    dict_entries = [entry for entry in entries if isinstance(entry, dict)]
    for label in labels:
        aliases = _label_aliases(label)
        for entry in dict_entries:
            if str(entry.get("label")) in aliases:
                return entry
    return dict_entries[0] if dict_entries else {}


def _label_aliases(label: str) -> set[str]:
    aliases = {
        "best": {"best", "best_predictions"},
        "final": {"final", "predictions", "final_predictions"},
    }
    return aliases.get(label, {label})


def _bin_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    overall = _as_dict(payload.get("overall"))
    bins = overall.get("bins")
    if isinstance(bins, list):
        for item in bins:
            if isinstance(item, dict) and item.get("bin", item.get("bin_name")) == name:
                return item
        if name == "bin_0" and bins and isinstance(bins[0], dict):
            return bins[0]
    bin_summary = _as_dict(overall.get("bin_summary"))
    return _as_dict(bin_summary.get(name))


def _find_named_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = _number(payload.get(key))
            if value is not None:
                return value
        for value in payload.values():
            found = _find_named_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_named_number(item, keys)
            if found is not None:
                return found
    return None


def _mean_metric(rows: list[Any], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows if isinstance(row, dict)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _probe_metric(rows: list[Any], probe_id: str, key: str) -> float | None:
    for row in rows:
        if isinstance(row, dict) and str(row.get("probe_id")) == probe_id:
            return _number(row.get(key))
    return None


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
