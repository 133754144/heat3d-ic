#!/usr/bin/env python3
"""Summarize Heat3D v3 long-run diagnostics without starting training.

The script reads existing ``loss_summary.json`` and optional final/best
diagnostics JSON files from run directories. It writes compact JSON and
Markdown summaries for completed and pending runs. It intentionally does not
import JAX, build graphs, load predictions, or execute training.

External run-list JSON may be either a list of run objects or an object with a
``runs`` list. Each run object should provide at least ``label``, ``run_name``
or ``run_dir``, ``expected_seed``, ``expected_schedule``, ``expected_epochs``,
and ``notes``. Optional expected fields include ``expected_model_seed``,
``expected_batch_order_seed``, ``expected_graph_seed``, ``expected_lr``,
``expected_min_lr``, ``expected_warmup_epochs``, and ``expected_graph_policy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MILESTONE_EPOCHS = (1, 20, 50, 100, 200, 400, 800, 1200, 1600)
CONDITION_GROUP_KEYS = (
    "split",
    "source_category",
    "q_power_range",
    "k_mode",
    "k_region_mode",
    "bc_category",
)
MARKDOWN_WEAK_GROUP_LIMIT = 3

DEFAULT_RUNS: tuple[dict[str, Any], ...] = (
    {
        "label": "W1_seed1_e1200_warmup_flat",
        "root": "wsl2_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_W1_seed1_e1200_upstream_warmup_flat_lr1e-3_wd1e-4",
        "notes": "trusted diagnostic; post-fix upstream_onecycle; not benchmark",
        "expected_seed": 1,
        "expected_model_seed": 1,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "upstream_onecycle",
        "expected_lr": 1.0e-3,
        "expected_epochs": 1200,
        "expected_graph_policy": "nearest_rnode",
    },
    {
        "label": "L2_seed1_e1200_constant",
        "root": "run_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_L2_seed1_e1200_constant_lr1e-3_wd1e-4",
        "notes": "trusted diagnostic; seed1 constant-lr repaired reference",
        "expected_seed": 1,
        "expected_model_seed": 1,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "constant",
        "expected_lr": 1.0e-3,
        "expected_epochs": 1200,
        "expected_graph_policy": "nearest_rnode",
    },
    {
        "label": "S1_seed1_e1600_constant",
        "root": "run_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S1_seed1_e1600_constant_lr1e-3_wd1e-4",
        "notes": "trusted diagnostic; extended seed1 constant-lr path; not benchmark",
        "expected_seed": 1,
        "expected_model_seed": 1,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "constant",
        "expected_lr": 1.0e-3,
        "expected_epochs": 1600,
        "expected_graph_policy": "nearest_rnode",
    },
    {
        "label": "B6_seed0_e400_warmup_cosine",
        "root": "run_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_LR_B6_e400_model_seed0_batchbuild0_batchorder0_graphseed0_lr5e-4_warmup10_minlr5e-5_wd1e-4",
        "notes": "trusted diagnostic; stronger seed0 baseline",
        "expected_seed": 0,
        "expected_model_seed": 0,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "warmup_cosine",
        "expected_lr": 5.0e-4,
        "expected_min_lr": 5.0e-5,
        "expected_warmup_epochs": 10,
        "expected_epochs": 400,
        "expected_graph_policy": "nearest_rnode",
    },
    {
        "label": "S2_seed0_e1200_constant",
        "root": "wsl2_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S2_seed0_e1200_constant_lr1e-3_wd1e-4",
        "notes": "pending; seed0 constant-lr control",
        "expected_seed": 0,
        "expected_model_seed": 0,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "constant",
        "expected_lr": 1.0e-3,
        "expected_epochs": 1200,
        "expected_graph_policy": "nearest_rnode",
    },
    {
        "label": "S3_seed0_e1200_warmup_cosine",
        "root": "wsl2_root",
        "run_name": "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S3_seed0_e1200_warmupcosine_lr1e-3_minlr1e-4_wd1e-4",
        "notes": "pending; seed0 L3-style warmup-cosine control",
        "expected_seed": 0,
        "expected_model_seed": 0,
        "expected_batch_order_seed": 0,
        "expected_graph_seed": 0,
        "expected_schedule": "warmup_cosine",
        "expected_lr": 1.0e-3,
        "expected_min_lr": 1.0e-4,
        "expected_warmup_epochs": 10,
        "expected_epochs": 1200,
        "expected_graph_policy": "nearest_rnode",
    },
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}g}"


def _epoch_to_index(epoch: int) -> int:
    if epoch < 1:
        raise ValueError(f"epoch must be >= 1, got {epoch}")
    return epoch - 1


def _entry_expected(entry: dict[str, Any]) -> dict[str, Any]:
    """Returns expected metadata from either flat or legacy nested fields."""

    nested = entry.get("expected")
    expected = dict(nested) if isinstance(nested, dict) else {}
    field_map = {
        "expected_seed": "seed",
        "expected_model_seed": "model_seed",
        "expected_batch_order_seed": "batch_order_seed",
        "expected_graph_seed": "graph_seed",
        "expected_schedule": "lr_schedule",
        "expected_lr": "lr",
        "expected_min_lr": "min_lr",
        "expected_warmup_epochs": "warmup_epochs",
        "expected_epochs": "epochs",
        "expected_graph_policy": "graph_policy",
    }
    for source, target in field_map.items():
        if source in entry:
            expected[target] = entry[source]
    return expected


def _value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return True
    actual_num = _to_float(actual)
    expected_num = _to_float(expected)
    if actual_num is not None and expected_num is not None:
        return abs(actual_num - expected_num) <= max(1e-12, abs(expected_num) * 1e-9)
    return str(actual) == str(expected)


def _metadata_mismatch_warnings(
    *,
    summary: dict[str, Any] | None,
    run_config: dict[str, Any] | None,
    expected: dict[str, Any],
    graph_policy: str | None,
    final_epoch: int | None,
    diagnostics_complete: bool,
) -> list[str]:
    warnings: list[str] = []
    if summary is None:
        warnings.append("loss_summary.json missing")
        return warnings

    checks = {
        "seed": summary.get("legacy_seed", summary.get("seed")),
        "model_seed": summary.get("model_seed"),
        "batch_order_seed": summary.get("batch_order_seed"),
        "graph_seed": summary.get("graph_seed"),
        "lr_schedule": summary.get("lr_schedule"),
        "lr": summary.get("lr"),
        "min_lr": summary.get("min_lr"),
        "warmup_epochs": summary.get("warmup_epochs"),
        "graph_policy": graph_policy,
    }
    for key, actual in checks.items():
        if key in expected and not _value_matches(actual, expected[key]):
            warnings.append(f"{key} expected {expected[key]!r}, got {actual!r}")

    expected_epochs = _to_int(expected.get("epochs"))
    if expected_epochs is not None and final_epoch is not None and final_epoch < expected_epochs:
        warnings.append(f"final_epoch {final_epoch} below configured {expected_epochs}")

    configured_epochs = None
    if run_config is not None:
        configured_epochs = _to_int(run_config.get("epochs"))
    if configured_epochs is not None and final_epoch is not None and final_epoch > configured_epochs:
        warnings.append(
            f"final_epoch {final_epoch} exceeds run_config epochs {configured_epochs}"
        )

    if not diagnostics_complete:
        warnings.append("final/best diagnostics incomplete")
    return warnings


def _prediction_row(baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline:
        return {}
    rows = baseline.get("overall")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("predictor") == "trained_prediction":
            return row
    for row in rows:
        if isinstance(row, dict) and row.get("predictor") != "zero_delta":
            return row
    return {}


def _split_rows(baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not baseline:
        return []
    rows = baseline.get("split_summary")
    if not isinstance(rows, list):
        return []

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        predictor = row.get("predictor")
        has_trained_fields = row.get("trained_rmse") is not None
        if predictor not in (None, "trained_prediction") and not has_trained_fields:
            continue
        rmse = _to_float(row.get("mean_DeltaT_rmse", row.get("trained_rmse")))
        mae = _to_float(row.get("mean_DeltaT_mae", row.get("trained_mae")))
        split = row.get("split")
        if split is None or rmse is None:
            continue
        parsed.append({"split": str(split), "rmse": rmse, "mae": mae})
    return sorted(parsed, key=lambda item: item["rmse"], reverse=True)


def _bin0(error_bins: dict[str, Any] | None) -> dict[str, Any]:
    if not error_bins:
        return {}
    overall = error_bins.get("overall")
    if not isinstance(overall, dict):
        return {}
    bins = overall.get("bins")
    if not isinstance(bins, list) or not bins:
        return {}
    row = bins[0]
    return row if isinstance(row, dict) else {}


def _weak_condition_groups(condition: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not condition:
        return {}
    groups = condition.get("condition_groups")
    if not isinstance(groups, dict):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for group_key in CONDITION_GROUP_KEYS:
        rows = groups.get(group_key)
        if not isinstance(rows, list):
            continue
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            rmse = _to_float(metrics.get("trained_rmse"))
            if rmse is None:
                continue
            parsed.append(
                {
                    "group_key": group_key,
                    "group_value": row.get("group_value"),
                    "sample_count": metrics.get("sample_count"),
                    "rmse": rmse,
                    "mae": _to_float(metrics.get("trained_mae")),
                    "bias": _to_float(metrics.get("trained_bias")),
                    "overprediction_ratio": _to_float(
                        metrics.get("trained_overprediction_ratio")
                    ),
                }
            )
        if parsed:
            result[group_key] = sorted(parsed, key=lambda item: item["rmse"], reverse=True)
    return result


def _prediction_metrics(run_dir: Path, label: str) -> dict[str, Any]:
    baseline = _read_json(run_dir / f"baseline_comparison_{label}.json")
    error_bins = _read_json(run_dir / f"error_bins_{label}.json")
    condition = _read_json(run_dir / f"condition_diagnostics_{label}.json")
    field_shape = _read_json(run_dir / f"field_shape_diagnostics_{label}.json")

    trained = _prediction_row(baseline)
    bin0 = _bin0(error_bins)
    field = field_shape.get("overall", {}) if field_shape else {}
    if not isinstance(field, dict):
        field = {}

    return {
        "available": bool(baseline and error_bins and condition and field_shape),
        "mean_deltaT_rmse": _to_float(trained.get("mean_DeltaT_rmse")),
        "mean_deltaT_mae": _to_float(trained.get("mean_DeltaT_mae")),
        "centered_spatial_correlation": _to_float(
            field.get("centered_spatial_correlation")
        ),
        "top_k_overlap": _to_float(field.get("top_k_overlap")),
        "per_sample_zscore_rmse": _to_float(field.get("per_sample_zscore_rmse")),
        "amplitude_ratio": _to_float(field.get("amplitude_ratio")),
        "bin0_signed_bias": _to_float(bin0.get("trained_signed_bias")),
        "bin0_overprediction_ratio": _to_float(
            bin0.get("trained_overprediction_ratio")
        ),
        "weak_splits": _split_rows(baseline)[:3],
        "weak_condition_groups": _weak_condition_groups(condition),
    }


def _history_value(summary: dict[str, Any], field: str, epoch: int) -> float | None:
    values = summary.get(field)
    index = _epoch_to_index(epoch)
    if not isinstance(values, list) or index >= len(values):
        return None
    return _to_float(values[index])


def _milestone_losses(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in MILESTONE_EPOCHS:
        rows.append(
            {
                "epoch": epoch,
                "valid_iid_loss": _history_value(summary, "valid_iid_losses", epoch),
                "valid_stress_loss": _history_value(
                    summary, "valid_stress_losses", epoch
                ),
                "lr": _history_value(summary, "epoch_lrs", epoch),
            }
        )
    return rows


def _graph_policy(summary: dict[str, Any] | None, expected: dict[str, Any]) -> str | None:
    if summary:
        graph = summary.get("graph_config")
        if isinstance(graph, dict):
            coverage = graph.get("coverage_repair_policy")
            radius = graph.get("radius_policy")
            if coverage:
                return str(coverage)
            if radius:
                return str(radius)
    value = expected.get("graph_policy")
    return str(value) if value is not None else None


def _resolve_run_dir(entry: dict[str, Any], run_root: Path, wsl2_root: Path) -> Path:
    run_dir = entry.get("run_dir")
    if run_dir:
        return Path(str(run_dir))
    root_name = entry.get("root", "run_root")
    root = wsl2_root if root_name == "wsl2_root" else run_root
    return root / str(entry["run_name"])


def _load_run_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(entry) for entry in DEFAULT_RUNS]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("runs")
    if not isinstance(data, list):
        raise ValueError("run list JSON must be a list or an object with a 'runs' list")
    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each run entry must be a JSON object")
        if not item.get("run_name") and not item.get("run_dir"):
            raise ValueError("each run entry needs run_name or run_dir")
        entries.append(item)
    return entries


def _summarize_run(entry: dict[str, Any], run_root: Path, wsl2_root: Path) -> dict[str, Any]:
    run_dir = _resolve_run_dir(entry, run_root, wsl2_root)
    expected = _entry_expected(entry)

    loss_summary = _read_json(run_dir / "loss_summary.json")
    run_config = _read_json(run_dir / "run_config.json")
    predictions_path = run_dir / "predictions.npz"
    best_predictions_path = run_dir / "best_predictions.npz"
    final_epoch = _to_int((loss_summary or {}).get("final_epoch"))
    configured_epochs = (run_config or {}).get("epochs", expected.get("epochs"))

    final_metrics = _prediction_metrics(run_dir, "final") if loss_summary else {}
    best_metrics = _prediction_metrics(run_dir, "best") if loss_summary else {}
    diagnostics_complete = bool(
        final_metrics.get("available") and best_metrics.get("available")
    )
    predictions_complete = bool(predictions_path.exists() and best_predictions_path.exists())
    graph_policy = _graph_policy(loss_summary, expected)

    if loss_summary is None:
        status = "missing"
    elif bool(loss_summary.get("status_ok")) is False:
        status = "partial"
    elif not predictions_complete:
        status = "needs_predictions"
    elif not diagnostics_complete:
        status = "needs_diagnostics"
    else:
        status = "complete"

    expected_epochs = _to_int(expected.get("epochs"))
    if (
        status == "complete"
        and expected_epochs is not None
        and final_epoch is not None
        and final_epoch < expected_epochs
    ):
        status = "partial"

    warnings = _metadata_mismatch_warnings(
        summary=loss_summary,
        run_config=run_config,
        expected=expected,
        graph_policy=graph_policy,
        final_epoch=final_epoch,
        diagnostics_complete=diagnostics_complete,
    )

    return {
        "label": entry.get("label") or entry.get("run_name") or str(run_dir),
        "run_name": entry.get("run_name") or run_dir.name,
        "run_dir": str(run_dir),
        "status": status,
        "predictions_complete": predictions_complete,
        "diagnostics_complete": diagnostics_complete,
        "metadata_mismatch_warnings": warnings,
        "trusted_note": entry.get("trusted_note") or entry.get("notes"),
        "notes": entry.get("notes") or entry.get("trusted_note"),
        "seed": (loss_summary or {}).get("legacy_seed", (loss_summary or {}).get("seed", expected.get("seed"))),
        "model_seed": (loss_summary or {}).get("model_seed", expected.get("model_seed")),
        "batch_order_seed": (loss_summary or {}).get(
            "batch_order_seed", expected.get("batch_order_seed")
        ),
        "graph_seed": (loss_summary or {}).get("graph_seed", expected.get("graph_seed")),
        "schedule": (loss_summary or {}).get("lr_schedule", expected.get("lr_schedule")),
        "lr": (loss_summary or {}).get("lr", expected.get("lr")),
        "min_lr": (loss_summary or {}).get("min_lr", expected.get("min_lr")),
        "warmup_epochs": (loss_summary or {}).get(
            "warmup_epochs", expected.get("warmup_epochs")
        ),
        "epochs": final_epoch or expected.get("epochs"),
        "final_epoch": final_epoch,
        "configured_epochs": configured_epochs,
        "graph_policy": graph_policy,
        "best_epoch": (loss_summary or {}).get("best_epoch"),
        "final_valid_iid_loss": (loss_summary or {}).get("final_valid_iid_loss"),
        "best_valid_iid_loss": (loss_summary or {}).get("best_valid_iid_loss"),
        "final_valid_stress_loss": (loss_summary or {}).get("final_valid_stress_loss"),
        "best_valid_stress_loss": (loss_summary or {}).get("best_valid_stress_loss"),
        "final_best_ratio": (loss_summary or {}).get("final_best_ratio"),
        "final_prediction": final_metrics,
        "best_prediction": best_metrics,
        "milestone_losses": _milestone_losses(loss_summary) if loss_summary else [],
    }


def _weak_split_text(result: dict[str, Any]) -> str:
    rows = result.get("best_prediction", {}).get("weak_splits") or []
    if not rows:
        return "-"
    parts = []
    for row in rows[:2]:
        parts.append(f"{row['split']} {_format_number(row['rmse'])}")
    return "; ".join(parts)


def _weak_condition_text(result: dict[str, Any]) -> str:
    groups = result.get("best_prediction", {}).get("weak_condition_groups") or {}
    flattened: list[dict[str, Any]] = []
    for group_key in CONDITION_GROUP_KEYS:
        for row in groups.get(group_key) or []:
            flattened.append(dict(row))
    flattened.sort(key=lambda item: _to_float(item.get("rmse")) or float("-inf"), reverse=True)

    parts = []
    for row in flattened[:MARKDOWN_WEAK_GROUP_LIMIT]:
        parts.append(
            "{group_key}={group_value} {rmse}".format(
                group_key=row.get("group_key"),
                group_value=row.get("group_value"),
                rmse=_format_number(row.get("rmse")),
            )
        )
    return "; ".join(parts) if parts else "-"


def _warnings_text(result: dict[str, Any]) -> str:
    warnings = result.get("metadata_mismatch_warnings") or []
    if not warnings:
        return "-"
    return "; ".join(str(item) for item in warnings[:3])


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = summary["runs"]
    lines = [
        "# Heat3D v3 Long-Run Audit Summary",
        "",
        "Generated from existing `loss_summary.json` and diagnostics JSON files.",
        "No training is started by this script.",
        "",
        "| run | status | seed | schedule | configured/final epochs | best_epoch | final/best iid | final/best stress | final RMSE/MAE | best RMSE/MAE | best corr/top-k | best bin0 bias/over | weak split | warnings | caveat |",
        "| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in rows:
        final = result.get("final_prediction", {})
        best = result.get("best_prediction", {})
        seed = result.get("model_seed")
        lines.append(
            "| {label} | {status} | {seed} | {schedule} | {configured_epochs}/{final_epoch} | {best_epoch} | "
            "{final_iid}/{best_iid} | {final_stress}/{best_stress} | "
            "{final_rmse}/{final_mae} | {best_rmse}/{best_mae} | "
            "{corr}/{topk} | {bias}/{over} | {weak_split} | {warnings} | {caveat} |".format(
                label=result["label"],
                status=result["status"],
                seed=seed if seed is not None else "-",
                schedule=result.get("schedule") or "-",
                configured_epochs=result.get("configured_epochs") or "-",
                final_epoch=result.get("final_epoch") or "-",
                best_epoch=result.get("best_epoch") or "-",
                final_iid=_format_number(result.get("final_valid_iid_loss")),
                best_iid=_format_number(result.get("best_valid_iid_loss")),
                final_stress=_format_number(result.get("final_valid_stress_loss")),
                best_stress=_format_number(result.get("best_valid_stress_loss")),
                final_rmse=_format_number(final.get("mean_deltaT_rmse")),
                final_mae=_format_number(final.get("mean_deltaT_mae")),
                best_rmse=_format_number(best.get("mean_deltaT_rmse")),
                best_mae=_format_number(best.get("mean_deltaT_mae")),
                corr=_format_number(best.get("centered_spatial_correlation")),
                topk=_format_number(best.get("top_k_overlap")),
                bias=_format_number(best.get("bin0_signed_bias")),
                over=_format_number(best.get("bin0_overprediction_ratio")),
                weak_split=_weak_split_text(result),
                warnings=_warnings_text(result),
                caveat=result.get("trusted_note") or "-",
            )
        )

    lines.extend(
        [
            "",
            "## Milestone Losses",
            "",
            "| run | e1 iid | e20 iid | e50 iid | e100 iid | e200 iid | e400 iid | e800 iid | e1200 iid | e1600 iid |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in rows:
        by_epoch = {row["epoch"]: row for row in result.get("milestone_losses", [])}
        lines.append(
            "| {label} | {e1} | {e20} | {e50} | {e100} | {e200} | {e400} | {e800} | {e1200} | {e1600} |".format(
                label=result["label"],
                e1=_format_number(by_epoch.get(1, {}).get("valid_iid_loss")),
                e20=_format_number(by_epoch.get(20, {}).get("valid_iid_loss")),
                e50=_format_number(by_epoch.get(50, {}).get("valid_iid_loss")),
                e100=_format_number(by_epoch.get(100, {}).get("valid_iid_loss")),
                e200=_format_number(by_epoch.get(200, {}).get("valid_iid_loss")),
                e400=_format_number(by_epoch.get(400, {}).get("valid_iid_loss")),
                e800=_format_number(by_epoch.get(800, {}).get("valid_iid_loss")),
                e1200=_format_number(by_epoch.get(1200, {}).get("valid_iid_loss")),
                e1600=_format_number(by_epoch.get(1600, {}).get("valid_iid_loss")),
            )
        )

    lines.extend(["", "## Weak Condition Groups", ""])
    for result in rows:
        lines.append(f"- `{result['label']}`: {_weak_condition_text(result)}")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-root", default="output/heat3d_v2_runs")
    parser.add_argument(
        "--wsl2-root",
        default="output/_from_wsl2/DESKTOP-2GE35DV/heat3d_v2_runs",
    )
    parser.add_argument(
        "--run-list-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of run entries. Supports fields "
            "label/run_name/run_dir/expected_seed/expected_schedule/"
            "expected_epochs/notes. Defaults to v3 W1/L2/S1/B6/S2/S3."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/heat3d_v3_long_run_audit/long_run_audit_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/heat3d_v3_long_run_audit/long_run_audit_summary.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    wsl2_root = Path(args.wsl2_root)
    entries = _load_run_entries(args.run_list_json)
    results = [_summarize_run(entry, run_root, wsl2_root) for entry in entries]
    summary = {
        "inputs": {
            "run_root": str(run_root),
            "wsl2_root": str(wsl2_root),
            "run_list_json": str(args.run_list_json) if args.run_list_json else None,
        },
        "runs": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    for result in results:
        print(
            "{label}: status={status} best_iid={best} diagnostics={diag}".format(
                label=result["label"],
                status=result["status"],
                best=_format_number(result.get("best_valid_iid_loss")),
                diag=result["diagnostics_complete"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
