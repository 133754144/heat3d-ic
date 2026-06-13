#!/usr/bin/env python3
"""Audit scalar-loss versus mechanism-metric mismatch for Heat3D v3 runs.

The script compares existing final/best prediction archives across multiple
run directories. It reads only ``run_config.json``, ``loss_summary.json``,
prediction ``.npz`` files, and subset sample arrays/metadata. It does not
import JAX, build graphs, execute a model, or train.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_heat3d_v3_prediction_mechanisms as mech  # noqa: E402


DEFAULT_LABEL_TO_PREDICTION = {
    "final": "predictions.npz",
    "best": "best_predictions.npz",
}
GROUP_KEYS = mech.GROUP_KEYS
DEFAULT_PAIRED_COMPARISONS = (
    "B6:best=S3:final",
    "B6:best=S3:best",
    "B6:best=S2:best",
    "S3:best=S3:final",
)


def _parse_run(token: str) -> tuple[str, Path]:
    if "=" not in token:
        path = Path(token)
        return path.name, path
    label, path = token.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty run label in {token!r}")
    return label, Path(path)


def _parse_endpoint(token: str) -> tuple[str, str]:
    if ":" not in token:
        raise ValueError(f"prediction endpoint must be RUN:LABEL, found {token!r}")
    run, label = token.split(":", 1)
    run = run.strip()
    label = label.strip()
    if not run or not label:
        raise ValueError(f"prediction endpoint must be RUN:LABEL, found {token!r}")
    if label not in DEFAULT_LABEL_TO_PREDICTION:
        raise ValueError(f"unsupported prediction label {label!r} in endpoint {token!r}")
    return run, label


def _parse_pair(token: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if "=" not in token:
        raise ValueError(f"paired comparison must be REF_RUN:REF_LABEL=TARGET_RUN:TARGET_LABEL, found {token!r}")
    left, right = token.split("=", 1)
    return _parse_endpoint(left), _parse_endpoint(right)


def _endpoint_text(endpoint: tuple[str, str]) -> str:
    return f"{endpoint[0]}:{endpoint[1]}"


def _prediction_labels(spec: str) -> list[str]:
    labels = [item.strip() for item in spec.split(",") if item.strip()]
    if not labels:
        raise ValueError("--prediction-labels must include at least one label")
    for label in labels:
        if label not in DEFAULT_LABEL_TO_PREDICTION:
            raise ValueError(f"unsupported prediction label {label!r}")
    return labels


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _scalar_losses(loss_summary: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if not loss_summary:
        return {}
    if label == "best":
        prefix = "best"
    else:
        prefix = "final"
    return {
        "valid_iid_loss": loss_summary.get(f"{prefix}_valid_iid_loss"),
        "valid_stress_loss": loss_summary.get(f"{prefix}_valid_stress_loss"),
        "valid_loss": loss_summary.get(f"{prefix}_valid_loss"),
        "best_epoch": loss_summary.get("best_epoch"),
        "final_epoch": loss_summary.get("final_epoch"),
        "final_best_ratio": loss_summary.get("final_best_ratio"),
    }


def _sample_row(row: dict[str, Any]) -> dict[str, Any]:
    rmse = mech._json_float(row.get("rmse"))
    return {
        "sample_id": row.get("sample_id"),
        "groups": row.get("groups", {}),
        "rmse": rmse,
        "squared_error": rmse * rmse if rmse is not None else None,
        "mae": row.get("mae"),
        "peak_rel_error": row.get("peak_rel_error"),
        "zscore_rmse": row.get("zscore_rmse"),
        "top_k_overlap": row.get("top_k_overlap"),
        "amplitude_ratio": row.get("amplitude_ratio"),
        "centered_corr": row.get("centered_corr"),
    }


def _top_samples(rows: list[dict[str, Any]], *, metric: str, limit: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        value = mech._json_float(row.get(metric))
        return value if value is not None else float("-inf")

    return [_sample_row(row) for row in sorted(rows, key=key, reverse=True)[:limit]]


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = mech._grouped(rows)
    result: dict[str, list[dict[str, Any]]] = {}
    for key, group_rows in grouped.items():
        result[key] = sorted(
            group_rows,
            key=lambda row: mech._json_float(row.get("rmse")) or float("-inf"),
            reverse=True,
        )
    return result


def _load_run_prediction(
    *,
    run_label: str,
    run_dir: Path,
    prediction_label: str,
    top_k: int,
    top_samples: int,
) -> dict[str, Any]:
    loss_summary = _maybe_json(run_dir / "loss_summary.json")
    prediction_name = DEFAULT_LABEL_TO_PREDICTION[prediction_label]
    prediction_path = run_dir / prediction_name
    if not prediction_path.is_file():
        return {
            "run": run_label,
            "prediction_label": prediction_label,
            "status": "missing_prediction",
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
            "scalar_losses": _scalar_losses(loss_summary, prediction_label),
        }

    subset = mech._resolve_subset(run_dir, None)
    rows, failures, q_edges = mech._load_sample_rows(
        subset=subset,
        prediction_path=prediction_path,
        top_k=top_k,
    )
    grouped = _group_summaries(rows)
    return {
        "run": run_label,
        "prediction_label": prediction_label,
        "status": "complete",
        "run_dir": str(run_dir),
        "prediction_path": str(prediction_path),
        "sample_count": len(rows),
        "failed_sample_count": len(failures),
        "q_power_edges": q_edges,
        "scalar_losses": _scalar_losses(loss_summary, prediction_label),
        "overall": mech._aggregate(rows),
        "top_error_samples": _top_samples(rows, metric="rmse", limit=top_samples),
        "top_peak_rel_samples": _top_samples(rows, metric="peak_rel_error", limit=top_samples),
        "top_shape_samples": _top_samples(rows, metric="zscore_rmse", limit=top_samples),
        "per_sample": [_sample_row(row) for row in rows],
        "grouped": grouped,
        "failures": failures,
    }


def _rank_complete(results: list[dict[str, Any]], metric_path: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result.get("status") != "complete":
            continue
        value: Any = result
        for key in metric_path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        numeric = mech._json_float(value)
        if numeric is None:
            continue
        rows.append(
            {
                "run": result["run"],
                "prediction_label": result["prediction_label"],
                "value": numeric,
            }
        )
    return sorted(rows, key=lambda row: row["value"])


def _group_metric_map(result: dict[str, Any], group_key: str) -> dict[str, dict[str, Any]]:
    rows = result.get("grouped", {}).get(group_key) or []
    return {str(row.get("group_value")): row for row in rows}


def _compare_to_reference(
    results: list[dict[str, Any]],
    *,
    reference_run: str,
    metric: str,
    limit: int,
) -> dict[str, Any]:
    complete = [row for row in results if row.get("status") == "complete"]
    by_label: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in complete:
        by_label[row["prediction_label"]][row["run"]] = row

    payload: dict[str, Any] = {}
    for label, runs in sorted(by_label.items()):
        reference = runs.get(reference_run)
        if reference is None:
            continue
        label_items = []
        for run_name, other in sorted(runs.items()):
            if run_name == reference_run:
                continue
            for group_key in GROUP_KEYS:
                ref_groups = _group_metric_map(reference, group_key)
                other_groups = _group_metric_map(other, group_key)
                for group_value, other_group in other_groups.items():
                    ref_group = ref_groups.get(group_value)
                    if ref_group is None:
                        continue
                    other_value = mech._json_float(other_group.get(metric))
                    ref_value = mech._json_float(ref_group.get(metric))
                    if other_value is None or ref_value is None:
                        continue
                    label_items.append(
                        {
                            "comparison": f"{run_name}_minus_{reference_run}",
                            "prediction_label": label,
                            "group_key": group_key,
                            "group_value": group_value,
                            "metric": metric,
                            "other_value": other_value,
                            "reference_value": ref_value,
                            "delta": other_value - ref_value,
                            "other_sample_count": other_group.get("sample_count"),
                            "reference_sample_count": ref_group.get("sample_count"),
                        }
                    )
        payload[label] = sorted(label_items, key=lambda item: item["delta"], reverse=True)[:limit]
    return payload


def _metric_delta(
    target: dict[str, Any],
    reference: dict[str, Any],
    metric: str,
) -> float | None:
    target_value = mech._json_float(target.get(metric))
    reference_value = mech._json_float(reference.get(metric))
    if target_value is None or reference_value is None:
        return None
    return target_value - reference_value


def _paired_row(
    *,
    reference_endpoint: tuple[str, str],
    target_endpoint: tuple[str, str],
    sample_id: str,
    reference_row: dict[str, Any],
    target_row: dict[str, Any],
) -> dict[str, Any]:
    groups = target_row.get("groups") or reference_row.get("groups") or {}
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "reference_run": reference_endpoint[0],
        "reference_label": reference_endpoint[1],
        "target_run": target_endpoint[0],
        "target_label": target_endpoint[1],
        "groups": groups,
    }
    for key in GROUP_KEYS:
        payload[key] = groups.get(key, "unknown")
    for metric in ("rmse", "zscore_rmse", "peak_rel_error", "top_k_overlap"):
        payload[f"reference_{metric}"] = reference_row.get(metric)
        payload[f"target_{metric}"] = target_row.get(metric)
        payload[f"delta_{metric}"] = _metric_delta(target_row, reference_row, metric)
    return payload


def _sample_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in result.get("per_sample") or []:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            rows[str(sample_id)] = row
    return rows


def _sort_delta_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    reverse: bool,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        value = mech._json_float(row.get(metric))
        if value is None:
            return float("-inf") if reverse else float("inf")
        return value

    return sorted(rows, key=key, reverse=reverse)


def _paired_comparisons(
    results: list[dict[str, Any]],
    *,
    pairs: list[tuple[tuple[str, str], tuple[str, str]]],
    limit: int,
) -> list[dict[str, Any]]:
    complete = {
        (str(result.get("run")), str(result.get("prediction_label"))): result
        for result in results
        if result.get("status") == "complete"
    }
    comparisons = []
    for reference_endpoint, target_endpoint in pairs:
        comparison_name = f"{_endpoint_text(reference_endpoint)}_vs_{_endpoint_text(target_endpoint)}"
        reference = complete.get(reference_endpoint)
        target = complete.get(target_endpoint)
        if reference is None or target is None:
            comparisons.append(
                {
                    "comparison": comparison_name,
                    "reference": _endpoint_text(reference_endpoint),
                    "target": _endpoint_text(target_endpoint),
                    "status": "missing_endpoint",
                    "reference_available": reference is not None,
                    "target_available": target is not None,
                    "rows": [],
                    "reference_beats_target_most": [],
                    "target_beats_reference_most": [],
                }
            )
            continue

        reference_rows = _sample_map(reference)
        target_rows = _sample_map(target)
        common_sample_ids = sorted(set(reference_rows).intersection(target_rows))
        rows = [
            _paired_row(
                reference_endpoint=reference_endpoint,
                target_endpoint=target_endpoint,
                sample_id=sample_id,
                reference_row=reference_rows[sample_id],
                target_row=target_rows[sample_id],
            )
            for sample_id in common_sample_ids
        ]
        reference_beats = _sort_delta_rows(rows, metric="delta_rmse", reverse=True)
        target_beats = _sort_delta_rows(rows, metric="delta_rmse", reverse=False)
        comparisons.append(
            {
                "comparison": comparison_name,
                "reference": _endpoint_text(reference_endpoint),
                "target": _endpoint_text(target_endpoint),
                "status": "complete",
                "common_sample_count": len(common_sample_ids),
                "reference_only_sample_count": len(set(reference_rows).difference(target_rows)),
                "target_only_sample_count": len(set(target_rows).difference(reference_rows)),
                "rows": rows,
                "reference_beats_target_most": reference_beats[:limit],
                "target_beats_reference_most": target_beats[:limit],
            }
        )
    return comparisons


def build_mismatch_payload(
    *,
    runs: list[tuple[str, Path]],
    prediction_labels: list[str],
    top_k: int,
    top_samples: int,
    comparison_limit: int,
    paired_limit: int,
    pairs: list[tuple[tuple[str, str], tuple[str, str]]],
) -> dict[str, Any]:
    results = []
    for run_label, run_dir in runs:
        for prediction_label in prediction_labels:
            results.append(
                _load_run_prediction(
                    run_label=run_label,
                    run_dir=run_dir,
                    prediction_label=prediction_label,
                    top_k=top_k,
                    top_samples=top_samples,
                )
            )

    rankings = {
        "scalar_valid_iid_loss": _rank_complete(results, ("scalar_losses", "valid_iid_loss")),
        "scalar_valid_stress_loss": _rank_complete(results, ("scalar_losses", "valid_stress_loss")),
        "raw_rmse": _rank_complete(results, ("overall", "rmse")),
        "zscore_rmse": _rank_complete(results, ("overall", "zscore_rmse")),
        "peak_rel_error": _rank_complete(results, ("overall", "peak_rel_error")),
        "top_k_overlap_desc": list(
            reversed(_rank_complete(results, ("overall", "top_k_overlap")))
        ),
    }
    return {
        "diagnostic_scope": "read-only scalar-loss versus mechanism-metric mismatch audit",
        "inputs": {
            "runs": [{"label": label, "run_dir": str(path)} for label, path in runs],
            "prediction_labels": prediction_labels,
            "top_k": top_k,
            "top_samples": top_samples,
            "paired_limit": paired_limit,
            "pairs": [
                {"reference": _endpoint_text(reference), "target": _endpoint_text(target)}
                for reference, target in pairs
            ],
        },
        "results": results,
        "rankings": rankings,
        "reference_group_deltas_vs_B6": {
            "rmse": _compare_to_reference(
                results,
                reference_run="B6",
                metric="rmse",
                limit=comparison_limit,
            ),
            "peak_rel_error": _compare_to_reference(
                results,
                reference_run="B6",
                metric="peak_rel_error",
                limit=comparison_limit,
            ),
            "zscore_rmse": _compare_to_reference(
                results,
                reference_run="B6",
                metric="zscore_rmse",
                limit=comparison_limit,
            ),
        },
        "paired_sample_deltas": _paired_comparisons(results, pairs=pairs, limit=paired_limit),
    }


def _fmt(value: Any) -> str:
    numeric = mech._json_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.6g}"


def _ranking_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    return "; ".join(
        f"{row['run']}:{row['prediction_label']}={_fmt(row['value'])}"
        for row in rows[:6]
    )


def _result_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| run | label | status | valid_iid | valid_stress | raw RMSE | zRMSE | top-k | peak_rel |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        scalar = result.get("scalar_losses", {})
        overall = result.get("overall", {})
        lines.append(
            "| {run} | {label} | {status} | {iid} | {stress} | {rmse} | {zrmse} | {topk} | {peak} |".format(
                run=result.get("run"),
                label=result.get("prediction_label"),
                status=result.get("status"),
                iid=_fmt(scalar.get("valid_iid_loss")),
                stress=_fmt(scalar.get("valid_stress_loss")),
                rmse=_fmt(overall.get("rmse")),
                zrmse=_fmt(overall.get("zscore_rmse")),
                topk=_fmt(overall.get("top_k_overlap")),
                peak=_fmt(overall.get("peak_rel_error")),
            )
        )
    return lines


def _top_delta_text(items: list[dict[str, Any]]) -> str:
    if not items:
        return "-"
    parts = []
    for item in items[:5]:
        parts.append(
            "{comparison} {group_key}={group_value} delta={delta}".format(
                comparison=item["comparison"],
                group_key=item["group_key"],
                group_value=item["group_value"],
                delta=_fmt(item["delta"]),
            )
        )
    return "; ".join(parts)


def _paired_sample_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| sample | split | source | q_power | k_mode | k_region | bc | dRMSE | dzRMSE | dPeakRel | dTopK |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {split} | {source} | {q} | {k_mode} | {k_region} | {bc} | {drmse} | {dz} | {dpeak} | {dtopk} |".format(
                sample=row.get("sample_id"),
                split=row.get("split"),
                source=row.get("source_category"),
                q=row.get("q_power_range"),
                k_mode=row.get("k_mode"),
                k_region=row.get("k_region_mode"),
                bc=row.get("bc_category"),
                drmse=_fmt(row.get("delta_rmse")),
                dz=_fmt(row.get("delta_zscore_rmse")),
                dpeak=_fmt(row.get("delta_peak_rel_error")),
                dtopk=_fmt(row.get("delta_top_k_overlap")),
            )
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v3 Scalar/Mechanism Mismatch Audit",
        "",
        "Read-only comparison of scalar loss, raw prediction error, shape, and hotspot metrics.",
        "",
        "## Run Summary",
        "",
    ]
    lines.extend(_result_table(payload["results"]))
    lines.extend(
        [
            "",
            "## Rankings",
            "",
            f"- scalar valid_iid: {_ranking_text(payload['rankings']['scalar_valid_iid_loss'])}",
            f"- scalar valid_stress: {_ranking_text(payload['rankings']['scalar_valid_stress_loss'])}",
            f"- raw RMSE: {_ranking_text(payload['rankings']['raw_rmse'])}",
            f"- zscore RMSE: {_ranking_text(payload['rankings']['zscore_rmse'])}",
            f"- peak relative error: {_ranking_text(payload['rankings']['peak_rel_error'])}",
            f"- top-k overlap: {_ranking_text(payload['rankings']['top_k_overlap_desc'])}",
            "",
            "## Largest Group Deltas vs B6",
            "",
        ]
    )
    for metric, by_label in payload["reference_group_deltas_vs_B6"].items():
        lines.append(f"- {metric}:")
        for label, items in by_label.items():
            lines.append(f"  - {label}: {_top_delta_text(items)}")
    lines.extend(
        [
            "",
            "## Paired Per-Sample Deltas",
            "",
            "Deltas are target minus reference, aligned by sample_id. Positive dRMSE means the reference is better on RMSE; negative dRMSE means the target is better.",
        ]
    )
    for comparison in payload.get("paired_sample_deltas", []):
        lines.extend(
            [
                "",
                f"### {comparison.get('comparison')}",
                "",
                f"- status: {comparison.get('status')}",
                f"- common samples: {comparison.get('common_sample_count', 0)}",
                "",
                "Reference beats target most:",
                "",
            ]
        )
        lines.extend(_paired_sample_table(comparison.get("reference_beats_target_most", [])))
        lines.extend(["", "Target beats reference most:", ""])
        lines.extend(_paired_sample_table(comparison.get("target_beats_reference_most", [])))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Scalar losses come from the training loss summary; mechanism metrics are raw DeltaT / recovered-field diagnostics.",
            "- A run can rank better on raw mechanism metrics while ranking worse on scalar loss because normalization and sample/point weighting differ.",
            "- Inspect top-error samples and group deltas before treating scalar valid loss as the only selection criterion.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run entry as LABEL=RUN_DIR.")
    parser.add_argument("--prediction-labels", default="final,best")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-samples", type=int, default=20)
    parser.add_argument("--comparison-limit", type=int, default=12)
    parser.add_argument(
        "--pair",
        action="append",
        help=(
            "Paired sample comparison as REF_RUN:REF_LABEL=TARGET_RUN:TARGET_LABEL. "
            "Defaults to B6/S2/S3 comparisons when omitted."
        ),
    )
    parser.add_argument("--paired-limit", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [_parse_run(token) for token in args.run]
    payload = build_mismatch_payload(
        runs=runs,
        prediction_labels=_prediction_labels(args.prediction_labels),
        top_k=args.top_k,
        top_samples=args.top_samples,
        comparison_limit=args.comparison_limit,
        paired_limit=args.paired_limit,
        pairs=[_parse_pair(token) for token in (args.pair or DEFAULT_PAIRED_COMPARISONS)],
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"scalar valid_iid: {_ranking_text(payload['rankings']['scalar_valid_iid_loss'])}")
    print(f"raw RMSE: {_ranking_text(payload['rankings']['raw_rmse'])}")
    print(f"zscore RMSE: {_ranking_text(payload['rankings']['zscore_rmse'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
