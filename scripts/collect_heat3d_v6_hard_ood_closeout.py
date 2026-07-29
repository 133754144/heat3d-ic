#!/usr/bin/env python3
"""Collect frozen valid/holdout/hard-stress V6 metrics without new inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values())))


def _row(
    *,
    resolution: int,
    role: str,
    classification: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    cycle = payload["cycles"][0]
    support = cycle["metrics"]["support"]
    full = cycle["metrics"]["full_field"]
    row = {
        "resolution": resolution,
        "role": role,
        "role_classification": classification,
        "sample_count": full["sample_count"],
        "support_point_global_pct": support[
            "point_global_cv_relative_rmse_pct"
        ],
        "support_sample_first_pct": support[
            "sample_first_cv_relative_rmse_pct"
        ],
        "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
        "support_field_bias_K": support["field_cv_weighted_bias_K"],
        "support_peak_rmse_K": support["peak"]["rmse_K"],
        "support_source_rmse_K": support["source_region"][
            "cv_weighted_rmse_K"
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
        "full_layer_rmse_K_mean": _mean(full["layer_cv_weighted_rmse_K"]),
        "full_interface_rmse_K_mean": _mean(
            full["interface_cv_weighted_rmse_K"]
        ),
        "full_top_rmse_K": full["top_cv_weighted_rmse_K"],
        "full_bottom_rmse_K": full["bottom_cv_weighted_rmse_K"],
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
        "used_for_selection_or_tuning": False,
    }
    if not all(
        math.isfinite(float(value))
        for key, value in row.items()
        if key
        not in {
            "role",
            "role_classification",
            "checkpoint_sha256",
            "used_for_selection_or_tuning",
        }
    ):
        raise RuntimeError(f"non-finite metric row: {resolution}/{role}")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--hard-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    performance = _load(CONFIG / "v6_final_performance_closeout.json")
    prereg = _load(CONFIG / "v6_hard_ood_preregistration.json")
    role_manifest = _load(CONFIG / "v6_hard_input_stress_role.json")
    preflight = _load(CONFIG / "v6_hard_ood_preflight.json")
    adapter = _load(CONFIG / "v6_hard_ood_evaluator_adapter.json")
    rows = []
    result_hashes: dict[str, str] = {}
    for resolution in (4096, 8192, 16384):
        valid_path = args.valid_dir / f"{resolution}_persistent_b8.json"
        valid = _load(valid_path)
        holdout = performance["corrected_confirmatory_holdout"][
            str(resolution)
        ]
        hard_path = (
            args.hard_dir / f"hard_input_stress_{resolution}.json"
        )
        hard = _load(hard_path)
        if (
            hard["role"] != "hard_input_stress"
            or hard["sample_count"] != 16
            or hard["role_manifest_sha256"]
            != prereg["roles"]["hard_input_stress"]["role_manifest_sha256"]
            or hard["checkpoint"]["sha256"]
            != prereg["checkpoint"]["sha256"]
            or hard["hard_accessed"] is not True
            or hard["ood_accessed"] is not False
        ):
            raise RuntimeError(f"hard result binding failed at {resolution}")
        rows.extend(
            [
                _row(
                    resolution=resolution,
                    role="valid_iid",
                    classification="selection_split_frozen_existing",
                    payload=valid,
                ),
                _row(
                    resolution=resolution,
                    role="corrected_confirmatory_holdout",
                    classification="confirmatory_not_for_selection",
                    payload=holdout,
                ),
                _row(
                    resolution=resolution,
                    role="hard_input_stress",
                    classification=(
                        "input_defined_in_distribution_stress_subset"
                    ),
                    payload=hard,
                ),
            ]
        )
        result_hashes[str(resolution)] = _sha256(hard_path)
    _write_csv(args.output_csv, rows)

    comparisons = {}
    for resolution in (4096, 8192, 16384):
        by_role = {
            row["role"]: row
            for row in rows
            if row["resolution"] == resolution
        }
        hard = by_role["hard_input_stress"]
        holdout = by_role["corrected_confirmatory_holdout"]
        valid = by_role["valid_iid"]
        comparisons[str(resolution)] = {
            "hard_minus_valid_full_point_global_pct": (
                hard["full_point_global_pct"]
                - valid["full_point_global_pct"]
            ),
            "hard_minus_holdout_full_point_global_pct": (
                hard["full_point_global_pct"]
                - holdout["full_point_global_pct"]
            ),
            "hard_to_valid_full_raw_rmse_ratio": (
                hard["full_raw_cv_rmse_K"]
                / valid["full_raw_cv_rmse_K"]
            ),
            "hard_to_holdout_full_raw_rmse_ratio": (
                hard["full_raw_cv_rmse_K"]
                / holdout["full_raw_cv_rmse_K"]
            ),
        }
    payload = {
        "schema_version": "heat3d_v6_hard_ood_closeout_v1",
        "status": "passed",
        "preregistration_commit": (
            "63ef72007c6973805f08e68fad9c2f0dfe5122b6"
        ),
        "preflight_execution_head": preflight["execution_head"],
        "evaluator_adapter": {
            "status": adapter["status"],
            "sha256": _sha256(
                CONFIG / "v6_hard_ood_evaluator_adapter.json"
            ),
            "metrics_formulas_changed": False,
        },
        "checkpoint": prereg["checkpoint"],
        "workflow": prereg["workflow"],
        "hard_role": role_manifest,
        "hard_result_sha256": result_hashes,
        "metric_rows": rows,
        "comparisons": comparisons,
        "canonical_ood": {
            "status": "not_available_not_run",
            "reason": prereg["roles"]["canonical_ood"]["reason"],
            "labels_accessed": False,
        },
        "support_bias_scope": (
            "support_field_cv_weighted_bias_K; the frozen full-field "
            "accumulator did not expose a full-field signed-bias metric"
        ),
        "protocol_deviation": performance["protocol_deviation"],
        "hard_used_for_selection_or_tuning": False,
        "confirmatory_holdout_used_for_selection_or_tuning": False,
        "posthoc_reselection_allowed": False,
        "training_executed": False,
        "checkpoint_sampling_graph_reconstruction_modified": False,
        "local_absolute_paths_persisted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V6 hard/OOD closeout",
        "",
        "The canonical checkpoint and 4096/8192/16384 Anchor-derived workflow "
        "were frozen before this descriptive evaluation. 32768 was excluded.",
        "",
        "Canonical P1h has no registered distribution-shift OOD role, so no "
        "OOD labels were opened. `hard_input_stress` is an input-defined "
        "in-distribution corner, not OOD.",
        "",
        "| Mode | Role | Support point-global % | Full point-global % | "
        "Full RMSE K | Peak K | Source K | Layer/interface K | Bias K |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['role']} | "
            f"{row['support_point_global_pct']:.4f} | "
            f"{row['full_point_global_pct']:.4f} | "
            f"{row['full_raw_cv_rmse_K']:.4f} | "
            f"{row['full_peak_rmse_K']:.4f} | "
            f"{row['full_source_rmse_K']:.4f} | "
            f"{row['full_layer_rmse_K_mean']:.4f}/"
            f"{row['full_interface_rmse_K_mean']:.4f} | "
            f"{row['support_field_bias_K']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The hard and corrected-confirmatory rows are descriptive only and "
            "cannot change the canonical model, checkpoint, resolution roles, "
            "or any tuning decision.",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "metric_rows": len(rows),
                "hard_samples": 16,
                "canonical_ood": "not_available_not_run",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
