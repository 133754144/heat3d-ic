#!/usr/bin/env python3
"""Collect the formal CPU-only V6 volume resolution ladder evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (1024, 2048, 4096, 8192)
MODELS = (
    "V6_03_V5best_P1h",
    "V6_04_V5best_P1h_DualAttention",
)
CHECKPOINT_SHA256 = {
    "V6_03_V5best_P1h": (
        "3ad58c2b34a46481acb74722c80bdcadbf55a0d613bc25c4fe2d7646b91aa1f2"
    ),
    "V6_04_V5best_P1h_DualAttention": (
        "a127b020da14f3c7bdc544c0068ea755d9f58f1be0ee6cd627add914a6aec122"
    ),
}
METRICS = (
    "point_global_cv_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "peak_rmse_K",
    "source_region_cv_rmse_K",
    "layer_mean_rmse_K",
    "layer_drop_rmse_K",
    "top_surface_cv_rmse_K",
    "bottom_surface_cv_rmse_K",
)


class CollectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(resolution: int, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_payload = payload["models"][model]
    checkpoint = model_payload["checkpoint"]
    metrics = model_payload["metrics"]
    row = {
        "resolution": resolution,
        "config_id": model,
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_epoch": int(checkpoint["checkpoint_epoch"]),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "training_commit": checkpoint["training_commit"],
        "parameter_count": int(checkpoint["parameter_count"]),
        "point_global_cv_relative_rmse_pct": float(
            metrics["point_global_cv_relative_rmse_pct"]
        ),
        "sample_first_cv_relative_rmse_pct": float(
            metrics["sample_first_cv_relative_rmse_pct"]
        ),
        "raw_cv_weighted_rmse_K": float(metrics["raw_cv_weighted_rmse_K"]),
        "peak_rmse_K": float(metrics["peak"]["rmse_K"]),
        "source_region_cv_rmse_K": float(
            metrics["source_region"]["cv_weighted_rmse_K"]
        ),
        "layer_mean_rmse_K": float(metrics["layer_mean"]["rmse_K"]),
        "layer_drop_rmse_K": float(metrics["layer_drop"]["rmse_K"]),
        "top_surface_cv_rmse_K": float(
            metrics["top_surface"]["cv_weighted_rmse_K"]
        ),
        "bottom_surface_cv_rmse_K": float(
            metrics["bottom_surface"]["cv_weighted_rmse_K"]
        ),
        "graph_build_seconds": float(checkpoint["graph_build_seconds"]),
        "inference_seconds": float(checkpoint["inference_seconds"]),
        "process_peak_rss_bytes": int(checkpoint["process_peak_rss_bytes"]),
        "process_peak_rss_GiB": float(
            checkpoint["process_peak_rss_bytes"] / (1024**3)
        ),
        "gpu_memory": "N/A",
        "device": checkpoint["device"],
        "batch_size": int(checkpoint["batch_size"]),
        "global_context_fit_population": checkpoint[
            "global_context_fit_population"
        ],
        "global_context_fit_sample_count": int(
            checkpoint["global_context_fit_sample_count"]
        ),
    }
    if not all(
        math.isfinite(float(row[key]))
        for key in (*METRICS, "graph_build_seconds", "inference_seconds")
    ):
        raise CollectionError(f"{resolution}/{model}: non-finite result")
    return row


def _relative_change(new: float, old: float) -> float:
    return (new - old) / old


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(row["resolution"]), str(row["config_id"])): row for row in rows
    }
    stability: dict[str, Any] = {}
    for model in MODELS:
        low = by_key[(4096, model)]
        high = by_key[(8192, model)]
        relative = {
            key: _relative_change(float(high[key]), float(low[key]))
            for key in METRICS
        }
        stability[model] = {
            "relative_change_4096_to_8192": relative,
            "max_abs_relative_change_4096_to_8192": max(
                abs(value) for value in relative.values()
            ),
            "descriptive_all_core_metrics_within_2pct": all(
                abs(value) <= 0.02 for value in relative.values()
            ),
        }
    v603 = by_key[(8192, MODELS[0])]
    v604 = by_key[(8192, MODELS[1])]
    model_delta = {
        key: float(v604[key]) - float(v603[key]) for key in METRICS
    }
    peak_rss = max(int(row["process_peak_rss_bytes"]) for row in rows)
    return {
        "stability": stability,
        "v6_04_minus_v6_03_at_8192": model_delta,
        "resolution_8192": {
            "completed": True,
            "finite": True,
            "batch_size": 1,
            "max_process_peak_rss_bytes": peak_rss,
            "max_process_peak_rss_GiB": peak_rss / (1024**3),
            "gpu_memory": "N/A",
            "descriptive_feasibility": (
                "feasible_on_this_local_CPU_host; graph construction dominates "
                "runtime and no OOM/non-finite value occurred"
            ),
        },
        "interpretation": {
            "resolution_stability": (
                "4096-to-8192 aggregate metrics are descriptively stable within "
                "2%; 1024 remains visibly support-resolution sensitive"
            ),
            "model_difference": (
                "V6_04 is modestly better than V6_03 at 8192 across the reported "
                "aggregate metrics, but both fail badly on the source-unbiased "
                "volume probe and this is not a checkpoint-selection result"
            ),
            "support_warning": (
                "These volume-representative probes intentionally remove the "
                "historical 50% source-dense quota; they are not interchangeable "
                "with the frozen common source-dense 4096 probe."
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# V6 volume resolution ladder formal CPU evaluation",
        "",
        "Status: **completed**. Evaluation used only `valid_iid`, batch size 1, "
        "`JAX_PLATFORMS=cpu`, and `CUDA_VISIBLE_DEVICES=\"\"`. No training, "
        "checkpoint mutation, test, hard, or GPU execution occurred.",
        "",
        "The table uses the frozen V6_03 seed0 and V6_04 point-global checkpoints.",
        "",
        "| nodes | model | point-global % | sample-first % | raw RMSE K | peak K | source K | layer mean K | layer drop K | top K | bottom K | graph s | inference s | peak RSS GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {resolution} | {config_id} | "
            "{point_global_cv_relative_rmse_pct:.4f} | "
            "{sample_first_cv_relative_rmse_pct:.4f} | "
            "{raw_cv_weighted_rmse_K:.4f} | {peak_rmse_K:.4f} | "
            "{source_region_cv_rmse_K:.4f} | {layer_mean_rmse_K:.4f} | "
            "{layer_drop_rmse_K:.4f} | {top_surface_cv_rmse_K:.4f} | "
            "{bottom_surface_cv_rmse_K:.4f} | {graph_build_seconds:.2f} | "
            "{inference_seconds:.2f} | {process_peak_rss_GiB:.3f} |".format(
                **row
            )
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Stability",
            "",
        ]
    )
    for model in MODELS:
        item = summary["stability"][model]
        lines.append(
            f"- `{model}`: maximum absolute relative aggregate-metric change "
            f"from 4096 to 8192 is "
            f"{100.0 * item['max_abs_relative_change_4096_to_8192']:.3f}%; "
            f"descriptive 2% stability check = "
            f"`{str(item['descriptive_all_core_metrics_within_2pct']).lower()}`."
        )
    delta = summary["v6_04_minus_v6_03_at_8192"]
    lines.extend(
        [
            "",
            "At 8192 nodes, V6_04−V6_03 is "
            f"{delta['point_global_cv_relative_rmse_pct']:+.4f} percentage "
            "points for point-global, "
            f"{delta['sample_first_cv_relative_rmse_pct']:+.4f} percentage "
            "points for sample-first, and "
            f"{delta['raw_cv_weighted_rmse_K']:+.4f} K for raw CV RMSE.",
            "",
            "## Feasibility and interpretation",
            "",
            f"- 8192 completed on local CPU with maximum observed process peak "
            f"RSS {summary['resolution_8192']['max_process_peak_rss_GiB']:.3f} "
            "GiB. GPU memory is N/A.",
            "- 8192 is feasible on this host, but graph construction dominates "
            "runtime; it is suitable for bounded validation rather than frequent "
            "per-epoch evaluation.",
            "- The ladder is stable from 4096 to 8192, but absolute errors remain "
            "very large. The result shows strong dependence on the historical "
            "source-dense operator support, not a useful model-quality gain.",
            "- V6_04 is modestly better at 8192, but the difference is small "
            "relative to the shared failure on the volume-representative support.",
            "- The 2% stability statement is descriptive, not a preregistered "
            "promotion threshold, and no checkpoint/model selection was changed.",
            "",
            "## Provenance",
            "",
            f"- evaluator SHA256: `{payload['provenance']['evaluator_sha256']}`",
            f"- base Git HEAD: `{payload['provenance']['evaluation_base_git_head']}`",
            "- checkpoints were hash-verified; train-only Global Context fit count "
            "was 768 for every run.",
            "- raw per-sample valid-only payloads are embedded in the unified JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for resolution in RESOLUTIONS:
        parser.add_argument(
            f"--input-{resolution}", type=Path, required=True
        )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--evaluation-base-git-head", required=True)
    args = parser.parse_args()

    raw: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        path = getattr(args, f"input_{resolution}").resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        environment = payload.get("execution_environment") or {}
        if (
            payload.get("status") != "passed"
            or payload.get("evaluation_role") != "valid_iid"
            or payload.get("test_hard_accessed")
            or payload.get("training_executed")
            or not payload.get("formal_inference_executed")
            or payload.get("resolution") != resolution
            or environment.get("jax_platforms") != "cpu"
            or environment.get("cuda_visible_devices") != ""
            or environment.get("gpu_used")
            or environment.get("batch_size") != 1
            or set(payload.get("models", {})) != set(MODELS)
        ):
            raise CollectionError(f"{resolution}: evaluation contract failed")
        for model in MODELS:
            checkpoint = payload["models"][model]["checkpoint"]
            if (
                checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA256[model]
                or checkpoint["checkpoint_kind"] != "point_global_best"
                or checkpoint["device"] != "TFRT_CPU_0"
                or checkpoint["gpu_memory"] != "N/A_CPU_only"
                or checkpoint["global_context_fit_population"] != "train_only"
                or checkpoint["global_context_fit_sample_count"] != 768
            ):
                raise CollectionError(
                    f"{resolution}/{model}: frozen binding failed"
                )
            rows.append(_row(resolution, model, payload))
        raw[str(resolution)] = payload
        input_sha[str(resolution)] = _sha256(path)

    evaluator_path = ROOT / "scripts/evaluate_heat3d_v6_volume_probe_ladder.py"
    result = {
        "schema_version": "heat3d_v6_volume_resolution_ladder_results_v1",
        "status": "completed_valid_only_cpu",
        "evaluation_role": "valid_iid",
        "resolution_order": list(RESOLUTIONS),
        "batch_size": 1,
        "models": list(MODELS),
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_selection_modified": False,
        "execution_contract": {
            "jax_platforms": "cpu",
            "cuda_visible_devices": "",
            "gpu_used": False,
            "gpu_memory": "N/A",
            "remote_inference_executed": False,
        },
        "provenance": {
            "evaluation_base_git_head": args.evaluation_base_git_head,
            "evaluator_path": str(evaluator_path.relative_to(ROOT)),
            "evaluator_sha256": _sha256(evaluator_path),
            "raw_input_sha256": input_sha,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "rows": rows,
        "summary": _summary(rows),
        "raw_evaluations": raw,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_csv, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "row_count": len(rows),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
