#!/usr/bin/env python3
"""Check the frozen V6 volume resolution ladder result artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = (
    ROOT / "configs/heat3d_v6/v6_volume_resolution_ladder_results.json"
)
CSV_PATH = (
    ROOT / "configs/heat3d_v6/v6_volume_resolution_ladder_results.csv"
)
MD_PATH = ROOT / "docs/v6_volume_resolution_ladder_results.md"
RESOLUTIONS = (1024, 2048, 4096, 8192)
MODELS = (
    "V6_03_V5best_P1h",
    "V6_04_V5best_P1h_DualAttention",
)
NUMERIC_FIELDS = (
    "point_global_cv_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "peak_rmse_K",
    "source_region_cv_rmse_K",
    "layer_mean_rmse_K",
    "layer_drop_rmse_K",
    "top_surface_cv_rmse_K",
    "bottom_surface_cv_rmse_K",
    "graph_build_seconds",
    "inference_seconds",
    "process_peak_rss_bytes",
)


def main() -> int:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if (
        payload["status"] != "completed_valid_only_cpu"
        or payload["evaluation_role"] != "valid_iid"
        or payload["resolution_order"] != list(RESOLUTIONS)
        or payload["batch_size"] != 1
        or payload["models"] != list(MODELS)
        or payload["test_hard_accessed"]
        or payload["training_executed"]
        or payload["checkpoint_selection_modified"]
    ):
        raise RuntimeError("top-level evaluation contract failed")
    execution = payload["execution_contract"]
    if execution != {
        "cuda_visible_devices": "",
        "gpu_memory": "N/A",
        "gpu_used": False,
        "jax_platforms": "cpu",
        "remote_inference_executed": False,
    }:
        raise RuntimeError("CPU-only execution contract drifted")
    rows = payload["rows"]
    expected = [(n, model) for n in RESOLUTIONS for model in MODELS]
    actual = [(row["resolution"], row["config_id"]) for row in rows]
    if actual != expected:
        raise RuntimeError("resolution/model order drifted")
    for row in rows:
        if (
            row["checkpoint_kind"] != "point_global_best"
            or row["checkpoint_epoch"] != 111
            or row["batch_size"] != 1
            or row["device"] != "TFRT_CPU_0"
            or row["gpu_memory"] != "N/A"
            or row["global_context_fit_population"] != "train_only"
            or row["global_context_fit_sample_count"] != 768
        ):
            raise RuntimeError(f"{row['resolution']}/{row['config_id']}: binding")
        if not all(
            math.isfinite(float(row[field])) and float(row[field]) >= 0.0
            for field in NUMERIC_FIELDS
        ):
            raise RuntimeError(
                f"{row['resolution']}/{row['config_id']}: numeric field"
            )
    raw = payload["raw_evaluations"]
    for resolution in RESOLUTIONS:
        item = raw[str(resolution)]
        if (
            item["resolution"] != resolution
            or item["evaluation_role"] != "valid_iid"
            or item["test_hard_accessed"]
            or item["training_executed"]
            or not item["formal_inference_executed"]
            or item["execution_environment"]["gpu_used"]
        ):
            raise RuntimeError(f"{resolution}: raw payload contract failed")
        for model in MODELS:
            per_sample = item["models"][model]["metrics"]["per_sample"]
            if len(per_sample) != 128:
                raise RuntimeError(f"{resolution}/{model}: sample count drifted")
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(rows):
        raise RuntimeError("CSV row count drifted")
    for source, mirror in zip(rows, csv_rows, strict=True):
        if (
            int(mirror["resolution"]) != source["resolution"]
            or mirror["config_id"] != source["config_id"]
        ):
            raise RuntimeError("CSV key mismatch")
        for field in NUMERIC_FIELDS:
            if not math.isclose(
                float(mirror[field]),
                float(source[field]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"CSV numeric mismatch: {field}")
    markdown = MD_PATH.read_text(encoding="utf-8")
    for token in (
        "JAX_PLATFORMS=cpu",
        'CUDA_VISIBLE_DEVICES=""',
        "test",
        "8192",
        "GPU memory is N/A",
    ):
        if token not in markdown:
            raise RuntimeError(f"Markdown missing {token!r}")
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(rows),
                "resolutions": list(RESOLUTIONS),
                "models": list(MODELS),
                "test_hard_accessed": False,
                "training_executed": False,
                "gpu_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
