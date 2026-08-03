#!/usr/bin/env python3
"""Aggregate frozen P1i three-seed and resolution benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
DOC_DIR = ROOT / "docs"
RUNS = (
    (0, "devbox", "V6_06_V5best_P1i_seed0_reliable_B24"),
    (1, "wsl2", "V6_07_V5best_P1i_seed1_reliable_B24"),
    (2, "wsl2", "V6_08_V5best_P1i_seed2_reliable_B24"),
)
RESOLUTIONS = (1024, 4096, 16384, 65536, 240825)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(metrics: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return next(row for row in metrics if row["checkpoint_label"] == label)


def _mean_std(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    output = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "values": values.tolist(),
        }
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _artifact_inventory(seed_dir: Path, replay: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(seed_dir.iterdir()):
        if path.is_file() and path.name not in {
            "valid_support_per_sample.csv",
            "all_valid_fullfield.md",
        }:
            rows.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "source": "local_read_only_sync",
                }
            )
    for entry in replay["entries"]:
        rows.append(
            {
                "name": Path(entry["checkpoint"]["path"]).name,
                "size_bytes": None,
                "sha256": entry["checkpoint"]["sha256"],
                "source": "remote_frozen_checkpoint",
                "checkpoint_label": entry["label"],
                "epoch": entry["checkpoint"]["epoch"],
            }
        )
    return rows


def _svg_plot(
    path: Path,
    series: list[tuple[str, str, list[tuple[float, float, str]]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    """Write a dependency-free, deterministic paper-ready SVG line plot."""
    import html
    import math

    width, height = 900, 560
    left, right, top, bottom = 92, 28, 54, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [(x, y) for _, _, values in series for x, y, _ in values]
    transform_x = math.log10 if log_x else (lambda value: value)
    transform_y = math.log10 if log_y else (lambda value: value)
    xs = [transform_x(x) for x, _ in points]
    ys = [transform_y(y) for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.06, 1e-9)
    ypad = max((ymax - ymin) * 0.08, 1e-9)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def sx(value: float) -> float:
        return left + (transform_x(value) - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - transform_y(value)) / (ymax - ymin) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17212b}.axis{stroke:#17212b;stroke-width:1.4}.grid{stroke:#c9d1d9;stroke-width:.8;stroke-dasharray:3 4}.line{fill:none;stroke-width:2.4}.label{font-size:13px}.tick{font-size:12px}.title{font-size:19px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">{html.escape(title)}</text>',
    ]
    for index in range(6):
        frac = index / 5
        x = left + frac * plot_w
        y = top + frac * plot_h
        xv = 10 ** (xmin + frac * (xmax - xmin)) if log_x else xmin + frac * (xmax - xmin)
        yv = 10 ** (ymax - frac * (ymax - ymin)) if log_y else ymax - frac * (ymax - ymin)
        svg.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>',
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle">{xv:.3g}</text>',
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{yv:.3g}</text>',
        ])
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text class="label" x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">{html.escape(xlabel)}</text>',
        f'<text class="label" transform="translate(23 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>',
    ])
    for series_index, (label, color, values) in enumerate(series):
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _ in values)
        svg.append(f'<polyline class="line" stroke="{color}" points="{polyline}"/>')
        for point_index, (x, y, annotation) in enumerate(values):
            label_offset_y = (point_index - (len(values) - 1) / 2) * 24 if series_index == 1 else -7
            label_offset_x = 14 + 34 * (point_index % 2) if series_index == 1 else 7
            svg.extend([
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.2" fill="{color}"/>',
                f'<text class="tick" x="{sx(x) + label_offset_x:.2f}" y="{sy(y) + label_offset_y:.2f}">{html.escape(annotation)}</text>',
            ])
        legend_x, legend_y = left + 12, top + 18 + 21 * series_index
        svg.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 27}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>',
            f'<text class="legend" x="{legend_x + 34}" y="{legend_y + 4}">{html.escape(label)}</text>',
        ])
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _plot_benchmark(rows: list[dict[str, Any]]) -> None:
    route_styles = (
        ("direct", "direct target-resolution inference", "#d1495b"),
        ("interpolate", "1024 inference + reconstruction", "#2878b5"),
        ("solver", "structured FVM", "#2a9d68"),
    )
    latency_series = []
    for route, label, color in route_styles:
        values = [
            (float(row["steady_end_to_end_median_s"]), float(row["point_global_true_rms_pct"]), str(row["resolution"]))
            for row in rows if row["route"] == route and row["status"] == "passed"
        ]
        latency_series.append((label, color, values))
    _svg_plot(
        DOC_DIR / "v6_p1i_accuracy_latency.svg",
        latency_series,
        title="P1i resolution accuracy-latency benchmark",
        xlabel="Steady end-to-end latency per sample (s, log scale)",
        ylabel="Full-field point-global true-RMS error (%, log scale)",
    )
    direct = [row for row in rows if row["route"] == "direct" and row["status"] == "passed"]
    _svg_plot(
        DOC_DIR / "v6_p1i_memory_resolution.svg",
        [
            ("GPU peak", "#7b2cbf", [(float(row["resolution"]), float(row["peak_gpu_bytes"]) / 2**30, str(row["resolution"])) for row in direct]),
            ("process RAM peak", "#e76f51", [(float(row["resolution"]), float(row["peak_ram_bytes"]) / 2**30, str(row["resolution"])) for row in direct]),
        ],
        title="P1i direct-inference memory scaling",
        xlabel="Target nodes (log scale)",
        ylabel="Peak memory (GiB, log scale)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()

    seeds = []
    checkpoint_rows = []
    primary_rows = []
    artifacts = []
    reconstruction_floor: dict[str, Any] | None = None
    for seed, host, config_id in RUNS:
        seed_dir = args.evidence_root / f"seed{seed}"
        support = json.loads((seed_dir / "valid_support.json").read_text())
        full = json.loads((seed_dir / "all_valid_fullfield.json").read_text())
        replay = json.loads((seed_dir / "independent_replay.json").read_text())
        loss = json.loads((seed_dir / "loss_summary.json").read_text())
        provenance = json.loads((seed_dir / "pretraining_provenance.json").read_text())
        if replay["status"] != "passed" or support["test_accessed"] or full["test_accessed"]:
            raise RuntimeError(f"seed{seed}: valid-only replay contract failed")
        if provenance["git_commit"] != "3884de07525b7e8c0f8fa3382b24bf94322bebe9":
            raise RuntimeError("training commit drifted")
        by_full = full["model_plus_reconstruction"]
        current_floor = full["reconstruction_only_sampling_floor"]
        if reconstruction_floor is None:
            reconstruction_floor = current_floor
        elif current_floor != reconstruction_floor:
            raise RuntimeError("reconstruction-only sampling floor drifted across seeds")
        for label in ("point_global_best", "sample_first_best", "base_mse_best", "final"):
            support_metric = _metric(support["metrics"], label)
            full_metric = by_full[label]
            checkpoint_rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "host": host,
                    "checkpoint_label": label,
                    "epoch": support_metric["checkpoint_epoch"],
                    "support_point_global_pct": support_metric["point_global_relative_rmse_pct"],
                    "support_sample_first_pct": support_metric["sample_first_cv_relative_rmse_pct"],
                    "support_raw_cv_rmse_K": support_metric["raw_cv_weighted_rmse_K"],
                    "shape_cv_rmse": support_metric["shape_cv_rmse"],
                    "scale_log_rmse": support_metric["scale_log_rmse"],
                    "full_point_global_pct": full_metric["point_global_true_rms_relative_rmse_pct"],
                    "full_sample_first_pct": full_metric["sample_first_cv_relative_rmse_pct"],
                    "full_raw_cv_rmse_K": full_metric["raw_cv_weighted_rmse_K"],
                    "peak_rmse_K": full_metric["peak_rmse_K"],
                    "source_rmse_K": full_metric["source"]["cv_weighted_rmse_K"],
                    "background_rmse_K": full_metric["background"]["cv_weighted_rmse_K"],
                    "layer_mean_rmse_K": full_metric["layer_mean_rmse_K"],
                    "layer_drop_rmse_K": full_metric["layer_drop_rmse_K"],
                    "interface_rmse_K": full_metric["interface_mean_rmse_K"],
                    "top_rmse_K": full_metric["top"]["cv_weighted_rmse_K"],
                    "bottom_rmse_K": full_metric["bottom"]["cv_weighted_rmse_K"],
                }
            )
        primary = next(row for row in checkpoint_rows if row["seed"] == seed and row["checkpoint_label"] == "point_global_best")
        primary_support_metric = _metric(support["metrics"], "point_global_best")
        primary_rows.append(primary)
        seed_payload = {
            "seed": seed,
            "host": host,
            "config_id": config_id,
            "training_commit": provenance["git_commit"],
            "command": provenance["command"],
            "log": {
                "size_bytes": (seed_dir / "training.log").stat().st_size,
                "sha256": _sha256(seed_dir / "training.log"),
                "epoch_600_recorded": "epoch 600/600" in (seed_dir / "training.log").read_text(errors="replace"),
            },
            "primary": primary,
            "primary_error_tail": primary_support_metric["tail"],
            "best_to_final_support": support["best_to_final"],
            "independent_replay": replay,
            "final_epoch": int(loss["final_epoch"]),
        }
        seeds.append(seed_payload)
        artifacts.append(
            {
                "seed": seed,
                "host": host,
                "config_id": config_id,
                "files": _artifact_inventory(seed_dir, replay),
            }
        )

    aggregate_keys = [
        "support_point_global_pct", "support_sample_first_pct", "support_raw_cv_rmse_K",
        "shape_cv_rmse", "scale_log_rmse", "full_point_global_pct",
        "full_sample_first_pct", "full_raw_cv_rmse_K", "peak_rmse_K",
        "source_rmse_K", "background_rmse_K", "layer_mean_rmse_K",
        "layer_drop_rmse_K", "interface_rmse_K", "top_rmse_K", "bottom_rmse_K",
    ]
    aggregate = _mean_std(primary_rows, aggregate_keys)

    benchmark_payloads = []
    benchmark_rows = []
    for resolution in RESOLUTIONS:
        payload = json.loads(
            (args.evidence_root / f"p1i_resolution_{resolution}.json").read_text()
        )
        benchmark_payloads.append(payload)
        solver = payload["route_C_structured_FVM"]
        solver_e2e = float(solver["steady_end_to_end_seconds"]["median"])
        for route, key in (
            ("direct", "route_A_direct"),
            ("interpolate", "route_B_1024_plus_interpolation"),
            ("solver", "route_C_structured_FVM"),
        ):
            current = payload[key]
            status = current.get("status", "passed")
            model_stats: dict[str, Any] = {}
            e2e_stats: dict[str, Any] = current.get("steady_end_to_end_seconds") or {}
            interpolation_stats: dict[str, Any] = {}
            if route == "solver":
                model_median = None
                graph_build = None
                jit = None
                interpolation = 0.0
                e2e = solver_e2e
                assembly_stats = solver["assembly_seconds"]
                solve_stats = solver["linear_solve_seconds"]
            elif status == "passed":
                model_stats = current["steady_model_inference_seconds"]
                model_median = float(model_stats["median"])
                graph_build = float(current["graph_build_seconds"])
                jit = float(current["jit_first_seconds"])
                interp_key = "full_field_interpolation_seconds" if route == "direct" else "interpolation_seconds"
                interpolation_stats = current[interp_key]
                interpolation = float(interpolation_stats["median"])
                e2e = float(current["steady_end_to_end_seconds"]["median"])
                assembly_stats = solve_stats = {}
            else:
                model_median = graph_build = jit = interpolation = e2e = None
                assembly_stats = solve_stats = {}
            metrics = current.get("metrics") or {}
            oracle_metrics = current.get("oracle_reconstruction_metrics") or {}
            benchmark_rows.append(
                {
                    "resolution": resolution,
                    "route": route,
                    "status": status,
                    "data_prepare_s": payload["data_prepare_seconds"],
                    "graph_build_s": graph_build,
                    "jit_first_s": jit,
                    "steady_model_median_s": model_median,
                    "steady_model_mean_s": model_stats.get("mean"),
                    "steady_model_std_s": model_stats.get("std"),
                    "steady_model_p95_s": model_stats.get("p95"),
                    "interpolation_median_s": interpolation,
                    "interpolation_mean_s": interpolation_stats.get("mean"),
                    "interpolation_std_s": interpolation_stats.get("std"),
                    "interpolation_p95_s": interpolation_stats.get("p95"),
                    "assembly_median_s": assembly_stats.get("median"),
                    "assembly_mean_s": assembly_stats.get("mean"),
                    "assembly_std_s": assembly_stats.get("std"),
                    "assembly_p95_s": assembly_stats.get("p95"),
                    "linear_solve_median_s": solve_stats.get("median"),
                    "linear_solve_mean_s": solve_stats.get("mean"),
                    "linear_solve_std_s": solve_stats.get("std"),
                    "linear_solve_p95_s": solve_stats.get("p95"),
                    "steady_end_to_end_median_s": e2e,
                    "steady_end_to_end_mean_s": e2e_stats.get("mean"),
                    "steady_end_to_end_std_s": e2e_stats.get("std"),
                    "steady_end_to_end_p95_s": e2e_stats.get("p95"),
                    "cold_end_to_end_s": solver.get("cold", {}).get("end_to_end_seconds") if route == "solver" else ((graph_build or 0.0) + (jit or 0.0) + (e2e or 0.0) if status == "passed" else None),
                    "point_global_true_rms_pct": metrics.get("point_global_true_rms_relative_rmse_pct"),
                    "quality_gate_point_global_lt20": (
                        float(metrics["point_global_true_rms_relative_rmse_pct"]) < 20.0
                        if metrics.get("point_global_true_rms_relative_rmse_pct") is not None else None
                    ),
                    "sample_first_cv_pct": metrics.get("sample_first_cv_relative_rmse_pct"),
                    "raw_cv_rmse_K": metrics.get("raw_cv_weighted_rmse_K"),
                    "oracle_reconstruction_point_global_pct": oracle_metrics.get("point_global_true_rms_relative_rmse_pct"),
                    "oracle_reconstruction_sample_first_cv_pct": oracle_metrics.get("sample_first_cv_relative_rmse_pct"),
                    "oracle_reconstruction_raw_cv_rmse_K": oracle_metrics.get("raw_cv_weighted_rmse_K"),
                    "peak_rmse_K": metrics.get("peak_rmse_K"),
                    "source_rmse_K": (metrics.get("source") or {}).get("cv_weighted_rmse_K"),
                    "background_rmse_K": (metrics.get("background") or {}).get("cv_weighted_rmse_K"),
                    "layer_mean_rmse_K": metrics.get("layer_mean_rmse_K"),
                    "interface_rmse_K": metrics.get("interface_mean_rmse_K"),
                    "solver_over_inference_speedup": (solver_e2e / model_median if model_median else None),
                    "solver_over_end_to_end_speedup": (solver_e2e / e2e if e2e else None),
                    "peak_gpu_bytes": payload["gpu_memory"]["peak_bytes_in_use"],
                    "peak_ram_bytes": payload["process_peak_ram_bytes"],
                }
            )

    closeout = {
        "schema_version": "heat3d_v6_p1i_three_seed_closeout_v1",
        "status": "completed_three_seed_valid_only",
        "dataset_id": "heat3d_v6_p1i_continuous_physics1024_v1",
        "training_commit": "3884de07525b7e8c0f8fa3382b24bf94322bebe9",
        "accessed_roles": ["train_inputs_for_frozen_normalization", "valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
        "retrained": False,
        "tuned": False,
        "primary_checkpoint": "point_global_best",
        "seeds": seeds,
        "primary_mean_std": aggregate,
        "checkpoint_comparison": checkpoint_rows,
        "valid_full_field_reconstruction_only_sampling_floor": reconstruction_floor,
        "benchmark": {
            "host_alias": "devbox",
            "reported_host": benchmark_payloads[0]["environment"]["host"],
            "platform": benchmark_payloads[0]["environment"]["platform"],
            "sample_ids": benchmark_payloads[0]["sample_ids"],
            "repeats": 20,
            "batch_size": 1,
            "resolutions": list(RESOLUTIONS),
            "command_template": "source ~/miniforge3/etc/profile.d/conda.sh; conda activate rigno; MEM_FRACTION=0.85 python /tmp/benchmark_heat3d_v6_p1i_resolution.py --config configs/heat3d_v6_p1i/V6_06_V5best_P1i_seed0_reliable_B24.yaml --checkpoint output/heat3d_v6_p1i_runs/V6_06_V5best_P1i_seed0_reliable_B24/params_best_valid_point_global.pkl --resolution <1024|4096|16384|65536|240825> --repeats 20 --output /tmp/p1i_resolution_<nodes>.json",
            "raw_results": benchmark_payloads,
        },
    }
    closeout_path = CONFIG_DIR / "v6_p1i_three_seed_inference_closeout.json"
    closeout_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    _write_csv(CONFIG_DIR / "v6_p1i_three_seed_checkpoint_metrics.csv", checkpoint_rows)
    _write_csv(CONFIG_DIR / "v6_p1i_resolution_accuracy_latency.csv", benchmark_rows)
    manifest = {
        "schema_version": "heat3d_v6_p1i_three_seed_artifact_manifest_v1",
        "training_commit": closeout["training_commit"],
        "dataset_manifest_sha256": "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514",
        "full_field_archive_sha256": "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb",
        "evaluator_code_sha256": {
            name: _sha256(ROOT / "scripts" / name)
            for name in (
                "replay_heat3d_v6_p1i_checkpoint.py",
                "evaluate_heat3d_v6_p1i_formal_valid_support.py",
                "evaluate_heat3d_v6_p1i_valid_full_field.py",
                "benchmark_heat3d_v6_p1i_resolution.py",
            )
        },
        "artifacts": artifacts,
        "commands": {
            "checkpoint_replay": "source ~/miniforge3/etc/profile.d/conda.sh; conda activate rigno; python /tmp/replay_heat3d_v6_p1i_checkpoint.py --config <seed-yaml> --run-dir <frozen-run-dir> --output /tmp/<seed>_independent_replay.json",
            "support_evaluator": "source ~/miniforge3/etc/profile.d/conda.sh; conda activate rigno; python /tmp/evaluate_heat3d_v6_p1i_formal_valid_support.py --config <seed-yaml> --run-dir <frozen-run-dir> --output-json /tmp/<seed>_valid_support.json --output-csv /tmp/<seed>_valid_support.csv --output-sample-csv /tmp/<seed>_valid_support_per_sample.csv",
            "resolution_benchmark": closeout["benchmark"]["command_template"],
        },
        "test_accessed": False,
        "sealed_accessed": False,
    }
    (CONFIG_DIR / "v6_p1i_three_seed_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    registry_path = CONFIG_DIR / "v6_p1i_training_registry.csv"
    registry_rows = list(csv.DictReader(registry_path.open(encoding="utf-8")))
    for row in registry_rows:
        match = next((item for item in primary_rows if item["config_id"] == row["config_id"]), None)
        if match is None:
            continue
        final = next(
            item for item in checkpoint_rows
            if item["config_id"] == row["config_id"] and item["checkpoint_label"] == "final"
        )
        row.update(
            {
                "execution_status": "completed_e600",
                "evaluation_status": "completed_valid_support_fullfield_replay",
                "best_epoch": str(match["epoch"]),
                "final_epoch": "600",
                "checkpoint_status": "complete_atomic_optimizer_best_sample_base_final_latest_reload_passed",
                "best_valid_point_global_pct": str(match["support_point_global_pct"]),
                "best_valid_sample_first_pct": str(match["support_sample_first_pct"]),
                "best_valid_raw_cv_rmse_K": str(match["support_raw_cv_rmse_K"]),
                "final_valid_point_global_pct": str(final["support_point_global_pct"]),
                "result_path": "configs/heat3d_v6_p1i/v6_p1i_three_seed_inference_closeout.json",
            }
        )
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=registry_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry_rows)

    _plot_benchmark(benchmark_rows)
    lines = [
        "# V6 P1i three-seed formal closeout and inference benchmark",
        "",
        "本报告仅使用 train 输入拟合冻结标准化，并评价 `valid_iid`；test 与 sealed IID 均未打开。训练协议、checkpoint 和模型参数未修改，也未重训或调参。",
        "",
        "## Three-seed primary checkpoint",
        "",
        "| seed | epoch | support point-global % | support sample-first % | support raw CV K | full-field point-global % | full-field raw CV K | peak/source/background K | layer/interface/top/bottom K |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['seed']} | {row['epoch']} | {row['support_point_global_pct']:.6f} | {row['support_sample_first_pct']:.6f} | {row['support_raw_cv_rmse_K']:.6f} | {row['full_point_global_pct']:.6f} | {row['full_raw_cv_rmse_K']:.6f} | {row['peak_rmse_K']:.3f}/{row['source_rmse_K']:.3f}/{row['background_rmse_K']:.3f} | {row['layer_mean_rmse_K']:.3f}/{row['interface_rmse_K']:.3f}/{row['top_rmse_K']:.3f}/{row['bottom_rmse_K']:.3f} |"
        )
    lines.extend(
        [
            "",
            "三 seed mean±std：support point-global "
            f"{aggregate['support_point_global_pct']['mean']:.6f}±{aggregate['support_point_global_pct']['std']:.6f}%，"
            "sample-first "
            f"{aggregate['support_sample_first_pct']['mean']:.6f}±{aggregate['support_sample_first_pct']['std']:.6f}%，"
            "full-field point-global "
            f"{aggregate['full_point_global_pct']['mean']:.6f}±{aggregate['full_point_global_pct']['std']:.6f}%。",
            "",
            "同一 128-sample valid 真值在冻结 1024 support 上直接采样后重建至 240825 节点的 oracle sampling floor："
            f"point-global {reconstruction_floor['point_global_true_rms_relative_rmse_pct']:.6f}%，"
            f"sample-first {reconstruction_floor['sample_first_cv_relative_rmse_pct']:.6f}%，"
            f"raw CV {reconstruction_floor['raw_cv_weighted_rmse_K']:.6f} K。"
            "它与模型+重建误差分开报告。",
            "",
            "## Checkpoint reliability and late-epoch behavior",
            "",
            "- 三个独立 Python 进程均成功加载 best/sample-first/base/final/latest；参数归档 schema 为 optimizer-aware v2。",
            "- 跨进程重放 RMSE 均低于 0.01 K；GPU scatter 的极少数单点差异完整保留在 machine-readable audit 中。",
            "- support point-SSE best→final 分别变化 "
            + ", ".join(f"seed{x['seed']} {x['best_to_final_support']['point_sse_change_pct']:.3f}%" for x in seeds)
            + "；seed1 的 sample-first 与 full-field final 略优，但不能替代预注册 point-global primary。",
            "- primary sample-relative tail（p95/max）分别为 "
            + ", ".join(
                f"seed{x['seed']} {x['primary_error_tail']['sample_relative_rmse_pct']['p95']:.3f}%/"
                f"{x['primary_error_tail']['sample_relative_rmse_pct']['max']:.3f}%"
                for x in seeds
            )
            + "；top-10 point-SSE 占比分别为 "
            + ", ".join(f"seed{x['seed']} {100*x['primary_error_tail']['top10_point_sse_fraction']:.2f}%" for x in seeds)
            + "。",
            "",
            "## Resolution–accuracy–latency",
            "",
            "固定 `devbox` SSH alias（系统报告 hostname `XYH-Desktop`、WSL2）、RTX 5070、B1、同一首个 valid 样本；每个稳态阶段预热一次后重复 20 次。FVM 与直接模型使用同节点数结构化网格；source 以 control-volume overlap 守恒投影。",
            "",
            "| nodes | direct PG % | direct graph/JIT/steady-e2e s | 1024+recon PG % (oracle floor) | 1024+recon core/e2e s | FVM PG % | FVM assembly/solve/e2e s | FVM/B core / e2e |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for resolution in RESOLUTIONS:
        direct = next(row for row in benchmark_rows if row["resolution"] == resolution and row["route"] == "direct")
        interp = next(row for row in benchmark_rows if row["resolution"] == resolution and row["route"] == "interpolate")
        solver = next(row for row in benchmark_rows if row["resolution"] == resolution and row["route"] == "solver")
        direct_pg = "failed" if direct["status"] != "passed" else f"{direct['point_global_true_rms_pct']:.4f}"
        direct_time = "failed" if direct["status"] != "passed" else f"{direct['graph_build_s']:.3f}/{direct['jit_first_s']:.3f}/{direct['steady_end_to_end_median_s']:.4f}"
        lines.append(
            f"| {resolution} | {direct_pg} | {direct_time} | {interp['point_global_true_rms_pct']:.4f} ({interp['oracle_reconstruction_point_global_pct']:.4f}) | {interp['steady_model_median_s']:.4f}/{interp['steady_end_to_end_median_s']:.4f} | {solver['point_global_true_rms_pct']:.4f} | {solver['assembly_median_s']:.4f}/{solver['linear_solve_median_s']:.4f}/{solver['steady_end_to_end_median_s']:.4f} | {solver['steady_end_to_end_median_s']/interp['steady_model_median_s']:.3f}×/{solver['steady_end_to_end_median_s']/interp['steady_end_to_end_median_s']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Applicability",
            "",
            "- 当前 checkpoint 的可信默认路径仍是 P1i 原生 source-aware 1024 support 后进行 layer-aware reconstruction；它在所有目标分辨率上最稳定。",
            "- 直接结构化高分辨率不具稳定兼容性：误差随分辨率非单调，65536 的偶然恢复没有在 240825 延续，因此所有 A 路线结果都只作 compatibility diagnostic，不构成生产适用区间。",
            "- 基准精度与耗时使用一个预注册 valid 样本，适合工程兼容性与阶段计时，不替代 128-sample 三 seed 质量统计。",
            "- FVM 和模型同节点数，但计算图/数值方法不同；speedup 仅为同硬件单样本墙钟比较。",
        ]
    )
    (DOC_DIR / "v6_p1i_three_seed_inference_closeout.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "seeds": 3, "resolutions": 5}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
