#!/usr/bin/env python3
"""Freeze the valid-only V6 unified quality/runtime benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"
DOCS = ROOT / "docs"
ENVIRONMENT = CONFIG / "v6_inference_qualification_environment.json"
STATES = ("process_cold", "jit_cached_new_topology", "known_topology_new_physics", "fully_cached")
RESOLUTION_STATES = ("process_cold", "new_topology", "known_topology_new_physics", "fully_cached")
RESOLUTIONS = (4096, 8192, 16384, 32768, 65536, 240825)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def state_distribution(item: Mapping[str, Any], state: str, *, resolution: bool) -> Mapping[str, Any] | None:
    if item.get("status", "passed") != "passed":
        return None
    if state == "process_cold":
        return item["external_continuous_wall_seconds" if resolution else "external_process_wall_seconds"]
    return item["stage_timing"]["continuous_wall_seconds"]


def stage_value(item: Mapping[str, Any], state: str, name: str) -> Mapping[str, Any] | None:
    stage = item.get("stage_timing", {})
    return stage.get(name)


def timing_row(
    *, family: str, resolution: int, route: str, state: str,
    item: Mapping[str, Any], resolution_payload: bool,
) -> dict[str, Any]:
    status = str(item.get("status", "passed"))
    dist = state_distribution(item, state, resolution=resolution_payload)
    row: dict[str, Any] = {
        "family": family, "resolution": resolution, "route": route, "state": state,
        "status": status, "sample_count": 32,
    }
    if dist is None:
        preflight = item.get("preflight", {})
        row.update({
            "reason": preflight.get("exception") or item.get("reason") or status,
            "failed_group_id": preflight.get("failed_group_id"),
        })
        return row
    row.update({
        "continuous_wall_median_s": dist["median"], "continuous_wall_mean_s": dist["mean"],
        "continuous_wall_std_s": dist["std"], "continuous_wall_p95_s": dist["p95"],
    })
    for stage_name, output_name in (
        ("data_seconds", "load_s"), ("graph_seconds", "graph_s"),
        ("jit_or_forward_seconds", "jit_forward_s"), ("map_build_seconds", "map_build_s"),
        ("map_apply_seconds", "map_apply_s"), ("output_seconds", "output_s"),
        ("assembly_seconds", "fvm_assembly_s"), ("linear_solve_seconds", "fvm_solve_s"),
    ):
        value = stage_value(item, state, stage_name)
        if value is not None:
            row[output_name + "_median"] = value["median"]
            row[output_name + "_mean"] = value["mean"]
            row[output_name + "_p95"] = value["p95"]
    if state == "process_cold":
        row["peak_ram_bytes"] = item.get("peak_ram_bytes")
        row["peak_device_bytes"] = item.get("peak_device_bytes")
    else:
        row["peak_ram_bytes"] = item.get("process_peak_ram_bytes")
        row["peak_device_bytes"] = (item.get("device_memory") or {}).get("peak_bytes_in_use")
    cg = item.get("cg_iterations")
    if cg:
        row["cg_iterations_median"] = cg["median"]
        row["cg_iterations_p95"] = cg["p95"]
    return row


def accuracy_rows(quality: Mapping[str, Any], timing: Mapping[str, Mapping[str, Any]], resolutions: Mapping[tuple[str, int], Mapping[str, Any]], direct: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary = quality["primary_mean_std"]
    rows = [{
        "scope": "formal_quality_128_valid_x_3_seed", "family": "p1i",
        "resolution": 1024, "route": "frozen_primary_point_global",
        "sample_count": 128, "seed_count": 3,
        "support_point_global_pct_mean": summary["support_point_global_pct"]["mean"],
        "support_point_global_pct_std": summary["support_point_global_pct"]["std"],
        "support_sample_first_pct_mean": summary["support_sample_first_pct"]["mean"],
        "support_sample_first_pct_std": summary["support_sample_first_pct"]["std"],
        "full_field_point_global_pct_mean": summary["full_point_global_pct"]["mean"],
        "full_field_point_global_pct_std": summary["full_point_global_pct"]["std"],
    }]
    for family, payload in timing.items():
        item = payload["routes"]["production_reconstruction"]["fully_cached"]
        metrics = item.get("metrics", {})
        support = metrics.get("support")
        full = metrics.get("full_field")
        if support:
            rows.append({"scope": "timing_queue_diagnostic_only", "family": family, "resolution": 1024, "route": "model_support", "sample_count": 32, **support})
        if full:
            rows.append({"scope": "timing_queue_diagnostic_only", "family": family, "resolution": 240825, "route": "production_reconstruction", "sample_count": 32, **full})
    for (family, resolution), payload in sorted(resolutions.items()):
        for route in ("production_reconstruction", "fvm"):
            item = payload["routes"][route].get("fully_cached", {})
            metrics = item.get("metrics")
            if metrics:
                rows.append({"scope": "timing_queue_resolution_diagnostic_only", "family": family, "resolution": resolution, "route": route, "sample_count": 32, **metrics})
    for (family, resolution), payload in sorted(direct.items()):
        item = payload["states"].get("fully_cached", {})
        metrics = item.get("metrics") if item.get("status") == "passed" else None
        if metrics:
            rows.append({"scope": "structured_support_OOD_diagnostic_only", "family": family, "resolution": resolution, "route": "direct_model", "sample_count": 32, **metrics})
    return rows


def line_svg(path: Path, rows: Sequence[Mapping[str, Any]], *, title: str, value_key: str, route_filter: str | None = None) -> None:
    selected = [row for row in rows if row.get(value_key) not in (None, "") and (route_filter is None or row.get("route") == route_filter)]
    width, height = 940, 520; left, right, top, bottom = 90, 900, 55, 445
    values = [float(row[value_key]) for row in selected]
    positive = [value for value in values if value > 0]
    low, high = (min(positive), max(positive)) if positive else (1.0, 10.0)
    if math.isclose(low, high): high = low * 1.1
    logs = (math.log10(low), math.log10(high))
    def x(node: int) -> float:
        return left + (math.log2(node / 4096) / math.log2(240825 / 4096)) * (right - left)
    def y(value: float) -> float:
        return bottom - (math.log10(value) - logs[0]) / max(logs[1] - logs[0], 1e-12) * (bottom - top)
    colors = {
        ("p1i", "production_reconstruction"): "#1665D8",
        ("p1i", "fvm"): "#238B45",
        ("p1i", "direct_model"): "#00A6A6",
        ("randomblock", "production_reconstruction"): "#D65A31",
        ("randomblock", "fvm"): "#7A5195",
        ("randomblock", "direct_model"): "#D4A72C",
    }
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>', f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>']
    groups = sorted({(str(row["family"]), str(row["route"])) for row in selected})
    for group in groups:
        group_rows = sorted([row for row in selected if (row["family"], row["route"]) == group], key=lambda row: int(row["resolution"]))
        if not group_rows: continue
        points = " ".join(f'{x(int(row["resolution"])):.1f},{y(float(row[value_key])):.1f}' for row in group_rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[group]}" stroke-width="2"/>')
        for row in group_rows:
            lines.append(f'<circle cx="{x(int(row["resolution"])):.1f}" cy="{y(float(row[value_key])):.1f}" r="4" fill="{colors[group]}"/>')
    for index, group in enumerate(groups):
        legend_x = left + 8 + (index % 3) * 245; legend_y = top + 14 + (index // 3) * 18
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+20}" y2="{legend_y}" stroke="{colors[group]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+25}" y="{legend_y+4}" font-family="sans-serif" font-size="10">{group[0]}:{group[1]}</text>')
    for node in RESOLUTIONS:
        lines.append(f'<text x="{x(node):.1f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="10">{node}</text>')
    lines += [f'<text x="{(left+right)/2}" y="495" text-anchor="middle" font-family="sans-serif" font-size="12">node count</text>', f'<text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle" font-family="sans-serif" font-size="12">{value_key} (log)</text>', '</svg>']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def accuracy_latency_svg(path: Path, accuracy: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]]) -> None:
    latency = {
        (row["family"], int(row["resolution"]), row["route"]): float(row["continuous_wall_median_s"])
        for row in timing_rows
        if row["state"] == "fully_cached" and row.get("continuous_wall_median_s")
    }
    points = []
    for row in accuracy:
        if row["route"] == "direct_model":
            continue
        metric = row.get("point_global_true_rms_relative_rmse_pct")
        key = (row["family"], int(row["resolution"]), row["route"])
        if metric is not None and key in latency:
            points.append((row["family"], row["route"], int(row["resolution"]), latency[key], float(metric)))
    width, height = 940, 520; left, right, top, bottom = 90, 900, 55, 445
    xs = [math.log10(item[3]) for item in points]; ys = [item[4] for item in points]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if math.isclose(x0, x1): x1 = x0 + 1.0
    if math.isclose(y0, y1): y1 = y0 + 1.0
    def x(value: float) -> float: return left + (math.log10(value) - x0) / (x1 - x0) * (right - left)
    def y(value: float) -> float: return bottom - (value - y0) / (y1 - y0) * (bottom - top)
    colors = {("p1i", "production_reconstruction"): "#1665D8", ("p1i", "fvm"): "#238B45", ("p1i", "direct_model"): "#00A6A6", ("randomblock", "production_reconstruction"): "#D65A31", ("randomblock", "fvm"): "#7A5195", ("randomblock", "direct_model"): "#D4A72C"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Accuracy-latency diagnostic (32 valid cases)</text>', f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>']
    for family, route, node, latency_s, error in points:
        color = colors[(family, route)]
        lines.append(f'<circle cx="{x(latency_s):.1f}" cy="{y(error):.1f}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{x(latency_s)+6:.1f}" y="{y(error)-5:.1f}" font-family="sans-serif" font-size="9">{family}:{route[:3]}:{node}</text>')
    lines += [f'<text x="{(left+right)/2}" y="495" text-anchor="middle" font-family="sans-serif" font-size="12">fully-cached continuous wall latency, seconds (log)</text>', f'<text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle" font-family="sans-serif" font-size="12">point-global true-RMS relative RMSE (%)</text>', '</svg>']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value in (None, "") else f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--p1i-timing", type=Path, required=True)
    parser.add_argument("--randomblock-timing", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, nargs="+", required=True)
    parser.add_argument("--direct-resolution", type=Path, nargs="+", required=True)
    parser.add_argument("--historical-audit", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--direct-execution-commit", required=True)
    args = parser.parse_args()
    quality = read(args.quality)
    timing = {"p1i": read(args.p1i_timing), "randomblock": read(args.randomblock_timing)}
    resolution_payloads = [read(path) for path in args.resolution]
    direct_payloads = [read(path) for path in args.direct_resolution]
    historical = read(args.historical_audit)
    resolutions = {(item["family"], int(item["resolution"])): item for item in resolution_payloads}
    direct = {(item["family"], int(item["resolution"])): item for item in direct_payloads}
    expected = {(family, node) for family in timing for node in RESOLUTIONS}
    if set(resolutions) != expected:
        raise RuntimeError(f"resolution coverage mismatch: {sorted(expected - set(resolutions))}")
    if set(direct) != expected:
        raise RuntimeError(f"direct-resolution coverage mismatch: {sorted(expected - set(direct))}")
    for family, payload in timing.items():
        selected = payload["sample_ids"]
        if len(selected) != 32 or len(set(selected)) != 32:
            raise RuntimeError(f"{family}: timing sample queue drift")
        for node in RESOLUTIONS:
            for candidate in (resolutions[(family, node)], direct[(family, node)]):
                if candidate["sample_ids"] != selected:
                    raise RuntimeError(f"{family} N={node}: sample queue mismatch")
                contract = candidate["contract"]
                if not contract["valid_only"] or contract["test_accessed"] or contract["sealed_accessed"]:
                    raise RuntimeError(f"{family} N={node}: closed-role contract drift")
    timing_rows: list[dict[str, Any]] = []
    for family, payload in timing.items():
        for route, route_payload in payload["routes"].items():
            for state in STATES:
                timing_rows.append(timing_row(family=family, resolution=1024 if route != "fvm" else 240825, route=route, state=state, item=route_payload[state], resolution_payload=False))
    resolution_rows: list[dict[str, Any]] = []
    for (family, node), payload in sorted(resolutions.items()):
        for route, route_payload in payload["routes"].items():
            for state in RESOLUTION_STATES:
                resolution_rows.append(timing_row(family=family, resolution=node, route=route, state=state, item=route_payload[state], resolution_payload=True))
    for (family, node), payload in sorted(direct.items()):
        direct_states = dict(payload["states"])
        direct_states["new_topology"] = payload["new_topology"]
        for state in RESOLUTION_STATES:
            resolution_rows.append(timing_row(family=family, resolution=node, route="direct_model", state=state, item=direct_states[state], resolution_payload=True))
    for family in timing:
        for node in RESOLUTIONS:
            for state in RESOLUTION_STATES:
                model = next(row for row in resolution_rows if row["family"] == family and row["resolution"] == node and row["route"] == "production_reconstruction" and row["state"] == state)
                fvm = next(row for row in resolution_rows if row["family"] == family and row["resolution"] == node and row["route"] == "fvm" and row["state"] == state)
                if model.get("continuous_wall_median_s") and fvm.get("continuous_wall_median_s"):
                    speedup = float(fvm["continuous_wall_median_s"]) / float(model["continuous_wall_median_s"])
                    model["gpu_model_over_cpu_fvm_speedup"] = speedup
                    fvm["gpu_model_over_cpu_fvm_speedup"] = speedup
    accuracy = accuracy_rows(quality, timing, resolutions, direct)
    first_over = {}
    for family in timing:
        candidates = [row for row in resolution_rows if row["family"] == family and row["route"] == "production_reconstruction" and row["state"] == "fully_cached" and float(row.get("gpu_model_over_cpu_fvm_speedup", 0.0)) > 1.0]
        first_over[family] = min((int(row["resolution"]) for row in candidates), default=None)
    first_over_by_state = {}
    for family in timing:
        first_over_by_state[family] = {}
        for state in RESOLUTION_STATES:
            candidates = [row for row in resolution_rows if row["family"] == family and row["route"] == "production_reconstruction" and row["state"] == state and float(row.get("gpu_model_over_cpu_fvm_speedup", 0.0)) > 1.0]
            first_over_by_state[family][state] = min((int(row["resolution"]) for row in candidates), default=None)
    inputs = [args.quality, args.p1i_timing, args.randomblock_timing, args.historical_audit, ENVIRONMENT, *args.resolution, *args.direct_resolution]
    closeout = {
        "schema_version": "heat3d_v6_unified_performance_closeout_v1", "status": "passed",
        "execution_commit": args.execution_commit,
        "direct_execution_commit": args.direct_execution_commit,
        "closeout_commit_parent": git_head(),
        "formal_quality": quality["primary_mean_std"],
        "formal_quality_contract": {"sample_count": 128, "seed_count": 3, "primary": "support_point_global", "secondary": "support_sample_first", "full_field": "point_global", "timing_queue_replaces_formal_quality": False},
        "timing_contract": {"host": "same_WSL2", "model_device": "RTX_5070_GPU", "fvm_device": "Ryzen_9700X_CPU", "batch_size": 1, "fixed_cpu_threads": 1, "sample_count": 32, "continuous_wall_clock_primary": True, "minimum_repeats": 20, "production_excludes_sha_metrics_oracle_json_checker": True},
        "environment": read(ENVIRONMENT),
        "timing_rows": timing_rows, "resolution_rows": resolution_rows, "accuracy_rows": accuracy,
        "first_resolution_speedup_over_1x": first_over,
        "first_resolution_speedup_over_1x_by_state": first_over_by_state,
        "randomblock_governance": {"checkpoint": "V6_03_layer_checkpoint", "runtime_only_structured_support_OOD_diagnostic": True, "formal_accuracy_failed_pct": 108.34012027547357, "production_acceleration_claim_allowed": False},
        "direct_model_governance": {"route": "direct_N_structured_support_model", "diagnostic_only": True, "production_speedup_claim_allowed": False, "checkpoint_or_model_modified": False},
        "execution_bindings": {
            family: {
                "dataset": payload["dataset"], "checkpoint": payload["checkpoint"],
                "command_count": len(payload["commands"]),
                "raw_command_source": repo_path(args.p1i_timing if family == "p1i" else args.randomblock_timing),
                "sample_ids_sha256": hashlib.sha256(json.dumps(payload["sample_ids"], separators=(",", ":")).encode()).hexdigest(),
            }
            for family, payload in timing.items()
        },
        "historical_layer_audit": historical,
        "historical_values_directly_comparable": False,
        "input_artifacts": [{"path": repo_path(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in inputs],
        "benchmark_sources": {path.name: sha256(path) for path in (ROOT / "scripts/benchmark_heat3d_v6_inference_qualification.py", ROOT / "scripts/benchmark_heat3d_v6_unified_resolution.py", ROOT / "scripts/benchmark_heat3d_v6_direct_resolution.py")},
        "closeout_sources": {path.name: sha256(path) for path in (Path(__file__).resolve(), ROOT / "scripts/check_heat3d_v6_unified_performance.py")},
        "accessed_roles": ["train_frozen_normalization_metadata", "valid_iid"], "test_accessed": False, "sealed_accessed": False, "training_executed": False, "tuning_executed": False,
    }
    CONFIG.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    closeout_path = CONFIG / "v6_unified_performance_closeout.json"
    closeout_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(CONFIG / "v6_unified_performance_timing.csv", timing_rows)
    write_csv(CONFIG / "v6_unified_performance_resolution.csv", resolution_rows)
    write_csv(CONFIG / "v6_unified_performance_accuracy.csv", accuracy)
    write_csv(CONFIG / "v6_unified_historical_layer_audit.csv", historical)
    (CONFIG / "v6_unified_historical_layer_audit.json").write_text(json.dumps(historical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_rows = [row for row in resolution_rows if row["state"] == "fully_cached"]
    line_svg(DOCS / "v6_unified_latency.svg", plot_rows, title="V6 unified latency", value_key="continuous_wall_median_s")
    line_svg(DOCS / "v6_unified_speedup.svg", [row for row in plot_rows if row["route"] == "production_reconstruction"], title="GPU model / CPU FVM speedup", value_key="gpu_model_over_cpu_fvm_speedup")
    line_svg(DOCS / "v6_unified_memory.svg", plot_rows, title="V6 peak memory", value_key="peak_ram_bytes")
    line_svg(DOCS / "v6_unified_cg.svg", [row for row in plot_rows if row["route"] == "fvm"], title="FVM CG iterations", value_key="cg_iterations_median")
    accuracy_latency_svg(DOCS / "v6_unified_accuracy_latency.svg", accuracy, resolution_rows)
    formal = quality["primary_mean_std"]
    p1i_cached = [row for row in resolution_rows if row["family"] == "p1i" and row["state"] == "fully_cached"]
    p1i_accuracy = {(int(row["resolution"]), row["route"]): row for row in accuracy if row["family"] == "p1i" and row["scope"] == "timing_queue_resolution_diagnostic_only"}
    resolution_table = [
        "| N | model cold (s) | FVM cold (s) | cold speedup | model cached (s) | FVM cached (s) | cached speedup | model PG (%) | FVM PG (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for node in RESOLUTIONS:
        model = next(row for row in p1i_cached if row["resolution"] == node and row["route"] == "production_reconstruction")
        fvm = next(row for row in p1i_cached if row["resolution"] == node and row["route"] == "fvm")
        model_cold = next(row for row in resolution_rows if row["family"] == "p1i" and row["resolution"] == node and row["route"] == "production_reconstruction" and row["state"] == "process_cold")
        fvm_cold = next(row for row in resolution_rows if row["family"] == "p1i" and row["resolution"] == node and row["route"] == "fvm" and row["state"] == "process_cold")
        resolution_table.append(
            f"| {node} | {fmt(model_cold.get('continuous_wall_median_s'), 4)} | {fmt(fvm_cold.get('continuous_wall_median_s'), 4)} | {fmt(model_cold.get('gpu_model_over_cpu_fvm_speedup'), 2)}× | {fmt(model.get('continuous_wall_median_s'), 6)} | {fmt(fvm.get('continuous_wall_median_s'), 6)} | {fmt(model.get('gpu_model_over_cpu_fvm_speedup'), 2)}× | {fmt(p1i_accuracy[(node, 'production_reconstruction')].get('point_global_true_rms_relative_rmse_pct'), 3)} | {fmt(p1i_accuracy[(node, 'fvm')].get('point_global_true_rms_relative_rmse_pct'), 3)} |"
        )
    state_table = [
        "| route | process cold (s) | new topology/JIT-cached (s) | known topology/new physics (s) | fully cached (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    base_rows = {(row["route"], row["state"]): row for row in timing_rows if row["family"] == "p1i"}
    for route in ("model_support", "production_reconstruction", "fvm"):
        state_table.append(
            f"| {route} | {fmt(base_rows[(route, 'process_cold')].get('continuous_wall_median_s'))} | {fmt(base_rows[(route, 'jit_cached_new_topology')].get('continuous_wall_median_s'))} | {fmt(base_rows[(route, 'known_topology_new_physics')].get('continuous_wall_median_s'))} | {fmt(base_rows[(route, 'fully_cached')].get('continuous_wall_median_s'))} |"
        )
    direct_accuracy = {(row["family"], int(row["resolution"])): row for row in accuracy if row["route"] == "direct_model"}
    direct_table = [
        "| family | N | cold status/median (s) | cached status/median (s) | point-global (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("p1i", "randomblock"):
        for node in RESOLUTIONS:
            cold = next(row for row in resolution_rows if row["family"] == family and row["resolution"] == node and row["route"] == "direct_model" and row["state"] == "process_cold")
            cached = next(row for row in resolution_rows if row["family"] == family and row["resolution"] == node and row["route"] == "direct_model" and row["state"] == "fully_cached")
            metric = direct_accuracy.get((family, node), {})
            direct_table.append(
                f"| {family} | {node} | {cold['status']} / {fmt(cold.get('continuous_wall_median_s'))} | {cached['status']} / {fmt(cached.get('continuous_wall_median_s'))} | {fmt(metric.get('point_global_true_rms_relative_rmse_pct'), 3)} |"
            )
    p1i_direct_under_20 = [node for node in RESOLUTIONS if float(direct_accuracy.get(("p1i", node), {}).get("point_global_true_rms_relative_rmse_pct", float("inf"))) < 20.0]
    lines = [
        "# V6 unified performance benchmark", "",
        "本报告不重训、不调参，仅访问 valid_iid；test/sealed 保持关闭。P1i 代表连续物理参数的 V6 layered family，random-block 为跨数据集运行时 OOD 诊断。正式质量仍来自 128 valid × 3 seed，32 样本只用于计时队列与分辨率诊断。", "",
        f"生产/FVM执行 commit：`{args.execution_commit}`；direct-N 诊断执行 commit：`{args.direct_execution_commit}`。", "",
        "## Frozen quality", "",
        f"- support point-global: {formal['support_point_global_pct']['mean']:.6f} ± {formal['support_point_global_pct']['std']:.6f}%",
        f"- support sample-first: {formal['support_sample_first_pct']['mean']:.6f} ± {formal['support_sample_first_pct']['std']:.6f}%",
        f"- full-field point-global: {formal['full_point_global_pct']['mean']:.6f} ± {formal['full_point_global_pct']['std']:.6f}%", "",
        "上述正式精度来自 128 个 valid_iid × 3 seeds；以下 32 样本只用于统一计时队列和分辨率诊断，不能替代正式精度。", "",
        "## Four-state timing", "", *state_table, "",
        "表内是连续 wall-clock 的逐样本中位数；N/A 表示冻结数值合同下该状态不可定义，并非零耗时。process cold 包含独立进程启动至预测序列化完成；fully cached 是持久进程内图/JIT/重建映射均缓存的重复推理。", "",
        "## P1i resolution, accuracy, and runtime", "", *resolution_table, "",
        "P1i 在 fully-cached system-level 口径下从 4096 节点首次超过 1×；process-cold 在已测全部分辨率均未超过 1×。这是相同节点数但非同 DOF 布置、且精度不同的系统级比较；不能称为 matched-accuracy speedup。", "",
        "## Direct-N structured-support OOD diagnostic", "", *direct_table, "",
        "该表只验证 checkpoint 在直接 N 节点结构化支撑上的运行兼容性。它不属于冻结生产路线，不参与 speedup、模型或分辨率选择；失败/OOM 也是正式诊断结果。", "",
        f"P1i direct-N 低于 20% 的诊断分辨率为 {p1i_direct_under_20}；该路线随 N 明显非单调，240825 再次失效，因此不能据局部低误差点宣称可泛化的高分辨率直接推理。", "",
        "## Governance", "",
        "Cold 在预测序列化完成时截止；SHA、指标、oracle、JSON 与 checker 均位于生产计时之外。连续 wall-clock 是主口径，阶段计时只用于归因，禁止相加替代。", "",
        "random-block 使用 V6_03 layer checkpoint，仅为 runtime-only structured-support OOD diagnostic；其约 108% point-global 失败结果禁止形成生产加速结论。所有 direct-N 模型结果同样只作 structured-support OOD compatibility diagnostic，不进入生产 speedup 或模型/分辨率选择。", "",
        "random-block 的 4096/8192/16384/32768 structured-FVM 对完整 32 样本队列均因至少一个几何块解析不足而 fail-closed；只有 65536 与 240825 可运行。其首次 >1× 仅是运行时诊断，不是生产结论。", "",
        "历史 layer 数字均绑定其原数据集、样本、硬件、batch、预热、重复次数、计时边界与求解器定义；没有一项可按名称直接并入本轮统一表。旧 4.86× 仅称 cached steady-state speedup。", "",
        "模型路线使用冻结 1024 source-aware support 推理，再重建/输出到 N 节点；FVM 使用准确 N 节点的合法结构化网格。节点数匹配但 DOF 放置不同。", "",
        f"首次 fully-cached system speedup >1×：P1i={first_over['p1i']}，random-block={first_over['randomblock']}。", "",
        "完整 median/mean/std/P95、阶段耗时、RAM/GPU 显存、CG iterations 与精度见 `configs/heat3d_v6_p1i/v6_unified_performance_timing.csv`、`v6_unified_performance_resolution.csv` 和 `v6_unified_performance_accuracy.csv`。原始执行 JSON 位于 `configs/heat3d_v6_p1i/v6_unified_raw/`。", "",
    ]
    (DOCS / "v6_unified_performance_closeout.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed", "timing_rows": len(timing_rows), "resolution_rows": len(resolution_rows), "accuracy_rows": len(accuracy), "first_over_1x": first_over}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
