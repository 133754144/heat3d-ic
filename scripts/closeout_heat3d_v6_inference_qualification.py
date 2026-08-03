#!/usr/bin/env python3
"""Freeze the valid-only P1i/random-block inference qualification benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "heat3d_v6_p1i"
DOC_DIR = ROOT / "docs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timing_stat(payload: Mapping[str, Any], state: str) -> Mapping[str, Any]:
    if state == "cold":
        return payload["external_process_wall_seconds"]
    return payload["stage_timing"]["continuous_wall_seconds"]


def metric_rows(family: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cached = payload["routes"]
    sources = (
        ("model_support", "support", cached["model_support"]["fully_cached_repeat"]["metrics"]["support"]),
        ("production_reconstruction", "support", cached["production_reconstruction"]["fully_cached_repeat"]["metrics"]["support"]),
        ("production_reconstruction", "full_field", cached["production_reconstruction"]["fully_cached_repeat"]["metrics"]["full_field"]),
        ("oracle_reconstruction", "full_field", cached["production_reconstruction"]["fully_cached_repeat"]["metrics"]["oracle_reconstruction"]),
        ("dataset_consistent_fvm", "full_field", cached["fvm"]["fully_cached_repeat"]["metrics"]["full_field"]),
    )
    for route, domain, metrics in sources:
        rows.append({"family": family, "route": route, "domain": domain, **metrics})
    return rows


def timing_rows(family: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route, route_payload in payload["routes"].items():
        for state in ("cold", "jit_cached_new_case", "fully_cached_repeat"):
            item = route_payload[state]
            wall = timing_stat(item, state)
            if state == "cold":
                peak_ram = item["peak_ram_bytes"]
                peak_device = item["peak_device_bytes"]
                stages = item["stage_timing"]
            else:
                peak_ram = item["process_peak_ram_bytes"]
                peak_device = item["device_memory"]["peak_bytes_in_use"]
                stages = item["stage_timing"]
            row: dict[str, Any] = {
                "family": family,
                "route": route,
                "state": state,
                "sample_count": int(wall["count"]),
                "continuous_wall_median_s": wall["median"],
                "continuous_wall_mean_s": wall["mean"],
                "continuous_wall_std_s": wall["std"],
                "continuous_wall_p95_s": wall["p95"],
                "peak_ram_bytes": peak_ram,
                "peak_device_bytes": peak_device,
                "input_nodes": 240825 if route == "fvm" else 1024,
                "output_nodes": 240825 if route in {"production_reconstruction", "fvm"} else 1024,
                "cache_preparation_seconds_outside_timing": (
                    0.0 if state == "cold" else float(item.get("cache_preparation_seconds_outside_timing", 0.0))
                ),
            }
            for name, stats in stages.items():
                for statistic in ("median", "mean", "std", "p95"):
                    row[f"stage_{name}_{statistic}"] = stats[statistic]
            if route == "production_reconstruction":
                solver_wall = timing_stat(payload["routes"]["fvm"][state], state)
                row["dataset_consistent_fvm_over_route_speedup"] = solver_wall["median"] / wall["median"]
            rows.append(row)
    return rows


def historical_audit() -> list[dict[str, Any]]:
    return [
        {
            "benchmark_id": "p1i_old_single_sample_resolution_benchmark",
            "dataset": "heat3d_v6_p1i_continuous_physics1024_v1",
            "checkpoint": "P1i seed0 e559",
            "sample_scope": "one preregistered valid_iid sample",
            "resolutions": "1024/4096/16384/65536/240825",
            "hardware": "devbox alias; hostname XYH-Desktop; WSL2; RTX 5070",
            "model_batch": 1,
            "timing_boundary": "stage-separated steady model and reconstruction; cold derived from stages",
            "warmup": 1,
            "repetitions": 20,
            "solver_definition": "structured FVM at each listed node count; CG rtol=1e-10",
            "directly_comparable_to_qualification": False,
            "reason": "single sample and stage-derived cold boundary; only cached steady-state values retain their label",
            "evidence": "docs/v6_p1i_three_seed_inference_closeout.md;scripts/benchmark_heat3d_v6_p1i_resolution.py",
        },
        {
            "benchmark_id": "v6_layer_final_performance",
            "dataset": "heat3d_v6_p1h_shared_support1024_v0",
            "checkpoint": "V6_03 seed0 e111",
            "sample_scope": "128 valid_iid samples",
            "resolutions": "4096/8192/16384;32768 experimental",
            "hardware": "GPU WSL2 RTX 5070; CPU results collected on macOS arm64",
            "model_batch": "8/16",
            "timing_boundary": "128-sample production cycles; cold/cached/persistent are protocol-specific",
            "warmup": "script mode dependent",
            "repetitions": 3,
            "solver_definition": "240825-node FVM cold/warm; nonmatched-DOF",
            "directly_comparable_to_qualification": False,
            "reason": "different dataset, batch, host boundary, repetition count and nonmatched-DOF solver denominator",
            "evidence": "docs/v6_final_performance_closeout.md;configs/heat3d_v6/v6_final_performance_environment.json;scripts/run_heat3d_v6_final_performance.py",
        },
        {
            "benchmark_id": "v6_layer_production_inference",
            "dataset": "heat3d_v6_p1h_shared_support1024_v0",
            "checkpoint": "V6_03 seed0/1/2; timing seed0",
            "sample_scope": "128 valid_iid samples",
            "resolutions": "1024/2048/4096/8192/16384/32768",
            "hardware": "mixed local CPU and WSL2 RTX 5070",
            "model_batch": "1/8/16 depending table",
            "timing_boundary": "cached graph/reconstruction production stages; archived protocol",
            "warmup": "1 or 10 depending command",
            "repetitions": "1/10 depending command",
            "solver_definition": "240825-node FVM cold/warm; nonmatched-DOF",
            "directly_comparable_to_qualification": False,
            "reason": "mixed hardware and mode-dependent repeats; usable only within the archived protocol",
            "evidence": "docs/v6_production_inference_final_closeout.md;configs/heat3d_v6/v6_production_bundle_manifest.json;scripts/run_heat3d_v6_production_highres_inference.py",
        },
    ]


def svg_plot(path: Path, rows: list[dict[str, Any]], *, mode: str) -> None:
    width, height = 760, 430
    left, right, top, bottom = 85, 25, 35, 65
    colors = {"p1i": "#2563eb", "randomblock": "#dc2626"}
    points = []
    if mode == "accuracy_latency":
        for row in rows:
            if row["route"] == "production_reconstruction" and row["state"] == "fully_cached_repeat":
                points.append((row["family"], row["continuous_wall_median_s"], None))
        x_values = [p[1] for p in points]
        # Accuracy values are inserted by caller through sibling lookup below.
        title, xlabel, ylabel = "Valid full-field accuracy-latency", "cached wall time / sample (s)", "point-global relative RMSE (%)"
    else:
        for row in rows:
            if row["state"] == "fully_cached_repeat":
                points.append((row["family"] + ":" + row["route"], row["output_nodes"], row["continuous_wall_median_s"]))
        x_values = [float(p[1]) for p in points]
        title, xlabel, ylabel = "Resolution-time qualification", "output nodes (log scale)", "cached wall time / sample (s)"
    # Caller replaces the accuracy placeholders before this function is used.
    if not points:
        raise RuntimeError("no plot points")
    import math
    if mode == "scale_time":
        transformed_x = [math.log10(float(x)) for x in x_values]
        y_values = [float(p[2]) for p in points]
    else:
        raise RuntimeError("accuracy plot is written by svg_accuracy_plot")
    xmin, xmax = min(transformed_x), max(transformed_x)
    ymin, ymax = 0.0, max(y_values) * 1.12
    def sx(v: float) -> float: return left + (v - xmin) / max(xmax - xmin, 1e-12) * (width - left - right)
    def sy(v: float) -> float: return height - bottom - (v - ymin) / max(ymax - ymin, 1e-12) * (height - top - bottom)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="17">{title}</text>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>']
    for label, nodes, value in points:
        family = label.split(":", 1)[0]
        x, y = sx(math.log10(float(nodes))), sy(float(value))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[family]}"/>')
        body.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="sans-serif" font-size="10">{label}</text>')
    body.extend([f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">{xlabel}</text>', f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{ylabel}</text>', '</svg>'])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def svg_accuracy_plot(path: Path, timing: list[dict[str, Any]], accuracy: list[dict[str, Any]]) -> None:
    values = []
    for family in ("p1i", "randomblock"):
        t = next(row for row in timing if row["family"] == family and row["route"] == "production_reconstruction" and row["state"] == "fully_cached_repeat")
        a = next(row for row in accuracy if row["family"] == family and row["route"] == "production_reconstruction" and row["domain"] == "full_240825")
        values.append((family, float(t["continuous_wall_median_s"]), float(a["point_global_true_rms_relative_rmse_pct"])))
    width, height, left, right, top, bottom = 720, 400, 80, 30, 35, 60
    xmin, xmax = 0.0, max(x for _, x, _ in values) * 1.15
    ymin, ymax = 0.0, max(y for _, _, y in values) * 1.15
    sx = lambda x: left + (x - xmin) / max(xmax - xmin, 1e-12) * (width-left-right)
    sy = lambda y: height-bottom - (y-ymin) / max(ymax-ymin, 1e-12) * (height-top-bottom)
    colors = {"p1i": "#2563eb", "randomblock": "#dc2626"}
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="360" y="22" text-anchor="middle" font-family="sans-serif" font-size="17">Valid full-field accuracy-latency</text>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>']
    for family, x, y in values:
        body += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="{colors[family]}"/>', f'<text x="{sx(x)+10:.1f}" y="{sy(y)-8:.1f}" font-family="sans-serif" font-size="12">{family}</text>']
    body += [f'<text x="360" y="385" text-anchor="middle" font-family="sans-serif" font-size="13">cached production wall time / sample (s)</text>', '<text transform="translate(18 200) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">point-global relative RMSE (%)</text>', '</svg>']
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1i", type=Path, required=True)
    parser.add_argument("--randomblock", type=Path, required=True)
    parser.add_argument("--p1i-scope", type=Path, required=True)
    parser.add_argument("--randomblock-scope", type=Path, required=True)
    parser.add_argument("--p1i-edge-contract", type=Path, required=True)
    parser.add_argument("--randomblock-edge-contract", type=Path, required=True)
    parser.add_argument("--p1i-edge-equivalence", type=Path, required=True)
    parser.add_argument("--randomblock-padding-rejection", type=Path, required=True)
    parser.add_argument("--p1i-rejected-log", type=Path, required=True)
    parser.add_argument("--randomblock-failed-log", type=Path, required=True)
    parser.add_argument("--p1i-execution-code-sha256", required=True)
    parser.add_argument("--randomblock-execution-code-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    families = {"p1i": read_json(args.p1i), "randomblock": read_json(args.randomblock)}
    sample_scope = {"p1i": read_json(args.p1i_scope), "randomblock": read_json(args.randomblock_scope)}
    edge_contract = {"p1i": read_json(args.p1i_edge_contract), "randomblock": read_json(args.randomblock_edge_contract)}
    p1i_edge_equivalence = read_json(args.p1i_edge_equivalence)
    randomblock_padding_rejection = read_json(args.randomblock_padding_rejection)
    environment = read_json(args.environment)
    for family, payload in families.items():
        if payload.get("family") != family or payload.get("sample_count") != 32:
            raise RuntimeError(f"{family}: family/sample qualification mismatch")
        if payload.get("test_accessed") or payload.get("sealed_accessed") or payload.get("training_executed"):
            raise RuntimeError(f"{family}: forbidden role/training access")
    timing = sum((timing_rows(name, value) for name, value in families.items()), [])
    accuracy = sum((metric_rows(name, value) for name, value in families.items()), [])
    full_pg = {
        family: next(
            row["point_global_true_rms_relative_rmse_pct"] for row in accuracy
            if row["family"] == family and row["route"] == "production_reconstruction" and row["domain"] == "full_240825"
        )
        for family in families
    }
    historical = historical_audit()
    for entry in historical:
        entry["evidence_sha256"] = {
            relative: sha256(ROOT / relative)
            for relative in str(entry["evidence"]).split(";")
        }
    write_csv(CONFIG_DIR / "v6_inference_qualification_timing.csv", timing)
    write_csv(CONFIG_DIR / "v6_inference_qualification_accuracy.csv", accuracy)
    write_csv(CONFIG_DIR / "v6_inference_historical_layer_audit.csv", historical)
    (CONFIG_DIR / "v6_inference_historical_layer_audit.json").write_text(json.dumps(historical, indent=2, sort_keys=True) + "\n")
    three_seed = read_json(CONFIG_DIR / "v6_p1i_three_seed_inference_closeout.json")
    closeout = {
        "schema_version": "heat3d_v6_inference_qualification_closeout_v1",
        "status": "qualified_valid_only",
        "repository_base_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "training_commit": three_seed.get("training_commit"),
        "accepted_model_contract": {
            "primary": "point_global_true_rms_relative_rmse",
            "secondary": "sample_first_cv_relative_rmse",
            "deployment": "1024_source_aware_support_plus_layer_aware_reconstruction",
            "three_seed_mean_std": three_seed["primary_mean_std"],
        },
        "scope": {"roles": ["valid_iid", "train_frozen_normalization_metadata"], "test_accessed": False, "sealed_accessed": False, "training_executed": False, "tuning_executed": False},
        "host_contract": {"host": families["p1i"]["routes"]["model_support"]["fully_cached_repeat"]["environment"], "same_host_for_families": True, "sample_count": 32, "minimum_repetitions": 20},
        "environment": {"payload": environment, "sha256": sha256(args.environment)},
        "inputs": {
            "p1i": {"sha256": sha256(args.p1i), "dataset": families["p1i"]["dataset"], "checkpoint": families["p1i"]["checkpoint"], "commands": families["p1i"]["commands"]},
            "randomblock": {"sha256": sha256(args.randomblock), "dataset": families["randomblock"]["dataset"], "checkpoint": families["randomblock"]["checkpoint"], "commands": families["randomblock"]["commands"]},
        },
        "code": {
            "archived_benchmark_source_sha256": sha256(ROOT / "scripts/benchmark_heat3d_v6_inference_qualification.py"),
            "p1i_execution_code_sha256": args.p1i_execution_code_sha256,
            "randomblock_execution_code_sha256": args.randomblock_execution_code_sha256,
            "scope_audit_sha256": sha256(ROOT / "scripts/audit_heat3d_v6_inference_qualification_scope.py"),
        },
        "timing": timing,
        "accuracy": accuracy,
        "qualification_decision": {
            "p1i_native_deployment": "qualified" if full_pg["p1i"] < 20.0 else "failed",
            "randomblock_structured_support_ood_compatibility": "failed" if full_pg["randomblock"] >= 20.0 else "diagnostic_pass_only",
            "randomblock_is_production_claim": False,
        },
        "historical_layer_audit": historical,
        "sample_scope": sample_scope,
        "jit_shape_contract": {
            "p1i": {"contract": edge_contract["p1i"], "equivalence": p1i_edge_equivalence},
            "randomblock": {
                "contract": edge_contract["randomblock"],
                "raw_graph_unmodified": True,
                "rejected_fixed_padding_attempt": randomblock_padding_rejection,
            },
        },
        "terminology_amendment": {"old_4_864x": "cached steady-state speedup only", "route_A": "structured-support OOD compatibility diagnostic only"},
        "protocol_deviations": [
            {
                "attempt": "p1i_prequalification_without_literal_full_forward_jit",
                "status": "rejected_and_rerun_from_scratch",
                "reason": "new-case still compiled per varying graph; no result entered formal tables",
                "log_sha256": sha256(args.p1i_rejected_log),
            },
            {
                "attempt": "randomblock_fixed_global_edge_padding",
                "status": "rejected_before_formal_timing",
                "reason": "forward equivalence exceeded frozen 0.01/0.002 K limits",
                "audit_sha256": sha256(args.randomblock_padding_rejection),
            },
            {
                "attempt": "randomblock_fvm_with_p1i_mesh_builder",
                "status": "failed_closed_and_full_family_rerun",
                "reason": "schema mismatch; replaced by frozen random-block solver core",
                "log_sha256": sha256(args.randomblock_failed_log),
            },
        ],
    }
    closeout_path = CONFIG_DIR / "v6_inference_qualification_closeout.json"
    closeout_path.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    svg_accuracy_plot(DOC_DIR / "v6_inference_qualification_accuracy_latency.svg", timing, accuracy)
    svg_plot(DOC_DIR / "v6_inference_qualification_scale_time.svg", timing, mode="scale_time")
    lines = [
        "# V6 inference benchmark qualification closeout", "",
        "本轮冻结三 seed valid 结果：point-global 为 primary、sample-first 为 secondary；可信部署路线是 1024 source-aware support + layer-aware reconstruction。test/sealed 未访问，未训练或调参。", "",
        f"Fixed host: `{environment['host']}`; CPU `{environment['cpu_model']}` ({environment['logical_cpu_count']} logical CPUs, {environment['memory_total']}); model device `{environment['model_device_kind']}`; Python {environment['python'].split()[0]}, JAX/JAXlib {environment['jax']}/{environment['jaxlib']}, NumPy/SciPy {environment['numpy']}/{environment['scipy']}.", "",
        "## Corrected timing", "",
        "所有结果来自同一 WSL2 主机、32 个固定 valid 样本、B1/单线程。cold 为每样本新进程；model new-case 为 JIT 已建立但案例/图/映射未缓存，FVM 对应为新案例且未缓存系统；cached 为 model 图/JIT/重建映射或 FVM 组装系统已缓存。每项均为连续 wall-clock，阶段和不作为总时间相加。production 区间不含 oracle 或指标计算。", "",
        "| family | route | state | median / mean / std / p95 s | peak RAM GiB | peak device GiB | GPU→CPU FVM/route speedup |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in timing:
        speed = row.get("dataset_consistent_fvm_over_route_speedup")
        lines.append(f"| {row['family']} | {row['route']} | {row['state']} | {row['continuous_wall_median_s']:.4f} / {row['continuous_wall_mean_s']:.4f} / {row['continuous_wall_std_s']:.4f} / {row['continuous_wall_p95_s']:.4f} | {row['peak_ram_bytes']/2**30:.3f} | {row['peak_device_bytes']/2**30:.3f} | {speed:.3f}× |" if speed is not None else f"| {row['family']} | {row['route']} | {row['state']} | {row['continuous_wall_median_s']:.4f} / {row['continuous_wall_mean_s']:.4f} / {row['continuous_wall_std_s']:.4f} / {row['continuous_wall_p95_s']:.4f} | {row['peak_ram_bytes']/2**30:.3f} | {row['peak_device_bytes']/2**30:.3f} | — |")
    lines += ["", "## Stage decomposition (median seconds)", "", "| family | route/state | data | graph | JIT/forward | map build/apply | output | FVM assembly/solve |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in timing:
        get = lambda name: float(row.get(f"stage_{name}_median", 0.0))
        lines.append(f"| {row['family']} | {row['route']}/{row['state']} | {get('data_seconds'):.4f} | {get('graph_seconds'):.4f} | {get('jit_or_forward_seconds'):.4f} | {get('map_build_seconds'):.4f}/{get('map_apply_seconds'):.4f} | {get('output_seconds'):.4f} | {get('assembly_seconds'):.4f}/{get('linear_solve_seconds'):.4f} |")
    lines += ["", "## Accuracy", "", "| family | route/domain | point-global % | sample-first % | raw CV K | peak/source/background K |", "|---|---|---:|---:|---:|---:|"]
    for row in accuracy:
        lines.append(f"| {row['family']} | {row['route']}/{row['domain']} | {row['point_global_true_rms_relative_rmse_pct']:.4f} | {row['sample_first_cv_relative_rmse_pct']:.4f} | {row['raw_cv_weighted_rmse_K']:.4f} | {row['peak_rmse_K']:.3f}/{row['source_rmse_K']:.3f}/{row['background_rmse_K']:.3f} |")
    lines += ["", f"P1i 1024+reconstruction full-field point-global={full_pg['p1i']:.4f}%，通过 <20% 资格门；random-block OOD diagnostic={full_pg['randomblock']:.4f}%，不具生产兼容性。"]
    lines += ["", "## Historical layer audit", "", "历史数字均可在各自归档协议内复现，但没有一组可直接与本轮资格计时合并：旧 P1i 只测 1 个样本且 cold 为分段派生；layer 基准使用不同数据集、batch、主机边界、重复数或 nonmatched-DOF FVM。原 4.86× 只称 cached steady-state speedup；Route A 只称 structured-support OOD compatibility diagnostic。", "", "## Sample complexity and solver iterations", ""]
    for family in ("p1i", "randomblock"):
        scope = sample_scope[family]
        cg = scope["cg_iterations"]
        lines.append(f"- {family}: {scope['sample_count']} valid samples, unique support hashes={scope['unique_support_hashes']}, CG iterations median/P95={cg['median']:.1f}/{cg['p95']:.1f}; source-region median={scope['source_region_count']['median']:.1f}, conductivity-region median={scope['conductivity_region_count']['median']:.1f}.")
    lines += ["", "| family | support/FVM nodes | support hashes | source/k regions median | CG iterations median/P95 | new-case graph s | cached FVM solve s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in ("p1i", "randomblock"):
        scope = sample_scope[family]
        graph = next(row for row in timing if row["family"] == family and row["route"] == "model_support" and row["state"] == "jit_cached_new_case")
        solver = next(row for row in timing if row["family"] == family and row["route"] == "fvm" and row["state"] == "fully_cached_repeat")
        lines.append(f"| {family} | 1024/240825 | {scope['unique_support_hashes']} | {scope['source_region_count']['median']:.1f}/{scope['conductivity_region_count']['median']:.1f} | {scope['cg_iterations']['median']:.1f}/{scope['cg_iterations']['p95']:.1f} | {graph['stage_graph_seconds_median']:.4f} | {solver['stage_linear_solve_seconds_median']:.4f} |")
    lines += ["", "## JIT shape qualification", "", f"- P1i uses fixed dummy-edge padding. Frozen equivalence: max={p1i_edge_equivalence['max_abs_error_K']:.6f} K, RMSE={p1i_edge_equivalence['rmse_K']:.6f} K (limits 0.01/0.002 K).", f"- random-block fixed padding was rejected (max={randomblock_padding_rejection['max_abs_error_K']:.6f} K, RMSE={randomblock_padding_rejection['rmse_K']:.6f} K). Formal timing keeps raw graphs and warms the 16 preregistered support-shape families with a third valid variant before measuring two new variants per group."]
    lines += ["", "## Rejected attempts", "", "P1i 无完整 forward JIT 的初次计时、random-block 全局 edge padding、以及误用 P1i mesh builder 的 random-block FVM 均 fail closed；它们的日志/审计 SHA 保留在 machine-readable closeout，且没有数字进入正式表。"]
    lines += ["", "## Qualification", "", "本报告只判定计时和复现资格，不扩大模型适用域。P1i native support 的质量与 1024+layer-aware reconstruction 分开报告；random-block 是冻结 layer checkpoint 的跨结构 OOD 诊断，不用于调参或宣称生产泛化。", ""]
    (DOC_DIR / "v6_inference_qualification_closeout.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed", "timing_rows": len(timing), "accuracy_rows": len(accuracy)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
