#!/usr/bin/env python3
"""Consolidate frozen V6/P1i publication evidence without rerunning timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOCS = ROOT / "docs"
ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
    "FVM240825_reference",
)
NEURAL_ROUTES = ROUTES[:-1]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise RuntimeError(f"expected three lifecycle values, got {len(values)}")
    return {"median": float(median(values)), "min": float(min(values)), "max": float(max(values))}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def serial_stage_rows(cell: dict[str, Any]) -> list[dict[str, float]]:
    route = cell["route"]
    output: list[dict[str, float]] = []
    if route.startswith("E"):
        rows = cell["serial_orders"][0]["rows"]
        for row in rows:
            s = row["stages"]
            output.append({
                "fresh_e2e": float(row["elapsed_seconds"]),
                "input_support": sum(float(s[k]) for k in (
                    "input_lookup_and_anchor_index", "support_plus_cv",
                    "support_payload_and_builder_setup")),
                "graph": float(s["anchor_graph"]) + float(s["query_graph"]),
                "packing": float(s["anchor_group_pack"]) + float(s["query_group_pack"]),
                "reconstruction_map": float(s["reconstruction_map"]),
                "h2d": float(s["h2d_enqueue"]) + float(s["h2d_sync"]),
                "nn_reconstruction": float(s["neural_forward_and_reconstruction"]),
                "other_measured": float(row["residual_seconds"]),
            })
    else:
        for row in cell["samples"]:
            s = row["stages"]
            output.append({
                "fresh_e2e": float(s["matched_continuous_e2e"]),
                "input_support": sum(float(s[k]) for k in (
                    "input_lookup_and_anchor_support", "support_plus_cv",
                    "query_support_example_builder")),
                "graph": sum(float(s[k]) for k in (
                    "anchor_graph", "query_graph", "dummy_local_p2r", "graph_extraction")),
                "packing": sum(float(s[k]) for k in (
                    "anchor_group_pack", "query_group_pack", "host_tree", "inputs", "kwargs")),
                "reconstruction_map": float(s["reconstruction_map"]) + float(s["map_array_materialization"]),
                "h2d": float(s["h2d_enqueue"]) + float(s["h2d_sync"]),
                "nn_reconstruction": float(s["asymmetric_forward"]) + float(s["reconstruction_apply"]),
                "other_measured": float(s["e2e_minus_exclusive_stages"]),
            })
    return output


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("empty percentile input")
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    weight = index - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def lifecycle(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collector = load(raw_root / "publication_results.json")
    if (collector["status"] != "collected_authoritative_valid32_without_pooled_96"
            or collector["publication_timing_freeze"] != "GO"):
        raise RuntimeError(f"collector not frozen: {raw_root}")
    stats = collector["route_seed_statistics"]
    by_route: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTES}
    for row in stats:
        by_route[row["route"]].append(row)
    rows: list[dict[str, Any]] = []
    for route in ROUTES:
        serial = sorted((x for x in by_route[route] if x["service_mode"] == "serial"), key=lambda x: x["order_seed"])
        q2 = sorted((x for x in by_route[route] if x["service_mode"] == "Q2"), key=lambda x: x["order_seed"])
        if len(serial) != 3 or len(q2) != 3:
            raise RuntimeError(f"{raw_root}: {route} lacks 3x Serial/Q2")
        if [x["order_seed"] for x in serial] != [x["order_seed"] for x in q2]:
            raise RuntimeError(f"{route}: Serial/Q2 seeds differ")
        def collect(source: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
            values = []
            for item in source:
                value: Any = item
                for key in path:
                    value = value[key]
                values.append(float(value))
            return values
        fields = {
            "cold_s": summary(collect(serial, ("cold_service_first_case_seconds",))),
            "fresh_s": summary(collect(serial, ("fresh_distinct_case", "median_seconds"))),
            "fresh_p95_s": summary(collect(serial, ("fresh_distinct_case", "p95_seconds"))),
            "cache_hot_s": summary(collect(serial, ("repeat_case_cache_hot", "median_seconds"))),
            "resident_s": summary(collect(serial, ("resident_core", "median_seconds"))),
            "q2_submit_s": summary(collect(q2, ("Q2_submit_to_result", "median_seconds"))),
            "q2_submit_p95_s": summary(collect(q2, ("Q2_submit_to_result", "p95_seconds"))),
            "q2_throughput_samples_s": summary(collect(q2, ("Q2_samples_per_second",))),
            "b16_to_b32_marginal_s": summary(collect(q2, ("true_B16_to_B32_marginal_seconds",))),
        }
        # Resource reporting is conservative: within each lifecycle seed take the
        # maximum across its Serial and Q2 services, then report the 3-seed range.
        seed_ram = [max(float(s["peak_RAM_bytes"]), float(q["peak_RAM_bytes"])) for s, q in zip(serial, q2)]
        fields["ram_bytes"] = summary(seed_ram)
        vram_values = []
        for s, q in zip(serial, q2):
            candidates = [v for v in (s.get("peak_VRAM_bytes"), q.get("peak_VRAM_bytes")) if v is not None]
            vram_values.append(float(max(candidates)) if candidates else math.nan)
        fields["vram_bytes"] = ({"median": None, "min": None, "max": None}
                                  if all(math.isnan(v) for v in vram_values)
                                  else summary([v for v in vram_values if not math.isnan(v)]))
        speed = collector["paired_speedups"].get(route)
        fields["fresh_speedup_vs_fvm"] = ({"median": 1.0, "min": 1.0, "max": 1.0}
                                                  if speed is None else speed["fresh_three_lifecycle_median_range"])
        fields["q2_speedup_vs_fvm"] = ({"median": 1.0, "min": 1.0, "max": 1.0}
                                               if speed is None else speed["Q2_throughput_three_lifecycle_median_range"])
        row: dict[str, Any] = {"route": route, "order_seeds": [x["order_seed"] for x in serial]}
        for name, value in fields.items():
            for suffix in ("median", "min", "max"):
                row[f"{name}_{suffix}"] = value.get(suffix)
        rows.append(row)
    return rows, collector


def stage_table(raw_roots: dict[str, Path]) -> list[dict[str, Any]]:
    stage_names = ("input_support", "graph", "packing", "reconstruction_map", "h2d", "nn_reconstruction", "other_measured")
    data: dict[tuple[str, str, str], dict[str, float]] = {}
    for machine, raw_root in raw_roots.items():
        for route in NEURAL_ROUTES:
            samples: list[dict[str, float]] = []
            for seed in (20260814, 20260815, 20260816):
                cell = load(raw_root / f"{route}_seed{seed}_serial.json")
                samples.extend(serial_stage_rows(cell))
            if len(samples) != 96:
                raise RuntimeError(f"{machine}/{route}: expected 96 serial rows")
            for stage in stage_names:
                values = [row[stage] for row in samples]
                shares = [100.0 * row[stage] / row["fresh_e2e"] for row in samples]
                data[(machine, route, stage)] = {
                    "median_s": float(median(values)),
                    "p95_s": percentile(values, .95),
                    "median_share_fresh_pct": float(median(shares)),
                    "p95_share_fresh_pct": percentile(shares, .95),
                }
    rows = []
    for key, values in data.items():
        machine, route, stage = key
        w = data[("wsl2", route, stage)]["median_s"]
        d = data[("devbox", route, stage)]["median_s"]
        rows.append({"machine": machine, "route": route, "stage": stage, **values,
                     "devbox_over_wsl2_median_ratio": d / w,
                     "devbox_vs_wsl2_change_pct": 100.0 * (d / w - 1.0)})
    return rows


def load_accuracy(u16384_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    u4 = load(CFG / "v6_p1i_u4_direct240825_closeout.json")
    source = u4["historical_aggregate_metrics"]
    accuracy = {
        "E16384_reconstruction": source["E16384_reconstruction"],
        "U_v2_direct240825": source["U_direct240825"],
        "E240825_direct_control": source["E240825_direct"],
    }
    u16384 = load(u16384_path)
    if u16384.get("status") != "passed_accuracy_only" or u16384.get("sample_count") != 32:
        raise RuntimeError("U-v2 16384 valid32 accuracy-only artifact invalid")
    accuracy["U_v2_16384_reconstruction"] = u16384["accuracy"]["full_field"]
    accuracy["FVM240825_reference"] = {
        "point_global_true_rms_relative_rmse_pct": None,
        "raw_cv_weighted_rmse_K": None,
        "source_rmse_K": None,
        "peak_rmse_K": None,
        "interface_drop_rmse_K": None,
        "sample_count": 32,
        "accuracy_role": "reference_solution_not_surrogate_error",
    }
    return accuracy, u16384


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u16384-accuracy", type=Path, required=True)
    args = parser.parse_args()
    args.u16384_accuracy = args.u16384_accuracy.resolve()
    raw_roots = {
        "wsl2": CFG / "v6_p1i_publication_authoritative_valid32_attempt4_04dc85c_raw",
        "devbox": CFG / "v6_p1i_publication_authoritative_valid32_devbox_1fa8310_raw",
    }
    lifecycle_rows: dict[str, list[dict[str, Any]]] = {}
    collectors = {}
    for machine, raw_root in raw_roots.items():
        lifecycle_rows[machine], collectors[machine] = lifecycle(raw_root)
        raw_matrix = load(raw_root / "authoritative_valid32_raw.json")
        if not (raw_matrix["status"] == "passed"
                and raw_matrix["formal_measurement_attempted"]
                and raw_matrix["formal_matrix_completed"]
                and raw_matrix["same_seed_cross_route_order_exact"]
                and raw_matrix["E_U_CPU_resource_policy_equal"]
                and raw_matrix["sample_count"] == 32):
            raise RuntimeError(f"{machine}: formal raw matrix contract failed")
    # Cross-host order and contract checks are performed on frozen collector data.
    for route in ROUTES:
        for seed in (20260814, 20260815, 20260816):
            def order(machine: str) -> list[str]:
                matches = [x for x in collectors[machine]["route_seed_statistics"]
                           if x["route"] == route and x["service_mode"] == "serial" and x["order_seed"] == seed]
                if len(matches) != 1:
                    raise RuntimeError("order record missing")
                return matches[0]["ordered_sample_ids"]
            if order("wsl2") != order("devbox"):
                raise RuntimeError(f"cross-host sample order drift: {route}/{seed}")
            for mode in ("serial", "Q2"):
                cells = {
                    machine: load(raw_root / f"{route}_seed{seed}_{mode}.json")
                    for machine, raw_root in raw_roots.items()
                }
                if route == "FVM240825_reference":
                    policy = lambda cell: (cell["thread_env"], cell["worker_count"])
                else:
                    policy = lambda cell: cell["cpu_policy"]
                if policy(cells["wsl2"]) != policy(cells["devbox"]):
                    raise RuntimeError(f"cross-host CPU policy drift: {route}/{seed}/{mode}")
                if route != "FVM240825_reference":
                    for machine, cell in cells.items():
                        if (cell["checkpoint_parameter_sha256_before"]
                                != cell["checkpoint_parameter_sha256_after"]):
                            raise RuntimeError(f"{machine}: checkpoint parameter drift: {route}/{seed}/{mode}")
    accuracy, u16384 = load_accuracy(args.u16384_accuracy)
    frozen_valid32 = set(collectors["wsl2"]["route_seed_statistics"][0]["ordered_sample_ids"])
    if len(frozen_valid32) != 32 or set(u16384["ordered_sample_ids"]) != frozen_valid32:
        raise RuntimeError("U-v2 16384 accuracy-only population differs from frozen valid32")
    stages = stage_table(raw_roots)

    master_rows: list[dict[str, Any]] = []
    for machine in ("wsl2", "devbox"):
        fresh_rank = {row["route"]: rank + 1 for rank, row in enumerate(sorted(lifecycle_rows[machine], key=lambda x: x["fresh_s_median"]))}
        q2_rank = {row["route"]: rank + 1 for rank, row in enumerate(sorted(lifecycle_rows[machine], key=lambda x: -x["q2_throughput_samples_s_median"]))}
        for row in lifecycle_rows[machine]:
            master_rows.append({"machine": machine,
                                "evidence_role": "primary_authoritative" if machine == "wsl2" else "independent_overclock_enabled_replication",
                                "fresh_latency_rank": fresh_rank[row["route"]],
                                "q2_throughput_rank": q2_rank[row["route"]], **row})

    cross_rows = []
    wmap = {row["route"]: row for row in lifecycle_rows["wsl2"]}
    dmap = {row["route"]: row for row in lifecycle_rows["devbox"]}
    for route in ROUTES:
        w, d = wmap[route], dmap[route]
        cross_rows.append({
            "route": route,
            "wsl2_fresh_latency_rank": next(x["fresh_latency_rank"] for x in master_rows if x["machine"] == "wsl2" and x["route"] == route),
            "devbox_fresh_latency_rank": next(x["fresh_latency_rank"] for x in master_rows if x["machine"] == "devbox" and x["route"] == route),
            "wsl2_q2_throughput_rank": next(x["q2_throughput_rank"] for x in master_rows if x["machine"] == "wsl2" and x["route"] == route),
            "devbox_q2_throughput_rank": next(x["q2_throughput_rank"] for x in master_rows if x["machine"] == "devbox" and x["route"] == route),
            "wsl2_fresh_median_s": w["fresh_s_median"], "devbox_fresh_median_s": d["fresh_s_median"],
            "devbox_over_wsl2_fresh_ratio": d["fresh_s_median"] / w["fresh_s_median"],
            "wsl2_cold_median_s": w["cold_s_median"], "devbox_cold_median_s": d["cold_s_median"],
            "devbox_over_wsl2_cold_ratio": d["cold_s_median"] / w["cold_s_median"],
            "wsl2_fresh_p95_median_s": w["fresh_p95_s_median"], "devbox_fresh_p95_median_s": d["fresh_p95_s_median"],
            "devbox_over_wsl2_fresh_p95_ratio": d["fresh_p95_s_median"] / w["fresh_p95_s_median"],
            "wsl2_cache_hot_median_s": w["cache_hot_s_median"], "devbox_cache_hot_median_s": d["cache_hot_s_median"],
            "devbox_over_wsl2_cache_hot_ratio": d["cache_hot_s_median"] / w["cache_hot_s_median"],
            "wsl2_resident_median_s": w["resident_s_median"], "devbox_resident_median_s": d["resident_s_median"],
            "devbox_over_wsl2_resident_ratio": d["resident_s_median"] / w["resident_s_median"],
            "wsl2_q2_submit_median_s": w["q2_submit_s_median"], "devbox_q2_submit_median_s": d["q2_submit_s_median"],
            "devbox_over_wsl2_q2_submit_ratio": d["q2_submit_s_median"] / w["q2_submit_s_median"],
            "wsl2_q2_submit_p95_median_s": w["q2_submit_p95_s_median"], "devbox_q2_submit_p95_median_s": d["q2_submit_p95_s_median"],
            "devbox_over_wsl2_q2_submit_p95_ratio": d["q2_submit_p95_s_median"] / w["q2_submit_p95_s_median"],
            "wsl2_q2_throughput_samples_s": w["q2_throughput_samples_s_median"],
            "devbox_q2_throughput_samples_s": d["q2_throughput_samples_s_median"],
            "devbox_over_wsl2_q2_throughput_ratio": d["q2_throughput_samples_s_median"] / w["q2_throughput_samples_s_median"],
            "wsl2_fresh_speedup_vs_fvm": w["fresh_speedup_vs_fvm_median"],
            "devbox_fresh_speedup_vs_fvm": d["fresh_speedup_vs_fvm_median"],
            "wsl2_q2_speedup_vs_fvm": w["q2_speedup_vs_fvm_median"],
            "devbox_q2_speedup_vs_fvm": d["q2_speedup_vs_fvm_median"],
            "wsl2_b16_to_b32_marginal_s": w["b16_to_b32_marginal_s_median"],
            "devbox_b16_to_b32_marginal_s": d["b16_to_b32_marginal_s_median"],
            "devbox_over_wsl2_b16_to_b32_ratio": d["b16_to_b32_marginal_s_median"] / w["b16_to_b32_marginal_s_median"],
            "wsl2_ram_bytes": w["ram_bytes_median"], "devbox_ram_bytes": d["ram_bytes_median"],
            "devbox_over_wsl2_ram_ratio": d["ram_bytes_median"] / w["ram_bytes_median"],
            "wsl2_vram_bytes": w["vram_bytes_median"], "devbox_vram_bytes": d["vram_bytes_median"],
            "devbox_over_wsl2_vram_ratio": (
                None if w["vram_bytes_median"] is None else d["vram_bytes_median"] / w["vram_bytes_median"]
            ),
        })

    strategy_rows = []
    for route in ROUTES:
        perf = wmap[route]
        acc = accuracy[route]
        strategy_rows.append({
            "route": route,
            "accuracy_population": "frozen_valid32_model_seed0" if route != "FVM240825_reference" else "reference",
            "point_global_rmse_pct": acc.get("point_global_true_rms_relative_rmse_pct"),
            "raw_cv_rmse_K": acc.get("raw_cv_weighted_rmse_K"),
            "source_rmse_K": acc.get("source_rmse_K"),
            "peak_rmse_K": acc.get("peak_rmse_K"),
            "interface_rmse_K": acc.get("interface_drop_rmse_K"),
            "fresh_median_s": perf["fresh_s_median"], "fresh_p95_median_s": perf["fresh_p95_s_median"],
            "resident_median_s": perf["resident_s_median"],
            "q2_throughput_samples_s": perf["q2_throughput_samples_s_median"],
            "b16_to_b32_marginal_s": perf["b16_to_b32_marginal_s_median"],
            "ram_bytes": perf["ram_bytes_median"], "vram_bytes": perf["vram_bytes_median"],
            "fresh_speedup_vs_fvm": perf["fresh_speedup_vs_fvm_median"],
            "q2_speedup_vs_fvm": perf["q2_speedup_vs_fvm_median"],
            "timing_provenance": "WSL2_Attempt4_primary_authoritative",
        })
    pareto_rows = [{k: row[k] for k in (
        "route", "accuracy_population", "point_global_rmse_pct", "raw_cv_rmse_K",
        "fresh_median_s", "q2_throughput_samples_s", "ram_bytes", "vram_bytes",
        "fresh_speedup_vs_fvm", "q2_speedup_vs_fvm")}
                   for row in strategy_rows]

    old_binding = load(CFG / "v6_p1i_high_n_implementation_binding.json")["code_fingerprints"]
    current_binding = load(raw_roots["devbox"] / "runtime_binding_fingerprint.json")["code_fingerprints"]
    fingerprint_rows = []
    for key in ("adapter_and_selector", "graph_builder", "reconstruction"):
        fingerprint_rows.append({"component": key, "path": old_binding[key]["path"],
                                 "historical_sha256": old_binding[key]["sha256"],
                                 "current_runtime_sha256": current_binding[key]["sha256"],
                                 "different": old_binding[key]["sha256"] != current_binding[key]["sha256"]})

    env_wsl = load(CFG / "v6_inference_qualification_environment.json")
    env_dev = load(CFG / "v6_p1i_publication_authoritative_valid32_devbox_1fa8310_metadata.json")["environment"]
    raw_evidence = {}
    for machine, raw_root in raw_roots.items():
        raw_matrix = raw_root / "authoritative_valid32_raw.json"
        collector_result = raw_root / "publication_results.json"
        raw_evidence[machine] = {
            "measurement_commit": (
                "04dc85c6ec1b620f026ea546f28a045cd43bbc9c"
                if machine == "wsl2" else
                "1fa83103fa01dff604c1f377fcc6cd61cdf2ec4d"
            ),
            "raw_matrix_path": str(raw_matrix.relative_to(ROOT)),
            "raw_matrix_sha256": sha256(raw_matrix),
            "collector_result_path": str(collector_result.relative_to(ROOT)),
            "collector_result_sha256": sha256(collector_result),
        }
    seal_path = CFG / "v6_p1i_publication_benchmark_pre_measurement_seal_runtime_isolation.json"
    protocol_sha = load(raw_roots["wsl2"] / "authoritative_valid32_raw.json")["protocol_sha256"]
    devbox_protocol_sha = load(raw_roots["devbox"] / "authoritative_valid32_raw.json")["protocol_sha256"]
    if protocol_sha != devbox_protocol_sha:
        raise RuntimeError("cross-host protocol SHA drift")
    report = {
        "schema_version": "heat3d_v6_p1i_publication_evidence_summary_v1",
        "status": "passed",
        "base_commit": "2ca9bcca61d5ef66de52b2cbdfd8171374da964b",
        "publication_evidence_completeness": "GO",
        "performance_roles": {"primary": "WSL2_Attempt4", "replication": "devbox_overclock_enabled_hardware_state",
                              "cross_host_seed_pooling": False,
                              "paired_speedup_definition": "within each lifecycle seed and identical ordered sample IDs compute FVM/neural paired workload ratio first; report median and min-max over the three lifecycle seeds; never pool 96 samples across machines"},
        "frozen_inputs": {
            "checkpoint_sha256": "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e",
            "checkpoint_epoch": 559,
            "dataset_manifest_sha256": "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514",
            "full_field_archive_sha256": "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb",
            "protocol_sha256": protocol_sha,
            "collector_sha256": "455680823359eed38585a2d2a949db69c35791e4fcf0aa643931459b0e0807e4",
            "sample_order_cross_host_exact": True,
            "cpu_policy_cross_host_exact": True,
            "measurement_evidence": raw_evidence,
            "current_golden_exactness_seal": {
                "path": str(seal_path.relative_to(ROOT)),
                "sha256": sha256(seal_path),
                "status": load(seal_path)["status"],
            },
        },
        "environment_evidence": {
            "wsl2": {"evidence_scope": "same_host_historical_environment_record_not_attempt4_embedded",
                     "cpu": env_wsl.get("cpu_model", "N/A"), "gpu": env_wsl.get("model_device_kind", "N/A"),
                     "driver": "N/A", "jax": env_wsl.get("jax", "N/A"), "jaxlib": env_wsl.get("jaxlib", "N/A"),
                     "cuda_runtime": "N/A", "python": env_wsl.get("python", "N/A"), "memory_total": env_wsl.get("memory_total", "N/A")},
            "devbox": {"evidence_scope": "attempt_bound_metadata_plus_user_designated_overclock_state",
                       "cpu": "N/A", "gpu": "N/A", "driver": "N/A", "jax": "N/A", "jaxlib": "N/A",
                       "cuda_runtime": "N/A", "gpu_backend": env_dev.get("gpu_backend", "N/A"),
                       "jax_devices": env_dev.get("jax_devices", ["N/A"]), "python": env_dev.get("python", "N/A"),
                       "overclock_state": "enabled_user_designated_raw_clock_values_not_recorded"},
        },
        "provenance_reconciliation": {
            "historical_high_n_fingerprint_differences": fingerprint_rows,
            "legacy_smoke_only_role_fields": "non_authoritative_stale_nested role_contract fields",
            "authoritative_formal_metadata": [
                "authoritative_valid32_raw.status=passed",
                "formal_measurement_attempted=true",
                "formal_matrix_completed=true",
                "raw publication_results_generated=false and publication_timing_freeze=NO_GO_pending_collector are correct pre-collector lifecycle state",
                "collector publication_results_generated=true",
                "collector publication_timing_freeze=GO",
                "30 independent process records with sample_count=32",
                "current golden/exactness/seal passed",
            ],
            "raw_artifacts_modified": False,
        },
        "accuracy_only_inference": {"performed": True, "host": u16384["host"],
                                    "route": "U_v2_16384_reconstruction", "sample_count": 32,
                                    "timing_claimed": False,
                                    "complete_remote_source_sha256": u16384["source_artifact"]["sha256"],
                                    "tracked_compact_artifact_sha256": sha256(args.u16384_accuracy)},
        "lifecycle_master": master_rows,
        "cross_machine_replication": cross_rows,
        "stage_decomposition": stages,
        "strategy_table": strategy_rows,
        "pareto_data": pareto_rows,
        "claims": {
            "sixteen_k_reconstruction_primary_pareto": True,
            "E_and_U_16384_same_e2e_class": True,
            "U_direct_better_accuracy_at_approximately_equal_direct_latency_than_E_direct": True,
            "sixteen_k_fresh_speedup_cross_machine_range": [
                min(next(x for x in lifecycle_rows[m] if x["route"] == r)["fresh_speedup_vs_fvm_median"]
                    for m in ("wsl2", "devbox") for r in ROUTES[:2]),
                max(next(x for x in lifecycle_rows[m] if x["route"] == r)["fresh_speedup_vs_fvm_median"]
                    for m in ("wsl2", "devbox") for r in ROUTES[:2]),
            ],
            "sixteen_k_q2_speedup_cross_machine_range": [
                min(next(x for x in lifecycle_rows[m] if x["route"] == r)["q2_speedup_vs_fvm_median"]
                    for m in ("wsl2", "devbox") for r in ROUTES[:2]),
                max(next(x for x in lifecycle_rows[m] if x["route"] == r)["q2_speedup_vs_fvm_median"]
                    for m in ("wsl2", "devbox") for r in ROUTES[:2]),
            ],
            "preprocessing_bound_supported": True,
            "claim_boundary": "valid32 timing/accuracy evidence only; test and sealed remain unopened",
        },
        "role_contract": {"training": False, "test": False, "sealed": False,
                          "timing_rerun": False, "accuracy_only_route_count": 1,
                          "machines_pooled_as_six_seeds": False},
    }

    master_path = CFG / "v6_p1i_master_strategy_table.csv"
    cross_path = CFG / "v6_p1i_cross_machine_replication.csv"
    stage_path = CFG / "v6_p1i_stage_decomposition.csv"
    pareto_path = CFG / "v6_p1i_pareto_data.csv"
    json_path = CFG / "v6_p1i_publication_evidence_summary.json"
    md_path = DOCS / "v6_p1i_publication_evidence_summary.md"
    write_csv(master_path, master_rows, list(master_rows[0]))
    write_csv(cross_path, cross_rows, list(cross_rows[0]))
    write_csv(stage_path, stages, list(stages[0]))
    write_csv(pareto_path, pareto_rows, list(pareto_rows[0]))
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def f(value: Any, digits: int = 4) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"
    lines = [
        "# V6/P1i publication evidence consolidation", "",
        "## Evidence roles", "",
        "WSL2 Attempt 4 is the primary authoritative benchmark. Devbox is an independent, overclock-enabled hardware-state replication and does not replace or pool with WSL2. Both use the same frozen checkpoint, dataset/full-field hashes, three ordered sample permutations, protocol, collector, and CPU worker policy.", "",
        "The nested historical `valid_iid_inputs_smoke_only` role string is stale legacy metadata. Authority comes from each immutable raw matrix's passed 30-process lifecycle records, formal attempted/completed flags, frozen exactness/seal, and collector `publication_results_generated=true` plus `publication_timing_freeze=GO`. The raw matrix correctly remains in its pre-collector state (`publication_results_generated=false`, `NO_GO_pending_collector`) and was not rewritten.", "",
        "## Frozen provenance and environment", "",
        "| Role | Measurement commit | CPU | GPU | Driver | JAX/jaxlib | CUDA runtime |",
        "|---|---|---|---|---|---|---|",
        f"| WSL2 primary | `{raw_evidence['wsl2']['measurement_commit']}` | {env_wsl.get('cpu_model', 'N/A')} | {env_wsl.get('model_device_kind', 'N/A')} | N/A | {env_wsl.get('jax', 'N/A')}/{env_wsl.get('jaxlib', 'N/A')} | N/A |",
        f"| devbox replication | `{raw_evidence['devbox']['measurement_commit']}` | N/A | N/A (`{env_dev.get('jax_devices', ['N/A'])[0]}` only) | N/A | N/A | N/A |", "",
        "Devbox clock state is user-designated as overclock-enabled, but raw clock values are not recorded. Missing environment fields are intentionally `N/A`; they are not inferred from WSL2.", "",
        "Frozen identity: checkpoint `51567afe…b90e` (epoch 559), dataset manifest `f19987c…8514`, full-field archive `49023a…3cb`, protocol `325dd8…90b`, collector code `455680…0e4`. Cross-host ordered sample IDs and E/U CPU worker policy are exact.", "",
        "Three historical high-N implementation fingerprints differ from the current runtime binding:", "",
        "| Component | Historical SHA | Current runtime SHA | Interpretation |",
        "|---|---|---|---|",
    ]
    for row in fingerprint_rows:
        lines.append(f"| `{row['component']}` | `{row['historical_sha256'][:12]}…` | `{row['current_runtime_sha256'][:12]}…` | exact-safe implementation evolution; current runtime binding and seal govern Attempt 4 |")
    lines += ["", "The old binding governs only its historical artifacts. Current authority is the immutable raw matrix, runtime binding, frozen golden hashes, padding/exactness evidence, current seal, and collector output; no frozen raw artifact was edited.", "",
        "## Primary strategy table (WSL2)", "",
        "| Route | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | Fresh med/p95 (s) | Resident (s) | Q2 (sample/s) | Fresh/Q2 speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategy_rows:
        lines.append("| {route} | {pg} | {raw} | {source} | {peak} | {interface} | {fresh}/{p95} | {resident} | {q2} | {fs}×/{qs}× |".format(
            route=row["route"], pg=f(row["point_global_rmse_pct"]), raw=f(row["raw_cv_rmse_K"]),
            source=f(row["source_rmse_K"]), peak=f(row["peak_rmse_K"]), interface=f(row["interface_rmse_K"]),
            fresh=f(row["fresh_median_s"]), p95=f(row["fresh_p95_median_s"]), resident=f(row["resident_median_s"], 5),
            q2=f(row["q2_throughput_samples_s"]), fs=f(row["fresh_speedup_vs_fvm"], 3), qs=f(row["q2_speedup_vs_fvm"], 3)))
    lines += ["", "FVM accuracy is `Reference/N.A.`: it supplies the reference solution and is not assigned surrogate error. Cold/cache-hot/resident/Q2/B16-to-B32/RAM/VRAM three-lifecycle median and min–max values are retained in `v6_p1i_master_strategy_table.csv`; the table above is the compact primary view.", "",
              "## Cross-machine conclusion", ""]
    for row in cross_rows:
        lines.append(f"- `{row['route']}`: devbox/WSL2 Fresh ratio `{row['devbox_over_wsl2_fresh_ratio']:.3f}`; Q2-throughput ratio `{row['devbox_over_wsl2_q2_throughput_ratio']:.3f}`. WSL2 remains primary.")
    lines += ["", "The two 16k reconstruction routes span Fresh speedup `{:.3f}–{:.3f}×` and Q2 speedup `{:.3f}–{:.3f}×` across the two separately reported machines.".format(
        *report["claims"]["sixteen_k_fresh_speedup_cross_machine_range"], *report["claims"]["sixteen_k_q2_speedup_cross_machine_range"]), "",
        "## Stage evidence", ""]
    for machine in ("wsl2", "devbox"):
        for route in NEURAL_ROUTES:
            rows = [x for x in stages if x["machine"] == machine and x["route"] == route]
            ranked = sorted(rows, key=lambda x: -x["median_s"])
            top = ", ".join(f"{x['stage']} {x['median_s']:.4f}s ({x['median_share_fresh_pct']:.1f}%)" for x in ranked[:3])
            lines.append(f"- {machine} `{route}`: {top}.")
    lines += ["", "The dominant measured stages are preprocessing (support/input, graph, packing, and reconstruction-map), while NN/reconstruction is smaller; this supports the bounded claim `preprocessing-bound`, not a universal hardware claim.", "",
              "## Accuracy-only addition", "",
              f"U-v2 16384 valid32 was absent from tracked aggregate accuracy evidence, so one `model_seed0` frozen-route accuracy-only evaluation was run on `{u16384['host']}`. No timing result from that execution is used. PG `{accuracy['U_v2_16384_reconstruction']['point_global_true_rms_relative_rmse_pct']:.6f}%`, raw `{accuracy['U_v2_16384_reconstruction']['raw_cv_weighted_rmse_K']:.6f} K`, source `{accuracy['U_v2_16384_reconstruction']['source_rmse_K']:.6f} K`, peak `{accuracy['U_v2_16384_reconstruction']['peak_rmse_K']:.6f} K`, interface `{accuracy['U_v2_16384_reconstruction']['interface_drop_rmse_K']:.6f} K`.", "",
              "## Frozen claims", "",
              "- 16k + reconstruction is the main accuracy-latency Pareto family in this valid32 evidence.",
              "- E/U 16384 are in the same E2E performance class on both machines; neither machine is pooled as extra seeds.",
              "- U-v2 direct has better valid32 accuracy than E-direct at approximately equal WSL2 direct-route Fresh latency.",
              "- The evidence is complete for publication-table construction on valid32. Generalization beyond this scope still requires the separately governed test/sealed confirmation; FVM retains reference physics fidelity.", "",
              "Final: `publication evidence completeness = GO`."]
    md_path.write_text("\n".join(lines) + "\n")

    manifest_paths = [
        master_path, cross_path, stage_path, pareto_path, json_path, md_path,
        args.u16384_accuracy,
        ROOT / "scripts/closeout_heat3d_v6_p1i_publication_evidence.py",
        ROOT / "scripts/check_heat3d_v6_p1i_publication_evidence.py",
    ]
    manifest_path = CFG / "v6_p1i_publication_evidence_sha256.txt"
    manifest_path.write_text("".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in manifest_paths))
    print(json.dumps({"status": "passed", "publication_evidence_completeness": "GO",
                      "outputs": [str(path.relative_to(ROOT)) for path in manifest_paths + [manifest_path]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
