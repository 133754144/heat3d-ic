#!/usr/bin/env python3
"""Build the semantically matched P2 workload matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"
DOC = ROOT / "docs"
STATES = ["virgin_process_cold", "resident_runtime_graph_rebuild",
          "known_support_new_physics", "same_input_resident_replay"]
FIELDS = ["route", "system", "state", "semantics", "sample_count", "median_s", "p95_s",
          "dynamic_prepare_median_s", "h2d_median_s", "forward_plus_reconstruction_median_s",
          "speedup_vs_semantically_matched_fvm", "comparison_status", "provenance"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol_path = CFG / "v6_p1i_performance_p2_protocol.json"
    protocol = load(protocol_path)
    assert protocol["status"] == "frozen_after_p1_before_p2_execution"
    p1_path = CFG / "v6_p1i_performance_p1_closeout.json"
    p1 = load(p1_path)
    p1_rows = {row["route"]: row for row in p1["rows"]}
    routes = ["B8192_recon", "E32768_recon", "B240825_direct", "E240825_direct"]
    raw_paths = {"B8192_recon": CFG / "v6_p1i_performance_p2_raw/B_8192.json",
                 "E32768_recon": CFG / "v6_p1i_performance_p2_raw/E_32768.json",
                 "B240825_direct": CFG / "v6_p1i_performance_p2_raw/B_240825.json",
                 "E240825_direct": CFG / "v6_p1i_performance_p2_raw/E_240825.json"}
    raw = {route: load(path) for route, path in raw_paths.items()}
    with (CFG / "v6_unified_performance_timing.csv").open(newline="") as handle:
        fvm_rows = {row["state"]: row for row in csv.DictReader(handle)
                    if row["family"] == "p1i" and row["route"] == "fvm"}
    fvm = {"virgin_process_cold": fvm_rows["process_cold"],
           "known_support_new_physics": fvm_rows["known_topology_new_physics"],
           "same_input_resident_replay": fvm_rows["fully_cached"]}
    semantics = {
        "virgin_process_cold": "independent process entry to first synchronized output",
        "resident_runtime_graph_rebuild": "resident runtime, fresh sample-varying support graph",
        "known_support_new_physics": "fixed first support/graph/map; 32 distinct groups with real per-sample k/q/BC/context/scale",
        "same_input_resident_replay": "fixed input/support/graph/JIT repeated replay",
    }
    rows = []
    for route in routes:
        old = p1_rows[route]; fresh = raw[route]
        values = {
            "virgin_process_cold": (old["process_cold_median_s"], old["process_cold_p95_s"], 1),
            "resident_runtime_graph_rebuild": (old["fresh_total_median_s"], old["fresh_total_p95_s"], 32),
            "known_support_new_physics": (fresh["timing"]["continuous_total"]["median_seconds"],
                                          fresh["timing"]["continuous_total"]["p95_seconds"], 32),
            "same_input_resident_replay": (old["warm_total_median_s"], old["warm_total_p95_s"], 20),
        }
        for state in STATES:
            median, p95, count = values[state]
            if state == "resident_runtime_graph_rebuild":
                speedup = "N/A:no_semantically_matched_FVM_state"; comparison = "not_comparable"
            else:
                reference = float(fvm[state]["continuous_wall_median_s"])
                speedup = reference / float(median); comparison = "semantically_matched"
            row = {"route": route, "system": "GPU_RIGNO", "state": state,
                   "semantics": semantics[state], "sample_count": count,
                   "median_s": median, "p95_s": p95,
                   "dynamic_prepare_median_s": "", "h2d_median_s": "",
                   "forward_plus_reconstruction_median_s": "",
                   "speedup_vs_semantically_matched_fvm": speedup,
                   "comparison_status": comparison,
                   "provenance": str(p1_path.relative_to(ROOT))}
            if state == "known_support_new_physics":
                row.update({"dynamic_prepare_median_s": fresh["timing"]["dynamic_prepare"]["median_seconds"],
                            "h2d_median_s": fresh["timing"]["h2d"]["median_seconds"],
                            "forward_plus_reconstruction_median_s": fresh["timing"]["forward_plus_reconstruction"]["median_seconds"],
                            "provenance": str(raw_paths[route].relative_to(ROOT))})
            rows.append(row)
    fvm_map = {"virgin_process_cold": "process_cold",
               "known_support_new_physics": "known_topology_new_physics",
               "same_input_resident_replay": "fully_cached"}
    for state in STATES:
        if state == "resident_runtime_graph_rebuild":
            rows.append({"route": "FVM240825", "system": "CPU_FVM", "state": state,
                         "semantics": "no FVM unseen-topology/JIT state under frozen solver contract",
                         "sample_count": 0, "median_s": "N/A", "p95_s": "N/A",
                         "dynamic_prepare_median_s": "", "h2d_median_s": "",
                         "forward_plus_reconstruction_median_s": "",
                         "speedup_vs_semantically_matched_fvm": "N/A", "comparison_status": "not_applicable",
                         "provenance": "configs/heat3d_v6_p1i/v6_unified_performance_timing.csv"})
        else:
            source = fvm_rows[fvm_map[state]]
            rows.append({"route": "FVM240825", "system": "CPU_FVM", "state": state,
                         "semantics": semantics[state], "sample_count": source["sample_count"],
                         "median_s": source["continuous_wall_median_s"], "p95_s": source["continuous_wall_p95_s"],
                         "dynamic_prepare_median_s": "", "h2d_median_s": "",
                         "forward_plus_reconstruction_median_s": "",
                         "speedup_vs_semantically_matched_fvm": 1.0, "comparison_status": "reference",
                         "provenance": "configs/heat3d_v6_p1i/v6_unified_performance_timing.csv"})

    csv_path = DOC / "v6_p1i_workload_semantics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    sources = {str(protocol_path.relative_to(ROOT)): sha(protocol_path), str(p1_path.relative_to(ROOT)): sha(p1_path)}
    for path in raw_paths.values(): sources[str(path.relative_to(ROOT))] = sha(path)
    closeout = {"schema_version": "heat3d_v6_p1i_performance_p2_closeout_v1", "status": "completed",
                "p1_frozen_sha256": sha(p1_path), "rows": rows, "sources": sources,
                "known_support_gates": {route: {"status": payload["status"],
                                                "unique_group_count": payload["unique_group_count"],
                                                "unique_physics_signature_count": payload["unique_physics_signature_count"],
                                                "graph_repeat_exact": payload["graph_repeat_exact"],
                                                "temperature_read": payload["temperature_read"],
                                                "metrics_computed": payload["metrics_computed"]}
                                        for route, payload in raw.items()},
                "role_contract": protocol["role_contract"]}
    out_json = CFG / "v6_p1i_performance_p2_closeout.json"
    out_json.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n")
    lines = ["# V6/P1i P2 benchmark semantics", "",
             "P1 accuracy 已冻结；本阶段只建立 workload 语义并补真实 known-support/new-physics timing。", "",
             "| route | cold s | graph-rebuild s | known-support/new-physics s | replay s | known-physics speedup vs FVM |",
             "|---|---:|---:|---:|---:|---:|"]
    for route in routes:
        part = {row["state"]: row for row in rows if row["route"] == route}
        lines.append(f"| {route} | {float(part['virgin_process_cold']['median_s']):.4f} | "
                     f"{float(part['resident_runtime_graph_rebuild']['median_s']):.4f} | "
                     f"{float(part['known_support_new_physics']['median_s']):.4f} | "
                     f"{float(part['same_input_resident_replay']['median_s']):.4f} | "
                     f"{float(part['known_support_new_physics']['speedup_vs_semantically_matched_fvm']):.2f}x |")
    lines += ["", "## Semantic gates", "",
              "- `known_support_new_physics` 使用 32 个不同 group；每个样本读取真实 sidecar k/q 与原 BC、anchor context/scale。",
              "- support、graph、reconstruction map 固定为第一个预注册 valid32 样本；32 个联合物理签名均唯一。",
              "- `resident_runtime_graph_rebuild` 没有语义匹配的 FVM 状态，因此 speedup 为 N/A。",
              "- temperature/metrics 不进入 P2 production timing；test/sealed 未访问。"]
    (DOC / "v6_p1i_workload_semantics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
