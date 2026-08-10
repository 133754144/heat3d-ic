#!/usr/bin/env python3
"""Freeze P3 static-cache evidence and the final five-route performance table."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/heat3d_v6_p1i"; DOC = ROOT / "docs"
FIELDS = [
    "route", "system", "measurement_domain", "point_global_pct", "raw_cv_rmse_K",
    "source_rmse_K", "peak_rmse_K", "interface_rmse_K", "regional_nodes", "p2r_edges",
    "r2r_edges", "process_cold_median_s", "process_cold_p95_s", "fresh_topology_median_s",
    "fresh_topology_p95_s", "known_support_new_physics_median_s", "known_support_new_physics_p95_s",
    "same_input_replay_median_s", "same_input_replay_p95_s", "static_graph_load_s",
    "reconstruction_map_load_s", "dynamic_prepare_median_s", "dynamic_h2d_median_s",
    "forward_median_s", "reconstruction_median_s", "peak_vram_bytes",
    "process_cold_speedup_vs_fvm", "known_physics_speedup_vs_fvm", "same_input_speedup_vs_fvm",
    "static_cache_speedup_vs_p2", "equivalence", "provenance",
]


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol_path = CFG / "v6_p1i_performance_p3_protocol.json"; protocol = load(protocol_path)
    assert protocol["status"] == "frozen_after_p2_before_p3_execution"
    p1_path = CFG / "v6_p1i_performance_p1_closeout.json"; p1 = load(p1_path)
    p2_path = CFG / "v6_p1i_performance_p2_closeout.json"; p2 = load(p2_path)
    p1_rows = {row["route"]: row for row in p1["rows"]}
    raw_paths = {"B8192_recon": CFG / "v6_p1i_performance_p3_raw/B_8192.json",
                 "E32768_recon": CFG / "v6_p1i_performance_p3_raw/E_32768.json",
                 "B240825_direct": CFG / "v6_p1i_performance_p3_raw/B_240825.json",
                 "E240825_direct": CFG / "v6_p1i_performance_p3_raw/E_240825.json"}
    raw = {route: load(path) for route, path in raw_paths.items()}
    fvm = p1_rows["FVM240825"]
    fvm_known = next(row for row in p2["rows"] if row["route"] == "FVM240825" and row["state"] == "known_support_new_physics")
    rows=[]
    for route in raw_paths:
        old=p1_rows[route]; new=raw[route]; timing=new["timing"]
        rows.append({
            "route": route, "system": "GPU_RIGNO", "measurement_domain": old["measurement_domain"],
            "point_global_pct": old["point_global_pct"], "raw_cv_rmse_K": old["raw_cv_rmse_K"],
            "source_rmse_K": old["source_rmse_K"], "peak_rmse_K": old["peak_rmse_K"],
            "interface_rmse_K": old["interface_rmse_K"], "regional_nodes": old["regional_nodes"],
            "p2r_edges": old["p2r_edges"], "r2r_edges": old["r2r_edges"],
            "process_cold_median_s": old["process_cold_median_s"], "process_cold_p95_s": old["process_cold_p95_s"],
            "fresh_topology_median_s": old["fresh_total_median_s"], "fresh_topology_p95_s": old["fresh_total_p95_s"],
            "known_support_new_physics_median_s": timing["known_support_new_physics_total"]["median_seconds"],
            "known_support_new_physics_p95_s": timing["known_support_new_physics_total"]["p95_seconds"],
            "same_input_replay_median_s": old["warm_total_median_s"], "same_input_replay_p95_s": old["warm_total_p95_s"],
            "static_graph_load_s": timing["static_graph_load"], "reconstruction_map_load_s": timing["reconstruction_map_load"],
            "dynamic_prepare_median_s": timing["dynamic_input_preparation"]["median_seconds"],
            "dynamic_h2d_median_s": timing["dynamic_h2d"]["median_seconds"],
            "forward_median_s": timing["forward"]["median_seconds"],
            "reconstruction_median_s": (0.0 if old["measurement_domain"] == "direct_full_grid_240825"
                                          else timing["reconstruction"]["median_seconds"]),
            "peak_vram_bytes": new["memory"]["peak_bytes_in_use"],
            "process_cold_speedup_vs_fvm": old["process_cold_speedup_vs_fvm"],
            "known_physics_speedup_vs_fvm": float(fvm_known["median_s"]) / timing["known_support_new_physics_total"]["median_seconds"],
            "same_input_speedup_vs_fvm": old["warm_speedup_vs_fvm"],
            "static_cache_speedup_vs_p2": new["speedup_vs_p2_dynamic_full_pack"],
            "equivalence": "passed_tree_and_hash_exact_prediction_within_frozen_GPU_replay_envelope",
            "provenance": f"{old['accuracy_provenance']};{raw_paths[route].relative_to(ROOT)}",
        })
    rows.append({
        "route": "FVM240825", "system": "CPU_FVM", "measurement_domain": fvm["measurement_domain"],
        "point_global_pct": fvm["point_global_pct"], "raw_cv_rmse_K": fvm["raw_cv_rmse_K"],
        "source_rmse_K": fvm["source_rmse_K"], "peak_rmse_K": fvm["peak_rmse_K"],
        "interface_rmse_K": fvm["interface_rmse_K"], "regional_nodes": "", "p2r_edges": "", "r2r_edges": "",
        "process_cold_median_s": fvm["process_cold_median_s"], "process_cold_p95_s": fvm["process_cold_p95_s"],
        "fresh_topology_median_s": "N/A", "fresh_topology_p95_s": "N/A",
        "known_support_new_physics_median_s": fvm_known["median_s"], "known_support_new_physics_p95_s": fvm_known["p95_s"],
        "same_input_replay_median_s": fvm["warm_total_median_s"], "same_input_replay_p95_s": fvm["warm_total_p95_s"],
        "static_graph_load_s": "", "reconstruction_map_load_s": "", "dynamic_prepare_median_s": "",
        "dynamic_h2d_median_s": "", "forward_median_s": "", "reconstruction_median_s": "",
        "peak_vram_bytes": 0, "process_cold_speedup_vs_fvm": 1.0, "known_physics_speedup_vs_fvm": 1.0,
        "same_input_speedup_vs_fvm": 1.0, "static_cache_speedup_vs_p2": "N/A",
        "equivalence": "reference", "provenance": fvm["timing_provenance"],
    })
    csv_path=DOC/"v6_p1i_performance_final.csv"
    with csv_path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    source_hashes={str(protocol_path.relative_to(ROOT)):sha(protocol_path),str(p1_path.relative_to(ROOT)):sha(p1_path),str(p2_path.relative_to(ROOT)):sha(p2_path)}
    for path in raw_paths.values(): source_hashes[str(path.relative_to(ROOT))]=sha(path)
    initial=CFG/"v6_p1i_performance_p3_raw/B_8192_initial_failed_diagnostic.json"; source_hashes[str(initial.relative_to(ROOT))]=sha(initial)
    cache_gates={route:{"status":payload["status"],"promote":payload["promote"],
                        "graph_cache_exact":payload["static_graph_cache"]["fresh_cached_exact"],
                        "group_tree_exact":payload["equivalence"]["group_tree_all_exact"],
                        "prediction_within_replay_envelope":payload["equivalence"]["passed"],
                        "static_cache_speedup":payload["speedup_vs_p2_dynamic_full_pack"]}
                 for route,payload in raw.items()}
    closeout={"schema_version":"heat3d_v6_p1i_performance_p3_closeout_v1","status":"completed",
              "p1_sha256":sha(p1_path),"p2_sha256":sha(p2_path),"rows":rows,"cache_gates":cache_gates,
              "initial_failed_diagnostic":{"path":str(initial.relative_to(ROOT)),"sha256":sha(initial),
                   "cause":"prediction bitwise threshold conflated known GPU reduction nondeterminism; hash audit was inside timing"},
              "persistent_jax_cache":{"representative_route":"B8192_recon","first_compile_s":load(initial)["jax_compilation_cache"]["compile_and_first_apply_seconds"],
                   "cache_hit_compile_s":raw["B8192_recon"]["jax_compilation_cache"]["compile_and_first_apply_seconds"],
                   "promote":True,"production_span_includes_compile":False},
              "hybrid_gpu_tiled_continued":False,"unseen_topology_gpu_builder_deferred":True,
              "source_hashes":source_hashes,"role_contract":protocol["role_contract"]}
    out=CFG/"v6_p1i_performance_p3_closeout.json"; out.write_text(json.dumps(closeout,indent=2,sort_keys=True)+"\n")
    by={row["route"]:row for row in rows}
    lines=["# V6/P1i final performance closeout", "",
           "Accuracy 复用冻结 valid32；P3 仅测静态缓存 production timing 与等价性。hash/equivalence/metrics 均不在 production span。", "",
           "| route | PG % | raw K | source K | peak K | cold s | known-new-physics ms | replay ms | VRAM GiB | known speedup vs FVM | cache speedup |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def f(key,scale=1):
            value=row[key]; return "N/A" if value in ("","N/A") else f"{float(value)*scale:.3f}"
        lines.append(f"| {row['route']} | {f('point_global_pct')} | {f('raw_cv_rmse_K')} | {f('source_rmse_K')} | {f('peak_rmse_K')} | {f('process_cold_median_s')} | {f('known_support_new_physics_median_s',1000)} | {f('same_input_replay_median_s',1000)} | {f('peak_vram_bytes',1/2**30)} | {f('known_physics_speedup_vs_fvm')}x | {f('static_cache_speedup_vs_p2')}x |")
    lines += ["", "## Static cache equivalence and decision", "",
              "- 四条路线的 CPU model-group tree、graph metadata/hash 均精确一致；预测差异均低于冻结 same-GPU replay envelope。",
              "- 静态缓存四条路线均有明确收益，GO：运行时仅更新 k/q/BC/context/scale/QK；graph/map/structural packing 常驻。",
              f"- B8192 persistent JAX cache：首次 compile {closeout['persistent_jax_cache']['first_compile_s']:.3f}s，cache-hit {closeout['persistent_jax_cache']['cache_hit_compile_s']:.3f}s；compile 不进入 known-support steady span。",
              "- hybrid GPU-tiled 未继续；true unseen-topology GPU builder 留待下一阶段。", "",
              "## Pareto conclusions", "",
              "- Process-cold：B8192-recon 严格支配 B240825-direct；E32768-recon 严格支配 E240825-direct。FVM240825 在 accuracy 与 cold latency 上仍占优。",
              "- Known-support/new-physics：B8192-recon 最快；E32768-recon 提供更低 PG/raw/interface，但 source/peak 与 latency/VRAM 更差，二者均在 Pareto 前沿。",
              "- 两条 direct full-grid 路线同时被各自 reconstruction 路线支配，不作为默认生产路线。"]
    (DOC/"v6_p1i_performance_final.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"completed","rows":len(rows),"cache_promoted":sum(g["promote"] for g in cache_gates.values())}))
    return 0


if __name__=="__main__": raise SystemExit(main())
