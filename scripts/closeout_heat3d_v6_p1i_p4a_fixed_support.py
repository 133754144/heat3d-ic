#!/usr/bin/env python3
"""Close P4-A and enforce fail-fast before P4-B."""

from __future__ import annotations

import csv, hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CFG=ROOT/"configs/heat3d_v6_p1i"; DOC=ROOT/"docs"

def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol_path=CFG/"v6_p1i_p4a_fixed_support_accuracy_protocol.json"; protocol=load(protocol_path)
    paths={route:CFG/f"v6_p1i_p4a_raw/{route}.json" for route in protocol["routes"]}
    raw={route:load(path) for route,path in paths.items()}; paired=[]; summaries=[]
    metric_keys=("full_rmse_K","source_rmse_K","peak_rmse_K","interface_rmse_K")
    for route,payload in raw.items():
        for row in payload["paired"]: paired.append({"route":route,**row})
        stats={}
        for key in metric_keys:
            values=[float(row[f"delta_{key}"]) for row in payload["paired"]]
            stats[key]={"mean_delta":statistics.mean(values),"median_delta":statistics.median(values),
                        "worst_regression":max(values),"best_improvement":min(values),
                        "fixed_win_rate":sum(v<0 for v in values)/len(values),
                        "worst_sample_id":payload["paired"][values.index(max(values))]["sample_id"]}
        summaries.append({"route":route,"production_go":payload["production_go"],
            "adaptive_pg_pct":payload["adaptive_accuracy"]["point_global_true_rms_relative_rmse_pct"],
            "fixed_pg_pct":payload["fixed_accuracy"]["point_global_true_rms_relative_rmse_pct"],
            "delta_pg_pp":payload["fixed_minus_adaptive"]["point_global_pct"],
            "adaptive_raw_K":payload["adaptive_accuracy"]["raw_cv_weighted_rmse_K"],
            "fixed_raw_K":payload["fixed_accuracy"]["raw_cv_weighted_rmse_K"],
            "delta_raw_K":payload["fixed_minus_adaptive"]["raw_cv_rmse_K"],
            "delta_source_K":payload["fixed_minus_adaptive"]["source_rmse_K"],
            "delta_peak_K":payload["fixed_minus_adaptive"]["peak_rmse_K"],
            "delta_interface_K":payload["fixed_minus_adaptive"]["interface_rmse_K"],
            "max_sample_raw_delta_K":payload["worst_case"]["max_raw_delta_K"],
            "failed_gates":";".join(k for k,v in payload["production_gates"].items() if not v),
            "paired_statistics":stats})
    summary_fields=[k for k in summaries[0] if k!="paired_statistics"]
    with (DOC/"v6_p1i_p4a_fixed_support_accuracy.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=summary_fields,lineterminator="\n"); writer.writeheader(); writer.writerows([{k:r[k] for k in summary_fields} for r in summaries])
    paired_fields=["route","sample_id"]+[f"{prefix}_{key}" for key in metric_keys for prefix in ("adaptive","fixed","delta")]
    with (DOC/"v6_p1i_p4a_fixed_support_paired.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=paired_fields,lineterminator="\n"); writer.writeheader(); writer.writerows([{k:r[k] for k in paired_fields} for r in paired])
    stop=not any(row["production_go"] for row in summaries)
    closeout={"schema_version":"heat3d_v6_p1i_p4a_fixed_support_closeout_v1","status":"completed_fail_fast",
              "protocol_commit":"c6733ad","summaries":summaries,"production_routes":[],
              "p4b_allowed":not stop,"stop_reason":"both_fixed_support_routes_failed_preregistered_noninferiority_gate" if stop else None,
              "subsequent_stages":{"P4-B":"not_executed","P4-C":"not_executed","P4-D":"not_executed","P4-E":"not_executed"},
              "sources":{str(protocol_path.relative_to(ROOT)):sha(protocol_path),**{str(p.relative_to(ROOT)):sha(p) for p in paths.values()}},
              "role_contract":protocol["role_contract"]}
    (CFG/"v6_p1i_p4a_fixed_support_accuracy_closeout.json").write_text(json.dumps(closeout,indent=2,sort_keys=True)+"\n")
    lines=["# V6/P1i P4 fixed-support closeout","","P4-A 使用与 P2/P3 相同的固定 support/graph，重新计算 valid32 标签指标；没有复用 adaptive accuracy 冒充 fixed accuracy。","",
           "| route | adaptive PG % | fixed PG % | ΔPG pp | adaptive raw K | fixed raw K | Δsource K | Δpeak K | Δinterface K | max sample Δraw K | decision |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summaries:
        lines.append(f"| {r['route']} | {r['adaptive_pg_pct']:.4f} | {r['fixed_pg_pct']:.4f} | {r['delta_pg_pp']:+.4f} | {r['adaptive_raw_K']:.4f} | {r['fixed_raw_K']:.4f} | {r['delta_source_K']:+.4f} | {r['delta_peak_K']:+.4f} | {r['delta_interface_K']:+.4f} | {r['max_sample_raw_delta_K']:+.4f} | NO-GO ({r['failed_gates']}) |")
    lines += ["","## Decision","","- B8192-recon fixed support：NO-GO；PG、source、peak 超出预注册 margin。",
              "- E32768-recon fixed support：NO-GO；PG、raw、source、peak 超出预注册 margin。",
              "- fixed-support timing 不能与 adaptive-support accuracy 拼接为同一路线；P3 的 113x/44.8x 已在上一阶段废弃，本轮不产生新的 production speedup。",
              "- 因两条路线均失败，按 fail-fast 合同不执行 P4-B/C/D/E，也不提出开启 test/sealed。","",
              "逐样本结果见 `docs/v6_p1i_p4a_fixed_support_paired.csv`。"]
    (DOC/"v6_p1i_p4_closeout.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":closeout["status"],"p4b_allowed":closeout["p4b_allowed"],"routes":len(summaries)})); return 0
if __name__=="__main__": raise SystemExit(main())
