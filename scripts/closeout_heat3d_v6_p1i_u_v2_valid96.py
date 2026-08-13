#!/usr/bin/env python3
"""Close U-v2 valid96 characterization and the unified 240825 timing matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "point_global_true_rms_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "source_rmse_K",
    "peak_rmse_K",
    "interface_drop_rmse_K",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "median": float(np.median(x)), "mean": float(np.mean(x)),
            "std": float(np.std(x)), "p95": float(np.quantile(x, .95))}


def distribution(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def continuous(payload: dict[str, Any]) -> list[float]:
    key = "timing" if "route" in payload else "stages"
    return [float(row[key]["matched_continuous_e2e"]) for row in payload["samples"]]


def route_timing(payloads: list[dict[str, Any]], route: str) -> dict[str, Any]:
    if len(payloads) != 3:
        raise RuntimeError(f"{route}: exactly three randomized orders required")
    fresh=[];fresh95=[];resident=[];resident95=[];marginal=[];q1=[];q195=[];q2=[];q295=[];throughput=[]
    for payload in payloads:
        if not payload.get("timing_only"):
            raise RuntimeError(f"{route}: service timing must be timing-only")
        if "route" in payload:
            f=payload["timing"]["matched_continuous_e2e"];r=payload["resident_core"]
        else:
            f=payload["runtime"]["fresh_sample"]["matched_continuous_e2e"];r=payload["runtime"]["same_input_replay"]
        fresh.append(float(f["median_seconds"]));fresh95.append(float(f["p95_seconds"]))
        resident.append(float(r["median_seconds"]));resident95.append(float(r["p95_seconds"]))
        values=continuous(payload);marginal.append((sum(values[:32])-sum(values[:16]))/16.0)
        stream=payload["streaming"];q1.append(float(stream["submit_to_result"]["median_seconds"]));q195.append(float(stream["submit_to_result"]["p95_seconds"]))
        completion=np.cumsum(values);submitted=np.zeros_like(completion);submitted[2:]=completion[:-2]
        latency=completion-submitted
        q2.append(float(np.median(latency)));q295.append(float(np.quantile(latency,.95)))
        throughput.append(float(len(values)/completion[-1]))
    peak_vram=max(int(payload.get("peak_vram_bytes",payload.get("memory",{}).get("peak_bytes_in_use",0))) for payload in payloads)
    return {
        "randomized_order_count":3,
        "fresh_single_case":{"median_seconds":float(np.median(fresh)),"p95_seconds":float(max(fresh95))},
        "resident_core":{"median_seconds":float(np.median(resident)),"p95_seconds":float(max(resident95))},
        "batch_scale_marginal_fresh_case_estimate":{"definition":"actual_distinct_case_B32_minus_B16_divided_by_16","median_seconds":float(np.median(marginal)),"p95_seconds":float(max(marginal))},
        "closed_loop_added_case_latency":{"queue_depth":1,"median_seconds":float(np.median(q1)),"p95_seconds":float(max(q195))},
        "saturated_streaming":{"queue_depth":2,"worker_count":1,"arrival":"two_submitted_then_refill_after_completion","median_submit_to_result_seconds":float(np.median(q2)),"p95_submit_to_result_seconds":float(max(q295)),"median_throughput_samples_per_second":float(np.median(throughput))},
        "peak_vram_bytes":peak_vram,
        "note":"Q2 arrival trace uses uninterrupted measured per-case service spans; no qualification, hash, metrics, or serialization is in service wall",
    }


def aggregate_components(rows: list[dict[str, Any]], draws: np.ndarray | None = None) -> dict[str,float] | dict[str,np.ndarray]:
    cols={key:np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    total=lambda key: np.sum(cols[key] if draws is None else cols[key][draws],axis=None if draws is None else 1)
    n=len(rows) if draws is None else draws.shape[1]
    return {
        METRICS[0]:np.sqrt(total("point_sse")/total("point_energy"))*100,
        METRICS[1]:np.sqrt(total("weighted_sse")/total("volume")),
        METRICS[2]:np.sqrt(total("source_sse")/total("source_volume")),
        METRICS[3]:np.sqrt(total("peak_error_squared")/n),
        METRICS[4]:np.sqrt(total("interface_error_squared_sum")/total("interface_error_count")),
    }


def paired(left: dict[str,Any], right: dict[str,Any], seed: int, repeats: int) -> dict[str,Any]:
    left_rows={row["sample_id"]:row["full_field_metric_components"] for row in left["samples"]}
    right_rows={row["sample_id"]:row["full_field_metric_components"] for row in right["samples"]}
    if set(left_rows)!=set(right_rows): raise RuntimeError("paired populations differ")
    ids=left.get("sample_ids") or [row["sample_id"] for row in left["samples"]]
    l=[left_rows[x] for x in ids];r=[right_rows[x] for x in ids]
    rng=np.random.default_rng(seed);draws=rng.integers(0,len(ids),size=(repeats,len(ids)))
    la=aggregate_components(l);ra=aggregate_components(r);lb=aggregate_components(l,draws);rb=aggregate_components(r,draws)
    out={}
    for metric in METRICS:
        delta=np.asarray(lb[metric])-np.asarray(rb[metric])
        # Per-sample RMSE contribution comparison provides an interpretable win rate.
        wins=[]
        for x,y in zip(l,r):
            if metric==METRICS[0]: wins.append(x["point_sse"]/x["point_energy"] < y["point_sse"]/y["point_energy"])
            elif metric==METRICS[1]: wins.append(x["weighted_sse"]/x["volume"] < y["weighted_sse"]/y["volume"])
            elif metric==METRICS[2]: wins.append(x["source_sse"]/x["source_volume"] < y["source_sse"]/y["source_volume"])
            elif metric==METRICS[3]: wins.append(x["peak_error_squared"] < y["peak_error_squared"])
            else: wins.append(x["interface_error_squared_sum"]/x["interface_error_count"] < y["interface_error_squared_sum"]/y["interface_error_count"])
        out[metric]={"left_minus_right":float(la[metric]-ra[metric]),"bootstrap_95pct_CI":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],"left_win_rate":float(np.mean(wins))}
    return out


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,required=True);p.add_argument("--u-qualification",type=Path,required=True)
    p.add_argument("--e16384-qualification",type=Path,required=True);p.add_argument("--e240825-qualification",type=Path,required=True)
    p.add_argument("--u-timing",type=Path,action="append",required=True);p.add_argument("--e16384-timing",type=Path,action="append",required=True);p.add_argument("--e240825-timing",type=Path,action="append",required=True)
    p.add_argument("--fvm",type=Path,required=True);p.add_argument("--fvm-batch",type=Path,required=True);p.add_argument("--output-json",type=Path,required=True);p.add_argument("--output-csv",type=Path,required=True);p.add_argument("--output-md",type=Path,required=True)
    a=p.parse_args();protocol=load(a.protocol);uq=load(a.u_qualification);eq=load(a.e16384_qualification);cq=load(a.e240825_qualification);fvm=load(a.fvm);fvm_batch=load(a.fvm_batch)
    for payload in (uq,eq,cq):
        if payload.get("status")!="passed" or payload.get("sample_count")!=96 or payload.get("population_mode")!="remaining_valid96":raise RuntimeError("qualification binding failed")
    routes={
        "E16384-reconstruction":{"accuracy":eq["accuracy"]["full_field"],"timing":route_timing([load(x) for x in a.e16384_timing],"E16384")},
        "U-v2-direct240825":{"accuracy":uq["accuracy"]["full_field"],"timing":route_timing([load(x) for x in a.u_timing],"U-v2")},
        "E240825-direct-control":{"accuracy":cq["accuracy"]["full_field"],"timing":route_timing([load(x) for x in a.e240825_timing],"E240825")},
    }
    p1=next(row for row in fvm["rows"] if row["process_count"]==1);sat=next(row for row in fvm["rows"] if row["process_count"]==fvm["saturation_process_count"])
    fresh=[rep["fresh_single_case"]["median_seconds"] for rep in p1["repeats"]];fresh95=[rep["fresh_single_case"]["p95_seconds"] for rep in p1["repeats"]]
    resident=[rep["resident_core_solve_only"]["median_seconds"] for rep in p1["repeats"]];resident95=[rep["resident_core_solve_only"]["p95_seconds"] for rep in p1["repeats"]]
    if fvm_batch.get("status")!="passed" or fvm_batch.get("process_count")!=fvm["saturation_process_count"]:raise RuntimeError("FVM B16/B32 batch binding failed")
    marginal=[float(row["marginal_B16_to_B32_seconds"]) for row in fvm_batch["rows"]]
    satlat=[rep["stream_submit_to_result"]["median_seconds"] for rep in sat["repeats"]];sat95=[rep["stream_submit_to_result"]["p95_seconds"] for rep in sat["repeats"]];satthr=[rep["streaming_samples_per_second"] for rep in sat["repeats"]]
    fvm_t={"randomized_order_count":3,"fresh_single_case":{"median_seconds":float(np.median(fresh)),"p95_seconds":float(max(fresh95))},"resident_core":{"definition":"prepared_system_solve_only_not_E2E","median_seconds":float(np.median(resident)),"p95_seconds":float(max(resident95))},"batch_scale_marginal_fresh_case_estimate":{"definition":"per_case_distribution_proxy_no_FVM_B16_B32_batch_API","median_seconds":float(np.median(marginal)),"p95_seconds":float(max(marginal))},"closed_loop_added_case_latency":{"queue_depth":1,"median_seconds":float(np.median(fresh)),"p95_seconds":float(max(fresh95))},"saturated_streaming":{"queue_depth":fvm["queue_depth"],"worker_count":fvm["saturation_process_count"],"median_submit_to_result_seconds":float(np.median(satlat)),"p95_submit_to_result_seconds":float(max(sat95)),"median_throughput_samples_per_second":float(np.median(satthr))}}
    for route in routes.values():
        timing=route["timing"]
        timing["ratios_vs_FVM"]={
            "fresh_speedup":fvm_t["fresh_single_case"]["median_seconds"]/timing["fresh_single_case"]["median_seconds"],
            "resident_core_ratio":fvm_t["resident_core"]["median_seconds"]/timing["resident_core"]["median_seconds"],
            "B16_to_B32_marginal_speedup":fvm_t["batch_scale_marginal_fresh_case_estimate"]["median_seconds"]/timing["batch_scale_marginal_fresh_case_estimate"]["median_seconds"],
            "closed_loop_Q1_speedup":fvm_t["closed_loop_added_case_latency"]["median_seconds"]/timing["closed_loop_added_case_latency"]["median_seconds"],
            "saturated_throughput_ratio":timing["saturated_streaming"]["median_throughput_samples_per_second"]/fvm_t["saturated_streaming"]["median_throughput_samples_per_second"],
        }
    comparisons={"U-v2_minus_E16384":paired(uq,eq,protocol["paired_bootstrap"]["seed"],protocol["paired_bootstrap"]["replicates"]),"U-v2_minus_E240825":paired(uq,cq,protocol["paired_bootstrap"]["seed"],protocol["paired_bootstrap"]["replicates"])}
    artifacts=[a.protocol,a.u_qualification,a.e16384_qualification,a.e240825_qualification,*a.u_timing,*a.e16384_timing,*a.e240825_timing,a.fvm,a.fvm_batch]
    result={"schema_version":"heat3d_v6_p1i_u_v2_valid96_closeout_v1","status":"passed_final_freeze","population":"valid96_diagnostic_characterization","output_nodes":240825,"routes":routes,"FVM240825":{"accuracy_role":"reference_solution_zero_surrogate_error","timing":fvm_t},"paired_statistics":comparisons,"decision":{"production_reference":"E16384-reconstruction","parallel_direct_strategy":"U-v2-direct240825","architecture_control":"E240825-direct","valid32_architecture_optimization_closed":True,"test_or_sealed_opened":False},"artifacts":[{"path":str(x),"sha256":sha(x),"bytes":x.stat().st_size} for x in artifacts],"role_contract":protocol["role_contract"]}
    a.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    rows=[]
    for name,row in routes.items():
        acc=row["accuracy"];tim=row["timing"]
        rows.append({"strategy":name,"domain":"240825_solver_nodes","PG_pct":acc[METRICS[0]],"raw_K":acc[METRICS[1]],"source_K":acc[METRICS[2]],"peak_K":acc[METRICS[3]],"interface_K":acc[METRICS[4]],"fresh_median_s":tim["fresh_single_case"]["median_seconds"],"fresh_p95_s":tim["fresh_single_case"]["p95_seconds"],"resident_median_s":tim["resident_core"]["median_seconds"],"marginal_B16_to_B32_s":tim["batch_scale_marginal_fresh_case_estimate"]["median_seconds"],"closed_loop_Q1_median_s":tim["closed_loop_added_case_latency"]["median_seconds"],"saturated_Q2_submit_median_s":tim["saturated_streaming"]["median_submit_to_result_seconds"],"saturated_throughput_samples_s":tim["saturated_streaming"]["median_throughput_samples_per_second"],"peak_vram_bytes":tim["peak_vram_bytes"],**tim["ratios_vs_FVM"]})
    rows.append({"strategy":"FVM240825","domain":"240825_solver_nodes_reference","PG_pct":0.0,"raw_K":0.0,"source_K":0.0,"peak_K":0.0,"interface_K":0.0,"fresh_median_s":fvm_t["fresh_single_case"]["median_seconds"],"fresh_p95_s":fvm_t["fresh_single_case"]["p95_seconds"],"resident_median_s":fvm_t["resident_core"]["median_seconds"],"marginal_B16_to_B32_s":fvm_t["batch_scale_marginal_fresh_case_estimate"]["median_seconds"],"closed_loop_Q1_median_s":fvm_t["closed_loop_added_case_latency"]["median_seconds"],"saturated_Q2_submit_median_s":fvm_t["saturated_streaming"]["median_submit_to_result_seconds"],"saturated_throughput_samples_s":fvm_t["saturated_streaming"]["median_throughput_samples_per_second"],"peak_vram_bytes":"N/A","fresh_speedup":1.0,"resident_core_ratio":1.0,"B16_to_B32_marginal_speedup":1.0,"closed_loop_Q1_speedup":1.0,"saturated_throughput_ratio":1.0})
    with a.output_csv.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["# P1i U-v2 valid96 performance closeout","","所有 accuracy 均为 frozen valid96 diagnostic/characterization；未访问 test/sealed，未训练。E16384 保持 production/reference，U-v2 是并列 direct inference strategy，E240825 仅作 architecture control。","","| strategy | PG % | raw K | source K | peak K | interface K | fresh med s | resident med s | Q1 med s | Q2 throughput/s |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:lines.append(f"| {row['strategy']} | {row['PG_pct']:.6f} | {row['raw_K']:.6f} | {row['source_K']:.6f} | {row['peak_K']:.6f} | {row['interface_K']:.6f} | {row['fresh_median_s']:.6f} | {row['resident_median_s']:.6f} | {row['closed_loop_Q1_median_s']:.6f} | {row['saturated_throughput_samples_s']:.6f} |")
    lines += ["","## 解释","","- `fresh_single_case` 与 Q1 是不同新 k/q/BC 的完整 in-memory compute service。","- neural Q2 使用相同固定深度 arrival rule 在不中断的实测逐 case service trace 上重放；FVM Q2 是两个 persistent workers 的直接 wall-clock。worker 数不同，submit latency 与 throughput 均保留，不混称相同 core latency。","- resident FVM 是 prepared-system solve-only，不是 E2E；surrogate 指标是相对 FVM reference field 的误差，绝不表示精度优于 FVM。"]
    a.output_md.write_text("\n".join(lines)+"\n");print(json.dumps({"status":result["status"],"json":str(a.output_json)}));return 0


if __name__=="__main__":raise SystemExit(main())
