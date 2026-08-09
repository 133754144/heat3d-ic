#!/usr/bin/env python3
"""Generate paired A/B graph-policy confirmation closeout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_p1i"
PROTOCOL = CONFIG / "v6_p1i_graph_policy_ab_confirmation_protocol.json"
COMPACT = CONFIG / "v6_p1i_graph_policy_confirmation_compact.json"
FINAL = CONFIG / "v6_p1i_graph_policy_final.json"
CSV_PATH = ROOT / "docs/v6_p1i_graph_policy_confirmation.csv"
PAIRED_PATH = ROOT / "docs/v6_p1i_graph_policy_confirmation_paired.csv"
MD_PATH = ROOT / "docs/v6_p1i_graph_policy_confirmation.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: np.ndarray, *, seed: int, repeats: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        count = min(500, repeats - start)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = np.mean(values[draws], axis=1)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
        "win_rate": float(np.mean(values < 0.0)), "worst_case": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    compact = json.loads(COMPACT.read_text())
    if protocol["status"] != "frozen_after_E_no_go_before_confirmation" or compact["status"] != "passed":
        raise RuntimeError("confirmation inputs not frozen/passed")
    cells = {(int(r["seed"]), r["policy"], int(r["resolution"])): r for r in compact["cells"]}
    if set(cells) != {(s, p, n) for s in (0, 1, 2) for p in ("A", "B") for n in (8192, 16384)}:
        raise RuntimeError("confirmation matrix identity drifted")
    table = []
    for (seed, policy, resolution), row in sorted(cells.items()):
        full = row["accuracy"]["full_field"]
        table.append({
            "seed": seed, "policy": policy, "resolution": resolution,
            "full_point_global_pct": full["point_global_true_rms_relative_rmse_pct"],
            "full_sample_first_pct": full["sample_first_cv_relative_rmse_pct"],
            "full_raw_cv_rmse_K": full["raw_cv_weighted_rmse_K"],
            "source_rmse_K": full["source_rmse_K"], "peak_rmse_K": full["peak_rmse_K"],
            "interface_rmse_K": full["interface_drop_rmse_K"],
            "fresh_median_s": row["timing"]["new_case_e2e"]["median_seconds"],
            "fresh_p95_s": row["timing"]["new_case_e2e"]["p95_seconds"],
            "warm_median_s": row["timing"]["warm_cache_e2e"]["median_seconds"],
            "warm_p95_s": row["timing"]["warm_cache_e2e"]["p95_seconds"],
            "peak_vram_bytes": row["device_memory"]["peak_bytes_in_use"],
            "raw_result_sha256": row["raw_result"]["sha256"],
        })
    write_csv(CSV_PATH, table)
    metric_margin = {
        "full_rmse_K": protocol["non_inferiority_margins"]["full_raw_cv_rmse_K"],
        "source_rmse_K": protocol["non_inferiority_margins"]["source_rmse_K"],
        "peak_rmse_K": protocol["non_inferiority_margins"]["peak_rmse_K"],
        "interface_rmse_K": protocol["non_inferiority_margins"]["interface_rmse_K"],
    }
    paired_rows, ni_pass = [], True
    for resolution in (8192, 16384):
        by_seed = {}
        for seed in (0, 1, 2):
            by_policy = {}
            for policy in ("A", "B"):
                rows = cells[(seed, policy, resolution)]["per_sample_metrics"]
                by_policy[policy] = {row["sample_id"]: row for row in rows}
            ids = list(cells[(seed, "A", resolution)]["sample_ids"])
            if ids != list(cells[(seed, "B", resolution)]["sample_ids"]):
                raise RuntimeError("paired sample order drifted")
            by_seed[seed] = {metric: np.asarray([
                float(by_policy["B"][sample_id][metric]) - float(by_policy["A"][sample_id][metric])
                for sample_id in ids
            ]) for metric in metric_margin}
        for metric, margin in metric_margin.items():
            # Cluster by sample: average the paired effect across the three seeds,
            # then bootstrap the 96 sample IDs using the frozen seed.
            clustered = np.mean(np.stack([by_seed[s][metric] for s in (0, 1, 2)]), axis=0)
            stats = bootstrap(
                clustered, seed=int(protocol["paired_bootstrap"]["seed"]),
                repeats=int(protocol["paired_bootstrap"]["replicates"]),
            )
            passed = stats["ci_high"] <= float(margin)
            ni_pass &= passed
            paired_rows.append({
                "resolution": resolution, "metric": metric, "difference": "B_minus_A",
                **stats, "non_inferiority_margin": margin, "passed": passed,
            })
    write_csv(PAIRED_PATH, paired_rows)
    latency = {}
    latency_pass = True
    for resolution in (8192, 16384):
        values = {}
        for policy in ("A", "B"):
            policy_rows = [r for r in table if r["policy"] == policy and int(r["resolution"]) == resolution]
            values[policy] = {
                "fresh_median_s": float(np.median([float(r["fresh_median_s"]) for r in policy_rows])),
                "warm_median_s": float(np.median([float(r["warm_median_s"]) for r in policy_rows])),
                "peak_vram_bytes": int(max(int(r["peak_vram_bytes"]) for r in policy_rows)),
            }
        fresh_gain = 1.0 - values["B"]["fresh_median_s"] / values["A"]["fresh_median_s"]
        warm_gain = 1.0 - values["B"]["warm_median_s"] / values["A"]["warm_median_s"]
        vram_ratio = values["B"]["peak_vram_bytes"] / values["A"]["peak_vram_bytes"]
        passed = bool(fresh_gain >= 0.0 and warm_gain >= 0.0 and max(fresh_gain, warm_gain) >= 0.05 and vram_ratio <= 1.05)
        latency_pass &= passed
        latency[str(resolution)] = {
            "A": values["A"], "B": values["B"], "fresh_improvement_fraction": fresh_gain,
            "warm_improvement_fraction": warm_gain, "peak_vram_ratio": vram_ratio, "passed": passed,
        }
    go = bool(ni_pass and latency_pass)
    final = {
        "schema_version": "heat3d_v6_p1i_graph_policy_final_v1",
        "status": "completed", "decision": "B_GO" if go else "B_NO_GO_RETAIN_A",
        "selected_policy": "B_factor8_discrete_physical_coverage" if go else "A_factor4_discrete_physical_coverage",
        "E_decision": "NO_GO", "accuracy_non_inferiority_passed": ni_pass,
        "latency_pareto_passed": latency_pass, "paired": paired_rows, "latency": latency,
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha(PROTOCOL)},
        "compact_input": {"path": str(COMPACT.relative_to(ROOT)), "sha256": sha(COMPACT)},
        "role_contract": {"training": False, "test": False, "sealed": False, "remaining_valid96_only": True},
    }
    FINAL.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    MD_PATH.write_text(
        "# P1i A/B graph-policy confirmatory closeout\n\n"
        "范围：剩余 valid96 × seeds0/1/2 × N{8192,16384}；冻结 valid32 未重算；无训练、test/sealed。\n\n"
        f"最终判定：**{final['decision']}**。accuracy non-inferiority={ni_pass}，latency Pareto={latency_pass}。\n\n"
        "详细逐 seed 指标见 `v6_p1i_graph_policy_confirmation.csv`；clustered paired bootstrap、win rate、"
        "median、worst-case 与 95% CI 见 `v6_p1i_graph_policy_confirmation_paired.csv`。\n"
    )
    print(json.dumps({"status": "completed", "decision": final["decision"], "ni": ni_pass, "latency": latency_pass}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
