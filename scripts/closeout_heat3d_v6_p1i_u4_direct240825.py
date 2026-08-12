#!/usr/bin/env python3
"""Close out U4 direct-240825 qualification from frozen valid32 predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROUTES = ("native1024_reconstruction", "E16384_reconstruction", "E240825_direct", "U_direct240825")
METRICS = ("point_global_pct", "raw_cv_rmse_K", "source_rmse_K", "peak_abs_error_K", "interface_rmse_K")
SSE_METRICS = ("point_global_SSE", "raw_cv_SSE", "source_cv_SSE", "peak_squared_error", "interface_squared_error")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap(delta: np.ndarray, *, seed: int, replicates: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = np.mean(delta[indices], axis=1)
    return {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "win_rate": float(np.mean(delta < 0.0)),
        "worst_case_delta": float(np.max(delta)),
    }


def sample_metrics(prediction: np.ndarray, truth: np.ndarray, cv: np.ndarray,
                   layer: np.ndarray, q: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    source = q > 0.0
    means = []
    for value in sorted(np.unique(layer)):
        mask = layer == value
        means.append(float(np.sum(cv[mask] * error[mask]) / np.sum(cv[mask])))
    interface_squared_error = float(np.mean(np.square(np.diff(means))))
    return {
        "point_global_pct": math.sqrt(float(np.sum(error ** 2)) / float(np.sum(truth ** 2))) * 100.0,
        "raw_cv_rmse_K": math.sqrt(float(np.sum(cv * error ** 2)) / float(np.sum(cv))),
        "source_rmse_K": math.sqrt(float(np.sum(cv[source] * error[source] ** 2)) / float(np.sum(cv[source]))),
        "peak_abs_error_K": abs(float(np.max(prediction) - np.max(truth))),
        "interface_rmse_K": math.sqrt(interface_squared_error),
        "point_global_SSE": float(np.sum(error ** 2)),
        "raw_cv_SSE": float(np.sum(cv * error ** 2)),
        "source_cv_SSE": float(np.sum(cv[source] * error[source] ** 2)),
        "peak_squared_error": float((np.max(prediction) - np.max(truth)) ** 2),
        "interface_squared_error": interface_squared_error,
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--u4-result", type=Path, required=True)
    parser.add_argument("--p8-closeout", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse()
    protocol = json.loads(args.protocol.read_text())
    expected_ids = None
    prediction: dict[str, np.ndarray] = {}
    prediction_artifacts = {}
    for route in ROUTES:
        path = args.prediction_root / f"{route}.npz"
        with np.load(path, allow_pickle=False) as payload:
            ids = [str(value) for value in payload["sample_ids"].tolist()]
            prediction[route] = np.asarray(payload["full_deltaT_K"], dtype=np.float64)
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids or prediction[route].shape != (32, 240825):
            raise RuntimeError(f"{route}: sample/order/shape drift")
        prediction_artifacts[route] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    with h5py.File(args.full_fields, "r") as archive:
        archive_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in archive["samples/sample_id"][:]]
        lookup = {value: index for index, value in enumerate(archive_ids)}
        truth = np.stack([np.asarray(archive["samples/deltaT_K"][lookup[sid]], dtype=np.float64) for sid in expected_ids])
        cv = np.asarray(archive["shared/control_volume_m3"], dtype=np.float64)
        layer = np.asarray(archive["shared/layer_id"], dtype=np.int32)
    q = np.stack([
        np.asarray(np.load(args.physics_root / f"{sid}.npz")["q_W_m3"], dtype=np.float64)
        for sid in expected_ids
    ])

    rows = []
    all_metrics = METRICS + SSE_METRICS
    by_route: dict[str, dict[str, np.ndarray]] = {route: {metric: np.empty(32) for metric in all_metrics} for route in ROUTES}
    for index, sid in enumerate(expected_ids):
        for route in ROUTES:
            values = sample_metrics(prediction[route][index], truth[index], cv, layer, q[index])
            row = {"sample_id": sid, "route": route, **values}
            rows.append(row)
            for metric in all_metrics:
                by_route[route][metric][index] = values[metric]

    comparisons = {}
    preregistered_sse = {}
    for baseline in ("native1024_reconstruction", "E16384_reconstruction", "E240825_direct"):
        key = f"U_direct240825_minus_{baseline}"
        comparisons[key] = {
            metric: bootstrap(
                by_route["U_direct240825"][metric] - by_route[baseline][metric],
                seed=int(protocol["bootstrap"]["seed"]),
                replicates=int(protocol["bootstrap"]["replicates"]),
            )
            for metric in METRICS
        }
        preregistered_sse[key] = {
            metric: bootstrap(
                by_route["U_direct240825"][metric] - by_route[baseline][metric],
                seed=int(protocol["bootstrap"]["seed"]),
                replicates=int(protocol["bootstrap"]["replicates"]),
            )
            for metric in SSE_METRICS
        }

    historical = {route: json.loads((args.historical_root / f"{route}.json").read_text()) for route in ROUTES[:-1]}
    current_u4 = json.loads(args.u4_result.read_text())
    aggregate = {
        **{route: historical[route]["accuracy"]["full_field"] for route in ROUTES[:-1]},
        "U_direct240825": current_u4["accuracy"]["full_field"],
    }
    latency = {
        route: historical[route]["timing"]["matched_continuous_e2e"] for route in ROUTES[:-1]
    }
    latency["U_direct240825"] = current_u4["runtime"]["fresh_sample"]["matched_continuous_e2e"]
    same_domain = {
        "U_direct_minus_E240825_pg_pp": aggregate["U_direct240825"]["point_global_true_rms_relative_rmse_pct"] - aggregate["E240825_direct"]["point_global_true_rms_relative_rmse_pct"],
        "U_direct_minus_E240825_raw_K": aggregate["U_direct240825"]["raw_cv_weighted_rmse_K"] - aggregate["E240825_direct"]["raw_cv_weighted_rmse_K"],
        "U_direct_minus_E240825_fresh_median_s": latency["U_direct240825"]["median_seconds"] - latency["E240825_direct"]["median_seconds"],
    }
    same_domain["u_direct_dominates_E240825"] = bool(all(value < 0.0 for value in same_domain.values()))

    p8 = json.loads(args.p8_closeout.read_text())
    result: dict[str, Any] = {
        "schema_version": "heat3d_v6_p1i_u4_direct240825_closeout_v1",
        "status": "passed_valid32",
        "decision": "GO_architecture_freeze_candidate" if same_domain["u_direct_dominates_E240825"] else "NO_GO",
        "decision_scope": "valid32 architecture freeze candidate; not test/sealed confirmation",
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "prediction_artifacts": prediction_artifacts,
        "historical_aggregate_metrics": aggregate,
        "matched_continuous_latency": latency,
        "same_240825_output_pareto": same_domain,
        "paired_valid32": comparisons,
        "paired_valid32_preregistered_sse": preregistered_sse,
        "packing": current_u4["packing_optimization"],
        "padding": current_u4["padding"],
        "checkpoint_parameters_unchanged": current_u4["checkpoint_parameters_unchanged"],
        "memory": current_u4["memory"],
        "p8_publication_safe": p8["publication_safe"],
        "role_contract": protocol["role_contract"],
        "paired_replay_note": "new prediction-only replay for paired statistics; frozen historical aggregate metrics remain authoritative",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fields = ["route", "pg_pct", "raw_K", "source_K", "peak_K", "interface_K", "fresh_median_s", "fresh_p95_s", "peak_vram_bytes"]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for route in ROUTES:
            metric = aggregate[route]; timing = latency[route]
            writer.writerow({"route": route, "pg_pct": metric["point_global_true_rms_relative_rmse_pct"], "raw_K": metric["raw_cv_weighted_rmse_K"], "source_K": metric["source_rmse_K"], "peak_K": metric["peak_rmse_K"], "interface_K": metric["interface_drop_rmse_K"], "fresh_median_s": timing["median_seconds"], "fresh_p95_s": timing["p95_seconds"], "peak_vram_bytes": current_u4["memory"]["peak_bytes_in_use"] if route == "U_direct240825" else historical[route]["peak_vram_bytes"]})
    lines = ["# V6 P1i U4 direct-240825 qualification", "", f"Decision: `{result['decision']}`.", "", "| route | PG % | raw K | source K | peak K | interface K | fresh median s | p95 s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for route in ROUTES:
        m, t = aggregate[route], latency[route]
        lines.append(f"| {route} | {m['point_global_true_rms_relative_rmse_pct']:.6f} | {m['raw_cv_weighted_rmse_K']:.6f} | {m['source_rmse_K']:.6f} | {m['peak_rmse_K']:.6f} | {m['interface_drop_rmse_K']:.6f} | {t['median_seconds']:.6f} | {t['p95_seconds']:.6f} |")
    lines += ["", "## Same-output direct Pareto", "", f"U-direct minus E240825-direct: PG {same_domain['U_direct_minus_E240825_pg_pp']:+.6f} pp, raw {same_domain['U_direct_minus_E240825_raw_K']:+.6f} K, fresh median {same_domain['U_direct_minus_E240825_fresh_median_s']:+.6f} s. U-direct dominates: `{same_domain['u_direct_dominates_E240825']}`.", "", "The historical U3 +0.1 pp comparison to E16384 is report-only in U4. Frozen historical artifacts were not modified. The paired replay is valid32-only and does not open test/sealed."]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"], "same_output_pareto": same_domain}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
