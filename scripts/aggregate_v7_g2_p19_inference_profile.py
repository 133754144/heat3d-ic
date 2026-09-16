#!/usr/bin/env python3
"""Compact a remote P19 valid-only inference profile receipt.

The raw receipt remains on the devbox/temporary evidence path.  This command
stores only reproducibility metadata and phase summaries in Git; it never opens
targets, predictions, or a test/sealed split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_summary(
    payload: dict[str, Any], rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    peak = payload.get("peak_gpu_memory", {})
    summary = {
        "peak_bytes_in_use": peak.get("peak_bytes_in_use"),
        "peak_bytes_reserved": peak.get("peak_bytes_reserved"),
        "peak_pool_bytes": peak.get("peak_pool_bytes"),
        "bytes_limit": peak.get("bytes_limit"),
        "peak_gib": (
            None
            if peak.get("peak_bytes_in_use") is None
            else float(peak["peak_bytes_in_use"]) / (1024.0**3)
        ),
    }
    live_values = [
        int(row[memory_key]["bytes_in_use"])
        for row in rows or []
        for memory_key in ("memory_before", "memory_after")
        if row.get(memory_key, {}).get("bytes_in_use") is not None
    ]
    if live_values:
        summary["peak_live_bytes_in_use"] = max(live_values)
        summary["peak_live_gib"] = max(live_values) / (1024.0**3)
    else:
        summary["peak_live_bytes_in_use"] = None
        summary["peak_live_gib"] = None
    summary["peak_stats_scope"] = "process cumulative; live peak is phase-row maximum"
    return summary


def phase_summary(payload: dict[str, Any], *, include_graph: bool) -> dict[str, Any]:
    rows = (
        payload.get("raw_rows", [])
        if include_graph
        else payload.get("raw_model_only_rows", []) + payload.get("raw_e2e_rows", [])
    )
    summary: dict[str, Any] = {
        "batch_size": payload.get("batch_size"),
        "valid_count": payload.get("valid_count"),
        "output_count": payload.get("query_count", payload.get("output_count")),
        "cold_e2e_seconds": payload.get("cold_e2e_seconds"),
        "postprocess_seconds": payload.get("postprocess_seconds"),
        "peak_gpu_memory": memory_summary(payload, rows),
        "checkpoint_role": payload.get("checkpoint_role"),
        "checkpoint": payload.get("checkpoint"),
        "status": payload.get("status"),
    }
    if include_graph:
        summary.update(
            {
                "query_graph_construction_seconds": payload.get(
                    "query_graph_construction_seconds"
                ),
                "model_forward_seconds": payload.get("model_forward_seconds"),
                "steady_e2e_seconds": payload.get("steady_e2e_seconds"),
                "cached_fixture": {
                    "valid_index": payload.get("cached_fixture", {}).get("valid_index"),
                    "repeats": payload.get("cached_fixture", {}).get("repeats"),
                    "e2e_seconds": payload.get("cached_fixture", {}).get("e2e_seconds"),
                },
                "edge_counts": payload.get("edge_counts"),
                "all_finite": all(
                    bool(row.get("finite")) for row in payload.get("raw_rows", [])
                ),
            }
        )
    else:
        summary.update(
            {
                "model_only_seconds": payload.get("model_only_seconds"),
                "e2e_seconds": payload.get("e2e_seconds"),
                "cached_fixture": {
                    "repeats": payload.get("cached_fixture", {}).get("repeats"),
                    "e2e_seconds": payload.get("cached_fixture", {}).get("e2e_seconds"),
                },
                "all_finite": payload.get("status", "").startswith("PASS_"),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--idw-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not str(raw.get("status", "")).startswith("PASS_"):
        raise SystemExit("P19 raw receipt is not a PASS valid-only profile")
    boundaries = raw.get("hard_boundaries", {})
    forbidden = (
        "training_started",
        "p1i_test_iid_accessed",
        "sealed_accessed",
        "deepoheat_official100_accessed",
    )
    if any(boundaries.get(key) is not False for key in forbidden):
        raise SystemExit("P19 hard-boundary receipt is not closed")
    if raw.get("data_provenance", {}).get("target_truth_loaded") is not False:
        raise SystemExit("P19 profile loaded target/truth data")
    if raw.get("checkpoint_reload_provenance", {}).get("weights_modified") is not False:
        raise SystemExit("P19 checkpoint weights were modified")
    heat3d = raw["profiles"]["heat3d"]
    deepoheat = raw["profiles"]
    out = {
        "schema_version": "heat3d_v7_g2_p19_inference_profile_aggregation_v2",
        "status": "COMPLETE_VALID_ONLY_INFERENCE_PROFILE_AGGREGATION",
        "scope": "P19 valid128 inference profiling only; no training or accuracy claim",
        "profile_raw_receipt": {
            "path": str(args.input),
            "sha256": sha256(args.input),
            "schema_version": raw.get("schema_version"),
            "wall_clock_started_unix": raw.get("wall_clock_started_unix"),
            "wall_clock_finished_unix": raw.get("wall_clock_finished_unix"),
        },
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "idw_provenance": {
            "path": str(args.idw_provenance),
            "sha256": sha256(args.idw_provenance),
            "role": "historical P14.1 diagnostic only; not used by this timing profile",
        },
        "runner": raw.get("runner"),
        "environment": raw.get("environment"),
        "data_provenance": raw.get("data_provenance"),
        "checkpoint_reload_provenance": raw.get("checkpoint_reload_provenance"),
        "timing_contract": raw.get("timing_contract"),
        "profiles": {
            "heat3d_e600_fixed_endpoint": phase_summary(heat3d, include_graph=True),
            "deepoheat_full_minus_valid128_best": phase_summary(
                deepoheat["deepoheat_best"], include_graph=False
            ),
            "deepoheat_full_minus_valid128_final": phase_summary(
                deepoheat["deepoheat_final"], include_graph=False
            ),
        },
        "hard_boundaries": {
            "training_started": False,
            "p1i_test_iid_accessed": False,
            "sealed_accessed": False,
            "deepoheat_official100_accessed": False,
            "therm_fm_downloaded": False,
            "multi_htc_started": False,
            "existing_checkpoint_overwritten": False,
        },
        "interpretation": {
            "u_v2_naming": "direct-query dense inference",
            "formal_accuracy": "not claimed; finite checks only",
            "latency_boundary": "explicit jax.block_until_ready; cold first valid pass includes compilation; steady pass excludes the first case; cached fixture repeats use a prepared first case/batch",
            "checkpoint_policy": "Heat3D fixed e600 endpoint; DeepOHeat validation-selected best and final endpoint both profiled",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    h = out["profiles"]["heat3d_e600_fixed_endpoint"]
    b = out["profiles"]["deepoheat_full_minus_valid128_best"]
    f = out["profiles"]["deepoheat_full_minus_valid128_final"]
    lines = [
        "# V7 G2-P19：valid-only inference profiling aggregation",
        "",
        "仅使用冻结 valid128 输入；不读取 target/truth，不训练，不访问 test_iid、sealed 或 DeepOHeat official100。",
        "U-v2 统一称 `direct-query dense inference`。所有时间均在同一 devbox GPU、显式 `jax.block_until_ready` 边界下记录。",
        "",
        "| checkpoint | cold E2E (s) | steady/model-only (s) | cached E2E median (s) | peak live GiB |",
        "|---|---:|---:|---:|---:|",
        f"| Heat3D e600 fixed endpoint | {h['cold_e2e_seconds']:.6f} | {h['steady_e2e_seconds']['median']:.6f} steady E2E | {h['cached_fixture']['e2e_seconds']['median']:.6f} | {h['peak_gpu_memory']['peak_live_gib']:.3f} |",
        f"| DeepOHeat best | {b['cold_e2e_seconds']:.6f} | {b['model_only_seconds']['median']:.6f} model-only | {b['cached_fixture']['e2e_seconds']['median']:.6f} | {b['peak_gpu_memory']['peak_live_gib']:.3f} |",
        f"| DeepOHeat final | {f['cold_e2e_seconds']:.6f} | {f['model_only_seconds']['median']:.6f} model-only | {f['cached_fixture']['e2e_seconds']['median']:.6f} | {f['peak_gpu_memory']['peak_live_gib']:.3f} |",
        "",
        "Heat3D phases: query-graph construction, model forward, postprocess and steady/cached E2E are retained in the JSON receipt. DeepOHeat reports model-only and E2E separately. The table uses each phase's live row maximum; JAX cumulative process peak is retained separately.",
        "The IDW utility is documented separately as a historical Heat3D V6/P1h diagnostic and is not used to rename or interpret U-v2 timing.",
        "",
        f"Raw receipt SHA256: `{out['profile_raw_receipt']['sha256']}`; runner commit: `{out['runner'].get('repo_commit_sha')}`.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "output": str(args.output), "markdown": str(args.markdown)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
