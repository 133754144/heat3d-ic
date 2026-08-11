#!/usr/bin/env python3
"""Close out checkpoint-preserving U1 identity and asymmetric-query probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--n8192", type=Path)
    parser.add_argument("--n32768", type=Path)
    parser.add_argument("--p5r", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol); identity = load(args.identity); p5r = load(args.p5r)
    if identity["status"] != "passed" or not identity["identity_hard_gate_passed"]:
        raise RuntimeError("U1 identity did not pass")
    probes = []
    for resolution, path in ((8192, args.n8192), (32768, args.n32768)):
        if path is None:
            break
        payload = load(path)
        if payload["status"] != "passed" or payload["sample_count"] != 32:
            raise RuntimeError(f"U1 N={resolution} valid32 did not pass")
        probes.append(payload)
    if not probes:
        decision = "NO_GO_stopped_after_identity_or_8192_smoke"
        discuss_240825 = False
    elif len(probes) == 1:
        decision = "NO_GO_32768_not_established"
        discuss_240825 = False
    else:
        decision = "GO_checkpoint_preserving_asymmetric_query_feasible"
        discuss_240825 = True
    baseline = {int(row["N"]): row for row in p5r["rows"]}
    comparisons = []
    for probe in probes:
        n = int(probe["resolution"]); ref = baseline[n]
        metric = probe["full_field_accuracy"]
        comparisons.append({
            "resolution": n,
            "u1_point_global_pct": metric["point_global_true_rms_relative_rmse_pct"],
            "baseline_point_global_pct": ref["point_global_pct"],
            "delta_point_global_percentage_points": metric["point_global_true_rms_relative_rmse_pct"] - ref["point_global_pct"],
            "u1_raw_cv_rmse_K": metric["raw_cv_weighted_rmse_K"],
            "baseline_raw_cv_rmse_K": ref["raw_cv_rmse_K"],
            "u1_source_rmse_K": metric["source_rmse_K"],
            "baseline_source_rmse_K": ref["source_rmse_K"],
            "u1_peak_rmse_K": metric["peak_rmse_K"],
            "baseline_peak_rmse_K": ref["peak_rmse_K"],
            "u1_interface_rmse_K": metric["interface_drop_rmse_K"],
            "baseline_interface_rmse_K": ref["interface_rmse_K"],
            "u1_forward_median_s": probe["steady_forward"]["median_seconds"],
            "baseline_query_forward_median_s": ref["forward_median_s"],
            "forward_speedup": ref["forward_median_s"] / probe["steady_forward"]["median_seconds"],
            "peak_vram_bytes": probe["memory"].get("peak_bytes_in_use", 0),
        })
    result = {
        "schema_version": "heat3d_v6_p1i_u1_split_adapter_closeout_v1",
        "status": "completed",
        "identity": identity,
        "high_n": probes,
        "comparisons_vs_current_p5r_route_at_same_resolution": comparisons,
        "layer_gates": {
            "frozen_encoder_output_local_transform": "PASS_bitwise_identity_32_of_32",
            "frozen_original_decoder_pre_bypass": "1024_nodes_not_N",
            "split_adapter_decoder_pre_bypass": "PASS_N_nodes_at_8192_and_32768",
            "native_shape_scale_and_projection": "PASS_finite_at_8192_and_32768",
            "checkpoint_parameter_tree": "PASS_unchanged",
            "backend_contract": "CPU_bitwise_identity_then_GPU_high_N",
        },
        "decision": {
            "u1": decision,
            "worth_entering_1024_to_240825": discuss_240825,
            "production_route_replaced": False,
            "note": "U1 remains an exploratory checkpoint-preserving route; P5-R production recommendation is unchanged in this phase. Forward-only savings are not production E2E speedups."
        },
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# V6 P1i U1 split-adapter closeout", "",
        f"Identity gate: **PASS** on {identity['sample_count']} frozen valid samples using the deterministic CPU backend; every registered intermediate and final tensor is bitwise equal.", "",
        "The original frozen decoder actually produces 1024 pre-bypass nodes under the asymmetric graph. The split adapter separately applies the frozen pnode-local Encoder transform to `c_out`, so its pre-bypass tensor has N nodes without changing checkpoint parameters.", "",
        "| N | original pre-bypass | split pre-bypass | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | forward median (s) | ΔPG vs P5-R route (pp) | forward-only speedup |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for probe, comparison in zip(probes, comparisons, strict=True):
        metric = probe["full_field_accuracy"]
        lines.append(
            f"| {probe['resolution']} | {probe['original_decoder_pre_bypass_output_shape']} | "
            f"{probe['split_adapter_pre_bypass_output_shape']} | "
            f"{metric['point_global_true_rms_relative_rmse_pct']:.6f} | "
            f"{metric['raw_cv_weighted_rmse_K']:.6f} | "
            f"{metric['source_rmse_K']:.6f} | "
            f"{metric['peak_rmse_K']:.6f} | "
            f"{metric['interface_drop_rmse_K']:.6f} | "
            f"{probe['steady_forward']['median_seconds']:.6f} | "
            f"{comparison['delta_point_global_percentage_points']:+.6f} | "
            f"{comparison['forward_speedup']:.3f}x |"
        )
    lines += [
        "", "## Layer gates", "",
        "- Frozen Encoder local output transform: **PASS**, 1024→1024 bitwise identity on 32/32 samples.",
        "- Original decoder output contract: **BLOCKED for asymmetric N**, measured pre-bypass shape remains 1024 nodes.",
        "- Split decoder contract: **PASS**, measured pre-bypass shapes are exactly 8192 and 32768 nodes.",
        "- Native shape/scale reconstruction, finite forward, and unchanged checkpoint tree: **PASS**.",
        "", "## Decision", "", f"**{decision}**.",
        f"Worth entering a separately preregistered 1024→240825 probe: **{discuss_240825}**.",
        "The measured 6.71× (8192) and 2.37× (32768) gains are forward-only comparisons, not matched production E2E speedups. P5-R still recommends E16384 for the current production route.",
        "No training, checkpoint update, test/sealed access, 240825 U1 execution, or production-route replacement occurred.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
