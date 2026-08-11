#!/usr/bin/env python3
"""P5-S2 exact support-ordering closeout on frozen valid32."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_p5_a4_p2r_r2p as a4  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import (  # noqa: E402
    build_reconstruction_map,
    prepare_reconstruction_domain_partition,
)
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
    deterministic_nested_query_order,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)


CONTINUOUS_STAGES = (
    "support_ordering", "cv_redistribution", "regional_prepare", "coverage",
    "p2r", "r2r", "r2p", "packing", "graph_total",
    "reconstruction_map", "continuous_total",
)
ORDERING_STAGES = (
    "mask_seconds", "sha256_seconds", "sort_seconds",
    "inner_interleave_seconds", "outer_interleave_seconds",
)


def _block(metadata: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        metadata,
    )


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "p5s_baseline_json", "output_json",
        "output_csv", "output_md",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_candidate_execution":
        raise RuntimeError("P5-S2 protocol is not preregistered")
    baseline = json.loads(args.p5s_baseline_json.read_text(encoding="utf-8"))
    if baseline["status"] != "passed" or len(baseline["samples"]) != 64:
        raise RuntimeError("P5-S baseline is not frozen/complete")
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    runtime = highn._checkpoint_runtime(args)
    anchors = highn._valid_examples(highn._dataset(args), binding)
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    support_rows = {
        str(resolution): {
            row["sample_id"]: row for row in preflight["supports"][str(resolution)]
        }
        for resolution in (8192, 32768)
    }
    with h5py.File(args.full_fields, "r") as archive:
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
    boundaries = highn._boundaries(anchors[0], float(np.min(coords[:, 2])))
    geometry_started = time.perf_counter()
    geometry = prepare_nested_query_geometry_cache(
        full_coords=coords, full_control_volume=cv, full_layer_id=layer,
        layer_boundaries_m=boundaries,
    )
    geometry_seconds = time.perf_counter() - geometry_started
    reconstruction_partition = prepare_reconstruction_domain_partition(
        coords=coords, layer_id=layer, boundaries=boundaries
    )
    boundary_hash = array_sha256(boundaries)
    geometry_boundary_exact = geometry.static_hashes["layer_boundaries_m"] == boundary_hash
    key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    rows: list[dict[str, Any]] = []

    for resolution, route, factor in (
        (8192, "B8192_adaptive", 8.0),
        (32768, "E32768_adaptive", 128.0),
    ):
        for number, anchor in enumerate(anchors, start=1):
            frozen = highn._load_support(
                Path(support_rows[str(resolution)][anchor.sample_id]["support_file"])
            )
            anchor_indices = np.asarray(frozen["selected_indices"][:1024], dtype=np.int64)
            with np.load(args.artifact_root / "physics" / f"{anchor.sample_id}.npz") as physics:
                q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            graph_config = dict(runtime["graph_config"])
            graph_config.update(
                subsample_factor=factor,
                discrete_graph_backend="sparse_kdtree_v1",
                reuse_exact_p2r_for_r2p=True,
            )

            full_order, full_audit = deterministic_nested_query_order(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices,
                full_coords=coords, full_control_volume=cv, full_layer_id=layer,
                full_q=q, layer_boundaries_m=boundaries,
            )
            selected = np.asarray(full_order[:resolution], dtype=np.int64)
            continuous_started = time.perf_counter()
            ordering_started = time.perf_counter()
            candidate, candidate_audit = deterministic_nested_query_prefix(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices,
                full_q=q, target_count=resolution, geometry_cache=geometry,
            )
            ordering_seconds = time.perf_counter() - ordering_started
            cv_started = time.perf_counter()
            selected_cv, cv_audit = conservative_selected_control_volume(
                full_coords=coords, full_control_volume=cv, full_layer_id=layer,
                selected_indices=candidate,
            )
            cv_seconds = time.perf_counter() - cv_started
            support = dict(frozen)
            support["selected_indices"] = candidate
            support["operator_control_volume"] = selected_cv
            example = highn._query_example(anchor, support, coords)
            builder = Heat3DGraphBuilder(**graph_config)
            graph_started = time.perf_counter()
            metadata = builder.build_metadata(
                highn.runner._graph_coords_for_example(example, runtime["stats"]), key=key
            )
            _block(metadata)
            graph_seconds = time.perf_counter() - graph_started
            map_started = time.perf_counter()
            mapping, _ = build_reconstruction_map(
                coords=coords, layer_id=layer, boundaries=boundaries,
                support_indices=candidate, empty_domain_fallback="same_layer",
                prepared_partition=reconstruction_partition, query_workers=-1,
            )
            map_seconds = time.perf_counter() - map_started
            build = builder.builder.last_build_timings
            stages = {
                "support_ordering": ordering_seconds,
                "cv_redistribution": cv_seconds,
                "regional_prepare": float(build["regional_prepare_seconds"]),
                "coverage": float(build["coverage_radius_seconds"]),
                "p2r": float(build["p2r_seconds"]),
                "r2r": float(build["r2r_seconds"]),
                "r2p": float(build["r2p_seconds"]),
                "packing": float(build["packing_seconds"]),
                "graph_total": graph_seconds,
                "reconstruction_map": map_seconds,
                "continuous_total": time.perf_counter() - continuous_started,
            }
            profile: dict[str, float] = {}
            profiled_candidate, _ = deterministic_nested_query_prefix(
                sample_id=anchor.sample_id, anchor_indices=anchor_indices,
                full_q=q, target_count=resolution, geometry_cache=geometry,
                profile=profile,
            )
            reference_cv, reference_cv_audit = conservative_selected_control_volume(
                full_coords=coords, full_control_volume=cv, full_layer_id=layer,
                selected_indices=selected,
            )
            reference_support = dict(frozen)
            reference_support["selected_indices"] = selected
            reference_support["operator_control_volume"] = reference_cv
            reference_example = highn._query_example(anchor, reference_support, coords)
            reference_builder = Heat3DGraphBuilder(**graph_config)
            reference_metadata = reference_builder.build_metadata(
                highn.runner._graph_coords_for_example(reference_example, runtime["stats"]),
                key=key,
            )
            _block(reference_metadata)
            reference_mapping, _ = build_reconstruction_map(
                coords=coords, layer_id=layer, boundaries=boundaries,
                support_indices=selected, empty_domain_fallback="same_layer",
                prepared_partition=reconstruction_partition, query_workers=-1,
            )
            frozen_selected = np.asarray(frozen["selected_indices"], dtype=np.int64)
            gates = {
                "prefix_array_equal_historical": bool(np.array_equal(selected, candidate)),
                "prefix_array_equal_frozen_support": bool(np.array_equal(candidate, frozen_selected)),
                "profiled_prefix_equal_candidate": bool(np.array_equal(profiled_candidate, candidate)),
                "prefix_sha256_equal": (
                    candidate_audit["prefix_sha256"] == array_sha256(selected)
                ),
                "anchor_prefix_exact": bool(np.array_equal(candidate[:1024], anchor_indices)),
                "geometry_boundary_hash_exact": geometry_boundary_exact,
                "cv_exact_historical_and_frozen": (
                    np.array_equal(selected_cv, reference_cv)
                    and cv_audit["weights_sha256"] == reference_cv_audit["weights_sha256"]
                    and cv_audit["weights_sha256"] == array_sha256(
                        np.asarray(frozen["operator_control_volume"], dtype=np.float64)
                    )
                ),
                "canonical_graph_hash_exact_historical": (
                    a4._canonical_hash(metadata) == a4._canonical_hash(reference_metadata)
                ),
                "reconstruction_map_exact_historical": (
                    highn._mapping_sha256(mapping) == highn._mapping_sha256(reference_mapping)
                ),
            }
            if not all(gates.values()):
                raise RuntimeError(f"P5-S2 fail-fast: {route}/{anchor.sample_id}: {gates}")
            rows.append({
                "route": route, "resolution": resolution, "sample_id": anchor.sample_id,
                "historical_full_order_sha256": full_audit["order_sha256"],
                "prefix_sha256": candidate_audit["prefix_sha256"],
                "gates": gates, "ordering_profile": profile, "stages": stages,
            })
            print(f"[P5-S2] {route} {number}/32", flush=True)

    summary: dict[str, Any] = {}
    for route in ("B8192_adaptive", "E32768_adaptive"):
        selected_rows = [row for row in rows if row["route"] == route]
        stage_stats = {
            stage: _stats([row["stages"][stage] for row in selected_rows])
            for stage in CONTINUOUS_STAGES
        }
        ordering_stats = {
            stage: _stats([row["ordering_profile"][stage] for row in selected_rows])
            for stage in ORDERING_STAGES
        }
        previous = baseline["summary"][route]["stages"]
        previous_total = previous["continuous_total"]["candidate"]["median_seconds"]
        previous_ordering = previous["support_ordering"]["candidate"]["median_seconds"]
        ranked = sorted(
            (
                (stage, values["median_seconds"])
                for stage, values in stage_stats.items()
                if stage not in {"continuous_total", "graph_total"}
            ),
            key=lambda item: item[1], reverse=True,
        )
        summary[route] = {
            "continuous_stages": stage_stats,
            "ordering_profile": ordering_stats,
            "p5s_baseline_continuous_median_seconds": previous_total,
            "p5s2_continuous_median_seconds": stage_stats["continuous_total"]["median_seconds"],
            "continuous_speedup_vs_p5s": previous_total / stage_stats["continuous_total"]["median_seconds"],
            "p5s_baseline_ordering_median_seconds": previous_ordering,
            "p5s2_ordering_median_seconds": stage_stats["support_ordering"]["median_seconds"],
            "ordering_speedup_vs_p5s": previous_ordering / stage_stats["support_ordering"]["median_seconds"],
            "remaining_bottleneck": ranked[0][0],
            "remaining_bottleneck_median_seconds": ranked[0][1],
        }

    result = {
        "schema_version": "heat3d_v6_p1i_p5s2_support_ordering_closeout_v1",
        "status": "passed",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "baseline_artifact_sha256": hashlib.sha256(args.p5s_baseline_json.read_bytes()).hexdigest(),
        "geometry_cache_prepare_seconds": geometry_seconds,
        "geometry_static_hashes": dict(geometry.static_hashes),
        "hard_gate_passed": True,
        "summary": summary,
        "samples": rows,
        "decision": {
            "implementation": "GO_exact_cached_lazy_interleave",
            "further_python_optimization": "STOP_after_profile",
            "stop_reason": "remaining work is dominated by SHA256/sort or would require C++/semantic change",
            "c_cpp": "NOT_IMPLEMENTED",
            "approximate_q_cluster_cache": "NOT_IMPLEMENTED",
        },
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "route", "stage", "median_seconds", "mean_seconds", "p95_seconds",
            "p5s_baseline_seconds", "speedup_vs_p5s",
        ])
        for route, route_result in summary.items():
            for stage, values in route_result["continuous_stages"].items():
                baseline_seconds = ""
                speedup = ""
                if stage == "support_ordering":
                    baseline_seconds = route_result["p5s_baseline_ordering_median_seconds"]
                    speedup = route_result["ordering_speedup_vs_p5s"]
                elif stage == "continuous_total":
                    baseline_seconds = route_result["p5s_baseline_continuous_median_seconds"]
                    speedup = route_result["continuous_speedup_vs_p5s"]
                writer.writerow([
                    route, stage, values["median_seconds"], values["mean_seconds"],
                    values["p95_seconds"], baseline_seconds, speedup,
                ])
            for stage, values in route_result["ordering_profile"].items():
                writer.writerow([
                    route, f"ordering::{stage}", values["median_seconds"],
                    values["mean_seconds"], values["p95_seconds"], "", "",
                ])
    lines = [
        "# V6 P1i P5-S2 support-ordering closeout", "",
        "Status: **PASS**. All 64 valid32 route/sample prefixes and downstream exact gates passed.", "",
    ]
    for route, route_result in summary.items():
        lines += [
            f"## {route}", "",
            "| Quantity | P5-S median (s) | P5-S2 median (s) | Speedup |",
            "|---|---:|---:|---:|",
            f"| support ordering | {route_result['p5s_baseline_ordering_median_seconds']:.6f} | {route_result['p5s2_ordering_median_seconds']:.6f} | {route_result['ordering_speedup_vs_p5s']:.3f}x |",
            f"| continuous preprocessing | {route_result['p5s_baseline_continuous_median_seconds']:.6f} | {route_result['p5s2_continuous_median_seconds']:.6f} | {route_result['continuous_speedup_vs_p5s']:.3f}x |",
            "", "### Ordering profile", "",
            "| Stage | median (s) | p95 (s) |", "|---|---:|---:|",
        ]
        for stage, values in route_result["ordering_profile"].items():
            lines.append(f"| {stage} | {values['median_seconds']:.6f} | {values['p95_seconds']:.6f} |")
        lines += [
            "",
            f"Remaining continuous-stage bottleneck: `{route_result['remaining_bottleneck']}` "
            f"({route_result['remaining_bottleneck_median_seconds']:.6f} s).", "",
        ]
    lines += [
        "## Decision", "",
        "The cached lazy-interleave implementation is retained. Further Python micro-optimization stops: "
        "remaining ordering time is attributed by the frozen profile, and C/C++ or semantic/approximate changes are out of scope.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
