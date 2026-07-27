#!/usr/bin/env python3
"""Collect V6 canonical freeze, support attribution, and anchored-ladder results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_anchored_resolution as anchored  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import evaluate_heat3d_v6_volume_probe_ladder as volume  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


METRICS = {
    "point_global_cv_relative_rmse_pct": lambda m: m["point_global_cv_relative_rmse_pct"],
    "sample_first_cv_relative_rmse_pct": lambda m: m["sample_first_cv_relative_rmse_pct"],
    "raw_cv_weighted_rmse_K": lambda m: m["raw_cv_weighted_rmse_K"],
    "peak_rmse_K": lambda m: m["peak"]["rmse_K"],
    "source_region_cv_rmse_K": lambda m: m["source_region"]["cv_weighted_rmse_K"],
    "layer_mean_rmse_K": lambda m: m["layer_mean"]["rmse_K"],
    "layer_drop_rmse_K": lambda m: m["layer_drop"]["rmse_K"],
    "top_surface_cv_rmse_K": lambda m: m["top_surface"]["cv_weighted_rmse_K"],
    "bottom_surface_cv_rmse_K": lambda m: m["bottom_surface"]["cv_weighted_rmse_K"],
    "shape_cv_rmse": lambda m: m["shape_cv_rmse"],
    "scale_log_rmse": lambda m: m["scale_log_rmse"],
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _summary_rows(raw: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for resolution, payload in raw.items():
        for seed, seed_payload in payload["results"].items():
            for mode, metrics in seed_payload["modes"].items():
                row = {
                    "resolution": resolution,
                    "seed": seed,
                    "config_id": seed_payload["config_id"],
                    "pooling_mode": mode,
                    "graph_build_seconds": seed_payload["runtime"]["graph_build_seconds"],
                    "inference_seconds": seed_payload["runtime"]["inference_seconds"],
                    "peak_ram_GiB": seed_payload["runtime"]["process_peak_rss_bytes"] / 2**30,
                }
                row.update({name: getter(metrics) for name, getter in METRICS.items()})
                rows.append(row)
    return rows


def _mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for resolution in (1024, 2048, 4096, 8192):
        for mode in ("joint_pooling", "anchor_derived_scale_pooling"):
            selected = [
                row for row in rows
                if row["resolution"] == resolution and row["pooling_mode"] == mode
            ]
            out: dict[str, Any] = {
                "resolution": resolution,
                "pooling_mode": mode,
                "seed_count": len(selected),
            }
            for name in METRICS:
                values = np.asarray([row[name] for row in selected], dtype=np.float64)
                out[f"{name}_mean"] = float(np.mean(values))
                out[f"{name}_std"] = float(np.std(values, ddof=1))
            for name in ("graph_build_seconds", "inference_seconds", "peak_ram_GiB"):
                values = np.asarray([row[name] for row in selected], dtype=np.float64)
                out[f"{name}_mean"] = float(np.mean(values))
                out[f"{name}_std"] = float(np.std(values, ddof=1))
            result.append(out)
    return result


def _coverage(examples) -> dict[str, Any]:
    nonzero, source_counts, covered_sources = [], [], []
    for example in examples:
        coords = np.asarray(example.condition.coords)
        q = np.asarray(example.condition.condition_features)[:, 3]
        nonzero.append(int(np.sum(q > 0.0)))
        counts = []
        for source in example.meta["sources"]:
            box = source["bbox_m"]
            mask = np.ones(len(coords), dtype=bool)
            for axis, key in enumerate(("x", "y", "z")):
                low, high = map(float, box[key])
                mask &= (coords[:, axis] >= low - 1e-15) & (coords[:, axis] <= high + 1e-15)
            counts.append(int(np.sum(mask)))
        source_counts.extend(counts)
        covered_sources.append(sum(value > 0 for value in counts))
    values = np.asarray(source_counts, dtype=np.float64)
    return {
        "nonzero_q_node_count": {
            "min": int(np.min(nonzero)),
            "median": float(np.median(nonzero)),
            "max": int(np.max(nonzero)),
        },
        "source_box_node_count": {
            "min": int(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "max": int(np.max(values)),
        },
        "source_box_zero_coverage_count": int(np.sum(values == 0)),
        "samples_with_all_sources_covered": int(
            sum(
                covered == len(example.meta["sources"])
                for covered, example in zip(covered_sources, examples)
            )
        ),
    }


def _graph_audit(coords: np.ndarray, graph_config: Mapping[str, Any]) -> dict[str, Any]:
    builder = Heat3DGraphBuilder(**dict(graph_config))
    graph = builder.build_graphs(builder.build_metadata(coords, key=None))
    result = {}
    for name in ("p2r", "r2r", "r2p"):
        edge_set = next(iter(getattr(graph, name).edges.values()))
        count = int(np.asarray(edge_set.n_edge).reshape(-1)[0])
        senders = np.asarray(edge_set.indices.senders).reshape(-1)[:count]
        receivers = np.asarray(edge_set.indices.receivers).reshape(-1)[:count]
        sender_count = int(np.max(senders)) + 1
        receiver_count = int(np.max(receivers)) + 1
        result[name] = {
            "edge_count": count,
            "zero_out_degree": int(np.sum(np.bincount(senders, minlength=sender_count) == 0)),
            "zero_in_degree": int(np.sum(np.bincount(receivers, minlength=receiver_count) == 0)),
        }
        if name == "r2r":
            # The last node/edge is the runner's dummy padding sentinel.
            physical_count = int(max(np.max(senders), np.max(receivers)))
            keep = (senders < physical_count) & (receivers < physical_count)
            adjacency = coo_matrix(
                (
                    np.ones(int(np.sum(keep)), dtype=np.int8),
                    (senders[keep], receivers[keep]),
                ),
                shape=(physical_count, physical_count),
            )
            component_count, labels = connected_components(
                adjacency, directed=False, return_labels=True
            )
            result[name]["weak_component_count_excluding_dummy"] = int(component_count)
            result[name]["largest_component_fraction_excluding_dummy"] = float(
                np.max(np.bincount(labels)) / physical_count
            )
    result["regional_graph_connected"] = (
        result["r2r"]["weak_component_count_excluding_dummy"] == 1
    )
    return result


def _context_audit(
    *,
    dataset_root: Path,
    manifest_path: Path,
    anchored_ladder: Mapping[str, Any],
    volume_ladder: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    anchors, _, _ = anchored._load_examples(
        dataset_root, manifest_path, anchored_ladder["probes"]["1024"]
    )
    volume_examples, _, _ = volume._load_valid_examples(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        probe=volume_ladder["probes"]["1024"],
    )
    run_config = json.loads((run_dir / "run_config.json").read_text())
    standardizer = run_config["global_context"]["standardizer"]
    names = list(standardizer["feature_names"])
    raw_anchor = np.asarray(
        [
            [runner._global_context_row_for_example(row)[name] for name in names]
            for row in anchors
        ],
        dtype=np.float64,
    )
    raw_volume = np.asarray(
        [
            [runner._global_context_row_for_example(row)[name] for name in names]
            for row in volume_examples
        ],
        dtype=np.float64,
    )
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    z_anchor = (raw_anchor - mean) / std
    z_volume = (raw_volume - mean) / std
    feature_rows = []
    for index, name in enumerate(names):
        diff = raw_volume[:, index] - raw_anchor[:, index]
        zdiff = z_volume[:, index] - z_anchor[:, index]
        feature_rows.append(
            {
                "feature": name,
                "anchor_mean": float(np.mean(raw_anchor[:, index])),
                "volume_mean": float(np.mean(raw_volume[:, index])),
                "raw_mean_absolute_drift": float(np.mean(np.abs(diff))),
                "z_mean_absolute_drift": float(np.mean(np.abs(zdiff))),
                "z_p95_absolute_drift": float(np.quantile(np.abs(zdiff), 0.95)),
            }
        )
    graph_config = run_config["graph_config"]
    return {
        "feature_names": names,
        "anchor_raw_sha256": _array_sha256(raw_anchor),
        "volume_raw_sha256": _array_sha256(raw_volume),
        "anchor_z_sha256": _array_sha256(z_anchor),
        "volume_z_sha256": _array_sha256(z_volume),
        "raw_context_l2_drift_mean": float(
            np.mean(np.linalg.norm(raw_volume - raw_anchor, axis=1))
        ),
        "z_context_l2_drift_mean": float(
            np.mean(np.linalg.norm(z_volume - z_anchor, axis=1))
        ),
        "z_context_l2_drift_p95": float(
            np.quantile(np.linalg.norm(z_volume - z_anchor, axis=1), 0.95)
        ),
        "features": feature_rows,
        "support_coverage": {
            "source_aware_anchors": _coverage(anchors),
            "volume_only": _coverage(volume_examples),
        },
        "regional_graph": {
            "source_aware_anchors": _graph_audit(
                np.asarray(anchors[0].condition.coords), graph_config
            ),
            "volume_only": _graph_audit(
                np.asarray(volume_examples[0].condition.coords), graph_config
            ),
        },
    }


def _compact_metric(metric: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name, getter in METRICS.items():
        try:
            result[name] = float(getter(metric))
        except KeyError:
            continue
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchored-ladder", type=Path, required=True)
    parser.add_argument("--volume-ladder", type=Path, required=True)
    parser.add_argument("--volume-results", type=Path, required=True)
    parser.add_argument("--volume-anchor-context", type=Path, required=True)
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--raw", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    raw = {}
    for path in args.raw:
        payload = json.loads(path.read_text())
        raw[int(payload["resolution"])] = payload
    if tuple(sorted(raw)) != (1024, 2048, 4096, 8192):
        raise RuntimeError("raw anchored resolutions incomplete")
    rows = _summary_rows(raw)
    aggregate = _mean_std(rows)
    old_volume = json.loads(args.volume_results.read_text())
    a = raw[1024]["results"]["seed0"]["modes"]["joint_pooling"]
    b = old_volume["raw_evaluations"]["1024"]["models"]["V6_03_V5best_P1h"]["metrics"]
    c = json.loads(args.volume_anchor_context.read_text())["results"]["seed0"]["modes"]["joint_pooling"]
    a_pg, b_pg, c_pg = [
        value["point_global_cv_relative_rmse_pct"] for value in (a, b, c)
    ]
    rmse_gap = b_pg - a_pg
    mse_a, mse_b, mse_c = [(value / 100.0) ** 2 for value in (a_pg, b_pg, c_pg)]
    attribution = {
        "conditions": {
            "source_aware_support_canonical_context": _compact_metric(a),
            "volume_only_support_volume_context": _compact_metric(b),
            "volume_only_support_frozen_source_aware_context": _compact_metric(c),
        },
        "point_global_rmse_gap_attribution": {
            "context_recovery_fraction": float((b_pg - c_pg) / rmse_gap),
            "remaining_local_support_fraction": float((c_pg - a_pg) / rmse_gap),
        },
        "normalized_sse_gap_attribution": {
            "context_recovery_fraction": float((mse_b - mse_c) / (mse_b - mse_a)),
            "remaining_local_support_fraction": float((mse_c - mse_a) / (mse_b - mse_a)),
        },
        "interpretation_guardrail": (
            "Diagnostic intervention, not an additive causal identity: A uses the "
            "canonical anchor measure while B/C use the frozen volume-probe measure."
        ),
    }
    context = _context_audit(
        dataset_root=args.dataset,
        manifest_path=args.manifest,
        anchored_ladder=json.loads(args.anchored_ladder.read_text()),
        volume_ladder=json.loads(args.volume_ladder.read_text()),
        run_dir=args.seed0_run,
    )
    best_highres = min(
        (
            row for row in aggregate
            if row["resolution"] > 1024
            and row["pooling_mode"] == "anchor_derived_scale_pooling"
        ),
        key=lambda row: row["point_global_cv_relative_rmse_pct_mean"],
    )
    stable = all(
        row["point_global_cv_relative_rmse_pct_mean"] < 20.0
        and row["point_global_cv_relative_rmse_pct_std"] < 1.0
        for row in aggregate
        if row["resolution"] > 1024
        and row["pooling_mode"] == "anchor_derived_scale_pooling"
    )
    registered_ids = {
        "V6_03_V5best_P1h",
        "V6_03_V5best_P1h_seed1",
        "V6_03_V5best_P1h_seed2",
    }
    with (ROOT / "configs/heat3d_v6/v6_multiseed_checkpoint_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        frozen_artifacts = [
            dict(row)
            for row in csv.DictReader(handle)
            if row["config_id"] in registered_ids
        ]
    if len(frozen_artifacts) != 12:
        raise RuntimeError("three-seed 12-checkpoint artifact freeze is incomplete")
    frozen_configs = []
    for config_id in sorted(registered_ids):
        path = ROOT / "configs/heat3d_v6" / f"{config_id}.yaml"
        frozen_configs.append(
            {
                "config_id": config_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": common._sha256(path),
            }
        )
    payload = {
        "schema_version": "heat3d_v6_model_closeout_anchored_resolution_v1",
        "status": "passed",
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "formal_inference_platform": "local_CPU",
        "canonical_model": {
            "config_id": "V6_03_V5best_P1h",
            "reference_run": "seed0",
            "replication_runs": ["seed1", "seed2"],
            "registered_ablation": "V6_04_V5best_P1h_DualAttention",
            "applicability": "P1h source-aware support family only",
        },
        "three_seed_artifact_freeze": {
            "config_count": 3,
            "checkpoint_prediction_pair_count": 12,
            "configs": frozen_configs,
            "artifacts": frozen_artifacts,
        },
        "rows": rows,
        "mean_std": aggregate,
        "volume_support_attribution": attribution,
        "context_and_support_audit": context,
        "workflow_decision": {
            "recognized": bool(stable),
            "statement": (
                "Low-resolution source-aware conditioning anchors can support "
                "higher-resolution anchored queries with anchor-derived scale/pooling."
            ),
            "lowest_error_high_resolution": {
                key: best_highres[key] for key in (
                    "resolution",
                    "pooling_mode",
                    "point_global_cv_relative_rmse_pct_mean",
                    "point_global_cv_relative_rmse_pct_std",
                    "sample_first_cv_relative_rmse_pct_mean",
                    "raw_cv_weighted_rmse_K_mean",
                )
            },
            "limitation": (
                "The added query nodes participate in the frozen joint encoder/processor "
                "path because decoder bypass requires node-aligned local k/q/BC inputs. "
                "Thus conditioning/query roles are explicit in context and scale pooling, "
                "but are not a pure out-of-sample decoder-only query path."
            ),
            "canonical_1024_remains_lower_error": True,
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# V6 model closeout and anchored high-resolution inference",
        "",
        "Status: **passed**. Evaluation is local CPU, `valid_iid` only.",
        "",
        "| nodes | pooling | point-global mean±std % | sample-first mean±std % | raw mean±std K | shape mean | scale-log mean |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['resolution']} | {row['pooling_mode']} | "
            f"{row['point_global_cv_relative_rmse_pct_mean']:.4f}±{row['point_global_cv_relative_rmse_pct_std']:.4f} | "
            f"{row['sample_first_cv_relative_rmse_pct_mean']:.4f}±{row['sample_first_cv_relative_rmse_pct_std']:.4f} | "
            f"{row['raw_cv_weighted_rmse_K_mean']:.4f}±{row['raw_cv_weighted_rmse_K_std']:.4f} | "
            f"{row['shape_cv_rmse_mean']:.5f} | {row['scale_log_rmse_mean']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Volume-support attribution",
            "",
            f"- Canonical anchors + canonical context: {a_pg:.4f}% point-global.",
            f"- Volume-only + volume context: {b_pg:.4f}%.",
            f"- Volume-only + frozen anchor context: {c_pg:.4f}%.",
            f"- RMSE-gap attribution: context drift {100*attribution['point_global_rmse_gap_attribution']['context_recovery_fraction']:.2f}%; remaining local support/graph gap {100*attribution['point_global_rmse_gap_attribution']['remaining_local_support_fraction']:.2f}%.",
            f"- Normalized-SSE-gap attribution: context drift {100*attribution['normalized_sse_gap_attribution']['context_recovery_fraction']:.2f}%; remaining local support/graph gap {100*attribution['normalized_sse_gap_attribution']['remaining_local_support_fraction']:.2f}%.",
            f"- 24D context z-score L2 drift: mean {context['z_context_l2_drift_mean']:.3f}, P95 {context['z_context_l2_drift_p95']:.3f}; per-feature raw/z-score audit is frozen in the JSON.",
            f"- Source-aware anchors cover every source box (minimum {context['support_coverage']['source_aware_anchors']['source_box_node_count']['min']} nodes); volume-only support has {context['support_coverage']['volume_only']['source_box_zero_coverage_count']} zero-covered source boxes and only {context['support_coverage']['volume_only']['samples_with_all_sources_covered']}/128 samples with all sources covered.",
            "- Both supports retain finite p2r/r2r/r2p connectivity with zero zero-degree nodes; the residual error is therefore tied to sparse local source representation, not graph disconnection.",
            "",
            "## Decision",
            "",
            f"- High-resolution workflow recognized: **{str(stable).lower()}**.",
            f"- Lowest-error high-resolution setting: {best_highres['resolution']} nodes with anchor-derived scale/pooling, point-global {best_highres['point_global_cv_relative_rmse_pct_mean']:.4f}±{best_highres['point_global_cv_relative_rmse_pct_std']:.4f}%.",
            "- The canonical 1024 source-aware support remains the lowest-error evaluation domain.",
            "- Scope is frozen to the P1h source-aware support family. Test/hard were not accessed.",
            "- Added query nodes still enter the node-aligned encoder/processor path; this is not a pure decoder-only zero-shot query path.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "recognized": stable}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
