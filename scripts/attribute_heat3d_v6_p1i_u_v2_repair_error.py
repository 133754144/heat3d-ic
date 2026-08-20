#!/usr/bin/env python3
"""Attribute frozen valid96 U-v2 error by output-side R2P repair class."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "query_padding_result", "population_preflight",
        "geometry_audit", "output_json", "output_csv",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--prediction", action="append", required=True, help="seed=predictions.npz")
    return parser.parse_args()


def metric(bucket: dict[str, float]) -> dict[str, float | int | None]:
    if bucket["volume"] <= 0.0:
        return {"node_count": int(bucket["count"]), "volume_fraction": 0.0,
                "cv_rmse_K": None, "cv_bias_K": None, "weighted_sse_fraction": 0.0}
    return {
        "node_count": int(bucket["count"]),
        "volume_fraction": float(bucket["volume"] / bucket["total_volume"]),
        "cv_rmse_K": float(np.sqrt(bucket["sse"] / bucket["volume"])),
        "cv_bias_K": float(bucket["signed"] / bucket["volume"]),
        "weighted_sse_fraction": float(bucket["sse"] / bucket["total_sse"]),
    }


def main() -> int:
    args = parse()
    protocol = json.loads(args.protocol.read_text())
    binding = json.loads(args.binding.read_text())
    geometry = json.loads(args.geometry_audit.read_text())
    geometry_by_id = {row["sample_id"]: row for row in geometry["rows"]}
    args.padding_result = args.query_padding_result
    runtime = p5r._runtime(args)
    dataset = highn._dataset(args)
    ordered = sorted(dataset.split_ids["valid_iid"], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if ordered[:32] != binding["development_subset"]["sample_ids"]:
        raise RuntimeError("valid32 subset drift")
    index = dataset.sample_index_by_id()
    anchors = [dataset[index[sample_id]] for sample_id in ordered[32:]]
    preflight = json.loads(args.population_preflight.read_text())
    sample_ids = [anchor.sample_id for anchor in anchors]
    if preflight["sample_ids"] != sample_ids or geometry["sample_ids"] != sample_ids:
        raise RuntimeError("valid96 population/order drift")

    predictions: dict[str, np.ndarray] = {}
    prediction_artifacts = []
    for spec in args.prediction:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as payload:
            if payload["sample_ids"].astype(str).tolist() != sample_ids:
                raise RuntimeError(f"{name}: prediction population/order drift")
            predictions[name] = np.asarray(payload["full_deltaT_K"], dtype=np.float64)
        prediction_artifacts.append({"seed": name, "path": str(path), "sha256": file_sha256(path)})

    full, archive_lookup = highn._full_shared(args)
    coords = np.asarray(full["coords"], dtype=np.float64)
    cv = np.asarray(full["cv"], dtype=np.float64)
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    graph_config = dict(runtime["graph_config"])
    graph_config.update(subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1",
                        reuse_exact_p2r_for_r2p=True)
    tolerance = float(protocol["u_v2"]["normalized_numerical_tolerance"])
    categories = ("covered", "repaired_inside", "repaired_outside")
    aggregate: dict[str, dict[str, dict[str, float]]] = {
        seed: {category: {"count": 0.0, "volume": 0.0, "sse": 0.0, "signed": 0.0}
               for category in categories}
        for seed in predictions
    }
    repaired_distance: list[np.ndarray] = []
    repaired_error: dict[str, list[np.ndarray]] = {seed: [] for seed in predictions}
    raw_uncovered_total = 0
    outside_total = 0

    with h5py.File(args.full_fields, "r") as archive:
        for number, anchor in enumerate(anchors):
            builder = Heat3DGraphBuilder(**graph_config)
            anchor_coords = highn.runner._graph_coords_for_example(anchor, runtime["stats"])
            native = builder.build_metadata(anchor_coords, key=graph_key)
            support = {
                "selected_indices": np.arange(len(coords), dtype=np.int64),
                "operator_control_volume": cv,
                "k_xyz": np.zeros((len(coords), 3), dtype=np.float64),
                "q_W_m3": np.zeros(len(coords), dtype=np.float64),
                "layer_id": np.asarray(full["layer"], dtype=np.int32),
            }
            query = highn._query_example(anchor, support, coords)
            query_coords = highn.runner._graph_coords_for_example(query, runtime["stats"])
            centers = np.asarray(native.x_rnodes)[0, :-1]
            base_radii = np.asarray(native.r_rnodes)[0, :-1]
            lower = np.asarray(anchor_coords, dtype=np.float64).min(axis=0)
            upper = np.asarray(anchor_coords, dtype=np.float64).max(axis=0)
            normalized = 2.0 * (np.asarray(query_coords, dtype=np.float64) - lower) / (upper - lower) - 1.0
            impl = builder.builder
            raw = np.asarray(impl._get_supported_pnodes_by_rnodes(
                centers=centers,
                points=normalized,
                radii=impl._get_effective_support_radii(base_radii, impl.overlap_factor_r2p),
                apply_legacy_hard_reset=(impl.radius_policy == "legacy_kdtree_mean4"),
            ))
            degree = np.bincount(raw[:, 0], minlength=len(coords))
            covered = degree >= 1
            outside = np.any((normalized < (-1.0 - tolerance)) | (normalized > (1.0 + tolerance)), axis=1)
            repaired = ~covered
            masks = {
                "covered": covered,
                "repaired_inside": repaired & ~outside,
                "repaired_outside": repaired & outside,
            }
            raw_uncovered_total += int(np.sum(repaired))
            outside_total += int(np.sum(outside))
            frozen = geometry_by_id[anchor.sample_id]
            if int(np.sum(repaired)) != int(frozen["raw_uncovered_count"]):
                raise RuntimeError(f"{anchor.sample_id}: raw coverage drift")
            if int(np.sum(outside)) != int(frozen["outside_node_count"]):
                raise RuntimeError(f"{anchor.sample_id}: outside classification drift")
            uncovered_points = normalized[repaired]
            if len(uncovered_points):
                distances = np.linalg.norm(uncovered_points[:, None, :] - centers[None, :, :], axis=-1)
                nearest_distance = np.min(distances, axis=1)
                repaired_distance.append(nearest_distance)
            truth = np.asarray(archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]], dtype=np.float64)
            for seed, fields in predictions.items():
                error = fields[number] - truth
                if len(uncovered_points):
                    repaired_error[seed].append(np.abs(error[repaired]))
                total_sse = float(np.sum(cv * error * error))
                total_volume = float(np.sum(cv))
                for category, mask in masks.items():
                    bucket = aggregate[seed][category]
                    bucket["count"] += int(np.sum(mask))
                    bucket["volume"] += float(np.sum(cv[mask]))
                    bucket["sse"] += float(np.sum(cv[mask] * error[mask] ** 2))
                    bucket["signed"] += float(np.sum(cv[mask] * error[mask]))
                    bucket["total_sse"] = bucket.get("total_sse", 0.0) + total_sse
                    bucket["total_volume"] = bucket.get("total_volume", 0.0) + total_volume
            print(f"[U-v2 repair attribution] {number + 1}/96", flush=True)

    all_distance = np.concatenate(repaired_distance)
    quantiles = np.quantile(all_distance, [0.0, 0.25, 0.5, 0.75, 1.0])
    summary: dict[str, Any] = {}
    csv_rows = []
    for seed in predictions:
        category_metrics = {category: metric(aggregate[seed][category]) for category in categories}
        errors = np.concatenate(repaired_error[seed])
        pearson = pearsonr(all_distance, errors)
        spearman = spearmanr(all_distance, errors)
        bins = []
        for bin_index in range(4):
            lo, hi = quantiles[bin_index], quantiles[bin_index + 1]
            mask = (all_distance >= lo) & ((all_distance <= hi) if bin_index == 3 else (all_distance < hi))
            bins.append({
                "bin": bin_index + 1, "distance_min": float(lo), "distance_max": float(hi),
                "node_count": int(np.sum(mask)), "absolute_error_mean_K": float(np.mean(errors[mask])),
                "absolute_error_p95_K": float(np.quantile(errors[mask], 0.95)),
            })
        summary[seed] = {
            "categories": category_metrics,
            "repair_distance_vs_absolute_error": {
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
            },
            "repair_distance_bins": bins,
        }
        for category, values in category_metrics.items():
            csv_rows.append({"seed": seed, "partition": category, **values})

    result = {
        "schema_version": "heat3d_v6_p1i_u_v2_repair_error_attribution_v1",
        "status": "passed",
        "population": "frozen_valid96_diagnostic_characterization",
        "sample_count": 96,
        "classification": {
            "covered": "raw radius R2P degree >= 1",
            "repaired_inside": "raw degree 0 and query within native bbox tolerance",
            "repaired_outside": "raw degree 0 and query outside native bbox tolerance",
            "numerical_tolerance": tolerance,
        },
        "raw_uncovered_node_count": raw_uncovered_total,
        "outside_node_count": outside_total,
        "repair_distance_quantiles": quantiles.tolist(),
        "summary": summary,
        "artifacts": {
            "geometry_audit": {"path": str(args.geometry_audit), "sha256": file_sha256(args.geometry_audit)},
            "predictions": prediction_artifacts,
        },
        "role_contract": protocol["role_contract"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": "passed", "samples": 96, "seeds": list(predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
