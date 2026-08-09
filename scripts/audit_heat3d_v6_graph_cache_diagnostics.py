#!/usr/bin/env python3
"""Offline P1i/P1h cached-graph diagnostics without model inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


EDGE_FIELDS = ("p2r_edge_indices", "r2r_edge_indices", "r2p_edge_indices")


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        none = {
            value.decode("utf-8")
            for value in np.asarray(payload["__none_fields_utf8"]).tolist()
        }
        return {
            field: None if field in none else np.asarray(payload[field])
            for field in (
                "x_pnodes_inp", "x_pnodes_out", "x_rnodes", "r_rnodes",
                "p2r_edge_indices", "r2r_edge_indices", "r2p_edge_indices",
            )
        }


def _physical(normalized: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (normalized + 1.0) * 0.5 * (upper - lower) + lower


def _real_edges(value: np.ndarray | None, n_sender: int, n_receiver: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(value)[0].astype(np.int64)
    return edges[(edges[:, 0] < n_sender) & (edges[:, 1] < n_receiver)]


def _categories(
    coords: np.ndarray,
    q: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    source = np.asarray(q).reshape(-1) > 0.0
    interface = np.any(
        np.isclose(coords[:, 2, None], boundaries[None, 1:-1], atol=1.0e-15),
        axis=1,
    )
    result = np.full(len(coords), "background", dtype="U10")
    result[interface] = "interface"
    result[source] = "source"
    return result


def _one_graph(
    cache_file: Path,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    metadata = _load_npz(cache_file)
    x_p = np.asarray(metadata["x_pnodes_inp"])[0, :-1]
    x_r = np.asarray(metadata["x_rnodes"])[0, :-1]
    radii = np.asarray(metadata["r_rnodes"])[0, :-1]
    n_p, n_r = len(x_p), len(x_r)
    if len(categories) != n_p:
        raise RuntimeError(f"{cache_file}: category/support length mismatch")
    p2r = _real_edges(metadata["p2r_edge_indices"], n_p, n_r)
    r2r = _real_edges(metadata["r2r_edge_indices"], n_r, n_r)
    r2p_value = metadata["r2p_edge_indices"]
    r2p = (
        _real_edges(r2p_value, n_r, n_p)
        if r2p_value is not None
        else np.flip(p2r, axis=1)
    )
    p_phys = _physical(x_p, lower, upper)
    r_phys = _physical(x_r, lower, upper)
    p2r_length = np.linalg.norm(p_phys[p2r[:, 0]] - r_phys[p2r[:, 1]], axis=1)
    r2r_length = np.linalg.norm(r_phys[r2r[:, 0]] - r_phys[r2r[:, 1]], axis=1)
    r2p_length = np.linalg.norm(r_phys[r2p[:, 0]] - p_phys[r2p[:, 1]], axis=1)
    p2r_p_degree = np.bincount(p2r[:, 0], minlength=n_p)
    p2r_r_degree = np.bincount(p2r[:, 1], minlength=n_r)
    r2r_out = np.bincount(r2r[:, 0], minlength=n_r)
    r2r_in = np.bincount(r2r[:, 1], minlength=n_r)
    r2p_r_degree = np.bincount(r2p[:, 0], minlength=n_r)
    r2p_p_degree = np.bincount(r2p[:, 1], minlength=n_p)
    observed_radius = np.zeros(n_r, dtype=np.float64)
    np.maximum.at(observed_radius, p2r[:, 1], p2r_length)
    partition: dict[str, Any] = {}
    for name in ("source", "interface", "background"):
        mask = categories == name
        partition[name] = {
            "physical_node_count": int(np.count_nonzero(mask)),
            "p2r_degree": _distribution(p2r_p_degree[mask]),
            "r2p_degree": _distribution(r2p_p_degree[mask]),
        }
    return {
        "cache_file": str(cache_file),
        "physical_node_count": n_p,
        "regional_node_count": n_r,
        "edge_count": {"p2r": int(len(p2r)), "r2r": int(len(r2r)), "r2p": int(len(r2p))},
        "degree": {
            "p2r_physical": _distribution(p2r_p_degree),
            "p2r_regional": _distribution(p2r_r_degree),
            "r2r_out": _distribution(r2r_out),
            "r2r_in": _distribution(r2r_in),
            "r2p_regional": _distribution(r2p_r_degree),
            "r2p_physical": _distribution(r2p_p_degree),
        },
        "coverage": {
            "p2r_zero_degree_nodes": int(np.count_nonzero(p2r_p_degree == 0)),
            "r2p_zero_degree_nodes": int(np.count_nonzero(r2p_p_degree == 0)),
            "p2r_min_degree": int(np.min(p2r_p_degree)),
            "r2p_min_degree": int(np.min(r2p_p_degree)),
        },
        "normalized_regional_radius": _distribution(radii),
        "observed_physical_support_radius_m": _distribution(observed_radius),
        "physical_edge_length_m": {
            "p2r": _distribution(p2r_length),
            "r2r": _distribution(r2r_length),
            "r2p": _distribution(r2p_length),
        },
        "partition": partition,
    }


def _aggregate(rows: list[dict[str, Any]], *, family: str, resolution: int) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> np.ndarray:
        result = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(value)
        return np.asarray(result, dtype=np.float64)

    summary: dict[str, Any] = {
        "family": family,
        "resolution": resolution,
        "graph_count": len(rows),
        "regional_node_count": _distribution(values(("regional_node_count",))),
        "edge_count": {
            name: _distribution(values(("edge_count", name))) for name in ("p2r", "r2r", "r2p")
        },
        "coverage": {
            key: _distribution(values(("coverage", key)))
            for key in ("p2r_zero_degree_nodes", "r2p_zero_degree_nodes", "p2r_min_degree", "r2p_min_degree")
        },
        "normalized_regional_radius": {
            key: float(np.mean(values(("normalized_regional_radius", key))))
            for key in ("mean", "median", "p95", "max")
        },
        "observed_physical_support_radius_m": {
            key: float(np.mean(values(("observed_physical_support_radius_m", key))))
            for key in ("mean", "median", "p95", "max")
        },
        "physical_edge_length_m": {},
        "degree": {},
        "partition": {},
    }
    for edge in ("p2r", "r2r", "r2p"):
        summary["physical_edge_length_m"][edge] = {
            key: float(np.mean(values(("physical_edge_length_m", edge, key))))
            for key in ("mean", "median", "p95", "max")
        }
    for degree in rows[0]["degree"]:
        summary["degree"][degree] = {
            key: float(np.mean(values(("degree", degree, key))))
            for key in ("mean", "median", "p95", "max")
        }
    for category in ("source", "interface", "background"):
        summary["partition"][category] = {
            "physical_node_count": _distribution(values(("partition", category, "physical_node_count"))),
            "p2r_degree_mean": float(np.mean(values(("partition", category, "p2r_degree", "mean")))),
            "r2p_degree_mean": float(np.mean(values(("partition", category, "r2p_degree", "mean")))),
        }
    return summary


def _p1i_rows(args: argparse.Namespace, resolution: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = args.p1i_baseline_root if resolution == 1024 else args.p1i_root
    result = json.loads((root / f"resolution_{resolution}.json").read_text())
    caches = {row["sample_id"]: Path(row["cache_file"]) for row in result["graph_cache"]["samples"]}
    preflight = json.loads((args.p1i_root / "actual_data_preflight.json").read_text())
    with h5py.File(args.p1i_full_fields, "r") as archive:
        full_coords = np.asarray(archive["shared/coords_m"], dtype=np.float64)
    lower, upper = np.min(full_coords, axis=0), np.max(full_coords, axis=0)
    manifest = json.loads(args.p1i_manifest.read_text())
    manifest_samples = {row["sample_id"]: row for row in manifest["samples"]}
    first_entry = manifest_samples[next(iter(caches))]
    first_meta = json.loads(
        (args.p1i_dataset_root / first_entry["relative_path"] / "sample_meta.json").read_text()
    )
    thickness = [
        float(row["thickness_m"])
        for row in first_meta["physics"]["layers_bottom_to_top"]
    ]
    boundaries = float(np.min(full_coords[:, 2])) + np.concatenate(
        [np.asarray([0.0]), np.cumsum(thickness)]
    )
    support_lookup = (
        {} if resolution == 1024 else {
            row["sample_id"]: row for row in preflight["supports"][str(resolution)]
        }
    )
    rows = []
    for sample_id in result["sample_ids"]:
        if resolution == 1024:
            entry = manifest_samples[sample_id]
            sample_root = args.p1i_dataset_root / entry["relative_path"]
            coords = np.load(sample_root / "coords.npy")
            q = np.load(sample_root / "q_field.npy").reshape(-1)
        else:
            with np.load(support_lookup[sample_id]["support_file"], allow_pickle=False) as support:
                indices = np.asarray(support["selected_indices"], dtype=np.int64)
                coords = full_coords[indices]
                q = np.asarray(support["q_W_m3"], dtype=np.float64)
        rows.append(_one_graph(
            caches[sample_id], lower=lower, upper=upper,
            categories=_categories(coords, q, boundaries),
        ))
    timing = {
        "historical_graph_stage_seconds": float(result["runtime"]["graph_build_or_load_seconds"]),
        "historical_group_prepare_seconds": float(result["runtime"]["group_prepare_seconds"]),
        "cache_load_seconds_sum": float(sum(row["load_seconds"] for row in result["graph_cache"]["samples"])),
        "fresh_build_seconds_sum_qualification_only": float(sum(row["build_seconds"] for row in result["graph_cache"]["samples"])),
    }
    return rows, timing


def _p1h_rows(args: argparse.Namespace, resolution: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ladder = json.loads(args.p1h_ladder.read_text())
    indices = np.asarray(ladder["probes"][str(resolution)]["indices"], dtype=np.int64)
    manifest = json.loads(args.p1h_manifest.read_text())
    valid = [row for row in manifest["samples"] if row["split_role"] == "valid_iid"][:32]
    cache_manifest = json.loads(args.p1h_cache_manifest.read_text())
    cache_entry = next(row for row in cache_manifest["entries"] if row["resolution"] == resolution)
    cache_file = args.p1h_cache_root / cache_entry["cache_file"]
    rows = []
    with h5py.File(args.p1h_full_fields, "r") as archive:
        coords = np.asarray(archive["mesh/coords"], dtype=np.float64)
        boundaries = np.asarray(archive["mesh/boundaries"], dtype=np.float64)
        sample_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in archive["samples/sample_id"][:]]
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        support_coords = coords[indices]
        lower, upper = np.min(coords, axis=0), np.max(coords, axis=0)
        for entry in valid:
            q = np.asarray(archive["samples/q_W_m3"][lookup[entry["sample_id"]], indices], dtype=np.float64)
            rows.append(_one_graph(
                cache_file, lower=lower, upper=upper,
                categories=_categories(support_coords, q, boundaries),
            ))
    timing_rows = list(csv.DictReader(args.p1h_timing_csv.open()))
    timing_row = next(
        row for row in timing_rows
        if row["platform"] == "gpu" and int(row["resolution"]) == resolution and int(row["batch_size"]) == 1
    )
    timing = {
        "historical_uncached_build_seconds": float(timing_row["graph_uncached_build_seconds"]),
        "historical_cached_load_seconds": float(timing_row["graph_cached_load_seconds"]),
        "scope": "one shared-support graph reused across samples",
    }
    return rows, timing


def _flatten(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": summary["family"],
        "resolution": summary["resolution"],
        "graph_count": summary["graph_count"],
        "regional_nodes_mean": summary["regional_node_count"]["mean"],
        "p2r_edges_mean": summary["edge_count"]["p2r"]["mean"],
        "r2r_edges_mean": summary["edge_count"]["r2r"]["mean"],
        "r2p_edges_mean": summary["edge_count"]["r2p"]["mean"],
        "p2r_regional_degree_mean": summary["degree"]["p2r_regional"]["mean"],
        "r2p_regional_degree_mean": summary["degree"]["r2p_regional"]["mean"],
        "r2r_out_degree_mean": summary["degree"]["r2r_out"]["mean"],
        "p2r_zero_degree_nodes_max": summary["coverage"]["p2r_zero_degree_nodes"]["max"],
        "r2p_zero_degree_nodes_max": summary["coverage"]["r2p_zero_degree_nodes"]["max"],
        "normalized_radius_median": summary["normalized_regional_radius"]["median"],
        "observed_radius_median_m": summary["observed_physical_support_radius_m"]["median"],
        "p2r_edge_length_median_m": summary["physical_edge_length_m"]["p2r"]["median"],
        "r2r_edge_length_median_m": summary["physical_edge_length_m"]["r2r"]["median"],
        "source_p2r_degree_mean": summary["partition"]["source"]["p2r_degree_mean"],
        "interface_p2r_degree_mean": summary["partition"]["interface"]["p2r_degree_mean"],
        "background_p2r_degree_mean": summary["partition"]["background"]["p2r_degree_mean"],
    }


def _write_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# V6 P1i vs P1h cached graph diagnostics", "",
        "This is an offline cache audit. It performs no model inference and reads no test/sealed labels.", "",
        "| family | N | Nr | P2R edges | R2R edges | P2R regional degree | radius median (m) | P2R length median (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        row = _flatten(summary)
        lines.append(
            f"| {row['family']} | {row['resolution']} | {row['regional_nodes_mean']:.1f} | "
            f"{row['p2r_edges_mean']:.1f} | {row['r2r_edges_mean']:.1f} | "
            f"{row['p2r_regional_degree_mean']:.3f} | {row['observed_radius_median_m']:.6e} | "
            f"{row['p2r_edge_length_median_m']:.6e} |"
        )
    lines += ["", "Interpretation is recorded in the publication-pipeline closeout after correlation with the frozen accuracy curve."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1i-root", type=Path, required=True)
    parser.add_argument("--p1i-baseline-root", type=Path, required=True)
    parser.add_argument("--p1i-dataset-root", type=Path, required=True)
    parser.add_argument("--p1i-manifest", type=Path, required=True)
    parser.add_argument("--p1i-full-fields", type=Path, required=True)
    parser.add_argument("--p1h-cache-root", type=Path, required=True)
    parser.add_argument("--p1h-cache-manifest", type=Path, required=True)
    parser.add_argument("--p1h-ladder", type=Path, required=True)
    parser.add_argument("--p1h-manifest", type=Path, required=True)
    parser.add_argument("--p1h-full-fields", type=Path, required=True)
    parser.add_argument("--p1h-timing-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = []
    for resolution in (1024, 4096, 8192, 16384, 32768, 65536):
        rows, timing = _p1i_rows(args, resolution)
        summary = _aggregate(rows, family="P1i_sample_varying", resolution=resolution)
        summary["graph_stage_timing"] = timing
        summaries.append(summary)
    for resolution in (1024, 4096, 8192, 16384, 32768):
        rows, timing = _p1h_rows(args, resolution)
        summary = _aggregate(rows, family="P1h_shared_support", resolution=resolution)
        summary["graph_stage_timing"] = timing
        summaries.append(summary)
    payload = {
        "schema_version": "heat3d_v6_cached_graph_diagnostics_v1",
        "status": "passed_offline_cache_only",
        "role_contract": {"training": False, "inference": False, "test": False, "sealed": False},
        "partition_priority": ["source_q_positive", "interface", "background"],
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    flat = [_flatten(summary) for summary in summaries]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    _write_md(args.output_md, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
