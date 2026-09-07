#!/usr/bin/env python3
"""Audit the frozen GINO graphs on all 768 train + 128 valid_iid geometries.

This is deliberately a geometry-only diagnostic.  It opens only ``coords.npy``
for the two allowed split roles and never loads physical features, labels,
predictions, test_iid, or sealed files.  The radii and latent grid are the
frozen formal values; disagreements are classified with independent CPU/GPU
distance arithmetic and the radius is never changed to make graphs agree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
R_IN, R_OUT, LATENT_RESOLUTION = 0.15, 0.033, 32
BOUNDARY_MARGIN = 1.0e-5
DATASET_MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATISTICS_PAYLOAD_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
UPSTREAM_COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def canonical_graph(search: Any, source: torch.Tensor, query: torch.Tensor, radius: float) -> tuple[np.ndarray, np.ndarray]:
    payload = search(source, query, radius)
    index = payload["neighbors_index"].detach().cpu().numpy().astype(np.int64)
    splits = payload["neighbors_row_splits"].detach().cpu().numpy().astype(np.int64)
    counts = np.diff(splits)
    query_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    pairs = np.column_stack((query_ids, index)) if len(index) else np.empty((0, 2), dtype=np.int64)
    if len(pairs):
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
    return pairs, counts


def structured_pairs(pairs: np.ndarray) -> np.ndarray:
    dtype = np.dtype([("query", "<i8"), ("source", "<i8")])
    return np.ascontiguousarray(pairs, dtype=np.int64).view(dtype).reshape(-1)


def pair_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sorted ``left - right`` edge IDs without relying on order."""
    if not len(left):
        return np.empty((0, 2), dtype=np.int64)
    values = np.setdiff1d(structured_pairs(left), structured_pairs(right), assume_unique=False)
    return values.view(np.int64).reshape(-1, 2)


def graph_stats(pairs: np.ndarray, counts: np.ndarray, source_count: int, query_count: int) -> dict[str, Any]:
    nonzero = counts > 0
    used_sources = np.unique(pairs[:, 1]) if len(pairs) else np.empty(0, dtype=np.int64)
    return {
        "source_count": int(source_count),
        "query_count": int(query_count),
        "edge_count": int(len(pairs)),
        "zero_neighbor_fraction": float(np.mean(~nonzero)) if len(counts) else 0.0,
        "query_coverage": float(np.mean(nonzero)) if len(counts) else 0.0,
        "source_coverage": float(len(used_sources) / source_count) if source_count else 0.0,
        "mean_neighbors": float(np.mean(counts)) if len(counts) else 0.0,
        "median_neighbors": float(np.median(counts)) if len(counts) else 0.0,
        "p5_neighbors": float(np.percentile(counts, 5)) if len(counts) else 0.0,
        "p95_neighbors": float(np.percentile(counts, 95)) if len(counts) else 0.0,
    }


def load_coordinate_stats(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = json_sha256(payload)
    if claimed != actual or actual != STATISTICS_PAYLOAD_SHA or payload.get("fit_role") != "train_only":
        raise ValueError("frozen train-only coordinate statistics mismatch")
    statistics = payload["statistics"]
    return {
        "coordinate_min": np.asarray(statistics["coordinate_min"], dtype=np.float32),
        "coordinate_max": np.asarray(statistics["coordinate_max"], dtype=np.float32),
    }


def load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if sha256(manifest_path) != DATASET_MANIFEST_SHA:
        raise ValueError("frozen P1i manifest SHA mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["samples"] if row["split_role"] in {"train", "valid_iid"}]
    if len(rows) != 896 or sum(row["split_role"] == "train" for row in rows) != 768 or sum(row["split_role"] == "valid_iid" for row in rows) != 128:
        raise ValueError("expected exactly 768 train and 128 valid_iid manifest rows")
    if any(row["split_role"] == "test_iid" for row in rows):
        raise ValueError("test_iid row entered geometry audit")
    return rows


def load_coords(dataset_root: Path, row: dict[str, Any], coordinate_stats: dict[str, np.ndarray]) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = dataset_root / "samples" / sample_id
    path = sample_dir / "coords.npy"
    claimed = row["file_sha256"]["coords.npy"]
    if not path.is_file() or sha256(path) != claimed:
        raise ValueError(f"coordinate SHA mismatch for {sample_id}")
    raw = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if raw.shape != (1024, 3):
        raise ValueError(f"coordinate shape mismatch for {sample_id}: {raw.shape}")
    span = np.maximum(coordinate_stats["coordinate_max"] - coordinate_stats["coordinate_min"], 1.0e-12)
    normalized = np.asarray((raw - coordinate_stats["coordinate_min"]) / span, dtype=np.float32)
    return {
        "sample_id": sample_id,
        "role": str(row["split_role"]),
        "raw": raw,
        "normalized": normalized,
        "raw_file_sha256": claimed,
        "raw_bytes_sha256": bytes_sha256(raw),
        "normalized_bytes_sha256": bytes_sha256(normalized),
    }


def oracle_for_edges(source: np.ndarray, query: np.ndarray, pairs: np.ndarray, radius: float) -> dict[str, Any]:
    if not len(pairs):
        return {"count": 0, "edges": [], "non_boundary_count": 0, "max_abs_cpu_f64_margin": 0.0}
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pair_query = pairs[:, 0]
    pair_source = pairs[:, 1]
    for name, dtype in (("cpu_float64", np.float64), ("cpu_float32", np.float32)):
        delta = source[pair_source].astype(dtype) - query[pair_query].astype(dtype)
        squared = np.sum(delta * delta, axis=1, dtype=dtype)
        values[name] = (np.sqrt(squared).astype(dtype, copy=False), squared)
    if torch.cuda.is_available():
        src = torch.as_tensor(source[pair_source], dtype=torch.float32, device="cuda")
        qry = torch.as_tensor(query[pair_query], dtype=torch.float32, device="cuda")
        delta = src - qry
        squared = torch.sum(delta * delta, dim=-1)
        distance = torch.sqrt(squared)
        torch.cuda.synchronize()
        values["cuda_float32"] = (
            distance.detach().cpu().numpy(), squared.detach().cpu().numpy()
        )
    edges = []
    f64_margins = []
    for index, (query_index, source_index) in enumerate(pairs.tolist()):
        edge: dict[str, Any] = {"query_index": int(query_index), "source_index": int(source_index)}
        for name, (distance, squared) in values.items():
            d = float(distance[index]); sq = float(squared[index]); margin = d - radius
            edge[name] = {
                "distance": d,
                "squared_distance": sq,
                "signed_margin": margin,
                "squared_signed_margin": sq - radius * radius,
                "inside_or_equal": bool(d <= radius),
            }
            if name == "cpu_float64":
                f64_margins.append(abs(margin))
        edges.append(edge)
    return {
        "count": len(edges),
        "edges": edges,
        "non_boundary_count": int(sum(margin > BOUNDARY_MARGIN for margin in f64_margins)),
        "max_abs_cpu_f64_margin": float(max(f64_margins)) if f64_margins else 0.0,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: 0.0 for key in ("max", "p50", "p90", "p95", "p99")}
    return {
        "max": float(np.max(array)), "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-CLOSED: full GINO geometry audit requires CUDA f32 oracle")
    if git_head(args.upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError("pinned neuraloperator commit mismatch")
    rows = load_manifest_rows(args.dataset_manifest)
    coordinate_stats = load_coordinate_stats(args.statistics)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    sys.path.insert(0, str(ROOT))
    from scripts.run_v7_g2_p1_local_qualification import build_gino, latent_queries

    torch.manual_seed(20260907); torch.cuda.manual_seed_all(20260907)
    device = torch.device("cuda")
    fallback = build_gino(R_IN, R_OUT, use_open3d=False, use_torch_scatter=False).to(device).eval()
    base_state = copy.deepcopy(fallback.state_dict())
    optimized = build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).to(device).eval()
    optimized.load_state_dict(base_state)
    state_equal = all(torch.equal(base_state[key], optimized.state_dict()[key]) for key in base_state)
    grid = latent_queries(LATENT_RESOLUTION).to(device)
    grid_np = grid.detach().cpu().numpy().astype(np.float32)
    sample_receipts: list[dict[str, Any]] = []
    margin_values: list[float] = []
    non_boundary = 0
    boundary_only_edges = 0
    differing_samples = 0
    total_edges = {"input_fallback": 0, "input_open3d": 0, "output_fallback": 0, "output_open3d": 0}
    total_differing = {"input_fallback_only": 0, "input_open3d_only": 0, "output_fallback_only": 0, "output_open3d_only": 0}
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample = load_coords(args.dataset_root, row, coordinate_stats)
        coords = torch.as_tensor(sample["normalized"], dtype=torch.float32, device=device)
        input_fallback, input_fallback_counts = canonical_graph(fallback.gno_in.neighbor_search, coords, grid, R_IN)
        input_open3d, input_open3d_counts = canonical_graph(optimized.gno_in.neighbor_search, coords, grid, R_IN)
        output_fallback, output_fallback_counts = canonical_graph(fallback.gno_out.neighbor_search, grid, coords, R_OUT)
        output_open3d, output_open3d_counts = canonical_graph(optimized.gno_out.neighbor_search, grid, coords, R_OUT)
        graphs = {
            "input_fallback": (input_fallback, input_fallback_counts, 1024, LATENT_RESOLUTION ** 3),
            "input_open3d": (input_open3d, input_open3d_counts, 1024, LATENT_RESOLUTION ** 3),
            "output_fallback": (output_fallback, output_fallback_counts, LATENT_RESOLUTION ** 3, 1024),
            "output_open3d": (output_open3d, output_open3d_counts, LATENT_RESOLUTION ** 3, 1024),
        }
        for key, (pairs, _, _, _) in graphs.items():
            total_edges[key] += len(pairs)
        input_fallback_only = pair_difference(input_fallback, input_open3d)
        input_open3d_only = pair_difference(input_open3d, input_fallback)
        output_fallback_only = pair_difference(output_fallback, output_open3d)
        output_open3d_only = pair_difference(output_open3d, output_fallback)
        coords_np = coords.detach().cpu().numpy()
        differences = (
            ("input_fallback_only", input_fallback_only, coords_np, grid_np, R_IN),
            ("input_open3d_only", input_open3d_only, coords_np, grid_np, R_IN),
            ("output_fallback_only", output_fallback_only, grid_np, coords_np, R_OUT),
            ("output_open3d_only", output_open3d_only, grid_np, coords_np, R_OUT),
        )
        oracle: dict[str, Any] = {}
        for key, pairs, source, query, radius in differences:
            total_differing[key] += len(pairs)
            record = oracle_for_edges(source, query, pairs, radius)
            oracle[key] = record
            margin_values.extend(abs(edge["cpu_float64"]["signed_margin"]) for edge in record["edges"])
            non_boundary += record["non_boundary_count"]
            boundary_only_edges += record["count"] - record["non_boundary_count"]
        differs = any(len(pairs) for _, pairs, *_ in differences)
        differing_samples += int(differs)
        sample_receipts.append({
            "sample_id": sample["sample_id"], "role": sample["role"],
            "coords_raw_file_sha256": sample["raw_file_sha256"],
            "coords_raw_bytes_sha256": sample["raw_bytes_sha256"],
            "coords_normalized_bytes_sha256": sample["normalized_bytes_sha256"],
            "input": {
                "fallback": graph_stats(*graphs["input_fallback"],),
                "open3d": graph_stats(*graphs["input_open3d"],),
                "fallback_only_edge_ids": input_fallback_only.tolist(),
                "open3d_only_edge_ids": input_open3d_only.tolist(),
                "oracle": {key: oracle[key] for key in ("input_fallback_only", "input_open3d_only")},
            },
            "output": {
                "fallback": graph_stats(*graphs["output_fallback"],),
                "open3d": graph_stats(*graphs["output_open3d"],),
                "fallback_only_edge_ids": output_fallback_only.tolist(),
                "open3d_only_edge_ids": output_open3d_only.tolist(),
                "oracle": {key: oracle[key] for key in ("output_fallback_only", "output_open3d_only")},
            },
        })
        if index % 16 == 0:
            print(json.dumps({"processed": index, "total": len(rows), "elapsed_seconds": time.perf_counter() - started}), flush=True)
        del coords
        torch.cuda.empty_cache()
    del fallback, optimized, grid
    torch.cuda.synchronize()
    margin_summary = percentile_summary(margin_values)
    status = "PASS_GEOMETRY_NON_BOUNDARY_ZERO" if non_boundary == 0 else "FAIL_CLOSED_NON_BOUNDARY_GRAPH_DISAGREEMENT"
    return {
        "schema_version": "heat3d_v7_g2_p9_gino_geometry_896_v1",
        "status": status,
        "scientific_contract": {
            "upstream": f"neuraloperator/neuraloperator@{UPSTREAM_COMMIT}",
            "r_in": R_IN, "r_out": R_OUT, "latent_grid": [LATENT_RESOLUTION] * 3,
            "radius_changed": False, "architecture_changed": False,
            "fallback_backend": "pure_PyTorch", "optimized_backend": "Open3D_FixedRadiusSearch",
        },
        "sample_scope": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA,
            "train_count": 768, "valid_iid_count": 128, "processed_count": len(rows),
            "roles": ["train", "valid_iid"], "test_iid_opened": False, "sealed_opened": False,
            "target_truth_accuracy_opened": False,
        },
        "coordinate_oracle": {
            "coordinate_statistics_payload_sha256": STATISTICS_PAYLOAD_SHA,
            "query_coordinates_bytes_sha256": bytes_sha256(grid_np),
            "query_coordinates_shape": list(grid_np.shape),
            "arithmetic": ["cpu_float64", "cpu_float32", "cuda_float32"],
            "boundary_margin_threshold": BOUNDARY_MARGIN,
        },
        "backend_state": {"identical_state_dict": state_equal},
        "totals": {
            "zero_difference_samples": len(rows) - differing_samples,
            "differing_samples": differing_samples,
            "total_edges": total_edges,
            "total_differing_edges": total_differing,
            "total_differing_edges_all_sides": int(sum(total_differing.values())),
            "disagreement_fraction_input": float((total_differing["input_fallback_only"] + total_differing["input_open3d_only"]) / max(total_edges["input_fallback"], 1)),
            "disagreement_fraction_output": float((total_differing["output_fallback_only"] + total_differing["output_open3d_only"]) / max(total_edges["output_fallback"], 1)),
            "boundary_only_count": boundary_only_edges,
            "non_boundary_disagreement": non_boundary,
            "radius_margin_abs_summary": margin_summary,
        },
        "samples": sample_receipts,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "open3d": importlib.metadata.version("open3d"),
            "torch_scatter": importlib.metadata.version("torch-scatter"),
            "upstream_head": git_head(args.upstream_root),
            "runner_sha256": sha256(Path(__file__)),
        },
        "formal_training_started": False,
        "test_or_sealed_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, default=ROOT / "docs/v7_g2_p3_p1i_train_statistics.json")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "totals": receipt["totals"]}, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_GEOMETRY_NON_BOUNDARY_ZERO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
