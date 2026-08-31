#!/usr/bin/env python3
"""Qualify GINO pure-PyTorch vs Open3D/torch-scatter semantics on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
R_IN, R_OUT, LATENT_RESOLUTION = 0.15, 0.033, 32
OUTPUT_ATOL, OUTPUT_RTOL = 1.0e-5, 1.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_graph(neighbors: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    index = neighbors["neighbors_index"].detach().cpu().numpy().astype(np.int64)
    splits = neighbors["neighbors_row_splits"].detach().cpu().numpy().astype(np.int64)
    query = np.repeat(np.arange(len(splits) - 1, dtype=np.int64), np.diff(splits))
    pairs = np.column_stack((query, index))
    order = np.lexsort((pairs[:, 1], pairs[:, 0])) if len(pairs) else np.empty(0, dtype=np.int64)
    return pairs[order], np.diff(splits)


def compare_graph(fallback_search: Any, optimized_search: Any, data: torch.Tensor, queries: torch.Tensor, radius: float) -> dict[str, Any]:
    fallback_pairs, fallback_counts = canonical_graph(fallback_search(data, queries, radius))
    optimized_pairs, optimized_counts = canonical_graph(optimized_search(data, queries, radius))
    exact = np.array_equal(fallback_pairs, optimized_pairs) and np.array_equal(fallback_counts, optimized_counts)
    return {
        "status": "PASS" if exact else "FAIL",
        "radius": radius,
        "query_count": int(len(queries)), "source_count": int(len(data)),
        "fallback_edge_count": int(len(fallback_pairs)), "optimized_edge_count": int(len(optimized_pairs)),
        "query_neighbor_counts_exact": bool(np.array_equal(fallback_counts, optimized_counts)),
        "edge_multiset_exact": bool(np.array_equal(fallback_pairs, optimized_pairs)),
    }


def cuda_timed(callable_: Any) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); result = callable_(); end.record(); torch.cuda.synchronize()
    return result, float(start.elapsed_time(end) / 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local-contract-check", "optimized-preflight"), required=True)
    parser.add_argument("--dataset-root", type=Path); parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--statistics", type=Path, default=ROOT / "docs/v7_g2_p3_p1i_train_statistics.json")
    parser.add_argument("--upstream-root", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "local-contract-check":
        receipt = {
            "schema_version": "heat3d_v7_g2_p5_gino_backend_local_contract_v1",
            "status": "FORMAL_OPTIMIZED_BACKEND_UNQUALIFIED_NO_CUDA" if not torch.cuda.is_available() else "CUDA_PRESENT_USE_OPTIMIZED_PREFLIGHT",
            "qualification_backend": "pure_PyTorch_fallback", "formal_backend": "Open3D_plus_torch_scatter",
            "formal_preflight_cuda_required": True, "cuda_available": torch.cuda.is_available(),
            "scientific_config_unchanged": {"r_in": R_IN, "r_out": R_OUT, "latent_grid": [32, 32, 32]},
            "graph_or_output_equivalence_executed": False,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True)); return 0
    if None in (args.dataset_root, args.dataset_manifest, args.upstream_root, args.output):
        parser.error("optimized-preflight requires dataset, upstream, and output paths")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: GINO formal optimized preflight requires CUDA")
    if importlib.util.find_spec("open3d") is None or importlib.util.find_spec("torch_scatter") is None:
        raise SystemExit("FAIL-CLOSED: Open3D and torch-scatter are both required")
    sys.path.insert(0, str(args.upstream_root.resolve())); sys.path.insert(0, str(ROOT))
    from scripts.run_v7_g2_p1_local_qualification import build_gino, latent_queries, relative_l2
    from scripts.run_v7_g2_p1i_external_formal import P1iRoleDataset, load_stats, normalize, predict
    device = torch.device("cuda"); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    train = P1iRoleDataset(args.dataset_root, args.dataset_manifest, "train")
    valid = P1iRoleDataset(args.dataset_root, args.dataset_manifest, "valid_iid")
    stats = load_stats(args.statistics)
    fallback = build_gino(R_IN, R_OUT, use_open3d=False, use_torch_scatter=False).to(device)
    optimized = build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).to(device)
    optimized.load_state_dict(fallback.state_dict())
    if not optimized.gno_in.neighbor_search.use_open3d or not optimized.gno_out.neighbor_search.use_open3d:
        raise RuntimeError("Open3D was requested but upstream silently selected fallback")
    if not optimized.gno_in.integral_transform.use_torch_scatter or not optimized.gno_out.integral_transform.use_torch_scatter:
        raise RuntimeError("torch-scatter was requested but not bound")
    train_row = train[0]; coords, features, _target, target_n, _local = normalize(train_row, stats, device)
    grid = latent_queries(LATENT_RESOLUTION).unsqueeze(0).to(device)
    flat_grid = grid.squeeze(0).reshape(-1, 3); flat_coords = coords.squeeze(0)
    graph = {
        "input_gno": compare_graph(fallback.gno_in.neighbor_search, optimized.gno_in.neighbor_search, flat_coords, flat_grid, R_IN),
        "output_gno": compare_graph(fallback.gno_out.neighbor_search, optimized.gno_out.neighbor_search, flat_grid, flat_coords, R_OUT),
    }
    fallback.eval(); optimized.eval()
    with torch.no_grad():
        fallback_output = predict("GINO", fallback, coords, features, grid)
        optimized_output = predict("GINO", optimized, coords, features, grid)
    output_close = torch.allclose(fallback_output, optimized_output, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    output_max_abs = float(torch.max(torch.abs(fallback_output - optimized_output)).cpu())
    del fallback, fallback_output; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(optimized.parameters(), lr=1e-3, weight_decay=1e-4)
    optimized.train()
    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True); prediction = predict("GINO", optimized, coords, features, grid)
        loss = relative_l2(prediction, target_n); loss.backward(); optimizer.step(); return loss.detach()
    loss, train_seconds = cuda_timed(train_step)
    valid_row = valid[0]; valid_coords, valid_features, _valid_target, _valid_target_n, _ = normalize(valid_row, stats, device)
    optimized.eval()
    with torch.no_grad():
        _valid_output, valid_seconds = cuda_timed(lambda: predict("GINO", optimized, valid_coords, valid_features, grid))
    graph_pass = all(row["status"] == "PASS" for row in graph.values())
    status = "PASS_OPTIMIZED_BACKEND_QUALIFIED" if graph_pass and output_close and torch.isfinite(loss) else "FAIL_CLOSED"
    receipt = {
        "schema_version": "heat3d_v7_g2_p5_gino_backend_fairness_v1", "status": status,
        "fixed_samples": {"train": train_row["sample_id"], "valid_iid": valid_row["sample_id"]},
        "scientific_config_unchanged": {"r_in": R_IN, "r_out": R_OUT, "latent_grid": [32, 32, 32]},
        "backends": {"qualification": "pure_PyTorch_native_neighbor_search_and_segment_csr", "formal": "Open3D_FixedRadiusSearch_plus_torch_scatter_segment_csr"},
        "graph_semantics": graph,
        "output_semantics": {"allclose": bool(output_close), "atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL, "max_absolute_difference": output_max_abs},
        "resource": {"gpu_name": torch.cuda.get_device_name(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "train_step_wall_seconds": train_seconds, "valid_forward_wall_seconds": valid_seconds},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        "train_objective_finite": bool(torch.isfinite(loss)), "accuracy_used_to_change_config": False,
        "formal_or_long_training_started": False, "test_or_sealed_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True)); return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
