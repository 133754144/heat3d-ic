#!/usr/bin/env python3
"""Repeatability diagnostic for the authoritative Open3D+torch-scatter GINO.

The fixtures contain only coordinates and the 11 physical input features from
one train and one valid_iid sample.  No target, truth, test, or accuracy is
opened.  This script does not change the frozen radius, architecture, or
tolerance; it only records graph identity and output run-to-run stability.
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
OUTPUT_ATOL = OUTPUT_RTOL = 1.0e-5
DATASET_MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATISTICS_PAYLOAD_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
UPSTREAM_COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
FIXTURE_SEED = 20260907


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
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


def state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tensor.dtype).encode()); digest.update(repr(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def graph_hash(pairs: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "edge_count": int(len(pairs)),
        "pair_sha256": hashlib.sha256(np.ascontiguousarray(pairs, dtype=np.int64).tobytes()).hexdigest(),
        "counts_sha256": hashlib.sha256(np.ascontiguousarray(counts, dtype=np.int64).tobytes()).hexdigest(),
    }


def load_statistics(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = json_sha256(payload)
    if claimed != actual or actual != STATISTICS_PAYLOAD_SHA or payload.get("fit_role") != "train_only":
        raise ValueError("frozen train-only statistics mismatch")
    stats = payload["statistics"]
    return {
        "coordinate_min": np.asarray(stats["coordinate_min"], dtype=np.float32),
        "coordinate_max": np.asarray(stats["coordinate_max"], dtype=np.float32),
        "feature_mean": np.asarray(stats["feature_mean"], dtype=np.float32),
        "feature_std": np.maximum(np.asarray(stats["feature_std"], dtype=np.float32), 1.0e-12),
    }


def load_fixture(dataset_root: Path, row: dict[str, Any], stats: dict[str, np.ndarray]) -> dict[str, Any]:
    sample_id, role = str(row["sample_id"]), str(row["split_role"])
    sample_dir = dataset_root / "samples" / sample_id
    allowed = ("coords.npy", "k_field.npy", "q_field.npy", "bc_features.npy", "sample_meta.json")
    paths = {name: sample_dir / name for name in allowed}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != row["file_sha256"][name]:
            raise ValueError(f"fixture SHA mismatch: {sample_id}/{name}")
    coords_raw = np.asarray(np.load(paths["coords.npy"], allow_pickle=False), dtype=np.float32)
    k_field = np.asarray(np.load(paths["k_field.npy"], allow_pickle=False), dtype=np.float32)
    q_field = np.asarray(np.load(paths["q_field.npy"], allow_pickle=False), dtype=np.float32).reshape(-1, 1)
    bc = np.asarray(np.load(paths["bc_features.npy"], allow_pickle=False), dtype=np.float32)
    metadata = json.loads(paths["sample_meta.json"].read_text(encoding="utf-8"))
    if bc.shape == (1024, 4):
        bc = np.column_stack((bc, np.full(1024, metadata["top_h_W_m2K"], dtype=np.float32), np.full(1024, metadata["bottom_h_W_m2K"], dtype=np.float32), np.zeros(1024, dtype=np.float32)))
    if coords_raw.shape != (1024, 3) or k_field.shape != (1024, 3) or q_field.shape != (1024, 1) or bc.shape != (1024, 7):
        raise ValueError(f"fixture shape mismatch: {sample_id}")
    features_raw = np.concatenate((k_field, q_field, bc), axis=-1)
    coord_span = np.maximum(stats["coordinate_max"] - stats["coordinate_min"], 1.0e-12)
    coords = np.asarray((coords_raw - stats["coordinate_min"]) / coord_span, dtype=np.float32)
    features = np.asarray((features_raw - stats["feature_mean"]) / stats["feature_std"], dtype=np.float32)
    return {
        "sample_id": sample_id, "role": role, "coords": coords, "features": features,
        "coords_file_sha256": row["file_sha256"]["coords.npy"],
        "features_bytes_sha256": bytes_sha256(features_raw),
    }


def output_stats(base: torch.Tensor, other: torch.Tensor) -> dict[str, Any]:
    diff = (base.detach().cpu() - other.detach().cpu()).abs().reshape(-1).double().numpy()
    base_norm = float(torch.linalg.vector_norm(base.detach().float()).item())
    relative = float(np.linalg.norm((base.detach().cpu() - other.detach().cpu()).numpy().reshape(-1)) / max(base_norm, 1.0e-12))
    return {
        "max_abs": float(np.max(diff)) if len(diff) else 0.0,
        "mean_abs": float(np.mean(diff)) if len(diff) else 0.0,
        "p95_abs": float(np.percentile(diff, 95)) if len(diff) else 0.0,
        "relative_output_norm_difference": relative,
        "bitwise_equal": bool(torch.equal(base, other)),
        "allclose_atol_rtol_1e-5": bool(torch.allclose(base, other, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-CLOSED: repeatability diagnostic requires CUDA")
    if git_head(args.upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError("pinned neuraloperator commit mismatch")
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if sha256(args.dataset_manifest) != DATASET_MANIFEST_SHA:
        raise ValueError("frozen P1i manifest SHA mismatch")
    rows_by_id = {str(row["sample_id"]): row for row in manifest["samples"] if row["split_role"] in {"train", "valid_iid"}}
    train_id = "v6p1if1_0000"
    valid_candidates = [row for row in manifest["samples"] if row["split_role"] == "valid_iid"]
    if train_id not in rows_by_id or not valid_candidates or rows_by_id[train_id]["split_role"] != "train":
        raise ValueError("fixed repeatability fixtures are not in allowed roles")
    valid_id = str(valid_candidates[0]["sample_id"])
    stats = load_statistics(args.statistics)
    fixtures = [load_fixture(args.dataset_root, rows_by_id[train_id], stats), load_fixture(args.dataset_root, rows_by_id[valid_id], stats)]
    sys.path.insert(0, str(args.upstream_root.resolve())); sys.path.insert(0, str(ROOT))
    from scripts.run_v7_g2_p1_local_qualification import build_gino, latent_queries

    torch.manual_seed(FIXTURE_SEED); torch.cuda.manual_seed_all(FIXTURE_SEED)
    device = torch.device("cuda")
    model = build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).to(device).eval()
    base_state = copy.deepcopy(model.state_dict())
    grid = latent_queries(LATENT_RESOLUTION).unsqueeze(0).to(device)
    fixture_receipts = []
    all_graph_repeatable = True; all_output_repeatable = True
    for fixture in fixtures:
        coords = torch.as_tensor(fixture["coords"], dtype=torch.float32, device=device).unsqueeze(0)
        features = torch.as_tensor(fixture["features"], dtype=torch.float32, device=device).unsqueeze(0)
        graph_runs = []; output_runs = []
        for _ in range(3):
            graph_runs.append({
                "input": canonical_graph(model.gno_in.neighbor_search, coords.squeeze(0), grid.squeeze(0).reshape(-1, 3), R_IN),
                "output": canonical_graph(model.gno_out.neighbor_search, grid.squeeze(0).reshape(-1, 3), coords.squeeze(0), R_OUT),
            })
            with torch.no_grad():
                torch.cuda.synchronize(); started = time.perf_counter()
                prediction = model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features).detach()
                torch.cuda.synchronize()
                output_runs.append({"tensor": prediction.cpu(), "wall_seconds": time.perf_counter() - started})
        graph_identity = []
        for run_index, graph in enumerate(graph_runs):
            graph_identity.append({"run": run_index, "input": graph_hash(*graph["input"]), "output": graph_hash(*graph["output"])})
        graph_exact = all(
            np.array_equal(graph_runs[0][side][0], graph_runs[index][side][0]) and np.array_equal(graph_runs[0][side][1], graph_runs[index][side][1])
            for index in range(1, 3) for side in ("input", "output")
        )
        output_diffs = [output_stats(output_runs[0]["tensor"], output_runs[index]["tensor"]) for index in (1, 2)]
        outputs_finite = all(bool(torch.isfinite(item["tensor"]).all().item()) for item in output_runs)
        output_stable = outputs_finite and all(item["allclose_atol_rtol_1e-5"] for item in output_diffs)
        all_graph_repeatable &= graph_exact; all_output_repeatable &= output_stable
        fixture_receipts.append({
            "sample_id": fixture["sample_id"], "role": fixture["role"],
            "coords_file_sha256": fixture["coords_file_sha256"], "features_raw_bytes_sha256": fixture["features_bytes_sha256"],
            "graph_identity": graph_identity, "graph_repeatable": graph_exact,
            "output_runs": [{"wall_seconds": item["wall_seconds"], "finite": bool(torch.isfinite(item["tensor"]).all().item())} for item in output_runs],
            "output_run_to_run": output_diffs, "output_repeatable": output_stable,
        })
        del coords, features
        torch.cuda.empty_cache()
    status = "PASS_REPEATABLE" if all_graph_repeatable and all_output_repeatable else "FAIL_CLOSED_OPTIMIZED_NONREPEATABLE"
    return {
        "schema_version": "heat3d_v7_g2_p9_gino_repeatability_v1", "status": status,
        "scientific_contract": {"r_in": R_IN, "r_out": R_OUT, "latent_grid": [LATENT_RESOLUTION] * 3, "tolerance": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL}, "accuracy_read": False},
        "backend": "Open3D_FixedRadiusSearch_plus_torch_scatter", "fixtures": fixture_receipts,
        "state_dict_sha256": state_hash(base_state), "fixture_seed": FIXTURE_SEED,
        "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark},
        "target_truth_accuracy_opened": False, "test_or_sealed_access": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "open3d": importlib.metadata.version("open3d"), "torch_scatter": importlib.metadata.version("torch-scatter"), "upstream_head": git_head(args.upstream_root), "runner_sha256": sha256(Path(__file__))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, default=ROOT / "docs/v7_g2_p3_p1i_train_statistics.json")
    parser.add_argument("--upstream-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); receipt = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "fixtures": [{"sample_id": x["sample_id"], "graph_repeatable": x["graph_repeatable"], "output_repeatable": x["output_repeatable"]} for x in receipt["fixtures"]]}, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_REPEATABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
