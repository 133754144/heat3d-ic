#!/usr/bin/env python3
"""GINO backend semantics diagnostics without targets or accuracy.

This P8 diagnostic deliberately reads only one frozen ``train`` sample's
coordinates and physical input features.  It never opens ``deltaT.npy``,
``temperature.npy``, a validation sample, or any test/sealed artifact.  The
four backend combinations separate neighbor-search effects from reduction
effects while retaining the frozen GINO radii and latent grid.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
OUTPUT_ATOL, OUTPUT_RTOL = 1.0e-5, 1.0e-5
DATASET_MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATISTICS_PAYLOAD_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
UPSTREAM_COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
FIXED_SAMPLE = "v6p1if1_0000"
TRAIN_ONLY_FEATURES = (
    "kx", "ky", "kz", "q", "is_top", "is_bottom", "is_side",
    "is_interior", "top_h", "bottom_h", "top_T_inf_minus_T_ref",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_graph(neighbors: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    index = neighbors["neighbors_index"].detach().cpu().numpy().astype(np.int64)
    splits = neighbors["neighbors_row_splits"].detach().cpu().numpy().astype(np.int64)
    query = np.repeat(np.arange(len(splits) - 1, dtype=np.int64), np.diff(splits))
    pairs = np.column_stack((query, index))
    order = np.lexsort((pairs[:, 1], pairs[:, 0])) if len(pairs) else np.empty(0, dtype=np.int64)
    return pairs[order], np.diff(splits)


def graph_payload(search: Any, source: torch.Tensor, query: torch.Tensor, radius: float) -> dict[str, Any]:
    pairs, counts = canonical_graph(search(source, query, radius))
    return {
        "radius": radius,
        "source_count": int(source.shape[0]),
        "query_count": int(query.shape[0]),
        "edge_count": int(len(pairs)),
        "pairs": pairs,
        "counts": counts,
    }


def graph_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(np.array_equal(left["pairs"], right["pairs"]) and np.array_equal(left["counts"], right["counts"]))


def serializable_graph(payload: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "radius": payload["radius"],
        "source_count": payload["source_count"],
        "query_count": payload["query_count"],
        "edge_count": payload["edge_count"],
    }
    if baseline is not None:
        result["edge_multiset_exact_vs_fallback"] = graph_equal(payload, baseline)
        result["query_neighbor_counts_exact_vs_fallback"] = bool(
            np.array_equal(payload["counts"], baseline["counts"])
        )
    return result


def state_tensor_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in state.items():
        if key == "_metadata":
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"unexpected learned state value for {key}: {type(value)!r}")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def output_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left.detach().cpu() - right.detach().cpu()).abs().reshape(-1).double().numpy()
    return {
        "max_abs": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs": float(np.mean(difference)) if difference.size else 0.0,
        "p95_abs": float(np.percentile(difference, 95)) if difference.size else 0.0,
        "bitwise_equal": bool(torch.equal(left, right)),
        "allclose_atol_rtol_1e-5": bool(torch.allclose(left, right, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)),
    }


def load_input_sample(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    if sha256(manifest_path) != DATASET_MANIFEST_SHA:
        raise ValueError("frozen P1i manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in manifest["samples"]}
    row = rows[FIXED_SAMPLE]
    if row["split_role"] != "train":
        raise ValueError("P8 diagnostic fixed sample is not a train row")
    sample_dir = dataset_root / "samples" / FIXED_SAMPLE
    allowed = ("coords.npy", "k_field.npy", "q_field.npy", "bc_features.npy", "sample_meta.json")
    for name in allowed:
        path = sample_dir / name
        if not path.is_file() or sha256(path) != row["file_sha256"][name]:
            raise ValueError(f"frozen input SHA mismatch: {FIXED_SAMPLE}/{name}")
    coords_raw = np.asarray(np.load(sample_dir / "coords.npy", allow_pickle=False), dtype=np.float32)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy", allow_pickle=False), dtype=np.float32)
    q_field = np.asarray(np.load(sample_dir / "q_field.npy", allow_pickle=False), dtype=np.float32).reshape(-1, 1)
    bc = np.asarray(np.load(sample_dir / "bc_features.npy", allow_pickle=False), dtype=np.float32)
    metadata = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
    if coords_raw.shape != (1024, 3) or k_field.shape != (1024, 3) or q_field.shape != (1024, 1):
        raise ValueError("fixed input shape mismatch")
    if bc.shape == (1024, 4):
        bc = np.column_stack((
            bc,
            np.full(1024, metadata["top_h_W_m2K"], dtype=np.float32),
            np.full(1024, metadata["bottom_h_W_m2K"], dtype=np.float32),
            np.zeros(1024, dtype=np.float32),
        ))
    if bc.shape != (1024, 7):
        raise ValueError("fixed boundary feature shape mismatch")
    features = np.concatenate((k_field, q_field, bc), axis=-1)
    if features.shape != (1024, 11):
        raise ValueError("fixed physical feature shape mismatch")
    return {
        "sample_id": FIXED_SAMPLE,
        "role": "train",
        "coords_raw": coords_raw,
        "features_raw": features,
        "input_file_sha256": {name: row["file_sha256"][name] for name in allowed},
    }


def load_train_statistics(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = json_sha256(payload)
    if claimed != actual or actual != STATISTICS_PAYLOAD_SHA or payload["fit_role"] != "train_only":
        raise ValueError("frozen train-only statistics mismatch")
    statistics = payload["statistics"]
    return {
        "coordinate_min": np.asarray(statistics["coordinate_min"], dtype=np.float32),
        "coordinate_max": np.asarray(statistics["coordinate_max"], dtype=np.float32),
        "feature_mean": np.asarray(statistics["feature_mean"], dtype=np.float32),
        "feature_std": np.maximum(np.asarray(statistics["feature_std"], dtype=np.float32), 1.0e-12),
    }


def normalized_inputs(sample: dict[str, Any], statistics: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    coords = (sample["coords_raw"] - statistics["coordinate_min"]) / np.maximum(
        statistics["coordinate_max"] - statistics["coordinate_min"], 1.0e-12
    )
    features = (sample["features_raw"] - statistics["feature_mean"]) / statistics["feature_std"]
    return np.asarray(coords, dtype=np.float32), np.asarray(features, dtype=np.float32)


def signed_distance_records(source: np.ndarray, query: np.ndarray, pairs: np.ndarray, radius: float) -> dict[str, Any]:
    """Evaluate only the differing edges in fixed CPU/GPU arithmetic modes."""
    if pairs.size == 0:
        return {"count": 0, "edges": [], "max_abs_cpu_f64_margin": 0.0, "non_boundary_count": 0}
    pair_list = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    result: list[dict[str, Any]] = []
    modes: list[tuple[str, torch.dtype, str]] = [
        ("cpu_float64", torch.float64, "cpu"),
        ("cpu_float32", torch.float32, "cpu"),
    ]
    if torch.cuda.is_available():
        modes.append(("cuda_float32", torch.float32, "cuda"))
    values: dict[str, dict[str, list[float]]] = {}
    for name, dtype, device_name in modes:
        src = torch.as_tensor(source, dtype=dtype, device=device_name)
        qry = torch.as_tensor(query, dtype=dtype, device=device_name)
        selected_src = src[pair_list[:, 1]]
        selected_qry = qry[pair_list[:, 0]]
        delta = selected_src - selected_qry
        squared = torch.sum(delta * delta, dim=-1)
        distance = torch.sqrt(squared)
        if device_name == "cuda":
            torch.cuda.synchronize()
        values[name] = {
            "distance": [float(value) for value in distance.detach().cpu().tolist()],
            "squared_distance": [float(value) for value in squared.detach().cpu().tolist()],
        }
    for index, pair in enumerate(pair_list.tolist()):
        edge = {"query_index": int(pair[0]), "source_index": int(pair[1])}
        for name, value in values.items():
            distance = value["distance"][index]
            squared = value["squared_distance"][index]
            edge[name] = {
                "distance": distance,
                "squared_distance": squared,
                "signed_margin": distance - radius,
                "squared_signed_margin": squared - radius * radius,
                "inside_or_equal": bool(distance <= radius),
            }
        result.append(edge)
    cpu_f64 = [abs(edge["cpu_float64"]["signed_margin"]) for edge in result]
    boundary_limit = 1.0e-5
    return {
        "count": len(result),
        "edges": result,
        "boundary_margin_limit": boundary_limit,
        "max_abs_cpu_f64_margin": max(cpu_f64) if cpu_f64 else 0.0,
        "non_boundary_count": sum(value > boundary_limit for value in cpu_f64),
        "all_cpu_f64_boundary_near": bool(all(value <= boundary_limit for value in cpu_f64)),
    }


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-CLOSED: GINO P8 semantics diagnostic requires CUDA")
    upstream_head = git_head(args.upstream_root)
    if upstream_head != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {upstream_head} != {UPSTREAM_COMMIT}")
    sample = load_input_sample(args.dataset_root, args.dataset_manifest)
    statistics = load_train_statistics(args.statistics)
    coords_np, features_np = normalized_inputs(sample, statistics)
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    sys.path.insert(0, str(ROOT))
    from scripts.run_v7_g2_p1_local_qualification import build_gino, latent_queries

    coords = torch.from_numpy(coords_np).unsqueeze(0).to(device)
    features = torch.from_numpy(features_np).unsqueeze(0).to(device)
    grid = latent_queries(LATENT_RESOLUTION).unsqueeze(0).to(device)
    flat_coords, flat_grid = coords.squeeze(0), grid.squeeze(0).reshape(-1, 3)
    variant_specs = (
        ("fallback_neighbor_fallback_reduction", False, False),
        ("open3d_neighbor_fallback_reduction", True, False),
        ("fallback_neighbor_torch_scatter_reduction", False, True),
        ("open3d_neighbor_torch_scatter_reduction", True, True),
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    seed_model = build_gino(R_IN, R_OUT, use_open3d=False, use_torch_scatter=False).to(device)
    base_state = copy.deepcopy(seed_model.state_dict())
    state_hash = state_tensor_hash(base_state)
    fallback_input_graph = graph_payload(seed_model.gno_in.neighbor_search, flat_coords, flat_grid, R_IN)
    fallback_output_graph = graph_payload(seed_model.gno_out.neighbor_search, flat_grid, flat_coords, R_OUT)
    graph_results: dict[str, Any] = {}
    output_results: dict[str, Any] = {}
    outputs: dict[str, torch.Tensor] = {}
    variant_errors: dict[str, str] = {}
    for name, use_open3d, use_torch_scatter in variant_specs:
        try:
            model = seed_model if name == variant_specs[0][0] else build_gino(
                R_IN, R_OUT, use_open3d=use_open3d, use_torch_scatter=use_torch_scatter
            ).to(device)
            if model is not seed_model:
                model.load_state_dict(base_state)
            learned_equal = state_tensor_hash(model.state_dict()) == state_hash
            model.eval()
            input_graph = graph_payload(model.gno_in.neighbor_search, flat_coords, flat_grid, R_IN)
            output_graph = graph_payload(model.gno_out.neighbor_search, flat_grid, flat_coords, R_OUT)
            with torch.no_grad():
                prediction = model(
                    input_geom=coords, latent_queries=grid, output_queries=coords, x=features
                )
                prediction = prediction.detach()
                torch.cuda.synchronize()
            outputs[name] = prediction.cpu()
            graph_results[name] = {
                "backend": {"use_open3d": use_open3d, "use_torch_scatter": use_torch_scatter},
                "learned_state_bitwise_equal": learned_equal,
                "input": serializable_graph(input_graph, fallback_input_graph),
                "output": serializable_graph(output_graph, fallback_output_graph),
                "finite": bool(torch.isfinite(prediction).all().item()),
            }
            if name != variant_specs[0][0]:
                output_results[name] = output_stats(outputs[variant_specs[0][0]], outputs[name])
            if model is not seed_model:
                del model
        except Exception as exc:  # receipt must preserve a fail-closed error
            variant_errors[name] = f"{type(exc).__name__}: {exc}"
    del seed_model
    gc.collect()
    torch.cuda.empty_cache()

    fallback_pairs = fallback_input_graph["pairs"]
    optimized_graph = graph_results.get(variant_specs[-1][0], {})
    optimized_pairs = None
    # Re-run only the graph query (not a model or loss) to retain exact pair IDs
    # for the high-precision boundary receipt.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    optimized_search_model = build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).to(device)
    optimized_input_graph = graph_payload(
        optimized_search_model.gno_in.neighbor_search, flat_coords, flat_grid, R_IN
    )
    optimized_pairs = optimized_input_graph["pairs"]
    fallback_set = {tuple(pair) for pair in fallback_pairs.tolist()}
    optimized_set = {tuple(pair) for pair in optimized_pairs.tolist()}
    fallback_only = np.asarray(sorted(fallback_set - optimized_set), dtype=np.int64).reshape(-1, 2)
    optimized_only = np.asarray(sorted(optimized_set - fallback_set), dtype=np.int64).reshape(-1, 2)
    del optimized_search_model
    gc.collect()
    torch.cuda.empty_cache()
    boundary = {
        "fallback_only": signed_distance_records(coords_np, flat_grid.detach().cpu().numpy(), fallback_only, R_IN),
        "optimized_only": signed_distance_records(coords_np, flat_grid.detach().cpu().numpy(), optimized_only, R_IN),
    }
    differing = np.concatenate((fallback_only, optimized_only), axis=0) if len(fallback_only) + len(optimized_only) else np.empty((0, 2), dtype=np.int64)
    boundary["combined"] = signed_distance_records(coords_np, flat_grid.detach().cpu().numpy(), differing, R_IN)
    input_graph_exact = len(fallback_only) == 0 and len(optimized_only) == 0
    output_graph_exact = bool(
        graph_results.get(variant_specs[-1][0], {}).get("output", {}).get("edge_multiset_exact_vs_fallback", False)
    )
    optimized_output_close = bool(
        output_results.get(variant_specs[-1][0], {}).get("allclose_atol_rtol_1e-5", False)
    )

    # B3 is a synthetic self-loss timing probe: it deliberately has no target.
    b3: dict[str, Any]
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        fallback = build_gino(R_IN, R_OUT, use_open3d=False, use_torch_scatter=False).to(device)
        fallback.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(fallback.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        fallback.train()
        optimizer.zero_grad(set_to_none=True)
        warm = fallback(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
        warm.square().mean().backward(); optimizer.step(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        prediction = fallback(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
        synthetic_loss = prediction.square().mean()
        synthetic_loss.backward(); optimizer.step(); torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        fallback.eval()
        with torch.no_grad():
            torch.cuda.synchronize(); started = time.perf_counter()
            valid_prediction = fallback(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
            torch.cuda.synchronize(); valid_seconds = time.perf_counter() - started
        b3 = {
            "backend": "pure_PyTorch_fallback",
            "objective": "synthetic_prediction_square_mean_only; no target/truth/accuracy",
            "finite_optimizer_step": bool(torch.isfinite(synthetic_loss).item()),
            "train_step_wall_seconds": train_seconds,
            "valid_forward_wall_seconds": valid_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "prediction_finite": bool(torch.isfinite(valid_prediction).all().item()),
        }
        del fallback
    except Exception as exc:
        b3 = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    gc.collect(); torch.cuda.empty_cache()

    boundary_only = bool(
        boundary["combined"]["count"] > 0
        and boundary["combined"]["non_boundary_count"] == 0
        and boundary["combined"]["all_cpu_f64_boundary_near"]
    )
    all_variants_finite = bool(graph_results) and not variant_errors and all(
        value.get("finite", False) for value in graph_results.values()
    )
    if input_graph_exact and optimized_output_close and output_graph_exact and all_variants_finite:
        status = "ORIGINAL_GATE_PASS"
    elif boundary_only and output_graph_exact and all_variants_finite and b3.get("finite_optimizer_step", False):
        status = "SEMANTIC_EQUIVALENCE_AMENDMENT_CANDIDATE"
    else:
        status = "FAIL_CLOSED"
    return {
        "schema_version": "heat3d_v7_g2_p8_gino_semantics_diagnostic_v1",
        "status": status,
        "scientific_contract": {
            "upstream": f"neuraloperator/neuraloperator@{UPSTREAM_COMMIT}",
            "upstream_head_verified": upstream_head,
            "input_radius": R_IN,
            "output_radius": R_OUT,
            "latent_grid": [LATENT_RESOLUTION] * 3,
            "configuration_changed": False,
        },
        "data_boundary": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA,
            "sample_id": sample["sample_id"],
            "role": sample["role"],
            "source_coordinates_dtype": str(sample["coords_raw"].dtype),
            "source_coordinates_shape": list(sample["coords_raw"].shape),
            "source_coordinates_file_sha256": sample["input_file_sha256"]["coords.npy"],
            "source_coordinates_raw_bytes_sha256": bytes_sha256(sample["coords_raw"]),
            "source_coordinates_normalized_bytes_sha256": bytes_sha256(coords_np),
            "query_coordinates_dtype": "float32",
            "query_coordinates_shape": [LATENT_RESOLUTION ** 3, 3],
            "query_coordinates_bytes_sha256": bytes_sha256(flat_grid.detach().cpu().numpy()),
            "physical_features_shape": list(sample["features_raw"].shape),
            "physical_features_raw_bytes_sha256": bytes_sha256(sample["features_raw"]),
            "physical_features_normalized_bytes_sha256": bytes_sha256(features_np),
            "feature_names": list(TRAIN_ONLY_FEATURES),
            "target_or_truth_opened": False,
            "valid_or_test_opened": False,
        },
        "state_dict": {"learned_tensor_sha256": state_hash, "same_fixed_weight_for_all_variants": True},
        "B1_radius_boundary": {
            "fallback_edge_count": int(len(fallback_pairs)),
            "optimized_edge_count": int(len(optimized_pairs)),
            "fallback_only_edges": fallback_only.tolist(),
            "optimized_only_edges": optimized_only.tolist(),
            "symmetric_difference_each_side": {"fallback_only": len(fallback_only), "optimized_only": len(optimized_only)},
            "distance_arithmetic": boundary,
            "classification": {
                "boundary_only_by_cpu_float64_margin": boundary_only,
                "radius_was_changed": False,
                "accuracy_read": False,
            },
        },
        "B2_backend_decomposition": {
            "variants": list(graph_results.keys()),
            "graph": graph_results,
            "output_vs_fallback": output_results,
            "variant_errors": variant_errors,
            "all_variants_finite": all_variants_finite,
            "interpretation_scope": "neighbor-search versus reduction only; no truth or accuracy",
        },
        "B3_fallback_performance": b3,
        "gate_inputs": {
            "input_graph_exact": input_graph_exact,
            "output_graph_exact": output_graph_exact,
            "optimized_output_allclose_1e-5": optimized_output_close,
            "output_allclose_tolerance": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
        },
        "formal_training_started": False,
        "test_or_sealed_access": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "open3d": importlib.metadata.version("open3d"),
            "torch_scatter": importlib.metadata.version("torch-scatter"),
            "runner_sha256": sha256(Path(__file__)),
        },
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
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"ORIGINAL_GATE_PASS", "SEMANTIC_EQUIVALENCE_AMENDMENT_CANDIDATE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
