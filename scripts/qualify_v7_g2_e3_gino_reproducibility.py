#!/usr/bin/env python3
"""Bounded fresh-process reproducibility qualification for upstream GINO.

The Open3D FixedRadiusSearch + torch-scatter implementation is authoritative.
The pure-PyTorch path is not used as a formal backend; the existing geometry
receipt supplies its boundary-only comparison.  This script uses only frozen
train/valid_iid rows and temporary files under /tmp.  It performs three fresh
process repeats on two fixtures, a five-update trajectory, and an in-process
checkpoint restore probe.  No test/sealed role or accuracy selection is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
STATS_SHA = "554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a"
UPSTREAM_COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
R_IN, R_OUT, LATENT = 0.15, 0.033, 32
FIXTURE_SEED = 20260907
STEPS = 5


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def tensor_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        digest.update(str(array.dtype).encode())
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(array.numpy().tobytes(order="C"))
        return digest.hexdigest()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            digest.update(tensor_hash(value[key]).encode())
        return digest.hexdigest()
    if isinstance(value, (list, tuple)):
        for item in value:
            digest.update(tensor_hash(item).encode())
        return digest.hexdigest()
    digest.update(repr(value).encode())
    return digest.hexdigest()


def tree_arrays(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy().copy()
    if isinstance(value, dict):
        return {key: tree_arrays(value[key]) for key in value}
    if isinstance(value, list):
        return [tree_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_arrays(item) for item in value)
    return copy.deepcopy(value)


def tree_probe(value: Any, limit: int = 16384) -> dict[str, Any]:
    """Keep a deterministic bounded numerical probe plus full-tree hashes.

    GINO state trees are large enough that retaining every five-step tensor for
    every fresh process would create multi-GB temporary traces.  The complete
    tree hash remains in the child receipt; the fixed prefix probe supplies a
    reproducible relative-difference envelope without changing computation.
    """
    leaves: list[np.ndarray] = []
    remaining = int(limit)

    def visit(item: Any) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        if isinstance(item, np.ndarray):
            flat = np.asarray(item).reshape(-1)
            take = min(len(flat), remaining)
            leaves.append(np.asarray(flat[:take], dtype=np.float64).copy())
            remaining -= take
            return
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {"leaf_values": leaves, "budget": int(limit), "used": int(limit - remaining)}


def tree_sub(left: Any, right: Any) -> Any:
    """Subtract two CPU snapshots while retaining the pytree structure."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return np.asarray([np.inf], dtype=np.float64)
        return np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return np.asarray([np.inf], dtype=np.float64)
        return {key: tree_sub(left[key], right[key]) for key in sorted(left)}
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return np.asarray([np.inf], dtype=np.float64)
        values = [tree_sub(a, b) for a, b in zip(left, right)]
        return tuple(values) if isinstance(left, tuple) else values
    return 0.0 if left == right else np.asarray([np.inf], dtype=np.float64)


def tree_rel(left: Any, right: Any) -> dict[str, float]:
    diffs: list[np.ndarray] = []
    base: list[np.ndarray] = []

    def visit(a: Any, b: Any) -> None:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype:
                diffs.append(np.asarray([np.inf])); base.append(np.asarray([1.0])); return
            diffs.append(np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1))
            base.append(np.asarray(b, dtype=np.float64).reshape(-1))
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                diffs.append(np.asarray([np.inf])); base.append(np.asarray([1.0])); return
            for key in sorted(a): visit(a[key], b[key])
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                diffs.append(np.asarray([np.inf])); base.append(np.asarray([1.0])); return
            for x, y in zip(a, b): visit(x, y)
            return
        if a != b:
            diffs.append(np.asarray([np.inf])); base.append(np.asarray([1.0]))

    visit(left, right)
    if not diffs:
        return {"relative_l2": 0.0, "max_abs": 0.0, "rms_normalized": 0.0}
    d = np.concatenate(diffs); b = np.concatenate(base)
    denom = max(float(np.linalg.norm(b)), 1.0e-12)
    rms_denom = max(float(np.sqrt(np.mean(b * b))), 1.0e-12)
    return {
        "relative_l2": float(np.linalg.norm(d) / denom),
        "max_abs": float(np.max(np.abs(d))),
        "rms_normalized": float(np.sqrt(np.mean(d * d)) / rms_denom),
    }


def tree_norm(value: Any) -> float:
    leaves: list[np.ndarray] = []

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            leaves.append(np.asarray(item, dtype=np.float64).reshape(-1))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return float(np.linalg.norm(np.concatenate(leaves))) if leaves else 0.0


def output_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return tree_rel(np.asarray(left), np.asarray(right))


def load_inputs(args: argparse.Namespace, external: Any, fixture_module: Any) -> tuple[dict[str, tuple[Any, ...]], dict[str, Any]]:
    if sha256(args.dataset_manifest) != DATASET_SHA:
        raise ValueError("frozen dataset manifest SHA mismatch")
    if git_head(args.upstream_root) != UPSTREAM_COMMIT:
        raise ValueError("pinned upstream commit mismatch")
    if any(token in str(path).lower() for path in (args.dataset_root, args.dataset_manifest, args.statistics) for token in ("test", "sealed")):
        raise ValueError("test/sealed path is forbidden")
    manifest, rows_by_id = external.load_manifest(args.dataset_manifest)
    del manifest
    statistics_payload = json.loads(args.statistics.read_text(encoding="utf-8"))
    claimed_stats_sha = statistics_payload.get("payload_sha256")
    if claimed_stats_sha != STATS_SHA or statistics_payload.get("fit_role") != "train_only":
        raise ValueError("frozen train-only statistics SHA/role mismatch")
    stats = fixture_module.load_statistics(args.statistics)
    target_mean = np.asarray(statistics_payload["statistics"]["target_mean"], dtype=np.float32)
    target_std = np.maximum(np.asarray(statistics_payload["statistics"]["target_std"], dtype=np.float32), 1.0e-12)
    rows = {
        "train": next(row for row in rows_by_id.values() if row["split_role"] == "train"),
        "valid_iid": next(row for row in rows_by_id.values() if row["split_role"] == "valid_iid"),
    }
    inputs: dict[str, tuple[Any, ...]] = {}
    coord_min = torch.as_tensor(stats["coordinate_min"], dtype=torch.float32, device="cuda")
    coord_span = torch.clamp(torch.as_tensor(stats["coordinate_max"] - stats["coordinate_min"], dtype=torch.float32, device="cuda"), min=1.0e-12)
    feature_mean = torch.as_tensor(stats["feature_mean"], dtype=torch.float32, device="cuda")
    feature_std = torch.as_tensor(stats["feature_std"], dtype=torch.float32, device="cuda")
    y_mean = torch.as_tensor(target_mean, dtype=torch.float32, device="cuda")
    y_std = torch.as_tensor(target_std, dtype=torch.float32, device="cuda")

    def load_row(row: dict[str, Any], role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        sample_id = str(row["sample_id"])
        if str(row["split_role"]) != role or role not in {"train", "valid_iid"}:
            raise ValueError(f"fixture role mismatch for {sample_id}")
        directory = args.dataset_root / "samples" / sample_id
        for name in external.REQUIRED_FILES:
            path = directory / name
            if not path.is_file() or sha256(path) != row["file_sha256"][name]:
                raise ValueError(f"frozen file SHA mismatch: {sample_id}/{name}")
        coords = np.asarray(np.load(directory / "coords.npy", allow_pickle=False), dtype=np.float32)
        k_field = np.asarray(np.load(directory / "k_field.npy", allow_pickle=False), dtype=np.float32)
        q_field = np.asarray(np.load(directory / "q_field.npy", allow_pickle=False), dtype=np.float32).reshape(-1, 1)
        bc = np.asarray(np.load(directory / "bc_features.npy", allow_pickle=False), dtype=np.float32)
        target = np.asarray(np.load(directory / "deltaT.npy", allow_pickle=False), dtype=np.float32).reshape(-1, 1)
        metadata = json.loads((directory / "sample_meta.json").read_text(encoding="utf-8"))
        if bc.shape == (1024, 4):
            bc = np.column_stack((bc, np.full(1024, metadata["top_h_W_m2K"], dtype=np.float32), np.full(1024, metadata["bottom_h_W_m2K"], dtype=np.float32), np.zeros(1024, dtype=np.float32)))
        if coords.shape != (1024, 3) or k_field.shape != (1024, 3) or q_field.shape != (1024, 1) or bc.shape != (1024, 7) or target.shape != (1024, 1):
            raise ValueError(f"fixture shape mismatch: {sample_id}")
        return coords, np.concatenate((k_field, q_field, bc), axis=-1), target, {"sample_id": sample_id, "role": role}

    for role, row in rows.items():
        coords_np, features_np, target_np, metadata = load_row(row, role)
        coords = torch.from_numpy(coords_np).unsqueeze(0).to("cuda")
        features = torch.from_numpy(features_np).unsqueeze(0).to("cuda")
        target = torch.from_numpy(target_np).unsqueeze(0).to("cuda")
        normalized = ((coords - coord_min) / coord_span, (features - feature_mean) / feature_std, target, (target - y_mean) / y_std, {"sample_id": metadata["sample_id"], "role": metadata["role"]})
        inputs[role] = normalized
    return inputs, {"manifest_sha256": DATASET_SHA, "statistics_sha256": STATS_SHA}


def canonical_graph(search: Any, source: torch.Tensor, query: torch.Tensor, radius: float) -> tuple[np.ndarray, np.ndarray]:
    payload = search(source, query, radius)
    index = payload["neighbors_index"].detach().cpu().numpy().astype(np.int64)
    splits = payload["neighbors_row_splits"].detach().cpu().numpy().astype(np.int64)
    counts = np.diff(splits)
    query_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    pairs = np.column_stack((query_ids, index)) if len(index) else np.empty((0, 2), dtype=np.int64)
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return pairs, counts


def graph_summary(pairs: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "edge_count": int(len(pairs)),
        "pair_sha256": hashlib.sha256(np.ascontiguousarray(pairs, dtype=np.int64).tobytes()).hexdigest(),
        "counts_sha256": hashlib.sha256(np.ascontiguousarray(counts, dtype=np.int64).tobytes()).hexdigest(),
    }


def make_model(external: Any, upstream: Path, seed: int) -> tuple[torch.nn.Module, Any, Any, torch.Tensor]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = external.build_gino(R_IN, R_OUT, use_open3d=True, use_torch_scatter=True).cuda()
    if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
        raise RuntimeError("optimized Open3D backend unavailable")
    if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
        raise RuntimeError("optimized torch-scatter backend unavailable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    grid = external.latent_queries(LATENT).cuda()
    return model, optimizer, scheduler, grid


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: GINO qualification requires CUDA")
    if args.output.exists() or args.trace.exists():
        raise FileExistsError("refusing to overwrite fresh-process receipt/trace")
    external = load_module("g2_e3_external", ROOT / "scripts/run_v7_g2_p1_local_qualification.py")
    fixture_module = load_module("g2_e3_fixture", ROOT / "scripts/check_v7_g2_p9_gino_repeatability.py")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    inputs, data_receipt = load_inputs(args, external, fixture_module)
    fixtures: list[dict[str, Any]] = []
    model, optimizer, scheduler, grid = make_model(external, args.upstream_root, args.seed)
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = tensor_hash(initial_state)
    for role, normalized in inputs.items():
        coords, features, target, target_n, local = normalized
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize(); started = time.perf_counter()
            static_pred = model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
            torch.cuda.synchronize(); static_wall = time.perf_counter() - started
        in_pairs, in_counts = canonical_graph(model.gno_in.neighbor_search, coords.squeeze(0), grid.squeeze(0).reshape(-1, 3), R_IN)
        out_pairs, out_counts = canonical_graph(model.gno_out.neighbor_search, grid.squeeze(0).reshape(-1, 3), coords.squeeze(0), R_OUT)
        model.train()
        predictions: list[np.ndarray] = []
        params_snapshots: list[Any] = []
        update_snapshots: list[Any] = []
        loss_rows: list[dict[str, Any]] = []
        previous_params = tree_arrays(model.state_dict())
        base_key = torch.Generator(device="cuda").manual_seed(args.seed + 901)
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); step_started = time.perf_counter()
            pred_n = model(input_geom=coords, latent_queries=grid, output_queries=coords, x=features)
            loss = external.relative_l2(pred_n, target_n)
            loss.backward(); optimizer.step(); scheduler.step()
            torch.cuda.synchronize(); wall = time.perf_counter() - step_started
            value = float(loss.detach().cpu())
            current_params = tree_arrays(model.state_dict())
            update_tree = tree_sub(current_params, previous_params)
            predictions.append(pred_n.detach().cpu().numpy().copy())
            params_snapshots.append(tree_probe(current_params))
            update_snapshots.append(tree_probe(update_tree))
            loss_rows.append({"step": step + 1, "loss": value, "loss_finite": bool(np.isfinite(value)), "wall_seconds": wall, "params_hash": tensor_hash(current_params), "prediction_hash": tensor_hash(predictions[-1]), "update_l2": tree_norm(update_tree)})
            previous_params = current_params
        continuous_final = copy.deepcopy(model.state_dict())
        continuous_pred = predictions[-1].copy()

        # Bounded checkpoint restore: two updates, atomic torch save, reload in
        # a fresh model/optimizer object, then three updates.  This is not a
        # formal checkpoint and remains under /tmp.
        checkpoint_path = args.trace.with_suffix(f".{role}.checkpoint.pt")
        if checkpoint_path.exists():
            raise FileExistsError("refusing to overwrite checkpoint probe")
        model_r, opt_r, sch_r, grid_r = make_model(external, args.upstream_root, args.seed)
        model_r.load_state_dict(copy.deepcopy(initial_state))
        for _ in range(2):
            opt_r.zero_grad(set_to_none=True)
            pred_r = model_r(input_geom=coords, latent_queries=grid_r, output_queries=coords, x=features)
            loss_r = external.relative_l2(pred_r, target_n); loss_r.backward(); opt_r.step(); sch_r.step()
        saved_state = {"model": copy.deepcopy(model_r.state_dict()), "optimizer": opt_r.state_dict(), "scheduler": sch_r.state_dict(), "step": 2, "seed": args.seed, "role": role}
        torch.save(saved_state, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        model_r2, opt_r2, sch_r2, grid_r2 = make_model(external, args.upstream_root, args.seed)
        model_r2.load_state_dict(loaded["model"]); opt_r2.load_state_dict(loaded["optimizer"]); sch_r2.load_state_dict(loaded["scheduler"])
        reload_equal = tensor_hash(tree_arrays(model_r2.state_dict())) == tensor_hash(tree_arrays(model_r.state_dict()))
        resumed_losses: list[float] = []
        resumed_pred = None
        for _ in range(2, STEPS):
            opt_r2.zero_grad(set_to_none=True)
            pred_r2 = model_r2(input_geom=coords, latent_queries=grid_r2, output_queries=coords, x=features)
            loss_r2 = external.relative_l2(pred_r2, target_n); loss_r2.backward(); opt_r2.step(); sch_r2.step()
            resumed_losses.append(float(loss_r2.detach().cpu())); resumed_pred = pred_r2.detach().cpu().numpy().copy()
        resumed_final = tree_arrays(model_r2.state_dict())
        checkpoint_path.unlink(missing_ok=True)
        fixtures.append({
            "role": role,
            "sample_id": str(local["sample_id"]),
            "graph": {"input": graph_summary(in_pairs, in_counts), "output": graph_summary(out_pairs, out_counts)},
            "static_output": {"wall_seconds": static_wall, "finite": bool(torch.isfinite(static_pred).all().item()), "array": static_pred.detach().cpu().numpy().copy()},
            "trajectory": {"steps": loss_rows, "predictions": predictions, "params": params_snapshots, "updates": update_snapshots, "continuous_final_probe": tree_probe(continuous_final), "checkpoint_resumed_final_probe": tree_probe(resumed_final), "checkpoint_resumed_prediction": resumed_pred, "checkpoint_reload_state_equal": reload_equal, "checkpoint_resumed_losses": resumed_losses, "checkpoint_final_vs_continuous": tree_rel(resumed_final, tree_arrays(continuous_final)), "checkpoint_prediction_vs_continuous": output_stats(resumed_pred, continuous_pred) if resumed_pred is not None else {"relative_l2": float("inf"), "max_abs": float("inf"), "rms_normalized": float("inf")}},
        })
        del coords, features, target, target_n, local
        torch.cuda.empty_cache()
    payload = {
        "schema_version": "heat3d_v7_g2_e3_gino_child_v1",
        "status": "PASS_FINITE" if all(row["static_output"]["finite"] and all(item["loss_finite"] for item in row["trajectory"]["steps"]) for row in fixtures) else "FAIL_NONFINITE",
        "process_role": "fresh_child",
        "seed": int(args.seed),
        "backend": "Open3D_FixedRadiusSearch_plus_torch_scatter",
        "scientific_contract": {"r_in": R_IN, "r_out": R_OUT, "latent_grid": [LATENT] * 3, "steps": STEPS, "test_or_sealed_access": False, "accuracy_used_for_decision": False, "fallback_formal": False},
        "data": data_receipt,
        "initial_state": {"tree_sha256": initial_hash},
        "fixtures": [{key: value for key, value in row.items() if key != "static_output" or True} for row in fixtures],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "upstream_head": git_head(args.upstream_root), "runner_sha256": sha256(Path(__file__))},
        "hard_boundaries": {"test_or_sealed_access": False, "formal_training_started": False, "science_config_changed": False},
    }
    # The arrays are kept in a temporary pickle so the parent can calculate
    # relative-L2 envelopes without placing large predictions in Git.
    import pickle
    trace_payload = {"seed": args.seed, "initial_state": initial_hash, "fixtures": fixtures}
    with args.trace.open("wb") as stream:
        pickle.dump(trace_payload, stream, protocol=5)
    for row in payload["fixtures"]:
        row["static_output"].pop("array", None)
        row["trajectory"].pop("predictions", None); row["trajectory"].pop("params", None); row["trajectory"].pop("updates", None)
        row["trajectory"].pop("continuous_final_probe", None); row["trajectory"].pop("checkpoint_resumed_final_probe", None); row["trajectory"].pop("checkpoint_resumed_prediction", None)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def compare_trace(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for left, right in zip(base["fixtures"], other["fixtures"]):
        step_rows = []
        left_traj, right_traj = left["trajectory"], right["trajectory"]
        for index, (ls, rs, lp, rp, lparams, rparams, lupd, rupd) in enumerate(zip(left_traj["steps"], right_traj["steps"], left_traj["predictions"], right_traj["predictions"], left_traj["params"], right_traj["params"], left_traj["updates"], right_traj["updates"]), start=1):
            step_rows.append({"step": index, "loss_relative_difference": float(abs(ls["loss"] - rs["loss"]) / max(abs(ls["loss"]), 1.0e-12)), "prediction": output_stats(lp, rp), "parameters": tree_rel(lparams, rparams), "updates": tree_rel(lupd, rupd)})
        rows.append({"role": left["role"], "steps": step_rows, "static_output": output_stats(left["static_output"]["array"], right["static_output"]["array"]), "graph_exact": left["graph"] == right["graph"]})
    return {"fixtures": rows}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    child_specs: list[tuple[str, int]] = [("qualification", FIXTURE_SEED)] * 3 + [("inter_seed", seed) for seed in (0, 1, 2)]
    children: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    import pickle
    for index, (group, seed) in enumerate(child_specs):
        child_json = args.work_dir / f"child_{index:02d}_{group}_seed{seed}.json"
        child_trace = args.work_dir / f"child_{index:02d}_{group}_seed{seed}.pkl"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--dataset-root", str(args.dataset_root), "--dataset-manifest", str(args.dataset_manifest), "--statistics", str(args.statistics), "--upstream-root", str(args.upstream_root), "--output", str(child_json), "--trace", str(child_trace), "--seed", str(seed)]
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        started = time.perf_counter(); result = subprocess.run(command, env=env, cwd=ROOT, text=True, capture_output=True); wall = time.perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(f"GINO child failed ({group}, seed={seed}): {result.stdout[-2000:]} {result.stderr[-2000:]}")
        payload = json.loads(child_json.read_text(encoding="utf-8")); payload["parent_group"] = group; payload["process_wall_seconds"] = wall; payload["raw_sha256"] = sha256(child_json); children.append(payload)
        with child_trace.open("rb") as stream: traces.append(pickle.load(stream))
    same_pairs = [compare_trace(traces[0], traces[index]) for index in (1, 2)]
    inter_pairs = [compare_trace(traces[3], traces[index]) for index in (4, 5)]
    def median_metric(items: list[dict[str, Any]], path: tuple[str, ...]) -> float:
        values = []
        for item in items:
            for fixture in item["fixtures"]:
                for step in fixture["steps"]:
                    value: Any = step
                    for key in path: value = value[key]
                    values.append(float(value))
        return float(np.median(values)) if values else float("nan")
    same_rel = {"prediction": median_metric(same_pairs, ("prediction", "relative_l2")), "parameters": median_metric(same_pairs, ("parameters", "relative_l2")), "updates": median_metric(same_pairs, ("updates", "relative_l2")), "loss": median_metric(same_pairs, ("loss_relative_difference",))}
    inter_rel = {"prediction": median_metric(inter_pairs, ("prediction", "relative_l2")), "parameters": median_metric(inter_pairs, ("parameters", "relative_l2")), "updates": median_metric(inter_pairs, ("updates", "relative_l2")), "loss": median_metric(inter_pairs, ("loss_relative_difference",))}
    ratios = {key: float(same_rel[key] / max(inter_rel[key], 1.0e-12)) for key in same_rel}
    geometry = json.loads((ROOT / "docs/v7_g2_e2_gino_p8_e2_reconciliation.json").read_text(encoding="utf-8"))
    stable = all(np.isfinite(list(same_rel.values()))) and all(ratios[key] < 0.1 for key in ratios) and all(child["status"] == "PASS_FINITE" for child in children)
    payload = {
        "schema_version": "heat3d_v7_g2_e3_gino_reproducibility_aggregate_v1",
        "status": "GINO_AUTHOR_SEMANTICS_QUALIFIED" if stable else "GINO_NUMERICAL_STABILITY_FAIL_CLOSED",
        "scope": "six fresh child processes: three same-seed repeats plus seeds 0/1/2 bounded references; two frozen train/valid_iid fixtures; five optimizer updates",
        "authoritative_backend": "Open3D_FixedRadiusSearch_plus_torch_scatter",
        "fallback_role": "diagnostic_oracle_only",
        "geometry_gate": {"receipt": "docs/v7_g2_e2_gino_p8_e2_reconciliation.json", "current_non_boundary_disagreement": geometry["current_reruns"]["p9_geometry_896"]["non_boundary_disagreement"], "current_boundary_only_count": geometry["current_reruns"]["p9_geometry_896"]["boundary_only_count"]},
        "fresh_process_children": [{"group": x["parent_group"], "seed": x["seed"], "status": x["status"], "raw_sha256": x["raw_sha256"], "process_wall_seconds": x["process_wall_seconds"]} for x in children],
        "same_seed_vs_inter_seed": {"same_seed_median_relative": same_rel, "inter_seed_median_relative": inter_rel, "same_over_inter_ratio": ratios, "target_ratio_lt_0.1": True},
        "pairwise_receipts": {"same_seed": same_pairs, "inter_seed": inter_pairs},
        "decision": {"no_accuracy_used": True, "architecture_radius_information_budget_changed": False, "checkpoint_restore_tested": True, "classification_rule": "same-seed noise <10% of inter-seed variability, finite, no systematic drift", "formal_training_started": False},
        "environment": children[0]["environment"],
        "test_or_sealed_access": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/g2_e3_gino_reproducibility"))
    parser.add_argument("--seed", type=int, default=FIXTURE_SEED)
    args = parser.parse_args()
    if args.child:
        if args.trace is None: parser.error("--child requires --trace")
        payload = run_child(args)
        print(json.dumps({"status": payload["status"], "seed": payload["seed"], "fixtures": len(payload["fixtures"])}, sort_keys=True), flush=True)
        return 0 if payload["status"] == "PASS_FINITE" else 2
    payload = aggregate(args)
    print(json.dumps({"status": payload["status"], "children": len(payload["fresh_process_children"]), "same_over_inter": payload["same_seed_vs_inter_seed"]["same_over_inter_ratio"]}, sort_keys=True), flush=True)
    return 0 if payload["status"] == "GINO_AUTHOR_SEMANTICS_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
