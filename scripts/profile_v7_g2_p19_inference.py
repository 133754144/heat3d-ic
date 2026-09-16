#!/usr/bin/env python3
"""Bounded P19 valid-only inference profiling for Heat3D and DeepOHeat-v1.

This script profiles frozen checkpoints only.  It never trains, evaluates a
test/sealed split, loads a temperature target, or writes predictions.  Timing
receipts contain phase timings and provenance, not accuracy claims.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_runtime.high_n import FullFieldGeometry  # noqa: E402
from rigno.heat3d_runtime.u_split import UHighNRuntime  # noqa: E402
from rigno.heat3d_training import block_until_ready, build_p1i_batches, model_apply_full  # noqa: E402
from rigno.heat3d_training.p1i import attach_input_contexts, attach_native_physics, attach_qk_features  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402


FS_TRAIN_SHA = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
LABEL_RECEIPT_SHA = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
SUBSET_SHA = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
NORMALIZATION_SHA = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"
MESH_SHAPE = (101, 101, 56)
MESH_COUNT = int(np.prod(MESH_SHAPE))
FORBIDDEN_NAMES = {"fs_test_volume.npy", "u_test_volume.npy", "test_iid", "sealed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def install_guard() -> None:
    forbidden = {item.lower() for item in FORBIDDEN_NAMES}

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(candidate))
            if path.name.lower() in forbidden or any(part.lower() in forbidden for part in path.parts):
                raise PermissionError(f"FAIL-CLOSED: forbidden split artifact opened: {candidate}")

    sys.addaudithook(audit)


def load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if claimed != actual or actual != NORMALIZATION_SHA or payload.get("valid_or_test_used_to_fit") is not False:
        raise ValueError("frozen train-only normalization mismatch")
    stats = dict(payload["statistics"])
    for key in ("coord_min", "coord_span", "condition_mean", "condition_std", "target_delta_mean", "target_delta_std"):
        stats[key] = np.asarray(stats[key], dtype=np.float32)
    return stats


class InputOnlyDataset:
    """Read only frozen train/valid inputs needed to build model conditions.

    The compact formal loader also exposes targets for evaluation.  P19 is a
    timing-only receipt, so this local view deliberately never opens target
    artifacts; it supplies a zero-shaped placeholder solely for the existing
    example dataclass, which is not consumed by inference.
    """

    def __init__(self, *, fs_train: Path, labels_root: Path, role: str, converter: Any):
        if role not in {"train", "valid"}:
            raise ValueError("P19 input-only dataset accepts train or valid")
        receipt_path = labels_root / "label_generation_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.rows = [row for row in receipt["rows"] if row.get("role") == role]
        expected = 768 if role == "train" else 128
        if len(self.rows) != expected:
            raise ValueError(f"{role} row count mismatch")
        self.fs_train = np.load(fs_train, mmap_mode="r", allow_pickle=False)
        self.labels_root = labels_root
        self.role = role
        self.converter = converter

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_index = int(row["source_index"])
        power = np.asarray(self.fs_train[source_index])
        if sha256_array(power) != row["source_input_sha256"]:
            raise ValueError(f"source-input SHA drift: {row['sample_id']}")
        arrays = self.converter.volume_v1_arrays(power)
        directory = self.labels_root / self.role / row["sample_id"]
        support_meta = row["artifacts"]["support_indices"]
        support_path = directory / support_meta["file"]
        support_raw = np.asarray(np.load(support_path, allow_pickle=False), dtype=np.int32)
        if sha256_array(support_raw) != support_meta["sha256"]:
            raise ValueError(f"support-index SHA drift: {row['sample_id']}")
        support = support_raw.astype(np.int64)
        if support.shape != (1024,):
            raise ValueError(f"support shape drift: {row['sample_id']}")
        return {
            "sample_id": row["sample_id"],
            "source_index": source_index,
            "coords": np.asarray(arrays["coords"])[support].astype(np.float32),
            "features": np.asarray(arrays["features"])[support].astype(np.float32),
            "support_indices": support,
            "target_1024": np.zeros(1024, dtype=np.float32),
        }


def checkpoint_params(path: Path) -> tuple[Any, dict[str, Any]]:
    import pickle

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if "params" not in payload:
        raise ValueError(f"checkpoint has no params: {path}")
    return payload["params"], {key: payload.get(key) for key in ("epoch", "step", "seed", "selection_metric", "selection_value")}


def memory_stats() -> dict[str, int | float | None]:
    device = jax.devices()[0]
    raw = device.memory_stats() or {}
    result: dict[str, int | float | None] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            result[str(key)] = int(value)
    return result


def gpu_environment() -> dict[str, Any]:
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise SystemExit("FAIL-CLOSED: P19 profile requires a CUDA GPU")
    smi: dict[str, Any] = {}
    command = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        text = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        smi["query"] = text
    except Exception as exc:
        smi["query_error"] = f"{type(exc).__name__}: {exc}"
    try:
        jaxlib_version = __import__("jaxlib").__version__
    except Exception:
        jaxlib_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "backend": jax.default_backend(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "devices": [str(item) for item in devices],
        "smi": smi,
        "memory_initial": memory_stats(),
    }


def max_memory(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = set(before) | set(after)
    return {key: max(int(before.get(key, 0) or 0), int(after.get(key, 0) or 0)) for key in keys}


def prepare_heat3d(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = load_script("evaluate_v7_g2_p14_heat3d_v1_fullfield.py")
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    support_module = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    if sha256_file(args.fs_train) != FS_TRAIN_SHA:
        raise ValueError("fs_train SHA mismatch")
    if sha256_file(args.labels_root / "label_generation_receipt.json") != LABEL_RECEIPT_SHA:
        raise ValueError("label receipt SHA mismatch")
    if sha256_file(args.subset_manifest) != SUBSET_SHA:
        raise ValueError("subset manifest SHA mismatch")
    stats = load_stats(args.normalization)
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    train_data = InputOnlyDataset(fs_train=args.fs_train, labels_root=args.labels_root, role="train", converter=converter)
    valid_data = InputOnlyDataset(fs_train=args.fs_train, labels_root=args.labels_root, role="valid", converter=converter)
    mesh = support_module.mesh_arrays()
    layer_id = np.asarray(mesh["layer_id"], dtype=np.int32)
    train_examples = evaluator.build_examples(dataset=train_data, role="train", full_coords=mesh["coords"], full_cv=mesh["control_volume"], layer_id=layer_id, converter=converter)
    valid_examples = evaluator.build_examples(dataset=valid_data, role="valid", full_coords=mesh["coords"], full_cv=mesh["control_volume"], layer_id=layer_id, converter=converter)
    builder = Heat3DGraphBuilder(**config["graph"])
    valid_batches = build_p1i_batches(valid_examples, stats, builder, label="p19_profile_valid", batch_size=32, graph_seed=int(args.seed))
    context = attach_input_contexts(valid_batches, train_examples, train_examples + valid_examples, config["model"])
    by_id = {row.sample_id: row for row in train_examples + valid_examples}
    attach_native_physics(valid_batches, by_id, context_by_id=context["raw_context_by_id"])
    attach_qk_features(valid_batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
    model_config.setdefault("shape_attention_mode", "none")
    model = RIGNO(**model_config)
    params, checkpoint_meta = checkpoint_params(args.heat3d_checkpoint)
    p14 = load_script("evaluate_v7_g2_p14_1_heat3d_u_v2.py")
    session = p14.build_session(
        model=model,
        params=params,
        stats=stats,
        config=config,
        context_standardizer=context["standardizer"],
        seed=int(args.seed),
    )
    runtime = UHighNRuntime.from_session(
        session,
        FullFieldGeometry(
            path=Path("<derived-deepoheat-v1-mesh>"),
            coords=np.asarray(mesh["coords"], dtype=np.float64),
            control_volume=np.asarray(mesh["control_volume"], dtype=np.float64),
            layer_id=layer_id,
            sample_ids=tuple(),
            split_roles=tuple(),
        ),
    )
    runtime.geometry = SimpleNamespace(coords=np.asarray(mesh["coords"], dtype=np.float64), control_volume=np.asarray(mesh["control_volume"], dtype=np.float64), layer_id=layer_id)
    return {
        "p14": p14,
        "converter": converter,
        "support_module": support_module,
        "runtime": runtime,
        "model": model,
        "params": params,
        "checkpoint_meta": checkpoint_meta,
        "valid_data": valid_data,
        "valid_examples": valid_examples,
        "mesh": mesh,
        "layer_id": layer_id,
        "MESH_COUNT": MESH_COUNT,
    }


def profile_heat3d(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_heat3d(args)
    p14 = prepared["p14"]
    runtime = prepared["runtime"]
    converter = prepared["converter"]
    support_module = prepared["support_module"]
    valid_data = prepared["valid_data"]
    valid_examples = prepared["valid_examples"]
    mesh = prepared["mesh"]
    layer_id = prepared["layer_id"]
    boundaries = np.asarray([0.0, 0.1, 0.55], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    cached_case: Any = None
    peak: dict[str, Any] = {}
    for index, compact in enumerate(valid_data.rows):
        compact_row = valid_data[index]
        anchor = valid_examples[index]
        support_weights, support_indices = p14.support_from_compact(compact_row, mesh)
        power = np.asarray(valid_data.fs_train[int(compact["source_index"])], dtype=np.float32)
        full_support = p14.full_support_artifact(power=power, converter=converter, mesh=mesh)
        # Compact transport coordinates are float32.  Rebind only the already
        # frozen support coordinates to the exact benchmark mesh before the
        # U-v2 membership check; no input feature or support index changes.
        u_anchor = replace(
            anchor,
            condition=replace(
                anchor.condition,
                coords=np.asarray(mesh["coords"], dtype=np.float64)[support_indices],
            ),
            meta={
                **anchor.meta,
                "top_h_W_m2K": 0.1 / 2.0,
                "bottom_h_W_m2K": 0.1 / 40.0,
            },
        )
        before = memory_stats()
        e2e_started = time.perf_counter()
        graph_started = time.perf_counter()
        case = runtime.build_case(u_anchor, MESH_COUNT, support=full_support, native_edge_targets=None, query_edge_targets=None)
        graph_seconds = time.perf_counter() - graph_started
        forward_started = time.perf_counter()
        output = runtime.apply(case)
        block_until_ready(output["raw_temperature"])
        forward_seconds = time.perf_counter() - forward_started
        post_started = time.perf_counter()
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        finite = bool(np.all(np.isfinite(raw)))
        output_shape = list(raw.shape)
        # The host conversion is the only postprocess required for timing;
        # no target/truth or metric is loaded.
        _ = raw.reshape(-1) - 298.15
        post_seconds = time.perf_counter() - post_started
        e2e_seconds = time.perf_counter() - e2e_started
        after = memory_stats()
        peak = max_memory(peak, max_memory(before, after))
        edge_counts = {
            key: (None if value is None else int(value))
            for key, value in runtime._edge_counts(case.query_metadata).items()
        }
        rows.append({
            "valid_index": index,
            "sample_id": str(compact["sample_id"]),
            "cold": index == 0,
            "query_graph_construction_seconds": float(graph_seconds),
            "model_forward_seconds": float(forward_seconds),
            "postprocess_seconds": float(post_seconds),
            "e2e_seconds": float(e2e_seconds),
            "output_shape": output_shape,
            "finite": finite,
            "query_edge_counts": edge_counts,
            "memory_before": before,
            "memory_after": after,
        })
        if cached_case is None:
            cached_case = case
        else:
            del case
    cached_rows: list[dict[str, Any]] = []
    for repeat in range(int(args.cached_repeats)):
        before = memory_stats()
        started = time.perf_counter()
        output = runtime.apply(cached_case)
        block_until_ready(output["raw_temperature"])
        forward_done = time.perf_counter()
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        _ = raw.reshape(-1) - 298.15
        total = time.perf_counter() - started
        post = time.perf_counter() - forward_done
        after = memory_stats()
        peak = max_memory(peak, max_memory(before, after))
        cached_rows.append({"repeat": repeat, "e2e_seconds": float(total), "model_forward_seconds": float(total - post), "postprocess_seconds": float(post), "finite": bool(np.all(np.isfinite(raw))), "memory_before": before, "memory_after": after})
    steady = rows[1:]
    return {
        "status": "PASS_VALID_ONLY_INFERENCE_PROFILE" if all(row["finite"] for row in rows + cached_rows) else "FAIL_CLOSED_NONFINITE",
        "regime": "Heat3D-on-DeepOHeat-v1 e600",
        "checkpoint_role": "fixed_e600_endpoint",
        "checkpoint": {"path": str(args.heat3d_checkpoint), "sha256": sha256_file(args.heat3d_checkpoint), "metadata": prepared["checkpoint_meta"]},
        "batch_size": 1,
        "valid_count": len(rows),
        "query_count": MESH_COUNT,
        "cold_e2e_seconds": rows[0]["e2e_seconds"],
        "query_graph_construction_seconds": {"cold": rows[0]["query_graph_construction_seconds"], "steady_median": median([row["query_graph_construction_seconds"] for row in steady]), "steady_p95": float(np.percentile([row["query_graph_construction_seconds"] for row in steady], 95))},
        "model_forward_seconds": {"cold": rows[0]["model_forward_seconds"], "steady_median": median([row["model_forward_seconds"] for row in steady]), "steady_p95": float(np.percentile([row["model_forward_seconds"] for row in steady], 95))},
        "postprocess_seconds": {"cold": rows[0]["postprocess_seconds"], "steady_median": median([row["postprocess_seconds"] for row in steady]), "steady_p95": float(np.percentile([row["postprocess_seconds"] for row in steady], 95))},
        "steady_e2e_seconds": {"median": median([row["e2e_seconds"] for row in steady]), "p95": float(np.percentile([row["e2e_seconds"] for row in steady], 95))},
        "cached_fixture": {"valid_index": 0, "repeats": cached_rows, "e2e_seconds": {"median": median([row["e2e_seconds"] for row in cached_rows]), "p95": float(np.percentile([row["e2e_seconds"] for row in cached_rows], 95))}},
        "edge_counts": {"first_case": rows[0]["query_edge_counts"], "all_cases_real_edge_count_min": min(int(row["query_edge_counts"].get("r2p_edge_indices") or 0) for row in rows), "all_cases_real_edge_count_max": max(int(row["query_edge_counts"].get("r2p_edge_indices") or 0) for row in rows)},
        "peak_gpu_memory": peak,
        "raw_rows": rows,
        "hard_boundaries": {"training_started": False, "p1i_test_iid_accessed": False, "sealed_accessed": False, "deepoheat_official100_accessed": False},
    }


def valid_source_indices(labels_root: Path) -> list[int]:
    receipt = json.loads((labels_root / "label_generation_receipt.json").read_text(encoding="utf-8"))
    rows = [row for row in receipt["rows"] if row.get("role") == "valid"]
    if len(rows) != 128:
        raise ValueError("valid row count is not 128")
    return [int(row["source_index"]) for row in rows]


def load_deepoheat_model(upstream_root: Path, checkpoint: Path) -> Any:
    import equinox as eqx
    import jax.numpy as jnp

    sys.path.insert(0, str(upstream_root))
    from models import DeepOHeat_v1  # type: ignore

    template = DeepOHeat_v1(dim=3, branch_dim=101**2, field_dim=1, branch_depth=8, branch_hidden=256, trunk_depth=3, trunk_hidden=64, rank=128, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(checkpoint, template)


def profile_one_deepoheat(args: argparse.Namespace, checkpoint: Path, role: str) -> dict[str, Any]:
    import equinox as eqx
    import jax.numpy as jnp

    model = load_deepoheat_model(args.deepoheat_upstream, checkpoint)
    fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    source_indices = valid_source_indices(args.labels_root)
    x = jnp.asarray(np.linspace(0.0, 1.0, 101).reshape(-1, 1), dtype=jnp.float32)
    y = jnp.asarray(np.linspace(0.0, 1.0, 101).reshape(-1, 1), dtype=jnp.float32)
    z = jnp.asarray(np.linspace(0.0, 0.55, 56).reshape(-1, 1), dtype=jnp.float32)

    @eqx.filter_jit
    def predict(current_model: Any, functions: Any) -> Any:
        return current_model(((x, y, z), functions))

    batches = [np.asarray(fs_train[source_indices[start : start + 4]], dtype=np.float32).reshape(-1, 101**2) for start in range(0, 128, 4)]
    peak: dict[str, Any] = {}
    cold_before = memory_stats()
    cold_started = time.perf_counter()
    cold_output = predict(model, jnp.asarray(batches[0]))
    block_until_ready(cold_output)
    cold_host = np.asarray(cold_output)
    cold_seconds = time.perf_counter() - cold_started
    cold_finite = bool(np.all(np.isfinite(cold_host)))
    peak = max_memory(peak, max_memory(cold_before, memory_stats()))
    model_only_rows: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        before = memory_stats()
        started = time.perf_counter()
        output = predict(model, jnp.asarray(batch))
        block_until_ready(output)
        model_seconds = time.perf_counter() - started
        host = np.asarray(output)
        finite = bool(np.all(np.isfinite(host)))
        after = memory_stats()
        peak = max_memory(peak, max_memory(before, after))
        model_only_rows.append({"batch": index, "model_only_seconds": float(model_seconds), "output_shape": list(host.shape), "finite": finite, "memory_before": before, "memory_after": after})
    e2e_rows: list[dict[str, Any]] = []
    for index, source_batch in enumerate([source_indices[start : start + 4] for start in range(0, 128, 4)]):
        before = memory_stats()
        started = time.perf_counter()
        functions = np.asarray(fs_train[source_batch], dtype=np.float32).reshape(-1, 101**2)
        device_input = jnp.asarray(functions)
        output = predict(model, device_input)
        block_until_ready(output)
        forward_done = time.perf_counter()
        host = np.asarray(output)
        _ = 25.0 * (host.astype(np.float64) - 0.2)
        post_done = time.perf_counter()
        after = memory_stats()
        peak = max_memory(peak, max_memory(before, after))
        e2e_rows.append({"batch": index, "e2e_seconds": float(post_done - started), "model_forward_plus_sync_seconds": float(forward_done - started), "postprocess_seconds": float(post_done - forward_done), "output_shape": list(host.shape), "finite": bool(np.all(np.isfinite(host))), "memory_before": before, "memory_after": after})
    cached_rows: list[dict[str, Any]] = []
    fixture = batches[0]
    for repeat in range(int(args.cached_repeats)):
        before = memory_stats()
        started = time.perf_counter()
        output = predict(model, jnp.asarray(fixture))
        block_until_ready(output)
        forward_done = time.perf_counter()
        host = np.asarray(output)
        _ = 25.0 * (host.astype(np.float64) - 0.2)
        total = time.perf_counter() - started
        post = time.perf_counter() - forward_done
        after = memory_stats()
        peak = max_memory(peak, max_memory(before, after))
        cached_rows.append({"repeat": repeat, "e2e_seconds": float(total), "model_only_seconds": float(total - post), "postprocess_seconds": float(post), "finite": bool(np.all(np.isfinite(host)))})
    steady_model = [row["model_only_seconds"] for row in model_only_rows]
    steady_e2e = [row["e2e_seconds"] for row in e2e_rows]
    return {
        "status": "PASS_VALID_ONLY_INFERENCE_PROFILE" if cold_finite and all(row["finite"] for row in model_only_rows + e2e_rows + cached_rows) else "FAIL_CLOSED_NONFINITE",
        "regime": "DeepOHeat-full-minus-valid128",
        "checkpoint_role": role,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "batch_size": 4,
        "valid_count": 128,
        "output_count": MESH_COUNT,
        "cold_e2e_seconds": float(cold_seconds),
        "model_only_seconds": {"median": median(steady_model), "p95": float(np.percentile(steady_model, 95)), "samples_per_second": 4.0 / median(steady_model)},
        "e2e_seconds": {"median": median(steady_e2e), "p95": float(np.percentile(steady_e2e, 95)), "samples_per_second": 4.0 / median(steady_e2e)},
        "postprocess_seconds": {"median": median([row["postprocess_seconds"] for row in e2e_rows]), "p95": float(np.percentile([row["postprocess_seconds"] for row in e2e_rows], 95))},
        "cached_fixture": {"repeats": cached_rows, "e2e_seconds": {"median": median([row["e2e_seconds"] for row in cached_rows]), "p95": float(np.percentile([row["e2e_seconds"] for row in cached_rows], 95))}},
        "peak_gpu_memory": peak,
        "raw_model_only_rows": model_only_rows,
        "raw_e2e_rows": e2e_rows,
        "hard_boundaries": {"training_started": False, "p1i_test_iid_accessed": False, "sealed_accessed": False, "deepoheat_official100_accessed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--heat3d-checkpoint", type=Path, required=True)
    parser.add_argument("--deepoheat-upstream", type=Path, required=True)
    parser.add_argument("--deepoheat-best", type=Path, required=True)
    parser.add_argument("--deepoheat-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cached-repeats", type=int, default=3)
    args = parser.parse_args()
    install_guard()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    for path in (args.fs_train, args.labels_root / "label_generation_receipt.json", args.subset_manifest, args.normalization, args.heat3d_config, args.heat3d_checkpoint, args.deepoheat_best, args.deepoheat_final):
        if not path.exists():
            raise FileNotFoundError(path)
    environment = gpu_environment()
    started = time.time()
    heat3d = profile_heat3d(args)
    deepoheat_best = profile_one_deepoheat(args, args.deepoheat_best, "validation_selected_best")
    deepoheat_final = profile_one_deepoheat(args, args.deepoheat_final, "final_endpoint")
    receipt = {
        "schema_version": "heat3d_v7_g2_p19_valid_only_inference_profile_v1",
        "status": "PASS_VALID_ONLY_INFERENCE_PROFILES" if all(item["status"] == "PASS_VALID_ONLY_INFERENCE_PROFILE" for item in (heat3d, deepoheat_best, deepoheat_final)) else "FAIL_CLOSED",
        "scope": "same devbox GPU, same valid128, frozen checkpoint inference only",
        "runner": {"script": "scripts/profile_v7_g2_p19_inference.py", "repo_commit_sha": repo_commit(), "runner_script_sha256": sha256_file(Path(__file__))},
        "environment": environment,
        "timing_contract": {"jax_block_until_ready": True, "cached_repeats": int(args.cached_repeats), "test_or_sealed_access": False, "accuracy_used": False},
        "profiles": {"heat3d": heat3d, "deepoheat_best": deepoheat_best, "deepoheat_final": deepoheat_final},
        "data_provenance": {"fs_train_sha256": sha256_file(args.fs_train), "label_receipt_sha256": sha256_file(args.labels_root / "label_generation_receipt.json"), "subset_manifest_sha256": sha256_file(args.subset_manifest), "normalization_payload_sha256": NORMALIZATION_SHA, "valid_count": 128, "target_truth_loaded": False},
        "checkpoint_reload_provenance": {"heat3d_checkpoint_loaded": True, "deepoheat_best_loaded": True, "deepoheat_final_loaded": True, "weights_modified": False},
        "wall_clock_started_unix": started,
        "wall_clock_finished_unix": time.time(),
        "hard_boundaries": {"training_started": False, "p1i_test_iid_accessed": False, "sealed_accessed": False, "deepoheat_official100_accessed": False, "existing_checkpoint_overwritten": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
