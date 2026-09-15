#!/usr/bin/env python3
"""DeepOHeat-v1 matched physical-case-budget training for V7 G2.

This runner keeps the released DeepOHeat-v1 architecture, full-mesh PDE/BC
loss, sampling semantics, optimizer and iteration schedule intact.  The only
planned experimental change is the source pool: the frozen 768 Heat3D train
case IDs replace the official 100,000-function pool.  The frozen 128 valid
case IDs are used only for a temperature-space selection/diagnostic metric.

The official test input/field files are deliberately not accepted or opened.
Large checkpoints and metrics are written to an external output directory on
the execution host, never to the repository.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


UPSTREAM_SHA = "3ef3d9c41666a56b5940b39a61166ccaa5aaedb2"
FS_TRAIN_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
SUBSET_SHA256 = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
LABEL_RECEIPT_SHA256 = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
MESH_SHAPE = (101, 101, 56)
BRANCH_DIM = 101**2
BATCH_FUNCTIONS = 50
ITERATIONS = 100_000
VALIDATION_INTERVAL = 10_000
VALID_BATCH_FUNCTIONS = 4
FORBIDDEN_TEST_FILES = {"fs_test_volume.npy", "u_test_volume.npy"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def repo_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def finite_tree(value: Any) -> bool:
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            return False
    return True


def install_test_file_guard() -> None:
    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(candidate)).name
            if name in FORBIDDEN_TEST_FILES:
                raise PermissionError(
                    f"FAIL-CLOSED: matched training attempted to open sealed {name}"
                )

    sys.addaudithook(audit)


def decode_indices(manifest: dict[str, Any], role: str) -> np.ndarray:
    row = manifest["roles"][role]
    values = np.frombuffer(
        base64.b64decode(row["indices_base64"]), dtype="<u4"
    ).astype(np.int64)
    if len(values) != int(row["count"]):
        raise ValueError(f"{role} subset count mismatch")
    return values


def load_contract(
    *, fs_train_path: Path, subset_path: Path, labels_root: Path
) -> tuple[np.memmap, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    if fs_train_path.name != "fs_train_volume.npy":
        raise ValueError("only the official fs_train_volume.npy is accepted")
    if file_sha256(fs_train_path) != FS_TRAIN_SHA256:
        raise ValueError("official fs_train_volume.npy SHA mismatch")
    if file_sha256(subset_path) != SUBSET_SHA256:
        raise ValueError("frozen 768/128 subset manifest SHA mismatch")
    receipt_path = labels_root / "label_generation_receipt.json"
    if file_sha256(receipt_path) != LABEL_RECEIPT_SHA256:
        raise ValueError("expanded label receipt SHA mismatch")
    manifest = json.loads(subset_path.read_text(encoding="utf-8"))
    if manifest["selection"]["accuracy_or_temperature_observed"] is not False:
        raise ValueError("subset selection was not temperature/accuracy blind")
    train_indices = decode_indices(manifest, "train")
    valid_indices = decode_indices(manifest, "valid")
    if len(train_indices) != 768 or len(valid_indices) != 128:
        raise ValueError("matched contract requires exactly 768 train and 128 valid IDs")
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError("train/valid source index overlap")
    fs_train = np.load(fs_train_path, mmap_mode="r", allow_pickle=False)
    if fs_train.shape != (100000, 101, 101) or fs_train.dtype != np.float64:
        raise ValueError(f"official source shape/dtype mismatch: {fs_train.shape}/{fs_train.dtype}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt.get("rows", [])
    for role, indices in (("train", train_indices), ("valid", valid_indices)):
        role_rows = [row for row in rows if row.get("role") == role]
        if [int(row["source_index"]) for row in role_rows] != indices.tolist():
            raise ValueError(f"{role} row ordering/source IDs drifted")
        for row in role_rows:
            source = np.asarray(fs_train[int(row["source_index"])])
            if canonical_array_sha256(source) != row["source_input_sha256"]:
                raise ValueError(f"source input SHA drift: {row['sample_id']}")
    return fs_train, manifest, receipt, train_indices, valid_indices


def mesh_arrays() -> tuple[np.ndarray, np.ndarray]:
    axes = (
        np.linspace(0.0, 1.0, 101),
        np.linspace(0.0, 1.0, 101),
        np.linspace(0.0, 0.55, 56),
    )
    one_d = []
    for axis in axes:
        spacing = float(axis[1] - axis[0])
        weights = np.full(len(axis), spacing, dtype=np.float64)
        weights[[0, -1]] *= 0.5
        one_d.append(weights)
    cv = np.einsum("i,j,k->ijk", *one_d).reshape(-1)
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return coords, cv


def load_valid_rows(labels_root: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in receipt["rows"]:
        if row.get("role") != "valid":
            continue
        directory = labels_root / "valid" / row["sample_id"]
        artifact = row["artifacts"]["full_reference"]
        path = directory / artifact["file"]
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != list(artifact["shape"]) or str(value.dtype) != artifact["dtype"]:
            raise ValueError(f"valid reference schema drift: {row['sample_id']}")
        if canonical_array_sha256(value) != artifact["sha256"]:
            raise ValueError(f"valid reference SHA drift: {row['sample_id']}")
        result.append({"row": row, "truth": value})
    if len(result) != 128:
        raise ValueError("valid reference row count is not 128")
    return result


def save_eqx_checkpoint(path: Path, model: Any) -> str:
    temporary = path.with_name(path.name + ".tmp")
    eqx.tree_serialise_leaves(temporary, model)
    temporary.replace(path)
    return file_sha256(path)


def save_optimizer_state(path: Path, opt_state: Any) -> str:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(opt_state), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return file_sha256(path)


def evaluate_valid(
    *, model: Any, valid_rows: list[dict[str, Any]], fs_train: np.memmap,
    labels_root: Path, x: jax.Array, y: jax.Array, z: jax.Array,
    cv: np.ndarray,
) -> dict[str, Any]:
    # This is an execution wrapper only.  It calls the unchanged upstream
    # model on batches of four valid functions, then computes the frozen
    # temperature-space sample-first CV-relative RMSE on the 571256-node field.
    @eqx.filter_jit
    def predict(current_model: Any, functions: jax.Array) -> jax.Array:
        return current_model(((x, y, z), functions))

    relative_values: list[float] = []
    rmse_values: list[float] = []
    peak_values: list[float] = []
    for start in range(0, len(valid_rows), VALID_BATCH_FUNCTIONS):
        chunk = valid_rows[start : start + VALID_BATCH_FUNCTIONS]
        functions = np.stack(
            [np.asarray(fs_train[int(entry["row"]["source_index"])], dtype=np.float32).reshape(-1) for entry in chunk],
            axis=0,
        )
        prediction = np.asarray(
            jax.block_until_ready(predict(model, jnp.asarray(functions)))
        ).reshape(len(chunk), *MESH_SHAPE)
        for index, entry in enumerate(chunk):
            truth = np.asarray(entry["truth"], dtype=np.float64).reshape(-1)
            pred_delta = 25.0 * (prediction[index].astype(np.float64).reshape(-1) - 0.2)
            error = pred_delta - truth
            sse = float(np.sum(cv * error * error))
            energy = float(np.sum(cv * truth * truth))
            relative_values.append(float(np.sqrt(sse / max(energy, 1.0e-30)) * 100.0))
            rmse_values.append(float(np.sqrt(sse / np.sum(cv))))
            peak_values.append(float(np.max(pred_delta) - np.max(truth)))
    if not all(np.isfinite(relative_values + rmse_values + peak_values)):
        raise FloatingPointError("nonfinite valid temperature-space metric")
    return {
        "sample_count": len(relative_values),
        "sample_first_relative_rmse_pct": float(np.mean(relative_values)),
        "sample_first_relative_rmse_sample_sd_pct": float(np.std(relative_values, ddof=1)),
        "rmse_K_mean": float(np.mean(rmse_values)),
        "peak_error_K_rms": float(np.sqrt(np.mean(np.asarray(peak_values) ** 2))),
        "valid_source_indices": [int(entry["row"]["source_index"]) for entry in valid_rows],
        "labels_read_by_inference": True,
        "selection_split": "valid",
        "temperature_space": "deltaT_K=25*(u-0.2); full 101x101x56 CV-weighted evaluator",
    }


def accelerator_receipt() -> dict[str, Any]:
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise SystemExit("FAIL-CLOSED: matched DeepOHeat-v1 requires JAX CUDA")
    stats = devices[0].memory_stats() or {}
    return {
        "device": str(devices[0]),
        "platform": devices[0].platform,
        "memory_stats": stats,
    }


def contract_payload(*, seed: int, source_indices: np.ndarray) -> dict[str, Any]:
    return {
        "schema_version": "heat3d_v7_g2_p14_deepoheat_v1_same_physical_case_budget_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "classification": "SAME_PHYSICAL_CASE_BUDGET",
        "not_same_information_budget": True,
        "upstream": {
            "repo": "xlyu0127/DeepOHeat-v1",
            "commit": UPSTREAM_SHA,
            "architecture": "DeepOHeat_v1(dim=3, branch_dim=101^2, field_dim=1, branch_depth=8, branch_hidden=256, trunk_depth=3, trunk_hidden=64, rank=128)",
            "mesh": list(MESH_SHAPE),
            "pde_bc_semantics": "official apply_model_deepoheat_st full PDE and Robin/adiabatic BC",
            "sampling": "JAX key split then random.choice replace=False",
            "optimizer": "Optax Adam",
            "lr": 1.0e-3,
            "schedule": "exponential_decay(0.9 per 1000 iterations)",
            "batch_functions": BATCH_FUNCTIONS,
            "iterations": ITERATIONS,
            "normalization": "none; native nondimensional u",
        },
        "planned_variation": "only training physical-case source pool changes from official 100000 to frozen 768 IDs",
        "seed": seed,
        "source_indices": {
            "count": int(len(source_indices)),
            "little_endian_int64_sha256": hashlib.sha256(np.asarray(source_indices, dtype="<i8").tobytes()).hexdigest(),
        },
        "valid_selection": {
            "count": 128,
            "interval_iterations": VALIDATION_INTERVAL,
            "metric": "sample_first_relative_rmse_pct",
            "domain": "full 101x101x56 valid field",
            "tie_break": "earliest iteration",
            "official_test_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract-check", "preflight", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve()
    if repo_sha(upstream) != UPSTREAM_SHA:
        raise ValueError("DeepOHeat-v1 upstream SHA mismatch")
    # Contract checking does not initialize JAX or touch data.
    if args.mode == "contract-check":
        print(json.dumps({
            "status": "PASS_MATCHED_CONTRACT_CHECK_NO_TRAINING",
            "classification": "SAME_PHYSICAL_CASE_BUDGET",
            "planned_variation": "768 train IDs only",
            "train": 768, "valid": 128, "batch_functions": BATCH_FUNCTIONS,
            "iterations": ITERATIONS, "official_test_access": False,
        }, indent=2, sort_keys=True))
        return 0

    install_test_file_guard()
    fs_train, manifest, label_receipt, train_indices, valid_indices = load_contract(
        fs_train_path=args.fs_train, subset_path=args.subset_manifest, labels_root=args.labels_root
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty matched output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = contract_payload(seed=args.seed, source_indices=train_indices)
    (args.output_dir / "matched_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    device_before = accelerator_receipt()
    sys.path.insert(0, str(upstream))
    from heat_volumetric import apply_model_deepoheat_st  # type: ignore
    from models import DeepOHeat_v1  # type: ignore
    from train import update as upstream_update  # type: ignore

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key, 2)
    model = DeepOHeat_v1(
        dim=3, branch_dim=BRANCH_DIM, field_dim=1,
        branch_depth=8, branch_hidden=256,
        trunk_depth=3, trunk_hidden=64, rank=128, key=model_key,
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    parameter_count = sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(params))
    optimizer = optax.adam(optax.exponential_decay(1.0e-3, 1000, 0.9))
    opt_state = optimizer.init(params)
    key, train_key = jax.random.split(key, 2)
    _coords_np, cv = mesh_arrays()
    x = jnp.asarray(np.linspace(0, 1, 101).reshape(-1, 1), dtype=jnp.float32)
    y = jnp.asarray(np.linspace(0, 1, 101).reshape(-1, 1), dtype=jnp.float32)
    z = jnp.asarray(np.linspace(0, 0.55, 56).reshape(-1, 1), dtype=jnp.float32)
    if args.mode == "preflight":
        iterations = 1
        valid_rows: list[dict[str, Any]] = []
    else:
        iterations = ITERATIONS
        valid_rows = load_valid_rows(args.labels_root, label_receipt)
    train_started = time.perf_counter()
    first_step_seconds: float | None = None
    last_loss: float | None = None
    validation_history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_iteration: int | None = None
    best_checkpoint: dict[str, str] | None = None
    best_validation: dict[str, Any] | None = None
    progress_path = args.output_dir / "progress.json"

    for iteration_zero in range(iterations):
        iteration = iteration_zero + 1
        train_key, sample_key = jax.random.split(train_key)
        choice_key, _unused = jax.random.split(sample_key)
        local_indices = np.asarray(
            jax.device_get(jax.random.choice(choice_key, train_indices.shape[0], (BATCH_FUNCTIONS,), replace=False)),
            dtype=np.int64,
        )
        source_batch = train_indices[local_indices]
        functions = jnp.asarray(
            np.asarray(fs_train[source_batch], dtype=np.float32).reshape(BATCH_FUNCTIONS, BRANCH_DIM)
        )
        step_started = time.perf_counter()
        loss, gradients = apply_model_deepoheat_st(model, x, y, z, functions)
        model, opt_state = upstream_update(gradients, optimizer, opt_state, model)
        jax.block_until_ready((loss, model, opt_state))
        step_seconds = time.perf_counter() - step_started
        if first_step_seconds is None:
            first_step_seconds = step_seconds
        if not np.isfinite(float(loss)) or not finite_tree(gradients) or not finite_tree(model):
            raise FloatingPointError(f"nonfinite loss/gradient/model at iteration {iteration}")
        last_loss = float(loss)
        if iteration % 100 == 0 or iteration == 1:
            print(f"iteration={iteration}/{iterations} physics_loss={last_loss:.9g}", flush=True)
        if args.mode == "train" and (iteration % VALIDATION_INTERVAL == 0 or iteration == ITERATIONS):
            valid_started = time.perf_counter()
            metrics = evaluate_valid(
                model=model, valid_rows=valid_rows, fs_train=fs_train,
                labels_root=args.labels_root, x=x, y=y, z=z, cv=cv,
            )
            metrics.update({"iteration": iteration, "wall_seconds": time.perf_counter() - valid_started, "physics_loss": last_loss})
            validation_history.append(metrics)
            metrics_path = args.output_dir / f"valid_iter_{iteration:06d}.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if metrics["sample_first_relative_rmse_pct"] < best_metric:
                best_metric = float(metrics["sample_first_relative_rmse_pct"])
                best_iteration = iteration
                model_sha = save_eqx_checkpoint(args.output_dir / "DeepOHeat_v1_best.eqx", model)
                opt_sha = save_optimizer_state(args.output_dir / "DeepOHeat_v1_best.opt.pkl", opt_state)
                best_checkpoint = {"model_file": "DeepOHeat_v1_best.eqx", "model_sha256": model_sha, "optimizer_file": "DeepOHeat_v1_best.opt.pkl", "optimizer_sha256": opt_sha}
                best_validation = dict(metrics)
        if args.mode == "train" and (iteration % 1000 == 0 or iteration == ITERATIONS):
            progress = {"status": "RUNNING", "seed": args.seed, "iteration": iteration, "iterations": ITERATIONS, "physics_loss": last_loss, "best_iteration": best_iteration, "best_metric": best_metric, "global_update_count": iteration}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(progress_path)

    if args.mode == "preflight":
        print(json.dumps({"status": "PASS_MATCHED_ONE_UPDATE", "seed": args.seed, "physics_loss": last_loss, "first_step_seconds": first_step_seconds, "parameter_count": parameter_count}, indent=2, sort_keys=True))
        return 0

    final_model_path = args.output_dir / "DeepOHeat_v1_final.eqx"
    final_opt_path = args.output_dir / "DeepOHeat_v1_final.opt.pkl"
    final_model_sha = save_eqx_checkpoint(final_model_path, model)
    final_opt_sha = save_optimizer_state(final_opt_path, opt_state)
    # Checkpoint integrity: reconstruct the same model structure and compare a
    # fixed valid function after loading. This does not inspect test data.
    reload_model = eqx.tree_deserialise_leaves(final_model_path, model)
    probe_functions = jnp.asarray(np.asarray(fs_train[valid_indices[:1]], dtype=np.float32).reshape(1, BRANCH_DIM))
    @eqx.filter_jit
    def probe(current_model: Any, functions: jax.Array) -> jax.Array:
        return current_model(((x, y, z), functions))
    pred_before = np.asarray(jax.block_until_ready(probe(model, probe_functions)))
    pred_after = np.asarray(jax.block_until_ready(probe(reload_model, probe_functions)))
    reload_max_abs = float(np.max(np.abs(pred_before - pred_after)))
    reload_rel = float(np.linalg.norm(pred_before - pred_after) / max(np.linalg.norm(pred_before), 1.0e-12))
    if reload_max_abs != 0.0 or reload_rel != 0.0:
        raise RuntimeError("checkpoint reload numerical state changed")
    device_after = accelerator_receipt()
    total_seconds = time.perf_counter() - train_started
    receipt = {
        **contract,
        "status": "COMPLETE_MATCHED_PHYSICAL_CASE_BUDGET_TRAINING",
        "execution": {
            "runner_sha": repo_sha(Path(__file__).resolve().parents[1]),
            "script": "scripts/run_v7_g2_p14_deepoheat_v1_matched.py",
            "upstream_sha": UPSTREAM_SHA,
            "fresh_start": True,
            "gpu_only": True,
            "serial_gpu_task": True,
            "official_test_access": False,
        },
        "data": {
            "fs_train_file": str(args.fs_train), "fs_train_sha256": FS_TRAIN_SHA256,
            "subset_manifest": str(args.subset_manifest), "subset_manifest_sha256": SUBSET_SHA256,
            "labels_root": str(args.labels_root), "label_receipt_sha256": LABEL_RECEIPT_SHA256,
            "train_case_count": 768, "valid_case_count": 128,
            "train_indices_sha256": contract["source_indices"]["little_endian_int64_sha256"],
        },
        "model": {"parameter_count": parameter_count, "mesh_shape": list(MESH_SHAPE), "pde_collocation_evaluations": BATCH_FUNCTIONS * ITERATIONS, "pde_functions_per_iteration": BATCH_FUNCTIONS},
        "metrics": {"selection_metric": "valid full-field sample_first_relative_rmse_pct", "selection_split": "valid", "best_iteration": best_iteration, "best_metric": best_metric, "best_validation": best_validation, "validation_history": validation_history, "final_physics_loss": last_loss},
        "runtime": {"total_wall_seconds": total_seconds, "first_step_wall_seconds": first_step_seconds, "average_step_seconds_excluding_first": (total_seconds - float(first_step_seconds or 0.0)) / max(ITERATIONS - 1, 1), "peak_rss_bytes": peak_rss_bytes(), "device_before": device_before, "device_after": device_after},
        "checkpoint": {"best": best_checkpoint, "final_model_file": final_model_path.name, "final_model_sha256": final_model_sha, "final_optimizer_file": final_opt_path.name, "final_optimizer_sha256": final_opt_sha, "reload": {"status": "PASS", "max_abs": reload_max_abs, "relative_l2": reload_rel}},
        "hard_boundaries": {"p1i_test_iid_accessed": False, "sealed_accessed": False, "deepoheat_official100_accessed": False, "multi_htc_started": False, "therm_fm_downloaded": False},
    }
    (args.output_dir / "formal_training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress = {"status": "COMPLETE", "seed": args.seed, "iteration": ITERATIONS, "best_iteration": best_iteration, "best_metric": best_metric, "global_update_count": ITERATIONS}
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Preserve a machine-readable failure receipt in the external output
        # directory when possible; never turn a failed run into a success.
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
