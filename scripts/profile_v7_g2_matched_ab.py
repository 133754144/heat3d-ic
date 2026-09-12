#!/usr/bin/env python3
"""Matched runtime-only A/B benchmark for frozen Heat3D workloads.

Case A is the frozen G1 P1i B24 workload and case B is the frozen
DeepOHeat-v1 adaptation B24 workload.  Each case is prepared and timed on the
same process/device/JAX runtime, with no accuracy metric or forbidden split
opened.  The benchmark is deliberately bounded to one warm train step and
one single-forward validation step per workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not command.is_file():
        command = Path("nvidia-smi")
    try:
        result = subprocess.run(
            [str(command), "--query-gpu=name,utilization.gpu,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows = []
    for line in result.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append({"name": values[0], "utilization_gpu_pct": values[1], "power_draw_w": values[2], "memory_used_mib": values[3], "memory_total_mib": values[4]})
    return {"available": bool(rows), "rows": rows, "command": str(command)}


def batch_shape_receipt(batches: list[Any]) -> dict[str, Any]:
    if not batches:
        return {"batch_count": 0}
    group = batches[0].groups[0]
    metadata = group["metadata"]
    names = ("x_pnodes_inp", "x_rnodes", "x_pnodes_out", "p2r_edge_indices", "r2r_edge_indices", "r2p_edge_indices")
    shapes = {}
    for name in names:
        value = getattr(metadata, name, None)
        shapes[name] = list(np.asarray(value).shape) if value is not None else None
    return {"batch_count": len(batches), "first_batch_id": batches[0].batch_id, "metadata_shapes": shapes}


def graph_counts(batches: list[Any]) -> dict[str, Any]:
    totals = {key: {"real": 0, "padded": 0} for key in ("p2r", "r2r", "r2p")}
    for batch in batches:
        metadata = batch.groups[0]["metadata"]
        n_p_in = int(np.asarray(metadata.x_pnodes_inp).shape[1] - 1)
        n_p_out = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
        n_r = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
        dummy_endpoints = {"p2r": (n_p_in, n_r), "r2r": (n_r, n_r), "r2p": (n_r, n_p_out)}
        for name, field in (("p2r", "p2r_edge_indices"), ("r2r", "r2r_edge_indices"), ("r2p", "r2p_edge_indices")):
            value = getattr(metadata, field, None)
            if value is None:
                continue
            for row in np.asarray(value):
                sender, receiver = dummy_endpoints[name]
                padded = int(np.count_nonzero((row[:, 0] == sender) & (row[:, 1] == receiver)))
                totals[name]["padded"] += padded
                totals[name]["real"] += int(len(row) - padded)
    for name, value in totals.items():
        value["padding_ratio"] = value["padded"] / max(value["real"], 1)
        value["total_slots"] = value["real"] + value["padded"]
    return totals


def prepare_g1(args: argparse.Namespace) -> dict[str, Any]:
    runner = load_script("run_heat3d_v7_formal_p1i_training.py")
    config = json.loads(args.g1_config.read_text(encoding="utf-8"))
    started = time.perf_counter()
    profile: dict[str, Any] = {}
    prepared = runner.prepare_p1i_data(
        args.g1_subset,
        args.g1_manifest,
        graph_config=config["graph"],
        model_config=config["model"],
        loss_config=config["loss"],
        batch_size=24,
        validation_batch_size=32,
        batch_build_seed=0,
        graph_seed=0,
        full_field_archive_path=args.g1_full_field,
        profile=profile,
    )
    return {"prepared": prepared, "config": config, "preparation_seconds": time.perf_counter() - started, "profile": profile, "variant": "Full"}


def prepare_v1(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_script("profile_v7_g2_training_efficiency.py")
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    config = json.loads(args.v1_config.read_text(encoding="utf-8"))
    stats = load_script("run_v7_g2_p6_heat3d_v1_formal.py").load_stats(args.v1_normalization)
    train_data = loader.CompactDeepOHeatV1Dataset(fs_train=args.v1_fs_train, labels_root=args.v1_labels_root, role="train")
    valid_data = loader.CompactDeepOHeatV1Dataset(fs_train=args.v1_fs_train, labels_root=args.v1_labels_root, role="valid", verify_source_file=False)
    mesh = support.mesh_arrays()
    started = time.perf_counter()
    train_examples = profile._build_examples(train_data, "train", mesh["coords"], mesh["control_volume"], mesh["layer_id"])
    valid_examples = profile._build_examples(valid_data, "valid", mesh["coords"], mesh["control_volume"], mesh["layer_id"])
    builder = __import__("rigno.graphBuilder_Heat3D", fromlist=["Heat3DGraphBuilder"]).Heat3DGraphBuilder(**config["graph"])
    from rigno.heat3d_training import build_p1i_batches
    batch_profile: dict[str, Any] = {}
    train_batches = build_p1i_batches(train_examples, stats, builder, label="g2_e2_ab_v1_train", batch_size=24, graph_seed=0, profile=batch_profile)
    valid_batches = build_p1i_batches(valid_examples, stats, builder, label="g2_e2_ab_v1_valid", batch_size=32, graph_seed=0, profile=batch_profile)
    all_examples = train_examples + valid_examples
    from rigno.heat3d_training.p1i import attach_input_contexts, attach_native_physics, attach_qk_features, fit_native_loss_references
    context = attach_input_contexts(train_batches + valid_batches, train_examples, all_examples, config["model"])
    by_id = {row.sample_id: row for row in all_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
        attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    return {"train_batches": train_batches, "valid_batches": valid_batches, "train_examples": train_examples, "valid_examples": valid_examples, "stats": stats, "config": config, "helper": helper, "batch_profile": batch_profile, "preparation_seconds": time.perf_counter() - started, "variant": "DeepOHeat-v1"}


def time_case(case: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    config = case["config"]
    train_batches = case.get("train_batches") or case["prepared"].train_batches
    valid_batches = case.get("valid_batches") or case["prepared"].valid_batches
    train_examples = case.get("train_examples") or case["prepared"].train_examples
    valid_examples = case.get("valid_examples") or case["prepared"].valid_examples
    stats = case.get("stats") or case["prepared"].stats
    if case["variant"] == "Full":
        runner = load_script("run_heat3d_v7_formal_p1i_training.py")
        model_config = runner._resolve_model_config(config["model"], tuple(stats["feature_names"]))
        loss_config = dict(case["prepared"].train_only_loss_references)
        from rigno.heat3d_training import model_apply_full, model_init_full, loss_fn_full
        apply_fn_builder = lambda model: (lambda current, batch, rng: model_apply_full(model, current, batch, rng))
        init_fn = model_init_full
        loss_builder = lambda prediction, batch, lc: loss_fn_full(prediction, batch, lc)
        optimizer_epochs = 200
        model_module = __import__("rigno.models.rigno", fromlist=["RIGNO"])
        from rigno.heat3d_training.p1i import fit_native_loss_references
        loss_config.update(fit_native_loss_references(train_examples, config["loss"]))
    else:
        helper = case["helper"]
        model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
        from rigno.heat3d_training import model_apply_full, model_init_full, loss_fn_full
        apply_fn_builder = lambda model: (lambda current, batch, rng: model_apply_full(model, current, batch, rng))
        init_fn = model_init_full
        loss_builder = lambda prediction, batch, lc: loss_fn_full(prediction, batch, lc)
        optimizer_epochs = 200
        loss_config = dict(config["loss"])
        from rigno.heat3d_training.p1i import fit_native_loss_references
        loss_config.update(fit_native_loss_references(train_examples, config["loss"]))
        model_module = __import__("rigno.models.rigno", fromlist=["RIGNO"])
    from rigno.heat3d_training import TrainingDependencies, V7FormalTrainer, block_until_ready, make_gradient_transform, make_p1i_optimizer
    model = model_module.RIGNO(**model_config)
    params = init_fn(model, jax.random.PRNGKey(seed), train_batches[0])["params"]
    optimizer = make_p1i_optimizer(config["optimizer"], epochs=optimizer_epochs, updates_per_epoch=len(train_batches))
    apply_fn = apply_fn_builder(model)
    batch_loss = lambda prediction, batch: loss_builder(prediction, batch, loss_config)
    deps = TrainingDependencies(data_source=case["variant"], feature_transform="frozen", normalization=stats, graph_builder=None, model=model, model_apply=apply_fn, loss_fn=batch_loss, optimizer=optimizer, batch_iterator=lambda x: x, validation_fn=lambda current, batch: batch_loss(apply_fn(current, batch, None), batch), checkpoint_writer=lambda p, v: None, metrics_fn=lambda c, b: {"runtime_only": True}, gradient_transform=make_gradient_transform(model_config, config["optimizer"]), validation_outputs_fn=lambda current, batch: (apply_fn(current, batch, None), batch_loss(apply_fn(current, batch, None), batch)))
    trainer = V7FormalTrainer(deps, jit_cache=True)
    state = trainer.initialize(params)
    block = block_until_ready
    before = gpu_snapshot()
    train_started = time.perf_counter()
    step = trainer.step(state, train_batches[0], jax.random.PRNGKey(seed))
    block((step.state.params, step.state.optimizer_state, step.loss, step.gradients, step.updates, step.prediction))
    train_seconds = time.perf_counter() - train_started
    after_train = gpu_snapshot()
    valid_started = time.perf_counter()
    prediction, valid_loss = trainer.validate_with_outputs(step.state, valid_batches[0])
    block((prediction, valid_loss))
    valid_seconds = time.perf_counter() - valid_started
    after_valid = gpu_snapshot()
    memory = jax.devices()[0].memory_stats() or {}
    return {"variant": case["variant"], "train_samples": len(train_examples), "valid_samples": len(valid_examples), "batch_size": 24, "validation_batch_size": 32, "train_batches": len(train_batches), "valid_batches": len(valid_batches), "preparation_seconds": case["preparation_seconds"], "graph_profile": case.get("profile") or case.get("batch_profile"), "graph_counts": graph_counts(train_batches), "batch_shapes": batch_shape_receipt(train_batches), "compile_count": trainer.compile_count, "xla_compile_occurrence": {"first_step_includes_compile": trainer.compile_count >= 1, "unique_executables": trainer.compile_count}, "warm_train_step_seconds": train_seconds, "valid_forward_seconds": valid_seconds, "host_device_transfer": {"status": "implicit_in_warm_step", "explicit_h2d_d2h_timer": False, "reason": "separate transfer timing requires profiler; no accuracy/protocol decision"}, "finite": bool(np.isfinite(float(np.asarray(step.loss))) and np.isfinite(float(np.asarray(valid_loss)))), "gpu_snapshots": {"before": before, "after_train": after_train, "after_valid": after_valid}, "jax_memory_stats": {str(k): int(v) for k, v in memory.items() if isinstance(v, (int, np.integer))}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-subset", type=Path, required=True)
    parser.add_argument("--g1-manifest", type=Path, required=True)
    parser.add_argument("--g1-full-field", type=Path, required=True)
    parser.add_argument("--g1-config", type=Path, required=True)
    parser.add_argument("--v1-fs-train", type=Path, required=True)
    parser.add_argument("--v1-labels-root", type=Path, required=True)
    parser.add_argument("--v1-normalization", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("matched benchmark output must remain under /tmp")
    if jax.default_backend() != "gpu":
        raise SystemExit("FAIL-CLOSED: matched benchmark requires JAX CUDA")
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit("FAIL-CLOSED: matched benchmark requires the frozen deterministic XLA flag")
    started = time.perf_counter()
    a = prepare_g1(args)
    a_receipt = time_case(a)
    b = prepare_v1(args)
    b_receipt = time_case(b)
    payload = {"schema_version": "heat3d_v7_g2_e2_matched_ab_runtime_v1", "status": "PASS_MATCHED_RUNTIME_FINITE" if a_receipt["finite"] and b_receipt["finite"] else "FAIL_NONFINITE", "same_process_device_jax": True, "case_A": a_receipt, "case_B": b_receipt, "interpretation": "runtime_only; no accuracy/truth/test; A and B use their respective frozen science configs with common B24/valid32 and legacy single-step path", "attribution": "deferred until measurements; stack versus graph-specific bottleneck is determined from preparation/graph/step components", "total_wall_seconds": time.perf_counter() - started, "environment": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS"), "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}, "official_test_accessed": False, "p1i_test_or_sealed_access": False, "formal_accuracy_claim_allowed": False}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "case_A": {k: a_receipt[k] for k in ("preparation_seconds", "warm_train_step_seconds", "valid_forward_seconds")}, "case_B": {k: b_receipt[k] for k in ("preparation_seconds", "warm_train_step_seconds", "valid_forward_seconds")}}, indent=2), flush=True)
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
