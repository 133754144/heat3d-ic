#!/usr/bin/env python3
"""Run one frozen V7 P1i e600 budget-replication seed.

This entrypoint is intentionally separate from the closed G1 runner.  It
loads the frozen V7 Full parent contract, permits exactly one budget change
(600 epochs and a 600-epoch cosine horizon), and writes only to the dedicated
ignored ``output/heat3d_v7_e600_budget/seed{seed}`` directory.  It never
enumerates test_iid or sealed rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from rigno.heat3d_training import (
    TrainingState,
    V7FormalTrainer,
    atomic_training_checkpoint,
    block_until_ready,
    learning_rate_for_epoch,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_init_full,
    prepare_p1i_data,
    tree_l2_norm,
    tree_parameter_count,
)
from rigno.heat3d_training.p1i import prediction_to_raw_delta
from scripts.evaluate_v7_g2_final_evidence import aggregate_seed, one_case_metrics
from scripts.run_heat3d_v7_formal_p1i_training import (
    _build_dependencies,
    _load_json,
    _resolve_model_config,
    _sha256,
    _variant_model_config,
)
from rigno.models.rigno import RIGNO


CONTRACT_PATH = ROOT / "configs" / "heat3d_v7" / "v7_e600_budget_replication.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 P1i e600 budget replication")
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--config", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--jit-cache", action="store_true", default=True)
    parser.add_argument("--no-jit-cache", action="store_false", dest="jit_cache")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_json(contract_path.resolve())
    if contract.get("status") != "frozen_prelaunch":
        raise ValueError("e600 contract is not frozen_prelaunch")
    if contract.get("experiment_id") != "V7-E600-Budget-Replication":
        raise ValueError("unexpected e600 experiment id")
    parent_path = (ROOT / str(contract["parent_config"])).resolve()
    if _sha256(parent_path) != contract.get("parent_config_sha256"):
        raise ValueError("parent V7 e200 config SHA drifted")
    parent = _load_json(parent_path)
    dataset = contract["dataset"]
    parent_dataset = parent["dataset"]
    for key in ("dataset_id", "manifest_sha256", "full_field_archive_sha256"):
        if dataset.get(key) != parent_dataset.get(key):
            raise ValueError(f"dataset binding drifted: {key}")
    for key in ("manifest_path", "subset_path", "full_field_archive_path"):
        if dataset.get(key) != parent_dataset.get(key):
            raise ValueError(f"dataset path binding drifted: {key}")
    if dataset["roles"] != {
        "train": 768,
        "valid_iid": 128,
        "test_iid": "forbidden",
        "sealed": "forbidden",
    }:
        raise ValueError("P1i role contract drifted")
    invariant = contract["frozen_scientific_invariants"]
    parent_batch = parent["batching"]
    if invariant["batch_size"] != parent_batch["batch_size"] or parent_batch["batch_size"] != 24:
        raise ValueError("batch size drifted")
    if invariant["validation_batch_size"] != parent_batch["validation_batch_size"] or parent_batch["validation_batch_size"] != 32:
        raise ValueError("validation batch size drifted")
    if invariant["train_batches_per_epoch"] != parent_batch["train_batches_per_epoch"] or parent_batch["train_batches_per_epoch"] != 32:
        raise ValueError("train batch count drifted")
    parent_optimizer = parent["optimizer"]
    checks = {
        "optimizer": parent_optimizer["optimizer"],
        "learning_rate": float(parent_optimizer["lr"]),
        "min_learning_rate": float(parent_optimizer["min_lr"]),
        "weight_decay": float(parent_optimizer["weight_decay"]),
        "gradient_clip_norm": float(parent_optimizer["gradient_clip_norm"]),
        "lr_schedule": parent_optimizer["lr_schedule"],
        "warmup_epochs": int(parent_optimizer["warmup_epochs"]),
    }
    for key, expected in checks.items():
        actual = invariant[key]
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=0.0):
                raise ValueError(f"optimizer contract drifted: {key}")
        elif actual != expected:
            raise ValueError(f"optimizer contract drifted: {key}")
    if invariant["cosine_horizon_epochs"] != 600:
        raise ValueError("e600 cosine horizon is not frozen")
    budget = contract["budget_change"]
    if budget.get("epochs") != 600 or budget.get("cosine_horizon_epochs") != 600:
        raise ValueError("e600 budget change drifted")
    if budget.get("resume_from_e200") is not False:
        raise ValueError("e600 must be fresh initialization")
    if any(bool(contract["frozen_scientific_invariants"].get(key)) for key in ("test_iid_access", "sealed_access", "deepoheat_official100_access")):
        raise ValueError("forbidden split access was enabled")
    return contract, parent


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _checkpoint_with_hash(path: Path, *, state: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    report = atomic_training_checkpoint(path, state=state, metadata=metadata)
    report = dict(report)
    report["sha256"] = _sha256_file(path)
    report["bytes"] = path.stat().st_size
    return report


def _valid_pass(
    trainer: V7FormalTrainer,
    state: TrainingState,
    prepared: Any,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(example.sample_id): example for example in prepared.valid_examples}
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for batch in prepared.valid_batches:
        predictions, loss = trainer.validate_with_outputs(state, batch)
        losses.append(float(loss))
        for prediction, group in zip(predictions, batch.groups, strict=True):
            raw_prediction = prediction_to_raw_delta(
                prediction, variant="Full", stats=prepared.stats
            )
            raw_prediction = np.asarray(raw_prediction)[:, 0, :, 0]
            truth = np.asarray(group["target_delta_raw"])[:, 0, :, 0]
            for row_index, sample_id in enumerate(group["sample_ids"]):
                example = by_id[str(sample_id)]
                metrics = one_case_metrics(
                    raw_prediction[row_index],
                    truth[row_index],
                    example.v6_operator_point_weights(),
                )
                metrics["sample_id"] = str(sample_id)
                metrics["split"] = "valid_iid"
                rows.append(metrics)
    aggregate = aggregate_seed(rows)
    aggregate["valid_loss"] = float(np.mean(losses)) if losses else 0.0
    return aggregate["valid_loss"], aggregate["metrics"], rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = args.config.resolve()
    contract, parent = _validate_contract(contract_path)
    seed = int(args.seed)
    expected_output = (
        ROOT / contract["output_contract"]["output_root"] / f"seed{seed}"
    ).resolve()
    output_dir = (args.output_dir or expected_output).resolve()
    if output_dir != expected_output:
        raise ValueError("output directory must be the frozen seed-specific path")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    contract_sha = _sha256_file(contract_path)
    parent_config_path = (ROOT / contract["parent_config"]).resolve()
    dataset_config = parent["dataset"]
    subset = (ROOT / dataset_config["subset_path"]).resolve()
    manifest = (ROOT / dataset_config["manifest_path"]).resolve()
    if not subset.exists() or not manifest.exists():
        raise FileNotFoundError(f"missing frozen P1i subset or manifest: {subset}, {manifest}")

    preparation_profile: dict[str, Any] = {}
    variant = "Full"
    parent_model_config = dict(parent["model"])
    preparation_model_config = _variant_model_config(parent_model_config, variant)
    prepared = prepare_p1i_data(
        subset,
        manifest,
        graph_config=parent["graph"],
        model_config=preparation_model_config,
        loss_config=parent["loss"],
        batch_size=24,
        validation_batch_size=32,
        batch_build_seed=seed,
        graph_seed=seed,
        profile=preparation_profile,
    )
    if len(prepared.train_examples) != 768 or len(prepared.valid_examples) != 128:
        raise ValueError("prepared P1i split counts drifted")
    model_config = _resolve_model_config(parent_model_config, tuple(prepared.stats["feature_names"]))
    loss_config = dict(prepared.train_only_loss_references)
    model = RIGNO(**model_config)
    init_start = time.perf_counter()
    params = model_init_full(model, jax.random.PRNGKey(seed), prepared.train_batches[0])["params"]
    init_seconds = time.perf_counter() - init_start
    optimizer = make_p1i_optimizer(parent["optimizer"], epochs=600, updates_per_epoch=len(prepared.train_batches))
    dependencies = _build_dependencies(
        model=model,
        prepared=prepared,
        model_config=model_config,
        loss_config=loss_config,
        optimizer=optimizer,
        parent_config=parent,
        variant=variant,
        dataset_config=dataset_config,
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=bool(args.jit_cache))
    state = trainer.initialize(params)
    progress_path = output_dir / "progress.json"
    _write_json(
        progress_path,
        {
            "schema_version": "heat3d_v7_e600_budget_progress_v1",
            "status": "RUNNING",
            "seed": seed,
            "epoch": 0,
            "epochs": 600,
            "contract_sha256": contract_sha,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )

    history: list[dict[str, Any]] = []
    step_times: list[float] = []
    validation_times: list[float] = []
    best_sample_value = float("inf")
    best_sample_epoch: int | None = None
    best_point_value = float("inf")
    best_point_epoch: int | None = None
    best_sample_report: dict[str, Any] | None = None
    best_point_report: dict[str, Any] | None = None
    started_total = time.perf_counter()
    for epoch in range(1, 601):
        order = np.random.default_rng(seed + epoch).permutation(len(prepared.train_batches))
        epoch_start = time.perf_counter()
        losses: list[float] = []
        for batch_index, raw_index in enumerate(order, start=1):
            batch = prepared.train_batches[int(raw_index)]
            step_start = time.perf_counter()
            step_key = jax.random.fold_in(jax.random.PRNGKey(seed), epoch)
            step_key = jax.random.fold_in(step_key, batch_index)
            result = trainer.step(state, batch, rng=step_key)
            state = result.state
            block_until_ready((state.params, state.optimizer_state, result.loss, result.gradients, result.updates, result.prediction))
            loss_value = float(result.loss)
            if not np.isfinite(loss_value) or not np.isfinite(tree_l2_norm(result.gradients)):
                raise FloatingPointError(f"non-finite train state at epoch={epoch}, batch={batch_index}")
            losses.append(loss_value)
            step_times.append(time.perf_counter() - step_start)
        valid_start = time.perf_counter()
        valid_loss, metrics, rows = _valid_pass(trainer, state, prepared)
        validation_seconds = time.perf_counter() - valid_start
        validation_times.append(validation_seconds)
        sample_value = float(metrics["sample_first_relative_rmse_pct"])
        point_value = float(metrics["point_global_relative_rmse_pct"])
        for name, value in (("sample_first_relative_rmse_pct", sample_value), ("point_global_relative_rmse_pct", point_value)):
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite validation metric {name} at epoch={epoch}")
        if sample_value < best_sample_value:
            best_sample_value = sample_value
            best_sample_epoch = epoch
            best_sample_report = _checkpoint_with_hash(
                output_dir / "params_best_sample_first.pkl",
                state=state,
                metadata={
                    "checkpoint_role": "primary_sample_first_best",
                    "selection_metric": "sample_first_relative_rmse_pct",
                    "selection_metric_value": sample_value,
                    "epoch": epoch,
                    "seed": seed,
                    "contract_sha256": contract_sha,
                    "test_iid_access": False,
                    "sealed_access": False,
                },
            )
        if point_value < best_point_value:
            best_point_value = point_value
            best_point_epoch = epoch
            best_point_report = _checkpoint_with_hash(
                output_dir / "params_best_point_global.pkl",
                state=state,
                metadata={
                    "checkpoint_role": "secondary_point_global_best",
                    "selection_metric": "point_global_relative_rmse_pct",
                    "selection_metric_value": point_value,
                    "epoch": epoch,
                    "seed": seed,
                    "contract_sha256": contract_sha,
                    "test_iid_access": False,
                    "sealed_access": False,
                },
            )
        latest_report = _checkpoint_with_hash(
            output_dir / "latest_epoch.pkl",
            state=state,
            metadata={
                "checkpoint_role": "latest_epoch_atomic",
                "epoch": epoch,
                "seed": seed,
                "contract_sha256": contract_sha,
                "test_iid_access": False,
                "sealed_access": False,
            },
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "valid_loss": valid_loss,
                "metrics": metrics,
                "best_sample_epoch": best_sample_epoch,
                "best_sample_value": best_sample_value,
                "best_point_epoch": best_point_epoch,
                "best_point_value": best_point_value,
                "learning_rate": learning_rate_for_epoch(epoch, epochs=600, updates_per_epoch=len(prepared.train_batches), config=parent["optimizer"]),
                "validation_seconds": validation_seconds,
                "epoch_wall_seconds": time.perf_counter() - epoch_start,
                "train_order_hash": hashlib.sha256(_json_canonical([prepared.train_batches[int(index)].batch_id for index in order])).hexdigest(),
                "valid_case_count": len(rows),
                "latest_checkpoint_sha256": latest_report["sha256"],
            }
        )
        _write_json(
            progress_path,
            {
                "schema_version": "heat3d_v7_e600_budget_progress_v1",
                "status": "RUNNING",
                "seed": seed,
                "epoch": epoch,
                "epochs": 600,
                "best_sample_epoch": best_sample_epoch,
                "best_sample_metric": best_sample_value,
                "best_point_epoch": best_point_epoch,
                "best_point_metric": best_point_value,
                "contract_sha256": contract_sha,
                "test_iid_access": False,
                "sealed_access": False,
            },
        )

    final_report = _checkpoint_with_hash(
        output_dir / "params_final_e600.pkl",
        state=state,
        metadata={
            "checkpoint_role": "final_e600",
            "epoch": 600,
            "seed": seed,
            "contract_sha256": contract_sha,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )
    with (output_dir / "params_final_e600.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    reloaded = TrainingState(
        params=payload["params"], optimizer_state=payload["optimizer_state"], step=int(payload["step"])
    )
    reload_valid_loss, reload_metrics, _reload_rows = _valid_pass(trainer, reloaded, prepared)
    if not np.isfinite(reload_valid_loss) or not np.isfinite(reload_metrics["sample_first_relative_rmse_pct"]):
        raise FloatingPointError("final checkpoint reload produced non-finite validation")
    runner_sha = _git("rev-parse", "HEAD")
    receipt = {
        "schema_version": "heat3d_v7_e600_budget_replication_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "experiment_id": contract["experiment_id"],
        "experiment_role": contract["experiment_role"],
        "seed": seed,
        "repo_commit_sha": runner_sha,
        "runner_path": "scripts/run_heat3d_v7_e600_budget_replication.py",
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": contract_sha,
        "parent_config_sha256": _sha256(parent_config_path),
        "dataset": contract["dataset"],
        "split_counts": {"train": len(prepared.train_examples), "valid_iid": len(prepared.valid_examples)},
        "batching": {"batch_size": 24, "validation_batch_size": 32, "train_batches_per_epoch": 32, "drop_last": False},
        "optimization": {"epochs": 600, "cosine_horizon_epochs": 600, **parent["optimizer"], "initialization": "random"},
        "checkpoint_selection": {
            "primary_metric": "sample_first_relative_rmse_pct",
            "tie_break": "earliest_epoch",
            "best_epoch": best_sample_epoch,
            "best_value": best_sample_value,
            "primary_checkpoint": best_sample_report,
            "point_global_best_epoch": best_point_epoch,
            "point_global_best_value": best_point_value,
            "point_global_checkpoint": best_point_report,
            "final_checkpoint": final_report,
        },
        "final_reload": {"validation_loss": reload_valid_loss, "metrics": reload_metrics, "state_tree_round_trip": bool(final_report["passed"])},
        "metrics_at_selected_checkpoint": history[best_sample_epoch - 1]["metrics"] if best_sample_epoch else None,
        "metrics_at_point_global_checkpoint": history[best_point_epoch - 1]["metrics"] if best_point_epoch else None,
        "metrics_at_final_checkpoint": history[-1]["metrics"],
        "history": history,
        "training_wall_seconds": time.perf_counter() - started_total,
        "step_timing": {"count": len(step_times), "total_seconds": float(sum(step_times)), "median_seconds": float(np.median(step_times)), "p95_seconds": float(np.percentile(step_times, 95))},
        "validation_timing": {"epochs": len(validation_times), "total_seconds": float(sum(validation_times)), "median_seconds": float(np.median(validation_times)), "p95_seconds": float(np.percentile(validation_times, 95))},
        "parameter_count": tree_parameter_count(params),
        "model_initialization_seconds": init_seconds,
        "compile_count": trainer.compile_count,
        "jit_cache": bool(args.jit_cache),
        "device": [str(device) for device in jax.devices()],
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
        "test_iid_access": False,
        "sealed_access": False,
        "deepoheat_official100_access": False,
        "u_v2_full_field_evaluation": False,
        "g2_v6_evidence_modified": False,
    }
    receipt_path = output_dir / "v7_e600_budget_replication_receipt.json"
    _write_json(receipt_path, receipt)
    _write_json(
        progress_path,
        {
            "schema_version": "heat3d_v7_e600_budget_progress_v1",
            "status": "COMPLETE",
            "seed": seed,
            "epoch": 600,
            "epochs": 600,
            "best_sample_epoch": best_sample_epoch,
            "best_sample_metric": best_sample_value,
            "best_point_epoch": best_point_epoch,
            "best_point_metric": best_point_value,
            "receipt_sha256": _sha256_file(receipt_path),
            "contract_sha256": contract_sha,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return receipt


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        contract, parent = _validate_contract(args.config.resolve())
        print(json.dumps({"mode": "dry_run", "contract": contract["experiment_id"], "parent_config_sha256": contract["parent_config_sha256"], "epochs": contract["budget_change"]["epochs"], "seed": args.seed, "test_iid_access": False, "sealed_access": False, "parent_model_keys": len(parent["model"])}, indent=2, sort_keys=True))
        return 0
    try:
        _run(args)
    except Exception as exc:
        output_dir = (ROOT / "output" / "heat3d_v7_e600_budget" / f"seed{args.seed}").resolve()
        if output_dir.exists():
            failure = {
                "schema_version": "heat3d_v7_e600_budget_replication_failure_v1",
                "status": "FAILED_FAIL_CLOSED",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "repo_commit_sha": _git("rev-parse", "HEAD"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "test_iid_access": False,
                "sealed_access": False,
            }
            _write_json(output_dir / "failure_receipt.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
