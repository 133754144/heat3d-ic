"""Registered V7 Full P1i training path.

The command is intentionally rehearsal-only until a later, explicit G1
authorization.  It requires a registered experiment ID and never exposes a
hidden dataset or split default.  The small V1 fixture remains in
``run_heat3d_v7_formal_training.py`` for CI/readiness tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any

import jax
import numpy as np

from rigno.heat3d_training import (
    TrainingDependencies,
    V7FormalTrainer,
    atomic_training_checkpoint,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_init_full,
    prepare_p1i_data,
    loss_fn_full,
)
from rigno.models.rigno import RIGNO


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "heat3d_v7" / "v7_experiment_registry.json"
FULL_CONFIG_PATH = ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 registered Full P1i rehearsal")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path, default=FULL_CONFIG_PATH)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/v7_g1_p1i_rehearsal"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jit-cache", action="store_true", default=True)
    parser.add_argument("--no-jit-cache", action="store_false", dest="jit_cache")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_model_config(model_config: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    resolved = dict(model_config)
    resolved.pop("architecture", None)
    resolved["global_context_feature_names"] = tuple(resolved["global_context_feature_names"])
    resolved["decoder_bypass_local_feature_names"] = tuple(
        resolved["decoder_bypass_local_feature_names"]
    )
    resolved["decoder_bypass_feature_names"] = resolved["decoder_bypass_local_feature_names"]
    resolved["decoder_bypass_feature_indices"] = tuple(
        feature_names.index(name) for name in resolved["decoder_bypass_local_feature_names"]
    )
    resolved["decoder_bypass_num_features"] = len(resolved["decoder_bypass_feature_indices"])
    return resolved


def _registered_full_config(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_json(REGISTRY_PATH)
    config = _load_json(args.config.resolve())
    registered = {
        entry.get("experiment_id"): entry
        for entry in registry.get("registered_runs", [])
    }
    entry = registered.get(args.experiment_id)
    if entry is None or args.experiment_id != "V7-G1-Full-P1i":
        raise ValueError("formal publication training requires the registered Full P1i experiment ID")
    if entry.get("status") != "registered_not_executed":
        raise ValueError("Full P1i is not in the registered-not-executed preflight state")
    if config.get("experiment_id") != args.experiment_id:
        raise ValueError("config and registered experiment ID do not match")
    if config.get("experiment_role") != "publication_training":
        raise ValueError("Full P1i config must declare publication_training")
    return config


def _dry_run(config: dict[str, Any], args: argparse.Namespace) -> int:
    dataset = config["dataset"]
    batching = config["batching"]
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "experiment_id": args.experiment_id,
                "experiment_role": config["experiment_role"],
                "dataset_id": dataset["dataset_id"],
                "train_count": dataset["roles"]["train"],
                "valid_iid_count": dataset["roles"]["valid_iid"],
                "test_iid_access": dataset["label_access"]["test_iid"],
                "sealed_access": dataset["label_access"]["sealed"],
                "batching": batching,
                "training_runs": 0,
                "publication_evidence": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _validation_loss(trainer: V7FormalTrainer, state: Any, batches: list[Any]) -> float:
    values = []
    for batch in batches:
        value = trainer.validate(state, batch)
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if not args.rehearsal:
        raise ValueError("this entrypoint only permits explicit non-publication --rehearsal runs")
    if args.epochs < 1 or args.epochs > 3:
        raise ValueError("preflight rehearsal is limited to 1-3 epochs")
    dataset_config = config["dataset"]
    subset = (args.subset or ROOT / dataset_config["subset_path"]).resolve()
    manifest = (args.manifest or ROOT / dataset_config["manifest_path"]).resolve()
    if dataset_config["label_access"]["test_iid"] != "forbidden" or dataset_config["label_access"]["sealed"] != "forbidden":
        raise ValueError("test/sealed access must remain closed")

    preparation_profile: dict[str, Any] = {}
    prepared = prepare_p1i_data(
        subset,
        manifest,
        graph_config=config["graph"],
        model_config=config["model"],
        loss_config=config["loss"],
        batch_size=int(config["batching"]["batch_size"]),
        validation_batch_size=int(config["batching"]["validation_batch_size"]),
        batch_build_seed=int(config["batching"]["batch_build_seed"]),
        graph_seed=0,
        profile=preparation_profile,
    )
    feature_names = tuple(prepared.stats["feature_names"])
    model_config = _resolve_model_config(dict(config["model"]), feature_names)
    loss_config = dict(prepared.train_only_loss_references)
    model = RIGNO(**model_config)
    init_start = time.perf_counter()
    params = model_init_full(model, jax.random.PRNGKey(int(args.seed)), prepared.train_batches[0])["params"]
    init_seconds = time.perf_counter() - init_start
    optimizer = make_p1i_optimizer(
        config["optimizer"],
        epochs=args.epochs,
        updates_per_epoch=len(prepared.train_batches),
    )

    def checkpoint_writer(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    dependencies = TrainingDependencies(
        data_source={"dataset_id": dataset_config["dataset_id"], "roles": ["train", "valid_iid"]},
        feature_transform="v6_dual_robin_relative_bc_features",
        normalization=prepared.stats,
        graph_builder=prepared.builder,
        model=model,
        model_apply=lambda current_params, batch, rng: model_apply_full(model, current_params, batch, rng),
        loss_fn=lambda prediction, batch: loss_fn_full(prediction, batch, loss_config),
        optimizer=optimizer,
        batch_iterator=lambda batches: batches,
        validation_fn=lambda current_params, batch: loss_fn_full(
            model_apply_full(model, current_params, batch, None), batch, loss_config
        ),
        checkpoint_writer=checkpoint_writer,
        metrics_fn=lambda current_params, batch: {
            "loss": float(
                loss_fn_full(
                    model_apply_full(model, current_params, batch, None), batch, loss_config
                )
            )
        },
        gradient_transform=make_gradient_transform(model_config, config["optimizer"]),
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=bool(args.jit_cache))
    state = trainer.initialize(params)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    step_times = []
    validation_times = []
    for epoch in range(1, args.epochs + 1):
        order = np.arange(len(prepared.train_batches))
        if bool(config["batching"]["shuffle_train_batches"]):
            order = np.random.default_rng(int(config["batching"]["batch_build_seed"]) + epoch).permutation(order)
        epoch_start = time.perf_counter()
        for batch_index, raw_index in enumerate(order, start=1):
            batch = prepared.train_batches[int(raw_index)]
            step_start = time.perf_counter()
            step_key = jax.random.fold_in(jax.random.PRNGKey(int(args.seed)), epoch)
            step_key = jax.random.fold_in(step_key, batch_index)
            result = trainer.step(state, batch, rng=step_key)
            state = result.state
            step_times.append({
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_id": batch.batch_id,
                "seconds": time.perf_counter() - step_start,
                "loss": float(result.loss),
                "compile_count": trainer.compile_count,
            })
        validation_start = time.perf_counter()
        valid_loss = _validation_loss(trainer, state, list(prepared.valid_batches))
        validation_times.append(time.perf_counter() - validation_start)
        epoch_record = {
            "epoch": epoch,
            "batch_count": len(order),
            "valid_loss": valid_loss,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
            "train_order_hash": hashlib.sha256(
                json.dumps([prepared.train_batches[int(index)].batch_id for index in order], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        history.append(epoch_record)
        checkpoint_report = atomic_training_checkpoint(
            output_dir / f"epoch_{epoch:04d}.pkl",
            state=state,
            metadata={
                "experiment_id": args.experiment_id,
                "execution_role": "readiness_fixture",
                "epoch": epoch,
                "checkpoint_selection_applied": False,
                "test_and_sealed_access": "closed",
            },
        )
        if epoch == 1:
            reloaded_payload = pickle.loads((output_dir / f"epoch_{epoch:04d}.pkl").read_bytes())
            from rigno.heat3d_training.core import TrainingState
            reloaded_state = TrainingState(
                params=reloaded_payload["params"],
                optimizer_state=reloaded_payload["optimizer_state"],
                step=int(reloaded_payload["step"]),
            )
            resumed_valid_loss = _validation_loss(trainer, reloaded_state, list(prepared.valid_batches))
            if resumed_valid_loss != valid_loss:
                raise RuntimeError("checkpoint resume validation loss changed")
            state = reloaded_state
            epoch_record["resume_round_trip_valid_loss"] = resumed_valid_loss
            epoch_record["checkpoint_round_trip"] = checkpoint_report

    receipt = {
        "schema_version": "heat3d_v7_g1_p1i_rehearsal_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "experiment_role": config["experiment_role"],
        "execution_role": "readiness_fixture",
        "status": "COMPLETE",
        "rehearsal": True,
        "publication_evidence": False,
        "headline_performance_claim": False,
        "git_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "config_sha256": _sha256(args.config.resolve()),
        "dataset_id": dataset_config["dataset_id"],
        "dataset_manifest_sha256": dataset_config["manifest_sha256"],
        "full_field_archive_sha256": dataset_config["full_field_archive_sha256"],
        "split_counts": {"train": len(prepared.train_examples), "valid_iid": len(prepared.valid_examples)},
        "test_and_sealed_access": "closed",
        "batching_contract": config["batching"],
        "model_contract": model_config,
        "preparation": preparation_profile,
        "model_init_seconds": init_seconds,
        "jit_cache": bool(args.jit_cache),
        "compile_count": trainer.compile_count,
        "unique_batch_signatures": len({batch.batch_id for batch in prepared.train_batches}),
        "step_timing": {
            "count": len(step_times),
            "first_seconds": step_times[0]["seconds"] if step_times else None,
            "median_after_first_seconds": float(np.median([row["seconds"] for row in step_times[1:]])) if len(step_times) > 1 else None,
            "total_seconds": float(sum(row["seconds"] for row in step_times)),
            "forward_backward_optimizer_boundary": "V7FormalTrainer.step; optimizer included in actual step callback",
        },
        "validation_seconds": validation_times,
        "epochs": history,
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
        "device": [str(device) for device in jax.devices()],
        "checkpoint_resume": history[0].get("checkpoint_round_trip") if history else None,
        "frozen_v6_evidence_modified": False,
        "model_selection": False,
        "training_variant_comparison": False,
        "scientific_evidence_eligible": False,
        "prohibited_actions": {
            "test_iid": False,
            "sealed": False,
            "solver": False,
            "new_data": False,
            "g1_scientific_experiment": False,
            "architecture_or_loss_change": False,
            "high_n_optimization": False,
        },
    }
    receipt_path = output_dir / "v7_g1_p1i_rehearsal_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    args = _parse_args()
    config = _registered_full_config(args)
    if args.dry_run:
        return _dry_run(config, args)
    receipt = _run(args, config)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
