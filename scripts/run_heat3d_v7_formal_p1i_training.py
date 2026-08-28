"""Single registered V7 P1i training entrypoint.

The entrypoint has two explicit, non-overlapping modes:

* ``V7-G1-Full-P1i`` is a rehearsal-only publication-training registration
  (one to three epochs until a later explicit G1 authorization).
* ``V7-G1-BudgetQual-e200-*`` is a seed-0, non-publication budget
  qualification.  It uses the complete e200 schedule and is never counted as
  formal G1 evidence.
* Registered formal variants can be resolved through ``--dry-run`` for
  registry validation, but all non-dry formal execution remains closed.

All numerical dependencies are assembled from ``rigno.heat3d_training`` and
the stable RIGNO library.  No historical script, smoke helper, development
runner, private cross-script symbol, or module-state patch is reachable here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
from typing import Any

import jax
import numpy as np

from rigno.heat3d_training import (
    TrainingDependencies,
    V7FormalTrainer,
    atomic_training_checkpoint,
    evaluate_level_a_validation,
    learning_rate_for_epoch,
    loss_fn_full,
    loss_fn_vanilla,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_apply_vanilla,
    model_init_full,
    model_init_vanilla,
    prepare_p1i_data,
    tree_l2_norm,
    tree_parameter_count,
)
from rigno.models.rigno import RIGNO


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "heat3d_v7" / "v7_experiment_registry.json"
FULL_CONFIG_PATH = ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json"
BUDGET_CONFIG_PATH = ROOT / "configs" / "heat3d_v7" / "v7_g1_budget_qualification.json"
SUPPORTED_BUDGET_VARIANTS = {"Full", "vanilla_RIGNO"}
FORMAL_VARIANT_BY_ID = {
    "V7-G1-Full-P1i": "Full",
    "V7-G1-Full-P1i:vanilla-RIGNO": "vanilla_RIGNO",
    "V7-G1-Full-P1i:generic-uniform-support": "generic_uniform_support",
    "V7-G1-Full-P1i:volume-only-support": "volume_only_support",
    "V7-G1-Full-P1i:no-context": "no_context",
    "V7-G1-Full-P1i:no-scale": "no_scale",
    "V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched": "vanilla_RIGNO_capacity_matched",
}


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
    parser = argparse.ArgumentParser(description="V7 registered P1i training entrypoint")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jit-cache", action="store_true", default=True)
    parser.add_argument("--no-jit-cache", action="store_false", dest="jit_cache")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_model_config(
    model_config: dict[str, Any], feature_names: tuple[str, ...]
) -> dict[str, Any]:
    resolved = dict(model_config)
    resolved.pop("architecture", None)
    resolved["global_context_feature_names"] = tuple(
        resolved.get("global_context_feature_names") or ()
    )
    local_names = tuple(resolved.get("decoder_bypass_local_feature_names") or ())
    resolved["decoder_bypass_local_feature_names"] = local_names
    resolved["decoder_bypass_feature_names"] = local_names
    resolved["decoder_bypass_feature_indices"] = tuple(
        feature_names.index(name) for name in local_names
    )
    resolved["decoder_bypass_num_features"] = len(
        resolved["decoder_bypass_feature_indices"]
    )
    return resolved


def _variant_model_config(parent: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Resolve a registered variant as a delta over the single Full parent."""

    config = dict(parent)
    if variant == "Full":
        return config
    if variant == "vanilla_RIGNO":
        # This is the explicit control used by the e200 qualification pilot.
        # Its graph/input/batch contract remains shared with Full, while all
        # Full-only conditioning and native shape--scale heads are disabled.
        config.update(
            {
                "decoder_bypass_mode": "none",
                "decoder_bypass_features": "none",
                "decoder_bypass_local_feature_names": [],
                "global_context_mode": "none",
                "global_context_feature_dim": 0,
                "global_context_feature_names": [],
                "native_output_mode": "legacy_normalized_deltaT",
                "decoder_bypass_output_space": "normalized_deltaT",
            }
        )
        return config
    if variant in {
        "generic_uniform_support",
        "volume_only_support",
        "no_context",
        "no_scale",
    }:
        raise ValueError(
            f"{variant} is registered as a future G1 delta but has no support "
            "provider in the e200 qualification path; refusing an implicit variant"
        )
    raise ValueError(f"unregistered V7 training variant {variant!r}")


def _resolve_registration(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    registry = _load_json(REGISTRY_PATH)
    entries = {
        str(entry.get("experiment_id")): entry
        for entry in registry.get("registered_runs", [])
    }
    entry = entries.get(args.experiment_id)
    if entry is None:
        raise ValueError(f"experiment is not registered: {args.experiment_id}")

    if (
        args.experiment_id in FORMAL_VARIANT_BY_ID
        and args.experiment_id != "V7-G1-Full-P1i"
    ):
        config_path = (args.config or FULL_CONFIG_PATH).resolve()
        config = _load_json(config_path)
        if entry.get("status") not in {"registered_not_executed", "planned_not_executed"}:
            raise ValueError("formal G1 variant is not in a planned state")
        if config.get("experiment_id") != "V7-G1-Full-P1i":
            raise ValueError("formal G1 variants must resolve to the frozen Full parent config")
        if not args.dry_run:
            raise ValueError(
                "formal G1 execution is closed; use --dry-run for registry validation "
                "until explicit scientific G1 authorization"
            )
        return config, entry, FORMAL_VARIANT_BY_ID[args.experiment_id], False

    if args.experiment_id == "V7-G1-Full-P1i":
        config_path = (args.config or FULL_CONFIG_PATH).resolve()
        config = _load_json(config_path)
        if entry.get("status") != "registered_not_executed":
            raise ValueError("Full P1i is not in the registered-not-executed state")
        if config.get("experiment_id") != args.experiment_id:
            raise ValueError("Full P1i config and registry ID do not match")
        if config.get("experiment_role") != "publication_training":
            raise ValueError("Full P1i config must declare publication_training")
        return config, entry, "Full", False

    if not args.experiment_id.startswith("V7-G1-BudgetQual-e200-"):
        raise ValueError(
            "formal G1 execution is closed; only the Full rehearsal or registered "
            "e200 budget qualification can be invoked"
        )
    config_path = (args.config or BUDGET_CONFIG_PATH).resolve()
    budget = _load_json(config_path)
    candidates = {
        str(row.get("experiment_id")): row
        for row in budget.get("qualification_runs", [])
    }
    candidate = candidates.get(args.experiment_id)
    if candidate is None:
        raise ValueError("budget qualification ID is missing from its config")
    if entry.get("status") != "registered_not_executed":
        raise ValueError("budget qualification is not in the registered-not-executed state")
    variant = str(candidate.get("variant"))
    if variant not in SUPPORTED_BUDGET_VARIANTS:
        raise ValueError(f"unsupported budget qualification variant {variant!r}")
    parent_path = (ROOT / str(budget["parent_config"])).resolve()
    parent = _load_json(parent_path)
    if parent.get("experiment_id") != "V7-G1-Full-P1i":
        raise ValueError("budget qualification parent is not the frozen Full P1i config")
    return parent, entry, variant, True


def _dry_run(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    variant: str,
    budget_only: bool,
) -> int:
    dataset = config["dataset"]
    batching = config["batching"]
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "experiment_id": args.experiment_id,
                "variant": variant,
                "experiment_role": (
                    "budget_qualification_only" if budget_only else config["experiment_role"]
                ),
                "dataset_id": dataset["dataset_id"],
                "train_count": dataset["roles"]["train"],
                "valid_iid_count": dataset["roles"]["valid_iid"],
                "test_iid_access": dataset["label_access"]["test_iid"],
                "sealed_access": dataset["label_access"]["sealed"],
                "batching": batching,
                "epochs": 200 if budget_only else (args.epochs or 1),
                "training_runs": 0,
                "budget_qualification_only": budget_only,
                "publication_evidence": False,
                "g1_formal": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _checkpoint_writer(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _build_dependencies(
    *,
    model: Any,
    prepared: Any,
    model_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    optimizer: Any,
    parent_config: Mapping[str, Any],
    variant: str,
    dataset_config: Mapping[str, Any],
) -> TrainingDependencies:
    if variant == "Full":
        apply_fn = lambda current_params, batch, rng: model_apply_full(
            model, current_params, batch, rng
        )
        loss_fn = lambda prediction, batch: loss_fn_full(
            prediction, batch, loss_config
        )

        def validation_outputs_fn(current_params: Any, batch: Any) -> tuple[Any, Any]:
            prediction = model_apply_full(model, current_params, batch, None)
            return prediction, loss_fn_full(prediction, batch, loss_config)

        gradient_transform = make_gradient_transform(
            model_config, parent_config["optimizer"]
        )
        feature_transform = "v6_dual_robin_relative_bc_features+native_shape_scale"
    else:
        apply_fn = lambda current_params, batch, rng: model_apply_vanilla(
            model, current_params, batch, rng
        )
        loss_fn = lambda prediction, batch: loss_fn_vanilla(
            prediction, batch, loss_config
        )

        def validation_outputs_fn(current_params: Any, batch: Any) -> tuple[Any, Any]:
            prediction = model_apply_vanilla(model, current_params, batch, None)
            return prediction, loss_fn_vanilla(prediction, batch, loss_config)

        gradient_transform = None
        feature_transform = "v6_dual_robin_relative_bc_features"

    return TrainingDependencies(
        data_source={
            "dataset_id": dataset_config["dataset_id"],
            "roles": ["train", "valid_iid"],
        },
        feature_transform=feature_transform,
        normalization=prepared.stats,
        graph_builder=prepared.builder,
        model=model,
        model_apply=apply_fn,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_iterator=lambda batches: batches,
        validation_fn=lambda current_params, batch: validation_outputs_fn(
            current_params, batch
        )[1],
        checkpoint_writer=_checkpoint_writer,
        metrics_fn=lambda current_params, batch: {
            "loss": float(validation_outputs_fn(current_params, batch)[1])
        },
        gradient_transform=gradient_transform,
        validation_outputs_fn=validation_outputs_fn,
    )


def _validation_pass(
    trainer: V7FormalTrainer,
    state: Any,
    batches: list[Any],
    *,
    prepared: Any,
    variant: str,
) -> tuple[float, dict[str, Any]]:
    losses = []
    predictions = []
    for batch in batches:
        prediction, loss = trainer.validate_with_outputs(state, batch)
        predictions.append(prediction)
        losses.append(float(loss))
    evaluation = evaluate_level_a_validation(
        predictions=predictions,
        batches=batches,
        examples=prepared.valid_examples,
        stats=prepared.stats,
        variant=variant,
    )
    return float(np.mean(losses)) if losses else 0.0, evaluation


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    variant: str,
    budget_only: bool,
) -> dict[str, Any]:
    if not args.rehearsal:
        raise ValueError("all non-dry runs require explicit --rehearsal")
    epochs = int(args.epochs if args.epochs is not None else (200 if budget_only else 1))
    if budget_only and epochs != 200:
        raise ValueError("e200 budget qualification requires exactly --epochs 200")
    if not budget_only and not 1 <= epochs <= 3:
        raise ValueError("Full P1i preflight rehearsal is limited to 1-3 epochs")
    seed = int(args.seed if args.seed is not None else 0)
    if budget_only and seed != 0:
        raise ValueError("budget qualification is registered for seed0 only")

    dataset_config = config["dataset"]
    subset = (args.subset or ROOT / dataset_config["subset_path"]).resolve()
    manifest = (args.manifest or ROOT / dataset_config["manifest_path"]).resolve()
    if (
        dataset_config["label_access"]["test_iid"] != "forbidden"
        or dataset_config["label_access"]["sealed"] != "forbidden"
    ):
        raise ValueError("test/sealed access must remain closed")
    if dataset_config["roles"]["train"] != 768 or dataset_config["roles"]["valid_iid"] != 128:
        raise ValueError("P1i publication population drifted")
    if int(config["batching"]["train_batches_per_epoch"]) != 32:
        raise ValueError("B24 contract must yield exactly 32 training batches")

    # The Full parent owns the frozen feature/context preparation.  Vanilla
    # deliberately consumes the same prepared graph/input batches, then calls
    # only the ordinary RIGNO method.
    parent_model_config = dict(config["model"])
    preparation_profile: dict[str, Any] = {}
    prepared = prepare_p1i_data(
        subset,
        manifest,
        graph_config=config["graph"],
        model_config=parent_model_config,
        loss_config=config["loss"],
        batch_size=int(config["batching"]["batch_size"]),
        validation_batch_size=int(config["batching"]["validation_batch_size"]),
        batch_build_seed=int(config["batching"]["batch_build_seed"]),
        graph_seed=0,
        profile=preparation_profile,
    )
    feature_names = tuple(prepared.stats["feature_names"])
    model_config = _resolve_model_config(
        _variant_model_config(parent_model_config, variant), feature_names
    )
    loss_config = dict(prepared.train_only_loss_references)
    model = RIGNO(**model_config)
    init_start = time.perf_counter()
    if variant == "Full":
        params = model_init_full(
            model, jax.random.PRNGKey(seed), prepared.train_batches[0]
        )["params"]
    else:
        params = model_init_vanilla(
            model, jax.random.PRNGKey(seed), prepared.train_batches[0]
        )["params"]
    init_seconds = time.perf_counter() - init_start
    optimizer = make_p1i_optimizer(
        config["optimizer"],
        epochs=epochs,
        updates_per_epoch=len(prepared.train_batches),
    )
    dependencies = _build_dependencies(
        model=model,
        prepared=prepared,
        model_config=model_config,
        loss_config=loss_config,
        optimizer=optimizer,
        parent_config=config,
        variant=variant,
        dataset_config=dataset_config,
    )
    trainer = V7FormalTrainer(dependencies, jit_cache=bool(args.jit_cache))
    state = trainer.initialize(params)
    output_dir = (
        args.output_dir
        or Path(
            "/tmp/"
            + (
                "v7_g1_budget_qual_e200_"
                + ("full_seed0" if variant == "Full" else "vanilla_seed0")
                if budget_only
                else "v7_g1_p1i_rehearsal"
            )
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    step_times: list[dict[str, Any]] = []
    validation_times: list[float] = []
    best_metric = float("inf")
    best_epoch: int | None = None
    best_checkpoint_report: dict[str, Any] | None = None
    best_checkpoint_path = output_dir / "params_best_sample_first.pkl"
    for epoch in range(1, epochs + 1):
        order = np.arange(len(prepared.train_batches))
        if bool(config["batching"]["shuffle_train_batches"]):
            order = np.random.default_rng(
                int(config["batching"]["batch_build_seed"]) + epoch
            ).permutation(order)
        epoch_start = time.perf_counter()
        epoch_losses = []
        epoch_grad_norms = []
        epoch_update_norms = []
        epoch_param_norms = []
        for batch_index, raw_index in enumerate(order, start=1):
            batch = prepared.train_batches[int(raw_index)]
            step_start = time.perf_counter()
            step_key = jax.random.fold_in(jax.random.PRNGKey(seed), epoch)
            step_key = jax.random.fold_in(step_key, batch_index)
            result = trainer.step(state, batch, rng=step_key)
            state = result.state
            step_seconds = time.perf_counter() - step_start
            loss_value = float(result.loss)
            epoch_losses.append(loss_value)
            epoch_grad_norms.append(tree_l2_norm(result.gradients))
            epoch_update_norms.append(tree_l2_norm(result.updates))
            epoch_param_norms.append(tree_l2_norm(state.params))
            step_times.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "batch_id": batch.batch_id,
                    "seconds": step_seconds,
                    "loss": loss_value,
                    "gradient_norm": epoch_grad_norms[-1],
                    "update_norm": epoch_update_norms[-1],
                    "parameter_norm": epoch_param_norms[-1],
                    "compile_count": trainer.compile_count,
                }
            )
        validation_start = time.perf_counter()
        valid_loss, evaluation = _validation_pass(
            trainer,
            state,
            list(prepared.valid_batches),
            prepared=prepared,
            variant=variant,
        )
        validation_seconds = time.perf_counter() - validation_start
        validation_times.append(validation_seconds)
        metric = float(evaluation["metrics"]["sample_first_relative_rmse_pct"])
        if not np.isfinite(metric):
            raise RuntimeError("sample-first validation metric is non-finite")
        selection_improved = metric < best_metric
        checkpoint_report = None
        if selection_improved:
            best_metric = metric
            best_epoch = epoch
            checkpoint_report = atomic_training_checkpoint(
                best_checkpoint_path,
                state=state,
                metadata={
                    "experiment_id": args.experiment_id,
                    "variant": variant,
                    "execution_role": (
                        "budget_qualification" if budget_only else "readiness_fixture"
                    ),
                    "budget_qualification_only": budget_only,
                    "epoch": epoch,
                    "selection_metric": "sample_first_relative_rmse_pct",
                    "selection_metric_value": metric,
                    "test_and_sealed_access": "closed",
                },
            )
            best_checkpoint_report = checkpoint_report
        epoch_record = {
            "epoch": epoch,
            "batch_count": len(order),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "valid_loss": valid_loss,
            "selection_metric": metric,
            "selection_metric_name": "sample_first_relative_rmse_pct",
            "selection_improved": selection_improved,
            "best_epoch_so_far": best_epoch,
            "learning_rate": learning_rate_for_epoch(
                epoch,
                epochs=epochs,
                updates_per_epoch=len(prepared.train_batches),
                config=config["optimizer"],
            ),
            "gradient_norm_mean": float(np.mean(epoch_grad_norms)),
            "gradient_norm_max": float(np.max(epoch_grad_norms)),
            "update_norm_mean": float(np.mean(epoch_update_norms)),
            "parameter_norm_last": epoch_param_norms[-1],
            "validation_seconds": validation_seconds,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
            "train_order_hash": hashlib.sha256(
                json.dumps(
                    [prepared.train_batches[int(index)].batch_id for index in order],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "level_a_metrics": evaluation["metrics"],
            "sufficient_statistics": evaluation["sufficient_statistics"],
        }
        history.append(epoch_record)

    final_checkpoint_report = atomic_training_checkpoint(
        output_dir / "params_final.pkl",
        state=state,
        metadata={
            "experiment_id": args.experiment_id,
            "variant": variant,
            "execution_role": "budget_qualification" if budget_only else "readiness_fixture",
            "budget_qualification_only": budget_only,
            "epoch": epochs,
            "selection_metric": "sample_first_relative_rmse_pct",
            "best_epoch": best_epoch,
            "test_and_sealed_access": "closed",
        },
    )

    final_payload = pickle.loads((output_dir / "params_final.pkl").read_bytes())
    from rigno.heat3d_training.core import TrainingState

    reloaded_state = TrainingState(
        params=final_payload["params"],
        optimizer_state=final_payload["optimizer_state"],
        step=int(final_payload["step"]),
    )
    resumed_valid_loss, resumed_evaluation = _validation_pass(
        trainer,
        reloaded_state,
        list(prepared.valid_batches),
        prepared=prepared,
        variant=variant,
    )
    final_valid_loss = history[-1]["valid_loss"] if history else None
    if final_valid_loss is not None and not np.isfinite(resumed_valid_loss):
        raise RuntimeError("final checkpoint resume validation loss became non-finite")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    receipt = {
        "schema_version": "heat3d_v7_g1_budget_qualification_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "parent_experiment_id": "V7-G1-Full-P1i",
        "variant": variant,
        "experiment_role": (
            "budget_qualification_only" if budget_only else config["experiment_role"]
        ),
        "registered_experiment_role": config["experiment_role"],
        "execution_role": "budget_qualification" if budget_only else "readiness_fixture",
        "status": "COMPLETE",
        "budget_qualification_only": budget_only,
        "g1_formal": False,
        "rehearsal": not budget_only,
        "publication_evidence": False,
        "scientific_evidence_eligible": False,
        "headline_performance_claim": False,
        "git_commit": commit,
        "config_sha256": _sha256(FULL_CONFIG_PATH),
        "budget_config_sha256": _sha256(BUDGET_CONFIG_PATH) if budget_only else None,
        "dataset_id": dataset_config["dataset_id"],
        "dataset_manifest_sha256": dataset_config["manifest_sha256"],
        "full_field_archive_sha256": dataset_config["full_field_archive_sha256"],
        "split_counts": {
            "train": len(prepared.train_examples),
            "valid_iid": len(prepared.valid_examples),
        },
        "test_and_sealed_access": "closed",
        "batching_contract": config["batching"],
        "optimization_contract": {
            "epochs": epochs,
            "warmup_epochs": int(config["optimizer"]["warmup_epochs"]),
            "lr_schedule": config["optimizer"]["lr_schedule"],
            "base_lr": float(config["optimizer"]["lr"]),
            "min_lr": float(config["optimizer"]["min_lr"]),
            "cosine_horizon_epochs": epochs,
            "optimizer": config["optimizer"]["optimizer"],
            "weight_decay": float(config["optimizer"]["weight_decay"]),
            "gradient_clip_norm": float(config["optimizer"]["gradient_clip_norm"]),
            "initialization": "random",
        },
        "checkpoint_selection": {
            "metric": "sample_first_relative_rmse_pct",
            "tie_break": "earliest_epoch",
            "split": "valid_iid",
            "best_epoch": best_epoch,
            "best_value": best_metric,
            "best_checkpoint": best_checkpoint_report,
            "final_checkpoint": final_checkpoint_report,
        },
        "model_contract": model_config,
        "parameter_count": tree_parameter_count(params),
        "preparation": preparation_profile,
        "model_init_seconds": init_seconds,
        "jit_cache": bool(args.jit_cache),
        "compile_count": trainer.compile_count,
        "unique_batch_signatures": len({batch.batch_id for batch in prepared.train_batches}),
        "step_timing": {
            "count": len(step_times),
            "first_seconds": step_times[0]["seconds"] if step_times else None,
            "median_after_first_seconds": (
                float(np.median([row["seconds"] for row in step_times[1:]]))
                if len(step_times) > 1
                else None
            ),
            "total_seconds": float(sum(row["seconds"] for row in step_times)),
            "forward_backward_optimizer_boundary": (
                "V7FormalTrainer.step; optimizer included in actual step callback"
            ),
        },
        "validation_seconds": validation_times,
        "epochs": history,
        "final_resume": {
            "validation_loss": resumed_valid_loss,
            "validation_loss_abs_diff": (
                None
                if final_valid_loss is None
                else abs(resumed_valid_loss - final_valid_loss)
            ),
            "selection_metric": resumed_evaluation["metrics"][
                "sample_first_relative_rmse_pct"
            ],
            "state_tree_round_trip": bool(final_checkpoint_report["passed"]),
        },
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "device": [str(device) for device in jax.devices()],
        "frozen_v6_evidence_modified": False,
        "model_selection": False,
        "training_variant_comparison": False,
        "prohibited_actions": {
            "test_iid": False,
            "sealed": False,
            "solver": False,
            "new_data": False,
            "formal_g1_multi_seed": False,
            "architecture_or_loss_change": False,
            "high_n_optimization": False,
        },
    }
    receipt_path = output_dir / (
        "v7_g1_budget_qualification_receipt.json"
        if budget_only
        else "v7_g1_p1i_rehearsal_receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    args = _parse_args()
    config, _entry, variant, budget_only = _resolve_registration(args)
    if args.dry_run:
        return _dry_run(config, args, variant=variant, budget_only=budget_only)
    _run(args, config, variant=variant, budget_only=budget_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
