"""Single registered V7 P1i training entrypoint.

The entrypoint has two explicit, non-overlapping modes:

* ``V7-G1-Full-P1i`` is a rehearsal-only publication-training registration
  (one to three epochs until a later explicit G1 authorization).
* ``V7-G1-BudgetQual-e200-*`` is a seed-0, non-publication budget
  qualification.  It uses the complete e200 schedule and is never counted as
  formal G1 evidence.
* Registered formal variants can be resolved through ``--dry-run`` for
  registry validation.  Non-dry formal execution requires ``--formal`` and
  an exact manifest-bound launch contract; rehearsal and formal execution
  remain separate modes.

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
    block_until_ready,
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
SUPPORTED_VARIANT_QUALIFICATION_VARIANTS = {
    "layout_agnostic_stratified_support",
    "cv_only_support",
    "no_film",
    "physics_scale_only",
    "vanilla_RIGNO_capacity_matched",
}
# A one-to-three-epoch canonical Vanilla rehearsal is permitted only as a
# nonpublication profiling/compatibility fixture.  It never opens formal G1.
SUPPORTED_NONPUBLICATION_REHEARSAL_VARIANTS = {
    "vanilla_RIGNO",
    *SUPPORTED_VARIANT_QUALIFICATION_VARIANTS,
}
NATIVE_VARIANTS = {
    "Full",
    "physics_scale_only",
    "no_film",
    "layout_agnostic_stratified_support",
    "cv_only_support",
}
VANILLA_VARIANTS = {"vanilla_RIGNO", "vanilla_RIGNO_capacity_matched"}
FORMAL_VARIANT_BY_ID = {
    "V7-G1-Full-P1i": "Full",
    "V7-G1-Full-P1i:vanilla-RIGNO": "vanilla_RIGNO",
    "V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched": "vanilla_RIGNO_capacity_matched",
    "V7-G1-Full-P1i:layout-agnostic-stratified-support": "layout_agnostic_stratified_support",
    "V7-G1-Full-P1i:cv-only-support": "cv_only_support",
    "V7-G1-Full-P1i:no-film": "no_film",
    "V7-G1-Full-P1i:physics-scale-only": "physics_scale_only",
}
FORMAL_LAUNCH_MANIFEST_DEFAULT = (
    ROOT / "configs" / "heat3d_v7" / "v7_g1_formal_launch_manifest.json"
)
SUPPORT_PROVIDER_BY_VARIANT = {
    "layout_agnostic_stratified_support": "generic_stratified_v2",
    "cv_only_support": "cv_only_v1",
}
ALTERNATIVE_SUPPORT_PROVIDERS = frozenset(SUPPORT_PROVIDER_BY_VARIANT.values())


def _execution_role(variant: str, *, budget_only: bool, registered_role: str) -> str:
    if budget_only:
        return "budget_qualification"
    if variant in SUPPORTED_VARIANT_QUALIFICATION_VARIANTS:
        return "variant_qualification"
    return registered_role


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
    parser.add_argument(
        "--formal",
        action="store_true",
        help="run one manifest-bound formal G1 seed (never a multi-seed process)",
    )
    parser.add_argument("--launch-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_model_config(
    model_config: dict[str, Any], feature_names: tuple[str, ...]
) -> dict[str, Any]:
    resolved = dict(model_config)
    # Data-preparation markers never leak into the Flax RIGNO constructor.
    resolved.pop("global_context_ablation", None)
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
    if variant == "vanilla_RIGNO_capacity_matched":
        # Width 100 is the pre-registered capacity-matched candidate.  This
        # remains the ordinary RIGNO control; only latent width is changed.
        config.update(
            {
                "node_latent_size": 100,
                "edge_latent_size": 100,
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
    if variant == "physics_scale_only":
        # Keep the native shape/physical-scale route and remove the learned
        # residual correction; this is not a direct-output architecture.
        config.update(
            {
                "learned_scale_correction_mode": "physics_only",
                "scale_head_mode": "physics_only",
                "scale_attention_mode": "none",
                "scale_deepsets_mode": "none",
                "scale_context_mode": "none",
                "scale_context_feature_dim": 0,
                "scale_context_feature_names": [],
            }
        )
        return config
    if variant == "no_film":
        # No-FiLM is intentionally a one-field delta.  The frozen 24-D
        # physical context remains prepared, standardized, and available to
        # the native scale/context path; only the global FiLM modulation is
        # disabled.
        config["global_context_mode"] = "none"
        return config
    if variant in {
        "layout_agnostic_stratified_support",
        "cv_only_support",
    }:
        return config
    raise ValueError(f"unregistered V7 training variant {variant!r}")


def _validate_variant_semantics(
    parent: Mapping[str, Any], resolved: Mapping[str, Any], variant: str
) -> None:
    """Fail closed if a registered single-factor delta drifts."""

    if variant != "no_film":
        return
    changed = [
        key for key in parent
        if key != "global_context_mode" and resolved.get(key) != parent.get(key)
    ]
    if changed or resolved.get("global_context_mode") != "none":
        raise ValueError(
            "no_film must differ from Full only by global_context_mode=none"
        )
    if resolved.get("global_context_feature_dim") != 24:
        raise ValueError("no_film must retain the frozen 24-D context schema")
    if tuple(resolved.get("global_context_feature_names") or ()) != tuple(
        parent.get("global_context_feature_names") or ()
    ):
        raise ValueError("no_film must retain the Full context feature names")


def _git_stdout(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_formal_request(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Validate one formal seed against the immutable launch manifest.

    The manifest is deliberately separate from the scientific configuration:
    the frozen scientific code SHA is its ancestor, while the manifest commit
    may only change launch metadata.  This keeps the executable matrix
    auditable without permitting a scientific code/config drift after freeze.
    """

    if not bool(getattr(args, "formal", False)):
        raise ValueError("formal request validation requires --formal")
    if bool(getattr(args, "rehearsal", False)):
        raise ValueError("--formal and --rehearsal are mutually exclusive")
    if args.experiment_id not in FORMAL_VARIANT_BY_ID:
        raise ValueError("only the seven registered G1 variants may run formally")
    if args.seed is None or args.epochs is None:
        raise ValueError("formal G1 requires explicit --seed and --epochs")
    if int(args.epochs) != 200 or int(args.seed) not in {0, 1, 2}:
        raise ValueError("formal G1 is frozen to epochs=200 and seed in {0,1,2}")
    if args.output_dir is None:
        raise ValueError("formal G1 requires the manifest-bound --output-dir")
    manifest_path = (
        Path(getattr(args, "launch_manifest", None) or FORMAL_LAUNCH_MANIFEST_DEFAULT)
        .resolve()
    )
    if not manifest_path.exists():
        raise ValueError(f"formal launch manifest is missing: {manifest_path}")
    launch = _load_json(manifest_path)
    if launch.get("schema_version") != "heat3d_v7_g1_formal_launch_manifest_v1":
        raise ValueError("formal launch manifest schema drifted")
    if launch.get("status") != "frozen_launch_manifest":
        raise ValueError("formal launch manifest is not frozen")
    if launch.get("branch") != "research/v7":
        raise ValueError("formal launch branch drifted")
    frozen_code_sha = str(launch.get("g1_formal_code_sha", ""))
    if len(frozen_code_sha) != 40 or any(
        character not in "0123456789abcdef" for character in frozen_code_sha
    ):
        raise ValueError("formal code SHA is not pinned")
    current_sha = _git_stdout("rev-parse", "HEAD")
    subprocess.run(
        ["git", "cat-file", "-e", f"{frozen_code_sha}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_code_sha, current_sha],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [
        name
        for name in subprocess.run(
            ["git", "diff", "--name-only", frozen_code_sha, current_sha],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if name
    ]
    manifest_relative = str(manifest_path.relative_to(ROOT))
    if changed not in ([], [manifest_relative]):
        raise ValueError(
            "scientific files changed after G1 freeze: " + ", ".join(changed)
        )

    matrix = launch.get("matrix", {})
    if (
        matrix.get("variants") != list(FORMAL_VARIANT_BY_ID.values())
        or matrix.get("seeds") != [0, 1, 2]
        or matrix.get("epochs") != 200
        or matrix.get("run_count") != 21
        or matrix.get("formal_execution_started") is not False
    ):
        raise ValueError("formal launch matrix drifted or was already opened")
    dataset = launch.get("dataset", {})
    config_dataset = config.get("dataset", {})
    for key in ("dataset_id", "manifest_sha256", "full_field_archive_sha256"):
        if dataset.get(key) != config_dataset.get(key):
            raise ValueError(f"formal dataset binding drifted: {key}")
    preregistration_path = (
        ROOT / "configs" / "heat3d_v7" / "v7_g1_statistical_preregistration.json"
    )
    if launch.get("preregistration_file_sha256") != _sha256(preregistration_path):
        raise ValueError("formal preregistration file SHA binding drifted")
    preregistration = _load_json(preregistration_path)
    if launch.get("preregistration_sha256") != preregistration.get(
        "preregistration_sha256"
    ):
        raise ValueError("formal preregistration canonical SHA binding drifted")
    if launch.get("support_provider_contract_sha256") != _sha256(
        ROOT / "configs" / "heat3d_v7" / "v7_g1_support_provider_contract.json"
    ):
        raise ValueError("formal support-provider contract SHA binding drifted")
    if launch.get("parent_config_sha256") != _sha256(FULL_CONFIG_PATH):
        raise ValueError("formal parent config SHA binding drifted")
    if entry.get("experiment_role") != "publication_training":
        raise ValueError("formal run is not registered as publication_training")
    rows = launch.get("runs", [])
    row = next(
        (
            candidate
            for candidate in rows
            if candidate.get("experiment_id") == args.experiment_id
            and int(candidate.get("seed", -1)) == int(args.seed)
        ),
        None,
    )
    if row is None or row.get("variant") != variant:
        raise ValueError("formal run is not present in the frozen launch matrix")
    expected_output = (
        Path(str(launch["output_root"])) / str(row["run_id"])
    ).resolve()
    if Path(args.output_dir).resolve() != expected_output:
        raise ValueError("formal output directory is not the unique manifest path")
    if len(rows) != 21 or len({str(candidate.get("run_id")) for candidate in rows}) != 21:
        raise ValueError("formal launch matrix run IDs are not unique")
    if launch.get("test_iid_access") is not False or launch.get("sealed_access") is not False:
        raise ValueError("formal launch manifest opened forbidden splits")
    return launch


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
        variant = FORMAL_VARIANT_BY_ID[args.experiment_id]
        config_path = (args.config or FULL_CONFIG_PATH).resolve()
        config = _load_json(config_path)
        if entry.get("status") not in {"registered_not_executed", "planned_not_executed"}:
            raise ValueError("formal G1 variant is not in a planned state")
        if config.get("experiment_id") != "V7-G1-Full-P1i":
            raise ValueError("formal G1 variants must resolve to the frozen Full parent config")
        if bool(getattr(args, "formal", False)):
            _validate_formal_request(args, config, entry, variant)
            return config, entry, variant, False
        if not args.dry_run:
            if (
                args.rehearsal
                and variant in SUPPORTED_NONPUBLICATION_REHEARSAL_VARIANTS
            ):
                return config, entry, variant, False
            raise ValueError(
                "formal G1 execution is closed; only explicitly supported "
                "nonpublication variant qualification may use --rehearsal"
            )
        return config, entry, variant, False

    if args.experiment_id == "V7-G1-Full-P1i":
        config_path = (args.config or FULL_CONFIG_PATH).resolve()
        config = _load_json(config_path)
        if entry.get("status") != "registered_not_executed":
            raise ValueError("Full P1i is not in the registered-not-executed state")
        if config.get("experiment_id") != args.experiment_id:
            raise ValueError("Full P1i config and registry ID do not match")
        if config.get("experiment_role") != "publication_training":
            raise ValueError("Full P1i config must declare publication_training")
        if bool(getattr(args, "formal", False)):
            _validate_formal_request(args, config, entry, "Full")
        return config, entry, "Full", False

    if not args.experiment_id.startswith("V7-G1-BudgetQual-e200-"):
        raise ValueError(
            "formal G1 execution is closed; only the Full rehearsal or registered "
            "e200 budget qualification can be invoked"
        )
    if bool(getattr(args, "formal", False)):
        raise ValueError("budget qualification registrations cannot run as formal G1")
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
                    "budget_qualification_only"
                    if budget_only
                    else _execution_role(
                        variant,
                        budget_only=False,
                        registered_role=config["experiment_role"],
                    )
                ),
                "dataset_id": dataset["dataset_id"],
                "train_count": dataset["roles"]["train"],
                "valid_iid_count": dataset["roles"]["valid_iid"],
                "test_iid_access": dataset["label_access"]["test_iid"],
                "sealed_access": dataset["label_access"]["sealed"],
                "batching": batching,
                "support_provider": SUPPORT_PROVIDER_BY_VARIANT.get(
                    variant, "historical_v6_stored_support"
                ),
                "epochs": 200 if budget_only else (args.epochs or 1),
                "training_runs": 0,
                "budget_qualification_only": budget_only,
                "publication_evidence": False,
                "g1_formal": bool(getattr(args, "formal", False)),
                "formal_manifest_validation": bool(getattr(args, "formal", False)),
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
    if variant in NATIVE_VARIANTS:
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
    elif variant in VANILLA_VARIANTS:
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
    else:
        raise ValueError(f"unsupported V7 training variant {variant!r}")

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
        full_field_data=(
            prepared.full_field_data
            if prepared.support_provider_id in ALTERNATIVE_SUPPORT_PROVIDERS
            else None
        ),
    )
    return float(np.mean(losses)) if losses else 0.0, evaluation


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    variant: str,
    budget_only: bool,
) -> dict[str, Any]:
    formal = bool(getattr(args, "formal", False))
    if not formal and not args.rehearsal:
        raise ValueError("all non-dry non-formal runs require explicit --rehearsal")
    registry = _load_json(REGISTRY_PATH)
    registry_entries = {
        str(candidate.get("experiment_id")): candidate
        for candidate in registry.get("registered_runs", [])
    }
    registered_entry = registry_entries.get(args.experiment_id)
    if registered_entry is None:
        raise ValueError("formal run registration disappeared before execution")
    formal_manifest = (
        _validate_formal_request(args, config, registered_entry, variant)
        if formal
        else None
    )
    epochs = int(args.epochs if args.epochs is not None else (200 if budget_only else 1))
    if formal and budget_only:
        raise ValueError("formal G1 cannot use a budget qualification registration")
    if formal and epochs != 200:
        raise ValueError("formal G1 requires the frozen e200 schedule")
    if budget_only and epochs != 200:
        raise ValueError("e200 budget qualification requires exactly --epochs 200")
    if not formal and not budget_only and not 1 <= epochs <= 3:
        raise ValueError("pre-G1 qualification rehearsal is limited to 1-3 epochs")
    seed = int(args.seed if args.seed is not None else 0)
    if formal and args.seed is None:
        raise ValueError("formal G1 requires an explicit registered seed")
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
    preparation_model_config = _variant_model_config(parent_model_config, variant)
    _validate_variant_semantics(parent_model_config, preparation_model_config, variant)
    preparation_profile: dict[str, Any] = {}
    prepared = prepare_p1i_data(
        subset,
        manifest,
        graph_config=config["graph"],
        model_config=preparation_model_config,
        loss_config=config["loss"],
        batch_size=int(config["batching"]["batch_size"]),
        validation_batch_size=int(config["batching"]["validation_batch_size"]),
        # V6-style synchronized seed contract: graph construction, population
        # order, and model initialization all use the registered run seed.
        batch_build_seed=seed,
        graph_seed=seed,
        support_provider_id=SUPPORT_PROVIDER_BY_VARIANT.get(variant),
        full_field_archive_path=(
            ROOT / dataset_config["full_field_archive_path"]
            if SUPPORT_PROVIDER_BY_VARIANT.get(variant)
            else None
        ),
        profile=preparation_profile,
    )
    feature_names = tuple(prepared.stats["feature_names"])
    model_config = _resolve_model_config(
        _variant_model_config(parent_model_config, variant), feature_names
    )
    loss_config = dict(prepared.train_only_loss_references)
    model = RIGNO(**model_config)
    init_start = time.perf_counter()
    if variant in NATIVE_VARIANTS:
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
                else "v7_g1_p1i_qualification_" + variant
            )
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "v7_g1_progress.json"

    def write_progress(
        *,
        status: str,
        epoch: int,
        best_epoch_value: int | None,
        best_metric_value: float | None,
        latest_metric: float | None = None,
    ) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": "heat3d_v7_g1_progress_v1",
                    "status": status,
                    "experiment_id": args.experiment_id,
                    "variant": variant,
                    "run_seed": seed,
                    "epoch": epoch,
                    "epochs": epochs,
                    "best_epoch": best_epoch_value,
                    "best_selection_metric": best_metric_value,
                    "latest_selection_metric": latest_metric,
                    "execution_role": (
                        "publication_training"
                        if formal
                        else _execution_role(
                            variant,
                            budget_only=budget_only,
                            registered_role=config["experiment_role"],
                        )
                    ),
                    "g1_formal": formal,
                    "test_iid_access": False,
                    "sealed_access": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    write_progress(
        status="RUNNING",
        epoch=0,
        best_epoch_value=None,
        best_metric_value=None,
    )

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
                seed + epoch
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
            block_until_ready(
                (
                    state.params,
                    state.optimizer_state,
                    result.loss,
                    result.gradients,
                    result.updates,
                    result.prediction,
                )
            )
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
                    "execution_role": _execution_role(
                        variant,
                        budget_only=budget_only,
                        registered_role=config["experiment_role"],
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
        write_progress(
            status="RUNNING",
            epoch=epoch,
            best_epoch_value=best_epoch,
            best_metric_value=best_metric,
            latest_metric=metric,
        )

    final_checkpoint_report = atomic_training_checkpoint(
        output_dir / "params_final.pkl",
        state=state,
        metadata={
            "experiment_id": args.experiment_id,
            "variant": variant,
            "execution_role": _execution_role(
                variant,
                budget_only=budget_only,
                registered_role=config["experiment_role"],
            ),
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
            "budget_qualification_only"
            if budget_only
            else _execution_role(
                variant,
                budget_only=False,
                registered_role=config["experiment_role"],
            )
        ),
        "registered_experiment_role": config["experiment_role"],
        "execution_role": _execution_role(
            variant,
            budget_only=budget_only,
            registered_role=config["experiment_role"],
        ),
        "status": "COMPLETE",
        "budget_qualification_only": budget_only,
        "g1_formal": formal,
        "formal_g1_run": formal,
        "rehearsal": not budget_only and not formal,
        "publication_evidence": formal,
        "scientific_evidence_eligible": formal,
        "headline_performance_claim": False,
        "git_commit": commit,
        "g1_formal_code_sha": (
            formal_manifest["g1_formal_code_sha"] if formal_manifest is not None else None
        ),
        "launch_manifest_path": (
            str(
                Path(
                    getattr(args, "launch_manifest", None)
                    or FORMAL_LAUNCH_MANIFEST_DEFAULT
                ).resolve()
            )
            if formal_manifest is not None
            else None
        ),
        "formal_run_id": (
            next(
                row["run_id"]
                for row in formal_manifest["runs"]
                if row["experiment_id"] == args.experiment_id
                and int(row["seed"]) == seed
            )
            if formal_manifest is not None
            else None
        ),
        "config_sha256": _sha256(FULL_CONFIG_PATH),
        "budget_config_sha256": _sha256(BUDGET_CONFIG_PATH) if budget_only else None,
        "dataset_id": dataset_config["dataset_id"],
        "support_provider": prepared.support_provider_id,
        "dataset_manifest_sha256": dataset_config["manifest_sha256"],
        "full_field_archive_sha256": dataset_config["full_field_archive_sha256"],
        "split_counts": {
            "train": len(prepared.train_examples),
            "valid_iid": len(prepared.valid_examples),
        },
        "test_and_sealed_access": "closed",
        "batching_contract": {
            **config["batching"],
            "batch_build_seed": seed,
            "graph_seed": seed,
            "seed_binding": "V6_style_synchronized_run_seed",
        },
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
            "formal_g1_run": formal,
            "architecture_or_loss_change": False,
            "high_n_optimization": False,
        },
    }
    receipt_path = output_dir / (
        "v7_g1_budget_qualification_receipt.json"
        if budget_only
        else "v7_g1_formal_receipt.json"
        if formal
        else "v7_g1_p1i_rehearsal_receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_progress(
        status="COMPLETE",
        epoch=epochs,
        best_epoch_value=best_epoch,
        best_metric_value=best_metric,
        latest_metric=(history[-1]["selection_metric"] if history else None),
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
