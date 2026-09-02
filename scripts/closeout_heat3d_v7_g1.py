#!/usr/bin/env python3
"""V7 G1 formal closeout helpers.

The ``export-native`` mode applies the already-complete formal checkpoints to
the frozen train/valid_iid preparation and writes native 1024-point
prediction/evaluation evidence.  It deliberately does not call a training
loop, optimizer, solver, test_iid loader, or sealed loader.

The ``analyze`` mode consumes only the exported valid_iid evaluation JSONs and
executes the frozen two-level paired bootstrap.  The statistical code keeps
the aggregate functional explicit: for point-global relative RMSE it
resamples samples and recomputes sqrt(sum(point_sse)/sum(point_truth_energy)),
rather than bootstrapping sample-first RMSE as a proxy.

The ``manifest`` mode hashes the complete ignored archive.  The archive
manifest itself is intentionally excluded from its own file list so its SHA
can be reported without a circular self-reference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np


FORMAL_CODE_SHA = "191a7a06a681556f575a1c04e2b61cb13363efe1"
NATIVE_DOMAIN_ID = "registered_support_1024"
COMMON_DOMAIN_ID = "heat3d_v6_p1i_full_field_240825"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2_026_0829
VALID_COUNT = 128
SEED_SET = (0, 1, 2)

METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_relative_rmse_pct",
    "raw_K_CV_RMSE_K",
    "source_region_RMSE_K",
    "peak_RMSE_K",
    "interface_RMSE_K",
)

HYPOTHESIS_SPECS = {
    "H1": {
        "comparison_id": "H1_Full_vs_vanilla_RIGNO",
        "ablation_variant": "vanilla_RIGNO",
        "primary_metric": "point_global_relative_rmse_pct",
    },
    "H1b": {
        "comparison_id": "H1b_Full_vs_vanilla_RIGNO_capacity_matched",
        "ablation_variant": "vanilla_RIGNO_capacity_matched",
        "primary_metric": "point_global_relative_rmse_pct",
    },
    "H2_generic": {
        "comparison_id": "H2_Full_vs_layout_agnostic_stratified_support",
        "ablation_variant": "layout_agnostic_stratified_support",
        "primary_metric": "source_region_RMSE_K",
    },
    "H2_volume_only": {
        "comparison_id": "H2_Full_vs_cv_only_support",
        "ablation_variant": "cv_only_support",
        "primary_metric": "source_region_RMSE_K",
    },
    "H3": {
        "comparison_id": "H3_Full_vs_no_film",
        "ablation_variant": "no_film",
        "primary_metric": "sample_first_relative_rmse_pct",
    },
    "H4": {
        "comparison_id": "H4_Full_vs_physics_scale_only",
        "ablation_variant": "physics_scale_only",
        "primary_metric": "raw_K_CV_RMSE_K",
    },
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export-native", "analyze", "manifest"))
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--launch-manifest", type=Path, default=None)
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    return parser.parse_args()


def _require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"--{name} is required for this mode")
    return value.resolve()


def _export_native(args: argparse.Namespace) -> int:
    repo = _require_path(args.repo, "repo")
    source_root = _require_path(args.source_root, "source-root")
    output_root = args.output_root.resolve()
    launch_path = (
        args.launch_manifest
        or repo / "configs" / "heat3d_v7" / "v7_g1_formal_launch_manifest.json"
    ).resolve()
    launch = _load_json(launch_path)
    if launch.get("g1_formal_code_sha") != FORMAL_CODE_SHA:
        raise ValueError("formal launch manifest code SHA drifted")
    if launch.get("test_iid_access") or launch.get("sealed_access"):
        raise ValueError("launch manifest opens test/sealed access")
    config = _load_json(repo / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json")
    runs = list(launch.get("runs") or [])
    if args.run_id is not None:
        runs = [row for row in runs if str(row.get("run_id")) == str(args.run_id)]
        if not runs:
            raise ValueError(f"unknown run id: {args.run_id}")

    sys.path.insert(0, str(repo))
    from rigno.heat3d_training import (  # pylint: disable=import-outside-toplevel
        block_until_ready,
        model_apply_full,
        model_apply_vanilla,
    )
    from rigno.heat3d_training.full_field import (  # pylint: disable=import-outside-toplevel
        load_alternative_p1i_examples,
    )
    from rigno.heat3d_training.evaluation import (  # pylint: disable=import-outside-toplevel
        evaluate_level_a_validation,
    )
    from rigno.heat3d_training.p1i import (  # pylint: disable=import-outside-toplevel
        COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
        Heat3DGraphBuilder,
        attach_input_contexts,
        attach_native_physics,
        attach_qk_features,
        build_p1i_batches,
        legacy_train_only_stats,
        load_selected_p1i_examples,
        prediction_to_raw_delta,
    )
    from rigno.models.rigno import RIGNO  # pylint: disable=import-outside-toplevel
    from scripts.run_heat3d_v7_formal_p1i_training import (  # pylint: disable=import-outside-toplevel
        SUPPORT_PROVIDER_BY_VARIANT,
        _resolve_model_config,
        _variant_model_config,
    )

    modules = {
        "block_until_ready": block_until_ready,
        "model_apply_full": model_apply_full,
        "model_apply_vanilla": model_apply_vanilla,
        "prediction_to_raw_delta": prediction_to_raw_delta,
        "RIGNO": RIGNO,
        "support_provider_by_variant": SUPPORT_PROVIDER_BY_VARIANT,
        "resolve_model_config": _resolve_model_config,
        "variant_model_config": _variant_model_config,
        "evaluate_level_a_validation": evaluate_level_a_validation,
        "load_alternative_p1i_examples": load_alternative_p1i_examples,
        "load_selected_p1i_examples": load_selected_p1i_examples,
        "build_p1i_batches": build_p1i_batches,
        "attach_input_contexts": attach_input_contexts,
        "attach_native_physics": attach_native_physics,
        "attach_qk_features": attach_qk_features,
        "legacy_train_only_stats": legacy_train_only_stats,
        "Heat3DGraphBuilder": Heat3DGraphBuilder,
        "coord_policy_train_minmax_unit_box": COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    }
    for run in runs:
        _export_one_native(
            repo=repo,
            source_root=source_root,
            output_root=output_root,
            launch=launch,
            config=config,
            run=run,
            modules=modules,
        )
    print(f"native export complete: {len(runs)} run(s)", flush=True)
    return 0


def _prepare_native_validation(
    *,
    repo: Path,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    preparation_model: Mapping[str, Any],
    support_provider: str | None,
    seed: int,
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare only the train statistics and valid batches needed for inference.

    ``prepare_p1i_data`` also builds train graphs because the formal trainer
    needs them.  A prediction-only closeout does not.  This helper preserves
    its loader, train-only normalization/context fit, graph builder, support
    selection, and valid batch construction while intentionally omitting
    train graph materialization and all optimizer objects.
    """

    subset = repo / dataset["subset_path"]
    manifest = repo / dataset["manifest_path"]
    if support_provider:
        full_field_data = modules["load_alternative_p1i_examples"](
            subset=subset,
            manifest_path=manifest,
            full_field_archive_path=repo / dataset["full_field_archive_path"],
            provider_id=support_provider,
            seed=seed,
        )
        train_examples = tuple(full_field_data.train_examples)
        valid_examples = tuple(full_field_data.valid_examples)
        context_rows_by_id = full_field_data.context_by_id
        resolved_provider = support_provider
    else:
        loaded = modules["load_selected_p1i_examples"](subset, manifest)
        train_examples = tuple(loaded["train"])
        valid_examples = tuple(loaded["valid_iid"])
        full_field_data = None
        context_rows_by_id = None
        resolved_provider = "historical_v6_stored_support"
    if len(train_examples) != 768 or len(valid_examples) != VALID_COUNT:
        raise ValueError("native validation preparation population drifted")
    stats = modules["legacy_train_only_stats"](
        list(train_examples),
        coord_policy=modules["coord_policy_train_minmax_unit_box"],
    )
    builder = modules["Heat3DGraphBuilder"](**dict(config["graph"]))
    valid_batches = modules["build_p1i_batches"](
        valid_examples,
        stats,
        builder,
        label="valid_iid",
        batch_size=int(config["batching"]["validation_batch_size"]),
        graph_seed=seed,
    )
    all_examples = [*train_examples, *valid_examples]
    context = modules["attach_input_contexts"](
        valid_batches,
        train_examples,
        all_examples,
        preparation_model,
        context_rows_by_id=context_rows_by_id,
    )
    by_id = {str(example.sample_id): example for example in all_examples}
    modules["attach_native_physics"](
        valid_batches,
        by_id,
        context_by_id=context["raw_context_by_id"],
    )
    modules["attach_qk_features"](
        valid_batches,
        by_id,
        feature_version=str(preparation_model["qk_region_feature_version"]),
    )
    return {
        "stats": stats,
        "valid_examples": valid_examples,
        "valid_batches": tuple(valid_batches),
        "support_provider_id": resolved_provider,
        "full_field_data": full_field_data,
        "preparation_profile": {
            "mode": "prediction_only_validation_preparation",
            "train_sample_count": len(train_examples),
            "valid_sample_count": len(valid_examples),
            "valid_batch_count": len(valid_batches),
            "support_provider_id": resolved_provider,
            "train_statistics_fit_role": "train_only",
            "train_graphs_built": False,
            "optimizer_constructed": False,
            "test_and_sealed_access": "closed",
        },
    }


def _export_one_native(
    *,
    repo: Path,
    source_root: Path,
    output_root: Path,
    launch: Mapping[str, Any],
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    modules: Mapping[str, Any],
) -> None:
    run_id = str(run["run_id"])
    variant = str(run["variant"])
    seed = int(run["seed"])
    raw_dir = source_root / run_id
    raw_receipt_path = raw_dir / "v7_g1_formal_receipt.json"
    receipt = _load_json(raw_receipt_path)
    if receipt.get("status") != "COMPLETE" or receipt.get("g1_formal") is not True:
        raise ValueError(f"{run_id}: formal receipt is not COMPLETE/g1_formal")
    if receipt.get("g1_formal_code_sha") != FORMAL_CODE_SHA:
        raise ValueError(f"{run_id}: formal code SHA drifted")
    if receipt.get("formal_run_id") != run_id:
        raise ValueError(f"{run_id}: formal receipt run id drifted")
    if receipt.get("split_counts") != {"train": 768, "valid_iid": 128}:
        raise ValueError(f"{run_id}: split counts drifted")
    if receipt.get("test_and_sealed_access") != "closed":
        raise ValueError(f"{run_id}: test/sealed access is not closed")

    parent_model = dict(config["model"])
    preparation_model = modules["variant_model_config"](parent_model, variant)
    support_provider = modules["support_provider_by_variant"].get(variant)
    dataset = config["dataset"]
    prepared = _prepare_native_validation(
        repo=repo,
        dataset=dataset,
        config=config,
        preparation_model=preparation_model,
        support_provider=support_provider,
        seed=seed,
        modules=modules,
    )
    valid_examples = prepared["valid_examples"]
    valid_batches = prepared["valid_batches"]
    valid_stats = prepared["stats"]
    valid_ids = [str(example.sample_id) for example in valid_examples]
    if len(valid_ids) != VALID_COUNT or len(set(valid_ids)) != VALID_COUNT:
        raise ValueError(f"{run_id}: valid_iid population drifted")
    if any(int(len(example.condition.coords)) != 1024 for example in valid_examples):
        raise ValueError(f"{run_id}: native support is not exactly 1024 points")

    feature_names = tuple(valid_stats["feature_names"])
    model_config = modules["resolve_model_config"](
        modules["variant_model_config"](parent_model, variant), feature_names
    )
    model = modules["RIGNO"](**model_config)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_config.json",
        {
            "schema_version": "heat3d_v7_g1_archival_run_config_v2",
            "run_id": run_id,
            "experiment_id": run["experiment_id"],
            "variant": variant,
            "seed": seed,
            "epochs": 200,
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "training_source_git_commit": receipt.get("git_commit"),
            "source_formal_receipt": str(raw_receipt_path),
            "source_formal_receipt_sha256": _sha256(raw_receipt_path),
            "training_performed_by_export": False,
            "optimizer_called": False,
            "solver_called": False,
            "test_iid_access": False,
            "sealed_access": False,
            "dataset": dataset,
            "graph": config["graph"],
            "model_parent": parent_model,
            "model_resolved_for_variant": model_config,
            "loss": config["loss"],
            "batching": {
                **config["batching"],
                "batch_build_seed": seed,
                "graph_seed": seed,
            },
            "optimizer_contract": config["optimizer"],
            "checkpoint_selection": receipt["checkpoint_selection"],
            "support_provider_id": prepared["support_provider_id"],
            "evaluation_domain": {
                "primary": NATIVE_DOMAIN_ID,
                "point_count_per_sample": 1024,
                "common_240825_exported_by_this_run": False,
                "common_240825_existing_results_retained": True,
            },
            "preparation_profile": prepared["preparation_profile"],
        },
    )

    prediction_arrays: dict[str, np.ndarray] = {}
    evaluation_results: dict[str, Any] = {}
    for checkpoint, checkpoint_name in (
        ("best", "params_best_sample_first.pkl"),
        ("final", "params_final.pkl"),
    ):
        checkpoint_path = raw_dir / checkpoint_name
        with checkpoint_path.open("rb") as stream:
            payload = pickle.load(stream)
        if "params" not in payload:
            raise ValueError(f"{run_id}: checkpoint lacks params")
        params = payload["params"]
        predictions = []
        raw_rows = []
        ids = []
        for batch in valid_batches:
            if variant in {"vanilla_RIGNO", "vanilla_RIGNO_capacity_matched"}:
                output = modules["model_apply_vanilla"](model, params, batch, None)
            else:
                output = modules["model_apply_full"](model, params, batch, None)
            modules["block_until_ready"](output)
            predictions.append(output)
            if len(output) != len(batch.groups):
                raise ValueError(f"{run_id}/{checkpoint}: output/group count drifted")
            for prediction, group in zip(output, batch.groups, strict=True):
                raw = np.asarray(
                    modules["prediction_to_raw_delta"](
                        prediction, variant=variant, stats=valid_stats
                    ),
                    dtype=np.float64,
                )
                target = np.asarray(group["target_delta_raw"])
                if raw.shape != target.shape or raw.shape[1:] != (1, 1024, 1):
                    raise ValueError(
                        f"{run_id}/{checkpoint}: prediction shape {raw.shape} does not match native target {target.shape}"
                    )
                raw_rows.append(raw[:, 0, :, 0])
                ids.extend(str(sample_id) for sample_id in batch.sample_ids)
        native = np.concatenate(raw_rows, axis=0)
        if native.shape != (VALID_COUNT, 1024) or ids != valid_ids:
            raise ValueError(f"{run_id}/{checkpoint}: native prediction population drifted")
        evaluation = modules["evaluate_level_a_validation"](
            predictions=predictions,
            batches=list(valid_batches),
            examples=valid_examples,
            stats=valid_stats,
            variant=variant,
            full_field_data=None,
        )
        if evaluation.get("sample_count") != VALID_COUNT:
            raise ValueError(f"{run_id}/{checkpoint}: evaluation sample count drifted")
        if evaluation.get("evaluation_split") != "valid_iid":
            raise ValueError(f"{run_id}/{checkpoint}: evaluation split drifted")
        prediction_arrays[checkpoint] = native.astype(np.float32)
        evaluation_results[checkpoint] = {
            "schema_version": "heat3d_v7_g1_archival_evaluation_v2",
            "run_id": run_id,
            "variant": variant,
            "seed": seed,
            "checkpoint": checkpoint,
            "domain_id": NATIVE_DOMAIN_ID,
            "point_count_per_sample": 1024,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": _sha256(checkpoint_path),
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "training_performed": False,
            "optimizer_called": False,
            "solver_called": False,
            "test_iid_access": False,
            "sealed_access": False,
            **evaluation,
        }
        _write_json(output_dir / f"evaluation_{checkpoint}.json", evaluation_results[checkpoint])

    np.savez_compressed(
        output_dir / "predictions_best.npz",
        sample_ids=np.asarray(valid_ids, dtype="U128"),
        prediction_deltaT_K=prediction_arrays["best"],
        split=np.asarray("valid_iid"),
        run_id=np.asarray(run_id),
        variant=np.asarray(variant),
        seed=np.asarray(seed, dtype=np.int32),
        checkpoint=np.asarray("best"),
        domain_id=np.asarray(NATIVE_DOMAIN_ID),
    )
    np.savez_compressed(
        output_dir / "predictions_final.npz",
        sample_ids=np.asarray(valid_ids, dtype="U128"),
        prediction_deltaT_K=prediction_arrays["final"],
        split=np.asarray("valid_iid"),
        run_id=np.asarray(run_id),
        variant=np.asarray(variant),
        seed=np.asarray(seed, dtype=np.int32),
        checkpoint=np.asarray("final"),
        domain_id=np.asarray(NATIVE_DOMAIN_ID),
    )
    _write_json(
        output_dir / "export_receipt.json",
        {
            "schema_version": "heat3d_v7_g1_prediction_export_receipt_v2",
            "status": "COMPLETE",
            "run_id": run_id,
            "experiment_id": run["experiment_id"],
            "variant": variant,
            "seed": seed,
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "source_git_commit": receipt.get("git_commit"),
            "source_formal_receipt_sha256": _sha256(raw_receipt_path),
            "prediction_checkpoints": ["best", "final"],
            "evaluation_domain": NATIVE_DOMAIN_ID,
            "point_count_per_sample": 1024,
            "valid_iid_count": VALID_COUNT,
            "train_count_used_for_frozen_normalization": 768,
            "training_performed": False,
            "optimizer_called": False,
            "solver_called": False,
            "new_data_generated": False,
            "test_iid_access": False,
            "sealed_access": False,
            "new_240825_export": False,
            "prior_240825_results_retained": True,
            "interpretation": "prediction-only archival export from frozen formal checkpoints; primary G1 closeout uses native 1024-point metrics",
        },
    )
    print(f"exported {run_id}", flush=True)


def _metric_components(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "point_global_relative_rmse_pct":
        return (
            np.asarray([float(row["point_sse"]) for row in rows]),
            np.asarray([float(row["point_truth_energy"]) for row in rows]),
        )
    if metric == "sample_first_relative_rmse_pct":
        return (
            np.asarray([float(row["sample_cv_relative_rmse"]) for row in rows]),
            np.ones(len(rows), dtype=np.float64),
        )
    if metric == "raw_K_CV_RMSE_K":
        return (
            np.asarray([float(row["cv_sse"]) for row in rows]),
            np.asarray([float(row["cv_volume"]) for row in rows]),
        )
    if metric == "source_region_RMSE_K":
        return (
            np.asarray([float(row["source_sse"]) for row in rows]),
            np.asarray([float(row["source_volume"]) for row in rows]),
        )
    if metric == "peak_RMSE_K":
        return (
            np.asarray([float(row["peak_error_squared"]) for row in rows]),
            np.ones(len(rows), dtype=np.float64),
        )
    if metric == "interface_RMSE_K":
        return (
            np.asarray([float(row["interface_error_sum_squared"]) for row in rows]),
            np.asarray([float(row["interface_error_count"]) for row in rows]),
        )
    raise ValueError(f"unsupported metric: {metric}")


def _aggregate_components(numerator: np.ndarray, denominator: np.ndarray, metric: str) -> float:
    numerator_total = float(np.sum(numerator, dtype=np.float64))
    denominator_total = float(np.sum(denominator, dtype=np.float64))
    if denominator_total <= 0.0:
        raise ValueError(f"{metric}: non-positive aggregate denominator")
    if metric == "point_global_relative_rmse_pct":
        return float(np.sqrt(numerator_total / denominator_total) * 100.0)
    if metric == "sample_first_relative_rmse_pct":
        return float(numerator_total / denominator_total * 100.0)
    return float(np.sqrt(numerator_total / denominator_total))


def _sample_error(row: Mapping[str, Any], metric: str) -> float | None:
    numerator, denominator = _metric_components([row], metric)
    if float(denominator[0]) <= 0.0:
        # A native support sample without a source node has no defined
        # source-region RMSE.  It must remain non-estimable; silently
        # assigning zero or dropping the row would change the preregistered
        # paired unit.
        return None
    return _aggregate_components(numerator, denominator, metric)


def _distribution(values: Sequence[float | None], sample_ids: Sequence[str] | None = None) -> dict[str, Any]:
    raw_values = list(values)
    valid_positions = [index for index, value in enumerate(raw_values) if value is not None]
    array = np.asarray([raw_values[index] for index in valid_positions], dtype=np.float64).reshape(-1)
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be finite")
    input_count = len(raw_values)
    if not array.size:
        return {
            "count": 0,
            "input_count": input_count,
            "non_estimable_count": input_count,
            "estimable": False,
        }
    order = np.argsort(-array, kind="mergesort")
    top = array[order[: min(10, len(array))]]
    result: dict[str, Any] = {
        "count": int(len(array)),
        "input_count": input_count,
        "non_estimable_count": input_count - int(len(array)),
        "estimable": len(valid_positions) == input_count,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "worst_10_mean": float(np.mean(top)),
        "worst_10_values_descending": [float(value) for value in top],
    }
    if sample_ids is not None:
        ids = [str(value) for value in sample_ids]
        valid_ids = [ids[index] for index in valid_positions]
        result["worst_10_sample_ids"] = [valid_ids[int(index)] for index in order[: min(10, len(valid_ids))]]
    return result


def _load_best_evaluation(derived_root: Path, run_id: str) -> dict[str, Any]:
    path = derived_root / run_id / "evaluation_best.json"
    payload = _load_json(path)
    if payload.get("checkpoint") != "best":
        raise ValueError(f"{run_id}: expected best evaluation")
    if payload.get("domain_id") != NATIVE_DOMAIN_ID:
        raise ValueError(f"{run_id}: non-native evaluation supplied")
    if payload.get("evaluation_split") != "valid_iid" or payload.get("sample_count") != VALID_COUNT:
        raise ValueError(f"{run_id}: evaluation population drifted")
    rows = payload.get("per_sample")
    if not isinstance(rows, list) or len(rows) != VALID_COUNT:
        raise ValueError(f"{run_id}: per-sample rows missing/drifted")
    if any(int(row.get("point_count", -1)) != 1024 for row in rows):
        raise ValueError(f"{run_id}: per-sample point count is not 1024")
    return payload


def _pair_rows(
    full_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    full_by_id = {str(row["sample_id"]): row for row in full_rows}
    ablation_by_id = {str(row["sample_id"]): row for row in ablation_rows}
    if set(full_by_id) != set(ablation_by_id) or len(full_by_id) != VALID_COUNT:
        raise ValueError(f"seed {seed}: paired sample IDs drifted")
    result = []
    for sample_id in sorted(full_by_id):
        full_error = _sample_error(full_by_id[sample_id], metric)
        ablation_error = _sample_error(ablation_by_id[sample_id], metric)
        result.append(
            {
                "seed": int(seed),
                "sample_id": sample_id,
                "full_error": full_error,
                "ablation_error": ablation_error,
                "effect_ablation_minus_full": (
                    float(ablation_error - full_error)
                    if full_error is not None and ablation_error is not None
                    else None
                ),
                "estimable": full_error is not None and ablation_error is not None,
            }
        )
    return result


def _bootstrap_two_level(
    full_rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    ablation_rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    full_num = []
    full_den = []
    ablation_num = []
    ablation_den = []
    for seed in SEED_SET:
        fn, fd = _metric_components(full_rows_by_seed[seed], metric)
        an, ad = _metric_components(ablation_rows_by_seed[seed], metric)
        if len(fn) != VALID_COUNT or len(an) != VALID_COUNT:
            raise ValueError(f"{metric}: bootstrap population drifted at seed {seed}")
        full_num.append(fn)
        full_den.append(fd)
        ablation_num.append(an)
        ablation_den.append(ad)
    full_num_array = np.stack(full_num)
    full_den_array = np.stack(full_den)
    ablation_num_array = np.stack(ablation_num)
    ablation_den_array = np.stack(ablation_den)
    effects = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        seed_draw = rng.integers(0, len(SEED_SET), size=len(SEED_SET))
        sample_draw = rng.integers(0, VALID_COUNT, size=(len(SEED_SET), VALID_COUNT))
        full_n = np.sum(full_num_array[seed_draw[:, None], sample_draw])
        full_d = np.sum(full_den_array[seed_draw[:, None], sample_draw])
        ablation_n = np.sum(ablation_num_array[seed_draw[:, None], sample_draw])
        ablation_d = np.sum(ablation_den_array[seed_draw[:, None], sample_draw])
        full_value = _aggregate_components(
            np.asarray([full_n]), np.asarray([full_d]), metric
        )
        ablation_value = _aggregate_components(
            np.asarray([ablation_n]), np.asarray([ablation_d]), metric
        )
        effects[replicate] = ablation_value - full_value
    return {
        "enabled": True,
        "resampling_levels": ["seed", "valid_iid_sample_within_seed"],
        "seed_resampling_with_replacement": True,
        "sample_resampling_with_replacement": True,
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "interval": "percentile_95_percent_CI",
        "ci_low": float(np.percentile(effects, 2.5)),
        "ci_high": float(np.percentile(effects, 97.5)),
        "bootstrap_effect_mean": float(np.mean(effects)),
        "bootstrap_effect_median": float(np.median(effects)),
        "replicate_effect_sha256": hashlib.sha256(
            np.asarray(effects, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def _claim_status(
    *,
    bootstrap: Mapping[str, Any],
    paired_median: float,
    seed_effects: Sequence[float],
    paired_sample_estimable: bool,
) -> str:
    # All hypotheses define effect = ablation - Full, so positive is the
    # registered direction (lower Full error).  Exact zero is not a direction.
    if not paired_sample_estimable:
        return "FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024"
    if (
        float(bootstrap["ci_low"]) > 0.0
        and float(paired_median) > 0.0
        and all(float(value) > 0.0 for value in seed_effects)
    ):
        return "SUPERIORITY_SUPPORTED"
    return "DESCRIPTIVE_ONLY"


def _analyze(args: argparse.Namespace) -> int:
    archive_root = _require_path(args.archive_root, "archive-root")
    derived_root = archive_root / "derived_1024"
    output_root = args.output_root.resolve()
    launch_path = (
        args.launch_manifest
        or Path("configs/heat3d_v7/v7_g1_formal_launch_manifest.json")
    ).resolve()
    prereg_path = (
        args.preregistration
        or Path("configs/heat3d_v7/v7_g1_statistical_preregistration.json")
    ).resolve()
    launch = _load_json(launch_path)
    prereg = _load_json(prereg_path)
    if launch.get("g1_formal_code_sha") != FORMAL_CODE_SHA:
        raise ValueError("launch formal code SHA drifted")
    if prereg.get("preregistration_sha256") != "03be1617b78f2e1f41431411e601a54136a59e363c8321457a19b717249ad31e":
        raise ValueError("statistical preregistration SHA drifted")
    run_rows = list(launch.get("runs") or [])
    if len(run_rows) != 21:
        raise ValueError("formal launch matrix is not 21 runs")
    by_variant_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for run in run_rows:
        run_id = str(run["run_id"])
        payload = _load_best_evaluation(derived_root, run_id)
        if payload.get("variant") != run["variant"] or int(payload.get("seed")) != int(run["seed"]):
            raise ValueError(f"{run_id}: evaluation provenance drifted")
        by_variant_seed[(str(run["variant"]), int(run["seed"]))] = payload
    if len(by_variant_seed) != 21:
        raise ValueError("variant/seed matrix is incomplete")

    # The user-confirmed closeout scope is native 1024 points.  The frozen
    # preregistration's historical H2 common-domain text is retained verbatim;
    # no new 240825 export is made here.  This is an explicit scope record,
    # not an edit to the frozen preregistration file.
    analysis_note = {
        "primary_domain": NATIVE_DOMAIN_ID,
        "point_count_per_sample": 1024,
        "user_scope_confirmation": "G1 performance comparisons are closed on native 1024-point evidence",
        "new_240825_results_generated": False,
        "prior_240825_results_retained": True,
        "preregistration_common_domain_text_retained": COMMON_DOMAIN_ID,
        "h2_native_interpretation": "paired by same valid_iid sample_id, with each registered run's native 1024-point support; the support arm therefore changes the sampled coordinates and H2 is reported as native-support descriptive attribution",
        "h1_h1b_metric_guard": "point_global_relative_rmse_pct is the primary aggregate functional; sample-first is not substituted",
    }
    _write_json(output_root / "analysis_scope.json", analysis_note)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_per_sample: list[dict[str, Any]] = []
    all_per_seed: list[dict[str, Any]] = []
    all_pooled: list[dict[str, Any]] = []
    all_bootstrap: list[dict[str, Any]] = []
    all_worst: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    variant_summary: list[dict[str, Any]] = []

    variants_in_launch = list(dict.fromkeys(str(row["variant"]) for row in run_rows))
    for variant in [variant for variant in variants_in_launch if variant != "Full"]:
        for metric in METRICS:
            seed_values = [
                _load_best_evaluation(derived_root, f"{variant}_seed{seed}")["metrics"][metric]
                for seed in SEED_SET
            ]
            variant_summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "seed_values": [float(value) for value in seed_values],
                    "mean": float(np.mean(seed_values)),
                    "sample_sd": float(np.std(seed_values, ddof=1)),
                    "domain": NATIVE_DOMAIN_ID,
                }
            )
    for metric in METRICS:
        seed_values = [
            _load_best_evaluation(derived_root, f"Full_seed{seed}")["metrics"][metric]
            for seed in SEED_SET
        ]
        variant_summary.append(
            {
                "variant": "Full",
                "metric": metric,
                "seed_values": [float(value) for value in seed_values],
                "mean": float(np.mean(seed_values)),
                "sample_sd": float(np.std(seed_values, ddof=1)),
                "domain": NATIVE_DOMAIN_ID,
            }
        )

    for hypothesis, spec in HYPOTHESIS_SPECS.items():
        ablation_variant = str(spec["ablation_variant"])
        full_by_seed = {
            seed: _load_best_evaluation(derived_root, f"Full_seed{seed}")["per_sample"]
            for seed in SEED_SET
        }
        ablation_by_seed = {
            seed: by_variant_seed[(ablation_variant, seed)]["per_sample"]
            for seed in SEED_SET
        }
        coordinate_grid_changed = ablation_variant in {
            "layout_agnostic_stratified_support",
            "cv_only_support",
        }
        for metric in METRICS:
            per_sample_effects: list[dict[str, Any]] = []
            per_seed_effects: list[dict[str, Any]] = []
            for seed in SEED_SET:
                pair = _pair_rows(full_by_seed[seed], ablation_by_seed[seed], metric, seed=seed)
                per_sample_effects.extend(
                    {
                        "hypothesis": hypothesis,
                        "comparison_id": spec["comparison_id"],
                        "ablation_variant": ablation_variant,
                        "metric": metric,
                        "native_domain": NATIVE_DOMAIN_ID,
                        "same_sample_id": True,
                        "same_coordinate_grid": not coordinate_grid_changed,
                        **row,
                    }
                    for row in pair
                )
                full_value = _aggregate_components(*_metric_components(full_by_seed[seed], metric), metric)
                ablation_value = _aggregate_components(*_metric_components(ablation_by_seed[seed], metric), metric)
                pair_effect_values = [row["effect_ablation_minus_full"] for row in pair]
                per_seed_effects.append(
                    {
                        "hypothesis": hypothesis,
                        "comparison_id": spec["comparison_id"],
                        "ablation_variant": ablation_variant,
                        "metric": metric,
                        "seed": seed,
                        "full_value": full_value,
                        "ablation_value": ablation_value,
                        "effect_ablation_minus_full": float(ablation_value - full_value),
                        "paired_sample_distribution": _distribution(
                            pair_effect_values,
                            [row["sample_id"] for row in pair],
                        ),
                        "full_sample_distribution": _distribution(
                            [_sample_error(row, metric) for row in full_by_seed[seed]],
                            [str(row["sample_id"]) for row in full_by_seed[seed]],
                        ),
                        "ablation_sample_distribution": _distribution(
                            [_sample_error(row, metric) for row in ablation_by_seed[seed]],
                            [str(row["sample_id"]) for row in ablation_by_seed[seed]],
                        ),
                    }
                )
            pooled_pair = per_sample_effects
            pooled_full_rows = [row for seed in SEED_SET for row in full_by_seed[seed]]
            pooled_ablation_rows = [row for seed in SEED_SET for row in ablation_by_seed[seed]]
            pooled_full_value = _aggregate_components(*_metric_components(pooled_full_rows, metric), metric)
            pooled_ablation_value = _aggregate_components(*_metric_components(pooled_ablation_rows, metric), metric)
            paired_values = [row["effect_ablation_minus_full"] for row in pooled_pair]
            bootstrap = _bootstrap_two_level(full_by_seed, ablation_by_seed, metric, rng)
            seed_effects = [float(row["effect_ablation_minus_full"]) for row in per_seed_effects]
            paired_summary = _distribution(paired_values, [str(row["sample_id"]) for row in pooled_pair])
            paired_sample_estimable = bool(paired_summary.get("estimable", False))
            paired_median = paired_summary.get("median")
            claim = _claim_status(
                bootstrap=bootstrap,
                paired_median=float(paired_median) if paired_median is not None else 0.0,
                seed_effects=seed_effects,
                paired_sample_estimable=paired_sample_estimable,
            )
            pooled = {
                "hypothesis": hypothesis,
                "comparison_id": spec["comparison_id"],
                "ablation_variant": ablation_variant,
                "metric": metric,
                "primary_metric": metric == spec["primary_metric"],
                "native_domain": NATIVE_DOMAIN_ID,
                "same_sample_id": True,
                "same_coordinate_grid": not coordinate_grid_changed,
                "paired_sample_estimable": paired_sample_estimable,
                "full_pooled_aggregate": pooled_full_value,
                "ablation_pooled_aggregate": pooled_ablation_value,
                "effect_ablation_minus_full": float(pooled_ablation_value - pooled_full_value),
                "paired_sample_distribution": paired_summary,
                "per_seed_effects": seed_effects,
                "claim_status_under_directional_rule": claim,
            }
            all_per_sample.extend(per_sample_effects)
            all_per_seed.extend(per_seed_effects)
            all_pooled.append(pooled)
            all_bootstrap.append(
                {
                    "hypothesis": hypothesis,
                    "comparison_id": spec["comparison_id"],
                    "ablation_variant": ablation_variant,
                    "metric": metric,
                    "primary_metric": metric == spec["primary_metric"],
                    "native_domain": NATIVE_DOMAIN_ID,
                    "bootstrap": bootstrap,
                    "paired_median": paired_median,
                    "per_seed_effects": seed_effects,
                    "claim_status": claim,
                }
            )
            if metric == spec["primary_metric"]:
                primary_rows.append(
                    {
                        "hypothesis": hypothesis,
                        "comparison_id": spec["comparison_id"],
                        "ablation_variant": ablation_variant,
                        "primary_metric": metric,
                        "domain": NATIVE_DOMAIN_ID,
                        "full_pooled_aggregate": pooled_full_value,
                        "ablation_pooled_aggregate": pooled_ablation_value,
                        "effect_ablation_minus_full": float(pooled_ablation_value - pooled_full_value),
                        "paired_median": paired_summary.get("median"),
                        "paired_p90": paired_summary.get("p90"),
                        "paired_p95": paired_summary.get("p95"),
                        "paired_worst_10_mean": paired_summary.get("worst_10_mean"),
                        "paired_sample_estimable": paired_sample_estimable,
                        "per_seed_effects": seed_effects,
                        "bootstrap_ci_low": float(bootstrap["ci_low"]),
                        "bootstrap_ci_high": float(bootstrap["ci_high"]),
                        "claim_status": claim,
                    }
                )
                all_worst.append(
                    {
                        "hypothesis": hypothesis,
                        "comparison_id": spec["comparison_id"],
                        "ablation_variant": ablation_variant,
                        "metric": metric,
                        "domain": NATIVE_DOMAIN_ID,
                        "per_seed": [
                            {
                                "seed": row["seed"],
                                "full": row["full_sample_distribution"],
                                "ablation": row["ablation_sample_distribution"],
                                "paired_effect": row["paired_sample_distribution"],
                            }
                            for row in per_seed_effects
                        ],
                        "pooled_384_sample_rows": {
                            "full": _distribution(
                                [_sample_error(row, metric) for row in pooled_full_rows],
                                [str(row["sample_id"]) for row in pooled_full_rows],
                            ),
                            "ablation": _distribution(
                                [_sample_error(row, metric) for row in pooled_ablation_rows],
                                [str(row["sample_id"]) for row in pooled_ablation_rows],
                            ),
                            "paired_effect": paired_summary,
                        },
                    }
                )

    h2_statuses = [
        row["claim_status"]
        for row in primary_rows
        if row["hypothesis"] in {"H2_generic", "H2_volume_only"}
    ]
    for row in primary_rows:
        if row["hypothesis"].startswith("H2_"):
            row["hypothesis_claim_group"] = "H2"
            row["h2_overall_claim_status"] = (
                "SUPERIORITY_SUPPORTED"
                if h2_statuses and all(status == "SUPERIORITY_SUPPORTED" for status in h2_statuses)
                else "DESCRIPTIVE_ONLY"
            )
    _write_json(output_root / "per_sample_effects.json", {"schema_version": "heat3d_v7_g1_per_sample_effects_v1", "rows": all_per_sample})
    _write_json(output_root / "per_seed_effects.json", {"schema_version": "heat3d_v7_g1_per_seed_effects_v1", "rows": all_per_seed})
    _write_json(output_root / "pooled_summaries.json", {"schema_version": "heat3d_v7_g1_pooled_summaries_v1", "rows": all_pooled})
    _write_per_seed_markdown(output_root / "per_seed_effects.md", all_per_seed, HYPOTHESIS_SPECS)
    _write_json(output_root / "variant_level_native_summary.json", {"schema_version": "heat3d_v7_g1_variant_level_native_summary_v1", "rows": variant_summary})
    _write_json(
        output_root / "bootstrap_ci_receipt.json",
        {
            "schema_version": "heat3d_v7_g1_bootstrap_ci_receipt_v1",
            "status": "COMPLETE",
            "domain": NATIVE_DOMAIN_ID,
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "resampling_levels": ["seed", "valid_iid_sample_within_seed"],
            "interval": "percentile_95_percent_CI",
            "effect_direction": "ablation_error_minus_Full_error; positive favors Full",
            "h1_h1b_primary_metric_guard": "point_global_relative_rmse_pct",
            "rows": all_bootstrap,
        },
    )
    _write_json(output_root / "hypothesis_effect_table.json", {"schema_version": "heat3d_v7_g1_hypothesis_effect_table_v1", "rows": primary_rows})
    _write_json(output_root / "worst_case_diagnostics.json", {"schema_version": "heat3d_v7_g1_worst_case_diagnostics_v1", "rows": all_worst})
    _write_worst_markdown(output_root / "worst_case_diagnostics.md", all_worst)
    h2_non_estimable = {
        row["hypothesis"]: int(
            sum(
                int(seed_row["paired_sample_distribution"].get("non_estimable_count", 0))
                for seed_row in all_per_seed
                if seed_row["hypothesis"] == row["hypothesis"]
                and seed_row["metric"] == row["primary_metric"]
            )
        )
        for row in primary_rows
        if row["hypothesis"].startswith("H2_")
    }
    _write_json(
        output_root / "analysis_receipt.json",
        {
            "schema_version": "heat3d_v7_g1_statistical_analysis_receipt_v1",
            "status": "COMPLETE",
            "analysis_implementation_path": "scripts/closeout_heat3d_v7_g1.py",
            "analysis_implementation_sha256": _sha256(Path(__file__).resolve()),
            "formal_code_sha": FORMAL_CODE_SHA,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "domain": NATIVE_DOMAIN_ID,
            "runs": 21,
            "variants": 7,
            "seeds": list(SEED_SET),
            "valid_iid_count_per_seed": VALID_COUNT,
            "checkpoint": "best; pre-registered valid_iid sample_first_relative_rmse_pct selection",
            "training_performed": False,
            "test_iid_access": False,
            "sealed_access": False,
            "new_240825_results_generated": False,
            "h2_native_scope": "descriptive native-support attribution; no common-domain 240825 result was generated in this closeout",
            "h2_primary_non_estimable_paired_samples": h2_non_estimable,
            "hypothesis_claim_status": {
                "H1": next(row["claim_status"] for row in primary_rows if row["hypothesis"] == "H1"),
                "H1b": next(row["claim_status"] for row in primary_rows if row["hypothesis"] == "H1b"),
                "H2": "SUPERIORITY_SUPPORTED" if h2_statuses and all(status == "SUPERIORITY_SUPPORTED" for status in h2_statuses) else "DESCRIPTIVE_ONLY",
                "H3": next(row["claim_status"] for row in primary_rows if row["hypothesis"] == "H3"),
                "H4": next(row["claim_status"] for row in primary_rows if row["hypothesis"] == "H4"),
            },
        },
    )
    _write_markdown_table(output_root / "hypothesis_effect_table.md", primary_rows)
    print(f"analysis complete: {len(primary_rows)} primary rows", flush=True)
    return 0


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.6g}"

    lines = [
        "# V7 G1 1024-point hypothesis effect table",
        "",
        "Effect is `ablation_error - Full_error`; positive favors Full. CI is the preregistered two-level percentile bootstrap CI.",
        "",
        "| Hypothesis | Comparison | Primary metric | Full pooled | Ablation pooled | Effect | Paired median | 95% CI | Seed effects | Claim status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        ci = f"[{row['bootstrap_ci_low']:.6g}, {row['bootstrap_ci_high']:.6g}]"
        seeds = ", ".join(f"{float(value):.6g}" for value in row["per_seed_effects"])
        lines.append(
            f"| {row['hypothesis']} | {row['comparison_id']} | `{row['primary_metric']}` | "
            f"{fmt(row['full_pooled_aggregate'])} | {fmt(row['ablation_pooled_aggregate'])} | "
            f"{fmt(row['effect_ablation_minus_full'])} | {fmt(row['paired_median'])} | "
            f"{ci} | {seeds} | {row['claim_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_seed_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    hypothesis_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# V7 G1 1024-point per-seed primary effects",
        "",
        "Effect is `ablation_error - Full_error`; positive favors Full. Values are computed independently for seeds 0, 1, and 2.",
        "",
        "| Hypothesis | Seed | Primary metric | Full | Ablation | Effect | Paired median | Non-estimable paired samples |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for hypothesis, spec in hypothesis_specs.items():
        metric = str(spec["primary_metric"])
        for row in rows:
            if row["hypothesis"] != hypothesis or row["metric"] != metric:
                continue
            paired = row["paired_sample_distribution"]
            median = paired.get("median")
            median_text = "NA" if median is None else f"{float(median):.6g}"
            lines.append(
                f"| {hypothesis} | {int(row['seed'])} | `{metric}` | "
                f"{float(row['full_value']):.6g} | {float(row['ablation_value']):.6g} | "
                f"{float(row['effect_ablation_minus_full']):.6g} | {median_text} | "
                f"{int(paired.get('non_estimable_count', 0))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_worst_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# V7 G1 1024-point worst-case diagnostics",
        "",
        "Worst-10 is selected by the corresponding pre-registered metric within each frozen 128-sample seed population. No row is dropped or winsorized.",
        "",
        "| Hypothesis | Seed | Metric | Full worst-10 mean | Ablation worst-10 mean | Paired-effect worst-10 mean | Non-estimable paired samples |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for seed_row in row["per_seed"]:
            paired = seed_row["paired_effect"]
            lines.append(
                f"| {row['hypothesis']} | {int(seed_row['seed'])} | `{row['metric']}` | "
                f"{float(seed_row['full']['worst_10_mean']):.6g} | "
                f"{float(seed_row['ablation']['worst_10_mean']):.6g} | "
                f"{float(paired['worst_10_mean']):.6g} | "
                f"{int(paired.get('non_estimable_count', 0))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_role(relative: str) -> str:
    name = Path(relative).name
    if relative.startswith("h2_fullfield_240825/"):
        h2_roles = {
            "H2_FORMAL_CLOSEOUT_FAIL_CLOSED.json": "h2_fail_closed_receipt",
            "run_config.json": "h2_run_config_provenance",
            "implementation_provenance.json": "h2_implementation_provenance",
            "evaluation_contract.json": "h2_common_domain_evaluation_contract",
            "per_sample_metrics.json": "h2_per_sample_metrics",
            "support_reconstruction_provenance.json": "h2_support_reconstruction_provenance",
            "evaluation_receipt.json": "h2_evaluation_receipt",
            "predictions_best.npz": "h2_best_predictions_240825",
            "query_predictions_best.npz": "h2_query_predictions",
            "query_support_indices_best.npz": "h2_query_support_indices",
        }
        return h2_roles.get(name, "h2_route_evidence")
    h2_top_level_roles = {
        "h2_execution_scope.json": "h2_execution_scope",
        "h2_analysis_receipt.json": "h2_statistical_analysis_receipt",
        "h2_per_sample_effects.json": "h2_paired_per_sample_effects",
        "h2_per_seed_effects.json": "h2_per_seed_effects",
        "h2_pooled_summaries.json": "h2_pooled_summaries",
        "h2_bootstrap_ci_receipt.json": "h2_bootstrap_ci_receipt",
        "h2_hypothesis_effect_table.json": "h2_hypothesis_effect_table",
        "h2_hypothesis_effect_table.md": "h2_hypothesis_effect_table_document",
        "h2_variant_route_summary.json": "h2_variant_route_summary",
        "h2_route_comparison.json": "h2_route_comparison",
        "h2_route_comparison.md": "h2_route_comparison_document",
        "h2_worst_case_diagnostics.json": "h2_worst_case_diagnostics",
    }
    if name in h2_top_level_roles:
        return h2_top_level_roles[name]
    roles = {
        "matrix_status.json": "formal_matrix_status",
        "formal_stderr.log": "training_stderr_log",
        "formal_stdout.log": "training_stdout_log",
        "params_best_sample_first.pkl": "best_checkpoint",
        "params_final.pkl": "final_checkpoint",
        "v7_g1_formal_receipt.json": "formal_receipt",
        "v7_g1_progress.json": "training_history_progress",
        "run_config.json": "run_config_provenance",
        "predictions_best.npz": "best_predictions_native_1024",
        "predictions_final.npz": "final_predictions_native_1024",
        "evaluation_best.json": "best_evaluation_metrics_native_1024",
        "evaluation_final.json": "final_evaluation_metrics_native_1024",
        "export_receipt.json": "prediction_export_receipt",
        "analysis_scope.json": "analysis_scope_receipt",
        "analysis_receipt.json": "statistical_analysis_receipt",
        "per_sample_effects.json": "paired_per_sample_effects",
        "per_seed_effects.json": "per_seed_effects",
        "pooled_summaries.json": "pooled_summaries",
        "variant_level_native_summary.json": "variant_level_native_summary",
        "bootstrap_ci_receipt.json": "bootstrap_ci_receipt",
        "hypothesis_effect_table.json": "hypothesis_effect_table",
        "hypothesis_effect_table.md": "hypothesis_effect_table_document",
        "per_seed_effects.md": "per_seed_effects_document",
        "worst_case_diagnostics.json": "worst_case_diagnostics",
        "worst_case_diagnostics.md": "worst_case_diagnostics_document",
    }
    if name == "derived_1024_export.log":
        return "native_1024_export_log"
    if name in roles:
        return roles[name]
    return "archived_evidence"


def _manifest(args: argparse.Namespace) -> int:
    archive_root = _require_path(args.archive_root, "archive-root")
    output_path = (
        args.manifest_output or archive_root / "archive_manifest.json"
    ).resolve()
    launch_path = (
        args.launch_manifest
        or Path("configs/heat3d_v7/v7_g1_formal_launch_manifest.json")
    ).resolve()
    launch = _load_json(launch_path)
    run_meta = {
        str(row["run_id"]): {
            "variant": str(row["variant"]),
            "seed": int(row["seed"]),
        }
        for row in launch.get("runs") or []
    }
    records = []
    archive_manifest_path = (archive_root / "archive_manifest.json").resolve()
    for path in sorted(archive_root.rglob("*")):
        if not path.is_file() or path.resolve() in {output_path, archive_manifest_path}:
            continue
        relative = path.relative_to(archive_root).as_posix()
        parts = Path(relative).parts
        run_id = None
        variant = None
        seed = None
        if len(parts) >= 2 and parts[0] in {"raw", "derived_1024"} and parts[1] in run_meta:
            run_id = parts[1]
            variant = run_meta[run_id]["variant"]
            seed = run_meta[run_id]["seed"]
        elif len(parts) >= 3 and parts[0] == "h2_fullfield_240825" and parts[2] in run_meta:
            run_id = parts[2]
            variant = run_meta[run_id]["variant"]
            seed = run_meta[run_id]["seed"]
        route = parts[1] if len(parts) >= 3 and parts[0] == "h2_fullfield_240825" else None
        records.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": _sha256(path),
                "run_id": run_id,
                "variant": variant,
                "seed": seed,
                "route": route,
                "evidence_role": _manifest_role(relative),
            }
        )
    formal_receipts = [row for row in records if row["evidence_role"] == "formal_receipt"]
    checkpoint_records = [
        row for row in records if row["evidence_role"] in {"best_checkpoint", "final_checkpoint"}
    ]
    h2_evaluation_receipt_count = sum(
        row["evidence_role"] == "h2_evaluation_receipt" for row in records
    )
    h2_fail_closed_receipt_count = sum(
        row["evidence_role"] == "h2_fail_closed_receipt" for row in records
    )
    if h2_fail_closed_receipt_count:
        h2_fullfield_status = "FAIL_CLOSED_NO_FORMAL_EVIDENCE"
    elif h2_evaluation_receipt_count == 18:
        h2_fullfield_status = "COMPLETE"
    else:
        h2_fullfield_status = "NOT_PRESENT"
    manifest = {
        "schema_version": "heat3d_v7_g1_archive_manifest_v2",
        "archive_root": str(archive_root),
        "formal_training_code_sha": FORMAL_CODE_SHA,
        "source_formal_output_root": "/tmp/v7_g1_formal_runs",
        "primary_analysis_domain": COMMON_DOMAIN_ID if h2_evaluation_receipt_count == 18 else NATIVE_DOMAIN_ID,
        "h2_primary_analysis_domain": COMMON_DOMAIN_ID,
        "new_240825_results_generated": any(
            row["evidence_role"] == "h2_evaluation_receipt" for row in records
        ),
        "prior_240825_results_retained": True,
        "h2_fullfield_closeout_status": h2_fullfield_status,
        "h2_formal_evidence_archived": h2_evaluation_receipt_count == 18,
        "file_count": len(records),
        "formal_receipt_count": len(formal_receipts),
        "checkpoint_file_count": len(checkpoint_records),
        "h2_evaluation_receipt_count": h2_evaluation_receipt_count,
        "h2_fail_closed_receipt_count": h2_fail_closed_receipt_count,
        "h2_route_evidence_count": sum(
            row["path"].startswith("h2_fullfield_240825/") for row in records
        ),
        "run_count": len(run_meta),
        "runs_with_formal_receipts": sorted({row["run_id"] for row in formal_receipts}),
        "files": records,
    }
    _write_json(output_path, manifest)
    print(f"archive manifest complete: {output_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.mode == "export-native":
        return _export_native(args)
    if args.mode == "analyze":
        return _analyze(args)
    return _manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
