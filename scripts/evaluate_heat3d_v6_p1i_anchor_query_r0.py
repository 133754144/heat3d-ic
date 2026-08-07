#!/usr/bin/env python3
"""Strict valid-only R0 equivalence gate for one frozen P1i checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import jax
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import audit_heat3d_v6_p1i_controlled_cross_resolution as cross  # noqa: E402
import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.heat3d_v6_dataset import CONTINUOUS_PHYSICS_V6_DATASET_ID, Heat3DV6DualRobinDataset  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import P1iSampleVaryingAnchorQueryAdapter, array_sha256  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks, stats_from_checkpoint_payload  # noqa: E402


MODEL_GROUP_KEYS = (
    "inputs", "graphs", "global_context", "native_physics",
    "qk_region_features", "scale_context", "scale_region_source_weights",
    "scale_region_volume_weights",
)
MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
FULL_FIELDS_SHA = "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_arrays(value: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(value)
            if leaf is not None and hasattr(leaf, "shape")]


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    for array in tree_arrays(value):
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_equivalence(reference: Any, adapted: Any) -> dict[str, Any]:
    left_structure = str(jax.tree_util.tree_structure(reference))
    right_structure = str(jax.tree_util.tree_structure(adapted))
    left, right = tree_arrays(reference), tree_arrays(adapted)
    shape_equal = len(left) == len(right) and all(
        a.shape == b.shape and a.dtype == b.dtype for a, b in zip(left, right, strict=False)
    )
    exact = left_structure == right_structure and shape_equal and all(
        np.array_equal(a, b) for a, b in zip(left, right, strict=False)
    )
    errors = [float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
              for a, b in zip(left, right, strict=False) if shape_equal and a.size]
    return {
        "passed": bool(exact),
        "tree_structure_exact": left_structure == right_structure,
        "leaf_shape_dtype_exact": shape_equal,
        "array_values_exact": exact,
        "reference_sha256": tree_sha256(reference),
        "adapter_sha256": tree_sha256(adapted),
        "max_abs_error": max(errors, default=0.0),
    }


def difference(expected: Mapping[str, np.ndarray], actual: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(expected) != set(actual):
        raise RuntimeError("prediction sample IDs differ")
    values = np.concatenate([
        (np.asarray(actual[key], dtype=np.float64) - np.asarray(expected[key], dtype=np.float64)).reshape(-1)
        for key in sorted(expected)
    ])
    absolute = np.abs(values)
    return {
        "max_abs_error_K": float(np.max(absolute)),
        "rmse_K": float(math.sqrt(np.mean(np.square(values)))),
        "mean_abs_error_K": float(np.mean(absolute)),
        "p99_abs_error_K": float(np.quantile(absolute, 0.99)),
        "p999_abs_error_K": float(np.quantile(absolute, 0.999)),
        "count_abs_gt_0p1_K": int(np.sum(absolute > 0.1)),
        "fraction_abs_gt_0p1_K": float(np.mean(absolute > 0.1)),
    }


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name], dtype=np.float64) for name in payload.files}


def prepare_groups(examples: Sequence[Any], *, stats: Mapping[str, Any],
                   run_config: Mapping[str, Any], model_config: Mapping[str, Any],
                   global_lookup: Mapping[str, np.ndarray], label: str,
                   batch_size: int) -> tuple[list[dict[str, Any]], Any]:
    builder = runner.RunSharedSupportGraphBuilder(
        runner.Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    )
    groups = runner._make_v6_padded_groups_with_progress(
        examples, dict(stats), builder, label, False, "off",
        int(run_config["graph_seed"]), batch_size=batch_size, drop_last=False,
    )
    runner._attach_global_context_to_groups(
        groups, dict(global_lookup),
        expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
    )
    by_id = {example.sample_id: example for example in examples}
    runner._attach_native_physics_to_groups(groups, by_id)
    if (model_config.get("scale_pooling") == "qk_gated"
            or model_config.get("shape_attention_mode") != "none"
            or model_config.get("scale_attention_mode") != "none"):
        runner._attach_qk_region_features_to_groups(
            groups, by_id, feature_version=str(model_config["qk_region_feature_version"])
        )
    if model_config.get("scale_deepsets_mode", "none") != "none":
        runner._attach_scale_deepsets_weights_to_groups(groups, by_id)
    return groups, builder


def predict_with_scales(model: Any, params: Any, groups: Sequence[Mapping[str, Any]]
                        ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions, scales = {}, {}
    for group in groups:
        output = runner._model_apply(model, params, group)
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        scale = np.asarray(output["s_hat"], dtype=np.float64).reshape(len(group["sample_ids"]), -1)
        for index, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = raw[index, 0, :, :]
            scales[str(sample_id)] = float(scale[index, 0])
    return predictions, scales


def scale_difference(expected: Mapping[str, float], actual: Mapping[str, float]) -> dict[str, float]:
    if set(expected) != set(actual):
        raise RuntimeError("predicted-scale sample IDs differ")
    values = np.asarray([actual[key] - expected[key] for key in sorted(expected)])
    return {"max_abs_error": float(np.max(np.abs(values))),
            "rmse": float(math.sqrt(np.mean(np.square(values))))}


def metric_differences(actual_support: Mapping[str, Any], actual_full: Mapping[str, Any],
                       frozen_closeout: Mapping[str, Any], seed: int) -> dict[str, Any]:
    frozen = next(row["primary"] for row in frozen_closeout["seeds"] if int(row["seed"]) == seed)
    pairs = {
        "support_point_global_pct": (actual_support["point_global_true_rms_relative_rmse_pct"], frozen["support_point_global_pct"]),
        "support_sample_first_pct": (actual_support["sample_first_cv_relative_rmse_pct"], frozen["support_sample_first_pct"]),
        "support_raw_cv_rmse_K": (actual_support["raw_cv_weighted_rmse_K"], frozen["support_raw_cv_rmse_K"]),
        "full_point_global_pct": (actual_full["point_global_true_rms_relative_rmse_pct"], frozen["full_point_global_pct"]),
        "full_sample_first_pct": (actual_full["sample_first_cv_relative_rmse_pct"], frozen["full_sample_first_pct"]),
        "full_raw_cv_rmse_K": (actual_full["raw_cv_weighted_rmse_K"], frozen["full_raw_cv_rmse_K"]),
    }
    return {name: {"actual": float(a), "frozen": float(b), "difference": float(a - b)}
            for name, (a, b) in pairs.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archived-predictions", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--frozen-closeout", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    hard = contract["hard_gate"]
    if sha256(args.manifest) != MANIFEST_SHA or sha256(args.full_fields) != FULL_FIELDS_SHA:
        raise RuntimeError("frozen P1i manifest/full-field SHA mismatch")
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA mismatch")
    run_config = json.loads((args.run_dir / "run_config.json").read_text())

    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root, args.manifest, include_roles={"train", "valid_iid"}
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise RuntimeError("dataset ID mismatch")
    train_ids, valid_ids = dataset.split_ids["train"], dataset.split_ids["valid_iid"]
    if len(train_ids) != 768 or len(valid_ids) != 128:
        raise RuntimeError("frozen train/valid population mismatch")
    index = dataset.sample_index_by_id()
    train_examples = [dataset[index[sample_id]] for sample_id in train_ids]
    reference_examples = [dataset[index[sample_id]] for sample_id in valid_ids]
    adapters = [P1iSampleVaryingAnchorQueryAdapter(example) for example in reference_examples]
    adapted_examples = [adapter.r0_example() for adapter in adapters]
    input_checks = [{"sample_id": example.sample_id, **adapter.r0_input_equivalence(example)}
                    for adapter, example in zip(adapters, adapted_examples, strict=True)]

    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != args.checkpoint_epoch:
        raise RuntimeError("checkpoint epoch mismatch")
    checkpoint_stats = dict(checkpoint.get("train_only_normalization") or {})
    install_checkpoint_feature_hooks(checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    model_config = runner._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    runner._validate_model_config(model_config)
    global_lookup, global_payload = runner._prepare_global_context_lookup(
        model_config, train_examples=train_examples, required_examples=reference_examples
    )
    expected_standardizer = run_config["global_context"]["standardizer"]
    standardizer = global_payload["standardizer"]
    standardizer_diff = {
        "mean_max_abs_error": float(np.max(np.abs(np.asarray(standardizer["mean"]) - np.asarray(expected_standardizer["mean"])))),
        "std_max_abs_error": float(np.max(np.abs(np.asarray(standardizer["std"]) - np.asarray(expected_standardizer["std"])))),
        "fit_population": standardizer["fit_population"],
        "fit_sample_count": int(standardizer["fit_sample_count"]),
    }
    adapted_global_lookup = {
        example.sample_id: qualification.common.standardize_v6_contexts(
            [runner._global_context_row_for_example(example)], standardizer
        )[0]
        for example in adapted_examples
    }

    reference_groups, reference_builder = prepare_groups(
        reference_examples, stats=stats, run_config=run_config, model_config=model_config,
        global_lookup=global_lookup, label="r0_reference", batch_size=args.prediction_batch_size,
    )
    adapted_groups, adapted_builder = prepare_groups(
        adapted_examples, stats=stats, run_config=run_config, model_config=model_config,
        global_lookup=adapted_global_lookup, label="r0_adapter", batch_size=args.prediction_batch_size,
    )
    if len(reference_groups) != len(adapted_groups):
        raise RuntimeError("reference/adapter group count differs")
    group_checks = []
    for reference, adapted in zip(reference_groups, adapted_groups, strict=True):
        sections = {key: tree_equivalence(reference[key], adapted[key])
                    for key in MODEL_GROUP_KEYS if key in reference or key in adapted}
        sample_ids_exact = tuple(reference["sample_ids"]) == tuple(adapted["sample_ids"])
        group_checks.append({
            "reference_name": reference["name"], "adapter_name": adapted["name"],
            "sample_ids_exact": sample_ids_exact, "sections": sections,
            "passed": bool(sample_ids_exact and all(row["passed"] for row in sections.values())),
        })

    graph_seed = int(run_config["graph_seed"])
    ref_raw_builder = runner.Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    adapter_raw_builder = runner.Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    graph_checks = []
    for reference, adapted in zip(reference_examples, adapted_examples, strict=True):
        ref_meta = ref_raw_builder.build_metadata(
            runner._graph_coords_for_example(reference, stats), key=jax.random.PRNGKey(graph_seed))
        adapter_meta = adapter_raw_builder.build_metadata(
            runner._graph_coords_for_example(adapted, stats), key=jax.random.PRNGKey(graph_seed))
        ref_hash, adapter_hash = cross.graph_sha256(ref_meta, 1024), cross.graph_sha256(adapter_meta, 1024)
        graph_checks.append({
            "sample_id": reference.sample_id,
            "reference_real_edge_sha256": ref_hash, "adapter_real_edge_sha256": adapter_hash,
            "real_edge_semantics_exact": ref_hash == adapter_hash,
            "metadata": tree_equivalence(ref_meta, adapter_meta),
        })

    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    reference_predictions, reference_scales = predict_with_scales(model, params, reference_groups)
    adapted_predictions, adapted_scales = predict_with_scales(model, params, adapted_groups)
    archived = load_predictions(args.archived_predictions)
    adapter_vs_reference = difference(reference_predictions, adapted_predictions)
    adapter_vs_archived = difference(archived, adapted_predictions)
    adapter_scale_vs_reference = scale_difference(reference_scales, adapted_scales)

    family = qualification.FamilyData(
        family="p1i", dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=None,
    )
    row_by_id = {str(row["sample_id"]): row for row in family.valid_rows}
    support_rows, full_rows = [], []
    full_reference, full_adapter, full_archived = {}, {}, {}
    reconstruction_checks = []
    for sample_id in valid_ids:
        _, public = cross.checkpoint_example(family, row_by_id[sample_id], "formal_valid128")
        reference_delta = np.asarray(reference_predictions[sample_id]).reshape(-1) - public["reference_K"]
        adapted_delta = np.asarray(adapted_predictions[sample_id]).reshape(-1) - public["reference_K"]
        archived_delta = np.asarray(archived[sample_id]).reshape(-1) - public["reference_K"]
        full_reference[sample_id] = public["mapping"].reconstruct(reference_delta)
        full_adapter[sample_id] = public["mapping"].reconstruct(adapted_delta)
        full_archived[sample_id] = public["mapping"].reconstruct(archived_delta)
        support_rows.append(cross.one_metric_row(
            adapted_delta, public["support_truth"], public["support_cv"], public["support_coords"],
            public["support_layer"], public["support_q"]))
        full_rows.append(cross.one_metric_row(
            full_adapter[sample_id], public["full_truth"], public["full_cv"], public["full_coords"],
            public["full_layer"], public["full_q"]))
        reconstruction_checks.append({
            "sample_id": sample_id,
            "support_indices_sha256": array_sha256(public["mapping"].support_indices),
            "mapping_sha256": tree_sha256(public["mapping"]),
            "mode": public["map_mode"],
            "label_independent": bool(public["map_audit"]["label_independent"]),
        })
    full_adapter_vs_reference = difference(full_reference, full_adapter)
    full_adapter_vs_archived = difference(full_archived, full_adapter)
    support_metrics = qualification.metric_accumulate(support_rows, full=False)
    full_metrics = qualification.metric_accumulate(full_rows, full=True)
    frozen_differences = metric_differences(
        support_metrics, full_metrics, json.loads(args.frozen_closeout.read_text()), args.seed)

    archived_limit = hard["adapter_vs_archived_prediction"]
    full_archived_limit = hard["adapter_vs_archived_full_field_reconstruction"]
    aggregate_tolerance = float(hard["frozen_aggregate_metrics"]["absolute_tolerance"])
    checks = {
        "input_exact": all(row["passed"] for row in input_checks),
        "group_features_exact": all(row["passed"] for row in group_checks),
        "graph_semantics_exact": all(row["real_edge_semantics_exact"] and row["metadata"]["passed"] for row in graph_checks),
        "adapter_reference_prediction_exact": adapter_vs_reference["max_abs_error_K"] == 0.0,
        "adapter_reference_scale_exact": adapter_scale_vs_reference["max_abs_error"] == 0.0,
        "adapter_reference_full_field_exact": full_adapter_vs_reference["max_abs_error_K"] == 0.0,
        "archived_prediction_replay": (
            adapter_vs_archived["rmse_K"] <= float(archived_limit["rmse_K_max"])
            and adapter_vs_archived["mean_abs_error_K"] <= float(archived_limit["mean_abs_error_K_max"])
            and adapter_vs_archived["fraction_abs_gt_0p1_K"] <= float(archived_limit["fraction_abs_gt_0p1_K_max"])),
        "archived_full_field_replay": (
            full_adapter_vs_archived["rmse_K"] <= float(full_archived_limit["rmse_K_max"])
            and full_adapter_vs_archived["mean_abs_error_K"] <= float(full_archived_limit["mean_abs_error_K_max"])
            and full_adapter_vs_archived["fraction_abs_gt_0p1_K"] <= float(full_archived_limit["fraction_abs_gt_0p1_K_max"])),
        "frozen_metrics_secondary": all(abs(row["difference"]) <= aggregate_tolerance for row in frozen_differences.values()),
        "train_only_standardizer_exact": (
            standardizer_diff["mean_max_abs_error"] <= float(hard["train_only_standardizer"]["mean_max_abs_error"])
            and standardizer_diff["std_max_abs_error"] <= float(hard["train_only_standardizer"]["std_max_abs_error"])
            and standardizer_diff["fit_population"] == "train_only"
            and standardizer_diff["fit_sample_count"] == 768),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_query_r0_seed_result_v1",
        "status": "passed" if passed else "failed", "config_id": args.config_id, "seed": args.seed,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "epoch": int(checkpoint["epoch"])},
        "archived_predictions": {"path": str(args.archived_predictions), "sha256": sha256(args.archived_predictions)},
        "dataset": {"dataset_id": dataset.manifest["dataset_id"], "manifest_sha256": sha256(args.manifest),
                    "full_field_archive_sha256": sha256(args.full_fields), "train_count": 768,
                    "valid_iid_count": 128, "anchor_count": 1024, "full_field_node_count": 240825},
        "contract_sha256": sha256(args.contract), "checks": checks,
        "input_equivalence": {"sample_count": len(input_checks), "all_passed": all(row["passed"] for row in input_checks), "samples": input_checks},
        "group_equivalence": group_checks,
        "graph_equivalence": {"sample_count": len(graph_checks),
                              "unique_reference_hash_count": len({row["reference_real_edge_sha256"] for row in graph_checks}),
                              "all_passed": checks["graph_semantics_exact"], "samples": graph_checks,
                              "reference_builder_audit": reference_builder.audit, "adapter_builder_audit": adapted_builder.audit},
        "feature_and_scale_equivalence": {"group_sections": [row["sections"] for row in group_checks],
                                          "predicted_scale": adapter_scale_vs_reference, "standardizer": standardizer_diff},
        "prediction_equivalence": {"adapter_vs_reference": adapter_vs_reference,
                                   "adapter_vs_archived": adapter_vs_archived, "hard_gate_is_prediction_level": True},
        "full_field_reconstruction_equivalence": {
            "adapter_vs_reference": full_adapter_vs_reference, "adapter_vs_archived": full_adapter_vs_archived,
            "mapping_sample_count": len(reconstruction_checks),
            "unique_mapping_hash_count": len({row["mapping_sha256"] for row in reconstruction_checks}),
            "samples": reconstruction_checks},
        "metrics_secondary": {"support": support_metrics, "full_240825": full_metrics,
                              "frozen_differences": frozen_differences,
                              "aggregate_metrics_are_not_prediction_gate": True},
        "role_contract": {"accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
                          "test_accessed": False, "sealed_accessed": False, "training_executed": False,
                          "checkpoint_modified": False, "high_n_inference_executed": False},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "config_id": args.config_id, "checks": checks}, indent=2))
    if not passed:
        raise RuntimeError(f"{args.config_id}: R0 anchor/query equivalence gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
