#!/usr/bin/env python3
"""Independent-process, valid-only replay of one frozen P1i checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    stats_from_checkpoint_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_difference(
    expected_path: Path, regenerated: dict[str, np.ndarray]
) -> dict[str, float | int]:
    with np.load(expected_path, allow_pickle=False) as expected:
        if set(expected.files) != set(regenerated) or len(expected.files) != 128:
            raise RuntimeError("archived/regenerated valid_iid keys differ")
        differences = np.concatenate(
            [
                (
                    np.asarray(regenerated[sample_id], dtype=np.float64)
                    - np.asarray(expected[sample_id], dtype=np.float64)
                ).reshape(-1)
                for sample_id in sorted(regenerated)
            ]
        )
    absolute = np.abs(differences)
    return {
        "max_abs_error_K": float(np.max(np.abs(differences))),
        "rmse_K": float(math.sqrt(np.mean(np.square(differences)))),
        "mean_abs_error_K": float(np.mean(np.abs(differences))),
        "p99_abs_error_K": float(np.quantile(absolute, 0.99)),
        "p999_abs_error_K": float(np.quantile(absolute, 0.999)),
        "count_abs_gt_0p1_K": int(np.sum(absolute > 0.1)),
        "fraction_abs_gt_0p1_K": float(np.mean(absolute > 0.1)),
    }


def _standardizer_difference(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    if observed["feature_names"] != expected["feature_names"]:
        raise RuntimeError("global-context feature schema drifted")
    return {
        "mean_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(observed["mean"], dtype=np.float64)
                    - np.asarray(expected["mean"], dtype=np.float64)
                )
            )
        ),
        "std_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(observed["std"], dtype=np.float64)
                    - np.asarray(expected["std"], dtype=np.float64)
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--archived-predictions", type=Path)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Repeat label=checkpoint_path=archived_predictions_path.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    args = parser.parse_args()

    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    if run_config.get("dataset_loader") != "v6_p1i_dual_robin_manifest_v1":
        raise RuntimeError("independent replay requires the explicit P1i loader")
    sample_root = Path(str(run_config["subset"]))
    manifest_path = Path(str(run_config["dataset_manifest"]))
    dataset = Heat3DV6DualRobinDataset(
        sample_root,
        manifest_path,
        include_roles={"train", "valid_iid"},
    )
    if dataset.manifest.get("dataset_id") != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise RuntimeError("dataset identity drifted")
    train_ids = dataset.split_ids["train"]
    valid_ids = dataset.split_ids["valid_iid"]
    if len(train_ids) != 768 or len(valid_ids) != 128:
        raise RuntimeError("frozen train/valid populations drifted")
    index = dataset.sample_index_by_id()
    train_examples = [dataset[index[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index[sample_id]] for sample_id in valid_ids]

    raw_entries: list[tuple[str, Path, Path]] = []
    if args.entry:
        for value in args.entry:
            parts = value.split("=", 2)
            if len(parts) != 3 or not parts[0]:
                raise ValueError("--entry must be label=checkpoint=predictions")
            raw_entries.append((parts[0], Path(parts[1]), Path(parts[2])))
    elif args.checkpoint is not None and args.archived_predictions is not None:
        raw_entries.append(("checkpoint", args.checkpoint, args.archived_predictions))
    else:
        raise ValueError("provide --entry or both --checkpoint/--archived-predictions")

    first_checkpoint = runner._load_params_checkpoint(raw_entries[0][1])
    checkpoint = first_checkpoint
    checkpoint_stats = dict(checkpoint.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise RuntimeError("checkpoint lacks train-only normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    runner._validate_model_config(model_config)

    builder = runner.RunSharedSupportGraphBuilder(
        runner.Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    )
    groups = runner._make_v6_padded_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid_iid_independent_replay",
        False,
        "off",
        int(run_config["graph_seed"]),
        batch_size=int(args.prediction_batch_size),
        drop_last=False,
    )
    global_lookup, global_payload = runner._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    runner._attach_global_context_to_groups(
        groups,
        global_lookup,
        expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
    )
    examples_by_id = {example.sample_id: example for example in valid_examples}
    if model_config.get("native_output_mode") == "native_shape_scale":
        runner._attach_native_physics_to_groups(groups, examples_by_id)
        if (
            model_config.get("scale_pooling") == "qk_gated"
            or model_config.get("shape_attention_mode") != "none"
            or model_config.get("scale_attention_mode") != "none"
        ):
            runner._attach_qk_region_features_to_groups(
                groups,
                examples_by_id,
                feature_version=str(
                    model_config.get("qk_region_feature_version", "bugged_v1")
                ),
            )
        if model_config.get("scale_deepsets_mode", "none") != "none":
            runner._attach_scale_deepsets_weights_to_groups(groups, examples_by_id)

    model = GraphNeuralOperator(**model_config)
    expected_standardizer = run_config["global_context"]["standardizer"]
    standardizer_difference = _standardizer_difference(
        global_payload["standardizer"], expected_standardizer
    )
    replay_entries = []
    for label, checkpoint_path, archived_predictions in raw_entries:
        current = runner._load_params_checkpoint(checkpoint_path)
        if current.get("model_config") != first_checkpoint.get("model_config"):
            raise RuntimeError(f"{label}: checkpoint model_config drifted")
        regenerated = runner._predict_temperatures(
            model,
            runner._device_params(current["params"]),
            groups,
            stats,
        )
        if set(regenerated) != set(valid_ids):
            raise RuntimeError("regenerated prediction keys are not exactly valid_iid")
        difference = _prediction_difference(archived_predictions, regenerated)
        entry_passed = (
            difference["rmse_K"] <= 0.01
            and difference["mean_abs_error_K"] <= 0.005
            and difference["fraction_abs_gt_0p1_K"] <= 1.0e-4
        )
        replay_entries.append(
            {
                "label": label,
                "status": "passed" if entry_passed else "failed",
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                    "epoch": int(current.get("epoch", -1)),
                    "schema_version": current.get("schema_version"),
                },
                "archived_predictions": {
                    "path": str(archived_predictions),
                    "sha256": _sha256(archived_predictions),
                },
                "prediction_difference": difference,
            }
        )
    passed = (
        all(entry["status"] == "passed" for entry in replay_entries)
        and standardizer_difference["mean_max_abs_error"] <= 1.0e-12
        and standardizer_difference["std_max_abs_error"] <= 1.0e-12
    )
    payload = {
        "schema_version": "heat3d_v6_p1i_independent_checkpoint_replay_v1",
        "status": "passed" if passed else "failed",
        "process_scope": "fresh_python_process",
        "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "entries": replay_entries,
        "sample_count": len(regenerated),
        "global_context_standardizer": {
            "fit_population": global_payload["standardizer"]["fit_population"],
            "fit_sample_count": global_payload["standardizer"]["fit_sample_count"],
            **standardizer_difference,
        },
        "graph_builder_audit": builder.audit,
        "tolerances": {
            "rmse_K": 0.01,
            "mean_abs_K": 0.005,
            "fraction_abs_gt_0p1_K": 1.0e-4,
            "max_abs_K": "reported_diagnostic_not_cross_process_gate",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("independent checkpoint replay failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
