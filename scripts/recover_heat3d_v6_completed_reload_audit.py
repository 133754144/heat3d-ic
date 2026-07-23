#!/usr/bin/env python3
"""Inference-only recovery for a completed V6 run's reload audit.

The original run directory is read-only except for the explicitly requested
recovery JSON. Checkpoints, saved predictions, run_config, and loss_summary are
never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v4_controlled_training as v4_wrapper  # noqa: E402
from scripts.run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    stats_from_checkpoint_payload,
)


runner = v4_wrapper.legacy_runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _json_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalized(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    loss_summary_path = run_dir / "loss_summary.json"
    for path in (run_config_path, loss_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    loss_summary = json.loads(loss_summary_path.read_text(encoding="utf-8"))
    artifacts = {
        "final": (run_dir / "params_final.pkl", run_dir / "predictions.npz"),
        "best": (run_dir / "params_best.pkl", run_dir / "best_predictions.npz"),
        "point_global_best": (
            run_dir / "params_best_valid_point_global.pkl",
            run_dir / "point_global_best_predictions.npz",
        ),
        "base_mse_best": (
            run_dir / "params_best_valid_base_mse.pkl",
            run_dir / "base_mse_best_predictions.npz",
        ),
        "sample_first_best": (
            run_dir / "params_best_valid_sample_first.pkl",
            run_dir / "sample_first_best_predictions.npz",
        ),
    }
    artifacts = {
        label: pair
        for label, pair in artifacts.items()
        if pair[0].is_file() and pair[1].is_file()
    }
    if not {"final", "best"} <= set(artifacts):
        raise FileNotFoundError("completed final and best artifact pairs are required")

    final_payload = runner._load_params_checkpoint(artifacts["final"][0])
    metadata = dict(final_payload.get("run_config_metadata") or {})
    checkpoint_stats = dict(final_payload.get("train_only_normalization") or {})
    model_config = dict(final_payload.get("model_config") or {})
    if metadata.get("dataset_loader") != "v6_dual_robin_manifest_v1":
        raise ValueError("checkpoint is not bound to the V6 dual-Robin loader")
    final_epoch = int(final_payload.get("epoch", -1))
    if final_epoch != int(run_config["final_epoch"]):
        raise ValueError("final checkpoint epoch differs from run_config")
    if run_config.get("test_iid_group_count") != 0:
        raise ValueError("the original run materialized test_iid groups")
    if run_config.get("all_groups_count") != 0:
        raise ValueError("the original run materialized all-role groups")

    sample_root = ROOT / str(metadata["subset"])
    manifest_path = ROOT / str(metadata["dataset_manifest"])
    dataset = runner.Heat3DV6DualRobinDataset(sample_root, manifest_path)
    index = dataset.sample_index_by_id()
    train_examples = [
        dataset[index[sample_id]] for sample_id in dataset.split_ids["train"]
    ]
    valid_examples = [
        dataset[index[sample_id]] for sample_id in dataset.split_ids["valid_iid"]
    ]
    if len(train_examples) != 768 or len(valid_examples) != 128:
        raise ValueError("unexpected V6 train/valid_iid sample counts")

    v4_wrapper._install_profile_hooks(
        str(checkpoint_stats["normalization_profile"]),
        str(checkpoint_stats["condition_feature_transform"]),
        str(checkpoint_stats["input_feature_schema"]),
        str(checkpoint_stats["coord_policy"]),
        str(checkpoint_stats["extent_feature_policy"]),
    )
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    builder = runner.Heat3DGraphBuilder(**dict(metadata.get("graph_config") or {}))
    groups = runner._make_v6_padded_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid_iid_completed_reload_recovery",
        False,
        "off",
        int((metadata.get("seed_config") or {}).get("graph_seed", 0)),
        batch_size=int(run_config["prediction_batch_size"]),
        drop_last=False,
        profile_counts=None,
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
    scale_lookup, scale_payload = runner._prepare_scale_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    runner._attach_scale_context_to_groups(
        groups,
        scale_lookup,
        expected_feature_dim=int(model_config.get("scale_context_feature_dim", 0)),
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

    stored_global = dict(
        (metadata.get("global_context") or {}).get("standardizer") or {}
    )
    stored_scale = dict(
        (metadata.get("scale_context") or {}).get("standardizer") or {}
    )
    if _json_normalized(global_payload.get("standardizer") or {}) != _json_normalized(
        stored_global
    ):
        raise ValueError("reconstructed train-only global context differs")
    if _json_normalized(scale_payload.get("standardizer") or {}) != _json_normalized(
        stored_scale
    ):
        raise ValueError("reconstructed train-only scale context differs")

    model = runner.GraphNeuralOperator(**model_config)
    entries = []
    artifact_hashes = {}
    checkpoint_epochs = {}
    for label, (checkpoint_path, predictions_path) in artifacts.items():
        checkpoint = runner._load_params_checkpoint(checkpoint_path)
        entries.append(
            (
                label,
                checkpoint_path,
                predictions_path,
                _load_npz(predictions_path),
                checkpoint["params"],
            )
        )
        checkpoint_epochs[label] = int(checkpoint.get("epoch", -1))
        artifact_hashes[label] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
        }
    reload_audit = runner._checkpoint_prediction_reload_audit(
        model=model,
        groups=groups,
        stats=stats,
        entries=entries,
    )
    report = {
        "schema_version": "heat3d_v6_completed_reload_recovery_v1",
        "status": (
            "passed_recovered_post_export"
            if reload_audit["status"] == "passed"
            else "failed_recovery"
        ),
        "training_restarted": False,
        "inference_only": True,
        "roles_inferred": ["valid_iid"],
        "forbidden_roles_inferred": [],
        "sample_counts": {"train_context": 768, "valid_iid_inference": 128},
        "node_count": 1024,
        "run_training_commit": run_config["code_version_or_git_commit"],
        "recovery_evaluator_commit": _git_head(),
        "run_dir": str(run_dir),
        "run_config_sha256": _sha256(run_config_path),
        "loss_summary_sha256": _sha256(loss_summary_path),
        "original_reload_status": (
            loss_summary.get("checkpoint_prediction_reload_audit") or {}
        ).get("status"),
        "checkpoint_epochs": checkpoint_epochs,
        "checkpoint_prediction_reload_audit": reload_audit,
        "artifacts": artifact_hashes,
        "global_context": global_payload,
        "scale_context": scale_payload,
        "original_run_artifacts_modified": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_normalized(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if reload_audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
