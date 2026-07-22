#!/usr/bin/env python3
"""Inference-only recovery of a completed V6 e1 checkpoint reload audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v4_controlled_training as v4_wrapper  # noqa: E402
from scripts.run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    stats_from_checkpoint_payload,
)


runner = v4_wrapper.legacy_runner


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def _memory_evidence(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_ends = [row for row in rows if row.get("stage") == "train_batch_end"]
    finite_keys = ("loss", "grad_norm", "param_norm", "update_norm")
    finite = bool(train_ends) and all(
        all(
            math.isfinite(float(row.get("detail", {}).get(key, math.nan)))
            for key in finite_keys
        )
        for row in train_ends
    )
    device_peaks = [
        float(device["peak_bytes_in_use_mb"])
        for row in rows
        for device in (row.get("jax_memory") or {}).get("jax_devices") or []
        if device.get("peak_bytes_in_use_mb") is not None
    ]
    device_limits = [
        float(device["bytes_limit_mb"])
        for row in rows
        for device in (row.get("jax_memory") or {}).get("jax_devices") or []
        if device.get("bytes_limit_mb") is not None
    ]
    return {
        "event_count": len(rows),
        "last_stage": rows[-1].get("stage") if rows else None,
        "train_batch_end_count": len(train_ends),
        "train_batch_details_finite": finite,
        "peak_rss_mb": max((float(row.get("rss_mb", 0.0)) for row in rows), default=0.0),
        "peak_device_bytes_in_use_mb": max(device_peaks, default=0.0),
        "device_limit_mb": max(device_limits, default=0.0),
        "sha256": _sha256(path),
    }


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    loss_summary_path = run_dir / "loss_summary.json"
    memory_path = run_dir / "memory_audit.jsonl"
    artifacts = {
        "final": (run_dir / "params_final.pkl", run_dir / "predictions.npz"),
        "legacy_best": (run_dir / "params_best.pkl", run_dir / "best_predictions.npz"),
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
    for path in [run_config_path, loss_summary_path, memory_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for checkpoint, predictions in artifacts.values():
        if not checkpoint.is_file() or not predictions.is_file():
            raise FileNotFoundError(f"missing checkpoint/predictions pair: {checkpoint}, {predictions}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    loss_summary = json.loads(loss_summary_path.read_text(encoding="utf-8"))
    final_payload = runner._load_params_checkpoint(artifacts["final"][0])
    metadata = dict(final_payload.get("run_config_metadata") or {})
    checkpoint_stats = dict(final_payload.get("train_only_normalization") or {})
    model_config = dict(final_payload.get("model_config") or {})
    if metadata.get("dataset_loader") != "v6_dual_robin_manifest_v1":
        raise ValueError("checkpoint is not bound to the V6 dual-Robin loader")
    if int(metadata.get("epochs", -1)) != 1 or int(final_payload.get("epoch", -1)) != 1:
        raise ValueError("recovery only accepts completed e1 artifacts")
    if run_config.get("test_iid_group_count") != 0 or run_config.get("all_groups_count") != 0:
        raise ValueError("forbidden test/all groups were materialized")

    sample_root = ROOT / str(metadata["subset"])
    manifest_path = ROOT / str(metadata["dataset_manifest"])
    dataset = runner.Heat3DV6DualRobinDataset(sample_root, manifest_path)
    train_ids = list(dataset.split_ids["train"])
    valid_ids = list(dataset.split_ids["valid_iid"])
    if len(train_ids) != 768 or len(valid_ids) != 128:
        raise ValueError("unexpected canonical V6 split counts")
    index = dataset.sample_index_by_id()
    train_examples = [dataset[index[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index[sample_id]] for sample_id in valid_ids]

    v4_wrapper._install_profile_hooks(
        str(checkpoint_stats["normalization_profile"]),
        str(checkpoint_stats["condition_feature_transform"]),
        str(checkpoint_stats["input_feature_schema"]),
        str(checkpoint_stats["coord_policy"]),
        str(checkpoint_stats["extent_feature_policy"]),
    )
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    builder = runner.Heat3DGraphBuilder(**dict(metadata.get("graph_config") or {}))
    groups = runner._make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid_iid_reload_recovery",
        False,
        "off",
        int((metadata.get("seed_config") or {}).get("graph_seed", 0)),
        batch_size=32,
        drop_last=False,
        profile_counts=None,
    )
    context, context_meta = runner._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    stored_context = dict((metadata.get("global_context") or {}).get("standardizer") or {})
    if _json_normalized(context_meta.get("standardizer") or {}) != _json_normalized(stored_context):
        raise ValueError("reconstructed train-only global context differs")
    runner._attach_global_context_to_groups(
        groups,
        context,
        expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
    )
    examples_by_id = {example.sample_id: example for example in valid_examples}
    runner._attach_native_physics_to_groups(groups, examples_by_id)
    runner._attach_qk_region_features_to_groups(
        groups,
        examples_by_id,
        feature_version=str(
            model_config.get("qk_region_feature_version", "bugged_v1")
        ),
    )

    model = runner.GraphNeuralOperator(**model_config)
    entries = []
    artifact_hashes = {}
    for label, (checkpoint, predictions) in artifacts.items():
        payload = runner._load_params_checkpoint(checkpoint)
        entries.append(
            (
                label,
                checkpoint,
                predictions,
                _load_npz(predictions),
                payload["params"],
            )
        )
        artifact_hashes[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "predictions": str(predictions),
            "predictions_sha256": _sha256(predictions),
        }
    reload_audit = runner._checkpoint_prediction_reload_audit(
        model=model,
        groups=groups,
        stats=stats,
        entries=entries,
    )
    memory = _memory_evidence(memory_path)
    finite_history = all(
        math.isfinite(float(value))
        for record in loss_summary.get("epoch_history") or []
        for key, value in record.items()
        if value is not None and ("loss" in key or "grad_norm" in key)
    )
    passed = bool(
        reload_audit.get("status") == "passed"
        and memory["train_batch_end_count"] == 28
        and memory["train_batch_details_finite"]
        and finite_history
        and (metadata.get("global_context") or {}).get("standardizer", {}).get(
            "fit_population"
        )
        == "train_only"
    )
    if not passed:
        raise RuntimeError("V6 e1 recovery evidence did not pass")

    report = {
        "schema_version": "heat3d_v6_e1_reload_recovery_v1",
        "status": "passed_recovered_post_export",
        "training_restarted": False,
        "inference_only": True,
        "roles_materialized": ["train", "valid_iid"],
        "forbidden_roles_materialized": [],
        "sample_counts": {"train": 768, "valid_iid": 128},
        "node_count": 1024,
        "dataset": {
            "id": sample_root.name,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "group_locked": True,
            "bottom_boundary_semantics": "robin_not_fixed_temperature",
        },
        "effective_batch": {
            "configured": 28,
            "effective": 28,
            "micro_max": 8,
            "updates_per_epoch": 28,
            "tail": 12,
        },
        "checkpoint_prediction_reload_audit": reload_audit,
        "global_context": context_meta,
        "memory_audit": memory,
        "finite_epoch_history": finite_history,
        "artifacts": artifact_hashes,
        "run_config_sha256": _sha256(run_config_path),
        "loss_summary_sha256": _sha256(loss_summary_path),
        "training_commit": run_config.get("code_version_or_git_commit"),
        "formal_training_started": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_normalized(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "reload": reload_audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
