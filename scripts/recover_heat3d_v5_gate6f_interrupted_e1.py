#!/usr/bin/env python3
"""Recover the read-only reload audit for an interrupted Gate 6F e1 smoke.

This script never trains.  It reconstructs only the train-fitted preprocessing
state and the valid_iid graph batch, reloads the already-written final/best
checkpoints, and compares their predictions with the already-written NPZs.
It writes the recovery report outside the original run directory.
"""

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
EXPECTED_CONFIG_ID = "V4P5_16_gate6f_mean_max_smoke"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
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
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_ends = [row for row in rows if row.get("stage") == "train_batch_end"]
    finite_keys = ("loss", "grad_norm", "param_norm", "update_norm")
    finite = bool(train_ends) and all(
        all(math.isfinite(float(row.get("detail", {}).get(key, math.nan))) for key in finite_keys)
        for row in train_ends
    )
    device_peaks = []
    for row in rows:
        for device in (row.get("jax_memory") or {}).get("jax_devices") or []:
            value = device.get("peak_pool_bytes_mb")
            if value is not None:
                device_peaks.append(float(value))
    return {
        "event_count": len(rows),
        "last_stage": rows[-1].get("stage") if rows else None,
        "train_batch_end_count": len(train_ends),
        "train_batch_details_finite": finite,
        "peak_rss_mb": max(float(row.get("rss_mb", 0.0)) for row in rows),
        "peak_device_memory_all_mb": max(device_peaks, default=0.0),
        "sha256": _sha256(path),
    }


def _load_examples(
    metadata: dict[str, Any], checkpoint_stats: dict[str, Any]
) -> tuple[list[Any], list[Any]]:
    subset = ROOT / str(metadata["subset"])
    sample_root = runner._sample_root(subset)
    split_map = ROOT / str(metadata["split_map_path"])
    split_ids, _source, _primary, _stress = runner._resolve_training_splits(sample_root, split_map)
    train_ids = list(split_ids.get("train") or [])
    valid_ids = list(split_ids.get("valid_iid") or [])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise ValueError(f"unexpected split counts train={len(train_ids)} valid_iid={len(valid_ids)}")
    names = tuple(checkpoint_stats.get("feature_names") or ())
    k_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(names) else "native"
    dataset = runner.Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=k_mode,
        boundary_mask_fallback=bool(metadata.get("boundary_mask_fallback", True)),
    )
    index = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in [*train_ids, *valid_ids] if sample_id not in index]
    if missing:
        raise FileNotFoundError(f"missing requested train/valid samples: {missing[:5]}")
    return (
        [dataset[index[sample_id]] for sample_id in train_ids],
        [dataset[index[sample_id]] for sample_id in valid_ids],
    )


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    memory_path = args.memory_jsonl.resolve()
    artifacts = {
        "final_checkpoint": run_dir / "params_final.pkl",
        "best_checkpoint": run_dir / "params_best.pkl",
        "final_predictions": run_dir / "predictions.npz",
        "best_predictions": run_dir / "best_predictions.npz",
    }
    for path in [memory_path, *artifacts.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (run_dir / "loss_summary.json").exists() or (run_dir / "run_config.json").exists():
        raise ValueError("recovery is only valid for the interrupted post-output V16 run")

    final_payload = runner._load_params_checkpoint(artifacts["final_checkpoint"])
    best_payload = runner._load_params_checkpoint(artifacts["best_checkpoint"])
    metadata = dict(final_payload.get("run_config_metadata") or {})
    model_config = dict(final_payload.get("model_config") or {})
    checkpoint_stats = dict(final_payload.get("train_only_normalization") or {})
    if final_payload.get("checkpoint_kind") != "final" or int(final_payload.get("epoch", -1)) != 1:
        raise ValueError("final checkpoint is not the expected e1 payload")
    if best_payload.get("checkpoint_kind") != "best" or int(best_payload.get("epoch", -1)) != 1:
        raise ValueError("best checkpoint is not the expected e1 payload")
    if Path(str(metadata.get("output_dir"))).name != run_dir.name:
        raise ValueError("checkpoint output_dir does not bind to the recovery run")
    if model_config.get("scale_pooling") != "mean_max":
        raise ValueError("checkpoint is not the V16 mean+max candidate")
    if model_config != dict(best_payload.get("model_config") or {}):
        raise ValueError("best/final model configs differ")
    if checkpoint_stats != dict(best_payload.get("train_only_normalization") or {}):
        raise ValueError("best/final train-only normalization differs")

    v4_wrapper._install_profile_hooks(
        str(checkpoint_stats["normalization_profile"]),
        str(checkpoint_stats["condition_feature_transform"]),
        str(checkpoint_stats["input_feature_schema"]),
        str(checkpoint_stats["coord_policy"]),
        str(checkpoint_stats["extent_feature_policy"]),
    )
    train_examples, valid_examples = _load_examples(metadata, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    builder = runner.Heat3DGraphBuilder(**dict(metadata.get("graph_config") or {}))
    groups = runner._make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid_iid_recovery",
        False,
        "off",
        int((metadata.get("seed_config") or {}).get("graph_seed", 0)),
        batch_size=128,
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
        raise ValueError("reconstructed train-only global-context standardizer differs")
    runner._attach_global_context_to_groups(
        groups,
        context,
        expected_feature_dim=int(model_config["global_context_feature_dim"]),
    )
    examples_by_id = {example.sample_id: example for example in valid_examples}
    runner._attach_native_physics_to_groups(groups, examples_by_id)
    if model_config.get("scale_pooling") == "qk_gated_pooling":
        runner._attach_qk_region_features_to_groups(groups, examples_by_id)

    model = runner.GraphNeuralOperator(**model_config)
    expected_final = _load_npz(artifacts["final_predictions"])
    expected_best = _load_npz(artifacts["best_predictions"])
    reload_audit = runner._checkpoint_prediction_reload_audit(
        model=model,
        groups=groups,
        stats=stats,
        entries=[
            (
                "final",
                artifacts["final_checkpoint"],
                artifacts["final_predictions"],
                expected_final,
                final_payload["params"],
            ),
            (
                "best",
                artifacts["best_checkpoint"],
                artifacts["best_predictions"],
                expected_best,
                best_payload["params"],
            ),
        ],
    )
    native_audit = runner._native_runtime_architecture_audit(
        model, runner._device_params(final_payload["params"]), groups[0]
    )
    memory = _memory_evidence(memory_path)
    passed = bool(
        reload_audit.get("status") == "passed"
        and native_audit.get("passed")
        and memory["train_batch_end_count"] == 24
        and memory["train_batch_details_finite"]
        and memory["last_stage"] == "best_prediction_save_end"
    )
    if not passed:
        raise RuntimeError("V16 recovery evidence did not pass")

    report = {
        "schema_version": "heat3d_v5_gate6f_interrupted_e1_recovery_v1",
        "config_id": EXPECTED_CONFIG_ID,
        "status": "passed_recovered_post_interrupt",
        "interrupt": "KeyboardInterrupt",
        "interrupt_stage": "checkpoint_prediction_reload_audit",
        "training_restarted": False,
        "inference_only": True,
        "roles_materialized": ["train", "valid_iid"],
        "forbidden_roles_materialized": [],
        "sealed_iid_accessed": False,
        "sample_counts": {"train": len(train_examples), "valid_iid": len(valid_examples)},
        "node_count": int(valid_examples[0].condition.coords.shape[0]),
        "batch_size": 128,
        "global_context": context_meta,
        "checkpoint_prediction_reload_audit": reload_audit,
        "native_runtime_architecture_audit": native_audit,
        "memory_audit_summary": memory,
        "record": final_payload.get("record"),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "git_commit": final_payload.get("git_commit"),
        "long_training_started": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_normalized(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "checkpoint_reload": reload_audit,
        "memory": memory,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
