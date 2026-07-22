#!/usr/bin/env python3
"""Frozen V42 e257 test_iid evaluation and batch-1 inference timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_heat3d_v5_gate6q_closeout import (  # noqa: E402
    _build_groups,
    _checkpoint,
    _load_examples_for_ids,
    _resolve_training_splits,
    _sample_root,
    _suite,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v5_scale_context import standardize_scale_contexts  # noqa: E402
from rigno.heat3d_v5_metrics import control_volume_weights  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    _attach_qk_region_features_to_groups,
    _attach_scale_context_to_groups,
    _attach_scale_deepsets_weights_to_groups,
    _device_params,
    _make_batch_group_with_seed,
    _model_apply,
    _resolve_decoder_bypass_model_config,
    _scale_context_row_for_example,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    stats_from_checkpoint_payload,
)
from run_heat3d_v5_clean_first import _attach_v5_physics, _physics_cache  # noqa: E402


CONFIG_ID = "V4P5_42_gate6q_objective_only_e600"
EXPECTED_EPOCH = 257
EPS = 1.0e-12


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-valid-json", type=Path, required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _target_cache(sample_root: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    scales: list[float] = []
    for sample_id in ids:
        sample_dir = sample_root / sample_id
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        if str(meta.get("split")) != "test_iid":
            raise RuntimeError(f"{sample_id}: expected test_iid, found {meta.get('split')!r}")
        bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        coords = np.load(sample_dir / "coords.npy").astype(np.float64)
        target = np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1) - bottom
        weights = control_volume_weights(coords)
        scale = math.sqrt(float(np.sum(np.square(target) * weights) / np.sum(weights)))
        scales.append(scale)
        result[sample_id] = {
            "bottom_temperature_K": bottom,
            "target_deltaT_K": target,
            "control_volumes_m3": weights,
            "q_W_m3": np.load(sample_dir / "q_field.npy").astype(np.float64).reshape(-1),
            "true_scale_cv_rms_K": scale,
        }
    q25, q50, q75 = np.quantile(np.asarray(scales), [0.25, 0.5, 0.75])
    for sample_id in ids:
        value = float(result[sample_id]["true_scale_cv_rms_K"])
        result[sample_id]["deltaT_quartile"] = (
            "Q1" if value <= q25 else "Q2" if value <= q50 else "Q3" if value <= q75 else "Q4"
        )
    return result


def _summary_stats(seconds: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(seconds, dtype=np.float64) * 1000.0
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid timing samples")
    return {
        "sample_count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def _block(raw: Any) -> None:
    jax.block_until_ready(raw)


def _metrics_summary(suite: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(suite["summary"])
    keep = (
        "point_global_relative_rmse_pct",
        "sample_first_cv_relative_rmse_pct",
        "raw_cv_weighted_rmse_K",
        "amplitude_ratio",
        "spatial_correlation",
        "hotspot_cv_weighted_rmse_K",
        "top5_cv_weighted_rmse_K",
        "strong_q_cv_weighted_rmse_K",
        "low_deltaT_background_bias_K",
        "low_deltaT_background_rmse_K",
        "low_deltaT_background_over_ratio",
        "shape_cv_rmse",
        "scale_log_rmse",
        "legacy_normalized_valid_base_mse",
    )
    return {key: summary[key] for key in keep if key in summary}


def _valid_v42_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload["models"]["V42"] if "models" in payload else payload
    if model["config_id"] != CONFIG_ID:
        raise RuntimeError("frozen valid artifact is not V42")
    checkpoint = model["checkpoint_metadata"]["point_global_best"]
    if int(checkpoint["epoch"]) != EXPECTED_EPOCH:
        raise RuntimeError("frozen valid point-global epoch is not e257")
    return {
        "artifact_sha256": _sha256(path),
        "evaluator_commit": model["evaluator_commit"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "summary": _metrics_summary(model["metrics"]["point_global_best"]),
    }


def _build_single_group(
    *,
    example: Any,
    stats: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    graph_seed: int,
    model_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    group = _make_batch_group_with_seed(
        "v5_final_timing_test_iid",
        [example],
        dict(stats),
        builder,
        graph_seed=graph_seed,
    )
    groups = [group]
    _attach_v5_physics(
        groups,
        _physics_cache([example]),
        dict(run_config["global_context"]["standardizer"]),
    )
    group["native_physics"] = group["v5_physics"]
    group["global_context"] = group["v5_physics"]["global_context"]
    if model_config.get("scale_context_mode", "none") != "none":
        stored_scale = dict((run_config.get("scale_context") or {}).get("standardizer") or {})
        encoded = standardize_scale_contexts([_scale_context_row_for_example(example)], stored_scale)[0]
        _attach_scale_context_to_groups(
            groups,
            {str(example.sample_id): encoded},
            expected_feature_dim=int(model_config.get("scale_context_feature_dim", 0)),
        )
    by_id = {str(example.sample_id): example}
    if (
        model_config.get("scale_pooling") == "qk_gated"
        or model_config.get("shape_attention_mode", "none") != "none"
        or model_config.get("scale_attention_mode", "none") != "none"
    ):
        _attach_qk_region_features_to_groups(
            groups,
            by_id,
            feature_version=str(model_config.get("qk_region_feature_version", "bugged_v1")),
        )
    if model_config.get("scale_deepsets_mode", "none") != "none":
        _attach_scale_deepsets_weights_to_groups(groups, by_id)
    return group


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    if run_dir.name != CONFIG_ID:
        raise RuntimeError("config_id/run directory binding failed")
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "loss_summary.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = [int(row["epoch"]) for row in summary.get("epoch_history", ())]
    if int(summary.get("final_epoch", -1)) != 600 or history != list(range(1, 601)):
        raise RuntimeError("V42 run is not a complete e600 run")
    payload, checkpoint = _checkpoint(run_dir, "point_global_best", summary)
    if int(checkpoint["epoch"]) != EXPECTED_EPOCH:
        raise RuntimeError(f"checkpoint selection changed: {checkpoint['epoch']} != {EXPECTED_EPOCH}")

    stats_payload = dict(payload["train_only_normalization"])
    install_checkpoint_feature_hooks(stats_payload)
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(
        sample_root, Path(str(run_config["split_map_path"]))
    )
    train_ids = list(split_ids.get("train") or ())
    test_ids = list(split_ids.get("test_iid") or ())
    if len(train_ids) != 672 or not test_ids or set(train_ids).intersection(test_ids):
        raise RuntimeError("unexpected train/test split")
    boundary_fallback = bool(run_config.get("boundary_mask_fallback", True))
    train_examples = _load_examples_for_ids(
        sample_root, train_ids, role="train", boundary_mask_fallback=boundary_fallback
    )
    test_examples = _load_examples_for_ids(
        sample_root, test_ids, role="test_iid", boundary_mask_fallback=boundary_fallback
    )
    physics_cache = _physics_cache(train_examples + test_examples)
    stats = stats_from_checkpoint_payload(stats_payload, train_examples)
    model_config = _resolve_decoder_bypass_model_config(dict(payload["model_config"]), stats)
    groups, context = _build_groups(
        run_config=run_config,
        model_config=model_config,
        stats=stats,
        train_examples=train_examples,
        valid_examples=test_examples,
        physics_cache=physics_cache,
        prediction_batch_size=1,
    )
    ordered = [str(value) for group in groups for value in group["sample_ids"]]
    if ordered != test_ids or any(int(group["inputs"].x_inp.shape[2]) != 1024 for group in groups):
        raise RuntimeError("test batch order/node count mismatch")

    model = GraphNeuralOperator(**model_config)
    params = _device_params(payload["params"])
    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    graph_seed = int(run_config.get("graph_seed", 0))
    for _ in range(args.warmup):
        _block(_model_apply(model, params, groups[0])["raw_temperature"])
    raw_fields: dict[str, np.ndarray] = {}
    forward_seconds: list[float] = []
    for sample_id, group in zip(test_ids, groups, strict=True):
        started = time.perf_counter()
        raw = _model_apply(model, params, group)["raw_temperature"]
        _block(raw)
        forward_seconds.append(time.perf_counter() - started)
        raw_fields[sample_id] = np.asarray(raw, dtype=np.float64).reshape(-1)

    # End-to-end excludes checkpoint/data I/O but includes one-sample graph
    # construction, all context/physics attachments, and synchronized forward.
    for _ in range(args.warmup):
        warm_group = _build_single_group(
            example=test_examples[0],
            stats=stats,
            builder=builder,
            graph_seed=graph_seed,
            model_config=model_config,
            run_config=run_config,
        )
        _block(_model_apply(model, params, warm_group)["raw_temperature"])
    end_to_end_seconds: list[float] = []
    for example in test_examples:
        started = time.perf_counter()
        one_group = _build_single_group(
            example=example,
            stats=stats,
            builder=builder,
            graph_seed=graph_seed,
            model_config=model_config,
            run_config=run_config,
        )
        raw = _model_apply(model, params, one_group)["raw_temperature"]
        _block(raw)
        end_to_end_seconds.append(time.perf_counter() - started)

    targets = _target_cache(sample_root, test_ids)
    suite = _suite(raw_temperature=raw_fields, valid_ids=test_ids, targets=targets, stats=stats_payload)
    for row in suite["per_sample"]:
        row["split"] = "test_iid"
    test_summary = _metrics_summary(suite)
    frozen_valid = _valid_v42_metrics(args.frozen_valid_json)
    differences = {
        key: float(test_summary[key]) - float(frozen_valid["summary"][key])
        for key in sorted(set(test_summary).intersection(frozen_valid["summary"]))
        if isinstance(test_summary[key], (int, float))
    }
    leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(params)]
    dtype_values = sorted({str(value.dtype) for value in leaves})
    device = str(jax.devices()[0])
    result = {
        "schema_version": "heat3d_v5_final_test_timing_v1",
        "status": "completed_frozen_checkpoint_test_and_timing",
        "scope": {
            "roles_accessed": ["train", "test_iid"],
            "valid_metrics_source": "frozen_existing_artifact_only",
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "checkpoint_reselected": False,
            "hyperparameters_tuned": False,
        },
        "binding": {
            "config_id": CONFIG_ID,
            "checkpoint_epoch": EXPECTED_EPOCH,
            "checkpoint_sha256": checkpoint["sha256"],
            "training_commit": checkpoint["training_commit"],
            "evaluator_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_host": args.source_host,
            "run_config_sha256": _sha256(run_config_path),
            "loss_summary_sha256": _sha256(summary_path),
            "train_ids_sha256": _ids_hash(train_ids),
            "test_iid_ids_sha256": _ids_hash(test_ids),
            "split_source": split_source,
            "test_iid_count": len(test_ids),
            "node_count": 1024,
        },
        "normalization_context": {
            "train_only_normalization": True,
            "train_sample_count": len(train_ids),
            "global_context_fit_population": context["global_context"].get("standardizer", {}).get("fit_population"),
            "global_context_recompute_max_abs": context["global_context_recompute_max_abs"],
        },
        "test_iid": {"summary": test_summary, "per_sample": suite["per_sample"]},
        "frozen_valid_iid": frozen_valid,
        "test_minus_valid": differences,
        "timing": {
            "device": device,
            "backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "parameter_dtypes": dtype_values,
            "batch_size": 1,
            "warmup_iterations": args.warmup,
            "synchronization": "jax.block_until_ready(raw_temperature) per sample",
            "excluded": ["checkpoint_load", "dataset_file_io"],
            "model_forward": _summary_stats(forward_seconds),
            "graph_preprocess_and_model_forward": _summary_stats(end_to_end_seconds),
        },
    }
    if not all(math.isfinite(float(value)) for value in test_summary.values() if isinstance(value, (int, float))):
        raise RuntimeError("non-finite test metric")
    for output in (args.output_json, args.output_csv, args.output_markdown):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        rows = suite["per_sample"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# V42 e257 final test_iid evaluation and batch-1 timing",
        "",
        "Frozen checkpoint only; test_iid was opened after V5 selection was complete. No hard/sealed access.",
        "",
        "## Core metrics",
        "",
        "| split | point-global | sample-first | raw CV RMSE K |",
        "|---|---:|---:|---:|",
        f"| valid_iid (frozen) | {frozen_valid['summary']['point_global_relative_rmse_pct']:.6f}% | {frozen_valid['summary']['sample_first_cv_relative_rmse_pct']:.6f}% | {frozen_valid['summary']['raw_cv_weighted_rmse_K']:.6f} |",
        f"| test_iid | {test_summary['point_global_relative_rmse_pct']:.6f}% | {test_summary['sample_first_cv_relative_rmse_pct']:.6f}% | {test_summary['raw_cv_weighted_rmse_K']:.6f} |",
        "",
        "## Timing",
        "",
        "| path | mean ms | median ms | P90 ms | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("model_forward", "model forward"), ("graph_preprocess_and_model_forward", "end-to-end")):
        row = result["timing"][key]
        lines.append(f"| {label} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | {row['sample_count']} |")
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "test_count": len(test_ids), "device": device}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
